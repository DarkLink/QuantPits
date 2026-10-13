"""Real temporary post-trade/seal/capsule writers and readers, with finite inputs."""

import copy
import json
from dataclasses import replace
from decimal import Decimal

import pandas as pd
import pytest

from quantpits.evidence import ProductionCycleEvidenceSealer
from quantpits.post_trade.audit import write_post_trade_manifest
from quantpits.post_trade.command import PostTradeRunOptions, execute_prepared, prepare_post_trade_run
from quantpits.post_trade.contracts import ExecutionEvidenceGapError
from quantpits.post_trade.service import PostTradeService
from quantpits.post_trade.state import ValuationSnapshot
from quantpits.research import signal_input_capsule as signal
from quantpits.research import decision_surface as surface
from quantpits.utils.workspace import WorkspaceContext
from tests.quantpits.evidence.test_sealing import cycle_factory, _write_json


DAYS = ("2099-01-01", "2099-01-02")


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def _layout(cycle_factory, tmp_path, *, partial=True, settlement_mode="empty"):
    root, qlib, request = cycle_factory(deep=not partial, scores={"AAA": 2.0} if partial else None)
    ctx = WorkspaceContext.from_root(root)
    ctx.data_dir.mkdir(exist_ok=True)
    _write_json(ctx.config_path("prod_config.json"), {
        "last_processed_date": "2098-12-31", "current_date": "2098-12-31",
        "current_cash": 1000, "current_holding": [
            {"instrument": "SH600000", "value": 10, "amount": 100},
        ], "broker": "gtja", "model": "SYNTHETIC", "benchmark": "SH000300",
    })
    _write_json(ctx.config_path("cashflow.json"), {})
    _write_json(ctx.data_path("post_trade_ingestion_state.json"), {
        "schema_version": 1, "sources": {},
    })
    _write_json(ctx.data_path(".order_trade_state.json"), {})
    prices = ctx.data_path("prices.csv")
    prices.write_text("date,close,benchmark\n2099-01-01,10,3000\n2099-01-02,11,3100\n")

    class Prices:
        def snapshot(self, day, instruments, _benchmark):
            row = pd.read_csv(prices).set_index("date").loc[day]
            return ValuationSnapshot(day, tuple((key, Decimal(str(row.close))) for key in instruments),
                                     Decimal(str(row.benchmark)))

    class Adapter:
        def parse_settlement(self, _path):
            return pd.DataFrame({"交收日期": [20990101], "交易类别": ["利息归本"],
                "成交价格": [0], "成交数量": [0], "成交金额": [0], "资金发生数": [0]})

    bundle = None
    if settlement_mode == "daily":
        ctx.data_path("2099-01-01-table.xlsx").write_bytes(b"synthetic daily export")
    elif settlement_mode == "bundle":
        bundle = "data/2099-01-01-2099-01-02-table.xlsx"
        (root / bundle).write_bytes(b"synthetic interval export")
    options = PostTradeRunOptions(scope="state", end_date=DAYS[-1], allow_missing_settlement=True,
                                  run_id="empty-days", settlement_bundle=bundle)
    args = ("--scope", "state", "--end-date", DAYS[-1], "--allow-missing-settlement")
    service = PostTradeService(Prices())
    state = lambda prepared, parsed: service.run_state(prepared, parsed, DAYS)
    preview = execute_prepared(prepare_post_trade_run(ctx, replace(options, dry_run=True), cli_args=args),
        Adapter(), resolve_trade_dates=lambda *_: DAYS, state_callback=state)
    assert preview.state_result.dry_run and preview.state_result.artifacts is None
    assert json.loads(ctx.config_path("prod_config.json").read_text())["last_processed_date"] == "2098-12-31"
    summary = execute_prepared(prepare_post_trade_run(ctx, options, cli_args=args), Adapter(),
                              resolve_trade_dates=lambda *_: DAYS, state_callback=state)
    assert summary.state_result.change_set.final_state.cash == Decimal("1000.00")
    assert summary.state_result.change_set.transitions[-1].valuation.closes == (("SH600000", Decimal("11")),)
    m1, _ = write_post_trade_manifest(summary.prepared, summary, started_at="2099-01-02T00:00:00")
    request = replace(request, post_trade_manifest=m1.relative_to(root).as_posix())

    for logical in (request.prediction_manifest, request.order_manifest):
        raw = json.loads((root / logical).read_text())
        raw.update(schema_version=1, args=[])
        _write_json(root / logical, raw)
    m3 = root / request.ensemble_manifest
    raw = json.loads(m3.read_text())
    raw.update(schema_version=1, args=["--from-config"])
    template = raw["records"]["input_models"][0]
    models = []
    for position in range(4):
        model = copy.deepcopy(template)
        model.update(resolved_key="MODEL_%d" % position, recorder_id="PRED_%d" % position,
            source_recorder_id="SOURCE_%d" % position, operation="predict_only",
            artifact_path="mlruns/pred_%d" % position, source_artifact_path="mlruns/source_%d" % position)
        models.append(model)
        pred = root / model["artifact_path"]
        source = root / model["source_artifact_path"]
        pred.mkdir(); source.mkdir()
        (source / "model.pkl").write_bytes(("synthetic model %d" % position).encode())
        pd.DataFrame({"score": [float(position)]}, index=pd.MultiIndex.from_tuples(
            [("AAA", DAYS[-1])], names=["instrument", "datetime"])).to_pickle(pred / "pred.pkl")
    raw["records"]["input_models"] = models
    combo = raw["records"]["combos"][0]
    combo["models"] = combo["resolved_models"] = [row["resolved_key"] for row in models]
    _write_json(m3, raw)
    research = tmp_path / "research"
    store = research / "research/shadow_v1/signal_input_capsules"
    store.mkdir(parents=True, mode=0o700); store.chmod(0o700)
    return root, qlib, request, research, store, summary


def _legacy_manifest(root, request):
    path = root / request.post_trade_manifest
    raw = json.loads(path.read_text())
    raw["records"]["settlement_logical_partitions"] = {}
    raw["records"].pop("settlement_logical_day_count")
    raw["records"]["settlement_source_count"] = len(DAYS)
    raw["inputs"].extend({"description": "settlement", "kind": "data",
        "path": "data/%s-table.xlsx" % day, "required": False} for day in DAYS)
    _write_json(path, raw)
    return raw


def _capture_old(root, qlib, request, monkeypatch):
    # Produce the fixed v1 layout with the old reference handling and generic
    # diagnostics. The consumer and capsule writers are never mocked.
    from quantpits.post_trade import audit
    from quantpits.evidence import sealing
    original = sealing._problem
    def problem(code, evidence_class, detail, **kwargs):
        if code == "referenced_file_missing":
            detail = "referenced filesystem evidence is missing"
        return original(code, evidence_class, detail, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(audit, "legacy_assumed_empty_settlement_inputs", lambda _: ())
        patch.setattr(sealing, "_problem", problem)
        result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    return root / result.bundle_path


def _retain(root, request, research, store):
    before = _snapshot(root)
    plan = signal.prepare_signal_input_capsule(root, research, request.cycle_id, store)
    assert plan.status == "PREPARED", plan.to_safe_dict()
    assert not plan.to_safe_dict()["did_write"] and _snapshot(root) == before
    result = signal.publish_signal_input_capsule(root, research, request.cycle_id, store,
        plan.capsule_id, plan.to_safe_dict()["request_digest"], signal.AUTHORIZATION_ACTION)
    assert result.status == "COMMITTED", result.to_safe_dict()
    assert result.critical_signal_retention_complete
    receipt = json.loads((store / plan.capsule_id / signal.RECEIPT_NAME).read_bytes())
    assert receipt["requested_member_count"] == 5
    assert receipt["phase37a_status"] == "sealed_partial"
    adopted = signal.adopt_signal_input_capsule(root, research, request.cycle_id, store, plan.capsule_id)
    assert adopted.status == "ADOPTED" and not adopted.did_write
    assert _snapshot(root) == before
    return receipt


@pytest.mark.parametrize("settlement_mode", ["empty", "daily", "bundle"])
def test_new_daily_or_bundle_audit_seal_and_real_five_member_retention(
    cycle_factory, tmp_path, settlement_mode,
):
    root, qlib, request, research, store, summary = _layout(
        cycle_factory, tmp_path, settlement_mode=settlement_mode)
    raw = json.loads((root / request.post_trade_manifest).read_text())
    logical = raw["records"]["settlement_logical_partitions"]
    assert logical == summary.prepared.plan.metadata["settlement_logical_partitions"]
    assert logical[DAYS[-1]]["status"] == "assumed_empty" and logical[DAYS[-1]]["row_count"] == 0
    assert raw["records"]["settlement_logical_day_count"] == 2
    assert raw["records"]["settlement_source_count"] == int(settlement_mode != "empty")
    assert all(row.get("path") != "data/%s-table.xlsx" % DAYS[-1] for row in raw["inputs"])
    if settlement_mode != "empty":
        assert logical[DAYS[0]]["status"] == "observed" and logical[DAYS[0]]["row_count"] == 1
        assert logical[DAYS[0]]["source_fingerprint"]
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert {row["code"] for row in result.problems} == {"deep_analysis_missing", "ranking_coverage_partial"}
    receipt = _retain(root, request, research, store)
    assert "post_trade_context" not in receipt  # Existing normal receipt contract.


def test_old_v1_seal_keeps_two_occurrences_and_remains_partial_after_retention(
    cycle_factory, tmp_path, monkeypatch,
):
    root, qlib, request, research, store, _ = _layout(cycle_factory, tmp_path)
    _legacy_manifest(root, request)
    cycle = _capture_old(root, qlib, request, monkeypatch)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    missing = [row for row in manifest["problems"] if row["code"] == "referenced_file_missing"]
    assert len(missing) == 2 and missing[0] == missing[1]
    # The allowance belongs exclusively to signal consumption.
    with pytest.raises(surface._ComponentIncomparable, match="NOT_SCOPED"):
        surface._cycle_authority(root, request.cycle_id, allow_partial_ranking=True)
    receipt = _retain(root, request, research, store)
    assert receipt["phase37a_problems"] == manifest["problems"]
    assert receipt["problem_inventory_digest"] == signal.TypedDigest.canonical(manifest["problems"]).to_dict()
    assert receipt["post_trade_context"]["settlement_day_detail_status"] == "legacy_not_recorded"
    assert receipt["post_trade_context"]["production_replay_complete"] is False


def test_new_sealer_excludes_old_committed_logical_empty_input_refs(
    cycle_factory, tmp_path,
):
    root, qlib, request, research, store, _ = _layout(cycle_factory, tmp_path)
    _legacy_manifest(root, request)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert {row["code"] for row in result.problems} == {"deep_analysis_missing", "ranking_coverage_partial"}
    _retain(root, request, research, store)


@pytest.mark.parametrize("mutation", ["no_allow", "uncommitted", "range", "required", "fingerprinted",
                                     "other_missing", "required_alias"])
def test_legacy_optional_input_alone_cannot_explain_missing_required_evidence(
    cycle_factory, tmp_path, monkeypatch, mutation,
):
    root, qlib, request, research, store, _ = _layout(cycle_factory, tmp_path)
    raw = _legacy_manifest(root, request)
    if mutation == "no_allow":
        raw["args"].remove("--allow-missing-settlement")
    elif mutation == "uncommitted":
        raw["records"]["state_committed"] = False
    elif mutation == "range":
        raw["records"]["processed_date_count"] = 1
    elif mutation == "required":
        raw["inputs"][-1]["required"] = True
    elif mutation == "fingerprinted":
        raw["inputs"][-1]["fingerprint"] = "a" * 64
    elif mutation == "other_missing":
        raw["inputs"].append({"kind": "data", "path": "data/real-required.csv", "required": True})
    else:
        raw["inputs"].append({"kind": "data", "path": raw["inputs"][-1]["path"], "required": True})
    _write_json(root / request.post_trade_manifest, raw)
    _capture_old(root, qlib, request, monkeypatch)
    before = _snapshot(store)
    plan = signal.prepare_signal_input_capsule(root, research, request.cycle_id, store)
    assert plan.status == "PRECONDITION_BLOCKED" and not plan.to_safe_dict()["did_write"]
    assert plan.reason_code == "CYCLE_BLOCKING_PROBLEM_NOT_SCOPED"
    assert _snapshot(store) == before


@pytest.mark.parametrize("mutation", ["missing_prediction", "wrong_anchor", "wrong_prediction_anchor",
                                     "changed_content", "damaged_seal", "duplicate_member", "ranking_mismatch",
                                     "changed_live_prediction", "missing_live_prediction"])
def test_empty_day_compatibility_cannot_admit_bad_signal_or_seal(
    cycle_factory, tmp_path, monkeypatch, mutation,
):
    root, qlib, request, research, store, _ = _layout(cycle_factory, tmp_path)
    _legacy_manifest(root, request)
    if mutation == "missing_prediction":
        (root / "mlruns/pred_2/pred.pkl").unlink()
    elif mutation == "wrong_anchor":
        path = root / request.prediction_manifest
        raw = json.loads(path.read_text()); raw["records"]["anchor_date"] = DAYS[0]
        _write_json(path, raw)
    elif mutation == "wrong_prediction_anchor":
        path = root / "mlruns/pred_2/pred.pkl"
        frame = pd.read_pickle(path)
        frame.index = pd.MultiIndex.from_tuples([("AAA", DAYS[0])], names=["instrument", "datetime"])
        frame.to_pickle(path)
    elif mutation == "duplicate_member":
        path = root / request.ensemble_manifest
        raw = json.loads(path.read_text())
        first, second = raw["records"]["input_models"][:2]
        second["recorder_id"] = first["recorder_id"]
        second["artifact_path"] = first["artifact_path"]
        _write_json(path, raw)
    elif mutation in {"changed_live_prediction", "missing_live_prediction"}:
        path = root / "mlruns/pred_2/pred.pkl"
        frame = pd.read_pickle(path)
        # Exceed the real sealer's embedding threshold, exercising its live
        # preservation path without mocking file observation or retention.
        frame["payload"] = "x" * (2 * 1024 * 1024 + 100)
        frame.to_pickle(path)
    cycle = _capture_old(root, qlib, request, monkeypatch)
    if mutation == "changed_content":
        # Remove the retained object's exact original bytes, preserving size.
        manifest = json.loads((cycle / "manifest.json").read_bytes())
        row = next(row for row in manifest["model_and_ensemble_lineage"]["source_artifacts"]
                   if row.get("position") == 2 and "role" not in row)
        digest = row["members"][0]["digest"]["value"]
        path = cycle / "objects" / digest[:2] / digest
        path.write_bytes(b"x" * path.stat().st_size)
    elif mutation == "damaged_seal":
        (cycle / "seal.json").write_bytes(b"{}")
    elif mutation == "ranking_mismatch":
        (cycle / "ranking.csv").write_bytes(b"changed ranking")
    elif mutation == "changed_live_prediction":
        path = root / "mlruns/pred_2/pred.pkl"
        frame = pd.read_pickle(path); frame["score"] = 99.0
        frame.to_pickle(path)
    elif mutation == "missing_live_prediction":
        (root / "mlruns/pred_2/pred.pkl").unlink()
    plan = signal.prepare_signal_input_capsule(root, research, request.cycle_id, store)
    assert plan.status == "PRECONDITION_BLOCKED", plan.to_safe_dict()
    assert not plan.to_safe_dict()["did_write"] and not list(store.iterdir())


def test_real_filled_trade_without_settlement_fails_before_state_commit(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    ctx.config_dir.mkdir(); ctx.data_dir.mkdir()
    _write_json(ctx.config_path("prod_config.json"), {
        "last_processed_date": DAYS[0], "current_cash": 1000, "current_holding": [],
    })
    _write_json(ctx.config_path("cashflow.json"), {})
    ctx.data_path("%s-trade.xlsx" % DAYS[-1]).write_bytes(b"filled trade")
    ctx.data_path("%s-order.xlsx" % DAYS[-1]).write_bytes(b"filled order")
    class Adapter:
        def parse_trades(self, _path):
            return pd.DataFrame({"交易类别": ["深圳A股普通股票竞价买入"], "成交数量": [10]})
        parse_orders = parse_trades
    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="all", end_date=DAYS[-1],
                                                             allow_missing_settlement=True))
    before = _snapshot(ctx.root)
    with pytest.raises(ExecutionEvidenceGapError, match="without settlement"):
        execute_prepared(prepared, Adapter(), resolve_trade_dates=lambda *_: (DAYS[-1],),
                         state_callback=lambda *_: pytest.fail("must not commit"))
    assert _snapshot(ctx.root) == before
