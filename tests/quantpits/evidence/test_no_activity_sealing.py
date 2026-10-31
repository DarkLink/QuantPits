"""Real post-trade auditing and no-activity evidence sealing in temporary workspaces."""

import copy
import json
from dataclasses import replace
from decimal import Decimal

import pandas as pd
import pytest

from quantpits.evidence import ProductionCycleEvidenceSealer
from quantpits.post_trade.audit import write_post_trade_manifest
from quantpits.post_trade.command import PostTradeRunOptions, execute_prepared, prepare_post_trade_run
from quantpits.post_trade.service import PostTradeService
from quantpits.post_trade.state import ValuationSnapshot
from quantpits.utils.workspace import WorkspaceContext
from tests.quantpits.evidence.test_sealing import cycle_factory, _write_json


DAYS = ("2099-01-01", "2099-01-02")


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
    return root, qlib, request, summary


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


@pytest.mark.parametrize("settlement_mode", ["empty", "daily", "bundle"])
def test_new_daily_or_bundle_audit_and_seal(
    cycle_factory, tmp_path, settlement_mode,
):
    root, qlib, request, summary = _layout(
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


def test_new_sealer_excludes_old_committed_logical_empty_input_refs(
    cycle_factory, tmp_path,
):
    root, qlib, request, _ = _layout(cycle_factory, tmp_path)
    _legacy_manifest(root, request)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert {row["code"] for row in result.problems} == {"deep_analysis_missing", "ranking_coverage_partial"}
