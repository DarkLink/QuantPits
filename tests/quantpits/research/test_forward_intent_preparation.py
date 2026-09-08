"""C3 joins use real frozen definition/bootstrap and actual B1/price arithmetic.

Surface/current seal and capsule admission have independent integration suites;
this fixture substitutes those boundaries, not ranking, prior or planning.
"""
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.research import forward_intent_preparation as m
from quantpits.research import decision_surface as surface
from tests.quantpits.research.test_forward_definition_evidence import fresh_evidence_workspace
from tests.quantpits.research.test_forward_bootstrap import fresh_bootstrap_workspace, _fresh_bootstrap_publish
from tests.quantpits.research.test_model_continuity import copied_models

ANCHOR = "2026-09-04"
TRADE = "2026-09-07"
INSTRUMENTS = ("SH600000", "SZ000001", "SZ000002")


def _json(path, value):
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)


def _prediction(values=(1., 2., 3.)):
    frame = pd.DataFrame({"score": values}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp(ANCHOR), instrument) for instrument in INSTRUMENTS], names=("datetime", "instrument")))
    stream = io.BytesIO()
    frame.to_pickle(stream)
    return stream.getvalue()


@pytest.fixture
def prepared_inputs(fresh_bootstrap_workspace, tmp_path, monkeypatch):
    production, research, activation, definitions, evidence, bootstraps = fresh_bootstrap_workspace
    bootstrap = _fresh_bootstrap_publish(fresh_bootstrap_workspace)
    assert bootstrap.status == "COMMITTED"
    identifier = bootstrap.to_safe_summary_dict()["bootstrap_set_id"]
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / ".git").mkdir()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    day = b"2026-08-28\n2026-09-04\n"
    (provider / "calendars/day.txt").write_bytes(day)
    (provider / "calendars/day_future.txt").write_bytes(day + b"2026-09-07\n")
    universe = ("\n".join(i + "\t2020-01-01\t2030-01-01" for i in INSTRUMENTS) + "\n").encode()
    (provider / "instruments/csi300.txt").write_bytes(universe)
    states = [m._read_json(bootstraps / identifier / (role.lower() + "_state.json")) for role in m.ROLES]
    union = set(INSTRUMENTS) | {p["instrument"] for state in states for p in state["positions"]}
    for instrument in union:
        directory = provider / "features" / instrument.lower()
        directory.mkdir(parents=True)
        for field, number in (("close", 10.), ("factor", 1.)):
            (directory / (field + ".day.bin")).write_bytes(struct.pack("<fff", 0., number, number))
    signals = research / "research/shadow_v1/signal_input_capsules"
    models = research / "research/shadow_v1/model_artifact_capsules"
    for path in (signals, models):
        path.mkdir(mode=0o700)
    sid, mid = "signalcapsule." + "a" * 64, "modelcapsule." + "b" * 64
    signal_target = signals / sid
    signal_target.mkdir(mode=0o700)
    (models / mid).mkdir(mode=0o700)
    from quantpits.research.signal_input_capsule import MEMBER_NAMES
    metadata = []
    for name in MEMBER_NAMES:
        data = _prediction()
        (signal_target / name).write_bytes(data)
        metadata.append({"logical_path": name, "raw_digest": surface._digest(data, "raw_bytes")})
    _json(signal_target / "capsule_manifest.json", {"members": metadata})
    ranking = m.replay.rank_complete_anchor(prediction_bytes_by_member={str(i): _prediction() for i in range(4)},
              member_order=tuple(str(i) for i in range(4)), anchor_date=ANCHOR, eligible_instruments=INSTRUMENTS)
    cycle = production / "data/evidence/v1/cycles" / ANCHOR
    cycle.mkdir(mode=0o700)
    (cycle / "ranking.csv").write_bytes(ranking.to_csv_bytes())
    manifest = {"data_identity": {"qlib_materialization_identity": {
        "status": "observed", "universe_name": "csi300",
        "universe_digest": surface._digest(universe, "raw_bytes"), "calendar_digest": surface._digest(day, "raw_bytes"),
    }}}
    seal = {"named_file_digests": {"ranking.csv": surface._digest(ranking.to_csv_bytes(), "raw_bytes")}}
    original = surface._cycle_authority
    monkeypatch.setattr(surface, "_cycle_authority", lambda root, date: (cycle, manifest, seal) if date == ANCHOR else original(root, date))
    monkeypatch.setattr(surface, "observe_production_decision_surface", lambda *args, **kw: SimpleNamespace(status="SAME_CHAMPION_SEGMENT", same_champion_segment=True))
    original_projection = surface._source_projection
    monkeypatch.setattr(surface, "_source_projection", lambda value: {} if value is manifest else original_projection(value))
    monkeypatch.setattr(surface, "_source_matches_definition", lambda *args: True)
    monkeypatch.setattr(m, "_engine", lambda *args: ("a" * 40, "b" * 40, "c" * 64))
    import quantpits.research.signal_input_capsule as signal_module
    import quantpits.research.model_artifact_capsule as model_module
    calls = []
    def adopt_signal(prod, res, date, store, selector):
        calls.append(("signal", date))
        return SimpleNamespace(status="ADOPTED", critical_signal_retention_complete=True, did_write=False,
                               manifest_digest=surface._digest((store / selector / "capsule_manifest.json").read_bytes(), "raw_bytes"))
    def adopt_model(prod, res, date, *args):
        calls.append(("model", date))
        return SimpleNamespace(status="ADOPTED", model_artifact_retention_complete=True, did_write=False, manifest_digest=surface._digest(b"model", "raw_bytes"))
    monkeypatch.setattr(signal_module, "adopt_signal_input_capsule", adopt_signal)
    monkeypatch.setattr(model_module, "adopt_definition_bound_model_artifact_capsule", adopt_model)
    args = (production, research, engine, provider, ANCHOR, activation, definitions, evidence,
            bootstraps, identifier, signals, sid, models, mid)
    return args, calls, manifest, seal


def test_complete_pair_prepared(prepared_inputs, capsys):
    args, calls, _, _ = prepared_inputs
    before = m._metadata((args[0], args[1], args[3]))
    result = m.prepare_first_forward_intent(*args)
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "PREPARED", summary["reason_codes"]
    assert [role["status"] for role in summary["roles"]] == ["COMPLETE", "COMPLETE"]
    assert calls == [("model", "2026-08-14"), ("signal", ANCHOR)]
    assert result.pair[0][0].portfolio_id != result.pair[0][1].portfolio_id
    assert summary["roles"][0]["ranking_digest"] == summary["roles"][1]["ranking_digest"]
    assert summary["comparison"]["top_overlap_count"] == 3
    assert summary["trade_date"] == TRADE
    assert all(summary[key] is False for key in ("intent_publication_capability", "epoch_started", "prospective_claim", "promotion_capability", "did_write"))
    assert before == m._metadata((args[0], args[1], args[3]))
    assert m.prepare_first_forward_intent(*args).to_safe_summary_dict()["preparation_digest"] == summary["preparation_digest"]
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert not any(token in json.dumps(summary) for token in (*INSTRUMENTS, str(args[1]), "MODEL_A", args[9]))


def test_arm_failure_keeps_pair_inventory(prepared_inputs, monkeypatch):
    args = prepared_inputs[0]
    original = m.CurrentRuleShadowIntentPlanner.plan
    calls = []
    def plan(**kwargs):
        calls.append(kwargs["portfolio_id"])
        if len(calls) == 1:
            raise ValueError("PRIVATE_HOLDINGS")
        return original(**kwargs)
    monkeypatch.setattr(m.CurrentRuleShadowIntentPlanner, "plan", plan)
    result = m.prepare_first_forward_intent(*args)
    assert result.status == "PRECONDITION_BLOCKED"
    assert [r["status"] for r in result.to_safe_summary_dict()["roles"]] == ["FAILED", "COMPLETE"]
    assert len(calls) == 2 and result.pair is None


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates(prepared_inputs, monkeypatch, exception):
    calls = []
    def stop(**kw):
        calls.append(1)
        raise exception()
    monkeypatch.setattr(m.CurrentRuleShadowIntentPlanner, "plan", stop)
    with pytest.raises(exception):
        m.prepare_first_forward_intent(*prepared_inputs[0])
    assert calls == [1]


@pytest.mark.parametrize("kind", ["anchor_absent", "future_absent", "order", "duplicate", "prefix"])
def test_calendar_invalid_blocks_both_arms(prepared_inputs, kind):
    args = prepared_inputs[0]
    path = args[3] / "calendars/day_future.txt"
    values = {"anchor_absent": b"2026-09-07\n", "future_absent": b"2026-08-28\n2026-09-04\n",
              "order": b"2026-09-04\n2026-08-28\n2026-09-07\n", "duplicate": b"2026-08-28\n2026-09-04\n2026-09-04\n2026-09-07\n",
              "prefix": b"2026-08-27\n2026-09-04\n2026-09-07\n"}
    path.write_bytes(values[kind])
    result = m.prepare_first_forward_intent(*args)
    assert result.status == "PRECONDITION_BLOCKED"
    assert all(r["status"] == "NOT_RUN" for r in result.to_safe_summary_dict()["roles"])


def test_union_price_projection(prepared_inputs, monkeypatch):
    original = m.QlibCashPriceObserver.observe_anchor_close
    calls = []
    def observe(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)
    monkeypatch.setattr(m.QlibCashPriceObserver, "observe_anchor_close", observe)
    monkeypatch.setattr(m.QlibCashPriceObserver, "observe_next_open", lambda *a, **kw: pytest.fail("next-open forbidden"))
    result = m.prepare_first_forward_intent(*prepared_inputs[0])
    assert result.status == "PREPARED", result.to_safe_summary_dict()
    assert len(calls) == 1
    for prior, ranking, snapshot in zip(*result.pair[:3]):
        assert snapshot.requested_instruments == tuple(sorted({r["instrument"] for r in ranking.rows} | {p.instrument for p in prior.positions}))


def test_business_partial_plans_remain_complete(prepared_inputs):
    args = prepared_inputs[0]
    for path in (args[3] / "features").rglob("close.day.bin"):
        path.unlink()
    result = m.prepare_first_forward_intent(*args)
    summary = result.to_safe_summary_dict()
    assert result.status == "PREPARED", summary["reason_codes"]
    assert all(prior.cash < 0 for prior in result.pair[0])
    assert all(role["order_count"] == 0 and role["pending_count"] > 0 and role["shortage_count"] > 0 for role in summary["roles"])
    assert summary["price_counts"]["missing"] > 0


@pytest.mark.parametrize("status,expected", [("VERSION_BREAK", "VERSION_BREAK"), ("INCOMPARABLE", "PRECONDITION_BLOCKED")])
def test_surface_status_preserved(prepared_inputs, monkeypatch, status, expected):
    monkeypatch.setattr(surface, "observe_production_decision_surface", lambda *a, **kw: SimpleNamespace(status=status, same_champion_segment=False))
    result = m.prepare_first_forward_intent(*prepared_inputs[0])
    assert result.status == expected and result.pair is None
    assert all(role["status"] == "NOT_RUN" for role in result.to_safe_summary_dict()["roles"])


def test_runtime_code_mismatch_blocks(prepared_inputs, monkeypatch):
    def mismatch(*args):
        raise m._Blocked("RUNTIME_CODE_MISMATCH")
    monkeypatch.setattr(m, "_engine", mismatch)
    summary = m.prepare_first_forward_intent(*prepared_inputs[0]).to_safe_summary_dict()
    assert summary["reason_codes"] == ["RUNTIME_CODE_MISMATCH"]
    assert all(r["status"] == "NOT_RUN" for r in summary["roles"])


@pytest.mark.parametrize("kind", ["missing", "foreign", "duplicate", "nan", "inf", "wrong_anchor"])
def test_missing_scores_are_explicitly_blocked(prepared_inputs, kind):
    args = prepared_inputs[0]
    target = args[10] / args[11]
    member = target / "source_prediction_0.bin"
    frame = pd.read_pickle(io.BytesIO(member.read_bytes()))
    if kind == "missing":
        frame = frame.iloc[:-1]
    elif kind == "foreign":
        frame.index = pd.MultiIndex.from_tuples([(pd.Timestamp(ANCHOR), "FOREIGN"), *list(frame.index)[1:]], names=frame.index.names)
    elif kind == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif kind == "wrong_anchor":
        frame.index = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-09-03"), i) for i in INSTRUMENTS], names=frame.index.names)
    else:
        frame.iloc[0, 0] = float(kind)
    frame.to_pickle(member)
    raw = m._read_json(target / "capsule_manifest.json")
    raw["members"][0]["raw_digest"] = surface._digest(member.read_bytes(), "raw_bytes")
    _json(target / "capsule_manifest.json", raw)
    summary = m.prepare_first_forward_intent(*args).to_safe_summary_dict()
    assert summary["status"] == "PRECONDITION_BLOCKED"
    assert summary["roles"][0]["coverage_counts"]["eligible"] == 3
    assert all(r["status"] == "NOT_RUN" for r in summary["roles"])


@pytest.mark.parametrize("kind", ["score", "rank", "unscored"])
def test_champion_parity_is_actual(prepared_inputs, kind):
    args, _, _, seal = prepared_inputs
    scores = {INSTRUMENTS[0]: 0., INSTRUMENTS[1]: 0.5, INSTRUMENTS[2]: 1.}
    if kind == "score":
        scores[INSTRUMENTS[1]] = 0.6
    elif kind == "rank":
        scores[INSTRUMENTS[1]] = 1.
        scores[INSTRUMENTS[2]] = 0.5
    else:
        del scores[INSTRUMENTS[1]]
    data = m.replay.canonical_full_ranking(INSTRUMENTS, scores).to_csv_bytes()
    (args[0] / "data/evidence/v1/cycles" / ANCHOR / "ranking.csv").write_bytes(data)
    seal["named_file_digests"]["ranking.csv"] = surface._digest(data, "raw_bytes")
    summary = m.prepare_first_forward_intent(*args).to_safe_summary_dict()
    assert summary["status"] == "PRECONDITION_BLOCKED"
    assert summary["reason_codes"][0] == {"score": "CHAMPION_PARITY_SCORE", "rank": "CHAMPION_PARITY_ORDER", "unscored": "SEALED_RANKING_INCOMPLETE"}[kind]


def test_final_input_change_denies_pair(prepared_inputs, monkeypatch):
    args = prepared_inputs[0]
    original = m.CurrentRuleShadowIntentPlanner.plan
    def plan(**kwargs):
        result = original(**kwargs)
        path = args[3] / "calendars/day_future.txt"
        path.write_bytes(path.read_bytes() + b"2026-09-08\n")
        return result
    monkeypatch.setattr(m.CurrentRuleShadowIntentPlanner, "plan", plan)
    result = m.prepare_first_forward_intent(*args)
    assert result.status == "PRECONDITION_BLOCKED" and result.pair is None
    assert result.to_safe_summary_dict()["reason_codes"] == ["INPUT_STABILITY_LOST"]


@pytest.mark.parametrize("field", ["role", "economic_state_digest", "bootstrap_source_cycle_id"])
def test_bootstrap_join_corruption_blocks(prepared_inputs, field):
    args = prepared_inputs[0]
    path = args[8] / args[9] / "bootstrap_manifest.json"
    raw = m._read_json(path)
    if field == "role":
        raw["roles"][0]["role"] = "CHALLENGER"
    elif field == "economic_state_digest":
        raw[field]["value"] = "0" * 64
    else:
        raw[field] = ANCHOR
    _json(path, raw)
    summary = m.prepare_first_forward_intent(*args).to_safe_summary_dict()
    assert summary["status"] == "PRECONDITION_BLOCKED"
    assert all(role["status"] == "NOT_RUN" for role in summary["roles"])


def test_calendar_extension_is_not_historical_price_claim(prepared_inputs):
    args = prepared_inputs[0]
    path = args[3] / "calendars/day.txt"
    path.write_bytes(path.read_bytes() + b"2026-09-07\n")
    summary = m.prepare_first_forward_intent(*args).to_safe_summary_dict()
    assert summary["status"] == "PREPARED", summary["reason_codes"]
    assert summary["calendar_match_status"] == "DIFFERENT"
    assert summary["historical_materialization_relation"] == "unverified"


def test_actual_engine_sources_match_seal(tmp_path, monkeypatch):
    from tests.quantpits.research.test_decision_surface import _git
    import shutil
    loaded = Path(m.__file__).resolve().parents[2]
    engine = tmp_path / "engine"
    engine.mkdir()
    _git(engine, "init", "-q")
    _git(engine, "config", "user.email", "test@example.invalid")
    _git(engine, "config", "user.name", "Test")
    for logical in surface.CURATED_CODE_PATHS:
        target = engine / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(loaded / logical), str(target))
    _git(engine, "add", ".")
    _git(engine, "commit", "-qm", "fixture")
    commit = _git(engine, "rev-parse", "HEAD")
    manifest = {"engine_identity": {"commit": commit, "status_inventory": []}}
    original_engine = m._engine
    observed = original_engine(engine, manifest)
    assert observed[0] == commit and len(observed[2]) == 64
    (engine / "quantpits/utils/strategy.py").write_text("# changed economic code\n")
    with pytest.raises(m._Blocked, match="RUNTIME_CODE_MISMATCH"):
        original_engine(engine, manifest)


def test_final_pair_construction_failure_cannot_leave_prepared_flag(prepared_inputs, monkeypatch):
    def fail(*args):
        raise ValueError("PRIVATE_FINALIZATION_FAILURE")
    monkeypatch.setattr(m, "_PreparedPair", fail)
    result = m.prepare_first_forward_intent(*prepared_inputs[0])
    summary = result.to_safe_summary_dict()
    assert result.status == "PRECONDITION_BLOCKED" and result.pair is None
    assert summary["intent_pair_prepared"] is False
    assert summary["input_digest"] is summary["preparation_digest"] is None
    assert all(role["status"] == "COMPLETE" for role in summary["roles"])


@pytest.mark.parametrize("anchor", ["2026-08-28", "2026-08-14", "2026-08-07"])
def test_cycle_joins_block_before_planning(prepared_inputs, anchor):
    args = list(prepared_inputs[0])
    args[4] = anchor
    result = m.prepare_first_forward_intent(*args)
    summary = result.to_safe_summary_dict()
    assert result.status == "PRECONDITION_BLOCKED"
    assert summary["reason_codes"] == ["CYCLE_JOIN_INVALID"]
    assert all(role["status"] == "NOT_RUN" for role in summary["roles"])


def test_bootstrap_state_must_match_observed_source(prepared_inputs, monkeypatch):
    # Retain all selector/hash joins, then corrupt only the freshly checked
    # source economics. The C3 join must independently reject that mismatch.
    import quantpits.research.forward_bootstrap as bootstrap_module
    original = bootstrap_module._build_states
    def different(source, candidate):
        champion, challenger, assumption = original(source, candidate)
        changed = []
        for state in (champion, challenger):
            raw = state.to_dict()
            raw.pop("digest")
            raw["cash"] = "100.00"
            changed.append(m.ShadowPortfolioState.from_dict(raw))
        return changed[0], changed[1], assumption
    monkeypatch.setattr(bootstrap_module, "_build_states", different)
    summary = m.prepare_first_forward_intent(*prepared_inputs[0]).to_safe_summary_dict()
    assert summary["status"] == "PRECONDITION_BLOCKED"
    assert summary["reason_codes"] == ["BOOTSTRAP_SOURCE_ECONOMICS_INVALID"]
    assert all(role["status"] == "NOT_RUN" for role in summary["roles"])



def test_copy_branch_preparation_digest_uses_content_inventory(prepared_inputs, copied_models, monkeypatch):
    import os
    from quantpits.research import model_continuity as continuity
    args = prepared_inputs[0]
    root, (reference, current) = copied_models
    original = continuity.observe_model_copy_pair
    original_projection = surface._source_projection
    def projection(value):
        if value is prepared_inputs[2]:
            return {}
        if 'model_and_ensemble_lineage' not in value:
            return {'fixture_reference': True}
        return original_projection(value)
    monkeypatch.setattr(surface, '_source_projection', projection)
    original_cycle = surface._cycle_authority
    def cycle_authority(workspace, date):
        if date == '2026-08-28':
            return workspace / 'data/evidence/v1/cycles' / date, {'fixture_reference': True}, {}
        return original_cycle(workspace, date)
    monkeypatch.setattr(surface, '_cycle_authority', cycle_authority)
    monkeypatch.setattr(surface, '_source_matches_definition', lambda projection, candidate: bool(projection))
    def copied_pair(*unused, **kwargs):
        return original(root, reference, root, current, **kwargs)
    monkeypatch.setattr(continuity, 'observe_model_copy_pair', copied_pair)
    first = m.prepare_first_forward_intent(*args).to_safe_summary_dict()
    assert first['status'] == 'PREPARED', first['reason_codes']
    tag = root / 'mlruns/1/source_0_b/tags/model'
    info = tag.stat()
    os.utime(tag, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
    second = m.prepare_first_forward_intent(*args).to_safe_summary_dict()
    assert second['status'] == 'PREPARED', second
    assert first['input_digest'] == second['input_digest']
    assert first['preparation_digest'] == second['preparation_digest']
    (tag.parent / 'audit_note').write_text('new bytes')
    third = m.prepare_first_forward_intent(*args).to_safe_summary_dict()
    assert third['status'] == 'PREPARED', third
    assert first['input_digest'] != third['input_digest']
    assert first['preparation_digest'] != third['preparation_digest']
