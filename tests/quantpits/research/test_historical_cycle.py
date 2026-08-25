from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.evidence.ranking import canonical_full_ranking
from quantpits.research.historical_cycle import (
    ARM_IDS,
    CashPriceReceipt,
    HistoricalCycleContractError,
    HistoricalCycleInputError,
    HistoricalCycleResult,
    HistoricalShadowCycleReplay,
    PublicationReceipt,
    QlibCashPriceObserver,
    load_replay_profile,
    rounding_adjustment_within_bound,
    write_historical_cycle_output,
)
from quantpits.research.intents import CurrentRuleShadowIntentPlanner
from quantpits.research.accounting import ShadowPortfolioState, ShadowPortfolioTransition
import quantpits.research.historical_cycle as historical_cycle_module
from quantpits.research.replay import ResearchRankingReplay, load_sealed_replay_inputs
from quantpits.scripts.research_shadow_cycle import main as shadow_cycle_main
from quantpits.utils.predict_utils import rank_norm


DATES = (
    "2026-07-03", "2026-07-10", "2026-07-17", "2026-07-24",
    "2026-07-31", "2026-08-07", "2026-08-14", "2026-08-21",
)
MODELS = ("MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D")
INSTRUMENTS = ("SH000001", "SZ000002", "SH000003", "SZ000004")
ANCHOR = DATES[-2]


def digest(data):
    return {
        "algorithm": "sha256", "domain": "raw_bytes",
        "value": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }


def definition_raw(**changes):
    value = {
        "schema_version": 1, "definition_id": "b2-current-rule-v1",
        "strategy_name": "topk_dropout", "topk": 2, "n_drop": 1,
        "buy_suggestion_factor": 2, "sell_out_of_universe": True,
        "production_buy_lot_size": 100,
        "cashflow_mode": "ZERO_EXTERNAL_CASHFLOW",
        "planning_price_rule": "CASH_CLOSE_WITH_0_9_1_1_ESTIMATES_V1",
        "buy_selection_rule": "FIRST_N_GENERATED_SUGGESTIONS_V1",
        "production_primitive_id": "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1",
    }
    value.update(changes)
    return value


def assumption_raw(**changes):
    value = {
        "schema_version": 1, "assumption_id": "b2-friction-v1", "deal_price": "NEXT_OPEN",
        "sell_before_buy": True, "partial_fill": False, "short_sell": False,
        "lot_size": 100, "money_quantum": "0.01", "price_quantum": "0.01",
        "rounding_mode": "HALF_UP", "buy_slippage_rate": "0.001",
        "sell_slippage_rate": "0.001", "buy_fee_rate": "0.0003",
        "sell_fee_rate": "0.0003", "minimum_buy_fee": "0",
        "minimum_sell_fee": "0", "corporate_action_mode": "UNMODELED",
    }
    value.update(changes)
    return value


def profile_raw(**changes):
    value = {
        "schema_version": 1, "profile_id": "synthetic-b2-profile",
        "research_protocol_id": "RETROSPECTIVE_SINGLE_CYCLE_V1",
        "bootstrap": {"as_of_date": DATES[-3], "cash": "10000", "positions": []},
        "intent_definition": definition_raw(), "execution_assumption": assumption_raw(),
    }
    value.update(changes)
    return value


def write_profile(tmp_path, value=None, raw=None):
    path = tmp_path / "private-profile.json"
    path.write_bytes(raw if raw is not None else canonical_json_bytes(value or profile_raw()))
    return path


def prediction(model_position):
    rows, values = [], []
    for date_position, date in enumerate(DATES):
        for instrument_position, instrument in enumerate(INSTRUMENTS):
            rows.append((pd.Timestamp(date), instrument))
            values.append(float((instrument_position + 1) * (model_position + 2) + date_position % 3))
    return pd.DataFrame(
        {"score": values},
        index=pd.MultiIndex.from_tuples(rows, names=["datetime", "instrument"]),
    )


def sealed_champion(frames):
    normalized = []
    for frame in frames:
        selected = frame.xs(pd.Timestamp(DATES[-1]), level="datetime")["score"]
        selected.index = pd.MultiIndex.from_arrays(
            [[pd.Timestamp(DATES[-1])] * len(selected), selected.index],
            names=["datetime", "instrument"],
        )
        normalized.append(rank_norm(selected))
    score = sum(normalized) / len(normalized)
    return canonical_full_ranking(
        INSTRUMENTS, {instrument: float(value) for (_date, instrument), value in score.items()},
    )


def fixture(tmp_path):
    workspace, qlib = tmp_path / "workspace", tmp_path / "qlib"
    cycle = workspace / "data/evidence/v1/cycles" / DATES[-1]
    cycle.mkdir(parents=True)
    (qlib / "instruments").mkdir(parents=True)
    (qlib / "calendars").mkdir()
    universe = ("\n".join("%s\t2026-01-01\t2026-12-31" % item for item in INSTRUMENTS) + "\n").encode()
    calendar = ("\n".join(DATES) + "\n").encode()
    (qlib / "instruments/fixture.txt").write_bytes(universe)
    (qlib / "calendars/day.txt").write_bytes(calendar)
    values = [10.0 + index for index in range(len(DATES))]
    prices = struct.pack("<%sf" % (len(values) + 1), 0.0, *values)
    factors = struct.pack("<%sf" % (len(values) + 1), 0.0, *([1.0] * len(values)))
    for instrument in INSTRUMENTS:
        root = qlib / "features" / instrument.lower()
        root.mkdir(parents=True)
        (root / "close.day.bin").write_bytes(prices)
        (root / "open.day.bin").write_bytes(prices)
        (root / "factor.day.bin").write_bytes(factors)
    frames, source_models, source_artifacts = [], [], []
    for index, model in enumerate(MODELS):
        frame = prediction(index)
        frames.append(frame)
        artifact = workspace / "mlruns" / str(index) / "artifacts"
        artifact.mkdir(parents=True)
        path = artifact / "pred.pkl"
        frame.to_pickle(path)
        data = path.read_bytes()
        recorder = "recorder-%d" % index
        source_models.append({
            "resolved_key": model, "status": "ready", "recorder_id": recorder,
            "experiment_name": "fixture-experiment", "artifact_path": artifact.relative_to(workspace).as_posix(),
        })
        source_artifacts.append({
            "recorder_id": recorder,
            "members": [{"path": path.relative_to(workspace).as_posix(), "digest": digest(data)}],
        })
    ranking = sealed_champion(frames)
    ranking_bytes = ranking.to_csv_bytes()
    (cycle / "ranking.csv").write_bytes(ranking_bytes)
    manifest = {
        "cycle_identity": {"cycle_id": DATES[-1]},
        "data_identity": {
            "source_to_materialization_relation": "unverified",
            "qlib_materialization_identity": {
                "status": "observed", "calendar_cutoff": DATES[-1], "universe_name": "fixture",
                "universe_digest": digest(universe), "calendar_digest": digest(calendar),
            },
        },
        "model_and_ensemble_lineage": {
            "status": "complete", "source_models": source_models,
            "source_artifacts": source_artifacts + [{"recorder_id": "ensemble", "position": "ensemble", "members": []}],
            "combo": {"resolved_members": list(MODELS)},
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (cycle / "manifest.json").write_bytes(manifest_bytes)
    (cycle / "seal.json").write_bytes(canonical_json_bytes({
        "cycle_id": DATES[-1], "manifest_digest": digest(manifest_bytes),
        "named_file_digests": {"ranking.csv": digest(ranking_bytes)},
    }))
    return workspace, qlib


def build_stage(tmp_path):
    workspace, qlib = fixture(tmp_path)
    inputs = load_sealed_replay_inputs(
        workspace, DATES[-1], qlib_data_dir=qlib,
        coverage_start=DATES[0], coverage_end=DATES[-1],
    )
    stage_a = ResearchRankingReplay(inputs, top_k=2).run(
        preferred_start=DATES[0], preferred_end=DATES[-1], window_size=6,
    )
    return stage_a, inputs, workspace, qlib


def build_runner(tmp_path, profile=None):
    stage_a, inputs, workspace, qlib = build_stage(tmp_path)
    profile = load_replay_profile(write_profile(tmp_path, profile))
    runner = HistoricalShadowCycleReplay(
        stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
        provider_root=qlib, anchor_date=ANCHOR, market="csi300",
    )
    return runner, workspace, qlib


def test_optional_prior_states_none_preserves_exact_b2_terminal_payload(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    expected = runner.run()
    close = runner._observer.observe_anchor_close(
        anchor_date=ANCHOR,
        requested_instruments=runner.requested_anchor_instruments_for(None),
    )
    prepared = runner.prepare_arms(close, prior_states=None)
    receipts = {}
    for item in prepared.terminal_arms:
        requested = tuple(sorted(
            {position.instrument for position in item["prior_state"].positions}
            | {intent.instrument for intent in item["intent_planning_result"].intents.intents}
        ))
        receipts[item["arm_id"]] = runner._observer.observe_next_open(
            trade_date=runner._trade_date, requested_instruments=requested,
        )
    actual = runner.settle_arms(prepared, receipts)
    assert [runner._arm_payload(item) for item in actual] == expected["terminal_arms"]
    assert [dict(runner._comparison(actual[0], item)) for item in actual[1:]] == expected["comparisons"]


def test_optional_prior_states_use_current_holdings_for_close_request(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    states = {}
    for arm in ARM_IDS:
        base = runner._state(arm)
        states[arm] = ShadowPortfolioState.from_dict({
            "portfolio_id": base.portfolio_id, "as_of_date": base.as_of_date,
            "cash": "10000",
            "positions": [{"instrument": "SH999999", "quantity": 100, "book_cost": "1000"}],
        })
    requested = runner.requested_anchor_instruments_for(states)
    assert "SH999999" in requested
    assert requested == tuple(sorted({
        row["instrument"] for ranking in runner._rankings.values()
        for row in ranking.rows if row["scored"]
    } | {"SH999999"}))


def test_optional_prior_state_set_rejects_missing_extra_reordered_foreign_and_duplicate_portfolios(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    states = {arm: runner._state(arm) for arm in ARM_IDS}
    invalid = [
        {arm: states[arm] for arm in ARM_IDS[:-1]},
        {**states, "FOREIGN": states[ARM_IDS[0]]},
        {arm: states[arm] for arm in reversed(ARM_IDS)},
        {arm: states[ARM_IDS[0]] if arm == ARM_IDS[1] else states[arm] for arm in ARM_IDS},
    ]
    for value in invalid:
        with pytest.raises(HistoricalCycleContractError, match="prior[_ ]state"):
            runner.requested_anchor_instruments_for(value)


def test_strict_profile_preserves_private_path_and_exact_digests(tmp_path):
    path = write_profile(tmp_path)
    receipt = load_replay_profile(path)
    public = receipt.public_payload()
    assert public["profile_id"] == "synthetic-b2-profile"
    assert str(path) not in json.dumps(public)
    assert public["raw_digest"] == digest(path.read_bytes())
    assert receipt.definition.digest == public["definition"]["digest"]
    assert receipt.assumption.digest == public["execution_assumption"]["digest"]


@pytest.mark.parametrize("raw", [b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}', b'[]'])
def test_profile_duplicate_nonfinite_and_nonobject_are_rejected(tmp_path, raw):
    with pytest.raises((HistoricalCycleInputError, HistoricalCycleContractError)):
        load_replay_profile(write_profile(tmp_path, raw=raw))


def test_profile_forbidden_extra_and_lot_mismatch_are_denied(tmp_path):
    value = profile_raw(real_account_id="secret")
    with pytest.raises(HistoricalCycleContractError, match="extra"):
        load_replay_profile(write_profile(tmp_path, value))
    mismatch = profile_raw(execution_assumption=assumption_raw(lot_size=200))
    with pytest.raises(HistoricalCycleContractError, match="lot sizes"):
        build_runner(tmp_path / "mismatch", mismatch)


def test_profile_and_provider_symlink_aliases_are_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    profile = write_profile(real)
    alias = tmp_path / "profile-alias.json"
    alias.symlink_to(profile)
    with pytest.raises(HistoricalCycleInputError, match="physical path"):
        load_replay_profile(alias)

    _workspace, qlib = fixture(tmp_path / "provider")
    provider_alias = tmp_path / "provider-alias"
    provider_alias.symlink_to(qlib, target_is_directory=True)
    with pytest.raises(HistoricalCycleInputError, match="provider root"):
        QlibCashPriceObserver(provider_alias, DATES)


def test_price_observer_derives_exact_close_and_open_from_distinct_receipts(tmp_path):
    _workspace, qlib = fixture(tmp_path)
    observer = QlibCashPriceObserver(qlib, DATES)
    requested = tuple(sorted(INSTRUMENTS))
    close = observer.observe_anchor_close(anchor_date=ANCHOR, requested_instruments=requested)
    opened = observer.observe_next_open(trade_date=DATES[-1], requested_instruments=requested)
    assert close.observation_kind == "CASH_CLOSE"
    assert opened.observation_kind == "NEXT_OPEN"
    assert all(row["numerator"]["field"] == "close" for row in close.rows)
    assert all(row["numerator"]["field"] == "open" for row in opened.rows)
    assert all(row["denominator"]["field"] == "factor" for row in close.rows + opened.rows)
    assert close.to_anchor_snapshot().anchor_date == ANCHOR
    with pytest.raises(HistoricalCycleContractError):
        opened.to_anchor_snapshot()


@pytest.mark.parametrize(
    "mutation,expected",
    [("missing_factor", "MISSING"), ("zero_factor", "INVALID"), ("malformed_open", "INVALID")],
)
def test_price_observer_retains_missing_and_invalid_members(tmp_path, mutation, expected):
    _workspace, qlib = fixture(tmp_path)
    target = qlib / "features/sh000001"
    if mutation == "missing_factor":
        (target / "factor.day.bin").unlink()
        field, date = "close", ANCHOR
    elif mutation == "zero_factor":
        values = [1.0] * len(DATES)
        values[DATES.index(ANCHOR)] = 0.0
        (target / "factor.day.bin").write_bytes(struct.pack("<%sf" % (len(values) + 1), 0.0, *values))
        field, date = "close", ANCHOR
    else:
        (target / "open.day.bin").write_bytes(b"bad")
        field, date = "open", DATES[-1]
    observer = QlibCashPriceObserver(qlib, DATES)
    receipt = (
        observer.observe_anchor_close(anchor_date=date, requested_instruments=tuple(sorted(INSTRUMENTS)))
        if field == "close" else observer.observe_next_open(trade_date=date, requested_instruments=tuple(sorted(INSTRUMENTS)))
    )
    assert len(receipt.rows) == len(INSTRUMENTS)
    assert receipt.rows[0]["status"] == expected
    assert tuple(row["instrument"] for row in receipt.rows) == tuple(sorted(INSTRUMENTS))


@pytest.mark.parametrize("mutation,expected_status,expected_reason", [
    ("missing_numerator", "MISSING", "NUMERATOR_OR_FACTOR_MISSING"),
    ("negative_factor", "INVALID", "INVALID_FACTOR_OR_DERIVED_PRICE"),
    ("nonfinite_numerator", "INVALID", "NUMERATOR_OR_FACTOR_INVALID"),
    ("position_before_start", "MISSING", "NUMERATOR_OR_FACTOR_MISSING"),
    ("missing_tail", "MISSING", "NUMERATOR_OR_FACTOR_MISSING"),
])
def test_price_word_boundary_failures_remain_explicit(tmp_path, mutation, expected_status, expected_reason):
    _workspace, qlib = fixture(tmp_path)
    root = qlib / "features/sh000001"
    if mutation == "missing_numerator":
        (root / "close.day.bin").unlink()
    elif mutation == "negative_factor":
        values = [1.0] * len(DATES)
        values[DATES.index(ANCHOR)] = -1.0
        (root / "factor.day.bin").write_bytes(struct.pack("<%sf" % (len(values) + 1), 0.0, *values))
    elif mutation == "nonfinite_numerator":
        values = [10.0] * len(DATES)
        values[DATES.index(ANCHOR)] = float("nan")
        (root / "close.day.bin").write_bytes(struct.pack("<%sf" % (len(values) + 1), 0.0, *values))
    elif mutation == "position_before_start":
        (root / "close.day.bin").write_bytes(struct.pack("<2f", 99.0, 10.0))
    else:
        (root / "close.day.bin").write_bytes(struct.pack("<2f", 0.0, 10.0))
    receipt = QlibCashPriceObserver(qlib, DATES).observe_anchor_close(
        anchor_date=ANCHOR, requested_instruments=tuple(sorted(INSTRUMENTS)),
    )
    row = next(item for item in receipt.rows if item["instrument"] == "SH000001")
    assert (row["status"], row["reason_code"]) == (expected_status, expected_reason)


@pytest.mark.parametrize("requested", [(), ("SH000001", "SH000001")])
def test_price_requested_set_zero_and_duplicate_are_rejected(tmp_path, requested):
    _workspace, qlib = fixture(tmp_path)
    observer = QlibCashPriceObserver(qlib, DATES)
    with pytest.raises(HistoricalCycleContractError, match="requested instruments"):
        observer.observe_anchor_close(anchor_date=ANCHOR, requested_instruments=requested)
    with pytest.raises(HistoricalCycleContractError, match="absent from calendar"):
        observer.observe_anchor_close(
            anchor_date="2026-08-15", requested_instruments=("SH000001",),
        )


def test_symlink_and_hardlink_price_sources_are_fail_closed(tmp_path):
    _workspace, qlib = fixture(tmp_path / "symlink")
    path = qlib / "features/sh000001/close.day.bin"
    moved = path.with_name("real.bin")
    path.rename(moved)
    path.symlink_to(moved.name)
    observer = QlibCashPriceObserver(qlib, DATES)
    with pytest.raises(HistoricalCycleInputError, match="symlink"):
        observer.observe_anchor_close(anchor_date=ANCHOR, requested_instruments=tuple(sorted(INSTRUMENTS)))

    _workspace, qlib = fixture(tmp_path / "hardlink")
    path = qlib / "features/sh000001/close.day.bin"
    os.link(str(path), str(path.with_name("alias.bin")))
    observer = QlibCashPriceObserver(qlib, DATES)
    with pytest.raises(HistoricalCycleInputError, match="exclusive"):
        observer.observe_anchor_close(anchor_date=ANCHOR, requested_instruments=tuple(sorted(INSTRUMENTS)))


def test_price_source_and_calendar_identity_cannot_drift_between_observations(tmp_path):
    _workspace, qlib = fixture(tmp_path)
    observer = QlibCashPriceObserver(qlib, DATES)
    requested = tuple(sorted(INSTRUMENTS))
    observer.observe_anchor_close(anchor_date=ANCHOR, requested_instruments=requested)
    path = qlib / "features/sh000001/factor.day.bin"
    path.write_bytes(path.read_bytes() + struct.pack("<f", 1.0))
    with pytest.raises(HistoricalCycleInputError, match="within logical replay"):
        observer.observe_next_open(trade_date=DATES[-1], requested_instruments=requested)

    _workspace, qlib = fixture(tmp_path / "calendar")
    observer = QlibCashPriceObserver(qlib, DATES)
    calendar = qlib / "calendars/day.txt"
    calendar.write_bytes(calendar.read_bytes() + b"\n")
    with pytest.raises(HistoricalCycleInputError, match="calendar identity changed"):
        observer.observe_anchor_close(anchor_date=ANCHOR, requested_instruments=requested)

def test_exact_five_arm_run_is_complete_independent_and_deterministic(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    first = runner.run()
    second = runner.run()
    assert first.to_canonical_json_bytes() == second.to_canonical_json_bytes()
    assert first["status"] == "COMPLETE"
    assert tuple(first["requested_arm_ids"]) == ARM_IDS
    assert first["arm_counts"] == {"requested": 5, "terminal": 5, "complete": 5, "blocked": 0}
    assert len(first["comparisons"]) == 4
    portfolios = [item["prior_state"]["portfolio_id"] for item in first["terminal_arms"]]
    assert len(set(portfolios)) == 5
    assert all(item["terminal_status"] == "COMPLETE" for item in first["terminal_arms"])
    assert first["prospective_claim"] is False and first["promotion_capability"] is False
    assert first["source_to_materialization_relation"] == "unverified"


@pytest.mark.parametrize("mutation", ["missing", "renamed", "extra", "reordered"])
def test_stage_a_arm_set_must_remain_exact_and_ordered(tmp_path, mutation):
    stage_a, inputs, _workspace, qlib = build_stage(tmp_path)
    arms = stage_a._payload["rankings"][ANCHOR]
    if mutation == "missing":
        arms.pop("DROP_2_3")
    elif mutation == "renamed":
        arms["FOREIGN"] = arms.pop("DROP_2_3")
    elif mutation == "extra":
        arms["EXTRA"] = arms["DROP_2_3"]
    else:
        stage_a._payload["rankings"][ANCHOR] = {key: arms[key] for key in reversed(tuple(arms))}
    profile = load_replay_profile(write_profile(tmp_path))
    with pytest.raises(
        HistoricalCycleContractError, match="result-digest revalidation|five arms",
    ):
        HistoricalShadowCycleReplay(
            stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
            provider_root=qlib, anchor_date=ANCHOR, market="csi300",
        )


@pytest.mark.parametrize("filled", [0, 1, 7])
@pytest.mark.parametrize("sign", [Decimal("1"), Decimal("-1")])
def test_rounding_adjustment_bound_accepts_equality_and_rejects_one_unit_beyond(filled, sign):
    quantum = Decimal("0.01")
    bound = Decimal(filled + 3) * quantum / Decimal(2)
    assert rounding_adjustment_within_bound(sign * bound, filled, quantum) is True
    assert rounding_adjustment_within_bound(sign * (bound + Decimal("0.0000001")), filled, quantum) is False


def test_nonpositive_sell_proceeds_blocks_all_affected_arms(tmp_path):
    value = profile_raw(
        bootstrap={
            "as_of_date": DATES[-3], "cash": "10000",
            "positions": [{"instrument": "SH999999", "quantity": 100, "book_cost": "1000"}],
        },
        execution_assumption=assumption_raw(minimum_sell_fee="1698"),
    )
    runner, _workspace, qlib = build_runner(tmp_path, value)
    values = [10.0 + index for index in range(len(DATES))]
    prices = struct.pack("<%sf" % (len(values) + 1), 0.0, *values)
    factors = struct.pack("<%sf" % (len(values) + 1), 0.0, *([1.0] * len(values)))
    root = qlib / "features/sh999999"
    root.mkdir(parents=True)
    (root / "close.day.bin").write_bytes(prices)
    (root / "open.day.bin").write_bytes(prices)
    (root / "factor.day.bin").write_bytes(factors)
    result = runner.run()
    assert result["status"] == "BLOCKED"
    affected = [item for item in result["terminal_arms"] if any(
        row["side"] == "SELL" and row["status"] == "FILLED"
        for row in (item["transition_result"] or {"terminal_results": []})["terminal_results"]
    )]
    assert affected
    assert all(item["terminal_status"] == "BLOCKED_EXECUTION_PROFILE_NONPOSITIVE_SELL_PROCEEDS" for item in affected)


def test_one_ordinary_planning_failure_preserves_later_arm_order(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path)
    close = runner._observer.observe_anchor_close(
        anchor_date=ANCHOR, requested_instruments=runner.requested_anchor_instruments,
    )
    original = CurrentRuleShadowIntentPlanner.plan
    calls = []

    def injected(**kwargs):
        calls.append(kwargs["cycle_id"])
        if len(calls) == 2:
            raise RuntimeError("injected ordinary failure")
        return original(**kwargs)

    monkeypatch.setattr(CurrentRuleShadowIntentPlanner, "plan", injected)
    prepared = runner.prepare_arms(close)
    assert prepared.status == "BLOCKED"
    assert tuple(item["arm_id"] for item in prepared.terminal_arms) == ARM_IDS
    assert prepared.terminal_arms[1]["terminal_status"] == "BLOCKED_PLANNING"
    assert len(calls) == 5


def test_process_control_from_arm_planning_propagates(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path)
    close = runner._observer.observe_anchor_close(
        anchor_date=ANCHOR, requested_instruments=runner.requested_anchor_instruments,
    )
    monkeypatch.setattr(CurrentRuleShadowIntentPlanner, "plan", lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner.prepare_arms(close)


def test_one_ordinary_settlement_failure_preserves_later_arm_order(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path)
    close = runner._observer.observe_anchor_close(
        anchor_date=ANCHOR, requested_instruments=runner.requested_anchor_instruments,
    )
    prepared = runner.prepare_arms(close)
    receipts = {}
    for item in prepared.terminal_arms:
        requested = tuple(sorted(
            {position.instrument for position in item["prior_state"].positions}
            | {intent.instrument for intent in item["intent_planning_result"].intents.intents}
        ))
        receipts[item["arm_id"]] = runner._observer.observe_next_open(
            trade_date=DATES[-1], requested_instruments=requested,
        )
    original = ShadowPortfolioTransition.apply
    calls = []

    def injected(**kwargs):
        calls.append(kwargs["intents"].cycle_id)
        if len(calls) == 2:
            raise RuntimeError("injected settlement failure")
        return original(**kwargs)

    monkeypatch.setattr(ShadowPortfolioTransition, "apply", injected)
    arms = runner.settle_arms(prepared, receipts)
    assert tuple(item["arm_id"] for item in arms) == ARM_IDS
    assert arms[1]["terminal_status"] == "BLOCKED_SETTLEMENT"
    assert len(calls) == 5


def test_process_control_from_settlement_propagates(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path)
    close = runner._observer.observe_anchor_close(
        anchor_date=ANCHOR, requested_instruments=runner.requested_anchor_instruments,
    )
    prepared = runner.prepare_arms(close)
    receipts = {}
    for item in prepared.terminal_arms:
        requested = tuple(sorted(
            {position.instrument for position in item["prior_state"].positions}
            | {intent.instrument for intent in item["intent_planning_result"].intents.intents}
        ))
        receipts[item["arm_id"]] = runner._observer.observe_next_open(
            trade_date=DATES[-1], requested_instruments=requested,
        )
    monkeypatch.setattr(ShadowPortfolioTransition, "apply", lambda **_kwargs: (_ for _ in ()).throw(SystemExit(7)))
    with pytest.raises(SystemExit):
        runner.settle_arms(prepared, receipts)


def test_tampered_anchor_receipt_is_denied_before_planning(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    close = runner._observer.observe_anchor_close(
        anchor_date=ANCHOR, requested_instruments=runner.requested_anchor_instruments,
    )
    object.__setattr__(close, "observed_instruments", ())
    with pytest.raises(HistoricalCycleContractError, match="partitions"):
        runner.prepare_arms(close)


def test_prepare_signature_accepts_only_anchor_receipt_and_future_receipt_is_denied(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    close = runner._observer.observe_anchor_close(
        anchor_date=ANCHOR, requested_instruments=runner.requested_anchor_instruments,
    )
    opened = runner._observer.observe_next_open(
        trade_date=DATES[-1], requested_instruments=runner.requested_anchor_instruments,
    )
    assert runner.prepare_arms(close).status == "COMPLETE"
    with pytest.raises(HistoricalCycleContractError, match="anchor-close"):
        runner.prepare_arms(opened)


def test_missing_next_open_is_visible_no_fill_and_partial_nav_without_blocking(tmp_path):
    value = profile_raw(bootstrap={
        "as_of_date": DATES[-3], "cash": "10000",
        "positions": [{"instrument": "SZ000004", "quantity": 100, "book_cost": "1000"}],
    })
    runner, _workspace, qlib = build_runner(tmp_path, value)
    (qlib / "features/sz000004/open.day.bin").unlink()
    result = runner.run()
    assert result["status"] == "COMPLETE"
    champion = result["terminal_arms"][0]
    assert "SZ000004" in champion["next_open_receipt"]["missing_instruments"]
    assert champion["transition_result"]["valuation_status"] == "PARTIAL"
    assert result["comparisons"][0]["valuation_comparison_status"] == "NOT_COMPARABLE_MISSING_VALUATION"


def test_profile_drift_is_detected_before_replay(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    path = runner._profile._private_path
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(HistoricalCycleInputError, match="identity changed"):
        runner.run()


def test_public_result_construction_and_existing_output_are_fail_closed(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path)
    result = runner.run()
    with pytest.raises(HistoricalCycleContractError):
        HistoricalCycleResult(result.to_dict())
    impossible = result.to_dict()
    impossible["status"] = "BLOCKED"
    impossible["result_digest"] = historical_cycle_module._digest_payload({
        key: value for key, value in impossible.items() if key != "result_digest"
    })
    with pytest.raises(HistoricalCycleContractError, match="capability"):
        HistoricalCycleResult(impossible, historical_cycle_module._AUTHORITY)
    with pytest.raises(HistoricalCycleContractError):
        PublicationReceipt(
            operation_id="forged", status="COMMITTED", did_write=True,
            result_digest=result["result_digest"], manifest_digest=result["result_digest"],
            member_count=1, output_root_identity=(1, 2, 3, 0, 0),
            root_parent_identity_before=(1, 2, 3, 0, 0),
            root_parent_identity_after=(1, 2, 3, 0, 0),
        )
    output = Path("/tmp") / ("quantpits_b2_conflict_%s" % tmp_path.name)
    output.mkdir()
    try:
        receipt = write_historical_cycle_output(runner, result, output)
        assert receipt.status == "CONFLICT" and receipt.did_write is False
    finally:
        output.rmdir()


def test_stage_a_current_ranking_tamper_is_denied_before_runner_capability(tmp_path):
    stage_a, inputs, _workspace, qlib = build_stage(tmp_path / "source")
    profile = load_replay_profile(write_profile(tmp_path, profile_raw()))
    carried = stage_a._trusted_payload()
    current = carried["rankings"][ANCHOR]["DROP_1_3"]
    changed_scores = {
        row["instrument"]: float(row["raw_score"]) + (
            0.25 if index == 0 else 0.0
        )
        for index, row in enumerate(current.rows) if row["scored"]
    }
    carried["rankings"][ANCHOR]["DROP_1_3"] = canonical_full_ranking(
        tuple(row["instrument"] for row in current.rows), changed_scores,
    )
    output = Path("/tmp") / ("quantpits_b2_stage_a_tamper_%s" % tmp_path.name)
    with pytest.raises(HistoricalCycleContractError, match="result-digest revalidation"):
        HistoricalShadowCycleReplay(
            stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
            provider_root=qlib, anchor_date=ANCHOR, market="csi300",
        )
    assert not output.exists()


def test_publication_is_manifest_last_exact_and_recomputed(tmp_path):
    runner, workspace, qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_success_%s" % tmp_path.name)
    workspace_before = sorted((path.relative_to(workspace).as_posix(), path.stat().st_size) for path in workspace.rglob("*") if path.is_file())
    qlib_before = sorted((path.relative_to(qlib).as_posix(), path.stat().st_size) for path in qlib.rglob("*") if path.is_file())
    try:
        receipt = write_historical_cycle_output(runner, result, output)
        assert receipt.status == "COMMITTED" and receipt.did_write is True
        manifest = json.loads((output / "output_manifest.json").read_text())
        assert manifest["member_count"] == 4 + 5 * 5
        assert len(manifest["members"]) == manifest["member_count"]
        for item in manifest["members"]:
            data = (output / item["logical_path"]).read_bytes()
            assert item["digest"] == digest(data)
        assert {path.name for path in output.iterdir()} == {"input_receipt.json", "result.json", "summary.csv", "report.md", "arms", "output_manifest.json"}
        assert {path.name for path in (output / "arms").iterdir()} == set(ARM_IDS)
        for arm in ARM_IDS:
            assert {path.name for path in (output / "arms" / arm).iterdir()} == {
                "ranking.csv", "intent_plan.json", "quotes.json",
                "settlement.json", "after_state.json",
            }
        assert workspace_before == sorted((path.relative_to(workspace).as_posix(), path.stat().st_size) for path in workspace.rglob("*") if path.is_file())
        assert qlib_before == sorted((path.relative_to(qlib).as_posix(), path.stat().st_size) for path in qlib.rglob("*") if path.is_file())
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_tampered_result_is_denied_before_output_root_creation(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    object.__setattr__(result, "_bytes", result.to_canonical_json_bytes() + b" ")
    output = Path("/tmp") / ("quantpits_b2_tamper_%s" % tmp_path.name)
    with pytest.raises(HistoricalCycleContractError, match="before write"):
        write_historical_cycle_output(runner, result, output)
    assert not output.exists()


def test_publication_rejects_symlink_parent_without_creating_target(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    real_parent = Path("/tmp") / ("quantpits_b2_real_%s" % tmp_path.name)
    alias = Path("/tmp") / ("quantpits_b2_alias_%s" % tmp_path.name)
    real_parent.mkdir()
    alias.symlink_to(real_parent, target_is_directory=True)
    try:
        with pytest.raises(HistoricalCycleContractError, match="direct child"):
            write_historical_cycle_output(runner, result, alias / "result")
        assert not (real_parent / "result").exists()
    finally:
        alias.unlink()
        real_parent.rmdir()


def test_publication_rejects_nested_parent_before_and_after_replacement(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    parent = Path("/tmp") / ("quantpits_b2_nested_%s" % tmp_path.name)
    displaced = parent.with_name(parent.name + "_displaced")
    target = parent.with_name(parent.name + "_target")
    parent.mkdir()
    target.mkdir()
    try:
        with pytest.raises(HistoricalCycleContractError, match="direct child"):
            write_historical_cycle_output(runner, result, parent / "result")
        assert not (parent / "result").exists()
        parent.rename(displaced)
        parent.symlink_to(target, target_is_directory=True)
        with pytest.raises(HistoricalCycleContractError, match="direct child"):
            write_historical_cycle_output(runner, result, parent / "result")
        assert not (target / "result").exists()
    finally:
        if parent.is_symlink():
            parent.unlink()
        shutil.rmtree(parent, ignore_errors=True)
        shutil.rmtree(displaced, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)


def test_publication_root_create_is_relative_to_observed_tmp_descriptor(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_dirfd_%s" % tmp_path.name)
    original = os.mkdir
    observed = []

    def recording(path, mode=0o777, *, dir_fd=None):
        if path == output.name:
            observed.append(dir_fd)
            assert dir_fd is not None
            assert historical_cycle_module._directory_descriptor_identity(dir_fd) == historical_cycle_module._root_identity(Path("/tmp"))
        return original(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", recording)
    try:
        receipt = write_historical_cycle_output(runner, result, output)
        assert receipt.status == "COMMITTED"
        assert len(observed) == 1
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_process_control_during_publication_propagates_and_manifest_is_absent(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_interrupt_%s" % tmp_path.name)
    original = os.fsync
    calls = []

    def interrupted(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(fd)

    monkeypatch.setattr(os, "fsync", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            write_historical_cycle_output(runner, result, output)
        assert output.exists()
        assert not (output / "output_manifest.json").exists()
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_canonical_root_replacement_yields_uncertain_not_committed(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_replace_%s" % tmp_path.name)
    displaced = output.with_name(output.name + "_displaced")
    original = historical_cycle_module._stable_read
    triggered = []

    def replacing(root, logical, **kwargs):
        if root == output and logical == "input_receipt.json" and not triggered:
            triggered.append(True)
            output.rename(displaced)
            output.mkdir()
        return original(root, logical, **kwargs)

    monkeypatch.setattr(historical_cycle_module, "_stable_read", replacing)
    try:
        receipt = write_historical_cycle_output(runner, result, output)
        assert receipt.status == "UNCERTAIN"
        assert receipt.did_write is True
    finally:
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(displaced, ignore_errors=True)


@pytest.mark.parametrize("foreign_kind", ["file", "directory", "symlink"])
def test_foreign_final_namespace_member_yields_uncertain(tmp_path, monkeypatch, foreign_kind):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_foreign_%s_%s" % (foreign_kind, tmp_path.name))
    original = historical_cycle_module._stable_read
    injected = []

    def injecting(root, logical, **kwargs):
        data = original(root, logical, **kwargs)
        if root == output and logical == "output_manifest.json" and not injected:
            injected.append(True)
            foreign = output / "foreign"
            if foreign_kind == "file":
                foreign.write_bytes(b"foreign")
            elif foreign_kind == "directory":
                foreign.mkdir()
            else:
                foreign.symlink_to(output / "result.json")
        return data

    monkeypatch.setattr(historical_cycle_module, "_stable_read", injecting)
    try:
        receipt = write_historical_cycle_output(runner, result, output)
        assert receipt.status == "UNCERTAIN" and receipt.did_write is True
        assert (output / "output_manifest.json").is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_member_move_away_and_back_during_final_read_yields_uncertain(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_member_move_%s" % tmp_path.name)
    original = historical_cycle_module._stable_read
    calls = []

    def moving(root, logical, **kwargs):
        if root == output and logical == "result.json":
            calls.append(True)
            if len(calls) == 2:
                member = output / logical
                displaced = output / "result.displaced"
                member.rename(displaced)
                try:
                    return original(root, logical, **kwargs)
                finally:
                    displaced.rename(member)
        return original(root, logical, **kwargs)

    monkeypatch.setattr(historical_cycle_module, "_stable_read", moving)
    try:
        receipt = write_historical_cycle_output(runner, result, output)
        assert receipt.status == "UNCERTAIN" and receipt.did_write is True
        assert (output / "result.json").is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)


@pytest.mark.parametrize("mutation", ["missing", "symlink", "hardlink", "special"])
def test_invalid_expected_member_after_manifest_yields_uncertain(tmp_path, monkeypatch, mutation):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_member_%s_%s" % (mutation, tmp_path.name))
    original = historical_cycle_module._stable_read
    injected = []

    def mutating(root, logical, **kwargs):
        data = original(root, logical, **kwargs)
        if root == output and logical == "output_manifest.json" and not injected:
            injected.append(True)
            target = output / "result.json"
            target.unlink()
            if mutation == "symlink":
                target.symlink_to(output / "input_receipt.json")
            elif mutation == "hardlink":
                os.link(str(output / "input_receipt.json"), str(target))
            elif mutation == "special":
                os.mkfifo(str(target))
        return data

    monkeypatch.setattr(historical_cycle_module, "_stable_read", mutating)
    try:
        receipt = write_historical_cycle_output(runner, result, output)
        assert receipt.status == "UNCERTAIN" and receipt.did_write is True
        assert (output / "output_manifest.json").is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_process_control_after_manifest_write_propagates(tmp_path, monkeypatch):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path("/tmp") / ("quantpits_b2_manifest_interrupt_%s" % tmp_path.name)

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        historical_cycle_module, "_require_exact_directory_inventory", interrupted,
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            write_historical_cycle_output(runner, result, output)
        assert (output / "output_manifest.json").is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_output_outside_tmp_is_rejected_before_write(tmp_path):
    runner, _workspace, _qlib = build_runner(tmp_path / "source")
    result = runner.run()
    output = Path.cwd() / "forbidden-b2-output"
    with pytest.raises(HistoricalCycleContractError, match="direct child"):
        write_historical_cycle_output(runner, result, output)
    assert not output.exists()


def test_cli_executes_the_same_complete_bundle_contract(tmp_path, capsys):
    runner, workspace, qlib = build_runner(tmp_path / "source")
    output = Path("/tmp") / ("quantpits_b2_cli_%s" % tmp_path.name)
    try:
        code = shadow_cycle_main([
            "--workspace", str(workspace), "--qlib-data-dir", str(qlib),
            "--profile", str(runner._profile._private_path), "--sealed-cycle", DATES[-1],
            "--anchor", ANCHOR, "--preferred-start", DATES[0],
            "--preferred-end", DATES[-1], "--top-k", "2",
            "--output-dir", str(output),
        ])
        rendered = json.loads(capsys.readouterr().out)
        assert code == 0 and rendered["status"] == "complete"
        assert rendered["publication"]["status"] == "COMMITTED"
        assert (output / "output_manifest.json").is_file()
    finally:
        shutil.rmtree(output, ignore_errors=True)
