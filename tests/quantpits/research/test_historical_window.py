from __future__ import annotations

import json
import struct
from decimal import Decimal

import pytest

from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.research.historical_cycle import ARM_IDS, HistoricalShadowCycleReplay, load_replay_profile
from quantpits.research.historical_window import (
    HistoricalShadowWindowReplay,
    HistoricalWindowContractError,
    HistoricalWindowResult,
    SequentialPriorStateSet,
    compact_window_summary,
    _WINDOW_AUTHORITY,
)
from quantpits.research.historical_cycle import _digest_payload
from quantpits.research.replay import (
    ReplayResult, ResearchRankingReplay, load_sealed_replay_inputs,
    _RESULT_AUTHORITY, _canonical_digest, _copy_result_value, _public_result,
)
from quantpits.research.accounting import ShadowPortfolioTransition
from quantpits.research.intents import CurrentRuleShadowIntentPlanner
from tests.quantpits.research.test_historical_cycle import (
    DATES, digest, fixture, profile_raw, write_profile,
)


def build_window(tmp_path, window_size=4):
    workspace, qlib = fixture(tmp_path)
    sessions = (
        "2026-07-03", "2026-07-06", "2026-07-10", "2026-07-13",
        "2026-07-17", "2026-07-20", "2026-07-24", "2026-07-27",
        "2026-07-31", "2026-08-03", "2026-08-07", "2026-08-10",
        "2026-08-14", "2026-08-17", "2026-08-21", "2026-08-24",
    )
    calendar = ("\n".join(sessions) + "\n").encode()
    (qlib / "calendars/day.txt").write_bytes(calendar)
    values = [10.0 + index / 10.0 for index in range(len(sessions))]
    prices = struct.pack("<%sf" % (len(values) + 1), 0.0, *values)
    factors = struct.pack("<%sf" % (len(values) + 1), 0.0, *([1.0] * len(values)))
    for root in (qlib / "features").iterdir():
        (root / "close.day.bin").write_bytes(prices)
        (root / "open.day.bin").write_bytes(prices)
        (root / "factor.day.bin").write_bytes(factors)
    cycle = workspace / "data/evidence/v1/cycles" / DATES[-1]
    manifest_path = cycle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_identity"]["qlib_materialization_identity"]["calendar_digest"] = digest(calendar)
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    seal_path = cycle / "seal.json"
    seal = json.loads(seal_path.read_text())
    seal["manifest_digest"] = digest(manifest_bytes)
    seal_path.write_bytes(canonical_json_bytes(seal))
    inputs = load_sealed_replay_inputs(
        workspace, DATES[-1], qlib_data_dir=qlib,
        coverage_start=DATES[0], coverage_end=DATES[-1],
    )
    stage_a = ResearchRankingReplay(inputs, top_k=2).run(
        preferred_start=DATES[0], preferred_end=DATES[-1], window_size=window_size,
    )
    profile_value = profile_raw(bootstrap={"as_of_date": DATES[0], "cash": "10000", "positions": []})
    profile = load_replay_profile(write_profile(tmp_path, profile_value))
    runner = HistoricalShadowWindowReplay(
        stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
        provider_root=qlib, market="csi300", window_size=window_size,
    )
    return runner, stage_a, inputs, profile, workspace, qlib


def test_constructor_revalidates_stage_a_and_exact_selected_anchor_window(tmp_path):
    runner, stage_a, inputs, profile, _workspace, qlib = build_window(tmp_path, 4)
    assert runner._anchors == tuple(stage_a["inventory"]["selected_anchors"])
    with pytest.raises(HistoricalWindowContractError, match="window_size"):
        HistoricalShadowWindowReplay(
            stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
            provider_root=qlib, market="csi300", window_size=3,
        )
    with pytest.raises(HistoricalWindowContractError, match="requested window"):
        HistoricalShadowWindowReplay(
            stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
            provider_root=qlib, market="csi300", window_size=5,
        )


def test_two_cycle_after_state_is_exact_next_prior_for_all_five_arms(tmp_path):
    result = build_window(tmp_path, 4)[0].run()
    assert result["status"] == "COMPLETE"
    for previous, current in zip(result["terminal_cycles"], result["terminal_cycles"][1:]):
        previous_by_arm = {item["arm_id"]: item for item in previous["terminal_arms"]}
        for link, arm in zip(current["chain_links"], ARM_IDS):
            assert link["predecessor_after_state_digest"] == previous_by_arm[arm]["after_state"]["digest"]
            assert link["prior_state_digest"] == previous_by_arm[arm]["after_state"]["digest"]
            assert link["digest_equal_checked"] is True


@pytest.mark.parametrize("window_size", [4, 5, 6])
def test_four_five_six_cycle_grid_is_exact_and_deterministic(tmp_path, window_size):
    runner = build_window(tmp_path, window_size)[0]
    first = runner.run()
    second = runner.run()
    assert first.to_canonical_json_bytes() == second.to_canonical_json_bytes()
    assert first["counts"] == {
        "requested_cycles": window_size, "terminal_cycles": window_size,
        "requested_cycle_arms": window_size * 5, "terminal_cycle_arms": window_size * 5,
        "complete_cycle_arms": window_size * 5, "blocked_cycle_arms": 0,
    }


def test_first_cycle_terminal_and_comparison_payload_is_exact_b2_parity(tmp_path):
    window, stage_a, inputs, profile, _workspace, qlib = build_window(tmp_path, 4)
    result = window.run()
    anchor = result["requested_anchor_ids"][0]
    b2 = HistoricalShadowCycleReplay(
        stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
        provider_root=qlib, anchor_date=anchor, market="csi300",
    ).run()
    assert result["terminal_cycles"][0]["terminal_arms"] == b2["terminal_arms"]
    assert result["terminal_cycles"][0]["comparisons"] == b2["comparisons"]


def test_portfolio_ids_persist_and_cycle_ids_are_unique(tmp_path):
    result = build_window(tmp_path, 4)[0].run()
    for arm_index in range(5):
        rows = [cycle["terminal_arms"][arm_index] for cycle in result["terminal_cycles"]]
        assert len({item["prior_state"]["portfolio_id"] for item in rows}) == 1
        assert len({item["transition_result"]["cycle_id"] for item in rows}) == 4


def test_observer_trace_is_close_plan_open_settle_without_future_preload(tmp_path):
    result = build_window(tmp_path, 4)[0].run()
    assert [cycle["timeline"] for cycle in result["terminal_cycles"]] == [
        ["CLOSE", "PLAN", "OPEN", "SETTLE"]
    ] * 4
    assert result["future_price_order_checked"] is True


def test_common_observer_failure_materializes_exact_blocked_grid_without_private_message(tmp_path, monkeypatch):
    runner = build_window(tmp_path, 4)[0]
    calls = []
    original = runner._observer.observe_anchor_close

    def injected(**kwargs):
        calls.append(kwargs["anchor_date"])
        if len(calls) == 2:
            raise OSError("/private/account/path")
        return original(**kwargs)

    monkeypatch.setattr(runner._observer, "observe_anchor_close", injected)
    result = runner.run()
    assert result["status"] == "BLOCKED"
    assert result["counts"]["terminal_cycle_arms"] == 20
    assert result["counts"]["complete_cycle_arms"] == 5
    assert calls == list(result["requested_anchor_ids"][:2])
    assert "/private" not in result.to_canonical_json_bytes().decode()
    assert all(
        arm["terminal_status"] == "BLOCKED_WINDOW_PREDECESSOR_INCOMPLETE"
        for cycle in result["terminal_cycles"][2:] for arm in cycle["terminal_arms"]
    )


def test_common_source_failure_materializes_exact_grid_without_price_read_or_private_message(tmp_path, monkeypatch):
    runner = build_window(tmp_path, 4)[0]
    price_calls = []
    monkeypatch.setattr(
        runner._observer, "observe_anchor_close",
        lambda **kwargs: price_calls.append(kwargs) or (_ for _ in ()).throw(AssertionError()),
    )
    runner._profile._private_path.write_bytes(
        runner._profile._private_path.read_bytes() + b" /private/account/path"
    )
    result = runner.run()
    assert result["status"] == "BLOCKED"
    assert result["counts"]["terminal_cycle_arms"] == 20
    assert result["counts"]["complete_cycle_arms"] == 0
    assert not price_calls
    assert result["terminal_cycles"][0]["terminal_arms"][0]["terminal_status"] == "BLOCKED_COMMON_SOURCE"
    assert "/private" not in result.to_canonical_json_bytes().decode()


@pytest.mark.parametrize("stage", ["close", "plan", "open", "settle", "link", "metric"])
def test_process_control_propagates_from_close_plan_open_settle_link_and_metric(tmp_path, monkeypatch, stage):
    runner = build_window(tmp_path, 4)[0]
    stop = lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    if stage == "close":
        monkeypatch.setattr(runner._observer, "observe_anchor_close", stop)
    elif stage == "plan":
        monkeypatch.setattr(CurrentRuleShadowIntentPlanner, "plan", stop)
    elif stage == "open":
        monkeypatch.setattr(runner._observer, "observe_next_open", stop)
    elif stage == "settle":
        monkeypatch.setattr(ShadowPortfolioTransition, "apply", stop)
    elif stage == "link":
        monkeypatch.setattr(HistoricalShadowWindowReplay, "_chain_links", staticmethod(stop))
    else:
        monkeypatch.setattr(HistoricalShadowWindowReplay, "_metrics", staticmethod(stop))
    with pytest.raises(KeyboardInterrupt):
        runner.run()


def test_complete_common_window_metrics_match_exact_decimal_oracle(tmp_path):
    result = build_window(tmp_path, 4)[0].run()
    assert result["metric_capability"] is True
    assert tuple(item["arm_id"] for item in result["arm_metrics"]) == ARM_IDS
    champion = result["arm_metrics"][0]
    first = result["terminal_cycles"][0]["terminal_arms"][0]["transition_result"]
    last = result["terminal_cycles"][-1]["terminal_arms"][0]["transition_result"]
    expected = Decimal(last["nav_after"]) / Decimal(first["nav_before"])
    assert Decimal(champion["terminal_normalized_nav"]) == expected
    assert Decimal(champion["max_drawdown"]) <= 0
    summary = compact_window_summary(result)
    assert set(summary) == {
        "status", "evidence_class", "warnings", "prospective_claim",
        "promotion_capability", "window_digest", "requested_cycle_count",
        "terminal_cycle_count", "complete_cycle_arm_count", "blocked_cycle_arm_count",
        "chain_continuity_checked", "metric_capability", "arms",
    }
    assert "cash" not in json.dumps(summary).lower()


def test_partial_valuation_nulls_all_window_metrics_without_blocking_complete_state_chain(tmp_path):
    _runner, stage_a, inputs, _profile, _workspace, qlib = build_window(tmp_path / "base", 4)
    profile_root = tmp_path / "holding"
    profile_root.mkdir()
    holding = load_replay_profile(write_profile(profile_root, profile_raw(
        bootstrap={
            "as_of_date": DATES[0], "cash": "10000",
            "positions": [{"instrument": "SH000001", "quantity": 100, "book_cost": "1000"}],
        },
    )))
    runner = HistoricalShadowWindowReplay(
        stage_a_result=stage_a, stage_a_inputs=inputs, profile=holding,
        provider_root=qlib, market="csi300", window_size=4,
    )
    (runner._provider_root / "features/sh000001/open.day.bin").unlink()
    result = runner.run()
    assert result["status"] == "COMPLETE"
    assert result["chain_continuity_checked"] is True
    assert result["common_valuation_window_status"] == "NOT_COMPARABLE_MISSING_VALUATION"
    assert result["metric_capability"] is False
    assert result["arm_metrics"] == [] and result["champion_comparisons"] == []
    assert all(item["terminal_normalized_nav"] is None for item in compact_window_summary(result)["arms"])


def test_zero_negative_nav_and_empty_holding_overlap_edges_are_exact(tmp_path):
    _runner, stage_a, inputs, _profile, _workspace, qlib = build_window(tmp_path / "base", 4)
    profile_root = tmp_path / "zero"
    profile_root.mkdir()
    zero = load_replay_profile(write_profile(profile_root, profile_raw(
        bootstrap={"as_of_date": DATES[0], "cash": "0", "positions": []},
    )))
    result = HistoricalShadowWindowReplay(
        stage_a_result=stage_a, stage_a_inputs=inputs, profile=zero,
        provider_root=qlib, market="csi300", window_size=4,
    ).run()
    assert result["status"] == "BLOCKED"
    assert result["metric_capability"] is False
    assert result["arm_metrics"] == [] and result["champion_comparisons"] == []
    negative_root = tmp_path / "negative"
    negative_root.mkdir()
    with pytest.raises(Exception):
        load_replay_profile(write_profile(negative_root, profile_raw(
            bootstrap={"as_of_date": DATES[0], "cash": "-1", "positions": []},
        )))


def test_public_replay_and_impossible_result_capability_are_denied(tmp_path):
    runner = build_window(tmp_path, 4)[0]
    result = runner.run()
    with pytest.raises(HistoricalWindowContractError):
        HistoricalWindowResult(result.to_dict())
    with pytest.raises(HistoricalWindowContractError):
        SequentialPriorStateSet()
    with pytest.raises(HistoricalWindowContractError, match="replay-owned"):
        SequentialPriorStateSet._create(
            window_id="forged", source="INITIAL_BOOTSTRAP", source_cycle_id=None,
            as_of_boundary=runner._profile.bootstrap["as_of_date"],
            states=runner._initial_state_set.states,
        )
    forged = result.to_dict()
    forged["terminal_cycles"][1]["terminal_arms"][0], forged["terminal_cycles"][1]["terminal_arms"][1] = (
        forged["terminal_cycles"][1]["terminal_arms"][1], forged["terminal_cycles"][1]["terminal_arms"][0]
    )
    forged["result_digest"] = _digest_payload({key: value for key, value in forged.items() if key != "result_digest"})
    with pytest.raises(HistoricalWindowContractError, match="grid identity"):
        HistoricalWindowResult(forged, _WINDOW_AUTHORITY)


def test_ordinary_member_failure_retains_current_five_blocks_remainder_and_stops_future_reads(tmp_path, monkeypatch):
    runner = build_window(tmp_path, 4)[0]
    plans = []
    opens = []
    original_plan = CurrentRuleShadowIntentPlanner.plan
    original_open = runner._observer.observe_next_open

    def fail_middle(**kwargs):
        plans.append(kwargs["portfolio_id"])
        if len(plans) == 3:
            raise RuntimeError("private planning failure")
        return original_plan(**kwargs)

    def observe_open(**kwargs):
        opens.append(kwargs["trade_date"])
        return original_open(**kwargs)

    monkeypatch.setattr(CurrentRuleShadowIntentPlanner, "plan", fail_middle)
    monkeypatch.setattr(runner._observer, "observe_next_open", observe_open)
    result = runner.run()
    first = result["terminal_cycles"][0]
    assert len(first["terminal_arms"]) == 5
    assert first["terminal_arms"][2]["terminal_status"] == "BLOCKED_PLANNING"
    assert not opens
    assert all(
        arm["terminal_status"] == "BLOCKED_WINDOW_PREDECESSOR_INCOMPLETE"
        for cycle in result["terminal_cycles"][1:] for arm in cycle["terminal_arms"]
    )


@pytest.mark.parametrize("failure_index", [1, 3])
def test_cross_arm_bootstrap_stale_fork_and_digest_tamper_deny_chain_capability(tmp_path, monkeypatch, failure_index):
    runner = build_window(tmp_path, 4)[0]
    original = HistoricalShadowWindowReplay._chain_links

    def tampered(index, arms, predecessor):
        links = list(original(index, arms, predecessor))
        if index == failure_index:
            links[1] = {**links[1], "digest_equal_checked": False}
        return tuple(links)

    monkeypatch.setattr(HistoricalShadowWindowReplay, "_chain_links", staticmethod(tampered))
    result = runner.run()
    assert result["status"] == "BLOCKED"
    assert result["chain_continuity_checked"] is False
    assert result["metric_capability"] is False
    if failure_index < 3:
        assert result["terminal_cycles"][failure_index + 1]["terminal_arms"][0]["terminal_status"] == "BLOCKED_WINDOW_PREDECESSOR_INCOMPLETE"


def test_gap_duplicate_reorder_foreign_anchor_and_invalid_window_size_fail_before_price_read(tmp_path):
    _runner, stage_a, inputs, profile, _workspace, qlib = build_window(tmp_path, 6)
    original = list(stage_a["inventory"]["selected_anchors"])
    mutations = (
        [original[0], original[1], original[3], original[4]],
        [original[0], original[1], original[1], original[3]],
        [original[1], original[0], original[2], original[3]],
        [original[0], original[1], original[2], "2026-08-20"],
    )
    for selected in mutations:
        payload = _copy_result_value(stage_a._trusted_payload())
        payload["inventory"]["selected_anchors"] = selected
        payload["result_digest"] = _canonical_digest(
            _public_result(payload, include_digest=False, include_csv=False)
        )
        forged = ReplayResult(payload, _RESULT_AUTHORITY)
        with pytest.raises(HistoricalWindowContractError):
            HistoricalShadowWindowReplay(
                stage_a_result=forged, stage_a_inputs=inputs, profile=profile,
                provider_root=qlib, market="csi300", window_size=4,
            )
