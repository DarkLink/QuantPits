from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.evidence.ranking import canonical_full_ranking
from quantpits.research import replay
from quantpits.research.replay import (
    ReplayContractError,
    ReplayInputError,
    ResearchRankingReplay,
    load_sealed_replay_inputs,
    write_replay_output,
)
from quantpits.utils.predict_utils import rank_norm


DATES = (
    "2026-07-03", "2026-07-10", "2026-07-17", "2026-07-24",
    "2026-07-31", "2026-08-07", "2026-08-14", "2026-08-21",
)
MODELS = ("MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D")
INSTRUMENTS = ("AAA", "BBB", "CCC", "DDD")


def _digest(data):
    return {
        "algorithm": "sha256", "domain": "raw_bytes",
        "value": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }


def _prediction(model_position, *, missing=None, duplicate=False, non_finite=None, foreign=None):
    rows = []
    values = []
    for date_position, date in enumerate(DATES):
        for instrument_position, instrument in enumerate(INSTRUMENTS):
            if missing == (date, MODELS[model_position], instrument):
                continue
            rows.append((pd.Timestamp(date), instrument))
            value = float((instrument_position + 1) * (model_position + 2) + date_position % 3)
            if non_finite == (date, MODELS[model_position], instrument):
                value = float("nan")
            values.append(value)
    if duplicate:
        rows.append(rows[0])
        values.append(values[0])
    if foreign and foreign[1] == MODELS[model_position]:
        rows.append((pd.Timestamp(foreign[0]), foreign[2]))
        values.append(1.0)
    return pd.DataFrame(
        {"score": values},
        index=pd.MultiIndex.from_tuples(rows, names=["datetime", "instrument"]),
    )


def _sealed_champion(frames, anchor=DATES[-1]):
    normalized = []
    for frame in frames:
        selected = frame.xs(pd.Timestamp(anchor), level="datetime")["score"]
        selected.index = pd.MultiIndex.from_arrays(
            [[pd.Timestamp(anchor)] * len(selected), selected.index],
            names=["datetime", "instrument"],
        )
        normalized.append(rank_norm(selected))
    score = sum(normalized) / len(normalized)
    scores = {instrument: float(value) for (_date, instrument), value in score.items()}
    return canonical_full_ranking(INSTRUMENTS, scores)


def _fixture(tmp_path, *, missing=None, duplicate=False, non_finite=None, foreign=None, parity_shift=0.0):
    workspace = tmp_path / "workspace"
    qlib = tmp_path / "qlib"
    cycle_root = workspace / "data/evidence/v1/cycles" / DATES[-1]
    cycle_root.mkdir(parents=True)
    (qlib / "instruments").mkdir(parents=True)
    (qlib / "calendars").mkdir()
    universe = ("\n".join("%s\t2026-01-01\t2026-12-31" % item for item in INSTRUMENTS) + "\n").encode()
    calendar = ("\n".join(DATES) + "\n").encode()
    (qlib / "instruments/fixture.txt").write_bytes(universe)
    (qlib / "calendars/day.txt").write_bytes(calendar)
    frames = []
    source_models = []
    source_artifacts = []
    for position, model in enumerate(MODELS):
        frame = _prediction(
            position, missing=missing, duplicate=duplicate and position == 0,
            non_finite=non_finite, foreign=foreign,
        )
        frames.append(frame)
        artifact = workspace / "mlruns" / str(position) / "artifacts"
        artifact.mkdir(parents=True)
        prediction_path = artifact / "pred.pkl"
        frame.to_pickle(prediction_path)
        data = prediction_path.read_bytes()
        recorder = "recorder-%s" % position
        logical = prediction_path.relative_to(workspace).as_posix()
        source_models.append({
            "resolved_key": model, "status": "ready", "recorder_id": recorder,
            "experiment_name": "fixture-experiment", "artifact_path": artifact.relative_to(workspace).as_posix(),
        })
        source_artifacts.append({
            "recorder_id": recorder,
            "members": [{"path": logical, "digest": _digest(data)}],
        })
    champion = _sealed_champion(frames)
    if parity_shift:
        scores = {
            row["instrument"]: float(row["raw_score"]) + (parity_shift if row["instrument"] == "AAA" else 0.0)
            for row in champion.rows
        }
        champion = canonical_full_ranking(INSTRUMENTS, scores)
    ranking_bytes = champion.to_csv_bytes()
    (cycle_root / "ranking.csv").write_bytes(ranking_bytes)
    manifest = {
        "cycle_identity": {"cycle_id": DATES[-1]},
        "data_identity": {
            "source_to_materialization_relation": "unverified",
            "qlib_materialization_identity": {
                "status": "observed", "calendar_cutoff": DATES[-1],
                "universe_name": "fixture", "universe_digest": _digest(universe),
                "calendar_digest": _digest(calendar),
            },
        },
        "model_and_ensemble_lineage": {
            "status": "complete", "source_models": source_models,
            "source_artifacts": source_artifacts + [{
                "recorder_id": "ensemble-recorder", "position": "ensemble", "members": [],
            }],
            "combo": {"resolved_members": list(MODELS)},
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (cycle_root / "manifest.json").write_bytes(manifest_bytes)
    seal = {
        "cycle_id": DATES[-1], "manifest_digest": _digest(manifest_bytes),
        "named_file_digests": {"ranking.csv": _digest(ranking_bytes)},
    }
    (cycle_root / "seal.json").write_bytes(canonical_json_bytes(seal))
    return workspace, qlib


def _rewrite_source_relation(workspace, value):
    cycle_root = workspace / "data/evidence/v1/cycles" / DATES[-1]
    manifest_path = cycle_root / "manifest.json"
    seal_path = cycle_root / "seal.json"
    manifest = json.loads(manifest_path.read_text())
    if value is None:
        manifest["data_identity"].pop("source_to_materialization_relation")
    else:
        manifest["data_identity"]["source_to_materialization_relation"] = value
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    seal = json.loads(seal_path.read_text())
    seal["manifest_digest"] = _digest(manifest_bytes)
    seal_path.write_bytes(canonical_json_bytes(seal))


def _run(tmp_path, **fixture_kwargs):
    workspace, qlib = _fixture(tmp_path, **fixture_kwargs)
    inputs = load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)
    result = ResearchRankingReplay(inputs, top_k=2).run(
        preferred_start=DATES[0], preferred_end=DATES[-1], window_size=6,
    )
    return result, workspace, qlib


def test_a0_through_a3_are_complete_traceable_and_deterministic(tmp_path):
    first, _workspace, _qlib = _run(tmp_path / "first")
    second, _workspace, _qlib = _run(tmp_path / "second")

    assert first["status"] == "complete"
    assert first["parity"]["status"] == "passed"
    assert first["inventory"]["selected_anchors"] == list(DATES[-6:])
    assert first["inventory"]["excluded_anchors"] == [
        {"anchor": DATES[0], "reason": "outside_selected_contiguous_window"},
        {"anchor": DATES[1], "reason": "outside_selected_contiguous_window"},
    ]
    assert len(first["source_models"]) == 4
    assert all(len(row["members"]) == 4 for row in first["inventory"]["dates"])
    assert all(len(arms) == 5 for arms in first["rankings"].values())
    assert len(first["comparisons"]) == 6 * 4
    assert first["result_digest"] == second["result_digest"]
    assert first["prospective_claim"] is False
    assert first["promotion_capability"] is False


def test_incomplete_member_stays_visible_and_window_remains_contiguous(tmp_path):
    missing = (DATES[-2], MODELS[1], "BBB")
    result, _workspace, _qlib = _run(tmp_path, missing=missing)

    assert result["status"] == "complete"  # the earlier six-week contiguous run remains usable
    excluded = result["inventory"]["excluded_anchors"]
    assert {item["anchor"]: item["reason"] for item in excluded}[DATES[-2]] == "incomplete_model_or_universe_coverage"
    observed = next(row for row in result["inventory"]["dates"] if row["anchor"] == DATES[-2])
    assert observed["members"][1]["missing_eligible"] == ["BBB"]
    assert DATES[-2] not in result["rankings"]
    assert result["inventory"]["selected_anchors"] == list(DATES[:6])
    assert set(result["rankings"][DATES[-1]]) == {
        "CHAMPION_4", "DROP_1_3", "DROP_2_3", "DROP_3_3", "DROP_4_3",
    }


def test_fewer_than_four_complete_weeks_blocks_without_challenger_capability(tmp_path):
    workspace, qlib = _fixture(tmp_path)
    inputs = load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)
    result = ResearchRankingReplay(inputs).run(
        preferred_start=DATES[-3], preferred_end=DATES[-1], window_size=4,
    )
    assert result["status"] == "blocked_inventory"
    assert result["rankings"] == {}
    assert result["comparisons"] == []


def test_champion_parity_failure_prevents_challenger_results(tmp_path):
    result, _workspace, _qlib = _run(tmp_path, parity_shift=0.01)
    assert result["status"] == "blocked_parity"
    assert result["parity"]["score_mismatches"] == ["AAA"]
    assert set(result["rankings"]) == {DATES[-1]}
    assert set(result["rankings"][DATES[-1]]) == {"CHAMPION_4"}
    assert result["comparisons"] == []


@pytest.mark.parametrize("target", ["manifest", "prediction", "universe"])
def test_frozen_input_hash_mismatch_is_fail_closed(tmp_path, target):
    workspace, qlib = _fixture(tmp_path)
    if target == "manifest":
        path = workspace / "data/evidence/v1/cycles" / DATES[-1] / "manifest.json"
    elif target == "prediction":
        path = workspace / "mlruns/0/artifacts/pred.pkl"
    else:
        path = qlib / "instruments/fixture.txt"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ReplayInputError):
        load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)


@pytest.mark.parametrize("relation", [None, "", "verified", False])
def test_source_to_materialization_relation_is_observed_and_fail_closed(tmp_path, relation):
    workspace, qlib = _fixture(tmp_path)
    _rewrite_source_relation(workspace, relation)
    with pytest.raises(ReplayInputError, match="source-to-materialization"):
        load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)


def test_one_source_failure_preserves_later_requested_identity_and_cardinality(tmp_path):
    workspace, qlib = _fixture(tmp_path)
    path = workspace / "mlruns/0/artifacts/pred.pkl"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ReplayInputError) as caught:
        load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)
    assert [item["model_name"] for item in caught.value.evidence] == list(MODELS)
    assert len(caught.value.evidence) == 4
    assert caught.value.evidence[0]["status"] == "failed"
    assert [item["status"] for item in caught.value.evidence[1:]] == ["ready", "ready", "ready"]


def test_symlinked_source_public_name_is_denied_without_dropping_later_members(tmp_path):
    workspace, qlib = _fixture(tmp_path)
    path = workspace / "mlruns/0/artifacts/pred.pkl"
    displaced = path.with_name("displaced.pkl")
    path.rename(displaced)
    path.symlink_to(displaced.name)
    with pytest.raises(ReplayInputError) as caught:
        load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)
    assert "symlink" in caught.value.evidence[0]["detail"]
    assert [item["status"] for item in caught.value.evidence[1:]] == ["ready", "ready", "ready"]


def test_duplicate_prediction_identity_is_rejected_at_typed_leaf(tmp_path):
    workspace, qlib = _fixture(tmp_path, duplicate=True)
    with pytest.raises(ReplayInputError) as caught:
        load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)
    assert caught.value.evidence[0]["status"] == "failed"
    assert caught.value.evidence[0]["error_type"] == "ReplayInputError"
    assert caught.value.evidence[0]["detail"] == "prediction contains duplicate datetime/instrument rows"


def test_non_finite_value_is_inventory_fact_not_silently_neutralized(tmp_path):
    target = (DATES[-2], MODELS[2], "CCC")
    result, _workspace, _qlib = _run(tmp_path, non_finite=target)
    row = next(item for item in result["inventory"]["dates"] if item["anchor"] == DATES[-2])
    assert row["members"][2]["non_finite_instruments"] == ["CCC"]
    assert DATES[-2] not in result["rankings"]


def test_foreign_prediction_remainder_stays_visible_and_date_is_excluded(tmp_path):
    target = (DATES[-2], MODELS[2], "FOREIGN")
    result, _workspace, _qlib = _run(tmp_path, foreign=target)
    row = next(item for item in result["inventory"]["dates"] if item["anchor"] == DATES[-2])
    assert row["members"][2]["foreign_instruments"] == ["FOREIGN"]
    assert row["members"][2]["observed_row_count"] == len(INSTRUMENTS) + 1
    assert row["members"][2]["finite_scored_count"] == len(INSTRUMENTS)
    assert DATES[-2] not in result["rankings"]


def test_a0_observes_anchor_close_and_next_open_coverage_without_qlib_init(tmp_path):
    workspace, qlib = _fixture(tmp_path)
    values = [float(position + 1) for position in range(len(DATES))]
    payload = struct.pack("<%sf" % (len(values) + 1), 0.0, *values)
    for instrument in INSTRUMENTS:
        root = qlib / "features" / instrument.lower()
        root.mkdir(parents=True)
        (root / "close.day.bin").write_bytes(payload)
        (root / "open.day.bin").write_bytes(payload)
    inputs = load_sealed_replay_inputs(
        workspace, DATES[-1], qlib_data_dir=qlib,
        coverage_start=DATES[0], coverage_end=DATES[-1],
    )
    result = ResearchRankingReplay(inputs).run(
        preferred_start=DATES[0], preferred_end=DATES[-1], window_size=6,
    )
    first = next(item for item in result["inventory"]["dates"] if item["anchor"] == DATES[0])
    last = next(item for item in result["inventory"]["dates"] if item["anchor"] == DATES[-1])
    assert first["price_availability"]["anchor_close_status"] == "complete"
    assert first["price_availability"]["next_open_status"] == "complete"
    assert last["price_availability"]["anchor_close_status"] == "complete"
    assert last["price_availability"]["next_session"] is None
    assert last["price_availability"]["next_open_missing"] == list(INSTRUMENTS)
    assert result["calendar_identity"] == {
        "logical_path": "calendars/day.txt", "digest": _digest(
            ("\n".join(DATES) + "\n").encode()
        ),
    }
    assert result["source_to_materialization_relation"] == "unverified"
    assert len(result["price_source_inventory"]) == len(INSTRUMENTS) * 2
    assert all(set(item) == {
        "instrument", "field", "logical_path", "status", "digest",
    } for item in result["price_source_inventory"])
    assert result["price_source_inventory_digest"]["domain"] == "canonical_json"
    inventory_bytes = canonical_json_bytes(result["price_source_inventory"])
    assert result["price_source_inventory_digest"] == {
        "algorithm": "sha256", "domain": "canonical_json",
        "value": hashlib.sha256(inventory_bytes).hexdigest(),
        "size_bytes": len(inventory_bytes),
    }


def test_output_is_create_new_tmp_only_and_content_manifest_is_repeatable(tmp_path):
    result, _workspace, _qlib = _run(tmp_path / "source")
    first = write_replay_output(result, Path("/tmp") / ("qp_replay_test_%s_a" % tmp_path.name))
    second = write_replay_output(result, Path("/tmp") / ("qp_replay_test_%s_b" % tmp_path.name))
    try:
        assert first == second
        assert first["status"] == "complete"
        assert any(item["path"].endswith("CHAMPION_4.csv") for item in first["output_files"])
        with pytest.raises(ReplayContractError):
            write_replay_output(result, Path.cwd() / "not_allowed_replay_output")
    finally:
        import shutil
        shutil.rmtree(Path("/tmp") / ("qp_replay_test_%s_a" % tmp_path.name), ignore_errors=True)
        shutil.rmtree(Path("/tmp") / ("qp_replay_test_%s_b" % tmp_path.name), ignore_errors=True)


def test_output_rejects_symlink_parent_even_when_target_resolves_below_tmp(tmp_path):
    result, _workspace, _qlib = _run(tmp_path / "source")
    real_parent = Path("/tmp") / ("qp_replay_real_%s" % tmp_path.name)
    alias = Path("/tmp") / ("qp_replay_alias_%s" % tmp_path.name)
    real_parent.mkdir()
    alias.symlink_to(real_parent, target_is_directory=True)
    try:
        with pytest.raises(ReplayContractError, match="canonical"):
            write_replay_output(result, alias / "result")
    finally:
        alias.unlink()
        real_parent.rmdir()


def test_output_detects_canonical_root_replacement_after_member_commit(tmp_path, monkeypatch):
    result, _workspace, _qlib = _run(tmp_path / "source")
    output = Path("/tmp") / ("qp_replay_drift_%s" % tmp_path.name)
    displaced = output.with_name(output.name + "_displaced")
    original_replace = replay.os.replace
    triggered = {"value": False}

    def replace_then_drift(source, target):
        original_replace(source, target)
        if not triggered["value"]:
            triggered["value"] = True
            output.rename(displaced)
            output.mkdir()

    monkeypatch.setattr(replay.os, "replace", replace_then_drift)
    try:
        with pytest.raises(ReplayContractError, match="namespace identity"):
            write_replay_output(result, output)
    finally:
        import shutil
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(displaced, ignore_errors=True)


def test_strict_falsey_and_bool_contracts_are_distinct(tmp_path):
    workspace, qlib = _fixture(tmp_path)
    inputs = load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)
    with pytest.raises(ReplayContractError):
        ResearchRankingReplay(inputs, top_k=True)
    with pytest.raises(TypeError):
        ResearchRankingReplay(inputs, score_tolerance=1e300)
    with pytest.raises(ReplayContractError):
        ResearchRankingReplay(inputs).run(
            preferred_start="", preferred_end=DATES[-1], window_size=4,
        )


def test_public_replay_and_dataclass_replace_cannot_forge_observed_authority(tmp_path):
    workspace, qlib = _fixture(tmp_path)
    inputs = load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)
    result = ResearchRankingReplay(inputs).run(
        preferred_start=DATES[0], preferred_end=DATES[-1], window_size=6,
    )
    with pytest.raises(ReplayContractError, match="inspector-owned"):
        replace(inputs, parity_anchor=DATES[-2])
    with pytest.raises(ReplayContractError, match="truth-owner"):
        write_replay_output(dict(result), Path("/tmp") / ("forged_%s" % tmp_path.name))


def test_price_inventory_aggregate_revalidates_member_and_digest(tmp_path):
    workspace, qlib = _fixture(tmp_path)
    inputs = load_sealed_replay_inputs(
        workspace, DATES[-1], qlib_data_dir=qlib,
        coverage_start=DATES[0], coverage_end=DATES[-1],
    )
    forged = dict(inputs.__dict__)
    inventory = [dict(item) for item in inputs.price_source_inventory]
    inventory[0]["logical_path"] = "features/foreign/close.day.bin"
    forged["price_source_inventory"] = tuple(inventory)
    with pytest.raises(ReplayContractError, match="logical path"):
        replay.SealedReplayInputs(
            **forged, _authority=replay._OBSERVATION_AUTHORITY
        )


def test_result_access_cannot_mutate_truth_owner_state(tmp_path):
    result, _workspace, _qlib = _run(tmp_path)
    inventory = result["inventory"]
    inventory["status"] = "forged"
    assert result["inventory"]["status"] == "complete"


def test_process_control_interrupt_from_prediction_decoder_propagates(tmp_path, monkeypatch):
    workspace, qlib = _fixture(tmp_path)
    monkeypatch.setattr(replay.pd, "read_pickle", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        load_sealed_replay_inputs(workspace, DATES[-1], qlib_data_dir=qlib)


def test_cli_executes_complete_replay_without_backend_initialization(tmp_path):
    from quantpits.scripts import research_replay

    workspace, qlib = _fixture(tmp_path)
    output = Path("/tmp") / ("qp_replay_cli_%s" % tmp_path.name)
    try:
        status = research_replay.main([
            "--workspace", str(workspace),
            "--qlib-data-dir", str(qlib),
            "--sealed-cycle", DATES[-1],
            "--preferred-start", DATES[0],
            "--preferred-end", DATES[-1],
            "--top-k", "2",
            "--output-dir", str(output),
        ])
        assert status == 0
        payload = json.loads((output / "result.json").read_text())
        assert payload["status"] == "complete"
        assert payload["schema_version"] == 2
        assert payload["source_to_materialization_relation"] == "unverified"
        assert payload["calendar_identity"]["logical_path"] == "calendars/day.txt"
        assert len(payload["price_source_inventory"]) == len(INSTRUMENTS) * 2
        inventory_bytes = canonical_json_bytes(payload["price_source_inventory"])
        assert payload["price_source_inventory_digest"]["value"] == hashlib.sha256(
            inventory_bytes
        ).hexdigest()
    finally:
        import shutil
        shutil.rmtree(output, ignore_errors=True)


@pytest.mark.parametrize("omitted", ["--preferred-start", "--preferred-end", "--top-k"])
def test_cli_requires_workspace_specific_replay_parameters(omitted):
    from quantpits.scripts import research_replay

    values = {
        "--sealed-cycle": DATES[-1],
        "--preferred-start": DATES[0],
        "--preferred-end": DATES[-1],
        "--top-k": "2",
        "--output-dir": "/tmp/not_created_by_parser_test",
    }
    argv = [item for pair in values.items() if pair[0] != omitted for item in pair]
    with pytest.raises(SystemExit) as caught:
        research_replay.build_parser().parse_args(argv)
    assert caught.value.code == 2


def test_cli_has_no_score_tolerance_override():
    from quantpits.scripts import research_replay

    with pytest.raises(SystemExit) as caught:
        research_replay.build_parser().parse_args([
            "--sealed-cycle", DATES[-1],
            "--preferred-start", DATES[0],
            "--preferred-end", DATES[-1],
            "--top-k", "2",
            "--output-dir", "/tmp/not_created_by_parser_test",
            "--score-tolerance", "1e300",
        ])
    assert caught.value.code == 2
