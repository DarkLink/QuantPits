import json
import os
import shutil
from dataclasses import replace

import pytest
import pandas as pd

from quantpits.evidence import CaptureRequest, ContractError, ProductionCycleEvidenceSealer, TypedDigest
import quantpits.evidence.sealing as sealing_module
from quantpits.utils.workspace import fingerprint_value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def cycle_factory(tmp_path, monkeypatch):
    def build(*, deep=True, decision=True, scores=None, dirty=False):
        root = tmp_path / "workspace"
        root.mkdir()
        git_control = root / ".synthetic-git-control"
        (git_control / "refs" / "heads").mkdir(parents=True)
        (git_control / "HEAD").write_text("ref: refs/heads/main\n")
        (git_control / "index").write_bytes(b"synthetic-index")
        (git_control / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
        qlib = tmp_path / "qlib"
        (qlib / "calendars").mkdir(parents=True)
        (qlib / "instruments").mkdir()
        (qlib / "calendars" / "day.txt").write_text("2099-01-01\n2099-01-02\n")
        (qlib / "instruments" / "synthetic.txt").write_text(
            "AAA\t2090-01-01\t2100-01-01\nBBB\t2090-01-01\t2100-01-01\n"
        )
        (root / "config").mkdir()
        _write_json(root / "config" / "prod_config.json", {
            "current_cash": 1000.0,
            "current_holding": [{"instrument": "AAA", "amount": 10, "value": 100}],
        })
        model_config = {"market": "synthetic", "freq": "week"}
        _write_json(root / "config" / "model_config.json", model_config)
        (root / "output").mkdir()
        (root / "output" / "post.txt").write_text("post\n")
        (root / "output" / "prediction.txt").write_text("prediction\n")
        (root / "output" / "orders.csv").write_text("instrument,amount\nAAA,1\n")
        (root / "mlruns" / "train").mkdir(parents=True)
        (root / "mlruns" / "ensemble").mkdir()
        (root / "mlruns" / "source").mkdir()
        (root / "mlruns" / "train" / "model.pkl").write_bytes(b"synthetic-model")
        (root / "mlruns" / "train" / "pred.pkl").write_bytes(b"synthetic-model-prediction")
        (root / "mlruns" / "source" / "model.pkl").write_bytes(b"synthetic-source-model")
        values = scores if scores is not None else {"AAA": 2.0, "BBB": 1.0}
        rows = ["instrument,datetime,score"]
        rows.extend("%s,2099-01-02,%s" % item for item in values.items())
        (root / "output" / "ensemble.csv").write_text("\n".join(rows) + "\n")
        prediction_index = pd.MultiIndex.from_tuples(
            [(instrument, "2099-01-02") for instrument in values],
            names=["instrument", "datetime"],
        )
        pd.DataFrame({"score": list(values.values())}, index=prediction_index).to_pickle(
            root / "mlruns" / "ensemble" / "pred.pkl"
        )
        manifests = root / "output" / "manifests"
        common = {"status": "success", "records": {"anchor_date": "2099-01-02"}}
        _write_json(manifests / "m1.json", {
            "status": "success", "run_id": "M1", "command": "post-trade",
            "records": {
                "processed_date_count": 1,
                "processed_date_from": "2099-01-02",
                "processed_date_to": "2099-01-02",
            },
            "outputs": [{"path": "output/post.txt"}],
        })
        _write_json(manifests / "m2.json", {
            **common, "run_id": "M2", "command": "static_train",
            "outputs": [{"path": "output/prediction.txt"}],
        })
        _write_json(manifests / "m3.json", {
            "status": "success", "run_id": "M3", "command": "ensemble_fusion",
            "config_fingerprints": {
                "config/model_config.json": fingerprint_value(model_config),
            },
            "outputs": [{"path": "output/ensemble.csv"}],
            "records": {
                "anchor_date": "2099-01-02", "expected_anchor": "2099-01-02",
                "experiment_name": "SYNTHETIC_ENSEMBLE",
                "input_models": [{
                    "resolved_key": "MODEL_A", "recorder_id": "TRAIN_RECORDER_A",
                    "artifact_path": "mlruns/train", "status": "ready",
                    "experiment_name": "SYNTHETIC_TRAINING",
                    "experiment_id": "EXPERIMENT_A", "prediction_end": "2099-01-02",
                    "source_recorder_id": "SOURCE_TRAINING_A",
                    "source_experiment_name": "SYNTHETIC_SOURCE_TRAINING",
                    "source_artifact_path": "mlruns/source",
                }],
                "combos": [{
                    "name": "SYNTHETIC_COMBO", "method": "equal",
                    "is_default": True, "models": ["MODEL_A"],
                    "resolved_models": ["MODEL_A"],
                    "pred_file": "output/ensemble.csv",
                    "recorder_id": "ENSEMBLE_RECORDER_A",
                    "output_evidence": {
                        "artifact_path": "mlruns/ensemble", "contained": True,
                        "recorder_id": "ENSEMBLE_RECORDER_A",
                    },
                }],
            },
        })
        _write_json(manifests / "m4.json", {
            **common, "run_id": "M4", "command": "order_gen",
            "outputs": [{"path": "output/orders.csv"}],
        })
        deep_path = None
        if deep:
            deep_path = "output/deep"
            _write_json(root / deep_path / "trace.json", {
                "model": "synthetic-model", "prompt_digest": "2" * 64,
                "input_digest": "3" * 64, "output_digest": "4" * 64,
            })
        decision_path = None
        if decision:
            decision_path = "data/decision.json"
            _write_json(root / decision_path, {
                "decision_id": "DECISION_1", "decision_time": "2099-01-02T12:00:00Z",
                "evidence_cycle_id": "2099-01-02", "actor": "synthetic-owner",
                "decision": "NO_ACTION", "target": "synthetic-target",
                "reason_code": "FROZEN_OBSERVATION",
            })
        if dirty:
            (root / "output" / "operator-note.txt").write_text("observed dirty member\n")
        def synthetic_git(start):
            is_workspace = start.resolve() == root.resolve()
            dirty_members = []
            if is_workspace:
                dirty_members = sorted(
                    path.relative_to(root).as_posix()
                    for path in (root / "output").glob("operator-*.txt")
                )
            inventory = "\0".join(dirty_members).encode("utf-8")
            return {
                "status": "dirty_observed_and_fingerprinted" if dirty_members else "clean",
                "commit": "a" * 40 if is_workspace else "b" * 40,
                "tree": "c" * 40 if is_workspace else "d" * 40,
                "status_inventory_digest": TypedDigest.raw(inventory).to_dict(),
                "status_inventory": dirty_members,
                "tracked_diff_digest": TypedDigest.raw(b"").to_dict(),
                "repository_scope": "workspace", "remote_relation": "no_upstream",
            }
        monkeypatch.setattr(sealing_module, "inspect_git", synthetic_git)
        monkeypatch.setattr(
            sealing_module, "git_control_specs",
            lambda _start: ((git_control, ("HEAD", "index", "packed-refs", "refs")),),
        )
        return root, qlib, CaptureRequest(
            cycle_id="2099-01-02", research_epoch_id="SYNTHETIC_V1",
            post_trade_manifest="output/manifests/m1.json",
            prediction_manifest="output/manifests/m2.json",
            ensemble_manifest="output/manifests/m3.json",
            order_manifest="output/manifests/m4.json",
            deep_analysis_run=deep_path, decision_event=decision_path,
        )
    return build


def test_complete_cycle_publishes_one_verified_deterministic_bundle(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    result = sealer.capture(request)
    assert result.status == "sealed_complete"
    assert result.did_write is True
    bundle = root / result.bundle_path
    manifest = json.loads((bundle / "manifest.json").read_text())
    seal = json.loads((bundle / "seal.json").read_text())
    assert manifest["data_identity"]["source_to_materialization_relation"] == "unverified"
    assert manifest["ranking"]["eligible_count"] == 2
    assert (bundle / "ranking.csv").is_file()
    assert (bundle / "portfolio_state.json").is_file()
    assert seal["status"] == "sealed_complete"


def test_dataclass_replace_cannot_replay_inspector_authority(cycle_factory):
    root, qlib, request = cycle_factory()
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    with pytest.raises(ContractError, match="inspector-owned"):
        replace(result, status="adopted", did_write=False)


def test_capture_write_set_is_only_lock_staging_and_one_final_bundle(cycle_factory):
    root, qlib, request = cycle_factory()
    before = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    after = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    created = after - before
    assert result.status == "sealed_complete"
    assert created
    assert all(path == "data/evidence" or path.startswith("data/evidence/") for path in created)
    assert not any(path.endswith(".lock") or "/.staging/" in path for path in created)


def test_exact_replay_adopts_without_final_write(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    before = (root / first.bundle_path / "seal.json").read_bytes()
    second = sealer.capture(request)
    assert second.status == "adopted"
    assert second.did_write is False
    assert second.sealed_status == "sealed_complete"
    assert second.capability == "local_evidence_replay"
    assert second.seal_digest == first.seal_digest
    assert (root / first.bundle_path / "seal.json").read_bytes() == before


def test_exact_final_appearing_after_lock_is_verified_and_adopted(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    final = root / first.bundle_path
    saved = final.with_name("saved-exact-final")
    final.rename(saved)

    def fault(point):
        if point == "after_lock":
            saved.rename(final)

    replay = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert replay.status == "adopted"
    assert replay.capability == "local_evidence_replay"
    assert not (root / "data/evidence/v1/.locks/2099-01-02.lock").exists()


def test_different_final_appearing_after_lock_is_conflict(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    final = root / first.bundle_path
    saved = final.with_name("saved-different-final")
    final.rename(saved)
    source = root / request.order_manifest
    source.write_text(source.read_text() + " ")

    def fault(point):
        if point == "after_lock":
            saved.rename(final)

    conflict = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert conflict.status == "conflict"
    assert conflict.capability == "none"


def test_exact_final_winning_atomic_publish_race_is_adopted(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    final = root / first.bundle_path
    saved = final.with_name("saved-racing-final")
    final.rename(saved)
    original = sealing_module._rename_noreplace

    def race(source_parent_fd, source_name, target_parent_fd, target_name):
        saved.rename(final)
        original(
            source_parent_fd, source_name, target_parent_fd, target_name,
        )

    monkeypatch.setattr(sealing_module, "_rename_noreplace", race)
    replay = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert replay.status == "adopted"
    assert replay.capability == "local_evidence_replay"


def test_stage_member_file_exists_is_failure_not_cycle_conflict(
    cycle_factory,
):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "after_stage_created":
            stage = next((root / "data/evidence/v1/.staging").iterdir())
            (stage / "manifest.json").write_text("foreign\n")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert any(item["code"] == "staging_member_conflict" for item in result.problems)


def test_replay_source_mutation_during_existing_verification_denies_adoption(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    assert sealer.capture(request).status == "sealed_complete"
    source = root / request.order_manifest
    original = ProductionCycleEvidenceSealer._existing

    def mutate_after_verify(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        source.write_text(source.read_text() + " ")
        return result

    monkeypatch.setattr(ProductionCycleEvidenceSealer, "_existing", mutate_after_verify)
    replay = sealer.capture(request)
    assert replay.status == "blocked"
    assert replay.capability == "none"


def test_replay_final_replacement_during_existing_verification_denies_adoption(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    final = root / first.bundle_path
    original = ProductionCycleEvidenceSealer._existing

    def replace_after_verify(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if args[0] == final:
            displaced = final.with_name("displaced-final")
            final.rename(displaced)
            shutil.copytree(displaced, final)
        return result

    monkeypatch.setattr(
        ProductionCycleEvidenceSealer, "_existing", replace_after_verify,
    )
    replay = sealer.capture(request)
    assert replay.status == "blocked"
    assert replay.capability == "none"


def test_transient_valid_final_cannot_mask_tampered_canonical_bundle(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    final = root / first.bundle_path
    valid = final.with_name("valid-final")
    tampered = final.with_name("tampered-final")
    shutil.copytree(final, valid)
    (final / "ranking.csv").write_text("tampered\n")
    original = ProductionCycleEvidenceSealer._existing

    def verify_transient_valid_copy(self, *args, **kwargs):
        if args[0] != final:
            return original(self, *args, **kwargs)
        final.rename(tampered)
        valid.rename(final)
        try:
            return original(self, *args, **kwargs)
        finally:
            final.rename(valid)
            tampered.rename(final)

    monkeypatch.setattr(
        ProductionCycleEvidenceSealer, "_existing", verify_transient_valid_copy,
    )
    replay = sealer.capture(request)
    assert replay.status == "blocked"
    assert replay.capability == "none"
    assert (final / "ranking.csv").read_text() == "tampered\n"


def test_replay_member_tamper_after_first_verification_denies_adoption(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    final = root / first.bundle_path
    original = ProductionCycleEvidenceSealer._existing
    calls = 0

    def tamper_after_first(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 1:
            (final / "ranking.csv").write_text("tampered\n")
        return result

    monkeypatch.setattr(
        ProductionCycleEvidenceSealer, "_existing", tamper_after_first,
    )
    replay = sealer.capture(request)
    assert replay.status == "blocked"
    assert replay.capability == "none"


def test_existing_bundle_member_tamper_denies_adoption_capability(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    (root / first.bundle_path / "ranking.csv").write_text("tampered\n")
    second = sealer.capture(request)
    assert second.status == "conflict"
    assert second.capability == "none"
    assert second.seal_digest is None


def test_existing_manifest_semantic_tamper_cannot_be_hidden_by_resealing(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    bundle = root / first.bundle_path
    manifest_path = bundle / "manifest.json"
    seal_path = bundle / "seal.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["ranking"]["eligible_count"] = 999
    manifest_data = sealing_module.canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_data)
    seal = json.loads(seal_path.read_text())
    seal["manifest_digest"] = TypedDigest.raw(manifest_data).to_dict()
    seal_path.write_bytes(sealing_module.canonical_json_bytes(seal))

    replay = sealer.capture(request)
    assert replay.status == "conflict"
    assert replay.capability == "none"


def test_existing_noncanonical_manifest_representation_denies_adoption(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    bundle = root / first.bundle_path
    manifest_path = bundle / "manifest.json"
    seal_path = bundle / "seal.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_data)
    seal = json.loads(seal_path.read_text())
    seal["manifest_digest"] = TypedDigest.raw(manifest_data).to_dict()
    seal_path.write_bytes(sealing_module.canonical_json_bytes(seal))

    assert sealer.capture(request).status == "conflict"


def test_existing_extra_empty_object_directory_denies_adoption(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    (root / first.bundle_path / "objects" / "foreign").mkdir()

    assert sealer.capture(request).status == "conflict"


def test_resealed_bundle_cannot_omit_manifest_embedded_object(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    bundle = root / first.bundle_path
    seal_path = bundle / "seal.json"
    seal = json.loads(seal_path.read_text())
    removed = seal["object_digests"].pop()
    (bundle / "objects" / removed[:2] / removed).unlink()
    prefix = bundle / "objects" / removed[:2]
    if not any(prefix.iterdir()):
        prefix.rmdir()
    seal["artifact_root_digest"] = TypedDigest.canonical({
        "objects": seal["object_digests"],
        "named_files": seal["named_file_digests"],
    }).to_dict()
    seal_path.write_bytes(sealing_module.canonical_json_bytes(seal))

    assert sealer.capture(request).status == "conflict"


def test_resealed_bundle_cannot_omit_manifest_named_evidence(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    bundle = root / first.bundle_path
    seal_path = bundle / "seal.json"
    seal = json.loads(seal_path.read_text())
    del seal["named_file_digests"]["ranking.csv"]
    (bundle / "ranking.csv").unlink()
    seal["artifact_root_digest"] = TypedDigest.canonical({
        "objects": seal["object_digests"],
        "named_files": seal["named_file_digests"],
    }).to_dict()
    seal_path.write_bytes(sealing_module.canonical_json_bytes(seal))

    assert sealer.capture(request).status == "conflict"


def test_same_cycle_changed_input_conflicts_without_overwrite(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    first = sealer.capture(request)
    seal_path = root / first.bundle_path / "seal.json"
    before = seal_path.read_bytes()
    path = root / request.ensemble_manifest
    value = json.loads(path.read_text())
    value["records"]["combos"][0]["method"] = "changed"
    path.write_text(json.dumps(value))
    result = sealer.capture(request)
    assert result.status == "conflict"
    assert result.did_write is False
    assert seal_path.read_bytes() == before


def test_missing_deep_analysis_is_visible_partial(cycle_factory):
    root, qlib, request = cycle_factory(deep=False)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert any(item["code"] == "deep_analysis_missing" for item in result.problems)


def test_explicit_missing_deep_analysis_path_is_sealed_as_stable_partial(cycle_factory):
    root, qlib, request = cycle_factory(deep=False)
    request = replace(request, deep_analysis_run="output/deep/missing")
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert result.did_write is True
    assert any(item["code"] == "deep_analysis_incomplete" for item in result.problems)


def test_missing_required_manifest_is_preserved_in_partial_bundle(cycle_factory):
    root, qlib, request = cycle_factory()
    (root / request.order_manifest).unlink()
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert result.did_write is True
    assert any(
        item["code"] == "manifest_invalid" and item["evidence_class"] == "order"
        for item in result.problems
    )


def test_partial_replay_adoption_never_upgrades_complete_capability(cycle_factory):
    root, qlib, request = cycle_factory(deep=False)
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    assert sealer.capture(request).status == "sealed_partial"
    replay = sealer.capture(request)
    assert replay.status == "adopted"
    assert replay.sealed_status == "sealed_partial"
    assert replay.capability == "partial_replay_only"


def test_foreign_manifest_command_cannot_fill_an_m1_to_m4_slot(cycle_factory):
    root, qlib, request = cycle_factory()
    path = root / request.post_trade_manifest
    value = json.loads(path.read_text())
    value["command"] = "ensemble_fusion"
    path.write_text(json.dumps(value))
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert any(item["code"] == "manifest_command_mismatch" for item in result.problems)


def test_missing_prediction_does_not_forge_ranking(cycle_factory):
    root, qlib, request = cycle_factory(scores={"AAA": 1.0})
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    manifest = json.loads((root / result.bundle_path / "manifest.json").read_text())
    rows = (root / result.bundle_path / "ranking.csv").read_text().splitlines()
    assert result.status == "sealed_partial"
    assert manifest["ranking"]["missing_count"] == 1
    assert any("BBB" in row and "missing_prediction" in row for row in rows)


def test_missing_frozen_market_fingerprint_cannot_select_universe_by_filename_count(
    cycle_factory,
):
    root, qlib, request = cycle_factory()
    path = root / request.ensemble_manifest
    value = json.loads(path.read_text())
    del value["config_fingerprints"]
    path.write_text(json.dumps(value))
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert any(
        item["code"] == "market_config_fingerprint_missing"
        for item in result.problems
    )
    assert not (root / result.bundle_path / "ranking.csv").exists()


def test_malformed_universe_rows_cannot_manufacture_full_coverage(cycle_factory):
    root, qlib, request = cycle_factory()
    (qlib / "instruments/synthetic.txt").write_text("AAA\nBBB\n")
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert any(item["code"] == "ranking_unavailable" for item in result.problems)


def test_empty_prediction_creates_no_fake_ranking_rows(cycle_factory):
    root, qlib, request = cycle_factory(scores={})
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert not (root / result.bundle_path / "ranking.csv").exists()
    assert any(item["code"] == "ranking_unavailable" for item in result.problems)


def test_invalid_m3_prediction_keeps_diagnostic_and_no_fake_ranking(cycle_factory):
    root, qlib, request = cycle_factory()
    path = root / request.ensemble_manifest
    value = json.loads(path.read_text())
    value["records"]["combos"][0]["output_evidence"]["artifact_path"] = "mlruns/missing"
    path.write_text(json.dumps(value))
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert not (root / result.bundle_path / "ranking.csv").exists()
    assert any(item["code"] == "ranking_unavailable" for item in result.problems)


def test_local_recorder_evidence_loads_exact_pred_without_backend_initialization(cycle_factory):
    import pandas as pd

    root, qlib, request = cycle_factory()
    artifact = root / "mlruns/ensemble"
    frame = pd.DataFrame(
        {"score": [2.0, 1.0]},
        index=pd.MultiIndex.from_tuples(
            [("AAA", "2099-01-02"), ("BBB", "2099-01-02")],
            names=["instrument", "datetime"],
        ),
    )
    frame.to_pickle(artifact / "pred.pkl")
    path = root / request.ensemble_manifest
    value = json.loads(path.read_text())
    combo = value["records"]["combos"][0]
    combo["pred_file"] = "ENSEMBLE_RECORDER_A"
    combo["output_evidence"] = {
        "artifact_path": "mlruns/ensemble", "contained": True,
        "recorder_id": "ENSEMBLE_RECORDER_A",
    }
    path.write_text(json.dumps(value))
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_complete"
    manifest = json.loads((root / result.bundle_path / "manifest.json").read_text())
    assert manifest["ranking"]["scored_count"] == 2
    assert manifest["ranking"]["prediction_digest"]["domain"] == "raw_bytes"


def test_dirty_observed_and_fingerprinted_can_remain_complete(cycle_factory):
    root, qlib, request = cycle_factory(dirty=True)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    manifest = json.loads((root / result.bundle_path / "manifest.json").read_text())
    assert result.status == "sealed_complete"
    assert manifest["workspace_identity"]["status"] == "dirty_observed_and_fingerprinted"
    assert manifest["workspace_identity"]["status_inventory_digest"]["domain"] == "raw_bytes"


def test_absent_decision_is_not_recorded_not_no_action(cycle_factory):
    root, qlib, request = cycle_factory(decision=False)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    manifest = json.loads((root / result.bundle_path / "manifest.json").read_text())
    assert result.status == "sealed_complete"
    assert manifest["decision_state"] == {"status": "not_recorded", "as_of_capture": True}


def test_foreign_cycle_decision_is_partial_and_never_recorded(cycle_factory):
    root, qlib, request = cycle_factory()
    path = root / request.decision_event
    value = json.loads(path.read_text())
    value["evidence_cycle_id"] = "2099-01-03"
    path.write_text(json.dumps(value))
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    manifest = json.loads((root / result.bundle_path / "manifest.json").read_text())
    assert result.status == "sealed_partial"
    assert manifest["decision_state"]["status"] == "invalid"


def test_valid_decision_retains_exact_event_and_distinct_raw_canonical_digests(cycle_factory):
    root, qlib, request = cycle_factory()
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    decision = json.loads((root / result.bundle_path / "manifest.json").read_text())["decision_state"]
    assert decision["status"] == "recorded"
    assert decision["event"]["decision"] == "NO_ACTION"
    assert decision["raw_digest"]["domain"] == "raw_bytes"
    assert decision["canonical_digest"]["domain"] == "canonical_json"


def test_model_recorder_chain_and_llm_trace_are_retained(cycle_factory):
    root, qlib, request = cycle_factory()
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    manifest = json.loads((root / result.bundle_path / "manifest.json").read_text())
    lineage = manifest["model_and_ensemble_lineage"]
    assert lineage["source_models"][0]["recorder_id"] == "TRAIN_RECORDER_A"
    assert lineage["source_models"][0]["source_recorder_id"] == "SOURCE_TRAINING_A"
    assert lineage["combo"]["ensemble_recorder_id"] == "ENSEMBLE_RECORDER_A"
    assert any(
        item.get("role") == "source_training"
        and item["recorder_id"] == "SOURCE_TRAINING_A"
        and item["artifact_tree_digest"]["domain"] == "file_inventory"
        for item in lineage["source_artifacts"]
    )
    deep = next(item for item in manifest["run_evidence"] if item["class"] == "deep_analysis")
    assert deep["members"][0]["digest"]["domain"] == "raw_bytes"
    assert deep["members"][0]["preservation_status"] == "embedded"


def test_dry_run_has_zero_filesystem_write(cycle_factory):
    root, qlib, request = cycle_factory()
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request, dry_run=True)
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert result.status == "preview_complete"
    assert result.did_write is False
    assert result.bundle_path is None
    assert result.capability == "none"
    assert result.write_scope == "none"
    assert before == after


def test_dry_run_with_existing_bundle_remains_non_authoritative_preview(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    assert sealer.capture(request).status == "sealed_complete"
    preview = sealer.capture(request, dry_run=True)
    assert preview.status == "preview_complete"
    assert preview.capability == "none"
    assert preview.bundle_path is None


def test_duplicate_json_keys_make_source_visible_partial(cycle_factory):
    root, qlib, request = cycle_factory()
    path = root / request.order_manifest
    path.write_text('{"status":"success","status":"failed"}\n')
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert any(
        item["code"] == "manifest_invalid" and item["evidence_class"] == "order"
        for item in result.problems
    )


def test_internal_qlib_symlink_is_incomparable_without_outside_read(cycle_factory, tmp_path):
    root, qlib, request = cycle_factory()
    instruments = qlib / "instruments"
    outside = tmp_path / "outside-instruments"
    outside.mkdir()
    (outside / "synthetic.txt").write_text("SECRET\t2090-01-01\t2100-01-01\n")
    instruments.rename(qlib / "displaced-instruments")
    instruments.symlink_to(outside, target_is_directory=True)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "blocked"
    assert any(item["code"] == "path_boundary" for item in result.problems)


@pytest.mark.parametrize("fault_point,expected", [
    ("before_publish", "failed_no_final"),
    ("after_publish", "uncertain"),
])
def test_artifact_tree_member_addition_denies_capability(
    cycle_factory, fault_point, expected,
):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == fault_point:
            (root / "mlruns/train/foreign.bin").write_bytes(b"foreign")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == expected
    assert result.capability == "none"


def test_tampered_staging_is_rejected_before_irreversible_publish(cycle_factory):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "before_publish":
            stage = next((root / "data/evidence/v1/.staging").iterdir())
            (stage / "ranking.csv").write_text("tampered\n")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert any(item["code"] == "staging_verification_failed" for item in result.problems)
    assert not (root / "data/evidence/v1/cycles/2099-01-02").exists()


def test_staging_symlink_parent_cannot_write_outside_workspace(cycle_factory, tmp_path):
    root, qlib, request = cycle_factory()
    outside = tmp_path / "outside-stage"
    outside.mkdir()

    def fault(point):
        if point == "after_stage_created":
            stage = next((root / "data/evidence/v1/.staging").iterdir())
            (stage / "objects").symlink_to(outside, target_is_directory=True)

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert list(outside.iterdir()) == []


def test_staging_public_replacement_during_write_is_never_modified(cycle_factory):
    root, qlib, request = cycle_factory()
    replacement = None
    displaced = None

    def fault(point):
        nonlocal replacement, displaced
        if point == "after_stage_created":
            staging = root / "data/evidence/v1/.staging"
            original = next(staging.iterdir())
            displaced = original.with_name(original.name + ".displaced")
            original.rename(displaced)
            original.mkdir()
            replacement = original

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"
    assert replacement is not None and list(replacement.iterdir()) == []
    assert displaced is not None and (displaced / "manifest.json").is_file()
    assert not (root / "data/evidence/v1/cycles/2099-01-02").exists()


def test_write_parent_creation_race_cannot_escape_workspace(
    cycle_factory, tmp_path, monkeypatch,
):
    root, qlib, request = cycle_factory()
    outside = tmp_path / "outside-write-parent"
    outside.mkdir()
    displaced = root / "data-displaced"
    real_mkdir = sealing_module.os.mkdir
    replaced = False

    def replace_parent_before_descriptor_relative_create(path, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if path == "evidence" and dir_fd is not None and not replaced:
            replaced = True
            (root / "data").rename(displaced)
            (root / "data").symlink_to(outside, target_is_directory=True)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sealing_module.os, "mkdir", replace_parent_before_descriptor_relative_create)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "blocked"
    assert result.capability == "none"
    assert list(outside.iterdir()) == []
    assert (displaced / "evidence").is_dir()


def test_workspace_root_replacement_before_parent_creation_is_never_modified(
    cycle_factory,
):
    root, qlib, request = cycle_factory()
    displaced = root.with_name(root.name + "-displaced")
    replacement = root

    def fault(point):
        if point == "before_write_parent_creation":
            root.rename(displaced)
            replacement.mkdir()

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "blocked"
    assert result.capability == "none"
    assert list(replacement.iterdir()) == []
    assert (displaced / "data/evidence").is_dir()


def test_source_mutation_before_publish_fails_without_final(cycle_factory):
    root, qlib, request = cycle_factory()
    path = root / request.order_manifest

    def fault(point):
        if point == "before_publish":
            path.write_text(path.read_text() + " ")

    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib, fault_hook=fault).capture(request)
    assert result.status == "failed_no_final"
    assert result.write_scope == "staging_only"
    assert result.capability == "none"
    assert not (root / "data/evidence/v1/cycles/2099-01-02").exists()


def test_portfolio_mutation_before_publish_fails_without_final(cycle_factory):
    root, qlib, request = cycle_factory()
    portfolio = root / "config/prod_config.json"

    def fault(point):
        if point == "before_publish":
            portfolio.write_text(portfolio.read_text() + " ")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"


def test_same_bytes_portfolio_replacement_breaks_identity_continuity(cycle_factory):
    root, qlib, request = cycle_factory()
    portfolio = root / "config/prod_config.json"

    def fault(point):
        if point == "before_publish":
            replacement = portfolio.with_suffix(".replacement")
            replacement.write_bytes(portfolio.read_bytes())
            os.replace(str(replacement), str(portfolio))

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"


def test_transient_portfolio_move_away_and_back_is_not_erased(cycle_factory):
    root, qlib, request = cycle_factory()
    portfolio = root / "config/prod_config.json"

    def fault(point):
        if point == "before_publish":
            displaced = portfolio.with_suffix(".displaced")
            portfolio.rename(displaced)
            displaced.rename(portfolio)

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"


def test_transient_artifact_member_add_remove_is_not_erased(cycle_factory):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "before_publish":
            transient = root / "mlruns/train/transient.bin"
            transient.write_bytes(b"transient")
            transient.unlink()

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"


def test_prediction_artifact_change_between_consumers_prevents_publication(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    prediction = root / "mlruns/ensemble/pred.pkl"
    original = ProductionCycleEvidenceSealer._lineage_artifacts

    def mutate_then_observe(self, *args, **kwargs):
        prediction.write_bytes(prediction.read_bytes() + b"changed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        ProductionCycleEvidenceSealer, "_lineage_artifacts", mutate_then_observe,
    )
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"
    assert not (root / "data/evidence/v1/cycles/2099-01-02").exists()


def test_qlib_mutation_before_publish_fails_without_final(cycle_factory):
    root, qlib, request = cycle_factory()
    calendar = qlib / "calendars/day.txt"

    def fault(point):
        if point == "before_publish":
            calendar.write_text(calendar.read_text() + "2099-01-03\n")

    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib, fault_hook=fault).capture(request)
    assert result.status == "failed_no_final"
    assert not (root / "data/evidence/v1/cycles/2099-01-02").exists()


def test_source_mutation_after_publish_is_uncertain(cycle_factory):
    root, qlib, request = cycle_factory()
    source = root / request.order_manifest

    def fault(point):
        if point == "after_publish":
            source.write_text(source.read_text() + " ")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "uncertain"
    assert result.capability == "none"
    assert any(
        item["code"] == "post_publish_source_continuity_lost"
        for item in result.problems
    )


def test_portfolio_mutation_after_publish_is_uncertain(cycle_factory):
    root, qlib, request = cycle_factory()
    portfolio = root / "config/prod_config.json"

    def fault(point):
        if point == "after_publish":
            portfolio.write_text(portfolio.read_text() + " ")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "uncertain"
    assert result.capability == "none"


def test_process_interruption_propagates_and_claims_no_success(cycle_factory):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "before_publish":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib, fault_hook=fault).capture(request)
    assert not (root / "data/evidence/v1/cycles/2099-01-02").exists()


def test_post_publish_parent_loss_is_uncertain_without_capability(cycle_factory):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "after_publish":
            cycles = root / "data/evidence/v1/cycles"
            cycles.rename(root / "data/evidence/v1/displaced-cycles")

    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib, fault_hook=fault).capture(request)
    assert result.status == "uncertain"
    assert result.seal_digest is None


def test_post_publish_path_boundary_exception_is_uncertain(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    original = sealing_module._directory_chain
    after_publish = False

    def fault(point):
        nonlocal after_publish
        if point == "after_publish":
            after_publish = True

    def fail_after_publish(*args, **kwargs):
        if after_publish:
            raise sealing_module.PathBoundaryError("post-publish boundary drift")
        return original(*args, **kwargs)

    monkeypatch.setattr(sealing_module, "_directory_chain", fail_after_publish)
    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "uncertain"
    assert result.did_write is True
    assert result.write_scope == "final_uncertain"
    assert result.capability == "none"


def test_post_publish_public_name_replacement_is_uncertain(cycle_factory):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "after_publish":
            final = root / "data/evidence/v1/cycles/2099-01-02"
            displaced = final.with_name("displaced-bundle")
            final.rename(displaced)
            final.symlink_to(displaced, target_is_directory=True)

    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib, fault_hook=fault).capture(request)
    assert result.status == "uncertain"
    assert result.capability == "none"


def test_post_publish_sealed_member_tamper_is_uncertain(cycle_factory):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "after_publish":
            ranking = root / "data/evidence/v1/cycles/2099-01-02/ranking.csv"
            ranking.write_text("tampered\n")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "uncertain"
    assert result.capability == "none"
    assert any(
        item["code"] == "post_publish_verification_failed"
        for item in result.problems
    )


def test_member_tamper_after_first_post_publish_verification_is_uncertain(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    final = root / "data/evidence/v1/cycles/2099-01-02"
    original = ProductionCycleEvidenceSealer._existing
    final_calls = 0

    def tamper_after_first_final(self, path, *args, **kwargs):
        nonlocal final_calls
        result = original(self, path, *args, **kwargs)
        if path == final:
            final_calls += 1
            if final_calls == 1:
                (final / "ranking.csv").write_text("tampered\n")
        return result

    monkeypatch.setattr(
        ProductionCycleEvidenceSealer, "_existing", tamper_after_first_final,
    )
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "uncertain"
    assert result.capability == "none"
    assert any(
        item["code"] == "post_publish_confirmation_lost"
        for item in result.problems
    )


def test_transient_valid_final_cannot_mask_tampered_post_publish_bundle(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    final = root / "data/evidence/v1/cycles/2099-01-02"
    valid = final.with_name("valid-final")
    tampered = final.with_name("tampered-final")
    original = ProductionCycleEvidenceSealer._existing

    def fault(point):
        if point == "after_publish":
            shutil.copytree(final, valid)
            (final / "ranking.csv").write_text("tampered\n")

    def verify_transient_valid_copy(self, *args, **kwargs):
        if args[0] != final or not valid.exists():
            return original(self, *args, **kwargs)
        final.rename(tampered)
        valid.rename(final)
        try:
            return original(self, *args, **kwargs)
        finally:
            final.rename(valid)
            tampered.rename(final)

    monkeypatch.setattr(
        ProductionCycleEvidenceSealer, "_existing", verify_transient_valid_copy,
    )
    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "uncertain"
    assert result.capability == "none"
    assert (final / "ranking.csv").read_text() == "tampered\n"


def test_staging_public_name_replacement_before_publish_prevents_final(cycle_factory):
    root, qlib, request = cycle_factory()
    displaced = None

    def fault(point):
        nonlocal displaced
        if point == "before_publish":
            staging = root / "data/evidence/v1/.staging"
            stage = next(staging.iterdir())
            displaced = stage.with_name(stage.name + ".displaced")
            stage.rename(displaced)
            stage.mkdir()

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"
    assert not (root / "data/evidence/v1/cycles/2099-01-02").exists()
    assert displaced is not None and displaced.exists()


def test_symlink_source_escape_is_blocked(cycle_factory, tmp_path):
    root, qlib, request = cycle_factory()
    outside = tmp_path / "foreign.json"
    outside.write_text("{}")
    alias = root / "foreign.json"
    alias.symlink_to(outside)
    request = replace(request, order_manifest="foreign.json")
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "blocked"
    assert result.did_write is False


def test_ensemble_artifact_escape_is_blocked_without_final(cycle_factory, tmp_path):
    root, qlib, request = cycle_factory()
    outside = tmp_path / "foreign-recorder"
    outside.mkdir()
    (outside / "pred.pkl").write_bytes(b"foreign")
    path = root / request.ensemble_manifest
    value = json.loads(path.read_text())
    value["records"]["combos"][0]["output_evidence"]["artifact_path"] = str(outside)
    path.write_text(json.dumps(value))
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "blocked"
    assert result.did_write is False
    assert not (root / "data/evidence").exists()


def test_symlinked_evidence_parent_is_blocked_before_outside_write(cycle_factory, tmp_path):
    root, qlib, request = cycle_factory()
    outside = tmp_path / "outside-evidence"
    outside.mkdir()
    (root / "data/evidence").symlink_to(outside, target_is_directory=True)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "blocked"
    assert list(outside.iterdir()) == []


def test_foreign_existing_lock_is_blocked_and_never_removed(cycle_factory):
    root, qlib, request = cycle_factory()
    lock = root / "data/evidence/v1/.locks/2099-01-02.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("foreign-owner")
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "blocked"
    assert lock.read_text() == "foreign-owner"


@pytest.mark.parametrize("fault_point,expected_status", [
    ("before_publish", "failed_no_final"),
    ("after_publish", "uncertain"),
])
def test_replaced_owned_lock_fails_closed_and_preserves_foreign_lock(
    cycle_factory, fault_point, expected_status,
):
    root, qlib, request = cycle_factory()
    lock = root / "data/evidence/v1/.locks/2099-01-02.lock"

    def fault(point):
        if point == fault_point:
            lock.unlink()
            lock.write_text("foreign-owner")

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == expected_status
    assert result.capability == "none"
    assert lock.read_text() == "foreign-owner"


def test_lock_cleanup_failure_after_publish_is_uncertain(
    cycle_factory, monkeypatch,
):
    root, qlib, request = cycle_factory()
    monkeypatch.setattr(sealing_module, "_release_owned_lock", lambda *_args: False)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "uncertain"
    assert result.capability == "none"
    assert any(item["code"] == "lock_cleanup_failed" for item in result.problems)


def test_publication_ancestor_symlink_replacement_fails_before_publish(cycle_factory):
    root, qlib, request = cycle_factory()

    def fault(point):
        if point == "before_publish":
            evidence = root / "data/evidence"
            current = evidence / "v1"
            displaced = evidence / "displaced-v1"
            current.rename(displaced)
            current.symlink_to(displaced, target_is_directory=True)

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"
    assert not (root / "data/evidence/displaced-v1/cycles/2099-01-02").exists()


def test_research_epoch_change_on_replay_conflicts(cycle_factory):
    root, qlib, request = cycle_factory()
    sealer = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib)
    assert sealer.capture(request).status == "sealed_complete"
    changed = replace(request, research_epoch_id="SYNTHETIC_V2")
    assert sealer.capture(changed).status == "conflict"


def test_workspace_git_mutation_during_capture_is_visible_and_partial(cycle_factory, monkeypatch):
    root, qlib, request = cycle_factory()
    original = ProductionCycleEvidenceSealer._ranking

    def mutating(self, *args, **kwargs):
        (root / "output/operator-race.txt").write_text("changed during observation\n")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ProductionCycleEvidenceSealer, "_ranking", mutating)
    result = ProductionCycleEvidenceSealer(root, qlib_data_dir=qlib).capture(request)
    assert result.status == "sealed_partial"
    assert any(item["code"] == "workspace_git_mutated" for item in result.problems)


def test_transient_git_head_mutation_before_publish_denies_capability(cycle_factory):
    root, qlib, request = cycle_factory()
    head = root / ".synthetic-git-control" / "HEAD"
    original = head.read_bytes()

    def fault(point):
        if point == "before_publish":
            head.write_bytes(b"ref: refs/heads/transient\n")
            head.write_bytes(original)

    result = ProductionCycleEvidenceSealer(
        root, qlib_data_dir=qlib, fault_hook=fault,
    ).capture(request)
    assert result.status == "failed_no_final"
    assert result.capability == "none"
    assert any(
        item["code"] == "source_continuity_lost"
        for item in result.problems
    )
