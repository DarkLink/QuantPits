import copy
import json
import os
import shutil
import stat

import pytest

import quantpits.research.model_artifact_capsule as module
from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from tests.quantpits.research.test_forward_definition_evidence import (
    CYCLE, _fresh_publish, fresh_evidence_workspace,
)


def _roots(tmp_path):
    production = tmp_path / "production"
    research = tmp_path / "research-workspace"
    shadow = research / "research" / "shadow_v1"
    activation = shadow / "activations" / "shadow.fresh.segment.v1.json"
    definitions = shadow / "definitions"
    evidence = shadow / "definition_evidence"
    capsules = shadow / "model_artifact_capsules"
    production.mkdir()
    for directory in (research, shadow.parent, shadow, activation.parent, definitions, evidence, capsules):
        directory.mkdir(exist_ok=True)
        if directory != research:
            directory.chmod(0o700)
    activation.write_bytes(b"{}")
    activation.chmod(0o600)
    return production, research, activation, definitions, evidence, capsules


def _request():
    values = tuple(("model-%d" % index).encode() for index in range(4))
    metadata = tuple(("meta-%d" % index).encode() for index in range(4))
    config = b"config-a"
    members = tuple(module._payload_member(
        "SOURCE_TRAINING_ARTIFACT", index,
        "source-%d-artifact-000000" % index,
        "mlruns/%d/run/artifacts/model.bin" % index, values[index],
        "embedded" if index % 2 == 0 else "workspace_file",
    ) for index in range(4)) + (module._payload_member(
        "MODEL_CONFIG", None, "config-000000", "config/model.json",
        config, "embedded",
    ),) + tuple(module._payload_member(
        "SOURCE_RUN_METADATA", index,
        "source-%d-metadata-000000" % index,
        "mlruns/%d/run/meta.yaml" % index, metadata[index], "source_run",
    ) for index in range(4))
    payload_data, payload = module._payload_bytes(members)
    digest = TypedDigest.raw(b"authority").to_dict()
    canonical_digest = TypedDigest.canonical({"authority": True}).to_dict()
    authority = {
        "definition_set_id": "shadow.fresh.segment.v1",
        "definition_request_digest": canonical_digest,
        "definition_evidence_request_digest": canonical_digest,
        "definition_evidence_manifest_digest": digest,
        "definition_evidence_operation_id": "a" * 64,
        "champion_source_ids": ["A", "B", "C", "D"],
        "challenger_source_ids": ["A", "B", "D"],
        "omitted_champion_position": 2,
    }
    receipt = {
        "schema_version": 1,
        "source_claim": "EXACT_DEFINITION_BOUND_PHASE37A_MODEL_SOURCES_V1",
        "evidence_cycle_id": "2026-08-28",
        **authority,
        "phase37a_status": "sealed_complete",
        "phase37a_problems": [],
        "phase37a_seal_digest": digest,
        "phase37a_manifest_digest": digest,
        "source_partition": {
            "source_training_count": 4, "prediction_join_count": 4,
            "ensemble_join_count": 1, "unassigned_count": 0,
        },
        "source_training_tree_digests": [
            TypedDigest.canonical([], "file_inventory").to_dict()
            for _ in range(4)
        ],
        "prediction_tree_digests": [
            TypedDigest.canonical([], "file_inventory").to_dict()
            for _ in range(4)
        ],
        "ensemble_tree_digest": TypedDigest.canonical([], "file_inventory").to_dict(),
        "source_run_metadata_inventory_digests": [
            TypedDigest.canonical([{
                "path": "mlruns/%d/run/meta.yaml" % position,
                "digest": TypedDigest.raw(metadata[position]).to_dict(),
            }], "file_inventory").to_dict()
            for position in range(4)
        ],
        "run_metadata_claims": [{
            "recorder_join_verified": True,
            "source_member_mode": "LEGACY_UNQUALIFIED",
            "source_member_mode_authority": "SEALED_SOURCE_MEMBER_ID",
            "source_member_mode_bound": True,
            "run_metadata_mode_consistency": "NOT_APPLICABLE",
            "fit_start_time": "2020-01-01",
            "fit_end_time": "2026-08-28", "training_window_verified": True,
        } for _ in range(4)],
        "raw_member_count": len(members),
        "total_raw_bytes": sum(len(value) for value in values + metadata) + len(config),
        "embedded_source_count": 3,
        "live_source_count": 6,
        "payload_digest": TypedDigest.raw(payload_data).to_dict(),
        "definition_bound": True,
        "source_observation_complete": True,
        "did_write": False,
    }
    return module._Request(
        "2026-08-28", authority, receipt, payload_data, payload,
        _authority=module._AUTHORITY,
    )


def test_payload_base64_digest_order_and_budgets_are_strict(monkeypatch):
    request = _request()
    decoded = module._validate_payload(request.payload)
    assert len(decoded) == 9
    for mutation in ("bool_schema", "bad_base64", "duplicate", "reorder"):
        payload = copy.deepcopy(request.payload)
        if mutation == "bool_schema":
            payload["schema_version"] = True
        elif mutation == "bad_base64":
            payload["members"][0]["content_base64"] = "***"
        elif mutation == "duplicate":
            payload["members"][1]["logical_slot"] = payload["members"][0]["logical_slot"]
        else:
            payload["members"][0], payload["members"][1] = payload["members"][1], payload["members"][0]
        if mutation != "bad_base64":
            core = {key: value for key, value in payload.items() if key != "payload_digest"}
            payload["payload_digest"] = TypedDigest.canonical(core).to_dict()
        with pytest.raises(module.ModelArtifactCapsuleContractError):
            module._validate_payload(payload)
    monkeypatch.setattr(module, "MAX_TOTAL_RAW_BYTES", 2)
    with pytest.raises(module.ModelArtifactCapsuleContractError, match="budget"):
        module._validate_payload(request.payload)


def test_phase37_partition_is_exact_and_challenger_is_not_caller_controlled(monkeypatch):
    authority = {
        "champion_source_ids": ["A", "B", "C", "D"],
        "challenger_source_ids": ["A", "C", "D"],
    }
    rows = []
    models = []
    digest = TypedDigest.canonical([], "file_inventory").to_dict()
    for position, source in enumerate(authority["champion_source_ids"]):
        models.append({
            "resolved_key": source, "recorder_id": "P%d" % position,
            "source_recorder_id": "S%d" % position,
        })
        rows.extend(({
            "position": position, "role": "source_training",
            "recorder_id": "S%d" % position,
            "artifact_tree_digest": digest,
        }, {
            "position": position, "recorder_id": "P%d" % position,
            "source_recorder_id": "S%d" % position,
            "artifact_tree_digest": digest,
        }))
    rows.append({
        "position": "ensemble", "recorder_id": "E",
        "artifact_tree_digest": digest,
    })
    manifest = {"model_and_ensemble_lineage": {
        "status": "complete",
        "combo": {"resolved_members": ["A", "B", "C", "D"]},
        "source_models": models, "source_artifacts": rows,
    }}
    monkeypatch.setattr(module._phase37, "_source_projection", lambda _value: {
        "members": [{"source_id": value} for value in authority["champion_source_ids"]],
    })
    result = module._partition(manifest, authority)
    assert len(result["training"]) == 4
    assert len(result["predictions"]) == 4
    invalid = copy.deepcopy(manifest)
    invalid["model_and_ensemble_lineage"]["source_artifacts"].append(None)
    with pytest.raises(module._Blocked, match="PARTITION"):
        module._partition(invalid, authority)
    invalid = copy.deepcopy(manifest)
    invalid["model_and_ensemble_lineage"]["status"] = "partial"
    with pytest.raises(module._Blocked, match="PARTITION"):
        module._partition(invalid, authority)


def test_phase37_partial_is_scoped_to_deep_analysis_and_preserved_as_partial():
    allowed = [{
        "code": "deep_analysis_missing",
        "evidence_class": "deep_analysis",
        "detail": "no Deep Analysis run was provided",
        "blocks_complete": True,
    }]
    module._validate_phase37a_status("sealed_partial", allowed)
    for status, problems in (
        ("sealed_complete", allowed),
        ("sealed_partial", [{
            "code": "model_lineage_missing",
            "evidence_class": "model_lineage",
            "detail": "model lineage is absent",
            "blocks_complete": True,
        }]),
        ("sealed_partial", allowed + allowed),
        (True, []),
        ([], []),
    ):
        with pytest.raises(module.ModelArtifactCapsuleContractError, match="Phase37A"):
            module._validate_phase37a_status(status, problems)


def test_selected_configuration_is_embedded_deduplicated_and_conflict_visible():
    data = b"config"
    semantic = TypedDigest.raw(data).to_dict()
    row = {
        "evidence_class": "prediction", "collection": "inputs", "kind": "config",
        "locator": "config/model.json", "locator_type": "workspace_file",
        "observation": {
            "status": "observed", "preservation_status": "embedded",
            "digest": semantic,
        },
    }
    ensemble_config = copy.deepcopy(row)
    ensemble_config.update({
        "evidence_class": "ensemble", "locator": "config/ensemble_config.json",
    })
    ensemble_records = copy.deepcopy(row)
    ensemble_records.update({
        "evidence_class": "order", "kind": "record",
        "locator": "config/ensemble_records.json",
    })
    manifest = {"referenced_evidence": [
        row, copy.deepcopy(row), ensemble_config, ensemble_records,
    ]}
    selected = module._selected_references(manifest)
    assert len(selected) == 3
    assert selected[0]["digest"] == TypedDigest.raw(data).to_dict()
    manifest["referenced_evidence"][1]["observation"]["digest"]["value"] = "0" * 64
    with pytest.raises(module._Blocked, match="CONFLICT"):
        module._selected_references(manifest)


def test_run_metadata_binds_sealed_mode_and_training_window(tmp_path):
    run = tmp_path / "RUN"
    paths = (run / "meta.yaml", run / "params" / "fit_start_time",
             run / "params" / "fit_end_time", run / "params" / "training_mode")
    values = {
        paths[0]: ("run_id: RUN\nexperiment_id: %s\n" % run.parent.name).encode(),
        paths[1]: b"2020-01-01\n",
        paths[2]: b"2026-08-28\n",
        paths[3]: b"FULL\n",
    }
    claim = module._run_metadata_claim(run, paths, "RUN", "MODEL@FULL", values)
    assert claim["source_member_mode"] == "FULL"
    assert claim["source_member_mode_authority"] == "SEALED_SOURCE_MEMBER_ID"
    assert claim["source_member_mode_bound"] is True
    assert claim["run_metadata_mode_consistency"] == "CONSISTENT"
    invalid = dict(values)
    invalid[paths[3]] = b"CPCV\n"
    with pytest.raises(module._Blocked, match="MODE"):
        module._run_metadata_claim(run, paths, "RUN", "MODEL@FULL", invalid)


def test_run_metadata_accepts_absent_mode_without_forging_a_param_join(tmp_path):
    run = tmp_path / "RUN"
    paths = (run / "meta.yaml", run / "params" / "fit_start_time",
             run / "params" / "fit_end_time")
    values = {
        paths[0]: ("run_id: RUN\nexperiment_id: %s\n" % run.parent.name).encode(),
        paths[1]: b"2020-01-01\n",
        paths[2]: b"2026-08-28\n",
    }
    claim = module._run_metadata_claim(run, paths, "RUN", "MODEL@static", values)
    assert claim["source_member_mode"] == "static"
    assert claim["source_member_mode_authority"] == "SEALED_SOURCE_MEMBER_ID"
    assert claim["source_member_mode_bound"] is True
    assert claim["run_metadata_mode_consistency"] == "NOT_DECLARED"


def test_run_metadata_rejects_ambiguous_mode_declarations(tmp_path):
    run = tmp_path / "RUN"
    paths = (run / "meta.yaml", run / "params" / "fit_start_time",
             run / "params" / "fit_end_time", run / "params" / "mode",
             run / "params" / "training_mode")
    values = {
        paths[0]: ("run_id: RUN\nexperiment_id: %s\n" % run.parent.name).encode(),
        paths[1]: b"2020-01-01\n",
        paths[2]: b"2026-08-28\n",
        paths[3]: b"static\n",
        paths[4]: b"static\n",
    }
    with pytest.raises(module._Blocked, match="MODE"):
        module._run_metadata_claim(run, paths, "RUN", "MODEL@static", values)


def test_receipt_rejects_forged_sealed_mode_authority_and_status():
    request = _request()
    for field, value in (
        ("source_member_mode_authority", "RUN_PARAM"),
        ("source_member_mode_bound", False),
        ("run_metadata_mode_consistency", "NOT_DECLARED"),
    ):
        receipt = copy.deepcopy(request.receipt)
        receipt["run_metadata_claims"][0][field] = value
        with pytest.raises(module.ModelArtifactCapsuleContractError, match="claim"):
            module._validate_receipt(
                receipt, request.authority, request.payload_data,
                request.payload, request.cycle_id,
            )


def test_store_commits_three_files_manifest_last_and_exactly_adopts(tmp_path):
    root = tmp_path / "capsules"
    root.mkdir(mode=0o700)
    request = _request()
    first = module._publish_store(root, request)
    assert first.status == "COMMITTED"
    assert first.did_write is True
    assert first.model_artifact_retention_complete is True
    target = root / request.capsule_id
    assert tuple(sorted(path.name for path in target.iterdir())) == module.FINAL_NAMES
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in target.iterdir())
    manifest = json.loads((target / module.MANIFEST_NAME).read_bytes())
    assert manifest["members"][0]["logical_path"] == module.SOURCE_RECEIPT_NAME
    assert manifest["members"][1]["logical_path"] == module.PAYLOAD_NAME
    before = tuple((path.name, path.stat().st_ino) for path in target.iterdir())
    second = module._publish_store(root, request)
    assert second.status == "ADOPTED" and second.did_write is False
    assert before == tuple((path.name, path.stat().st_ino) for path in target.iterdir())


def test_store_conflict_special_partial_and_post_create_failure_fail_closed(tmp_path, monkeypatch):
    request = _request()
    for kind in ("partial", "special"):
        root = tmp_path / kind
        root.mkdir(mode=0o700)
        target = root / request.capsule_id
        target.mkdir(mode=0o700)
        if kind == "partial":
            (target / module.SOURCE_RECEIPT_NAME).write_bytes(b"{}")
            (target / module.SOURCE_RECEIPT_NAME).chmod(0o600)
        else:
            (target / module.SOURCE_RECEIPT_NAME).mkdir()
        result = module._publish_store(root, request)
        assert result.status in {"CONFLICT", "UNCERTAIN"}
        assert result.model_artifact_retention_complete is False
    root = tmp_path / "failure"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(module, "_write_all", lambda *_args: (_ for _ in ()).throw(OSError("fault")))
    result = module._publish_store(root, request)
    assert result.status == "UNCERTAIN" and result.did_write is True
    assert result.model_artifact_retention_complete is False


@pytest.mark.parametrize("mutation", ["target_move_back", "member_recreate", "manifest_move_back"])
def test_store_rejects_namespace_mutation_during_every_create_interval(
    tmp_path, monkeypatch, mutation,
):
    root = tmp_path / mutation
    root.mkdir(mode=0o700)
    request = _request()
    target = root / request.capsule_id
    if mutation in {"target_move_back", "member_recreate"}:
        original = module._write_all
        calls = []
        def mutate(fd, data):
            original(fd, data)
            if calls:
                return
            calls.append(True)
            if mutation == "target_move_back":
                displaced = root / "displaced"
                target.rename(displaced)
                displaced.rename(target)
            else:
                victim = target / module.SOURCE_RECEIPT_NAME
                victim.unlink()
                victim.write_bytes(data)
                victim.chmod(0o600)
        monkeypatch.setattr(module, "_write_all", mutate)
    else:
        original_verify = module._verify_bundle
        def verify(path, value):
            victim = path / module.MANIFEST_NAME
            displaced = path / "manifest.displaced"
            victim.rename(displaced)
            displaced.rename(victim)
            return original_verify(path, value)
        monkeypatch.setattr(module, "_verify_bundle", verify)
    result = module._publish_store(root, request)
    assert result.status == "UNCERTAIN"
    assert result.did_write is True
    assert result.model_artifact_retention_complete is False


def test_existing_target_mutation_during_final_verification_is_not_adopted(
    tmp_path, monkeypatch,
):
    root = tmp_path / "capsules"
    root.mkdir(mode=0o700)
    request = _request()
    assert module._publish_store(root, request).status == "COMMITTED"
    original = module._verify_bundle
    def mutate(target, value):
        victim = target / module.PAYLOAD_NAME
        displaced = target / "payload.displaced"
        victim.rename(displaced)
        displaced.rename(victim)
        return original(target, value)
    monkeypatch.setattr(module, "_verify_bundle", mutate)
    result = module._publish_store(root, request)
    assert result.status == "UNCERTAIN"
    assert result.did_write is False
    assert result.model_artifact_retention_complete is False


def test_prepare_repeats_observation_and_publish_joins_owner_before_store(tmp_path, monkeypatch):
    paths = _roots(tmp_path)
    request = _request()
    calls = {"derive": 0, "store": 0}
    class Adopted:
        status = "ADOPTED"
    def derive(*_args):
        calls["derive"] += 1
        return request, Adopted()
    monkeypatch.setattr(module, "_assemble_live", derive)
    plan = module.prepare_definition_bound_model_artifact_capsule(
        paths[0], paths[1], "2026-08-28", *paths[2:],
    )
    assert plan.target_state == "ABSENT" and calls["derive"] == 2
    original = module._publish_store
    def store(*args):
        calls["store"] += 1
        return original(*args)
    monkeypatch.setattr(module, "_publish_store", store)
    bad = dict(plan.request_digest)
    bad["value"] = "0" * 64
    with pytest.raises(module.ModelArtifactCapsuleContractError):
        module.publish_definition_bound_model_artifact_capsule(
            paths[0], paths[1], "2026-08-28", *paths[2:],
            plan.capsule_id, bad, module.AUTHORIZATION_ACTION,
        )
    assert calls["store"] == 0


def test_precondition_blocked_and_absent_adopt_are_zero_write_capability_denials(
    tmp_path, monkeypatch,
):
    paths = _roots(tmp_path)
    before = module._protected_inventory(paths[1], None)
    monkeypatch.setattr(
        module, "_assemble_live",
        lambda *_args: (_ for _ in ()).throw(module._Blocked("SOURCE_MISSING")),
    )
    plan = module.prepare_definition_bound_model_artifact_capsule(
        paths[0], paths[1], "2026-08-28", *paths[2:],
    )
    assert plan.status == "PRECONDITION_BLOCKED"
    assert plan.to_safe_summary_dict()["model_artifact_retention_complete"] is False
    absent = module.adopt_definition_bound_model_artifact_capsule(
        paths[0], paths[1], "2026-08-28", *paths[2:],
        "modelcapsule." + "0" * 64,
    )
    assert absent.status == "CONFLICT"
    assert absent.did_write is False
    assert absent.model_artifact_retention_complete is False
    assert module._protected_inventory(paths[1], None) == before


def test_post_writer_source_failure_preserves_write_fact_and_denies_capability(
    tmp_path, monkeypatch,
):
    paths = _roots(tmp_path)
    request = _request()
    calls = []
    class Adopted:
        status = "ADOPTED"
    def derive(*_args):
        calls.append(True)
        if len(calls) == 2:
            raise module._Blocked("SOURCE_DRIFT")
        return request, Adopted()
    monkeypatch.setattr(module, "_assemble_live", derive)
    result = module.publish_definition_bound_model_artifact_capsule(
        paths[0], paths[1], "2026-08-28", *paths[2:], request.capsule_id,
        request.request_digest, module.AUTHORIZATION_ACTION,
    )
    assert result.status == "UNCERTAIN"
    assert result.did_write is True
    assert result.model_artifact_retention_complete is False


def test_store_to_outer_handoff_has_no_target_member_continuity_gap(
    tmp_path, monkeypatch,
):
    paths = _roots(tmp_path)
    request = _request()
    calls = []
    class Adopted:
        status = "ADOPTED"
    def derive(*_args):
        calls.append(True)
        if len(calls) == 2:
            target = paths[-1] / request.capsule_id
            victim = target / module.PAYLOAD_NAME
            displaced = target / "payload.displaced"
            victim.rename(displaced)
            displaced.rename(victim)
        return request, Adopted()
    monkeypatch.setattr(module, "_assemble_live", derive)
    result = module.publish_definition_bound_model_artifact_capsule(
        paths[0], paths[1], "2026-08-28", *paths[2:], request.capsule_id,
        request.request_digest, module.AUTHORIZATION_ACTION,
    )
    assert result.status == "UNCERTAIN"
    assert result.did_write is True
    assert result.model_artifact_retention_complete is False


def test_adopt_observer_covers_request_read_to_final_store_replay(
    tmp_path, monkeypatch,
):
    paths = _roots(tmp_path)
    request = _request()
    assert module._publish_store(paths[-1], request).status == "COMMITTED"
    class Adopted:
        status = "ADOPTED"
    monkeypatch.setattr(
        module, "_request_from_target",
        lambda *_args: (request, Adopted()),
    )
    original = module._adopt_store
    def mutate(root, value):
        target = root / value.capsule_id
        victim = target / module.PAYLOAD_NAME
        displaced = target / "payload.displaced"
        victim.rename(displaced)
        displaced.rename(victim)
        return original(root, value)
    monkeypatch.setattr(module, "_adopt_store", mutate)
    result = module.adopt_definition_bound_model_artifact_capsule(
        paths[0], paths[1], "2026-08-28", *paths[2:], request.capsule_id,
    )
    assert result.status == "UNCERTAIN"
    assert result.did_write is False
    assert result.model_artifact_retention_complete is False


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_live_source_alias_hardlink_and_special_node_fail_without_blocking(
    tmp_path, kind,
):
    source = tmp_path / "source"
    source.write_bytes(b"value")
    candidate = tmp_path / "candidate"
    if kind == "symlink":
        candidate.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, candidate)
    else:
        os.mkfifo(candidate)
    with pytest.raises(module._Blocked):
        module._read_regular(candidate, 1024)


def test_ordinary_member_failure_still_observes_later_requested_members(
    tmp_path, monkeypatch,
):
    paths = _roots(tmp_path)
    production = paths[0]
    cycle_path = production / "data" / "evidence" / "v1" / "cycles" / "2026-08-28"
    (cycle_path / "objects").mkdir(parents=True)
    config = b"config"
    config_digest = TypedDigest.raw(config).to_dict()
    config_path = cycle_path / "objects" / config_digest["value"][:2] / config_digest["value"]
    config_path.parent.mkdir()
    config_path.write_bytes(config)
    training = []
    metadata_roots = {}
    artifact_paths = []
    tree = TypedDigest.canonical([], "file_inventory").to_dict()
    for position in range(4):
        artifact_root = production / "mlruns" / str(position) / ("S%d" % position) / "artifacts"
        artifact_root.mkdir(parents=True)
        member_path = artifact_root / "model.bin"
        data = ("model-%d" % position).encode()
        member_path.write_bytes(data)
        artifact_paths.append(member_path)
        training.append({
            "artifact_locator": artifact_root.relative_to(production).as_posix(),
            "recorder_id": "S%d" % position,
            "artifact_tree_digest": tree,
            "members": [{
                "path": member_path.relative_to(production).as_posix(),
                "status": "observed", "digest": TypedDigest.raw(data).to_dict(),
                "preservation_status": "workspace_file", "detail": "",
            }],
        })
        run = artifact_root.parent
        (run / "params").mkdir()
        (run / "tags").mkdir()
        (run / "meta.yaml").write_text(
            "run_id: S%d\nexperiment_id: %s\n" % (position, run.parent.name),
        )
        (run / "params" / "fit_start_time").write_text("2020-01-01\n")
        (run / "params" / "fit_end_time").write_text("2026-08-28\n")
        (run / "tags" / "source").write_text("synthetic\n")
        metadata_roots[position] = run
    authority = {
        "definition_set_id": "shadow.fresh.segment.v1",
        "definition_request_digest": TypedDigest.canonical({}).to_dict(),
        "definition_evidence_request_digest": TypedDigest.canonical({}).to_dict(),
        "definition_evidence_manifest_digest": TypedDigest.raw(b"evidence").to_dict(),
        "definition_evidence_operation_id": "a" * 64,
        "champion_source_ids": ["A", "B", "C", "D"],
        "challenger_source_ids": ["A", "C", "D"],
        "omitted_champion_position": 1,
    }
    class Adopted:
        status = "ADOPTED"
    monkeypatch.setattr(module, "_definition_authority", lambda *_args: (Adopted(), authority))
    monkeypatch.setattr(module._phase37, "_cycle_authority", lambda *_args: (cycle_path, {}, {}))
    monkeypatch.setattr(module._phase37, "_read_regular", lambda path, **_kwargs: (path.name.encode(), ()))
    monkeypatch.setattr(module, "_partition", lambda *_args: {
        "training": tuple(training), "predictions": tuple({"artifact_tree_digest": tree} for _ in range(4)),
        "ensemble": {"artifact_tree_digest": tree},
    })
    monkeypatch.setattr(module, "_selected_references", lambda _manifest: ({
        "locator": "config/model.json", "digest": config_digest,
    },))
    def metadata(_production, _row, position):
        run = metadata_roots[position]
        return run, (
            run / "meta.yaml", run / "params" / "fit_start_time",
            run / "params" / "fit_end_time", run / "tags" / "source",
        )
    monkeypatch.setattr(module, "_metadata_paths", metadata)
    original = module._read_regular
    observed = []
    def read(path, *args, **kwargs):
        observed.append(path)
        if path == artifact_paths[0]:
            raise module._Blocked("INJECTED_MEMBER_FAILURE")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module, "_read_regular", read)
    guard = module.SourceMutationObserver(
        production, ("data/evidence/v1/cycles/2026-08-28",),
    )
    try:
        with pytest.raises(module._Blocked, match="INJECTED"):
            module._assemble_live(paths, "2026-08-28", guard)
    finally:
        guard.close()
    assert artifact_paths[-1] in observed
    assert metadata_roots[3] / "tags" / "source" in observed


def test_plan_result_and_request_authority_cannot_be_publicly_replayed_or_mutated(tmp_path):
    request = _request()
    root = tmp_path / "capsules"
    root.mkdir(mode=0o700)
    result = module._publish_store(root, request)
    with pytest.raises(module.ModelArtifactCapsuleContractError, match="inspector-owned"):
        module.ModelArtifactCapsulePlan(request=request, target_state="PRESENT")
    with pytest.raises(module.ModelArtifactCapsuleContractError, match="writer-owned"):
        module.ModelArtifactCapsuleResult(**result.to_safe_summary_dict())
    request.payload["members"][0]["size_bytes"] += 1
    with pytest.raises(module.ModelArtifactCapsuleContractError):
        result.to_safe_summary_dict()


def test_physical_root_overlap_is_rejected_before_source_observation(tmp_path, monkeypatch):
    production = tmp_path / "production"
    research = production / "nested"
    shadow = research / "research" / "shadow_v1"
    activation = shadow / "activations" / "x.json"
    for directory in (production, research, shadow.parent, shadow, activation.parent,
                      shadow / "definitions", shadow / "definition_evidence",
                      shadow / "model_artifact_capsules"):
        directory.mkdir(exist_ok=True)
        if directory not in (production, research):
            directory.chmod(0o700)
    activation.write_bytes(b"{}")
    activation.chmod(0o600)
    monkeypatch.setattr(module, "_assemble_live", lambda *_args: pytest.fail("must not observe"))
    with pytest.raises(module.ModelArtifactCapsuleContractError, match="separated"):
        module.prepare_definition_bound_model_artifact_capsule(
            production, research, "2026-08-28", activation,
            shadow / "definitions", shadow / "definition_evidence",
            shadow / "model_artifact_capsules",
        )


def test_process_control_propagates_at_writer_seam(tmp_path, monkeypatch):
    root = tmp_path / "capsules"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(module, "_write_all", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        module._publish_store(root, _request())


def test_real_definition_evidence_and_sealed_cycle_join_into_live_request(
    fresh_evidence_workspace, monkeypatch,
):
    assert _fresh_publish(fresh_evidence_workspace).status == "COMMITTED"
    production, research, activation, definitions, evidence = fresh_evidence_workspace
    capsules = research / "research" / "shadow_v1" / "model_artifact_capsules"
    capsules.mkdir(mode=0o700)
    cycle_path = production / "data" / "evidence" / "v1" / "cycles" / CYCLE
    manifest = json.loads((cycle_path / "manifest.json").read_bytes())
    first_member = manifest["model_and_ensemble_lineage"]["source_artifacts"][0]["members"][0]
    monkeypatch.setattr(module, "_selected_references", lambda _manifest: ({
        "locator": "config/model.json", "digest": first_member["digest"],
    },))
    roots = {}
    for position in range(4):
        run = production / "runs" / ("SOURCE_%d" % position)
        (run / "params").mkdir(parents=True)
        (run / "tags").mkdir()
        (run / "meta.yaml").write_text(
            "run_id: SOURCE_%d\nexperiment_id: %s\n"
            % (position, run.parent.name),
        )
        (run / "params" / "fit_start_time").write_text("2020-01-01\n")
        (run / "params" / "fit_end_time").write_text(CYCLE + "\n")
        (run / "tags" / "source").write_text("synthetic\n")
        roots[position] = run
    def metadata(_production, _row, position):
        run = roots[position]
        return run, (
            run / "meta.yaml", run / "params" / "fit_end_time",
            run / "params" / "fit_start_time", run / "tags" / "source",
        )
    monkeypatch.setattr(module, "_metadata_paths", metadata)
    before = module._protected_inventory(production, None), module._protected_inventory(research, None)
    plan = module.prepare_definition_bound_model_artifact_capsule(
        production, research, CYCLE, activation, definitions, evidence, capsules,
    )
    assert plan.status == "PREPARED"
    assert plan.target_state == "ABSENT"
    assert plan.to_safe_summary_dict()["champion_source_count"] == 4
    assert before == (
        module._protected_inventory(production, None),
        module._protected_inventory(research, None),
    )
    assert module._publish_store(capsules, plan._request).status == "COMMITTED"
    shutil.rmtree(production / "runs")
    adopted = module.adopt_definition_bound_model_artifact_capsule(
        production, research, CYCLE, activation, definitions, evidence,
        capsules, plan.capsule_id,
    )
    assert adopted.status == "ADOPTED"
    assert adopted.did_write is False
    assert adopted.model_artifact_retention_complete is True
