import copy
import os
from pathlib import Path

import pytest

import quantpits.research.signal_input_capsule as module
from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes


def _roots(tmp_path):
    production = tmp_path / "production"
    research = tmp_path / "research-workspace"
    store = research / "research" / "shadow_v1" / "signal_input_capsules"
    production.mkdir()
    store.mkdir(parents=True, mode=0o700)
    store.chmod(0o700)
    return production, research, store


def _request(cycle="2026-08-21", *, live=True):
    values = tuple(("signal-%d" % index).encode() for index in range(5))
    metadata = tuple({
        "role": role,
        "preservation_status": "embedded" if index % 2 == 0 else "workspace_file",
        "logical_locator": "private/%d/pred.pkl" % index,
        "raw_digest": TypedDigest.raw(values[index]).to_dict(),
        "size_bytes": len(values[index]),
    } for index, role in enumerate(module.ROLES))
    receipt = {
        "schema_version": 1, "source_claim": "EXACT_PHASE37A_SIGNAL_INPUTS_V1",
        "cycle_id": cycle, "phase37a_status": "sealed_complete",
        "problem_inventory_digest": TypedDigest.canonical([]).to_dict(),
        "phase37a_seal_digest": TypedDigest.raw(b"seal").to_dict(),
        "phase37a_manifest_digest": TypedDigest.raw(b"manifest").to_dict(),
        "run_context_verified": True, "config_context_verified": True,
        "ranking_context_verified": True, "universe_identity_verified": True,
        "requested_member_count": 5,
        "members": [{
            "role": row["role"], "original_preservation": row["preservation_status"],
            "raw_digest": row["raw_digest"], "size_bytes": row["size_bytes"],
        } for row in metadata],
        "source_observation_complete": True, "did_write": False,
        "definition_bound": False, "same_champion_segment": False,
    }
    return module._Request(
        cycle, metadata, values if live else None, receipt,
        _authority=module._AUTHORITY,
    )


def _artifact_rows(values=None):
    values = values or tuple(("raw-%d" % index).encode() for index in range(5))
    rows = []
    for index in range(5):
        position = index if index < 4 else "ensemble"
        digest = TypedDigest.raw(values[index]).to_dict()
        row = {
            "position": position, "recorder_id": "PRIVATE-%d" % index,
            "artifact_locator": "private/%d" % index,
            "members": [{
                "path": "private/%d/metric.json" % index,
                "status": "observed", "digest": TypedDigest.raw(b"metric").to_dict(),
                "preservation_status": "embedded", "detail": "",
            }, {
                "path": "private/%d/pred.pkl" % index,
                "status": "observed", "digest": digest,
                "preservation_status": "embedded" if index % 2 == 0 else "workspace_file",
                "detail": "",
            }],
            "artifact_tree_digest": TypedDigest.canonical([]).to_dict(),
        }
        if index < 4:
            row["source_recorder_id"] = "SOURCE-%d" % index
        rows.append(row)
    return rows


def test_exact_ordered_five_signal_metadata_and_fixed_budget():
    rows = _artifact_rows()
    metadata = module._member_metadata(rows)
    assert tuple(row["role"] for row in metadata) == module.ROLES
    assert tuple(row["preservation_status"] for row in metadata) == (
        "embedded", "workspace_file", "embedded", "workspace_file", "embedded",
    )
    assert sum(row["size_bytes"] for row in metadata) == sum(
        len("raw-%d" % index) for index in range(5)
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "ambiguous"])
def test_missing_duplicate_foreign_or_ambiguous_prediction_blocks_without_redefining_roles(mutation):
    rows = _artifact_rows()
    if mutation == "missing":
        rows.pop(3)
    elif mutation == "duplicate":
        rows[2]["position"] = 1
    elif mutation == "foreign":
        rows.append({"position": "foreign"})
    else:
        rows[0]["members"].append(copy.deepcopy(rows[0]["members"][1]))
        rows[0]["members"][-1]["path"] = "other/pred.pkl"
    manifest = {"model_and_ensemble_lineage": {"source_artifacts": rows}}
    if mutation == "foreign":
        # The raw inventory must be validated by the Phase37A partition owner.
        with pytest.raises(Exception):
            module._phase37._source_projection(manifest)
    else:
        with pytest.raises(module._Blocked):
            metadata = module._member_metadata(module._artifact_rows(manifest))
            assert tuple(row["role"] for row in metadata) == module.ROLES


def test_each_member_and_total_budget_are_fixed(monkeypatch):
    rows = _artifact_rows()
    rows[0]["members"][1]["digest"]["size_bytes"] = module.MAX_MEMBER_BYTES + 1
    with pytest.raises(module._Blocked, match="BUDGET"):
        module._member_metadata(rows)


def test_run_config_ranking_universe_and_lineage_context_are_all_required(tmp_path, monkeypatch):
    calls = []
    def run_manifest(_cycle, _manifest, evidence_class):
        calls.append(evidence_class)
        commands = {
            "prediction": "static_train", "post_trade": "prod_post_trade",
            "ensemble": "ensemble_fusion", "order": "order_gen",
        }
        return {
            "schema_version": 1, "command": commands[evidence_class],
            "status": "success", "run_id": "RUN",
        }
    monkeypatch.setattr(module._phase37, "_run_manifest", run_manifest)
    monkeypatch.setattr(module._phase37, "_source_projection", lambda _value: {})
    monkeypatch.setattr(module._phase37, "_prediction_projection", lambda _value: {})
    monkeypatch.setattr(module._phase37, "_ensemble_projection", lambda *_value: {})
    monkeypatch.setattr(module._phase37, "_market_projection", lambda _value: {})
    manifest = {
        "ranking": {"status": "complete", "ranking_digest": TypedDigest.raw(b"rank").to_dict()},
        "data_identity": {"qlib_materialization_identity": {"status": "observed"}},
        "referenced_evidence": [{
            "kind": "config", "observation": {
                "status": "observed", "preservation_status": "embedded",
            },
        }],
    }
    module._run_context(tmp_path, manifest)
    assert calls == ["prediction", "post_trade", "ensemble", "order"]
    for mutation in ("ranking", "universe", "config"):
        invalid = copy.deepcopy(manifest)
        if mutation == "ranking":
            invalid["ranking"]["status"] = "partial"
        elif mutation == "universe":
            invalid["data_identity"]["qlib_materialization_identity"]["status"] = "missing"
        else:
            invalid["referenced_evidence"][0]["observation"]["preservation_status"] = "workspace_file"
        with pytest.raises(module._Blocked):
            module._run_context(tmp_path, invalid)
    rows = _artifact_rows()
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", 4)
    with pytest.raises(module._Blocked, match="TOTAL_BUDGET"):
        module._member_metadata(rows)


def test_prepare_is_zero_write_and_plan_authority_is_not_replayable(tmp_path, monkeypatch):
    production, research, store = _roots(tmp_path)
    request = _request()
    monkeypatch.setattr(module, "_derive_request", lambda *_args, **_kwargs: request)
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    plan = module.prepare_signal_input_capsule(production, research, request.cycle_id, store)
    assert plan.status == "PREPARED"
    assert plan.capsule_id == request.capsule_id
    assert plan.to_safe_dict()["critical_signal_retention_complete"] is False
    assert tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))) == before
    with pytest.raises(module.SignalInputCapsuleContractError, match="inspector-owned"):
        module.SignalInputCapsulePlan(**plan.to_safe_dict())


def test_precondition_blocked_preserves_exact_five_terminal_identities(tmp_path, monkeypatch):
    production, research, store = _roots(tmp_path)
    monkeypatch.setattr(
        module, "_derive_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(module._Blocked("SIGNAL_MISSING")),
    )
    plan = module.prepare_signal_input_capsule(production, research, "2026-08-21", store)
    payload = plan.to_safe_dict()
    assert payload["status"] == "PRECONDITION_BLOCKED"
    assert [row["role"] for row in payload["requested_members"]] == list(module.ROLES)
    assert all(row["status"] == "blocked" for row in payload["requested_members"])
    assert payload["critical_signal_retention_complete"] is False
    assert payload["did_write"] is False


def test_request_private_mutation_cannot_preserve_authority():
    request = _request()
    request.metadata[0]["raw_digest"] = TypedDigest.raw(b"forged").to_dict()
    with pytest.raises(module.SignalInputCapsuleContractError):
        request.member_bytes()


def test_mixed_embedded_and_workspace_sources_are_freshly_read(tmp_path, monkeypatch):
    production, _research, _store = _roots(tmp_path)
    cycle_id = "2026-08-21"
    cycle = production / "data" / "evidence" / "v1" / "cycles" / cycle_id
    cycle.mkdir(parents=True, mode=0o700)
    cycle.chmod(0o700)
    (cycle / "objects").mkdir(mode=0o700)
    values = tuple(("fresh-%d" % index).encode() for index in range(5))
    rows = _artifact_rows(values)
    for index, row in enumerate(rows):
        member = row["members"][1]
        digest = member["digest"]
        if member["preservation_status"] == "embedded":
            path = cycle / "objects" / digest["value"][:2] / digest["value"]
        else:
            path = production / member["path"]
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        path.write_bytes(values[index])
        path.chmod(0o600 if member["preservation_status"] == "embedded" else 0o644)
    (cycle / "manifest.json").write_bytes(b"manifest")
    (cycle / "seal.json").write_bytes(b"seal")
    (cycle / "manifest.json").chmod(0o600)
    (cycle / "seal.json").chmod(0o600)
    manifest = {
        "status": "sealed_complete", "problems": [],
        "model_and_ensemble_lineage": {"source_artifacts": rows},
    }
    monkeypatch.setattr(
        module._phase37, "_cycle_authority",
        lambda *_args: (cycle, manifest, {}),
    )
    monkeypatch.setattr(module, "_run_context", lambda *_args: None)
    request = module._derive_request(production, cycle_id, read_live=True)
    assert request.data == values
    assert tuple(row["role"] for row in request.metadata) == module.ROLES
    assert request.receipt["requested_member_count"] == 5


def test_ordinary_member_failure_still_observes_later_requested_members(tmp_path, monkeypatch):
    production, _research, _store = _roots(tmp_path)
    cycle_id = "2026-08-21"
    cycle = production / "data" / "evidence" / "v1" / "cycles" / cycle_id
    cycle.mkdir(parents=True, mode=0o700)
    for name in ("manifest.json", "seal.json"):
        (cycle / name).write_bytes(name.encode())
        (cycle / name).chmod(0o600)
    rows = _artifact_rows()
    for row in rows:
        row["members"][1]["preservation_status"] = "workspace_file"
    manifest = {
        "status": "sealed_complete", "problems": [],
        "model_and_ensemble_lineage": {"source_artifacts": rows},
    }
    monkeypatch.setattr(module._phase37, "_cycle_authority", lambda *_: (cycle, manifest, {}))
    monkeypatch.setattr(module, "_run_context", lambda *_: None)
    observed = []
    def read_live(_root, logical, _expected):
        observed.append(logical)
        if len(observed) == 2:
            raise module._Blocked("SIGNAL_SOURCE_READ_FAILED")
        return ("raw-%d" % (len(observed) - 1)).encode()
    monkeypatch.setattr(module, "_read_live", read_live)
    with pytest.raises(module._Blocked, match="READ_FAILED"):
        module._derive_request(production, cycle_id, read_live=True)
    assert observed == [row["members"][1]["path"] for row in rows]


def test_publish_commits_exact_seven_files_manifest_last_and_adopts(tmp_path, monkeypatch):
    production, research, store = _roots(tmp_path)
    request = _request()
    monkeypatch.setattr(module, "_derive_request", lambda *_args, **_kwargs: request)
    first = module.publish_signal_input_capsule(
        production, research, request.cycle_id, store, request.capsule_id,
        request.request_digest, module.AUTHORIZATION_ACTION,
    )
    assert first.status == "COMMITTED"
    assert first.did_write is True
    assert first.critical_signal_retention_complete is True
    target = store / request.capsule_id
    assert tuple(sorted(path.name for path in target.iterdir())) == module.FINAL_NAMES
    assert stat_modes(target) == (0o700, (0o600,) * 7)
    manifest = module._phase37._strict_json(
        (target / module.MANIFEST_NAME).read_bytes(), "manifest",
    )
    assert manifest["member_count"] == 6
    assert [row["logical_path"] for row in manifest["members"]] == [
        module.RECEIPT_NAME, *module.MEMBER_NAMES,
    ]
    before = tuple((path.name, path.stat().st_ino) for path in sorted(target.iterdir()))
    second = module.publish_signal_input_capsule(
        production, research, request.cycle_id, store, request.capsule_id,
        request.request_digest, module.AUTHORIZATION_ACTION,
    )
    assert second.status == "ADOPTED"
    assert second.did_write is False
    assert tuple((path.name, path.stat().st_ino) for path in sorted(target.iterdir())) == before


def test_publish_rejects_research_store_below_production_without_writing(tmp_path, monkeypatch):
    production = tmp_path / "production"
    research = production / "research-workspace"
    store = research / "research" / "shadow_v1" / "signal_input_capsules"
    store.mkdir(parents=True, mode=0o700)
    store.chmod(0o700)
    monkeypatch.setattr(
        module, "_derive_request",
        lambda *_args, **_kwargs: pytest.fail("source authority must not be read"),
    )

    request = _request()
    with pytest.raises(module.SignalInputCapsuleContractError, match="isolated"):
        module.publish_signal_input_capsule(
            production, research, request.cycle_id, store, request.capsule_id,
            request.request_digest, module.AUTHORIZATION_ACTION,
        )

    assert tuple(store.iterdir()) == ()


def test_zero_length_signal_member_is_retained_as_exact_raw_bytes(tmp_path):
    _production, _research, store = _roots(tmp_path)
    request = _request()
    request.data = (b"",) + request.data[1:]
    request.metadata[0]["raw_digest"] = TypedDigest.raw(b"").to_dict()
    request.metadata[0]["size_bytes"] = 0
    request.receipt["members"][0]["raw_digest"] = TypedDigest.raw(b"").to_dict()
    request.receipt["members"][0]["size_bytes"] = 0
    # Rebuild inspector authority after changing the synthetic source fixture.
    rebuilt = module._Request(
        request.cycle_id, request.metadata, request.data, request.receipt,
        _authority=module._AUTHORITY,
    )
    result = module._publish_store(store, rebuilt)
    assert result.status == "COMMITTED"
    assert (store / rebuilt.capsule_id / module.MEMBER_NAMES[0]).read_bytes() == b""


def stat_modes(target):
    return (
        target.stat().st_mode & 0o777,
        tuple((path.stat().st_mode & 0o777) for path in sorted(target.iterdir())),
    )


def test_adopt_does_not_require_live_signal_bytes(tmp_path, monkeypatch):
    production, research, store = _roots(tmp_path)
    live_request = _request()
    monkeypatch.setattr(module, "_derive_request", lambda *_args, **_kwargs: live_request)
    module.publish_signal_input_capsule(
        production, research, live_request.cycle_id, store, live_request.capsule_id,
        live_request.request_digest, module.AUTHORIZATION_ACTION,
    )
    metadata_request = _request(live=False)
    def derive(*_args, **kwargs):
        assert kwargs["read_live"] is False
        return metadata_request
    monkeypatch.setattr(module, "_derive_request", derive)
    result = module.adopt_signal_input_capsule(
        production, research, metadata_request.cycle_id, store, metadata_request.capsule_id,
    )
    assert result.status == "ADOPTED"
    assert result.critical_signal_retention_complete is True
    assert result.did_write is False


@pytest.mark.parametrize("mutation", ["partial", "extra", "replace", "symlink"])
def test_partial_extra_replaced_or_symlink_target_denies_capability(tmp_path, monkeypatch, mutation):
    production, research, store = _roots(tmp_path)
    request = _request()
    monkeypatch.setattr(module, "_derive_request", lambda *_args, **_kwargs: request)
    target = store / request.capsule_id
    target.mkdir(mode=0o700)
    if mutation == "partial":
        (target / module.RECEIPT_NAME).write_bytes(b"partial")
        (target / module.RECEIPT_NAME).chmod(0o600)
    elif mutation == "extra":
        (target / "foreign").write_bytes(b"")
        (target / "foreign").chmod(0o600)
    elif mutation == "replace":
        for name, data in request.member_bytes():
            (target / name).write_bytes(data)
            (target / name).chmod(0o600)
        members = request.member_bytes()
        (target / module.MANIFEST_NAME).write_bytes(module._manifest(request, members))
        (target / module.MANIFEST_NAME).chmod(0o600)
        (target / module.MEMBER_NAMES[0]).write_bytes(b"foreign")
    else:
        target.rmdir()
        target.symlink_to(production, target_is_directory=True)
    result = module.publish_signal_input_capsule(
        production, research, request.cycle_id, store, request.capsule_id,
        request.request_digest, module.AUTHORIZATION_ACTION,
    )
    assert result.status in {"CONFLICT", "UNCERTAIN"}
    assert result.critical_signal_retention_complete is False
    assert result.did_write is False


def test_result_mutation_and_public_replay_cannot_grant_capability():
    request = _request()
    result = module._result(request, "ADOPTED", False, 7, b"manifest")
    with pytest.raises(module.SignalInputCapsuleContractError, match="inspector-owned"):
        module.SignalInputCapsuleResult(**result.to_safe_dict())
    result.verified_count = 6
    with pytest.raises(module.SignalInputCapsuleContractError):
        result.to_safe_dict()


def test_target_move_away_back_during_create_is_uncertain_with_truthful_write(tmp_path, monkeypatch):
    _production, _research, store = _roots(tmp_path)
    request = _request()
    original = module.os.mkdir
    def move_after_create(name, mode=0o777, *, dir_fd=None):
        original(name, mode, dir_fd=dir_fd)
        if name == request.capsule_id:
            target = store / request.capsule_id
            displaced = store / "displaced"
            target.rename(displaced)
            displaced.rename(target)
    monkeypatch.setattr(module.os, "mkdir", move_after_create)
    result = module._publish_store(store, request)
    assert result.status == "UNCERTAIN"
    assert result.did_write is True
    assert result.verified_count == 0
    assert result.critical_signal_retention_complete is False


def test_member_move_away_back_during_write_is_uncertain(tmp_path, monkeypatch):
    _production, _research, store = _roots(tmp_path)
    request = _request()
    original = module._write_all
    moved = []
    def move_member(fd, data):
        original(fd, data)
        if not moved:
            target = store / request.capsule_id
            member = target / module.RECEIPT_NAME
            displaced = target / "displaced"
            member.rename(displaced)
            displaced.rename(member)
            moved.append(True)
    monkeypatch.setattr(module, "_write_all", move_member)
    result = module._publish_store(store, request)
    assert result.status == "UNCERTAIN"
    assert result.did_write is True
    assert result.critical_signal_retention_complete is False


def test_existing_target_member_move_away_back_during_publish_is_uncertain(tmp_path, monkeypatch):
    _production, _research, store = _roots(tmp_path)
    request = _request()
    assert module._publish_store(store, request).status == "COMMITTED"
    original = module._verify_bundle

    def move_after_verify(target, wanted, *, use_request_data):
        manifest = original(target, wanted, use_request_data=use_request_data)
        member = target / module.MEMBER_NAMES[0]
        displaced = target / "displaced"
        member.rename(displaced)
        displaced.rename(member)
        return manifest

    monkeypatch.setattr(module, "_verify_bundle", move_after_verify)
    result = module._publish_store(store, request)
    assert result.status == "UNCERTAIN"
    assert result.did_write is False
    assert result.verified_count == 0
    assert result.critical_signal_retention_complete is False


def test_fresh_publication_mutation_during_final_verify_is_uncertain(tmp_path, monkeypatch):
    _production, _research, store = _roots(tmp_path)
    request = _request()
    original = module._verify_bundle

    def move_after_verify(target, wanted, *, use_request_data):
        manifest = original(target, wanted, use_request_data=use_request_data)
        member = target / module.MEMBER_NAMES[0]
        displaced = target / "displaced"
        member.rename(displaced)
        displaced.rename(member)
        return manifest

    monkeypatch.setattr(module, "_verify_bundle", move_after_verify)
    result = module._publish_store(store, request)
    assert result.status == "UNCERTAIN"
    assert result.did_write is True
    assert result.verified_count == 7
    assert result.critical_signal_retention_complete is False


def test_extra_file_after_create_cannot_be_reported_as_zero_write_conflict(tmp_path, monkeypatch):
    _production, _research, store = _roots(tmp_path)
    request = _request()
    original = module._verify_bundle
    def inject_extra(target, wanted, *, use_request_data):
        extra = target / "foreign"
        extra.write_bytes(b"")
        extra.chmod(0o600)
        return original(target, wanted, use_request_data=use_request_data)
    monkeypatch.setattr(module, "_verify_bundle", inject_extra)
    result = module._publish_store(store, request)
    assert result.status == "UNCERTAIN"
    assert result.did_write is True
    assert result.critical_signal_retention_complete is False


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_process_control_propagates_from_publication(tmp_path, monkeypatch, exc):
    _production, _research, store = _roots(tmp_path)
    monkeypatch.setattr(module.os, "mkdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(exc))
    with pytest.raises(type(exc)):
        module._publish_store(store, _request())


def test_safe_renderer_contains_no_private_locator_or_signal_bytes():
    encoded = canonical_json_bytes(module._result(_request(), "ADOPTED", False, 7, b"manifest").to_safe_dict())
    assert b"PRIVATE" not in encoded
    assert b"private/" not in encoded
    assert b"signal-" not in encoded
    assert all(value is False for key, value in module._result(
        _request(), "ADOPTED", False, 7, b"manifest",
    ).to_safe_dict().items() if key in {
        "same_champion_segment", "definition_bound", "intent_capability",
        "epoch_started", "prospective_claim", "promotion_capability", "did_write",
    })
