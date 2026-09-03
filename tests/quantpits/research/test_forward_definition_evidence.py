import copy
import hashlib
import json
import os
import shutil
import stat

import pytest

from tests.quantpits.research.test_forward_observation import (
    CYCLE, _bundle, _fresh_split_bundle,
)
from quantpits.research.forward_observation import (
    observe_frozen_shadow_forward_definition_candidate,
)
from quantpits.research.forward_definition_evidence import (
    AUTHORIZATION_ACTION,
    FRESH_EVIDENCE_AUTHORIZATION_ACTION,
    ForwardDefinitionEvidenceContractError,
    ForwardDefinitionEvidencePlan,
    ForwardDefinitionEvidenceStoreReceipt,
    adopt_forward_definition_evidence,
    adopt_fresh_champion_segment_definition_evidence,
    prepare_forward_definition_evidence,
    prepare_fresh_champion_segment_definition_evidence,
    publish_forward_definition_evidence,
    publish_fresh_champion_segment_definition_evidence,
)


@pytest.fixture
def evidence_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, _ = _bundle(root, frozen=True)
    shadow = root / "research" / "shadow_v1"
    definitions = shadow / "definitions"
    definitions.mkdir(mode=0o700)
    definitions.chmod(0o700)
    candidate = observe_frozen_shadow_forward_definition_candidate(root, CYCLE, activation)
    from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
    assert CreateOnlyDefinitionBundleStore(definitions).publish(
        candidate.to_store_request(),
    ).status == "COMMITTED"
    evidence = shadow / "definition_evidence"
    evidence.mkdir(mode=0o700)
    evidence.chmod(0o700)
    return root, activation, definitions, evidence


@pytest.fixture
def fresh_evidence_workspace(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    shadow = research / "research" / "shadow_v1"
    definitions = shadow / "definitions"
    definitions.mkdir(mode=0o700)
    evidence = shadow / "definition_evidence"
    evidence.mkdir(mode=0o700)
    from quantpits.research.forward_definition_publication import (
        FRESH_AUTHORIZATION_ACTION,
        prepare_fresh_champion_segment_definition_publication,
        publish_fresh_champion_segment_definition_bundle,
    )
    plan = prepare_fresh_champion_segment_definition_publication(
        production, research, CYCLE, activation, definitions,
    )
    result = publish_fresh_champion_segment_definition_bundle(
        production, research, CYCLE, activation, definitions,
        plan.definition_set_id, dict(plan.request_digest),
        FRESH_AUTHORIZATION_ACTION,
    )
    assert result.status == "COMMITTED"
    return production, research, activation, definitions, evidence


def _snapshot(root):
    rows = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        rows.append((
            path.relative_to(root).as_posix(), info.st_dev, info.st_ino,
            info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        ))
    return rows


def _prepare(value):
    return prepare_forward_definition_evidence(*value[:1], CYCLE, *value[1:])


def _publish(value, plan=None, **changed):
    plan = _prepare(value) if plan is None else plan
    values = {
        "identifier": plan.definition_set_id,
        "definition_digest": dict(plan.definition_request_digest),
        "evidence_digest": dict(plan.evidence_request_digest),
        "action": AUTHORIZATION_ACTION,
    }
    values.update(changed)
    root, activation, definitions, evidence = value
    return publish_forward_definition_evidence(
        root, CYCLE, activation, definitions, evidence,
        values["identifier"], values["definition_digest"],
        values["evidence_digest"], values["action"],
    )


def _fresh_prepare(value):
    return prepare_fresh_champion_segment_definition_evidence(
        value[0], value[1], CYCLE, *value[2:],
    )


def _fresh_publish(value, plan=None, **changed):
    plan = _fresh_prepare(value) if plan is None else plan
    values = {
        "identifier": plan.definition_set_id,
        "definition_digest": dict(plan.definition_request_digest),
        "evidence_digest": dict(plan.evidence_request_digest),
        "action": FRESH_EVIDENCE_AUTHORIZATION_ACTION,
    }
    values.update(changed)
    return publish_fresh_champion_segment_definition_evidence(
        value[0], value[1], CYCLE, *value[2:], values["identifier"],
        values["definition_digest"], values["evidence_digest"],
        values["action"],
    )


def _fresh_adopt(value):
    return adopt_fresh_champion_segment_definition_evidence(
        value[0], value[1], CYCLE, *value[2:],
    )


def test_fresh_evidence_adopter_is_exact_and_strictly_zero_write(
    fresh_evidence_workspace,
):
    assert _fresh_publish(fresh_evidence_workspace).status == "COMMITTED"
    before = (
        _snapshot(fresh_evidence_workspace[0]),
        _snapshot(fresh_evidence_workspace[1]),
    )
    result = _fresh_adopt(fresh_evidence_workspace)
    assert result.status == "ADOPTED"
    assert result.evidence_receipt.did_write is False
    assert result.definition_evidence_complete is True
    assert before == (
        _snapshot(fresh_evidence_workspace[0]),
        _snapshot(fresh_evidence_workspace[1]),
    )


def test_fresh_evidence_adopter_cannot_create_absent_or_upgrade_conflict(
    fresh_evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    calls = []
    monkeypatch.setattr(
        module._CreateOnlyEvidenceStore, "publish",
        lambda *_args: calls.append(True),
    )
    before = _snapshot(fresh_evidence_workspace[1])
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _fresh_adopt(fresh_evidence_workspace)
    assert calls == [] and _snapshot(fresh_evidence_workspace[1]) == before

    monkeypatch.undo()
    assert _fresh_publish(fresh_evidence_workspace).status == "COMMITTED"
    target = (
        fresh_evidence_workspace[4]
        / "shadow.fresh.segment.v1" / "reference_receipt.json"
    )
    target.write_bytes(b"{}")
    target.chmod(0o600)
    before = _snapshot(fresh_evidence_workspace[1])
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _fresh_adopt(fresh_evidence_workspace)
    assert _snapshot(fresh_evidence_workspace[1]) == before


def test_fresh_evidence_preflight_uses_production_source_and_research_definition_without_writes(
    fresh_evidence_workspace,
):
    production, research, _activation, _definitions, _evidence = fresh_evidence_workspace
    before = _snapshot(production), _snapshot(research)
    plan = _fresh_prepare(fresh_evidence_workspace)
    summary = plan.to_safe_summary_dict()
    assert summary["status"] == "PREPARED"
    assert summary["definition_store_status"] == "ADOPTED"
    assert summary["definition_store_did_write"] is False
    assert summary["target_state"] == "ABSENT"
    assert plan.publication_capability is False
    assert before == (_snapshot(production), _snapshot(research))


def test_fresh_evidence_publish_reobserves_and_calls_adopter_and_writer_once(
    fresh_evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_observation as observation
    import quantpits.research.forward_definition_evidence as module
    from quantpits.research import definition_store
    plan = _fresh_prepare(fresh_evidence_workspace)
    observed, adopted, written = [], [], []
    original_observe = observation.observe_fresh_champion_segment_candidate
    original_adopt = definition_store.CreateOnlyDefinitionBundleStore.adopt_existing
    original_write = module._CreateOnlyEvidenceStore.publish

    def observe(*args):
        observed.append(args)
        return original_observe(*args)

    def adopt(store, request):
        adopted.append(request)
        return original_adopt(store, request)

    def write(store, request):
        written.append(request)
        return original_write(store, request)

    monkeypatch.setattr(observation, "observe_fresh_champion_segment_candidate", observe)
    monkeypatch.setattr(definition_store.CreateOnlyDefinitionBundleStore, "adopt_existing", adopt)
    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "publish", write)
    assert _fresh_publish(fresh_evidence_workspace, plan).status == "COMMITTED"
    assert len(observed) == len(adopted) == len(written) == 1
    assert dict(adopted[0].request_digest) == dict(plan.definition_request_digest)
    assert dict(written[0].request_digest) == dict(plan.evidence_request_digest)


@pytest.mark.parametrize("relation", ["same", "production_parent", "research_parent"])
def test_fresh_evidence_requires_physically_separate_roots_and_exact_research_layout(
    fresh_evidence_workspace, relation,
):
    production, research, activation, definitions, evidence = fresh_evidence_workspace
    if relation == "same":
        production = research
    elif relation == "production_parent":
        production = research.parent
    else:
        production = research / "nested-production"
        production.mkdir()
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        prepare_fresh_champion_segment_definition_evidence(
            production, research, CYCLE, activation, definitions, evidence,
        )


@pytest.mark.parametrize("field", ["id", "definition", "evidence", "action", "bool"])
def test_fresh_candidate_request_receipt_and_owner_digests_must_join(
    fresh_evidence_workspace, field,
):
    plan = _fresh_prepare(fresh_evidence_workspace)
    changed = {}
    if field == "id":
        changed["identifier"] = "foreign.definition"
    elif field == "definition":
        digest = dict(plan.definition_request_digest)
        digest["value"] = "0" * 64
        changed["definition_digest"] = digest
    elif field == "evidence":
        digest = dict(plan.evidence_request_digest)
        digest["size_bytes"] += 1
        changed["evidence_digest"] = digest
    elif field == "action":
        changed["action"] = AUTHORIZATION_ACTION
    else:
        digest = dict(plan.evidence_request_digest)
        digest["size_bytes"] = True
        changed["evidence_digest"] = digest
    before = _snapshot(fresh_evidence_workspace[1])
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _fresh_publish(fresh_evidence_workspace, plan, **changed)
    assert _snapshot(fresh_evidence_workspace[1]) == before


def test_first_fresh_evidence_publish_commits_exact_receipts_and_replay_adopts(
    fresh_evidence_workspace,
):
    production, research, _activation, _definitions, evidence = fresh_evidence_workspace
    production_before = _snapshot(production)
    first = _fresh_publish(fresh_evidence_workspace)
    summary = first.to_safe_summary_dict()
    assert summary["status"] == "COMMITTED" and summary["did_write"] is True
    assert summary["definition_bundle_adopted_verified"] is True
    assert summary["durable_reference_recorded"] is True
    assert summary["definition_evidence_complete"] is True
    assert summary["original_commit_receipt_recorded"] is False
    target = evidence / summary["definition_set_id"]
    assert {path.name for path in target.iterdir()} == {
        "reference_receipt.json", "definition_store_receipt.json",
        "evidence_manifest.json",
    }
    assert _snapshot(production) == production_before
    before = _snapshot(research)
    replay = _fresh_publish(fresh_evidence_workspace)
    assert replay.status == "ADOPTED"
    assert replay.to_safe_summary_dict()["did_write"] is False
    assert replay.definition_evidence_complete is True
    assert _snapshot(research) == before


def test_definition_conflict_never_reaches_fresh_evidence_writer(
    fresh_evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    target = fresh_evidence_workspace[3] / "shadow.fresh.segment.v1"
    (target / "protocol.json").write_bytes(b"foreign")
    called = []
    monkeypatch.setattr(
        module._CreateOnlyEvidenceStore, "publish",
        lambda *_args: called.append(True),
    )
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _fresh_prepare(fresh_evidence_workspace)
    assert called == []


def test_fresh_definition_store_is_never_a_write_domain(
    fresh_evidence_workspace, monkeypatch,
):
    from quantpits.research import definition_store
    monkeypatch.setattr(
        definition_store.CreateOnlyDefinitionBundleStore, "publish",
        lambda *_args: (_ for _ in ()).throw(AssertionError("definition publish called")),
    )
    plan = _fresh_prepare(fresh_evidence_workspace)
    assert _fresh_publish(fresh_evidence_workspace, plan).status == "COMMITTED"


def test_fresh_evidence_conflict_denies_complete_capability(
    fresh_evidence_workspace,
):
    plan = _fresh_prepare(fresh_evidence_workspace)
    target = fresh_evidence_workspace[4] / plan.definition_set_id
    target.mkdir(mode=0o700)
    before = _snapshot(fresh_evidence_workspace[0])
    result = _fresh_publish(fresh_evidence_workspace, plan)
    assert result.status == "CONFLICT"
    assert result.to_safe_summary_dict()["did_write"] is False
    assert result.definition_evidence_complete is False
    assert _snapshot(fresh_evidence_workspace[0]) == before


def test_fresh_extra_research_write_after_writer_is_uncertain_and_preserves_write_fact(
    fresh_evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    production, research, _activation, _definitions, _evidence = fresh_evidence_workspace
    production_before = _snapshot(production)
    original = module._CreateOnlyEvidenceStore.publish

    def publish_with_extra(store, request):
        receipt = original(store, request)
        (research / "foreign.txt").write_text("foreign")
        return receipt

    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "publish", publish_with_extra)
    result = _fresh_publish(fresh_evidence_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_evidence_complete is False
    assert _snapshot(production) == production_before


def test_fresh_final_public_verification_failure_is_uncertain(
    fresh_evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    monkeypatch.setattr(
        module, "_verify_public_evidence",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic")),
    )
    result = _fresh_publish(fresh_evidence_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_evidence_complete is False


def test_fresh_non_target_mutation_during_final_verification_is_uncertain(
    fresh_evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    research = fresh_evidence_workspace[1]
    original = module._verify_public_evidence

    def verify_then_mutate(*args):
        exact = original(*args)
        (research / "research" / "shadow_v1" / "late.txt").write_text("late")
        return exact

    monkeypatch.setattr(module, "_verify_public_evidence", verify_then_mutate)
    result = _fresh_publish(fresh_evidence_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_evidence_complete is False


def test_fresh_production_root_move_away_back_after_writer_is_uncertain(
    fresh_evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    production = fresh_evidence_workspace[0]
    displaced = production.with_name(production.name + "-away")
    original = module._CreateOnlyEvidenceStore.publish

    def publish_then_move(store, request):
        receipt = original(store, request)
        production.rename(displaced)
        displaced.rename(production)
        return receipt

    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "publish", publish_then_move)
    result = _fresh_publish(fresh_evidence_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_evidence_complete is False


@pytest.mark.parametrize("seam", ["observation", "adopter", "writer", "postcondition", "result"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fresh_process_control_propagates_at_all_evidence_seams(
    fresh_evidence_workspace, monkeypatch, seam, exception,
):
    import quantpits.research.forward_definition_evidence as module
    plan = _fresh_prepare(fresh_evidence_workspace)
    if seam == "observation":
        import quantpits.research.forward_observation as observation
        monkeypatch.setattr(
            observation, "observe_fresh_champion_segment_candidate",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    elif seam == "adopter":
        from quantpits.research import definition_store
        monkeypatch.setattr(
            definition_store.CreateOnlyDefinitionBundleStore, "adopt_existing",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    elif seam == "writer":
        monkeypatch.setattr(
            module._CreateOnlyEvidenceStore, "publish",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    elif seam == "postcondition":
        monkeypatch.setattr(
            module, "_verify_public_evidence",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    else:
        monkeypatch.setattr(
            module.ForwardDefinitionEvidenceResult, "__init__",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(exception()),
        )
    with pytest.raises(exception):
        _fresh_publish(fresh_evidence_workspace, plan)


def test_preflight_freshly_rebuilds_receipts_and_is_strictly_zero_write(evidence_workspace):
    before = _snapshot(evidence_workspace[0])
    plan = _prepare(evidence_workspace)
    summary = plan.to_safe_summary_dict()
    assert summary["status"] == "PREPARED"
    assert summary["definition_store_status"] == "ADOPTED"
    assert summary["target_state"] == "ABSENT"
    assert summary["definition_evidence_complete"] is False
    assert _snapshot(evidence_workspace[0]) == before
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        copy.copy(plan)


def test_publish_reobserves_instead_of_accepting_carried_authority(
    evidence_workspace, monkeypatch,
):
    plan = _prepare(evidence_workspace)
    import quantpits.research.forward_observation as observation
    original = observation.observe_frozen_shadow_forward_definition_candidate
    calls = []

    def fresh(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(observation, "observe_frozen_shadow_forward_definition_candidate", fresh)
    assert _publish(evidence_workspace, plan).status == "COMMITTED"
    assert len(calls) == 1
    with pytest.raises(TypeError):
        publish_forward_definition_evidence(
            *evidence_workspace[:1], CYCLE, *evidence_workspace[1:],
            plan.definition_set_id, dict(plan.definition_request_digest),
            dict(plan.evidence_request_digest), AUTHORIZATION_ACTION,
            carried_plan=plan,
        )


@pytest.mark.parametrize("field", ["id", "definition", "evidence", "action", "bool"])
def test_publish_requires_exact_id_full_typed_digests_and_action(evidence_workspace, field):
    plan = _prepare(evidence_workspace)
    changed = {}
    if field == "id":
        changed["identifier"] = "foreign.id"
    elif field == "definition":
        digest = dict(plan.definition_request_digest)
        digest["value"] = "0" * 64
        changed["definition_digest"] = digest
    elif field == "evidence":
        digest = dict(plan.evidence_request_digest)
        digest["size_bytes"] += 1
        changed["evidence_digest"] = digest
    elif field == "action":
        changed["action"] = "PUBLISH"
    else:
        digest = dict(plan.evidence_request_digest)
        digest["size_bytes"] = True
        changed["evidence_digest"] = digest
    before = _snapshot(evidence_workspace[0])
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _publish(evidence_workspace, plan, **changed)
    assert _snapshot(evidence_workspace[0]) == before


@pytest.mark.parametrize("invalid", [None, False, 0, 1.0, "", [], {}])
def test_strict_fields_reject_absent_null_falsey_bool_numeric_and_containers(
    evidence_workspace, invalid,
):
    plan = _prepare(evidence_workspace)
    root, activation, definitions, evidence = evidence_workspace
    before = _snapshot(root)
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        publish_forward_definition_evidence(
            root, CYCLE, activation, definitions, evidence,
            invalid, dict(plan.definition_request_digest),
            dict(plan.evidence_request_digest), AUTHORIZATION_ACTION,
        )
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        publish_forward_definition_evidence(
            root, CYCLE, activation, definitions, evidence,
            plan.definition_set_id, invalid,
            dict(plan.evidence_request_digest), AUTHORIZATION_ACTION,
        )
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        publish_forward_definition_evidence(
            root, CYCLE, activation, definitions, evidence,
            plan.definition_set_id, dict(plan.definition_request_digest),
            invalid, AUTHORIZATION_ACTION,
        )
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        publish_forward_definition_evidence(
            root, CYCLE, activation, definitions, evidence,
            plan.definition_set_id, dict(plan.definition_request_digest),
            dict(plan.evidence_request_digest), invalid,
        )
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        prepare_forward_definition_evidence(
            root, invalid, activation, definitions, evidence,
        )
    assert _snapshot(root) == before


def test_first_publish_commits_exact_two_receipts_and_manifest(evidence_workspace):
    result = _publish(evidence_workspace)
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "COMMITTED" and summary["did_write"] is True
    assert summary["definition_bundle_adopted_verified"] is True
    assert summary["durable_reference_recorded"] is True
    assert summary["definition_evidence_complete"] is True
    assert summary["original_commit_receipt_recorded"] is False
    assert summary["portfolio_bootstrap_complete"] is False
    target = evidence_workspace[3] / summary["definition_set_id"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert {path.name for path in target.iterdir()} == {
        "reference_receipt.json", "definition_store_receipt.json",
        "evidence_manifest.json",
    }
    assert all(
        path.is_file() and path.stat().st_nlink == 1
        and stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in target.iterdir()
    )
    manifest = json.loads((target / "evidence_manifest.json").read_bytes())
    assert set(manifest) == {
        "schema_version", "evidence_kind", "storage_claim", "definition_set_id",
        "evidence_cycle_id", "evidence_request_digest", "definition_request_digest",
        "reference_receipt_digest", "definition_store_receipt_digest",
        "definition_manifest_digest", "definition_store_operation_id",
        "definition_store_status", "definition_store_did_write", "member_count",
        "members", "original_commit_receipt_recorded", "portfolio_bootstrap_claim",
        "prospective_claim", "promotion_capability",
    }
    assert manifest["definition_store_status"] == "ADOPTED"
    assert manifest["definition_store_did_write"] is False
    assert manifest["member_count"] == 2
    assert [row["logical_path"] for row in manifest["members"]] == [
        "reference_receipt.json", "definition_store_receipt.json",
    ]
    assert manifest["evidence_request_digest"] == summary["evidence_request_digest"]
    assert manifest["reference_receipt_digest"] == summary["reference_receipt_digest"]
    for row in manifest["members"]:
        data = (target / row["logical_path"]).read_bytes()
        assert row["size_bytes"] == len(data)
        assert row["digest"]["domain"] == "raw_bytes"
        assert row["digest"]["value"] == hashlib.sha256(data).hexdigest()


def test_exact_replay_is_adopted_without_filesystem_change(evidence_workspace):
    assert _publish(evidence_workspace).status == "COMMITTED"
    before = _snapshot(evidence_workspace[0])
    replay = _publish(evidence_workspace)
    assert replay.status == "ADOPTED"
    assert replay.to_safe_summary_dict()["did_write"] is False
    assert replay.definition_evidence_complete is True
    assert _snapshot(evidence_workspace[0]) == before


def test_adopt_forward_evidence_exact_target_is_adopted_and_strictly_zero_write(
    evidence_workspace,
):
    assert _publish(evidence_workspace).status == "COMMITTED"
    before = _snapshot(evidence_workspace[0])
    result = adopt_forward_definition_evidence(
        evidence_workspace[0], CYCLE, *evidence_workspace[1:],
    )
    assert result.status == "ADOPTED"
    assert result.evidence_receipt.did_write is False
    assert result.definition_evidence_complete is True
    assert _snapshot(evidence_workspace[0]) == before


def test_adopt_forward_evidence_absent_never_calls_publish_or_create(
    evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    called = []
    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "publish", lambda *_: called.append(True))
    before = _snapshot(evidence_workspace[0])
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        adopt_forward_definition_evidence(
            evidence_workspace[0], CYCLE, *evidence_workspace[1:],
        )
    assert called == [] and _snapshot(evidence_workspace[0]) == before


def test_adopt_forward_evidence_conflict_and_move_away_back_deny_capability(
    evidence_workspace, monkeypatch,
):
    assert _publish(evidence_workspace).status == "COMMITTED"
    exact = adopt_forward_definition_evidence(
        evidence_workspace[0], CYCLE, *evidence_workspace[1:],
    )
    target = evidence_workspace[3] / exact.definition_receipt.definition_set_id
    member = target / "reference_receipt.json"
    member.write_bytes(member.read_bytes() + b"\n")
    assert adopt_forward_definition_evidence(
        evidence_workspace[0], CYCLE, *evidence_workspace[1:],
    ).status == "CONFLICT"

    member.write_bytes(exact.evidence_request.members[0][1])
    import quantpits.research.forward_definition_evidence as module
    original = module._CreateOnlyEvidenceStore._observe_existing
    moved = []

    def transient(store, *args):
        if not moved:
            displaced = target.with_name(target.name + ".displaced")
            target.rename(displaced); displaced.rename(target)
            moved.append(True)
        return original(store, *args)

    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "_observe_existing", transient)
    result = adopt_forward_definition_evidence(
        evidence_workspace[0], CYCLE, *evidence_workspace[1:],
    )
    assert result.status == "UNCERTAIN"
    assert result.evidence_receipt.did_write is False
    assert result.definition_evidence_complete is False


def test_definition_absent_and_conflict_never_reach_evidence_writer(evidence_workspace, monkeypatch):
    plan = _prepare(evidence_workspace)
    target = evidence_workspace[2] / plan.definition_set_id
    for member in target.iterdir():
        member.unlink()
    target.rmdir()
    before = _snapshot(evidence_workspace[3])
    import quantpits.research.forward_definition_evidence as module
    monkeypatch.setattr(
        module._CreateOnlyEvidenceStore, "publish",
        lambda *_args: (_ for _ in ()).throw(AssertionError("evidence writer reached")),
    )
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _publish(evidence_workspace, plan)
    assert _snapshot(evidence_workspace[3]) == before


@pytest.mark.parametrize("classification", ["CONFLICT", "UNCERTAIN"])
def test_definition_conflict_and_uncertain_never_reach_evidence_writer(
    evidence_workspace, monkeypatch, classification,
):
    plan = _prepare(evidence_workspace)
    import quantpits.research.forward_definition_evidence as module
    from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
    if classification == "CONFLICT":
        target = evidence_workspace[2] / plan.definition_set_id / "protocol.json"
        target.write_bytes(b'{}\n')
    else:
        monkeypatch.setattr(
            CreateOnlyDefinitionBundleStore, "_observe_existing",
            staticmethod(lambda *_args: (_ for _ in ()).throw(OSError("synthetic"))),
        )
    monkeypatch.setattr(
        module._CreateOnlyEvidenceStore, "publish",
        lambda *_args: (_ for _ in ()).throw(AssertionError("evidence writer reached")),
    )
    before = _snapshot(evidence_workspace[3])
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _publish(evidence_workspace, plan)
    assert _snapshot(evidence_workspace[3]) == before


def test_evidence_conflict_never_grants_complete_capability(evidence_workspace):
    plan = _prepare(evidence_workspace)
    target = evidence_workspace[3] / plan.definition_set_id
    target.mkdir(mode=0o700)
    result = _publish(evidence_workspace, plan)
    assert result.status == "CONFLICT"
    assert result.to_safe_summary_dict()["did_write"] is False
    assert result.definition_evidence_complete is False


def test_evidence_partial_create_is_uncertain_and_never_grants_capability(
    evidence_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_evidence as module
    original = module._write_exclusive
    calls = {"count": 0}

    def fail_second(*args):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic")
        return original(*args)

    monkeypatch.setattr(module, "_write_exclusive", fail_second)
    result = _publish(evidence_workspace)
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "UNCERTAIN" and summary["did_write"] is True
    assert summary["evidence_member_count"] == 1
    assert summary["definition_evidence_complete"] is False
    target = evidence_workspace[3] / summary["definition_set_id"]
    assert {path.name for path in target.iterdir()} == {"reference_receipt.json"}


@pytest.mark.parametrize("mutation", [
    "extra", "bytes", "mode", "hardlink", "symlink", "fifo", "target_symlink",
])
def test_existing_evidence_mutation_classes_are_stable_conflicts(
    evidence_workspace, mutation,
):
    first = _publish(evidence_workspace)
    target = evidence_workspace[3] / first.to_safe_summary_dict()["definition_set_id"]
    member = target / "reference_receipt.json"
    if mutation == "extra":
        (target / "foreign.json").write_text("{}")
    elif mutation == "bytes":
        member.write_text("{}\n")
    elif mutation == "mode":
        member.chmod(0o644)
    elif mutation == "hardlink":
        member.unlink()
        os.link(target / "definition_store_receipt.json", member)
    elif mutation == "symlink":
        member.unlink()
        member.symlink_to("definition_store_receipt.json")
    elif mutation == "fifo":
        member.unlink()
        os.mkfifo(member)
    else:
        displaced = target.with_name(target.name + "-old")
        target.rename(displaced)
        target.symlink_to(displaced.name, target_is_directory=True)
    replay = _publish(evidence_workspace)
    assert replay.status == "CONFLICT"
    assert replay.to_safe_summary_dict()["did_write"] is False
    assert replay.definition_evidence_complete is False


def test_post_write_observer_failure_preserves_uncertain_write_fact(evidence_workspace, monkeypatch):
    import quantpits.research.forward_definition_evidence as module
    plan = _prepare(evidence_workspace)
    original = module._workspace_inventory
    calls = {"count": 0}

    def fail_final(*args):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("synthetic private detail")
        return original(*args)

    monkeypatch.setattr(module, "_workspace_inventory", fail_final)
    result = _publish(evidence_workspace, plan)
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "UNCERTAIN"
    assert summary["did_write"] is True and summary["evidence_member_count"] == 2
    assert summary["definition_evidence_complete"] is False


def test_workspace_and_evidence_root_move_away_back_fail_closed(evidence_workspace, monkeypatch):
    import quantpits.research.forward_definition_evidence as module
    original = module._CreateOnlyEvidenceStore.publish
    evidence = evidence_workspace[3]
    displaced = evidence.with_name("definition_evidence-away")

    def move_back(store, request):
        receipt = original(store, request)
        evidence.rename(displaced)
        displaced.rename(evidence)
        return receipt

    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "publish", move_back)
    result = _publish(evidence_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_evidence_complete is False


def test_final_public_name_replacement_with_exact_bytes_is_uncertain(evidence_workspace, monkeypatch):
    import quantpits.research.forward_definition_evidence as module
    original = module._CreateOnlyEvidenceStore.publish

    def replace_exact(store, request):
        receipt = original(store, request)
        target = store.root / request.definition_set_id
        displaced = store.root / (request.definition_set_id + "-away")
        target.rename(displaced)
        shutil.copytree(displaced, target, copy_function=shutil.copy2)
        target.chmod(0o700)
        shutil.rmtree(displaced)
        return receipt

    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "publish", replace_exact)
    result = _publish(evidence_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_evidence_complete is False


def test_only_exact_evidence_target_can_change_and_extra_write_is_detected(evidence_workspace, monkeypatch):
    import quantpits.research.forward_definition_evidence as module
    original = module._CreateOnlyEvidenceStore.publish

    def extra_write(store, request):
        receipt = original(store, request)
        (evidence_workspace[0] / "foreign.log").write_text("private")
        return receipt

    monkeypatch.setattr(module._CreateOnlyEvidenceStore, "publish", extra_write)
    result = _publish(evidence_workspace)
    assert result.status == "UNCERTAIN"
    assert result.definition_evidence_complete is False


def test_writer_owned_receipts_and_plans_reject_public_construction():
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        ForwardDefinitionEvidencePlan()
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        ForwardDefinitionEvidenceStoreReceipt()


def test_writer_owned_receipt_rejects_impossible_cross_fields_and_mutation(evidence_workspace):
    import quantpits.research.forward_definition_evidence as module
    result = _publish(evidence_workspace)
    receipt = result.evidence_receipt
    fields = {
        "operation_id": receipt.operation_id, "status": "COMMITTED",
        "reason_code": "COMMITTED", "did_write": False,
        "definition_set_id": receipt.definition_set_id,
        "request_digest": receipt.request_digest,
        "manifest_digest": receipt.manifest_digest,
        "member_count": receipt.member_count,
        "store_root_identity_before": receipt.store_root_identity_before,
        "store_root_identity_after": receipt.store_root_identity_after,
        "bundle_root_identity": receipt.bundle_root_identity,
    }
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        ForwardDefinitionEvidenceStoreReceipt(
            _authority=module._RECEIPT_AUTHORITY, **fields,
        )
    object.__setattr__(receipt, "did_write", False)
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _ = receipt.durable_record_capability


def test_foreign_or_mutated_aggregate_denies_downstream_capability(evidence_workspace):
    result = _publish(evidence_workspace)
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        copy.copy(result)
    object.__setattr__(result.evidence_request, "definition_operation_id", "0" * 64)
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _ = result.definition_evidence_complete


@pytest.mark.parametrize("source", ["observer", "definition_store", "evidence_store", "result"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates_from_every_stage(
    evidence_workspace, monkeypatch, source, exception,
):
    plan = _prepare(evidence_workspace)
    import quantpits.research.forward_definition_evidence as module
    if source == "observer":
        import quantpits.research.forward_observation as observation
        monkeypatch.setattr(
            observation, "observe_frozen_shadow_forward_definition_candidate",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    elif source == "definition_store":
        from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
        monkeypatch.setattr(
            CreateOnlyDefinitionBundleStore, "adopt_existing",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    elif source == "evidence_store":
        monkeypatch.setattr(
            module._CreateOnlyEvidenceStore, "publish",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    else:
        monkeypatch.setattr(
            module, "ForwardDefinitionEvidenceResult",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(exception()),
        )
    with pytest.raises(exception):
        _publish(evidence_workspace, plan)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates_from_definition_store_and_evidence_store(
    evidence_workspace, monkeypatch, exception,
):
    import quantpits.research.forward_definition_evidence as module
    monkeypatch.setattr(
        module._CreateOnlyEvidenceStore, "publish",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        _publish(evidence_workspace)


def test_safe_summaries_exclude_private_paths_and_future_claims(evidence_workspace):
    summary = _publish(evidence_workspace).to_safe_summary_dict()
    rendered = json.dumps(summary)
    assert str(evidence_workspace[0]) not in rendered
    assert all(token not in rendered for token in ("actor", "hypothesis", "broker", "capital"))
    assert summary["portfolio_bootstrap_complete"] is False
    assert summary["epoch_started"] is False
    assert summary["prospective_claim"] is False
    assert summary["promotion_capability"] is False


def test_invalid_formal_paths_engineering_activation_and_budget_fail_zero_write(
    tmp_path, evidence_workspace, monkeypatch,
):
    before = _snapshot(evidence_workspace[0])
    root, activation, definitions, evidence = evidence_workspace
    for changed in (
        (True, CYCLE, activation, definitions, evidence),
        (root, True, activation, definitions, evidence),
        (root, CYCLE, activation, definitions, tmp_path),
    ):
        with pytest.raises(ForwardDefinitionEvidenceContractError):
            prepare_forward_definition_evidence(*changed)
    import quantpits.research.forward_definition_evidence as module
    original_budget = module._MAX_INVENTORY_BYTES
    monkeypatch.setattr(module, "_MAX_INVENTORY_BYTES", 1)
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        _prepare(evidence_workspace)
    assert _snapshot(evidence_workspace[0]) == before
    monkeypatch.setattr(module, "_MAX_INVENTORY_BYTES", original_budget)

    engineering = tmp_path / "engineering"
    engineering.mkdir()
    engineering_activation, _ = _bundle(engineering, frozen=False)
    shadow = engineering / "research" / "shadow_v1"
    shadow.mkdir(parents=True, mode=0o700)
    activations = shadow / "activations"
    activations.mkdir(mode=0o700)
    formal_activation = activations / engineering_activation.name
    formal_activation.write_bytes(engineering_activation.read_bytes())
    formal_activation.chmod(0o600)
    engineering_definitions = shadow / "definitions"
    engineering_definitions.mkdir(mode=0o700)
    engineering_evidence = shadow / "definition_evidence"
    engineering_evidence.mkdir(mode=0o700)
    with pytest.raises(ForwardDefinitionEvidenceContractError):
        prepare_forward_definition_evidence(
            engineering, CYCLE, formal_activation,
            engineering_definitions, engineering_evidence,
        )
