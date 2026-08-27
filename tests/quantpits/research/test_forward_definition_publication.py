import copy
import json
import os
import sys
from pathlib import Path

import pytest

from tests.quantpits.research.test_forward_observation import CYCLE, _bundle
from quantpits.research.forward_definition_publication import (
    AUTHORIZATION_ACTION,
    FrozenDefinitionBytePublicationResult,
    FrozenDefinitionPublicationContractError,
    FrozenDefinitionPublicationInputError,
    FrozenDefinitionPublicationPlan,
    prepare_frozen_shadow_forward_definition_publication,
    publish_frozen_shadow_forward_definition_bundle,
)


@pytest.fixture
def publication_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, _ = _bundle(root, frozen=True)
    store = root / "research" / "shadow_v1" / "definitions"
    store.mkdir(mode=0o700)
    return root, activation, store


def _snapshot(root):
    result = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        result.append((
            path.relative_to(root).as_posix(), info.st_dev, info.st_ino,
            info.st_mode, info.st_size, info.st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        ))
    return result


def _prepared(value):
    root, activation, store = value
    return prepare_frozen_shadow_forward_definition_publication(
        root, CYCLE, activation, store,
    )


def _publish(value, plan=None):
    root, activation, store = value
    plan = _prepared(value) if plan is None else plan
    return publish_frozen_shadow_forward_definition_bundle(
        root, CYCLE, activation, store, plan.definition_set_id,
        dict(plan.request_digest), AUTHORIZATION_ACTION,
    )


def test_preflight_freshly_rebuilds_one_frozen_request_and_is_strictly_zero_write(publication_workspace):
    before = _snapshot(publication_workspace[0])
    plan = _prepared(publication_workspace)
    assert plan.to_safe_summary_dict()["status"] == "PREPARED"
    assert plan.publication_capability is False
    assert _snapshot(publication_workspace[0]) == before
    assert not (publication_workspace[2] / plan.definition_set_id).exists()


def test_publish_reobserves_instead_of_accepting_a_carried_candidate_plan_or_request(publication_workspace, monkeypatch):
    import quantpits.research.forward_definition_publication as module
    plan = _prepared(publication_workspace)
    calls = []
    original = module._fresh_candidate

    def observed(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, "_fresh_candidate", observed)
    result = _publish(publication_workspace, plan)
    assert result.status == "COMMITTED"
    assert len(calls) == 1
    with pytest.raises(FrozenDefinitionPublicationContractError):
        copy.copy(plan)


@pytest.mark.parametrize("field", ["action", "id", "algorithm", "domain", "size", "bool"])
def test_publish_requires_exact_action_definition_id_and_full_typed_request_digest(publication_workspace, field):
    root, activation, store = publication_workspace
    plan = _prepared(publication_workspace)
    identifier = plan.definition_set_id
    digest = dict(plan.request_digest)
    action = AUTHORIZATION_ACTION
    if field == "action": action = "PUBLISH"
    elif field == "id": identifier = "foreign.id"
    elif field == "algorithm": digest["algorithm"] = "sha512"
    elif field == "domain": digest["domain"] = "raw_bytes"
    elif field == "size": digest["size_bytes"] += 1
    else: digest["size_bytes"] = True
    with pytest.raises(FrozenDefinitionPublicationContractError):
        publish_frozen_shadow_forward_definition_bundle(
            root, CYCLE, activation, store, identifier, digest, action,
        )
    assert tuple(store.iterdir()) == ()


def test_first_publish_commits_exact_c0_bundle_and_outer_result_joins_all_identities(publication_workspace):
    result = _publish(publication_workspace)
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "COMMITTED"
    assert summary["did_write"] is True
    assert summary["store_member_count"] == 4
    assert summary["definition_bytes_durable"] is True
    target = publication_workspace[2] / summary["definition_set_id"]
    assert {path.name for path in target.iterdir()} == {
        "protocol.json", "execution_assumption.json", "champion_strategy.json",
        "challenger_strategy.json", "definition_manifest.json",
    }


def test_exact_replay_is_adopted_without_any_filesystem_change(publication_workspace):
    first = _publish(publication_workspace)
    before = _snapshot(publication_workspace[0])
    second = _publish(publication_workspace)
    assert first.status == "COMMITTED" and second.status == "ADOPTED"
    assert second.to_safe_summary_dict()["did_write"] is False
    assert _snapshot(publication_workspace[0]) == before


def test_existing_conflict_and_c0_uncertain_never_grant_definition_byte_capability(publication_workspace, monkeypatch):
    plan = _prepared(publication_workspace)
    target = publication_workspace[2] / plan.definition_set_id
    target.mkdir(mode=0o700)
    conflict = _publish(publication_workspace, plan)
    assert conflict.status == "CONFLICT"
    assert conflict.definition_bytes_durable is False
    target.rmdir()
    from quantpits.research import definition_store
    original = definition_store._write_exclusive
    calls = {"count": 0}

    def fail_after_create(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("synthetic")
        return original(*args, **kwargs)

    monkeypatch.setattr(definition_store, "_write_exclusive", fail_after_create)
    uncertain = _publish(publication_workspace, plan)
    assert uncertain.status == "UNCERTAIN"
    assert uncertain.to_safe_summary_dict()["did_write"] is True
    assert uncertain.definition_bytes_durable is False


@pytest.mark.parametrize("node", ["research", "shadow", "activations", "store", "activation"])
def test_workspace_research_ancestor_store_and_target_identity_drift_fail_closed(publication_workspace, monkeypatch, node):
    import quantpits.research.forward_definition_publication as module
    original = module._fresh_candidate

    def drift(root, cycle, activation):
        candidate = original(root, cycle, activation)
        if node == "activation":
            info = activation.stat()
            activation.write_bytes(activation.read_bytes())
            os.utime(activation, ns=(info.st_atime_ns, info.st_mtime_ns))
        else:
            path = {
                "research": root / "research", "shadow": root / "research" / "shadow_v1",
                "activations": activation.parent, "store": publication_workspace[2],
            }[node]
            path.rename(path.with_name(path.name + "-away"))
        return candidate

    monkeypatch.setattr(module, "_fresh_candidate", drift)
    with pytest.raises(FrozenDefinitionPublicationInputError):
        _publish(publication_workspace)


def test_engineering_activation_and_nonformal_paths_can_never_reach_c0_publish(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, _ = _bundle(root)
    research = root / "research"
    research.mkdir(mode=0o700)
    shadow = research / "shadow_v1"
    shadow.mkdir(mode=0o700)
    activations = shadow / "activations"
    activations.mkdir(mode=0o700)
    store = shadow / "definitions"
    store.mkdir(mode=0o700)
    from quantpits.research import definition_store
    monkeypatch.setattr(
        definition_store.CreateOnlyDefinitionBundleStore, "publish",
        lambda *_args: (_ for _ in ()).throw(AssertionError("C0 reached")),
    )
    with pytest.raises(FrozenDefinitionPublicationInputError):
        prepare_frozen_shadow_forward_definition_publication(root, CYCLE, activation, store)


def test_outer_result_is_writer_owned_and_rejects_impossible_status_cross_fields(publication_workspace):
    with pytest.raises(FrozenDefinitionPublicationContractError):
        FrozenDefinitionPublicationPlan()
    with pytest.raises(FrozenDefinitionPublicationContractError):
        FrozenDefinitionBytePublicationResult()
    result = _publish(publication_workspace)
    object.__setattr__(result._receipt, "member_count", 3)
    with pytest.raises(FrozenDefinitionPublicationContractError):
        _ = result.status


def test_only_the_exact_target_subtree_can_change_and_extra_write_is_detected(publication_workspace, monkeypatch):
    from quantpits.research import definition_store
    original = definition_store.CreateOnlyDefinitionBundleStore.publish

    def publish_with_extra_write(store, request):
        receipt = original(store, request)
        (publication_workspace[2].parent / "foreign.txt").write_text("foreign")
        return receipt

    monkeypatch.setattr(
        definition_store.CreateOnlyDefinitionBundleStore, "publish",
        publish_with_extra_write,
    )
    plan = _prepared(publication_workspace)
    result = _publish(publication_workspace, plan)
    assert result.status == "UNCERTAIN"
    assert result.definition_bytes_durable is False


def test_ordinary_failure_before_c0_is_zero_write_and_after_create_remains_uncertain(publication_workspace, monkeypatch):
    import quantpits.research.forward_definition_publication as module
    before = _snapshot(publication_workspace[0])
    monkeypatch.setattr(
        module, "_fresh_candidate",
        lambda *_args: (_ for _ in ()).throw(FrozenDefinitionPublicationInputError("synthetic")),
    )
    with pytest.raises(FrozenDefinitionPublicationInputError):
        _publish(publication_workspace)
    assert _snapshot(publication_workspace[0]) == before


@pytest.mark.parametrize("seam", ["observation", "store", "result"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates_from_observation_store_and_result_seams(publication_workspace, monkeypatch, seam, exception):
    import quantpits.research.forward_definition_publication as module
    if seam == "observation":
        monkeypatch.setattr(module, "_fresh_candidate", lambda *_args: (_ for _ in ()).throw(exception()))
    elif seam == "store":
        from quantpits.research import definition_store
        monkeypatch.setattr(definition_store.CreateOnlyDefinitionBundleStore, "publish", lambda *_args: (_ for _ in ()).throw(exception()))
    else:
        monkeypatch.setattr(FrozenDefinitionBytePublicationResult, "__init__", lambda *_args, **_kwargs: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception):
        _publish(publication_workspace)


def test_safe_plan_and_result_exclude_private_sources_paths_actor_hypothesis_and_economics(publication_workspace):
    plan_text = json.dumps(_prepared(publication_workspace).to_safe_summary_dict(), sort_keys=True)
    result_text = json.dumps(_publish(publication_workspace).to_safe_summary_dict(), sort_keys=True)
    for private in (
        str(publication_workspace[0]), "MODEL_A", "PRIVATE_OWNER",
        "Engineering-only leave-one-out observation.", "0.001",
    ):
        assert private not in plan_text + result_text


def test_publication_import_preserves_environment_cwd_and_optional_dependency_state():
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    __import__("quantpits.research.forward_definition_publication")
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before
