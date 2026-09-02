import copy
import json
import os
import sys
from pathlib import Path

import pytest

from tests.quantpits.research.test_forward_observation import (
    CYCLE, _bundle, _fresh_split_bundle,
)
from quantpits.research.forward_definition_publication import (
    AUTHORIZATION_ACTION,
    FRESH_AUTHORIZATION_ACTION,
    FrozenDefinitionBytePublicationResult,
    FrozenDefinitionPublicationContractError,
    FrozenDefinitionPublicationInputError,
    FrozenDefinitionPublicationPlan,
    prepare_frozen_shadow_forward_definition_publication,
    prepare_fresh_champion_segment_definition_publication,
    publish_frozen_shadow_forward_definition_bundle,
    publish_fresh_champion_segment_definition_bundle,
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


def test_workspace_root_move_away_and_back_after_c0_is_uncertain(publication_workspace, monkeypatch):
    from quantpits.research import definition_store
    original = definition_store.CreateOnlyDefinitionBundleStore.publish
    root = publication_workspace[0]
    displaced = root.with_name(root.name + "-away")
    before = root.lstat()

    def publish_then_move_workspace_away_and_back(store, request):
        receipt = original(store, request)
        root.rename(displaced)
        displaced.rename(root)
        return receipt

    monkeypatch.setattr(
        definition_store.CreateOnlyDefinitionBundleStore, "publish",
        publish_then_move_workspace_away_and_back,
    )
    result = _publish(publication_workspace)
    summary = result.to_safe_summary_dict()
    after = root.lstat()
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev, before.st_ino, before.st_mode,
    )
    assert after.st_ctime_ns != before.st_ctime_ns
    assert summary["status"] == "UNCERTAIN"
    assert summary["did_write"] is True
    assert summary["store_member_count"] == 4
    assert summary["definition_bytes_durable"] is False


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

    monkeypatch.undo()
    plan = _prepared(publication_workspace)
    original = module._same_identities
    calls = {"count": 0}

    def fail_post_c0_observation(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic post-C0 observation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_same_identities", fail_post_c0_observation)
    result = _publish(publication_workspace, plan)
    summary = result.to_safe_summary_dict()
    assert calls["count"] == 2
    assert summary["status"] == "UNCERTAIN"
    assert summary["did_write"] is True
    assert summary["store_member_count"] == 4
    assert summary["definition_bytes_durable"] is False
    assert (publication_workspace[2] / summary["definition_set_id"]).is_dir()


@pytest.mark.parametrize("seam", ["observation", "store", "post_observation", "result"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates_from_observation_store_and_result_seams(publication_workspace, monkeypatch, seam, exception):
    import quantpits.research.forward_definition_publication as module
    if seam == "observation":
        monkeypatch.setattr(module, "_fresh_candidate", lambda *_args: (_ for _ in ()).throw(exception()))
    elif seam == "store":
        from quantpits.research import definition_store
        monkeypatch.setattr(definition_store.CreateOnlyDefinitionBundleStore, "publish", lambda *_args: (_ for _ in ()).throw(exception()))
    elif seam == "post_observation":
        plan = _prepared(publication_workspace)
        original = module._same_identities
        calls = {"count": 0}

        def interrupt_post_c0(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise exception()
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "_same_identities", interrupt_post_c0)
    else:
        monkeypatch.setattr(FrozenDefinitionBytePublicationResult, "__init__", lambda *_args, **_kwargs: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception):
        _publish(publication_workspace, plan if seam == "post_observation" else None)


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


@pytest.fixture
def fresh_publication_workspace(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    store = research / "research" / "shadow_v1" / "definitions"
    store.mkdir(mode=0o700)
    return production, research, activation, store


def _fresh_prepared(value):
    production, research, activation, store = value
    return prepare_fresh_champion_segment_definition_publication(
        production, research, CYCLE, activation, store,
    )


def _fresh_publish(value, plan=None):
    production, research, activation, store = value
    plan = _fresh_prepared(value) if plan is None else plan
    return publish_fresh_champion_segment_definition_bundle(
        production, research, CYCLE, activation, store,
        plan.definition_set_id, dict(plan.request_digest),
        FRESH_AUTHORIZATION_ACTION,
    )


def test_fresh_publication_preflight_uses_split_authority_without_writes(
    fresh_publication_workspace,
):
    production, research, _activation, store = fresh_publication_workspace
    before = _snapshot(production), _snapshot(research)
    plan = _fresh_prepared(fresh_publication_workspace)
    summary = plan.to_safe_summary_dict()
    assert summary["status"] == "PREPARED"
    assert summary["target_state"] == "ABSENT"
    assert summary["publication_capability"] is False
    assert before == (_snapshot(production), _snapshot(research))
    assert not (store / plan.definition_set_id).exists()


def test_fresh_publish_reobserves_and_calls_c0_exactly_once(
    fresh_publication_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_publication as module
    from quantpits.research import definition_store
    plan = _fresh_prepared(fresh_publication_workspace)
    observations = []
    publications = []
    original_observe = module._fresh_segment_candidate
    original_publish = definition_store.CreateOnlyDefinitionBundleStore.publish

    def observe(*args):
        observations.append(args)
        return original_observe(*args)

    def publish(store, request):
        publications.append(request)
        return original_publish(store, request)

    monkeypatch.setattr(module, "_fresh_segment_candidate", observe)
    monkeypatch.setattr(definition_store.CreateOnlyDefinitionBundleStore, "publish", publish)
    result = _fresh_publish(fresh_publication_workspace, plan)
    assert result.status == "COMMITTED"
    assert len(observations) == len(publications) == 1
    assert dict(publications[0].request_digest) == dict(plan.request_digest)


@pytest.mark.parametrize("mutation", ["old_action", "id", "digest", "bool"])
def test_fresh_publish_requires_new_action_and_exact_owner_identity(
    fresh_publication_workspace, mutation,
):
    production, research, activation, store = fresh_publication_workspace
    plan = _fresh_prepared(fresh_publication_workspace)
    identifier = plan.definition_set_id
    digest = dict(plan.request_digest)
    action = FRESH_AUTHORIZATION_ACTION
    if mutation == "old_action":
        action = AUTHORIZATION_ACTION
    elif mutation == "id":
        identifier = "foreign.definition"
    elif mutation == "digest":
        digest["value"] = "0" * 64
    else:
        digest["size_bytes"] = True
    with pytest.raises(FrozenDefinitionPublicationContractError):
        publish_fresh_champion_segment_definition_bundle(
            production, research, CYCLE, activation, store,
            identifier, digest, action,
        )
    assert tuple(store.iterdir()) == ()


def test_fresh_first_publish_commits_exact_bundle_and_replay_adopts_without_delta(
    fresh_publication_workspace,
):
    production, research, _activation, store = fresh_publication_workspace
    production_before = _snapshot(production)
    first = _fresh_publish(fresh_publication_workspace)
    first_summary = first.to_safe_summary_dict()
    assert first_summary["status"] == "COMMITTED"
    assert first_summary["did_write"] is True
    assert first_summary["store_member_count"] == 4
    assert first_summary["definition_bytes_durable"] is True
    assert _snapshot(production) == production_before
    before = _snapshot(research)
    second = _fresh_publish(fresh_publication_workspace)
    assert second.status == "ADOPTED"
    assert second.to_safe_summary_dict()["did_write"] is False
    assert second.definition_bytes_durable is True
    assert _snapshot(research) == before
    assert len(tuple((store / first_summary["definition_set_id"]).iterdir())) == 5


def test_fresh_existing_partial_target_is_conflict_without_durable_capability(
    fresh_publication_workspace,
):
    _production, _research, _activation, store = fresh_publication_workspace
    plan = _fresh_prepared(fresh_publication_workspace)
    (store / plan.definition_set_id).mkdir(mode=0o700)
    result = _fresh_publish(fresh_publication_workspace, plan)
    assert result.status == "CONFLICT"
    assert result.to_safe_summary_dict()["did_write"] is False
    assert result.definition_bytes_durable is False


def test_fresh_c0_partial_write_remains_uncertain_with_exact_write_fact(
    fresh_publication_workspace, monkeypatch,
):
    from quantpits.research import definition_store
    original = definition_store._write_exclusive
    calls = {"count": 0}

    def fail_after_first_create(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("synthetic")
        return original(*args, **kwargs)

    monkeypatch.setattr(definition_store, "_write_exclusive", fail_after_first_create)
    result = _fresh_publish(fresh_publication_workspace)
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "UNCERTAIN"
    assert summary["did_write"] is True
    assert summary["store_member_count"] == 0
    assert summary["definition_bytes_durable"] is False


@pytest.mark.parametrize("relation", ["same", "production_parent", "research_parent"])
def test_fresh_publication_requires_physically_separate_roots(
    fresh_publication_workspace, relation,
):
    production, research, activation, store = fresh_publication_workspace
    if relation == "same":
        production = research
    elif relation == "production_parent":
        production = research.parent
    else:
        production = research / "nested-production"
        production.mkdir()
    with pytest.raises(FrozenDefinitionPublicationInputError):
        prepare_fresh_champion_segment_definition_publication(
            production, research, CYCLE, activation, store,
        )


@pytest.mark.parametrize("root_name", ["production", "research"])
def test_fresh_root_move_away_and_back_before_c0_fails_zero_write(
    fresh_publication_workspace, monkeypatch, root_name,
):
    import quantpits.research.forward_definition_publication as module
    production, research, _activation, store = fresh_publication_workspace
    root = production if root_name == "production" else research
    displaced = root.with_name(root.name + "-away")
    original = module._fresh_segment_candidate
    plan = _fresh_prepared(fresh_publication_workspace)

    def observe_then_move(*args):
        candidate = original(*args)
        root.rename(displaced)
        displaced.rename(root)
        return candidate

    monkeypatch.setattr(module, "_fresh_segment_candidate", observe_then_move)
    with pytest.raises(FrozenDefinitionPublicationInputError):
        _fresh_publish(fresh_publication_workspace, plan)
    assert tuple(store.iterdir()) == ()


@pytest.mark.parametrize("node", ["activation", "store"])
def test_fresh_research_activation_and_store_drift_before_c0_fails_zero_write(
    fresh_publication_workspace, monkeypatch, node,
):
    import quantpits.research.forward_definition_publication as module
    _production, _research, activation, store = fresh_publication_workspace
    plan = _fresh_prepared(fresh_publication_workspace)
    original = module._fresh_segment_candidate

    def observe_then_drift(*args):
        candidate = original(*args)
        path = activation if node == "activation" else store
        displaced = path.with_name(path.name + "-away")
        path.rename(displaced)
        displaced.rename(path)
        return candidate

    monkeypatch.setattr(module, "_fresh_segment_candidate", observe_then_drift)
    with pytest.raises(FrozenDefinitionPublicationInputError):
        _fresh_publish(fresh_publication_workspace, plan)
    assert tuple(store.iterdir()) == ()


def test_fresh_extra_research_write_after_c0_is_uncertain_and_production_unchanged(
    fresh_publication_workspace, monkeypatch,
):
    from quantpits.research import definition_store
    production, research, _activation, _store = fresh_publication_workspace
    production_before = _snapshot(production)
    original = definition_store.CreateOnlyDefinitionBundleStore.publish

    def publish_with_extra(store, request):
        receipt = original(store, request)
        (research / "foreign.txt").write_text("foreign")
        return receipt

    monkeypatch.setattr(
        definition_store.CreateOnlyDefinitionBundleStore, "publish", publish_with_extra,
    )
    result = _fresh_publish(fresh_publication_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_bytes_durable is False
    assert _snapshot(production) == production_before


def test_fresh_production_root_drift_after_c0_is_uncertain(
    fresh_publication_workspace, monkeypatch,
):
    from quantpits.research import definition_store
    production, _research, _activation, _store = fresh_publication_workspace
    displaced = production.with_name(production.name + "-away")
    original = definition_store.CreateOnlyDefinitionBundleStore.publish

    def publish_then_move(store, request):
        receipt = original(store, request)
        production.rename(displaced)
        displaced.rename(production)
        return receipt

    monkeypatch.setattr(
        definition_store.CreateOnlyDefinitionBundleStore, "publish", publish_then_move,
    )
    result = _fresh_publish(fresh_publication_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_bytes_durable is False


def test_fresh_final_public_target_mutation_is_uncertain(
    fresh_publication_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_publication as module
    original = module._public_target_exact

    def mutate_then_verify(store, request, receipt):
        member = store / request.definition_set_id / "protocol.json"
        member.write_bytes(member.read_bytes() + b" ")
        return original(store, request, receipt)

    monkeypatch.setattr(module, "_public_target_exact", mutate_then_verify)
    result = _fresh_publish(fresh_publication_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_bytes_durable is False


def test_fresh_non_target_mutation_during_final_target_verification_is_uncertain(
    fresh_publication_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_publication as module
    _production, research, _activation, _store = fresh_publication_workspace
    original = module._public_target_exact

    def verify_then_mutate(store, request, receipt):
        exact = original(store, request, receipt)
        (research / "research" / "shadow_v1" / "late.txt").write_text("late")
        return exact

    monkeypatch.setattr(module, "_public_target_exact", verify_then_mutate)
    result = _fresh_publish(fresh_publication_workspace)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True
    assert result.definition_bytes_durable is False


def test_fresh_ordinary_failure_before_c0_is_zero_write_and_post_c0_is_uncertain(
    fresh_publication_workspace, monkeypatch,
):
    import quantpits.research.forward_definition_publication as module
    production, research, _activation, _store = fresh_publication_workspace
    plan = _fresh_prepared(fresh_publication_workspace)
    before = _snapshot(production), _snapshot(research)
    monkeypatch.setattr(
        module, "_fresh_segment_candidate",
        lambda *_args: (_ for _ in ()).throw(FrozenDefinitionPublicationInputError("synthetic")),
    )
    with pytest.raises(FrozenDefinitionPublicationInputError):
        _fresh_publish(fresh_publication_workspace, plan)
    assert before == (_snapshot(production), _snapshot(research))

    monkeypatch.undo()
    monkeypatch.setattr(
        module, "_public_target_exact",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic")),
    )
    result = _fresh_publish(fresh_publication_workspace, plan)
    assert result.status == "UNCERTAIN"
    assert result.to_safe_summary_dict()["did_write"] is True


@pytest.mark.parametrize("seam", ["observation", "store", "postcondition", "result"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fresh_process_control_propagates_at_all_outer_seams(
    fresh_publication_workspace, monkeypatch, seam, exception,
):
    import quantpits.research.forward_definition_publication as module
    plan = _fresh_prepared(fresh_publication_workspace)
    if seam == "observation":
        monkeypatch.setattr(
            module, "_fresh_segment_candidate",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    elif seam == "store":
        from quantpits.research import definition_store
        monkeypatch.setattr(
            definition_store.CreateOnlyDefinitionBundleStore, "publish",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    elif seam == "postcondition":
        monkeypatch.setattr(
            module, "_public_target_exact",
            lambda *_args: (_ for _ in ()).throw(exception()),
        )
    else:
        monkeypatch.setattr(
            FrozenDefinitionBytePublicationResult, "__init__",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(exception()),
        )
    with pytest.raises(exception):
        _fresh_publish(fresh_publication_workspace, plan)


def test_fresh_safe_summaries_exclude_both_roots_and_private_values(
    fresh_publication_workspace,
):
    production, research, _activation, _store = fresh_publication_workspace
    text = json.dumps(
        {
            "plan": _fresh_prepared(fresh_publication_workspace).to_safe_summary_dict(),
            "result": _fresh_publish(fresh_publication_workspace).to_safe_summary_dict(),
        },
        sort_keys=True,
    )
    for private in (str(production), str(research), "MODEL_A", "PRIVATE_OWNER", "0.001"):
        assert private not in text
