import copy
import json
import os
import shutil
import stat

import pytest

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes

from tests.quantpits.research.test_forward_definition_evidence import (
    CYCLE,
    _fresh_publish as publish_fresh_evidence,
    _publish as publish_evidence,
    _snapshot,
    evidence_workspace,
    fresh_evidence_workspace,
)
from tests.quantpits.research.test_forward_portfolio_source import (
    CYCLE as SOURCE_CYCLE,
    _problem,
    _write_bundle,
)
from quantpits.research.forward_bootstrap import (
    AUTHORIZATION_ACTION,
    FRESH_BOOTSTRAP_AUTHORIZATION_ACTION,
    ForwardBootstrapContractError,
    MatchedForwardBootstrapPlan,
    MatchedForwardBootstrapResult,
    MatchedForwardBootstrapStoreReceipt,
    prepare_matched_forward_bootstrap,
    prepare_fresh_champion_segment_matched_bootstrap,
    publish_matched_forward_bootstrap,
    publish_fresh_champion_segment_matched_bootstrap,
)


@pytest.fixture
def bootstrap_workspace(evidence_workspace, tmp_path):
    root, activation, definitions, evidence = evidence_workspace
    assert publish_evidence(evidence_workspace).status == "COMMITTED"
    source = tmp_path / "portfolio-source"
    source_cycle = _write_bundle(source, problems=(_problem(),))
    cycle_parent = root / "data" / "evidence" / "v1" / "cycles"
    shutil.move(str(source_cycle), str(cycle_parent / SOURCE_CYCLE))
    bootstraps = root / "research" / "shadow_v1" / "bootstraps"
    bootstraps.mkdir(mode=0o700)
    bootstraps.chmod(0o700)
    return root, activation, definitions, evidence, bootstraps


def _prepare(value):
    root, activation, definitions, evidence, bootstraps = value
    return prepare_matched_forward_bootstrap(
        root, CYCLE, SOURCE_CYCLE, activation, definitions, evidence, bootstraps,
    )


def _publish(value, plan=None, **changes):
    plan = _prepare(value) if plan is None else plan
    root, activation, definitions, evidence, bootstraps = value
    fields = {"identifier": plan.bootstrap_set_id,
              "digest": dict(plan.bootstrap_request_digest),
              "action": AUTHORIZATION_ACTION}
    fields.update(changes)
    return publish_matched_forward_bootstrap(
        root, CYCLE, SOURCE_CYCLE, activation, definitions, evidence, bootstraps,
        fields["identifier"], fields["digest"], fields["action"],
    )


def _rewrite_source_portfolio(root, portfolio):
    cycle = root / "data" / "evidence" / "v1" / "cycles" / SOURCE_CYCLE
    data = canonical_json_bytes(portfolio)
    path = cycle / "portfolio_state.json"
    path.write_bytes(data); path.chmod(0o600)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    manifest["portfolio_state"]["canonical_digest"] = TypedDigest.canonical(
        portfolio, "semantic_config",
    ).to_dict()
    manifest["portfolio_state"]["holding_count"] = len(portfolio["current_holding"])
    replay = {key: value for key, value in manifest.items()
              if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}}
    manifest["request_content_digest"] = TypedDigest.canonical(replay).to_dict()
    manifest_data = canonical_json_bytes(manifest)
    (cycle / "manifest.json").write_bytes(manifest_data)
    (cycle / "manifest.json").chmod(0o600)
    seal = json.loads((cycle / "seal.json").read_bytes())
    seal["manifest_digest"] = TypedDigest.raw(manifest_data).to_dict()
    seal["named_file_digests"]["portfolio_state.json"] = TypedDigest.raw(data).to_dict()
    seal["artifact_root_digest"] = TypedDigest.canonical({
        "objects": seal["object_digests"], "named_files": seal["named_file_digests"],
    }).to_dict()
    (cycle / "seal.json").write_bytes(canonical_json_bytes(seal))
    (cycle / "seal.json").chmod(0o600)


@pytest.fixture
def fresh_bootstrap_workspace(fresh_evidence_workspace, tmp_path):
    assert publish_fresh_evidence(fresh_evidence_workspace).status == "COMMITTED"
    production, research, activation, definitions, evidence = fresh_evidence_workspace
    source = tmp_path / "fresh-portfolio-source"
    source_cycle = _write_bundle(source, problems=(_problem(),))
    cycle_parent = production / "data" / "evidence" / "v1" / "cycles"
    shutil.move(str(source_cycle), str(cycle_parent / SOURCE_CYCLE))
    bootstraps = research / "research" / "shadow_v1" / "bootstraps"
    bootstraps.mkdir(mode=0o700)
    bootstraps.chmod(0o700)
    return production, research, activation, definitions, evidence, bootstraps


def _fresh_bootstrap_prepare(value):
    return prepare_fresh_champion_segment_matched_bootstrap(
        value[0], value[1], CYCLE, SOURCE_CYCLE, *value[2:],
    )


def _fresh_bootstrap_publish(value, plan=None, **changed):
    plan = _fresh_bootstrap_prepare(value) if plan is None else plan
    values = {
        "identifier": plan.bootstrap_set_id,
        "digest": dict(plan.bootstrap_request_digest),
        "action": FRESH_BOOTSTRAP_AUTHORIZATION_ACTION,
    }
    values.update(changed)
    return publish_fresh_champion_segment_matched_bootstrap(
        value[0], value[1], CYCLE, SOURCE_CYCLE, *value[2:],
        values["identifier"], values["digest"], values["action"],
    )


def test_fresh_bootstrap_preflight_is_split_root_and_zero_write(
    fresh_bootstrap_workspace,
):
    production, research = fresh_bootstrap_workspace[:2]
    before = _snapshot(production), _snapshot(research)
    plan = _fresh_bootstrap_prepare(fresh_bootstrap_workspace)
    summary = plan.to_safe_summary_dict()
    assert summary["status"] == "PREPARED"
    assert summary["target_state"] == "ABSENT"
    assert summary["definition_evidence_status"] == "ADOPTED"
    assert summary["matched_economics_checked"] is True
    assert before == (_snapshot(production), _snapshot(research))


def test_fresh_bootstrap_commit_and_exact_adopt_preserve_all_nontargets(
    fresh_bootstrap_workspace,
):
    production, research = fresh_bootstrap_workspace[:2]
    plan = _fresh_bootstrap_prepare(fresh_bootstrap_workspace)
    production_before = _snapshot(production)
    research_before = _snapshot(research)
    result = _fresh_bootstrap_publish(fresh_bootstrap_workspace, plan)
    assert result.status == "COMMITTED" and result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is True
    target = fresh_bootstrap_workspace[-1] / plan.bootstrap_set_id
    assert {path.name for path in target.iterdir()} == {
        "source_receipt.json", "champion_state.json",
        "challenger_state.json", "bootstrap_manifest.json",
    }
    assert _snapshot(production) == production_before
    assert len(_snapshot(research)) == len(research_before) + 5
    before_replay = _snapshot(production), _snapshot(research)
    replay = _fresh_bootstrap_publish(fresh_bootstrap_workspace, plan)
    assert replay.status == "ADOPTED" and replay.receipt.did_write is False
    assert replay.portfolio_bootstrap_complete is True
    assert before_replay == (_snapshot(production), _snapshot(research))


@pytest.mark.parametrize("relation", ["same", "production_parent", "research_parent"])
def test_fresh_bootstrap_requires_physical_root_separation(
    fresh_bootstrap_workspace, relation,
):
    production, research, activation, definitions, evidence, bootstraps = (
        fresh_bootstrap_workspace
    )
    if relation == "same":
        production = research
    elif relation == "production_parent":
        production = research.parent
    else:
        production = research / "nested-production"
        production.mkdir()
    with pytest.raises(ForwardBootstrapContractError):
        prepare_fresh_champion_segment_matched_bootstrap(
            production, research, CYCLE, SOURCE_CYCLE, activation, definitions,
            evidence, bootstraps,
        )


def test_fresh_bootstrap_owner_join_blocks_before_writer(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    plan = _fresh_bootstrap_prepare(fresh_bootstrap_workspace)
    calls = []
    monkeypatch.setattr(module._Store, "publish", lambda *_args: calls.append(True))
    before = _snapshot(fresh_bootstrap_workspace[1])
    digest = dict(plan.bootstrap_request_digest)
    digest["value"] = "0" * 64
    with pytest.raises(ForwardBootstrapContractError):
        _fresh_bootstrap_publish(
            fresh_bootstrap_workspace, plan, digest=digest,
        )
    assert calls == [] and _snapshot(fresh_bootstrap_workspace[1]) == before


def test_fresh_bootstrap_publish_reobserves_each_authority_and_writes_once(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    import quantpits.research.forward_definition_evidence as evidence_module
    import quantpits.research.forward_observation as observation_module
    import quantpits.research.forward_portfolio_source as portfolio_module
    from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
    plan = _fresh_bootstrap_prepare(fresh_bootstrap_workspace)
    calls = {
        "evidence": 0, "candidate": 0, "definition": 0,
        "portfolio": 0, "store": 0,
    }
    original_evidence = (
        evidence_module.adopt_fresh_champion_segment_definition_evidence
    )
    original_candidate = observation_module.observe_fresh_champion_segment_candidate
    original_definition = CreateOnlyDefinitionBundleStore.adopt_existing
    original_portfolio = portfolio_module.observe_forward_portfolio_source
    original_store = module._Store.publish

    def evidence(*args):
        calls["evidence"] += 1
        return original_evidence(*args)

    def candidate(*args):
        calls["candidate"] += 1
        return original_candidate(*args)

    def portfolio(*args):
        calls["portfolio"] += 1
        return original_portfolio(*args)

    def definition(*args):
        calls["definition"] += 1
        return original_definition(*args)

    def store(*args):
        calls["store"] += 1
        return original_store(*args)

    monkeypatch.setattr(
        evidence_module,
        "adopt_fresh_champion_segment_definition_evidence", evidence,
    )
    monkeypatch.setattr(
        observation_module, "observe_fresh_champion_segment_candidate", candidate,
    )
    monkeypatch.setattr(
        CreateOnlyDefinitionBundleStore, "adopt_existing", definition,
    )
    monkeypatch.setattr(
        portfolio_module, "observe_forward_portfolio_source", portfolio,
    )
    monkeypatch.setattr(module._Store, "publish", store)
    result = _fresh_bootstrap_publish(fresh_bootstrap_workspace, plan)
    assert result.status == "COMMITTED"
    # One candidate observation is owned by the evidence adopter and one
    # independently rejoins the pair compiler in the same logical call.
    assert calls == {
        "evidence": 1, "candidate": 2, "definition": 1,
        "portfolio": 1, "store": 1,
    }


def test_fresh_bootstrap_chronology_blocks_before_writer(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    calls = []
    monkeypatch.setattr(module._Store, "publish", lambda *_args: calls.append(True))
    values = list(fresh_bootstrap_workspace)
    with pytest.raises(ForwardBootstrapContractError):
        prepare_fresh_champion_segment_matched_bootstrap(
            values[0], values[1], CYCLE, "2026-08-13", *values[2:],
        )
    assert calls == []


def test_fresh_bootstrap_source_drift_before_writer_is_zero_write_blocked(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    import quantpits.research.forward_portfolio_source as portfolio_module
    original = portfolio_module.observe_forward_portfolio_source
    calls = []
    plan = _fresh_bootstrap_prepare(fresh_bootstrap_workspace)

    def mutate(*args):
        result = original(*args)
        path = fresh_bootstrap_workspace[0] / "unexpected-before-writer"
        path.write_text("private")
        return result

    monkeypatch.setattr(portfolio_module, "observe_forward_portfolio_source", mutate)
    monkeypatch.setattr(module._Store, "publish", lambda *_args: calls.append(True))
    bootstrap_before = _snapshot(fresh_bootstrap_workspace[-1])
    with pytest.raises(ForwardBootstrapContractError):
        _fresh_bootstrap_publish(fresh_bootstrap_workspace, plan)
    assert calls == []
    assert _snapshot(fresh_bootstrap_workspace[-1]) == bootstrap_before


def test_fresh_bootstrap_post_writer_drift_is_uncertain_with_write_fact(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    original = module._Store.publish

    def extra(store, request):
        receipt = original(store, request)
        path = fresh_bootstrap_workspace[1] / "unexpected"
        path.write_text("private")
        return receipt

    monkeypatch.setattr(module._Store, "publish", extra)
    result = _fresh_bootstrap_publish(fresh_bootstrap_workspace)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is False


def test_fresh_bootstrap_production_drift_after_writer_is_uncertain(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    original = module._Store.publish

    def mutate(store, request):
        receipt = original(store, request)
        path = fresh_bootstrap_workspace[0] / "config" / "strategy_config.yaml"
        path.write_bytes(path.read_bytes() + b"\n")
        return receipt

    monkeypatch.setattr(module._Store, "publish", mutate)
    result = _fresh_bootstrap_publish(fresh_bootstrap_workspace)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is False


def test_fresh_bootstrap_final_public_verification_failure_is_uncertain(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    monkeypatch.setattr(module, "_verify_public_bootstrap", lambda *_args: False)
    result = _fresh_bootstrap_publish(fresh_bootstrap_workspace)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.receipt.member_count == 3
    assert result.portfolio_bootstrap_complete is False


def test_fresh_bootstrap_post_store_observer_failure_preserves_write_fact(
    fresh_bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    plan = _fresh_bootstrap_prepare(fresh_bootstrap_workspace)
    original = module.SourceMutationObserver
    matches = []

    def observer(root, paths):
        if root == fresh_bootstrap_workspace[-1] and tuple(paths) == (
            plan.bootstrap_set_id,
        ):
            matches.append(True)
            if len(matches) == 4:
                raise OSError("synthetic post-store observer failure")
        return original(root, paths)

    monkeypatch.setattr(module, "SourceMutationObserver", observer)
    result = _fresh_bootstrap_publish(fresh_bootstrap_workspace, plan)
    assert len(matches) == 4
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.receipt.member_count == 3
    assert result.portfolio_bootstrap_complete is False


@pytest.mark.parametrize("seam", ["evidence", "candidate", "portfolio", "writer", "result"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fresh_bootstrap_process_control_propagates_at_every_seam(
    fresh_bootstrap_workspace, monkeypatch, seam, exception,
):
    import quantpits.research.forward_bootstrap as module
    import quantpits.research.forward_definition_evidence as evidence_module
    import quantpits.research.forward_observation as observation_module
    import quantpits.research.forward_portfolio_source as portfolio_module
    plan = _fresh_bootstrap_prepare(fresh_bootstrap_workspace)
    fail = lambda *_args, **_kwargs: (_ for _ in ()).throw(exception())
    if seam == "evidence":
        monkeypatch.setattr(
            evidence_module,
            "adopt_fresh_champion_segment_definition_evidence", fail,
        )
    elif seam == "candidate":
        monkeypatch.setattr(
            observation_module, "observe_fresh_champion_segment_candidate", fail,
        )
    elif seam == "portfolio":
        monkeypatch.setattr(
            portfolio_module, "observe_forward_portfolio_source", fail,
        )
    elif seam == "writer":
        monkeypatch.setattr(module._Store, "publish", fail)
    else:
        monkeypatch.setattr(module, "MatchedForwardBootstrapResult", fail)
    with pytest.raises(exception):
        _fresh_bootstrap_publish(fresh_bootstrap_workspace, plan)


def test_preflight_freshly_adopts_c1b_and_reads_exact_sealed_portfolio_zero_write(
    bootstrap_workspace,
):
    before = _snapshot(bootstrap_workspace[0])
    plan = _prepare(bootstrap_workspace)
    summary = plan.to_safe_summary_dict()
    assert summary["status"] == "PREPARED"
    assert summary["definition_evidence_status"] == "ADOPTED"
    assert summary["matched_economics_checked"] is True
    assert summary["target_state"] == "ABSENT"
    assert summary["portfolio_bootstrap_complete"] is False
    assert _snapshot(bootstrap_workspace[0]) == before


def test_first_publish_commits_exact_three_members_and_manifest_last(bootstrap_workspace):
    plan = _prepare(bootstrap_workspace)
    result = _publish(bootstrap_workspace, plan)
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "COMMITTED" and summary["did_write"] is True
    assert summary["portfolio_bootstrap_complete"] is True
    assert all(summary[name] is False for name in (
        "state_chain_advanced", "intent_sealed", "epoch_started",
        "prospective_claim", "promotion_capability",
    ))
    target = bootstrap_workspace[-1] / plan.bootstrap_set_id
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert {item.name for item in target.iterdir()} == {
        "source_receipt.json", "champion_state.json", "challenger_state.json",
        "bootstrap_manifest.json",
    }
    champion = json.loads((target / "champion_state.json").read_bytes())
    challenger = json.loads((target / "challenger_state.json").read_bytes())
    assert champion["portfolio_id"] != challenger["portfolio_id"]
    for key in ("as_of_date", "cash", "positions"):
        assert champion[key] == challenger[key]
    assert champion["cash"] == "-1.25"
    assert [row["instrument"] for row in champion["positions"]] == ["SH600001", "SZ000002"]
    terminal_identity = result.receipt.to_dict()["bootstrap_bundle_root_identity"]
    assert terminal_identity[3] >= 2 and terminal_identity[4] > 0


@pytest.mark.parametrize("field", [
    "definition_evidence_operation_id",
    "phase37a_seal_digest",
    "phase37a_manifest_digest",
    "source_portfolio_semantic_digest",
    "source_observation_status",
    "source_cycle_status",
    "source_problem_inventory_digest",
    "source_portfolio_holding_count",
    "portfolio_member_verified",
])
def test_bootstrap_set_id_binds_complete_definition_and_source_provenance(
    bootstrap_workspace, monkeypatch, field,
):
    import quantpits.research.forward_bootstrap as module
    original_plan = _prepare(bootstrap_workspace)
    original = module._source_receipt

    def drift(*args, **kwargs):
        receipt = original(*args, **kwargs)
        if field.endswith("_digest"):
            receipt[field] = dict(receipt[field])
            receipt[field]["value"] = "0" * 64
        elif field == "definition_evidence_operation_id":
            receipt[field] = "0" * 64
        elif field == "source_portfolio_holding_count":
            receipt[field] += 1
        elif field == "portfolio_member_verified":
            receipt[field] = False
        else:
            receipt[field] = "SYNTHETIC_PROVENANCE_DRIFT"
        return receipt

    monkeypatch.setattr(module, "_source_receipt", drift)
    drifted_plan = _prepare(bootstrap_workspace)
    assert drifted_plan.bootstrap_set_id != original_plan.bootstrap_set_id


def test_exact_replay_is_adopted_without_filesystem_change(bootstrap_workspace):
    plan = _prepare(bootstrap_workspace)
    assert _publish(bootstrap_workspace, plan).status == "COMMITTED"
    before = _snapshot(bootstrap_workspace[0])
    replay = _publish(bootstrap_workspace, plan)
    assert replay.status == "ADOPTED" and replay.receipt.did_write is False
    assert replay.portfolio_bootstrap_complete is True
    assert _snapshot(bootstrap_workspace[0]) == before


@pytest.mark.parametrize("field", ["identifier", "digest", "action", "bool"])
def test_publish_requires_exact_bootstrap_id_full_digest_and_action(
    bootstrap_workspace, field,
):
    plan = _prepare(bootstrap_workspace)
    changes = {}
    if field == "identifier": changes[field] = "bootstrap.foreign"
    elif field == "digest":
        value = dict(plan.bootstrap_request_digest); value["value"] = "0" * 64
        changes[field] = value
    elif field == "action": changes[field] = "PUBLISH"
    else:
        value = dict(plan.bootstrap_request_digest); value["size_bytes"] = True
        changes["digest"] = value
    before = _snapshot(bootstrap_workspace[0])
    with pytest.raises(ForwardBootstrapContractError):
        _publish(bootstrap_workspace, plan, **changes)
    assert _snapshot(bootstrap_workspace[0]) == before


@pytest.mark.parametrize("change", ["cash", "cost", "fractional"])
def test_portfolio_mapping_rejects_unaligned_and_fractional_state_before_writer(
    bootstrap_workspace, monkeypatch, change,
):
    portfolio = {
        "current_cash": "-1.25",
        "current_holding": [
            {"instrument": "SH600001", "value": "100", "amount": "800.5"},
            {"instrument": "SZ000002", "value": "200", "amount": "1500"},
        ],
    }
    if change == "cash": portfolio["current_cash"] = "-1.251"
    elif change == "cost": portfolio["current_holding"][0]["amount"] = "800.501"
    else: portfolio["current_holding"][0]["value"] = "100.5"
    _rewrite_source_portfolio(bootstrap_workspace[0], portfolio)
    import quantpits.research.forward_bootstrap as module
    calls = []
    monkeypatch.setattr(module._Store, "publish", lambda *_: calls.append(True))
    with pytest.raises(ForwardBootstrapContractError):
        _prepare(bootstrap_workspace)
    assert calls == []


def test_source_chronology_is_checked_before_writer(bootstrap_workspace, monkeypatch):
    import quantpits.research.forward_bootstrap as module
    calls = []
    monkeypatch.setattr(module._Store, "publish", lambda *_: calls.append(True))
    root, activation, definitions, evidence, bootstraps = bootstrap_workspace
    with pytest.raises(ForwardBootstrapContractError):
        prepare_matched_forward_bootstrap(
            root, CYCLE, "2026-08-13", activation, definitions, evidence, bootstraps,
        )
    assert calls == []


def test_same_id_different_bytes_is_conflict_without_write(bootstrap_workspace):
    plan = _prepare(bootstrap_workspace)
    target = bootstrap_workspace[-1] / plan.bootstrap_set_id
    target.mkdir(mode=0o700)
    (target / "foreign").write_text("x")
    before = _snapshot(bootstrap_workspace[0])
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "CONFLICT" and result.receipt.did_write is False
    assert result.portfolio_bootstrap_complete is False
    assert _snapshot(bootstrap_workspace[0]) == before


def test_existing_fifo_member_is_conflict_without_blocking_or_write(bootstrap_workspace):
    plan = _prepare(bootstrap_workspace)
    assert _publish(bootstrap_workspace, plan).status == "COMMITTED"
    member = bootstrap_workspace[-1] / plan.bootstrap_set_id / "source_receipt.json"
    member.unlink()
    os.mkfifo(str(member), 0o600)
    before = _snapshot(bootstrap_workspace[0])
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "CONFLICT" and result.receipt.did_write is False
    assert result.portfolio_bootstrap_complete is False
    assert _snapshot(bootstrap_workspace[0]) == before


def test_oversized_member_is_rejected_before_target_creation(
    bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    monkeypatch.setattr(module, "_MAX_MEMBER", 1)
    before = _snapshot(bootstrap_workspace[0])
    with pytest.raises(ForwardBootstrapContractError, match="size limit"):
        _prepare(bootstrap_workspace)
    assert tuple(bootstrap_workspace[-1].iterdir()) == ()
    assert _snapshot(bootstrap_workspace[0]) == before


def test_failure_before_create_is_typed_zero_write_and_partial_prefix_is_uncertain(
    bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    plan = _prepare(bootstrap_workspace)
    before = _snapshot(bootstrap_workspace[0])
    monkeypatch.setattr(module.os, "mkdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(ForwardBootstrapContractError):
        _publish(bootstrap_workspace, plan)
    assert _snapshot(bootstrap_workspace[0]) == before

    monkeypatch.undo()
    original = module._write_all
    calls = []

    def fail_second(descriptor, data):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("synthetic partial failure")
        return original(descriptor, data)

    monkeypatch.setattr(module, "_write_all", fail_second)
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "UNCERTAIN" and result.receipt.did_write is True
    assert result.receipt.member_count == 1
    assert result.portfolio_bootstrap_complete is False
    assert (bootstrap_workspace[-1] / plan.bootstrap_set_id).exists()


@pytest.mark.parametrize("mutation", ["move_away_back", "delete_recreate"])
def test_extra_target_namespace_event_inside_mkdir_is_uncertain_before_member_write(
    bootstrap_workspace, monkeypatch, mutation,
):
    import quantpits.research.forward_bootstrap as module
    plan = _prepare(bootstrap_workspace)
    original = module.os.mkdir

    def transient(path, mode=0o777, *, dir_fd=None):
        original(path, mode, dir_fd=dir_fd)
        target = bootstrap_workspace[-1] / path
        if mutation == "move_away_back":
            displaced = bootstrap_workspace[-1] / "bootstrap-displaced"
            target.rename(displaced)
            displaced.rename(target)
        else:
            target.rmdir()
            original(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "mkdir", transient)
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.receipt.member_count == 0
    assert result.portfolio_bootstrap_complete is False
    assert tuple((bootstrap_workspace[-1] / plan.bootstrap_set_id).iterdir()) == ()


@pytest.mark.parametrize("mutation", ["move_away_back", "delete_recreate"])
def test_current_member_namespace_mutation_is_uncertain_with_zero_verified_prefix(
    bootstrap_workspace, monkeypatch, mutation,
):
    import quantpits.research.forward_bootstrap as module
    plan = _prepare(bootstrap_workspace)
    original = module._write_all
    calls = []

    def transient(descriptor, data):
        original(descriptor, data)
        calls.append(True)
        if len(calls) != 1:
            return
        member = bootstrap_workspace[-1] / plan.bootstrap_set_id / "source_receipt.json"
        if mutation == "move_away_back":
            displaced = member.with_name("source-receipt-displaced")
            member.rename(displaced)
            displaced.rename(member)
        else:
            member.unlink()
            member.write_bytes(data)
            member.chmod(0o600)

    monkeypatch.setattr(module, "_write_all", transient)
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.receipt.member_count == 0
    assert result.portfolio_bootstrap_complete is False
    target = bootstrap_workspace[-1] / plan.bootstrap_set_id
    assert tuple(path.name for path in target.iterdir()) == ("source_receipt.json",)


def test_completed_member_move_away_back_during_later_write_preserves_exact_prefix(
    bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    plan = _prepare(bootstrap_workspace)
    original = module._write_all
    calls = []

    def transient(descriptor, data):
        calls.append(True)
        if len(calls) == 2:
            member = bootstrap_workspace[-1] / plan.bootstrap_set_id / "source_receipt.json"
            displaced = member.with_name("source-receipt-displaced")
            member.rename(displaced)
            displaced.rename(member)
        original(descriptor, data)

    monkeypatch.setattr(module, "_write_all", transient)
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.receipt.member_count == 1
    assert result.portfolio_bootstrap_complete is False
    target = bootstrap_workspace[-1] / plan.bootstrap_set_id
    assert tuple(sorted(path.name for path in target.iterdir())) == (
        "champion_state.json", "source_receipt.json",
    )


def test_manifest_move_away_back_is_uncertain_with_three_member_prefix(
    bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    plan = _prepare(bootstrap_workspace)
    original = module._write_all
    calls = []

    def transient(descriptor, data):
        original(descriptor, data)
        calls.append(True)
        if len(calls) == 4:
            manifest = bootstrap_workspace[-1] / plan.bootstrap_set_id / module.MANIFEST_NAME
            displaced = manifest.with_name("manifest-displaced")
            manifest.rename(displaced)
            displaced.rename(manifest)

    monkeypatch.setattr(module, "_write_all", transient)
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.receipt.member_count == 3
    assert result.portfolio_bootstrap_complete is False
    target = bootstrap_workspace[-1] / plan.bootstrap_set_id
    assert tuple(sorted(path.name for path in target.iterdir())) == tuple(sorted(
        module.MEMBER_PATHS + (module.MANIFEST_NAME,)
    ))


def test_source_drift_is_uncertain_and_denies_capability(bootstrap_workspace, monkeypatch):
    import quantpits.research.forward_bootstrap as module
    original = module._Store.publish

    def mutate(store, request):
        result = original(store, request)
        cycle = bootstrap_workspace[0] / "data" / "evidence" / "v1" / "cycles" / SOURCE_CYCLE
        displaced = cycle.with_name("displaced")
        cycle.rename(displaced); displaced.rename(cycle)
        return result

    monkeypatch.setattr(module._Store, "publish", mutate)
    result = _publish(bootstrap_workspace)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is False


def test_bootstrap_root_move_away_back_is_uncertain(bootstrap_workspace, monkeypatch):
    import quantpits.research.forward_bootstrap as module
    original = module._Store._verify
    moved = []

    def transient(store, request, descriptor, *, conflict):
        if not moved:
            displaced = store.root.with_name("bootstraps-displaced")
            store.root.rename(displaced); displaced.rename(store.root)
            moved.append(True)
        return original(store, request, descriptor, conflict=conflict)

    monkeypatch.setattr(module._Store, "_verify", transient)
    result = _publish(bootstrap_workspace)
    assert result.status == "UNCERTAIN" and result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is False


def test_post_write_observer_failure_and_extra_write_are_uncertain(
    bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    original = module._Store.publish

    def extra(store, request):
        result = original(store, request)
        path = bootstrap_workspace[0] / "unexpected.txt"
        path.write_text("unexpected")
        return result

    monkeypatch.setattr(module._Store, "publish", extra)
    result = _publish(bootstrap_workspace)
    assert result.status == "UNCERTAIN" and result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is False


def test_post_store_target_extra_member_and_public_observer_failure_are_uncertain(
    bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    original = module._Store.publish

    def extra_target(store, request):
        receipt = original(store, request)
        path = store.root / request.bootstrap_set_id / "extra.json"
        path.write_text("{}")
        return receipt

    monkeypatch.setattr(module._Store, "publish", extra_target)
    result = _publish(bootstrap_workspace)
    assert result.status == "UNCERTAIN" and result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is False


@pytest.mark.parametrize("existing", [False, True])
def test_store_to_outer_handoff_move_away_back_is_uncertain(
    bootstrap_workspace, monkeypatch, existing,
):
    import quantpits.research.forward_bootstrap as module
    plan = _prepare(bootstrap_workspace)
    if existing:
        assert _publish(bootstrap_workspace, plan).status == "COMMITTED"
    original = module._Store.publish

    def transient(store, request):
        receipt = original(store, request)
        target = store.root / request.bootstrap_set_id
        displaced = store.root / "bootstrap-displaced"
        target.rename(displaced)
        displaced.rename(target)
        return receipt

    monkeypatch.setattr(module._Store, "publish", transient)
    result = _publish(bootstrap_workspace, plan)
    assert result.status == "UNCERTAIN"
    assert result.receipt.did_write is (not existing)
    assert result.portfolio_bootstrap_complete is False


def test_post_write_public_observer_failure_preserves_write_fact(
    bootstrap_workspace, monkeypatch,
):
    import quantpits.research.forward_bootstrap as module
    monkeypatch.setattr(
        module, "_verify_public_bootstrap",
        lambda *_: (_ for _ in ()).throw(OSError("synthetic observer failure")),
    )
    result = _publish(bootstrap_workspace)
    assert result.status == "UNCERTAIN" and result.receipt.did_write is True
    assert result.portfolio_bootstrap_complete is False


def test_writer_owned_plan_receipt_and_result_reject_public_construction():
    with pytest.raises(ForwardBootstrapContractError): MatchedForwardBootstrapPlan()
    with pytest.raises(ForwardBootstrapContractError): MatchedForwardBootstrapStoreReceipt()
    with pytest.raises(ForwardBootstrapContractError): MatchedForwardBootstrapResult()


def test_safe_summary_excludes_economic_values_instruments_and_paths(bootstrap_workspace):
    summary = _prepare(bootstrap_workspace).to_safe_summary_dict()
    rendered = json.dumps(summary)
    for secret in ("-1.25", "SH600001", "SZ000002", str(bootstrap_workspace[0])):
        assert secret not in rendered


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates_from_writer(bootstrap_workspace, monkeypatch, exception):
    import quantpits.research.forward_bootstrap as module
    monkeypatch.setattr(module._Store, "publish", lambda *_: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception): _publish(bootstrap_workspace)


def test_lazy_public_exports_resolve_without_optional_dependencies():
    import quantpits.research as research
    assert research.prepare_matched_forward_bootstrap is prepare_matched_forward_bootstrap
    assert research.publish_matched_forward_bootstrap is publish_matched_forward_bootstrap
