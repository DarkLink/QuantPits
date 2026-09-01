import copy
import json
import os
import sys
from pathlib import Path

import pytest

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from quantpits.research.forward_observation import (
    FreshChampionSegmentCandidate,
    ForwardObservationContractError,
    ForwardObservationInputError,
    ObservedForwardDefinitionCandidate,
    observe_frozen_shadow_forward_definition_candidate,
    observe_fresh_champion_segment_candidate,
    observe_shadow_forward_definition_candidate,
)


CYCLE = "2026-08-14"


def _assumption():
    return {
        "schema_version": 1, "assumption_id": "execution.engineering.v1",
        "deal_price": "NEXT_OPEN", "sell_before_buy": True,
        "partial_fill": False, "short_sell": False, "lot_size": 100,
        "money_quantum": "0.01", "price_quantum": "0.0001",
        "rounding_mode": "HALF_UP", "buy_slippage_rate": "0.001",
        "sell_slippage_rate": "0.002", "buy_fee_rate": "0.0003",
        "sell_fee_rate": "0.0004", "minimum_buy_fee": "1",
        "minimum_sell_fee": "2", "corporate_action_mode": "UNMODELED",
    }


def _activation(omitted=1):
    return {
        "schema_version": 1, "purpose": "ENGINEERING_VALIDATION",
        "definition_set_id": "shadow.engineering.c1b0",
        "protocol_id": "protocol.engineering.v1",
        "execution_assumption_id": "execution.engineering.v1",
        "intent_definition_id": "intent.current.engineering.v1",
        "champion_strategy_id": "strategy.champion.engineering.v1",
        "challenger_strategy_id": "strategy.challenger.engineering.v1",
        "created_at": "2026-08-27T12:00:00Z", "effective_cycle": "2026-08-28",
        "evidence_cycle_id": CYCLE, "omitted_position": omitted,
        "selection_decision": {
            "decision_id": "DECISION.C1B0.001",
            "decision_time": "2026-08-27T11:00:00Z",
            "evidence_cycle_id": CYCLE, "actor": "PRIVATE_OWNER",
            "decision": "APPROVE", "target": "strategy.challenger.engineering.v1",
            "reason_code": "ENGINEERING_ONLY", "optional_note": None,
        },
        "hypothesis": "Engineering-only leave-one-out observation.",
        "execution_assumption": _assumption(),
    }


def _write(path, data, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mode is not None:
        path.chmod(mode)


def _bundle(root, *, omitted=1, frozen=False):
    config = (
        "strategy:\n"
        "  name: topk_dropout\n"
        "  params:\n"
        "    topk: 22\n"
        "    n_drop: 4\n"
        "    buy_suggestion_factor: 2\n"
        "    sell_out_of_universe: true\n"
        "backtest:\n"
        "  account: 99999999\n"
    ).encode()
    _write(root / "config" / "strategy_config.yaml", config)
    activation_raw = _activation(omitted)
    if frozen:
        activation_raw["purpose"] = "FROZEN_OBSERVATION"
        activation_raw["definition_set_id"] = "shadow.frozen.c1b1a"
        activation_raw["selection_decision"]["reason_code"] = "FROZEN_OBSERVATION"
        activation_path = (
            root / "research" / "shadow_v1" / "activations"
            / "shadow.frozen.c1b1a.json"
        )
    else:
        activation_path = root / "private" / "activation.json"
    _write(activation_path, canonical_json_bytes(activation_raw), 0o600)
    if frozen:
        for directory in (
            root / "research", root / "research" / "shadow_v1",
            root / "research" / "shadow_v1" / "activations",
        ):
            directory.chmod(0o700)
    source_ids = ["MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D"]
    artifact_rows = []
    objects = {}
    source_models = []
    for position, source_id in enumerate(source_ids):
        data = ("artifact-%d" % position).encode()
        digest = TypedDigest.raw(data)
        objects[digest.value] = data
        member = {
            "path": "mlruns/source-%d/model.bin" % position,
            "status": "observed", "digest": digest.to_dict(),
            "preservation_status": "embedded", "detail": "",
        }
        inventory = [{"path": member["path"], "digest": digest.to_dict()}]
        artifact_rows.append({
            "position": position, "role": "source_training",
            "recorder_id": "SOURCE_%d" % position,
            "experiment_name": "TRAIN_%d" % position,
            "artifact_locator": "mlruns/source-%d" % position,
            "members": [member],
            "artifact_tree_digest": TypedDigest.canonical(
                inventory, "file_inventory",
            ).to_dict(),
        })
        source_models.append({
            "resolved_key": source_id, "status": "ready",
            "recorder_id": "PRED_%d" % position,
            "source_recorder_id": "SOURCE_%d" % position,
        })
    semantic_data = canonical_json_bytes({"synthetic": True})
    semantic_raw = TypedDigest.raw(semantic_data)
    objects[semantic_raw.value] = semantic_data
    semantic_observation = {
        "path": "config/synthetic.json", "status": "observed",
        "digest": {**semantic_raw.to_dict(), "domain": "semantic_config"},
        "preservation_status": "embedded", "detail": "",
    }
    ranking = b"instrument,score,rank\nAAA,1,1\n"
    portfolio = canonical_json_bytes({"cash": "1000", "holdings": []})
    ranking_digest = TypedDigest.raw(ranking)
    portfolio_raw = TypedDigest.raw(portfolio)
    named = {
        "ranking.csv": ranking_digest.to_dict(),
        "portfolio_state.json": {
            **portfolio_raw.to_dict(), "domain": "raw_bytes",
        },
    }
    manifest = {
        "schema_version": 1,
        "cycle_identity": {"cycle_id": CYCLE, "research_epoch_id": "SYNTHETIC", "evidence_as_of": CYCLE},
        "engine_identity": {"semantic_observation": semantic_observation},
        "workspace_identity": {},
        "data_identity": {
            "qlib_materialization_identity": {"status": "observed", "calendar_cutoff": CYCLE},
            "source_dolt_identity": {"status": "missing"},
            "source_to_materialization_relation": "unverified",
        },
        "run_evidence": [],
        "model_and_ensemble_lineage": {
            "status": "complete",
            "combo": {"resolved_members": source_ids},
            "source_models": source_models, "source_artifacts": artifact_rows,
        },
        "ranking": {"status": "complete", "ranking_digest": ranking_digest.to_dict()},
        "portfolio_state": {
            "status": "observed",
            "canonical_digest": {**portfolio_raw.to_dict(), "domain": "semantic_config"},
        },
        "decision_state": {"status": "not_recorded", "as_of_capture": True},
        "referenced_evidence": [],
        "preservation": {
            "embedded_object_count": len(objects), "named_file_count": 2,
        },
        "problems": [], "capture_time": "2026-08-27T10:00:00+00:00",
        "status": "sealed_complete", "request_content_digest": {},
    }
    replay_core = {
        key: value for key, value in manifest.items()
        if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
    }
    manifest["request_content_digest"] = TypedDigest.canonical(replay_core).to_dict()
    manifest_data = canonical_json_bytes(manifest)
    object_names = sorted(objects)
    seal = {
        "schema_version": 1, "cycle_id": CYCLE, "status": "sealed_complete",
        "manifest_digest": TypedDigest.raw(manifest_data).to_dict(),
        "artifact_root_digest": TypedDigest.canonical({
            "objects": object_names, "named_files": named,
        }).to_dict(),
        "object_digests": object_names, "named_file_digests": named,
    }
    cycle = root / "data" / "evidence" / "v1" / "cycles" / CYCLE
    for digest, data in objects.items():
        _write(cycle / "objects" / digest[:2] / digest, data, 0o600)
    _write(cycle / "ranking.csv", ranking, 0o600)
    _write(cycle / "portfolio_state.json", portfolio, 0o600)
    _write(cycle / "manifest.json", manifest_data, 0o600)
    _write(cycle / "seal.json", canonical_json_bytes(seal), 0o600)
    (cycle / "objects").chmod(0o700)
    for prefix in (cycle / "objects").iterdir():
        prefix.chmod(0o700)
    cycle.chmod(0o700)
    return activation_path, cycle


@pytest.fixture
def observed_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, cycle = _bundle(root)
    return root, activation, cycle


def _observe(value):
    root, activation, _cycle = value
    return observe_shadow_forward_definition_candidate(root, CYCLE, activation)


def _rewrite_seal(cycle, manifest):
    replay = {
        key: value for key, value in manifest.items()
        if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
    }
    manifest["request_content_digest"] = TypedDigest.canonical(replay).to_dict()
    manifest_data = canonical_json_bytes(manifest)
    seal = json.loads((cycle / "seal.json").read_bytes())
    seal["manifest_digest"] = TypedDigest.raw(manifest_data).to_dict()
    object_names = seal["object_digests"]
    seal["artifact_root_digest"] = TypedDigest.canonical({
        "objects": object_names, "named_files": seal["named_file_digests"],
    }).to_dict()
    _write(cycle / "manifest.json", manifest_data)
    _write(cycle / "seal.json", canonical_json_bytes(seal))


def test_valid_engineering_activation_observes_exact_sealed_four_to_three_definition_candidate(observed_workspace):
    candidate = _observe(observed_workspace)
    assert candidate.compiled_capability is True
    assert len(candidate.compiled_definitions.champion.to_dict()["source_members"]) == 4
    assert len(candidate.compiled_definitions.challenger.to_dict()["source_members"]) == 3
    assert candidate.publication_capability is False


def test_engineering_and_frozen_observation_entries_enforce_disjoint_fixed_policies(tmp_path):
    engineering = tmp_path / "engineering"
    engineering.mkdir()
    engineering_activation, _ = _bundle(engineering)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    frozen_activation, _ = _bundle(frozen, frozen=True)
    assert observe_shadow_forward_definition_candidate(
        engineering, CYCLE, engineering_activation,
    ).reference_receipt["purpose"] == "ENGINEERING_VALIDATION"
    assert observe_frozen_shadow_forward_definition_candidate(
        frozen, CYCLE, frozen_activation,
    ).reference_receipt["purpose"] == "FROZEN_OBSERVATION"
    with pytest.raises(ForwardObservationInputError):
        observe_frozen_shadow_forward_definition_candidate(
            engineering, CYCLE, engineering_activation,
        )
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(frozen, CYCLE, frozen_activation)


def test_valid_frozen_activation_observes_a_new_formal_candidate_with_all_future_claims_false(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, _ = _bundle(root, frozen=True)
    candidate = observe_frozen_shadow_forward_definition_candidate(
        root, CYCLE, activation,
    )
    summary = candidate.to_safe_summary_dict()
    assert summary["definition_set_id"] == "shadow.frozen.c1b1a"
    assert candidate.reference_receipt["purpose"] == "FROZEN_OBSERVATION"
    assert summary["publication_capability"] is False
    assert summary["epoch_started"] is False
    assert summary["prospective_claim"] is False
    assert summary["promotion_capability"] is False


@pytest.mark.parametrize("mutation", ["reason", "decision", "target", "layout"])
def test_frozen_activation_requires_approve_frozen_reason_exact_target_and_private_layout(tmp_path, mutation):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, _ = _bundle(root, frozen=True)
    raw = json.loads(activation.read_bytes())
    if mutation == "reason":
        raw["selection_decision"]["reason_code"] = "ENGINEERING_ONLY"
    elif mutation == "decision":
        raw["selection_decision"]["decision"] = "DEFER"
    elif mutation == "target":
        raw["selection_decision"]["target"] = raw["champion_strategy_id"]
    else:
        displaced = root / "private" / activation.name
        _write(displaced, activation.read_bytes(), 0o600)
        activation = displaced
    if mutation != "layout":
        _write(activation, canonical_json_bytes(raw), 0o600)
    with pytest.raises(ForwardObservationInputError):
        observe_frozen_shadow_forward_definition_candidate(root, CYCLE, activation)


def test_frozen_candidate_mutation_replay_and_coordinated_replacement_deny_capability(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, _ = _bundle(root, frozen=True)
    candidate = observe_frozen_shadow_forward_definition_candidate(root, CYCLE, activation)
    with pytest.raises(ForwardObservationContractError):
        copy.copy(candidate)
    receipt = dict(candidate.reference_receipt)
    receipt["purpose"] = "ENGINEERING_VALIDATION"
    object.__setattr__(candidate, "_reference_receipt", receipt)
    object.__setattr__(candidate, "_reference_receipt_bytes", canonical_json_bytes(receipt))
    object.__setattr__(
        candidate, "_reference_receipt_digest", TypedDigest.canonical(receipt).to_dict(),
    )
    with pytest.raises(ForwardObservationContractError):
        candidate.to_store_request()


def test_frozen_observation_preserves_all_c1b0_seal_source_config_and_privacy_contracts(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, cycle = _bundle(root, frozen=True)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    manifest["model_and_ensemble_lineage"]["source_artifacts"].append(None)
    _rewrite_seal(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_frozen_shadow_forward_definition_candidate(root, CYCLE, activation)


def test_each_explicit_omitted_position_preserves_the_exact_remaining_source_order(tmp_path):
    for omitted in range(4):
        root = tmp_path / ("workspace-%d" % omitted)
        root.mkdir()
        activation, _ = _bundle(root, omitted=omitted)
        candidate = observe_shadow_forward_definition_candidate(root, CYCLE, activation)
        champion = [row["source_id"] for row in candidate.compiled_definitions.champion.to_dict()["source_members"]]
        challenger = [row["source_id"] for row in candidate.compiled_definitions.challenger.to_dict()["source_members"]]
        assert challenger == [value for index, value in enumerate(champion) if index != omitted]


@pytest.mark.parametrize("mutation", [
    "extra", "mode", "noncanonical", "omitted_bool", "omitted_low",
    "omitted_high", "reason", "decision", "purpose",
])
def test_activation_is_exact_private_canonical_and_cannot_carry_observer_owned_facts(observed_workspace, mutation):
    root, activation, _ = observed_workspace
    raw = json.loads(activation.read_bytes())
    if mutation == "extra":
        raw["sealed_reference_join_verified"] = True
        _write(activation, canonical_json_bytes(raw), 0o600)
    elif mutation == "mode":
        activation.chmod(0o644)
    elif mutation == "noncanonical":
        _write(activation, json.dumps(raw, indent=2).encode(), 0o600)
    else:
        if mutation == "omitted_bool": raw["omitted_position"] = True
        elif mutation == "omitted_low": raw["omitted_position"] = -1
        elif mutation == "omitted_high": raw["omitted_position"] = 4
        elif mutation == "reason": raw["selection_decision"]["reason_code"] = "FROZEN_OBSERVATION"
        elif mutation == "decision": raw["selection_decision"]["decision"] = "DEFER"
        else: raw["purpose"] = "FORWARD_OBSERVATION"
        _write(activation, canonical_json_bytes(raw), 0o600)
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)


@pytest.mark.parametrize("mutation", [
    "artifact", "manifest_digest", "noncanonical", "partial",
    "seal_schema_bool", "seal_schema_null", "seal_schema_string",
    "manifest_schema_bool", "manifest_schema_null", "manifest_schema_string",
])
def test_sealed_cycle_requires_canonical_complete_manifest_seal_and_exact_artifact_root(observed_workspace, mutation):
    root, activation, cycle = observed_workspace
    seal = json.loads((cycle / "seal.json").read_bytes())
    if mutation == "artifact":
        seal["artifact_root_digest"]["value"] = "0" * 64
        _write(cycle / "seal.json", canonical_json_bytes(seal))
    elif mutation == "manifest_digest":
        seal["manifest_digest"]["value"] = "0" * 64
        _write(cycle / "seal.json", canonical_json_bytes(seal))
    elif mutation == "noncanonical":
        _write(cycle / "seal.json", json.dumps(seal, indent=2).encode())
    elif mutation == "partial":
        manifest = json.loads((cycle / "manifest.json").read_bytes())
        manifest["problems"] = [{
            "code": "synthetic", "evidence_class": "test", "detail": "blocked",
            "blocks_complete": True,
        }]
        manifest["status"] = "sealed_partial"
        _rewrite_seal(cycle, manifest)
        seal = json.loads((cycle / "seal.json").read_bytes())
        seal["status"] = "sealed_partial"
        _write(cycle / "seal.json", canonical_json_bytes(seal))
    elif mutation.startswith("seal_schema_"):
        seal["schema_version"] = {
            "seal_schema_bool": True,
            "seal_schema_null": None,
            "seal_schema_string": "1",
        }[mutation]
        _write(cycle / "seal.json", canonical_json_bytes(seal))
    else:
        manifest = json.loads((cycle / "manifest.json").read_bytes())
        manifest["schema_version"] = {
            "manifest_schema_bool": True,
            "manifest_schema_null": None,
            "manifest_schema_string": "1",
        }[mutation]
        _rewrite_seal(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)


@pytest.mark.parametrize("mutation", ["object", "named", "foreign", "hardlink"])
def test_embedded_object_and_named_file_inventory_bytes_are_fully_reverified(observed_workspace, mutation):
    root, activation, cycle = observed_workspace
    if mutation == "object":
        seal = json.loads((cycle / "seal.json").read_bytes())
        digest = seal["object_digests"][0]
        path = cycle / "objects" / digest[:2] / digest
        path.write_bytes(b"X" * len(path.read_bytes()))
    elif mutation == "named":
        path = cycle / "ranking.csv"
        path.write_bytes(b"X" * len(path.read_bytes()))
    elif mutation == "foreign":
        _write(cycle / "foreign.bin", b"foreign", 0o600)
    else:
        seal = json.loads((cycle / "seal.json").read_bytes())
        digest = seal["object_digests"][0]
        os.link(cycle / "objects" / digest[:2] / digest, root / "private" / "foreign-link")
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign"])
def test_source_training_references_are_exactly_one_per_ordered_champion_member(observed_workspace, mutation):
    root, activation, cycle = observed_workspace
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    rows = manifest["model_and_ensemble_lineage"]["source_artifacts"]
    if mutation == "missing":
        rows.pop(0)
    else:
        extra = copy.deepcopy(rows[0])
        if mutation == "foreign": extra["position"] = 4
        rows.append(extra)
    _rewrite_seal(cycle, manifest)
    with pytest.raises(ForwardObservationInputError) as caught:
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)
    assert caught.value.requested_source_count == 4


def _auxiliary_artifact(manifest, position):
    template = copy.deepcopy(
        manifest["model_and_ensemble_lineage"]["source_artifacts"][0],
    )
    template.pop("role")
    template.pop("experiment_name")
    template["position"] = position
    if position == "ensemble":
        template["recorder_id"] = "ENSEMBLE"
    else:
        template["recorder_id"] = manifest["model_and_ensemble_lineage"][
            "source_models"
        ][position]["recorder_id"]
        template["source_recorder_id"] = manifest["model_and_ensemble_lineage"][
            "source_models"
        ][position]["source_recorder_id"]
    return template


def test_legal_prediction_and_ensemble_auxiliary_artifacts_are_fully_partitioned(observed_workspace):
    root, activation, cycle = observed_workspace
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    lineage = manifest["model_and_ensemble_lineage"]
    lineage["combo"]["ensemble_recorder_id"] = "ENSEMBLE"
    lineage["source_artifacts"].extend([
        _auxiliary_artifact(manifest, 0),
        _auxiliary_artifact(manifest, "ensemble"),
    ])
    _rewrite_seal(cycle, manifest)
    assert observe_shadow_forward_definition_candidate(
        root, CYCLE, activation,
    ).sealed_reference_join_verified is True


@pytest.mark.parametrize("mutation", [
    "null", "unknown_role", "foreign_position", "duplicate_prediction",
    "bad_auxiliary_tree", "prediction_source_mismatch",
    "ensemble_recorder_mismatch",
])
def test_source_artifact_unassigned_or_malformed_remainder_denies_verified_capability(
    observed_workspace, mutation,
):
    root, activation, cycle = observed_workspace
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    lineage = manifest["model_and_ensemble_lineage"]
    if mutation == "null":
        lineage["source_artifacts"].append(None)
    else:
        auxiliary_position = "ensemble" if mutation == "ensemble_recorder_mismatch" else 0
        auxiliary = _auxiliary_artifact(manifest, auxiliary_position)
        if mutation == "unknown_role":
            auxiliary["role"] = "prediction"
        elif mutation == "foreign_position":
            auxiliary["position"] = 4
        elif mutation == "duplicate_prediction":
            lineage["source_artifacts"].append(copy.deepcopy(auxiliary))
        elif mutation == "bad_auxiliary_tree":
            auxiliary["artifact_tree_digest"]["value"] = "1" * 64
        elif mutation == "prediction_source_mismatch":
            auxiliary["source_recorder_id"] = "FOREIGN_SOURCE"
        else:
            lineage["combo"]["ensemble_recorder_id"] = "OTHER_ENSEMBLE"
        lineage["source_artifacts"].append(auxiliary)
    _rewrite_seal(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)


def test_artifact_inventory_digest_is_rebuilt_from_exact_sealed_member_rows(observed_workspace):
    root, activation, cycle = observed_workspace
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    manifest["model_and_ensemble_lineage"]["source_artifacts"][0]["artifact_tree_digest"]["value"] = "1" * 64
    _rewrite_seal(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)


@pytest.mark.parametrize("mutation", ["missing", "bool", "duplicate", "unsafe", "multiple"])
def test_strategy_config_is_strict_no_default_and_revalidated_by_b1(observed_workspace, mutation):
    root, activation, _ = observed_workspace
    path = root / "config" / "strategy_config.yaml"
    if mutation == "missing":
        path.write_text("strategy:\n  name: topk_dropout\n  params:\n    topk: 22\n")
    elif mutation == "bool":
        path.write_text("strategy:\n  name: topk_dropout\n  params:\n    topk: true\n    n_drop: 4\n    buy_suggestion_factor: 2\n    sell_out_of_universe: true\n")
    elif mutation == "duplicate":
        path.write_text("strategy:\n  name: topk_dropout\n  name: topk_dropout\n  params: {}\n")
    elif mutation == "unsafe":
        path.write_text("strategy: !!python/object:builtins.object {}\n")
    else:
        path.write_text("strategy: {}\n---\nstrategy: {}\n")
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)


def test_reference_receipt_counts_digests_and_claims_are_observer_owned(observed_workspace):
    candidate = _observe(observed_workspace)
    receipt = candidate.reference_receipt
    assert receipt["source_reference_count"] == 4
    assert receipt["sealed_reference_join_verified"] is True
    assert receipt["live_artifact_member_bytes_reverified"] is False
    assert receipt["publication_capability"] is False


def test_foreign_mutated_or_replayed_candidate_cannot_grant_verified_or_compiled_capability(observed_workspace):
    candidate = _observe(observed_workspace)
    with pytest.raises(ForwardObservationContractError):
        ObservedForwardDefinitionCandidate()
    with pytest.raises(ForwardObservationContractError):
        copy.copy(candidate)
    with pytest.raises(ForwardObservationContractError):
        copy.deepcopy(candidate)
    object.__setattr__(candidate, "_reference_receipt_digest", {"value": "forged"})
    with pytest.raises(ForwardObservationContractError):
        _ = candidate.compiled_capability


@pytest.mark.parametrize("replacement", [
    "compiled", "compiled_nested_equal", "receipt", "receipt_nested_equal",
    "coordinated",
])
def test_candidate_fields_cannot_replace_the_original_inspector_observation_binding(
    observed_workspace, replacement,
):
    from quantpits.research.forward_definitions import compile_shadow_forward_definitions

    candidate = _observe(observed_workspace)
    receipt = dict(candidate.reference_receipt)
    raw = {
        "definition_set_id": candidate.compiled_definitions.definition_set_id,
        "protocol": candidate.compiled_definitions.protocol.to_dict(),
        "execution_assumption": candidate.compiled_definitions.execution_assumption.to_dict(),
        "champion_strategy": candidate.compiled_definitions.champion.to_dict(),
        "challenger_strategy": candidate.compiled_definitions.challenger.to_dict(),
    }
    raw["challenger_strategy"]["hypothesis"] += " Forged."
    forged = compile_shadow_forward_definitions(raw)
    if replacement in {"compiled", "coordinated"}:
        object.__setattr__(candidate, "compiled_definitions", forged)
    elif replacement == "compiled_nested_equal":
        equal = compile_shadow_forward_definitions({
            "definition_set_id": candidate.compiled_definitions.definition_set_id,
            "protocol": candidate.compiled_definitions.protocol.to_dict(),
            "execution_assumption": candidate.compiled_definitions.execution_assumption.to_dict(),
            "champion_strategy": candidate.compiled_definitions.champion.to_dict(),
            "challenger_strategy": candidate.compiled_definitions.challenger.to_dict(),
        })
        object.__setattr__(
            candidate.compiled_definitions, "challenger", equal.challenger,
        )
    if replacement in {"receipt", "coordinated"}:
        if replacement == "coordinated":
            receipt["compiled_request_digest"] = dict(forged.request_digest)
        receipt_bytes = canonical_json_bytes(receipt)
        object.__setattr__(candidate, "_reference_receipt", receipt)
        object.__setattr__(candidate, "_reference_receipt_bytes", receipt_bytes)
        object.__setattr__(
            candidate, "_reference_receipt_digest",
            TypedDigest.canonical(receipt).to_dict(),
        )
    elif replacement == "receipt_nested_equal":
        digest = candidate._reference_receipt["evidence_seal_digest"]
        original_value = digest["value"]
        equal_value = original_value.encode("ascii").decode("ascii")
        assert equal_value == original_value and equal_value is not original_value
        digest["value"] = equal_value
    with pytest.raises(ForwardObservationContractError):
        _ = candidate.sealed_reference_join_verified
    with pytest.raises(ForwardObservationContractError):
        _ = candidate.compiled_capability
    with pytest.raises(ForwardObservationContractError):
        candidate.to_store_request()
    with pytest.raises(ForwardObservationContractError):
        candidate.to_safe_summary_dict()


def test_observation_reads_are_bounded_and_workspace_cycle_config_activation_drift_is_denied(observed_workspace, monkeypatch):
    import quantpits.research.forward_observation as module
    root, activation, _ = observed_workspace
    requested = []
    original = module.os.read

    def bounded(descriptor, size):
        requested.append(size)
        return original(descriptor, size)

    monkeypatch.setattr(module.os, "read", bounded)
    assert _observe(observed_workspace).compiled_capability is True
    assert requested and max(requested) <= 16 * 1024 * 1024 + 1
    assert any(size == len(b"artifact-0") + 1 for size in requested)
    config = root / "config" / "strategy_config.yaml"
    original_strategy = module._strategy_config

    def transient_drift(data, definition_id):
        before = config.stat()
        original_bytes = config.read_bytes()
        config.write_bytes(original_bytes)
        os.utime(config, ns=(before.st_atime_ns, before.st_mtime_ns))
        return original_strategy(data, definition_id)

    monkeypatch.setattr(module, "_strategy_config", transient_drift)
    with pytest.raises(ForwardObservationInputError):
        _observe(observed_workspace)
    activation.write_bytes(b"{" + b"x" * (128 * 1024))
    with pytest.raises(ForwardObservationInputError):
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)


def test_ordinary_member_failure_preserves_requested_identity_diagnostics_and_denies_candidate(observed_workspace, monkeypatch):
    import quantpits.research.forward_observation as module
    root, activation, cycle = observed_workspace
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    member = manifest["model_and_ensemble_lineage"]["source_artifacts"][2]["members"][0]
    digest = member["digest"]["value"]
    original = module._read_regular

    def fail_one(path, **kwargs):
        if path.name == digest:
            raise ForwardObservationInputError("synthetic ordinary read failure")
        return original(path, **kwargs)

    monkeypatch.setattr(module, "_read_regular", fail_one)
    with pytest.raises(ForwardObservationInputError) as caught:
        observe_shadow_forward_definition_candidate(root, CYCLE, activation)
    # The terminal diagnostic retains the four requested source positions.
    assert caught.value.requested_source_positions == (0, 1, 2, 3)


@pytest.mark.parametrize("seam", ["read_close", "close", "bundle", "strategy", "compile"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates_from_each_observation_and_compilation_seam(observed_workspace, monkeypatch, exception, seam):
    import quantpits.research.forward_observation as module
    if seam == "read_close":
        monkeypatch.setattr(module.os, "read", lambda *_args: (_ for _ in ()).throw(exception()))
        monkeypatch.setattr(module.os, "close", lambda *_args: (_ for _ in ()).throw(OSError("secondary close")))
    elif seam == "close":
        original_close = module.os.close
        injected = {"done": False}

        def close(descriptor):
            if not injected["done"]:
                injected["done"] = True
                raise exception()
            return original_close(descriptor)

        monkeypatch.setattr(module.os, "close", close)
    elif seam == "bundle":
        monkeypatch.setattr(module, "_verify_bundle", lambda *_args: (_ for _ in ()).throw(exception()))
    elif seam == "strategy":
        monkeypatch.setattr(module, "_strategy_config", lambda *_args: (_ for _ in ()).throw(exception()))
    else:
        import quantpits.research.forward_definitions as definitions
        monkeypatch.setattr(definitions, "compile_shadow_forward_definitions", lambda *_args: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception):
        _observe(observed_workspace)


def test_definition_store_publish_is_never_called_and_all_filesystem_surfaces_remain_read_only(observed_workspace, monkeypatch):
    from quantpits.research import definition_store
    root, _activation, _ = observed_workspace
    before = sorted((path.relative_to(root).as_posix(), path.stat().st_mtime_ns) for path in root.rglob("*"))
    monkeypatch.setattr(definition_store.CreateOnlyDefinitionBundleStore, "publish", lambda *_args: (_ for _ in ()).throw(AssertionError("publish called")))
    candidate = _observe(observed_workspace)
    after = sorted((path.relative_to(root).as_posix(), path.stat().st_mtime_ns) for path in root.rglob("*"))
    assert candidate.compiled_capability is True
    assert before == after
    assert not (root / "research").exists()


def test_safe_summary_excludes_sources_paths_actor_hypothesis_economics_and_strategy_values(observed_workspace):
    summary = _observe(observed_workspace).to_safe_summary_dict()
    text = json.dumps(summary, sort_keys=True)
    for private in ("MODEL_A", "PRIVATE_OWNER", "leave-one-out", "0.001", str(observed_workspace[0])):
        assert private not in text
    assert "strategy_config_digest" not in summary


def test_forward_observation_import_preserves_environment_cwd_and_optional_dependency_state():
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    __import__("quantpits.research.forward_observation")
    after = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    assert dict(os.environ) == before_env
    assert os.getcwd() == before_cwd
    assert after == before


def _reseal_fresh_cycle(cycle, manifest, extra_objects=()):
    for data in extra_objects:
        digest = TypedDigest.raw(data)
        _write(cycle / "objects" / digest.value[:2] / digest.value, data, 0o600)
        (cycle / "objects" / digest.value[:2]).chmod(0o700)
    embedded = {}
    for row in __import__(
        "quantpits.research.forward_observation", fromlist=["_embedded_rows"],
    )._embedded_rows(manifest):
        digest = row["digest"]
        embedded[digest["value"]] = digest["size_bytes"]
    manifest["preservation"]["embedded_object_count"] = len(embedded)
    replay = {
        key: value for key, value in manifest.items()
        if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
    }
    manifest["request_content_digest"] = TypedDigest.canonical(replay).to_dict()
    manifest_data = canonical_json_bytes(manifest)
    old_seal = json.loads((cycle / "seal.json").read_bytes())
    objects = sorted(embedded)
    seal = {
        **old_seal,
        "status": manifest["status"],
        "manifest_digest": TypedDigest.raw(manifest_data).to_dict(),
        "object_digests": objects,
    }
    seal["artifact_root_digest"] = TypedDigest.canonical({
        "objects": objects,
        "named_files": seal["named_file_digests"],
    }).to_dict()
    _write(cycle / "manifest.json", manifest_data, 0o600)
    _write(cycle / "seal.json", canonical_json_bytes(seal), 0o600)


def _fresh_split_bundle(tmp_path):
    production = tmp_path / "production"
    research = tmp_path / "research"
    production.mkdir()
    research.mkdir()
    production_activation, cycle = _bundle(production, frozen=True)
    activation_raw = json.loads(production_activation.read_bytes())
    activation_raw.update({
        "definition_set_id": "shadow.fresh.segment.v1",
        "protocol_id": "protocol.fresh.segment.v1",
        "intent_definition_id": "intent.fresh.segment.v1",
        "champion_strategy_id": "strategy.champion.fresh.v1",
        "challenger_strategy_id": "strategy.challenger.fresh.v1",
    })
    activation_raw["selection_decision"].update({
        "decision_id": "DECISION.FRESH.001",
        "target": "strategy.challenger.fresh.v1",
    })
    fresh_production_activation = production_activation.with_name(
        "shadow.fresh.segment.v1.json",
    )
    production_activation.rename(fresh_production_activation)
    production_activation = fresh_production_activation
    _write(production_activation, canonical_json_bytes(activation_raw), 0o600)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    lineage = manifest["model_and_ensemble_lineage"]
    members = lineage["combo"]["resolved_members"]
    lineage["combo"].update({"method": "equal", "ensemble_recorder_id": "ENSEMBLE"})
    auxiliaries = []
    for position, model in enumerate(lineage["source_models"]):
        model["operation"] = "predict_only"
        source = lineage["source_artifacts"][position]
        auxiliary = {
            "position": position,
            "recorder_id": model["recorder_id"],
            "source_recorder_id": model["source_recorder_id"],
            "artifact_locator": "mlruns/prediction-%d" % position,
            "members": copy.deepcopy(source["members"]),
            "artifact_tree_digest": copy.deepcopy(source["artifact_tree_digest"]),
        }
        auxiliaries.append(auxiliary)
    ensemble_source = lineage["source_artifacts"][0]
    auxiliaries.append({
        "position": "ensemble", "recorder_id": "ENSEMBLE",
        "artifact_locator": "mlruns/ensemble",
        "members": copy.deepcopy(ensemble_source["members"]),
        "artifact_tree_digest": copy.deepcopy(ensemble_source["artifact_tree_digest"]),
    })
    lineage["source_artifacts"].extend(auxiliaries)
    manifest["data_identity"]["qlib_materialization_identity"].update({
        "status": "observed", "universe_name": "csi300",
    })
    ensemble_run = canonical_json_bytes({
        "schema_version": 1, "command": "ensemble_fusion", "status": "success",
        "run_id": "FRESH", "args": ["--from-config"],
        "records": {"combos": [{
            "name": "CHAMPION_4", "method": "equal", "is_default": True,
            "models": members, "resolved_models": members,
        }]},
    })
    run_digest = TypedDigest.raw(ensemble_run)
    manifest["run_evidence"] = [{
        "class": "ensemble", "path": "run/ensemble.json", "status": "observed",
        "digest": run_digest.to_dict(), "preservation_status": "embedded", "detail": "",
    }]
    _reseal_fresh_cycle(cycle, manifest, (ensemble_run,))
    reference = observe_frozen_shadow_forward_definition_candidate(
        production, CYCLE, production_activation,
    )
    activation = (
        research / "research" / "shadow_v1" / "activations"
        / "shadow.fresh.segment.v1.json"
    )
    _write(activation, production_activation.read_bytes(), 0o600)
    for directory in (
        research / "research", research / "research" / "shadow_v1",
        research / "research" / "shadow_v1" / "activations",
    ):
        directory.chmod(0o700)
    production_activation.unlink()
    (production / "research" / "shadow_v1" / "activations").rmdir()
    (production / "research" / "shadow_v1").rmdir()
    (production / "research").rmdir()
    return production, research, activation, cycle, reference


def test_split_root_observer_builds_one_fresh_candidate_from_production_source_and_research_activation(tmp_path):
    production, research, activation, _cycle, reference = _fresh_split_bundle(tmp_path)
    candidate = observe_fresh_champion_segment_candidate(
        production, research, CYCLE, activation,
    )
    assert candidate.fresh_segment_candidate_ready is True
    assert dict(candidate.compiled_request_digest) == dict(reference.compiled_definitions.request_digest)
    assert candidate.selected_omitted_position == 1
    assert all(getattr(candidate, name) is False for name in (
        "publication_capability", "definition_bound", "intent_capability",
        "epoch_started", "prospective_claim", "promotion_capability",
    ))


def test_research_decoy_cycle_and_config_are_never_used_as_production_authority(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    _write(research / "config" / "strategy_config.yaml", b"private: decoy\n")
    _write(
        research / "data" / "evidence" / "v1" / "cycles" / CYCLE / "seal.json",
        b"decoy", 0o600,
    )
    before = tuple(sorted(path.relative_to(research).as_posix() for path in research.rglob("*")))
    assert observe_fresh_champion_segment_candidate(
        production, research, CYCLE, activation,
    ).fresh_segment_candidate_ready is True
    assert before == tuple(sorted(path.relative_to(research).as_posix() for path in research.rglob("*")))


@pytest.mark.parametrize("allowed", [True, False])
def test_complete_and_deep_analysis_only_partial_cycle_admission_are_exact(tmp_path, allowed):
    production, research, activation, cycle, _reference = _fresh_split_bundle(tmp_path)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    manifest["problems"] = [{
        "code": "deep_analysis_missing" if allowed else "ranking_unavailable",
        "evidence_class": "deep_analysis" if allowed else "ranking",
        "detail": "scoped", "blocks_complete": True,
    }]
    manifest["status"] = "sealed_partial"
    _reseal_fresh_cycle(cycle, manifest)
    if allowed:
        assert observe_fresh_champion_segment_candidate(
            production, research, CYCLE, activation,
        ).fresh_segment_candidate_ready is True
    else:
        with pytest.raises(ForwardObservationInputError):
            observe_fresh_champion_segment_candidate(
                production, research, CYCLE, activation,
            )


@pytest.mark.parametrize("mutation", ["method", "normalization", "universe", "prediction"])
def test_fresh_source_fusion_market_and_prediction_relations_are_exact(tmp_path, mutation):
    production, research, activation, cycle, _reference = _fresh_split_bundle(tmp_path)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    if mutation == "method":
        manifest["model_and_ensemble_lineage"]["combo"]["method"] = "weighted"
    elif mutation == "universe":
        manifest["data_identity"]["qlib_materialization_identity"].pop("universe_name")
    elif mutation == "prediction":
        rows = manifest["model_and_ensemble_lineage"]["source_artifacts"]
        rows.pop(next(index for index, row in enumerate(rows) if row.get("position") == 2 and "role" not in row))
    else:
        observation = manifest["run_evidence"][0]
        old = cycle / "objects" / observation["digest"]["value"][:2] / observation["digest"]["value"]
        raw = json.loads(old.read_bytes())
        raw["args"] = ["--from-config", "--norm-method", "zscore"]
        data = canonical_json_bytes(raw)
        observation["digest"] = TypedDigest.raw(data).to_dict()
        _reseal_fresh_cycle(cycle, manifest, (data,))
    if mutation != "normalization":
        _reseal_fresh_cycle(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)


def test_same_or_nested_roots_are_rejected_without_effects(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    before = tuple(sorted(path.relative_to(production).as_posix() for path in production.rglob("*")))
    with pytest.raises(ForwardObservationInputError):
        observe_fresh_champion_segment_candidate(production, production, CYCLE, activation)
    nested = production / "nested"
    nested.mkdir()
    with pytest.raises(ForwardObservationInputError):
        observe_fresh_champion_segment_candidate(production, nested, CYCLE, activation)
    assert before + ("nested",) == tuple(sorted(path.relative_to(production).as_posix() for path in production.rglob("*")))


def test_fresh_candidate_replay_or_mutation_cannot_grant_ready_capability(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    candidate = observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)
    with pytest.raises(ForwardObservationContractError):
        FreshChampionSegmentCandidate()
    with pytest.raises(ForwardObservationContractError):
        copy.copy(candidate)
    object.__setattr__(candidate, "_observed_candidate", object())
    with pytest.raises(ForwardObservationContractError):
        _ = candidate.fresh_segment_candidate_ready


def test_fresh_safe_summary_is_private_and_fixed_false(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    summary = observe_fresh_champion_segment_candidate(
        production, research, CYCLE, activation,
    ).to_safe_summary_dict()
    text = json.dumps(summary, sort_keys=True)
    assert summary["status"] == "READY"
    for private in ("MODEL_A", "PRIVATE_OWNER", str(production), str(research), "0.001"):
        assert private not in text


@pytest.mark.parametrize("source", ["production_cycle", "production_config", "research_activation"])
def test_production_cycle_config_and_research_activation_mutation_fail_closed(
    tmp_path, monkeypatch, source,
):
    import quantpits.research.forward_observation as module
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    if source in {"production_cycle", "production_config"}:
        path = (
            production / "data" / "evidence" / "v1" / "cycles" / CYCLE / "seal.json"
            if source == "production_cycle"
            else production / "config" / "strategy_config.yaml"
        )
        original = module._strategy_config

        def mutate(data, definition_id):
            before = path.stat()
            path.write_bytes(path.read_bytes())
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            return original(data, definition_id)

        monkeypatch.setattr(module, "_strategy_config", mutate)
    else:
        original = module._verify_bundle

        def mutate(root, cycle_id, **kwargs):
            before = activation.stat()
            activation.write_bytes(activation.read_bytes())
            os.utime(activation, ns=(before.st_atime_ns, before.st_mtime_ns))
            return original(root, cycle_id, **kwargs)

        monkeypatch.setattr(module, "_verify_bundle", mutate)
    with pytest.raises(ForwardObservationInputError):
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fresh_observer_process_control_propagates(tmp_path, monkeypatch, exception):
    import quantpits.research.forward_observation as module
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    monkeypatch.setattr(
        module, "_strategy_config",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)


def test_split_root_api_keeps_both_workspace_inventories_and_metadata_unchanged(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)

    def snapshot(root):
        return tuple(sorted(
            (
                path.relative_to(root).as_posix(), path.stat().st_mode,
                path.stat().st_size, path.stat().st_mtime_ns,
            )
            for path in root.rglob("*")
        ))

    before = snapshot(production), snapshot(research)
    assert observe_fresh_champion_segment_candidate(
        production, research, CYCLE, activation,
    ).fresh_segment_candidate_ready is True
    assert before == (snapshot(production), snapshot(research))


def test_current_production_strategy_fields_build_the_existing_intent_contract(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    candidate = observe_fresh_champion_segment_candidate(
        production, research, CYCLE, activation,
    )
    intent = candidate.compiled_definitions.protocol.to_dict()["intent_definition"]
    assert {
        key: intent[key]
        for key in ("strategy_name", "topk", "n_drop", "buy_suggestion_factor", "sell_out_of_universe")
    } == {
        "strategy_name": "topk_dropout", "topk": 22, "n_drop": 4,
        "buy_suggestion_factor": 2, "sell_out_of_universe": True,
    }


@pytest.mark.parametrize("mutation", ["chronology", "target", "layout"])
def test_frozen_activation_requires_exact_internal_ids_decision_chronology_and_formal_layout(
    tmp_path, mutation,
):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    raw = json.loads(activation.read_bytes())
    if mutation == "chronology":
        raw["selection_decision"]["decision_time"] = "2026-08-28T13:00:00Z"
        _write(activation, canonical_json_bytes(raw), 0o600)
    elif mutation == "target":
        raw["selection_decision"]["target"] = raw["champion_strategy_id"]
        _write(activation, canonical_json_bytes(raw), 0o600)
    else:
        displaced = research / "private" / activation.name
        _write(displaced, activation.read_bytes(), 0o600)
        activation = displaced
    with pytest.raises((ForwardObservationInputError, ForwardObservationContractError)):
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)


def test_ordinary_failure_denies_candidate_and_process_control_propagates(tmp_path, monkeypatch):
    import quantpits.research.forward_observation as module
    production, research, activation, cycle, _reference = _fresh_split_bundle(tmp_path)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    digest = manifest["model_and_ensemble_lineage"]["source_artifacts"][2]["members"][0]["digest"]["value"]
    original = module._read_regular

    def fail_one(path, **kwargs):
        if path.name == digest:
            raise ForwardObservationInputError("synthetic ordinary failure")
        return original(path, **kwargs)

    monkeypatch.setattr(module, "_read_regular", fail_one)
    with pytest.raises(ForwardObservationInputError) as caught:
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)
    assert caught.value.requested_source_positions == (0, 1, 2, 3)


def test_existing_single_root_observer_public_behavior_is_unchanged(tmp_path):
    root = tmp_path / "single"
    root.mkdir()
    activation, _cycle = _bundle(root, frozen=True)
    candidate = observe_frozen_shadow_forward_definition_candidate(root, CYCLE, activation)
    assert candidate.to_safe_summary_dict()["status"] == "complete"
    assert "fresh_segment_candidate_ready" not in candidate.to_safe_summary_dict()
    assert not hasattr(candidate, "definition_bound")


def test_any_other_blocking_problem_denies_fresh_segment_candidate(tmp_path):
    production, research, activation, cycle, _reference = _fresh_split_bundle(tmp_path)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    manifest["problems"] = [{
        "code": "portfolio_incomplete", "evidence_class": "portfolio",
        "detail": "not admitted", "blocks_complete": True,
    }]
    manifest["status"] = "sealed_partial"
    _reseal_fresh_cycle(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)


@pytest.mark.parametrize("relation", ["training", "prediction", "ensemble"])
def test_exact_four_training_prediction_and_ensemble_relations_are_required(tmp_path, relation):
    production, research, activation, cycle, _reference = _fresh_split_bundle(tmp_path)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    rows = manifest["model_and_ensemble_lineage"]["source_artifacts"]
    if relation == "training":
        rows.pop(next(index for index, row in enumerate(rows) if row.get("role") == "source_training"))
    elif relation == "prediction":
        rows.pop(next(index for index, row in enumerate(rows) if row.get("position") == 1 and "role" not in row))
    else:
        rows.pop(next(index for index, row in enumerate(rows) if row.get("position") == "ensemble"))
    _reseal_fresh_cycle(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)


@pytest.mark.parametrize("fact", ["method", "normalization", "universe", "cutoff"])
def test_current_equal_rank_fusion_and_universe_cutoff_are_joined(tmp_path, fact):
    production, research, activation, cycle, _reference = _fresh_split_bundle(tmp_path)
    manifest = json.loads((cycle / "manifest.json").read_bytes())
    if fact == "method":
        manifest["model_and_ensemble_lineage"]["combo"]["method"] = "weighted"
    elif fact == "universe":
        manifest["data_identity"]["qlib_materialization_identity"]["universe_name"] = ""
    elif fact == "cutoff":
        manifest["data_identity"]["qlib_materialization_identity"]["calendar_cutoff"] = "2026-08-13"
    else:
        observation = manifest["run_evidence"][0]
        old = cycle / "objects" / observation["digest"]["value"][:2] / observation["digest"]["value"]
        raw = json.loads(old.read_bytes())
        raw["args"] = ["--from-config", "--norm-method", "zscore"]
        data = canonical_json_bytes(raw)
        observation["digest"] = TypedDigest.raw(data).to_dict()
        _reseal_fresh_cycle(cycle, manifest, (data,))
    if fact != "normalization":
        _reseal_fresh_cycle(cycle, manifest)
    with pytest.raises(ForwardObservationInputError):
        observe_fresh_champion_segment_candidate(production, research, CYCLE, activation)
