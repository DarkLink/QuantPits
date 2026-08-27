import copy
import dataclasses
import json
import os
import sys

import pytest

from quantpits.research.definition_store import DEFINITION_PATHS
from quantpits.research.forward_definitions import (
    CompiledShadowForwardDefinitions,
    DigestReference,
    ForwardDefinitionContractError,
    ShadowForwardProtocol,
    compile_shadow_forward_definitions,
)


def _digest(seed):
    return {
        "algorithm": "sha256", "domain": "raw_bytes", "value": seed * 64,
        "size_bytes": 10,
    }


def _intent():
    return {
        "schema_version": 1,
        "definition_id": "intent.current.v1",
        "strategy_name": "topk_dropout",
        "topk": 22,
        "n_drop": 4,
        "buy_suggestion_factor": 2,
        "sell_out_of_universe": True,
        "production_buy_lot_size": 100,
        "cashflow_mode": "ZERO_EXTERNAL_CASHFLOW",
        "planning_price_rule": "CASH_CLOSE_WITH_0_9_1_1_ESTIMATES_V1",
        "buy_selection_rule": "FIRST_N_GENERATED_SUGGESTIONS_V1",
        "production_primitive_id": "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1",
    }


def _assumption():
    return {
        "schema_version": 1,
        "assumption_id": "execution.synthetic.v1",
        "deal_price": "NEXT_OPEN",
        "sell_before_buy": True,
        "partial_fill": False,
        "short_sell": False,
        "lot_size": 100,
        "money_quantum": "0.01",
        "price_quantum": "0.0001",
        "rounding_mode": "HALF_UP",
        "buy_slippage_rate": "0.001",
        "sell_slippage_rate": "0.002",
        "buy_fee_rate": "0.0003",
        "sell_fee_rate": "0.0004",
        "minimum_buy_fee": "1",
        "minimum_sell_fee": "2",
        "corporate_action_mode": "UNMODELED",
        "definition_kind": "SHADOW_FORWARD_EXECUTION_ASSUMPTION_V1",
        "definition_claim": "TYPED_OWNER_DECLARATION_V1",
        "created_at": "2026-08-26T12:00:00Z",
        "effective_cycle": "2026-09-04",
        "external_references_verified": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
    }


def _strategy(role, member_indices):
    champion = role == "CHAMPION"
    return {
        "schema_version": 1,
        "definition_kind": "SHADOW_FORWARD_STRATEGY_V1",
        "definition_claim": "TYPED_OWNER_DECLARATION_V1",
        "strategy_id": "strategy.champion.v1" if champion else "strategy.challenger.v1",
        "role": role,
        "parent_strategy_id": None if champion else "strategy.champion.v1",
        "mutation_level": "CHAMPION_BASELINE" if champion else "L2_ENSEMBLE_COMPOSITION",
        "created_at": "2026-08-26T12:00:00Z",
        "effective_cycle": "2026-09-04",
        "data_cutoff": "2026-08-13",
        "evidence_cycle_id": "2026-08-14",
        "evidence_seal_digest": _digest("e"),
        "intent_definition_id": "intent.current.v1",
        "selection_decision_id": None if champion else "DECISION.C1A.001",
        "fusion_definition": "PER_MODEL_CROSS_SECTIONAL_PERCENTILE_RANK_EQUAL_MEAN_V1",
        "ranking_rule": (
            "SEALED_PRODUCTION_FULL_RANKING_V1" if champion
            else "FROZEN_MEMBER_EQUAL_FUSION_V1"
        ),
        "source_members": [
            {
                "position": position,
                "source_id": "MODEL_%s" % chr(ord("A") + source_index),
                "model_artifact_digest": _digest(str(source_index + 1)),
            }
            for position, source_index in enumerate(member_indices)
        ],
        "hypothesis": None if champion else "Synthetic leave-one-out behavior observation.",
        "external_references_verified": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
    }


def valid_raw():
    return {
        "definition_set_id": "shadow.synthetic.c1a",
        "protocol": {
            "schema_version": 1,
            "definition_kind": "SHADOW_FORWARD_PROTOCOL_V1",
            "definition_claim": "TYPED_OWNER_DECLARATION_V1",
            "protocol_id": "protocol.shadow.v1",
            "definition_set_id": "shadow.synthetic.c1a",
            "research_epoch_id": "PROSPECTIVE_SHADOW_V1",
            "created_at": "2026-08-26T12:00:00Z",
            "effective_cycle": "2026-09-04",
            "champion_strategy_id": "strategy.champion.v1",
            "challenger_strategy_id": "strategy.challenger.v1",
            "execution_assumption_id": "execution.synthetic.v1",
            "selection_decision": {
                "decision_id": "DECISION.C1A.001",
                "decision_time": "2026-08-26T11:00:00Z",
                "evidence_cycle_id": "2026-08-14",
                "actor": "SYNTHETIC_OWNER",
                "decision": "APPROVE",
                "target": "strategy.challenger.v1",
                "reason_code": "FROZEN_OBSERVATION",
                "optional_note": None,
            },
            "intent_definition": _intent(),
            "bootstrap_rule": "MATCHED_CHAMPION_SNAPSHOT_ONCE_V1",
            "champion_ranking_rule": "SEALED_PRODUCTION_FULL_RANKING_V1",
            "challenger_ranking_rule": "FROZEN_MEMBER_EQUAL_FUSION_V1",
            "settlement_rule": "NEXT_OPEN_SELL_BEFORE_BUY_V1",
            "promotion_rule": "EXPLICIT_OWNER_DECISION_ONLY_V1",
            "external_references_verified": False,
            "epoch_started": False,
            "prospective_claim": False,
            "promotion_capability": False,
        },
        "execution_assumption": _assumption(),
        "champion_strategy": _strategy("CHAMPION", (0, 1, 2, 3)),
        "challenger_strategy": _strategy("CHALLENGER", (0, 2, 3)),
    }


def _denied(raw):
    with pytest.raises(ForwardDefinitionContractError):
        compile_shadow_forward_definitions(raw)


def test_valid_typed_definition_set_compiles_exact_c0_members_and_fixed_false_claims():
    compiled = compile_shadow_forward_definitions(valid_raw())
    assert compiled.compiled_capability is True
    assert tuple(path for path, _ in compiled.canonical_members) == DEFINITION_PATHS
    request = compiled.to_store_request()
    # The C0 import-isolation contract deliberately reloads its module.  Resolve
    # the authoritative class after that reload instead of retaining a stale
    # collection-time class identity in this downstream test module.
    from quantpits.research import definition_store
    assert type(request) is definition_store.DefinitionBundleRequest
    for _, data in compiled.canonical_members:
        payload = json.loads(data)
        assert payload["external_references_verified"] is False
        assert payload["epoch_started"] is False
        assert payload["prospective_claim"] is False
        assert payload["promotion_capability"] is False


def test_same_semantics_compile_to_same_canonical_bytes_and_request_digest():
    left = compile_shadow_forward_definitions(valid_raw())
    right = compile_shadow_forward_definitions(copy.deepcopy(valid_raw()))
    assert left.canonical_members == right.canonical_members
    assert dict(left.request_digest) == dict(right.request_digest)


@pytest.mark.parametrize("mutation", ["none", "bool", "list", "missing", "extra", "leaf-list"])
def test_top_level_and_leaf_closest_invalid_representations_are_denied_without_effects(mutation):
    raw = valid_raw()
    if mutation == "none": raw = None
    elif mutation == "bool": raw = False
    elif mutation == "list": raw = []
    elif mutation == "missing": del raw["protocol"]
    elif mutation == "extra": raw["compiled"] = True
    else: raw["protocol"] = []
    cwd, environment = os.getcwd(), dict(os.environ)
    _denied(raw)
    assert os.getcwd() == cwd
    assert dict(os.environ) == environment


def test_top_level_mapping_subclasses_and_falsey_mapping_are_not_exact_inputs():
    class ForeignMapping(dict):
        pass

    _denied({})
    _denied(ForeignMapping(valid_raw()))
    raw = valid_raw()
    raw["protocol"] = ForeignMapping(raw["protocol"])
    _denied(raw)


@pytest.mark.parametrize("path,value", [
    (("definition_set_id",), "UPPER"),
    (("protocol", "created_at"), "2026-08-26T12:00:00.0Z"),
    (("champion_strategy", "effective_cycle"), "2026-02-30"),
    (("champion_strategy", "evidence_seal_digest", "domain"), "canonical_json"),
    (("champion_strategy", "evidence_seal_digest", "size_bytes"), True),
    (("champion_strategy", "evidence_seal_digest", "value"), "A" * 64),
])
def test_common_ids_timestamps_cycles_cutoff_and_digest_references_are_strict(path, value):
    raw = valid_raw()
    target = raw
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    _denied(raw)


def test_invalid_public_digest_leaf_cannot_enter_an_aggregate():
    with pytest.raises(ForwardDefinitionContractError):
        DigestReference("md5", "raw_bytes", "0" * 64, 0)


def test_execution_assumption_is_revalidated_by_b0_and_matches_intent_lot_size(monkeypatch):
    raw = valid_raw()
    raw["execution_assumption"]["deal_price"] = "CLOSE"
    _denied(raw)
    raw = valid_raw()
    raw["execution_assumption"]["lot_size"] = 200
    _denied(raw)


def test_intent_definition_is_revalidated_by_b1_without_trusting_carried_digest():
    raw = valid_raw()
    raw["protocol"]["intent_definition"]["digest"] = "0" * 64
    _denied(raw)
    raw = valid_raw()
    raw["protocol"]["intent_definition"]["strategy_name"] = "foreign"
    _denied(raw)


@pytest.mark.parametrize("field,value", [
    ("decision", "DEFER"), ("target", "strategy.champion.v1"),
    ("evidence_cycle_id", "2026-08-07"), ("decision_time", "2026-08-27T00:00:00Z"),
])
def test_selection_decision_must_approve_the_exact_challenger_from_same_evidence_cycle(field, value):
    raw = valid_raw()
    raw["protocol"]["selection_decision"][field] = value
    _denied(raw)


@pytest.mark.parametrize("section,field,value", [
    ("protocol", "champion_strategy_id", "strategy.foreign.v1"),
    ("champion_strategy", "role", "CHALLENGER"),
    ("challenger_strategy", "parent_strategy_id", "strategy.foreign.v1"),
    ("challenger_strategy", "mutation_level", "CHAMPION_BASELINE"),
    ("challenger_strategy", "ranking_rule", "SEALED_PRODUCTION_FULL_RANKING_V1"),
])
def test_protocol_links_roles_parent_mutation_and_ranking_rules_are_exact(section, field, value):
    raw = valid_raw()
    raw[section][field] = value
    _denied(raw)


@pytest.mark.parametrize("members", [(0, 1), (0, 1, 1), (2, 0, 3), (0, 1, 2, 3)])
def test_challenger_members_are_an_order_preserving_three_of_exact_four_champion_members(members):
    raw = valid_raw()
    raw["challenger_strategy"]["source_members"] = _strategy("CHALLENGER", members)["source_members"]
    _denied(raw)

    raw = valid_raw()
    raw["challenger_strategy"]["source_members"][1]["position"] = 7
    _denied(raw)


@pytest.mark.parametrize("field,value", [
    ("created_at", "2026-08-26T12:00:01Z"),
    ("effective_cycle", "2026-09-11"),
    ("data_cutoff", "2026-08-20"),
    ("evidence_cycle_id", "2026-08-21"),
    ("intent_definition_id", "intent.foreign.v1"),
])
def test_cross_definition_evidence_cycle_cutoff_and_semantics_must_match(field, value):
    raw = valid_raw()
    raw["challenger_strategy"][field] = value
    _denied(raw)


def test_mutated_foreign_or_publicly_constructed_aggregate_cannot_grant_compiled_capability():
    class ForeignProtocol(ShadowForwardProtocol):
        pass

    with pytest.raises(ForwardDefinitionContractError):
        ShadowForwardProtocol(payload={})
    with pytest.raises(ForwardDefinitionContractError):
        ForeignProtocol(payload={})
    with pytest.raises(ForwardDefinitionContractError):
        CompiledShadowForwardDefinitions()
    compiled = compile_shadow_forward_definitions(valid_raw())
    with pytest.raises(TypeError):
        dataclasses.replace(compiled)
    with pytest.raises(AttributeError):
        object.__setattr__(compiled, "prospective_claim", True)
    object.__setattr__(compiled, "protocol", object())
    with pytest.raises(ForwardDefinitionContractError):
        compiled.to_store_request()
    compiled = compile_shadow_forward_definitions(valid_raw())
    object.__setattr__(compiled.challenger, "_payload", {"strategy_id": "foreign"})
    with pytest.raises(ForwardDefinitionContractError):
        _ = compiled.compiled_capability
    with pytest.raises(ForwardDefinitionContractError):
        compiled.to_store_request()


def test_to_store_request_recompiles_current_payload_and_never_calls_store_publish(monkeypatch):
    from quantpits.research import definition_store
    monkeypatch.setattr(
        definition_store.CreateOnlyDefinitionBundleStore, "publish",
        lambda *args, **kwargs: pytest.fail("publish must not be called"),
    )
    compiled = compile_shadow_forward_definitions(valid_raw())
    assert dict(compiled.to_store_request().request_digest) == dict(compiled.request_digest)
    object.__setattr__(compiled, "definition_set_id", "shadow.changed.c1a")
    with pytest.raises(ForwardDefinitionContractError):
        compiled.to_store_request()


@pytest.mark.parametrize("seam", ["decision", "b0", "b1", "final"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates_from_decision_b0_b1_and_final_compilation(monkeypatch, seam, exception):
    if seam == "decision":
        from quantpits.evidence import contracts
        monkeypatch.setattr(contracts.DecisionEvent, "from_mapping", lambda value: (_ for _ in ()).throw(exception()))
    elif seam == "b0":
        from quantpits.research import accounting
        monkeypatch.setattr(accounting.ExecutionAssumption, "from_dict", lambda value: (_ for _ in ()).throw(exception()))
    elif seam == "b1":
        from quantpits.research import intents
        monkeypatch.setattr(intents.CurrentRuleIntentDefinition, "from_dict", lambda value: (_ for _ in ()).throw(exception()))
    else:
        from quantpits.research import definition_store
        monkeypatch.setattr(definition_store.DefinitionBundleRequest, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception):
        compile_shadow_forward_definitions(valid_raw())


def test_safe_summary_excludes_raw_sources_hypothesis_actor_and_economic_parameters():
    raw = valid_raw()
    summary = compile_shadow_forward_definitions(raw).to_summary_dict()
    rendered = json.dumps(summary, sort_keys=True)
    for secret in ("MODEL_A", "Synthetic leave-one-out", "SYNTHETIC_OWNER", "0.001", "minimum_buy_fee"):
        assert secret not in rendered
    assert summary["compiled_capability"] is True


def test_lazy_import_preserves_environment_cwd_workspace_and_backend_state():
    cwd, environment = os.getcwd(), dict(os.environ)
    sys.modules.pop("quantpits.research.forward_definitions", None)
    import quantpits.research as research
    assert "quantpits.research.forward_definitions" not in sys.modules
    assert os.getcwd() == cwd
    assert dict(os.environ) == environment
    assert research.__name__ == "quantpits.research"
