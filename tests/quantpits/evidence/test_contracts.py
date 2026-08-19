from dataclasses import replace

import pytest

from quantpits.evidence.contracts import (
    CaptureRequest, CaptureResult, ContractError, DecisionEvent, TypedDigest,
)
from quantpits.evidence.sealing import _result


def request(**changes):
    values = dict(
        cycle_id="2099-01-02", research_epoch_id="SYNTHETIC_V1",
        post_trade_manifest="m1.json", prediction_manifest="m2.json",
        ensemble_manifest="m3.json", order_manifest="m4.json",
    )
    values.update(changes)
    return CaptureRequest(**values)


@pytest.mark.parametrize("value", [None, "", "lowercase", "bad/id", False, 1])
def test_strict_research_epoch_rejects_closest_invalid_representations(value):
    with pytest.raises(ContractError):
        request(research_epoch_id=value)


def test_typed_digest_domains_are_not_interchangeable():
    raw = TypedDigest.raw(b"{}\n")
    semantic = TypedDigest.canonical({}, "semantic_config")
    assert raw.value == semantic.value
    assert raw != semantic
    with pytest.raises(ContractError):
        replace(raw, domain="fingerprint")


@pytest.mark.parametrize("decision", ["NO_ACTION", "APPROVE", "REJECT", "DEFER"])
@pytest.mark.parametrize("reason", [
    "FROZEN_OBSERVATION", "INSUFFICIENT_EVIDENCE", "RISK_BOUNDARY",
    "ENGINEERING_ONLY", "OWNER_OVERRIDE",
])
def test_all_decision_and_reason_variants_are_canonical(decision, reason):
    event = DecisionEvent.from_mapping({
        "decision_id": "DECISION_1", "decision_time": "2099-01-02T00:00:00Z",
        "evidence_cycle_id": "2099-01-02", "actor": "synthetic-owner",
        "decision": decision, "target": "synthetic-target", "reason_code": reason,
    })
    assert event.decision == decision
    assert event.reason_code == reason


def test_malformed_decision_cannot_manufacture_recorded_authority():
    with pytest.raises(ContractError):
        DecisionEvent.from_mapping({"decision_id": "DECISION_1"})


def test_aggregate_result_rejects_impossible_capability_combinations():
    with pytest.raises(ContractError):
        CaptureResult("2099-01-02", "adopted", True, "bundle", TypedDigest.raw(b"x"))
    with pytest.raises(ContractError):
        CaptureResult("2099-01-02", "conflict", False, None, TypedDigest.raw(b"x"))


def test_inspector_result_rejects_cross_field_status_and_problem_forgery():
    blocking = ({
        "code": "missing", "evidence_class": "synthetic",
        "detail": "missing", "blocks_complete": True,
    },)
    with pytest.raises(ContractError, match="complete"):
        _result(
            "2099-01-02", "sealed_complete", True, "bundle",
            TypedDigest.raw(b"seal"), blocking,
        )
    with pytest.raises(ContractError, match="partial"):
        _result(
            "2099-01-02", "sealed_partial", True, "bundle",
            TypedDigest.raw(b"seal"), (),
        )
    with pytest.raises(ContractError, match="write"):
        _result("2099-01-02", "uncertain", False, None, None)
