from dataclasses import FrozenInstanceError, replace

import pytest

from quantpits.rolling import (
    RollingArtifactExpectation,
    RollingArtifactObservation,
    RollingEvidenceContractError,
    RollingEvidenceSetInspection,
    RollingPredictionCoverage,
    RollingRecoveryProposal,
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingUnitEvidenceInspection,
    RollingUnitEvidenceRequest,
    RollingWindowIdentity,
    classify_rolling_recovery,
    workspace_fingerprint,
)
from quantpits.rolling.evidence import RollingOrphanObservation
from quantpits.rolling.evidence import _unit_evidence_fingerprint
from quantpits.utils.workspace import fingerprint_value


def requests_for(root, protocols=("execution_bound_v1", "execution_bound_v1")):
    window = RollingWindowIdentity(
        "rolling", "2025-01-01", "2025-10-31", "2026-01-05", "2026-01-06",
        "a" * 64, "2025-11-01", "2025-12-31",
    )
    targets = tuple(RollingTargetIdentity(name, "rolling").target_key for name in ("alpha", "beta"))
    run = RollingRunIdentity(
        workspace_fingerprint(root), "rolling", "merge", "b" * 64, "c" * 64,
        "2026-01-06", targets, (window.window_key,), "d" * 64, "attempt-1",
    )
    expectation = RollingArtifactExpectation("pred.pkl", "prediction", 1, "e" * 64)
    return tuple(
        RollingUnitEvidenceRequest(
            run, target, window, protocol, target, "merge", "exp", "exp-1",
            "rec-%s" % index, (expectation,), ("2026-01-05", "2026-01-06"),
        )
        for index, (target, protocol) in enumerate(zip(targets, protocols), 1)
    )


def result_for(request, classification):
    if classification == "valid":
        coverage = RollingPredictionCoverage(
            request.expected_prediction_sessions, request.expected_prediction_sessions,
            "f" * 64, 2, "score",
        )
        observations = (RollingArtifactObservation("pred.pkl", "prediction", "valid", 1, "e" * 64),)
        checked = (
            "candidate_cardinality", "source_identity", "artifact_root_containment",
            "artifact_node_kind", "artifact_byte_fingerprint", "prediction_schema",
            "prediction_index_unique", "prediction_scores_finite",
            "prediction_session_coverage", "artifact_public_path_recheck",
        )
        summary = (
            ("backend_fingerprint", "9" * 64),
            ("experiment_name", request.experiment_name),
            ("experiment_id", request.experiment_id),
            ("recorder_id", request.recorder_id),
            ("target_key", request.target_key),
            ("window_key", request.window_key),
            ("source_protocol", request.source_protocol),
            ("source_publication_key", request.source_publication_key),
            ("source_operation", request.source_operation),
            ("source_manifest_fingerprint", request.source_manifest_fingerprint),
        )
        evidence_fingerprint = _unit_evidence_fingerprint(
            request.source_manifest_fingerprint, summary, observations, coverage, checked,
        )
        return RollingUnitEvidenceInspection(
            request.unit_key, "valid", "rolling_evidence_valid", request.source_protocol,
            request.source_manifest_fingerprint, checked, summary, observations,
            coverage, evidence_fingerprint,
        )
    coverage = None
    if classification == "coverage_short":
        coverage = RollingPredictionCoverage(
            request.expected_prediction_sessions,
            request.expected_prediction_sessions[:-1],
            "f" * 64, 1, "score",
        )
    return RollingUnitEvidenceInspection(
        request.unit_key, classification, "rolling_evidence_%s" % classification,
        request.source_protocol, request.source_manifest_fingerprint,
        ("classification",), prediction_coverage=coverage,
        blockers=(classification,),
    )


def evidence_for(requests, classes, *, drift=False, orphan=False):
    results = tuple(result_for(request, classification) for request, classification in zip(requests, classes))
    before = fingerprint_value("before")
    after = fingerprint_value("after") if drift else before
    if drift:
        results = tuple(result_for(request, "drifted") for request in requests)
    valid = sum(item.classification == "valid" for item in results)
    status = "observation_drifted" if drift else "all_valid" if valid == len(results) else "none_valid" if not valid else "incomplete"
    orphans = (RollingOrphanObservation(("orphan@rolling", "rolling:2026-01-05:2026-01-06:123456789abc")),) if orphan else ()
    return RollingEvidenceSetInspection(
        tuple(item.unit_key for item in requests), results, orphans, before, after,
        status, "rolling_evidence_set_%s" % ("drifted" if drift else status),
    )


def test_all_valid_units_form_proposal_only_complete_reuse(tmp_path):
    requests = requests_for(tmp_path.resolve())
    proposal = classify_rolling_recovery(requests, evidence_for(requests, ("valid", "valid"), orphan=True))
    assert proposal.status == "all_reusable"
    assert proposal.reusable_unit_keys == tuple(item.unit_key for item in requests)
    assert proposal.unresolved_unit_keys == ()
    assert proposal.recovery_complete is True
    assert proposal.orphan_unit_keys == (("orphan@rolling", "rolling:2026-01-05:2026-01-06:123456789abc"),)


def test_mixed_units_preserve_valid_reuse_and_ordered_unresolved_members(tmp_path):
    requests = requests_for(tmp_path.resolve())
    evidence = evidence_for(requests, ("valid", "coverage_short"))
    proposal = classify_rolling_recovery(requests, evidence)
    assert proposal.status == "incomplete"
    assert proposal.reusable_unit_keys == (requests[0].unit_key,)
    assert proposal.unresolved_unit_keys == (requests[1].unit_key,)
    assert proposal.recovery_complete is False


def test_no_valid_or_legacy_only_evidence_grants_no_reuse(tmp_path):
    requests = requests_for(tmp_path.resolve(), protocols=("legacy_unverified", "legacy_unverified"))
    proposal = classify_rolling_recovery(requests, evidence_for(requests, ("legacy_unverified", "legacy_unverified")))
    assert proposal.status == "no_reusable_evidence"
    assert proposal.reusable_unit_keys == ()
    assert proposal.unresolved_unit_keys == tuple(item.unit_key for item in requests)


def test_drift_or_aggregate_ambiguity_blocks_all_reuse(tmp_path):
    requests = requests_for(tmp_path.resolve())
    proposal = classify_rolling_recovery(requests, evidence_for(requests, ("valid", "valid"), drift=True))
    assert proposal.status == "blocked"
    assert proposal.reusable_unit_keys == ()
    assert proposal.capabilities == ("render", "proposal_only")
    with pytest.raises(RollingEvidenceContractError):
        replace(proposal, reusable_unit_keys=(requests[0].unit_key,))


def test_recovery_rejects_request_result_identity_or_fingerprint_mismatch(tmp_path):
    requests = requests_for(tmp_path.resolve())
    evidence = evidence_for(requests, ("valid", "missing"))
    with pytest.raises(RollingEvidenceContractError):
        classify_rolling_recovery(tuple(reversed(requests)), evidence)
    changed = replace(requests[0], source_operation="resume")
    with pytest.raises(RollingEvidenceContractError):
        classify_rolling_recovery((changed, requests[1]), evidence)
    with pytest.raises(RollingEvidenceContractError):
        replace(evidence, unit_results=(evidence.unit_results[1], evidence.unit_results[0]))


def test_proposal_exposes_no_execute_transition_publish_or_apply_capability(tmp_path):
    requests = requests_for(tmp_path.resolve())
    proposal = classify_rolling_recovery(requests, evidence_for(requests, ("valid", "valid")))
    assert proposal.capabilities == ("render", "proposal_only")
    for name in ("apply", "execute", "publish", "transition", "repository", "backend"):
        assert not hasattr(proposal, name)
    with pytest.raises(FrozenInstanceError):
        proposal.status = "blocked"
