import hashlib
import io
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from quantpits.rolling import (
    RollingArtifactExpectation,
    RollingEvidenceContractError,
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingUnitEvidenceRequest,
    RollingWindowIdentity,
    classify_rolling_recovery,
    inspect_rolling_evidence,
    workspace_fingerprint,
)
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


class RecoveryBackend:
    def __init__(self, identity, candidates, *, drift=False):
        self.identity = identity
        self.candidates = tuple(candidates)
        self.drift = drift
        self.calls = 0

    def tracking_identity(self):
        return dict(self.identity)

    def inventory(self, requests):
        self.calls += 1
        token = self.calls if self.drift else "stable"
        return {"fingerprint": fingerprint_value(token), "candidates": self.candidates}


def prediction_bytes(sessions):
    frame = pd.DataFrame(
        {"score": [float(index) for index, _ in enumerate(sessions, 1)]},
        index=pd.MultiIndex.from_tuples(
            tuple((pd.Timestamp(session), "asset-%s" % index) for index, session in enumerate(sessions, 1)),
            names=("datetime", "instrument"),
        ),
    )
    buffer = io.BytesIO()
    frame.to_pickle(buffer)
    return buffer.getvalue()


def evidence_for(tmp_path, classes, *, protocols=None, drift=False, orphan=False):
    root = (tmp_path / "workspace").resolve()
    root.mkdir(parents=True)
    window = RollingWindowIdentity(
        "rolling", "2025-01-01", "2025-10-31", "2026-01-05", "2026-01-06",
        "a" * 64, "2025-11-01", "2025-12-31",
    )
    targets = tuple(RollingTargetIdentity(name, "rolling").target_key for name in ("alpha", "beta"))
    run = RollingRunIdentity(
        workspace_fingerprint(root), "rolling", "merge", "b" * 64, "c" * 64,
        "2026-01-06", targets, (window.window_key,), "d" * 64, "attempt-1",
    )
    protocols = protocols or ("execution_bound_v1", "execution_bound_v1")
    requests = []
    candidates = []
    for index, (target, protocol, classification) in enumerate(zip(targets, protocols, classes), 1):
        sessions = ("2026-01-05",) if classification == "coverage_short" else ("2026-01-05", "2026-01-06")
        data = prediction_bytes(sessions)
        expectation = RollingArtifactExpectation(
            "pred.pkl", "prediction", len(data), hashlib.sha256(data).hexdigest(),
        )
        request = RollingUnitEvidenceRequest(
            run, target, window, protocol, target, "merge", "exp", "exp-1",
            "rec-%s" % index, (expectation,), ("2026-01-05", "2026-01-06"),
        )
        requests.append(request)
        if classification != "missing":
            artifact_root = root / "records" / request.recorder_id
            artifact_root.mkdir(parents=True)
            (artifact_root / "pred.pkl").write_bytes(data)
            candidates.append({
                "workspace_fingerprint": run.workspace_fingerprint,
                "backend_fingerprint": "9" * 64,
                "experiment_name": request.experiment_name,
                "experiment_id": request.experiment_id,
                "recorder_id": request.recorder_id,
                "run_fingerprint": run.fingerprint,
                "attempt_id": run.attempt_id,
                "plan_fingerprint": run.plan_fingerprint,
                "config_fingerprint": run.config_fingerprint,
                "target_key": request.target_key,
                "window_key": request.window_key,
                "source_protocol": request.source_protocol,
                "source_publication_key": request.source_publication_key,
                "source_operation": request.source_operation,
                "source_manifest_fingerprint": request.source_manifest_fingerprint,
                "artifact_root_uri": artifact_root.as_uri(),
            })
    if orphan:
        candidates.append(dict(
            candidates[0], target_key="orphan@rolling",
            window_key="rolling:2026-01-05:2026-01-06:123456789abc",
            recorder_id="orphan-recorder", source_manifest_fingerprint="f" * 64,
        ))
    backend = RecoveryBackend({
        "workspace_fingerprint": run.workspace_fingerprint,
        "backend_fingerprint": "9" * 64,
        "present": True,
        "contained": True,
    }, candidates, drift=drift)
    requests = tuple(requests)
    evidence = inspect_rolling_evidence(WorkspaceContext.from_root(root), requests, backend)
    return requests, evidence


def test_all_valid_units_form_proposal_only_complete_reuse(tmp_path):
    requests, evidence = evidence_for(tmp_path, ("valid", "valid"), orphan=True)
    proposal = classify_rolling_recovery(requests, evidence)
    assert proposal.status == "all_reusable"
    assert proposal.reusable_unit_keys == tuple(item.unit_key for item in requests)
    assert proposal.unresolved_unit_keys == ()
    assert proposal.recovery_complete is True
    assert proposal.orphan_unit_keys == (("orphan@rolling", "rolling:2026-01-05:2026-01-06:123456789abc"),)


def test_mixed_units_preserve_valid_reuse_and_ordered_unresolved_members(tmp_path):
    requests, evidence = evidence_for(tmp_path, ("valid", "coverage_short"))
    proposal = classify_rolling_recovery(requests, evidence)
    assert proposal.status == "incomplete"
    assert proposal.reusable_unit_keys == (requests[0].unit_key,)
    assert proposal.unresolved_unit_keys == (requests[1].unit_key,)
    assert proposal.recovery_complete is False


def test_no_valid_or_legacy_only_evidence_grants_no_reuse(tmp_path):
    requests, evidence = evidence_for(
        tmp_path, ("legacy_unverified", "legacy_unverified"),
        protocols=("legacy_unverified", "legacy_unverified"),
    )
    proposal = classify_rolling_recovery(requests, evidence)
    assert proposal.status == "no_reusable_evidence"
    assert proposal.reusable_unit_keys == ()
    assert proposal.unresolved_unit_keys == tuple(item.unit_key for item in requests)


def test_drift_or_aggregate_ambiguity_blocks_all_reuse(tmp_path):
    requests, evidence = evidence_for(tmp_path, ("valid", "valid"), drift=True)
    proposal = classify_rolling_recovery(requests, evidence)
    assert proposal.status == "blocked"
    assert proposal.reusable_unit_keys == ()
    assert proposal.capabilities == ("render", "proposal_only")
    with pytest.raises(RollingEvidenceContractError):
        replace(proposal, reusable_unit_keys=(requests[0].unit_key,))


def test_recovery_rejects_request_result_identity_or_fingerprint_mismatch(tmp_path):
    requests, evidence = evidence_for(tmp_path, ("valid", "missing"))
    with pytest.raises(RollingEvidenceContractError):
        classify_rolling_recovery(tuple(reversed(requests)), evidence)
    changed = replace(requests[0], source_operation="resume")
    with pytest.raises(RollingEvidenceContractError):
        classify_rolling_recovery((changed, requests[1]), evidence)
    with pytest.raises(RollingEvidenceContractError):
        replace(evidence, unit_results=(evidence.unit_results[1], evidence.unit_results[0]))
    with pytest.raises(RollingEvidenceContractError):
        replace(evidence.unit_results[0])


def test_proposal_exposes_no_execute_transition_publish_or_apply_capability(tmp_path):
    requests, evidence = evidence_for(tmp_path, ("valid", "valid"))
    proposal = classify_rolling_recovery(requests, evidence)
    assert proposal.capabilities == ("render", "proposal_only")
    for name in ("apply", "execute", "publish", "transition", "repository", "backend"):
        assert not hasattr(proposal, name)
    with pytest.raises(FrozenInstanceError):
        proposal.status = "blocked"
