import hashlib
import io

import pandas as pd

from quantpits.rolling import (
    RollingArtifactExpectation,
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingUnitEvidenceRequest,
    RollingWindowIdentity,
    classify_rolling_recovery,
    inspect_rolling_evidence,
    workspace_fingerprint,
)
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


class SemanticBackend:
    def __init__(self, identity, candidates):
        self.identity = identity
        self.candidates = tuple(candidates)

    def tracking_identity(self):
        return dict(self.identity)

    def inventory(self, requests):
        return {"fingerprint": fingerprint_value("stable-semantic-inventory"), "candidates": self.candidates}


def semantic_invocation(tmp_path, *, drift=False):
    root = (tmp_path / "workspace").resolve()
    artifact_root = root / "records" / "one"
    artifact_root.mkdir(parents=True)
    frame = pd.DataFrame(
        {"score": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            ((pd.Timestamp("2026-01-05"), "asset-a"), (pd.Timestamp("2026-01-06"), "asset-b")),
            names=("datetime", "instrument"),
        ),
    )
    buffer = io.BytesIO()
    frame.to_pickle(buffer)
    data = buffer.getvalue()
    (artifact_root / "pred.pkl").write_bytes(data)
    window = RollingWindowIdentity(
        "rolling", "2025-01-01", "2025-10-31", "2026-01-05", "2026-01-06",
        "a" * 64, "2025-11-01", "2025-12-31",
    )
    targets = (RollingTargetIdentity("alpha", "rolling").target_key, RollingTargetIdentity("beta", "rolling").target_key)
    run = RollingRunIdentity(
        workspace_fingerprint(root), "rolling", "merge", "b" * 64, "c" * 64,
        "2026-01-06", targets, (window.window_key,), "d" * 64, "attempt-1",
    )
    expectation = RollingArtifactExpectation("pred.pkl", "prediction", len(data), hashlib.sha256(data).hexdigest())
    requests = tuple(
        RollingUnitEvidenceRequest(
            run, target, window, "execution_bound_v1", target, "merge", "exp", "exp-1",
            "rec-%s" % index, (expectation,), ("2026-01-05", "2026-01-06"),
        )
        for index, target in enumerate(targets, 1)
    )
    first = requests[0]
    candidate = {
        "workspace_fingerprint": run.workspace_fingerprint,
        "backend_fingerprint": "e" * 64,
        "experiment_name": first.experiment_name,
        "experiment_id": first.experiment_id,
        "recorder_id": first.recorder_id,
        "run_fingerprint": run.fingerprint,
        "attempt_id": run.attempt_id,
        "plan_fingerprint": run.plan_fingerprint,
        "config_fingerprint": run.config_fingerprint,
        "target_key": first.target_key,
        "window_key": first.window_key,
        "source_protocol": first.source_protocol,
        "source_publication_key": first.source_publication_key,
        "source_operation": first.source_operation,
        "source_manifest_fingerprint": first.source_manifest_fingerprint,
        "artifact_root_uri": artifact_root.as_uri(),
    }
    backend = SemanticBackend(
        {
            "workspace_fingerprint": run.workspace_fingerprint,
            "backend_fingerprint": "e" * 64,
            "present": True,
            "contained": True,
        },
        (candidate,),
    )
    if drift:
        calls = {"count": 0}
        def inventory(requests):
            calls["count"] += 1
            return {"fingerprint": fingerprint_value(calls["count"]), "candidates": (candidate,)}
        backend.inventory = inventory
    evidence = inspect_rolling_evidence(WorkspaceContext.from_root(root), requests, backend)
    return requests, evidence, classify_rolling_recovery(requests, evidence)


def test_requested_evidence_and_recovery_scope_are_identical_for_mixed_units(tmp_path):
    requests, evidence, proposal = semantic_invocation(tmp_path)
    expected = tuple(item.unit_key for item in requests)
    assert evidence.requested_unit_keys == expected
    assert tuple(item.unit_key for item in evidence.unit_results) == expected
    assert proposal.requested_unit_keys == expected
    assert proposal.reusable_unit_keys == tuple(key for key, result in zip(expected, evidence.unit_results) if result.classification == "valid")
    assert proposal.unresolved_unit_keys == tuple(key for key, result in zip(expected, evidence.unit_results) if result.classification != "valid")
    assert set(proposal.reusable_unit_keys).union(proposal.unresolved_unit_keys) == set(expected)
    assert evidence.n_requested == len(expected) == len(proposal.requested_unit_keys)
    assert proposal.evidence_set_fingerprint == evidence.fingerprint


def test_blocked_evidence_cannot_cross_the_recovery_capability_boundary(tmp_path):
    requests, evidence, proposal = semantic_invocation(tmp_path, drift=True)
    assert evidence.status == "observation_drifted"
    assert all(item.classification == "drifted" for item in evidence.unit_results)
    assert proposal.status == "blocked"
    assert proposal.reusable_unit_keys == ()
    assert proposal.capabilities == ("render", "proposal_only")
    assert not any(hasattr(proposal, name) for name in ("execute", "transition", "publish", "apply"))
