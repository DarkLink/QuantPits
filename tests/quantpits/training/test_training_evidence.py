import json
from dataclasses import replace

import pytest

from quantpits.training.evidence import TrainingTargetEvidence, TrainingTargetEvidenceRepository
from quantpits.training.errors import TrainingStateConflictError
from quantpits.training.records import ModelRecordEntry
from quantpits.utils.workspace import WorkspaceContext


def _evidence(run_id="run-1"):
    entry = ModelRecordEntry(
        "demo@static", "demo", "static", "train", "ready", "rid", "experiment",
        requested_anchor="2026-07-10", prediction_start="2026-07-10",
        prediction_end="2026-07-10", prediction_rows=1,
    )
    return TrainingTargetEvidence(
        run_id, "attempt-1", "demo@static", "train", "success", entry.to_dict(),
        {"IC": 0.1}, {"metric": 1}, None, "2026-07-10", "experiment",
        "plan", "resume", "execution", "2026-07-14T00:00:00",
    )


def test_target_evidence_is_immutable_and_reconstructs_result(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    repo = TrainingTargetEvidenceRepository(ctx, "run-1")
    relative, fingerprint = repo.write(_evidence())
    assert relative.startswith("data/training_runs/run-1/targets/demo_static-")
    assert relative.endswith(".json")
    loaded = repo.load("demo@static", expected_fingerprint=fingerprint)
    assert loaded.to_result().entry.recorder_id == "rid"
    assert repo.write(_evidence()) == (relative, fingerprint)
    path = ctx.path(relative)
    value = json.loads(path.read_text())
    value["attempt_id"] = "other"
    path.write_text(json.dumps(value))
    with pytest.raises(TrainingStateConflictError, match="fingerprint"):
        repo.load("demo@static", expected_fingerprint=fingerprint)


def test_target_evidence_read_does_not_create_workspace(tmp_path):
    root = tmp_path / "missing"
    repo = TrainingTargetEvidenceRepository(WorkspaceContext.from_root(root), "run-1")
    with pytest.raises(TrainingStateConflictError, match="missing"):
        repo.load("demo@static")
    assert not root.exists()


def test_target_evidence_normalizes_scalar_item_values(tmp_path):
    class Scalar:
        def item(self):
            return 0.25

    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    repo = TrainingTargetEvidenceRepository(ctx, "run-1")
    relative, fingerprint = repo.write(replace(_evidence(), performance={"IC": Scalar()}))
    assert repo.load("demo@static", expected_fingerprint=fingerprint).performance == {"IC": 0.25}
    assert json.loads(ctx.path(relative).read_text())["performance"] == {"IC": 0.25}


@pytest.mark.parametrize("field,value,match", [
    ("run_id", "other-run", "identity mismatch"),
    ("target_key", "other@static", "identity"),
    ("operation", "predict", "record entry"),
    ("outcome", "unknown", "unsupported"),
])
def test_target_evidence_rejects_corrupt_identity_fields(tmp_path, field, value, match):
    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    repo = TrainingTargetEvidenceRepository(ctx, "run-1")
    path = repo.path_for("demo@static")
    path.parent.mkdir(parents=True)
    payload = _evidence().to_dict()
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(TrainingStateConflictError, match=match):
        repo.load("demo@static")


def test_target_evidence_path_is_contained_for_adversarial_target_key(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    repo = TrainingTargetEvidenceRepository(ctx, "run-1")
    path = repo.path_for("../../outside@static").resolve()
    assert path.relative_to(repo.root.resolve())
