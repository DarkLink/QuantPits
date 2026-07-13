import json
from datetime import datetime

import pytest

from quantpits.training.record_repository import (
    TrainingRecordBaseline, TrainingRecordConflictError, TrainingRecordRepository,
)
from quantpits.training.records import ModelRecordEntry, ModelRecordOutcome


def _outcome(key="m@static", rid="r1"):
    name, mode = key.rsplit("@", 1)
    entry = ModelRecordEntry(
        key, name, mode, "train", "ready", rid, "exp",
        requested_anchor="2026-07-10", prediction_start="2026-07-10",
        prediction_end="2026-07-10", prediction_rows=1,
    )
    return ModelRecordOutcome(key, "train", "success", entry)


def test_repository_merge_preserves_and_atomically_projects(tmp_path):
    path = tmp_path / "latest_train_records.json"
    repo = TrainingRecordRepository(path)
    repo.merge([_outcome()])
    repo.merge([ModelRecordOutcome("m@static", "train", "failed", error_code="failed"), _outcome("n@static", "r2")])
    data = json.loads(path.read_text())
    assert data["models"] == {"m@static": "r1", "n@static": "r2"}


def test_incomplete_overwrite_is_rejected(tmp_path):
    repo = TrainingRecordRepository(tmp_path / "records.json")
    with pytest.raises(ValueError):
        repo.overwrite([ModelRecordOutcome("m@static", "train", "failed")])


def test_repository_uses_one_timestamp_for_stored_and_returned_snapshot(tmp_path):
    path = tmp_path / "records.json"
    calls = []
    def clock():
        calls.append(True)
        return datetime(2026, 7, 13, 12, 0, 0)
    snapshot = TrainingRecordRepository(path, clock=clock).merge([_outcome()])
    assert len(calls) == 1
    assert snapshot.updated_at == "2026-07-13T12:00:00"
    assert json.loads(path.read_text())["updated_at"] == snapshot.updated_at


def test_duplicate_outcomes_fail_before_lock_creation(tmp_path):
    path = tmp_path / "records.json"
    with pytest.raises(ValueError, match="duplicate"):
        TrainingRecordRepository(path).merge([_outcome(), _outcome()])
    assert not path.with_name("records.json.lock").exists()


def test_expected_absence_is_a_real_conflict_baseline(tmp_path):
    path = tmp_path / "records.json"
    repo = TrainingRecordRepository(path)
    baseline = TrainingRecordBaseline(None, {})
    path.write_text(json.dumps({"experiment_name": "legacy", "models": {}}))
    with pytest.raises(TrainingRecordConflictError):
        repo.merge([_outcome()], baseline=baseline)


def test_preview_is_deterministic_and_creates_no_lock(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"experiment_name": "legacy", "models": {"m@static": "r"}}))
    repo = TrainingRecordRepository(path)
    assert repo.preview_upgrade().to_public_dict(include_preview=True) == repo.preview_upgrade().to_public_dict(include_preview=True)
    assert not path.with_name("records.json.lock").exists()
