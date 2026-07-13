import json

import pytest

from quantpits.training.record_repository import TrainingRecordConflictError, TrainingRecordRepository
from quantpits.training.records import ModelRecordEntry, ModelRecordOutcome


def _outcome(key="m@static", rid="r1"):
    name, mode = key.rsplit("@", 1)
    entry = ModelRecordEntry(key, name, mode, "train", "ready", rid, "exp", prediction_end="2026-07-10")
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
