from quantpits.training.history import TrainingHistoryRepository


def test_history_append_is_idempotent_by_event_identity(tmp_path):
    repository = TrainingHistoryRepository(tmp_path / "training_history.jsonl")
    event = {
        "event_id": "stable-event", "operation": "train", "model_name": "demo",
        "mode": "static", "record_id": "rid",
    }
    assert repository.append(event) is True
    assert repository.append(event) is False
    assert len((tmp_path / "training_history.jsonl").read_text().splitlines()) == 1
