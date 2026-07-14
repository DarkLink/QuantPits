import pytest

from quantpits.training.closure import (
    TrainingClosureRepository, TrainingClosureState, closure_complete,
    required_closure_steps,
)
from quantpits.training.errors import TrainingStateConflictError
from quantpits.utils.workspace import WorkspaceContext


def test_closure_steps_round_trip_without_creating_on_read(tmp_path):
    root = tmp_path / "Demo_Workspace"
    ctx = WorkspaceContext.from_root(root)
    repository = TrainingClosureRepository(ctx, "run-1", "tx-1")
    state, baseline = repository.load()
    assert state is None
    assert baseline.existed is False
    assert not root.exists()

    repository.save(TrainingClosureState(
        "run-1", "tx-1", "receipt-fp",
        {"receipt_verified": "completed", "history_appended": "warning"},
        ("history_failed",),
        ({"path": "latest_train_records.json", "kind": "record", "fingerprint": "fp"},),
    ))
    loaded, _ = repository.load()
    assert loaded.steps["history_appended"] == "warning"
    assert loaded.warnings == ("history_failed",)


def test_closure_warning_can_be_replaced_by_completed_retry(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    repository = TrainingClosureRepository(ctx, "run-1", "tx-1")
    repository.save(TrainingClosureState(
        "run-1", "tx-1", "receipt-fp", {"promotion_applied": "warning"},
    ))
    repository.save(TrainingClosureState(
        "run-1", "tx-1", "receipt-fp", {"promotion_applied": "completed"},
    ))
    loaded, _ = repository.load()
    assert loaded.steps == {"promotion_applied": "completed"}


def test_closure_uses_one_canonical_required_step_contract():
    steps = {name: "completed" for name in required_closure_steps(
        no_manifest=False, include_terminal=True,
    )}
    assert closure_complete(steps, no_manifest=False)
    steps.pop("state_publication_bound")
    assert not closure_complete(steps, no_manifest=False)


def test_closure_rejects_completed_step_regression_and_unproven_operator_log(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    repository = TrainingClosureRepository(ctx, "run-1", "tx-1")
    repository.save(TrainingClosureState(
        "run-1", "tx-1", "receipt-fp", {"history_appended": "completed"},
    ))
    with pytest.raises(TrainingStateConflictError, match="regressed"):
        repository.save(TrainingClosureState(
            "run-1", "tx-1", "receipt-fp", {},
        ))
    with pytest.raises(TrainingStateConflictError, match="durable evidence"):
        TrainingClosureState(
            "run-1", "tx-1", "receipt-fp", {"operator_log_linked": "completed"},
        )
