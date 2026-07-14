import pytest

from quantpits.training.errors import TrainingStateConflictError
from quantpits.training.lease import TrainingExecutionLease
from quantpits.utils.workspace import WorkspaceContext


def test_workspace_training_lease_is_nonblocking_and_reusable(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    first = TrainingExecutionLease.for_workspace(ctx)
    second = TrainingExecutionLease.for_workspace(ctx)
    first.acquire(run_id="one")
    try:
        with pytest.raises(TrainingStateConflictError, match="owns"):
            second.acquire(run_id="two")
    finally:
        first.release()
    second.acquire(run_id="two")
    second.release()


def test_constructing_lease_is_read_only(tmp_path):
    TrainingExecutionLease.for_workspace(WorkspaceContext.from_root(tmp_path))
    assert not (tmp_path / "data").exists()
