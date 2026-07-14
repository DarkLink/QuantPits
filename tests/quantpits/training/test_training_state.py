import pytest

from quantpits.training.errors import TrainingStateConflictError
from quantpits.training.state import TrainingRunState, TrainingStateRepository


def state(phase="prepared", run_id="run-1", **kwargs):
    values = dict(
        run_id=run_id, family="static", action="incremental",
        plan_fingerprint="plan", execution_fingerprint="execution",
        resume_fingerprint="resume", anchor_date="2026-07-10",
        target_keys=("demo@static",), outcomes={}, phase=phase,
    )
    values.update(kwargs)
    return TrainingRunState(**values)


def test_state_repository_uses_cas_and_validates_transitions(tmp_path):
    repo = TrainingStateRepository(tmp_path / "run_state.json")
    current, absent = repo.inspect_readonly()
    assert current is None
    prepared = repo.save(state(), expected=absent)
    executing = repo.save(state("executing"), expected=prepared)
    with pytest.raises(TrainingStateConflictError, match="changed"):
        repo.save(state("targets_complete"), expected=prepared)
    with pytest.raises(TrainingStateConflictError, match="transition"):
        repo.save(state("completed"), expected=executing)


def test_state_rejects_other_run_and_published_without_recorder(tmp_path):
    repo = TrainingStateRepository(tmp_path / "run_state.json")
    baseline = repo.save(state())
    with pytest.raises(TrainingStateConflictError, match="another run"):
        repo.save(state("executing", run_id="run-2"), expected=baseline)
    with pytest.raises(TrainingStateConflictError, match="recorder evidence"):
        state("executing", outcomes={"demo@static": {"published": True}})


def test_readonly_state_inspect_creates_nothing(tmp_path):
    repo = TrainingStateRepository(tmp_path / "missing" / "run_state.json")
    assert repo.inspect_readonly()[0] is None
    assert not (tmp_path / "missing").exists()
