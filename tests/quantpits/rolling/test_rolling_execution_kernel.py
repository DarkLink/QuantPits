from pathlib import Path

from quantpits.rolling import (
    RollingExecutionKernel,
    RollingStateRepository,
)
from quantpits.utils.workspace import WorkspaceContext

from tests.quantpits.rolling.execution_support import (
    FakeExecutionBackend,
    FakeRunner,
    linear_capability_result,
    make_scope,
)


def _case(tmp_path, n_windows=1):
    root = (tmp_path / "workspace").resolve()
    for name in ("config", "data", "mlruns", "output"):
        (root / name).mkdir(parents=True, exist_ok=True)
    context = WorkspaceContext.from_root(root)
    scope = make_scope(context, linear_capability_result(), 1, n_windows)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    backend = FakeExecutionBackend(context)
    return context, scope, repository, backend


def test_runner_success_without_exact_evidence_is_failed_not_success(tmp_path):
    _context, scope, repository, backend = _case(tmp_path)

    class CorruptingBackend(FakeExecutionBackend):
        def commit_execution_manifest(self, scope, unit, observation):
            request = super().commit_execution_manifest(scope, unit, observation)
            root = Path(self.candidates[unit.unit_key]["artifact_root_uri"].removeprefix("file://"))
            (root / "pred.pkl").write_bytes(b"corrupt-after-manifest")
            return request

    backend = CorruptingBackend(backend.context)
    result = RollingExecutionKernel(repository, backend, FakeRunner()).execute(scope, "attempt-1")
    assert result.status == "failed"
    assert result.n_executed_success == 0
    assert repository.inspect_readonly().inspection.snapshot.phase == "failed"


def test_valid_execution_commits_evidence_then_state_success(tmp_path):
    context, scope, _repository, _backend = _case(tmp_path)
    timeline = []

    class TimelineRepository(RollingStateRepository):
        def commit_evidence_authorized(self, proposed, expected, evidence_set, blocking=True):
            receipt = super().commit_evidence_authorized(
                proposed, expected, evidence_set, blocking=blocking,
            )
            timeline.append("state:%s:%s" % (proposed.phase, receipt.status))
            return receipt

    repository = TimelineRepository.for_workspace(context, "rolling")
    backend = FakeExecutionBackend(context, timeline=timeline)
    runner = FakeRunner(timeline=timeline)
    result = RollingExecutionKernel(repository, backend, runner).execute(scope, "attempt-1")
    state = repository.inspect_readonly().inspection.snapshot
    assert result.status == "success"
    assert result.n_executed_success == 1
    assert result.n_runner_calls == 1
    assert backend.manifest_calls == [scope.units[0].unit_key]
    assert state.phase == "units_complete"
    assert state.units[0].record_id == result.unit_results[0].record_id
    assert state.units[0].evidence_id == result.unit_results[0].evidence_id
    assert timeline.index("runner:0") < timeline.index("manifest:0")
    assert timeline.index("manifest:0") < timeline.index("inspect:1")
    assert timeline.index("inspect:1") < timeline.index("state:executing:committed")
    assert timeline.index("state:executing:committed") < timeline.index(
        "state:units_complete:committed",
    )


def test_resume_reuses_original_source_without_current_record_lookup_or_runner_call(tmp_path):
    _context, scope, repository, backend = _case(tmp_path)
    first_runner = FakeRunner()
    first = RollingExecutionKernel(repository, backend, first_runner).execute(scope, "attempt-1")
    recorder = first.unit_results[0].record_id
    second_runner = FakeRunner()
    resumed = RollingExecutionKernel(repository, backend, second_runner).resume(scope, "attempt-2")
    assert resumed.status == "success"
    assert resumed.n_reused_success == 1
    assert resumed.n_runner_calls == 0
    assert resumed.unit_results[0].record_id == recorder
    assert second_runner.calls == []
    assert backend.current_lookup_calls == 0


def test_resume_retries_only_missing_evidence_units_with_new_attempt(tmp_path):
    _context, scope, repository, backend = _case(tmp_path, n_windows=3)
    first_runner = FakeRunner(failures=(1,))
    first = RollingExecutionKernel(repository, backend, first_runner).execute(scope, "attempt-1")
    assert first.status == "failed"
    second_runner = FakeRunner()
    resumed = RollingExecutionKernel(repository, backend, second_runner).resume(scope, "attempt-2")
    assert resumed.status == "success"
    assert resumed.n_reused_success == 2
    assert resumed.n_executed_success == 1
    assert second_runner.calls == [(scope.units[1].unit_key, "attempt-2")]


def test_nonmissing_invalid_evidence_blocks_retry_and_preserves_scope(tmp_path):
    _context, scope, repository, backend = _case(tmp_path)
    RollingExecutionKernel(repository, backend, FakeRunner()).execute(scope, "attempt-1")
    candidate = backend.candidates[scope.units[0].unit_key]
    root = Path(candidate["artifact_root_uri"].removeprefix("file://"))
    data = (root / "pred.pkl").read_bytes()
    (root / "pred.pkl").write_bytes(data + b"drift")
    runner = FakeRunner()
    resumed = RollingExecutionKernel(repository, backend, runner).resume(scope, "attempt-2")
    assert resumed.status == "blocked"
    assert resumed.requested_unit_keys == scope.requested_unit_keys
    assert runner.calls == []


def test_unit_exception_fails_one_and_preserves_later_unit_order(tmp_path):
    _context, scope, repository, backend = _case(tmp_path, n_windows=3)
    runner = FakeRunner(controls={1: RuntimeError("ordinary")})
    result = RollingExecutionKernel(repository, backend, runner).execute(scope, "attempt-1")
    assert result.status == "failed"
    assert tuple(item.unit_key for item in result.unit_results) == scope.requested_unit_keys
    assert [item.status for item in result.unit_results] == [
        "executed_success", "failed", "executed_success",
    ]
    assert runner.calls[-1][0] == scope.units[2].unit_key
