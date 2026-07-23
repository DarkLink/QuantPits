from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from quantpits.rolling import (
    RollingExecutionKernel,
    QlibMlflowExecutionBackend,
    RollingUnitRunnerObservation,
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
        def commit_execution_manifest(self, scope, unit, observation, recorder_baseline):
            request = super().commit_execution_manifest(
                scope, unit, observation, recorder_baseline,
            )
            root = Path(self.candidates[unit.unit_key]["artifact_root_uri"].removeprefix("file://"))
            (root / "pred.pkl").write_bytes(b"corrupt-after-manifest")
            return request

    backend = CorruptingBackend(backend.context)
    result = RollingExecutionKernel(repository, backend, FakeRunner(backend.context)).execute(scope, "attempt-1")
    assert result.status == "failed"
    assert result.n_executed_success == 0
    assert repository.inspect_readonly().inspection.snapshot.phase == "failed"


def test_existing_recorder_cannot_be_rewrapped_as_executed_success(tmp_path):
    _context, scope, repository, backend = _case(tmp_path)
    unit = scope.units[0]
    backend.candidates[unit.unit_key] = {
        "recorder_id": "recorder-attempt-1-0",
        "preexisting": True,
    }
    result = RollingExecutionKernel(
        repository, backend, FakeRunner(backend.context),
    ).execute(scope, "attempt-1")
    assert result.status == "failed"
    assert result.n_executed_success == 0
    state = repository.inspect_readonly().inspection.snapshot
    assert state.units[0].status == "failed"
    assert state.units[0].record_id is None


def test_mlflow_manifest_requires_unique_new_recorder_experiment_and_tags(monkeypatch, tmp_path):
    context, scope, _repository, _backend = _case(tmp_path)
    backend = QlibMlflowExecutionBackend(context)
    unit = scope.units[0]
    expected = ("exact-unit-experiment", "experiment-1", "recorder-1")
    artifact_root = context.mlruns_dir / "experiment-1" / "recorder-1" / "artifacts"
    artifact_root.mkdir(parents=True)
    (artifact_root / "model.pkl").write_bytes(b"model")
    (artifact_root / "pred.pkl").write_bytes(b"prediction")

    class Recorder:
        info = {"id": "recorder-1"}

        def __init__(self, tags):
            self.tags = tags

        def get_artifact_uri(self):
            return artifact_root.as_uri()

        def list_tags(self):
            return dict(self.tags)

        def log_artifact(self, path):
            (artifact_root / "execution_manifest.json").write_bytes(
                Path(path).read_bytes(),
            )

    tags = {
        "execution_protocol": scope.execution_protocol_version,
        "run_fingerprint": scope.run_identity.fingerprint,
        "attempt_id": "attempt-1",
        "target_key": unit.unit_key[0],
        "window_key": unit.unit_key[1],
        "source_operation": scope.run_identity.action,
    }
    recorder = Recorder(tags)
    monkeypatch.setattr(backend, "_recorder", lambda *_args: recorder)
    workflow = ModuleType("qlib.workflow")
    workflow.R = SimpleNamespace(
        get_exp=lambda **_kwargs: SimpleNamespace(id="experiment-1"),
    )
    qlib = ModuleType("qlib")
    qlib.workflow = workflow
    monkeypatch.setitem(sys.modules, "qlib", qlib)
    monkeypatch.setitem(sys.modules, "qlib.workflow", workflow)
    observation = RollingUnitRunnerObservation(
        unit.unit_key, "attempt-1", "candidate_success", *expected,
    )

    inventories = iter(((), (expected,), (expected,)))
    monkeypatch.setattr(backend, "_recorder_inventory", lambda: next(inventories))
    baseline = backend.capture_recorder_inventory(scope, unit, "attempt-1")
    request = backend.commit_execution_manifest(
        scope, unit, observation, baseline,
    )
    assert request.recorder_id == "recorder-1"

    monkeypatch.setattr(backend, "_recorder_inventory", lambda: (expected,))
    preexisting = backend.capture_recorder_inventory(scope, unit, "attempt-1")
    with pytest.raises(Exception, match="exactly its one claimed recorder"):
        backend.commit_execution_manifest(scope, unit, observation, preexisting)

    bad_recorder = Recorder(dict(tags, attempt_id="foreign-attempt"))
    monkeypatch.setattr(backend, "_recorder", lambda *_args: bad_recorder)
    inventories = iter(((), (expected,)))
    monkeypatch.setattr(backend, "_recorder_inventory", lambda: next(inventories))
    baseline = backend.capture_recorder_inventory(scope, unit, "attempt-1")
    with pytest.raises(Exception, match="provenance tags disagree"):
        backend.commit_execution_manifest(scope, unit, observation, baseline)


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
    runner = FakeRunner(context, timeline=timeline)
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
    first_runner = FakeRunner(backend.context)
    first = RollingExecutionKernel(repository, backend, first_runner).execute(scope, "attempt-1")
    recorder = first.unit_results[0].record_id
    second_runner = FakeRunner(backend.context)
    resumed = RollingExecutionKernel(repository, backend, second_runner).resume(scope, "attempt-2")
    assert resumed.status == "success"
    assert resumed.n_reused_success == 1
    assert resumed.n_runner_calls == 0
    assert resumed.unit_results[0].record_id == recorder
    assert second_runner.calls == []
    assert backend.current_lookup_calls == 0


def test_resume_requires_the_exact_phase32_recovery_proposal(monkeypatch, tmp_path):
    _context, scope, repository, backend = _case(tmp_path)
    first = RollingExecutionKernel(
        repository, backend, FakeRunner(backend.context),
    ).execute(scope, "attempt-1")
    assert first.status == "success"

    def reject_recovery(_requests, _evidence):
        raise RuntimeError("injected foreign recovery proposal")

    monkeypatch.setattr(
        "quantpits.rolling.execution_backend.classify_rolling_recovery",
        reject_recovery,
    )
    runner = FakeRunner(backend.context)
    resumed = RollingExecutionKernel(
        repository, backend, runner,
    ).resume(scope, "attempt-2")
    assert resumed.status == "blocked"
    assert tuple(item.reason_code for item in resumed.unit_results) == (
        "rolling_execution_recovery_proposal_blocked",
    )
    assert runner.calls == []


def test_resume_retries_only_missing_evidence_units_with_new_attempt(tmp_path):
    _context, scope, repository, backend = _case(tmp_path, n_windows=3)
    first_runner = FakeRunner(backend.context, failures=(1,))
    first = RollingExecutionKernel(repository, backend, first_runner).execute(scope, "attempt-1")
    assert first.status == "failed"
    second_runner = FakeRunner(backend.context)
    resumed = RollingExecutionKernel(repository, backend, second_runner).resume(scope, "attempt-2")
    assert resumed.status == "success"
    assert resumed.n_reused_success == 2
    assert resumed.n_executed_success == 1
    assert second_runner.calls == [(scope.units[1].unit_key, "attempt-2")]


def test_retry_success_preserves_every_prior_failure_attempt_in_order(tmp_path):
    _context, scope, repository, backend = _case(tmp_path)
    first = RollingExecutionKernel(
        repository, backend, FakeRunner(backend.context, failures=(0,)),
    ).execute(scope, "attempt-1")
    assert first.status == "failed"
    first_audit = repository.inspect_readonly().inspection.snapshot.units[0].extensions
    assert first_audit["attempt_id"] == "attempt-1"
    assert first_audit["failure_code"] == "RollingExecutionPreflightError"
    assert "prior_attempts" not in first_audit

    second = RollingExecutionKernel(
        repository, backend, FakeRunner(backend.context, failures=(0,)),
    ).resume(scope, "attempt-2")
    assert second.status == "failed"
    second_audit = repository.inspect_readonly().inspection.snapshot.units[0].extensions
    assert second_audit["attempt_id"] == "attempt-2"
    assert second_audit["prior_attempts"] == [first_audit]

    third = RollingExecutionKernel(
        repository, backend, FakeRunner(backend.context),
    ).resume(scope, "attempt-3")
    assert third.status == "success"
    terminal = repository.inspect_readonly().inspection.snapshot.units[0]
    assert terminal.status == "success"
    assert terminal.extensions["attempt_id"] == "attempt-3"
    assert terminal.extensions["prior_attempts"] == [
        first_audit,
        {key: value for key, value in second_audit.items() if key != "prior_attempts"},
    ]
    assert terminal.record_id == third.unit_results[0].record_id
    assert terminal.evidence_id == third.unit_results[0].evidence_id


def test_nonmissing_invalid_evidence_blocks_retry_and_preserves_scope(tmp_path):
    _context, scope, repository, backend = _case(tmp_path)
    RollingExecutionKernel(repository, backend, FakeRunner(backend.context)).execute(scope, "attempt-1")
    candidate = backend.candidates[scope.units[0].unit_key]
    root = Path(candidate["artifact_root_uri"].removeprefix("file://"))
    data = (root / "pred.pkl").read_bytes()
    (root / "pred.pkl").write_bytes(data + b"drift")
    runner = FakeRunner(backend.context)
    resumed = RollingExecutionKernel(repository, backend, runner).resume(scope, "attempt-2")
    assert resumed.status == "blocked"
    assert resumed.requested_unit_keys == scope.requested_unit_keys
    assert runner.calls == []


def test_unit_exception_fails_one_and_preserves_later_unit_order(tmp_path):
    _context, scope, repository, backend = _case(tmp_path, n_windows=3)
    runner = FakeRunner(backend.context, controls={1: RuntimeError("ordinary")})
    result = RollingExecutionKernel(repository, backend, runner).execute(scope, "attempt-1")
    assert result.status == "failed"
    assert tuple(item.unit_key for item in result.unit_results) == scope.requested_unit_keys
    assert [item.status for item in result.unit_results] == [
        "executed_success", "failed", "executed_success",
    ]
    assert runner.calls[-1][0] == scope.units[2].unit_key
