import pytest

from quantpits.rolling import RollingExecutionKernel, RollingStateRepository
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
    return (
        make_scope(context, linear_capability_result(), 1, n_windows),
        RollingStateRepository.for_workspace(context, "rolling"),
        FakeExecutionBackend(context),
    )


def _assert_lease_released(repository, probe):
    from quantpits.training.lease import TrainingExecutionLease
    lease = TrainingExecutionLease.for_workspace(repository.context)
    lease.acquire(run_id=probe)
    lease.release()


@pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_keyboard_systemexit_generatorexit_propagate_and_leave_reconcilable_state(tmp_path, control):
    scope, repository, backend = _case(tmp_path, n_windows=2)
    runner = FakeRunner(backend.context, controls={0: control})
    with pytest.raises(control.__class__):
        RollingExecutionKernel(
            repository, backend, runner,
        ).execute(scope, "attempt-1")
    state = repository.inspect_readonly().inspection.snapshot
    assert state.phase == "failed"
    assert tuple(item.status for item in state.units) == ("failed", "pending")
    assert runner.calls == [(scope.units[0].unit_key, "attempt-1")]
    assert backend.manifest_calls == []
    assert backend.candidates == {}
    assert state.units[0].record_id is None
    assert state.units[0].evidence_id is None
    assert state.units[0].extensions == {
        "attempt_id": "attempt-1",
        "interrupted_attempt_id": "attempt-1",
    }
    assert state.units[1].record_id is None
    assert state.units[1].evidence_id is None
    _assert_lease_released(repository, "post-interrupt-probe")
    resumed_runner = FakeRunner(backend.context)
    resumed = RollingExecutionKernel(
        repository, backend, resumed_runner,
    ).resume(scope, "attempt-2")
    assert resumed.status == "success"
    assert [item[0] for item in resumed_runner.calls] == list(scope.requested_unit_keys)
    terminal = repository.inspect_readonly().inspection.snapshot
    assert terminal.phase == "units_complete"
    assert terminal.units[0].extensions["prior_attempts"] == [{
        "attempt_id": "attempt-1",
        "interrupted_attempt_id": "attempt-1",
    }]
    assert "prior_attempts" not in terminal.units[1].extensions


def test_each_runner_manifest_inspection_and_state_commit_fault_is_truthful(tmp_path):
    scope, _repository, backend = _case(tmp_path)
    points = []

    def fault(point):
        points.append(point)
        if point == "after_source_read_before_temp_create":
            raise OSError("injected")

    repository = RollingStateRepository.for_workspace(
        backend.context, "rolling", fault_hook=fault,
    )
    runner = FakeRunner(backend.context)
    result = RollingExecutionKernel(repository, backend, runner).execute(
        scope, "attempt-1",
    )
    assert result.status == "blocked"
    assert tuple(item.status for item in result.unit_results) == ("blocked",)
    assert runner.calls == []
    assert repository.inspect_readonly().inspection.classification == "missing"
    assert "after_source_read_before_temp_create" in points

    prepared_root = tmp_path / "prepared-crash-case"
    prepared_root.mkdir()
    prepared_scope, prepared_repository, prepared_backend = _case(prepared_root)

    class AfterPreparedFaultKernel(RollingExecutionKernel):
        commits = 0

        def _commit(self, state, baseline, evidence=None):
            observed = super()._commit(state, baseline, evidence)
            self.commits += 1
            if self.commits == 1:
                raise OSError("injected crash after prepared")
            return observed

    prepared = AfterPreparedFaultKernel(
        prepared_repository, prepared_backend, FakeRunner(prepared_backend.context),
    ).execute(prepared_scope, "attempt-1")
    assert prepared.status == "blocked"
    assert prepared_repository.inspect_readonly().inspection.snapshot.phase == "prepared"
    resumed = RollingExecutionKernel(
        prepared_repository, prepared_backend, FakeRunner(prepared_backend.context),
    ).resume(prepared_scope, "attempt-2")
    assert resumed.status == "success"
    assert resumed.n_executed_success == 1
    assert prepared_repository.inspect_readonly().inspection.snapshot.phase == "units_complete"

    executing_root = tmp_path / "executing-crash-case"
    executing_root.mkdir()
    executing_scope, executing_repository, executing_backend = _case(executing_root)

    class AfterExecutingFaultKernel(RollingExecutionKernel):
        def _commit(self, state, baseline, evidence=None):
            observed = super()._commit(state, baseline, evidence)
            if state.phase == "executing" and all(
                item.status == "pending" for item in state.units
            ):
                raise OSError("injected crash after executing")
            return observed

    executing_result = AfterExecutingFaultKernel(
        executing_repository, executing_backend, FakeRunner(executing_backend.context),
    ).execute(executing_scope, "attempt-1")
    assert executing_result.status == "blocked"
    executing_state = executing_repository.inspect_readonly().inspection.snapshot
    assert executing_state.phase == "executing"
    assert executing_state.units[0].status == "pending"
    executing_resumed = RollingExecutionKernel(
        executing_repository, executing_backend, FakeRunner(executing_backend.context),
    ).resume(executing_scope, "attempt-2")
    assert executing_resumed.status == "success"

    manifest_root = tmp_path / "manifest-case"
    manifest_root.mkdir()
    manifest_scope, manifest_repository, manifest_backend = _case(
        manifest_root, n_windows=2,
    )

    class ManifestFailureBackend(FakeExecutionBackend):
        def commit_execution_manifest(self, scope, unit, observation, recorder_baseline):
            if unit.position == 0:
                partial = (
                    self.context.root / "mlruns" / observation.experiment_id
                    / observation.recorder_id / "artifacts"
                )
                partial.mkdir(parents=True, exist_ok=True)
                (partial / "model.pkl").write_bytes(b"partial-model-only")
                raise OSError("injected manifest failure")
            return super().commit_execution_manifest(
                scope, unit, observation, recorder_baseline,
            )

    manifest_runner = FakeRunner(manifest_backend.context)
    manifest_result = RollingExecutionKernel(
        manifest_repository,
        ManifestFailureBackend(manifest_backend.context), manifest_runner,
    ).execute(manifest_scope, "attempt-1")
    assert tuple(item.status for item in manifest_result.unit_results) == (
        "failed", "executed_success",
    )
    assert [item[0] for item in manifest_runner.calls] == list(
        manifest_scope.requested_unit_keys,
    )
    assert manifest_result.unit_results[0].record_id is None
    assert manifest_result.unit_results[0].evidence_id is None
    assert manifest_result.unit_results[1].record_id is not None
    assert manifest_result.unit_results[1].evidence_id is not None
    manifest_state = manifest_repository.inspect_readonly().inspection.snapshot
    assert manifest_state.phase == "failed"
    assert tuple(item.status for item in manifest_state.units) == ("failed", "success")
    assert manifest_state.units[0].record_id is None
    assert manifest_state.units[0].evidence_id is None
    assert manifest_state.units[1].record_id == manifest_result.unit_results[1].record_id
    assert manifest_state.units[1].evidence_id == manifest_result.unit_results[1].evidence_id
    _assert_lease_released(manifest_repository, "post-manifest-fault-probe")

    success_root = tmp_path / "success-cas-case"
    success_root.mkdir()
    success_scope, success_repository, success_backend = _case(
        success_root, n_windows=2,
    )

    class AfterFirstSuccessFaultKernel(RollingExecutionKernel):
        def _commit(self, state, baseline, evidence=None):
            observed = super()._commit(state, baseline, evidence)
            if (
                state.phase == "executing"
                and state.units[0].status == "success"
                and state.units[1].status == "pending"
            ):
                raise OSError("injected crash after success CAS")
            return observed

    success_result = AfterFirstSuccessFaultKernel(
        success_repository, success_backend, FakeRunner(success_backend.context),
    ).execute(success_scope, "attempt-1")
    assert tuple(item.status for item in success_result.unit_results) == (
        "failed", "blocked",
    )
    durable = success_repository.inspect_readonly().inspection.snapshot
    assert durable.phase == "executing"
    assert tuple(item.status for item in durable.units) == ("success", "pending")
    assert len(success_backend.candidates) == 1
    assert durable.units[0].record_id is not None
    assert durable.units[0].evidence_id is not None
    assert durable.units[1].record_id is None
    assert durable.units[1].evidence_id is None
    _assert_lease_released(success_repository, "post-success-cas-fault-probe")
    recovered = RollingExecutionKernel(
        success_repository, success_backend, FakeRunner(success_backend.context),
    ).resume(success_scope, "attempt-2")
    assert recovered.status == "success"
    assert recovered.n_reused_success == 1
    assert recovered.n_executed_success == 1

    final_root = tmp_path / "finalization-case"
    final_root.mkdir()
    final_scope, final_repository, final_backend = _case(final_root)

    class FinalizationFaultKernel(RollingExecutionKernel):
        def _commit(self, state, baseline, evidence=None):
            if state.phase == "units_complete":
                raise OSError("injected finalization failure")
            return super()._commit(state, baseline, evidence)

    final_result = FinalizationFaultKernel(
        final_repository, final_backend, FakeRunner(final_backend.context),
    ).execute(final_scope, "attempt-1")
    assert final_result.status == "failed"
    assert final_result.unit_results[0].status == "failed"
    final_state = final_repository.inspect_readonly().inspection.snapshot
    assert final_state.phase == "executing"
    assert final_state.units[0].status == "success"
    assert len(final_backend.candidates) == 1
    assert final_state.units[0].record_id is not None
    assert final_state.units[0].evidence_id is not None
    _assert_lease_released(final_repository, "post-finalization-fault-probe")

    lease_root = tmp_path / "lease-postcondition-case"
    lease_root.mkdir()
    lease_scope, lease_repository, lease_backend = _case(lease_root)

    class LeasePostconditionKernel(RollingExecutionKernel):
        checks = 0

        def _verify_lease(self, lease, expected=None):
            observed = super()._verify_lease(lease, expected)
            self.checks += 1
            if self.checks == 2:
                raise OSError("injected lease ancestry loss")
            return observed

    lease_result = LeasePostconditionKernel(
        lease_repository, lease_backend, FakeRunner(lease_backend.context),
    ).execute(lease_scope, "attempt-1")
    assert lease_result.status == "failed"
    assert lease_result.unit_results[0].record_id is None
    assert lease_result.unit_results[0].evidence_id is None
    _assert_lease_released(lease_repository, "post-fault-probe")


def test_runner_ordinary_failure_preserves_later_units_and_exact_postconditions(tmp_path):
    scope, repository, backend = _case(tmp_path, n_windows=3)
    runner = FakeRunner(backend.context, failures=(0,))
    result = RollingExecutionKernel(repository, backend, runner).execute(
        scope, "attempt-1",
    )
    assert result.status == "failed"
    assert tuple(item.unit_key for item in result.unit_results) == scope.requested_unit_keys
    assert tuple(item.status for item in result.unit_results) == (
        "failed", "executed_success", "executed_success",
    )
    assert [item[0] for item in runner.calls] == list(scope.requested_unit_keys)
    assert backend.manifest_calls == list(scope.requested_unit_keys[1:])
    assert len(backend.candidates) == 2
    state = repository.inspect_readonly().inspection.snapshot
    assert state.phase == "failed"
    assert tuple(item.status for item in state.units) == ("failed", "success", "success")
    assert state.units[0].record_id is None
    assert state.units[0].evidence_id is None
    assert all(
        item.record_id is not None and item.evidence_id is not None
        for item in state.units[1:]
    )
    _assert_lease_released(repository, "post-runner-failure-probe")


@pytest.mark.parametrize((
    "point", "result_statuses", "runner_calls", "manifest_calls",
    "candidate_count", "state_phase", "state_statuses",
    "durable_record_count", "durable_evidence_count",
), (
    ("before_shared_lease", ("blocked", "blocked"), 0, 0, 0, None, (), 0, 0),
    ("after_lease_before_baseline_recheck", ("blocked", "blocked"), 0, 0, 0, None, (), 0, 0),
    ("after_prepared_state_commit", ("blocked", "blocked"), 0, 0, 0, "prepared", (), 0, 0),
    ("after_executing_state_commit", ("blocked", "blocked"), 0, 0, 0, "executing", ("pending", "pending"), 0, 0),
    ("before_runner", ("failed", "failed"), 0, 0, 0, "failed", ("failed", "failed"), 0, 0),
    ("after_recorder_before_manifest", ("failed", "failed"), 2, 0, 0, "failed", ("failed", "failed"), 0, 0),
    ("after_manifest_before_evidence_inventory", ("failed", "failed"), 2, 2, 2, "failed", ("failed", "failed"), 0, 0),
    ("after_valid_evidence_before_state_cas", ("failed", "failed"), 2, 2, 2, "failed", ("failed", "failed"), 0, 0),
    ("after_success_cas_before_next_unit", ("failed", "blocked"), 1, 1, 1, "executing", ("success", "pending"), 1, 1),
    ("after_final_unit_before_units_complete", ("failed", "failed"), 2, 2, 2, "executing", ("success", "success"), 2, 2),
))
def test_frozen_kernel_fault_points_never_manufacture_success(
        tmp_path, point, result_statuses, runner_calls, manifest_calls,
        candidate_count, state_phase, state_statuses,
        durable_record_count, durable_evidence_count):
    scope, repository, backend = _case(tmp_path, n_windows=2)
    runner = FakeRunner(backend.context)

    def fault(observed):
        if observed == point:
            raise OSError("injected %s" % point)

    result = RollingExecutionKernel(
        repository, backend, runner, fault_hook=fault,
    ).execute(scope, "attempt-1")
    assert result.status in ("blocked", "failed")
    assert result.requested_unit_keys == scope.requested_unit_keys
    assert tuple(item.unit_key for item in result.unit_results) == scope.requested_unit_keys
    assert tuple(item.status for item in result.unit_results) == result_statuses
    assert result.n_executed_success == 0
    assert len(runner.calls) == runner_calls
    assert [item[0] for item in runner.calls] == list(
        scope.requested_unit_keys[:runner_calls]
    )
    assert len(backend.manifest_calls) == manifest_calls
    assert backend.manifest_calls == list(scope.requested_unit_keys[:manifest_calls])
    assert len(backend.candidates) == candidate_count
    view = repository.inspect_readonly()
    if state_phase is None:
        assert view.inspection.classification == "missing"
    else:
        assert view.inspection.classification == "valid_versioned"
        state = view.inspection.snapshot
        assert state.phase == state_phase
        assert tuple(item.status for item in state.units) == state_statuses
        assert sum(item.record_id is not None for item in state.units) == durable_record_count
        assert sum(item.evidence_id is not None for item in state.units) == durable_evidence_count
        assert all(
            (item.record_id is not None) == (item.status == "success")
            and (item.evidence_id is not None) == (item.status == "success")
            for item in state.units
        )
    _assert_lease_released(repository, "post-matrix-probe")


def test_artifact_read_inventory_drift_and_cas_conflict_fail_closed(tmp_path):
    scope, repository, backend = _case(tmp_path, n_windows=2)

    class ArtifactReadFailureBackend(FakeExecutionBackend):
        def inspect(self, scope, requests):
            raise OSError("artifact read failed")

    read_backend = ArtifactReadFailureBackend(backend.context)
    read_runner = FakeRunner(backend.context)
    read_result = RollingExecutionKernel(
        repository, read_backend, read_runner,
    ).execute(scope, "attempt-1")
    assert tuple(item.status for item in read_result.unit_results) == (
        "failed", "failed",
    )
    assert tuple(item.unit_key for item in read_result.unit_results) == scope.requested_unit_keys
    assert len(read_runner.calls) == 2
    assert read_backend.manifest_calls == list(scope.requested_unit_keys)
    assert len(read_backend.candidates) == 2
    read_state = repository.inspect_readonly().inspection.snapshot
    assert read_state.phase == "failed"
    assert tuple(item.status for item in read_state.units) == ("failed", "failed")
    assert all(item.record_id is None and item.evidence_id is None for item in read_state.units)
    _assert_lease_released(repository, "post-artifact-read-fault-probe")

    drift_root = tmp_path / "inventory-drift"
    drift_root.mkdir()
    drift_scope, drift_repository, drift_backend = _case(drift_root, n_windows=2)

    class InventoryDriftBackend(FakeExecutionBackend):
        inventory_calls = 0

        def inventory(self, requests):
            observed = super().inventory(requests)
            self.inventory_calls += 1
            if self.inventory_calls % 2 == 0:
                observed = dict(observed)
                observed["fingerprint"] = "f" * 64
            return observed

    drifting = InventoryDriftBackend(drift_backend.context)
    drift_runner = FakeRunner(drifting.context)
    drift_result = RollingExecutionKernel(
        drift_repository, drifting, drift_runner,
    ).execute(drift_scope, "attempt-1")
    assert drift_result.status == "failed"
    assert drift_result.n_executed_success == 0
    assert tuple(item.unit_key for item in drift_result.unit_results) == drift_scope.requested_unit_keys
    assert tuple(item.status for item in drift_result.unit_results) == ("failed", "failed")
    assert [item[0] for item in drift_runner.calls] == list(drift_scope.requested_unit_keys)
    assert drifting.manifest_calls == list(drift_scope.requested_unit_keys)
    assert len(drifting.candidates) == 2
    drift_state = drift_repository.inspect_readonly().inspection.snapshot
    assert drift_state.phase == "failed"
    assert tuple(item.status for item in drift_state.units) == ("failed", "failed")
    assert all(item.record_id is None and item.evidence_id is None for item in drift_state.units)
    _assert_lease_released(drift_repository, "post-inventory-drift-probe")

    conflict_root = tmp_path / "cas-conflict"
    conflict_root.mkdir()
    conflict_scope, _conflict_repository, conflict_backend = _case(
        conflict_root, n_windows=2,
    )

    class EvidenceCasConflictRepository(RollingStateRepository):
        def commit_evidence_authorized(
            self, proposed, expected, evidence_set, blocking=True,
        ):
            return self._receipt(
                "transition", "conflict", before=expected,
                before_phase="executing", did_write=False,
            )

    conflict_repository = EvidenceCasConflictRepository.for_workspace(
        conflict_backend.context, "rolling",
    )
    conflict_runner = FakeRunner(conflict_backend.context)
    conflict_result = RollingExecutionKernel(
        conflict_repository, conflict_backend, conflict_runner,
    ).execute(conflict_scope, "attempt-1")
    assert conflict_result.status == "failed"
    assert conflict_result.n_executed_success == 0
    assert tuple(item.unit_key for item in conflict_result.unit_results) == conflict_scope.requested_unit_keys
    assert tuple(item.status for item in conflict_result.unit_results) == ("failed", "failed")
    assert [item[0] for item in conflict_runner.calls] == list(
        conflict_scope.requested_unit_keys
    )
    assert conflict_backend.manifest_calls == list(conflict_scope.requested_unit_keys)
    conflict_state = conflict_repository.inspect_readonly().inspection.snapshot
    assert conflict_state.phase == "failed"
    assert tuple(item.status for item in conflict_state.units) == ("failed", "failed")
    assert all(
        item.record_id is None and item.evidence_id is None
        for item in conflict_state.units
    )
    assert len(conflict_backend.candidates) == 2
    _assert_lease_released(conflict_repository, "post-cas-conflict-probe")


def test_state_recorder_manifest_and_temp_symlink_escape_fail_before_execution(tmp_path):
    root = (tmp_path / "workspace").resolve()
    outside = (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)
    for name in ("config", "mlruns", "output"):
        (root / name).mkdir()
    context = WorkspaceContext.from_root(root)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    with pytest.raises(Exception):
        repository.inspect_readonly()

    foreign_root = tmp_path / "foreign-backend-case"
    foreign_root.mkdir()
    scope, repository, backend = _case(foreign_root)

    class ForeignBackend(FakeExecutionBackend):
        def tracking_identity(self):
            facts = super().tracking_identity()
            facts.update({"contained": False, "foreign": True})
            return facts

    runner = FakeRunner(backend.context)
    result = RollingExecutionKernel(
        repository, ForeignBackend(backend.context), runner,
    ).execute(scope, "attempt-1")
    assert result.status == "blocked"
    assert runner.calls == []
    assert repository.inspect_readonly().inspection.classification == "missing"

    artifact_parent = tmp_path / "artifact-parent"
    artifact_parent.mkdir()
    escaped = artifact_parent / "escaped"
    escaped.symlink_to(outside, target_is_directory=True)
    from quantpits.rolling.mlflow_execution_backend import _local_artifact_root
    with pytest.raises(Exception):
        _local_artifact_root(escaped.as_uri(), artifact_parent)
