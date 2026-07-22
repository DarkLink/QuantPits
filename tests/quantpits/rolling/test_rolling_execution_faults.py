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


@pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_keyboard_systemexit_generatorexit_propagate_and_leave_reconcilable_state(tmp_path, control):
    scope, repository, backend = _case(tmp_path)
    with pytest.raises(control.__class__):
        RollingExecutionKernel(
            repository, backend, FakeRunner(backend.context, controls={0: control}),
        ).execute(scope, "attempt-1")
    state = repository.inspect_readonly().inspection.snapshot
    assert state.phase == "failed"
    assert state.units[0].status == "failed"
    assert state.units[0].record_id is None
    assert state.units[0].evidence_id is None
    from quantpits.training.lease import TrainingExecutionLease
    lease = TrainingExecutionLease.for_workspace(repository.context)
    lease.acquire(run_id="post-interrupt-probe")
    lease.release()


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
    assert manifest_repository.inspect_readonly().inspection.snapshot.phase == "failed"

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
    from quantpits.training.lease import TrainingExecutionLease
    lease = TrainingExecutionLease.for_workspace(lease_repository.context)
    lease.acquire(run_id="post-fault-probe")
    lease.release()


@pytest.mark.parametrize("point", (
    "before_shared_lease",
    "after_lease_before_baseline_recheck",
    "after_prepared_state_commit",
    "after_executing_state_commit",
    "before_runner",
    "after_recorder_before_manifest",
    "after_manifest_before_evidence_inventory",
    "after_valid_evidence_before_state_cas",
    "after_success_cas_before_next_unit",
    "after_final_unit_before_units_complete",
))
def test_frozen_kernel_fault_points_never_manufacture_success(tmp_path, point):
    scope, repository, backend = _case(tmp_path)
    runner = FakeRunner(backend.context)

    def fault(observed):
        if observed == point:
            raise OSError("injected %s" % point)

    result = RollingExecutionKernel(
        repository, backend, runner, fault_hook=fault,
    ).execute(scope, "attempt-1")
    assert result.status in ("blocked", "failed")
    assert result.n_executed_success == 0
    view = repository.inspect_readonly()
    if view.inspection.classification == "valid_versioned":
        state = view.inspection.snapshot
        assert state.phase in ("prepared", "executing", "failed")
        if state.units:
            assert all(item.record_id is None or item.status == "success" for item in state.units)
    assert runner.calls == [] if point in (
        "before_shared_lease", "after_lease_before_baseline_recheck",
        "after_prepared_state_commit", "after_executing_state_commit",
        "before_runner",
    ) else runner.calls
    from quantpits.training.lease import TrainingExecutionLease
    lease = TrainingExecutionLease.for_workspace(repository.context)
    lease.acquire(run_id="post-matrix-probe")
    lease.release()


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
    assert len(read_runner.calls) == 2

    drift_root = tmp_path / "inventory-drift"
    drift_root.mkdir()
    drift_scope, drift_repository, drift_backend = _case(drift_root)

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
    drift_result = RollingExecutionKernel(
        drift_repository, drifting, FakeRunner(drifting.context),
    ).execute(drift_scope, "attempt-1")
    assert drift_result.status == "failed"
    assert drift_result.n_executed_success == 0

    conflict_root = tmp_path / "cas-conflict"
    conflict_root.mkdir()
    conflict_scope, _conflict_repository, conflict_backend = _case(conflict_root)

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
    conflict_result = RollingExecutionKernel(
        conflict_repository, conflict_backend, FakeRunner(conflict_backend.context),
    ).execute(conflict_scope, "attempt-1")
    assert conflict_result.status == "failed"
    assert conflict_result.n_executed_success == 0
    conflict_state = conflict_repository.inspect_readonly().inspection.snapshot
    assert conflict_state.phase == "failed"
    assert conflict_state.units[0].status == "failed"
    assert conflict_state.units[0].record_id is None
    assert conflict_state.units[0].evidence_id is None


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
