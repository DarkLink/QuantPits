"""Deterministic fakes for immutable Rolling aggregate candidate contracts."""

from pathlib import Path

from quantpits.rolling import (
    RollingExecutionKernel,
    RollingStateRepository,
    build_rolling_aggregate_scope,
)
from quantpits.rolling.aggregate import (
    _candidate_manifest_contract_fingerprint,
)
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value

from tests.quantpits.rolling.execution_support import (
    FakeExecutionBackend,
    FakeRunner,
    linear_capability_result,
    make_scope,
)


class FakeCandidateBackend:
    def __init__(self, context):
        self.context = context
        self.candidates = {}
        self.calls = []
        self.controls = {}
        self.fault_point = None
        self.experiment_present = False

    def _fault(self, point):
        if self.fault_point == point:
            raise RuntimeError(point)

    def inventory(self, aggregate_scope):
        rows = tuple(self.candidates.values())
        return {
            "raw_count": len(rows),
            "fingerprint": fingerprint_value(rows),
            "candidates": rows,
            "experiment_present": self.experiment_present,
        }

    def protected_snapshot(self, aggregate_scope):
        return fingerprint_value({
            "state": aggregate_scope.state_baseline_fingerprint,
            "current": "missing",
        })

    def backend_identity(self, aggregate_scope):
        return fingerprint_value({
            "workspace": self.workspace_identity(aggregate_scope),
            "backend": "fake-candidate",
        })

    def workspace_identity(self, aggregate_scope):
        from quantpits.rolling.identity import workspace_fingerprint
        return workspace_fingerprint(self.context.root)

    def with_candidate_lock(
        self, aggregate_scope, callback, create_if_missing=False,
    ):
        self._fault("under_terminal_candidate_lock")
        return callback()

    def inspect_candidate(
        self, aggregate_scope, target_key, candidate_key,
        expected_manifest_contract_fingerprint,
    ):
        observation = dict(
            self.candidates.get(
                candidate_key, {"classification": "missing"},
            )
        )
        if (
            observation.get("classification") == "valid"
            and observation.get("manifest_contract_fingerprint")
            != expected_manifest_contract_fingerprint
        ):
            return {"classification": "identity_mismatch"}
        return observation

    def create_candidate(
        self, aggregate_scope, target_key, candidate_key,
        prediction_bytes, manifest,
    ):
        position = aggregate_scope.target_keys.index(target_key)
        self.calls.append((target_key, candidate_key))
        if position in self.controls:
            raise self.controls[position]
        self._fault("before_candidate_namespace")
        self.experiment_present = True
        self._fault("after_candidate_experiment_namespace")
        partial = {
            "classification": "partial",
            "candidate_key": candidate_key,
            "target_key": target_key,
            "scope_fingerprint": aggregate_scope.scope_fingerprint,
            "aggregate_attempt_id": aggregate_scope.aggregate_attempt_id,
            "recorder_id": "aggregate-recorder-%d" % position,
        }
        self.candidates[candidate_key] = partial
        self._fault("after_candidate_namespace")
        self._fault("after_candidate_tags")
        self._fault("after_candidate_prediction")
        self._fault("after_candidate_manifest")
        observation = {
            "classification": "valid",
            "candidate_key": candidate_key,
            "target_key": target_key,
            "scope_fingerprint": aggregate_scope.scope_fingerprint,
            "aggregate_attempt_id": aggregate_scope.aggregate_attempt_id,
            "recorder_id": "aggregate-recorder-%d" % position,
            "manifest_fingerprint": fingerprint_value(manifest),
            "content_fingerprint": manifest["content_fingerprint"],
            "row_count": manifest["row_count"],
            "manifest_contract_fingerprint":
                _candidate_manifest_contract_fingerprint(manifest),
            "run_status": "FINISHED",
            "lifecycle_stage": "active",
            "prediction_bytes": prediction_bytes,
            "manifest": dict(manifest),
        }
        self.candidates[candidate_key] = observation
        self._fault("after_candidate_reinspection")
        return dict(observation)


def aggregate_case(tmp_path, n_targets=1, n_windows=2):
    root = (tmp_path / "Demo_Workspace").resolve()
    for name in ("config", "data", "mlruns", "output"):
        (root / name).mkdir(parents=True, exist_ok=True)
    context = WorkspaceContext.from_root(root)
    scope = make_scope(
        context, linear_capability_result(), n_targets, n_windows,
    )
    repository = RollingStateRepository.for_workspace(context, "rolling")
    source = FakeExecutionBackend(context)
    execution = RollingExecutionKernel(
        repository, source, FakeRunner(context),
    ).execute(scope, "execution-attempt")
    assert execution.status == "success"
    view = repository.inspect_readonly()
    aggregate = build_rolling_aggregate_scope(
        scope, view, "aggregate-attempt",
    )
    return context, scope, repository, source, aggregate
