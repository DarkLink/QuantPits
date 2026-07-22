"""Permanent deterministic fakes for exact Rolling execution contracts."""

import hashlib
import io
from functools import lru_cache
from pathlib import Path

import pandas as pd

from quantpits.rolling import (
    PreparedRollingRun,
    ResolvedRollingRun,
    RollingAnchorPolicy,
    RollingArtifactExpectation,
    RollingExecutionTargetDescriptor,
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingTarget,
    RollingRunOptions,
    RollingUnitEvidenceRequest,
    RollingUnitRunnerObservation,
    RollingWindowDescriptor,
    RollingWindowIdentity,
    build_rolling_execution_scope,
    inspect_rolling_evidence,
    map_workflow_capability,
    observe_rolling_business_sessions,
    workspace_fingerprint,
)
from quantpits.utils.workspace import fingerprint_value
from quantpits.runtime.command import CommandPlan, fingerprint_command_plan


RUNTIME_PARAMS = {"market": "csi300", "benchmark": "SH000300"}


@lru_cache(maxsize=1)
def linear_capability_matrix():
    from quantpits.model_capabilities.catalog import AUTHORITATIVE_CATALOG
    from quantpits.model_capabilities.inspector import ModelCapabilityInspector

    declaration = next(
        item for item in AUTHORITATIVE_CATALOG
        if item.model_module == "qlib.contrib.model.linear"
        and item.model_class == "LinearModel"
        and item.dataset_module == "qlib.data.dataset"
        and item.dataset_class == "DatasetH"
        and item.action == "train"
        and item.execution_family == "rolling"
    )
    unavailable = next(
        item for item in AUTHORITATIVE_CATALOG
        if item.model_module == declaration.model_module
        and item.model_class == declaration.model_class
        and item.dataset_module == declaration.dataset_module
        and item.dataset_class == declaration.dataset_class
        and item.dataset_protocol == declaration.dataset_protocol
        and item.action == "predict_only"
        and item.execution_family == declaration.execution_family
        and item.processor_profile == declaration.processor_profile
        and item.artifact_protocol == declaration.artifact_protocol
        and item.dependency_profile == declaration.dependency_profile
    )
    matrix = ModelCapabilityInspector().inspect((declaration, unavailable))
    assert matrix.results[0].preflight_allowed
    assert not matrix.results[1].preflight_allowed
    return matrix


def linear_capability_result():
    return linear_capability_matrix().results[0]


def prediction_bytes(sessions):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(session), "SYNTH") for session in sessions],
        names=("datetime", "instrument"),
    )
    frame = pd.DataFrame(
        {"score": [float(index + 1) for index in range(len(sessions))]},
        index=index,
    )
    buffer = io.BytesIO()
    frame.to_pickle(buffer)
    return buffer.getvalue()


def make_scope(context, capability_result, n_targets=1, n_windows=1):
    targets = []
    prepared_targets = []
    for index in range(n_targets):
        target = RollingTargetIdentity("linear-%s" % index, "rolling")
        relative = "config/linear-%s.yaml" % index
        workflow = context.root / relative
        workflow.parent.mkdir(parents=True, exist_ok=True)
        workflow.write_text(
            "task:\n"
            "  model: {class: LinearModel, module_path: qlib.contrib.model.linear}\n"
            "  dataset:\n"
            "    class: DatasetH\n"
            "    module_path: qlib.data.dataset\n"
            "    kwargs:\n"
            "      handler:\n"
            "        class: Alpha158\n"
            "        module_path: qlib.contrib.data.handler\n"
            "        kwargs: {}\n"
            "      segments: {}\n",
            encoding="utf-8",
        )
        targets.append(map_workflow_capability(
            context, target.target_key, relative, linear_capability_matrix(),
        ))
        prepared_targets.append(RollingTarget(
            target, relative, targets[-1].workflow_fingerprint,
            "test_fixture", {},
        ))
    windows = []
    for index in range(n_windows):
        month = index + 1
        test_start = "2026-%02d-05" % month
        test_end = "2026-%02d-06" % month
        identity = RollingWindowIdentity(
            family="rolling", train_start="2024-01-01", train_end="2024-12-31",
            valid_start="2025-01-01", valid_end="2025-12-31",
            test_start=test_start, test_end=test_end,
            effective_config_fingerprint="a" * 64,
        )
        descriptor = RollingWindowDescriptor(index, identity, {
            "window_idx": index,
            "train_start": identity.train_start, "train_end": identity.train_end,
            "valid_start": identity.valid_start, "valid_end": identity.valid_end,
            "test_start": identity.test_start, "test_end": identity.test_end,
        })
        windows.append(descriptor)
    resolved_windows = tuple(windows)
    plan = CommandPlan(
        "rolling_train", context.root.name, "test-fixture", mode="rolling:merge",
    )
    plan_fingerprint = fingerprint_command_plan(plan, length=64)
    prepared = PreparedRollingRun(
        context, RollingRunOptions(action="merge"), (), {},
        tuple(prepared_targets), None, RollingAnchorPolicy("test_fixture"),
        plan, plan_fingerprint, {},
    )
    resolved_identity = RollingRunIdentity(
        workspace_fingerprint=workspace_fingerprint(context.root),
        family="rolling", action="merge", plan_fingerprint=plan_fingerprint,
        config_fingerprint=fingerprint_value({}), anchor_date=windows[-1].identity.test_end,
        target_keys=tuple(item.target_key for item in targets),
        window_keys=tuple(item.window_key for item in resolved_windows),
        runtime_params_fingerprint=fingerprint_value(RUNTIME_PARAMS),
    )
    resolved = ResolvedRollingRun(
        prepared, resolved_identity.anchor_date, dict(RUNTIME_PARAMS),
        resolved_windows, resolved_identity,
    )
    observed_windows = observe_rolling_business_sessions(
        resolved_windows, lambda start, end: (start, end),
    )
    selected = tuple(item.window_key for item in observed_windows)
    return build_rolling_execution_scope(
        prepared, resolved, selected, tuple(targets), observed_windows,
    )


class FakeRunner:
    def __init__(self, context, runtime_params=None, failures=(), controls=None, timeline=None):
        self.context = context
        self.runtime_params = dict(runtime_params or RUNTIME_PARAMS)
        self.failures = set(failures)
        self.controls = dict(controls or {})
        self.calls = []
        self.timeline = timeline

    @property
    def runtime_params_fingerprint(self):
        return fingerprint_value(self.runtime_params)

    def execute(self, scope, unit, attempt_id):
        self.calls.append((unit.unit_key, attempt_id))
        if self.timeline is not None:
            self.timeline.append("runner:%s" % unit.position)
        if unit.position in self.controls:
            raise self.controls[unit.position]
        if unit.position in self.failures:
            return RollingUnitRunnerObservation(
                unit.unit_key, attempt_id, "failed",
                failure_code="injected_runner_failure",
            )
        return RollingUnitRunnerObservation(
            unit.unit_key, attempt_id, "candidate_success",
            "exact-unit-experiment", "experiment-1",
            "recorder-%s-%s" % (attempt_id, unit.position),
        )


class FakeExecutionBackend:
    def __init__(self, context, timeline=None):
        self.context = context
        self.requests = {}
        self.candidates = {}
        self.manifest_calls = []
        self.current_lookup_calls = 0
        self.timeline = timeline

    @property
    def backend_fingerprint(self):
        return fingerprint_value({"tracking_uri": str(self.context.mlflow_uri)})

    @staticmethod
    def calendar_sessions(start, end):
        return (start, end)

    def tracking_identity(self):
        return {
            "workspace_fingerprint": workspace_fingerprint(self.context.root),
            "backend_fingerprint": self.backend_fingerprint,
            "present": True, "contained": True, "foreign": False,
        }

    def capture_recorder_inventory(self, scope, unit, attempt_id):
        return frozenset(self.candidates)

    def commit_execution_manifest(self, scope, unit, observation, recorder_baseline):
        if recorder_baseline != frozenset(self.candidates):
            raise RuntimeError("fake recorder inventory baseline drifted")
        self.manifest_calls.append(unit.unit_key)
        if self.timeline is not None:
            self.timeline.append("manifest:%s" % unit.position)
        root = (
            self.context.root / "mlruns" / observation.experiment_id
            / observation.recorder_id / "artifacts"
        )
        root.mkdir(parents=True, exist_ok=True)
        pred = prediction_bytes(unit.window.expected_sessions)
        model = b"bounded-model"
        manifest = ("manifest:%s:%s" % (scope.scope_fingerprint, unit.position)).encode("ascii")
        values = (
            ("execution_manifest.json", "supporting", manifest),
            ("model.pkl", "supporting", model),
            ("pred.pkl", "prediction", pred),
        )
        expectations = []
        for key, role, data in values:
            (root / key).write_bytes(data)
            expectations.append(RollingArtifactExpectation(
                key, role, len(data), hashlib.sha256(data).hexdigest(),
            ))
        run = RollingRunIdentity(
            workspace_fingerprint=scope.run_identity.workspace_fingerprint,
            family=scope.run_identity.family, action=scope.run_identity.action,
            plan_fingerprint=scope.run_identity.plan_fingerprint,
            config_fingerprint=scope.run_identity.config_fingerprint,
            anchor_date=scope.run_identity.anchor_date,
            target_keys=scope.run_identity.target_keys,
            window_keys=scope.run_identity.window_keys,
            runtime_params_fingerprint=scope.run_identity.runtime_params_fingerprint,
            attempt_id=observation.attempt_id,
        )
        request = RollingUnitEvidenceRequest(
            run, unit.unit_key[0], unit.window.identity, "execution_bound_v1",
            unit.unit_key[0], scope.run_identity.action,
            observation.experiment_name, observation.experiment_id,
            observation.recorder_id, tuple(expectations),
            unit.window.expected_sessions,
        )
        self.requests[unit.unit_key] = request
        self.candidates[unit.unit_key] = {
            "workspace_fingerprint": run.workspace_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "experiment_name": request.experiment_name,
            "experiment_id": request.experiment_id,
            "recorder_id": request.recorder_id,
            "run_fingerprint": run.fingerprint,
            "attempt_id": run.attempt_id,
            "plan_fingerprint": run.plan_fingerprint,
            "config_fingerprint": run.config_fingerprint,
            "target_key": request.target_key,
            "window_key": request.window_key,
            "source_protocol": request.source_protocol,
            "source_publication_key": request.source_publication_key,
            "source_operation": request.source_operation,
            "source_manifest_fingerprint": request.source_manifest_fingerprint,
            "artifact_root_uri": root.resolve().as_uri(),
        }
        if tuple(key for key in self.candidates if key not in recorder_baseline) != (unit.unit_key,):
            raise RuntimeError("fake runner did not create exactly one recorder")
        return request

    def _placeholder(self, scope, unit, state):
        run = RollingRunIdentity(
            workspace_fingerprint=scope.run_identity.workspace_fingerprint,
            family=scope.run_identity.family, action=scope.run_identity.action,
            plan_fingerprint=scope.run_identity.plan_fingerprint,
            config_fingerprint=scope.run_identity.config_fingerprint,
            anchor_date=scope.run_identity.anchor_date,
            target_keys=scope.run_identity.target_keys,
            window_keys=scope.run_identity.window_keys,
            runtime_params_fingerprint=scope.run_identity.runtime_params_fingerprint,
            attempt_id=state.attempt_id,
        )
        return RollingUnitEvidenceRequest(
            run, unit.unit_key[0], unit.window.identity, "execution_bound_v1",
            unit.unit_key[0], scope.run_identity.action,
            "missing-experiment", "missing-experiment-id",
            "missing-recorder-%s" % unit.position,
            (RollingArtifactExpectation("pred.pkl", "prediction", 0, "0" * 64),),
            unit.window.expected_sessions,
        )

    def requests_for_state(self, scope, state):
        return tuple(
            self.requests.get(unit.unit_key) or self._placeholder(scope, unit, state)
            for unit in scope.units
        )

    def inventory(self, requests):
        candidates = tuple(
            self.candidates[request.unit_key]
            for request in requests if request.unit_key in self.candidates
        )
        return {"fingerprint": fingerprint_value(candidates), "candidates": candidates}

    def inspect(self, scope, requests):
        if self.timeline is not None:
            self.timeline.append("inspect:%s" % len(requests))
        return inspect_rolling_evidence(self.context, requests, self)
