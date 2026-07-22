"""Production-shaped LinearModel/DatasetH/Slide exact-unit runner."""

from __future__ import annotations

import hashlib
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from quantpits.rolling.errors import (
    RollingExecutionContractError,
    RollingExecutionPreflightError,
)
from quantpits.rolling.execution import (
    RollingExecutionScope,
    RollingExecutionUnit,
    RollingUnitRunnerObservation,
)
from quantpits.utils.workspace import WorkspaceContext
from quantpits.utils.workspace import fingerprint_value
from quantpits.rolling.identity import (
    RollingTargetIdentity,
    RollingWindowIdentity,
    workspace_fingerprint,
)


_EXACT_IDENTITY = {
    "model_module": "qlib.contrib.model.linear",
    "model_class": "LinearModel",
    "wrapper_kind": "external_passthrough",
    "dataset_module": "qlib.data.dataset",
    "dataset_class": "DatasetH",
    "dataset_protocol": "point_in_time",
    "action": "train",
    "execution_family": "rolling",
    "processor_profile": "standard_infer_no_label_drop",
    "artifact_protocol": "qlib_recorder_model_v1",
    "dependency_profile": "python_qlib",
}
_WORKER_MARKER = "QUANTPITS_ROLLING_UNIT="


def _public_worker_text(value, field_name):
    if (
        not isinstance(value, str) or not value or value != value.strip()
        or value.startswith(("/", "\\")) or "://" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise RollingExecutionContractError(
            "%s is not a public worker identifier" % field_name
        )
    return value


def _worker_payload(payload):
    """Execute a validated child payload; returned facts contain no local paths."""

    if not isinstance(payload, dict):
        raise RollingExecutionContractError("unit worker payload must be a mapping")
    required = {
        "workspace_root", "workspace_fingerprint", "mlflow_uri",
        "qlib_data_dir", "qlib_region", "workflow_relative_path",
        "workflow_fingerprint", "target_key", "window", "window_key",
        "run_fingerprint", "source_operation", "attempt_id",
        "experiment_name", "runtime_params",
    }
    if set(payload) != required:
        raise RollingExecutionContractError("unit worker payload fields are not exact")
    context = WorkspaceContext.from_root(
        payload["workspace_root"], mlflow_uri=payload["mlflow_uri"],
        qlib_data_dir=payload["qlib_data_dir"], qlib_region=payload["qlib_region"],
    )
    if workspace_fingerprint(context.root) != payload["workspace_fingerprint"]:
        raise RollingExecutionPreflightError("child workspace identity changed")
    target = RollingTargetIdentity.parse(payload["target_key"])
    if target.family != "rolling":
        raise RollingExecutionPreflightError("child target is not Slide rolling")
    if not isinstance(payload["window"], dict):
        raise RollingExecutionContractError("child window must be a mapping")
    window_values = dict(payload["window"])
    folds = window_values.pop("folds", ())
    if folds not in ((), []):
        raise RollingExecutionPreflightError("child runner does not support CPCV folds")
    window = RollingWindowIdentity(folds=(), **window_values)
    if window.family != "rolling" or window.window_key != payload["window_key"]:
        raise RollingExecutionPreflightError("child window identity changed")
    for field_name in (
        "run_fingerprint", "source_operation", "attempt_id", "experiment_name",
    ):
        _public_worker_text(payload[field_name], field_name)
    relative = Path(payload["workflow_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RollingExecutionPreflightError("child workflow path is not canonical")
    root = context.root.resolve(strict=True)
    workflow = (root / relative).absolute()
    try:
        physical = workflow.resolve(strict=True)
        physical.relative_to(root)
    except (OSError, ValueError):
        raise RollingExecutionPreflightError("child workflow is not physically contained")
    workflow_bytes = physical.read_bytes()
    if hashlib.sha256(workflow_bytes).hexdigest() != payload["workflow_fingerprint"]:
        raise RollingExecutionPreflightError("child workflow fingerprint changed")
    params = payload["runtime_params"]
    if not isinstance(params, dict) or any(name not in params for name in ("market", "benchmark")):
        raise RollingExecutionPreflightError("child runtime params are incomplete")
    task = LinearSlideUnitRunner._inject_exact_config(workflow_bytes, params, window)
    if (
        task["model"].get("module_path") != _EXACT_IDENTITY["model_module"]
        or task["model"].get("class") != _EXACT_IDENTITY["model_class"]
        or task["dataset"].get("module_path") != _EXACT_IDENTITY["dataset_module"]
        or task["dataset"].get("class") != _EXACT_IDENTITY["dataset_class"]
    ):
        raise RollingExecutionPreflightError("child task identity changed")

    import qlib
    from qlib.constant import REG_CN, REG_US
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    region = {"cn": REG_CN, "us": REG_US}.get(str(context.qlib_region).lower(), REG_CN)
    qlib.init(
        provider_uri=context.qlib_data_dir, region=region,
        exp_manager={
            "class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": context.mlflow_uri,
                "default_exp_name": "Experiment",
            },
        },
    )
    from quantpits.rolling.mlflow_execution_backend import _tracking_uri_identity
    backend_present, backend_contained = _tracking_uri_identity(
        R.get_uri(), context.root,
    )
    if (
        str(R.get_uri()) != str(context.mlflow_uri)
        or not backend_present or not backend_contained
    ):
        raise RollingExecutionPreflightError("child tracking backend identity changed")
    with R.start(experiment_name=payload["experiment_name"]):
        recorder = R.get_recorder()
        from quantpits.rolling.mlflow_execution_backend import _local_artifact_root
        artifact_root = _local_artifact_root(
            recorder.get_artifact_uri(), context.root,
        )
        R.set_tags(
            execution_protocol="rolling_execution_v1",
            run_fingerprint=payload["run_fingerprint"],
            attempt_id=payload["attempt_id"], target_key=payload["target_key"],
            window_key=payload["window_key"],
            source_operation=payload["source_operation"],
        )
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        model.fit(dataset=dataset)
        prediction = model.predict(dataset=dataset, segment="test")
        recorder.save_objects(**{"model.pkl": model, "pred.pkl": prediction})
        if _local_artifact_root(recorder.get_artifact_uri(), context.root) != artifact_root:
            raise RollingExecutionPreflightError("child recorder artifact root changed")
        recorder_id = recorder.info["id"]
        experiment_id = str(
            R.get_exp(experiment_name=payload["experiment_name"], create=False).id
        )
    return {
        "status": "candidate_success", "experiment_name": payload["experiment_name"],
        "experiment_id": experiment_id, "recorder_id": recorder_id,
    }


def _worker_main(request_path):
    try:
        payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
        result = _worker_payload(payload)
    except (KeyboardInterrupt, SystemExit, GeneratorExit) as exc:
        print(_WORKER_MARKER + json.dumps({
            "process_control": exc.__class__.__name__,
        }, sort_keys=True))
        return {"KeyboardInterrupt": 130, "SystemExit": 131, "GeneratorExit": 132}[
            exc.__class__.__name__
        ]
    except BaseException as exc:
        result = {
            "status": "failed",
            "failure_code": "unit_worker_%s" % exc.__class__.__name__,
        }
    print(_WORKER_MARKER + json.dumps(result, sort_keys=True))
    return 0


class LinearSlideUnitRunner:
    """Execute one exact unit without records, backtest, stitching or publication."""

    def __init__(
        self, context, runtime_params, experiment_name, budget_guard=None,
        timeout_seconds=3600,
    ):
        if not isinstance(context, WorkspaceContext):
            raise RollingExecutionContractError("runner requires WorkspaceContext")
        if not isinstance(runtime_params, dict):
            raise RollingExecutionContractError("runtime_params must be an explicit mapping")
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise RollingExecutionContractError("experiment_name must be explicit")
        if budget_guard is not None and not callable(budget_guard):
            raise RollingExecutionContractError("budget_guard must be callable")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise RollingExecutionContractError("timeout_seconds must be a positive integer")
        self.context = context
        self.runtime_params = dict(runtime_params)
        self.experiment_name = experiment_name.strip()
        self.budget_guard = budget_guard
        self.timeout_seconds = timeout_seconds

    @property
    def runtime_params_fingerprint(self):
        return fingerprint_value(self.runtime_params)

    def _workflow(self, unit):
        root = Path(self.context.root).resolve(strict=True)
        path = (root / unit.target.workflow_relative_path).absolute()
        try:
            physical = path.resolve(strict=True)
            physical.relative_to(root)
        except (OSError, ValueError):
            raise RollingExecutionPreflightError("workflow path is not physically contained")
        data = physical.read_bytes()
        if hashlib.sha256(data).hexdigest() != unit.target.workflow_fingerprint:
            raise RollingExecutionPreflightError("workflow fingerprint changed before execution")
        return physical, data

    @staticmethod
    def _inject_exact_config(data, params, window):
        """Inject only model/dataset dates; never import legacy path globals."""

        try:
            import yaml
            config = yaml.safe_load(data)
            task = copy.deepcopy(config["task"])
            dataset_kwargs = task["dataset"]["kwargs"]
            handler_kwargs = dataset_kwargs["handler"]["kwargs"]
            segments = dataset_kwargs["segments"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RollingExecutionPreflightError(
                "workflow lacks exact Linear/DatasetH runtime configuration"
            )
        handler_kwargs.update({
            "start_time": window.train_start,
            "end_time": window.test_end,
            "fit_start_time": window.train_start,
            "fit_end_time": window.train_end,
            "instruments": params["market"],
        })
        if "label_formula" in params:
            handler_kwargs["label"] = [params["label_formula"]]
        segments.update({
            "train": [window.train_start, window.train_end],
            "valid": [window.valid_start, window.valid_end],
            "test": [window.test_start, window.test_end],
        })
        return task

    @staticmethod
    def _validate_identity(unit):
        if unit.target.capability_identity.to_public_dict() != _EXACT_IDENTITY:
            raise RollingExecutionPreflightError("unit runner only supports the exact Linear rolling row")
        if not unit.target.capability_result.preflight_allowed:
            raise RollingExecutionPreflightError("unit capability lost execution authority")

    def execute(self, scope, unit, attempt_id):
        if not isinstance(scope, RollingExecutionScope) or not isinstance(unit, RollingExecutionUnit):
            raise RollingExecutionContractError("runner requires typed scope/unit")
        if unit.unit_key not in scope.requested_unit_keys:
            raise RollingExecutionContractError("runner unit is outside scope")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise RollingExecutionContractError("runner attempt_id is invalid")
        if self.runtime_params_fingerprint != scope.runtime_binding.runtime_params_fingerprint:
            raise RollingExecutionPreflightError(
                "runner runtime params disagree with execution scope"
            )
        self._validate_identity(unit)
        if self.budget_guard is not None:
            self.budget_guard(scope, unit)
        _workflow, workflow_bytes = self._workflow(unit)
        params = dict(self.runtime_params)
        window = unit.window.identity
        params.update({
            "start_time": window.train_start,
            "end_time": window.test_end,
            "fit_start_time": window.train_start,
            "fit_end_time": window.train_end,
            "valid_start_time": window.valid_start,
            "valid_end_time": window.valid_end,
            "test_start_time": window.test_start,
            "test_end_time": window.test_end,
        })
        required = ("market", "benchmark")
        if any(name not in params for name in required):
            raise RollingExecutionPreflightError("runner runtime params are incomplete")
        task = self._inject_exact_config(workflow_bytes, params, window)
        if (
            task["model"].get("module_path") != _EXACT_IDENTITY["model_module"]
            or task["model"].get("class") != _EXACT_IDENTITY["model_class"]
            or task["dataset"].get("module_path") != _EXACT_IDENTITY["dataset_module"]
            or task["dataset"].get("class") != _EXACT_IDENTITY["dataset_class"]
        ):
            raise RollingExecutionPreflightError("injected task identity changed")
        payload = {
            "workspace_root": str(self.context.root),
            "workspace_fingerprint": workspace_fingerprint(self.context.root),
            "mlflow_uri": self.context.mlflow_uri,
            "qlib_data_dir": str(self.context.qlib_data_dir),
            "qlib_region": self.context.qlib_region,
            "workflow_relative_path": unit.target.workflow_relative_path,
            "workflow_fingerprint": unit.target.workflow_fingerprint,
            "target_key": unit.unit_key[0],
            "window": {
                "family": window.family,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "valid_start": window.valid_start,
                "valid_end": window.valid_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "effective_config_fingerprint": window.effective_config_fingerprint,
                "folds": [],
            },
            "window_key": unit.unit_key[1],
            "run_fingerprint": scope.run_identity.fingerprint,
            "source_operation": scope.run_identity.action,
            "attempt_id": attempt_id,
            "experiment_name": self.experiment_name,
            "runtime_params": params,
        }
        with tempfile.TemporaryDirectory(prefix="quantpits-rolling-unit-") as temp_dir:
            request = Path(temp_dir) / "request.json"
            request_bytes = json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            descriptor = os.open(
                str(request), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
            )
            try:
                offset = 0
                while offset < len(request_bytes):
                    offset += os.write(descriptor, request_bytes[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            environment = dict(os.environ)
            environment.update({
                "QLIB_WORKSPACE_DIR": str(self.context.root),
                "QLIB_DATA_DIR": str(self.context.qlib_data_dir),
                "QLIB_REGION": str(self.context.qlib_region),
                "MLFLOW_TRACKING_URI": str(self.context.mlflow_uri),
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": temp_dir,
                "TMPDIR": temp_dir,
                "XDG_CACHE_HOME": str(Path(temp_dir) / "cache"),
                "MPLCONFIGDIR": str(Path(temp_dir) / "matplotlib"),
            })
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "quantpits.rolling.unit_runner", str(request)],
                    cwd=str(self.context.root), env=environment,
                    capture_output=True, text=True, check=False,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return RollingUnitRunnerObservation(
                    unit.unit_key, attempt_id, "failed",
                    failure_code="unit_worker_timeout",
                )
        lines = [
            line for line in completed.stdout.splitlines()
            if line.startswith(_WORKER_MARKER)
        ]
        if len(lines) != 1:
            return RollingUnitRunnerObservation(
                unit.unit_key, attempt_id, "failed",
                failure_code="unit_worker_invalid_envelope",
            )
        try:
            result = json.loads(lines[0][len(_WORKER_MARKER):])
        except (TypeError, ValueError, json.JSONDecodeError):
            result = None
        if isinstance(result, dict) and result.get("process_control") in (
            "KeyboardInterrupt", "SystemExit", "GeneratorExit",
        ):
            control = {
                "KeyboardInterrupt": KeyboardInterrupt,
                "SystemExit": SystemExit,
                "GeneratorExit": GeneratorExit,
            }[result["process_control"]]
            raise control()
        if (
            completed.returncode != 0 or not isinstance(result, dict)
            or set(result) not in (
                {"status", "failure_code"},
                {"status", "experiment_name", "experiment_id", "recorder_id"},
            )
        ):
            return RollingUnitRunnerObservation(
                unit.unit_key, attempt_id, "failed",
                failure_code="unit_worker_invalid_envelope",
            )
        if result["status"] == "failed":
            return RollingUnitRunnerObservation(
                unit.unit_key, attempt_id, "failed",
                failure_code=_public_worker_text(
                    result["failure_code"], "failure_code",
                ),
            )
        if result["status"] != "candidate_success":
            return RollingUnitRunnerObservation(
                unit.unit_key, attempt_id, "failed",
                failure_code="unit_worker_invalid_status",
            )
        return RollingUnitRunnerObservation(
            unit.unit_key, attempt_id, "candidate_success",
            _public_worker_text(result["experiment_name"], "experiment_name"),
            _public_worker_text(result["experiment_id"], "experiment_id"),
            _public_worker_text(result["recorder_id"], "recorder_id"),
        )


if __name__ == "__main__":  # pragma: no cover - exercised through parent runner
    if len(sys.argv) != 2:
        raise SystemExit(2)
    raise SystemExit(_worker_main(sys.argv[1]))
