"""Generated protocol probes and controlled observation boundaries."""

from __future__ import annotations

import ctypes
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import InitVar, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import ModelCapabilityIdentity, RawModelCapabilityDeclaration


_IDENTITY_FIELDS = (
    "model_module", "model_class", "wrapper_kind", "dataset_module", "dataset_class",
    "dataset_protocol", "action", "execution_family", "processor_profile",
    "artifact_protocol", "dependency_profile",
)
_ACTION_PROTOCOLS = {
    "train": "generated_fit_then_reload_predict",
    "incremental": "generated_refit_then_reload_predict",
    "predict_only": "generated_artifact_reload_predict",
    "resume": "generated_artifact_reload_retry_predict",
}
_LINEAR_ROLLING_IDENTITY = {
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
_ACTUAL_MEASUREMENT_TOKEN = object()


def _action_protocol(action: str) -> str:
    return _ACTION_PROTOCOLS[action]


@dataclass(frozen=True)
class ImportObservation:
    imported: bool
    class_resolved: bool
    constructor_signature: bool
    fit_signature: bool
    predict_signature: bool
    gpu_available: bool
    dependency_missing: bool
    reason: str


@dataclass(frozen=True)
class ProtocolProbeFailure:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason or self.reason != self.reason.strip():
            raise ValueError("protocol probe failure requires a stable reason")


@dataclass(frozen=True)
class _ProtocolMeasurements:
    """Internal measurement envelope.

    Base-envelope construction is deliberately harness-only.  Only the private
    actual subclass can attach inspector-routed observation authority.
    """

    model_module: str
    model_class: str
    wrapper_kind: str
    dataset_module: str
    dataset_class: str
    dataset_protocol: str
    action: str
    execution_family: str
    processor_profile: str
    artifact_protocol: str
    dependency_profile: str
    action_protocol: str
    expected_index: Tuple[str, ...]
    observed_index: Tuple[str, ...]
    scores: Tuple[float, ...]
    processor_input_index: Tuple[str, ...]
    processor_output_index: Tuple[str, ...]
    artifact_expected_type: str
    artifact_observed_type: str
    artifact_expected_source: str
    artifact_observed_source: str

    def __post_init__(self) -> None:
        identity = ModelCapabilityIdentity(**{
            field_name: getattr(self, field_name) for field_name in _IDENTITY_FIELDS
        })
        for field_name in _IDENTITY_FIELDS:
            object.__setattr__(self, field_name, getattr(identity, field_name))
        for field_name in (
            "expected_index", "observed_index", "scores", "processor_input_index",
            "processor_output_index",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError("%s must be an ordered tuple" % field_name)
        for field_name in (
            "expected_index", "observed_index", "processor_input_index", "processor_output_index",
        ):
            if any(not isinstance(item, str) or not item for item in getattr(self, field_name)):
                raise ValueError("%s must contain public index strings" % field_name)
        for field_name in (
            "action_protocol",
            "artifact_expected_type", "artifact_observed_type",
            "artifact_expected_source", "artifact_observed_source",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("%s must be a non-empty trimmed string" % field_name)

    @property
    def measurement_source(self) -> str:
        return "harness_self_test_only"

    @property
    def has_actual_authority(self) -> bool:
        return False

    @property
    def identity(self) -> ModelCapabilityIdentity:
        return ModelCapabilityIdentity(**{
            field_name: getattr(self, field_name) for field_name in _IDENTITY_FIELDS
        })

    def as_harness_only(self) -> "_ProtocolMeasurements":
        return _ProtocolMeasurements(**{
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        })


@dataclass(frozen=True)
class _ActualProtocolMeasurements(_ProtocolMeasurements):
    """Inspector-routed observation from the exact generated Linear adapter."""

    _authority: InitVar[Any] = None

    def __post_init__(self, _authority: Any) -> None:
        super().__post_init__()
        if _authority is not _ACTUAL_MEASUREMENT_TOKEN:
            raise TypeError("actual protocol measurements are inspector-owned")

    @property
    def measurement_source(self) -> str:
        return "actual_wrapper_generated_protocol_probe"

    @property
    def has_actual_authority(self) -> bool:
        return True

    def as_harness_only(self) -> _ProtocolMeasurements:
        return _ProtocolMeasurements(**{
            field_name: getattr(self, field_name)
            for field_name in _ProtocolMeasurements.__dataclass_fields__
        })


def _harness_protocol_measurements(
    declaration: RawModelCapabilityDeclaration,
    expected_index: Tuple[str, ...],
    observed_index: Tuple[str, ...],
    scores: Tuple[float, ...],
    **overrides: Any
) -> _ProtocolMeasurements:
    """Build an untrusted measurement for deliberate negative harness probes."""
    if not isinstance(declaration, RawModelCapabilityDeclaration):
        raise TypeError("harness measurements require a canonical declaration")
    values = ModelCapabilityIdentity.from_declaration(declaration).to_public_dict()
    values.update({
        "expected_index": expected_index,
        "observed_index": observed_index,
        "scores": scores,
        "action_protocol": _action_protocol(declaration.action),
        "processor_input_index": expected_index,
        "processor_output_index": expected_index,
        "artifact_expected_type": declaration.model_class,
        "artifact_observed_type": declaration.model_class,
        "artifact_expected_source": ModelCapabilityIdentity.from_declaration(declaration).fingerprint,
        "artifact_observed_source": ModelCapabilityIdentity.from_declaration(declaration).fingerprint,
    })
    values.update(overrides)
    return _ProtocolMeasurements(**values)


@dataclass(frozen=True)
class GeneratedProtocolFixture:
    dataset_protocol: str
    features: Any
    labels: Any
    expected_index: Tuple[str, ...]
    market_labels: Any = None


def generated_protocol_fixture(dataset_protocol: str) -> GeneratedProtocolFixture:
    """Build deterministic tiny pandas/numpy data for protocol probes."""
    import numpy as np
    import pandas as pd

    protocols = (
        "point_in_time", "time_series", "memory_time_series", "daily_market_label", "multi_label",
    )
    if not isinstance(dataset_protocol, str) or dataset_protocol not in protocols:
        raise ValueError("unknown generated dataset protocol")
    dates = pd.to_datetime(("2026-07-17", "2026-07-20", "2026-07-21"))
    index = pd.MultiIndex.from_arrays(
        (dates, ("SYNTH_A", "SYNTH_A", "SYNTH_A")),
        names=("datetime", "instrument"),
    )
    features = pd.DataFrame(
        np.asarray(((1.0, 0.0), (2.0, 1.0), (3.0, 1.0)), dtype="float64"),
        index=index, columns=("feature_0", "feature_1"),
    )
    if dataset_protocol == "multi_label":
        labels = pd.DataFrame(
            np.asarray(((0.1, 0.2), (0.2, 0.3), (np.nan, np.nan)), dtype="float64"),
            index=index, columns=("label_0", "label_1"),
        )
    else:
        labels = pd.Series((0.1, 0.2, np.nan), index=index, name="label", dtype="float64")
    market_labels = None
    if dataset_protocol == "daily_market_label":
        market_labels = pd.Series((0.01, 0.02, np.nan), index=dates, name="market_label", dtype="float64")
    return GeneratedProtocolFixture(
        dataset_protocol, features, labels,
        tuple(item.strftime("%Y-%m-%d") for item in dates), market_labels,
    )


def classify_prediction_coverage(observation: _ProtocolMeasurements) -> Dict[str, bool]:
    """Execute exact tail/gap/uniqueness/finiteness predicates."""
    if not isinstance(observation, _ProtocolMeasurements):
        raise TypeError("coverage classification requires an internal measurement envelope")
    expected = observation.expected_index
    observed = observation.observed_index
    observed_set = set(observed)
    return {
        "prediction_tail": bool(expected and observed and observed[-1] == expected[-1]),
        "prediction_gap": observed == expected,
        "prediction_unique": len(observed) == len(observed_set),
        "prediction_finite": len(observed) == len(observation.scores) and all(
            isinstance(item, (int, float)) and math.isfinite(item) for item in observation.scores
        ),
    }


class _ConstructorRequiresArgument:
    def __init__(self, required_value: object) -> None:
        self.required_value = required_value

    def fit(self, dataset: object, evals_result: object = None) -> None:
        return None

    def predict(self, dataset: object) -> None:
        return None


class _FitWithoutEvalsResult:
    def fit(self, dataset: object) -> None:
        return None

    def predict(self, dataset: object) -> None:
        return None


class ControlledImportProbe:
    """Import one exact class in a short-lived, backend-denying subprocess."""

    _MARKER = "QUANTPITS_CAPABILITY_RESULT="

    def __init__(self, timeout_seconds: int = 15) -> None:
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _environment(temp_dir: str, pythonpath_entries: Sequence[Path] = ()) -> Dict[str, str]:
        pythonpath = tuple(str(Path(item)) for item in pythonpath_entries) + tuple(sys.path)
        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HOME": temp_dir,
            "TMPDIR": temp_dir,
            "XDG_CACHE_HOME": str(Path(temp_dir) / "cache"),
            "MPLCONFIGDIR": str(Path(temp_dir) / "matplotlib"),
        }

    def observe(
        self,
        module_name: str,
        class_name: str,
        pythonpath_entries: Sequence[Path] = (),
    ) -> ImportObservation:
        script = (
            "import importlib, inspect, json, sys\n"
            "class ForbiddenBackendAccess(RuntimeError): pass\n"
            "class DenyBackendFinder:\n"
            " def find_spec(self,fullname,path=None,target=None):\n"
            "  if fullname == 'quantpits.utils.env': raise ForbiddenBackendAccess(fullname)\n"
            "  return None\n"
            "sys.meta_path.insert(0,DenyBackendFinder())\n"
            "def deny_backend(*args,**kwargs): raise ForbiddenBackendAccess('backend_hook_called')\n"
            "import qlib\n"
            "qlib.init=deny_backend\n"
            "try:\n"
            " import mlflow; mlflow.set_tracking_uri=deny_backend; mlflow.start_run=deny_backend; mlflow.set_registry_uri=deny_backend\n"
            "except ImportError: pass\n"
            "result={'imported':False,'class_resolved':False,'constructor_signature':False,'fit_signature':False,'predict_signature':False,'gpu_available':False,'dependency_missing':False,'reason':'import_failed'}\n"
            "def accepts_named(method,names):\n"
            " try: params=inspect.signature(method).parameters\n"
            " except (TypeError,ValueError): return False\n"
            " has_kwargs=any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())\n"
            " return all(name in params or has_kwargs for name in names)\n"
            "try:\n"
            " m=importlib.import_module(%r); result['imported']=True\n"
            " c=getattr(m,%r); result['class_resolved']=isinstance(c,type)\n"
            " try: inspect.signature(c).bind(); result['constructor_signature']=True\n"
            " except (TypeError,ValueError): result['constructor_signature']=False\n"
            " exact_linear=(%r == 'qlib.contrib.model.linear' and %r == 'LinearModel')\n"
            " result['fit_signature']=callable(getattr(c,'fit',None)) and accepts_named(c.fit,('dataset','reweighter') if exact_linear else ('dataset','evals_result'))\n"
            " result['predict_signature']=callable(getattr(c,'predict',None)) and accepts_named(c.predict,('dataset',))\n"
            " t=sys.modules.get('torch'); result['gpu_available']=bool(t is not None and t.cuda.is_available())\n"
            " result['reason']='observed'\n"
            "except ForbiddenBackendAccess:\n"
            " result['reason']='forbidden_backend_access'\n"
            "except (ImportError,ModuleNotFoundError):\n"
            " result['dependency_missing']=True; result['reason']='dependency_missing'\n"
            "except AttributeError:\n"
            " result['reason']='class_missing'\n"
            "except KeyboardInterrupt:\n"
            " print(%r+json.dumps({'process_control':'KeyboardInterrupt'},sort_keys=True)); sys.exit(130)\n"
            "except SystemExit:\n"
            " print(%r+json.dumps({'process_control':'SystemExit'},sort_keys=True)); sys.exit(131)\n"
            "except GeneratorExit:\n"
            " print(%r+json.dumps({'process_control':'GeneratorExit'},sort_keys=True)); sys.exit(132)\n"
            "except Exception:\n"
            " result['reason']='import_probe_exception'\n"
            "print(%r+json.dumps(result,sort_keys=True))\n"
        ) % (
            module_name, class_name, module_name, class_name,
            self._MARKER, self._MARKER, self._MARKER, self._MARKER,
        )
        with tempfile.TemporaryDirectory(prefix="quantpits-capability-") as temp_dir:
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", script], cwd=temp_dir,
                    env=self._environment(temp_dir, pythonpath_entries),
                    capture_output=True, text=True, timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                return ImportObservation(False, False, False, False, False, False, False, "import_probe_timeout")
        marker_lines = [line for line in completed.stdout.splitlines() if line.startswith(self._MARKER)]
        if len(marker_lines) == 1:
            try:
                payload = json.loads(marker_lines[0][len(self._MARKER):])
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, Mapping) and payload.get("process_control") in (
                "KeyboardInterrupt", "SystemExit", "GeneratorExit",
            ):
                control = payload["process_control"]
                if control == "KeyboardInterrupt":
                    raise KeyboardInterrupt()
                if control == "SystemExit":
                    raise SystemExit("controlled import subprocess exited")
                raise GeneratorExit()
        if completed.returncode != 0 or len(marker_lines) != 1:
            return ImportObservation(False, False, False, False, False, False, False, "invalid_probe_envelope")
        try:
            payload = json.loads(marker_lines[0][len(self._MARKER):])
            required = {
                "imported", "class_resolved", "constructor_signature", "fit_signature", "predict_signature",
                "gpu_available", "dependency_missing", "reason",
            }
            if set(payload) != required or any(type(payload[key]) is not bool for key in required - {"reason"}):
                raise ValueError("invalid envelope")
            if not isinstance(payload["reason"], str):
                raise ValueError("invalid reason")
            return ImportObservation(**payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ImportObservation(False, False, False, False, False, False, False, "invalid_probe_envelope")


class ControlledProtocolProbe:
    """Run the one bounded generated protocol adapter authorized by the catalog."""

    @staticmethod
    def supports(declaration: RawModelCapabilityDeclaration) -> bool:
        if not isinstance(declaration, RawModelCapabilityDeclaration):
            raise TypeError("protocol support requires a canonical declaration")
        return ModelCapabilityIdentity.from_declaration(declaration).to_public_dict() == _LINEAR_ROLLING_IDENTITY

    def __init__(self, timeout_seconds: int = 15) -> None:
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        self.timeout_seconds = timeout_seconds

    def observe(self, declaration: RawModelCapabilityDeclaration) -> Any:
        if not isinstance(declaration, RawModelCapabilityDeclaration):
            raise TypeError("protocol probe requires a canonical declaration")
        if not self.supports(declaration):
            return ProtocolProbeFailure("protocol_adapter_not_available")
        script = r'''
import hashlib, json
from pathlib import Path
from urllib.parse import unquote, urlparse
import numpy as np
import pandas as pd
import qlib
from qlib.contrib.model.linear import LinearModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler
from qlib.data.dataset.loader import StaticDataLoader
from qlib.workflow import R

root = Path.cwd()
qlib.init(
    provider_uri=str(root / "qlib_data"),
    exp_manager={
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": "sqlite:///" + str(root / "probe_mlflow.db"),
            "default_exp_name": "Phase34CapabilityProbe",
        },
    },
)

dates = pd.to_datetime(("2026-07-17", "2026-07-20", "2026-07-21"))
index = pd.MultiIndex.from_arrays(
    (dates, ("SYNTH_A", "SYNTH_A", "SYNTH_A")),
    names=("datetime", "instrument"),
)
columns = pd.MultiIndex.from_tuples(
    (("feature", "feature_0"), ("feature", "feature_1"), ("label", "label"))
)
frame = pd.DataFrame(
    np.asarray(((1.0, 0.0, 0.1), (2.0, 1.0, 0.2), (3.0, 1.0, 0.3))),
    index=index, columns=columns,
)
handler = DataHandler(
    instruments=None, start_time=dates[0], end_time=dates[-1],
    data_loader=StaticDataLoader(frame),
)
dataset = DatasetH(
    handler=handler,
    segments={"train": (dates[0], dates[1]), "test": (dates[0], dates[-1])},
)
model = LinearModel(estimator="ridge", alpha=1e-6)
prepared = dataset.prepare(
    "test", col_set="feature", data_key=DataHandler.DK_I,
)
with R.start(experiment_name="Phase34CapabilityProbe"):
    recorder = R.get_recorder()
    model.fit(dataset=dataset)
    recorder.save_objects(**{"model.pkl": model})
    recorder_id = recorder.info["id"]
    artifact_uri = recorder.get_artifact_uri()
    parsed = urlparse(artifact_uri)
    if parsed.scheme not in ("", "file"):
        raise RuntimeError("non-local probe artifact backend")
    artifact_path = Path(unquote(parsed.path if parsed.scheme else artifact_uri)) / "model.pkl"
    artifact = artifact_path.read_bytes()
    reloaded = recorder.load_object("model.pkl")
    observed_artifact = artifact_path.read_bytes()
prediction = reloaded.predict(dataset=dataset, segment="test")
observed_dates = prediction.index.get_level_values("datetime")
payload = {
    "expected_index": [item.strftime("%Y-%m-%d") for item in dates],
    "observed_index": [item.strftime("%Y-%m-%d") for item in observed_dates],
    "scores": [float(item) for item in prediction.to_numpy()],
    "processor_input_index": [item.strftime("%Y-%m-%d") for item in frame.index.get_level_values("datetime")],
    "processor_output_index": [item.strftime("%Y-%m-%d") for item in prepared.index.get_level_values("datetime")],
    "artifact_expected_type": "LinearModel",
    "artifact_observed_type": type(reloaded).__name__,
    "artifact_expected_source": hashlib.sha256(artifact).hexdigest(),
    "artifact_observed_source": hashlib.sha256(observed_artifact).hexdigest(),
    "recorder_id": recorder_id,
}
print("QUANTPITS_LINEAR_PROTOCOL=" + json.dumps(payload, sort_keys=True))
'''
        marker = "QUANTPITS_LINEAR_PROTOCOL="
        with tempfile.TemporaryDirectory(prefix="quantpits-linear-protocol-") as temp_dir:
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", script], cwd=temp_dir,
                    env=ControlledImportProbe._environment(temp_dir),
                    capture_output=True, text=True, timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ProtocolProbeFailure("protocol_probe_timeout")
        lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
        if completed.returncode != 0 or len(lines) != 1:
            return ProtocolProbeFailure("protocol_probe_failed")
        try:
            payload = json.loads(lines[0][len(marker):])
            required = {
                "expected_index", "observed_index", "scores",
                "processor_input_index", "processor_output_index",
                "artifact_expected_type", "artifact_observed_type",
                "artifact_expected_source", "artifact_observed_source",
                "recorder_id",
            }
            if set(payload) != required:
                raise ValueError("invalid fields")
            if (
                not isinstance(payload["recorder_id"], str)
                or not payload["recorder_id"].strip()
            ):
                raise ValueError("invalid recorder identity")
            values = dict(_LINEAR_ROLLING_IDENTITY)
            values.update({
                "action_protocol": _action_protocol(declaration.action),
                "expected_index": tuple(payload["expected_index"]),
                "observed_index": tuple(payload["observed_index"]),
                "scores": tuple(payload["scores"]),
                "processor_input_index": tuple(payload["processor_input_index"]),
                "processor_output_index": tuple(payload["processor_output_index"]),
                "artifact_expected_type": payload["artifact_expected_type"],
                "artifact_observed_type": payload["artifact_observed_type"],
                "artifact_expected_source": payload["artifact_expected_source"],
                "artifact_observed_source": payload["artifact_observed_source"],
                "_authority": _ACTUAL_MEASUREMENT_TOKEN,
            })
            return _ActualProtocolMeasurements(**values)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ProtocolProbeFailure("invalid_protocol_probe_envelope")


def snapshot_nodes(
    root: Path,
    excluded_relative_paths: Sequence[str] = (),
) -> Tuple[Tuple[str, str, Optional[str], int, Optional[str]], ...]:
    """Snapshot public node metadata without reading file contents or following symlinks."""
    root = Path(root)
    exclusions = tuple(sorted(set(excluded_relative_paths)))
    for value in exclusions:
        candidate = Path(value)
        if (
            not isinstance(value, str) or not value or value != value.strip()
            or candidate.is_absolute() or value == "." or ".." in candidate.parts
        ):
            raise ValueError("observer exclusions must be safe relative paths")

    def is_excluded(relative: str) -> bool:
        return any(relative == item or relative.startswith(item + "/") for item in exclusions)

    def descendants(directory: Path) -> Sequence[Path]:
        found = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root).as_posix()
            if is_excluded(relative):
                continue
            found.append(child)
            if child.is_dir() and not child.is_symlink():
                found.extend(descendants(child))
        return found

    observations = []
    if not root.exists() and not root.is_symlink():
        return tuple()
    paths = [root]
    if root.is_dir() and not root.is_symlink():
        paths.extend(descendants(root))
    for path in paths:
        relative = "." if path == root else path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = path.resolve(strict=False)
            try:
                target.relative_to(root.resolve())
                kind = "symlink"
            except ValueError:
                kind = "symlink_escape"
            observations.append((relative, kind, os.readlink(str(path)), info.st_size, str(info.st_mtime_ns)))
        elif stat.S_ISDIR(info.st_mode):
            observations.append((relative, "directory", None, info.st_size, str(info.st_mtime_ns)))
        elif stat.S_ISREG(info.st_mode):
            observations.append((relative, "file", None, info.st_size, str(info.st_mtime_ns)))
        else:
            observations.append((relative, "special", None, info.st_size, str(info.st_mtime_ns)))
    return tuple(observations)


class ZeroWriteObserver:
    """Compare explicit protected roots across a complete probe lifecycle."""

    def __init__(
        self,
        protected_roots: Sequence[Path],
        excluded_relative_paths: Sequence[str] = (),
    ) -> None:
        self._roots = tuple(Path(item) for item in protected_roots)
        self._exclusions = tuple(excluded_relative_paths)
        self._before = None  # type: Optional[Tuple[Tuple[Tuple[str, str, Optional[str], int, Optional[str]], ...], ...]]
        self._monitors = ()  # type: Tuple[_InotifyWriteMonitor, ...]

    def __enter__(self) -> "ZeroWriteObserver":
        self._before = tuple(snapshot_nodes(root, self._exclusions) for root in self._roots)
        if any(node[1] == "symlink_escape" for root in self._before for node in root):
            raise RuntimeError("capability probe protected root contains an external symlink")
        monitors = []
        try:
            for root, observations in zip(self._roots, self._before):
                monitor = _InotifyWriteMonitor(root, observations)
                monitor.start()
                monitors.append(monitor)
        except BaseException:
            for monitor in monitors:
                monitor.close()
            raise
        self._monitors = tuple(monitors)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Optional[bool]:
        after = None
        event_write = False
        observer_error = None  # type: Optional[BaseException]
        try:
            after = tuple(snapshot_nodes(root, self._exclusions) for root in self._roots)
            event_write = any(monitor.has_write_event() for monitor in self._monitors)
        except BaseException as error:
            observer_error = error
        finally:
            for monitor in self._monitors:
                try:
                    monitor.close()
                except BaseException as error:
                    if observer_error is None:
                        observer_error = error
            self._monitors = ()
        if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            return False
        if observer_error is not None:
            raise observer_error
        if self._before != after or event_write:
            raise RuntimeError("capability probe wrote to a protected root")
        return False


class _InotifyWriteMonitor:
    """Observe mutations without reading protected file contents."""

    _MASK = (
        0x00000002  # IN_MODIFY
        | 0x00000004  # IN_ATTRIB
        | 0x00000008  # IN_CLOSE_WRITE
        | 0x00000040  # IN_MOVED_FROM
        | 0x00000080  # IN_MOVED_TO
        | 0x00000100  # IN_CREATE
        | 0x00000200  # IN_DELETE
        | 0x00000400  # IN_DELETE_SELF
        | 0x00000800  # IN_MOVE_SELF
    )

    def __init__(
        self,
        root: Path,
        observations: Sequence[Tuple[str, str, Optional[str], int, Optional[str]]],
    ) -> None:
        self._root = Path(root)
        self._observations = tuple(observations)
        self._fd = -1

    def start(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            init = libc.inotify_init1
            add_watch = libc.inotify_add_watch
        except AttributeError as exc:
            raise RuntimeError("zero-write observation requires inotify") from exc
        init.argtypes = (ctypes.c_int,)
        init.restype = ctypes.c_int
        add_watch.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
        add_watch.restype = ctypes.c_int
        fd = init(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "unable to initialize zero-write observer")
        self._fd = fd
        try:
            for relative, kind, _target, _size, _stamp in self._observations:
                if kind not in ("directory", "file"):
                    continue
                path = self._root if relative == "." else self._root / relative
                if add_watch(fd, os.fsencode(str(path)), self._MASK) < 0:
                    raise OSError(ctypes.get_errno(), "unable to protect repository node")
        except BaseException:
            self.close()
            raise

    def has_write_event(self) -> bool:
        observed = False
        while self._fd >= 0:
            try:
                data = os.read(self._fd, 65536)
            except BlockingIOError:
                break
            if not data:
                break
            observed = True
        return observed

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
