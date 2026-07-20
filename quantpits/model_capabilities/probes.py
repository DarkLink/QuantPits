"""Generated protocol probes and controlled observation boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import InitVar, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import ModelCapabilityIdentity, RawModelCapabilityDeclaration


_PROTOCOL_AUTHORITY = object()
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

    Public/test construction is deliberately harness-only.  Actual observation
    provenance can only be attached by :class:`ControlledProtocolProbe` after a
    strict subprocess envelope has been validated against the requested row.
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
    _authority: InitVar[object] = None
    _actual_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: object) -> None:
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
        object.__setattr__(self, "_actual_authority", _authority is _PROTOCOL_AUTHORITY)

    @property
    def measurement_source(self) -> str:
        if self._actual_authority:
            return "actual_wrapper_generated_protocol_probe"
        return "harness_self_test_only"

    @property
    def identity(self) -> ModelCapabilityIdentity:
        return ModelCapabilityIdentity(**{
            field_name: getattr(self, field_name) for field_name in _IDENTITY_FIELDS
        })

    def as_harness_only(self) -> "_ProtocolMeasurements":
        return replace(self)


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
            " result['fit_signature']=callable(getattr(c,'fit',None)) and accepts_named(c.fit,('dataset','evals_result'))\n"
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
        ) % (module_name, class_name, self._MARKER, self._MARKER, self._MARKER, self._MARKER)
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
    """Run bounded actual-wrapper adapters against generated data.

    The allowlist is intentionally narrow.  Rows without an exact adapter return
    a typed failure and remain ``not_comparable`` rather than inheriting support
    from a nearby action, family, dataset, or wrapper.
    """

    _MARKER = "QUANTPITS_PROTOCOL_RESULT="

    @staticmethod
    def supports(declaration: RawModelCapabilityDeclaration) -> bool:
        return (
            declaration.model_module in (
                "quantpits.utils.model_wrappers.custom.pytorch_lstm",
                "quantpits.utils.model_wrappers.lh.pytorch_lstm",
            )
            and declaration.model_class == "LSTM"
            and declaration.dataset_module == "qlib.data.dataset"
            and declaration.dataset_class == "DatasetH"
            and declaration.dataset_protocol == "point_in_time"
            and declaration.action == "train"
            and declaration.execution_family == "static"
            and declaration.processor_profile == "standard_infer_no_label_drop"
            and declaration.artifact_protocol == "qlib_recorder_model_v1"
            and declaration.dependency_profile == "python_qlib_torch"
        )

    def __init__(self, timeout_seconds: int = 15) -> None:
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        self.timeout_seconds = timeout_seconds

    def observe(self, declaration: RawModelCapabilityDeclaration) -> Any:
        if not isinstance(declaration, RawModelCapabilityDeclaration):
            raise TypeError("protocol probe requires a canonical declaration")
        if not self.supports(declaration):
            return ProtocolProbeFailure("protocol_adapter_not_available")
        identity = ModelCapabilityIdentity.from_declaration(declaration)
        public_identity = identity.to_public_dict()
        script = (
            "import importlib,json,sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
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
            "identity=%r\n"
            "stage='import'; result={'identity':identity,'reason':'protocol_probe_failed'}\n"
            "try:\n"
            " DatasetH=getattr(importlib.import_module(identity['dataset_module']),identity['dataset_class'])\n"
            " Model=getattr(importlib.import_module(identity['model_module']),identity['model_class'])\n"
            " stage='fixture'; dates=pd.to_datetime(('2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17','2026-07-20','2026-07-21','2026-07-22','2026-07-23','2026-07-24'))\n"
            " index=pd.MultiIndex.from_arrays((dates,('SYNTH_A',)*len(dates)),names=('datetime','instrument'))\n"
            " values=np.asarray(tuple((float(i),float(i %% 3)) for i in range(len(dates))),dtype='float32')\n"
            " labels=np.asarray(tuple(float(i %% 2) for i in range(len(dates))),dtype='float32')\n"
            " columns=pd.MultiIndex.from_tuples((('feature','f0'),('feature','f1'),('label','label')))\n"
            " frame=pd.DataFrame(np.column_stack((values,labels)),index=index,columns=columns)\n"
            " segments={'train':frame.iloc[:4],'valid':frame.iloc[4:6],'test':frame.iloc[6:]}\n"
            " class TinyDataset(DatasetH):\n"
            "  def __init__(self): pass\n"
            "  def prepare(self,segments,col_set=None,data_key=None,**kwargs):\n"
            "   def one(name):\n"
            "    selected=segments_map[name]\n"
            "    if col_set == 'feature': return selected['feature']\n"
            "    if col_set == ['feature','label']: return selected\n"
            "    return selected\n"
            "   return tuple(one(name) for name in segments) if isinstance(segments,list) else one(segments)\n"
            " segments_map=segments\n"
            " dataset=TinyDataset()\n"
            " stage='construct'; model=Model(d_feat=2,hidden_size=4,num_layers=1,dropout=0.0,n_epochs=1,batch_size=2,early_stop=1,metric='mse',loss='mse',GPU=-1,seed=0)\n"
            " evals_result={}\n"
            " stage='fit'; model.fit(dataset,evals_result=evals_result,save_path='generated-model.bin')\n"
            " stage='artifact_save'; model.to_pickle('generated-artifact.pkl',dump_all=True)\n"
            " with open('generated-artifact-source.json','w',encoding='utf-8') as handle: json.dump({'source':%r},handle,sort_keys=True)\n"
            " stage='artifact_load'; observed_model=Model.load('generated-artifact.pkl')\n"
            " with open('generated-artifact-source.json','r',encoding='utf-8') as handle: observed_source=json.load(handle)['source']\n"
            " stage='processor'; processed=dataset.prepare('test',col_set='feature',data_key='infer')\n"
            " processor_output=tuple(item.strftime('%%Y-%%m-%%d') for item in processed.index.get_level_values('datetime'))\n"
            " stage='predict'; prediction=observed_model.predict(dataset)\n"
            " expected=tuple(item.strftime('%%Y-%%m-%%d') for item in segments['test'].index.get_level_values('datetime'))\n"
            " observed=tuple(item.strftime('%%Y-%%m-%%d') for item in prediction.index.get_level_values('datetime'))\n"
            " actual_type=observed_model.__class__.__module__+'.'+observed_model.__class__.__name__\n"
            " expected_type=identity['model_module']+'.'+identity['model_class']\n"
            " action_protocol={'train':'generated_fit_then_reload_predict','incremental':'generated_refit_then_reload_predict','predict_only':'generated_artifact_reload_predict','resume':'generated_artifact_reload_retry_predict'}[identity['action']]\n"
            " result.update({'reason':'observed','action_protocol':action_protocol,'expected_index':expected,'observed_index':observed,'scores':tuple(float(item) for item in prediction.values),'processor_input_index':expected,'processor_output_index':processor_output,'artifact_expected_type':expected_type,'artifact_observed_type':actual_type,'artifact_expected_source':%r,'artifact_observed_source':observed_source})\n"
            "except ForbiddenBackendAccess:\n"
            " result['reason']='forbidden_backend_access'\n"
            "except KeyboardInterrupt:\n"
            " print(%r+json.dumps({'process_control':'KeyboardInterrupt'},sort_keys=True)); sys.exit(130)\n"
            "except SystemExit:\n"
            " print(%r+json.dumps({'process_control':'SystemExit'},sort_keys=True)); sys.exit(131)\n"
            "except GeneratorExit:\n"
            " print(%r+json.dumps({'process_control':'GeneratorExit'},sort_keys=True)); sys.exit(132)\n"
            "except Exception as exc:\n"
            " result['reason']='actual_protocol_'+stage+'_'+exc.__class__.__name__\n"
            "print(%r+json.dumps(result,sort_keys=True))\n"
        ) % (
            public_identity, identity.fingerprint, identity.fingerprint,
            self._MARKER, self._MARKER, self._MARKER, self._MARKER,
        )
        with tempfile.TemporaryDirectory(prefix="quantpits-protocol-") as temp_dir:
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", script], cwd=temp_dir,
                    env=ControlledImportProbe._environment(temp_dir),
                    capture_output=True, text=True, timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                return ProtocolProbeFailure("protocol_probe_timeout")
        marker_lines = [line for line in completed.stdout.splitlines() if line.startswith(self._MARKER)]
        if len(marker_lines) == 1:
            try:
                payload = json.loads(marker_lines[0][len(self._MARKER):])
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, Mapping) and payload.get("process_control") in (
                "KeyboardInterrupt", "SystemExit", "GeneratorExit",
            ):
                if payload["process_control"] == "KeyboardInterrupt":
                    raise KeyboardInterrupt()
                if payload["process_control"] == "SystemExit":
                    raise SystemExit("controlled protocol subprocess exited")
                raise GeneratorExit()
        if completed.returncode != 0 or len(marker_lines) != 1:
            return ProtocolProbeFailure("invalid_protocol_probe_envelope")
        try:
            payload = json.loads(marker_lines[0][len(self._MARKER):])
            if payload.get("reason") != "observed":
                reason = payload.get("reason")
                return ProtocolProbeFailure(reason if isinstance(reason, str) else "invalid_protocol_probe_reason")
            if payload.get("identity") != public_identity:
                return ProtocolProbeFailure("protocol_probe_identity_envelope_mismatch")
            expected = {
                "identity", "reason", "action_protocol", "expected_index", "observed_index", "scores",
                "processor_input_index", "processor_output_index", "artifact_expected_type",
                "artifact_observed_type", "artifact_expected_source", "artifact_observed_source",
            }
            if set(payload) != expected:
                return ProtocolProbeFailure("invalid_protocol_probe_envelope")
            values = dict(public_identity)
            for field_name in expected - {"identity", "reason"}:
                value = payload[field_name]
                values[field_name] = tuple(value) if field_name in (
                    "expected_index", "observed_index", "scores",
                    "processor_input_index", "processor_output_index",
                ) else value
            return _ProtocolMeasurements(**values, _authority=_PROTOCOL_AUTHORITY)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ProtocolProbeFailure("invalid_protocol_probe_envelope")


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_nodes(root: Path) -> Tuple[Tuple[str, str, Optional[str], int, Optional[str]], ...]:
    """Snapshot node kind, symlink target and size without following symlinks."""
    root = Path(root)
    observations = []
    if not root.exists() and not root.is_symlink():
        return tuple()
    paths = [root]
    if root.is_dir() and not root.is_symlink():
        paths.extend(sorted(root.rglob("*")))
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
            observations.append((relative, kind, os.readlink(str(path)), info.st_size, None))
        elif stat.S_ISDIR(info.st_mode):
            observations.append((relative, "directory", None, info.st_size, None))
        elif stat.S_ISREG(info.st_mode):
            observations.append((relative, "file", None, info.st_size, _file_fingerprint(path)))
        else:
            observations.append((relative, "special", None, info.st_size, None))
    return tuple(observations)


class ZeroWriteObserver:
    """Compare explicit protected roots across a complete probe lifecycle."""

    def __init__(self, protected_roots: Sequence[Path]) -> None:
        self._roots = tuple(Path(item) for item in protected_roots)
        self._before = None  # type: Optional[Tuple[Tuple[Tuple[str, str, Optional[str], int, Optional[str]], ...], ...]]

    def __enter__(self) -> "ZeroWriteObserver":
        self._before = tuple(snapshot_nodes(root) for root in self._roots)
        if any(node[1] == "symlink_escape" for root in self._before for node in root):
            raise RuntimeError("capability probe protected root contains an external symlink")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        after = tuple(snapshot_nodes(root) for root in self._roots)
        if self._before != after:
            raise RuntimeError("capability probe wrote to a protected root")
