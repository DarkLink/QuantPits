"""Generated protocol probes and controlled import observation."""

from __future__ import annotations

import json
import hashlib
import math
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


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
class ProtocolMeasurements:
    model_module: str
    model_class: str
    measurement_source: str
    expected_index: Tuple[str, ...]
    observed_index: Tuple[str, ...]
    scores: Tuple[float, ...]
    dataset_protocol: str
    processor_input_index: Tuple[str, ...]
    processor_output_index: Tuple[str, ...]
    artifact_expected_type: str
    artifact_observed_type: str
    artifact_expected_source: str
    artifact_observed_source: str

    def __post_init__(self) -> None:
        for field_name in (
            "expected_index", "observed_index", "scores", "processor_input_index",
            "processor_output_index",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError("%s must be an ordered tuple" % field_name)
        for field_name in (
            "model_module", "model_class", "measurement_source", "dataset_protocol",
            "artifact_expected_type", "artifact_observed_type",
            "artifact_expected_source", "artifact_observed_source",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("%s must be a non-empty trimmed string" % field_name)
        if self.measurement_source not in (
            "actual_wrapper_generated_protocol_probe", "harness_self_test_only",
        ):
            raise ValueError("measurement_source is unsupported")
        for field_name in ("expected_index", "observed_index", "processor_input_index", "processor_output_index"):
            if any(not isinstance(item, str) or not item for item in getattr(self, field_name)):
                raise ValueError("%s must contain public index strings" % field_name)


@dataclass(frozen=True)
class GeneratedProtocolFixture:
    dataset_protocol: str
    features: Any
    labels: Any
    expected_index: Tuple[str, ...]
    market_labels: Any = None


def generated_protocol_fixture(dataset_protocol: str) -> GeneratedProtocolFixture:
    """Build deterministic tiny pandas/numpy data for harness self-tests only."""
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


def classify_prediction_coverage(observation: ProtocolMeasurements) -> Dict[str, bool]:
    """Execute exact tail/gap/uniqueness/finiteness predicates."""
    expected = observation.expected_index
    observed = observation.observed_index
    observed_set = set(observed)
    return {
        "prediction_tail": bool(expected and observed and observed[-1] == expected[-1]),
        "prediction_gap": observed == expected,
        "prediction_unique": len(observed) == len(observed_set),
        "prediction_finite": len(observed) == len(observation.scores) and all(math.isfinite(item) for item in observation.scores),
    }


class ControlledImportProbe:
    """Import one exact class in a short-lived, cache-isolated subprocess."""

    _MARKER = "QUANTPITS_CAPABILITY_RESULT="

    def __init__(self, timeout_seconds: int = 15) -> None:
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        self.timeout_seconds = timeout_seconds

    def observe(self, module_name: str, class_name: str) -> ImportObservation:
        script = (
            "import importlib, inspect, json\n"
            "result={'imported':False,'class_resolved':False,'constructor_signature':False,'fit_signature':False,'predict_signature':False,'gpu_available':False,'dependency_missing':False,'reason':'import_failed'}\n"
            "try:\n"
            " m=importlib.import_module(%r); result['imported']=True\n"
            " c=getattr(m,%r); result['class_resolved']=isinstance(c,type)\n"
            " result['constructor_signature']=bool(inspect.signature(c))\n"
            " result['fit_signature']=callable(getattr(c,'fit',None)) and 'dataset' in inspect.signature(c.fit).parameters\n"
            " result['predict_signature']=callable(getattr(c,'predict',None)) and 'dataset' in inspect.signature(c.predict).parameters\n"
            " t=__import__('sys').modules.get('torch'); result['gpu_available']=bool(t is not None and t.cuda.is_available())\n"
            " result['reason']='observed'\n"
            "except (ImportError, ModuleNotFoundError):\n"
            " result['dependency_missing']=True; result['reason']='dependency_missing'\n"
            "except AttributeError:\n"
            " result['reason']='class_missing'\n"
            "except Exception:\n"
            " result['reason']='import_probe_exception'\n"
            "print(%r+json.dumps(result,sort_keys=True))\n"
        ) % (module_name, class_name, self._MARKER)
        with tempfile.TemporaryDirectory(prefix="quantpits-capability-") as temp_dir:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join(sys.path),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "HOME": temp_dir,
                "TMPDIR": temp_dir,
                "XDG_CACHE_HOME": str(Path(temp_dir) / "cache"),
                "MPLCONFIGDIR": str(Path(temp_dir) / "matplotlib"),
            }
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", script], cwd=temp_dir, env=env,
                    capture_output=True, text=True, timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                return ImportObservation(False, False, False, False, False, False, False, "import_probe_timeout")
        marker_lines = [line for line in completed.stdout.splitlines() if line.startswith(self._MARKER)]
        if completed.returncode != 0 or len(marker_lines) != 1:
            return ImportObservation(False, False, False, False, False, False, False, "invalid_probe_envelope")
        try:
            payload = json.loads(marker_lines[0][len(self._MARKER):])
            required = {
                "imported", "class_resolved", "constructor_signature", "fit_signature", "predict_signature", "gpu_available",
                "dependency_missing", "reason",
            }
            if set(payload) != required or any(type(payload[key]) is not bool for key in required - {"reason"}):
                raise ValueError("invalid envelope")
            if not isinstance(payload["reason"], str):
                raise ValueError("invalid reason")
            return ImportObservation(**payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ImportObservation(False, False, False, False, False, False, False, "invalid_probe_envelope")


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
