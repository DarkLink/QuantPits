"""Versioned, per-model training-record contracts.

The legacy ``models`` mapping remains a compatibility view.  ``model_records``
is authoritative whenever ``schema_version == 2``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple

KNOWN_MODES = ("static", "rolling", "cpcv", "cpcv_rolling")
KNOWN_OPERATIONS = (
    "train", "predict_only", "rolling_combine", "cpcv_predict",
    "cpcv_rolling_combine", "legacy_import",
)
CURRENT_STATUSES = ("ready", "legacy_unverified")


def split_model_key(key: str) -> Tuple[str, str]:
    if not isinstance(key, str) or "@" not in key:
        raise ValueError("canonical model key must use model@mode")
    model_name, mode = key.rsplit("@", 1)
    if not model_name or mode not in KNOWN_MODES:
        raise ValueError("invalid canonical model key: %s" % key)
    return model_name, mode


def _date(value: Optional[str], field: str) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("%s must be YYYY-MM-DD" % field) from exc
    return str(value)


@dataclass(frozen=True)
class ModelRecordEntry:
    key: str
    model_name: str
    training_mode: str
    operation: str
    status: str
    recorder_id: str
    experiment_name: str
    experiment_id: Optional[str] = None
    artifact_path: Optional[str] = None
    requested_anchor: Optional[str] = None
    prediction_start: Optional[str] = None
    prediction_end: Optional[str] = None
    prediction_rows: Optional[int] = None
    dataset_test_end: Optional[str] = None
    fit_start: Optional[str] = None
    fit_end: Optional[str] = None
    source_recorder_id: Optional[str] = None
    source_experiment_name: Optional[str] = None
    source_operation: Optional[str] = None
    produced_at: Optional[str] = None
    config_fingerprint: Optional[str] = None

    def __post_init__(self):
        name, mode = split_model_key(self.key)
        if (name, mode) != (self.model_name, self.training_mode):
            raise ValueError("entry fields do not reconstruct key")
        if self.operation not in KNOWN_OPERATIONS:
            raise ValueError("unsupported training-record operation")
        if self.status not in CURRENT_STATUSES:
            raise ValueError("unsupported current-record status")
        if not self.recorder_id or not self.experiment_name:
            raise ValueError("recorder_id and experiment_name are required")
        for field in ("requested_anchor", "prediction_start", "prediction_end", "dataset_test_end", "fit_start", "fit_end"):
            _date(getattr(self, field), field)
        if self.prediction_rows is not None and self.prediction_rows < 0:
            raise ValueError("prediction_rows must be non-negative")
        if bool(self.source_recorder_id) != bool(self.source_experiment_name):
            raise ValueError("source recorder and experiment identity must be provided together")
        if self.source_operation is not None and self.source_operation not in KNOWN_OPERATIONS:
            raise ValueError("unsupported source operation")
        if self.source_recorder_id == self.recorder_id:
            raise ValueError("record cannot source itself")

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRecordEntry":
        return cls(**dict(value))


@dataclass(frozen=True)
class ModelRecordOutcome:
    key: str
    requested_operation: str
    outcome: str
    entry: Optional[ModelRecordEntry] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None

    def __post_init__(self):
        split_model_key(self.key)
        if self.outcome not in ("success", "failed", "skipped", "preserved"):
            raise ValueError("unsupported record outcome")
        if self.outcome == "success" and self.entry is None:
            raise ValueError("successful outcome requires an entry")


@dataclass(frozen=True)
class ResolvedModelRecord:
    key: str
    recorder_id: str
    experiment_name: str
    status: str
    entry: Optional[ModelRecordEntry]
    record_schema: int


@dataclass(frozen=True)
class TrainingRecordSnapshot:
    entries: Tuple[ModelRecordEntry, ...]
    updated_at: Optional[str] = None
    schema_version: int = 2

    def __post_init__(self):
        keys = [entry.key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate current model record")
        _validate_source_lineage(self.entries)

    @property
    def entry_map(self) -> Dict[str, ModelRecordEntry]:
        return {entry.key: entry for entry in self.entries}

    def to_dict(self) -> Dict[str, Any]:
        entries = self.entry_map
        modes: Dict[str, set] = {mode: set() for mode in KNOWN_MODES}
        for entry in entries.values():
            modes[entry.training_mode].add(entry.experiment_name)
        all_experiments = {entry.experiment_name for entry in entries.values()}
        anchors = [entry.prediction_end or entry.requested_anchor for entry in entries.values()]
        anchors = sorted(value for value in anchors if value)
        updated_at = self.updated_at or datetime.now().replace(microsecond=0).isoformat()
        result: Dict[str, Any] = {
            "schema_version": 2,
            "updated_at": updated_at,
            "timestamp": updated_at.replace("T", " "),
            "models": {key: entries[key].recorder_id for key in sorted(entries)},
            "model_records": {key: entries[key].to_dict() for key in sorted(entries)},
            "experiment_name": next(iter(all_experiments)) if len(all_experiments) == 1 else "",
            "anchor_date": anchors[-1] if anchors else "",
        }
        for mode in KNOWN_MODES:
            values = modes[mode]
            result["%s_experiment_name" % mode] = next(iter(values)) if len(values) == 1 else ""
        return result


def legacy_entry(key: str, recorder_id: str, records: Mapping[str, Any]) -> ModelRecordEntry:
    if "@" not in key:
        key = "%s@static" % key
    model_name, mode = split_model_key(key)
    experiment = records.get("%s_experiment_name" % mode) or records.get("experiment_name") or ""
    if not experiment:
        raise ValueError("missing experiment identity for %s" % key)
    return ModelRecordEntry(
        key=key, model_name=model_name, training_mode=mode, operation="legacy_import",
        status="legacy_unverified", recorder_id=str(recorder_id), experiment_name=str(experiment),
        requested_anchor=records.get("anchor_date") or None,
    )


def snapshot_from_dict(records: Mapping[str, Any]) -> TrainingRecordSnapshot:
    entries = []
    if int(records.get("schema_version", 1)) == 2 and isinstance(records.get("model_records"), dict):
        for key, value in records["model_records"].items():
            entry = ModelRecordEntry.from_dict(value)
            if entry.key != key:
                raise ValueError("model_records key does not match entry key")
            entries.append(entry)
        compatibility = records.get("models", {})
        projected = {entry.key: entry.recorder_id for entry in entries}
        if compatibility != projected:
            raise ValueError("models compatibility view differs from model_records")
    else:
        for key, recorder_id in records.get("models", {}).items():
            entries.append(legacy_entry(key, recorder_id, records))
    return TrainingRecordSnapshot(tuple(sorted(entries, key=lambda item: item.key)), records.get("updated_at"))


def resolve_model_record(records: Mapping[str, Any], model_key: str) -> ResolvedModelRecord:
    if int(records.get("schema_version", 1)) == 2 and isinstance(records.get("model_records"), dict):
        raw = records["model_records"].get(model_key)
        if raw is None:
            raise KeyError("model has no V2 record: %s" % model_key)
        entry = ModelRecordEntry.from_dict(raw)
        compatibility_id = records.get("models", {}).get(model_key)
        if compatibility_id != entry.recorder_id:
            raise ValueError("models compatibility recorder differs from V2 identity")
        return ResolvedModelRecord(model_key, entry.recorder_id, entry.experiment_name, entry.status, entry, 2)
    models = records.get("models", {})
    if model_key not in models:
        raise KeyError("model has no recorder ID: %s" % model_key)
    entry = legacy_entry(model_key, models[model_key], records)
    return ResolvedModelRecord(model_key, entry.recorder_id, entry.experiment_name, entry.status, entry, 1)


def build_model_record_entry(
    *, key: str, operation: str, experiment_name: str, recorder: Any,
    requested_anchor: str, dataset_test_end: Optional[str] = None,
    fit_start: Optional[str] = None, fit_end: Optional[str] = None,
    source_recorder_id: Optional[str] = None,
    source_experiment_name: Optional[str] = None,
    source_operation: Optional[str] = None,
    workspace_root: Optional[Any] = None,
    config_fingerprint: Optional[str] = None,
) -> ModelRecordEntry:
    """Build a ready entry from the exact persisted recorder output."""
    import pandas as pd

    prediction = recorder.load_object("pred.pkl")
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame("score")
    if not isinstance(prediction, pd.DataFrame) or prediction.empty:
        raise ValueError("persisted pred.pkl must be a non-empty pandas object")
    if not isinstance(prediction.index, pd.MultiIndex) or "datetime" not in prediction.index.names:
        raise ValueError("persisted pred.pkl requires a datetime MultiIndex level")
    dates = pd.to_datetime(prediction.index.get_level_values("datetime"), errors="raise")
    prediction_start = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
    prediction_end = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
    if prediction_end != requested_anchor:
        raise ValueError("persisted prediction end differs from requested anchor")
    if dataset_test_end and prediction_end != dataset_test_end:
        raise ValueError("persisted prediction end differs from dataset test end")
    info = getattr(recorder, "info", {}) or {}
    recorder_id = str(info.get("id") or info.get("run_id") or getattr(recorder, "id", ""))
    if not recorder_id:
        raise ValueError("output recorder has no identity")
    artifact_path = None
    if workspace_root is not None:
        from quantpits.runtime.mlflow_integrity import resolve_mlflow_resource_uri
        artifact_uri = str(info.get("artifact_uri") or getattr(recorder, "artifact_uri", ""))
        artifact = resolve_mlflow_resource_uri(
            artifact_uri, workspace_root=workspace_root, resource_kind="recorder"
        )
        if not artifact.contained:
            raise ValueError("output recorder artifact is outside active workspace")
        artifact_path = artifact.public_path()
    tags = {}
    try:
        tags = recorder.list_tags() or {}
    except (AttributeError, TypeError):
        pass
    model_name, mode = split_model_key(key)
    if tags.get("model") and str(tags["model"]) != model_name:
        raise ValueError("output recorder model tag differs from record identity")
    if tags.get("anchor_date") and str(tags["anchor_date"]) != requested_anchor:
        raise ValueError("output recorder anchor tag differs from requested anchor")
    return ModelRecordEntry(
        key=key, model_name=model_name, training_mode=mode, operation=operation,
        status="ready", recorder_id=recorder_id, experiment_name=experiment_name,
        experiment_id=str(info.get("experiment_id")) if info.get("experiment_id") is not None else None,
        artifact_path=artifact_path,
        requested_anchor=requested_anchor, prediction_start=prediction_start,
        prediction_end=prediction_end, prediction_rows=len(prediction),
        dataset_test_end=dataset_test_end, fit_start=fit_start, fit_end=fit_end,
        source_recorder_id=source_recorder_id,
        source_experiment_name=source_experiment_name,
        source_operation=source_operation,
        produced_at=datetime.now().replace(microsecond=0).isoformat(),
        config_fingerprint=config_fingerprint,
    )


def _validate_source_lineage(entries, max_depth=10):
    by_recorder = {entry.recorder_id: entry for entry in entries}
    for entry in entries:
        seen = {entry.recorder_id}
        current = entry
        depth = 0
        while current.source_recorder_id:
            depth += 1
            if depth > max_depth:
                raise ValueError("source lineage exceeds maximum depth")
            source_id = current.source_recorder_id
            if source_id in seen:
                raise ValueError("source lineage cycle detected")
            seen.add(source_id)
            current = by_recorder.get(source_id)
            if current is None:
                break
