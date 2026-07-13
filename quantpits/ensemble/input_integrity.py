"""Strict, evidence-producing prediction input loading for ensemble execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quantpits.runtime.mlflow_integrity import resolve_mlflow_resource_uri
from quantpits.ensemble.execution import EnsembleExecutionError
from quantpits.utils.predict_utils import rank_norm, zscore_norm
from quantpits.training.records import parse_training_record_document, resolve_model_record


class EnsembleInputIntegrityError(EnsembleExecutionError):
    def __init__(self, message: str, evidence: Sequence["LoadedModelEvidence"] = ()):
        super().__init__(message)
        self.evidence = tuple(evidence)


class PredictionLoadIntegrityError(EnsembleInputIntegrityError):
    pass


class PredictionFreshnessError(EnsembleInputIntegrityError):
    pass


class ComboMembershipMismatchError(EnsembleInputIntegrityError):
    pass


@dataclass(frozen=True)
class LoadedModelEvidence:
    declared_name: str
    resolved_key: str
    recorder_id: str
    experiment_name: str
    experiment_id: str
    artifact_path: str
    prediction_start: str | None
    prediction_end: str | None
    prediction_rows: int
    status: str
    error_type: str | None = None
    error_message: str | None = None
    record_schema: int = 1
    operation: str | None = None
    requested_anchor: str | None = None
    declared_prediction_end: str | None = None
    declared_prediction_start: str | None = None
    declared_prediction_rows: int | None = None
    declared_experiment_id: str | None = None
    declared_artifact_path: str | None = None
    dataset_test_end: str | None = None
    source_recorder_id: str | None = None
    source_experiment_name: str | None = None
    identity_status: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }


@dataclass(frozen=True)
class StrictPredictionBundle:
    norm_df: pd.DataFrame
    model_metrics: Mapping[str, float]
    evidence: tuple[LoadedModelEvidence, ...]

    @property
    def loaded_models(self) -> tuple[str, ...]:
        return tuple(item.resolved_key for item in self.evidence if item.status == "ready")


def _info(recorder: Any) -> dict[str, Any]:
    value = getattr(recorder, "info", {})
    return value if isinstance(value, dict) else {}


def _prediction_frame(pred: Any, model_key: str) -> pd.DataFrame:
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if not isinstance(pred, pd.DataFrame):
        raise TypeError("pred.pkl must contain a pandas Series or DataFrame")
    if pred.empty:
        raise ValueError("pred.pkl is empty")
    if len(pred.columns) > 1:
        if list(pred.columns).count("score") != 1:
            raise ValueError("multi-column pred.pkl must contain one unique 'score' column")
        pred = pred[["score"]]
    if len(pred.columns) != 1:
        raise ValueError("pred.pkl must contain exactly one score column")
    if not isinstance(pred.index, pd.MultiIndex) or "datetime" not in pred.index.names:
        raise ValueError("prediction index must be a MultiIndex containing 'datetime'")
    dates = pd.to_datetime(pred.index.get_level_values("datetime"), errors="raise")
    if dates.isna().any():
        raise ValueError("prediction datetime index contains invalid values")
    result = pred.copy()
    result.columns = [model_key]
    return result


def load_strict_prediction_bundle(
    train_records: dict,
    selected_models: Sequence[str],
    *,
    workspace_root: str | Path,
    expected_anchor: str,
    norm_method: str = "rank",
    recorder_getter=None,
) -> StrictPredictionBundle:
    """Load exactly the requested recorders or raise one aggregate integrity error."""

    if recorder_getter is None:
        from qlib.workflow import R

        recorder_getter = lambda recorder_id, experiment_name: R.get_recorder(
            recorder_id=recorder_id, experiment_name=experiment_name
        )

    frames: list[pd.DataFrame] = []
    metrics: dict[str, float] = {}
    evidence: list[LoadedModelEvidence] = []
    expected = pd.Timestamp(expected_anchor).normalize()
    # Validate the complete document once.  This prevents selecting one valid
    # entry from an otherwise malformed/cyclic V2 snapshot.
    parse_training_record_document(train_records)
    schema_version = 2 if "schema_version" in train_records else 1

    for model_key in selected_models:
        try:
            resolved = resolve_model_record(train_records, model_key)
            record_id = resolved.recorder_id
            experiment_name = resolved.experiment_name
            entry = resolved.entry
        except Exception:
            resolved = None
            entry = None
            record_id = ""
            experiment_name = ""
        base = dict(
            declared_name=model_key,
            resolved_key=model_key,
            recorder_id=record_id,
            experiment_name=experiment_name,
            experiment_id="",
            artifact_path="<external>",
            prediction_start=None,
            prediction_end=None,
            prediction_rows=0,
            record_schema=resolved.record_schema if resolved else schema_version,
            operation=entry.operation if entry else None,
            requested_anchor=entry.requested_anchor if entry else None,
            declared_prediction_end=entry.prediction_end if entry else None,
            declared_prediction_start=entry.prediction_start if entry else None,
            declared_prediction_rows=entry.prediction_rows if entry else None,
            declared_experiment_id=entry.experiment_id if entry else None,
            declared_artifact_path=entry.artifact_path if entry else None,
            dataset_test_end=entry.dataset_test_end if entry else None,
            source_recorder_id=entry.source_recorder_id if entry else None,
            source_experiment_name=entry.source_experiment_name if entry else None,
            identity_status=entry.status if entry else None,
        )
        try:
            if not record_id:
                raise KeyError("model has no recorder ID in train records")
            recorder = recorder_getter(record_id, experiment_name)
            info = _info(recorder)
            actual_id = str(info.get("id") or info.get("run_id") or record_id)
            if actual_id != record_id:
                raise ValueError("recorder ID differs from requested record ID")
            artifact_uri = str(info.get("artifact_uri") or getattr(recorder, "artifact_uri", ""))
            artifact = resolve_mlflow_resource_uri(
                artifact_uri, workspace_root=workspace_root, resource_kind="recorder"
            )
            if not artifact.contained:
                raise ValueError("recorder artifact URI is outside the active workspace")
            actual_artifact_path = artifact.public_path()
            actual_experiment_id = str(info.get("experiment_id", ""))
            if entry and entry.artifact_path and entry.artifact_path != actual_artifact_path:
                raise ValueError("declared artifact path differs from persisted recorder")
            if entry and entry.experiment_id and entry.experiment_id != actual_experiment_id:
                raise ValueError("declared experiment ID differs from persisted recorder")
            pred = _prediction_frame(recorder.load_object("pred.pkl"), model_key)
            dates = pd.to_datetime(pred.index.get_level_values("datetime"))
            start = pd.Timestamp(dates.min()).normalize()
            end = pd.Timestamp(dates.max()).normalize()
            base.update(
                experiment_id=actual_experiment_id,
                artifact_path=actual_artifact_path,
                prediction_start=start.strftime("%Y-%m-%d"),
                prediction_end=end.strftime("%Y-%m-%d"),
                prediction_rows=len(pred),
            )
            if end != expected:
                evidence.append(LoadedModelEvidence(**base, status="stale"))
                continue
            if entry and entry.prediction_end and end.strftime("%Y-%m-%d") != entry.prediction_end:
                raise ValueError("declared prediction end differs from persisted pred.pkl")
            if entry and entry.prediction_start and start.strftime("%Y-%m-%d") != entry.prediction_start:
                raise ValueError("declared prediction start differs from persisted pred.pkl")
            if entry and entry.prediction_rows is not None and len(pred) != entry.prediction_rows:
                raise ValueError("declared prediction rows differ from persisted pred.pkl")
            if entry and entry.dataset_test_end and end.strftime("%Y-%m-%d") != entry.dataset_test_end:
                raise ValueError("dataset test end differs from persisted pred.pkl")
            raw_metrics = recorder.list_metrics()
            metrics[model_key] = next(
                (float(value) for key, value in raw_metrics.items() if "ICIR" in key), 0.0
            )
            frames.append(pred)
            evidence.append(LoadedModelEvidence(**base, status="ready"))
        except Exception as exc:
            raw_message = str(exc).lower()
            if isinstance(exc, KeyError):
                status = "missing_record"
            elif "outside the active workspace" in raw_message:
                status = "external_artifact"
            elif "pred.pkl" in raw_message and ("missing" in raw_message or "not found" in raw_message):
                status = "missing_pred"
            elif isinstance(exc, (TypeError, ValueError)):
                status = "invalid_schema"
            else:
                status = "recorder_error"
            evidence.append(LoadedModelEvidence(
                **base,
                status=status,
                error_type=type(exc).__name__,
                error_message=f"{status} while validating model input",
            ))

    failed = [item for item in evidence if item.status != "ready"]
    if failed:
        codes = ", ".join(f"{item.resolved_key}:{item.status}" for item in failed)
        error_cls = PredictionFreshnessError if all(item.status == "stale" for item in failed) else PredictionLoadIntegrityError
        raise error_cls(f"Strict prediction input validation failed ({codes})", evidence)

    merged = pd.concat(frames, axis=1)
    norm_func = rank_norm if norm_method == "rank" else zscore_norm
    norm_df = pd.DataFrame(index=merged.index)
    for column in merged.columns:
        norm_df[column] = norm_func(merged[column].dropna())
    if norm_method == "rank":
        norm_df = norm_df.fillna(0.5)
    if tuple(norm_df.columns) != tuple(selected_models):
        raise ComboMembershipMismatchError("Loaded prediction columns differ from required models", evidence)
    return StrictPredictionBundle(norm_df, metrics, tuple(evidence))


def assert_exact_members(required: Sequence[str], actual: Sequence[str], *, layer: str) -> None:
    required_tuple = tuple(required)
    actual_tuple = tuple(actual)
    if (
        len(required_tuple) != len(set(required_tuple))
        or len(actual_tuple) != len(set(actual_tuple))
        or set(required_tuple) != set(actual_tuple)
    ):
        raise ComboMembershipMismatchError(
            f"{layer} membership mismatch: required={list(required_tuple)}, actual={list(actual_tuple)}"
        )
