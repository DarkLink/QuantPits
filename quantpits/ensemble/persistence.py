"""Persistence helpers for ensemble fusion outputs."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PredictionSaveRequest:
    """Inputs required to persist an ensemble prediction run."""

    final_score: Any
    anchor_date: str
    experiment_name: str
    method: str
    model_names: tuple[str, ...]
    model_metrics: Mapping[str, float]
    static_weights: Mapping[str, float] | None
    is_dynamic: bool
    output_dir: str | Path
    combo_name: str | None = None
    is_default: bool = False
    prediction_dir: str | Path | None = None
    save_csv: bool = False
    workspace_root: str | Path | None = None
    source_recorders: Mapping[str, str] | None = None
    source_anchors: Mapping[str, str] | None = None
    run_id: str | None = None
    plan_fingerprint: str | None = None


@dataclass(frozen=True)
class PredictionSaveResult:
    """Artifacts produced by saving ensemble predictions."""

    recorder_id: str
    returned_ref: str
    records_path: Path
    config_path: Path
    prediction_csv_path: Path | None = None
    compatibility_csv_path: Path | None = None
    output_evidence: Mapping[str, Any] | None = None


def workspace_root_path(workspace_root: str | Path | None = None) -> Path:
    """Resolve a workspace root without binding writes to the process cwd."""
    if workspace_root is None:
        from quantpits.utils import env

        return env.get_workspace_context().root
    return Path(workspace_root).expanduser().resolve()


def workspace_bound_path(workspace_root: Path, path_value: str | Path) -> Path:
    """Bind relative paths to the workspace root while preserving absolute paths."""
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else workspace_root / candidate


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def inspect_saved_recorder(
    record_id: str,
    *,
    workspace_root: Path,
    experiment_name: str,
    expected_tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify an exact output run before any local pointer is updated."""
    from mlflow.tracking import MlflowClient
    from quantpits.runtime.mlflow_integrity import resolve_mlflow_resource_uri

    client = MlflowClient()
    run = client.get_run(record_id)
    experiment = client.get_experiment(run.info.experiment_id)
    if experiment.name != experiment_name:
        raise ValueError("output recorder belongs to an unexpected experiment")
    artifact = resolve_mlflow_resource_uri(
        run.info.artifact_uri, workspace_root=workspace_root, resource_kind="recorder"
    )
    if not artifact.contained:
        raise ValueError("output recorder artifact URI is outside the active workspace")
    artifacts = {item.path for item in client.list_artifacts(record_id)}
    if "pred.pkl" not in artifacts:
        raise ValueError("output recorder does not contain pred.pkl")
    tags = dict(run.data.tags)
    for key, expected in (expected_tags or {}).items():
        if tags.get(key) != expected:
            raise ValueError(f"output recorder tag mismatch: {key}")
    return {
        "recorder_id": record_id,
        "experiment_name": experiment.name,
        "experiment_id": str(run.info.experiment_id),
        "artifact_path": artifact.public_path(),
        "contained": True,
    }


def save_ensemble_predictions(request: PredictionSaveRequest, *, output_inspector=None) -> PredictionSaveResult:
    """Save ensemble predictions, records, optional CSVs, and config snapshot."""
    from quantpits.utils.predict_utils import save_predictions_to_recorder

    ensemble_df = request.final_score.to_frame("score")
    combo_display = request.combo_name if request.combo_name else "ensemble_raw"
    root = workspace_root_path(request.workspace_root)
    workspace_fingerprint = hashlib.sha256(root.as_posix().encode("utf-8")).hexdigest()[:12]
    recorder_tags = {
        "anchor_date": request.anchor_date,
        "weight_mode": request.method,
        "combo_name": combo_display,
        "workspace_fingerprint": workspace_fingerprint,
    }
    if request.run_id:
        recorder_tags["run_id"] = request.run_id
    if request.plan_fingerprint:
        recorder_tags["plan_fingerprint"] = request.plan_fingerprint
    record_id = save_predictions_to_recorder(
        pred=ensemble_df,
        experiment_name="Ensemble_Fusion",
        model_name=combo_display,
        tags=recorder_tags,
    )
    print(f"\nEnsemble 预测已保存至 Recorder: {record_id} (Experiment: Ensemble_Fusion)")

    output_evidence = None
    if output_inspector is not None:
        output_evidence = output_inspector(
            record_id,
            workspace_root=root,
            experiment_name="Ensemble_Fusion",
            expected_tags={
                "anchor_date": request.anchor_date,
                "weight_mode": request.method,
                "combo_name": combo_display,
                "mode": "fused_prediction",
                "workspace_fingerprint": workspace_fingerprint,
                **({"run_id": request.run_id} if request.run_id else {}),
                **({"plan_fingerprint": request.plan_fingerprint} if request.plan_fingerprint else {}),
            },
        )
    records_file = root / "config" / "ensemble_records.json"
    records_file.parent.mkdir(parents=True, exist_ok=True)

    ensemble_records: dict[str, Any] = {}
    if records_file.exists():
        with records_file.open("r", encoding="utf-8") as f:
            ensemble_records = json.load(f)

    ensemble_records.setdefault("combos", {})
    ensemble_records.setdefault("combo_meta", {})
    ensemble_records["_schema_version"] = max(int(ensemble_records.get("_schema_version", 1)), 2)
    ensemble_records["combos"][combo_display] = record_id
    ensemble_records["combo_meta"][combo_display] = {
        "record_id": record_id,
        "anchor_date": request.anchor_date,
        "models": list(request.model_names),
        "source_recorders": dict(request.source_recorders or {}),
        "source_anchors": dict(request.source_anchors or {}),
        "experiment_id": (output_evidence or {}).get("experiment_id"),
        "artifact_path": (output_evidence or {}).get("artifact_path"),
        "run_id": request.run_id,
        "plan_fingerprint": request.plan_fingerprint,
    }
    if request.is_default:
        ensemble_records["default_combo"] = combo_display
        ensemble_records["default_record_id"] = record_id

    _atomic_write_json(records_file, ensemble_records)

    pred_file: Path | None = None
    compat_file: Path | None = None
    if request.save_csv:
        pred_dir = workspace_bound_path(
            root,
            request.prediction_dir or Path("output") / "predictions",
        )
        pred_dir.mkdir(parents=True, exist_ok=True)
        if request.combo_name:
            pred_file = pred_dir / f"ensemble_{request.combo_name}_{request.anchor_date}.csv"
        else:
            pred_file = pred_dir / f"ensemble_{request.anchor_date}.csv"

        ensemble_df.to_csv(pred_file)
        print(f"Ensemble CSV 已额外保存: {pred_file}")

        if request.combo_name and request.is_default:
            compat_file = pred_dir / f"ensemble_{request.anchor_date}.csv"
            ensemble_df.to_csv(compat_file)

    output_dir = workspace_bound_path(root, request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_out: dict[str, Any] = {
        "recorder_id": record_id,
        "anchor_date": request.anchor_date,
        "experiment_name": request.experiment_name,
        "weight_mode": request.method,
        "models_used": list(request.model_names),
        "model_metrics": {m: round(v, 4) for m, v in request.model_metrics.items()},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_recorders": dict(request.source_recorders or {}),
        "source_anchors": dict(request.source_anchors or {}),
        "output_recorder": dict(output_evidence or {}),
        "run_id": request.run_id,
        "plan_fingerprint": request.plan_fingerprint,
    }
    if request.combo_name:
        config_out["combo_name"] = request.combo_name
        config_out["is_default"] = request.is_default
    if not request.is_dynamic and request.static_weights:
        config_out["static_weights"] = {
            m: round(w, 4) for m, w in request.static_weights.items()
        }

    suffix = f"_{request.combo_name}" if request.combo_name else ""
    config_file = output_dir / f"ensemble_fusion_config{suffix}_{request.anchor_date}.json"
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=4, ensure_ascii=False)
    print(f"Config 已保存: {config_file}")

    returned_ref = str(pred_file) if pred_file else record_id
    return PredictionSaveResult(
        recorder_id=record_id,
        returned_ref=returned_ref,
        records_path=records_file,
        config_path=config_file,
        prediction_csv_path=pred_file,
        compatibility_csv_path=compat_file,
        output_evidence=output_evidence,
    )
