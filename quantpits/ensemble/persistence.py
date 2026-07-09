"""Persistence helpers for ensemble fusion outputs."""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class PredictionSaveResult:
    """Artifacts produced by saving ensemble predictions."""

    recorder_id: str
    returned_ref: str
    records_path: Path
    config_path: Path
    prediction_csv_path: Path | None = None
    compatibility_csv_path: Path | None = None


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


def save_ensemble_predictions(request: PredictionSaveRequest) -> PredictionSaveResult:
    """Save ensemble predictions, records, optional CSVs, and config snapshot."""
    from quantpits.utils.predict_utils import save_predictions_to_recorder

    ensemble_df = request.final_score.to_frame("score")
    combo_display = request.combo_name if request.combo_name else "ensemble_raw"
    record_id = save_predictions_to_recorder(
        pred=ensemble_df,
        experiment_name="Ensemble_Fusion",
        model_name=combo_display,
        tags={
            "anchor_date": request.anchor_date,
            "weight_mode": request.method,
            "combo_name": combo_display,
        },
    )
    print(f"\nEnsemble 预测已保存至 Recorder: {record_id} (Experiment: Ensemble_Fusion)")

    root = workspace_root_path(request.workspace_root)
    records_file = root / "config" / "ensemble_records.json"
    records_file.parent.mkdir(parents=True, exist_ok=True)

    ensemble_records: dict[str, Any] = {}
    if records_file.exists():
        with records_file.open("r", encoding="utf-8") as f:
            ensemble_records = json.load(f)

    ensemble_records.setdefault("combos", {})
    ensemble_records["combos"][combo_display] = record_id
    if request.is_default:
        ensemble_records["default_combo"] = combo_display
        ensemble_records["default_record_id"] = record_id

    with records_file.open("w", encoding="utf-8") as f:
        json.dump(ensemble_records, f, indent=4)

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
    )
