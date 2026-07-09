"""Workspace-explicit config loading for ensemble fusion."""

from __future__ import annotations

import json
from pathlib import Path

from quantpits.ensemble.types import EnsembleRunConfig
from quantpits.utils.config_loader import load_workspace_config
from quantpits.utils.workspace import WorkspaceContext


def _resolve_record_path(ctx: WorkspaceContext, record_file: str) -> Path:
    path = Path(record_file)
    return path if path.is_absolute() else ctx.path(record_file)


def load_ensemble_run_config(
    ctx: WorkspaceContext,
    *,
    record_file: str = "latest_train_records.json",
) -> EnsembleRunConfig:
    record_path = _resolve_record_path(ctx, record_file)
    if record_path.exists():
        with record_path.open("r", encoding="utf-8") as handle:
            train_records = json.load(handle)
    else:
        train_records = {"models": {}, "experiment_name": "unknown"}

    model_config = load_workspace_config(ctx.root)

    ensemble_config = {}
    ensemble_path = ctx.config_path("ensemble_config.json")
    if ensemble_path.exists():
        with ensemble_path.open("r", encoding="utf-8") as handle:
            ensemble_config = json.load(handle)

    return EnsembleRunConfig(
        train_records=train_records,
        model_config=model_config,
        ensemble_config=ensemble_config,
    )
