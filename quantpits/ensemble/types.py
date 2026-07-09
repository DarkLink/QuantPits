"""Typed boundaries for ensemble fusion runtime orchestration."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any

from quantpits.config_contracts.core import WorkspaceValidationResult
from quantpits.runtime import CommandPlan
from quantpits.utils.ensemble_plan import ResolvedCombo
from quantpits.utils.workspace import WorkspaceContext


@dataclass(frozen=True)
class EnsembleRunOptions:
    models: str | None = None
    from_config: bool = False
    from_config_all: bool = False
    combo: str | None = None
    method: str = "equal"
    weights: str | None = None
    freq: str | None = None
    record_file: str = "latest_train_records.json"
    training_mode: str | None = None
    output_dir: str = "output/ensemble"
    prediction_dir: str | None = None
    no_backtest: bool = False
    no_charts: bool = False
    start_date: str | None = None
    end_date: str | None = None
    only_last_years: int = 0
    only_last_months: int = 0
    detailed_analysis: bool = False
    verbose_backtest: bool = False
    save_csv: bool = False
    norm_method: str = "rank"
    explain_plan: bool = False
    json_plan: bool = False
    run_id: str | None = None
    no_manifest: bool = False


@dataclass(frozen=True)
class EnsembleRunConfig:
    train_records: dict
    model_config: dict
    ensemble_config: dict


@dataclass(frozen=True)
class PreparedEnsembleRun:
    ctx: WorkspaceContext
    options: EnsembleRunOptions
    cli_args: tuple[str, ...]
    validation_result: WorkspaceValidationResult | None
    config: EnsembleRunConfig
    combos: tuple[ResolvedCombo, ...]
    plan: CommandPlan
    plan_fingerprint: str


@dataclass(frozen=True)
class EnsembleExecutionHooks:
    init_qlib: Callable[[], None]
    load_selected_predictions: Callable[..., tuple[Any, dict, list[str]]]
    filter_norm_df_by_args: Callable[..., Any]
    run_single_combo: Callable[..., dict | None]
    compare_combos: Callable[..., None]


@dataclass(frozen=True)
class EnsembleRunSummary:
    run_id: str
    anchor_date: str
    experiment_name: str
    combo_results: tuple[dict, ...]
    manifest_path: str | None


def _field_default(name: str) -> Any:
    for item in fields(EnsembleRunOptions):
        if item.name == name:
            if item.default is not MISSING:
                return item.default
            if item.default_factory is not MISSING:  # type: ignore[attr-defined]
                return item.default_factory()  # type: ignore[misc]
    raise KeyError(name)


def options_from_namespace(args: argparse.Namespace) -> EnsembleRunOptions:
    values = {
        item.name: getattr(args, item.name, _field_default(item.name))
        for item in fields(EnsembleRunOptions)
    }
    return EnsembleRunOptions(**values)


def options_to_namespace(options: EnsembleRunOptions) -> argparse.Namespace:
    return argparse.Namespace(**asdict(options))
