"""Execution primitives for ensemble fusion service orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quantpits.ensemble.types import EnsembleRunOptions, PreparedEnsembleRun


class EnsembleExecutionError(RuntimeError):
    """Base class for expected ensemble execution failures."""


class NoRequiredModelsError(EnsembleExecutionError):
    """Raised when resolved combos contain no executable model."""


class EmptyPredictionWindowError(EnsembleExecutionError):
    """Raised when prediction filtering removes every row."""


@dataclass(frozen=True)
class EnsembleExecutionContext:
    prepared: PreparedEnsembleRun
    options: EnsembleRunOptions
    execution_options: EnsembleRunOptions
    args: Any
    train_records: dict
    model_config: dict
    ensemble_config: dict
    started_at: str
    anchor_date: str
    experiment_name: str


@dataclass(frozen=True)
class LoadedPredictionBundle:
    norm_df: Any
    model_metrics: dict
    loaded_models: tuple[str, ...]


def required_models_from_combos(combos: Sequence[Any]) -> tuple[str, ...]:
    """Return the sorted model union required by resolved combos."""

    models: set[str] = set()
    for combo in combos:
        models.update(getattr(combo, "models", ()))
    return tuple(sorted(models))


def valid_models_for_combo(combo: Any, loaded_models: Sequence[str]) -> tuple[str, ...]:
    """Return combo models available in loaded predictions, preserving combo order."""

    loaded = set(loaded_models)
    return tuple(model for model in getattr(combo, "models", ()) if model in loaded)


def combo_manifest_records(combo_results: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Build manifest combo records from legacy combo result dictionaries."""

    return [
        {
            "name": item.get("name"),
            "models": item.get("models", []),
            "method": item.get("method"),
            "is_default": item.get("is_default", False),
            "pred_file": item.get("pred_file"),
        }
        for item in combo_results
    ]


def success_manifest_records(
    *,
    anchor_date: str,
    experiment_name: str,
    combo_results: Sequence[Mapping[str, Any]],
) -> dict:
    """Build the records payload for a successful ensemble fusion manifest."""

    return {
        "anchor_date": anchor_date,
        "n_combos": len(combo_results),
        "experiment_name": experiment_name,
        "combos": combo_manifest_records(combo_results),
    }
