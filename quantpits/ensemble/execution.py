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
    evidence: tuple[Any, ...] = ()


def required_models_from_combos(combos: Sequence[Any]) -> tuple[str, ...]:
    """Return the sorted model union required by resolved combos."""

    models: set[str] = set()
    for combo in combos:
        models.update(getattr(combo, "models", ()))
    return tuple(sorted(models))


def valid_models_for_combo(combo: Any, loaded_models: Sequence[str]) -> tuple[str, ...]:
    """Require exact combo membership, preserving the legacy function name."""
    from quantpits.ensemble.input_integrity import assert_exact_members

    required = tuple(getattr(combo, "models", ()))
    available = tuple(model for model in loaded_models if model in set(required))
    assert_exact_members(required, available, layer=f"combo {getattr(combo, 'name', None)}")
    return required


def combo_manifest_records(combo_results: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Build manifest combo records from legacy combo result dictionaries."""

    return [
        {
            "name": item.get("name"),
            "models": item.get("models", []),
            "declared_models": item.get("declared_models", item.get("models", [])),
            "resolved_models": item.get("resolved_models", item.get("models", [])),
            "loaded_models": item.get("loaded_models", item.get("models", [])),
            "method": item.get("method"),
            "is_default": item.get("is_default", False),
            "pred_file": item.get("pred_file"),
            "recorder_id": item.get("recorder_id"),
            "output_evidence": item.get("output_evidence", {}),
        }
        for item in combo_results
    ]


def success_manifest_records(
    *,
    anchor_date: str,
    experiment_name: str,
    combo_results: Sequence[Mapping[str, Any]],
    expected_anchor: str | None = None,
    input_evidence: Sequence[Any] = (),
) -> dict:
    """Build the records payload for a successful ensemble fusion manifest."""

    return {
        "anchor_date": anchor_date,
        "n_combos": len(combo_results),
        "experiment_name": experiment_name,
        "combos": combo_manifest_records(combo_results),
        "expected_anchor": expected_anchor or anchor_date,
        "input_models": [
            item.to_public_dict() if hasattr(item, "to_public_dict") else dict(item)
            for item in input_evidence
        ],
    }
