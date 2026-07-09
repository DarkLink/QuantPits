"""Deterministic analytics helpers for ensemble fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationAnalysisRequest:
    """Inputs for prediction correlation analysis."""

    norm_df: pd.DataFrame
    output_dir: str | Path
    anchor_date: str
    combo_name: str | None = None


@dataclass(frozen=True)
class CorrelationAnalysisResult:
    """Prediction correlation analysis output."""

    matrix: pd.DataFrame
    path: Path
    average: float | None = None
    maximum: float | None = None
    minimum: float | None = None


@dataclass(frozen=True)
class ModelContributionSaveResult:
    """Saved model contribution snapshot metadata."""

    path: Path
    payload: dict[str, Any]


def compute_prediction_correlation(norm_df: pd.DataFrame) -> pd.DataFrame:
    """Return the model prediction correlation matrix."""

    return norm_df.corr()


def summarize_correlation_matrix(corr_matrix: pd.DataFrame) -> dict[str, float] | None:
    """Summarize pairwise correlations using the upper triangle only."""

    if len(corr_matrix) <= 1:
        return None

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    stacked = upper.stack()
    return {
        "average": float(stacked.mean()),
        "maximum": float(stacked.max()),
        "minimum": float(stacked.min()),
    }


def correlation_output_path(
    output_dir: str | Path,
    anchor_date: str,
    combo_name: str | None = None,
) -> Path:
    """Return the legacy correlation matrix CSV path."""

    suffix = f"_{combo_name}" if combo_name else ""
    return Path(output_dir) / f"correlation_matrix{suffix}_{anchor_date}.csv"


def run_correlation_analysis(
    request: CorrelationAnalysisRequest,
    *,
    verbose: bool = True,
) -> CorrelationAnalysisResult:
    """Compute, save, and optionally print prediction correlation analysis."""

    if verbose:
        print(f"\n{'=' * 60}")
        print("Stage 2: 相关性分析（仅选定模型）")
        print(f"{'=' * 60}")

    corr_matrix = compute_prediction_correlation(request.norm_df)
    if verbose:
        print("\n模型预测相关性矩阵:")
        print(corr_matrix.round(4))

    output_path = correlation_output_path(
        request.output_dir,
        request.anchor_date,
        combo_name=request.combo_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(output_path)
    if verbose:
        print(f"\n相关性矩阵已保存: {output_path}")

    summary = summarize_correlation_matrix(corr_matrix)
    if summary is not None and verbose:
        print(
            "\n相关性统计: "
            f"均值={summary['average']:.4f}, "
            f"最大={summary['maximum']:.4f}, "
            f"最小={summary['minimum']:.4f}"
        )

    return CorrelationAnalysisResult(
        matrix=corr_matrix,
        path=output_path,
        average=None if summary is None else summary["average"],
        maximum=None if summary is None else summary["maximum"],
        minimum=None if summary is None else summary["minimum"],
    )


def calculate_loo_contribution(
    norm_df: pd.DataFrame,
    final_score: pd.Series,
) -> dict[str, dict[str, float]]:
    """
    Calculate Leave-One-Out contribution using the legacy IC proxy.

    The current production semantics use equal-weight averages for both the
    full ensemble and each leave-one-out score. Do not change the formula here
    unless the report and ledger contracts are changed together.
    """

    models = norm_df.columns.tolist()
    if len(models) <= 1:
        return {}

    contributions = {}
    full_ensemble = norm_df.mean(axis=1)
    full_ic = float(full_ensemble.corr(final_score))

    for model_name in models:
        other_models = [model for model in models if model != model_name]
        loo_score = norm_df[other_models].mean(axis=1)
        loo_ic = float(loo_score.corr(final_score))
        delta = full_ic - loo_ic

        contributions[model_name] = {
            "loo_ic": round(loo_ic, 6),
            "full_ic": round(full_ic, 6),
            "delta": round(delta, 6),
        }

    return contributions


def build_model_contribution_payload(
    *,
    combo_name: str | None,
    anchor_date: str,
    contributions: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Build the legacy model contribution JSON payload."""

    return {
        "combo": combo_name or "default",
        "anchor_date": anchor_date,
        "method": "loo_ic_proxy",
        "contributions": contributions,
    }


def model_contribution_output_path(
    output_dir: str | Path,
    anchor_date: str,
    combo_name: str | None = None,
) -> Path:
    """Return the legacy model contribution JSON path."""

    suffix = f"_{combo_name}" if combo_name else ""
    return Path(output_dir) / f"model_contribution{suffix}_{anchor_date}.json"


def save_model_contribution_snapshot(
    *,
    output_dir: str | Path,
    anchor_date: str,
    combo_name: str | None,
    contributions: Mapping[str, Mapping[str, float]],
) -> ModelContributionSaveResult:
    """Save the legacy model contribution JSON snapshot."""

    payload = build_model_contribution_payload(
        combo_name=combo_name,
        anchor_date=anchor_date,
        contributions=contributions,
    )
    output_path = model_contribution_output_path(
        output_dir,
        anchor_date,
        combo_name=combo_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)

    return ModelContributionSaveResult(path=output_path, payload=payload)
