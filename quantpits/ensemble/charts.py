"""Chart generation helpers for ensemble fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ChartGenerationRequest:
    """Inputs for ensemble chart generation."""

    all_reports: dict[str, pd.DataFrame]
    report_df: pd.DataFrame | None
    final_weights: pd.DataFrame | None
    is_dynamic: bool
    freq: str
    output_dir: str | Path
    anchor_date: str
    combo_name: str | None = None


@dataclass(frozen=True)
class ChartGenerationResult:
    """Chart artifact paths written by ensemble chart generation."""

    nav_chart_path: Path | None
    weights_chart_path: Path | None


def _suffix_for_combo(combo_name: str | None) -> str:
    return f"_{combo_name}" if combo_name else ""


def ensemble_nav_chart_path(
    output_dir: str | Path,
    anchor_date: str,
    *,
    combo_name: str | None = None,
) -> Path:
    """Return the legacy ensemble NAV chart path."""

    suffix = _suffix_for_combo(combo_name)
    return Path(output_dir) / f"ensemble_nav{suffix}_{anchor_date}.png"


def ensemble_weights_chart_path(
    output_dir: str | Path,
    anchor_date: str,
    *,
    combo_name: str | None = None,
) -> Path:
    """Return the legacy dynamic weights chart path."""

    suffix = _suffix_for_combo(combo_name)
    return Path(output_dir) / f"ensemble_weights{suffix}_{anchor_date}.png"


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_nav_chart(
    request: ChartGenerationRequest,
    *,
    verbose: bool,
):
    if not request.all_reports:
        return None

    plt = _load_pyplot()
    plt.figure(figsize=(12, 6))
    plotted_any = False
    for name, r_df in request.all_reports.items():
        if "return" not in r_df.columns:
            continue
        cum_ret = (1 + r_df["return"]).cumprod()
        style = (
            {"color": "red", "linewidth": 2.5, "zorder": 10}
            if name == "Ensemble"
            else {"alpha": 0.4, "linewidth": 1}
        )
        plt.plot(cum_ret.index, cum_ret.values, label=name, **style)
        plotted_any = True

    if request.report_df is not None and "bench" in request.report_df.columns:
        bench_cum = (1 + request.report_df["bench"]).cumprod()
        plt.plot(
            bench_cum.index,
            bench_cum.values,
            label="Benchmark",
            color="black",
            linestyle="--",
            alpha=0.8,
        )
        plotted_any = True

    plt.title(f"Cumulative Return Comparison ({request.freq})")
    if plotted_any:
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = ensemble_nav_chart_path(
        request.output_dir,
        request.anchor_date,
        combo_name=request.combo_name,
    )
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"净值曲线已保存: {chart_path}")
    return chart_path


def _write_weights_chart(
    request: ChartGenerationRequest,
    *,
    verbose: bool,
):
    if not request.is_dynamic or request.final_weights is None:
        return None

    plt = _load_pyplot()
    _fig, ax = plt.subplots(figsize=(14, 6))
    request.final_weights.plot.area(ax=ax, alpha=0.7, linewidth=0.5)
    ax.set_title("Dynamic Weight Distribution (Rolling Sharpe)", fontsize=14)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = ensemble_weights_chart_path(
        request.output_dir,
        request.anchor_date,
        combo_name=request.combo_name,
    )
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"权重分布图已保存: {chart_path}")
    return chart_path


def build_chart_outputs(
    request: ChartGenerationRequest,
    *,
    verbose: bool = True,
) -> ChartGenerationResult:
    """Generate ensemble charts and return the written artifact paths."""

    if verbose:
        print(f"\n{'=' * 60}")
        print("Stage 8: 可视化")
        print(f"{'=' * 60}")

    Path(request.output_dir).mkdir(parents=True, exist_ok=True)
    nav_chart_path = _write_nav_chart(request, verbose=verbose)
    weights_chart_path = _write_weights_chart(request, verbose=verbose)
    return ChartGenerationResult(
        nav_chart_path=nav_chart_path,
        weights_chart_path=weights_chart_path,
    )


def generate_charts(
    all_reports,
    report_df,
    final_weights,
    is_dynamic,
    freq,
    anchor_date,
    output_dir,
    combo_name=None,
):
    """Generate ensemble charts using the legacy script signature."""

    build_chart_outputs(
        ChartGenerationRequest(
            all_reports=all_reports,
            report_df=report_df,
            final_weights=final_weights,
            is_dynamic=is_dynamic,
            freq=freq,
            output_dir=output_dir,
            anchor_date=anchor_date,
            combo_name=combo_name,
        )
    )
    return None
