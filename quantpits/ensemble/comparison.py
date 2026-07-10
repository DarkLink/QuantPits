"""Combo comparison reporting helpers for ensemble fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ComboComparisonRequest:
    """Inputs for multi-combo comparison reporting."""

    combo_results: Sequence[Mapping[str, Any]]
    anchor_date: str
    output_dir: str | Path
    freq: str


@dataclass(frozen=True)
class ComboComparisonResult:
    """Saved multi-combo comparison report metadata."""

    frame: pd.DataFrame
    csv_path: Path
    chart_path: Path | None


def combo_comparison_csv_path(output_dir: str | Path, anchor_date: str) -> Path:
    """Return the legacy combo comparison CSV path."""

    return Path(output_dir) / f"combo_comparison_{anchor_date}.csv"


def combo_comparison_chart_path(output_dir: str | Path, anchor_date: str) -> Path:
    """Return the legacy combo comparison chart path."""

    return Path(output_dir) / f"combo_comparison_{anchor_date}.png"


def build_base_combo_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the legacy base comparison row for one combo."""

    return {
        "combo": result["name"],
        "models": ", ".join(result["models"]),
        "method": result["method"],
        "is_default": result["is_default"],
    }


def build_daily_amount_frame(report_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Build the daily amount frame consumed by PortfolioAnalyzer."""

    da_df = pd.DataFrame(index=report_df.index)
    da_df["收盘价值"] = report_df["account"]
    da_df[benchmark] = (1 + report_df["bench"]).cumprod()
    if not isinstance(da_df.index, pd.DatetimeIndex):
        da_df.index = pd.to_datetime(da_df.index)
    da_df.index.name = "成交日期"
    return da_df


def upsample_daily_amount_frame(daily_amount_df: pd.DataFrame) -> pd.DataFrame:
    """Upsample the legacy amount frame to daily frequency using Qlib calendar."""

    from qlib.data import D

    bt_start_dt = daily_amount_df.index.min()
    bt_end_dt = daily_amount_df.index.max()
    daily_dates = D.calendar(start_time=bt_start_dt, end_time=bt_end_dt, freq="day")
    return (
        daily_amount_df.reindex(daily_dates, method="ffill")
        .dropna(subset=["收盘价值"])
        .reset_index()
        .rename(columns={"index": "成交日期"})
    )


def calculate_combo_report_metrics(
    report_df: pd.DataFrame,
    *,
    freq: str,
    benchmark: str,
) -> dict[str, Any]:
    """Calculate legacy report metrics for one combo result."""

    from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer

    da_df = build_daily_amount_frame(report_df, benchmark)
    da_df = upsample_daily_amount_frame(da_df)

    analyzer = PortfolioAnalyzer(
        daily_amount_df=da_df,
        trade_log_df=pd.DataFrame(),
        holding_log_df=pd.DataFrame(),
        benchmark_col=benchmark,
        freq=freq,
    )
    metrics = analyzer.calculate_traditional_metrics()

    return {
        "total_return": round(metrics.get("Absolute_Return", 0) * 100, 2),
        "annualized_return": round(metrics.get("CAGR_252", 0) * 100, 2),
        "annualized_excess": round(metrics.get("Excess_Return_CAGR_252", 0) * 100, 2),
        "max_drawdown": round(metrics.get("Max_Drawdown", 0) * 100, 2),
        "calmar_ratio": round(
            metrics.get("Calmar", 1.0) if not pd.isna(metrics.get("Calmar")) else 0.0,
            4,
        ),
        "excess_return": round(
            metrics.get("Absolute_Return", 0) * 100
            - metrics.get("Benchmark_Absolute_Return", 0) * 100,
            2,
        ),
    }


def build_combo_comparison_frame(combo_results, *, freq: str) -> pd.DataFrame:
    """Build the legacy multi-combo comparison table."""

    comparison_data = []
    benchmark = None
    for result in combo_results:
        row = build_base_combo_row(result)
        report_df = result.get("report_df")
        if report_df is not None:
            if benchmark is None:
                from quantpits.utils import strategy

                st_config = strategy.load_strategy_config()
                benchmark = st_config.get("benchmark", "SH000300")
            row.update(
                calculate_combo_report_metrics(
                    report_df,
                    freq=freq,
                    benchmark=benchmark,
                )
            )
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def save_combo_comparison_chart(
    combo_results,
    *,
    output_dir: str | Path,
    anchor_date: str,
    verbose: bool = True,
) -> Path | None:
    """Save the legacy cumulative return comparison chart."""

    has_reports = [result for result in combo_results if result.get("report_df") is not None]
    if not has_reports:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 7))
    for result in has_reports:
        report_df = result["report_df"]
        if "return" in report_df.columns:
            cum_ret = (1 + report_df["return"]).cumprod()
            label = f"{result['name']}{'  ★' if result['is_default'] else ''}"
            linewidth = 2.5 if result["is_default"] else 1.2
            plt.plot(cum_ret.index, cum_ret.values, label=label, linewidth=linewidth)

    first_report = has_reports[0]["report_df"]
    if "bench" in first_report.columns:
        bench_cum = (1 + first_report["bench"]).cumprod()
        plt.plot(
            bench_cum.index,
            bench_cum.values,
            label="Benchmark",
            color="black",
            linestyle="--",
            alpha=0.7,
        )

    plt.title(f"Combo Comparison - Cumulative Returns ({anchor_date})")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = combo_comparison_chart_path(output_dir, anchor_date)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"对比图已保存: {chart_path}")

    return chart_path


def run_combo_comparison(
    request: ComboComparisonRequest,
    *,
    verbose: bool = True,
) -> ComboComparisonResult:
    """Build and save the legacy multi-combo comparison report."""

    if verbose:
        print(f"\n{'#' * 60}")
        print("# 跨组合对比")
        print(f"{'#' * 60}")

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_combo_comparison_frame(request.combo_results, freq=request.freq)
    csv_path = combo_comparison_csv_path(output_dir, request.anchor_date)
    frame.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n对比表已保存: {csv_path}")
        print(f"\n{'=' * 20} 组合对比 {'=' * 20}")

        display_cols = [
            col
            for col in [
                "combo",
                "is_default",
                "total_return",
                "annualized_return",
                "annualized_excess",
                "max_drawdown",
                "calmar_ratio",
                "excess_return",
            ]
            if col in frame.columns
        ]
        if display_cols:
            print(frame[display_cols].to_string(index=False))

    chart_path = save_combo_comparison_chart(
        request.combo_results,
        output_dir=output_dir,
        anchor_date=request.anchor_date,
        verbose=verbose,
    )

    return ComboComparisonResult(frame=frame, csv_path=csv_path, chart_path=chart_path)


def compare_combos(
    combo_results,
    anchor_date: str,
    output_dir: str | Path,
    freq: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Legacy-compatible entry point returning only the comparison frame."""

    result = run_combo_comparison(
        ComboComparisonRequest(
            combo_results=combo_results,
            anchor_date=anchor_date,
            output_dir=output_dir,
            freq=freq,
        ),
        verbose=verbose,
    )
    return result.frame
