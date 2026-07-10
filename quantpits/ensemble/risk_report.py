"""Risk and leaderboard reporting helpers for ensemble fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class RiskLeaderboardRequest:
    """Inputs for ensemble risk and leaderboard reporting."""

    report_df: pd.DataFrame | None
    norm_df: pd.DataFrame
    train_records: Mapping[str, Any]
    loaded_models: Sequence[str]
    freq: str
    output_dir: str | Path
    anchor_date: str
    combo_name: str | None = None


@dataclass(frozen=True)
class RiskLeaderboardResult:
    """Risk report outputs consumed by later ensemble stages."""

    all_reports: dict[str, pd.DataFrame]
    leaderboard: pd.DataFrame | None
    csv_path: Path | None


def leaderboard_output_path(
    output_dir: str | Path,
    anchor_date: str,
    *,
    combo_name: str | None = None,
) -> Path:
    """Return the legacy leaderboard CSV path."""

    suffix = f"_{combo_name}" if combo_name else ""
    return Path(output_dir) / f"leaderboard{suffix}_{anchor_date}.csv"


def calculate_safe_risk(returns, freq: str) -> dict[str, Any]:
    """Ensure Series-like input and return a flat risk dictionary."""

    from qlib.contrib.evaluate import risk_analysis

    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0] if not returns.empty else pd.Series(dtype=float)

    try:
        risk = risk_analysis(returns, freq=freq)
        if isinstance(risk, pd.DataFrame):
            risk = risk.iloc[:, 0]
        return risk.to_dict()
    except Exception as exc:
        print(f"Risk calculation failed: {exc}")
        return {}


def load_benchmark() -> str:
    """Load the configured benchmark with the legacy default."""

    from quantpits.utils import strategy

    st_config = strategy.load_strategy_config()
    return st_config.get("benchmark", "SH000300")


def build_daily_amount_frame(report_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Build the daily amount frame consumed by PortfolioAnalyzer."""

    daily_amount_df = pd.DataFrame(index=report_df.index)
    daily_amount_df["收盘价值"] = report_df["account"]
    daily_amount_df[benchmark] = (1 + report_df["bench"]).cumprod()
    if not isinstance(daily_amount_df.index, pd.DatetimeIndex):
        daily_amount_df.index = pd.to_datetime(daily_amount_df.index)
    daily_amount_df.index.name = "成交日期"
    return daily_amount_df


def upsample_daily_amount_frame(daily_amount_df: pd.DataFrame) -> pd.DataFrame:
    """Upsample the amount frame to Qlib daily calendar frequency."""

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


def calculate_portfolio_metrics(
    report_df: pd.DataFrame,
    *,
    benchmark: str,
    freq: str,
) -> dict[str, Any]:
    """Calculate PortfolioAnalyzer metrics for one report frame."""

    from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer

    daily_amount_df = build_daily_amount_frame(report_df, benchmark)
    daily_amount_df = upsample_daily_amount_frame(daily_amount_df)

    analyzer = PortfolioAnalyzer(
        daily_amount_df=daily_amount_df,
        trade_log_df=pd.DataFrame(),
        holding_log_df=pd.DataFrame(),
        benchmark_col=benchmark,
        freq=freq,
    )
    return analyzer.calculate_traditional_metrics()


def build_ensemble_risk_tables(
    report_df: pd.DataFrame,
    *,
    benchmark: str,
    freq: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the legacy ensemble risk table and leaderboard row."""

    metrics = calculate_portfolio_metrics(report_df, benchmark=benchmark, freq=freq)

    r_strat = report_df["return"]
    r_bench = report_df["bench"]
    r_excess = r_strat - r_bench
    bench_nav = (1 + r_bench).cumprod()

    risk_strat = pd.Series(
        {
            "mean": r_strat.mean(),
            "std": r_strat.std(),
            "annualized_return": metrics.get("CAGR_252", 0),
            "information_ratio": metrics.get("Information_Ratio_(Arithmetic)", 0),
            "max_drawdown": metrics.get("Max_Drawdown", 0),
        }
    )
    risk_bench = pd.Series(
        {
            "mean": r_bench.mean(),
            "std": r_bench.std(),
            "annualized_return": metrics.get("Benchmark_CAGR_252", 0),
            "information_ratio": metrics.get("Benchmark_Sharpe", 0),
            "max_drawdown": metrics.get("Benchmark_Max_Drawdown", 0),
        }
    )
    risk_excess = pd.Series(
        {
            "mean": r_excess.mean(),
            "std": r_excess.std(),
            "annualized_return": metrics.get("Excess_Return_CAGR_252", 0),
            "information_ratio": metrics.get("Information_Ratio_(Arithmetic)", 0),
            "max_drawdown": (
                report_df["account"] / report_df["account"].cummax()
                - bench_nav / bench_nav.cummax()
            ).min(),
        }
    )

    wide_df = pd.concat(
        [risk_strat, risk_bench, risk_excess],
        axis=1,
        keys=["account", "bench", "excess"],
    )

    leaderboard_row = risk_strat.to_dict()
    leaderboard_row["annualized_excess"] = metrics.get("Excess_Return_CAGR_252", 0)
    leaderboard_row["name"] = "Ensemble"
    return wide_df, leaderboard_row


def model_report_filename(freq: str) -> str:
    """Return the legacy model portfolio report artifact path."""

    freq_val = "week" if freq == "week" else "day"
    freq_suffix = "1week" if freq_val == "week" else "1day"
    return f"portfolio_analysis/report_normal_{freq_suffix}.pkl"


def evaluation_window_from_predictions(norm_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the evaluation window implied by normalized predictions."""

    datetimes = pd.to_datetime(norm_df.index.get_level_values("datetime"))
    return datetimes.min(), datetimes.max()


def load_model_report(
    train_records: Mapping[str, Any],
    model_name: str,
    report_filename: str,
) -> pd.DataFrame:
    """Load one model's historical portfolio report from its Qlib recorder."""

    from qlib.workflow import R
    from quantpits.utils.train_utils import get_experiment_name_for_model

    models = train_records.get("models", {})
    record_id = models.get(model_name)
    if not record_id:
        raise KeyError(f"missing recorder id for {model_name}")

    model_exp_name = get_experiment_name_for_model(train_records, model_name)
    recorder = R.get_recorder(recorder_id=record_id, experiment_name=model_exp_name)
    return recorder.load_object(report_filename)


def crop_report_to_window(
    report_df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Crop a report to the current combo evaluation window."""

    index = pd.to_datetime(report_df.index)
    return report_df[(index >= start) & (index <= end)]


def build_model_leaderboard_row(
    model_name: str,
    report_df: pd.DataFrame,
    *,
    benchmark: str,
    freq: str,
) -> dict[str, Any] | None:
    """Build the legacy leaderboard row for one sub-model report."""

    if "return" not in report_df.columns or report_df.empty:
        return None

    metrics = calculate_portfolio_metrics(report_df, benchmark=benchmark, freq=freq)
    return {
        "name": model_name,
        "annualized_return": metrics.get("CAGR_252", 0),
        "annualized_excess": metrics.get("Excess_Return_CAGR_252", 0),
        "information_ratio": metrics.get("Information_Ratio_(Arithmetic)", 0),
        "max_drawdown": metrics.get("Max_Drawdown", 0),
    }


def build_leaderboard_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame | None:
    """Build the legacy sorted leaderboard DataFrame."""

    if not rows:
        return None

    leaderboard_df = pd.DataFrame(rows).set_index("name")
    leaderboard_df = leaderboard_df.apply(pd.to_numeric, errors="coerce")
    sort_col = (
        "annualized_return"
        if "annualized_return" in leaderboard_df.columns
        else leaderboard_df.columns[0]
    )
    return leaderboard_df.sort_values(sort_col, ascending=False)


def print_leaderboard(leaderboard_df: pd.DataFrame) -> None:
    """Print the legacy leaderboard display."""

    print(f"\n{'=' * 10} 绩效对比 {'=' * 10}")
    display_cols = [
        column
        for column in [
            "annualized_return",
            "annualized_excess",
            "information_ratio",
            "max_drawdown",
        ]
        if column in leaderboard_df.columns
    ]
    if display_cols:
        print(leaderboard_df[display_cols])
    else:
        print(leaderboard_df)


def build_risk_leaderboard(
    request: RiskLeaderboardRequest,
    *,
    verbose: bool = True,
) -> RiskLeaderboardResult:
    """Run risk analysis and save the legacy leaderboard CSV."""

    if verbose:
        print(f"\n{'=' * 60}")
        print("Stage 7: 风险分析 & 排行榜")
        print(f"{'=' * 60}")

    benchmark = load_benchmark()
    leaderboard_data: list[dict[str, Any]] = []
    all_reports: dict[str, pd.DataFrame] = {}

    if request.report_df is not None:
        if verbose:
            print(">>> Ensemble 模型风险分析:")
        ensemble_wide_df, ensemble_row = build_ensemble_risk_tables(
            request.report_df,
            benchmark=benchmark,
            freq=request.freq,
        )
        if verbose:
            print(ensemble_wide_df)
        leaderboard_data.append(ensemble_row)
        all_reports["Ensemble"] = request.report_df

    if verbose:
        print("\n>>> 生成子模型对比排行榜...")

    report_filename = model_report_filename(request.freq)
    eval_start, eval_end = evaluation_window_from_predictions(request.norm_df)
    models = request.train_records.get("models", {})

    for model_name in request.loaded_models:
        record_id = models.get(model_name)
        if not record_id:
            continue
        try:
            hist_report = load_model_report(
                request.train_records,
                model_name,
                report_filename,
            )
            hist_report = crop_report_to_window(
                hist_report,
                start=eval_start,
                end=eval_end,
            )
            all_reports[model_name] = hist_report

            row = build_model_leaderboard_row(
                model_name,
                hist_report,
                benchmark=benchmark,
                freq=request.freq,
            )
            if row is not None:
                leaderboard_data.append(row)
        except Exception as exc:
            if verbose:
                print(f"  [跳过] {model_name}: {exc}")

    leaderboard_df = build_leaderboard_frame(leaderboard_data)
    csv_path = None
    if leaderboard_df is not None:
        if verbose:
            print_leaderboard(leaderboard_df)

        csv_path = leaderboard_output_path(
            request.output_dir,
            request.anchor_date,
            combo_name=request.combo_name,
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard_df.to_csv(csv_path)
        if verbose:
            print(f"\n排行榜已保存: {csv_path}")

    return RiskLeaderboardResult(
        all_reports=all_reports,
        leaderboard=leaderboard_df,
        csv_path=csv_path,
    )


def risk_analysis_and_leaderboard(
    report_df,
    norm_df,
    train_records,
    loaded_models,
    freq,
    output_dir,
    anchor_date,
    combo_name=None,
):
    """Legacy-compatible risk analysis entry point."""

    result = build_risk_leaderboard(
        RiskLeaderboardRequest(
            report_df=report_df,
            norm_df=norm_df,
            train_records=train_records,
            loaded_models=loaded_models,
            freq=freq,
            output_dir=output_dir,
            anchor_date=anchor_date,
            combo_name=combo_name,
        )
    )
    return result.all_reports, result.leaderboard
