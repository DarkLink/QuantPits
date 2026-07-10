"""Backtest execution helpers for ensemble fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestExecutionRequest:
    """Inputs for ensemble backtest execution."""

    final_score: pd.Series
    top_k: int
    drop_n: int
    benchmark: str
    freq: str
    strategy_config: dict[str, Any] | None = None
    backtest_config: dict[str, Any] | None = None
    verbose: bool = False


@dataclass(frozen=True)
class BacktestPerformanceSummary:
    """Operator-facing scalar summary for a backtest report."""

    bt_start: str
    bt_end: str
    initial_cash: float
    final_nav: float
    total_return: float
    benchmark_return: float
    excess_return: float
    annualized_return: float
    max_drawdown: float
    calmar: float | None


@dataclass(frozen=True)
class BacktestExecutionResult:
    """Structured outputs from ensemble backtest execution."""

    report_df: pd.DataFrame | None
    executor_obj: Any | None
    summary: BacktestPerformanceSummary | None


def _date_label(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return str(value)
    return str(timestamp.date())


def execution_window_from_signal(final_score: pd.Series) -> tuple[str, str]:
    """Return backtest start/end dates from a prediction signal MultiIndex."""

    dates = final_score.index.get_level_values(0)
    return _date_label(dates.min()), _date_label(dates.max())


def build_backtest_performance_summary(
    report_df: pd.DataFrame,
    *,
    benchmark: str,
    freq: str,
    backtest_config: dict[str, Any],
    bt_start: str,
    bt_end: str,
) -> BacktestPerformanceSummary:
    """Calculate the operator-facing performance summary for a backtest report."""

    from quantpits.utils.backtest_utils import standard_evaluate_portfolio

    metrics = standard_evaluate_portfolio(report_df, benchmark, freq)
    annualized_return = metrics.get("CAGR_252", 0)
    max_drawdown = metrics.get("Max_Drawdown", 0)
    benchmark_return = metrics.get("Benchmark_Absolute_Return", 0)
    total_return = metrics.get("Absolute_Return", 0)
    calmar = metrics.get("Calmar", 0)

    calmar_value = None if pd.isna(calmar) else float(calmar)
    initial_cash = float(backtest_config["account"])
    final_nav = float(report_df.iloc[-1]["nav"])
    total_return_value = float(total_return)
    benchmark_return_value = float(benchmark_return)

    return BacktestPerformanceSummary(
        bt_start=bt_start,
        bt_end=bt_end,
        initial_cash=initial_cash,
        final_nav=final_nav,
        total_return=total_return_value,
        benchmark_return=benchmark_return_value,
        excess_return=total_return_value - benchmark_return_value,
        annualized_return=float(annualized_return),
        max_drawdown=float(max_drawdown),
        calmar=calmar_value,
    )


def print_backtest_performance_summary(summary: BacktestPerformanceSummary) -> None:
    """Print the legacy human-readable backtest performance report."""

    print(f'\n{"="*20} 回测绩效报告 {"="*20}')
    print(f"回测区间     : {summary.bt_start} ~ {summary.bt_end}")
    print(f"初始资金     : {summary.initial_cash:,.2f}")
    print(f"最终净值     : {summary.final_nav:,.2f}")
    print(f"策略累计收益 : {summary.total_return*100:.2f}%")
    print(
        f"基准累计收益 : {summary.benchmark_return*100:.2f}% "
        f"(超额: {summary.excess_return*100:.2f}%)"
    )
    print(f"年化收益率   : {summary.annualized_return*100:.2f}%")
    print(f"最大回撤     : {summary.max_drawdown*100:.2f}%")
    if summary.calmar is not None:
        print(f"Calmar Ratio : {summary.calmar:.4f}")


def _print_backtest_header(
    *,
    bt_start: str,
    bt_end: str,
    freq: str,
    verbose: bool,
) -> None:
    print(f"\n{'='*60}")
    print("Stage 6: 回测")
    print(f"{'='*60}")
    print(f"Backtest Range: {bt_start} ~ {bt_end}")
    print(f"Freq: {freq}")
    print(f"Verbose: {verbose}")


def run_backtest_execution(
    request: BacktestExecutionRequest,
    *,
    verbose: bool = True,
) -> BacktestExecutionResult:
    """Run Qlib backtest execution and return structured outputs."""

    from qlib.backtest.exchange import Exchange

    from quantpits.utils import strategy
    from quantpits.utils.backtest_utils import run_backtest_with_strategy

    strategy_config = request.strategy_config
    if strategy_config is None:
        strategy_config = strategy.load_strategy_config()
    backtest_config = request.backtest_config
    if backtest_config is None:
        backtest_config = strategy.get_backtest_config(strategy_config)

    bt_start, bt_end = execution_window_from_signal(request.final_score)
    if verbose:
        _print_backtest_header(
            bt_start=bt_start,
            bt_end=bt_end,
            freq=request.freq,
            verbose=request.verbose,
        )

    strategy_inst = strategy.create_backtest_strategy(request.final_score, strategy_config)

    all_codes = sorted(request.final_score.index.get_level_values(1).unique().tolist())
    exchange_kwargs = backtest_config["exchange_kwargs"].copy()
    exchange_freq = exchange_kwargs.pop("freq", "day")

    trade_exchange = Exchange(
        freq=exchange_freq,
        start_time=bt_start,
        end_time=bt_end,
        codes=all_codes,
        **exchange_kwargs,
    )

    if verbose:
        print("\n开始回测...")
    report_df, executor_obj = run_backtest_with_strategy(
        strategy_inst=strategy_inst,
        trade_exchange=trade_exchange,
        freq=request.freq,
        account_cash=backtest_config["account"],
        bt_start=bt_start,
        bt_end=bt_end,
    )

    if report_df is None:
        if verbose:
            print("【错误】未能提取回测数据")
        return BacktestExecutionResult(
            report_df=None,
            executor_obj=executor_obj,
            summary=None,
        )

    summary = build_backtest_performance_summary(
        report_df,
        benchmark=request.benchmark,
        freq=request.freq,
        backtest_config=backtest_config,
        bt_start=bt_start,
        bt_end=bt_end,
    )
    if verbose:
        print_backtest_performance_summary(summary)

    return BacktestExecutionResult(
        report_df=report_df,
        executor_obj=executor_obj,
        summary=summary,
    )


def run_backtest(
    final_score,
    top_k,
    drop_n,
    benchmark,
    freq,
    st_config=None,
    bt_config=None,
    verbose=False,
):
    """Legacy-compatible tuple-returning backtest API."""

    result = run_backtest_execution(
        BacktestExecutionRequest(
            final_score=final_score,
            top_k=top_k,
            drop_n=drop_n,
            benchmark=benchmark,
            freq=freq,
            strategy_config=st_config,
            backtest_config=bt_config,
            verbose=verbose,
        ),
        verbose=True,
    )
    return result.report_df, result.executor_obj


def run_detailed_backtest_analysis(
    executor_obj,
    combo_name,
    anchor_date,
    output_dir,
    freq,
    benchmark="SH000300",
):
    """Run the detailed ensemble backtest analysis report."""

    from quantpits.utils.backtest_report import run_detailed_backtest_analysis as _run

    return _run(executor_obj, combo_name, anchor_date, output_dir, freq, benchmark)
