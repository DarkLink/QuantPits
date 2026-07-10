from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantpits.ensemble.backtest import (
    BacktestExecutionRequest,
    build_backtest_performance_summary,
    execution_window_from_signal,
    run_backtest,
    run_backtest_execution,
    run_detailed_backtest_analysis,
)


def _signal(date_values=None):
    dates = date_values or pd.to_datetime(["2020-01-01", "2020-01-02"])
    index = pd.MultiIndex.from_product(
        [dates, ["B", "A"]],
        names=["datetime", "instrument"],
    )
    return pd.Series([0.2, 0.1, 0.4, 0.3], index=index)


def _report_frame():
    return pd.DataFrame(
        {
            "account": [100000.0, 105000.0],
            "nav": [100000.0, 105000.0],
            "return": [0.0, 0.05],
            "bench": [0.0, 0.02],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )


def test_execution_window_from_signal_uses_multiindex_dates():
    assert execution_window_from_signal(_signal()) == ("2020-01-01", "2020-01-02")
    assert execution_window_from_signal(_signal(["2020-01-01", "2020-01-02"])) == (
        "2020-01-01",
        "2020-01-02",
    )


@patch("quantpits.utils.backtest_utils.standard_evaluate_portfolio")
def test_build_backtest_performance_summary_uses_standard_metrics(mock_evaluate):
    mock_evaluate.return_value = {
        "CAGR_252": 0.12,
        "Max_Drawdown": -0.08,
        "Benchmark_Absolute_Return": 0.04,
        "Absolute_Return": 0.10,
        "Calmar": 1.5,
    }

    summary = build_backtest_performance_summary(
        _report_frame(),
        benchmark="SH000300",
        freq="day",
        backtest_config={"account": 100000.0},
        bt_start="2020-01-01",
        bt_end="2020-01-02",
    )

    assert summary.initial_cash == 100000.0
    assert summary.final_nav == 105000.0
    assert summary.total_return == 0.10
    assert summary.benchmark_return == 0.04
    assert summary.excess_return == pytest.approx(0.06)
    assert summary.calmar == 1.5
    mock_evaluate.assert_called_once()


@patch("qlib.backtest.exchange.Exchange")
@patch("quantpits.utils.backtest_utils.standard_evaluate_portfolio")
@patch("quantpits.utils.backtest_utils.run_backtest_with_strategy")
@patch("quantpits.utils.strategy.create_backtest_strategy")
def test_run_backtest_execution_builds_exchange_and_returns_result(
    mock_create_strategy,
    mock_run_backtest,
    mock_evaluate,
    mock_exchange,
):
    strategy_inst = MagicMock()
    executor = MagicMock()
    mock_create_strategy.return_value = strategy_inst
    mock_run_backtest.return_value = (_report_frame(), executor)
    mock_evaluate.return_value = {
        "CAGR_252": 0.12,
        "Max_Drawdown": -0.08,
        "Benchmark_Absolute_Return": 0.04,
        "Absolute_Return": 0.10,
        "Calmar": 1.5,
    }
    backtest_config = {
        "account": 100000.0,
        "exchange_kwargs": {"freq": "week", "limit_threshold": 0.095},
    }

    result = run_backtest_execution(
        BacktestExecutionRequest(
            final_score=_signal(),
            top_k=50,
            drop_n=5,
            benchmark="SH000300",
            freq="day",
            strategy_config={"strategy": "topk"},
            backtest_config=backtest_config,
            verbose=False,
        ),
        verbose=False,
    )

    assert result.report_df is not None
    assert result.executor_obj is executor
    assert result.summary is not None
    assert result.summary.bt_start == "2020-01-01"
    assert result.summary.bt_end == "2020-01-02"
    mock_exchange.assert_called_once_with(
        freq="week",
        start_time="2020-01-01",
        end_time="2020-01-02",
        codes=["A", "B"],
        limit_threshold=0.095,
    )
    assert backtest_config["exchange_kwargs"] == {"freq": "week", "limit_threshold": 0.095}
    mock_run_backtest.assert_called_once_with(
        strategy_inst=strategy_inst,
        trade_exchange=mock_exchange.return_value,
        freq="day",
        account_cash=100000.0,
        bt_start="2020-01-01",
        bt_end="2020-01-02",
    )


@patch("qlib.backtest.exchange.Exchange")
@patch("quantpits.utils.backtest_utils.run_backtest_with_strategy")
@patch("quantpits.utils.strategy.create_backtest_strategy")
def test_run_backtest_execution_returns_none_summary_when_report_missing(
    mock_create_strategy,
    mock_run_backtest,
    mock_exchange,
):
    executor = MagicMock()
    mock_create_strategy.return_value = MagicMock()
    mock_run_backtest.return_value = (None, executor)

    result = run_backtest_execution(
        BacktestExecutionRequest(
            final_score=_signal(),
            top_k=50,
            drop_n=5,
            benchmark="SH000300",
            freq="day",
            strategy_config={},
            backtest_config={"account": 100000.0, "exchange_kwargs": {}},
        ),
        verbose=False,
    )

    assert result.report_df is None
    assert result.executor_obj is executor
    assert result.summary is None
    mock_exchange.assert_called_once()


@patch("quantpits.ensemble.backtest.run_backtest_execution")
def test_legacy_run_backtest_returns_tuple(mock_execution):
    executor = MagicMock()
    mock_execution.return_value = MagicMock(report_df=_report_frame(), executor_obj=executor)

    report_df, returned_executor = run_backtest(
        _signal(),
        top_k=50,
        drop_n=5,
        benchmark="SH000300",
        freq="day",
        st_config={"strategy": "topk"},
        bt_config={"account": 100000.0, "exchange_kwargs": {}},
        verbose=True,
    )

    assert report_df is mock_execution.return_value.report_df
    assert returned_executor is executor
    request = mock_execution.call_args.args[0]
    assert request.top_k == 50
    assert request.drop_n == 5
    assert request.verbose is True
    assert mock_execution.call_args.kwargs == {"verbose": True}


@patch("quantpits.utils.backtest_report.run_detailed_backtest_analysis")
def test_run_detailed_backtest_analysis_delegates_to_utils(mock_run):
    executor = MagicMock()
    mock_run.return_value = {"ok": True}

    result = run_detailed_backtest_analysis(
        executor,
        "combo_a",
        "2020-01-02",
        "/tmp/out",
        "day",
        benchmark="SH000905",
    )

    assert result == {"ok": True}
    mock_run.assert_called_once_with(
        executor,
        "combo_a",
        "2020-01-02",
        "/tmp/out",
        "day",
        "SH000905",
    )
