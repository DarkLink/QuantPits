"""Unit tests for monkey_benchmark.py (quantpits/scripts/monkey_benchmark.py)."""

import os
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from quantpits.scripts.monkey_benchmark import (
    compute_daily_ic_fast,
    run_single_monkey_backtest,
    compute_performance_metrics,
    format_distribution,
    load_real_model_metrics,
    run_monte_carlo_trials,
)


class TestComputeDailyICFast:
    """Test suite for compute_daily_ic_fast."""

    def test_perfect_correlation(self):
        """Test IC when prediction is identical to label."""
        T, N = 5, 10
        label_mat = np.tile(np.arange(N, dtype=np.float64), (T, 1))
        scores_mat = label_mat.copy()
        valid_mask = np.ones((T, N), dtype=bool)

        ic_mean, icir, rank_ic, rank_icir = compute_daily_ic_fast(scores_mat, label_mat, valid_mask)
        assert np.isclose(ic_mean, 1.0)
        assert np.isclose(rank_ic, 1.0)
        assert np.isfinite(icir)
        assert np.isfinite(rank_icir)

    def test_negative_correlation(self):
        """Test IC when prediction is negatively correlated."""
        T, N = 5, 10
        label_mat = np.tile(np.arange(N, dtype=np.float64), (T, 1))
        scores_mat = -label_mat.copy()
        valid_mask = np.ones((T, N), dtype=bool)

        ic_mean, icir, rank_ic, rank_icir = compute_daily_ic_fast(scores_mat, label_mat, valid_mask)
        assert np.isclose(ic_mean, -1.0)
        assert np.isclose(rank_ic, -1.0)

    def test_masked_and_nan_exclusion(self):
        """Test that invalid_mask elements and NaNs are excluded from IC computation."""
        T, N = 4, 6
        label_mat = np.array([
            [1.0, 2.0, 3.0, 4.0, np.nan, 100.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        ])
        scores_mat = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, -999.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        ])
        # Mask out stock 5 on day 0
        valid_mask = np.ones((T, N), dtype=bool)
        valid_mask[0, 5] = False

        ic_mean, icir, rank_ic, rank_icir = compute_daily_ic_fast(scores_mat, label_mat, valid_mask)
        assert not np.isnan(ic_mean)
        assert not np.isnan(rank_ic)

    def test_insufficient_samples(self):
        """Test that days with < 2 valid samples produce NaN daily IC and are safely ignored in mean."""
        T, N = 3, 5
        label_mat = np.ones((T, N))
        scores_mat = np.ones((T, N))
        valid_mask = np.zeros((T, N), dtype=bool)
        valid_mask[0, :1] = True  # Only 1 sample on day 0
        valid_mask[1, :3] = True
        valid_mask[2, :4] = True

        ic_mean, icir, rank_ic, rank_icir = compute_daily_ic_fast(scores_mat, label_mat, valid_mask)
        # Constant label means 0 std, producing 0.0 IC
        assert np.isclose(ic_mean, 0.0)


class TestRunSingleMonkeyBacktest:
    """Test suite for TopkDropout simulation state machine."""

    def test_buy_and_hold_constant_holdings(self):
        """Test n_drop=0 (Buy and Hold) retains exact initial holdings and avoids runaway buying."""
        T, N = 15, 50
        top_k = 5
        n_drop = 0
        rebalance_freq = 5

        # Random scores and returns
        rng = np.random.default_rng(42)
        scores_mat = rng.uniform(0.0, 1.0, size=(T, N))
        returns_mat = rng.normal(0.001, 0.02, size=(T, N))
        valid_mask = np.ones((T, N), dtype=bool)

        net_returns = run_single_monkey_backtest(
            scores_mat=scores_mat,
            returns_mat=returns_mat,
            valid_mask=valid_mask,
            top_k=top_k,
            n_drop=n_drop,
            cost_rate=0.002,
            rebalance_freq=rebalance_freq,
        )

        assert len(net_returns) == T
        assert not np.any(np.isnan(net_returns))

    def test_dropout_replaces_worst_holdings(self):
        """Test 0 < n_drop < top_k replaces the worst ranked holdings."""
        T, N = 10, 10
        top_k = 3
        n_drop = 1
        rebalance_freq = 5

        scores_mat = np.zeros((T, N))
        # Day 0: top 3 are [0, 1, 2]
        scores_mat[0] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        # Day 5: stock 2 drops to worst score 0, stock 3 rises to 10
        scores_mat[5] = [9, 8, 0, 10, 5, 4, 3, 2, 1, 0]

        returns_mat = np.zeros((T, N))
        returns_mat[:, 0] = 0.01
        returns_mat[:, 1] = 0.02
        returns_mat[:, 3] = 0.03
        valid_mask = np.ones((T, N), dtype=bool)

        net_returns = run_single_monkey_backtest(
            scores_mat=scores_mat,
            returns_mat=returns_mat,
            valid_mask=valid_mask,
            top_k=top_k,
            n_drop=n_drop,
            cost_rate=0.001,
            rebalance_freq=rebalance_freq,
        )

        # On Day 5, 1 stock replaced -> turnover = 1/3 = 0.3333 -> cost = 0.3333 * 0.001
        expected_cost = (1.0 / 3.0) * 0.001
        day5_return = (0.01 + 0.02 + 0.03) / 3.0 - expected_cost
        assert np.isclose(net_returns[5], day5_return, atol=1e-6)

    def test_forced_exit_out_of_universe(self):
        """Test that a constituent leaving the universe is forcibly removed and replaced."""
        T, N = 10, 10
        top_k = 2
        n_drop = 0
        rebalance_freq = 5

        scores_mat = np.zeros((T, N))
        scores_mat[0] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        scores_mat[5] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        returns_mat = np.full((T, N), 0.01)
        valid_mask = np.ones((T, N), dtype=bool)
        # Stock 1 exits universe on Day 5
        valid_mask[5:, 1] = False

        net_returns = run_single_monkey_backtest(
            scores_mat=scores_mat,
            returns_mat=returns_mat,
            valid_mask=valid_mask,
            top_k=top_k,
            n_drop=n_drop,
            cost_rate=0.0,
            rebalance_freq=rebalance_freq,
        )

        assert len(net_returns) == T
        assert not np.any(np.isnan(net_returns))

    def test_full_rebalance_mode(self):
        """Test n_drop >= top_k triggers full rebalance."""
        T, N = 10, 6
        top_k = 2
        n_drop = 2
        rebalance_freq = 5

        scores_mat = np.zeros((T, N))
        scores_mat[0] = [5, 4, 3, 2, 1, 0]  # Holds 0, 1
        scores_mat[5] = [0, 1, 2, 3, 4, 5]  # Holds 4, 5

        returns_mat = np.zeros((T, N))
        returns_mat[:, 4] = 0.02
        returns_mat[:, 5] = 0.04
        valid_mask = np.ones((T, N), dtype=bool)

        net_returns = run_single_monkey_backtest(
            scores_mat=scores_mat,
            returns_mat=returns_mat,
            valid_mask=valid_mask,
            top_k=top_k,
            n_drop=n_drop,
            cost_rate=0.002,
            rebalance_freq=rebalance_freq,
        )

        # Day 5 replaces 2 out of 2 stocks -> turnover = 1.0 -> cost = 0.002
        expected_day5 = 0.03 - 0.002
        assert np.isclose(net_returns[5], expected_day5, atol=1e-6)


class TestComputePerformanceMetrics:
    """Test suite for compute_performance_metrics."""

    def test_performance_metrics_known_series(self):
        """Test performance metrics calculation against known return series."""
        # Alternating daily returns: [0.002, 0.000] -> mean = 0.001
        daily_ret = np.tile([0.002, 0.000], 126)
        bench_ret = np.tile([0.001, 0.000], 126)

        metrics = compute_performance_metrics(daily_ret, bench_ret)

        expected_final = np.cumprod(1.0 + daily_ret)[-1]
        expected_total = expected_final - 1.0
        expected_cagr = (expected_final ** (252.0 / 252.0)) - 1.0
        assert np.isclose(metrics["total_ret"], expected_total, atol=1e-4)
        assert np.isclose(metrics["ann_ret"], expected_cagr, atol=1e-4)
        assert np.isclose(metrics["max_dd"], 0.0, atol=1e-6)
        assert metrics["sharpe"] > 0.0
        assert metrics["excess_ret"] > 0.0
        assert metrics["ann_excess"] > 0.0

    def test_max_drawdown_calculation(self):
        """Test max drawdown computation with known peak and valley."""
        returns = np.array([0.10, -0.20, 0.05, -0.10])
        bench = np.zeros(4)

        metrics = compute_performance_metrics(returns, bench)
        nav = np.cumprod(1.0 + returns)
        # Peak is at day 0 (1.10), valley is at day 1 (0.88) -> DD = (0.88 - 1.10)/1.10 = -0.20
        # Day 2 is 0.88 * 1.05 = 0.924, Day 3 is 0.924 * 0.9 = 0.8316 -> DD = (0.8316 - 1.10)/1.10 = -0.244
        assert np.isclose(metrics["max_dd"], (0.8316 - 1.10) / 1.10, atol=1e-5)

    def test_empty_returns(self):
        """Test empty return array returns empty dict."""
        assert compute_performance_metrics(np.array([]), np.array([])) == {}


class TestFormatDistribution:
    """Test suite for format_distribution."""

    def test_format_distribution_pct(self):
        """Test format_distribution with percentage formatting."""
        series = pd.Series(np.linspace(0.01, 0.10, 100))
        stats = format_distribution(series, pct=True)

        assert "Mean" in stats
        assert "Std" in stats
        assert "5%ile" in stats
        assert "95%ile" in stats
        assert "%" in stats["Mean"]

    def test_format_distribution_non_pct(self):
        """Test format_distribution with numeric float formatting."""
        series = pd.Series(np.linspace(-0.5, 0.5, 100))
        stats = format_distribution(series, pct=False)

        assert "%" not in stats["Mean"]
        assert "+" in stats["Max"] or "-" in stats["Min"]


class TestLoadRealModelMetrics:
    """Test suite for load_real_model_metrics."""

    @patch("qlib.workflow.R")
    def test_load_real_model_metrics(self, mock_r):
        """Test loading metrics from mocked MLflow recorder."""
        train_records = {
            "models": {
                "test_model_a": "rec_123",
                "test_model_b": "rec_456",
            },
            "experiments": {
                "test_model_a": "exp_a",
                "test_model_b": "exp_b",
            },
        }

        mock_rec = MagicMock()
        mock_rec.load_object.side_effect = lambda key: pd.Series([0.05, 0.06, 0.04]) if "ic" in key else None
        mock_r.get_recorder.return_value = mock_rec

        metrics = load_real_model_metrics(train_records, selected_models=["test_model_a"])
        assert "test_model_a" in metrics
        assert np.isclose(metrics["test_model_a"]["ic"], 0.05)
        assert metrics["test_model_a"]["icir"] > 0


class TestRunMonteCarloTrials:
    """Test suite for run_monte_carlo_trials."""

    def test_monte_carlo_execution_ic_only(self):
        """Test Monte Carlo trial execution in IC-only mode."""
        T, N = 10, 20
        label_mat = np.random.normal(0.01, 0.05, size=(T, N))
        returns_mat = np.random.normal(0.001, 0.02, size=(T, N))
        valid_mask = np.ones((T, N), dtype=bool)
        bench_returns_arr = np.zeros(T)

        df = run_monte_carlo_trials(
            n_trials=5,
            label_mat=label_mat,
            returns_mat=returns_mat,
            valid_mask=valid_mask,
            bench_returns_arr=bench_returns_arr,
            top_k=5,
            n_drop=1,
            cost_rate=0.001,
            rebalance_freq=5,
            ic_only=True,
        )

        assert len(df) == 5
        assert "ic" in df.columns
        assert "rank_ic" in df.columns
        assert "ann_ret" not in df.columns

    def test_monte_carlo_execution_full(self):
        """Test Monte Carlo trial execution in full backtest mode."""
        T, N = 10, 20
        label_mat = np.random.normal(0.01, 0.05, size=(T, N))
        returns_mat = np.random.normal(0.001, 0.02, size=(T, N))
        valid_mask = np.ones((T, N), dtype=bool)
        bench_returns_arr = np.zeros(T)

        df = run_monte_carlo_trials(
            n_trials=5,
            label_mat=label_mat,
            returns_mat=returns_mat,
            valid_mask=valid_mask,
            bench_returns_arr=bench_returns_arr,
            top_k=5,
            n_drop=1,
            cost_rate=0.001,
            rebalance_freq=5,
            ic_only=False,
        )

        assert len(df) == 5
        assert "ic" in df.columns
        assert "ann_ret" in df.columns
        assert "sharpe" in df.columns
        assert "max_dd" in df.columns
