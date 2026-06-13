"""
Tests for run_detailed_backtest_analysis in backtest_report.py.

Covers edge cases: empty portfolio metrics, position extraction errors,
sell-side trade reconstruction, missing price attribute on position units.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_unit(count, price, has_price=True):
    u = MagicMock()
    u.count = count
    if has_price:
        u.price = price
    return u


def _make_mock_pos_obj(positions):
    obj = MagicMock()
    obj.position = positions
    return obj


def _make_mock_executor(pm_source_df, hist_positions):
    pm_list = [pm_source_df] if pm_source_df is not None else []
    ta = MagicMock()
    ta.get_portfolio_metrics.return_value = pm_list
    ta.get_hist_positions.return_value = hist_positions
    ex = MagicMock()
    ex.trade_account = ta
    return ex


def _base_source_df():
    dates = pd.date_range("2026-01-05", "2026-01-09", freq="B")
    df = pd.DataFrame({
        "account": [1000000.0] * 5,
        "bench": [0.001, 0.002, -0.001, 0.003, 0.0],
    }, index=dates)
    return df


def _default_mock_pa():
    """Create a mock PortfolioAnalyzer that returns empty/default metrics."""
    pa = MagicMock()
    pa.calculate_traditional_metrics.return_value = {}
    pa.calculate_factor_exposure.return_value = {}
    pa.calculate_style_exposures.return_value = {}
    pa.calculate_holding_metrics.return_value = {}
    return pa


def _setup_and_run(mock_script_context, executor_obj, combo_name, anchor_date,
                   output_dir, freq, mock_pa_instance, tmp_path):
    """Set up mock workspace, import backtest_report, run with mocks."""
    import quantpits.utils.backtest_report as btr
    mock_script_context(btr, ["quantpits.utils.env"])

    source_df = executor_obj.trade_account.get_portfolio_metrics()[0]

    # Pre-populate sys.modules so the internal import succeeds
    fake_pa_module = MagicMock()
    fake_pa_module.PortfolioAnalyzer = MagicMock(return_value=mock_pa_instance)
    sys.modules["quantpits.scripts.analysis.portfolio_analyzer"] = fake_pa_module

    with patch("qlib.data.D") as mock_D:
        mock_D.calendar.return_value = source_df.index
        return btr.run_detailed_backtest_analysis(
            executor_obj, combo_name, anchor_date, output_dir, freq
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyPortfolioMetrics:
    def test_empty_tuple_returns_none(self, mock_script_context, tmp_path, capsys):
        """Line 25-27: empty pm_tuple → return None."""
        ex = _make_mock_executor(None, {})
        import quantpits.utils.backtest_report as btr
        mock_script_context(btr, ["quantpits.utils.env"])
        result = btr.run_detailed_backtest_analysis(
            ex, "combo1", "2026-01-10", str(tmp_path), "week"
        )
        assert result is None
        captured = capsys.readouterr()
        assert "No portfolio metrics" in captured.out


class TestPositionExtraction:
    def test_exception_during_extraction(self, mock_script_context,
                                         tmp_path, capsys):
        """Lines 76-79: exception extracting unit → count=0.0, price=0.0."""
        u = MagicMock()
        type(u).count = property(
            fget=lambda self: (_ for _ in ()).throw(Exception("boom"))
        )
        pos_obj = _make_mock_pos_obj({"SZ000001": u})
        source_df = _base_source_df()
        ex = _make_mock_executor(source_df, {"2026-01-05": pos_obj})

        mock_pa = _default_mock_pa()
        _setup_and_run(mock_script_context, ex, "combo1", "2026-01-10",
                       str(tmp_path), "week", mock_pa, tmp_path)
        captured = capsys.readouterr()
        assert "Error extracting" in captured.out


class TestSellSide:
    def test_sell_side_trade(self, mock_script_context, tmp_path):
        """Lines 105-118: curr_n < prev_n → sell trade reconstruction."""
        u1 = _make_mock_unit(100, 10.0)
        pos_day1 = _make_mock_pos_obj({"SZ000001": u1})
        u2 = _make_mock_unit(50, 12.0)
        pos_day2 = _make_mock_pos_obj({"SZ000001": u2})
        source_df = _base_source_df()
        hist = {"2026-01-05": pos_day1, "2026-01-06": pos_day2}
        ex = _make_mock_executor(source_df, hist)

        mock_pa = _default_mock_pa()
        result = _setup_and_run(mock_script_context, ex, "combo1",
                                "2026-01-10", str(tmp_path), "week",
                                mock_pa, tmp_path)
        assert result is not None
        assert os.path.exists(result)

    def test_sell_side_missing_price_attr(self, mock_script_context, tmp_path):
        """Lines 109-110: sell trade where unit has no .price attr."""
        u1 = _make_mock_unit(100, 10.0)
        pos_day1 = _make_mock_pos_obj({"SZ000001": u1})
        u2 = MagicMock()
        u2.count = 50
        pos_day2 = _make_mock_pos_obj({"SZ000001": u2})
        source_df = _base_source_df()
        hist = {"2026-01-05": pos_day1, "2026-01-06": pos_day2}
        ex = _make_mock_executor(source_df, hist)

        mock_pa = _default_mock_pa()
        result = _setup_and_run(mock_script_context, ex, "combo1",
                                "2026-01-10", str(tmp_path), "week",
                                mock_pa, tmp_path)
        assert result is not None


class TestHappyPath:
    def test_basic_happy_path(self, mock_script_context, tmp_path):
        """Full happy path: good data, writes report file."""
        u = _make_mock_unit(100, 10.0)
        pos_obj = _make_mock_pos_obj({"SZ000001": u})
        source_df = _base_source_df()
        ex = _make_mock_executor(source_df, {"2026-01-05": pos_obj})

        mock_pa = MagicMock()
        mock_pa.calculate_traditional_metrics.return_value = {
            "CAGR": 0.15, "Benchmark_CAGR": 0.08,
            "Volatility": 0.20, "Sharpe": 0.75, "Max_Drawdown": -0.10,
        }
        mock_pa.calculate_factor_exposure.return_value = {"Beta_Market": 0.95}
        mock_pa.calculate_style_exposures.return_value = {}
        mock_pa.calculate_holding_metrics.return_value = {}

        result = _setup_and_run(mock_script_context, ex, "test_combo",
                                "2026-01-10", str(tmp_path), "week",
                                mock_pa, tmp_path)
        assert result is not None
        with open(result, "r") as f:
            content = f.read()
        assert "test_combo" in content

    def test_no_combo_name(self, mock_script_context, tmp_path):
        """combo_name=None/empty → no combo suffix."""
        u = _make_mock_unit(100, 10.0)
        pos_obj = _make_mock_pos_obj({"SZ000001": u})
        source_df = _base_source_df()
        ex = _make_mock_executor(source_df, {"2026-01-05": pos_obj})

        mock_pa = MagicMock()
        mock_pa.calculate_traditional_metrics.return_value = {}
        mock_pa.calculate_factor_exposure.return_value = {}
        mock_pa.calculate_style_exposures.return_value = {}
        mock_pa.calculate_holding_metrics.return_value = {}

        result = _setup_and_run(mock_script_context, ex, "",
                                "2026-01-10", str(tmp_path), "week",
                                mock_pa, tmp_path)
        assert result is not None
        assert os.path.exists(result)

    def test_with_style_exposures(self, mock_script_context, tmp_path):
        """Full metrics + exposure + style_exp → performance attribution section."""
        u = _make_mock_unit(100, 10.0)
        pos_obj = _make_mock_pos_obj({"SZ000001": u})
        source_df = _base_source_df()
        ex = _make_mock_executor(source_df, {"2026-01-05": pos_obj})

        mock_pa = MagicMock()
        mock_pa.calculate_traditional_metrics.return_value = {
            "CAGR": 0.20, "Benchmark_CAGR": 0.10,
            "Volatility": 0.18, "Sharpe": 1.11, "Max_Drawdown": -0.08,
        }
        mock_pa.calculate_factor_exposure.return_value = {
            "Beta_Market": 0.90, "Annualized_Alpha": 0.05, "R_Squared": 0.6,
        }
        mock_pa.calculate_style_exposures.return_value = {
            "Multi_Factor_Beta": 0.85,
            "Barra_Liquidity_Exp": 0.01, "Barra_Momentum_Exp": 0.02,
            "Barra_Volatility_Exp": -0.01, "Barra_Style_R_Squared": 0.3,
            "Multi_Factor_Intercept": 0.03,
            "Factor_Annualized": {
                "size": 0.05, "momentum": 0.03, "volatility": -0.02,
            },
        }
        mock_pa.calculate_holding_metrics.return_value = {
            "Avg_Daily_Holdings_Count": 15.0,
            "Avg_Top1_Concentration": 0.25,
        }

        result = _setup_and_run(mock_script_context, ex, "full",
                                "2026-06-01", str(tmp_path), "day",
                                mock_pa, tmp_path)
        assert result is not None
        with open(result, "r") as f:
            content = f.read()
        assert "Performance Attribution" in content
        assert "Beta Return" in content
        assert "Idiosyncratic Alpha" in content
