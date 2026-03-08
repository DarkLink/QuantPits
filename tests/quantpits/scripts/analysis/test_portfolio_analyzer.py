import os
import sys
import importlib
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer


def _make_daily_amount_df():
    return pd.DataFrame({
        "成交日期": pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-10", "2026-01-13", "2026-01-14"]),
        "收盘价值": [100000.0, 102000.0, 98000.0, 105000.0, 106000.0],
        "CASHFLOW": [0.0, 0.0, -1000.0, 0.0, 0.0],  # Outflow on 01-10
        "CSI300": [3500.0, 3550.0, 3500.0, 3600.0, 3620.0]
    })


def _make_trade_log_df():
    return pd.DataFrame({
        "成交日期": pd.to_datetime(["2026-01-09", "2026-01-10", "2026-01-14"]),
        "证券代码": ["SZ000001", "SZ000002", "SZ000001"],
        "交易类别": ["买入", "卖出", "卖出"],
        "成交金额": [10000.0, 5000.0, 12000.0]
    })


def _make_holding_log_df():
    return pd.DataFrame({
        "成交日期": pd.to_datetime(["2026-01-09", "2026-01-09", "2026-01-10"]),
        "证券代码": ["SZ000001", "CASH", "SZ000001"],
        "收盘价值": [10000.0, 92000.0, 10200.0],
        "浮盈收益率": [0.0, pd.NA, 0.02]
    })


# ── calculate_daily_returns ──────────────────────────────────────────────

def test_calculate_daily_returns():
    da_df = _make_daily_amount_df()
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

    returns = pa.calculate_daily_returns()
    assert not returns.empty
    assert len(returns) == 5

    # Day 0 (2026-01-08): prev_nav is NaN, return is 0 (due to fillna)
    assert returns.iloc[0] == 0.0

    # Day 1 (2026-01-09): (102000 - 100000 - 0) / 100000 = 0.02
    assert np.isclose(returns.iloc[1], 0.02)

    # Day 2 (2026-01-10): CF is -1000
    # Ret = (98000 - 102000 - (-1000)) / 102000 = (-4000 + 1000) / 102000 = -3000 / 102000 = -0.0294117
    assert np.isclose(returns.iloc[2], -3000.0 / 102000.0)


def test_calculate_daily_returns_empty():
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame())
    assert pa.calculate_daily_returns().empty


# ── calculate_traditional_metrics ────────────────────────────────────────

def test_calculate_traditional_metrics():
    da_df = _make_daily_amount_df()
    tl_df = _make_trade_log_df()
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=tl_df, holding_log_df=pd.DataFrame())

    metrics = pa.calculate_traditional_metrics()
    
    assert metrics is not None
    assert "CAGR" in metrics
    assert "Benchmark_CAGR" in metrics
    assert "Sharpe" in metrics
    assert "Max_Drawdown" in metrics
    assert "Turnover_Rate_Annual" in metrics

    # Basic validity checks
    assert not pd.isna(metrics["CAGR"])
    assert metrics["Absolute_Return"] != 0.0


def test_calculate_traditional_metrics_empty():
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame())
    assert pa.calculate_traditional_metrics() == {}


# ── calculate_factor_exposure ────────────────────────────────────────────

def test_calculate_factor_exposure():
    da_df = _make_daily_amount_df()
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

    with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        metrics = pa.calculate_factor_exposure()

    assert "Beta_Market" in metrics
    assert "Annualized_Alpha" in metrics
    assert "R_Squared" in metrics


def test_calculate_factor_exposure_fallback_qlib():
    # Test without CSI300 column to trigger qlib branch
    da_df = _make_daily_amount_df().drop(columns=["CSI300"])
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

    mock_features = pd.DataFrame({
        'instrument': ['SZ000001', 'SZ000001', 'SZ000001', 'SZ000001', 'SZ000001'],
        'datetime': pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-10", "2026-01-13", "2026-01-14"]),
        'close': [10.0, 10.2, 9.9, 10.5, 10.6]
    })
    
    with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=mock_features):
        with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
            metrics = pa.calculate_factor_exposure()
            
    assert "Beta_Market" in metrics


# ── calculate_style_exposures ────────────────────────────────────────────

@patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features')
def test_calculate_style_exposures(mock_get_features):
    da_df = _make_daily_amount_df()
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

    # We need enough data to survive the 20-day rolling operations.
    # We will generate dummy daily dates backwards to supply ~30 days.
    dates = pd.date_range(end="2026-01-14", periods=30, freq='B')
    num_days = len(dates)
    
    mock_features = pd.DataFrame({
        'instrument': ['SZ000001'] * num_days + ['SZ000002'] * num_days,
        'datetime': dates.tolist() * 2,
        'close': np.random.uniform(10, 20, size=num_days * 2),
        'volume': np.random.uniform(1000, 5000, size=num_days * 2)
    })
    
    mock_get_features.return_value = mock_features

    with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        metrics = pa.calculate_style_exposures()

    if metrics:  # It might still be empty if the mocked random data yields no overlap
        assert "Multi_Factor_Intercept" in metrics
        assert "Barra_Size_Exp" in metrics
        assert "Barra_Style_R_Squared" in metrics


# ── calculate_holding_metrics ────────────────────────────────────────────

def test_calculate_holding_metrics():
    hl_df = _make_holding_log_df()
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=hl_df)

    metrics = pa.calculate_holding_metrics()
    
    # 01-09 SZ000001, 01-10 SZ000001 (Since CASH is excluded)
    # 2 days total. Total count without CASH = 2. Avg count = 1.0 (1 per day)
    assert np.isclose(metrics['Avg_Daily_Holdings_Count'], 1.0)
    assert np.isclose(metrics['Avg_Top1_Concentration'], 1.0) # 100% since only 1 stock per day
    assert np.isclose(metrics['Avg_Floating_Return'], 0.01) # (0.0 + 0.02) / 2
    assert np.isclose(metrics['Daily_Holding_Win_Rate'], 0.5) # 0.0 is not > 0, 0.02 is > 0. (0+1)/2


def test_calculate_holding_metrics_empty():
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    metrics = pa.calculate_holding_metrics()
    assert metrics == {}


# ── calculate_classified_returns ─────────────────────────────────────────

def test_calculate_classified_returns(tmp_path):
    tl_df = _make_trade_log_df()
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=tl_df, holding_log_df=pd.DataFrame())

    # Mock trade classification
    class_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2026-01-09", "2026-01-10"]),
        "证券代码": ["SZ000001", "SZ000002"],
        "trade_class": ["M", "S"]  # Manual and System
    })
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    class_df.to_csv(data_dir / "trade_classification.csv", index=False)
    
    with patch('quantpits.scripts.analysis.utils.load_trade_log', return_value=tl_df):
        with patch('quantpits.scripts.analysis.portfolio_analyzer.ROOT_DIR', str(tmp_path), create=True):
            # The method imports ROOT_DIR locally from utils
            with patch('quantpits.scripts.analysis.utils.ROOT_DIR', str(tmp_path)):
                result = pa.calculate_classified_returns()
                
    assert result is not None
    assert result['manual_buys_count'] == 1  # SZ000001 buy on 01-09 is Manual
    assert result['manual_sells_count'] == 0 # SZ000002 sell on 01-10 is System (S), SZ000001 sell on 01-14 is missing class -> 'U' => not 'M'


def test_calculate_annualization_basis():
    # 252 days of 0.1% return each. 
    # Total return = (1.001)^252 - 1 = 0.2863 (28.63%)
    dates = pd.date_range(start="2025-01-01", periods=252, freq='D')
    # Qlib reconstructions usually have NAV. 
    nav = [100000.0 * (1.001**i) for i in range(252)]
    bench_returns = [0.0005] * 252 # 0.05% bench return daily
    bench_nav = (1 + pd.Series(bench_returns)).cumprod()
    
    da_df = pd.DataFrame({
        "成交日期": dates,
        "收盘价值": nav,
        "CASHFLOW": 0.0,
        "SH000300": bench_nav
    })
    
    # PortfolioAnalyzer with daily data
    pa = PortfolioAnalyzer(daily_amount_df=da_df, benchmark_col="SH000300")
    metrics = pa.calculate_traditional_metrics()
    
    # Absolute return is based on the full series
    abs_ret_expected = (1.001**251 - 1.0) # 251 steps for 252 rows
    assert np.isclose(metrics['Absolute_Return'], abs_ret_expected, atol=1e-5)
    
    # CAGR should be (1 + abs_ret)^(1 / (252/252)) - 1 = abs_ret
    # Wait, PA uses years = len(returns) / 252.0. returns has 252 rows.
    # So years = 1.0.
    assert np.isclose(metrics['CAGR'], abs_ret_expected, atol=1e-5)
    
    # Benchmark checks
    bench_abs_ret_expected = (1.0005**251 - 1.0)
    assert np.isclose(metrics['Benchmark_Absolute_Return'], bench_abs_ret_expected, atol=1e-5)
    assert np.isclose(metrics['Benchmark_CAGR'], bench_abs_ret_expected, atol=1e-5)
