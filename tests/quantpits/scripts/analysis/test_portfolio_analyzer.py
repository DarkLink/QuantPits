import os
import sys
import importlib
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
from quantpits.utils.constants import TRADING_DAYS_PER_YEAR


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
    # Day 0 is now dropped since it has no previous day for return calculation.
    assert len(returns) == 4

    # The new first element is Day 1 (2026-01-09): (102000 - 100000 - 0) / 100000 = 0.02
    assert np.isclose(returns.iloc[0], 0.02)

    # The new second element is Day 2 (2026-01-10): CF is -1000
    # Ret = (98000 - 102000 - (-1000)) / (102000 + (-1000)) = -3000 / 101000
    assert np.isclose(returns.iloc[1], -3000.0 / 101000.0)


def test_calculate_daily_returns_empty():
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    assert pa.calculate_daily_returns().empty


# ── calculate_traditional_metrics ────────────────────────────────────────

def test_calculate_traditional_metrics():
    da_df = _make_daily_amount_df()
    tl_df = _make_trade_log_df()
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=tl_df, holding_log_df=pd.DataFrame())

    metrics = pa.calculate_traditional_metrics()
    
    assert metrics is not None
    assert "CAGR_252" in metrics
    assert "CAGR_Calendar" in metrics
    assert "Benchmark_CAGR_252" in metrics
    assert "Sharpe" in metrics
    assert "Max_Drawdown" in metrics
    assert "Turnover_Rate_Annual" in metrics

    # Basic validity checks
    assert not pd.isna(metrics["CAGR_252"])
    assert metrics["Absolute_Return"] != 0.0


def test_calculate_traditional_metrics_empty():
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
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
        assert "Barra_Liquidity_Exp_(High-Low)" in metrics
        assert "Barra_Style_R_Squared" in metrics


# ── calculate_holding_metrics ────────────────────────────────────────────

def test_calculate_holding_metrics():
    hl_df = _make_holding_log_df()
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=hl_df)

    metrics = pa.calculate_holding_metrics()
    
    # 01-09 SZ000001, 01-10 SZ000001 (Since CASH is excluded from count)
    # 2 days total. Total count without CASH = 2. Avg count = 1.0 (1 per day)
    assert np.isclose(metrics['Avg_Daily_Holdings_Count'], 1.0)
    
    # Concentation includes CASH now:
    # 01-09: 10000 / (10000 + 92000) = 10000 / 102000
    # 01-10: 10200 / 10200 = 1.0
    expected_conc = (10000.0 / 102000.0 + 1.0) / 2.0
    assert np.isclose(metrics['Avg_Top1_Concentration'], expected_conc) 
    
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
    # TRADING_DAYS_PER_YEAR days of 0.1% return each. 
    # Total return = (1.001)^TRADING_DAYS_PER_YEAR - 1 = 0.2863 (28.63%)
    dates = pd.date_range(start="2025-01-01", periods=TRADING_DAYS_PER_YEAR, freq='D')
    # Qlib reconstructions usually have NAV. 
    nav = [100000.0 * (1.001**i) for i in range(TRADING_DAYS_PER_YEAR)]
    bench_returns = [0.0005] * TRADING_DAYS_PER_YEAR # 0.05% bench return daily
    bench_nav = (1 + pd.Series(bench_returns)).cumprod()
    
    da_df = pd.DataFrame({
        "成交日期": dates,
        "收盘价值": nav,
        "CASHFLOW": 0.0,
        "SH000300": bench_nav
    })
    
    # PortfolioAnalyzer with daily data
    pa = PortfolioAnalyzer(
        daily_amount_df=da_df, 
        trade_log_df=pd.DataFrame(), 
        holding_log_df=pd.DataFrame(), 
        benchmark_col="SH000300"
    )
    metrics = pa.calculate_traditional_metrics()
    
    # Absolute return is based on the full series of returns
    # We have TRADING_DAYS_PER_YEAR rows, which means 251 return intervals.
    abs_ret_expected = (1.001**251 - 1.0)
    assert np.isclose(metrics['Absolute_Return'], abs_ret_expected, atol=1e-5)
    
    # CAGR_252 should be (1 + abs_ret)^(TRADING_DAYS_PER_YEAR / 251) - 1
    # because 'years_252' = days / TRADING_DAYS_PER_YEAR = 251 / TRADING_DAYS_PER_YEAR.
    expected_cagr = (1 + abs_ret_expected)**(TRADING_DAYS_PER_YEAR/251) - 1
    assert np.isclose(metrics['CAGR_252'], expected_cagr, atol=1e-5)
    
    # Benchmark checks
    bench_abs_ret_expected = (1.0005**251 - 1.0)
    assert np.isclose(metrics['Benchmark_Absolute_Return'], bench_abs_ret_expected, atol=1e-5)
    
    # Bench CAGR should also use (TRADING_DAYS_PER_YEAR / 251)
    expected_bench_cagr = (1 + bench_abs_ret_expected)**(TRADING_DAYS_PER_YEAR/251) - 1
    assert np.isclose(metrics['Benchmark_CAGR_252'], expected_bench_cagr, atol=1e-5)

# ── New Tests for Missing Coverage ───────────────────────────────────────

@patch('quantpits.scripts.analysis.portfolio_analyzer.load_daily_amount', return_value=_make_daily_amount_df())
@patch('quantpits.scripts.analysis.utils.load_trade_log', return_value=_make_trade_log_df())
@patch('quantpits.scripts.analysis.portfolio_analyzer.load_holding_log', return_value=_make_holding_log_df())
def test_init_default_loaders_and_date_slicing(m_hl, m_tl, m_da):
    pa = PortfolioAnalyzer(start_date='2026-01-09', end_date='2026-01-10')
    assert len(pa.holding_log) > 0 

def test_calculate_daily_returns_variations():
    base_df = pd.DataFrame({"成交日期": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "收盘价值": [100.0, 110.0, 120.0]})
    
    df1 = base_df.copy()
    df1['今日出入金'] = [0.0, 5.0, 0.0]
    pa1 = PortfolioAnalyzer(daily_amount_df=df1)
    assert not pa1.calculate_daily_returns().empty
    
    df2 = base_df.copy()
    df2['资金发生数'] = [0.0, 5.0, 0.0]
    pa2 = PortfolioAnalyzer(daily_amount_df=df2)
    assert not pa2.calculate_daily_returns().empty
    
    df3 = pd.DataFrame({"成交日期": pd.to_datetime(["2020-01-01", "2020-01-02"]), "无关列": [1, 2]})
    pa3 = PortfolioAnalyzer(daily_amount_df=df3)
    assert pa3.calculate_daily_returns().empty

def test_calculate_traditional_metrics_edge_cases():
    da_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        "收盘价值": [100.0, 110.0, 105.0, 115.0],
        "CSI300": [100.0, 110.0, 105.0, 115.0]
    })
    tl_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "证券代码": ["A", "B", "A"],
        "交易类别": ["买入", "买入", "卖出"],
        "成交数量": [100.0, 200.0, 100.0],
        "成交价格": [10.0, 20.0, 9.0], 
        "费用合计": [0.0, 0.0, 0.0],
        "成交金额": [1000.0, 4000.0, 900.0]
    })
    
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=tl_df, holding_log_df=pd.DataFrame())
    pa.daily_amount = pa.daily_amount.drop(columns=['收盘价值'])
    
    with patch.object(pa, 'calculate_daily_returns', return_value=pd.Series([0.1, -0.05], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))):
        res = pa.calculate_traditional_metrics()
        assert not np.isnan(res['Daily_Profit_Factor_(PnL)']) 
        
    pa2 = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=tl_df, holding_log_df=pd.DataFrame())
    pa2.daily_amount = pa2.daily_amount.iloc[1:].copy() 
    res2 = pa2.calculate_traditional_metrics()
    assert res2['Tracking_Error_(Geometric)'] == 0.0
    assert np.isnan(res2['Information_Ratio_(Log_Based)'])

    pa_short = PortfolioAnalyzer(daily_amount_df=da_df.head(2).copy(), trade_log_df=tl_df, holding_log_df=pd.DataFrame())
    assert pa_short.calculate_traditional_metrics() == {}

def test_trade_metrics_and_mtm_edge_cases():
    da_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "收盘价值": [100.0, 110.0, 120.0],
        "CSI300": [100.0, 110.0, 120.0]
    })
    tl_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "证券代码": ["A", "A"],
        "交易类别": ["买入", "卖出"],
        "成交数量": [100.0, 100.0],
        "成交价格": [10.0, 15.0],
        "费用合计": [0.0, 0.0],
        "成交金额": [1000.0, 1500.0]
    })
    hl_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "证券代码": ["A", "B"],
        "持仓数量": [100.0, 100.0],
        "收盘价值": [1500.0, 2000.0]
    })

    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=tl_df, holding_log_df=hl_df, start_date='2020-01-01', end_date='2020-01-03')
    res = pa.calculate_traditional_metrics()
    assert np.isinf(res['Trade_Profit_Factor'])
    assert np.isinf(res['Trade_Profit_Factor_MTM'])

def test_calculate_exposures_empty_edge():
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    assert pa.calculate_factor_exposure() == {}
    assert pa.calculate_style_exposures() == {}

    da_df = _make_daily_amount_df()
    pa2 = PortfolioAnalyzer(daily_amount_df=da_df.head(2).copy(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=pd.DataFrame()):
        assert pa2.calculate_factor_exposure() == {}
        assert pa2.calculate_style_exposures() == {}
        
    da_df_no_bench = da_df.drop(columns=['CSI300']).head(3).copy()
    pa3 = PortfolioAnalyzer(daily_amount_df=da_df_no_bench, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=pd.DataFrame()):
        assert pa3.calculate_factor_exposure() == {}
        assert pa3.calculate_style_exposures() == {}

def test_calculate_style_exposures_fallback_market():
    da_df = _make_daily_amount_df().drop(columns=['CSI300'])
    pa = PortfolioAnalyzer(daily_amount_df=da_df, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    
    dates = pd.date_range(end="2026-01-14", periods=30, freq='B')
    num_days = len(dates)
    
    mock_features = pd.DataFrame({
        'instrument': ['SZ000001'] * num_days,
        'datetime': dates.tolist(),
        'close': np.random.uniform(10, 20, size=num_days),
        'volume': np.random.uniform(1000, 5000, size=num_days)
    })
    with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=mock_features):
        with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
            pa.calculate_style_exposures()

def test_calculate_holding_metrics_missing_float_ret():
    hl_df = _make_holding_log_df().drop(columns=['浮盈收益率'])
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=hl_df)
    res = pa.calculate_holding_metrics()
    assert np.isnan(res['Avg_Floating_Return'])
    assert np.isnan(res['Daily_Holding_Win_Rate'])

def test_calculate_classified_returns_branches(tmp_path):
    pa_empty = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    assert pa_empty.calculate_classified_returns() is None

    tl_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "证券代码": ["A", "B"],
        "交易类别": ["买入", "买入"]  
    })
    pa_no_sells = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=tl_df, holding_log_df=pd.DataFrame())
    
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame().to_csv(data_dir / "trade_classification.csv")
    
    with patch('quantpits.scripts.analysis.utils.ROOT_DIR', str(tmp_path)): 
        assert pa_no_sells.calculate_classified_returns() is None

    tl_df_2 = pd.DataFrame({
        "成交日期": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "证券代码": ["A", "B"],
        "交易类别": ["买入", "卖出"]
    })
    pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=tl_df_2, holding_log_df=pd.DataFrame(), start_date='2020-01-01', end_date='2020-01-02') 
    
    pd.DataFrame({
        "trade_date": ["2020-01-01"],
        "instrument": ["A"],
        "trade_class": ["M"]
    }).to_csv(data_dir / "trade_classification.csv", index=False)
            
    with patch('quantpits.scripts.analysis.utils.ROOT_DIR', str(tmp_path)):
        res = pa.calculate_classified_returns()
        assert res is not None
