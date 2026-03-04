import os
import sys
import importlib
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer


def _make_daily_amount(n_days=200):
    """Build a synthetic daily amount log with proper datetime index."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    nav = 1_000_000 * np.cumprod(1 + rng.standard_normal(n_days) * 0.005)
    df = pd.DataFrame({
        "成交日期": dates,
        "收盘价值": nav,
        "CASHFLOW": [0.0] * n_days,
    })
    return df


def _make_trade_log():
    """Build a synthetic trade log."""
    return pd.DataFrame({
        "成交日期": pd.to_datetime(["2025-02-10", "2025-02-15", "2025-03-01"]),
        "证券代码": ["SZ000001", "SZ000002", "SZ000001"],
        "交易类别": ["买入", "买入", "卖出"],
        "成交价格": [10.0, 20.0, 11.0],
        "成交数量": [100, 200, 100],
        "成交金额": [1000.0, 4000.0, 1100.0],
    })


def _make_holding_log():
    """Build a synthetic holding log."""
    rows = []
    for d in pd.bdate_range("2025-02-10", periods=20):
        rows.append({
            "成交日期": d,
            "证券代码": "SZ000001",
            "持仓数量": 100,
            "收盘价值": 1050.0,
            "浮盈收益率": 0.05,
        })
        rows.append({
            "成交日期": d,
            "证券代码": "CASH",
            "持仓数量": 0,
            "收盘价值": 500000.0,
            "浮盈收益率": 0,
        })
    return pd.DataFrame(rows)


# ── calculate_daily_returns ──────────────────────────────────────────────

def test_calculate_daily_returns():
    daily = _make_daily_amount()
    pa = PortfolioAnalyzer(daily_amount_df=daily, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    ret = pa.calculate_daily_returns()
    assert not ret.empty
    assert len(ret) > 0


# ── calculate_traditional_metrics ────────────────────────────────────────

def test_calculate_traditional_metrics():
    daily = _make_daily_amount(n_days=200)
    pa = PortfolioAnalyzer(daily_amount_df=daily, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    metrics = pa.calculate_traditional_metrics()
    assert metrics  # Should not be empty
    assert "CAGR" in metrics or "cagr" in str(metrics).lower()


# ── calculate_holding_metrics ────────────────────────────────────────────

def test_calculate_holding_metrics():
    daily = _make_daily_amount()
    holding = _make_holding_log()
    pa = PortfolioAnalyzer(daily_amount_df=daily, trade_log_df=pd.DataFrame(), holding_log_df=holding)
    metrics = pa.calculate_holding_metrics()
    assert metrics is not None
    assert "Avg_Daily_Holdings_Count" in metrics
    assert "Daily_Holding_Win_Rate" in metrics


def test_calculate_holding_metrics_empty():
    daily = _make_daily_amount()
    pa = PortfolioAnalyzer(daily_amount_df=daily, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
    metrics = pa.calculate_holding_metrics()
    assert metrics == {}
