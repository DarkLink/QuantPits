import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from quantpits.scripts.analysis.execution_analyzer import ExecutionAnalyzer


def _make_trade_log():
    """Build a synthetic trade log with columns matching source expectations."""
    return pd.DataFrame({
        "成交日期": pd.to_datetime(["2026-01-10", "2026-01-15"]),
        "证券代码": ["SZ000001", "SZ000002"],
        "交易类别": ["买入", "卖出"],
        "成交价格": [10.5, 20.1],
        "成交数量": [100, 200],
        "成交金额": [1050.0, 4020.0],
        "费用合计": [5.0, 8.0],
        "资金发生数": [-1055.0, 4012.0],
    })


# ── analyze_explicit_costs ───────────────────────────────────────────────

def test_analyze_explicit_costs():
    trade_log = _make_trade_log()
    ea = ExecutionAnalyzer(trade_log_df=trade_log)
    result = ea.analyze_explicit_costs()
    assert result is not None
    assert "fee_ratio" in result
    assert "total_fees" in result
    assert result["total_fees"] == 13.0  # 5 + 8


def test_analyze_explicit_costs_empty():
    ea = ExecutionAnalyzer(trade_log_df=pd.DataFrame())
    result = ea.analyze_explicit_costs()
    assert result["fee_ratio"] == 0.0
    assert result["total_fees"] == 0.0


# ── slippage with mock ───────────────────────────────────────────────────

def test_slippage_with_mock():
    trade_log = _make_trade_log()
    ea = ExecutionAnalyzer(trade_log_df=trade_log)

    with patch.object(ea, 'calculate_slippage_and_delay') as mock_method:
        mock_result = trade_log.copy()
        mock_result["delay_cost"] = [0.02, -0.005]
        mock_result["exec_slippage"] = [-0.01, 0.005]
        mock_method.return_value = mock_result

        result = ea.calculate_slippage_and_delay()
        assert "delay_cost" in result.columns
        assert "exec_slippage" in result.columns
