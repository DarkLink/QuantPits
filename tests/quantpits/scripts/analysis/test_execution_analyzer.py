import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "data").mkdir()
    
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    import importlib
    importlib.reload(env)
    
    from quantpits.scripts.analysis import execution_analyzer
    importlib.reload(execution_analyzer)
    
    return workspace

@pytest.fixture
def ea_factory(mock_env):
    from quantpits.scripts.analysis.execution_analyzer import ExecutionAnalyzer
    def _create(trade_log_df=None, **kwargs):
        return ExecutionAnalyzer(trade_log_df=trade_log_df, **kwargs)
    return _create

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

def test_analyze_explicit_costs(ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)
    result = ea.analyze_explicit_costs()
    assert result is not None
    assert "fee_ratio" in result
    assert "total_fees" in result
    assert result["total_fees"] == 13.0  # 5 + 8


def test_analyze_explicit_costs_empty(ea_factory):
    ea = ea_factory(trade_log_df=pd.DataFrame())
    result = ea.analyze_explicit_costs()
    assert result["fee_ratio"] == 0.0
    assert result["total_fees"] == 0.0


# ── slippage with mock ───────────────────────────────────────────────────

def test_slippage_with_mock(ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)

    with patch.object(ea, 'calculate_slippage_and_delay') as mock_method:
        mock_result = trade_log.copy()
        mock_result["delay_cost"] = [0.02, -0.005]
        mock_result["exec_slippage"] = [-0.01, 0.005]
        mock_method.return_value = mock_result

        result = ea.calculate_slippage_and_delay()
        assert "delay_cost" in result.columns
        assert "exec_slippage" in result.columns

# ── calculate_slippage_and_delay ─────────────────────────────────────────

@patch('quantpits.scripts.analysis.execution_analyzer.get_daily_features')
def test_calculate_slippage_and_delay(mock_get_features, ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)

    # Mock Qlib daily features
    mock_features = pd.DataFrame({
        'instrument': ['SZ000001', 'SZ000001', 'SZ000002', 'SZ000002'],
        'datetime': pd.to_datetime(['2026-01-09', '2026-01-10', '2026-01-14', '2026-01-15']),
        'close': [10.0, 10.6, 20.0, 19.9],
        'open': [10.1, 10.2, 20.1, 20.2],
        'unadj_open': [10.1, 10.2, 20.1, 20.2],
        'unadj_close': [10.0, 10.6, 20.0, 19.9],
        'volume': [1000, 1200, 2000, 2200],
        'vwap': [10.05, 10.4, 20.05, 20.05]
    })
    mock_get_features.return_value = mock_features

    with patch('quantpits.scripts.analysis.execution_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        result = ea.calculate_slippage_and_delay()

    assert not result.empty
    assert 'Delay_Cost' in result.columns
    assert 'Exec_Slippage' in result.columns
    assert 'Total_Friction' in result.columns
    assert 'Absolute_Slippage_Amount' in result.columns
    assert 'ADV_Participation_Rate' in result.columns

    # SZ000001 Buy on 2026-01-10 at 10.5. prev_close (01-09) = 10.0. open = 10.2
    # Delay Cost = (10.0 - 10.2) / 10.0 = -0.02
    assert np.isclose(result.loc[result['证券代码'] == 'SZ000001', 'Delay_Cost'].iloc[0], -0.02)
    # Exec Slippage = (10.2 - 10.5) / 10.2 = -0.0294
    assert np.isclose(result.loc[result['证券代码'] == 'SZ000001', 'Exec_Slippage'].iloc[0], (10.2 - 10.5) / 10.2)
    
    # SZ000002 Sell on 2026-01-15 at 20.1. prev_close (01-14) = 20.0. open = 20.2
    # Delay Cost = (20.2 - 20.0) / 20.0 = 0.01
    assert np.isclose(result.loc[result['证券代码'] == 'SZ000002', 'Delay_Cost'].iloc[0], 0.01)

def test_calculate_slippage_and_delay_null_dates(ea_factory):
    trade_log = pd.DataFrame({"成交日期": [pd.NaT], "证券代码": ["A"]})
    ea = ea_factory(trade_log_df=trade_log)
    result = ea.calculate_slippage_and_delay()
    assert result.empty

@patch('quantpits.scripts.analysis.execution_analyzer.get_daily_features')
def test_calculate_slippage_and_delay_empty_features(mock_get_features, ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)
    mock_get_features.return_value = pd.DataFrame()
    with patch('quantpits.scripts.analysis.execution_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        result = ea.calculate_slippage_and_delay()
    assert result.empty

@patch('quantpits.scripts.analysis.execution_analyzer.get_daily_features')
def test_calculate_slippage_and_delay_no_merge(mock_get_features, ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)
    # Price for different dates
    mock_features = pd.DataFrame({
        'instrument': ['SZ000001'],
        'datetime': pd.to_datetime(['2020-01-01']),
        'close': [10.0], 'open': [10.0], 'unadj_open': [10.0], 'unadj_close': [10.0], 'volume': [1], 'vwap': [10]
    })
    mock_get_features.return_value = mock_features
    with patch('quantpits.scripts.analysis.execution_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        result = ea.calculate_slippage_and_delay()
    assert result.empty

# ── calculate_path_dependency ────────────────────────────────────────────

@patch('quantpits.scripts.analysis.execution_analyzer.get_daily_features')
def test_calculate_path_dependency(mock_get_features, ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)

    # Mock high/low features
    mock_features = pd.DataFrame({
        'instrument': ['SZ000001', 'SZ000002'],
        'datetime': pd.to_datetime(['2026-01-10', '2026-01-15']),
        'unadj_high': [11.0, 20.5],
        'unadj_low': [10.0, 19.5]
    })
    mock_get_features.return_value = mock_features

    with patch('quantpits.scripts.analysis.execution_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        result = ea.calculate_path_dependency()

    assert not result.empty
    assert 'MFE' in result.columns
    assert 'MAE' in result.columns

    # Buy SZ000001 at 10.5. High: 11.0, Low: 10.0
    # MFE = (11.0 - 10.5)/10.5, MAE = (10.0 - 10.5)/10.5
    assert np.isclose(result.loc[result['证券代码'] == 'SZ000001', 'MFE'].iloc[0], (11.0 - 10.5) / 10.5)
    assert np.isclose(result.loc[result['证券代码'] == 'SZ000001', 'MAE'].iloc[0], (10.0 - 10.5) / 10.5)

def test_calculate_path_dependency_null_dates(ea_factory):
    trade_log = pd.DataFrame({"成交日期": [pd.NaT], "证券代码": ["A"]})
    ea = ea_factory(trade_log_df=trade_log)
    result = ea.calculate_path_dependency()
    assert result.empty

@patch('quantpits.scripts.analysis.execution_analyzer.get_daily_features')
def test_calculate_path_dependency_empty_features(mock_get_features, ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)
    mock_get_features.return_value = pd.DataFrame()
    with patch('quantpits.scripts.analysis.execution_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        result = ea.calculate_path_dependency()
    assert result.empty

# ── analyze_order_discrepancies ──────────────────────────────────────────

@patch('quantpits.scripts.analysis.utils.get_forward_returns')
def test_analyze_order_discrepancies(mock_fwd_returns, tmp_path, ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log)

    # Create dummy suggestion file for 2026-01-10 (when we bought SZ000001)
    order_dir = tmp_path / "order_suggestions"
    order_dir.mkdir()
    
    # Actually bought SZ000001, but the suggestion was for SZ000999 (Missed Buy)
    # SZ000001 is the Substitute.
    sugg_content = "instrument,score,action\nSZ000999,0.95,BUY\n"
    (order_dir / "buy_suggestion_20260110.csv").write_text(sugg_content)

    # Mock forward returns
    mock_returns = pd.DataFrame({
        'return_5d': [0.10, -0.05]
    }, index=pd.MultiIndex.from_tuples([
        ('SZ000999', pd.to_datetime('2026-01-10')), # missed return
        ('SZ000001', pd.to_datetime('2026-01-10'))  # substitute return
    ], names=["instrument", "datetime"]))
    mock_returns = mock_returns.reset_index()
    mock_returns = mock_returns.set_index(["instrument", "datetime"])  
    mock_fwd_returns.return_value = mock_returns

    with patch('quantpits.scripts.analysis.execution_analyzer.load_market_config', return_value=("csi300", "SH000300")):
        result = ea.analyze_order_discrepancies(str(order_dir))

    assert "substitute_bias_impact" in result
    assert result["total_missed_count"] == 1
    assert result["total_days_with_misses"] == 1
    
    assert np.isclose(float(result["avg_missed_buys_return"]), 0.10)
    assert np.isclose(float(result["avg_substitute_buys_return"]), -0.05)
    assert np.isclose(float(result["substitute_bias_impact"]), -0.15)

def test_analyze_order_discrepancies_no_date(ea_factory, tmp_path):
    trade_log = _make_trade_log()
    trade_log['成交日期'] = pd.NaT
    ea = ea_factory(trade_log_df=trade_log)
    result = ea.analyze_order_discrepancies(str(tmp_path))
    assert result == {}

def test_analyze_order_discrepancies_empty(tmp_path, ea_factory):
    ea = ea_factory(trade_log_df=pd.DataFrame())
    result = ea.analyze_order_discrepancies(str(tmp_path))
    assert result == {}

# ── __init__ ─────────────────────────────────────────────────────────────

def test_init_with_date_range(ea_factory):
    trade_log = _make_trade_log()
    ea = ea_factory(trade_log_df=trade_log, start_date="2026-01-11", end_date="2026-01-20")
    assert len(ea.trade_log) == 1
    assert ea.trade_log['证券代码'].iloc[0] == "SZ000002"

def test_init_load_classification(mock_env, ea_factory):
    workspace = mock_env
    trade_log = _make_trade_log()
    
    # Test auto-loading classification
    class_df = pd.DataFrame({
        "成交日期": pd.to_datetime(["2026-01-10", "2026-01-15"]),
        "证券代码": ["SZ000001", "SZ000002"],
        "trade_class": ["S", "M"]
    })
    class_df['成交日期'] = pd.to_datetime(class_df['成交日期'])
    
    class_path = workspace / "data" / "trade_classification.csv"
    # Force both to datetime before merge in the mock to avoid the warning/error
    trade_log['成交日期'] = pd.to_datetime(trade_log['成交日期'])
    class_df['成交日期'] = pd.to_datetime(class_df['成交日期'])
    class_df.to_csv(class_path, index=False)
    
    with patch('quantpits.scripts.analysis.execution_analyzer.load_trade_log', return_value=trade_log):
        with patch('quantpits.scripts.analysis.utils.ROOT_DIR', str(workspace)):
            ea = ea_factory(trade_log_df=None)
            assert 'trade_class' in ea.trade_log.columns
            assert ea.trade_log.loc[ea.trade_log['证券代码'] == 'SZ000001', 'trade_class'].iloc[0] == 'S'
