import pytest
import os
import json
import pandas as pd
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_env_analytics(tmp_path, monkeypatch):
    workspace = tmp_path / "MockWorkspaceAnalytics"
    workspace.mkdir()
    
    config_dir = workspace / "config"
    config_dir.mkdir()
    
    data_dir = workspace / "data"
    data_dir.mkdir()
    
    # Create mock configs
    prod_config = {
        "current_date": "2026-03-01",
        "last_processed_date": "2026-03-01",
        "broker": "gtja"
    }
    with open(config_dir / "prod_config.json", "w") as f:
        json.dump(prod_config, f)
        
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    import importlib
    importlib.reload(env)
    
    from quantpits.scripts import prod_post_trade_analytics
    importlib.reload(prod_post_trade_analytics)
    
    # Monkeypatch paths
    monkeypatch.setattr(prod_post_trade_analytics, 'DATA_DIR', str(data_dir))
    monkeypatch.setattr(prod_post_trade_analytics, 'ORDER_LOG_FILE', str(data_dir / "raw_order_log_full.csv"))
    monkeypatch.setattr(prod_post_trade_analytics, 'TRADE_LOG_FILE', str(data_dir / "raw_trade_log_full.csv"))
    
    yield prod_post_trade_analytics, workspace

def test_process_analytics_for_day(mock_env_analytics):
    analytics, workspace = mock_env_analytics
    data_dir = workspace / "data"
    
    # Create dummy excel files
    (data_dir / "2026-03-02-order.xlsx").write_text("dummy")
    (data_dir / "2026-03-02-trade.xlsx").write_text("dummy")
    
    mock_adapter = MagicMock()
    mock_adapter.read_orders.return_value = pd.DataFrame({"证券代码": ["000001"], "type": ["order"]})
    mock_adapter.read_trades.return_value = pd.DataFrame({"证券代码": ["600000"], "type": ["trade"]})
    
    # First run (create files)
    analytics.process_analytics_for_day("2026-03-02", mock_adapter)
    
    order_log = pd.read_csv(data_dir / "raw_order_log_full.csv", dtype={"证券代码": str})
    trade_log = pd.read_csv(data_dir / "raw_trade_log_full.csv", dtype={"证券代码": str})
    
    assert len(order_log) == 1
    assert order_log["证券代码"].iloc[0] == "000001"
    assert len(trade_log) == 1
    assert trade_log["证券代码"].iloc[0] == "600000"
    
    # Second run (append and de-duplicate)
    # We use same data, drop_duplicates should keep it at 1
    analytics.process_analytics_for_day("2026-03-02", mock_adapter)
    order_log_v2 = pd.read_csv(data_dir / "raw_order_log_full.csv", dtype={"证券代码": str})
    assert len(order_log_v2) == 1

def test_process_analytics_for_day_missing_files(mock_env_analytics, capsys):
    analytics, _ = mock_env_analytics
    mock_adapter = MagicMock()
    
    analytics.process_analytics_for_day("2026-03-99", mock_adapter)
    captured = capsys.readouterr()
    assert "File not found (2026-03-99-order.xlsx)" in captured.out

@patch('quantpits.scripts.prod_post_trade_analytics.get_trade_dates')
@patch('quantpits.scripts.prod_post_trade_analytics.process_analytics_for_day')
@patch('quantpits.scripts.brokers.get_adapter')
@patch('quantpits.utils.env.init_qlib')
def test_main_analytics(mock_qlib, mock_get_adapter, mock_process, mock_get_dates, mock_env_analytics):
    analytics, _ = mock_env_analytics
    mock_get_dates.return_value = ["2026-03-02"]
    
    import sys
    with patch.object(sys, 'argv', ['prod_post_trade_analytics.py', '--explain-plan']):
        analytics.main()
    mock_process.assert_not_called()
    mock_qlib.assert_not_called()
