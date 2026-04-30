"""
Supplemental tests targeting uncovered branches in prod_post_trade_analytics.py.

Coverage targets:
- Line 25: sys.path.insert
- Lines 44-49: get_trade_dates exception fallback
- Line 72: orders empty after filtering
- Line 91: trades empty after filtering
- Lines 111-113: invalid broker adapter exit
- Lines 124-125: no trade dates early return
- Line 135: main() entry
"""

import os
import json
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def analytics_env(tmp_path, monkeypatch):
    """Setup workspace for analytics tests."""
    workspace = tmp_path / "AnalyticsWS"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "data").mkdir()

    # Create prod config
    prod_config = {
        "current_date": "2026-03-01",
        "last_processed_date": "2026-03-01",
        "broker": "gtja",
    }
    with open(workspace / "config" / "prod_config.json", "w") as f:
        json.dump(prod_config, f)

    import sys
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    from quantpits.utils import env
    from quantpits.scripts import prod_post_trade_analytics
    import importlib
    importlib.reload(env)
    importlib.reload(prod_post_trade_analytics)

    # Patch paths
    monkeypatch.setattr(prod_post_trade_analytics, "DATA_DIR", str(workspace / "data"))
    monkeypatch.setattr(prod_post_trade_analytics, "ORDER_LOG_FILE",
                        str(workspace / "data" / "raw_order_log_full.csv"))
    monkeypatch.setattr(prod_post_trade_analytics, "TRADE_LOG_FILE",
                        str(workspace / "data" / "raw_trade_log_full.csv"))

    return prod_post_trade_analytics, workspace


def test_get_trade_dates_exception(analytics_env):
    """Lines 44-49: D.calendar raises → return empty list."""
    analytics, _ = analytics_env
    with patch("qlib.data.D") as mock_D:
        mock_D.calendar.side_effect = Exception("qlib not initialized")
        result = analytics.get_trade_dates("2026-01-01", "2026-01-31")
    assert result == []


def test_process_analytics_orders_empty_after_filter(analytics_env, capsys):
    """Line 72: adapter returns empty DataFrame for orders."""
    analytics, workspace = analytics_env
    data_dir = workspace / "data"

    # Create order file
    (data_dir / "2026-03-02-order.xlsx").write_text("dummy")

    mock_adapter = MagicMock()
    mock_adapter.read_orders.return_value = pd.DataFrame()  # empty
    mock_adapter.read_trades.return_value = pd.DataFrame({"证券代码": ["000001"]})

    analytics.process_analytics_for_day("2026-03-02", mock_adapter)

    captured = capsys.readouterr()
    assert "No valid rows found" in captured.out


def test_process_analytics_trades_empty_after_filter(analytics_env, capsys):
    """Line 91: adapter returns empty DataFrame for trades."""
    analytics, workspace = analytics_env
    data_dir = workspace / "data"

    (data_dir / "2026-03-02-order.xlsx").write_text("dummy")
    (data_dir / "2026-03-02-trade.xlsx").write_text("dummy")

    mock_adapter = MagicMock()
    mock_adapter.read_orders.return_value = pd.DataFrame({"证券代码": ["000001"]})
    mock_adapter.read_trades.return_value = pd.DataFrame()  # empty

    analytics.process_analytics_for_day("2026-03-02", mock_adapter)

    captured = capsys.readouterr()
    assert "Trades: No valid rows" in captured.out or "Trades found: 0" in captured.out


@patch("quantpits.scripts.prod_post_trade_analytics.get_trade_dates")
@patch("quantpits.scripts.prod_post_trade_analytics.load_prod_config")
@patch("quantpits.scripts.brokers.get_adapter")
@patch("quantpits.utils.env.init_qlib")
def test_main_no_trade_dates(mock_qlib, mock_get_adapter, mock_load_cfg, mock_get_dates,
                               analytics_env, capsys):
    """Lines 124-125: no trade dates → early return."""
    analytics, _ = analytics_env
    mock_get_dates.return_value = []
    mock_load_cfg.return_value = {"current_date": "2026-03-01",
                                   "last_processed_date": "2026-03-01"}

    import sys
    with patch.object(sys, "argv", ["prod_post_trade_analytics.py"]):
        analytics.main()

    captured = capsys.readouterr()
    assert "No trade dates to process" in captured.out


@patch("quantpits.scripts.prod_post_trade_analytics.get_trade_dates")
@patch("quantpits.scripts.prod_post_trade_analytics.load_prod_config")
@patch("quantpits.scripts.brokers.get_adapter")
@patch("quantpits.utils.env.init_qlib")
def test_main_invalid_broker(mock_qlib, mock_get_adapter, mock_load_cfg, mock_get_dates,
                               analytics_env):
    """Lines 111-113: invalid broker adapter → sys.exit(1)."""
    analytics, _ = analytics_env
    mock_get_dates.return_value = ["2026-03-02"]
    mock_load_cfg.return_value = {"current_date": "2026-03-01",
                                   "last_processed_date": "2026-03-01"}
    mock_get_adapter.side_effect = ValueError("Unknown broker: invalid")

    import sys
    with patch.object(sys, "argv", ["prod_post_trade_analytics.py", "--broker", "invalid"]):
        with pytest.raises(SystemExit) as excinfo:
            analytics.main()
    assert excinfo.value.code == 1
