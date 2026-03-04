import pytest
import os
import json
import pandas as pd
from decimal import Decimal
from unittest.mock import patch, MagicMock

# Apply environment mocking before loading the module
@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    config_dir = workspace / "config"
    config_dir.mkdir()
    
    data_dir = workspace / "data"
    data_dir.mkdir()
    
    # Create mock configs
    prod_config = {
        "current_date": "2026-03-01",
        "last_processed_date": "2026-03-01",
        "current_cash": 100000.0,
        "current_holding": [
            {"instrument": "000001", "value": "100", "amount": "1000.0"}
        ],
        "model": "GATs",
        "market": "csi300",
        "benchmark": "SH000300"
    }
    with open(config_dir / "prod_config.json", "w") as f:
        json.dump(prod_config, f)
        
    cashflow_config = {
        "cashflows": {"2026-03-02": 50000}
    }
    with open(config_dir / "cashflow.json", "w") as f:
        json.dump(cashflow_config, f)
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, prod_post_trade
    import importlib
    importlib.reload(env)
    importlib.reload(prod_post_trade)
    
    # Monkeypatch the constants in the module to point to our temp dir
    monkeypatch.setattr(prod_post_trade, 'CONFIG_DIR', str(config_dir))
    monkeypatch.setattr(prod_post_trade, 'DATA_DIR', str(data_dir))
    monkeypatch.setattr(prod_post_trade, 'PROD_CONFIG_FILE', str(config_dir / "prod_config.json"))
    monkeypatch.setattr(prod_post_trade, 'CASHFLOW_CONFIG_FILE', str(config_dir / "cashflow.json"))
    monkeypatch.setattr(prod_post_trade, 'TRADE_LOG_FILE', str(data_dir / "trade_log_full.csv"))
    monkeypatch.setattr(prod_post_trade, 'HOLDING_LOG_FILE', str(data_dir / "holding_log_full.csv"))
    monkeypatch.setattr(prod_post_trade, 'DAILY_LOG_FILE', str(data_dir / "daily_amount_log_full.csv"))
    monkeypatch.setattr(prod_post_trade, 'EMPTY_TRADE_FILE', str(data_dir / "emp-table.xlsx"))
    
    yield prod_post_trade, workspace

def test_load_configs(mock_env):
    post_trade, workspace = mock_env
    
    prod_config = post_trade.load_prod_config()
    assert prod_config["current_cash"] == 100000.0
    assert len(prod_config["current_holding"]) == 1
    
    cashflow_config = post_trade.load_cashflow_config()
    assert "2026-03-02" in cashflow_config["cashflows"]

def test_get_cashflow_for_date(mock_env):
    post_trade, _ = mock_env
    
    new_format = {"cashflows": {"2026-03-02": 5000}}
    assert post_trade.get_cashflow_for_date(new_format, "2026-03-02", False) == Decimal("5000")
    assert post_trade.get_cashflow_for_date(new_format, "2026-03-03", False) == Decimal("0")
    
    old_format = {"cash_flow_today": 1234}
    assert post_trade.get_cashflow_for_date(old_format, "2026-03-02", True) == Decimal("1234")
    assert post_trade.get_cashflow_for_date(old_format, "2026-03-02", False) == Decimal("0")

def test_process_single_day_no_trades(mock_env):
    post_trade, workspace = mock_env
    
    current_cash = Decimal("100000.0")
    current_holding = [{"instrument": "000001", "value": "100", "amount": "1000.0"}]
    
    mock_adapter = MagicMock()
    mock_adapter.read_settlement.return_value = pd.DataFrame()
    
    mock_features_df = pd.DataFrame({
        "instrument": ["000001"],
        "datetime": [pd.to_datetime("2026-03-02")],
        "Div($close,$factor)": [11.0]
    }).set_index(["instrument", "datetime"])
    
    import sys
    mock_qlib = MagicMock()
    mock_qlib.data.D.features.return_value = mock_features_df
    mock_qlib.data.D.instruments.return_value = []

    with patch.dict(sys.modules, {'qlib': mock_qlib, 'qlib.data': mock_qlib.data, 'qlib.data.ops': mock_qlib.data.ops}):
        with patch("quantpits.scripts.prod_post_trade.load_trade_file", return_value=pd.DataFrame()):
            cash_after, holding_after = post_trade.process_single_day(
                    current_date_string="2026-03-02",
                    current_cash=current_cash,
                    current_holding=current_holding,
                    model="GATs",
                    market="csi300",
                    benchmark="SH000300",
                    cashflow_today=Decimal("5000"),
                    adapter=mock_adapter
                )
    
    # 100000 + 5000
    assert cash_after == Decimal("105000.0")
    assert len(holding_after) == 1
    assert holding_after[0]["instrument"] == "000001"
    assert holding_after[0]["value"] == "100"
    
    # Check that logs were created
    assert os.path.exists(workspace / "data" / "holding_log_full.csv")
    assert os.path.exists(workspace / "data" / "daily_amount_log_full.csv")

def test_process_single_day_with_buy_sell(mock_env):
    post_trade, workspace = mock_env
    
    current_cash = Decimal("100000.0")
    # Hold SZ000001
    current_holding = [{"instrument": "SZ000001", "value": "100", "amount": "1000.0"}]
    
    mock_adapter = MagicMock()
    # Mock trade detail (Sell 000001, Buy 000002)
    trade_df = pd.DataFrame({
        "证券代码": ["000001", "000002"],
        "model": ["GATs", "GATs"],
        "交易类别": ["卖出", "买入"],
        "成交价格": [11.0, 20.0],
        "成交数量": [100.0, 200.0],
        "成交金额": [1100.0, 4000.0],
        "资金发生数": [1100.0, -4000.0],
        "交收日期": ["20260302", "20260302"]
    })
    
    mock_features_df = pd.DataFrame({
        "instrument": ["SZ000002"],
        "datetime": [pd.to_datetime("2026-03-02")],
        "Div($close,$factor)": [21.0]
    }).set_index(["instrument", "datetime"])
    
    
    import sys
    # Create fake qlib module layout
    mock_qlib = MagicMock()
    mock_qlib.data.D.features.return_value = mock_features_df
    mock_qlib.data.D.instruments.return_value = []
    
    with patch.dict(sys.modules, {'qlib': mock_qlib, 'qlib.data': mock_qlib.data, 'qlib.data.ops': mock_qlib.data.ops}):
        with patch("quantpits.scripts.prod_post_trade.load_trade_file", return_value=trade_df):
            # We mock SELL_TYPES / BUY_TYPES to match our mock data
            with patch("quantpits.scripts.prod_post_trade.SELL_TYPES", {"卖出"}):
                with patch("quantpits.scripts.prod_post_trade.BUY_TYPES", {"买入"}):
                        cash_after, holding_after = post_trade.process_single_day(
                            current_date_string="2026-03-02",
                            current_cash=current_cash,
                            current_holding=current_holding,
                            model="GATs",
                            market="csi300",
                            benchmark="SH000300",
                            cashflow_today=Decimal("0"),
                            adapter=mock_adapter
                        )
    
    # 100000 + 1100 (sell) - 4000 (buy) = 97100
    assert cash_after == Decimal("97100.0")
    assert len(holding_after) == 1
    assert holding_after[0]["instrument"] == "SZ000002"
    assert holding_after[0]["value"] == "200.0"
    assert holding_after[0]["amount"] == "4000.0"


# ── Supplementary tests for previously uncovered functions ───────────────

def test_add_prefix(mock_env):
    post_trade, _ = mock_env
    assert post_trade.add_prefix("600001") == "SH600001"
    assert post_trade.add_prefix("000001") == "SZ000001"
    assert post_trade.add_prefix("300123") == "SZ300123"
    assert post_trade.add_prefix("800999") == "800999"  # Unknown prefix
    assert pd.isna(post_trade.add_prefix(pd.NA))


def test_to_decimal_columns(mock_env):
    post_trade, _ = mock_env
    df = pd.DataFrame({
        "成交价格": [10.5, 20.1],
        "成交金额": [1050.0, 4020.0],
        "备注": ["foo", "bar"]
    })
    post_trade.to_decimal_columns(df, ["成交价格", "成交金额", "不存在的列"])
    assert isinstance(df["成交价格"].iloc[0], Decimal)
    assert isinstance(df["成交金额"].iloc[0], Decimal)
    assert df["备注"].iloc[0] == "foo"  # Unchanged


def test_archive_cashflows_new_format(mock_env):
    post_trade, workspace = mock_env
    cashflow_config = {
        "cashflows": {"2026-03-02": 50000, "2026-03-03": -10000},
        "processed": {}
    }
    post_trade.archive_cashflows(cashflow_config)
    assert cashflow_config["cashflows"] == {}
    assert cashflow_config["processed"]["2026-03-02"] == 50000
    assert cashflow_config["processed"]["2026-03-03"] == -10000


def test_archive_cashflows_old_format(mock_env):
    post_trade, workspace = mock_env
    cashflow_config = {"cash_flow_today": 12345}
    post_trade.archive_cashflows(cashflow_config)
    assert cashflow_config["cash_flow_today"] == 0


def test_save_prod_config(mock_env):
    post_trade, workspace = mock_env
    new_config = {"current_cash": 99999.0, "current_holding": []}
    post_trade.save_prod_config(new_config)
    with open(workspace / "config" / "prod_config.json") as f:
        saved = json.load(f)
    assert saved["current_cash"] == 99999.0
    assert saved["current_holding"] == []
