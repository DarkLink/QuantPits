"""
Supplemental tests targeting uncovered branches in prod_post_trade.py.

Coverage targets:
- Line 23: sys.path insertion
- Lines 290-291: verbose sell display
- Lines 315-316: verbose buy display
- Line 423: benchmark close price extraction
- Lines 477-478: append holding log to existing CSV
- Lines 501-502: append daily amount log to existing CSV
- Lines 598-600: invalid broker adapter exit
- Line 635: no cashflows message
- Lines 646-647: no trade dates → early return
- Lines 682-686: trade classification exception
"""

import os
import json
import pytest
import pandas as pd
from decimal import Decimal
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def ppt_env(monkeypatch, tmp_path):
    """Setup workspace for prod_post_trade tests."""
    workspace = tmp_path / "PPTCovWS"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "data").mkdir()

    prod_config = {
        "current_date": "2026-03-01",
        "last_processed_date": "2026-03-01",
        "current_cash": 100000.0,
        "current_holding": [
            {"instrument": "000001", "value": "100", "amount": "1000.0"}
        ],
        "model": "GATs",
        "market": "csi300",
        "benchmark": "SH000300",
    }
    with open(workspace / "config" / "prod_config.json", "w") as f:
        json.dump(prod_config, f)

    cashflow_config = {"cashflows": {"2026-03-02": 50000}}
    with open(workspace / "config" / "cashflow.json", "w") as f:
        json.dump(cashflow_config, f)

    import sys
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    from quantpits.utils import env
    from quantpits.scripts import prod_post_trade
    import importlib
    importlib.reload(env)
    importlib.reload(prod_post_trade)

    monkeypatch.setattr(prod_post_trade, "CONFIG_DIR", str(workspace / "config"))
    monkeypatch.setattr(prod_post_trade, "DATA_DIR", str(workspace / "data"))
    monkeypatch.setattr(prod_post_trade, "PROD_CONFIG_FILE",
                        str(workspace / "config" / "prod_config.json"))
    monkeypatch.setattr(prod_post_trade, "CASHFLOW_CONFIG_FILE",
                        str(workspace / "config" / "cashflow.json"))
    monkeypatch.setattr(prod_post_trade, "TRADE_LOG_FILE",
                        str(workspace / "data" / "trade_log_full.csv"))
    monkeypatch.setattr(prod_post_trade, "HOLDING_LOG_FILE",
                        str(workspace / "data" / "holding_log_full.csv"))
    monkeypatch.setattr(prod_post_trade, "DAILY_LOG_FILE",
                        str(workspace / "data" / "daily_amount_log_full.csv"))

    return prod_post_trade, workspace


def test_process_single_day_no_benchmark(ppt_env):
    """Line 423: benchmark price extraction — fallback to 0."""
    post_trade, workspace = ppt_env

    current_cash = Decimal("100000.0")
    current_holding = [{"instrument": "SZ000001", "value": "100", "amount": "1000.0"}]

    mock_adapter = MagicMock()
    mock_adapter.read_settlement.return_value = pd.DataFrame()

    # Mock features to succeed for close price but fail for benchmark
    mock_features_df = pd.DataFrame({
        "instrument": ["SZ000001"],
        "datetime": [pd.to_datetime("2026-03-02")],
        "Div($close,$factor)": [11.0],
    }).set_index(["instrument", "datetime"])

    import sys
    mock_qlib = MagicMock()
    # First call (close prices) succeeds; benchmark query raises
    benchmark_features = pd.DataFrame(
        {"Div($close,$factor)": []},
        index=pd.MultiIndex.from_tuples([], names=["instrument", "datetime"]),
    )

    def features_side_effect(instruments, fields, start_time=None, end_time=None):
        if isinstance(instruments, list) and any("SH000300" in str(i) for i in instruments):
            raise Exception("benchmark not found")
        return mock_features_df

    mock_qlib.data.D.features = MagicMock(side_effect=features_side_effect)
    mock_qlib.data.D.instruments.return_value = []

    with patch.dict(sys.modules, {"qlib": mock_qlib, "qlib.data": mock_qlib.data,
                                    "qlib.data.ops": mock_qlib.data.ops}):
        with patch("quantpits.scripts.prod_post_trade.load_trade_file",
                   return_value=pd.DataFrame()):
            cash_after, holding_after, closing_value = post_trade.process_single_day(
                current_date_string="2026-03-02",
                current_cash=current_cash,
                current_holding=current_holding,
                model="GATs",
                market="csi300",
                benchmark="SH000300",
                cashflow_today=Decimal("0"),
                adapter=mock_adapter,
            )

    # Should complete without error, benchmark fallback to 0
    assert cash_after is not None


def test_process_single_day_verbose_sell(ppt_env, capsys):
    """Lines 290-291: verbose=True triggers per-trade sell detail print."""
    post_trade, workspace = ppt_env

    current_cash = Decimal("100000.0")
    current_holding = [{"instrument": "SZ000001", "value": "100", "amount": "1000.0"}]

    mock_adapter = MagicMock()

    trade_df = pd.DataFrame({
        "证券代码": ["000001"],
        "model": ["GATs"],
        "交易类别": ["卖出"],
        "成交价格": [11.0],
        "成交数量": [100.0],
        "成交金额": [1100.0],
        "资金发生数": [1100.0],
        "交收日期": ["20260302"],
    })

    mock_features_df = pd.DataFrame({
        "instrument": ["SZ000002"],
        "datetime": [pd.to_datetime("2026-03-02")],
        "Div($close,$factor)": [21.0],
    }).set_index(["instrument", "datetime"])

    import sys
    mock_qlib = MagicMock()
    mock_qlib.data.D.features.return_value = mock_features_df
    mock_qlib.data.D.instruments.return_value = []

    with patch.dict(sys.modules, {"qlib": mock_qlib, "qlib.data": mock_qlib.data,
                                    "qlib.data.ops": mock_qlib.data.ops}):
        with patch("quantpits.scripts.prod_post_trade.load_trade_file",
                   return_value=trade_df):
            with patch("quantpits.scripts.prod_post_trade.SELL_TYPES", {"卖出"}):
                post_trade.process_single_day(
                    current_date_string="2026-03-02",
                    current_cash=current_cash,
                    current_holding=current_holding,
                    model="GATs",
                    market="csi300",
                    benchmark="SH000300",
                    cashflow_today=Decimal("0"),
                    adapter=mock_adapter,
                    verbose=True,
                )

    captured = capsys.readouterr()
    # Should contain verbose sell detail
    assert "000001" in captured.out


def test_process_single_day_verbose_buy(ppt_env, capsys):
    """Lines 315-316: verbose=True triggers per-trade buy detail print."""
    post_trade, workspace = ppt_env

    current_cash = Decimal("100000.0")
    # Keep a dummy holding to avoid KeyError in empty-holding path
    current_holding = [{"instrument": "DUMMY", "value": "1", "amount": "0.0"}]

    mock_adapter = MagicMock()

    trade_df = pd.DataFrame({
        "证券代码": ["000001"],
        "model": ["GATs"],
        "交易类别": ["买入"],
        "成交价格": [10.0],
        "成交数量": [100.0],
        "成交金额": [1000.0],
        "资金发生数": [-1000.0],
        "交收日期": ["20260302"],
    })

    mock_features_df = pd.DataFrame({
        "instrument": ["SH000001"],
        "datetime": [pd.to_datetime("2026-03-02")],
        "Div($close,$factor)": [10.5],
    }).set_index(["instrument", "datetime"])

    import sys
    mock_qlib = MagicMock()
    mock_qlib.data.D.features.return_value = mock_features_df
    mock_qlib.data.D.instruments.return_value = []

    with patch.dict(sys.modules, {"qlib": mock_qlib, "qlib.data": mock_qlib.data,
                                    "qlib.data.ops": mock_qlib.data.ops}):
        with patch("quantpits.scripts.prod_post_trade.load_trade_file",
                   return_value=trade_df):
            with patch("quantpits.scripts.prod_post_trade.BUY_TYPES", {"买入"}):
                cash_after, holding_after, closing_value = post_trade.process_single_day(
                    current_date_string="2026-03-02",
                    current_cash=current_cash,
                    current_holding=current_holding,
                    model="GATs",
                    market="csi300",
                    benchmark="SH000300",
                    cashflow_today=Decimal("0"),
                    adapter=mock_adapter,
                    verbose=True,
                )

    captured = capsys.readouterr()
    # Assert verbose output was printed
    assert "Buy amount" in captured.out or "1000.0" in captured.out


def test_process_single_day_append_to_existing_logs(ppt_env):
    """Lines 477-478, 501-502: append holding/daily log when CSVs exist."""
    post_trade, workspace = ppt_env

    data_dir = workspace / "data"

    # Pre-create existing log files
    pd.DataFrame({
        "成交日期": [pd.Timestamp("2026-03-01")],
        "证券代码": ["SH600000"],
        "收盘价值": [5000.0],
        "current_benchmark": [3000.0],
        "current_cash": [50000.0],
        "holdings": [1],
        "现金余额": [50000.0],
        "总资产": [55000.0],
    }).to_csv(data_dir / "holding_log_full.csv", index=False)

    pd.DataFrame({
        "date": [pd.Timestamp("2026-03-01")],
        "收盘价值": [55000.0],
        "CSI300": [3000.0],
        "净入金": [0.0],
    }).to_csv(data_dir / "daily_amount_log_full.csv", index=False)

    current_cash = Decimal("100000.0")
    current_holding = [{"instrument": "SZ000001", "value": "100", "amount": "1000.0"}]

    mock_adapter = MagicMock()
    mock_adapter.read_settlement.return_value = pd.DataFrame()

    mock_features_df = pd.DataFrame({
        "instrument": ["SZ000001"],
        "datetime": [pd.to_datetime("2026-03-02")],
        "Div($close,$factor)": [11.0],
    }).set_index(["instrument", "datetime"])

    import sys
    mock_qlib = MagicMock()
    mock_qlib.data.D.features.return_value = mock_features_df
    mock_qlib.data.D.instruments.return_value = []

    with patch.dict(sys.modules, {"qlib": mock_qlib, "qlib.data": mock_qlib.data,
                                    "qlib.data.ops": mock_qlib.data.ops}):
        with patch("quantpits.scripts.prod_post_trade.load_trade_file",
                   return_value=pd.DataFrame()):
            cash_after, holding_after, closing_value = post_trade.process_single_day(
                current_date_string="2026-03-02",
                current_cash=current_cash,
                current_holding=current_holding,
                model="GATs",
                market="csi300",
                benchmark="SH000300",
                cashflow_today=Decimal("0"),
                adapter=mock_adapter,
            )

    # Verify both logs exist and have been appended to
    holding_df = pd.read_csv(data_dir / "holding_log_full.csv")
    daily_df = pd.read_csv(data_dir / "daily_amount_log_full.csv")
    assert len(holding_df) >= 2
    assert len(daily_df) >= 2


@patch("quantpits.scripts.prod_post_trade.get_trade_dates")
@patch("quantpits.scripts.brokers.get_adapter")
def test_main_invalid_broker(mock_get_adapter, mock_get_dates, ppt_env):
    """Lines 598-600: invalid broker adapter → sys.exit(1)."""
    post_trade, _ = ppt_env

    mock_get_dates.return_value = ["2026-03-02"]
    mock_get_adapter.side_effect = ValueError("Unknown broker: bad_broker")

    import sys
    with patch.object(sys, "argv", ["prod_post_trade.py", "--broker", "bad_broker"]):
        with pytest.raises(SystemExit) as excinfo:
            post_trade.main()
    assert excinfo.value.code == 1


@patch("quantpits.scripts.prod_post_trade.get_trade_dates")
def test_main_no_cashflows_message(mock_get_dates, ppt_env, capsys):
    """Line 635: no cashflows in period → info message."""
    post_trade, _ = ppt_env

    # Remove cashflow config
    cashflow_file = os.path.join(post_trade.CASHFLOW_CONFIG_FILE)
    with open(cashflow_file, "w") as f:
        json.dump({}, f)

    mock_get_dates.return_value = ["2026-03-02"]

    import sys
    with patch.object(sys, "argv", ["prod_post_trade.py", "--dry-run"]):
        post_trade.main()

    captured = capsys.readouterr()
    assert "No cashflows" in captured.out


@patch("quantpits.scripts.prod_post_trade.get_trade_dates")
def test_main_no_trade_dates(mock_get_dates, ppt_env, capsys):
    """Lines 646-647: no trade dates → early return message."""
    post_trade, _ = ppt_env

    mock_get_dates.return_value = []

    import sys
    with patch.object(sys, "argv", ["prod_post_trade.py"]):
        post_trade.main()

    captured = capsys.readouterr()
    # Should not crash, should print no dates message
    assert "No trade dates" in captured.out or "trade dates" in captured.out.lower()


@patch("quantpits.scripts.prod_post_trade.get_trade_dates")
@patch("quantpits.scripts.prod_post_trade.process_single_day")
@patch("quantpits.scripts.prod_post_trade.save_prod_config")
@patch("quantpits.scripts.brokers.get_adapter")
@patch("quantpits.scripts.analysis.trade_classifier.classify_trades")
def test_main_trade_classification_exception(mock_classify, mock_adapter, mock_save,
                                              mock_process, mock_get_dates, ppt_env, capsys):
    """Lines 682-686: trade classification raises → caught, traceback logged."""
    post_trade, _ = ppt_env

    mock_get_dates.return_value = ["2026-03-02"]
    mock_process.return_value = (Decimal("100000.0"), [], 110000.0)
    mock_classify.side_effect = RuntimeError("classification failed")

    import sys
    with patch.object(sys, "argv", ["prod_post_trade.py", "--end-date", "2026-03-02"]):
        post_trade.main()

    captured = capsys.readouterr()
    # Should not crash; exception should be caught and reported
    assert "classification failed" in captured.out or "Traceback" in captured.out
