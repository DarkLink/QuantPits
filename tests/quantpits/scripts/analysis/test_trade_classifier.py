import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

import quantpits.scripts.analysis.trade_classifier as tc


def _make_dummy_suggestions(tmp_path):
    # Setup directories
    data_dir = tmp_path / "data"
    order_dir = data_dir / "order_history"
    out_dir = tmp_path / "output"
    
    order_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some suggestion files
    buy_1 = "instrument,score\nSZ000001,0.9\nSZ000002,0.8\nSZ000003,0.7\n"
    (order_dir / "buy_suggestion_2026-01-10.csv").write_text(buy_1)
    
    buy_2 = "instrument,score\nSZ000004,0.95\n"
    (out_dir / "buy_suggestion_ensemble_2026-02-25.csv").write_text(buy_2)
    
    sell_1 = "instrument,score\nSZ000001,0.1\n"
    (order_dir / "sell_suggestion_2026-01-15.csv").write_text(sell_1)
    
    return data_dir, order_dir, out_dir


def _make_dummy_trade_log(data_dir):
    log_content = """成交日期,证券代码,交易类别,成交数量
2026-01-10,SZ000001,深圳A股普通股票竞价买入,100
2026-01-10,000002,深圳A股普通股票竞价买入,200
2026-01-10,SZ000099,深圳A股普通股票竞价买入,300
2026-01-15,SZ000001,深圳A股普通股票竞价卖出,100
2026-02-25,SZ000004,深圳A股普通股票竞价买入,100
2026-03-01,SZ000005,深圳A股普通股票竞价买入,100
"""
    log_path = data_dir / "trade_log_full.csv"
    log_path.write_text(log_content)
    return str(log_path)


# ── Internal Functions ───────────────────────────────────────────────────

def test_scan_suggestion_dates(tmp_path):
    data_dir, order_dir, out_dir = _make_dummy_suggestions(tmp_path)
    
    with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
        with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(out_dir)):
            dates = tc._scan_suggestion_dates("buy_suggestion")
            assert len(dates) == 2
            assert pd.Timestamp("2026-01-10") in dates
            assert pd.Timestamp("2026-02-25") in dates

            sell_dates = tc._scan_suggestion_dates("sell_suggestion")
            assert len(sell_dates) == 1
            assert pd.Timestamp("2026-01-15") in sell_dates


def test_find_suggestion_date():
    available = [pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-15"), pd.Timestamp("2026-02-25")]
    
    # Exact match
    res = tc._find_suggestion_date("2026-01-10", available)
    assert res == pd.Timestamp("2026-01-10")
    
    # Fallback historical (<= strict_after) - 1 day gap
    res = tc._find_suggestion_date("2026-01-11", available, strict_after="2026-02-13")
    assert res == pd.Timestamp("2026-01-10")
    
    # Fallback historical - 8 days gap (too far)
    res = tc._find_suggestion_date("2026-01-20", available, strict_after="2026-02-13")
    res2 = tc._find_suggestion_date("2026-01-22", available, strict_after="2026-02-13")
    assert res == pd.Timestamp("2026-01-15")
    assert res2 == pd.Timestamp("2026-01-15")
    
    # Future mismatch (strict matching)
    res = tc._find_suggestion_date("2026-02-26", available, strict_after="2026-02-13")
    assert res is None


def test_load_suggestion(tmp_path):
    data_dir, order_dir, out_dir = _make_dummy_suggestions(tmp_path)
    
    with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
        with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(out_dir)):
            
            # Load from order
            df = tc._load_suggestion("buy_suggestion", "2026-01-10")
            assert not df.empty
            assert len(df) == 3
            assert "instrument" in df.columns
            
            # Load from output
            df2 = tc._load_suggestion("buy_suggestion", "2026-02-25")
            assert not df2.empty
            assert len(df2) == 1
            assert df2.iloc[0]["instrument"] == "SZ000004"
            
            # Missing
            df3 = tc._load_suggestion("buy_suggestion", "2026-03-01")
            assert df3.empty


def test_normalize_instrument():
    assert tc._normalize_instrument("SH601066") == "SH601066"
    assert tc._normalize_instrument("601066") == "SH601066"
    assert tc._normalize_instrument("000001") == "SZ000001"
    assert tc._normalize_instrument("300001") == "SZ300001"
    assert pd.isna(tc._normalize_instrument(np.nan))


# ── Core Classification ──────────────────────────────────────────────────

def test_classify_trades(tmp_path):
    data_dir, order_dir, out_dir = _make_dummy_suggestions(tmp_path)
    log_file = _make_dummy_trade_log(data_dir)
    
    with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', log_file):
        with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
            with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(out_dir)):
                with patch('quantpits.scripts.analysis.trade_classifier.ROOT_DIR', str(tmp_path)):
                    
                    # By default factor is 3, so if length is 3, alg_n_buy = ceil(3/3) = 1.
                    # Rank 0 -> "S" (Signal). Rank > 0 -> "A" (Substitute).
                    result = tc.classify_trades(verbose=True)
                    
                    assert not result.empty
                    
                    # 2026-01-10 SZ000001 (Rank 0 -> Signal 'S')
                    r1 = result[(result["trade_date"] == "2026-01-10") & (result["instrument"] == "SZ000001")].iloc[0]
                    assert r1["trade_class"] == "S"
                    assert r1["trade_type"] == "BUY"
                    
                    # 2026-01-10 000002 -> SZ000002 (Rank 1 -> Substitute 'A' because alg_n = 1)
                    r2 = result[(result["trade_date"] == "2026-01-10") & (result["instrument"] == "SZ000002")].iloc[0]
                    assert r2["trade_class"] == "A"
                    
                    # 2026-01-10 SZ000099 (Not in file -> Manual 'M')
                    r3 = result[(result["trade_date"] == "2026-01-10") & (result["instrument"] == "SZ000099")].iloc[0]
                    assert r3["trade_class"] == "M"
                    
                    # 2026-01-15 SZ000001 (Sell file -> 'S')
                    r4 = result[(result["trade_date"] == "2026-01-15")].iloc[0]
                    assert r4["trade_class"] == "S"
                    assert r4["trade_type"] == "SELL"
                    
                    # 2026-03-01 SZ000005 (Future, no file -> Manual)
                    r5 = result[(result["trade_date"] == "2026-03-01")].iloc[0]
                    assert r5["trade_class"] == "M"


def test_classify_trades_with_date_filter(tmp_path):
    data_dir, order_dir, out_dir = _make_dummy_suggestions(tmp_path)
    log_file = _make_dummy_trade_log(data_dir)
    
    with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', log_file):
        with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
            with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(out_dir)):
                with patch('quantpits.scripts.analysis.trade_classifier.ROOT_DIR', str(tmp_path)):
                    result = tc.classify_trades(trade_dates=["2026-01-15"])
                    assert len(result) == 1
                    assert result.iloc[0]["trade_date"] == "2026-01-15"


# ── Save / Load ──────────────────────────────────────────────────────────

def test_save_load_classification(tmp_path):
    class_file = str(tmp_path / "trade_classification.csv")
    
    df1 = pd.DataFrame({
        "trade_date": ["2026-01-10"],
        "instrument": ["SZ000001"],
        "trade_class": ["S"]
    })
    
    df2 = pd.DataFrame({
        "trade_date": ["2026-01-11"],
        "instrument": ["SZ000002"],
        "trade_class": ["M"]
    })
    
    # 1. Save and Load
    tc.save_classification(df1, path=class_file)
    loaded = tc.load_classification(path=class_file)
    assert len(loaded) == 1
    assert loaded.iloc[0]["instrument"] == "SZ000001"
    
    # 2. Append without intersection
    tc.save_classification(df2, path=class_file, append=True, trade_dates=["2026-01-11"])
    loaded_app = tc.load_classification(path=class_file)
    assert len(loaded_app) == 2
    assert "SZ000002" in loaded_app["instrument"].values

    # 3. Append with intersection (overriding 01-11)
    df3 = pd.DataFrame({
        "trade_date": ["2026-01-11"],
        "instrument": ["SZ000003"],
        "trade_class": ["A"]
    })
    tc.save_classification(df3, path=class_file, append=True, trade_dates=["2026-01-11"])
    loaded_over = tc.load_classification(path=class_file)
    assert len(loaded_over) == 2
    assert "SZ000003" in loaded_over["instrument"].values
    assert "SZ000002" not in loaded_over["instrument"].values

# ── New P1 Robustness Tests ──────────────────────────────────────────────

def test_classify_trades_missing_file(tmp_path):
    with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', str(tmp_path / "nonexistent.csv")):
        with pytest.raises(FileNotFoundError, match="Trade log not found"):
            tc.classify_trades()

def test_classify_trades_missing_columns(tmp_path):
    log_path = tmp_path / "bad_log.csv"
    # Missing "成交日期"
    log_path.write_text("wrong_col,证券代码,交易类别\n2026-01-10,SH600000,买入\n")
    
    with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', str(log_path)):
        with pytest.raises(ValueError, match="Column '成交日期' not found"):
            tc.classify_trades()

def test_classify_trades_missing_instrument_column(tmp_path):
    log_path = tmp_path / "bad_log_2.csv"
    # Has "成交日期" but missing "证券代码"
    log_path.write_text("成交日期,交易类别\n2026-01-10,SH600000买入\n")
    
    with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', str(log_path)):
        with pytest.raises(ValueError, match="Column '证券代码' not found"):
            tc.classify_trades()

def test_scan_suggestion_dates_robustness(tmp_path):
    order_dir = tmp_path / "orders"
    order_dir.mkdir()
    # Malformed filename (missing date part)
    (order_dir / "buy_suggestion_invalid.csv").write_text("dummy")
    # Valid one
    (order_dir / "buy_suggestion_2026-01-01.csv").write_text("dummy")
    
    with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
        with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(tmp_path)):
            dates = tc._scan_suggestion_dates("buy_suggestion")
            assert len(dates) == 1
            assert dates[0] == pd.Timestamp("2026-01-01")

def test_classify_trades_malformed_config(tmp_path):
    # This triggers the 'except Exception: pass' in factor loading
    data_dir, order_dir, out_dir = _make_dummy_suggestions(tmp_path)
    log_file = _make_dummy_trade_log(data_dir)
    cfg_file = tmp_path / "config" / "prod_config.json"
    cfg_file.parent.mkdir(exist_ok=True)
    cfg_file.write_text("INVALID JSON")
    
    with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', log_file):
        with patch('quantpits.scripts.analysis.trade_classifier.ROOT_DIR', str(tmp_path)):
            with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
                with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(out_dir)):
                    # Should use default factor=3 and not crash
                    result = tc.classify_trades()
                    assert not result.empty

def test_classify_trades_no_trades(tmp_path):
    data_dir, order_dir, out_dir = _make_dummy_suggestions(tmp_path)
    # Date with no trades in dummy log
    with patch('quantpits.scripts.analysis.trade_classifier.ROOT_DIR', str(tmp_path)):
        with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
            with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(out_dir)):
                with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', str(tmp_path / "data" / "trade_log_empty.csv")):
                    # Create empty log
                    (tmp_path / "data").mkdir(exist_ok=True)
                    (tmp_path / "data" / "trade_log_empty.csv").write_text("成交日期,证券代码,交易类别,成交数量\n")
                    # Case with empty trade log or no matches
                    res = tc.classify_trades(trade_dates=["1999-01-01"])
                    assert res.empty

def test_save_classification_default_path(tmp_path):
    df = pd.DataFrame({"a": [1]})
    with patch('quantpits.scripts.analysis.trade_classifier.CLASSIFICATION_FILE', str(tmp_path / "default.csv")):
        tc.save_classification(df)
        assert os.path.exists(tmp_path / "default.csv")

def test_classify_trades_unbalanced(tmp_path):
    data_dir, order_dir, out_dir = _make_dummy_suggestions(tmp_path)
    # Create a log with ONLY buys
    log_content = "成交日期,证券代码,交易类别,成交数量\n2026-01-10,SZ000001,深圳A股普通股票竞价买入,100\n"
    log_path = tmp_path / "buys_only.csv"
    log_path.write_text(log_content)
    
    with patch('quantpits.scripts.analysis.trade_classifier.TRADE_LOG_FILE', str(log_path)):
        with patch('quantpits.scripts.analysis.trade_classifier.ORDER_DIR', str(order_dir)):
            with patch('quantpits.scripts.analysis.trade_classifier.OUTPUT_DIR', str(out_dir)):
                with patch('quantpits.scripts.analysis.trade_classifier.ROOT_DIR', str(tmp_path)):
                    # Should cover 'continue' when sells are empty (Line 337)
                    result = tc.classify_trades(verbose=True)
                    assert not result.empty
                    assert "SELL" not in result["trade_type"].values
