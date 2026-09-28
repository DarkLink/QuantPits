import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from quantpits.scripts.brokers.gtja import GtjaAdapter

def test_gtja_adapter_name():
    adapter = GtjaAdapter()
    assert adapter.name == "gtja"

def test_gtja_adapter_valid_read():
    adapter = GtjaAdapter()
    
    mock_df = pd.DataFrame({
        "证券代码": ["000001", "600000"],
        "交易类别": ["深圳A股普通股票竞价买入", "上海A股普通股票竞价卖出"],
        "成交价格": [10.5, 20.1],
        "成交数量": [100.0, 200.0],
        "成交金额": [1050.0, 4020.0],
        "资金发生数": [-1050.0, 4020.0],
        "交收日期": ["2026-03-01", "2026-03-02"],
        "无关字段": ["\t额外数据", "脏数据\t"]
    })
    
    with patch("pandas.read_excel", return_value=mock_df) as mock_read:
        df = adapter.read_settlement("dummy.xlsx")
        
        mock_read.assert_called_once_with(
            "dummy.xlsx", 
            sheet_name="Sheet1", 
            skiprows=5, 
            dtype={"证券代码": str}
        )
        
        # Test string cleaning
        assert df["无关字段"].iloc[0] == "额外数据"
        assert not df.empty
        assert "证券代码" in df.columns

def test_gtja_adapter_missing_columns():
    adapter = GtjaAdapter()
    
    mock_df = pd.DataFrame({
        "证券代码": ["000001"],
        # Missing other required columns
    })
    
    with patch("pandas.read_excel", return_value=mock_df):
        with patch("sys.stdout") as mock_stdout: # to suppress print
            df = adapter.read_settlement("dummy.xlsx")
            assert df.empty

def test_gtja_adapter_read_exception(capsys):
    adapter = GtjaAdapter()
    
    with patch("pandas.read_excel", side_effect=Exception("Mocked Error")):
        df = adapter.read_settlement("dummy.xlsx")
        
        assert df.empty
        captured = capsys.readouterr()
        assert "[gtja] Error loading dummy.xlsx" in captured.out


def test_gtja_adapter_read_orders_and_trades():
    adapter = GtjaAdapter()
    
    mock_df = pd.DataFrame({
        "证券代码": ["895", "600309.SH", "1", "999999", "nan", None],
        "证券名称": ["双汇发展", "万华化学", "平安银行", "其它", "N/A", "N/A"],
        "无关字段": ["\tTab", "NoTab", " ", "", "", ""]
    })
    
    with patch("pandas.read_excel", return_value=mock_df):
        # Test read_orders (uses _read_and_filter)
        df_orders = adapter.read_orders("dummy.xlsx")
        
        # Expected: 
        # 895 -> 000895 (starts with 0, keep)
        # 600309.SH -> 600309 (starts with 6, keep)
        # 1 -> 000001 (starts with 0, keep)
        # 999999 -> 999999 (starts with 9, discard)
        # nan/None -> discard
        
        assert len(df_orders) == 3
        assert set(df_orders["证券代码"]) == {"000895", "600309", "000001"}
        assert df_orders["无关字段"].iloc[0] == "Tab" # Tab stripped
        
        # Test read_trades (uses same _read_and_filter)
        df_trades = adapter.read_trades("dummy.xlsx")
        assert len(df_trades) == 3
        assert set(df_trades["证券代码"]) == {"000895", "600309", "000001"}


@pytest.mark.parametrize("code", [None, "", "\t", "nan", "999999"])
def test_settlement_retains_cash_interest_without_stock_code(code):
    from decimal import Decimal
    from quantpits.post_trade.state import normalize_settlement_frame
    raw = pd.DataFrame({
        "证券代码": [code], "交易类别": ["利息归本"],
        "成交价格": [0], "成交数量": [0], "成交金额": [0],
        "资金发生数": [4.05], "交收日期": ["2026-09-21"],
    })
    with patch("pandas.read_excel", return_value=raw):
        parsed = GtjaAdapter().parse_settlement("interest.xlsx")
    assert len(parsed) == 1
    assert parsed.iloc[0]["证券代码"] == ""
    events, _ = normalize_settlement_frame(parsed, "2026-09-21")
    assert events[0].kind == "cash_adjustment"
    assert events[0].instrument is None
    assert events[0].cash_effect == Decimal("4.05")
