from unittest.mock import patch

import pandas as pd
import pytest

from quantpits.post_trade.contracts import BrokerParseError, BrokerSchemaError
from quantpits.scripts.brokers.gtja import GtjaAdapter
from quantpits.scripts.brokers.base import BaseBrokerAdapter, ORDER_REQUIRED_COLUMNS


def test_strict_parser_distinguishes_parse_and_schema_errors():
    adapter = GtjaAdapter()
    with patch("pandas.read_excel", side_effect=OSError("broken")):
        with pytest.raises(BrokerParseError):
            adapter.parse_orders("broken.xlsx")
    with patch("pandas.read_excel", return_value=pd.DataFrame({"证券代码": ["000001"]})):
        with pytest.raises(BrokerSchemaError):
            adapter.parse_orders("wrong.xlsx")


def test_valid_empty_order_export_is_valid():
    columns = ["委托日期", "委托时间", "交易类别", "证券代码", "委托数量", "成交数量", "撤单数量", "委托状态"]
    with patch("pandas.read_excel", return_value=pd.DataFrame(columns=columns)):
        assert GtjaAdapter().parse_orders("empty.xlsx").empty


def test_base_adapter_accepts_valid_empty_export():
    class EmptyAdapter(BaseBrokerAdapter):
        name = "empty"
        def read_settlement(self, _):
            return pd.DataFrame()
        def read_orders(self, _):
            return pd.DataFrame(columns=ORDER_REQUIRED_COLUMNS)

    assert EmptyAdapter().parse_orders("empty.xlsx").empty
