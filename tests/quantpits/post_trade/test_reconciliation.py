from decimal import Decimal
import pandas as pd
import pytest

from quantpits.post_trade.contracts import ExecutionReconciliationError
from quantpits.post_trade.reconciliation import reconcile_quantities
from quantpits.post_trade.state import SettlementEvent


def _frame(qty):
    return pd.DataFrame({"成交日期": ["2026-01-02"], "交易类别": ["证券买入"], "证券代码": ["000001"], "成交数量": [qty]})


def _event(qty):
    return SettlementEvent("2026-01-02", "SZ000001", "buy", Decimal(str(qty)), Decimal("10"), Decimal("100"), Decimal("-100"), 1, "深圳A股普通股票竞价买入")


def test_three_stream_quantities_match():
    result = reconcile_quantities(_frame(10), _frame(10), (_event(10),))
    assert result[0][1] == Decimal("10")


def test_three_stream_mismatch_fails_closed():
    with pytest.raises(ExecutionReconciliationError):
        reconcile_quantities(_frame(10), _frame(9), (_event(10),))
