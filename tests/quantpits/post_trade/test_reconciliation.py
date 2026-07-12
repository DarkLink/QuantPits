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


def test_bonus_share_trade_row_is_not_execution_fill():
    corporate_action = pd.DataFrame({
        "日期": ["2026-07-09"], "交易类别": ["上海A股红股上市入账"],
        "证券代码": ["600426"], "成交数量": [180],
    })
    assert reconcile_quantities(pd.DataFrame(), corporate_action, (), trade_date="2026-07-09") == ()
