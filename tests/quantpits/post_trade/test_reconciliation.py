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


@pytest.mark.parametrize("label, side, standard_label", [
    ("本方卖出", "sell", "深圳A股普通股票竞价卖出"),
    ("全额卖出", "sell", "深圳A股普通股票竞价卖出"),
    ("对方买入", "buy", "深圳A股普通股票竞价买入"),
    ("对方卖出", "sell", "深圳A股普通股票竞价卖出"),
])
def test_gtja_order_labels_reconcile_with_standard_fills(label, side, standard_label):
    order = _frame(10).assign(交易类别=label)
    trade = _frame(10).assign(交易类别=standard_label)
    event = SettlementEvent(
        "2026-01-02", "SZ000001", side, Decimal("10"),
        Decimal("10"), Decimal("100"),
        Decimal("-105" if side == "buy" else "95"), 1, standard_label,
    )
    # A fully cancelled instruction contributes no fill quantity.
    cancelled = _frame(0).assign(
        交易类别=label, 证券代码="000002", 委托数量=100, 撤单数量=100, 委托状态="已撤",
    )
    order = pd.concat([order, cancelled], ignore_index=True)
    assert reconcile_quantities(order, trade, (event,)) == (
        (("2026-01-02", "SZ000001", side), Decimal("10")),
    )
    with pytest.raises(ExecutionReconciliationError, match="do not reconcile"):
        reconcile_quantities(order, trade.assign(成交数量=9), (event,))


@pytest.mark.parametrize("label", ["对方买入", "对方卖出", "未知买入", "未知卖出"])
def test_zero_fill_cancelled_order_does_not_require_direction(label):
    order = _frame(0).assign(交易类别=label, 委托数量=100, 撤单数量=100, 委托状态="已撤")
    assert reconcile_quantities(order, pd.DataFrame(), ()) == ()


@pytest.mark.parametrize("label", ["未知买入", "未知卖出"])
def test_unknown_execution_like_label_is_rejected(label):
    with pytest.raises(ExecutionReconciliationError, match="Unsupported filled execution side"):
        reconcile_quantities(_frame(10).assign(交易类别=label), _frame(10), (_event(10),))
