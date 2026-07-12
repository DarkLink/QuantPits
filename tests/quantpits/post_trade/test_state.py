from decimal import Decimal

import pandas as pd
import pytest

from quantpits.post_trade.contracts import PositionQuantityError, ValuationMissingError
from quantpits.post_trade.state import (
    AccountState, Position, SettlementEvent, ValuationSnapshot,
    normalize_settlement_frame, transition_day,
)


def _sell(quantity, proceeds):
    return SettlementEvent("2026-01-02", "SZ000001", "sell", Decimal(str(quantity)), Decimal("12"), Decimal(str(proceeds)), Decimal(str(proceeds)), 1, "深圳A股普通股票竞价卖出")


def test_partial_sell_removes_average_cost_not_proceeds():
    before = AccountState("2026-01-01", Decimal("1000"), (Position("SZ000001", Decimal("100"), Decimal("1000")),))
    valuation = ValuationSnapshot("2026-01-02", (("SZ000001", Decimal("12")),), Decimal("4000"))
    result = transition_day(before, "2026-01-02", (_sell(40, 480),), Decimal("0"), valuation)
    assert result.after.positions[0].quantity == Decimal("60")
    assert result.after.positions[0].cost == Decimal("600.00")
    assert result.realized_cost_removed == Decimal("400.00")
    assert result.realized_pnl == Decimal("80.00")


def test_full_liquidation_absorbs_cost_and_oversell_fails():
    before = AccountState("2026-01-01", Decimal("0"), (Position("SZ000001", Decimal("3"), Decimal("10.01")),))
    valuation = ValuationSnapshot("2026-01-02", (), Decimal("4000"))
    result = transition_day(before, "2026-01-02", (_sell(3, 12),), Decimal("0"), valuation)
    assert not result.after.positions and result.realized_cost_removed == Decimal("10.01")
    with pytest.raises(PositionQuantityError):
        transition_day(before, "2026-01-02", (_sell(4, 12),), Decimal("0"), valuation)


def test_negative_dividend_tax_and_date_normalization():
    frame = pd.DataFrame({
        "证券代码": ["000001"], "交易类别": ["深圳A股红利税补缴"],
        "成交价格": [0], "成交数量": [0], "成交金额": [0],
        "资金发生数": [-5], "交收日期": [20260102],
    })
    events, _ = normalize_settlement_frame(frame, "2026-01-02")
    before = AccountState("2026-01-01", Decimal("100"), ())
    valuation = ValuationSnapshot("2026-01-02", (), Decimal("4000"))
    result = transition_day(before, "2026-01-02", events, Decimal("0"), valuation)
    assert result.after.cash == Decimal("95.00")


def test_bonus_share_listing_increases_quantity_without_changing_cost_or_cash():
    frame = pd.DataFrame({
        "证券代码": ["600426"], "交易类别": ["上海A股红股上市入账"],
        "成交价格": [0], "成交数量": [180], "成交金额": [0],
        "资金发生数": [0], "交收日期": [20260709],
    })
    events, warnings = normalize_settlement_frame(frame, "2026-07-09")
    before = AccountState(
        "2026-07-08", Decimal("100"),
        (Position("SH600426", Decimal("600"), Decimal("17404.20")),),
    )
    valuation = ValuationSnapshot("2026-07-09", (("SH600426", Decimal("18.63")),), Decimal("4000"))

    result = transition_day(before, "2026-07-09", events, Decimal("0"), valuation)

    assert not warnings
    assert result.after.cash == Decimal("100.00")
    assert result.after.positions[0].quantity == Decimal("780")
    assert result.after.positions[0].cost == Decimal("17404.20")


def test_missing_close_never_drops_position():
    before = AccountState("2026-01-01", Decimal("100"), (Position("SZ000001", Decimal("1"), Decimal("10")),))
    with pytest.raises(ValuationMissingError):
        transition_day(before, "2026-01-02", (), Decimal("0"), ValuationSnapshot("2026-01-02", (), Decimal("4000")))
