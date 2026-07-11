"""Broker-neutral quantity reconciliation across order/trade/settlement evidence."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from quantpits.post_trade.contracts import ExecutionReconciliationError
from quantpits.post_trade.state import SettlementEvent, decimal_value, normalize_date, normalize_instrument
from quantpits.scripts.brokers.base import BUY_TYPES, SELL_TYPES

BUY_LABELS = set(BUY_TYPES) | {"证券买入", "买入"}
SELL_LABELS = set(SELL_TYPES) | {"证券卖出", "卖出"}


def normalize_side(value) -> str:
    text = str(value).strip()
    if text in BUY_LABELS:
        return "buy"
    if text in SELL_LABELS:
        return "sell"
    raise ExecutionReconciliationError("Unsupported filled execution side: %s" % text)


def _frame_quantities(frame, *, default_date=None):
    totals = defaultdict(Decimal)
    if frame is None or frame.empty:
        return totals
    for _, row in frame.iterrows():
        quantity = decimal_value(row.get("成交数量", 0), field="execution quantity")
        if quantity == 0:
            continue
        date_value = row.get("成交日期", row.get("日期", row.get("委托日期", default_date)))
        key = (normalize_date(date_value), normalize_instrument(row["证券代码"]), normalize_side(row["交易类别"]))
        totals[key] += quantity
    return totals


def reconcile_quantities(order_frame, trade_frame, settlement_events, *, trade_date=None):
    order = _frame_quantities(order_frame, default_date=trade_date)
    trade = _frame_quantities(trade_frame, default_date=trade_date)
    settlement = defaultdict(Decimal)
    for event in settlement_events:
        if event.kind in {"buy", "sell"}:
            settlement[(event.trade_date, event.instrument, event.kind)] += event.quantity
    keys = set(order) | set(trade) | set(settlement)
    mismatches = []
    for key in sorted(keys):
        values = (order[key], trade[key], settlement[key])
        if len(set(values)) != 1:
            mismatches.append("%s/%s/%s order=%s trade=%s settlement=%s" % (key + values))
    if mismatches:
        raise ExecutionReconciliationError("Execution quantities do not reconcile: " + "; ".join(mismatches))
    return tuple((key, order[key]) for key in sorted(keys))
