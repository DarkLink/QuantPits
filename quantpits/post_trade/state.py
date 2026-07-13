"""Pure deterministic post-trade account-state transitions.

This module deliberately has no filesystem, workspace, Qlib, or CLI imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Optional, Sequence, Tuple

from quantpits.post_trade.contracts import (
    CashReconciliationError, PositionCostError, PositionNotFoundError,
    PositionQuantityError, PostTradeStateInputError,
    SettlementDateMismatchError, SettlementNormalizationError,
    UnsupportedSettlementEventError, ValuationMissingError,
    ValuationSchemaError,
)
from quantpits.scripts.brokers.base import (
    BUY_TYPES, INTEREST_TYPES, POSITION_ADJUSTMENT_TYPES, SELL_TYPES,
)

MONEY_QUANTUM = Decimal("0.01")
ROUNDING_MODE = ROUND_HALF_UP
EventKind = Literal["buy", "sell", "cash_adjustment", "position_adjustment"]


def decimal_value(value, *, field="value") -> Decimal:
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError) as exc:
        raise PostTradeStateInputError("Invalid decimal for %s" % field) from exc
    if not result.is_finite():
        raise PostTradeStateInputError("Non-finite decimal for %s" % field)
    return result


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=ROUNDING_MODE)


def normalize_instrument(value) -> str:
    raw = str(value).strip().upper().split(".")[0]
    if raw.startswith(("SH", "SZ")) and len(raw) == 8 and raw[2:].isdigit():
        return raw
    raw = raw.zfill(6)
    if len(raw) == 6 and raw.isdigit():
        if raw.startswith("6"):
            return "SH" + raw
        if raw.startswith(("0", "3")):
            return "SZ" + raw
    raise SettlementNormalizationError("Unsupported instrument code")


def normalize_date(value) -> str:
    import pandas as pd
    text = str(value).strip().split(".")[0]
    if len(text) == 8 and text.isdigit():
        return "%s-%s-%s" % (text[:4], text[4:6], text[6:])
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception as exc:
        raise SettlementNormalizationError("Invalid settlement date") from exc


@dataclass(frozen=True)
class Position:
    instrument: str
    quantity: Decimal
    cost: Decimal

    def __post_init__(self):
        if self.quantity <= 0:
            raise PositionQuantityError("Position quantity must be positive: %s" % self.instrument)
        if self.cost < 0:
            raise PositionCostError("Position cost must not be negative: %s" % self.instrument)

    @property
    def average_cost(self) -> Decimal:
        return self.cost / self.quantity


@dataclass(frozen=True)
class AccountState:
    as_of_date: str
    cash: Decimal
    positions: Tuple[Position, ...]

    def __post_init__(self):
        names = [item.instrument for item in self.positions]
        if len(names) != len(set(names)):
            raise PostTradeStateInputError("Duplicate position instruments")
        if tuple(names) != tuple(sorted(names)):
            object.__setattr__(self, "positions", tuple(sorted(self.positions, key=lambda x: x.instrument)))


@dataclass(frozen=True)
class SettlementEvent:
    trade_date: str
    instrument: Optional[str]
    kind: EventKind
    quantity: Decimal
    price: Decimal
    gross_amount: Decimal
    cash_effect: Decimal
    source_row: int
    normalized_trade_type: str


@dataclass(frozen=True)
class ValuationSnapshot:
    trade_date: str
    closes: Tuple[Tuple[str, Decimal], ...]
    benchmark: Decimal
    quote_evidence: Tuple[object, ...] = ()
    benchmark_evidence: Optional[object] = None

    def __post_init__(self):
        names = tuple(name for name, _ in self.closes)
        if len(names) != len(set(names)):
            raise ValuationSchemaError("Duplicate valuation instruments")
        if self.quote_evidence:
            evidence = {item.instrument: item for item in self.quote_evidence}
            if set(evidence) != set(names) or len(evidence) != len(self.quote_evidence):
                raise ValuationSchemaError("Valuation evidence does not match close instruments")
            for instrument, close in self.closes:
                if evidence[instrument].price != close:
                    raise ValuationSchemaError("Valuation evidence price differs from operational close")

    def close_map(self) -> Mapping[str, Decimal]:
        return MappingProxyType(dict(self.closes))


@dataclass(frozen=True)
class DailyStateTransition:
    trade_date: str
    before: AccountState
    after: AccountState
    settlement_events: Tuple[SettlementEvent, ...]
    external_cashflow: Decimal
    realized_cost_removed: Decimal
    realized_pnl: Decimal
    valuation: ValuationSnapshot
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PostTradeStateChangeSet:
    initial_state: AccountState
    final_state: AccountState
    transitions: Tuple[DailyStateTransition, ...]
    processed_dates: Tuple[str, ...]
    consumed_cashflow_dates: Tuple[str, ...]
    next_prod_config: Mapping
    warnings: Tuple[str, ...] = ()


def account_state_from_config(config: Mapping) -> AccountState:
    cursor = config.get("last_processed_date", config.get("current_date"))
    if not cursor:
        raise PostTradeStateInputError("Missing account-state cursor")
    positions = []
    for item in config.get("current_holding", ()):
        positions.append(Position(
            normalize_instrument(item["instrument"]),
            decimal_value(item["value"], field="position quantity"),
            money(decimal_value(item["amount"], field="position cost")),
        ))
    return AccountState(str(cursor), money(decimal_value(config.get("current_cash", 0), field="cash")), tuple(positions))


def normalize_settlement_frame(frame, trade_date: str) -> Tuple[Tuple[SettlementEvent, ...], Tuple[str, ...]]:
    if frame is None or frame.empty:
        return (), ()
    required = {"交易类别", "成交价格", "成交数量", "成交金额", "资金发生数", "交收日期"}
    missing = required - set(frame.columns)
    if missing:
        raise SettlementNormalizationError("Settlement schema missing columns: %s" % sorted(missing))
    events, warnings = [], []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        row_date = normalize_date(row["交收日期"])
        if row_date != trade_date:
            raise SettlementDateMismatchError("Settlement row date does not match source date %s" % trade_date)
        category = str(row["交易类别"]).strip()
        quantity = decimal_value(row["成交数量"], field="quantity")
        price = decimal_value(row["成交价格"], field="price")
        gross = decimal_value(row["成交金额"], field="gross amount")
        cash = money(decimal_value(row["资金发生数"], field="cash effect"))
        if category in SELL_TYPES:
            kind, instrument = "sell", normalize_instrument(row["证券代码"])
        elif category in BUY_TYPES:
            kind, instrument = "buy", normalize_instrument(row["证券代码"])
        elif category in INTEREST_TYPES:
            kind, instrument, quantity = "cash_adjustment", None, Decimal("0")
        elif category in POSITION_ADJUSTMENT_TYPES:
            kind, instrument = "position_adjustment", normalize_instrument(row["证券代码"])
        elif cash == 0:
            warnings.append("Ignored zero-cash settlement category: %s" % category)
            continue
        else:
            raise UnsupportedSettlementEventError("Unsupported material settlement category: %s" % category)
        if kind in {"buy", "sell", "position_adjustment"} and quantity <= 0:
            raise SettlementNormalizationError("Position-changing quantity must be positive")
        if kind == "position_adjustment" and (price != 0 or gross != 0 or cash != 0):
            raise SettlementNormalizationError("Position adjustment must have zero price, gross amount, and cash effect")
        if kind == "buy" and cash > 0:
            raise SettlementNormalizationError("Buy cash effect must be non-positive")
        if kind == "sell" and cash < 0:
            raise SettlementNormalizationError("Sell cash effect must be non-negative")
        events.append(SettlementEvent(trade_date, instrument, kind, quantity, price, gross, cash, row_number, category))
    order = {"sell": 0, "buy": 1, "position_adjustment": 2, "cash_adjustment": 3}
    events.sort(key=lambda x: (order[x.kind], x.instrument or "", x.source_row))
    return tuple(events), tuple(warnings)


def transition_day(before: AccountState, trade_date: str, events: Sequence[SettlementEvent], external_cashflow: Decimal, valuation: ValuationSnapshot, warnings: Sequence[str] = ()) -> DailyStateTransition:
    positions = {item.instrument: item for item in before.positions}
    cash = before.cash
    removed_total = Decimal("0")
    realized_total = Decimal("0")
    settlement_cash = Decimal("0")
    for event in events:
        settlement_cash += event.cash_effect
        cash += event.cash_effect
        if event.kind == "sell":
            position = positions.get(event.instrument)
            if position is None:
                raise PositionNotFoundError("Cannot sell an instrument not held: %s" % event.instrument)
            if event.quantity > position.quantity:
                raise PositionQuantityError("Sell quantity exceeds position: %s" % event.instrument)
            if event.quantity == position.quantity:
                removed = position.cost
                del positions[event.instrument]
            else:
                removed = money(position.cost * event.quantity / position.quantity)
                positions[event.instrument] = Position(event.instrument, position.quantity - event.quantity, position.cost - removed)
            removed_total += removed
            realized_total += event.cash_effect - removed
        elif event.kind == "buy":
            bought_cost = -event.cash_effect
            old = positions.get(event.instrument)
            if old:
                positions[event.instrument] = Position(event.instrument, old.quantity + event.quantity, money(old.cost + bought_cost))
            else:
                positions[event.instrument] = Position(event.instrument, event.quantity, money(bought_cost))
        elif event.kind == "position_adjustment":
            old = positions.get(event.instrument)
            if old is None:
                raise PositionNotFoundError("Cannot adjust an instrument not held: %s" % event.instrument)
            positions[event.instrument] = Position(
                event.instrument, old.quantity + event.quantity, old.cost,
            )
    cash = money(cash + external_cashflow)
    expected = money(before.cash + settlement_cash + external_cashflow)
    if cash != expected:
        raise CashReconciliationError("Daily cash equation failed")
    closes = valuation.close_map()
    missing = sorted(set(positions) - set(closes))
    if missing:
        raise ValuationMissingError("Missing close prices for: %s" % ", ".join(missing))
    after = AccountState(trade_date, cash, tuple(positions.values()))
    return DailyStateTransition(trade_date, before, after, tuple(events), external_cashflow, money(removed_total), money(realized_total), valuation, tuple(warnings))


def build_change_set(initial: AccountState, dates: Sequence[str], events_by_date: Mapping[str, Sequence[SettlementEvent]], cashflows: Mapping[str, object], valuations: Mapping[str, ValuationSnapshot], raw_prod_config: Mapping, warnings_by_date: Optional[Mapping[str, Sequence[str]]] = None) -> PostTradeStateChangeSet:
    current, transitions, consumed, warnings = initial, [], [], []
    old_cashflow = cashflows.get("cash_flow_today") if "cashflows" not in cashflows else None
    dated = cashflows.get("cashflows", {})
    for index, date in enumerate(dates):
        external = decimal_value(dated.get(date, old_cashflow if index == 0 and old_cashflow is not None else 0), field="cashflow")
        if date in dated or (index == 0 and old_cashflow not in (None, 0, 0.0, "0")):
            consumed.append(date)
        if date not in valuations:
            raise ValuationMissingError("Missing valuation snapshot for %s" % date)
        daily_warnings = tuple((warnings_by_date or {}).get(date, ()))
        transition = transition_day(current, date, events_by_date.get(date, ()), external, valuations[date], daily_warnings)
        transitions.append(transition); current = transition.after; warnings.extend(daily_warnings)
    next_config = dict(raw_prod_config)
    if dates:
        next_config.update({
            "current_cash": float(current.cash),
            "current_holding": [
                {"instrument": x.instrument, "value": str(x.quantity), "amount": str(x.cost)}
                for x in current.positions
            ],
            "current_date": dates[-1], "last_processed_date": dates[-1],
        })
    return PostTradeStateChangeSet(initial, current, tuple(transitions), tuple(dates), tuple(consumed), MappingProxyType(next_config), tuple(warnings))
