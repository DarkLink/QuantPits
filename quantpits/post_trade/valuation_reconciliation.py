"""Pure as-of and price-basis-aware account reconciliation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Optional, Tuple


@dataclass(frozen=True)
class ReconciliationIssue:
    code: str
    severity: str
    scope: str
    instrument: Optional[str]
    message: str


@dataclass(frozen=True)
class FieldComparison:
    field: str
    status: str
    internal_value: Optional[Decimal]
    broker_value: Optional[Decimal]
    difference: Optional[Decimal]
    reason_code: Optional[str] = None


@dataclass(frozen=True)
class AccountReconciliationReport:
    schema_version: int
    account_state_date: str
    snapshot_observed_at: str
    snapshot_effective_date: Optional[str]
    state_comparability: str
    price_comparability: str
    cash: FieldComparison
    quantities: Tuple[FieldComparison, ...]
    prices: Tuple[FieldComparison, ...]
    nav: FieldComparison
    issues: Tuple[ReconciliationIssue, ...]
    analytics_eligibility: str

    def to_public_dict(self):
        def convert(value):
            if isinstance(value, Decimal): return str(value)
            if isinstance(value, tuple): return [convert(item) for item in value]
            if isinstance(value, dict): return {key: convert(item) for key, item in value.items()}
            return value
        return convert(asdict(self))


def _comparison(field, internal, broker, tolerance=Decimal("0"), reason=None):
    if reason:
        return FieldComparison(field, "not_comparable", internal, broker, None, reason)
    difference = broker - internal
    return FieldComparison(field, "matched" if abs(difference) <= tolerance else "mismatch", internal, broker, difference)


def reconcile_account(account_state, valuation, broker_snapshot, *, cash_tolerance=Decimal("0.01"), value_tolerance=Decimal("0.01")):
    issues = []
    state_ok = broker_snapshot.effective_date == account_state.as_of_date
    if not state_ok:
        issues.append(ReconciliationIssue("state_date_mismatch" if broker_snapshot.effective_date else "missing_effective_date", "warning", "state", None, "Account state dates are not comparable."))
    cash = _comparison("cash", account_state.cash, broker_snapshot.cash, cash_tolerance, None if state_ok else "state_date_mismatch")
    internal_positions = {item.instrument: item for item in account_state.positions}
    broker_positions = {item.instrument: item for item in broker_snapshot.positions}
    quantities = []
    for name in sorted(set(internal_positions) | set(broker_positions)):
        internal = internal_positions.get(name)
        broker = broker_positions.get(name)
        quantities.append(_comparison(
            "quantity:%s" % name,
            internal.quantity if internal else None,
            broker.quantity if broker else None,
            reason="state_date_mismatch" if not state_ok else ("missing_instrument" if internal is None or broker is None else None),
        ))
    price_rows = []
    close_map = valuation.close_map()
    for name in sorted(set(internal_positions) | set(broker_positions)):
        broker = broker_positions.get(name)
        reason = None
        if not state_ok: reason = "state_date_mismatch"
        elif broker_snapshot.asserted_market_date is None: reason = "missing_market_date"
        elif broker_snapshot.asserted_market_date != valuation.trade_date: reason = "market_date_mismatch"
        elif broker and broker.corporate_action_status == "adjusted_in_advance": reason = "corporate_action_adjusted_in_advance"
        elif broker and broker.corporate_action_status != "none": reason = "price_basis_unknown"
        elif name not in close_map or broker is None: reason = "missing_instrument"
        price_rows.append(_comparison("price:%s" % name, close_map.get(name), broker.display_price if broker else None, value_tolerance, reason))
    cash_only = not internal_positions and not broker_positions
    comparable_prices = cash_only or (
        bool(price_rows) and all(item.status in {"matched", "mismatch"} for item in price_rows)
    )
    comparable_state = state_ok and cash.status in {"matched", "mismatch"} and all(item.status in {"matched", "mismatch"} for item in quantities)
    if comparable_state and comparable_prices:
        internal_nav = account_state.cash + sum((internal_positions[name].quantity * close_map[name] for name in internal_positions), Decimal("0"))
        nav = _comparison("nav", internal_nav, broker_snapshot.total_assets, value_tolerance)
    else:
        nav = FieldComparison("nav", "not_comparable", None, broker_snapshot.total_assets, None, "incomplete_comparability")
    any_mismatch = any(item.status == "mismatch" for item in (cash, nav) + tuple(quantities) + tuple(price_rows))
    unresolved = not comparable_state or not comparable_prices
    eligibility = "eligible" if not unresolved and not any_mismatch else ("blocked" if any_mismatch else "review_required")
    return AccountReconciliationReport(
        1, account_state.as_of_date, broker_snapshot.observed_at, broker_snapshot.effective_date,
        "comparable" if state_ok else "not_comparable",
        "not_applicable" if cash_only else ("comparable" if comparable_prices else "not_comparable"),
        cash, tuple(quantities), tuple(price_rows), nav, tuple(issues), eligibility,
    )
