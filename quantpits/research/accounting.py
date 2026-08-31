"""Pure, exact accounting kernel for retrospective shadow portfolios.

The module deliberately has no workspace, market-data, clock, or filesystem
dependency.  Its public input constructors accept strict JSON-like values and
the transition is the sole constructor of capability-bearing results.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class ShadowAccountingContractError(ValueError):
    """Raised when a caller-provided accounting contract is invalid."""


_INSTRUMENT_RE = re.compile(r"^(?:SH|SZ)[0-9]{6}$")
_RESULT_TOKEN = object()
_ZERO = Decimal("0")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowAccountingContractError("%s must be an object" % field)
    return value


def _require_keys(value: Mapping[str, Any], expected: Sequence[str], field: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ShadowAccountingContractError(
            "%s fields mismatch (missing=%r, extra=%r)" % (field, missing, extra)
        )


def _strict_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ShadowAccountingContractError("%s must be a non-empty exact string" % field)
    return value


def _strict_decimal(
    value: Any,
    field: str,
    *,
    minimum: Optional[Decimal] = None,
    maximum_exclusive: Optional[Decimal] = None,
) -> Decimal:
    raw = _strict_string(value, field)
    try:
        result = Decimal(raw)
    except InvalidOperation as exc:
        raise ShadowAccountingContractError("%s must be a decimal string" % field) from exc
    if not result.is_finite():
        raise ShadowAccountingContractError("%s must be finite" % field)
    if minimum is not None and result < minimum:
        raise ShadowAccountingContractError("%s is below its minimum" % field)
    if maximum_exclusive is not None and result >= maximum_exclusive:
        raise ShadowAccountingContractError("%s is above its exclusive maximum" % field)
    return result


def _positive_decimal(value: Any, field: str) -> Decimal:
    result = _strict_decimal(value, field)
    if result <= _ZERO:
        raise ShadowAccountingContractError("%s must be positive" % field)
    return result


def _strict_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShadowAccountingContractError("%s must be a positive integer" % field)
    return value


def _strict_date(value: Any, field: str) -> str:
    raw = _strict_string(value, field)
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d")
    except ValueError as exc:
        raise ShadowAccountingContractError("%s must be YYYY-MM-DD" % field) from exc
    if parsed.strftime("%Y-%m-%d") != raw:
        raise ShadowAccountingContractError("%s must be YYYY-MM-DD" % field)
    return raw


def _instrument(value: Any, field: str) -> str:
    raw = _strict_string(value, field)
    if _INSTRUMENT_RE.fullmatch(raw) is None:
        raise ShadowAccountingContractError("%s must be a canonical SH/SZ instrument" % field)
    return raw


def _enum(value: Any, expected: str, field: str) -> str:
    raw = _strict_string(value, field)
    if raw != expected:
        raise ShadowAccountingContractError("%s must be %s" % (field, expected))
    return raw


def _decimal_text(value: Decimal) -> str:
    if value == _ZERO:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _new_frozen(cls: Any, **values: Any) -> Any:
    obj = object.__new__(cls)
    for field, value in values.items():
        object.__setattr__(obj, field, value)
    return obj


def _money(value: Decimal, assumption: "ExecutionAssumption") -> Decimal:
    units = (value / assumption.money_quantum).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return units * assumption.money_quantum


def _price(value: Decimal, assumption: "ExecutionAssumption") -> Decimal:
    units = (value / assumption.price_quantum).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return units * assumption.price_quantum


def _transition_precision(
    prior: "ShadowPortfolioState",
    intents: "ShadowIntentBatch",
    quotes: "ShadowQuoteSnapshot",
    assumption: "ExecutionAssumption",
) -> int:
    decimals = [prior.cash]
    decimals.extend(item.book_cost for item in prior.positions)
    decimals.extend(item.price for item in quotes.quotes if item.price is not None)
    decimals.extend((
        assumption.money_quantum,
        assumption.price_quantum,
        assumption.buy_slippage_rate,
        assumption.sell_slippage_rate,
        assumption.buy_fee_rate,
        assumption.sell_fee_rate,
        assumption.minimum_buy_fee,
        assumption.minimum_sell_fee,
    ))
    decimal_span = max(
        len(value.as_tuple().digits) + abs(value.as_tuple().exponent)
        for value in decimals
    )
    quantities = [assumption.lot_size]
    quantities.extend(item.quantity for item in prior.positions)
    quantities.extend(item.quantity for item in intents.intents)
    quantity_span = max(len(str(value)) for value in quantities)
    member_span = len(str(len(prior.positions) + len(intents.intents) + 1))
    return max(64, decimal_span * 2 + quantity_span * 3 + member_span + 32)


@dataclass(frozen=True, init=False)
class ShadowPosition:
    instrument: str
    quantity: int
    book_cost: Decimal

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("positions require strict aggregate construction")

    @classmethod
    def _from_mapping(cls, value: Any, field: str) -> "ShadowPosition":
        row = _require_mapping(value, field)
        _require_keys(row, ("instrument", "quantity", "book_cost"), field)
        return _new_frozen(
            cls,
            instrument=_instrument(row["instrument"], field + ".instrument"),
            quantity=_strict_positive_int(row["quantity"], field + ".quantity"),
            book_cost=_strict_decimal(row["book_cost"], field + ".book_cost", minimum=_ZERO),
        )

    def _raw(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "quantity": self.quantity,
            "book_cost": _decimal_text(self.book_cost),
        }


@dataclass(frozen=True, init=False)
class ShadowPortfolioState:
    portfolio_id: str
    as_of_date: str
    cash: Decimal
    positions: Tuple[ShadowPosition, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("states require from_dict")

    @classmethod
    def from_dict(cls, value: Any) -> "ShadowPortfolioState":
        raw = _require_mapping(value, "state")
        _require_keys(raw, ("portfolio_id", "as_of_date", "cash", "positions"), "state")
        rows = raw["positions"]
        if not isinstance(rows, (list, tuple)):
            raise ShadowAccountingContractError("state.positions must be a list")
        positions = tuple(
            ShadowPosition._from_mapping(row, "state.positions[%d]" % index)
            for index, row in enumerate(rows)
        )
        identities = tuple(item.instrument for item in positions)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ShadowAccountingContractError("state.positions must be unique and sorted")
        return _new_frozen(
            cls,
            portfolio_id=_strict_string(raw["portfolio_id"], "state.portfolio_id"),
            as_of_date=_strict_date(raw["as_of_date"], "state.as_of_date"),
            cash=_strict_decimal(raw["cash"], "state.cash"),
            positions=positions,
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "as_of_date": self.as_of_date,
            "cash": _decimal_text(self.cash),
            "positions": [item._raw() for item in self.positions],
        }

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["digest"] = self.digest
        return payload

    def _validated_copy(self) -> "ShadowPortfolioState":
        return self.from_dict(self._payload())


@dataclass(frozen=True, init=False)
class ShadowIntent:
    intent_id: str
    portfolio_id: str
    cycle_id: str
    trade_date: str
    side: str
    instrument: str
    quantity: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("intents require strict aggregate construction")

    @classmethod
    def _from_mapping(cls, value: Any, field: str) -> "ShadowIntent":
        raw = _require_mapping(value, field)
        expected = (
            "intent_id", "portfolio_id", "cycle_id", "trade_date",
            "side", "instrument", "quantity",
        )
        _require_keys(raw, expected, field)
        side = _strict_string(raw["side"], field + ".side")
        if side not in ("SELL", "BUY"):
            raise ShadowAccountingContractError("%s.side must be SELL or BUY" % field)
        return _new_frozen(
            cls,
            intent_id=_strict_string(raw["intent_id"], field + ".intent_id"),
            portfolio_id=_strict_string(raw["portfolio_id"], field + ".portfolio_id"),
            cycle_id=_strict_string(raw["cycle_id"], field + ".cycle_id"),
            trade_date=_strict_date(raw["trade_date"], field + ".trade_date"),
            side=side,
            instrument=_instrument(raw["instrument"], field + ".instrument"),
            quantity=_strict_positive_int(raw["quantity"], field + ".quantity"),
        )

    def _raw(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "portfolio_id": self.portfolio_id,
            "cycle_id": self.cycle_id,
            "trade_date": self.trade_date,
            "side": self.side,
            "instrument": self.instrument,
            "quantity": self.quantity,
        }


@dataclass(frozen=True, init=False)
class ShadowIntentBatch:
    portfolio_id: str
    cycle_id: str
    trade_date: str
    intents: Tuple[ShadowIntent, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("intent batches require from_iterable")

    @classmethod
    def from_iterable(
        cls,
        *,
        portfolio_id: Any,
        cycle_id: Any,
        trade_date: Any,
        rows: Iterable[Any],
    ) -> "ShadowIntentBatch":
        canonical_portfolio = _strict_string(portfolio_id, "intents.portfolio_id")
        canonical_cycle = _strict_string(cycle_id, "intents.cycle_id")
        canonical_date = _strict_date(trade_date, "intents.trade_date")
        if isinstance(rows, (str, bytes, Mapping)):
            raise ShadowAccountingContractError("intents.rows must be an iterable of objects")
        try:
            intents = tuple(
                ShadowIntent._from_mapping(row, "intents.rows[%d]" % index)
                for index, row in enumerate(rows)
            )
        except TypeError as exc:
            raise ShadowAccountingContractError("intents.rows must be iterable") from exc
        seen_buy = False
        ids = set()
        instruments = set()
        for item in intents:
            if (item.portfolio_id, item.cycle_id, item.trade_date) != (
                canonical_portfolio, canonical_cycle, canonical_date
            ):
                raise ShadowAccountingContractError("intent identity does not match its batch")
            if item.intent_id in ids or item.instrument in instruments:
                raise ShadowAccountingContractError("intent IDs and instruments must be unique")
            ids.add(item.intent_id)
            instruments.add(item.instrument)
            if item.side == "BUY":
                seen_buy = True
            elif seen_buy:
                raise ShadowAccountingContractError("all SELL intents must precede BUY intents")
        return _new_frozen(
            cls,
            portfolio_id=canonical_portfolio,
            cycle_id=canonical_cycle,
            trade_date=canonical_date,
            intents=intents,
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "cycle_id": self.cycle_id,
            "trade_date": self.trade_date,
            "intents": [item._raw() for item in self.intents],
        }

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["digest"] = self.digest
        return payload

    def _validated_copy(self) -> "ShadowIntentBatch":
        return self.from_iterable(
            portfolio_id=self.portfolio_id,
            cycle_id=self.cycle_id,
            trade_date=self.trade_date,
            rows=[item._raw() for item in self.intents],
        )


@dataclass(frozen=True, init=False)
class ExecutionAssumption:
    schema_version: int
    assumption_id: str
    deal_price: str
    sell_before_buy: bool
    partial_fill: bool
    short_sell: bool
    lot_size: int
    money_quantum: Decimal
    price_quantum: Decimal
    rounding_mode: str
    buy_slippage_rate: Decimal
    sell_slippage_rate: Decimal
    buy_fee_rate: Decimal
    sell_fee_rate: Decimal
    minimum_buy_fee: Decimal
    minimum_sell_fee: Decimal
    corporate_action_mode: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("assumptions require from_dict")

    _FIELDS = (
        "schema_version", "assumption_id", "deal_price", "sell_before_buy",
        "partial_fill", "short_sell", "lot_size", "money_quantum", "price_quantum",
        "rounding_mode", "buy_slippage_rate", "sell_slippage_rate", "buy_fee_rate",
        "sell_fee_rate", "minimum_buy_fee", "minimum_sell_fee", "corporate_action_mode",
    )

    @classmethod
    def from_dict(cls, value: Any) -> "ExecutionAssumption":
        raw = _require_mapping(value, "assumption")
        _require_keys(raw, cls._FIELDS, "assumption")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ShadowAccountingContractError("assumption.schema_version must be integer 1")
        for field, expected in (
            ("sell_before_buy", True), ("partial_fill", False), ("short_sell", False)
        ):
            if type(raw[field]) is not bool or raw[field] is not expected:
                raise ShadowAccountingContractError("assumption.%s must be %r" % (field, expected))
        rate_fields = (
            "buy_slippage_rate", "sell_slippage_rate", "buy_fee_rate", "sell_fee_rate"
        )
        rates = {
            field: _strict_decimal(
                raw[field], "assumption." + field, minimum=_ZERO, maximum_exclusive=Decimal("1")
            )
            for field in rate_fields
        }
        return _new_frozen(
            cls,
            schema_version=1,
            assumption_id=_strict_string(raw["assumption_id"], "assumption.assumption_id"),
            deal_price=_enum(raw["deal_price"], "NEXT_OPEN", "assumption.deal_price"),
            sell_before_buy=True,
            partial_fill=False,
            short_sell=False,
            lot_size=_strict_positive_int(raw["lot_size"], "assumption.lot_size"),
            money_quantum=_positive_decimal(raw["money_quantum"], "assumption.money_quantum"),
            price_quantum=_positive_decimal(raw["price_quantum"], "assumption.price_quantum"),
            rounding_mode=_enum(raw["rounding_mode"], "HALF_UP", "assumption.rounding_mode"),
            buy_slippage_rate=rates["buy_slippage_rate"],
            sell_slippage_rate=rates["sell_slippage_rate"],
            buy_fee_rate=rates["buy_fee_rate"],
            sell_fee_rate=rates["sell_fee_rate"],
            minimum_buy_fee=_strict_decimal(
                raw["minimum_buy_fee"], "assumption.minimum_buy_fee", minimum=_ZERO
            ),
            minimum_sell_fee=_strict_decimal(
                raw["minimum_sell_fee"], "assumption.minimum_sell_fee", minimum=_ZERO
            ),
            corporate_action_mode=_enum(
                raw["corporate_action_mode"], "UNMODELED", "assumption.corporate_action_mode"
            ),
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assumption_id": self.assumption_id,
            "deal_price": self.deal_price,
            "sell_before_buy": self.sell_before_buy,
            "partial_fill": self.partial_fill,
            "short_sell": self.short_sell,
            "lot_size": self.lot_size,
            "money_quantum": _decimal_text(self.money_quantum),
            "price_quantum": _decimal_text(self.price_quantum),
            "rounding_mode": self.rounding_mode,
            "buy_slippage_rate": _decimal_text(self.buy_slippage_rate),
            "sell_slippage_rate": _decimal_text(self.sell_slippage_rate),
            "buy_fee_rate": _decimal_text(self.buy_fee_rate),
            "sell_fee_rate": _decimal_text(self.sell_fee_rate),
            "minimum_buy_fee": _decimal_text(self.minimum_buy_fee),
            "minimum_sell_fee": _decimal_text(self.minimum_sell_fee),
            "corporate_action_mode": self.corporate_action_mode,
        }

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["digest"] = self.digest
        return payload

    def _validated_copy(self) -> "ExecutionAssumption":
        return self.from_dict(self._payload())


@dataclass(frozen=True, init=False)
class ShadowQuote:
    instrument: str
    trade_date: str
    field: str
    status: str
    price: Optional[Decimal]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("quotes require strict aggregate construction")

    @classmethod
    def _from_mapping(cls, value: Any, field_name: str) -> "ShadowQuote":
        raw = _require_mapping(value, field_name)
        _require_keys(raw, ("instrument", "trade_date", "field", "status", "price"), field_name)
        status = _strict_string(raw["status"], field_name + ".status")
        if status not in ("OBSERVED", "MISSING"):
            raise ShadowAccountingContractError("%s.status is invalid" % field_name)
        if status == "OBSERVED":
            price = _positive_decimal(raw["price"], field_name + ".price")
        else:
            if raw["price"] is not None:
                raise ShadowAccountingContractError("MISSING quote price must be null")
            price = None
        return _new_frozen(
            cls,
            instrument=_instrument(raw["instrument"], field_name + ".instrument"),
            trade_date=_strict_date(raw["trade_date"], field_name + ".trade_date"),
            field=_enum(raw["field"], "NEXT_OPEN", field_name + ".field"),
            status=status,
            price=price,
        )

    def _raw(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "trade_date": self.trade_date,
            "field": self.field,
            "status": self.status,
            "price": None if self.price is None else _decimal_text(self.price),
        }


@dataclass(frozen=True, init=False)
class ShadowQuoteSnapshot:
    portfolio_id: str
    cycle_id: str
    trade_date: str
    requested_instruments: Tuple[str, ...]
    quotes: Tuple[ShadowQuote, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("quote snapshots require from_iterable")

    @classmethod
    def from_iterable(
        cls,
        *,
        portfolio_id: Any,
        cycle_id: Any,
        trade_date: Any,
        requested_instruments: Iterable[Any],
        rows: Iterable[Any],
    ) -> "ShadowQuoteSnapshot":
        canonical_portfolio = _strict_string(portfolio_id, "quotes.portfolio_id")
        canonical_cycle = _strict_string(cycle_id, "quotes.cycle_id")
        canonical_date = _strict_date(trade_date, "quotes.trade_date")
        if isinstance(requested_instruments, (str, bytes, Mapping)):
            raise ShadowAccountingContractError("requested instruments must be an iterable")
        try:
            requested = tuple(
                _instrument(item, "quotes.requested_instruments[%d]" % index)
                for index, item in enumerate(requested_instruments)
            )
        except TypeError as exc:
            raise ShadowAccountingContractError("requested instruments must be iterable") from exc
        if requested != tuple(sorted(set(requested))):
            raise ShadowAccountingContractError("requested instruments must be unique and sorted")
        if isinstance(rows, (str, bytes, Mapping)):
            raise ShadowAccountingContractError("quotes.rows must be an iterable of objects")
        try:
            quotes = tuple(
                ShadowQuote._from_mapping(row, "quotes.rows[%d]" % index)
                for index, row in enumerate(rows)
            )
        except TypeError as exc:
            raise ShadowAccountingContractError("quotes.rows must be iterable") from exc
        identities = tuple(item.instrument for item in quotes)
        if identities != requested:
            raise ShadowAccountingContractError("quotes must exactly cover the requested instruments")
        if any(item.trade_date != canonical_date for item in quotes):
            raise ShadowAccountingContractError("quote trade date does not match its snapshot")
        return _new_frozen(
            cls,
            portfolio_id=canonical_portfolio,
            cycle_id=canonical_cycle,
            trade_date=canonical_date,
            requested_instruments=requested,
            quotes=quotes,
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "cycle_id": self.cycle_id,
            "trade_date": self.trade_date,
            "requested_instruments": list(self.requested_instruments),
            "quotes": [item._raw() for item in self.quotes],
        }

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["digest"] = self.digest
        return payload

    def _validated_copy(self) -> "ShadowQuoteSnapshot":
        return self.from_iterable(
            portfolio_id=self.portfolio_id,
            cycle_id=self.cycle_id,
            trade_date=self.trade_date,
            requested_instruments=self.requested_instruments,
            rows=[item._raw() for item in self.quotes],
        )


@dataclass(frozen=True, init=False)
class ShadowIntentResult:
    intent_id: str
    side: str
    instrument: str
    quantity: int
    status: str
    reference_price: Optional[Decimal]
    execution_price: Optional[Decimal]
    gross: Optional[Decimal]
    fee: Optional[Decimal]
    cash_effect: Optional[Decimal]
    removed_book_cost: Optional[Decimal]
    realized_pnl: Optional[Decimal]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("terminal results are transition-owned")

    @classmethod
    def _create(cls, token: object, **values: Any) -> "ShadowIntentResult":
        if token is not _RESULT_TOKEN:
            raise ShadowAccountingContractError("terminal results are transition-owned")
        obj = object.__new__(cls)
        for field, value in values.items():
            object.__setattr__(obj, field, value)
        obj._validate()
        return obj

    def _validate(self) -> None:
        statuses = {
            "FILLED", "NO_FILL_MISSING_PRICE", "NO_FILL_INSUFFICIENT_POSITION",
            "NO_FILL_INSUFFICIENT_CASH",
        }
        if self.status not in statuses:
            raise ShadowAccountingContractError("invalid terminal status")
        values = (
            self.reference_price, self.execution_price, self.gross, self.fee,
            self.cash_effect, self.removed_book_cost, self.realized_pnl,
        )
        if self.status == "FILLED" and any(value is None for value in values):
            raise ShadowAccountingContractError("filled result must contain all accounting values")
        if self.status != "FILLED" and any(value is not None for value in values):
            raise ShadowAccountingContractError("no-fill result accounting values must be null")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "side": self.side,
            "instrument": self.instrument,
            "quantity": self.quantity,
            "status": self.status,
            "reference_price": None if self.reference_price is None else _decimal_text(self.reference_price),
            "execution_price": None if self.execution_price is None else _decimal_text(self.execution_price),
            "gross": None if self.gross is None else _decimal_text(self.gross),
            "fee": None if self.fee is None else _decimal_text(self.fee),
            "cash_effect": None if self.cash_effect is None else _decimal_text(self.cash_effect),
            "removed_book_cost": None if self.removed_book_cost is None else _decimal_text(self.removed_book_cost),
            "realized_pnl": None if self.realized_pnl is None else _decimal_text(self.realized_pnl),
        }


@dataclass(frozen=True, init=False)
class ShadowTransitionResult:
    schema_version: int
    evidence_class: str
    prospective_claim: bool
    promotion_capability: bool
    corporate_action_mode: str
    warnings: Tuple[str, ...]
    portfolio_id: str
    cycle_id: str
    trade_date: str
    assumption: ExecutionAssumption
    prior_state: ShadowPortfolioState
    requested_intents: ShadowIntentBatch
    quotes: ShadowQuoteSnapshot
    terminal_results: Tuple[ShadowIntentResult, ...]
    after_state: ShadowPortfolioState
    filled_count: int
    no_fill_count: int
    all_filled: bool
    gross_buy: Decimal
    gross_sell: Decimal
    total_fee: Decimal
    slippage_cost: Decimal
    rounding_adjustment: Optional[Decimal]
    realized_pnl: Decimal
    valuation_status: str
    missing_valuation_instruments: Tuple[str, ...]
    nav_before: Optional[Decimal]
    nav_after: Optional[Decimal]
    nav_reconciled: bool
    transition_status: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ShadowAccountingContractError("transition results are transition-owned")

    @classmethod
    def _create(cls, token: object, **values: Any) -> "ShadowTransitionResult":
        if token is not _RESULT_TOKEN:
            raise ShadowAccountingContractError("transition results are transition-owned")
        obj = object.__new__(cls)
        for field, value in values.items():
            object.__setattr__(obj, field, value)
        obj._validate()
        return obj

    def _validate(self) -> None:
        if (
            self.schema_version != 1
            or self.evidence_class != "RETROSPECTIVE_TECHNICAL_REPLAY"
            or self.prospective_claim is not False
            or self.promotion_capability is not False
            or self.corporate_action_mode != "UNMODELED"
            or self.warnings != ("CORPORATE_ACTIONS_UNMODELED",)
        ):
            raise ShadowAccountingContractError("transition evidence authority fields are inconsistent")
        if self.transition_status != "COMPLETE":
            raise ShadowAccountingContractError("only a complete transition result can be created")
        identity = (self.portfolio_id, self.cycle_id, self.trade_date)
        if (
            (self.requested_intents.portfolio_id, self.requested_intents.cycle_id,
             self.requested_intents.trade_date) != identity
            or (self.quotes.portfolio_id, self.quotes.cycle_id, self.quotes.trade_date) != identity
            or self.prior_state.portfolio_id != self.portfolio_id
            or self.after_state.portfolio_id != self.portfolio_id
            or self.after_state.as_of_date != self.trade_date
            or self.assumption.corporate_action_mode != self.corporate_action_mode
        ):
            raise ShadowAccountingContractError("transition aggregate identity fields are inconsistent")
        requested = self.requested_intents.intents
        if len(requested) != len(self.terminal_results):
            raise ShadowAccountingContractError("terminal result cardinality mismatch")
        for intent, terminal in zip(requested, self.terminal_results):
            terminal._validate()
            if (intent.intent_id, intent.side, intent.instrument, intent.quantity) != (
                terminal.intent_id, terminal.side, terminal.instrument, terminal.quantity
            ):
                raise ShadowAccountingContractError("terminal result identity mismatch")

        positions = {
            item.instrument: (item.quantity, item.book_cost) for item in self.prior_state.positions
        }
        quote_map = {item.instrument: item for item in self.quotes.quotes}
        cash = self.prior_state.cash
        cash_floor = min(cash, _ZERO)
        expected_terminals = []
        for intent in requested:
            expected, cash = ShadowPortfolioTransition._execute_intent(
                intent, quote_map[intent.instrument], positions, cash, self.assumption
            )
            if cash < cash_floor:
                raise ShadowAccountingContractError("transition would worsen the opening deficit")
            expected_terminals.append(expected)
        if tuple(expected_terminals) != self.terminal_results:
            raise ShadowAccountingContractError("terminal accounting facts are inconsistent")
        expected_after = ShadowPortfolioState.from_dict({
            "portfolio_id": self.portfolio_id,
            "as_of_date": self.trade_date,
            "cash": _decimal_text(_money(cash, self.assumption)),
            "positions": [
                {
                    "instrument": instrument,
                    "quantity": quantity,
                    "book_cost": _decimal_text(_money(cost, self.assumption)),
                }
                for instrument, (quantity, cost) in sorted(positions.items())
            ],
        })
        if expected_after != self.after_state:
            raise ShadowAccountingContractError("after-state is inconsistent with terminal results")

        filled = sum(item.status == "FILLED" for item in self.terminal_results)
        if (self.filled_count, self.no_fill_count, self.all_filled) != (
            filled, len(requested) - filled, filled == len(requested)
        ):
            raise ShadowAccountingContractError("terminal result counts are inconsistent")
        filled_members = tuple(
            item for item in self.terminal_results if item.status == "FILLED"
        )
        expected_gross_buy = _money(sum((
            item.gross for item in filled_members if item.side == "BUY"
        ), _ZERO), self.assumption)
        expected_gross_sell = _money(sum((
            item.gross for item in filled_members if item.side == "SELL"
        ), _ZERO), self.assumption)
        expected_fee = _money(sum((item.fee for item in filled_members), _ZERO), self.assumption)
        expected_realized = _money(sum((
            item.realized_pnl for item in filled_members
        ), _ZERO), self.assumption)
        expected_slippage = _money(sum((
            item.quantity * (
                item.execution_price - item.reference_price
                if item.side == "BUY"
                else item.reference_price - item.execution_price
            )
            for item in filled_members
        ), _ZERO), self.assumption)
        if (
            self.gross_buy != expected_gross_buy
            or self.gross_sell != expected_gross_sell
            or self.total_fee != expected_fee
            or self.realized_pnl != expected_realized
            or self.slippage_cost != expected_slippage
        ):
            raise ShadowAccountingContractError("transition accounting aggregates are inconsistent")

        valuation_instruments = tuple(sorted(
            {item.instrument for item in self.prior_state.positions}
            | {item.instrument for item in self.after_state.positions}
        ))
        expected_missing = tuple(
            instrument for instrument in valuation_instruments
            if quote_map[instrument].status == "MISSING"
        )
        if self.missing_valuation_instruments != expected_missing:
            raise ShadowAccountingContractError("valuation missing set is inconsistent")
        if self.valuation_status == "COMPLETE":
            if self.missing_valuation_instruments or self.nav_before is None or self.nav_after is None:
                raise ShadowAccountingContractError("complete valuation fields are inconsistent")
            if self.rounding_adjustment is None or not self.nav_reconciled:
                raise ShadowAccountingContractError("complete valuation must reconcile")
            expected_before = _money(self.prior_state.cash + sum((
                item.quantity * quote_map[item.instrument].price
                for item in self.prior_state.positions
            ), _ZERO), self.assumption)
            expected_after_nav = _money(self.after_state.cash + sum((
                item.quantity * quote_map[item.instrument].price
                for item in self.after_state.positions
            ), _ZERO), self.assumption)
            expected_adjustment = _money(
                expected_before - expected_fee - expected_slippage - expected_after_nav,
                self.assumption,
            )
            if (
                self.nav_before != expected_before
                or self.nav_after != expected_after_nav
                or self.rounding_adjustment != expected_adjustment
                or self.nav_after
                != self.nav_before - self.total_fee - self.slippage_cost - self.rounding_adjustment
            ):
                raise ShadowAccountingContractError("complete NAV accounting is inconsistent")
        elif self.valuation_status == "PARTIAL":
            if not self.missing_valuation_instruments:
                raise ShadowAccountingContractError("partial valuation must identify missing instruments")
            if (
                self.nav_before is not None or self.nav_after is not None
                or self.rounding_adjustment is not None or self.nav_reconciled
            ):
                raise ShadowAccountingContractError("partial valuation cannot grant NAV capability")
        else:
            raise ShadowAccountingContractError("invalid valuation status")

    def _payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_class": self.evidence_class,
            "prospective_claim": self.prospective_claim,
            "promotion_capability": self.promotion_capability,
            "corporate_action_mode": self.corporate_action_mode,
            "warnings": list(self.warnings),
            "portfolio_id": self.portfolio_id,
            "cycle_id": self.cycle_id,
            "trade_date": self.trade_date,
            "assumption": self.assumption.to_dict(),
            "prior_state": self.prior_state.to_dict(),
            "requested_intents": self.requested_intents.to_dict(),
            "quotes": self.quotes.to_dict(),
            "terminal_results": [item.to_dict() for item in self.terminal_results],
            "after_state": self.after_state.to_dict(),
            "filled_count": self.filled_count,
            "no_fill_count": self.no_fill_count,
            "all_filled": self.all_filled,
            "gross_buy": _decimal_text(self.gross_buy),
            "gross_sell": _decimal_text(self.gross_sell),
            "total_fee": _decimal_text(self.total_fee),
            "slippage_cost": _decimal_text(self.slippage_cost),
            "rounding_adjustment": (
                None if self.rounding_adjustment is None
                else _decimal_text(self.rounding_adjustment)
            ),
            "realized_pnl": _decimal_text(self.realized_pnl),
            "valuation_status": self.valuation_status,
            "missing_valuation_instruments": list(self.missing_valuation_instruments),
            "nav_before": None if self.nav_before is None else _decimal_text(self.nav_before),
            "nav_after": None if self.nav_after is None else _decimal_text(self.nav_after),
            "nav_reconciled": self.nav_reconciled,
            "transition_status": self.transition_status,
        }

    @property
    def result_digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["result_digest"] = self.result_digest
        return payload

    def to_canonical_json_bytes(self) -> bytes:
        """Render the complete result, including its digest, deterministically."""
        return _canonical_json(self.to_dict())


class ShadowPortfolioTransition:
    """Truth owner for an exact, synthetic shadow-portfolio transition."""

    @staticmethod
    def _terminal(intent: ShadowIntent, status: str, **values: Any) -> ShadowIntentResult:
        empty = {
            "reference_price": None, "execution_price": None, "gross": None, "fee": None,
            "cash_effect": None, "removed_book_cost": None, "realized_pnl": None,
        }
        empty.update(values)
        return ShadowIntentResult._create(
            _RESULT_TOKEN,
            intent_id=intent.intent_id,
            side=intent.side,
            instrument=intent.instrument,
            quantity=intent.quantity,
            status=status,
            **empty,
        )

    @staticmethod
    def _execute_intent(
        intent: ShadowIntent,
        quote: ShadowQuote,
        positions: Dict[str, Tuple[int, Decimal]],
        cash: Decimal,
        assumption: ExecutionAssumption,
    ) -> Tuple[ShadowIntentResult, Decimal]:
        current = positions.get(intent.instrument)
        if intent.side == "SELL" and (current is None or intent.quantity > current[0]):
            return ShadowPortfolioTransition._terminal(
                intent, "NO_FILL_INSUFFICIENT_POSITION"
            ), cash
        if quote.status == "MISSING":
            return ShadowPortfolioTransition._terminal(intent, "NO_FILL_MISSING_PRICE"), cash
        if quote.price is None:
            raise ShadowAccountingContractError("OBSERVED quote must carry a price")
        if intent.side == "BUY":
            execution_price = _price(
                quote.price * (Decimal("1") + assumption.buy_slippage_rate), assumption
            )
            if execution_price <= _ZERO:
                raise ShadowAccountingContractError("rounded BUY execution price must be positive")
            gross = _money(execution_price * intent.quantity, assumption)
            fee = _money(max(assumption.minimum_buy_fee, gross * assumption.buy_fee_rate), assumption)
            required = gross + fee
            if cash < required:
                return ShadowPortfolioTransition._terminal(
                    intent, "NO_FILL_INSUFFICIENT_CASH"
                ), cash
            quantity, cost = positions.get(intent.instrument, (0, _ZERO))
            positions[intent.instrument] = (quantity + intent.quantity, cost + required)
            effect = -required
            terminal = ShadowPortfolioTransition._terminal(
                intent,
                "FILLED",
                reference_price=quote.price,
                execution_price=execution_price,
                gross=gross,
                fee=fee,
                cash_effect=effect,
                removed_book_cost=_money(_ZERO, assumption),
                realized_pnl=_money(_ZERO, assumption),
            )
            return terminal, cash + effect

        quantity, cost = current  # type: ignore[misc]
        execution_price = _price(
            quote.price * (Decimal("1") - assumption.sell_slippage_rate), assumption
        )
        if execution_price <= _ZERO:
            raise ShadowAccountingContractError("rounded SELL execution price must be positive")
        gross = _money(execution_price * intent.quantity, assumption)
        fee = _money(max(assumption.minimum_sell_fee, gross * assumption.sell_fee_rate), assumption)
        effect = gross - fee
        if effect < _ZERO:
            raise ShadowAccountingContractError(
                "SELL cash effect must be non-negative; negative cash would worsen the deficit"
            )
        if intent.quantity == quantity:
            removed = cost
            del positions[intent.instrument]
        else:
            removed = _money(cost * intent.quantity / quantity, assumption)
            positions[intent.instrument] = (quantity - intent.quantity, cost - removed)
        realized = effect - removed
        terminal = ShadowPortfolioTransition._terminal(
            intent,
            "FILLED",
            reference_price=quote.price,
            execution_price=execution_price,
            gross=gross,
            fee=fee,
            cash_effect=effect,
            removed_book_cost=removed,
            realized_pnl=realized,
        )
        return terminal, cash + effect

    @classmethod
    def apply(
        cls,
        *,
        prior: ShadowPortfolioState,
        intents: ShadowIntentBatch,
        quotes: ShadowQuoteSnapshot,
        assumption: ExecutionAssumption,
    ) -> ShadowTransitionResult:
        if type(prior) is not ShadowPortfolioState or type(intents) is not ShadowIntentBatch:
            raise ShadowAccountingContractError("prior and intents must be canonical accounting inputs")
        if type(quotes) is not ShadowQuoteSnapshot or type(assumption) is not ExecutionAssumption:
            raise ShadowAccountingContractError("quotes and assumption must be canonical accounting inputs")
        prior = prior._validated_copy()
        intents = intents._validated_copy()
        quotes = quotes._validated_copy()
        assumption = assumption._validated_copy()
        with localcontext() as context:
            context.prec = _transition_precision(prior, intents, quotes, assumption)
            context.rounding = ROUND_HALF_UP
            return cls._apply_canonical(
                prior=prior, intents=intents, quotes=quotes, assumption=assumption
            )

    @classmethod
    def _apply_canonical(
        cls,
        *,
        prior: ShadowPortfolioState,
        intents: ShadowIntentBatch,
        quotes: ShadowQuoteSnapshot,
        assumption: ExecutionAssumption,
    ) -> ShadowTransitionResult:
        identity = (intents.portfolio_id, intents.cycle_id, intents.trade_date)
        if prior.portfolio_id != intents.portfolio_id:
            raise ShadowAccountingContractError("prior portfolio does not match intent batch")
        if (quotes.portfolio_id, quotes.cycle_id, quotes.trade_date) != identity:
            raise ShadowAccountingContractError("quote snapshot identity does not match intent batch")
        if intents.trade_date <= prior.as_of_date:
            raise ShadowAccountingContractError("trade date must be later than prior as-of date")
        if prior.cash != _money(prior.cash, assumption) or any(
            item.book_cost != _money(item.book_cost, assumption) for item in prior.positions
        ):
            raise ShadowAccountingContractError(
                "prior cash and book cost must already match money_quantum"
            )
        expected_quotes = tuple(sorted(
            {item.instrument for item in prior.positions}
            | {item.instrument for item in intents.intents}
        ))
        if quotes.requested_instruments != expected_quotes:
            raise ShadowAccountingContractError("quote requested set does not match state and intents")
        for intent in intents.intents:
            if intent.side == "BUY" and intent.quantity % assumption.lot_size:
                raise ShadowAccountingContractError("BUY quantity must be a multiple of lot_size")

        positions = {
            item.instrument: (item.quantity, item.book_cost) for item in prior.positions
        }
        quote_map = {item.instrument: item for item in quotes.quotes}
        cash = prior.cash
        cash_floor = min(cash, _ZERO)
        terminals = []
        for intent in intents.intents:
            terminal, cash = cls._execute_intent(
                intent, quote_map[intent.instrument], positions, cash, assumption
            )
            if cash < cash_floor:
                raise ShadowAccountingContractError("transition would worsen the opening deficit")
            terminals.append(terminal)

        after = ShadowPortfolioState.from_dict({
            "portfolio_id": prior.portfolio_id,
            "as_of_date": intents.trade_date,
            "cash": _decimal_text(_money(cash, assumption)),
            "positions": [
                {
                    "instrument": instrument,
                    "quantity": quantity,
                    "book_cost": _decimal_text(_money(cost, assumption)),
                }
                for instrument, (quantity, cost) in sorted(positions.items())
            ],
        })
        filled = tuple(item for item in terminals if item.status == "FILLED")
        gross_buy = sum(
            (item.gross for item in filled if item.side == "BUY"), _ZERO
        )
        gross_sell = sum(
            (item.gross for item in filled if item.side == "SELL"), _ZERO
        )
        total_fee = sum((item.fee for item in filled), _ZERO)
        realized_pnl = sum((item.realized_pnl for item in filled), _ZERO)
        slippage_cost = _money(sum((
            item.quantity * (
                item.execution_price - item.reference_price
                if item.side == "BUY"
                else item.reference_price - item.execution_price
            )
            for item in filled
        ), _ZERO), assumption)

        valuation_instruments = tuple(sorted(
            {item.instrument for item in prior.positions}
            | {item.instrument for item in after.positions}
        ))
        missing = tuple(
            instrument for instrument in valuation_instruments
            if quote_map[instrument].status == "MISSING"
        )
        if missing:
            valuation_status = "PARTIAL"
            nav_before = None
            nav_after = None
            rounding_adjustment = None
            nav_reconciled = False
        else:
            valuation_status = "COMPLETE"
            nav_before = _money(prior.cash + sum((
                item.quantity * quote_map[item.instrument].price
                for item in prior.positions
            ), _ZERO), assumption)
            nav_after = _money(after.cash + sum((
                item.quantity * quote_map[item.instrument].price
                for item in after.positions
            ), _ZERO), assumption)
            rounding_adjustment = _money(
                nav_before - total_fee - slippage_cost - nav_after, assumption
            )
            if nav_after != nav_before - total_fee - slippage_cost - rounding_adjustment:
                raise ShadowAccountingContractError("exact NAV reconciliation failed")
            nav_reconciled = True

        return ShadowTransitionResult._create(
            _RESULT_TOKEN,
            schema_version=1,
            evidence_class="RETROSPECTIVE_TECHNICAL_REPLAY",
            prospective_claim=False,
            promotion_capability=False,
            corporate_action_mode=assumption.corporate_action_mode,
            warnings=("CORPORATE_ACTIONS_UNMODELED",),
            portfolio_id=prior.portfolio_id,
            cycle_id=intents.cycle_id,
            trade_date=intents.trade_date,
            assumption=assumption,
            prior_state=prior,
            requested_intents=intents,
            quotes=quotes,
            terminal_results=tuple(terminals),
            after_state=after,
            filled_count=len(filled),
            no_fill_count=len(terminals) - len(filled),
            all_filled=len(filled) == len(terminals),
            gross_buy=_money(gross_buy, assumption),
            gross_sell=_money(gross_sell, assumption),
            total_fee=_money(total_fee, assumption),
            slippage_cost=slippage_cost,
            rounding_adjustment=rounding_adjustment,
            realized_pnl=_money(realized_pnl, assumption),
            valuation_status=valuation_status,
            missing_valuation_instruments=missing,
            nav_before=nav_before,
            nav_after=nav_after,
            nav_reconciled=nav_reconciled,
            transition_status="COMPLETE",
        )


__all__ = [
    "ExecutionAssumption",
    "ShadowAccountingContractError",
    "ShadowIntentBatch",
    "ShadowPortfolioState",
    "ShadowPortfolioTransition",
    "ShadowQuoteSnapshot",
    "ShadowTransitionResult",
]
