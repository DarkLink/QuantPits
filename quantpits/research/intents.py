"""Deterministic current-rule intent planning for retrospective shadow research.

This sidecar adapts canonical research inputs to the existing production order
generator.  It has no workspace, market-data, clock, or persistence boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantpits.evidence.ranking import RankingResult
from quantpits.order.execution import PositionAnalysisResult, UniverseSnapshot
from quantpits.research.accounting import ShadowIntentBatch, ShadowPortfolioState


class IntentPlanningContractError(ValueError):
    """A caller-provided planning contract is invalid."""


class IntentPlanningParityError(RuntimeError):
    """The production primitive returned a malformed or inconsistent proposal."""


_INSTRUMENT_RE = re.compile(r"^(?:SH|SZ)[0-9]{6}$")
_RESULT_TOKEN = object()
_DEFINITION_FIELDS = (
    "schema_version", "definition_id", "strategy_name", "topk", "n_drop",
    "buy_suggestion_factor", "sell_out_of_universe", "production_buy_lot_size",
    "cashflow_mode", "planning_price_rule", "buy_selection_rule",
    "production_primitive_id",
)
_WARNINGS = (
    "CURRENT_RULE_TECHNICAL_REPLAY",
    "ZERO_EXTERNAL_CASHFLOW",
    "BUY_SELECTION_IS_RESEARCH_PROTOCOL",
    "NOT_PROSPECTIVE_EVIDENCE",
    "NOT_AUTOMATIC_PROMOTION_BASIS",
)


def _production_generator_class() -> Any:
    # The production strategy module also owns workspace config helpers.  Keep
    # that unrelated environment boundary out of strict contract imports.
    from quantpits.utils.strategy import TopkDropoutOrderGenerator

    return TopkDropoutOrderGenerator


def _new_frozen(cls: Any, **values: Any) -> Any:
    obj = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(obj, name, value)
    return obj


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntentPlanningContractError("%s must be an object" % field)
    return value


def _keys(value: Mapping[str, Any], expected: Sequence[str], field: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise IntentPlanningContractError(
            "%s fields mismatch (missing=%r, extra=%r)"
            % (field, sorted(wanted - actual), sorted(actual - wanted))
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise IntentPlanningContractError("%s must be a non-empty exact string" % field)
    return value


def _date(value: Any, field: str) -> str:
    raw = _text(value, field)
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d")
    except ValueError as exc:
        raise IntentPlanningContractError("%s must be YYYY-MM-DD" % field) from exc
    if parsed.strftime("%Y-%m-%d") != raw:
        raise IntentPlanningContractError("%s must be YYYY-MM-DD" % field)
    return raw


def _instrument(value: Any, field: str) -> str:
    raw = _text(value, field)
    if _INSTRUMENT_RE.fullmatch(raw) is None:
        raise IntentPlanningContractError("%s must be a canonical SH/SZ instrument" % field)
    return raw


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntentPlanningContractError("%s must be a positive integer" % field)
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntentPlanningContractError("%s must be a non-negative integer" % field)
    return value


def _decimal(value: Any, field: str) -> Decimal:
    raw = _text(value, field)
    try:
        result = Decimal(raw)
    except InvalidOperation as exc:
        raise IntentPlanningContractError("%s must be a decimal string" % field) from exc
    if not result.is_finite() or result <= 0:
        raise IntentPlanningContractError("%s must be a positive finite decimal" % field)
    return result


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _float_text(value: Any, field: str, *, nullable: bool = False) -> Optional[str]:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise IntentPlanningParityError("%s must be a finite number" % field)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntentPlanningParityError("%s must be a finite number" % field) from exc
    if not math.isfinite(number):
        raise IntentPlanningParityError("%s must be a finite number" % field)
    return format(number, ".17g")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _freeze_rows(rows: Iterable[Mapping[str, Any]]) -> Tuple[Mapping[str, Any], ...]:
    return tuple(MappingProxyType(dict(row)) for row in rows)


@dataclass(frozen=True, init=False)
class CurrentRuleIntentDefinition:
    schema_version: int
    definition_id: str
    strategy_name: str
    topk: int
    n_drop: int
    buy_suggestion_factor: int
    sell_out_of_universe: bool
    production_buy_lot_size: int
    cashflow_mode: str
    planning_price_rule: str
    buy_selection_rule: str
    production_primitive_id: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntentPlanningContractError("definitions require from_dict")

    @classmethod
    def from_dict(cls, value: Any) -> "CurrentRuleIntentDefinition":
        raw = _mapping(value, "definition")
        _keys(raw, _DEFINITION_FIELDS, "definition")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise IntentPlanningContractError("definition.schema_version must be integer 1")
        if type(raw["sell_out_of_universe"]) is not bool:
            raise IntentPlanningContractError("definition.sell_out_of_universe must be boolean")
        constants = {
            "strategy_name": "topk_dropout",
            "production_buy_lot_size": 100,
            "cashflow_mode": "ZERO_EXTERNAL_CASHFLOW",
            "planning_price_rule": "CASH_CLOSE_WITH_0_9_1_1_ESTIMATES_V1",
            "buy_selection_rule": "FIRST_N_GENERATED_SUGGESTIONS_V1",
            "production_primitive_id": "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1",
        }
        values = {
            "schema_version": 1,
            "definition_id": _text(raw["definition_id"], "definition.definition_id"),
            "topk": _positive_int(raw["topk"], "definition.topk"),
            "n_drop": _nonnegative_int(raw["n_drop"], "definition.n_drop"),
            "buy_suggestion_factor": _positive_int(
                raw["buy_suggestion_factor"], "definition.buy_suggestion_factor"
            ),
            "sell_out_of_universe": raw["sell_out_of_universe"],
        }
        for field, expected in constants.items():
            if raw[field] != expected or type(raw[field]) is not type(expected):
                raise IntentPlanningContractError("definition.%s must be %r" % (field, expected))
            values[field] = expected
        return _new_frozen(cls, **values)

    def _payload(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in _DEFINITION_FIELDS}

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        result = self._payload()
        result["digest"] = self.digest
        return result

    def _validated_copy(self) -> "CurrentRuleIntentDefinition":
        return self.from_dict(self._payload())


@dataclass(frozen=True, init=False)
class AnchorPriceObservation:
    instrument: str
    anchor_date: str
    status: str
    cash_close: Optional[Decimal]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntentPlanningContractError("price rows require aggregate construction")

    @classmethod
    def _from_mapping(cls, value: Any, field: str) -> "AnchorPriceObservation":
        raw = _mapping(value, field)
        _keys(raw, ("instrument", "anchor_date", "status", "cash_close"), field)
        status = _text(raw["status"], field + ".status")
        if status not in ("OBSERVED", "MISSING"):
            raise IntentPlanningContractError("%s.status must be OBSERVED or MISSING" % field)
        if status == "OBSERVED":
            close = _decimal(raw["cash_close"], field + ".cash_close")
            try:
                operational = float(close)
            except (ValueError, OverflowError) as exc:
                raise IntentPlanningContractError("%s.cash_close cannot enter production arithmetic" % field) from exc
            operational_bounds = (operational, operational * 0.9, operational * 1.1)
            if any(not math.isfinite(item) or item <= 0 for item in operational_bounds):
                raise IntentPlanningContractError("%s.cash_close cannot enter production arithmetic" % field)
        else:
            if raw["cash_close"] is not None:
                raise IntentPlanningContractError("%s MISSING row must have null cash_close" % field)
            close = None
        return _new_frozen(
            cls,
            instrument=_instrument(raw["instrument"], field + ".instrument"),
            anchor_date=_date(raw["anchor_date"], field + ".anchor_date"),
            status=status,
            cash_close=close,
        )

    def _raw(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "anchor_date": self.anchor_date,
            "status": self.status,
            "cash_close": None if self.cash_close is None else _decimal_text(self.cash_close),
        }


@dataclass(frozen=True, init=False)
class AnchorPriceSnapshot:
    anchor_date: str
    requested_instruments: Tuple[str, ...]
    rows: Tuple[AnchorPriceObservation, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntentPlanningContractError("price snapshots require from_iterable")

    @classmethod
    def from_iterable(
        cls, *, anchor_date: Any, requested_instruments: Any, rows: Iterable[Any]
    ) -> "AnchorPriceSnapshot":
        anchor = _date(anchor_date, "prices.anchor_date")
        if isinstance(requested_instruments, (str, bytes, Mapping)):
            raise IntentPlanningContractError("prices.requested_instruments must be a sequence")
        try:
            requested = tuple(
                _instrument(item, "prices.requested_instruments[%d]" % index)
                for index, item in enumerate(requested_instruments)
            )
        except TypeError as exc:
            raise IntentPlanningContractError("prices.requested_instruments must be iterable") from exc
        if not requested or requested != tuple(sorted(requested)) or len(set(requested)) != len(requested):
            raise IntentPlanningContractError("prices.requested_instruments must be non-empty, unique, sorted")
        if isinstance(rows, (str, bytes, Mapping)):
            raise IntentPlanningContractError("prices.rows must be an iterable of objects")
        try:
            observations = tuple(
                AnchorPriceObservation._from_mapping(row, "prices.rows[%d]" % index)
                for index, row in enumerate(rows)
            )
        except TypeError as exc:
            raise IntentPlanningContractError("prices.rows must be iterable") from exc
        identities = tuple(item.instrument for item in observations)
        if identities != requested:
            raise IntentPlanningContractError("prices.rows must exactly cover requested instruments in order")
        if any(item.anchor_date != anchor for item in observations):
            raise IntentPlanningContractError("price row anchor date does not match snapshot")
        return _new_frozen(cls, anchor_date=anchor, requested_instruments=requested, rows=observations)

    def _payload(self) -> Dict[str, Any]:
        result = {
            "anchor_date": self.anchor_date,
            "requested_instruments": list(self.requested_instruments),
            "rows": [item._raw() for item in self.rows],
        }
        return result

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    @property
    def observed_instruments(self) -> Tuple[str, ...]:
        return tuple(item.instrument for item in self.rows if item.status == "OBSERVED")

    @property
    def missing_instruments(self) -> Tuple[str, ...]:
        return tuple(item.instrument for item in self.rows if item.status == "MISSING")

    def to_dict(self) -> Dict[str, Any]:
        result = self._payload()
        result["observed_instruments"] = list(self.observed_instruments)
        result["missing_instruments"] = list(self.missing_instruments)
        result["digest"] = self.digest
        return result

    def _validated_copy(self) -> "AnchorPriceSnapshot":
        return self.from_iterable(
            anchor_date=self.anchor_date,
            requested_instruments=self.requested_instruments,
            rows=[item._raw() for item in self.rows],
        )


def _revalidate_ranking(value: Any) -> RankingResult:
    if not isinstance(value, RankingResult):
        raise IntentPlanningContractError("ranking must be a canonical RankingResult")
    try:
        return RankingResult(
            tuple(dict(row) for row in value.rows),
            value.eligible_count,
            value.scored_count,
            value.missing_count,
            value.complete,
        )
    except Exception as exc:
        raise IntentPlanningContractError("ranking failed canonical revalidation") from exc


def _ranking_payload(ranking: RankingResult) -> Dict[str, Any]:
    return {
        "eligible_count": ranking.eligible_count,
        "scored_count": ranking.scored_count,
        "missing_count": ranking.missing_count,
        "complete": ranking.complete,
        "rows": [dict(row) for row in ranking.rows],
    }


def _serialize_order(value: Any, field: str, expected_date: str) -> Dict[str, Any]:
    raw = _mapping(value, field)
    _keys(raw, ("instrument", "datetime", "value", "estimated_amount", "score", "current_close"), field)
    instrument = _instrument(raw["instrument"], field + ".instrument")
    if raw["datetime"] != expected_date:
        raise IntentPlanningParityError("%s datetime does not match trade date" % field)
    quantity = _positive_int(raw["value"], field + ".value")
    return {
        "instrument": instrument,
        "datetime": expected_date,
        "quantity": quantity,
        "estimated_amount": _float_text(raw["estimated_amount"], field + ".estimated_amount"),
        "score": _float_text(raw["score"], field + ".score", nullable=True),
        "current_close": _float_text(raw["current_close"], field + ".current_close"),
    }


def _frame_ids(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, pd.DataFrame):
        raise IntentPlanningParityError("%s must be a DataFrame" % field)
    identities = tuple(value.index.tolist())
    if any(not isinstance(item, str) for item in identities) or len(set(identities)) != len(identities):
        raise IntentPlanningParityError("%s identities must be unique strings" % field)
    return identities


def _intent_id(
    *, portfolio_id: str, cycle_id: str, trade_date: str, definition_digest: str,
    ranking_digest: str, prior_digest: str, prices_digest: str, side: str,
    ordinal: int, instrument: str, quantity: int,
) -> str:
    payload = {
        "domain": "SHADOW_INTENT_V1",
        "portfolio_id": portfolio_id,
        "cycle_id": cycle_id,
        "trade_date": trade_date,
        "definition_digest": definition_digest,
        "ranking_digest": ranking_digest,
        "prior_state_digest": prior_digest,
        "price_snapshot_digest": prices_digest,
        "side": side,
        "requested_ordinal": ordinal,
        "instrument": instrument,
        "quantity": quantity,
    }
    return _digest(payload)


def _expected_sell_proposal(
    candidates: pd.DataFrame, holdings: Sequence[Mapping[str, Any]], trade_date: str,
) -> Tuple[Dict[str, Any], ...]:
    quantities = {item["instrument"]: float(item["value"]) for item in holdings}
    expected = []
    for instrument, row in candidates.iterrows():
        if instrument not in quantities:
            continue
        quantity = quantities[instrument]
        amount = quantity * row["possible_min"]
        expected.append(_serialize_order({
            "instrument": instrument,
            "datetime": trade_date,
            "value": int(quantity),
            "estimated_amount": round(amount, 2),
            "score": round(row["score"], 6) if pd.notna(row.get("score")) else None,
            "current_close": round(row["current_close"], 2),
        }, "expected_sell", trade_date))
    return tuple(expected)


def _expected_buy_proposal(
    candidates: pd.DataFrame, target_count: int, available_cash: float, trade_date: str,
) -> Tuple[Dict[str, Any], ...]:
    average_cash = available_cash / target_count if target_count > 0 else 0
    expected = []
    for instrument, row in candidates.iterrows():
        quantity = int(np.floor(average_cash / row["possible_max"] / 100) * 100)
        if quantity < 100:
            continue
        amount = quantity * row["possible_max"]
        expected.append(_serialize_order({
            "instrument": instrument,
            "datetime": trade_date,
            "value": quantity,
            "estimated_amount": round(amount, 2),
            "score": round(row["score"], 6),
            "current_close": round(row["current_close"], 2),
        }, "expected_buy", trade_date))
    return tuple(expected)


@dataclass(frozen=True, init=False)
class IntentPlanningResult:
    request: Mapping[str, str]
    definition: CurrentRuleIntentDefinition
    definition_digest: str
    ranking_digest: str
    ranking_counts: Mapping[str, int]
    universe: Mapping[str, Any]
    prior_state_digest: str
    prices: AnchorPriceSnapshot
    price_partitions: Mapping[str, Tuple[str, ...]]
    ranking_partitions: Mapping[str, Tuple[str, ...]]
    holding_classifications: Mapping[str, Tuple[str, ...]]
    production_sell_proposal: Tuple[Mapping[str, Any], ...]
    production_buy_suggestions: Tuple[Mapping[str, Any], ...]
    raw_buy_candidate_instruments: Tuple[str, ...]
    target_buy_count: int
    selected_buy_instruments: Tuple[str, ...]
    excluded_buy_instruments: Tuple[str, ...]
    buy_intent_shortage: int
    intents: ShadowIntentBatch
    parity_checked: bool
    selection_checked: bool
    status: str
    evidence_class: str
    prospective_claim: bool
    promotion_capability: bool
    warnings: Tuple[str, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        token = kwargs.pop("_token", None)
        if token is not _RESULT_TOKEN or args:
            raise IntentPlanningContractError("planning results are planner-owned")
        expected = set(self.__dataclass_fields__) - {"result_digest"}
        if set(kwargs) != expected:
            raise IntentPlanningContractError("planning result fields are internal")
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

    def _payload(self) -> Dict[str, Any]:
        result = {
            "schema_version": 1,
            "evidence_class": self.evidence_class,
            "prospective_claim": self.prospective_claim,
            "promotion_capability": self.promotion_capability,
            "warnings": list(self.warnings),
            "request": dict(self.request),
            "definition": self.definition.to_dict(),
            "definition_digest": self.definition_digest,
            "production_primitive_id": self.definition.production_primitive_id,
            "ranking_digest": self.ranking_digest,
            "ranking_counts": dict(self.ranking_counts),
            "universe": dict(self.universe),
            "prior_state_digest": self.prior_state_digest,
            "prices": self.prices.to_dict(),
            "price_partitions": {key: list(value) for key, value in self.price_partitions.items()},
            "ranking_partitions": {key: list(value) for key, value in self.ranking_partitions.items()},
            "holding_classifications": {
                key: list(value) for key, value in self.holding_classifications.items()
            },
            "production_sell_proposal": [dict(item) for item in self.production_sell_proposal],
            "production_buy_suggestions": [dict(item) for item in self.production_buy_suggestions],
            "raw_buy_candidate_instruments": list(self.raw_buy_candidate_instruments),
            "target_buy_count": self.target_buy_count,
            "selected_buy_instruments": list(self.selected_buy_instruments),
            "excluded_buy_instruments": list(self.excluded_buy_instruments),
            "buy_intent_shortage": self.buy_intent_shortage,
            "intents": self.intents.to_dict(),
            "parity_checked": self.parity_checked,
            "selection_checked": self.selection_checked,
            "status": self.status,
        }
        result["universe"]["instruments"] = list(self.universe["instruments"])
        return result

    @property
    def result_digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        result = self._payload()
        result["result_digest"] = self.result_digest
        return result


class CurrentRuleShadowIntentPlanner:
    """Truth owner for the current-rule proposal and frozen first-N selection."""

    @classmethod
    def plan(
        cls, *, portfolio_id: Any, cycle_id: Any, market: Any, anchor_date: Any,
        trade_date: Any, definition: Any, ranking: Any, prior: Any, prices: Any,
    ) -> IntentPlanningResult:
        request = {
            "portfolio_id": _text(portfolio_id, "portfolio_id"),
            "cycle_id": _text(cycle_id, "cycle_id"),
            "market": _text(market, "market"),
            "anchor_date": _date(anchor_date, "anchor_date"),
            "trade_date": _date(trade_date, "trade_date"),
        }
        if not request["anchor_date"] < request["trade_date"]:
            raise IntentPlanningContractError("anchor_date must precede trade_date")
        if not isinstance(definition, CurrentRuleIntentDefinition):
            raise IntentPlanningContractError("definition must be CurrentRuleIntentDefinition")
        definition = definition._validated_copy()
        ranking = _revalidate_ranking(ranking)
        if ranking.scored_count == 0:
            raise IntentPlanningContractError("ranking has no scored planning capability")
        for index, row in enumerate(ranking.rows):
            _instrument(row["instrument"], "ranking.rows[%d].instrument" % index)
        if not isinstance(prior, ShadowPortfolioState):
            raise IntentPlanningContractError("prior must be ShadowPortfolioState")
        prior = prior._validated_copy()
        if prior.portfolio_id != request["portfolio_id"]:
            raise IntentPlanningContractError("prior portfolio does not match request")
        if not prior.as_of_date <= request["anchor_date"]:
            raise IntentPlanningContractError("prior state is later than anchor date")
        try:
            planning_cash = float(prior.cash)
        except (ValueError, OverflowError) as exc:
            raise IntentPlanningContractError("prior cash cannot enter production arithmetic") from exc
        if not math.isfinite(planning_cash):
            raise IntentPlanningContractError("prior cash cannot enter production arithmetic")
        if not isinstance(prices, AnchorPriceSnapshot):
            raise IntentPlanningContractError("prices must be AnchorPriceSnapshot")
        prices = prices._validated_copy()
        if prices.anchor_date != request["anchor_date"]:
            raise IntentPlanningContractError("price snapshot anchor does not match request")

        scored_rows = tuple(row for row in ranking.rows if row["scored"])
        scored_ids = tuple(row["instrument"] for row in scored_rows)
        unscored_ids = tuple(row["instrument"] for row in ranking.rows if not row["scored"])
        holding_ids = tuple(item.instrument for item in prior.positions)
        requested_prices = tuple(sorted(set(scored_ids) | set(holding_ids)))
        if prices.requested_instruments != requested_prices:
            raise IntentPlanningContractError("price requested set must equal scored ranking union holdings")

        ranking_digest = hashlib.sha256(ranking.to_csv_bytes()).hexdigest()
        universe = UniverseSnapshot.observe(
            request["market"], request["anchor_date"],
            tuple(row["instrument"] for row in ranking.rows),
        )
        prediction_index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(request["anchor_date"]), row["instrument"]) for row in scored_rows],
            names=("datetime", "instrument"),
        )
        predictions = pd.DataFrame(
            {"score": [float(row["raw_score"]) for row in scored_rows]}, index=prediction_index,
        )
        price_records = []
        for row in prices.rows:
            if row.cash_close is None:
                current = minimum = maximum = np.nan
            else:
                current = float(row.cash_close)
                minimum = current * 0.9
                maximum = current * 1.1
            price_records.append({
                "instrument": row.instrument,
                "current_close": current,
                "possible_min": minimum,
                "possible_max": maximum,
            })
        price_frame = pd.DataFrame(price_records).set_index("instrument")
        holdings = [
            {"instrument": item.instrument, "value": item.quantity}
            for item in prior.positions
        ]
        generator = _production_generator_class()(
            definition.topk, definition.n_drop, definition.buy_suggestion_factor,
            definition.sell_out_of_universe,
        )
        analysis = generator.analyze_positions_with_universe(
            predictions, price_frame, holdings,
            universe if definition.sell_out_of_universe else None,
        )
        cls._validate_analysis(
            analysis, holding_ids, universe, definition, set(scored_ids),
            set(prices.observed_instruments),
        )

        forced_ids = _frame_ids(analysis.forced_exit_candidates, "forced exits")
        normal_ids = _frame_ids(analysis.normal_sell_candidates, "normal sells")
        buy_candidate_ids = _frame_ids(analysis.buy_candidates, "buy candidates")
        sell_candidates = pd.concat(
            [analysis.forced_exit_candidates, analysis.normal_sell_candidates]
        )
        raw_sells, sell_amount = generator.generate_sell_orders(
            sell_candidates, holdings, request["trade_date"]
        )
        if isinstance(raw_sells, (str, bytes, Mapping)):
            raise IntentPlanningParityError("sell proposal must be a sequence")
        sells = tuple(
            _serialize_order(item, "sell_orders[%d]" % index, request["trade_date"])
            for index, item in enumerate(raw_sells)
        )
        sell_ids = tuple(item["instrument"] for item in sells)
        expected_sells = _expected_sell_proposal(sell_candidates, holdings, request["trade_date"])
        if (
            sell_ids != forced_ids + normal_ids
            or len(set(sell_ids)) != len(sell_ids)
            or sells != expected_sells
        ):
            raise IntentPlanningParityError("generated sells do not exactly match executable classifications")
        try:
            estimated_sell_amount = float(sell_amount)
        except (TypeError, ValueError, OverflowError) as exc:
            raise IntentPlanningParityError("estimated sell amount must be finite") from exc
        if not math.isfinite(estimated_sell_amount) or estimated_sell_amount < 0:
            raise IntentPlanningParityError("estimated sell amount must be finite and non-negative")
        # Replay the production primitive's scalar domain and left-to-right
        # ``+=`` exactly.  Python 3.12's built-in float sum uses compensated
        # summation and can differ by one ULP from the production loop.
        expected_sell_amount = 0
        for instrument in forced_ids + normal_ids:
            quantity = float(holdings[holding_ids.index(instrument)]["value"])
            expected_sell_amount += quantity * sell_candidates.loc[
                instrument, "possible_min"
            ]
        if estimated_sell_amount != expected_sell_amount:
            raise IntentPlanningParityError("estimated sell amount disagrees with proposal")

        available_cash = planning_cash + estimated_sell_amount
        if not math.isfinite(available_cash):
            raise IntentPlanningParityError("planning cash after sells must be finite")
        raw_buys = generator.generate_buy_orders(
            analysis.buy_candidates, analysis.target_buy_count,
            available_cash, request["trade_date"],
        )
        if isinstance(raw_buys, (str, bytes, Mapping)):
            raise IntentPlanningParityError("buy suggestions must be a sequence")
        buys = tuple(
            _serialize_order(item, "buy_orders[%d]" % index, request["trade_date"])
            for index, item in enumerate(raw_buys)
        )
        buy_ids = tuple(item["instrument"] for item in buys)
        expected_buys = _expected_buy_proposal(
            analysis.buy_candidates, analysis.target_buy_count,
            available_cash, request["trade_date"],
        )
        if (
            len(set(buy_ids)) != len(buy_ids)
            or any(item not in buy_candidate_ids for item in buy_ids)
            or tuple(item for item in buy_candidate_ids if item in set(buy_ids)) != buy_ids
            or set(buy_ids) & set(holding_ids)
            or buys != expected_buys
        ):
            raise IntentPlanningParityError("generated buys violate candidate identity/order")

        selected = buys[:analysis.target_buy_count]
        excluded = buys[analysis.target_buy_count:]
        shortage = max(analysis.target_buy_count - len(selected), 0)
        intent_rows = []
        definition_digest = definition.digest
        prior_digest = prior.digest
        for side, proposals in (("SELL", sells), ("BUY", selected)):
            for ordinal, proposal in enumerate(proposals):
                intent_rows.append({
                    "intent_id": _intent_id(
                        portfolio_id=request["portfolio_id"], cycle_id=request["cycle_id"],
                        trade_date=request["trade_date"], definition_digest=definition_digest,
                        ranking_digest=ranking_digest, prior_digest=prior_digest,
                        prices_digest=prices.digest, side=side, ordinal=ordinal,
                        instrument=proposal["instrument"], quantity=proposal["quantity"],
                    ),
                    "portfolio_id": request["portfolio_id"],
                    "cycle_id": request["cycle_id"],
                    "trade_date": request["trade_date"],
                    "side": side,
                    "instrument": proposal["instrument"],
                    "quantity": proposal["quantity"],
                })
        intents = ShadowIntentBatch.from_iterable(
            portfolio_id=request["portfolio_id"], cycle_id=request["cycle_id"],
            trade_date=request["trade_date"], rows=intent_rows,
        )

        observed = set(prices.observed_instruments)
        scored_usable = tuple(item for item in scored_ids if item in observed)
        scored_missing = tuple(item for item in scored_ids if item not in observed)
        membership_status = "ENABLED" if definition.sell_out_of_universe else "DISABLED_BY_DEFINITION"
        universe_payload = {
            "market": universe.market,
            "anchor_date": universe.anchor_date,
            "instruments": universe.instruments,
            "instrument_count": universe.instrument_count,
            "fingerprint_algorithm": universe.fingerprint_algorithm,
            "fingerprint": universe.fingerprint,
            "membership_check_status": membership_status,
        }
        result = IntentPlanningResult(
            _token=_RESULT_TOKEN,
            request=MappingProxyType(dict(request)),
            definition=definition,
            definition_digest=definition_digest,
            ranking_digest=ranking_digest,
            ranking_counts=MappingProxyType({
                "eligible": ranking.eligible_count,
                "scored": ranking.scored_count,
                "unscored": ranking.missing_count,
            }),
            universe=MappingProxyType(universe_payload),
            prior_state_digest=prior_digest,
            prices=prices,
            price_partitions=MappingProxyType({
                "requested": prices.requested_instruments,
                "observed": prices.observed_instruments,
                "missing": prices.missing_instruments,
            }),
            ranking_partitions=MappingProxyType({
                "scored_with_usable_price": scored_usable,
                "scored_with_missing_price": scored_missing,
                "unscored_eligible": unscored_ids,
            }),
            holding_classifications=MappingProxyType({
                "continuing": _frame_ids(analysis.ranked_continuing_holdings, "continuing"),
                "normal_sell": normal_ids,
                "forced_executable": forced_ids,
                "forced_pending": tuple(analysis.pending_forced_exit_instruments),
                "eligible_unscored_retained": tuple(analysis.eligible_unscored_instruments),
            }),
            production_sell_proposal=_freeze_rows(sells),
            production_buy_suggestions=_freeze_rows(buys),
            raw_buy_candidate_instruments=buy_candidate_ids,
            target_buy_count=analysis.target_buy_count,
            selected_buy_instruments=tuple(item["instrument"] for item in selected),
            excluded_buy_instruments=tuple(item["instrument"] for item in excluded),
            buy_intent_shortage=shortage,
            intents=intents,
            parity_checked=True,
            selection_checked=True,
            status="COMPLETE",
            evidence_class="RETROSPECTIVE_TECHNICAL_REPLAY",
            prospective_claim=False,
            promotion_capability=False,
            warnings=_WARNINGS,
        )
        # Force complete canonical rendering before granting the result.
        result.to_dict()
        return result

    @staticmethod
    def _validate_analysis(
        analysis: Any, holding_ids: Tuple[str, ...], universe: UniverseSnapshot,
        definition: CurrentRuleIntentDefinition, scored_ids: set,
        observed_price_ids: set,
    ) -> None:
        if not isinstance(analysis, PositionAnalysisResult):
            raise IntentPlanningParityError("production analysis has invalid type")
        count_fields = (
            "account_holding_count_before", "forced_exit_count", "normal_sell_count",
            "executable_sell_count", "remaining_after_sell", "target_buy_count",
            "planned_final_holding_count",
        )
        if any(
            isinstance(getattr(analysis, field), bool)
            or not isinstance(getattr(analysis, field), int)
            or getattr(analysis, field) < 0
            for field in count_fields
        ):
            raise IntentPlanningParityError("production analysis counts must be exact non-negative integers")
        for field in ("pending_forced_exit_instruments", "eligible_unscored_instruments"):
            value = getattr(analysis, field)
            if type(value) is not tuple or value != tuple(sorted(value)) or len(set(value)) != len(value):
                raise IntentPlanningParityError("%s must be an exact sorted unique tuple" % field)
        continuing_ids = _frame_ids(analysis.ranked_continuing_holdings, "continuing")
        normal_ids = _frame_ids(analysis.normal_sell_candidates, "normal sells")
        forced_ids = _frame_ids(analysis.forced_exit_candidates, "forced exits")
        buy_ids = _frame_ids(analysis.buy_candidates, "buy candidates")
        sorted_ids = _frame_ids(analysis.sorted_ranking, "sorted ranking")
        continuing = set(continuing_ids)
        normal = set(normal_ids)
        forced = set(forced_ids)
        pending = set(analysis.pending_forced_exit_instruments)
        unscored = set(analysis.eligible_unscored_instruments)
        if any(not isinstance(item, str) for item in pending | unscored):
            raise IntentPlanningParityError("classification identities must be strings")
        groups = (continuing, normal, forced, pending, unscored)
        holding_set = set(holding_ids)
        if set().union(*groups) != holding_set or sum(len(group) for group in groups) != len(holding_set):
            raise IntentPlanningParityError("every holding must have exactly one terminal classification")
        membership_enabled = definition.sell_out_of_universe
        expected_forced = holding_set - set(universe.instruments) if membership_enabled else set()
        if forced | pending != expected_forced:
            raise IntentPlanningParityError("forced exits do not match exact ranking universe")
        expected_ranked = scored_ids & observed_price_ids
        if set(sorted_ids) != expected_ranked:
            raise IntentPlanningParityError("production sorted ranking lost or added usable scored members")
        ranked_holding = holding_set & expected_ranked
        if continuing | normal != ranked_holding:
            raise IntentPlanningParityError("ranked holding partition is inconsistent")
        expected_unscored = holding_set - expected_forced - ranked_holding
        if unscored != expected_unscored:
            raise IntentPlanningParityError("unscored holding partition is inconsistent")
        if set(buy_ids) & holding_set:
            raise IntentPlanningParityError("buy candidates overlap prior holdings")
        executable = len(forced) + len(normal)
        remaining = len(holding_set) - executable
        expected_target = max(definition.topk - remaining, 0)
        buffer_ids = sorted_ids[:definition.topk + definition.n_drop * definition.buy_suggestion_factor]
        ranked_holding_order = tuple(item for item in sorted_ids if item in ranked_holding)
        normal_slots = max(definition.n_drop - len(expected_forced), 0)
        expected_normal_pool = tuple(item for item in ranked_holding_order if item not in set(buffer_ids))
        expected_normal = expected_normal_pool[-normal_slots:] if normal_slots else ()
        expected_continuing = tuple(item for item in ranked_holding_order if item not in set(expected_normal))
        expected_buy = tuple(item for item in buffer_ids if item not in holding_set)[
            :expected_target * definition.buy_suggestion_factor
        ]
        facts = (
            analysis.account_holding_count_before == len(holding_set),
            analysis.forced_exit_count == len(forced) + len(pending),
            analysis.normal_sell_count == len(normal),
            analysis.executable_sell_count == executable,
            analysis.remaining_after_sell == remaining,
            analysis.target_buy_count == expected_target,
            analysis.target_buy_count == expected_target,
            analysis.planned_final_holding_count == remaining + expected_target,
            normal_ids == expected_normal,
            continuing_ids == expected_continuing,
            buy_ids == expected_buy,
        )
        if not all(facts):
            raise IntentPlanningParityError("production analysis counts are inconsistent")
        if membership_enabled:
            if (
                isinstance(analysis.eligible_holding_count, bool)
                or not isinstance(analysis.eligible_holding_count, int)
                or isinstance(analysis.out_of_universe_holding_count, bool)
                or not isinstance(analysis.out_of_universe_holding_count, int)
                or analysis.eligible_holding_count != len(holding_set & set(universe.instruments))
                or analysis.out_of_universe_holding_count != len(expected_forced)
            ):
                raise IntentPlanningParityError("membership counts are inconsistent")
        elif analysis.eligible_holding_count is not None or analysis.out_of_universe_holding_count is not None:
            raise IntentPlanningParityError("disabled membership cannot grant observed counts")


__all__ = [
    "AnchorPriceSnapshot",
    "CurrentRuleIntentDefinition",
    "CurrentRuleShadowIntentPlanner",
    "IntentPlanningContractError",
    "IntentPlanningParityError",
    "IntentPlanningResult",
]
