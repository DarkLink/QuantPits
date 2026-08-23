"""Execution contracts and prediction loading for order generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from quantpits.order.command import ResolvedOrderSource


class OrderExecutionError(RuntimeError):
    """Expected order execution failure suitable for CLI handling."""


class OrderSourceUnavailableError(OrderExecutionError):
    """The prepared prediction source cannot be loaded."""


class InvalidPredictionDataError(OrderExecutionError):
    """The selected recorder did not contain usable prediction data."""


class TradingCalendarError(OrderExecutionError):
    """A next trading date could not be resolved."""


class UniverseObservationError(OrderExecutionError):
    """The exact anchor-date universe could not be observed safely."""


@dataclass(frozen=True, init=False)
class UniverseSnapshot:
    market: str
    anchor_date: str
    instruments: tuple[str, ...]
    instrument_count: int
    fingerprint_algorithm: str
    fingerprint: str

    def __init__(self, *args, **kwargs):
        raise TypeError("use UniverseSnapshot.observe()")

    @classmethod
    def observe(cls, market: str, anchor_date: str, members: Any) -> "UniverseSnapshot":
        if not isinstance(market, str) or not market.strip():
            raise UniverseObservationError("universe market must be a non-empty string")
        if isinstance(anchor_date, (date, datetime)):
            anchor = anchor_date.strftime("%Y-%m-%d")
        elif isinstance(anchor_date, str) and len(anchor_date) == 10:
            try:
                datetime.strptime(anchor_date, "%Y-%m-%d")
            except ValueError as exc:
                raise UniverseObservationError("universe anchor_date must be YYYY-MM-DD") from exc
            anchor = anchor_date
        else:
            raise UniverseObservationError("universe anchor_date must be YYYY-MM-DD")
        if isinstance(members, (str, bytes)):
            raise UniverseObservationError("universe membership must be a collection")
        try:
            raw = list(members)
        except (TypeError, ValueError) as exc:
            raise UniverseObservationError("universe membership is not iterable") from exc
        canonical: list[str] = []
        for member in raw:
            if not isinstance(member, str) or not member.strip() or member != member.strip():
                raise UniverseObservationError("universe contains a malformed instrument")
            canonical.append(member)
        if not canonical:
            raise UniverseObservationError("required universe membership is empty")
        if len(canonical) != len(set(canonical)):
            raise UniverseObservationError("universe contains duplicate instruments")
        instruments = tuple(sorted(canonical))
        digest = hashlib.sha256("\n".join(instruments).encode("utf-8")).hexdigest()
        instance = object.__new__(cls)
        object.__setattr__(instance, "market", market)
        object.__setattr__(instance, "anchor_date", anchor)
        object.__setattr__(instance, "instruments", instruments)
        object.__setattr__(instance, "instrument_count", len(instruments))
        object.__setattr__(instance, "fingerprint_algorithm", "sha256")
        object.__setattr__(instance, "fingerprint", digest)
        return instance


def resolve_exact_universe(market: str, anchor_date: str) -> UniverseSnapshot:
    try:
        from qlib.data import D

        members = D.list_instruments(
            D.instruments(market=market), start_time=anchor_date,
            end_time=anchor_date, freq="day", as_list=True,
        )
        return UniverseSnapshot.observe(market, anchor_date, members)
    except UniverseObservationError:
        raise
    except Exception as exc:
        raise UniverseObservationError(
            f"could not observe exact universe {market} at {anchor_date}: {exc}"
        ) from exc


@dataclass(frozen=True)
class LoadedOrderPrediction:
    data: Any
    source: ResolvedOrderSource
    description: str


@dataclass(frozen=True)
class OrderExecutionHooks:
    init_qlib: Callable[[], None]
    get_anchor_date: Callable[[], str]
    get_next_trade_date: Callable[[str], str]
    load_predictions: Callable[[ResolvedOrderSource], LoadedOrderPrediction]
    get_price_data: Callable[..., Any]
    create_order_generator: Callable[[dict], Any]
    get_strategy_params: Callable[[dict], dict]
    build_model_opinions: Callable[[Any], Any]
    persist_artifacts: Callable[[Any], Any]
    resolve_universe: Callable[[str, str], UniverseSnapshot] | None = None


@dataclass(frozen=True)
class PositionAnalysisResult:
    ranked_continuing_holdings: Any
    normal_sell_candidates: Any
    forced_exit_candidates: Any
    pending_forced_exit_instruments: tuple[str, ...]
    eligible_unscored_instruments: tuple[str, ...]
    buy_candidates: Any
    sorted_ranking: Any
    account_holding_count_before: int
    eligible_holding_count: int | None
    out_of_universe_holding_count: int | None
    forced_exit_count: int
    normal_sell_count: int
    executable_sell_count: int
    remaining_after_sell: int
    target_buy_count: int
    planned_final_holding_count: int

    def __post_init__(self) -> None:
        forced_candidates = tuple(self.forced_exit_candidates.index)
        normal_candidates = tuple(self.normal_sell_candidates.index)
        if len(set(forced_candidates)) != len(forced_candidates) or len(set(normal_candidates)) != len(normal_candidates):
            raise ValueError("sell candidate instruments must be unique")
        if self.forced_exit_count != len(forced_candidates) + len(self.pending_forced_exit_instruments):
            raise ValueError("forced exit count is inconsistent")
        if self.normal_sell_count != len(normal_candidates):
            raise ValueError("normal sell count is inconsistent")
        if self.executable_sell_count != self.forced_exit_count - len(self.pending_forced_exit_instruments) + self.normal_sell_count:
            raise ValueError("executable sell count is inconsistent")
        if self.remaining_after_sell != self.account_holding_count_before - self.executable_sell_count:
            raise ValueError("remaining holding count is inconsistent")
        if self.planned_final_holding_count != self.remaining_after_sell + self.target_buy_count:
            raise ValueError("planned final holding count is inconsistent")
        forced = set(forced_candidates) | set(self.pending_forced_exit_instruments)
        normal = set(normal_candidates)
        if forced & normal:
            raise ValueError("forced and normal sell classifications overlap")
        if len(set(self.pending_forced_exit_instruments)) != len(self.pending_forced_exit_instruments):
            raise ValueError("pending forced exits must be unique")
        if self.eligible_holding_count is None:
            if self.out_of_universe_holding_count is not None:
                raise ValueError("membership-dependent counts must be observed together")
        elif self.eligible_holding_count + (self.out_of_universe_holding_count or 0) != self.account_holding_count_before:
            raise ValueError("observed membership counts do not cover account holdings")


@dataclass(frozen=True)
class OrderCalculationResult:
    anchor_date: str
    trade_date: str
    source_label: str
    source_description: str
    holding_count: int
    target_buy_count: int
    sell_orders: tuple[dict, ...]
    buy_orders: tuple[dict, ...]
    estimated_sell_amount: float
    estimated_buy_min: float | None
    estimated_buy_max: float | None
    opinions: Any | None
    sell_out_of_universe: bool
    universe: UniverseSnapshot | None
    account_holding_count_before: int
    eligible_holding_count: int | None
    out_of_universe_holding_count: int | None
    eligible_unscored_holding_count: int | None
    forced_exit_count: int
    forced_exit_pending_count: int
    normal_sell_count: int
    executable_sell_count: int
    remaining_after_sell: int
    planned_final_holding_count: int
    sell_decisions: tuple[dict, ...]


def source_description(source: ResolvedOrderSource) -> str:
    if source.mode == "model":
        return f"单模型: {source.resolved_name} (Record: {source.record_id})"
    return f"Ensemble 融合: {source.resolved_name} (Record: {source.record_id})"


def normalize_prediction_data(value: Any) -> Any:
    import pandas as pd

    if isinstance(value, pd.Series):
        value = value.to_frame("score")
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise InvalidPredictionDataError("prediction recorder contains no usable data")
    if "score" not in value.columns:
        numeric = value.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise InvalidPredictionDataError("prediction data does not contain a score column")
        value = value.rename(columns={numeric[0]: "score"})
    if "datetime" not in value.index.names:
        raise InvalidPredictionDataError("prediction index does not contain a datetime level")
    if "instrument" not in value.index.names:
        raise InvalidPredictionDataError("prediction index does not contain an instrument level")
    return value


def load_resolved_prediction(source: ResolvedOrderSource) -> LoadedOrderPrediction:
    if not source.record_id or not source.experiment_name:
        raise OrderSourceUnavailableError("prepared order source has no recorder id")
    try:
        from qlib.workflow import R

        recorder = R.get_recorder(
            recorder_id=source.record_id,
            experiment_name=source.experiment_name,
        )
        data = normalize_prediction_data(recorder.load_object("pred.pkl"))
    except OrderExecutionError:
        raise
    except Exception as exc:
        raise OrderSourceUnavailableError(
            f"could not load prediction recorder {source.record_id}: {exc}"
        ) from exc
    return LoadedOrderPrediction(data=data, source=source, description=source_description(source))


def resolve_next_trade_date(anchor_date: str) -> str:
    try:
        from qlib.data import D

        dates = D.calendar(start_time=anchor_date, future=True)[:2]
    except Exception as exc:
        raise TradingCalendarError(f"could not read the trading calendar: {exc}") from exc
    if not len(dates):
        raise TradingCalendarError(f"no trading date is available on or after {anchor_date}")
    selected = dates[1] if len(dates) >= 2 else dates[0]
    return selected.strftime("%Y-%m-%d")
