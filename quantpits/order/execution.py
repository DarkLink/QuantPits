"""Execution contracts and prediction loading for order generation."""

from __future__ import annotations

from dataclasses import dataclass
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
