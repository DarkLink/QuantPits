"""Pure model-opinion discovery and decision calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from quantpits.order.command import OrderRunConfig


@dataclass(frozen=True)
class OpinionSourceSpec:
    label: str
    record_id: str
    experiment_name: str
    source_type: str
    detail: str


@dataclass(frozen=True)
class ModelOpinionsRequest:
    focus_instruments: tuple[str, ...]
    current_holding_instruments: tuple[str, ...]
    top_k: int
    drop_n: int
    buy_suggestion_factor: int
    sorted_predictions: Any
    trade_date: str
    run_config: OrderRunConfig
    load_prediction: Callable[[str, str], Any]


@dataclass(frozen=True)
class ModelOpinionsResult:
    dataframe: Any
    combo_composition: dict
    model_to_combos: dict
    source_summaries: tuple[dict, ...]
    thresholds: dict
    warnings: tuple[str, ...] = ()


def _record_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("record_id") or value.get("id")
        return str(candidate) if candidate else None
    return None


def _experiment_for_model(records: dict, model: str) -> str:
    experiments = records.get("experiments", {})
    explicit = experiments.get(model)
    if explicit:
        return explicit
    try:
        from quantpits.utils.train_utils import get_experiment_name_for_model

        resolved = get_experiment_name_for_model(records, model)
        if resolved:
            return resolved
    except Exception:
        pass
    return records.get("experiment_name") or "prod_train"


def discover_opinion_sources(config: OrderRunConfig) -> tuple[tuple[OpinionSourceSpec, ...], dict]:
    raw = config.ensemble_config
    if isinstance(raw.get("combos"), dict):
        combos = raw["combos"]
    elif isinstance(raw.get("models"), list):
        combos = {"legacy": {"models": raw["models"], "default": True}}
    else:
        combos = {}

    recorded = config.ensemble_records.get("combos", {})
    specs: list[OpinionSourceSpec] = []
    combo_info: dict[str, list[str]] = {}
    for name, combo in combos.items():
        models = list(combo.get("models", []))
        combo_info[name] = models
        record_id = (
            _record_id(recorded.get(name))
            or _record_id(recorded.get(f"ensemble_{name}"))
            or _record_id(recorded.get(name.replace("combo_", "")))
        )
        if not record_id and combo.get("default"):
            record_id = _record_id(config.ensemble_records.get("default_record_id"))
        if record_id:
            specs.append(OpinionSourceSpec(f"combo_{name}", record_id, "Ensemble_Fusion", "combo", name))

    models = sorted({model for values in combo_info.values() for model in values})
    train_models = config.train_records.get("models", {})
    for model in models:
        record_id = _record_id(train_models.get(model))
        if record_id:
            specs.append(
                OpinionSourceSpec(
                    f"model_{model}", record_id, _experiment_for_model(config.train_records, model), "model", model
                )
            )
    return tuple(specs), combo_info


def _latest_predictions(value: Any, valid_instruments: set[str]) -> Any | None:
    import pandas as pd

    if isinstance(value, pd.Series):
        value = value.to_frame("score")
    if not isinstance(value, pd.DataFrame) or value.empty:
        return None
    frame = value.copy()
    if "score" not in frame.columns:
        numeric = frame.select_dtypes(include="number").columns.tolist()
        if not numeric:
            return None
        frame = frame.rename(columns={numeric[0]: "score"})
    if "datetime" in frame.index.names:
        latest = frame.index.get_level_values("datetime").max()
        frame = frame.xs(latest, level="datetime") if len(frame.index.get_level_values("datetime").unique()) > 1 else frame.droplevel("datetime")
    elif "datetime" in frame.columns:
        latest = frame["datetime"].max()
        frame = frame[frame["datetime"] == latest]
    if "instrument" in frame.columns:
        frame = frame.set_index("instrument")
    frame = frame[frame.index.isin(valid_instruments)]
    return frame.sort_values("score", ascending=False)


def _decisions(frame: Any, holdings: set[str], top_k: int, drop_n: int, factor: int) -> dict[str, str]:
    instruments = frame.index.tolist()
    pool_size = top_k + drop_n * factor
    pool = set(instruments[:pool_size])
    held = [item for item in instruments if item in holdings]
    outside = [item for item in held if item not in pool]
    sell = set(outside[-drop_n:]) if outside else set()
    result = {item: ("SELL" if item in sell else "HOLD") for item in held}
    remaining = set(held) - sell
    buy_count = max(0, top_k - len(remaining))
    candidates = [item for item in instruments[:pool_size] if item not in holdings]
    primary = set(candidates[:buy_count])
    backup = set(candidates[buy_count : buy_count * factor])
    for item in candidates:
        result[item] = "BUY" if item in primary else ("BUY*" if item in backup else "--")
    for item in instruments[pool_size:]:
        if item not in holdings:
            result[item] = "--"
    ranks = {item: index + 1 for index, item in enumerate(instruments)}
    return {item: f"{decision} ({ranks[item]})" for item, decision in result.items()}


def build_model_opinions(request: ModelOpinionsRequest) -> ModelOpinionsResult | None:
    import pandas as pd

    specs, combo_info = discover_opinion_sources(request.run_config)
    if not specs:
        return None
    valid = set(request.sorted_predictions.index.tolist())
    frames: list[tuple[str, Any, str, str]] = [
        ("order_basis", request.sorted_predictions[["score"]].sort_values("score", ascending=False), "sorted_df", "order_basis")
    ]
    warnings: list[str] = []
    for spec in specs:
        try:
            frame = _latest_predictions(request.load_prediction(spec.record_id, spec.experiment_name), valid)
            if frame is not None:
                frames.append((spec.label, frame, spec.source_type, spec.detail))
        except Exception as exc:
            warnings.append(f"{spec.label}: {type(exc).__name__}: {exc}")
    holdings = set(request.current_holding_instruments)
    decisions = {
        label: _decisions(frame, holdings, request.top_k, request.drop_n, request.buy_suggestion_factor)
        for label, frame, _, _ in frames
    }
    rows = [
        {"instrument": instrument, **{label: decisions[label].get(instrument, "-") for label, _, _, _ in frames}}
        for instrument in request.focus_instruments
    ]
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.set_index("instrument")
    model_to_combos: dict[str, list[str]] = {}
    for combo, models in combo_info.items():
        for model in models:
            model_to_combos.setdefault(model, []).append(combo)
    summaries = tuple(
        {"label": label, "source": "sorted_df (与订单一致)" if kind == "sorted_df" else f"pkl:{detail}", "type": kind}
        for label, _, kind, detail in frames
    )
    return ModelOpinionsResult(
        dataframe=dataframe,
        combo_composition=combo_info,
        model_to_combos=model_to_combos,
        source_summaries=summaries,
        thresholds={"TopK": request.top_k, "DropN": request.drop_n, "buy_suggestion_factor": request.buy_suggestion_factor},
        warnings=tuple(warnings),
    )


def to_json_payload(result: ModelOpinionsResult, *, trade_date: str) -> dict:
    return {
        "trade_date": trade_date,
        "combo_composition": result.combo_composition,
        "model_to_combos": result.model_to_combos,
        "thresholds": result.thresholds,
        "legend": {
            "BUY": "非持仓, 排名靠前的买入候选 (数量 = 卖出数)",
            "BUY*": "非持仓, 备选买入 (应对停牌等情况)",
            "HOLD": "持仓, 继续持有",
            "SELL": "持仓, TopK 之外的最差 DropN",
            "--": "非持仓, 不在买入候选范围",
            "-": "无数据",
            "说明": "决策后的括号内数字表示该模型或组合下的预测排名",
        },
        "sources": [[item["label"], item["source"], item["type"]] for item in result.source_summaries],
        "warnings": list(result.warnings),
    }
