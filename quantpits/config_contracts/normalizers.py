"""In-memory normalizers for workspace configuration files."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from quantpits.utils.ensemble_utils import parse_ensemble_config


def normalize_model_config(data: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(data)


def normalize_strategy_config(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(data)
    strategy = normalized.setdefault("strategy", {})
    params = strategy.setdefault("params", {})

    if "topk" not in params and "TopK" in normalized:
        params["topk"] = normalized["TopK"]
    if "n_drop" not in params and "DropN" in normalized:
        params["n_drop"] = normalized["DropN"]

    if "topk" in params:
        normalized["topk"] = params["topk"]
        normalized["TopK"] = params["topk"]
    if "n_drop" in params:
        normalized["n_drop"] = params["n_drop"]
        normalized["DropN"] = params["n_drop"]

    if "buy_suggestion_factor" in params:
        normalized["buy_suggestion_factor"] = params["buy_suggestion_factor"]

    return normalized


def normalize_prod_config(data: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(data)


def normalize_ensemble_config(data: Dict[str, Any]) -> Dict[str, Any]:
    combos, global_config = parse_ensemble_config(deepcopy(data))
    normalized: Dict[str, Any] = deepcopy(global_config)
    normalized["combos"] = {}
    for name, combo in combos.items():
        combo_copy = deepcopy(combo)
        combo_copy.setdefault("method", "equal")
        combo_copy.setdefault("default", False)
        normalized["combos"][str(name)] = combo_copy
    return normalized


def normalize_ensemble_records(data: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(data)


def normalize_model_registry(data: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(data)


def normalize_rolling_config(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(data)
    if "test_step" in normalized:
        normalized["test_step"] = str(normalized["test_step"]).strip().upper()
    if "training_method" in normalized:
        normalized["training_method"] = str(normalized["training_method"]).lower()
    return normalized


def normalize_latest_train_records(data: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(data)


def normalize_cashflow_config(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(data)
    if "cash_flow_today" in normalized and "cashflows" not in normalized:
        normalized["legacy_cash_flow_today"] = normalized.get("cash_flow_today")
    return normalized


def summarize_config(name: str, normalized: Any) -> Dict[str, Any]:
    if not isinstance(normalized, dict):
        return {}
    if name == "ensemble_config":
        combos = normalized.get("combos", {})
        default_combos = [
            combo_name
            for combo_name, combo in combos.items()
            if isinstance(combo, dict) and combo.get("default")
        ]
        return {"combo_count": len(combos), "default_combos": default_combos}
    if name == "model_registry":
        models = normalized.get("models", {})
        enabled = [
            model_name
            for model_name, cfg in models.items()
            if isinstance(cfg, dict) and cfg.get("enabled", True)
        ] if isinstance(models, dict) else []
        return {"model_count": len(models) if isinstance(models, dict) else 0, "enabled_count": len(enabled)}
    if name == "latest_train_records":
        models = normalized.get("models", {})
        return {"model_record_count": len(models) if isinstance(models, dict) else 0}
    if name == "prod_config":
        holdings = normalized.get("current_holding", [])
        return {"holding_count": len(holdings) if isinstance(holdings, list) else 0}
    keys: List[str] = sorted(str(key) for key in normalized.keys())
    return {"keys": keys[:12], "key_count": len(keys)}
