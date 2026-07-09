"""Validation rules for core workspace configuration files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from quantpits.config_contracts.core import ValidationMessage

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STEP_RE = re.compile(r"^\d+[MY]$")


def _msg(severity: str, code: str, path: str, message: str, hint: str = "") -> ValidationMessage:
    return ValidationMessage(severity=severity, code=code, path=path, message=message, hint=hint)


def _require_mapping(data: Any, path: str) -> Optional[List[ValidationMessage]]:
    if not isinstance(data, dict):
        return [_msg("error", "invalid-type", path, "Expected a mapping object.")]
    return None


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_non_negative_number(value: Any) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return float(value) == int(value)


def _validate_date_field(messages: List[ValidationMessage], data: Dict[str, Any], field: str, path: str) -> None:
    if field in data and not (isinstance(data[field], str) and DATE_RE.match(data[field])):
        messages.append(_msg("error", "invalid-date", path, f"{field} must be a YYYY-MM-DD string."))


def validate_model_config(data: Dict[str, Any], *, path: str = "config/model_config.json") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    for field in ("market", "benchmark", "start_time", "freq"):
        if field not in data:
            messages.append(_msg("error", "missing-field", path, f"Missing required field: {field}"))
    for field in (
        "start_time",
        "fit_start_time",
        "fit_end_time",
        "valid_start_time",
        "valid_end_time",
        "test_start_time",
        "test_end_time",
        "backtest_start_time",
        "backtest_end_time",
    ):
        _validate_date_field(messages, data, field, path)
    if "freq" in data and data["freq"] not in ("day", "week"):
        messages.append(_msg("warning", "unknown-frequency", path, "freq is not day or week."))
    purged_cv = data.get("purged_cv")
    if purged_cv is not None:
        if not isinstance(purged_cv, dict):
            messages.append(_msg("error", "invalid-purged-cv", path, "purged_cv must be a mapping."))
        else:
            for field in ("n_groups", "n_test_groups", "n_val_groups", "purge_steps", "embargo_steps"):
                if field not in purged_cv:
                    messages.append(_msg("error", "missing-purged-cv-field", path, f"purged_cv missing {field}."))
                elif not _is_int_like(purged_cv[field]) or int(purged_cv[field]) < 0:
                    messages.append(_msg("error", "invalid-purged-cv-field", path, f"purged_cv.{field} must be a non-negative integer."))
    return messages


def validate_strategy_config(data: Dict[str, Any], *, path: str = "config/strategy_config.yaml") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    strategy = data.get("strategy")
    backtest = data.get("backtest")
    if not isinstance(strategy, dict):
        messages.append(_msg("error", "missing-strategy", path, "strategy must be present."))
        strategy = {}
    if not isinstance(backtest, dict):
        messages.append(_msg("error", "missing-backtest", path, "backtest must be present."))
        backtest = {}
    params = strategy.get("params") if isinstance(strategy.get("params"), dict) else {}
    if not strategy.get("name"):
        messages.append(_msg("error", "missing-strategy-name", path, "strategy.name is required."))
    elif strategy.get("name") != "topk_dropout":
        messages.append(_msg("warning", "unknown-strategy", path, "strategy.name is not topk_dropout."))
    topk = params.get("topk")
    n_drop = params.get("n_drop")
    if not _is_positive_number(topk):
        messages.append(_msg("error", "invalid-topk", path, "strategy.params.topk must be positive."))
    if not _is_non_negative_number(n_drop):
        messages.append(_msg("error", "invalid-n-drop", path, "strategy.params.n_drop must be non-negative."))
    if _is_number(topk) and _is_number(n_drop) and float(n_drop) > float(topk):
        messages.append(_msg("error", "invalid-drop-ratio", path, "strategy.params.n_drop cannot exceed topk."))
    if "buy_suggestion_factor" in params and not _is_positive_number(params["buy_suggestion_factor"]):
        messages.append(_msg("error", "invalid-buy-factor", path, "buy_suggestion_factor must be positive."))
    if not _is_number(backtest.get("account")):
        messages.append(_msg("error", "invalid-account", path, "backtest.account must be numeric."))
    exchange = backtest.get("exchange_kwargs")
    if not isinstance(exchange, dict):
        messages.append(_msg("error", "missing-exchange-kwargs", path, "backtest.exchange_kwargs is required."))
    else:
        if "deal_price" not in exchange:
            messages.append(_msg("error", "missing-deal-price", path, "exchange_kwargs.deal_price is required."))
        for field in ("open_cost", "close_cost", "min_cost"):
            if field in exchange and not _is_non_negative_number(exchange[field]):
                messages.append(_msg("error", "invalid-cost", path, f"exchange_kwargs.{field} must be non-negative."))
    return messages


def validate_prod_config(data: Dict[str, Any], *, path: str = "config/prod_config.json") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    for field in ("current_date", "last_processed_date", "current_cash", "current_holding"):
        if field not in data:
            messages.append(_msg("error", "missing-field", path, f"Missing required field: {field}"))
    _validate_date_field(messages, data, "current_date", path)
    _validate_date_field(messages, data, "last_processed_date", path)
    if "current_date" in data and "last_processed_date" in data:
        if isinstance(data["current_date"], str) and isinstance(data["last_processed_date"], str):
            if DATE_RE.match(data["current_date"]) and DATE_RE.match(data["last_processed_date"]):
                if data["current_date"] < data["last_processed_date"]:
                    messages.append(_msg("warning", "date-inversion", path, "current_date is earlier than last_processed_date."))
    if "current_cash" in data and not _is_number(data["current_cash"]):
        messages.append(_msg("error", "invalid-current-cash", path, "current_cash must be numeric."))
    holdings = data.get("current_holding")
    if holdings is not None and not isinstance(holdings, list):
        messages.append(_msg("error", "invalid-holdings", path, "current_holding must be a list."))
    elif isinstance(holdings, list):
        for idx, holding in enumerate(holdings):
            item_path = f"{path}:current_holding[{idx}]"
            if not isinstance(holding, dict):
                messages.append(_msg("error", "invalid-holding", item_path, "Holding entry must be a mapping."))
                continue
            if not holding.get("instrument") or not isinstance(holding.get("instrument"), str):
                messages.append(_msg("error", "missing-instrument", item_path, "Holding instrument must be a non-empty string."))
            for field in ("value", "amount"):
                if field not in holding:
                    messages.append(_msg("error", "missing-holding-field", item_path, f"Holding missing {field}."))
                elif not _is_non_negative_number(holding[field]):
                    messages.append(_msg("error", "invalid-holding-number", item_path, f"Holding {field} must be non-negative numeric."))
    return messages


def validate_ensemble_config(data: Dict[str, Any], *, path: str = "config/ensemble_config.json") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    combos = data.get("combos")
    if not isinstance(combos, dict) or not combos:
        messages.append(_msg("error", "missing-combos", path, "ensemble config must define at least one combo."))
        return messages
    default_count = 0
    allowed_methods = {"equal", "manual", "icir_weighted", "dynamic"}
    for name, combo in combos.items():
        combo_path = f"{path}:combos.{name}"
        if not isinstance(combo, dict):
            messages.append(_msg("error", "invalid-combo", combo_path, "Combo must be a mapping."))
            continue
        if combo.get("default"):
            default_count += 1
        models = combo.get("models")
        if not isinstance(models, list) or not models:
            messages.append(_msg("error", "invalid-combo-models", combo_path, "Combo models must be a non-empty list."))
            models = []
        else:
            for model in models:
                if not isinstance(model, str) or not model.strip():
                    messages.append(_msg("error", "invalid-model-key", combo_path, "Combo model keys must be non-empty strings."))
        method = combo.get("method", "equal")
        if method not in allowed_methods:
            messages.append(_msg("warning", "unknown-ensemble-method", combo_path, f"Unknown ensemble method: {method}"))
        if method == "manual":
            weights = combo.get("manual_weights")
            if not isinstance(weights, dict):
                messages.append(_msg("error", "missing-manual-weights", combo_path, "manual combo requires manual_weights."))
            else:
                missing = [model for model in models if model not in weights]
                if missing:
                    messages.append(_msg("error", "incomplete-manual-weights", combo_path, "manual_weights must cover all combo models."))
    if default_count == 0:
        messages.append(_msg("warning", "missing-default-combo", path, "No default combo marked; current code falls back to the first combo."))
    elif default_count > 1:
        messages.append(_msg("warning", "multiple-default-combos", path, "Multiple default combos are marked."))
    if "min_model_ic" in data and not _is_number(data["min_model_ic"]):
        messages.append(_msg("error", "invalid-min-model-ic", path, "min_model_ic must be numeric."))
    return messages


def validate_ensemble_records(data: Dict[str, Any], *, path: str = "config/ensemble_records.json") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    combos = data.get("combos")
    if combos is not None and not isinstance(combos, dict):
        messages.append(_msg("error", "invalid-record-combos", path, "combos must be a mapping."))
    if isinstance(combos, dict):
        default_combo = data.get("default_combo")
        if default_combo and default_combo not in combos:
            messages.append(_msg("warning", "unknown-default-record-combo", path, "default_combo is not present in ensemble records combos."))
        for name, record_id in combos.items():
            if not isinstance(record_id, str) or not record_id:
                messages.append(_msg("error", "invalid-record-id", f"{path}:combos.{name}", "Combo record id must be a non-empty string."))
    if "default_record_id" in data and not isinstance(data["default_record_id"], str):
        messages.append(_msg("error", "invalid-default-record-id", path, "default_record_id must be a string."))
    return messages


def validate_model_registry(data: Dict[str, Any], *, workspace_root: Path, path: str = "config/model_registry.yaml") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        return [_msg("error", "missing-models", path, "model_registry.yaml must define models.")]
    for name, cfg in models.items():
        item_path = f"{path}:models.{name}"
        if not isinstance(cfg, dict):
            messages.append(_msg("error", "invalid-model-entry", item_path, "Model registry entry must be a mapping."))
            continue
        yaml_file = cfg.get("yaml_file")
        if not yaml_file:
            messages.append(_msg("error", "missing-yaml-file", item_path, "Model registry entry missing yaml_file."))
        else:
            workflow_path = workspace_root / str(yaml_file)
            if not workflow_path.exists():
                messages.append(_msg("warning", "missing-workflow-yaml", item_path, f"Workflow YAML not found: {yaml_file}"))
        if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
            messages.append(_msg("error", "invalid-enabled", item_path, "enabled must be boolean."))
        if "tags" in cfg and not isinstance(cfg["tags"], list):
            messages.append(_msg("error", "invalid-tags", item_path, "tags must be a list."))
    return messages


def validate_rolling_config(data: Dict[str, Any], *, path: str = "config/rolling_config.yaml") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    for field in ("rolling_start", "train_years", "valid_years", "test_step"):
        if field not in data:
            messages.append(_msg("error", "missing-field", path, f"Missing required field: {field}"))
    _validate_date_field(messages, data, "rolling_start", path)
    for field in ("train_years", "valid_years"):
        if field in data and (not _is_int_like(data[field]) or int(data[field]) <= 0):
            messages.append(_msg("error", "invalid-rolling-years", path, f"{field} must be a positive integer."))
    step = str(data.get("test_step", "")).strip().upper()
    if "test_step" in data and not STEP_RE.match(step):
        messages.append(_msg("error", "invalid-test-step", path, "test_step must match examples like 3M or 1Y."))
    method = str(data.get("training_method", "slide")).lower()
    if method not in ("slide", "cpcv"):
        messages.append(_msg("error", "invalid-training-method", path, "training_method must be slide or cpcv."))
    for field in ("cpcv_n_groups", "cpcv_n_val_groups", "cpcv_purge_steps", "cpcv_embargo_steps"):
        if field in data and (not _is_int_like(data[field]) or int(data[field]) < 0):
            messages.append(_msg("error", "invalid-cpcv-field", path, f"{field} must be a non-negative integer."))
    return messages


def validate_latest_train_records(data: Dict[str, Any], *, path: str = "latest_train_records.json") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    models = data.get("models")
    if not isinstance(models, dict):
        messages.append(_msg("error", "missing-train-record-models", path, "latest_train_records must contain models mapping."))
    else:
        for name, record_id in models.items():
            if not isinstance(name, str) or not name:
                messages.append(_msg("error", "invalid-train-record-model", path, "Train record model keys must be non-empty strings."))
            if not isinstance(record_id, str) or not record_id:
                messages.append(_msg("error", "invalid-train-record-id", path, "Train record ids must be non-empty strings."))
    _validate_date_field(messages, data, "anchor_date", path)
    for field in ("experiment_name", "static_experiment_name", "rolling_experiment_name", "cpcv_experiment_name"):
        if field in data and not isinstance(data[field], str):
            messages.append(_msg("error", "invalid-experiment-name", path, f"{field} must be a string."))
    return messages


def validate_cashflow_config(data: Dict[str, Any], *, path: str = "config/cashflow.json") -> List[ValidationMessage]:
    type_error = _require_mapping(data, path)
    if type_error:
        return type_error
    messages: List[ValidationMessage] = []
    if "cash_flow_today" in data:
        messages.append(_msg("info", "legacy-cashflow-format", path, "cash_flow_today legacy format is supported."))
        if not _is_number(data["cash_flow_today"]):
            messages.append(_msg("error", "invalid-cash-flow-today", path, "cash_flow_today must be numeric."))
    for field in ("cashflows", "processed"):
        if field in data and not isinstance(data[field], dict):
            messages.append(_msg("error", "invalid-cashflow-section", path, f"{field} must be a mapping."))
    cashflows = data.get("cashflows", {})
    if isinstance(cashflows, dict):
        for date_key, amount in cashflows.items():
            if not isinstance(date_key, str) or not DATE_RE.match(date_key):
                messages.append(_msg("error", "invalid-cashflow-date", path, "cashflow dates must be YYYY-MM-DD strings."))
            if not _is_number(amount):
                messages.append(_msg("error", "invalid-cashflow-amount", path, "cashflow amounts must be numeric."))
    return messages


def validate_ensemble_cross_refs(
    ensemble_config: Dict[str, Any],
    *,
    registry_models: Optional[Iterable[str]] = None,
    train_record_models: Optional[Iterable[str]] = None,
    path: str = "config/ensemble_config.json",
) -> List[ValidationMessage]:
    messages: List[ValidationMessage] = []
    registry_set: Optional[Set[str]] = set(registry_models) if registry_models is not None else None
    train_set: Optional[Set[str]] = set(train_record_models) if train_record_models is not None else None
    combos = ensemble_config.get("combos", {})
    if not isinstance(combos, dict):
        return messages
    for combo_name, combo in combos.items():
        if not isinstance(combo, dict) or not isinstance(combo.get("models"), list):
            continue
        for model_key in combo["models"]:
            if not isinstance(model_key, str):
                continue
            base_name = model_key.split("@", 1)[0]
            combo_path = f"{path}:combos.{combo_name}"
            if registry_set is not None and base_name not in registry_set and model_key not in registry_set:
                messages.append(_msg("warning", "ensemble-model-not-in-registry", combo_path, f"Model {model_key} is not present in model_registry.yaml."))
            if train_set is not None and model_key not in train_set:
                messages.append(_msg("warning", "ensemble-model-not-in-train-records", combo_path, f"Model {model_key} is not present in latest_train_records.json."))
    return messages
