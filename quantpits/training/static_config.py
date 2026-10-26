"""Static model/data and record configuration builders with explicit inputs.

These functions do not resolve a workspace, query accounts, initialize Qlib or
load models. Pretrained paths and portfolio settings are supplied by adapters.
"""
import copy

import yaml

from quantpits.utils.constants import TRADING_DAYS_PER_YEAR, TRADING_WEEKS_PER_YEAR


_UNSET = object()


def _strategy_params(config, params, strict=True):
    strategy = config.get("strategy")
    if not strict and (not isinstance(strategy, dict) or not isinstance(strategy.get("params"), dict)):
        return
    if "strategy" in config and "params" in config["strategy"]:
        for name, default in (("topk", 20), ("n_drop", 3), ("buy_suggestion_factor", 2)):
            config["strategy"]["params"][name] = params.get(name, default)


def _model_data(config, params, no_pretrain, pretrain_path):
    freq = params.get("freq", "week").lower()
    config["market"] = params["market"]
    if "benchmark" in params:
        config["benchmark"] = params["benchmark"]
    handler = config["data_handler_config"]
    for name in ("start_time", "end_time", "fit_start_time", "fit_end_time"):
        handler[name] = params[name]
    handler["instruments"] = params["market"]
    if "label_formula" in params:
        handler["label"] = [params["label_formula"]]
    elif isinstance(handler.get("label"), list) and len(handler["label"]) > 1:
        pass
    else:
        handler["label"] = ["Ref($close, -6) / Ref($close, -1) - 1" if freq == "week"
                            else "Ref($close, -2) / Ref($close, -1) - 1"]
    if "task" in config and "dataset" in config["task"]:
        segments = config["task"]["dataset"]["kwargs"]["segments"]
        for name, start, end in (("train", "fit_start_time", "fit_end_time"),
                                 ("valid", "valid_start_time", "valid_end_time"),
                                 ("test", "test_start_time", "test_end_time")):
            segments[name] = [params[start], params[end]]
        for name, source in (("pretrain", "train"), ("pretrain_validation", "valid")):
            if name in segments:
                segments[name] = list(segments[source])
    if "task" in config and "model" in config["task"]:
        kwargs = config["task"]["model"].get("kwargs", {})
        if "base_model" in kwargs:
            if no_pretrain or pretrain_path is None:
                kwargs.pop("model_path", None)
            elif pretrain_path is not _UNSET:
                kwargs["model_path"] = pretrain_path


def _records(config, params, port_analysis_config):
    freq = params.get("freq", "week").lower()
    _strategy_params(config, params)
    if "port_analysis_config" in config:
        if port_analysis_config is None:
            raise ValueError("record construction requires an explicit port_analysis_config")
        portfolio = copy.deepcopy(port_analysis_config)
        portfolio["backtest"]["start_time"] = params["test_start_time"]
        portfolio["backtest"]["end_time"] = params["test_end_time"]
        portfolio["backtest"]["account"] = params.get("account", portfolio["backtest"]["account"])
        portfolio["backtest"]["benchmark"] = params["benchmark"]
        config["port_analysis_config"] = portfolio
        for record in config.get("task", {}).get("record", []):
            if record.get("class") == "PortAnaRecord" and "kwargs" in record and "config" in record["kwargs"]:
                record["kwargs"]["config"] = portfolio
    for record in config.get("task", {}).get("record", []):
        if record.get("class") == "SigAnaRecord":
            record.setdefault("kwargs", {})["ann_scaler"] = (
                TRADING_WEEKS_PER_YEAR if freq == "week" else TRADING_DAYS_PER_YEAR)


def build_model_task(template, params, *, no_pretrain=False, pretrain_path=_UNSET):
    """Copy and inject model/data inputs, without constructing any records.

    All unknown model/dataset/processor fields survive. Existing YAML handler
    aliases are preserved; a non-aliased handler retains its legacy behavior.
    An omitted pretrain_path preserves a configured path, None removes it.
    """
    selected = {key: value for key, value in template.items() if key != "port_analysis_config"}
    if isinstance(selected.get("task"), dict):
        selected["task"] = {key: value for key, value in selected["task"].items() if key != "record"}
    config = copy.deepcopy(selected)
    # Preserve fit inputs if a legacy YAML aliases strategy params into model
    # kwargs, while ignoring invalid, unused research settings on this path.
    _strategy_params(config, params, strict=False)
    config.pop("strategy", None)
    _model_data(config, params, no_pretrain, pretrain_path)
    return config


def load_model_task(yaml_path, params):
    """Read an explicit template for dataset/cache preparation only.

    No pretraining lookup is needed to prepare a handler. Configured model
    options are retained; only the returned dataset is used by the cache.
    """
    with open(yaml_path, "r") as stream:
        return build_model_task(yaml.safe_load(stream), params)


def build_record_config(template, params, *, port_analysis_config=None):
    """Build only record/strategy settings from explicit research inputs."""
    selected = {key: template[key] for key in ("strategy", "port_analysis_config") if key in template}
    if "task" in template:
        selected["task"] = {key: value for key, value in template["task"].items() if key == "record"}
    config = copy.deepcopy(selected)
    _records(config, params, port_analysis_config)
    return config


def compose_static_config(template, params, *, no_pretrain=False, pretrain_path=_UNSET,
                          port_analysis_config=None):
    """Legacy full view using the same injectors and original alias graph."""
    config = copy.deepcopy(template)
    # The facade historically requires benchmark even without portfolio records.
    config["benchmark"] = params["benchmark"]
    _strategy_params(config, params)
    _model_data(config, params, no_pretrain, pretrain_path)
    _records(config, params, port_analysis_config)
    return config
