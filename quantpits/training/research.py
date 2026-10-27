"""Prepared static research settings, without loading account state."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import yaml

from quantpits.config_contracts.validators import validate_strategy_config
from quantpits.training.errors import TrainingPlanError
from quantpits.training.persistence import read_with_baseline


@dataclass(frozen=True)
class PreparedResearchConfig:
    path: str
    fingerprint: str
    # Store immutable content; each consumer receives its own dictionary.
    content_json: str

    def to_dict(self):
        return json.loads(self.content_json)

    def description(self):
        return dict(path=self.path, fingerprint=self.fingerprint,
                    account_source="backtest.account", **self.to_dict())

    def training_config(self, model_config):
        selected = self.to_dict()
        config = dict(model_config)
        config.update(selected)
        config.update(research_params({}, selected))
        # calculate_dates retains its legacy cash alias. Here that value comes
        # exclusively from the selected research input, never account state.
        config["current_full_cash"] = selected["backtest"]["account"]
        return config


def research_params(params, config):
    result = dict(params)
    strategy = config["strategy"]["params"]
    result.update(account=config["backtest"]["account"], topk=strategy["topk"],
                  n_drop=strategy["n_drop"],
                  buy_suggestion_factor=strategy.get("buy_suggestion_factor", 2))
    return result


def prepare_research_config(ctx, path):
    relative = path.relative_to(ctx.root).as_posix()
    try:
        raw, baseline = read_with_baseline(path, display_path=relative)
        config = yaml.safe_load(raw) if raw is not None else None
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise TrainingPlanError("--research-config YAML is missing or invalid: %s" % relative) from exc
    if not isinstance(config, dict) or set(config) != {"strategy", "backtest"}:
        raise TrainingPlanError("--research-config requires only strategy and backtest mappings")
    strategy, backtest = config["strategy"], config["backtest"]
    if (not isinstance(strategy, dict) or set(strategy) != {"name", "params"}
            or not isinstance(strategy.get("params"), dict)
            or not isinstance(backtest, dict) or set(backtest) != {"account", "exchange_kwargs"}
            or not isinstance(backtest.get("exchange_kwargs"), dict)):
        raise TrainingPlanError("--research-config requires strategy.name/params and backtest.account/exchange_kwargs")
    account = backtest["account"]
    try:
        valid_cash = type(account) in (int, float) and math.isfinite(account) and account > 0
    except OverflowError:
        valid_cash = False
    if not valid_cash:
        raise TrainingPlanError("--research-config backtest.account must be finite positive numeric cash")
    for field in ("topk", "n_drop"):
        if type(strategy["params"].get(field)) is not int:
            raise TrainingPlanError("--research-config strategy.params.%s must be an integer" % field)
    errors = [message.message for message in validate_strategy_config(config, path=relative)
              if message.severity == "error"]
    if errors:
        raise TrainingPlanError("--research-config: " + "; ".join(errors))
    try:
        content = json.dumps(config, sort_keys=True, allow_nan=False)
        # Use the existing factory to reject unsupported names and structures.
        # An explicit dictionary never invokes its workspace/account loader.
        from quantpits.utils.strategy import generate_port_analysis_config
        generate_port_analysis_config(config_dict=config)
    except (TypeError, ValueError, KeyError) as exc:
        raise TrainingPlanError("--research-config: %s" % exc) from exc
    return PreparedResearchConfig(relative, baseline.fingerprint, content)
