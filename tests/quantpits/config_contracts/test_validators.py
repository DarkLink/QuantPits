from pathlib import Path

from quantpits.config_contracts.normalizers import normalize_ensemble_config
from quantpits.config_contracts.validators import (
    validate_ensemble_config,
    validate_model_registry,
    validate_prod_config,
    validate_strategy_config,
)


def _codes(messages):
    return {message.code for message in messages}


def test_strategy_n_drop_cannot_exceed_topk():
    messages = validate_strategy_config(
        {
            "strategy": {"name": "topk_dropout", "params": {"topk": 3, "n_drop": 4}},
            "backtest": {
                "account": 100000,
                "exchange_kwargs": {
                    "deal_price": "close",
                    "open_cost": 0.001,
                    "close_cost": 0.001,
                    "min_cost": 5,
                },
            },
        }
    )

    assert "invalid-drop-ratio" in _codes(messages)


def test_prod_holding_numeric_strings_are_valid():
    messages = validate_prod_config(
        {
            "current_date": "2024-01-01",
            "last_processed_date": "2024-01-01",
            "current_cash": "100000",
            "current_holding": [
                {"instrument": "SH000001", "value": "100", "amount": "1234.56"}
            ],
        }
    )

    assert not [message for message in messages if message.severity == "error"]


def test_manual_ensemble_requires_complete_weights():
    normalized = normalize_ensemble_config(
        {
            "combos": {
                "manual": {
                    "models": ["demo_linear_Alpha158", "demo_tree_Alpha158"],
                    "method": "manual",
                    "manual_weights": {"demo_linear_Alpha158": 1.0},
                    "default": True,
                }
            }
        }
    )

    messages = validate_ensemble_config(normalized)

    assert "incomplete-manual-weights" in _codes(messages)


def test_ensemble_missing_or_multiple_default_is_warning():
    no_default = validate_ensemble_config(
        normalize_ensemble_config({"combos": {"a": {"models": ["m1"]}}})
    )
    multi_default = validate_ensemble_config(
        normalize_ensemble_config(
            {
                "combos": {
                    "a": {"models": ["m1"], "default": True},
                    "b": {"models": ["m2"], "default": True},
                }
            }
        )
    )

    assert "missing-default-combo" in _codes(no_default)
    assert "multiple-default-combos" in _codes(multi_default)
    assert all(message.severity == "error" for message in no_default + multi_default)


def test_model_registry_missing_workflow_is_warning(tmp_path):
    messages = validate_model_registry(
        {
            "models": {
                "demo_linear_Alpha158": {
                    "yaml_file": "config/workflow_config_missing.yaml",
                    "enabled": True,
                    "tags": ["baseline"],
                }
            }
        },
        workspace_root=Path(tmp_path),
    )

    assert "missing-workflow-yaml" in _codes(messages)
    assert all(message.severity == "warning" for message in messages)
