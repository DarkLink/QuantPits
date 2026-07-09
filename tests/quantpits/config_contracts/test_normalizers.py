import json

from quantpits.config_contracts.normalizers import (
    normalize_ensemble_config,
    normalize_strategy_config,
)
from quantpits.utils.workspace import fingerprint_value


def test_normalize_ensemble_current_schema():
    cfg = {
        "combos": {
            "default": {
                "models": ["demo_linear_Alpha158"],
                "method": "equal",
                "default": True,
            }
        },
        "min_model_ic": 0.0,
    }

    normalized = normalize_ensemble_config(cfg)

    assert normalized["combos"]["default"]["models"] == ["demo_linear_Alpha158"]
    assert normalized["combos"]["default"]["default"] is True
    assert normalized["min_model_ic"] == 0.0


def test_normalize_ensemble_legacy_schema():
    cfg = {
        "models": ["demo_linear_Alpha158"],
        "ensemble_method": "manual",
        "manual_weights": {"demo_linear_Alpha158": 1.0},
        "use_ensemble": True,
    }

    normalized = normalize_ensemble_config(cfg)

    assert list(normalized["combos"]) == ["legacy"]
    assert normalized["combos"]["legacy"]["method"] == "manual"
    assert normalized["combos"]["legacy"]["default"] is True
    assert normalized["combos"]["legacy"]["manual_weights"] == {"demo_linear_Alpha158": 1.0}
    assert "use_ensemble" not in normalized


def test_normalize_strategy_promotes_top_level_compat_fields():
    normalized = normalize_strategy_config(
        {
            "strategy": {"name": "topk_dropout", "params": {"topk": 20, "n_drop": 3}},
            "backtest": {"account": 100000, "exchange_kwargs": {}},
        }
    )

    assert normalized["topk"] == 20
    assert normalized["TopK"] == 20
    assert normalized["n_drop"] == 3
    assert normalized["DropN"] == 3


def test_fingerprint_is_stable_across_dict_key_order():
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = json.loads('{"a": {"c": 3, "d": 4}, "b": 2}')

    assert fingerprint_value(left) == fingerprint_value(right)
