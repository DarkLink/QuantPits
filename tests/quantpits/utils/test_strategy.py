import os
import sys
import yaml
import json
import pytest
import importlib
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    config_dir = workspace / "config"
    config_dir.mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    # Ensure scripts dir is in sys.path for bare `import env`
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'quantpits', 'scripts')
    scripts_dir = os.path.abspath(scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from quantpits.utils import env
    importlib.reload(env)

    # Also reload bare `env` module used by strategy.py via `import env`
    

    from quantpits.utils import strategy
    importlib.reload(strategy)

    yield strategy, workspace, config_dir


# ── load_strategy_config ─────────────────────────────────────────────────

def test_load_strategy_config_yaml(mock_env):
    strategy, workspace, config_dir = mock_env

    yaml_config = {
        "strategy": {
            "name": "topk_dropout",
            "params": {"topk": 10, "n_drop": 2, "buy_suggestion_factor": 3}
        },
        "backtest": {
            "account": 50_000_000,
            "exchange_kwargs": {"limit_threshold": 0.1}
        }
    }
    with open(config_dir / "strategy_config.yaml", "w") as f:
        yaml.dump(yaml_config, f)

    config = strategy.load_strategy_config()
    assert config["strategy"]["params"]["topk"] == 10
    assert config["strategy"]["params"]["n_drop"] == 2
    assert config["backtest"]["account"] == 50_000_000
    assert config["backtest"]["exchange_kwargs"]["limit_threshold"] == 0.1


def test_load_strategy_config_defaults(mock_env):
    """When no YAML exists and no legacy JSON, defaults should be returned."""
    strategy, workspace, config_dir = mock_env

    config = strategy.load_strategy_config()
    assert config["strategy"]["name"] == "topk_dropout"
    assert config["strategy"]["params"]["topk"] == 20
    assert config["strategy"]["params"]["sell_out_of_universe"] is True
    assert config["strategy"]["params"]["n_drop"] == 3
    assert config["backtest"]["account"] == 100_000_000


def test_load_strategy_config_fallback_legacy(mock_env):
    """When no YAML but legacy JSONs exist, should fallback to those (now handled fully via mock for isolation)."""
    strategy, workspace, config_dir = mock_env

    with patch('quantpits.utils.config_loader.load_workspace_config') as mock_load:
        mock_load.return_value = {
            "TopK": 15,
            "DropN": 5,
            "buy_suggestion_factor": 4
        }
        config = strategy.load_strategy_config()
        
    assert config["strategy"]["params"]["topk"] == 15
    assert config["strategy"]["params"]["n_drop"] == 5
    assert config["strategy"]["params"]["buy_suggestion_factor"] == 4


# ── Factory methods ──────────────────────────────────────────────────────

def test_get_strategy_params(mock_env):
    strategy, _, _ = mock_env

    config = {
        "strategy": {
            "name": "topk_dropout",
            "params": {"topk": 25, "n_drop": 4, "buy_suggestion_factor": 2}
        },
        "backtest": {"account": 100_000}
    }
    params = strategy.get_strategy_params(config_dict=config)
    assert params["topk"] == 25
    assert params["n_drop"] == 4
    # Ensure it returns a copy
    params["topk"] = 99
    assert config["strategy"]["params"]["topk"] == 25


def test_get_backtest_config(mock_env):
    strategy, _, _ = mock_env

    config = {
        "strategy": {"name": "topk_dropout", "params": {}},
        "backtest": {
            "account": 200_000,
            "exchange_kwargs": {"limit_threshold": 0.1}
        }
    }
    bt = strategy.get_backtest_config(config_dict=config, fallback_freq="week")
    assert bt["account"] == 200_000
    assert bt["exchange_kwargs"]["freq"] == "week"


def test_get_backtest_config_freq_already_set(mock_env):
    strategy, _, _ = mock_env

    config = {
        "strategy": {"name": "topk_dropout", "params": {}},
        "backtest": {
            "account": 100,
            "exchange_kwargs": {"freq": "day", "limit_threshold": 0.1}
        }
    }
    bt = strategy.get_backtest_config(config_dict=config, fallback_freq="week")
    # Should keep existing freq
    assert bt["exchange_kwargs"]["freq"] == "day"


def test_create_order_generator(mock_env):
    strategy, _, _ = mock_env

    config = {
        "strategy": {
            "name": "topk_dropout",
            "params": {"topk": 12, "n_drop": 2, "buy_suggestion_factor": 3}
        },
        "backtest": {}
    }
    gen = strategy.create_order_generator(config_dict=config)
    assert isinstance(gen, strategy.TopkDropoutOrderGenerator)
    assert gen.topk == 12
    assert gen.drop_n == 2
    assert gen.buy_suggestion_factor == 3


def test_create_order_generator_unknown_strategy(mock_env):
    strategy, _, _ = mock_env

    config = {
        "strategy": {"name": "unknown_strategy", "params": {}},
        "backtest": {}
    }
    with pytest.raises(ValueError, match="not found in STRATEGY_REGISTRY"):
        strategy.create_order_generator(config_dict=config)


def test_generate_port_analysis_config(mock_env):
    strategy, _, _ = mock_env

    config = {
        "strategy": {
            "name": "topk_dropout",
            "params": {"topk": 20, "n_drop": 3, "buy_suggestion_factor": 2, "sell_out_of_universe": True, "only_tradable": True}
        },
        "backtest": {
            "account": 100_000_000,
            "exchange_kwargs": {"limit_threshold": 0.095, "deal_price": "close"}
        }
    }
    pa = strategy.generate_port_analysis_config(config_dict=config, freq="week")
    assert pa["strategy"]["class"] == "TopkDropoutStrategy"
    assert pa["strategy"]["kwargs"]["topk"] == 20
    assert "buy_suggestion_factor" not in pa["strategy"]["kwargs"]
    assert "sell_out_of_universe" not in pa["strategy"]["kwargs"]
    assert pa["executor"]["kwargs"]["time_per_step"] == "week"
    assert pa["backtest"]["account"] == 100_000_000


# ── TopkDropoutOrderGenerator edge cases ─────────────────────────────────

def test_topk_dropout_empty_holding(mock_env):
    """With no current holdings, all positions should be bought."""
    strategy, _, _ = mock_env

    gen = strategy.TopkDropoutOrderGenerator(topk=3, n_drop=1, buy_suggestion_factor=2)

    pred_data = {
        'instrument': ['A', 'B', 'C', 'D', 'E'],
        'datetime': ['2026-03-01'] * 5,
        'score': [0.9, 0.8, 0.7, 0.6, 0.5]
    }
    pred_df = pd.DataFrame(pred_data).set_index(['instrument', 'datetime'])

    price_data = {
        'instrument': ['A', 'B', 'C', 'D', 'E'],
        'current_close': [10.0] * 5,
        'possible_max': [11.0] * 5,
        'possible_min': [9.0] * 5
    }
    price_df = pd.DataFrame(price_data).set_index('instrument')

    hold_final, sell_cand, buy_cand, _, buy_count = gen.analyze_positions(
        pred_df, price_df, []
    )

    assert len(hold_final) == 0
    assert len(sell_cand) == 0
    assert buy_count == 3  # topk - 0 hold = 3


def test_topk_dropout_all_held_in_topk(mock_env):
    """If all current holdings are in topK, nothing should be sold."""
    strategy, _, _ = mock_env

    gen = strategy.TopkDropoutOrderGenerator(topk=3, n_drop=1, buy_suggestion_factor=2)

    pred_data = {
        'instrument': ['A', 'B', 'C', 'D', 'E'],
        'datetime': ['2026-03-01'] * 5,
        'score': [0.9, 0.8, 0.7, 0.6, 0.5]
    }
    pred_df = pd.DataFrame(pred_data).set_index(['instrument', 'datetime'])

    price_data = {
        'instrument': ['A', 'B', 'C', 'D', 'E'],
        'current_close': [10.0] * 5,
        'possible_max': [11.0] * 5,
        'possible_min': [9.0] * 5
    }
    price_df = pd.DataFrame(price_data).set_index('instrument')

    current_holding = [
        {'instrument': 'A', 'value': 100},
        {'instrument': 'B', 'value': 100},
        {'instrument': 'C', 'value': 100},
    ]

    hold_final, sell_cand, buy_cand, _, buy_count = gen.analyze_positions(
        pred_df, price_df, current_holding
    )

    assert len(hold_final) == 3
    assert len(sell_cand) == 0
    assert buy_count == 0


def test_generate_buy_orders_insufficient_cash(mock_env):
    """If cash is too low to buy 100 shares, no buy orders should be generated."""
    strategy, _, _ = mock_env

    gen = strategy.TopkDropoutOrderGenerator(topk=2, n_drop=1, buy_suggestion_factor=2)

    buy_candidates = pd.DataFrame({
        'instrument': ['A'],
        'possible_max': [100.0],  # 100 shares * 100 = 10000, need > 10000
        'score': [0.9],
        'current_close': [95.0]
    }).set_index('instrument')

    # Cash is only 50, can't even buy 100 shares at 100.0
    buys = gen.generate_buy_orders(buy_candidates, buy_count=1, available_cash=50, next_trade_date_string="2026-03-02")
    assert len(buys) == 0


def test_generate_buy_orders_success(mock_env):
    strategy, _, _ = mock_env
    gen = strategy.TopkDropoutOrderGenerator(topk=2, n_drop=1)
    
    buy_candidates = pd.DataFrame({
        'instrument': ['A'],
        'possible_max': [10.0],
        'score': [0.9],
        'current_close': [9.5]
    }).set_index('instrument')
    
    # Cash is 2000, can buy 200 shares at 10.0
    buys = gen.generate_buy_orders(buy_candidates, buy_count=1, available_cash=2000, next_trade_date_string="2026-03-02")
    assert len(buys) == 1
    assert buys[0]['instrument'] == 'A'
    assert buys[0]['value'] == 200
    assert buys[0]['estimated_amount'] == 2000.0


def test_generate_sell_orders(mock_env):
    strategy, _, _ = mock_env
    gen = strategy.TopkDropoutOrderGenerator(topk=20, n_drop=3)
    
    sell_candidates = pd.DataFrame({
        'instrument': ['A'],
        'possible_min': [9.0],
        'score': [0.1],
        'current_close': [10.0]
    }).set_index('instrument')
    
    current_holding = [{'instrument': 'A', 'value': 1000}]
    
    orders, amount = gen.generate_sell_orders(sell_candidates, current_holding, "2026-03-02")
    assert len(orders) == 1
    assert orders[0]['instrument'] == 'A'
    assert orders[0]['value'] == 1000
    assert orders[0]['estimated_amount'] == 9000.0
    assert amount == 9000.0


def _universe_frames():
    pred = pd.DataFrame({
        "instrument": ["A", "B", "C", "D", "E", "F", "G"],
        "datetime": ["2026-03-01"] * 7,
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    }).set_index(["instrument", "datetime"])
    prices = pd.DataFrame({
        "instrument": ["A", "B", "C", "D", "E", "F", "G", "X", "Y"],
        "current_close": [10.0] * 9,
        "possible_max": [11.0] * 9,
        "possible_min": [9.0] * 9,
    }).set_index("instrument")
    return pred, prices


def test_universe_exits_consume_drop_budget_and_drive_exact_buy_count(mock_env):
    strategy, _, _ = mock_env
    from quantpits.order.execution import UniverseSnapshot

    gen = strategy.TopkDropoutOrderGenerator(topk=3, n_drop=3, buy_suggestion_factor=1)
    pred, prices = _universe_frames()
    holdings = [{"instrument": item, "value": 100} for item in ("D", "G", "X", "Y")]
    result = gen.analyze_positions_with_universe(
        pred, prices, holdings, UniverseSnapshot.observe("demo", "2026-03-01", ["A", "B", "C", "D", "E", "F", "G"])
    )
    assert set(result.forced_exit_candidates.index) == {"X", "Y"}
    assert result.normal_sell_candidates.index.tolist() == ["G"]
    assert result.executable_sell_count == 3
    assert result.remaining_after_sell == 1
    assert result.target_buy_count == 2
    assert set(result.buy_candidates.index).isdisjoint({"D", "G", "X", "Y"})
    orders, _ = gen.generate_sell_orders(result.forced_exit_candidates, holdings, "2026-03-02")
    assert {item["instrument"] for item in orders} == {"X", "Y"}
    assert all(item["score"] is None for item in orders)


def test_pending_forced_exit_and_unscored_holding_do_not_create_buy_capacity(mock_env):
    strategy, _, _ = mock_env
    from quantpits.order.execution import UniverseSnapshot

    gen = strategy.TopkDropoutOrderGenerator(topk=2, n_drop=1, buy_suggestion_factor=1)
    pred, prices = _universe_frames()
    prices = prices.drop(index="X")
    holdings = [{"instrument": "X", "value": 100}, {"instrument": "Z", "value": 100}]
    result = gen.analyze_positions_with_universe(
        pred, prices, holdings, UniverseSnapshot.observe("demo", "2026-03-01", ["A", "B", "C", "D", "E", "Z"])
    )
    assert result.pending_forced_exit_instruments == ("X",)
    assert result.eligible_unscored_instruments == ("Z",)
    assert result.executable_sell_count == 0
    assert result.target_buy_count == 0


def test_forced_exits_can_exceed_dropn_without_normal_sells(mock_env):
    strategy, _, _ = mock_env
    from quantpits.order.execution import UniverseSnapshot

    gen = strategy.TopkDropoutOrderGenerator(topk=2, n_drop=1, buy_suggestion_factor=1)
    pred, prices = _universe_frames()
    holdings = [{"instrument": "X", "value": 100}, {"instrument": "Y", "value": 100}]
    result = gen.analyze_positions_with_universe(
        pred, prices, holdings,
        UniverseSnapshot.observe("demo", "2026-03-01", ["A", "B", "C", "D", "E", "F", "G"]),
    )
    assert result.forced_exit_count == 2
    assert result.normal_sell_count == 0
    assert result.executable_sell_count == 2
    assert result.target_buy_count == 2


def test_disabled_universe_check_still_counts_unranked_holdings(mock_env):
    strategy, _, _ = mock_env
    gen = strategy.TopkDropoutOrderGenerator(
        topk=2, n_drop=0, buy_suggestion_factor=1, sell_out_of_universe=False,
    )
    pred, prices = _universe_frames()
    holdings = [{"instrument": "D", "value": 100}, {"instrument": "Z", "value": 100}]
    result = gen.analyze_positions_with_universe(pred, prices, holdings, None)
    assert result.forced_exit_count == 0
    assert result.eligible_holding_count is None
    assert result.eligible_unscored_instruments == ("Z",)
    assert result.target_buy_count == 0


# ── create_backtest_strategy ─────────────────────────────────────────────

def test_create_backtest_strategy(mock_env, monkeypatch):
    strategy, _, _ = mock_env

    # Mock Qlib strategy class
    mock_qlib_strat = MagicMock()
    mock_module = MagicMock()
    setattr(mock_module, "TopkDropoutStrategy", mock_qlib_strat)
    
    monkeypatch.setitem(sys.modules, "qlib.contrib.strategy", mock_module)

    config = {
        "strategy": {
            "name": "topk_dropout",
            "params": {"topk": 20, "n_drop": 3, "buy_suggestion_factor": 2, "sell_out_of_universe": True}
        },
        "backtest": {}
    }
    
    signal = MagicMock()
    strat_instance = strategy.create_backtest_strategy(signal, config_dict=config)
    
    # Verify that the correct class was called with the correct parameters
    # Note: buy_suggestion_factor should have been popped
    mock_qlib_strat.assert_called_once_with(signal=signal, topk=20, n_drop=3)


def test_create_backtest_strategy_unknown(mock_env):
    strategy, _, _ = mock_env

    config = {
        "strategy": {"name": "non_existent_strategy", "params": {}},
        "backtest": {}
    }
    
    with pytest.raises(ValueError, match="not found in STRATEGY_REGISTRY"):
        strategy.create_backtest_strategy(MagicMock(), config_dict=config)
