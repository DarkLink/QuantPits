import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from decimal import Decimal

# Apply environment mocking before loading the module
@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    (workspace / "config").mkdir()
    (workspace / "data").mkdir()
    (workspace / "output").mkdir()
    (workspace / "output" / "predictions").mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, order_gen, strategy
    import importlib
    importlib.reload(env)
    importlib.reload(order_gen)
    importlib.reload(strategy)
    
    yield order_gen, strategy, workspace

def test_get_cashflow_today(mock_env):
    order_gen, strategy, _ = mock_env
    
    # Test new format
    cashflow_config_new = {"cashflows": {"2026-03-01": 50000, "2026-03-02": -20000}}
    assert order_gen.get_cashflow_today(cashflow_config_new, "2026-03-01") == 50000.0
    assert order_gen.get_cashflow_today(cashflow_config_new, "2026-03-02") == -20000.0
    assert order_gen.get_cashflow_today(cashflow_config_new, "2026-03-03") == 0.0
    
    # Test old format
    cashflow_config_old = {"cash_flow_today": 12345}
    assert order_gen.get_cashflow_today(cashflow_config_old, "2026-03-01") == 12345.0
    
    # Test empty
    assert order_gen.get_cashflow_today({}, "2026-03-01") == 0.0

def test_analyze_positions(mock_env):
    order_gen, strategy, _ = mock_env
    
    order_generator = strategy.TopkDropoutOrderGenerator(topk=2, n_drop=1, buy_suggestion_factor=2)
    
    # Mock Predictions (sorted by score)
    pred_data = {
        'instrument': ['000001', '000002', '000003', '000004', '000005', '000006'],
        'datetime': ['2026-03-01'] * 6,
        'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    }
    pred_df = pd.DataFrame(pred_data).set_index(['instrument', 'datetime'])
    
    # Mock Prices
    price_data = {
        'instrument': ['000001', '000002', '000003', '000004', '000005', '000006'],
        'current_close': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        'possible_max': [11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
        'possible_min': [9.0, 18.0, 27.0, 36.0, 45.0, 54.0]
    }
    price_df = pd.DataFrame(price_data).set_index('instrument')
    
    # Current Holdings (000003 is still good, 000006 is bad and should be sold)
    current_holding = [
        {'instrument': '000003', 'value': 100},
        {'instrument': '000006', 'value': 200}
    ]
    
    # Params: top_k=2, drop_n=1, factor=2
    # Candidate pool: Top (2 + 1*2) = Top 4 [000001, 000002, 000003, 000004]
    # Holding 000003 is in pool -> HOLD
    # Holding 000006 is out of pool -> SELL
    # Final hold = [000003] -> Need (top_k 2 - 1) = 1 buy
    # Buy candidates = non-held in TopK buffer -> [000001, 000002, 000004]
    # We need 1 * factor(2) = 2 buy candidates -> [000001, 000002]
    
    hold_final, sell_cand, buy_cand, merged_df, buy_count = order_generator.analyze_positions(
        pred_df, price_df, current_holding
    )
    
    assert len(hold_final) == 1
    assert '000003' in hold_final.index
    
    assert len(sell_cand) == 1
    assert '000006' in sell_cand.index
    
    assert buy_count == 1
    assert len(buy_cand) == 2
    assert list(buy_cand.index) == ['000001', '000002']
    
    assert len(merged_df) == 6

def test_generate_orders(mock_env):
    order_gen, strategy, _ = mock_env
    
    order_generator = strategy.TopkDropoutOrderGenerator(topk=2, n_drop=1, buy_suggestion_factor=2)
    
    # Generate Sell Orders
    sell_candidates = pd.DataFrame({
        'instrument': ['000006'],
        'possible_min': [54.0],
        'score': [0.4],
        'current_close': [60.0]
    }).set_index('instrument')
    
    current_holding = [{'instrument': '000006', 'value': 200}]
    
    sells, sell_amount = order_generator.generate_sell_orders(sell_candidates, current_holding, "2026-03-02")
    
    assert len(sells) == 1
    assert sells[0]['instrument'] == '000006'
    assert sells[0]['value'] == 200
    assert sells[0]['estimated_amount'] == 200 * 54.0
    assert sell_amount == 200 * 54.0

    # Generate Buy Orders
    buy_candidates = pd.DataFrame({
        'instrument': ['000001'],
        'possible_max': [11.0],
        'score': [0.9],
        'current_close': [10.0]
    }).set_index('instrument')
    
    buys = order_generator.generate_buy_orders(buy_candidates, buy_count=1, available_cash=1500, next_trade_date_string="2026-03-02")
    
    assert len(buys) == 1
    assert buys[0]['instrument'] == '000001'
    # Cash/buy_count = 1500. 1500 / 11.0 = 136.36 -> floor to 100 shares
    assert buys[0]['value'] == 100
    assert buys[0]['estimated_amount'] == 100 * 11.0


# ── Supplementary tests for save_orders ──────────────────────────────────

def test_save_orders_dry_run(mock_env, capsys):
    order_gen, _, workspace = mock_env
    sell_orders = [{"instrument": "000001", "datetime": "2026-03-02", "value": 100}]
    buy_orders = [{"instrument": "000002", "datetime": "2026-03-02", "value": 200}]

    sell_file, buy_file = order_gen.save_orders(
        sell_orders, buy_orders,
        next_trade_date_string="2026-03-02",
        output_dir=str(workspace / "output"),
        source_label="ensemble",
        dry_run=True
    )

    assert "sell_suggestion" in sell_file
    assert "buy_suggestion" in buy_file
    # In dry-run mode, files should NOT be written
    assert not os.path.exists(sell_file)
    assert not os.path.exists(buy_file)

    captured = capsys.readouterr()
    assert "DRY-RUN" in captured.out


def test_save_orders_normal(mock_env):
    import os
    order_gen, _, workspace = mock_env
    sell_orders = [{"instrument": "000001", "datetime": "2026-03-02", "value": 100}]
    buy_orders = [{"instrument": "000002", "datetime": "2026-03-02", "value": 200}]

    sell_file, buy_file = order_gen.save_orders(
        sell_orders, buy_orders,
        next_trade_date_string="2026-03-02",
        output_dir=str(workspace / "output"),
        source_label="ensemble",
        dry_run=False
    )

    assert os.path.exists(sell_file)
    assert os.path.exists(buy_file)

    sell_df = pd.read_csv(sell_file)
    buy_df = pd.read_csv(buy_file)
    assert len(sell_df) == 1
    assert len(buy_df) == 1
