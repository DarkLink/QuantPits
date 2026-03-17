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
    
    from quantpits.utils import env, strategy
    from quantpits.scripts import order_gen
    import importlib
    importlib.reload(env)
    importlib.reload(strategy)
    
    # Needs to be set after reload to ensure module uses fake workspace ROOT_DIR
    env.ROOT_DIR = str(workspace)
    order_gen.ROOT_DIR = str(workspace)
    order_gen.ENSEMBLE_CONFIG_FILE = os.path.join(env.ROOT_DIR, "config", "ensemble_fusion_config.json")
    order_gen.CASHFLOW_FILE = os.path.join(env.ROOT_DIR, "config", "cashflow.json")
    
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

def test_load_configs(mock_env):
    import json
    order_gen, strategy, workspace = mock_env
    
    cashflow_file = workspace / "config" / "cashflow.json"
    
    with open(cashflow_file, "w") as f:
        json.dump({"cash_flow_today": 123}, f)
        
    with patch('quantpits.utils.config_loader.load_workspace_config') as mock_load:
        mock_load.return_value = {"market": "csi300"}
        config, cf_config = order_gen.load_configs()
        
    assert config == {"market": "csi300"}
    assert cf_config == {"cash_flow_today": 123}

@patch('qlib.data.D', create=True)
def test_get_anchor_date(mock_D, mock_env):
    order_gen, strategy, _ = mock_env
    mock_D.calendar.return_value = [pd.Timestamp("2020-01-01")]
    res = order_gen.get_anchor_date()
    assert res == "2020-01-01"

@patch('qlib.workflow.R')
def test_load_predictions(mock_R, mock_env):
    order_gen, strategy, workspace = mock_env
    
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    mock_rec = MagicMock()
    mock_df = pd.DataFrame({"score": [1.0]}).set_index(pd.Index(["A"], name="instrument"))
    mock_rec.load_object.return_value = mock_df
    mock_rec.list_metrics.return_value = {}
    mock_rec.info = {"experiment_name": "ex", "id": "rec_id"}
    mock_R.get_recorder.return_value = mock_rec
    
    # 1. model name using latest_train_records.json
    train_file = workspace / "latest_train_records.json"
    with open(train_file, "w") as f:
        import json
        json.dump({"models": {"gru": "rec_123"}, "experiment_name": "ex"}, f)
        
    df, desc = order_gen.load_predictions(model_name="gru")
    assert "gru" in desc
    assert len(df) == 1
    
    # 2. ensemble default using ensemble_records.json
    ens_file = config_dir / "ensemble_records.json"
    with open(ens_file, "w") as f:
        json.dump({
            "combos": {
                "test_combo": {"record_id": "rec_456", "models": ["gru"]}
            }, 
            "default_combo": "test_combo"
        }, f)
        
    df, desc = order_gen.load_predictions()
    assert "Ensemble 融合" in desc or "test_combo" in desc
    assert len(df) == 1

def test_load_pred_latest_day(mock_env):
    order_gen, strategy, workspace = mock_env
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"])
    df = pd.DataFrame({"score": [1, 2, 3], "instrument": ["a", "b", "c"], "datetime": dates})
    
    # from df (model_pkl)
    df_multi = df.set_index(["instrument", "datetime"])
    res2 = order_gen._load_pred_latest_day(df_multi, valid_instruments=['a', 'b'])
    assert len(res2) == 1
    assert set(res2.index) == {"b"}

@patch('qlib.data.D', create=True)
@patch('qlib.data.ops.Feature', create=True)
def test_get_price_data(mock_Feature, mock_D, mock_env):
    order_gen, strategy, _ = mock_env
    mock_D.instruments.return_value = ['000001']
    
    mock_df = pd.DataFrame({
        'Div($close,$factor)': [10.0],
        'Mul(Div($close,$factor),1.1)': [11.0],
        'Mul(Div($close,$factor),0.9)': [9.0]
    })
    mock_D.features.return_value = mock_df
    
    price_df = order_gen.get_price_data('2020-01-01', 'csi300')
    assert list(price_df.columns) == ['current_close', 'possible_max', 'possible_min']
    assert price_df.iloc[0]['current_close'] == 10.0

@patch('qlib.workflow.R')
def test_generate_model_opinions(mock_R, mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    focus_instruments = ["A", "B", "C"]
    holding_instruments = ["B"]
    sorted_df = pd.DataFrame({"score": [0.9, 0.8, 0.7]}, index=["A", "B", "C"])
    
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write dummy ensemble config
    import json
    with open(order_gen.ENSEMBLE_CONFIG_FILE, "w") as f:
        json.dump({"combos": {"test_combo": {"models": ["gru"]}}}, f)
        
    ens_file = config_dir / "ensemble_records.json"
    with open(ens_file, "w") as f:
        json.dump({
            "combos": {
                "test_combo": {"record_id": "rec_ensemble", "models": ["gru"]}
            }, 
            "default_combo": "test_combo"
        }, f)
        
    train_file = workspace / "latest_train_records.json"
    with open(train_file, "w") as f:
        json.dump({"models": {"gru": "rec_gru"}}, f)
        
    mock_rec = MagicMock()
    mock_df = pd.DataFrame({"score": [0.9, 0.8, 0.7]}, index=["A", "B", "C"])
    mock_df.index.name = "instrument"
    mock_rec.load_object.return_value = mock_df
    mock_rec.list_metrics.return_value = {}
    mock_rec.info = {"experiment_name": "ex", "id": "rec_id"}
    mock_R.get_recorder.return_value = mock_rec
    
    opinions_df, combo_info = order_gen.generate_model_opinions(
        focus_instruments, holding_instruments,
        top_k=2, drop_n=1, buy_suggestion_factor=1,
        sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01"
    )
    
    assert opinions_df is not None
    assert len(opinions_df) == 3
    assert "order_basis" in opinions_df.columns
    assert list(opinions_df.loc["B"].values) == ["HOLD (2)", "HOLD (2)", "HOLD (2)"]
    assert list(opinions_df.loc["A"].values) == ["BUY (1)", "BUY (1)", "BUY (1)"]

@patch('qlib.workflow.R')
def test_generate_model_opinions_legacy_config(mock_R, mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    # 1. Legacy config (models in config, not combos)
    import json
    with open(order_gen.ENSEMBLE_CONFIG_FILE, "w") as f:
        json.dump({"models": ["gru"]}, f)
    
    train_file = workspace / "latest_train_records.json"
    with open(train_file, "w") as f:
        json.dump({"models": {"gru": "rec_gru"}}, f)
        
    mock_rec = MagicMock()
    mock_df = pd.DataFrame({"score": [0.9]}, index=["A"])
    mock_df.index.name = "instrument"
    mock_rec.load_object.return_value = mock_df
    mock_rec.list_metrics.return_value = {}
    mock_rec.info = {"experiment_name": "ex", "id": "rec_id"}
    mock_R.get_recorder.return_value = mock_rec
    
    sorted_df = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["A"])
    
    opinions_df, combo_info = order_gen.generate_model_opinions(
        ["A"], [], top_k=1, drop_n=0, buy_suggestion_factor=1,
        sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01"
    )
    
    assert "model_gru" in opinions_df.columns
    assert list(combo_info.keys()) == ["legacy"]

@patch('qlib.workflow.R')
def test_generate_model_opinions_default_generic_loading(mock_R, mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    import json
    # Mock ENSEMBLE_CONFIG_FILE content
    ensemble_cfg = workspace / "config" / "ensemble_fusion_config.json"
    with open(ensemble_cfg, "w") as f:
        json.dump({"combos": {"test_combo": {"models": ["gru"]}}}, f)
    
    ens_file = workspace / "config" / "ensemble_records.json"
    with open(ens_file, "w") as f:
        json.dump({
            "combos": {
                "test_combo": {"record_id": "rec_ensemble", "models": ["gru"]}
            }, 
            "default_combo": "test_combo"
        }, f)
        
    mock_rec = MagicMock()
    mock_df = pd.DataFrame({"score": [0.9], "instrument": ["A"]}).set_index("instrument")
    mock_rec.load_object.return_value = mock_df
    mock_rec.list_metrics.return_value = {}
    mock_rec.info = {"experiment_name": "ex", "id": "rec_id"}
    mock_R.get_recorder.return_value = mock_rec
    
    sorted_df = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["A"])
    
    opinions_df, combo_info = order_gen.generate_model_opinions(
        ["A"], [], top_k=1, drop_n=0, buy_suggestion_factor=1,
        sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01"
    )
    
    assert "combo_test_combo" in opinions_df.columns

@patch('qlib.workflow.R', create=True)
def test_generate_model_opinions_from_qlib_recorder(mock_R, mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    # 3. From Qlib Recorder
    with open(order_gen.ENSEMBLE_CONFIG_FILE, "w") as f:
        import json
        json.dump({"models": ["gru"]}, f)
    
    (workspace / "config").mkdir(exist_ok=True)
    train_records_file = workspace / "latest_train_records.json"
    with open(train_records_file, "w") as f:
        json.dump({"models": {"gru": "rec_id"}, "experiment_name": "exp"}, f)
    
    # Mock R behavior
    mock_recorder = MagicMock()
    mock_R.get_recorder.return_value = mock_recorder
    mock_recorder.load_object.return_value = pd.DataFrame({"score": [0.85], "instrument": ["A"]}).set_index(["instrument"])
    
    # Ensure NO CSV file exists for gru
    # (mock_env might have created some if we are not careful, but this is a fresh tmp_path Workspace)
    
    sorted_df = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["A"])
    
    # Patch ROOT_DIR to find our mock config
    with patch('quantpits.scripts.order_gen.ROOT_DIR', str(workspace)):
        opinions_df, combo_info = order_gen.generate_model_opinions(
            ["A"], [], top_k=1, drop_n=0, buy_suggestion_factor=1,
            sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01",
            record_file=str(train_records_file)
        )
    
    assert "model_gru" in opinions_df.columns

def test_generate_model_opinions_no_sources(mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    # 4. No sources found
    # Delete the ensemble config if it exists
    if os.path.exists(order_gen.ENSEMBLE_CONFIG_FILE):
        os.remove(order_gen.ENSEMBLE_CONFIG_FILE)
    
    sorted_df = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["A"])
    
    opinions_df, combo_info = order_gen.generate_model_opinions(
        ["A"], [], top_k=1, drop_n=0, buy_suggestion_factor=1,
        sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01"
    )
    
    assert opinions_df is None

@patch('quantpits.scripts.order_gen.init_qlib')
@patch('quantpits.scripts.order_gen.get_anchor_date')
@patch('quantpits.scripts.order_gen.load_configs')
@patch('quantpits.scripts.order_gen.load_predictions')
@patch('quantpits.scripts.order_gen.get_price_data')
@patch('quantpits.utils.env.safeguard')
@patch('qlib.data.D', create=True)
def test_main_dry_run_full(mock_D, mock_safeguard, mock_price, mock_pred, mock_configs, mock_anchor, mock_init, mock_env):
    order_gen, strategy, workspace = mock_env
    
    mock_anchor.return_value = "2020-01-01"
    mock_configs.return_value = (
        {"market": "csi300", "current_cash": 1000000, "current_holding": []},
        {"cash_flow_today": 0}
    )
    
    idx = pd.MultiIndex.from_tuples([("000001", pd.to_datetime("2020-01-01"))], names=["instrument", "datetime"])
    mock_pred.return_value = (pd.DataFrame({"score": [0.9]}, index=idx), "Mock Source")
    
    idx_p = pd.Index(["000001"], name="instrument")
    mock_price.return_value = pd.DataFrame({
        "current_close": [10.0], "possible_max": [11.0], "possible_min": [9.0]
    }, index=idx_p)
    
    mock_D.calendar.return_value = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--dry-run', '--verbose']):
        order_gen.main()
    
    # Check if a few key print messages were hit (via capsys if we had it, but mostly we want coverage)
    # The test should pass if it runs through the whole main()
