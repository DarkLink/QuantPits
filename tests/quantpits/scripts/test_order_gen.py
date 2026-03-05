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

def test_load_configs(mock_env):
    import json
    order_gen, strategy, workspace = mock_env
    
    config_file = workspace / "config" / "prod_config.json"
    cashflow_file = workspace / "config" / "cashflow.json"
    
    with open(config_file, "w") as f:
        json.dump({"market": "csi300"}, f)
        
    with open(cashflow_file, "w") as f:
        json.dump({"cash_flow_today": 123}, f)
        
    config, cf_config = order_gen.load_configs()
    assert config == {"market": "csi300"}
    assert cf_config == {"cash_flow_today": 123}

@patch('qlib.data.D', create=True)
def test_get_anchor_date(mock_D, mock_env):
    order_gen, strategy, _ = mock_env
    mock_D.calendar.return_value = [pd.Timestamp("2020-01-01")]
    res = order_gen.get_anchor_date()
    assert res == "2020-01-01"

def test_load_predictions(mock_env):
    order_gen, strategy, workspace = mock_env
    
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. specify file
    spec_file = pred_dir / "spec.csv"
    pd.DataFrame({"score": [1.0]}).to_csv(spec_file)
    df, desc = order_gen.load_predictions(prediction_file=str(spec_file))
    assert len(df) == 1
    assert "指定文件" in desc
    
    # 2. model name
    model_file = pred_dir / "gru_2020.csv"
    pd.DataFrame({"score": [1.0]}).to_csv(model_file)
    df, desc = order_gen.load_predictions(model_name="gru")
    assert "gru" in desc
    
    # 3. ensemble default
    ens_file = pred_dir / "ensemble_2020.csv"
    pd.DataFrame({"score": [1.0]}).to_csv(ens_file)
    
    ens_dir = workspace / "output" / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    ens_cfg = ens_dir / "ensemble_fusion_config_2020.json"
    import json
    with open(ens_cfg, "w") as f:
        json.dump({"weight_mode": "equal", "models_used": ["gru"]}, f)
        
    df, desc = order_gen.load_predictions()
    assert "Ensemble 融合" in desc
    assert "gru" in desc

def test_load_pred_latest_day(mock_env):
    order_gen, strategy, workspace = mock_env
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"])
    df = pd.DataFrame({"score": [1, 2, 3], "instrument": ["a", "b", "c"], "datetime": dates})
    
    spec_file = str(workspace / "test.csv")
    df.to_csv(spec_file, index=False)
    
    # from csv
    res = order_gen._load_pred_latest_day(spec_file, 'model')
    assert len(res) == 2
    assert set(res.index) == {"b", "c"}
    
    # from df (model_pkl)
    df_multi = df.set_index(["instrument", "datetime"])
    res2 = order_gen._load_pred_latest_day(df_multi, 'model_pkl')
    assert len(res2) == 2

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

def test_generate_model_opinions(mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    focus_instruments = ["A", "B", "C"]
    holding_instruments = ["B"]
    sorted_df = pd.DataFrame({"score": [0.9, 0.8, 0.7]}, index=["A", "B", "C"])
    
    # Setup dummy predictions to trigger multiple source paths
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"score": [0.9, 0.8, 0.7], "instrument": ["A", "B", "C"]}).to_csv(pred_dir / "ensemble_test_2020.csv", index=False)
    pd.DataFrame({"score": [0.9, 0.8, 0.7], "instrument": ["A", "B", "C"]}).to_csv(pred_dir / "gru_2020.csv", index=False)
    
    # Write dummy ensemble config
    with open(order_gen.ENSEMBLE_CONFIG_FILE, "w") as f:
        import json
        json.dump({"combos": {"test": {"models": ["gru"]}}}, f)
    
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

def test_generate_model_opinions_legacy_config(mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    # 1. Legacy config (models in config, not combos)
    with open(order_gen.ENSEMBLE_CONFIG_FILE, "w") as f:
        import json
        json.dump({"models": ["gru"]}, f)
    
    # Mock single model prediction
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"score": [0.9], "instrument": ["A"]}).to_csv(pred_dir / "gru_2020.csv", index=False)
    
    sorted_df = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["A"])
    
    opinions_df, combo_info = order_gen.generate_model_opinions(
        ["A"], [], top_k=1, drop_n=0, buy_suggestion_factor=1,
        sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01"
    )
    
    # It should find model_gru, not combo_legacy (unless we had ensemble_legacy_*.csv)
    assert "model_gru" in opinions_df.columns
    assert list(combo_info.keys()) == ["legacy"]

def test_generate_model_opinions_default_generic_loading(mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    # 2. Default combo loading generic ensemble_*.csv
    with open(order_gen.ENSEMBLE_CONFIG_FILE, "w") as f:
        import json
        json.dump({"combos": {"test": {"models": ["gru"], "default": True}}}, f)
    
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    # Generic file name with date format 2020-01-01
    pd.DataFrame({"score": [0.9], "instrument": ["A"]}).to_csv(pred_dir / "ensemble_2020-01-01.csv", index=False)
    # Also mock the specific group file to ensure generic is NOT picked if specific exists? 
    # Actually L345 says "if files: ... continue", so if ensemble_test_*.csv exists, it skips generic.
    # To test generic, we must NOT have ensemble_test_*.csv
    
    sorted_df = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["A"])
    
    opinions_df, combo_info = order_gen.generate_model_opinions(
        ["A"], [], top_k=1, drop_n=0, buy_suggestion_factor=1,
        sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01"
    )
    
    assert "combo_test" in opinions_df.columns
    # Check that it actually loaded the generic file
    # (Since we didn't provide ensemble_test_*.csv, it should have hit L349-357)

@patch('qlib.workflow.R', create=True)
def test_generate_model_opinions_from_qlib_recorder(mock_R, mock_env, tmp_path):
    order_gen, strategy, workspace = mock_env
    
    # 3. From Qlib Recorder
    with open(order_gen.ENSEMBLE_CONFIG_FILE, "w") as f:
        import json
        json.dump({"models": ["gru"]}, f)
    
    (workspace / "config").mkdir(exist_ok=True)
    train_records_file = workspace / "config" / "latest_train_records.json"
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
            sorted_df=sorted_df, output_dir=str(tmp_path), next_trade_date_string="2020-01-01"
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
@patch('quantpits.scripts.env.safeguard')
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
