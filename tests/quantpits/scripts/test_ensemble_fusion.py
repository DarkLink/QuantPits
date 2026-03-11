import os
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from datetime import datetime

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "output").mkdir()
    
    import sys
    import importlib
    script_dir = os.path.join(os.getcwd(), "quantpits/scripts")
    if script_dir not in sys.path:
        sys.path.append(script_dir)
        
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    
    # Reload to pick up new env
    for mod_name in ['env', 'quantpits.scripts.env', 'ensemble_fusion', 'quantpits.scripts.ensemble_fusion']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
            
    from quantpits.scripts import ensemble_fusion as ef
    
    # Global mocks for side-effect-heavy functions
    monkeypatch.setattr('quantpits.scripts.env.safeguard', lambda x: None)
    monkeypatch.setattr('quantpits.scripts.env.init_qlib', lambda: None)
    import qlib.data
    mock_D = MagicMock()
    mock_D.calendar.return_value = pd.date_range("2020-01-01", periods=10, freq="D")
    monkeypatch.setattr(qlib.data, 'D', mock_D)
    
    yield ef, workspace

def test_parse_ensemble_config():
    import quantpits.scripts.ensemble_fusion as ef
    
    legacy_config = {
        "models": ["A", "B"],
        "ensemble_method": "manual",
        "manual_weights": {"A": 0.6, "B": 0.4},
        "min_model_ic": 0.05
    }
    
    combos, global_cfg = ef.parse_ensemble_config(legacy_config)
    
    assert "legacy" in combos
    assert combos["legacy"]["method"] == "manual"
    assert combos["legacy"]["models"] == ["A", "B"]
    assert combos["legacy"]["manual_weights"]["A"] == 0.6
    assert global_cfg["min_model_ic"] == 0.05
    
    new_config = {
        "min_model_ic": 0.02,
        "combos": {
            "combo_X": {"models": ["C", "D"], "method": "equal", "default": True}
        }
    }
    
    combos2, global_cfg2 = ef.parse_ensemble_config(new_config)
    assert "combo_X" in combos2
    assert combos2["combo_X"]["default"] is True
    assert global_cfg2["min_model_ic"] == 0.02

def test_get_default_combo():
    import quantpits.scripts.ensemble_fusion as ef
    
    combos = {
        "combo1": {"default": False, "models": ["A"]},
        "combo2": {"default": True, "models": ["B"]}
    }
    name, cfg = ef.get_default_combo(combos)
    assert name == "combo2"
    assert cfg["models"] == ["B"]
    
    combos_none = {
        "combo1": {"models": ["A"]},
        "combo2": {"models": ["B"]}
    }
    name2, cfg2 = ef.get_default_combo(combos_none)
    assert name2 == "combo1"

def test_zscore_norm():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"])
    instruments = ["A", "B", "A", "B"]
    idx = pd.MultiIndex.from_arrays([dates, instruments], names=['datetime', 'instrument'])
    
    series = pd.Series([10.0, 20.0, 5.0, 5.0], index=idx)
    
    norm = ef.zscore_norm(series)
    
    # 2020-01-01 -> mean=15, std=7.07
    # 2020-01-02 -> mean=5, std=0 (so it should return 0)
    assert np.isclose(norm.loc[("2020-01-01", "A")], -0.7071, atol=1e-3)
    assert np.isclose(norm.loc[("2020-01-01", "B")], 0.7071, atol=1e-3)
    
    assert norm.loc[("2020-01-02", "A")] == 0.0
    assert norm.loc[("2020-01-02", "B")] == 0.0

@patch('qlib.workflow.R')
def test_load_selected_predictions(mock_R, mock_env):
    ef, workspace = mock_env
    
    train_records = {
        "experiment_name": "Exp1",
        "models": {"ModelA": "rid_a", "ModelB": "rid_b"}
    }
    selected_models = ["ModelA", "ModelB", "ModelC"] # ModelC omitted intentionally
    
    mock_recorder_a = MagicMock()
    # Need to return a pandas Series with MultiIndex for prediction
    idx_a = pd.MultiIndex.from_product([[pd.Timestamp("2026-03-01")], ["Inst1", "Inst2"]], names=["datetime", "instrument"])
    mock_recorder_a.load_object.return_value = pd.Series([0.1, 0.2], index=idx_a)
    mock_recorder_a.list_metrics.return_value = {"ICIR": 1.5, "IC": 0.02}
    
    mock_recorder_b = MagicMock()
    idx_b = pd.MultiIndex.from_product([[pd.Timestamp("2026-03-01")], ["Inst1", "Inst2"]], names=["datetime", "instrument"])
    mock_recorder_b.load_object.return_value = pd.DataFrame({"score": [0.4, 0.5]}, index=idx_b)
    mock_recorder_b.list_metrics.return_value = {"ICIR": 0.8}
    
    def r_getter(recorder_id, experiment_name):
        if recorder_id == "rid_a": return mock_recorder_a
        return mock_recorder_b
        
    mock_R.get_recorder.side_effect = r_getter
    
    norm_df, metrics, loaded = ef.load_selected_predictions(train_records, selected_models)
    
    assert loaded == ["ModelA", "ModelB"]
    assert len(norm_df.columns) == 2
    assert "ModelA" in norm_df.columns
    assert "ModelB" in norm_df.columns
    assert metrics["ModelA"] == 1.5
    assert metrics["ModelB"] == 0.8
    # Z-scores computed
    assert np.isclose(norm_df.loc[("2026-03-01", "Inst1"), "ModelA"], -0.7071, atol=1e-3)

def test_calculate_weights_equal():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0]}, index=idx)
    
    model_metrics = {"M1": 0.5, "M2": 0.3}
    model_config = {}
    ensemble_config = {}
    
    final, static, is_dyn = ef.calculate_weights(norm_df, model_metrics, 'equal', model_config, ensemble_config)
    assert not is_dyn
    assert final is None
    assert static["M1"] == 0.5
    assert static["M2"] == 0.5

def test_calculate_weights_icir():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0], "M3": [3.0]}, index=idx)
    
    model_metrics = {"M1": 0.6, "M2": 0.4, "M3": -0.1} # M3 should be excluded
    model_config = {}
    ensemble_config = {"min_model_ic": 0.05}
    
    _, static, is_dyn = ef.calculate_weights(norm_df, model_metrics, 'icir_weighted', model_config, ensemble_config)
    
    assert not is_dyn
    assert static["M1"] == 0.6
    assert static["M2"] == 0.4
    assert static.get("M3", 0) == 0

def test_calculate_weights_manual():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0]}, index=idx)
    
    _, static, is_dyn = ef.calculate_weights(norm_df, {}, 'manual', {}, {}, manual_weights_str="M1:0.7,M2:0.3")
    assert not is_dyn
    assert static["M1"] == 0.7
    assert static["M2"] == 0.3

def test_generate_ensemble_signal():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [10.0], "M2": [20.0]}, index=idx)
    
    static_w = {"M1": 0.4, "M2": 0.6}
    
    final = ef.generate_ensemble_signal(norm_df, None, static_w, False)
    
    assert final.loc[("2020-01-01", "A")] == (10.0 * 0.4 + 20.0 * 0.6)

@patch('quantpits.scripts.strategy.load_strategy_config')
@patch('quantpits.scripts.strategy.get_backtest_config')
@patch('quantpits.scripts.strategy.create_backtest_strategy')
@patch('qlib.backtest.executor.SimulatorExecutor')
@patch('qlib.backtest.backtest')
def test_run_backtest(mock_bt, mock_executor, mock_create_strat, mock_get_bt_cfg, mock_load_st_cfg, mock_env):
    ef, workspace = mock_env
    
    # Mock strategy dependencies
    mock_load_st_cfg.return_value = {}
    mock_get_bt_cfg.return_value = {"account": 100000.0, "exchange_kwargs": {}}
    mock_create_strat.return_value = MagicMock()
    mock_executor.return_value = MagicMock()
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    idx = pd.MultiIndex.from_product([dates, ["InstA"]], names=['datetime', 'instrument'])
    final_score = pd.Series([1.0, 2.0, 3.0], index=idx)
    
    mock_report = pd.DataFrame({
        "account": [100000, 105000, 110000],
        "return": [0.0, 0.05, 0.047],
        "bench": [1.0, 1.05, 1.08]
    }, index=dates)
    
    mock_bt.return_value = (mock_report, None)
    
    report_df, executor = ef.run_backtest(final_score, top_k=50, drop_n=0, benchmark="SH000300", freq="day")
    
    assert report_df is not None
    assert executor is not None
    assert len(report_df) == 3
    assert "nav" in report_df.columns
    assert report_df.iloc[-1]["nav"] == 110000

def test_save_predictions(mock_env, tmp_path):
    ef, workspace = mock_env
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_product([dates, ["InstA"]], names=['datetime', 'instrument'])
    final_score = pd.Series([1.0], index=idx)
    
    # Instance-level mock to avoid global pandas locks
    mock_df = MagicMock()
    final_score.to_frame = MagicMock(return_value=mock_df)
    
    out_dir = tmp_path / "output" / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with patch('quantpits.scripts.ensemble_fusion.os.makedirs') as mock_makedir:
        pred_file = ef.save_predictions(
            final_score, 
            "2020-01-01", 
            "exp1", 
            "equal", 
            ["M1", "M2"], 
            {"M1": 0.5, "M2": 0.6}, 
            {"M1": 0.5, "M2": 0.5}, 
            False, 
            str(out_dir)
        )
    
    assert "ensemble_2020-01-01.csv" in pred_file
    assert mock_makedir.called
    assert mock_df.to_csv.called
    
    config_file = out_dir / "ensemble_fusion_config_2020-01-01.json"
    assert config_file.exists()

@patch("builtins.open")
@patch("os.path.exists")
def test_load_config(mock_exists, mock_open):
    import quantpits.scripts.ensemble_fusion as ef
    
    mock_exists.return_value = True
    
    with patch("config_loader.load_workspace_config") as mock_load_workspace:
        mock_load_workspace.return_value = {"model": "config"}
        
        with patch("quantpits.scripts.ensemble_fusion.json.load") as mock_json_load:
            mock_json_load.side_effect = [
                {"train": "record"},
                {"ensemble": "config"}
            ]
            
            tr, mc, ec = ef.load_config("dummy.json")
            
            assert tr == {"train": "record"}
            assert mc == {"model": "config"}
            assert ec == {"ensemble": "config"}
            assert mock_json_load.call_count == 2
            mock_load_workspace.assert_called_once()

def test_filter_norm_df_by_args():
    import quantpits.scripts.ensemble_fusion as ef
    from types import SimpleNamespace
    
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2021-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B", "A", "B"]], names=["datetime", "instrument"])
    df = pd.DataFrame({"M1": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    
    args = SimpleNamespace(start_date="2020-02-01", end_date="2020-03-01")
    filtered = ef.filter_norm_df_by_args(df, args)
    assert len(filtered) == 2
    
    args_last_months = SimpleNamespace(start_date=None, end_date=None, only_last_years=0, only_last_months=11)
    filtered2 = ef.filter_norm_df_by_args(df, args_last_months)
    assert len(filtered2) == 2
    assert filtered2.index.get_level_values("datetime").min() == pd.Timestamp("2020-03-01")

def test_correlation_analysis(tmp_path):
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
    df = pd.DataFrame({"M1": [1.0, 2.0], "M2": [2.0, 4.0]}, index=idx)
    
    out_dir = tmp_path / "corr"
    corr = ef.correlation_analysis(df, str(out_dir), "2020-01-01", "test_combo")
    
    assert corr.loc["M1", "M2"] == 1.0
    assert (out_dir / "correlation_matrix_test_combo_2020-01-01.csv").exists()

@patch('qlib.data.D', create=True)
def test_calculate_weights_dynamic(mock_D):
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.date_range("2020-01-01", periods=65, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    
    norm_df = pd.DataFrame({"M1": np.random.randn(130), "M2": np.random.randn(130)}, index=idx)
    
    label_df = pd.DataFrame({"label": np.random.randn(130)}, index=idx)
    mock_D.features.return_value = label_df
    
    model_config = {"TopK": 1}
    ensemble_config = {}
    
    final, static, is_dyn = ef.calculate_weights(norm_df, {}, "dynamic", model_config, ensemble_config)
    
    assert is_dyn is True
    assert static is None
    assert final.shape == (65, 2)

def test_extract_report_df():
    import quantpits.scripts.ensemble_fusion as ef
    df = pd.DataFrame({'a': [1]})
    
    assert ef.extract_report_df({'m': df}).equals(df)
    assert ef.extract_report_df((df, 'other')).equals(df)
    assert ef.extract_report_df(((df, 'x'), 'y')).equals(df)
    assert ef.extract_report_df(df).equals(df)

@patch('qlib.contrib.evaluate.risk_analysis')
def test_calculate_safe_risk(mock_risk):
    import quantpits.scripts.ensemble_fusion as ef
    
    mock_risk.return_value = pd.DataFrame({'risk': [0.1, 0.2]}, index=['a', 'b'])
    res = ef.calculate_safe_risk(pd.Series([0.01]), 'day')
    assert res == {'a': 0.1, 'b': 0.2}
    
    mock_risk.return_value = pd.Series([0.3, 0.4], index=['c', 'd'])
    res2 = ef.calculate_safe_risk(pd.DataFrame({'ret': [0.01]}), 'week')
    assert res2 == {'c': 0.3, 'd': 0.4}
    
    mock_risk.side_effect = Exception("error")
    res3 = ef.calculate_safe_risk(pd.Series([0.01]), 'day')
    assert res3 == {}

@patch('strategy.get_backtest_config')
@patch('strategy.load_strategy_config')
def test_compare_combos(mock_load_st, mock_get_bt, tmp_path):
    import quantpits.scripts.ensemble_fusion as ef
    
    mock_load_st.return_value = {}
    mock_get_bt.return_value = {'account': 100000.0}
    
    report_df = pd.DataFrame({
        'account': [100000.0, 105000.0],
        'return': [0.0, 0.05],
        'bench': [0.0, 0.02]
    }, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    
    combo_results = [{
        'name': 'combo1',
        'models': ['M1', 'M2'],
        'method': 'equal',
        'is_default': True,
        'report_df': report_df
    }]
    
    out_dir = tmp_path / "combos"
    comp_df = ef.compare_combos(combo_results, "2020-01-01", str(out_dir), "day")
    
    assert len(comp_df) == 1
    assert comp_df.iloc[0]['combo'] == 'combo1'
    assert comp_df.iloc[0]['total_return'] == 5.0
    assert (out_dir / "combo_comparison_2020-01-01.csv").exists()
    assert (out_dir / "combo_comparison_2020-01-01.png").exists()

@patch('qlib.workflow.R')
def test_risk_analysis_and_leaderboard(mock_R, mock_env, tmp_path):
    ef, workspace = mock_env
    out_dir = tmp_path / "output"
    
    report_df = pd.DataFrame({
        'account': [100.0, 110.0],
        'bench': [0.0, 0.05],
        'return': [0.0, 0.1]
    }, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    
    train_records = {"experiment_name": "Exp1", "models": {"M1": "rid1"}}
    
    mock_recorder = MagicMock()
    mock_recorder.load_object.return_value = report_df
    mock_R.get_recorder.return_value = mock_recorder
    
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A"), (pd.Timestamp("2020-01-02"), "A")], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"M1": [0.5, 0.6]}, index=idx)

    with patch('quantpits.scripts.ensemble_fusion.calculate_safe_risk') as mock_risk:
        mock_risk.return_value = {"annualized_return": 0.5}
        reports, lb = ef.risk_analysis_and_leaderboard(
            report_df, norm_df, train_records, ["M1"], "day", str(out_dir), "2020-01-01"
        )
        
    assert "Ensemble" in reports
    assert "M1" in reports
    assert len(lb) == 2
    assert (out_dir / "leaderboard_2020-01-01.csv").exists()

def test_generate_charts(mock_env, tmp_path):
    ef, workspace = mock_env
    out_dir = tmp_path / "charts"
    
    report_df = pd.DataFrame({'return': [0.01], 'bench': [0.005]}, index=[pd.Timestamp("2020-01-01")])
    all_reports = {"Ensemble": report_df}
    
    final_weights = pd.DataFrame({"M1": [0.6], "M2": [0.4]}, index=[pd.Timestamp("2020-01-01")])
    
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        ef.generate_charts(all_reports, report_df, final_weights, True, "day", "2020-01-01", str(out_dir))
    
    assert mock_savefig.called

@patch('quantpits.scripts.ensemble_fusion.init_qlib', create=True)
@patch('quantpits.scripts.ensemble_fusion.load_config')
@patch('quantpits.scripts.ensemble_fusion.load_selected_predictions')
@patch('quantpits.scripts.ensemble_fusion.run_single_combo')
def test_main_from_config_all(mock_run_combo, mock_load_pred, mock_load_cfg, mock_init, mock_env):
    ef, workspace = mock_env
    mock_load_cfg.return_value = (
        {"experiment_name": "Exp1", "models": {"m1": "rid1", "m2": "rid2"}, "anchor_date": "2020-01-01"},
        {"freq": "day"},
        {"combos": {"c1": {"models": ["m1"]}, "c2": {"models": ["m2"]}}}
    )
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5], "m2": [0.6]}, index=idx)
    mock_load_pred.return_value = (norm_df, {}, ["m1", "m2"])
    mock_run_combo.return_value = {'name': 'c1', 'models': ['m1'], 'method': 'equal', 'is_default': False, 'pred_file': 'x.csv', 'report_df': None}

    import sys
    with patch.object(sys, 'argv', ['script.py', '--from-config-all']):
        ef.main()
    assert mock_run_combo.call_count == 2

@patch('quantpits.scripts.ensemble_fusion.init_qlib', create=True)
@patch('quantpits.scripts.ensemble_fusion.load_config')
@patch('quantpits.scripts.ensemble_fusion.load_selected_predictions')
@patch('quantpits.scripts.ensemble_fusion.run_single_combo')
def test_main_combo_specified(mock_run_combo, mock_load_pred, mock_load_cfg, mock_init, mock_env):
    ef, workspace = mock_env
    mock_load_cfg.return_value = (
        {"experiment_name": "Exp1", "models": {"m1": "rid1"}, "anchor_date": "2020-01-01"},
        {"freq": "day"},
        {"combos": {"c1": {"models": ["m1"]}}}
    )
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
    mock_load_pred.return_value = (norm_df, {}, ["m1"])
    mock_run_combo.return_value = {'name': 'c1', 'models': ['m1'], 'method': 'equal', 'is_default': False, 'pred_file': 'x.csv', 'report_df': None}

    import sys
    with patch.object(sys, 'argv', ['script.py', '--combo', 'c1']):
        ef.main()
    mock_run_combo.assert_called_once()

@patch('quantpits.scripts.ensemble_fusion.correlation_analysis')
@patch('quantpits.scripts.ensemble_fusion.calculate_weights')
@patch('quantpits.scripts.ensemble_fusion.generate_ensemble_signal')
@patch('quantpits.scripts.ensemble_fusion.save_predictions')
@patch('quantpits.scripts.ensemble_fusion.run_backtest')
@patch('quantpits.scripts.ensemble_fusion.risk_analysis_and_leaderboard')
@patch('quantpits.scripts.ensemble_fusion.generate_charts')
def test_run_single_combo_pipeline(mock_charts, mock_risk, mock_bt, mock_save, mock_signal, mock_weights, mock_corr, mock_env):
    ef, workspace = mock_env
    
    norm_df = pd.DataFrame({"m1": [0.5]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")]))
    mock_weights.return_value = (None, {"m1": 1.0}, False)
    mock_signal.return_value = pd.Series([0.5])
    mock_save.return_value = "pred.csv"
    mock_bt.return_value = (pd.DataFrame({"account": [100]}), MagicMock())
    mock_risk.return_value = ({"Ensemble": pd.DataFrame()}, pd.DataFrame())
    
    args = MagicMock()
    args.output_dir = "out"
    args.no_backtest = False
    args.no_charts = False
    args.freq = "day"
    args.verbose_backtest = False
    args.detailed_analysis = False
    
    res = ef.run_single_combo(
        "c1", ["m1"], "equal", None, norm_df, {"m1": 0.1}, ["m1"],
        {"experiment_name": "exp"}, {"TopK": 20}, {}, "2020-01-01", "exp", args
    )
    
    assert res['name'] == "c1"
    assert mock_corr.called
    assert mock_weights.called
    assert mock_signal.called
    assert mock_save.called
    assert mock_bt.called
    assert mock_risk.called
    assert mock_charts.called

@patch('quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer')
def test_run_detailed_backtest_analysis(mock_pa, tmp_path):
    from quantpits.scripts import ensemble_fusion as ef
    
    mock_executor = MagicMock()
    mock_ta = MagicMock()
    mock_executor.trade_account = mock_ta
    
    # Mock portfolio metrics
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    pm_df = pd.DataFrame({
        'account': [100.0, 110.0],
        'bench': [0.0, 0.05]
    }, index=idx)
    mock_ta.get_portfolio_metrics.return_value = (pm_df,)
    
    # Mock positions
    mock_pos_obj = MagicMock()
    # Fixed count and price
    mock_unit = MagicMock()
    mock_unit.count = 10
    mock_unit.price = 5.0
    mock_pos_obj.position = {"M1": mock_unit}
    mock_ta.get_hist_positions.return_value = {pd.Timestamp("2020-01-01"): mock_pos_obj}
    
    # Mock PA results
    mock_pa_inst = mock_pa.return_value
    mock_pa_inst.calculate_traditional_metrics.return_value = {"CAGR": 0.5}
    mock_pa_inst.calculate_factor_exposure.return_value = {"Beta_Market": 1.1}
    mock_pa_inst.calculate_style_exposures.return_value = {"Barra_Size_Exp": 0.1}
    mock_pa_inst.calculate_holding_metrics.return_value = {
        "Avg_Daily_Holdings_Count": 5,
        "Avg_Top1_Concentration": 0.1,
        "Avg_Floating_Return": 0.01,
        "Daily_Holding_Win_Rate": 0.6
    }
    
    out_dir = tmp_path / "detailed"
    out_dir.mkdir()
    
    report_file = ef.run_detailed_backtest_analysis(
        mock_executor, "test_combo", "2020-01-01", str(out_dir), "day"
    )
    
    assert report_file is not None
    assert os.path.exists(report_file)
    with open(report_file, 'r') as f:
        content = f.read()
        assert "CAGR" in content
        assert "Beta (Market)" in content
        assert "Performance Attribution" in content
        assert "Beta Return" in content
        assert "Style Alpha" in content
        assert "Idiosyncratic Alpha" in content
        assert "### Holding Analytics" in content
        assert "Avg Floating Return" in content
        assert "Backtest Period" in content
        assert "Benchmark**: SH000300" in content


# --- Stage 0: Initialization & Config ---

def test_init_qlib(mock_env):
    ef, _ = mock_env
    # Line 83: ef.init_qlib() calls env.init_qlib()
    ef.init_qlib()

def test_load_config_no_file(mock_env):
    ef, _ = mock_env
    # Line 93: record_file does not exist
    with patch("config_loader.load_workspace_config") as mock_load_ws:
        mock_load_ws.return_value = {}
        tr, mc, ec = ef.load_config("non_existent_records.json")
        assert tr == {"models": {}, "experiment_name": "unknown"}

def test_parse_ensemble_config_empty(mock_env):
    ef, _ = mock_env
    # Line 140: empty ensemble_config
    combos, global_cfg = ef.parse_ensemble_config({})
    assert combos == {}
    assert global_cfg == {}

def test_get_default_combo_none(mock_env):
    ef, _ = mock_env
    # Line 152: combos is empty
    name, cfg = ef.get_default_combo({})
    assert name is None
    assert cfg is None

# --- Stage 1: Load Predictions ---

@patch('qlib.workflow.R')
def test_load_selected_predictions_failure_exception(mock_R, mock_env):
    ef, _ = mock_env
    # Lines 224-225: recorder.load_object failure
    train_records = {"experiment_name": "Exp1", "models": {"m1": "rid1"}}
    mock_recorder = MagicMock()
    mock_recorder.load_object.side_effect = Exception("Load fail")
    mock_R.get_recorder.return_value = mock_recorder
    
    # Line 230: raise ValueError if no models loaded
    with pytest.raises(ValueError, match="未加载到任何预测数据"):
        ef.load_selected_predictions(train_records, ["m1"])

def test_filter_norm_df_by_args_last_years_months(mock_env):
    ef, _ = mock_env
    # Lines 257, 259: only_last_years / only_last_months
    dates = pd.date_range("2010-01-01", "2020-01-01", freq="YS")
    idx = pd.MultiIndex.from_arrays([dates, ["A"]*len(dates)], names=["datetime", "instrument"])
    df = pd.DataFrame({"M1": range(len(dates))}, index=idx)
    
    args = SimpleNamespace(start_date=None, end_date=None, only_last_years=1, only_last_months=0)
    filtered = ef.filter_norm_df_by_args(df, args)
    assert len(filtered.index.get_level_values("datetime").unique()) == 1
    
    args = SimpleNamespace(start_date=None, end_date=None, only_last_years=0, only_last_months=24)
    filtered = ef.filter_norm_df_by_args(df, args)
    assert len(filtered.index.get_level_values("datetime").unique()) == 2

def test_filter_norm_df_by_args_empty_result(mock_env):
    ef, _ = mock_env
    # Line 276: filtered_df is empty
    dates = pd.date_range("2010-01-01", "2010-01-05", freq="D")
    idx = pd.MultiIndex.from_arrays([dates, ["A"]*len(dates)], names=["datetime", "instrument"])
    df = pd.DataFrame({"M1": range(len(dates))}, index=idx)
    
    args = SimpleNamespace(start_date="2020-01-01", end_date="2020-01-05")
    filtered = ef.filter_norm_df_by_args(df, args)
    assert filtered.empty

# --- Stage 3: Weight Calculation ---

@patch('qlib.data.D')
def test_calculate_weights_dynamic_missing_dates(mock_D, mock_env):
    ef, _ = mock_env
    # Lines 370-372: date not in eval_df
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"M1": [1.0]*5}, index=idx)
    
    # Mock D.features to return data for only one date
    label_df = pd.DataFrame({"label": [0.1]}, index=idx[:1])
    mock_D.features.return_value = label_df
    
    # This should trigger lines 370-372 for other dates
    with patch('builtins.print'):
        ef.calculate_weights(norm_df, {}, "dynamic", {"TopK": 1}, {})

def test_calculate_weights_icir_weighted_invalid(mock_env):
    ef, _ = mock_env
    # Lines 404-405: no valid ICIR
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")])
    norm_df = pd.DataFrame({"M1": [0.5]}, index=idx)
    model_metrics = {"M1": 0.001} # < min_ic=0.01
    
    with patch('builtins.print'):
        res, static, is_dyn = ef.calculate_weights(norm_df, model_metrics, "icir_weighted", {}, {"min_model_ic": 0.01})
        assert static["M1"] == 1.0

def test_calculate_weights_manual_config(mock_env):
    ef, _ = mock_env
    # Line 425: manual_weights in ensemble_config
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")])
    norm_df = pd.DataFrame({"M1": [0.5]}, index=idx)
    ensemble_cfg = {"manual_weights": {"M1": 0.8}}
    
    with patch('builtins.print'):
        res, static, is_dyn = ef.calculate_weights(norm_df, {}, "manual", {}, ensemble_cfg)
        assert static["M1"] == 1.0

def test_calculate_weights_manual_sum_zero(mock_env):
    ef, _ = mock_env
    # Lines 429-430: fatal manual weights (sum to 0)
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")])
    norm_df = pd.DataFrame({"M1": [0.5]}, index=idx)
    ensemble_cfg = {"manual_weights": {"M1": 0.0}}
    
    with patch('builtins.print'):
        res, static, is_dyn = ef.calculate_weights(norm_df, {}, "manual", {}, ensemble_cfg)
        assert static["M1"] == 1.0

# --- Stage 4: Signal Fusion ---

def test_generate_ensemble_signal_dynamic(mock_env):
    ef, _ = mock_env
    # Lines 465-468: is_dynamic application
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"M1": [1.0, 2.0]}, index=idx)
    final_weights = pd.DataFrame({"M1": [0.5, 0.5]}, index=dates)
    
    signal = ef.generate_ensemble_signal(norm_df, final_weights, None, True)
    assert signal.iloc[0] == 0.5
    assert signal.iloc[1] == 1.0

def test_generate_ensemble_signal_zero_std(mock_env, capsys):
    ef, _ = mock_env
    # Line 482: final_score.std() == 0
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01", "2020-01-02"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"M1": [1.0, 1.0]}, index=idx)
    static_w = {"M1": 1.0}
    
    ef.generate_ensemble_signal(norm_df, None, static_w, False)
    captured = capsys.readouterr()
    assert "加权可能失败" in captured.out

# --- Stage 5: Save Predictions ---

def test_save_predictions_combo_and_default(mock_env, tmp_path):
    ef, workspace = mock_env
    # Lines 508, 518-520, 533-534
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")])
    final_score = pd.Series([1.0], index=idx)
    
    os.makedirs("output/predictions", exist_ok=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    ef.save_predictions(
        final_score, "2020-01-01", "exp1", "equal", ["M1"], 
        {"M1": 0.5}, {"M1": 1.0}, False, str(out_dir), 
        combo_name="my_combo", is_default=True
    )
    
    assert os.path.exists("output/predictions/ensemble_my_combo_2020-01-01.csv")
    assert os.path.exists("output/predictions/ensemble_2020-01-01.csv")
    
    with open(out_dir / "ensemble_fusion_config_my_combo_2020-01-01.json") as f:
        cfg = json.load(f)
        assert cfg["combo_name"] == "my_combo"

# --- Stage 6: Backtest & Analysis ---

def test_extract_report_df_default(mock_env):
    ef, _ = mock_env
    # Line 561: default return
    assert ef.extract_report_df(None) is None
    assert ef.extract_report_df("string") == "string"

@patch('qlib.backtest.backtest')
@patch('qlib.backtest.executor.SimulatorExecutor')
@patch('strategy.create_backtest_strategy')
@patch('strategy.load_strategy_config')
@patch('strategy.get_backtest_config')
def test_run_backtest_non_datetime_idx(mock_get_bt, mock_load_st, mock_strat, mock_exec, mock_bt, mock_env):
    ef, _ = mock_env
    # Line 616: non datetime index
    dates = ["2020-01-01", "2020-01-02"]
    idx = pd.MultiIndex.from_product([pd.to_datetime(dates), ["A"]], names=["datetime", "instrument"])
    final_score = pd.Series([1.0, 1.0], index=idx)
    
    report_df = pd.DataFrame({
        "account": [100.0, 101.0],
        "bench": [0.0, 0.01]
    }, index=dates)
    
    mock_bt.return_value = (report_df, None)
    mock_load_st.return_value = {}
    mock_get_bt.return_value = {"account": 100.0, "exchange_kwargs": {}}
    
    with patch('qlib.data.D.calendar', return_value=pd.to_datetime(dates)):
        with patch('quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer') as mock_pa:
            pa_inst = MagicMock()
            pa_inst.calculate_traditional_metrics.return_value = {"Calmar": 2.0}
            mock_pa.return_value = pa_inst
            
            with patch('builtins.print') as mock_p:
                ef.run_backtest(final_score, 1, 0, "SH000300", "day")
                assert any("Calmar Ratio" in str(c) for c in mock_p.call_args_list)

def test_run_detailed_backtest_analysis_metrics_calc(mock_env, tmp_path):
    ef, _ = mock_env
    # Lines 862-864: style_ret calculation
    executor = MagicMock()
    ta = MagicMock()
    executor.trade_account = ta
    
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    report_df = pd.DataFrame({"account": [100.0, 105.0], "bench": [0.0, 0.01]}, index=dates)
    ta.get_portfolio_metrics.return_value = (report_df,)
    
    pos_obj = MagicMock()
    pos_obj.position = {"CASH": 100.0}
    ta.get_hist_positions.return_value = {d: pos_obj for d in dates}
    
    out_dir = tmp_path / "detailed_metrics"
    out_dir.mkdir()
    
    with patch('quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer') as mock_pa:
        pa_inst = MagicMock()
        pa_inst.calculate_traditional_metrics.return_value = {"CAGR": 0.1, "Benchmark_CAGR": 0.05}
        pa_inst.calculate_factor_exposure.return_value = {"Beta_Market": 1.0}
        pa_inst.calculate_style_exposures.return_value = {
            "Barra_Size_Exp": 0.1, "Barra_Momentum_Exp": 0.2, "Barra_Volatility_Exp": 0.3,
            "Factor_Annualized": {"size": 0.01, "momentum": 0.02, "volatility": 0.03},
            "Multi_Factor_Intercept": 0.01
        }
        pa_inst.calculate_holding_metrics.return_value = {"Avg_Daily_Holdings_Count": 1}
        mock_pa.return_value = pa_inst
        
        with patch('qlib.data.D.calendar', return_value=dates):
            ef.run_detailed_backtest_analysis(executor, "c", "date", str(out_dir), "day")

# --- Stage 7: Risk Analysis & Leaderboard ---

@patch('qlib.workflow.R')
def test_risk_analysis_and_leaderboard_submodel_skip(mock_R, mock_env):
    ef, _ = mock_env
    # Lines 1010, 1054-1055
    report_df = pd.DataFrame({"account": [100.0], "bench": [0.0], "return": [0.0]}, index=pd.to_datetime(["2020-01-01"]))
    norm_df = pd.DataFrame({"M1": [0.5]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")], names=["datetime", "instrument"]))
    train_records = {"experiment_name": "E", "models": {"M1": "rid1", "M2": None}}
    
    mock_recorder = MagicMock()
    mock_recorder.load_object.return_value = pd.DataFrame({"account": [100.0], "bench": [0.0], "return": [0.0]}, index=pd.to_datetime(["2020-01-01"]))
    mock_R.get_recorder.side_effect = [mock_recorder, Exception("Recorder fail")]
    
    with patch('quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer'):
        with patch('strategy.load_strategy_config', return_value={}):
            with patch('qlib.data.D.calendar', return_value=pd.to_datetime(["2020-01-01"])):
                with patch('builtins.print') as mock_p:
                    ef.risk_analysis_and_leaderboard(report_df, norm_df, train_records, ["M1", "M1"], "day", "out", "date")
                    assert any("[跳过]" in str(c) for c in mock_p.call_args_list)

def test_risk_analysis_and_leaderboard_display_cols(mock_env):
    ef, _ = mock_env
    # Line 1071: fallback print
    report_df = pd.DataFrame({"account": [100.0], "bench": [0.0], "return": [0.0]}, index=pd.to_datetime(["2020-01-01"]))
    norm_df = pd.DataFrame({"M1": [0.5]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")], names=["datetime", "instrument"]))
    
    with patch('quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer') as mock_pa:
        pa_inst = MagicMock()
        pa_inst.calculate_traditional_metrics.return_value = {"SomethingElse": 1.0}
        mock_pa.return_value = pa_inst
        with patch('strategy.load_strategy_config', return_value={}):
            with patch('qlib.data.D.calendar', return_value=pd.to_datetime(["2020-01-01"])):
                with patch('builtins.print') as mock_p:
                    ef.risk_analysis_and_leaderboard(report_df, norm_df, {"experiment_name":"E", "models":{"M1":"r1"}}, ["M1"], "day", "out", "date")

# --- Main & Others ---

def test_main_arg_error(mock_env):
    ef, _ = mock_env
    # Line 1444: no args
    import sys
    with patch.object(sys, 'argv', ['ensemble_fusion.py']):
        with pytest.raises(SystemExit):
            ef.main()

def test_main_manual_models_missing(mock_env):
    ef, _ = mock_env
    # Lines 1471-1479, 1531
    train_records = {"experiment_name": "E", "models": {"m1": "r1"}}
    with patch('quantpits.scripts.ensemble_fusion.load_config', return_value=(train_records, {}, {})):
        with patch('sys.argv', ['ensemble_fusion.py', '--models', 'm1,m2']):
            norm_df = pd.DataFrame({"m1":[0.5]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"),"A")], names=["datetime", "instrument"]))
            with patch('quantpits.scripts.ensemble_fusion.load_selected_predictions', return_value=(norm_df, {"m1": 0.1}, ["m1"])):
                with patch('quantpits.scripts.ensemble_fusion.filter_norm_df_by_args', return_value=norm_df):
                    report_df = pd.DataFrame({"account": [100.0, 110.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
                    res = {'name': 'c1', 'models': ['m1'], 'method': 'equal', 'is_default': True, 'pred_file': 'f.csv', 'report_df': report_df}
                    with patch('quantpits.scripts.ensemble_fusion.run_single_combo', return_value=res):
                        with patch('strategy.get_backtest_config', return_value={'account': 100.0}):
                            with patch('builtins.print', side_effect=lambda *a, **k: None):
                                ef.main()

def test_main_combo_skip_no_models(mock_env):
    ef, _ = mock_env
    # Lines 1301-1302, 1557-1558
    train_records = {"experiment_name": "E", "models": {"m1": "r1", "m2": "r2"}}
    ec = {"combos": {"c1": {"models": ["m2"]}}}
    
    with patch('quantpits.scripts.ensemble_fusion.load_config', return_value=(train_records, {}, ec)):
        with patch('sys.argv', ['ensemble_fusion.py', '--from-config-all']):
             norm_df = pd.DataFrame({"m1":[0.5]}, index=pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"),"A")], names=["datetime", "instrument"]))
             with patch('quantpits.scripts.ensemble_fusion.load_selected_predictions', return_value=(norm_df, {"m1": 0.1}, ["m1"])):
                 with patch('quantpits.scripts.ensemble_fusion.filter_norm_df_by_args', return_value=norm_df):
                    with patch('builtins.print', side_effect=lambda *a, **k: None) as mock_p:
                        ef.main()
                        assert any("没有有效模型，跳过" in str(arg) for call in mock_p.call_args_list for arg in call[0])


def test_main_combo_missing_and_empty(mock_env):
    ef, _ = mock_env
    # Lines 1485-1487, 1497-1498
    train_records = {"experiment_name": "E", "models": {"m1": "r1"}}
    ec_empty = {"combos": {}}
    ec_missing = {"combos": {"c1": {"models": ["m1"]}}}
    
    with patch('quantpits.scripts.ensemble_fusion.load_config', return_value=(train_records, {}, ec_empty)):
        with patch('sys.argv', ['ensemble_fusion.py', '--from-config-all']):
            with pytest.raises(SystemExit):
                ef.main()
                
    with patch('quantpits.scripts.ensemble_fusion.load_config', return_value=(train_records, {}, ec_missing)):
        with patch('sys.argv', ['ensemble_fusion.py', '--combo', 'non_existent']):
            with pytest.raises(SystemExit):
                ef.main()

def test_main_no_valid_models(mock_env):
    ef, _ = mock_env
    # Line 1535-1536
    train_records = {"experiment_name": "E", "models": {"m1": "r1"}}
    with patch('quantpits.scripts.ensemble_fusion.load_config', return_value=(train_records, {}, {})):
        with patch('sys.argv', ['ensemble_fusion.py', '--models', 'm2']):
            with pytest.raises(SystemExit):
                ef.main()

def test_main_empty_norm_df_after_filter(mock_env):
    ef, _ = mock_env
    # Line 1548-1549
    train_records = {"experiment_name": "E", "models": {"m1": "r1"}}
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"),"A")], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1":[0.5]}, index=idx)
    with patch('quantpits.scripts.ensemble_fusion.load_config', return_value=(train_records, {}, {})):
        with patch('sys.argv', ['ensemble_fusion.py', '--models', 'm1']):
            with patch('quantpits.scripts.ensemble_fusion.load_selected_predictions', return_value=(norm_df, {"m1": 0.1}, ["m1"])):
                with patch('quantpits.scripts.ensemble_fusion.filter_norm_df_by_args', return_value=pd.DataFrame()):
                    with pytest.raises(SystemExit):
                        ef.main()

def test_main_final_prints(mock_env):
    ef, _ = mock_env
    # Lines 1593-1598
    train_records = {"experiment_name": "E", "models": {"m1": "r1"}}
    report_df = pd.DataFrame({"account": [100.0, 110.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    res = {'name': 'c1', 'models': ['m1'], 'method': 'equal', 'is_default': True, 'pred_file': 'f.csv', 'report_df': report_df}
    
    with patch('quantpits.scripts.ensemble_fusion.load_config', return_value=(train_records, {}, {})):
        with patch('sys.argv', ['ensemble_fusion.py', '--models', 'm1']):
            idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"),"A")], names=["datetime", "instrument"])
            norm_df = pd.DataFrame({"m1":[0.5]}, index=idx)
            with patch('quantpits.scripts.ensemble_fusion.load_selected_predictions', return_value=(norm_df, {"m1": 0.1}, ["m1"])):
                with patch('quantpits.scripts.ensemble_fusion.filter_norm_df_by_args', return_value=norm_df):
                    with patch('quantpits.scripts.ensemble_fusion.run_single_combo', return_value=res):
                        with patch('strategy.get_backtest_config', return_value={'account': 100.0}):
                            with patch('builtins.print', side_effect=lambda *a, **k: None) as mock_p:
                                ef.main()
                                assert mock_p.call_count > 5
