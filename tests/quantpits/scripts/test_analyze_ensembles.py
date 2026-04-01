import pytest
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    from quantpits.scripts import analyze_ensembles
    import importlib
    importlib.reload(env)
    importlib.reload(analyze_ensembles)
    
    yield analyze_ensembles, workspace

def test_run_single_backtest_oos_success(mock_env):
    analyze, _ = mock_env
    # Since run_single_backtest_oos now delegates to search_utils.run_single_backtest,
    # we need to mock at the search_utils level
    with patch('quantpits.utils.strategy.load_strategy_config') as mock_st_cfg:
        with patch('quantpits.utils.strategy.get_backtest_config') as mock_bt_cfg:
            with patch('quantpits.utils.strategy.create_backtest_strategy'):
                with patch('quantpits.utils.backtest_utils.run_backtest_with_strategy') as mock_run:
                    with patch('quantpits.utils.backtest_utils.standard_evaluate_portfolio') as mock_eval:
                        
                        mock_st_cfg.return_value = {"strategy": {"params": {}}, "benchmark": "SH000300"}
                        mock_bt_cfg.return_value = {"account": 10000}
                        
                        mock_report = pd.DataFrame({"account": [10000, 10100]})
                        mock_run.return_value = (mock_report, None)
                        
                        mock_eval.return_value = {
                            "CAGR_252": 0.1,
                            "Max_Drawdown": -0.05,
                            "Absolute_Return": 0.01,
                            "Benchmark_Absolute_Return": 0.005,
                            "Excess_Return_CAGR_252": 0.05,
                            "Calmar": 2.0
                        }
                        
                        dates = pd.to_datetime(["2020-01-01"])
                        idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=["datetime", "instrument"])
                        norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
                        
                        trade_exchange = MagicMock()
                        
                        res = analyze.run_single_backtest_oos(
                            ["m1"], norm_df, 1, 0, "SH000300", "day", trade_exchange, "2020-01-01", "2020-01-02"
                        )
                        
                        assert res is not None
                        assert res["Ann_Ret"] == 0.1
                        assert res["Max_DD"] == -0.05
                        assert res["Ann_Excess"] == 0.05
                        assert res["Calmar"] == 2.0
                        assert res["models"] == "m1"

def test_run_single_backtest_oos_exception(mock_env):
    analyze, _ = mock_env
    with patch('quantpits.utils.strategy.load_strategy_config'):
        with patch('quantpits.utils.strategy.get_backtest_config'):
            with patch('quantpits.utils.strategy.create_backtest_strategy'):
                with patch('quantpits.utils.backtest_utils.run_backtest_with_strategy', side_effect=Exception("Failed")):
                    dates = pd.to_datetime(["2020-01-01"])
                    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=["datetime", "instrument"])
                    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
                    res = analyze.run_single_backtest_oos(
                    ["m1"], norm_df, 1, 0, "SH000300", "day", MagicMock(), "2020-01-01", "2020-01-02",
                    st_config={"strategy": {"params": {}}}, bt_config={"account": 100}
                )
                assert res is None

def test_main(mock_env, tmp_path):
    analyze, workspace = mock_env
    
    meta_path = tmp_path / "run_metadata_2020-01-01.json"
    record_file = tmp_path / "train_records.json"
    
    meta_data = {
        "anchor_date": "2020-01-01",
        "script_used": "brute_force_fast",
        "freq": "day",
        "record_file": str(record_file),
        "oos_start_date": "2020-01-02",
        "oos_end_date": "2020-01-03"
    }
    with open(meta_path, "w") as f:
        json.dump(meta_data, f)
        
    with open(record_file, "w") as f:
        json.dump({"experiment": "test"}, f)
        
    # Generate mock CSV for brute_force results
    csv_path = tmp_path / "brute_force_fast_results_2020-01-01.csv"
    df = pd.DataFrame({
        "models": ["m1", "m2", "m3", "m4", "m1,m2"],
        "n_models": [1, 1, 1, 1, 2],
        "Ann_Ret": [0.1, 0.2, -0.1, 0.05, 0.3],
        "Ann_Excess": [0.05, 0.15, -0.05, 0.06, 0.2],
        "Calmar": [1.0, 2.0, -1.0, 0.5, 3.0],
        "Max_DD": [-0.1, -0.05, -0.2, -0.01, -0.08],
        "avg_corr": [1.0, 1.0, 1.0, 1.0, 0.2]
    })
    df.to_csv(csv_path, index=False)
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path)]):
        # We need to mock load_workspace_config from quantpits.utils.config_loader
        with patch('quantpits.scripts.analyze_ensembles.strategy.load_strategy_config', return_value={"benchmark": "SH000300", "strategy": {"params": {}}}):
            with patch('quantpits.scripts.analyze_ensembles.strategy.get_backtest_config', return_value={"account": 1000, "exchange_kwargs": {"freq": "day"}}):
                with patch('quantpits.utils.config_loader.load_workspace_config', return_value={"TopK": 1, "DropN": 0}):
                    with patch('quantpits.utils.predict_utils.load_predictions_from_recorder') as mock_load_pred:
                        
                        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
                        idx = pd.MultiIndex.from_arrays([
                            [dates[0], dates[1], dates[2]], 
                            ["A", "A", "A"]
                        ], names=["datetime", "instrument"])
                        
                        norm_df = pd.DataFrame({"m1": [0.5, 0.6, 0.7], "m2": [0.1, 0.2, 0.3]}, index=idx)
                        
                        mock_load_pred.return_value = (norm_df, None, None)
                        
                        with patch('quantpits.scripts.analyze_ensembles.Exchange'):
                            with patch('quantpits.scripts.analyze_ensembles.run_single_backtest_oos') as mock_run_oos:
                                mock_run_oos.return_value = {
                                    "models": "m1", "n_models": 1, "Ann_Ret": 0.1, "Max_DD": -0.05, "Excess_Ret": 0.01,
                                    "Ann_Excess": 0.05, "Total_Ret": 0.1, "Final_NAV": 10000, "Calmar": 2.0
                                }
                                with patch('builtins.print'):
                                    analyze.main()
                                    # Should run smoothly and create the outputs with prefix
                                    assert (tmp_path / "brute_force_fast_oos_multi_analysis_2020-01-01.csv").exists()
                                    assert (tmp_path / "brute_force_fast_oos_report_2020-01-01.txt").exists()

def test_main_missing_oos_dates(mock_env, tmp_path):
    analyze, _ = mock_env
    meta_path = tmp_path / "run_meta.json"
    meta_data = {
        "anchor_date": "2020",
        "script_used": "brute",
        "freq": "day",
        "record_file": "file",
        "oos_start_date": None,
        "oos_end_date": None
    }
    with open(meta_path, "w") as f:
        json.dump(meta_data, f)
        
    # Create the expected IS result file to avoid early sys.exit(1)
    csv_path = tmp_path / "brute_force_results_2020.csv"
    pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1], "Ann_Ret": [0.1], "Calmar": [1], "Max_DD": [-0.1], "n_models": [1]}).to_csv(csv_path, index=False)
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path)]):
        with patch('builtins.print') as mock_print:
            # Now it returns early instead of sys.exit
            analyze.main()
            # Verify its warning message
            any_warn = any("元数据中未找到有效的 OOS 数据周期" in str(arg) for call in mock_print.call_args_list for arg in call[0])
            assert any_warn

def test_main_missing_is_csv(mock_env, tmp_path):
    analyze, _ = mock_env
    meta_path = tmp_path / "run_meta.json"
    meta_data = {
        "anchor_date": "2020",
        "script_used": "brute",
        "freq": "day",
        "record_file": "file",
        "oos_start_date": "2020",
        "oos_end_date": "2021"
    }
    with open(meta_path, "w") as f:
        json.dump(meta_data, f)
        
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path)]):
        with pytest.raises(SystemExit):
            analyze.main()

def test_generate_is_visualizations(mock_env, tmp_path):
    analyze, _ = mock_env
    df = pd.DataFrame({
        "models": ["m1", "m2", "m1,m2"],
        "n_models": [1, 1, 2],
        "Ann_Ret": [0.1, 0.2, 0.3],
        "Ann_Excess": [0.05, 0.1, 0.15],
        "Calmar": [1.0, 2.0, 3.0],
        "Max_DD": [-0.1, -0.05, -0.08]
    })
    
    # Mock correlation matrix for diversity bonus coverage
    corr_df = pd.DataFrame({"m1": [1.0, 0.2], "m2": [0.2, 1.0]}, index=["m1", "m2"])
    corr_path = tmp_path / "correlation_matrix_2020.csv"
    corr_df.to_csv(corr_path)
    
    with patch('matplotlib.pyplot.savefig'):
        res_df = analyze.generate_is_visualizations_and_report(df, str(tmp_path), "2020", prefix="test_", top_n=2)
        assert "avg_corr" in res_df.columns
        assert "diversity_bonus" in res_df.columns
        assert (tmp_path / "test_analysis_report_2020.txt").exists()

def test_generate_dendrogram(mock_env, tmp_path):
    analyze, _ = mock_env
    corr_df = pd.DataFrame({"m1": [1.0, 0.2], "m2": [0.2, 1.0]}, index=["m1", "m2"])
    corr_path = tmp_path / "test_correlation_matrix_2020.csv"
    corr_df.to_csv(corr_path)
    
    with patch('matplotlib.pyplot.savefig'):
        analyze.generate_dendrogram(str(tmp_path), "2020", prefix="test_")

def test_generate_oos_visualizations(mock_env, tmp_path):
    analyze, _ = mock_env
    oos_df = pd.DataFrame({
        "Pool_Sources": ["Yield_Top", "Robust_Top"],
        "Ann_Excess": [0.1, 0.05],
        "Max_DD": [-0.05, -0.02]
    })
    with patch('matplotlib.pyplot.savefig'):
        analyze.generate_oos_visualizations(oos_df, str(tmp_path), "2020", prefix="test_")

def test_main_training_mode_filter(mock_env, tmp_path):
    analyze, _ = mock_env
    meta_path = tmp_path / "meta_filter.json"
    with open(meta_path, "w") as f:
        json.dump({
            "anchor_date": "2020", "script_used": "brute", "freq": "day",
            "record_file": "none", "oos_start_date": "none", "oos_end_date": "none"
        }, f)
    
    csv_path = tmp_path / "brute_force_results_2020.csv"
    pd.DataFrame({
        "models": ["m1@static", "m1@inc"], 
        "Ann_Excess": [0.1, 0.2], 
        "Ann_Ret": [0.1, 0.2],
        "Calmar": [1, 2], 
        "Max_DD": [-0.1, -0.1], 
        "n_models": [1, 1]
    }).to_csv(csv_path, index=False)
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path), '--training-mode', 'static']):
        with patch('quantpits.scripts.analyze_ensembles.generate_is_visualizations_and_report') as mock_vis:
            # We expect it to filter and call vis with only 1 row
            mock_vis.return_value = pd.DataFrame() 
            analyze.main()
            
            assert mock_vis.called
            call_df = mock_vis.call_args[0][0]
            assert len(call_df) == 1
            assert call_df.iloc[0]["models"] == "m1@static"
