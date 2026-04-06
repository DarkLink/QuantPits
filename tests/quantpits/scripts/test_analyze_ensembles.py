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
    
    # Create new-style directory structure
    run_dir = tmp_path / "brute_force_fast_2020-01-01"
    is_dir = run_dir / "is"
    oos_dir = run_dir / "oos"
    is_dir.mkdir(parents=True)
    oos_dir.mkdir(parents=True)
    
    record_file = tmp_path / "train_records.json"
    meta_path = run_dir / "run_metadata.json"
    
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
        
    # Generate mock CSV for IS results in the new location
    csv_path = is_dir / "results.csv"
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
                                    # New structure: outputs go to oos/ subdirectory
                                    assert (oos_dir / "oos_multi_analysis.csv").exists()
                                    assert (oos_dir / "oos_report.txt").exists()
                                    assert (run_dir / "summary.md").exists()

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
        
    # Create the expected IS result file (legacy flat structure, _find_is_results_csv will look here)
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
    corr_path = tmp_path / "correlation_matrix.csv"
    corr_df.to_csv(corr_path)
    
    with patch('matplotlib.pyplot.savefig'):
        res_df = analyze.generate_is_visualizations_and_report(
            df, str(tmp_path), "2020", top_n=2, corr_file=str(corr_path)
        )
        assert "avg_corr" in res_df.columns
        assert "diversity_bonus" in res_df.columns
        assert (tmp_path / "analysis_report.txt").exists()

def test_generate_dendrogram(mock_env, tmp_path):
    analyze, _ = mock_env
    corr_df = pd.DataFrame({"m1": [1.0, 0.2], "m2": [0.2, 1.0]}, index=["m1", "m2"])
    corr_path = tmp_path / "correlation_matrix.csv"
    corr_df.to_csv(corr_path)
    
    with patch('matplotlib.pyplot.savefig'):
        analyze.generate_dendrogram(str(tmp_path), corr_file=str(corr_path))

def test_generate_oos_visualizations(mock_env, tmp_path):
    analyze, _ = mock_env
    oos_df = pd.DataFrame({
        "Pool_Sources": ["Yield_Top", "Robust_Top"],
        "Ann_Excess": [0.1, 0.05],
        "Max_DD": [-0.05, -0.02]
    })
    with patch('matplotlib.pyplot.savefig'):
        analyze.generate_oos_visualizations(oos_df, str(tmp_path))

def test_main_training_mode_filter(mock_env, tmp_path):
    analyze, _ = mock_env
    meta_path = tmp_path / "meta_filter.json"
    with open(meta_path, "w") as f:
        json.dump({
            "anchor_date": "2020", "script_used": "brute", "freq": "day",
            "record_file": "none", "oos_start_date": "none", "oos_end_date": "none"
        }, f)
    
    # Legacy structure: LegacyRunContext will look in the same directory as metadata
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


def test_find_is_results_csv(mock_env, tmp_path):
    """Test the IS results CSV discovery function."""
    analyze, _ = mock_env
    from quantpits.utils.run_context import RunContext
    
    # New structure
    ctx = RunContext(base_dir=str(tmp_path), script_name="brute_force", anchor_date="2026-04-03")
    ctx.ensure_dirs()
    
    # Create results.csv in is/
    pd.DataFrame({"models": ["m1"]}).to_csv(ctx.is_path("results.csv"), index=False)
    
    meta = {"anchor_date": "2026-04-03", "script_used": "brute_force_ensemble"}
    found = analyze._find_is_results_csv(ctx, meta)
    assert found is not None
    assert found.endswith("results.csv")


def test_generate_summary_md(mock_env, tmp_path):
    """Test summary.md generation."""
    analyze, _ = mock_env
    from quantpits.utils.run_context import RunContext
    
    ctx = RunContext(base_dir=str(tmp_path), script_name="brute_force", anchor_date="2026-04-03")
    ctx.ensure_dirs()
    
    meta = {
        "anchor_date": "2026-04-03",
        "script_used": "brute_force_ensemble",
        "is_start_date": "2021-01-01",
        "is_end_date": "2025-04-03",
        "oos_start_date": None,
        "oos_end_date": None,
    }
    
    df = pd.DataFrame({
        "models": ["m1", "m1,m2"],
        "n_models": [1, 2],
        "Ann_Excess": [0.1, 0.2],
        "Max_DD": [-0.05, -0.08],
        "Calmar": [2.0, 2.5],
    })
    
    analyze.generate_summary_md(ctx, meta, df)
    summary_path = ctx.run_path("summary.md")
    assert os.path.exists(summary_path)
    
    content = open(summary_path).read()
    assert "2026-04-03" in content
    assert "brute_force_ensemble" in content

def test_exceptions_and_edge_cases(mock_env, tmp_path):
    analyze, _ = mock_env
    
    # Test exceptions in generate_is_visualizations_and_report
    df = pd.DataFrame({"models": ["a", "b"], "n_models": [1, 2], "Ann_Excess": [0.1, 0.2], "Max_DD": [-0.1, -0.2], "Calmar": [1, 2]})
    with patch("quantpits.scripts.analyze_ensembles.plt.subplots", side_effect=Exception("plot error")):
        analyze.generate_is_visualizations_and_report(df, str(tmp_path), "2020", top_n=2)
        
    with patch("quantpits.scripts.analyze_ensembles.pd.DataFrame.sort_values", side_effect=Exception("df error")):
        analyze.generate_is_visualizations_and_report(df, str(tmp_path), "2020", top_n=2)

    with patch("quantpits.scripts.analyze_ensembles.pd.merge", side_effect=Exception("corr error")):
        corr_path = tmp_path / "corr.csv"
        pd.DataFrame({"a": [1]}).to_csv(corr_path)
        analyze.generate_is_visualizations_and_report(df, str(tmp_path), "2020", top_n=2, corr_file=str(corr_path))
        
    with patch("quantpits.scripts.analyze_ensembles.linkage", side_effect=Exception("clustering error")):
        analyze.generate_dendrogram(str(tmp_path), corr_file=str(corr_path))

    with patch("quantpits.scripts.analyze_ensembles.plt.subplots", side_effect=Exception("plot error")):
        oos_df = pd.DataFrame({"Pool_Sources": ["A"], "Ann_Excess": [0.1], "Max_DD": [-0.1]})
        analyze.generate_oos_visualizations(oos_df, str(tmp_path))

def test_generate_summary_md_long_names(mock_env, tmp_path):
    analyze, _ = mock_env
    from quantpits.utils.run_context import RunContext
    ctx = RunContext(base_dir=str(tmp_path), script_name="brute", anchor_date="2020")
    ctx.ensure_dirs()
    
    meta = {"anchor_date": "2020"}
    df = pd.DataFrame({
        "models": ["m" * 100], "n_models": [1], "Ann_Excess": [0.1], "Max_DD": [-0.1], "Calmar": [1]
    })
    oos_df = pd.DataFrame({
        "models": ["m" * 100], "Ann_Excess": [0.1], "Max_DD": [-0.1], "Pool_Sources": ["test"], "IS_Ann_Excess": [0.1]
    })
    analyze.generate_summary_md(ctx, meta, df, oos_df)
    assert os.path.exists(ctx.run_path("summary.md"))

def test_is_results_csv_legacy(mock_env, tmp_path):
    analyze, _ = mock_env
    from quantpits.utils.run_context import RunContext
    ctx = RunContext(base_dir=str(tmp_path), script_name="test", anchor_date="2020")
    ctx.ensure_dirs()
    
    pd.DataFrame().to_csv(os.path.join(ctx.is_dir, "minentropy_results_2020.csv"))
    res = analyze._find_is_results_csv(ctx, {"anchor_date": "2020", "script_used": "minentropy"})
    assert res is not None
    assert res.endswith("minentropy_results_2020.csv")

def test_missing_data_sys_exits(mock_env, tmp_path):
    analyze, workspace = mock_env
    meta_path = tmp_path / "meta.json"
    with open(meta_path, "w") as f:
        json.dump({"anchor_date": "2020", "script_used": "x", "freq": "day", "record_file": "missing.json", "oos_start_date": "2020-01-01", "oos_end_date": "2020-01-02"}, f)
    
    from quantpits.utils.run_context import RunContext
    ctx = RunContext.from_metadata(str(meta_path))
    ctx.ensure_dirs()
    pd.DataFrame({"models": ["a", "b"], "Ann_Excess": [0.1, 0.2], "Calmar": [1, 2], "Max_DD": [-0.1, -0.2], "n_models": [1, 2], "Ann_Ret": [0.1, 0.2]}).to_csv(os.path.join(ctx.is_dir, "brute_force_results_2020.csv"))
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path)]):
        with pytest.raises(SystemExit):
            analyze.main()

def test_empty_oos_results_early_return(mock_env, tmp_path):
    analyze, workspace = mock_env
    meta_path = tmp_path / "meta.json"
    rec_path = tmp_path / "rec.json"
    with open(meta_path, "w") as f:
        json.dump({"anchor_date": "2020", "script_used": "x", "freq": "day", "record_file": str(rec_path), "oos_start_date": "2020-01-01", "oos_end_date": "2020-01-02"}, f)
    with open(rec_path, "w") as f:
        json.dump({}, f)
        
    from quantpits.utils.run_context import RunContext
    ctx = RunContext.from_metadata(str(meta_path))
    ctx.ensure_dirs()    
    pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1], "Ann_Ret": [0.1], "Calmar": [1], "Max_DD": [0], "n_models": [1]}).to_csv(os.path.join(ctx.is_dir, "brute_force_results_2020.csv"))
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path)]):
        with patch('quantpits.utils.config_loader.load_workspace_config', return_value={}):
            idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"),"A")], names=["datetime", "instrument"])
            norm_df = pd.DataFrame({"m1":[0.5]}, index=idx)
            with patch('quantpits.utils.predict_utils.load_predictions_from_recorder', return_value=(norm_df, None, None)):
                with patch('quantpits.scripts.analyze_ensembles.run_single_backtest_oos', return_value=None):
                    with patch('quantpits.scripts.analyze_ensembles.Exchange'):
                        analyze.main()

def test_max_plot_points(mock_env, tmp_path):
    analyze, _ = mock_env
    df = pd.DataFrame({
        "models": ["m1"] * 50001,
        "n_models": [2] * 50001,
        "Ann_Excess": [0.1] * 50001,
        "Max_DD": [-0.1] * 50001,
        "Calmar": [1] * 50001,
        "avg_corr": [np.nan] * 50001
    })
    with patch("quantpits.scripts.analyze_ensembles.plt.savefig"):
        analyze.generate_is_visualizations_and_report(df, str(tmp_path), "2020", top_n=2)

def test_main_training_mode_empty(mock_env, tmp_path):
    analyze, _ = mock_env
    meta_path = tmp_path / "meta_filter.json"
    with open(meta_path, "w") as f:
        json.dump({
            "anchor_date": "2020", "script_used": "brute", "freq": "day",
            "record_file": "none", "oos_start_date": "none", "oos_end_date": "none"
        }, f)
    
    csv_path = tmp_path / "brute_force_results_2020.csv"
    pd.DataFrame({"models": ["m1@static"], "Ann_Excess": [0.1], "Ann_Ret": [0.1], "Calmar": [1], "Max_DD": [-0.1], "n_models": [1]}).to_csv(csv_path, index=False)
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path), '--training-mode', 'nonexistent']):
        with pytest.raises(SystemExit):
            analyze.main()

def test_oos_norm_df_empty_exits(mock_env, tmp_path):
    analyze, workspace = mock_env
    meta_path = tmp_path / "meta.json"
    rec_path = tmp_path / "rec.json"
    with open(meta_path, "w") as f:
        json.dump({"anchor_date": "2020", "script_used": "x", "freq": "day", "record_file": str(rec_path), "oos_start_date": "2021-01-01", "oos_end_date": "2021-01-02"}, f)
    with open(rec_path, "w") as f:
        json.dump({}, f)
        
    from quantpits.utils.run_context import RunContext
    ctx = RunContext.from_metadata(str(meta_path))
    ctx.ensure_dirs()
    pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1], "Ann_Ret": [0.1], "Calmar": [1], "Max_DD": [0], "n_models": [1]}).to_csv(os.path.join(ctx.is_dir, "brute_force_results_2020.csv"))
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--metadata', str(meta_path)]):
        with patch('quantpits.utils.config_loader.load_workspace_config', return_value={}):
            idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"),"A")], names=["datetime", "instrument"])
            norm_df = pd.DataFrame({"m1":[0.5]}, index=idx)
            with patch('quantpits.utils.predict_utils.load_predictions_from_recorder', return_value=(norm_df, None, None)):
                with pytest.raises(SystemExit):
                    analyze.main()

def test_html_report_helpers(mock_env):
    """Test the small HTML helper functions."""
    analyze, _ = mock_env
    
    # 1. _pool_badge_html
    badge = analyze._pool_badge_html("Yield_Top")
    assert "Yield_Top" in badge
    assert "#78350f" in badge  # Correct bg color
    
    # 2. _pool_sources_html
    sources = analyze._pool_sources_html("Yield_Top | Robust_Top")
    assert sources.count("<span") == 2
    assert "Yield_Top" in sources
    assert "Robust_Top" in sources
    
    # 3. _is_oos_arrow_html
    # Positive OOS
    arrow_pos = analyze._is_oos_arrow_html(0.1, 0.05)
    assert "#10b981" in arrow_pos  # Green
    assert "10.0%" in arrow_pos
    assert "5.0%" in arrow_pos
    
    # Negative OOS
    arrow_neg = analyze._is_oos_arrow_html(0.1, -0.05)
    assert "#ef4444" in arrow_neg  # Red
    
    # Missing IS
    arrow_miss = analyze._is_oos_arrow_html(float('nan'), 0.05)
    assert "?" in arrow_miss

def test_oos_table_rows(mock_env):
    """Test table row generation."""
    analyze, _ = mock_env
    df = pd.DataFrame({
        "models": ["model_A", "model_B"],
        "n_models": [1, 2],
        "Ann_Excess": [0.05, -0.01],
        "Max_DD": [-0.1, -0.2],
        "Calmar": [0.5, -0.05],
        "IS_Ann_Excess": [0.1, 0.15],
        "IS_Calmar": [1.0, 1.5],
        "Pool_Sources": ["Yield_Top", "Robust_Top | Defensive_Top"],
        "_pool_count": [1, 2] 
    })
    
    rows = analyze._oos_table_rows(df)
    assert "model_A" in rows
    assert "model_B" in rows
    assert rows.count("<tr>") == 2
    assert "2×" in rows 

def test_generate_oos_html_report(mock_env, tmp_path):
    """Test full HTML report generation."""
    analyze, _ = mock_env
    oos_df = pd.DataFrame({
        "models": ["m1", "m2"],
        "n_models": [1, 2],
        "Ann_Excess": [0.1, -0.05],
        "Max_DD": [-0.05, -0.1],
        "Calmar": [2.0, -0.5],
        "IS_Ann_Excess": [0.15, 0.2],
        "IS_Calmar": [3.0, 4.0],
        "Pool_Sources": ["Yield_Top", "Yield_Top | Robust_Top"]
    })
    
    oos_dir = str(tmp_path)
    report_path = analyze.generate_oos_html_report(oos_df, oos_dir, "2026-04-03")
    
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "OOS Multi-Pool Analysis" in content
        assert "m1" in content
        assert "m2" in content
        assert "Yield_Top" in content
        assert "Robust_Top" in content
        assert "Cross-Dimension Consensus" in content
        assert "2026-04-03" in content

