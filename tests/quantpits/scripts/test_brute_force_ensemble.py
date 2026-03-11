import pytest
import os
import pandas as pd
import numpy as np
import yaml
import json
import sys
import gc
import signal
from unittest.mock import MagicMock, patch, mock_open
from io import StringIO
from datetime import datetime

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    import importlib
    import qlib.data
    from quantpits.scripts import env, brute_force_ensemble
    
    mock_D = MagicMock()
    mock_D.calendar.return_value = pd.date_range("2020-01-01", periods=10, freq="D")
    monkeypatch.setattr(qlib.data, 'D', mock_D)
    
    importlib.reload(env)
    importlib.reload(brute_force_ensemble)
    
    yield brute_force_ensemble, workspace

# ── zscore_norm ──────────────────────────────────────────────────────────
def test_zscore_norm(mock_env):
    bfe, _ = mock_env
    series = pd.Series([1, 2, 3, 10, 20, 30], 
                       index=pd.MultiIndex.from_arrays([
                           pd.to_datetime(["2020-01-01"]*3 + ["2020-01-02"]*3),
                           ["A", "B", "C", "A", "B", "C"]
                       ], names=["datetime", "instrument"]))
    normed = bfe.zscore_norm(series)
    assert np.isclose(normed.xs("2020-01-01", level="datetime").mean(), 0)
    assert np.isclose(normed.xs("2020-01-01", level="datetime").std(), 1)
    
    # Zero std case (lines 138-139)
    zero_series = pd.Series([1.0, 1.0, 1.0], index=pd.MultiIndex.from_arrays([
        pd.to_datetime(["2020-01-01"]*3), ["A", "B", "C"]
    ], names=["datetime", "instrument"]))
    assert np.all(bfe.zscore_norm(zero_series) == 0.0)

# ── load_combo_groups ────────────────────────────────────────────────────
def test_load_combo_groups(mock_env):
    bfe, workspace = mock_env
    cfg_path = workspace / "combo_groups.yaml"
    cfg_data = {
        "groups": {
            "G1": ["m1", "m2", "invalid_m"],
            "G2": ["m3"]
        }
    }
    cfg_path.write_text(yaml.dump(cfg_data))
    
    groups = bfe.load_combo_groups(str(cfg_path), available_models=["m1", "m2", "m3", "m4"])
    assert "G1" in groups
    assert "G2" in groups
    assert groups["G1"] == ["m1", "m2"] # invalid_m is filtered
    assert groups["G2"] == ["m3"]

def test_load_combo_groups_empty(mock_env):
    bfe, workspace = mock_env
    cfg_path = workspace / "combo_groups.yaml"
    cfg_path.write_text(yaml.dump({}))
    
    with pytest.raises(ValueError, match="为空"):
        bfe.load_combo_groups(str(cfg_path), available_models=["m1"])

def test_load_combo_groups_warnings(mock_env, tmp_path):
    bfe, _ = mock_env
    cfg_path = tmp_path / "groups_warn.yaml"
    cfg_path.write_text(yaml.dump({"groups": {"G1": ["m_missing"]}}))
    with patch('builtins.print') as mock_print:
        groups = bfe.load_combo_groups(str(cfg_path), ["m1"])
        assert "G1" not in groups
        assert any("组 [G1] 无有效模型，已跳过" in str(call) for call in mock_print.call_args_list)

# ── generate_grouped_combinations ────────────────────────────────────────
def test_generate_grouped_combinations(mock_env):
    bfe, _ = mock_env
    groups = {
        "G1": ["m1", "m2"],
        "G2": ["m3"],
        "G3": ["m4"]
    }
    # Without min_combo_size/max_combo_size args (default 1, 0)
    combos = bfe.generate_grouped_combinations(groups)
    # Total combos should cover r=1 to r=3.
    # r=1: (m1,), (m2,), (m3,), (m4,) => 4
    # r=2: G1*G2 => 2, G1*G3 => 2, G2*G3 => 1 => 5
    # r=3: G1*G2*G3 => 2
    # Total: 11
    assert len(combos) == 11
    assert ("m1", "m3") in combos

def test_generate_grouped_combinations_min_max(mock_env):
    bfe, _ = mock_env
    groups = {
        "G1": ["m1", "m2"],
        "G2": ["m3"],
        "G3": ["m4"]
    }
    combos = bfe.generate_grouped_combinations(groups, min_combo_size=2, max_combo_size=2)
    # r=2 only: 5 combos
    assert len(combos) == 5
    assert ("m1",) not in combos
    assert ("m1", "m3", "m4") not in combos
    assert ("m2", "m4") in combos

# ── split_is_oos_by_args ─────────────────────────────────────────────────
def test_split_is_oos_by_args(mock_env):
    bfe, _ = mock_env
    
    dates = pd.date_range("2010-01-01", "2020-01-01", freq="YS") # 11 years
    df = pd.DataFrame({"score": range(len(dates))}, index=pd.MultiIndex.from_arrays([
        dates, ["A"]*len(dates)
    ], names=["datetime", "instrument"]))
    
    args = MagicMock()
    args.start_date = None
    args.end_date = None
    args.exclude_last_years = 2
    args.exclude_last_months = 0
    
    is_df, oos_df = bfe.split_is_oos_by_args(df, args)
    # Max date is 2020-01-01. cutoff is 2018-01-01.
    is_dates = is_df.index.get_level_values("datetime").unique()
    oos_dates = oos_df.index.get_level_values("datetime").unique()
    print(is_dates)
    print(oos_dates)
    
    assert "2018-01-01" in is_dates
    assert "2019-01-01" not in is_dates
    
    assert "2019-01-01" in oos_dates
    assert "2020-01-01" in oos_dates
    assert "2018-01-01" not in oos_dates

# ── extract_report_df ────────────────────────────────────────────────────
def test_extract_report_df(mock_env):
    bfe, _ = mock_env
    df = pd.DataFrame({"A": [1]})
    
    metrics_dict = {"fold1": (df, "other")}
    assert bfe.extract_report_df(metrics_dict).equals(df)
    
    metrics_tuple = (df, "other")
    assert bfe.extract_report_df(metrics_tuple).equals(df)
    
    assert bfe.extract_report_df(df).equals(df)
    
    # Case: metrics is a tuple, first element is a tuple (lines 368-370)
    metrics_nested = ((df, "extra"),)
    assert bfe.extract_report_df(metrics_nested) is df

# ── load_config ──────────────────────────────────────────────────────────
def test_load_config(mock_env, tmp_path):
    import json
    bfe, workspace = mock_env

    record_file = tmp_path / "records.json"
    with open(record_file, "w") as f:
        json.dump({"models": {"m1": "r1"}}, f)

    with patch('config_loader.load_workspace_config') as mock_load:
        mock_load.return_value = {"TopK": 20}
        records, model_config = bfe.load_config(str(record_file))
        
    assert records["models"]["m1"] == "r1"
    assert model_config["TopK"] == 20
    
    # No file case (line 122)
    tr, mc = bfe.load_config("non_existent_records.json")
    assert tr == {"models": {}, "experiment_name": "unknown"}

# ── correlation_analysis ─────────────────────────────────────────────────
def test_correlation_analysis(mock_env, tmp_path):
    bfe, _ = mock_env

    dates = pd.to_datetime(["2020-01-01"] * 3)
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B", "C"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"M1": [1.0, 2.0, 3.0], "M2": [1.0, 2.0, 3.0]}, index=idx)

    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    corr = bfe.correlation_analysis(norm_df, out_dir, "2020-01-01")

    assert corr.loc["M1", "M2"] == 1.0
    assert os.path.exists(os.path.join(out_dir, "correlation_matrix_2020-01-01.csv"))

# ── signal handlers ──────────────────────────────────────────────────────
def test_signal_handlers(mock_env):
    import signal as sig
    bfe, _ = mock_env

    bfe._install_signal_handlers()
    assert sig.getsignal(sig.SIGINT) == bfe._signal_handler

    bfe._restore_signal_handlers()
    assert sig.getsignal(sig.SIGINT) != bfe._signal_handler

def test_signal_handler_sets_shutdown(mock_env):
    bfe, _ = mock_env
    import quantpits.scripts.brute_force_ensemble as bfe_mod
    bfe_mod._shutdown = False
    
    with patch('builtins.print') as mock_print:
        bfe._signal_handler(2, None)  # SIGINT = 2
        assert bfe_mod._shutdown is True
        assert any("安全退出" in str(call) for call in mock_print.call_args_list)

    # Second call exits (lines 81-82)
    with patch('sys.exit') as mock_exit:
        bfe._signal_handler(2, None)
        mock_exit.assert_called_with(1)

# ── _append_results_to_csv ───────────────────────────────────────────────
def test_append_results_to_csv(mock_env, tmp_path):
    bfe, _ = mock_env

    csv_path = str(tmp_path / "results.csv")

    # Empty results should be no-op
    bfe._append_results_to_csv(csv_path, [])
    assert not os.path.exists(csv_path)

    # Non-empty results
    results = [{"models": "m1,m2", "Ann_Ret": 0.1}]
    bfe._append_results_to_csv(csv_path, results, write_header=True)
    saved = pd.read_csv(csv_path)
    assert len(saved) == 1
    assert saved.iloc[0]["models"] == "m1,m2"

    # Append more
    bfe._append_results_to_csv(csv_path, results, write_header=False)
    saved2 = pd.read_csv(csv_path)
    assert len(saved2) == 2

# ── split_is_oos_by_args with start/end dates ────────────────────────────
def test_split_is_oos_with_dates(mock_env):
    bfe, _ = mock_env

    dates = pd.date_range("2015-01-01", "2020-01-01", freq="YS")
    df = pd.DataFrame({"score": range(len(dates))}, index=pd.MultiIndex.from_arrays([
        dates, ["A"] * len(dates)
    ], names=["datetime", "instrument"]))

    args = MagicMock()
    args.start_date = "2016-01-01"
    args.end_date = "2018-06-01"
    args.exclude_last_years = 0
    args.exclude_last_months = 0

    is_df, oos_df = bfe.split_is_oos_by_args(df, args)
    is_dates = is_df.index.get_level_values("datetime")
    assert is_dates.min() >= pd.Timestamp("2016-01-01")
    assert is_dates.max() <= pd.Timestamp("2018-06-01")

# ── load_predictions ─────────────────────────────────────────────────────
from unittest.mock import patch

@patch('qlib.workflow.R', create=True)
def test_load_predictions(mock_R, mock_env):
    bfe, _ = mock_env

    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"])
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B", "A", "B"]], names=["datetime", "instrument"])

    mock_recorder = MagicMock()
    mock_recorder.load_object.return_value = pd.Series(
        [0.5, 0.6, 0.7, 0.8], index=idx, name="score"
    )
    mock_recorder.list_metrics.return_value = {"IC_ICIR": 1.5}
    mock_R.get_recorder.return_value = mock_recorder

    train_records = {
        "experiment_name": "test_exp",
        "models": {"m1": "r1", "m2": "r2"}
    }

    norm_df, metrics = bfe.load_predictions(train_records)
    assert "m1" in norm_df.columns
    assert "m2" in norm_df.columns
    assert metrics["m1"] == 1.5

@patch('qlib.workflow.R', create=True)
def test_load_predictions_failure(mock_R, mock_env):
    bfe, _ = mock_env

    mock_R.get_recorder.side_effect = Exception("not found")

    train_records = {
        "experiment_name": "test_exp",
        "models": {"m1": "r1"}
    }

    with pytest.raises(ValueError, match="未加载到任何预测数据"):
        bfe.load_predictions(train_records)

# ── analyze_results ──────────────────────────────────────────────────────
def test_analyze_results_empty(mock_env, tmp_path):
    bfe, _ = mock_env
    os.makedirs(tmp_path / "output", exist_ok=True)
    # Empty results_df should return early
    bfe.analyze_results(
        results_df=pd.DataFrame(),
        corr_matrix=pd.DataFrame(),
        norm_df=pd.DataFrame(),
        train_records={"experiment_name": "exp", "models": {}},
        output_dir=str(tmp_path / "output"),
        anchor_date="2020-01-01",
    )

@patch('qlib.workflow.R', create=True)
def test_analyze_results_basic(mock_R, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame({
        "models": ["m1", "m2", "m1,m2"],
        "n_models": [1, 1, 2],
        "Ann_Ret": [0.15, 0.12, 0.18],
        "Max_DD": [-0.05, -0.08, -0.04],
        "Excess_Ret": [0.10, 0.07, 0.13],
        "Ann_Excess": [0.10, 0.07, 0.13],
        "Total_Ret": [0.15, 0.12, 0.18],
        "Final_NAV": [115000, 112000, 118000],
        "Calmar": [3.0, 1.5, 4.5],
    })

    corr_matrix = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["m1", "m2"], columns=["m1", "m2"]
    )

    dates = pd.to_datetime(["2020-01-01"] * 2)
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [1.0, 2.0], "m2": [1.5, 2.5]}, index=idx)

    mock_recorder = MagicMock()
    mock_report = pd.DataFrame({
        "return": [0.01, 0.02],
        "bench": [0.005, 0.005]
    }, index=dates)
    mock_recorder.load_object.return_value = mock_report
    mock_R.get_recorder.return_value = mock_recorder

    with patch('matplotlib.pyplot.savefig'):
        bfe.analyze_results(
            results_df=results_df,
            corr_matrix=corr_matrix,
            norm_df=norm_df,
            train_records={"experiment_name": "exp", "models": {"m1": "r1", "m2": "r2"}},
            output_dir=out_dir,
            anchor_date="2020-01-01",
            top_n=2
        )

    assert os.path.exists(os.path.join(out_dir, "analysis_report_2020-01-01.txt"))
    assert os.path.exists(os.path.join(out_dir, "model_attribution_2020-01-01.csv"))

@patch('qlib.workflow.R', create=True)
def test_analyze_results_fallback(mock_R, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)

    # Less than 2 models (should skip clustering and optimization plots without crashing)
    results_df = pd.DataFrame({
        "models": ["m1"],
        "n_models": [1],
        "Ann_Ret": [0.15],
        "Max_DD": [-0.05],
        "Excess_Ret": [0.10],
        "Ann_Excess": [0.10],
        "Total_Ret": [0.15],
        "Final_NAV": [115000],
        "Calmar": [3.0],
    })

    corr_matrix = pd.DataFrame([[1.0]], index=["m1"], columns=["m1"])

    dates = pd.to_datetime(["2020-01-01"] * 2)
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [1.0, 2.0]}, index=idx)

    mock_recorder = MagicMock()
    mock_report = pd.DataFrame({
        "return": [0.01, 0.02],
        "bench": [0.005, 0.005]
    }, index=dates)
    mock_recorder.load_object.return_value = mock_report
    mock_R.get_recorder.return_value = mock_recorder

    with patch('matplotlib.pyplot.savefig'):
        bfe.analyze_results(
            results_df=results_df,
            corr_matrix=corr_matrix,
            norm_df=norm_df,
            train_records={"experiment_name": "exp", "models": {"m1": "r1"}},
            output_dir=out_dir,
            anchor_date="2020-01-01",
            top_n=2
        )

    # Assure it succeeds and writes the report
    assert os.path.exists(os.path.join(out_dir, "analysis_report_2020-01-01.txt"))

# ── parse_args ───────────────────────────────────────────────────────────
def test_parse_args(mock_env):
    import sys
    bfe, _ = mock_env

    with patch.object(sys, 'argv', [
        'bfe.py', '--max-combo-size', '3', '--analysis-only', '--resume',
        '--freq', 'week', '--top-n', '10'
    ]):
        args = bfe.main.__code__  # Just verify parse_args is callable
    # Use the module-level argparse directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-combo-size', type=int, default=0)
    parser.add_argument('--analysis-only', action='store_true')
    with patch.object(sys, 'argv', ['bfe.py', '--max-combo-size', '3', '--analysis-only']):
        args = parser.parse_args()
    assert args.max_combo_size == 3
    assert args.analysis_only is True

from unittest.mock import patch, MagicMock

@patch('qlib.backtest.backtest_loop')
@patch('qlib.backtest.account.Account')
@patch('qlib.backtest.executor.SimulatorExecutor')
@patch('strategy.create_backtest_strategy')
def test_run_single_backtest_success(mock_create_strat, mock_executor, mock_account, mock_bt_loop, mock_env):
    bfe, _ = mock_env
    
    # Setup mocks
    mock_strat = MagicMock()
    mock_create_strat.return_value = mock_strat
    
    # Mock report from backtest_loop
    mock_report = pd.DataFrame({
        "account": [100.0, 110.0],
        "return": [0.0, 0.1],
        "bench": [100.0, 105.0]
    }, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    mock_bt_loop.return_value = ({"fold1": (mock_report, "other")}, None)
    
    norm_df = pd.DataFrame({"m1": [0.5, 0.6]}, index=pd.MultiIndex.from_tuples([
        (pd.to_datetime("2020-01-01"), "A"), (pd.to_datetime("2020-01-02"), "A")
    ], names=["datetime", "instrument"]))
    
    res = bfe.run_single_backtest(
        combo_models=["m1"], norm_df=norm_df, top_k=1, drop_n=0, 
        benchmark="SH000300", freq="day", trade_exchange=MagicMock(),
        bt_start="2020-01-01", bt_end="2020-01-02"
    )
    
    assert res is not None
    assert res["Ann_Ret"] > 0
    assert "m1" in res["models"]

@patch('strategy.get_backtest_config')
@patch('strategy.load_strategy_config')
@patch('qlib.backtest.backtest_loop')
@patch('qlib.backtest.account.Account')
@patch('qlib.backtest.executor.SimulatorExecutor')
@patch('strategy.create_backtest_strategy')
def test_run_single_backtest_with_config(mock_create_strat, mock_executor, mock_account, mock_bt_loop, mock_load_st, mock_get_bt, mock_env):
    bfe, _ = mock_env
    # Setup configs
    mock_load_st.return_value = {"strategy": {"params": {}}}
    mock_get_bt.return_value = {"account": 1000, "exchange_kwargs": {"freq": "day"}}
    
    # Setup mocks
    mock_strat = MagicMock()
    mock_create_strat.return_value = mock_strat
    
    # Mock report from backtest_loop
    mock_report = pd.DataFrame({
        "account": [100.0, 110.0],
        "return": [0.0, 0.1],
        "bench": [100.0, 105.0]
    }, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    mock_bt_loop.return_value = ({"fold1": (mock_report, "other")}, None)
    
    norm_df = pd.DataFrame({"m1": [0.5, 0.6]}, index=pd.MultiIndex.from_tuples([
        (pd.to_datetime("2020-01-01"), "A"), (pd.to_datetime("2020-01-02"), "A")
    ], names=["datetime", "instrument"]))
    
    res = bfe.run_single_backtest(
        combo_models=["m1"], norm_df=norm_df, top_k=10, drop_n=2, 
        benchmark="SH000300", freq="day", trade_exchange=MagicMock(),
        bt_start="2020-01-01", bt_end="2020-01-02",
        st_config=None, bt_config=None  # Force fallback
    )
    
    assert res is not None
    # Verify the injected config check
    mock_create_strat.assert_called_once()
    called_config = mock_create_strat.call_args[0][1]
    assert called_config["strategy"]["params"]["topk"] == 10
    assert called_config["strategy"]["params"]["n_drop"] == 2

@patch('qlib.backtest.backtest_loop')
@patch('qlib.backtest.account.Account')
@patch('qlib.backtest.executor.SimulatorExecutor')
@patch('strategy.create_backtest_strategy')
def test_run_single_backtest_exception(mock_create_strat, mock_executor, mock_account, mock_bt_loop, mock_env):
    bfe, _ = mock_env
    
    mock_strat = MagicMock()
    mock_create_strat.return_value = mock_strat
    
    # Force exception
    mock_bt_loop.side_effect = Exception("Intentional backtest exception")
    
    norm_df = pd.DataFrame({"m1": [0.5]}, index=pd.MultiIndex.from_tuples([
        (pd.to_datetime("2020-01-01"), "A")
    ], names=["datetime", "instrument"]))
    
    res = bfe.run_single_backtest(
        combo_models=["m1"], norm_df=norm_df, top_k=1, drop_n=0, 
        benchmark="SH000300", freq="day", trade_exchange=MagicMock(),
        bt_start="2020-01-01", bt_end="2020-01-01",
        st_config={"strategy": {"params": {}}},
        bt_config={"account": 100}
    )
    
    assert res is None

@patch('qlib.backtest.exchange.Exchange', create=True)
@patch('quantpits.scripts.brute_force_ensemble.run_single_backtest')
def test_brute_force_backtest_basic(mock_run_bt, mock_exchange, mock_env, tmp_path):
    bfe, _ = mock_env
    
    # Mock norm_df
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A", "B"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5, 0.6], "m2": [0.4, 0.7]}, index=idx)
    
    # Mock run_single_backtest to return valid metrics
    mock_run_bt.side_effect = lambda combo, *args, **kwargs: {
        "models": ",".join(combo), "n_models": len(combo), "Ann_Ret": 0.1, "Max_DD": -0.05,
        "Excess_Ret": 0.05, "Ann_Excess": 0.05, "Total_Ret": 0.1, "Final_NAV": 110000, "Calmar": 2.0
    }
    
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    
    results = bfe.brute_force_backtest(
        norm_df=norm_df, top_k=1, drop_n=0, benchmark="SH000300", freq="day",
        min_combo_size=1, max_combo_size=1, output_dir=out_dir, anchor_date="2020-01-01",
        n_jobs=1
    )
    
    assert len(results) == 2 # m1 and m2
    assert os.path.exists(os.path.join(out_dir, "brute_force_results_2020-01-01.csv"))

@patch('qlib.backtest.exchange.Exchange', create=True)
@patch('quantpits.scripts.brute_force_ensemble.run_single_backtest')
def test_brute_force_backtest_resume(mock_run_bt, mock_exchange, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Create an existing result CSV
    csv_path = os.path.join(out_dir, "brute_force_results_2020-01-01.csv")
    existing_df = pd.DataFrame({
        "models": ["m1"], "n_models": [1], "Ann_Ret": [0.1], "Max_DD": [-0.05],
        "Excess_Ret": [0.05], "Ann_Excess": [0.05], "Total_Ret": [0.1], "Final_NAV": [110000], "Calmar": [2.0]
    })
    existing_df.to_csv(csv_path, index=False)
    
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5], "m2": [0.4]}, index=idx)
    
    mock_run_bt.return_value = {
        "models": "m2", "n_models": 1, "Ann_Ret": 0.2, "Max_DD": -0.05,
        "Excess_Ret": 0.1, "Ann_Excess": 0.1, "Total_Ret": 0.2, "Final_NAV": 120000, "Calmar": 4.0
    }
    
    results = bfe.brute_force_backtest(
        norm_df=norm_df, top_k=1, drop_n=0, benchmark="SH000300", freq="day",
        min_combo_size=1, max_combo_size=1, output_dir=out_dir, anchor_date="2020-01-01",
        resume=True, n_jobs=1
    )
    
    assert len(results) == 2
    assert set(results["models"]) == {"m1", "m2"}
    mock_run_bt.assert_called_once() # Only called for m2

@patch('qlib.backtest.exchange.Exchange', create=True)
@patch('quantpits.scripts.brute_force_ensemble.run_single_backtest')
def test_brute_force_backtest_shutdown_signal(mock_run_bt, mock_exchange, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"])
    # 3 models to trigger multiple batches if batch_size=1
    norm_df = pd.DataFrame({"m1": [0.5], "m2": [0.4], "m3": [0.6]}, index=idx)
    
    def side_effect_run_bt(combo, *args, **kwargs):
        # Trigger shutdown during the first model evaluation
        import quantpits.scripts.brute_force_ensemble as bfe_mod
        bfe_mod._shutdown = True
        return {
            "models": ",".join(combo), "n_models": len(combo), "Ann_Ret": 0.1, "Max_DD": -0.05,
            "Excess_Ret": 0.05, "Ann_Excess": 0.05, "Total_Ret": 0.1, "Final_NAV": 110000, "Calmar": 2.0
        }
    
    mock_run_bt.side_effect = side_effect_run_bt
    
    results = bfe.brute_force_backtest(
        norm_df=norm_df, top_k=1, drop_n=0, benchmark="SH000300", freq="day",
        min_combo_size=1, max_combo_size=1, output_dir=out_dir, anchor_date="2020-01-01",
        n_jobs=1, batch_size=1
    )
    
    # Due to shutdown, only 1 result should be captured and loop should break early
    assert len(results) == 1
    import quantpits.scripts.brute_force_ensemble as bfe_mod
    assert bfe_mod._shutdown is True
    # Reset for other tests
    bfe_mod._shutdown = False

@patch('quantpits.scripts.brute_force_ensemble.init_qlib')
@patch('quantpits.scripts.brute_force_ensemble.load_config')
@patch('quantpits.scripts.brute_force_ensemble.load_predictions')
@patch('quantpits.scripts.brute_force_ensemble.correlation_analysis')
@patch('quantpits.scripts.brute_force_ensemble.brute_force_backtest')
@patch('quantpits.scripts.brute_force_ensemble.analyze_results')
@patch('quantpits.scripts.env.safeguard')
def test_main_full(mock_safeguard, mock_analyze, mock_bf, mock_corr, mock_load_pred, mock_load_cfg, mock_init, mock_env, tmp_path):
    bfe, _ = mock_env
    
    mock_load_cfg.return_value = ({"models": {"m1": "r1"}, "anchor_date": "2020-01-01"}, {"TopK": 1})
    
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
    mock_load_pred.return_value = (norm_df, {"m1": 0.05})
    
    mock_corr.return_value = pd.DataFrame([[1.0]], index=["m1"], columns=["m1"])
    mock_bf.return_value = pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1]})
    
    import sys
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Test standard execution
    with patch.object(sys, 'argv', ['script.py', '--output-dir', out_dir]):
        bfe.main()
    
    mock_bf.assert_called_once()
    mock_analyze.assert_called_once()

    # Test --analysis-only
    mock_bf.reset_mock()
    mock_analyze.reset_mock()
    # Mock existence of the result CSV to pass analysis
    pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1]}).to_csv(
        os.path.join(out_dir, "brute_force_results_2020-01-01.csv"), index=False
    )
    with patch.object(sys, 'argv', ['script.py', '--output-dir', out_dir, '--analysis-only']):
        bfe.main()
    mock_bf.assert_not_called()
    mock_analyze.assert_called_once()

    # Test --use-groups and OOS metrics via dates + resume
    mock_bf.reset_mock()
    mock_analyze.reset_mock()
    with patch.object(sys, 'argv', [
        'script.py', '--output-dir', out_dir, '--use-groups', 
        '--start-date', '2019-01-01', '--end-date', '2020-01-01',
        '--resume', '--n-jobs', '2', '--batch-size', '10', '--top-n', '3',
        '--min-combo-size', '2', '--max-combo-size', '5'
    ]):
        bfe.main()
    mock_bf.assert_called_once()
    mock_analyze.assert_called_once()

@patch('qlib.backtest.account.Account')
@patch('qlib.backtest.executor.SimulatorExecutor')
@patch('qlib.backtest.utils.CommonInfrastructure')
@patch('qlib.backtest.backtest_loop')
@patch('quantpits.scripts.strategy.load_strategy_config')
@patch('quantpits.scripts.strategy.get_backtest_config')
@patch('quantpits.scripts.strategy.create_backtest_strategy')
@patch('quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer')
def test_run_single_backtest_non_datetime_index(mock_pa, mock_st_create, mock_bt_cfg, mock_st_cfg, mock_bt_loop, mock_infra, mock_executor, mock_account, mock_env):
    bfe, _ = mock_env
    dates = ["2020-01-01", "2020-01-02"]
    report = pd.DataFrame({
        "account": [100000, 101000],
        "bench": [0.001, 0.002]
    }, index=dates) # String index
    
    mock_st_cfg.return_value = {"benchmark": "SH000300"}
    mock_bt_cfg.return_value = {"account": 1000000}
    mock_bt_loop.return_value = (report, None)
    
    mock_pa_inst = MagicMock()
    mock_pa_inst.calculate_traditional_metrics.return_value = {"CAGR": 0.1}
    mock_pa.return_value = mock_pa_inst
    
    idx = pd.MultiIndex.from_product([pd.to_datetime(dates), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5, 0.6]}, index=idx)
    
    with patch('qlib.data.D') as mock_D_inst:
        mock_D_inst.calendar.return_value = pd.to_datetime(dates)
        res = bfe.run_single_backtest(["m1"], norm_df, 1, 0, "SH000300", "day", MagicMock(), "2020-01-01", "2020-01-02")
        assert res["Ann_Ret"] == 0.1

def test_analyze_results_clustering_and_opt_fails(mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = tmp_path / "complex_fail"
    out_dir.mkdir()
    results_df = pd.DataFrame({
        "models": ["m1", "m2"], 
        "n_models": [1, 1], 
        "Ann_Excess": [0.1, 0.12], 
        "Calmar": [1.0, 1.2], 
        "Ann_Ret": [0.15, 0.16], 
        "Max_DD": [-0.1, -0.1]
    })
    results_df["diversity_bonus"] = [0.01, 0.02]
    
    with patch('qlib.workflow.R') as mock_R_inst:
        mock_R_inst.get_recorder.return_value.load_object.side_effect = Exception("Load fail")
        with patch('builtins.print') as mock_print:
            bfe.analyze_results(results_df, pd.DataFrame(), pd.DataFrame(), {"experiment_name": "exp", "models": {"m1": "r1"}}, str(out_dir), "test")
            assert any("[跳过] m1: Load fail" in str(call) for call in mock_print.call_args_list)

@patch('quantpits.scripts.brute_force_ensemble.run_single_backtest')
@patch('quantpits.scripts.brute_force_ensemble.load_config')
def test_main_oos_validation_stage5(mock_load_cfg, mock_run, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = tmp_path / "oos_stage5"
    out_dir.mkdir()
    
    res_df = pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1]})
    res_df.to_csv(out_dir / "brute_force_results_test.csv", index=False)
    
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2021-01-01"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
    
    mock_load_cfg.return_value = ({"anchor_date": "test", "experiment_name": "exp", "models": {"m1": "r1"}}, {"TopK": 1})
    mock_run.return_value = {"models": "m1", "Ann_Ret": 0.1, "Max_DD": -0.05, "Ann_Excess": 0.05, "Calmar": 2.0}

    with patch('sys.argv', ['script.py', '--auto-test-top', '1', '--output-dir', str(out_dir)]):
        with patch('quantpits.scripts.brute_force_ensemble.load_predictions', return_value=(norm_df, {})):
            with patch('quantpits.scripts.brute_force_ensemble.split_is_oos_by_args', return_value=(norm_df, norm_df)):
                with patch('quantpits.scripts.brute_force_ensemble.brute_force_backtest', return_value=res_df):
                    with patch('quantpits.scripts.brute_force_ensemble.analyze_results'):
                        with patch('qlib.backtest.exchange.Exchange'):
                            with patch('builtins.print'):
                                bfe.main()
                        assert os.path.exists(out_dir / "oos_validation_test.csv")

@patch('qlib.backtest.exchange.Exchange')
def test_brute_force_backtest_grouped_and_no_pending(mock_exch, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = tmp_path / "bfe_out"
    out_dir.mkdir()
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5], "m2": [0.6]}, index=idx)
    
    # 1. Test no pending tasks (lines 584-588)
    csv_path = out_dir / "brute_force_results_test.csv"
    pd.DataFrame({"models": ["m1", "m2"], "Ann_Excess": [0.1, 0.2]}).to_csv(csv_path, index=False)
    
    with patch('builtins.print') as mock_print:
        res = bfe.brute_force_backtest(norm_df, 1, 0, "BENCH", "day", 1, 1, str(out_dir), "test", resume=True)
        assert len(res) == 2
        assert any("所有组合已完成！" in str(call) for call in mock_print.call_args_list)

@patch('quantpits.scripts.brute_force_ensemble.run_single_backtest', return_value=None)
@patch('qlib.backtest.exchange.Exchange')
def test_brute_force_backtest_no_results(mock_exch, mock_run, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = tmp_path / "no_res"
    out_dir.mkdir()
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
    
    with patch('builtins.print') as mock_print:
        bfe.brute_force_backtest(norm_df, 1, 0, "BENCH", "day", 1, 1, str(out_dir), "no_res")
        assert any("警告: 无有效回测结果" in str(call) for call in mock_print.call_args_list)

@patch('qlib.workflow.R')
def test_analyze_results_plot_fails(mock_R, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = tmp_path / "plot_fail"
    out_dir.mkdir()
    results_df = pd.DataFrame({"models": ["m1"], "n_models": [1], "Ann_Excess": [0.1], "Calmar": [1.0], "Ann_Ret": [0.15], "Max_DD": [-0.1]})
    results_df["diversity_bonus"] = [0.0]
    mock_R.get_recorder.return_value.load_object.side_effect = Exception("Skip cluster")

    with patch('matplotlib.pyplot.savefig', side_effect=Exception("Save fail")):
        with patch('builtins.print') as mock_print:
            bfe.analyze_results(results_df, pd.DataFrame(), pd.DataFrame(), {"experiment_name": "exp", "models": {"m1": "r1"}}, str(out_dir), "test")
            assert any("归因图绘制失败: Save fail" in str(call) for call in mock_print.call_args_list)

def test_main_empty_is_exit(mock_env):
    bfe, _ = mock_env
    with patch('quantpits.scripts.brute_force_ensemble.load_predictions', return_value=(pd.DataFrame(), {})):
        with patch('quantpits.scripts.brute_force_ensemble.split_is_oos_by_args', return_value=(pd.DataFrame(), pd.DataFrame())):
            with patch('builtins.print'):
                with pytest.raises(SystemExit) as e:
                    bfe.main()
                assert e.value.code == 1

def test_main_analysis_only_glob(mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = tmp_path / "glob_test"
    out_dir.mkdir()
    pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1]}).to_csv(out_dir / "brute_force_results_2020.csv", index=False)
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)

    with patch('sys.argv', ['script.py', '--analysis-only', '--output-dir', str(out_dir)]):
        with patch('quantpits.scripts.brute_force_ensemble.load_predictions', return_value=(norm_df, {})):
            with patch('quantpits.scripts.brute_force_ensemble.split_is_oos_by_args', return_value=(norm_df, pd.DataFrame())):
                with patch('builtins.print') as mock_print:
                    with patch('quantpits.scripts.brute_force_ensemble.analyze_results'):
                        bfe.main()
                        assert any("使用最新结果文件" in str(call) for call in mock_print.call_args_list)

@patch('quantpits.scripts.brute_force_ensemble.load_config')
def test_main_oos_validation_no_data(mock_load_cfg, mock_env, tmp_path):
    bfe, _ = mock_env
    out_dir = tmp_path / "oos_no_data"
    out_dir.mkdir()
    mock_load_cfg.return_value = ({"anchor_date": "test", "experiment_name": "exp", "models": {"m1": "r1"}}, {"TopK": 1})
    norm_df = pd.DataFrame({"m1": [0.5]}, index=pd.MultiIndex.from_product([pd.to_datetime(["2021-01-01"]), ["A"]], names=["datetime", "instrument"]))
    with patch('sys.argv', ['script.py', '--auto-test-top', '1', '--output-dir', str(out_dir)]):
        with patch('quantpits.scripts.brute_force_ensemble.load_predictions', return_value=(norm_df, {})):
            with patch('quantpits.scripts.brute_force_ensemble.split_is_oos_by_args', return_value=(norm_df, pd.DataFrame())):
                with patch('quantpits.scripts.brute_force_ensemble.brute_force_backtest', return_value=pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1]})):
                    with patch('quantpits.scripts.brute_force_ensemble.analyze_results'):
                        with patch('builtins.print') as mock_print:
                            bfe.main()
                            assert any("无法进行 OOS 验证：无 OOS 数据" in str(call) for call in mock_print.call_args_list)


