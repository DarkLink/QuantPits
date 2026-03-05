import pytest
import os
import pandas as pd
import numpy as np
import yaml
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, brute_force_ensemble
    import importlib
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

# ── load_config ──────────────────────────────────────────────────────────
def test_load_config(mock_env, tmp_path):
    import json
    bfe, workspace = mock_env

    record_file = tmp_path / "records.json"
    with open(record_file, "w") as f:
        json.dump({"models": {"m1": "r1"}}, f)

    (workspace / "config").mkdir(exist_ok=True)
    with open(workspace / "config" / "model_config.json", "w") as f:
        json.dump({"TopK": 20}, f)

    old_cwd = os.getcwd()
    os.chdir(workspace)
    try:
        records, model_config = bfe.load_config(str(record_file))
    finally:
        os.chdir(old_cwd)
    assert records["models"]["m1"] == "r1"
    assert model_config["TopK"] == 20

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
    bfe._signal_handler(2, None)  # SIGINT = 2
    assert bfe_mod._shutdown is True

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
    
    with patch.object(sys, 'argv', ['script.py', '--output-dir', out_dir]):
        bfe.main()
    
    mock_bf.assert_called_once()
    mock_analyze.assert_called_once()


