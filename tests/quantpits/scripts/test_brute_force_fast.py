import pytest
import os
import pandas as pd
import numpy as np
import json
import yaml
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, brute_force_fast
    import importlib
    importlib.reload(env)
    importlib.reload(brute_force_fast)
    
    yield brute_force_fast, workspace

def test_zscore_norm(mock_env):
    bff, _ = mock_env
    dates = pd.to_datetime(["2020-01-01"]*3 + ["2020-01-02"]*3)
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B", "C"]*2], names=["datetime", "instrument"])
    series = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=idx)
    normed = bff.zscore_norm(series)
    assert np.isclose(normed.xs("2020-01-01", level="datetime").mean(), 0)
    assert np.isclose(normed.xs("2020-01-01", level="datetime").std(), 1)

def test_vectorized_topk_backtest_single(mock_env):
    bff, _ = mock_env
    # T=5, N=3. Top_k=1
    combo_score = np.array([
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.9],
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1]
    ], dtype=np.float32)
    returns = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0]
    ], dtype=np.float32)
    
    # rebalance_freq=1 (daily)
    net_ret = bff._vectorized_topk_backtest_single(combo_score, returns, top_k=1, cost_rate=0.0, rebalance_freq=1)
    print(f"Net ret (no cost): {net_ret}")
    assert np.allclose(net_ret, 0.1)
    
    # with cost
    net_ret_cost = bff._vectorized_topk_backtest_single(combo_score, returns, top_k=1, cost_rate=0.01, rebalance_freq=1)
    print(f"Net ret (cost): {net_ret_cost}")
    # Should have 4 rebalances (T=0 no cost, T=1,2,3,4 have cost if selection changes)
    assert np.isclose(net_ret_cost[0], 0.1)
    assert np.allclose(net_ret_cost[1:], 0.09)

def test_compute_metrics(mock_env):
    bff, _ = mock_env
    net_ret = np.array([0.01] * 100)
    bench_ret = np.array([0.005] * 100)
    metrics = bff.compute_metrics(net_ret, bench_ret)
    assert metrics["Ann_Ret"] > 0
    assert metrics["Total_Ret"] > 0
    assert metrics["Excess_Ret"] > 0

@patch('qlib.data.D', create=True)
def test_load_returns_matrix(mock_D, mock_env):
    bff, _ = mock_env
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5, 0.6, 0.7, 0.8]}, index=idx)
    
    mock_D.features.side_effect = [
        pd.DataFrame({"return": [0.01, 0.02, 0.03, 0.04]}, index=idx),
        pd.DataFrame({"$close": [100.0, 101.0]}, index=pd.MultiIndex.from_product([dates, ["SH000300"]], names=["datetime", "instrument"]))
    ]
    
    ret_wide, bench_ret, common_dates, insts = bff.load_returns_matrix(norm_df)
    assert ret_wide.shape == (2, 2)
    assert len(bench_ret) == 2

@patch('quantpits.scripts.brute_force_fast.init_qlib', create=True)
@patch('quantpits.scripts.brute_force_fast.load_config')
@patch('quantpits.scripts.brute_force_fast.load_predictions')
@patch('quantpits.scripts.brute_force_fast.load_returns_matrix')
@patch('quantpits.scripts.brute_force_fast.brute_force_fast_backtest')
@patch('quantpits.scripts.env.safeguard')
def test_main_full(mock_safeguard, mock_bf, mock_load_ret, mock_load_pred, mock_load_cfg, mock_init, mock_env, tmp_path):
    bff, _ = mock_env
    
    mock_load_cfg.return_value = ({"experiment_name": "Exp1", "models": {"m1": "r1"}}, {"TopK": 1})
    
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
    mock_load_pred.return_value = (norm_df, {"m1": 0.05})
    
    mock_load_ret.return_value = (pd.DataFrame(), pd.Series(dtype=float), pd.DatetimeIndex([]), [])
    mock_bf.return_value = pd.DataFrame({
        "models": ["m1"], "n_models": [1], "Ann_Excess": [0.1], "Ann_Ret": [0.15],
        "Max_DD": [-0.05], "Excess_Ret": [0.12], "Total_Ret": [0.15], "Final_NAV": [115000],
        "Calmar": [3.0], "Sharpe": [2.0]
    })
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--max-combo-size', '1']):
        bff.main()
    
    mock_bf.assert_called_once()
    mock_init.assert_called_once()

@patch('qlib.workflow.R')
def test_load_predictions(mock_R, mock_env):
    bff, _ = mock_env
    train_records = {"experiment_name": "Exp1", "models": {"m1": "rid1"}}
    
    mock_recorder = MagicMock()
    idx = pd.MultiIndex.from_tuples([("2020-01-01", "A")], names=["datetime", "instrument"])
    mock_recorder.load_object.return_value = pd.Series([0.5], index=idx, name="score")
    mock_recorder.list_metrics.return_value = {"ICIR": 0.1}
    mock_R.get_recorder.return_value = mock_recorder
    
    norm_df, metrics = bff.load_predictions(train_records)
    assert "m1" in norm_df.columns
    assert metrics["m1"] == 0.1

def test_load_combo_groups(mock_env, tmp_path):
    bff, _ = mock_env
    cfg_path = tmp_path / "groups.yaml"
    cfg_path.write_text(yaml.dump({"groups": {"G1": ["m1", "m2"]}}))
    
    groups = bff.load_combo_groups(str(cfg_path), ["m1", "m2", "m3"])
    assert groups["G1"] == ["m1", "m2"]

def test_generate_grouped_combinations(mock_env):
    bff, _ = mock_env
    groups = {"G1": ["m1", "m2"], "G2": ["m3"]}
    combos = bff.generate_grouped_combinations(groups, min_combo_size=2)
    assert len(combos) == 2
    assert ("m1", "m3") in combos

def test_brute_force_fast_backtest_basic(mock_env, tmp_path):
    bff, _ = mock_env
    scores_np = {"m1": np.array([[0.5]], dtype=np.float32)}
    returns_np = np.array([[0.1]], dtype=np.float32)
    bench_ret = np.array([0.05], dtype=np.float32)
    
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    
    results = bff.brute_force_fast_backtest(
        scores_np, returns_np, ["m1"], bench_ret,
        top_k=1, freq="day", cost_rate=0.0, batch_size=10,
        min_combo_size=1, max_combo_size=1, output_dir=out_dir, anchor_date="2020-01-01"
    )
    assert len(results) == 1
    assert results.iloc[0]["models"] == "m1"

def test_split_is_oos_by_args(mock_env):
    bff, _ = mock_env
    dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
    df = pd.DataFrame({"m1": [0.5, 0.6]}, index=idx)
    
    args = MagicMock()
    args.start_date = "2020-01-01"
    args.end_date = "2021-12-31"
    args.exclude_last_years = 0
    args.exclude_last_months = 0
    
    is_df, oos_df = bff.split_is_oos_by_args(df, args)
    assert len(is_df) == 2
    assert len(oos_df) == 0

def test_brute_force_fast_backtest_resume(mock_env, tmp_path):
    bff, _ = mock_env
    out_dir = tmp_path / "output_resume"
    out_dir.mkdir()
    csv_path = out_dir / "brute_force_fast_results_2020-01-01.csv"
    
    # Create existing results
    df = pd.DataFrame({"models": ["m1"], "Ann_Excess": [0.1]})
    df.to_csv(csv_path, index=False)
    
    scores_np = {"m1": np.array([[0.5]], dtype=np.float32), "m2": np.array([[0.6]], dtype=np.float32)}
    returns_np = np.array([[0.1]], dtype=np.float32)
    bench_ret = np.array([0.05], dtype=np.float32)
    
    results = bff.brute_force_fast_backtest(
        scores_np, returns_np, ["m1", "m2"], bench_ret,
        top_k=1, freq="day", cost_rate=0.0, batch_size=10,
        min_combo_size=1, max_combo_size=1, output_dir=str(out_dir), anchor_date="2020-01-01",
        resume=True
    )
    # Should skip m1 and only run m2
    assert len(results) == 2
    assert "m2" in results["models"].values

def test_brute_force_fast_backtest_groups(mock_env, tmp_path):
    bff, _ = mock_env
    out_dir = tmp_path / "output_groups"
    out_dir.mkdir()
    
    group_cfg = out_dir / "groups.yaml"
    group_cfg.write_text(yaml.dump({"groups": {"G1": ["m1"]}}))
    
    scores_np = {"m1": np.array([[0.5]], dtype=np.float32)}
    returns_np = np.array([[0.1]], dtype=np.float32)
    bench_ret = np.array([0.05], dtype=np.float32)
    
    results = bff.brute_force_fast_backtest(
        scores_np, returns_np, ["m1"], bench_ret,
        top_k=1, freq="day", cost_rate=0.0, batch_size=10,
        min_combo_size=1, max_combo_size=1, output_dir=str(out_dir), anchor_date="2020-01-01",
        use_groups=True, group_config=str(group_cfg)
    )
    assert len(results) == 1
    assert results.iloc[0]["models"] == "m1"

def test_init_gpu(mock_env):
    bff, _ = mock_env
    # Force no gpu
    bff._init_gpu(force_no_gpu=True)
    assert not bff._USE_GPU
    
    # Mock cupy
    with patch('builtins.__import__', side_effect=ImportError):
        bff._init_gpu(force_gpu=False)
        assert not bff._USE_GPU

def test_to_numpy(mock_env):
    bff, _ = mock_env
    arr = np.array([1])
    assert bff._to_numpy(arr) is arr
    
    # Mock GPU mode
    bff._USE_GPU = True
    mock_cp = MagicMock()
    mock_cp.ndarray = np.ndarray # simplify
    with patch.dict('sys.modules', {'cupy': mock_cp}):
        bff._to_numpy(arr)
def test_load_config_real(mock_env, tmp_path):
    bff, workspace = mock_env
    rec_file = workspace / "records.json"
    with open(rec_file, "w") as f:
        json.dump({"experiment_name": "Exp1", "models": {"m1": "r1"}}, f)
    
    cfg_dir = workspace / "config"
    cfg_dir.mkdir(exist_ok=True)
    with open(cfg_dir / "model_config.json", "w") as f:
        json.dump({"TopK": 50}, f)
        
    import os
    orig_cwd = os.getcwd()
    os.chdir(workspace)
    try:
        tr, mc = bff.load_config(str(rec_file))
        assert tr["experiment_name"] == "Exp1"
        assert mc["TopK"] == 50
    finally:
        os.chdir(orig_cwd)

@patch('qlib.data.D', create=True)
def test_load_returns_matrix_fail(mock_D, mock_env):
    bff, _ = mock_env
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5]}, index=idx)
    
    # First call for returns succeeds, second for benchmark fails
    mock_D.features.side_effect = [
        pd.DataFrame({"return": [0.01]}, index=idx),
        Exception("Benchmark load error")
    ]
    
    ret_wide, bench_ret, common_dates, insts = bff.load_returns_matrix(norm_df)
    assert len(bench_ret) == 1
    assert bench_ret.iloc[0] == 0.0

def test_init_gpu_force_gpu_fail(mock_env):
    bff, _ = mock_env
    with patch('builtins.__import__', side_effect=ImportError):
        with pytest.raises(SystemExit):
            bff._init_gpu(force_gpu=True)

def test_split_is_oos_extended(mock_env):
    bff, _ = mock_env
    dates = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
    df = pd.DataFrame({"m1": [0.5, 0.6, 0.7]}, index=idx)
    
    # Test exclude_last_years
    args_years = MagicMock(start_date=None, end_date=None, exclude_last_years=1, exclude_last_months=0)
    is_df, oos_df = bff.split_is_oos_by_args(df, args_years)
    assert len(is_df) == 2
    assert len(oos_df) == 1
    
    # Test exclude_last_months
    args_months = MagicMock(start_date=None, end_date=None, exclude_last_years=0, exclude_last_months=6)
    is_df, oos_df = bff.split_is_oos_by_args(df, args_months)
    assert len(is_df) == 2
    assert len(oos_df) == 1

    # Test end_date < cutoff_date
    args_end = MagicMock(start_date=None, end_date="2021-06-01", exclude_last_years=0, exclude_last_months=0)
    is_df, oos_df = bff.split_is_oos_by_args(df, args_end)
    assert len(is_df) == 2
    assert len(oos_df) == 1

def test_brute_force_fast_backtest_empty(mock_env):
    bff, _ = mock_env
    results = bff.brute_force_fast_backtest(
        {}, np.array([]), [], np.array([]),
        top_k=1, freq="day", cost_rate=0.0, batch_size=10,
        min_combo_size=1, max_combo_size=1, output_dir="out", anchor_date="2020-01-01"
    )
    assert results.empty

@patch('qlib.workflow.R', create=True)
def test_analyze_results_fast(mock_R, mock_env, tmp_path):
    bff, _ = mock_env
    out_dir = str(tmp_path / "output_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        "models": ["m1", "m2", "m1,m2"],
        "n_models": [1, 1, 2],
        "Ann_Ret": [0.15, 0.12, 0.18],
        "Max_DD": [-0.05, -0.08, -0.04],
        "Ann_Excess": [0.10, 0.07, 0.13],
        "Calmar": [3.0, 1.5, 4.5],
    })
    
    # Mocking necessary for attribution analysis
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    idx = pd.MultiIndex.from_tuples([(dates[0], "A"), (dates[1], "A")], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [0.5, 0.6], "m2": [0.6, 0.7]}, index=idx)
    
    mock_recorder = MagicMock()
    mock_report = pd.DataFrame({
        "return": [0.01, 0.02],
        "bench": [0.005, 0.005]
    }, index=dates)
    mock_recorder.load_object.return_value = mock_report
    mock_R.get_recorder.return_value = mock_recorder

    with patch('matplotlib.pyplot.savefig'):
        bff.analyze_results(
            results_df=results_df,
            corr_matrix=pd.DataFrame(),
            norm_df=norm_df,
            train_records={"experiment_name": "exp", "models": {"m1": "r1", "m2": "r2"}},
            output_dir=out_dir,
            anchor_date="2020-01-01",
            top_n=2
        )
    
    assert os.path.exists(os.path.join(out_dir, "analysis_report_fast_2020-01-01.txt"))
