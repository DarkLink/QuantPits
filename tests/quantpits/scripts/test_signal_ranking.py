import os
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Apply environment mocking before loading the module
@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    config_dir = workspace / "config"
    config_dir.mkdir()
    
    output_dir = workspace / "output"
    output_dir.mkdir()
    
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    from quantpits.scripts import signal_ranking
    import importlib
    importlib.reload(env)
    
    # Reload signal_ranking to pick up the new env.ROOT_DIR if needed
    importlib.reload(signal_ranking)
    
    # Since signal_ranking defines its constants at module level during import,
    # we need to patch them directly to use our tmp_path.
    monkeypatch.setattr(signal_ranking, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(signal_ranking, 'PREDICTION_DIR', str(pred_dir))
    monkeypatch.setattr(signal_ranking, 'ENSEMBLE_CONFIG_FILE', str(config_dir / "ensemble_config.json"))
    
    yield signal_ranking, workspace

# ── parse_ensemble_config ────────────────────────────────────────────────
def test_parse_ensemble_config_new_format(mock_env):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    
    cfg = {
        "combos": {
            "combo_A": {"models": ["m1"], "method": "equal", "default": True}
        }
    }
    config_file.write_text(json.dumps(cfg))
    
    # signal_ranking.parse_ensemble_config takes config_file
    result = sr.parse_ensemble_config(config_file=str(config_file))
    assert "combo_A" in result
    assert result["combo_A"]["method"] == "equal"

def test_parse_ensemble_config_old_format(mock_env):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    
    cfg = {
        "models": ["m1", "m2"],
        "ensemble_method": "icir_weighted"
    }
    config_file.write_text(json.dumps(cfg))
    
    result = sr.parse_ensemble_config(config_file=str(config_file))
    assert "legacy" in result
    assert result["legacy"]["method"] == "icir_weighted"
    assert result["legacy"]["models"] == ["m1", "m2"]

def test_parse_ensemble_config_empty(mock_env):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    result = sr.parse_ensemble_config(config_file=str(config_file))
    assert result == {}

# ── get_default_combo ────────────────────────────────────────────────────
def test_get_default_combo(mock_env):
    sr, _ = mock_env
    combos = {
        "combo_1": {"default": False},
        "combo_2": {"default": True},
        "combo_3": {"default": False}
    }
    name, cfg = sr.get_default_combo(combos)
    assert name == "combo_2"
    assert cfg["default"] is True

def test_get_default_combo_no_default(mock_env):
    sr, _ = mock_env
    combos = {
        "combo_1": {"default": False},
        "combo_2": {"default": False}
    }
    name, cfg = sr.get_default_combo(combos)
    assert name == "combo_1"

# ── generate_signal_scores ───────────────────────────────────────────────
def test_generate_signal_scores(mock_env):
    sr, _ = mock_env
    
    df = pd.DataFrame({
        "score": [0.1, 0.5, 0.9, 0.2]
    }, index=pd.MultiIndex.from_tuples([
        ("000001", "2026-03-01"),
        ("000002", "2026-03-01"),
        ("000003", "2026-03-01"),
        ("000004", "2026-03-01")
    ], names=["instrument", "datetime"]))
    
    # Raw range is [0.1, 0.9].
    # Normalizing to [-100, 100]:
    # 0.1 -> -100
    # 0.5 -> 0
    # 0.9 -> 100
    res_df, latest_date = sr.generate_signal_scores(df, top_n=3)
    
    assert latest_date == "2026-03-01"
    assert len(res_df) == 3
    assert list(res_df.index) == ["000003", "000002", "000004"]
    assert res_df.loc["000003", "推荐指数"] == 100.0
    assert res_df.loc["000002", "推荐指数"] == 0.0
    assert res_df.loc["000004", "推荐指数"] == -75.0

def test_generate_signal_scores_constant(mock_env):
    sr, _ = mock_env
    
    df = pd.DataFrame({
        "score": [0.5, 0.5, 0.5]
    }, index=pd.MultiIndex.from_tuples([
        ("000001", "2026-03-01"),
        ("000002", "2026-03-01"),
        ("000003", "2026-03-01"),
    ], names=["instrument", "datetime"]))
    
    res_df, latest_date = sr.generate_signal_scores(df, top_n=2)
    assert len(res_df) == 2
    assert res_df["推荐指数"].iloc[0] == 0.0

# ── find_prediction_file ─────────────────────────────────────────────────
def test_find_prediction_file(mock_env):
    sr, workspace = mock_env
    pred_dir = workspace / "output" / "predictions"
    
    # Create some mock files
    (pred_dir / "ensemble_2026-03-01.csv").write_text("1")
    (pred_dir / "ensemble_2026-03-02.csv").write_text("2")
    (pred_dir / "ensemble_combo_A_2026-03-01.csv").write_text("3")
    
    # Generic latest
    res = sr.find_prediction_file(combo_name=None, prediction_dir=str(pred_dir))
    assert "ensemble_2026-03-02.csv" in res
    
    # Specific combo
    res = sr.find_prediction_file(combo_name="combo_A", prediction_dir=str(pred_dir))
    assert "ensemble_combo_A_2026-03-01.csv" in res

def test_find_prediction_file_not_found(mock_env):
    sr, workspace = mock_env
    
    with pytest.raises(FileNotFoundError):
        sr.find_prediction_file(combo_name="combo_X", prediction_dir=str(workspace / "output" / "predictions"))
        
    with pytest.raises(FileNotFoundError):
        sr.find_prediction_file(combo_name=None, prediction_dir=str(workspace / "output" / "predictions"))

# ── main ─────────────────────────────────────────────────────────────────
def test_main_default(mock_env, tmp_path):
    sr, workspace = mock_env
    pred_dir = workspace / "output" / "predictions"
    pred_file = pred_dir / "ensemble_2026-03-01.csv"
    
    # Create mock prediction data
    df = pd.DataFrame({
        "score": [0.1, 0.9]
    }, index=pd.MultiIndex.from_tuples([
        ("000001", "2026-03-01"),
        ("000002", "2026-03-01")
    ], names=["instrument", "datetime"]))
    df.to_csv(pred_file)
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--output-dir', str(workspace / "output" / "ranking"),
                                    '--prediction-dir', str(pred_dir)]):
        sr.main()
    
    ranking_dir = workspace / "output" / "ranking"
    assert ranking_dir.exists()
    # Signal_default_2026-03-01_Top300.csv
    files = list(ranking_dir.glob("Signal_default_2026-03-01_*.csv"))
    assert len(files) == 1

def test_main_dry_run(mock_env, tmp_path):
    sr, workspace = mock_env
    pred_dir = workspace / "output" / "predictions"
    pred_file = pred_dir / "ensemble_2026-03-01.csv"
    df = pd.DataFrame({"score": [0.5]}, index=pd.MultiIndex.from_tuples([("A", "2026-01-01")], names=["instrument", "datetime"]))
    df.to_csv(pred_file)
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--dry-run', '--output-dir', str(workspace / "output" / "ranking"),
                                    '--prediction-dir', str(pred_dir)]):
        sr.main()
    
    ranking_dir = workspace / "output" / "ranking"
    # Even in dry-run, ranking_dir might be created by os.makedirs(args.output_dir, ...),
    # but no files should be written.
    if ranking_dir.exists():
        files = list(ranking_dir.glob("*.csv"))
        assert len(files) == 0

def test_main_prediction_file(mock_env, tmp_path):
    sr, workspace = mock_env
    custom_file = tmp_path / "custom.csv"
    df = pd.DataFrame({"score": [0.5]}, index=pd.MultiIndex.from_tuples([("A", "2026-01-01")], names=["instrument", "datetime"]))
    df.to_csv(custom_file)
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--prediction-file', str(custom_file), '--output-dir', str(workspace / "output" / "ranking")]):
        sr.main()
        
    ranking_dir = workspace / "output" / "ranking"
    files = list(ranking_dir.glob("Signal_custom_*.csv"))
    assert len(files) == 1

def test_main_all_combos(mock_env, tmp_path):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    cfg = {"combos": {"cA": {"models": ["m1"], "default": True}}}
    config_file.write_text(json.dumps(cfg))
    
    pred_dir = workspace / "output" / "predictions"
    pred_file = pred_dir / "ensemble_cA_2026-03-01.csv"
    df = pd.DataFrame({"score": [0.5]}, index=pd.MultiIndex.from_tuples([("A", "2026-03-01")], names=["instrument", "datetime"]))
    df.to_csv(pred_file)

    import sys
    with patch.object(sys, 'argv', ['script.py', '--all-combos', '--output-dir', str(workspace / "output" / "ranking"),
                                    '--prediction-dir', str(pred_dir)]):
        sr.main()
        
    ranking_dir = workspace / "output" / "ranking"
    files = list(ranking_dir.glob("Signal_cA_*.csv"))
    assert len(files) == 1

def test_main_combo_arg(mock_env, tmp_path):
    sr, workspace = mock_env
    pred_dir = workspace / "output" / "predictions"
    pred_file = pred_dir / "ensemble_cB_2026-03-01.csv"
    df = pd.DataFrame({"score": [0.5]}, index=pd.MultiIndex.from_tuples([("A", "2026-01-01")], names=["instrument", "datetime"]))
    df.to_csv(pred_file)

    import sys
    with patch.object(sys, 'argv', ['script.py', '--combo', 'cB', '--output-dir', str(workspace / "output" / "ranking"),
                                    '--prediction-dir', str(pred_dir)]):
        sr.main()
    
    ranking_dir = workspace / "output" / "ranking"
    files = list(ranking_dir.glob("Signal_cB_*.csv"))
    assert len(files) == 1

def test_main_error_no_config_combos(mock_env, tmp_path):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    cfg = {"other": {}} # No 'combos' or 'models'
    config_file.write_text(json.dumps(cfg))
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--all-combos']):
        with pytest.raises(SystemExit) as e:
            sr.main()
        assert e.value.code == 1

def test_main_all_combos_partial_missing(mock_env, tmp_path):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    cfg = {"combos": {
        "cA": {"models": ["m1"]},
        "cB": {"models": ["m2"]}
    }}
    config_file.write_text(json.dumps(cfg))
    
    pred_dir = workspace / "output" / "predictions"
    # Only cA has a file
    pred_file = pred_dir / "ensemble_cA_2026-03-01.csv"
    df = pd.DataFrame({"score": [0.5]}, index=pd.MultiIndex.from_tuples([("A", "2026-01-01")], names=["instrument", "datetime"]))
    df.to_csv(pred_file)

    import sys
    with patch.object(sys, 'argv', ['script.py', '--all-combos', '--output-dir', str(workspace / "output" / "ranking"),
                                    '--prediction-dir', str(pred_dir)]):
        sr.main()
    
    ranking_dir = workspace / "output" / "ranking"
    assert len(list(ranking_dir.glob("Signal_cA_*.csv"))) == 1
    assert len(list(ranking_dir.glob("Signal_cB_*.csv"))) == 0

def test_main_all_combos_none_found(mock_env, tmp_path):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    cfg = {"combos": {"cA": {"models": ["m1"]}}}
    config_file.write_text(json.dumps(cfg))
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--all-combos']):
        with pytest.raises(SystemExit) as e:
            sr.main()
        assert e.value.code == 1

def test_main_exit_not_found(mock_env):
    sr, _ = mock_env
    import sys
    with patch.object(sys, 'argv', ['script.py', '--prediction-file', 'nonexistent.csv']):
        with pytest.raises(SystemExit) as e:
            sr.main()
        assert e.value.code == 1

def test_main_all_combos_no_config(mock_env, tmp_path):
    sr, workspace = mock_env
    # No config file created
    import sys
    with patch.object(sys, 'argv', ['script.py', '--all-combos']):
        with pytest.raises(SystemExit) as e:
            sr.main()
        assert e.value.code == 1

# ── generate_signal_scores Edge Cases ─────────────────────────────────────
def test_generate_signal_scores_multiple_dates(mock_env):
    sr, _ = mock_env
    # Index with multiple dates
    idx = pd.MultiIndex.from_tuples([
        ("A", "2026-03-01"),
        ("A", "2026-03-02")
    ], names=["instrument", "datetime"])
    df = pd.DataFrame({"score": [0.1, 0.9]}, index=idx)
    
    res_df, latest_date = sr.generate_signal_scores(df)
    assert latest_date == "2026-03-02"
    assert len(res_df) == 1 # Only one instrument, A, on latest date

def test_generate_signal_scores_no_datetime_index(mock_env):
    sr, _ = mock_env
    # Index without datetime level
    idx = pd.Index(["A", "B"], name="instrument")
    df = pd.DataFrame({"score": [0.1, 0.9]}, index=idx)
    
    res_df, latest_date = sr.generate_signal_scores(df)
    assert latest_date is None
    assert len(res_df) == 2

# ── find_prediction_file Fallback ─────────────────────────────────────────
def test_find_prediction_file_fallback(mock_env, tmp_path):
    sr, workspace = mock_env
    pred_dir = workspace / "output" / "predictions"
    # Create a file that NOT follows the YYYY-MM-DD pattern
    (pred_dir / "ensemble_weird.csv").write_text("data")
    
    res = sr.find_prediction_file(combo_name=None, prediction_dir=str(pred_dir))
    assert "ensemble_weird.csv" in res

# ── get_default_combo Fallback ────────────────────────────────────────────
def test_get_default_combo_fallback(mock_env):
    sr, _ = mock_env
    # None of the combos have default: True
    combos = {"cA": {"models": []}, "cB": {"models": []}}
    name, cfg = sr.get_default_combo(combos)
    assert name == "cA" # Should return first one

def test_get_default_combo_empty(mock_env):
    sr, _ = mock_env
    name, cfg = sr.get_default_combo({})
    assert name is None
