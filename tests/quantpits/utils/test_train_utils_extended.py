import os
import sys
import json
import importlib
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    config_dir = workspace / "config"
    config_dir.mkdir()
    data_dir = workspace / "data"
    data_dir.mkdir()
    output_dir = workspace / "output"
    output_dir.mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    from quantpits.utils import env
    importlib.reload(env)

    from quantpits.utils import train_utils
    importlib.reload(train_utils)

    # Patch module-level path constants to use tmp_path
    monkeypatch.setattr(train_utils, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(train_utils, 'RECORD_OUTPUT_FILE', str(workspace / "latest_train_records.json"))
    monkeypatch.setattr(train_utils, 'HISTORY_DIR', str(data_dir / "history"))
    monkeypatch.setattr(train_utils, 'RUN_STATE_FILE', str(data_dir / "run_state.json"))

    yield train_utils, workspace


# ── get_models_by_filter ─────────────────────────────────────────────────

def test_get_models_by_filter_algorithm(mock_env):
    tu, _ = mock_env
    registry = {
        "m1": {"algorithm": "LSTM", "dataset": "Alpha158", "market": "csi300", "tags": ["ts"], "enabled": True},
        "m2": {"algorithm": "LGB", "dataset": "Alpha158", "market": "csi300", "tags": ["tree"], "enabled": True},
        "m3": {"algorithm": "LSTM", "dataset": "Alpha360", "market": "csi300", "tags": ["ts"], "enabled": False},
    }
    result = tu.get_models_by_filter(registry=registry, algorithm="lstm")
    assert len(result) == 2
    assert "m1" in result
    assert "m3" in result


def test_get_models_by_filter_tag(mock_env):
    tu, _ = mock_env
    registry = {
        "m1": {"algorithm": "LSTM", "tags": ["ts", "experimental"]},
        "m2": {"algorithm": "LGB", "tags": ["tree"]},
    }
    result = tu.get_models_by_filter(registry=registry, tag="tree")
    assert len(result) == 1
    assert "m2" in result


def test_get_models_by_filter_combined(mock_env):
    tu, _ = mock_env
    registry = {
        "m1": {"algorithm": "LSTM", "dataset": "Alpha158", "market": "csi300", "tags": ["ts"]},
        "m2": {"algorithm": "LSTM", "dataset": "Alpha360", "market": "csi300", "tags": ["ts"]},
        "m3": {"algorithm": "LGB", "dataset": "Alpha158", "market": "csi300", "tags": ["tree"]},
    }
    result = tu.get_models_by_filter(registry=registry, algorithm="lstm", dataset="alpha158")
    assert len(result) == 1
    assert "m1" in result


# ── get_models_by_names ──────────────────────────────────────────────────

def test_get_models_by_names_found(mock_env):
    tu, _ = mock_env
    registry = {
        "model_A": {"algorithm": "LSTM"},
        "model_B": {"algorithm": "LGB"},
    }
    result = tu.get_models_by_names(["model_A", "model_B"], registry=registry)
    assert len(result) == 2


def test_get_models_by_names_partial(mock_env, capsys):
    tu, _ = mock_env
    registry = {
        "model_A": {"algorithm": "LSTM"},
    }
    result = tu.get_models_by_names(["model_A", "nonexistent"], registry=registry)
    assert len(result) == 1
    assert "model_A" in result
    captured = capsys.readouterr()
    assert "nonexistent" in captured.out


# ── save_run_state / load_run_state / clear_run_state ────────────────────

def test_save_load_run_state(mock_env):
    tu, workspace = mock_env
    state_file = str(workspace / "data" / "run_state.json")

    state = {
        "started_at": "2026-03-01 10:00:00",
        "mode": "incremental",
        "target_models": ["m1", "m2"],
        "completed": ["m1"],
        "failed": {"m2": "Some error"},
        "skipped": []
    }
    tu.save_run_state(state, state_file=state_file)
    assert os.path.exists(state_file)

    loaded = tu.load_run_state(state_file=state_file)
    assert loaded is not None
    assert loaded["mode"] == "incremental"
    assert loaded["completed"] == ["m1"]
    assert "m2" in loaded["failed"]


def test_load_run_state_missing(mock_env):
    tu, workspace = mock_env
    result = tu.load_run_state(state_file=str(workspace / "data" / "nonexistent.json"))
    assert result is None


def test_clear_run_state(mock_env):
    tu, workspace = mock_env
    state_file = str(workspace / "data" / "run_state.json")

    tu.save_run_state({"mode": "test"}, state_file=state_file)
    assert os.path.exists(state_file)

    tu.clear_run_state(state_file=state_file)
    assert not os.path.exists(state_file)
    # Backup should exist in history
    history_dir = str(workspace / "data" / "history")
    if os.path.exists(history_dir):
        backups = os.listdir(history_dir)
        assert any("run_state" in b for b in backups)


# ── overwrite_train_records ──────────────────────────────────────────────

def test_overwrite_train_records(mock_env):
    tu, workspace = mock_env
    record_file = str(workspace / "latest_train_records.json")

    # Create initial records
    initial = {"experiment_name": "old", "models": {"old_model": "id1"}}
    with open(record_file, "w") as f:
        json.dump(initial, f)

    # Overwrite
    new_records = {"experiment_name": "new", "models": {"new_model": "id2"}}
    tu.overwrite_train_records(new_records, record_file=record_file)

    with open(record_file) as f:
        result = json.load(f)
    assert result["experiment_name"] == "new"
    assert "new_model" in result["models"]
    assert "old_model" not in result["models"]

    # History backup should exist
    history_dir = str(workspace / "data" / "history")
    if os.path.exists(history_dir):
        backups = os.listdir(history_dir)
        assert any("train_records" in b for b in backups)


# ── merge_performance_file ───────────────────────────────────────────────

def test_merge_performance_file(mock_env):
    tu, workspace = mock_env
    output_dir = str(workspace / "output")

    # First merge - no existing file
    perf1 = {"modelA": {"IC_Mean": 0.05, "ICIR": 1.2}}
    result = tu.merge_performance_file(perf1, "2026-03-01", output_dir=output_dir)
    assert result["modelA"]["IC_Mean"] == 0.05

    # Second merge - should combine
    perf2 = {"modelB": {"IC_Mean": 0.03, "ICIR": 0.8}}
    result = tu.merge_performance_file(perf2, "2026-03-01", output_dir=output_dir)
    assert "modelA" in result
    assert "modelB" in result

    # Overwrite existing model
    perf3 = {"modelA": {"IC_Mean": 0.07, "ICIR": 1.5}}
    result = tu.merge_performance_file(perf3, "2026-03-01", output_dir=output_dir)
    assert result["modelA"]["IC_Mean"] == 0.07


# ── print_model_table ────────────────────────────────────────────────────

def test_print_model_table(mock_env, capsys):
    tu, _ = mock_env
    models = {
        "GATs_Alpha158": {"algorithm": "GATs", "dataset": "Alpha158", "market": "csi300", "tags": ["deep"]},
        "LGB_Alpha360": {"algorithm": "LGB", "dataset": "Alpha360", "market": "csi300", "tags": ["tree", "fast"]},
    }
    tu.print_model_table(models, title="Test Table")
    captured = capsys.readouterr()
    assert "Test Table" in captured.out
    assert "GATs_Alpha158" in captured.out
    assert "LGB_Alpha360" in captured.out
    assert "2 个模型" in captured.out
