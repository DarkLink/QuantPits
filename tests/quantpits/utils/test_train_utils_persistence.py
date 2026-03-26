import os
import json
from unittest.mock import patch

def test_run_state(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    state_file = tmp_path / "run_state.json"
    assert train_utils.load_run_state(str(state_file)) is None
    state = {"mode": "incremental", "completed": ["m1"]}
    train_utils.save_run_state(state, str(state_file))
    assert train_utils.load_run_state(str(state_file))["mode"] == "incremental"
    train_utils.clear_run_state(str(state_file))
    assert not os.path.exists(state_file)

def test_run_state_default_file(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    state_file = tmp_path / "run_state_default.json"
    with patch('quantpits.utils.train_utils.RUN_STATE_FILE', str(state_file)):
        train_utils.save_run_state({"mode": "test"})
        assert train_utils.load_run_state()["mode"] == "test"
        train_utils.clear_run_state()
        assert not os.path.exists(state_file)

def test_backup_file_with_date(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello")
    backup_path = train_utils.backup_file_with_date(str(source_file), history_dir=str(tmp_path / "history"), prefix="pre")
    assert os.path.exists(backup_path) and "pre" in backup_path
    assert train_utils.backup_file_with_date("nonexistent.txt") is None

def test_merge_train_records(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    record_file = tmp_path / "records.json"
    record_file.write_text(json.dumps({"experiment_name": "e1", "models": {"mA": "id1"}}))
    new = {"experiment_name": "e2", "models": {"mB": "id2"}}
    merged = train_utils.merge_train_records(new, record_file=str(record_file))
    assert merged["experiment_name"] == "e2"
    assert "mA" in merged["models"] and "mB" in merged["models"]

def test_merge_performance_file(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    perf1 = {"mA": {"IC": 0.1}}
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    perf_file = out_dir / "model_performance_2026.json"
    perf_file.write_text(json.dumps(perf1))
    new = {"mB": {"IC": 0.2}}
    merged = train_utils.merge_performance_file(new, "2026", output_dir=str(out_dir))
    assert "mA" in merged and "mB" in merged
    # Test default output_dir using ROOT_DIR patch
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)):
        (tmp_path / "output").mkdir(exist_ok=True)
        assert train_utils.merge_performance_file(new, "2026") is not None

def test_overwrite_train_records(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    records = {"experiment_name": "test"}
    rec_file = tmp_path / 'rec.json'
    train_utils.overwrite_train_records(records, record_file=str(rec_file))
    with open(rec_file, 'r') as f:
        assert json.load(f)["experiment_name"] == "test"

def test_migrate_legacy_records(mock_env_constants, tmp_path, capsys):
    train_utils, _ = mock_env_constants
    legacy_file = tmp_path / "latest_rolling_records.json"
    legacy_file.write_text(json.dumps({"models": {"old_m": "old_id"}}))
    static_file = tmp_path / "latest_train_records.json"
    static_file.write_text(json.dumps({"models": {"static_m": "static_id"}}))
    
    # Pass workspace_dir explicitly
    train_utils.migrate_legacy_records(workspace_dir=str(tmp_path))
    assert os.path.exists(static_file)
    data = json.load(open(static_file))
    assert data["models"]["old_m@rolling"] == "old_id"
    assert data["models"]["static_m@static"] == "static_id"
    
    # Second run (already exists)
    train_utils.migrate_legacy_records(workspace_dir=str(tmp_path))
    captured = capsys.readouterr()
    assert "无需迁移" in captured.out
