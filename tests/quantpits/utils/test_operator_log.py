import os
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from quantpits.utils.operator_log import OperatorLog

@pytest.fixture
def mock_root_dir(monkeypatch, tmp_path):
    # Must set env var BEFORE importing env (which raises at import time if missing)
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(tmp_path))
    import quantpits.utils.env as env_module
    monkeypatch.setattr(env_module, 'ROOT_DIR', str(tmp_path))
    yield tmp_path

def test_operator_log_success(mock_root_dir):
    script_name = "test_script"
    args = ["--arg1", "val1"]
    
    with OperatorLog(script_name, args=args) as oplog:
        oplog.set_result({"status": "ok"})
        oplog.set_source("llm_critic")
    
    log_file = mock_root_dir / "data" / "operator_log.jsonl"
    assert log_file.exists()
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry['script'] == script_name
        assert entry['args'] == args
        assert entry['result_summary'] == {"status": "ok"}
        assert entry['source'] == "llm_critic"
        assert 'duration_seconds' in entry
        assert 'timestamp_start' in entry
        assert 'timestamp_end' in entry
        assert entry['run_id'] is None
        assert entry['manifest_path'] is None
        assert entry['plan_fingerprint'] is None

def test_operator_log_exception(mock_root_dir):
    script_name = "fail_script"
    
    try:
        with OperatorLog(script_name) as oplog:
            raise ValueError("Intentional Error")
    except ValueError:
        pass
    
    log_file = mock_root_dir / "data" / "operator_log.jsonl"
    assert log_file.exists()
    
    with open(log_file, 'r') as f:
        entry = json.loads(f.read())
        assert entry['script'] == script_name
        assert entry['exception']['type'] == "ValueError"
        assert "Intentional Error" in entry['exception']['value']

def test_operator_log_multiple_entries(mock_root_dir):
    log_file = mock_root_dir / "data" / "operator_log.jsonl"
    
    with OperatorLog("script1") as oplog:
        pass
    
    with OperatorLog("script2") as oplog:
        pass
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])['script'] == "script1"
        assert json.loads(lines[1])['script'] == "script2"

def test_operator_log_silent_failure(mock_root_dir):
    # Mock open to fail
    with patch('builtins.open', side_effect=OSError("Disk Full")):
        with OperatorLog("test") as oplog:
            pass # Should not raise exception
    
    log_file = mock_root_dir / "data" / "operator_log.jsonl"
    assert not log_file.exists()


def test_operator_log_records_manifest_linkage(mock_root_dir):
    with OperatorLog("ensemble_fusion", run_id="run-1") as oplog:
        oplog.set_run_manifest(
            run_id="run-1",
            manifest_path="output/manifests/ensemble_fusion/run-1.json",
        )
        oplog.set_plan_fingerprint("abc123")

    log_file = mock_root_dir / "data" / "operator_log.jsonl"
    entry = json.loads(log_file.read_text())

    assert entry["run_id"] == "run-1"
    assert entry["manifest_path"] == "output/manifests/ensemble_fusion/run-1.json"
    assert entry["plan_fingerprint"] == "abc123"


def test_explicit_operator_log_commit_exposes_write_failure(mock_root_dir):
    with patch("builtins.open", side_effect=OSError("Disk Full")):
        with pytest.raises(OSError, match="Disk Full"):
            with OperatorLog(
                "static_train", run_id="run-1", transaction_id="tx-1",
            ) as oplog:
                oplog.set_result({"status": "success"})
                oplog.commit()


def test_explicit_operator_log_commit_adopts_existing_durable_link(mock_root_dir):
    log_file = mock_root_dir / "data" / "operator_log.jsonl"
    kwargs = dict(
        script_name="static_train", run_id="run-1", transaction_id="tx-1",
        manifest_path="output/manifests/static_train/run-1.json",
        plan_fingerprint="plan-fp", log_file=str(log_file),
    )
    with OperatorLog(**kwargs) as oplog:
        oplog.set_result({"status": "success"})
        first = oplog.commit()
    with OperatorLog(**kwargs) as oplog:
        oplog.set_result({"status": "success"})
        adopted = oplog.commit()
    assert adopted == first
    assert len(log_file.read_text().splitlines()) == 1
