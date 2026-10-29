"""Phase 28 import-purity and filesystem-only plan contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPOSITORY = Path(__file__).resolve().parents[3]


def _snapshot(root):
    """Capture paths and file bytes without depending on workspace metadata."""

    return tuple(
        (path.relative_to(root).as_posix(), path.is_dir(),
         None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    )


def _clean_subprocess_env():
    env = os.environ.copy()
    env.pop("QLIB_WORKSPACE_DIR", None)
    env.pop("MLFLOW_TRACKING_URI", None)
    env["PYTHONPATH"] = str(REPOSITORY)
    return env


def test_import_does_not_require_workspace_or_change_cwd(tmp_path):
    code = """
import os
import sys
before = os.getcwd()
import quantpits.scripts.rolling_train as rolling_train
import quantpits.rolling.identity
import quantpits.rolling.state
assert os.getcwd() == before
assert 'quantpits.utils.env' not in sys.modules
assert callable(rolling_train.build_parser)
assert callable(rolling_train.resolve_target_models)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(tmp_path),
        env=_clean_subprocess_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_module_help_does_not_require_workspace(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "quantpits.scripts.rolling_train", "--help"],
        cwd=str(tmp_path), env=_clean_subprocess_env(),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--workspace" in result.stdout
    assert "--explain-plan" in result.stdout
    assert "--json-plan" in result.stdout
    assert "--run-id" in result.stdout


def test_script_path_help_does_not_require_workspace(tmp_path):
    script = REPOSITORY / "quantpits" / "scripts" / "rolling_train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=str(tmp_path),
        env=_clean_subprocess_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--workspace" in result.stdout
    assert "--json-plan" in result.stdout


def test_json_information_route_does_not_import_legacy_env(tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    config = workspace / "config"
    config.mkdir(parents=True)
    (config / "rolling_config.yaml").write_text(yaml.safe_dump({
        "rolling_start": "2020-01-01",
        "train_years": 3,
        "valid_years": 1,
        "test_step": "3M",
    }), encoding="utf-8")
    (config / "model_registry.yaml").write_text(yaml.safe_dump({
        "models": {
            "demo_model": {
                "enabled": True,
                "algorithm": "linear",
                "dataset": "Alpha158",
                "yaml_file": "config/workflow_config_demo.yaml",
            },
        },
    }), encoding="utf-8")
    (config / "workflow_config_demo.yaml").write_text(
        "model: demo\n", encoding="utf-8",
    )
    before = _snapshot(workspace)
    code = """
import json
import os
import sys
from quantpits.scripts import rolling_train
before_env = dict(os.environ)
exit_code = rolling_train.main([
    '--workspace', sys.argv[1], '--cold-start', '--all-enabled', '--json-plan',
])
assert exit_code in (None, 0), exit_code
assert 'quantpits.utils.env' not in sys.modules
assert dict(os.environ) == before_env
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(workspace)], cwd=str(tmp_path),
        env=_clean_subprocess_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["plan_fingerprint"]
    plan = payload["plan"]
    assert plan["metadata"]["action"] == "cold_start"
    assert plan["metadata"]["target_keys"] == ["demo_model@rolling"]
    assert plan["metadata"]["state_inspection"]["classification"] == "missing"
    assert len(plan["metadata"]["workspace_fingerprint"]) == 64
    assert [item["model_name"] for item in plan["metadata"]["targets"]] == [
        "demo_model",
    ]
    assert plan["metadata"]["zero_write_plan_route"] is True
    # Outputs describe a real run's effects; they are not writes performed by
    # this information route.
    assert plan["outputs"]
    assert _snapshot(workspace) == before


def test_action_conflict_fails_before_legacy_env_or_workspace_write(tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    config = workspace / "config"
    config.mkdir(parents=True)
    (config / "rolling_config.yaml").write_text(yaml.safe_dump({
        "rolling_start": "2020-01-01",
        "train_years": 3,
        "valid_years": 1,
        "test_step": "3M",
    }), encoding="utf-8")
    before = _snapshot(workspace)
    code = """
import sys
from quantpits.scripts import rolling_train
exit_code = rolling_train.main([
    '--workspace', sys.argv[1], '--cold-start', '--resume', '--all-enabled',
])
assert exit_code == 2, exit_code
assert 'quantpits.utils.env' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(workspace)], cwd=str(tmp_path),
        env=_clean_subprocess_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "rolling_action_conflict" in result.stderr
    assert _snapshot(workspace) == before
