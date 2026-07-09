from __future__ import annotations

import dataclasses
import importlib
import os
import sys
from pathlib import Path

import pytest

from quantpits.utils.workspace import (
    WorkspaceContext,
    fingerprint_file,
    fingerprint_value,
    short_fingerprint,
    stable_json_dumps,
)


@dataclasses.dataclass(frozen=True)
class SampleConfig:
    name: str
    params: dict[str, int]


def test_workspace_context_from_root_builds_paths_without_side_effects(tmp_path):
    workspace = tmp_path / "Workspace"
    cwd_before = Path.cwd()

    ctx = WorkspaceContext.from_root(workspace)

    assert Path.cwd() == cwd_before
    assert ctx.root == workspace.resolve()
    assert ctx.config_dir == workspace.resolve() / "config"
    assert ctx.data_dir == workspace.resolve() / "data"
    assert ctx.output_dir == workspace.resolve() / "output"
    assert ctx.mlruns_dir == workspace.resolve() / "mlruns"
    assert ctx.mlflow_uri == f"sqlite:///{workspace.resolve() / 'mlflow.db'}"
    assert ctx.qlib_region == "cn"
    assert not workspace.exists()


def test_workspace_context_path_helpers(tmp_path):
    workspace = tmp_path / "Workspace"
    ctx = WorkspaceContext.from_root(workspace)

    assert ctx.path("config", "model_config.json") == ctx.root / "config" / "model_config.json"
    assert ctx.config_path("model_config.json") == ctx.config_dir / "model_config.json"
    assert ctx.data_path("history", "runs.json") == ctx.data_dir / "history" / "runs.json"
    assert ctx.output_path("predictions") == ctx.output_dir / "predictions"


def test_workspace_context_uses_legacy_file_store_when_mlruns_has_data(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    workspace = tmp_path / "LegacyWorkspace"
    (workspace / "mlruns" / "0").mkdir(parents=True)

    ctx = WorkspaceContext.from_root(workspace)

    assert ctx.mlflow_uri == f"file://{workspace.resolve() / 'mlruns'}"


def test_workspace_context_respects_explicit_runtime_metadata(tmp_path):
    ctx = WorkspaceContext.from_root(
        tmp_path / "Workspace",
        mlflow_uri="sqlite:////tmp/custom.db",
        qlib_data_dir="/tmp/qlib_data",
        qlib_region="us",
    )

    assert ctx.mlflow_uri == "sqlite:////tmp/custom.db"
    assert ctx.qlib_data_dir == Path("/tmp/qlib_data")
    assert ctx.qlib_region == "us"


def test_env_get_workspace_context_uses_current_env_state(monkeypatch, tmp_path):
    workspace = tmp_path / "EnvWorkspace"
    workspace.mkdir()
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("QLIB_DATA_DIR", "/tmp/qlib_data")
    monkeypatch.setenv("QLIB_REGION", "us")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    from quantpits.utils import env

    importlib.reload(env)
    cwd_before = Path.cwd()
    ctx = env.get_workspace_context()

    assert Path.cwd() == cwd_before
    assert ctx.root == workspace.resolve()
    assert ctx.mlflow_uri == env.mlflow_backend
    assert ctx.qlib_data_dir == Path("/tmp/qlib_data")
    assert ctx.qlib_region == "us"


def test_env_get_workspace_context_accepts_explicit_root(monkeypatch, tmp_path):
    active_workspace = tmp_path / "ActiveWorkspace"
    explicit_workspace = tmp_path / "ExplicitWorkspace"
    active_workspace.mkdir()
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(active_workspace))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    from quantpits.utils import env

    importlib.reload(env)
    ctx = env.get_workspace_context(explicit_workspace)

    assert ctx.root == explicit_workspace.resolve()
    assert env.ROOT_DIR == str(active_workspace)
    assert os.environ["QLIB_WORKSPACE_DIR"] == str(active_workspace)


def test_workspace_context_from_env_delegates_to_env(monkeypatch, tmp_path):
    workspace = tmp_path / "DelegatedWorkspace"
    workspace.mkdir()
    monkeypatch.setattr(sys, "argv", ["script.py"])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    from quantpits.utils import env

    importlib.reload(env)
    ctx = WorkspaceContext.from_env()

    assert ctx.root == workspace.resolve()
    assert ctx.mlflow_uri == env.mlflow_backend


def test_stable_json_dumps_normalizes_supported_types(tmp_path):
    config = SampleConfig(
        name="demo",
        params={"b": 2, "a": 1},
    )

    dumped = stable_json_dumps(
        {
            "path": tmp_path / "config" / "model.json",
            "config": config,
            "items": {"z", "a"},
            "raw": b"\x00\xff",
        }
    )

    assert '"items":["a","z"]' in dumped
    assert '"params":{"a":1,"b":2}' in dumped
    assert '"path":"' in dumped
    assert '"raw":{"__bytes__":"00ff"}' in dumped


def test_fingerprint_value_is_stable_for_dict_order_and_set_order():
    left = {"b": 2, "a": 1, "tags": {"x", "y"}}
    right = {"tags": {"y", "x"}, "a": 1, "b": 2}

    assert fingerprint_value(left) == fingerprint_value(right)
    assert fingerprint_value(left) != fingerprint_value({**right, "a": 3})


def test_short_fingerprint_is_full_fingerprint_prefix():
    value = {"model": "lgb", "params": {"depth": 6}}
    full = fingerprint_value(value)

    assert short_fingerprint(value) == full[:12]
    assert short_fingerprint(value, length=8) == full[:8]


def test_fingerprint_file_hashes_bytes(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b'{"b":2,"a":1}')
    second.write_bytes(b'{"b":2,"a":1}')

    assert fingerprint_file(first) == fingerprint_file(second)
    second.write_bytes(b'{"a":1,"b":2}')
    assert fingerprint_file(first) != fingerprint_file(second)


def test_fingerprint_value_raises_for_unsupported_objects():
    with pytest.raises(TypeError):
        fingerprint_value(object())
