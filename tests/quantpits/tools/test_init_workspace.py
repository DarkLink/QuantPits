import os
import stat
import pytest
import yaml
from unittest.mock import patch

from quantpits.tools.init_workspace import init_workspace


def test_init_workspace_success(tmp_path):
    """Full lifecycle: config copy, directory creation, run_env.sh generation."""
    source = tmp_path / "SourceWorkspace"
    source.mkdir()
    source_config = source / "config"
    source_config.mkdir()
    (source_config / "model_config.json").write_text('{"market": "csi300"}')
    (source_config / "workflow.yaml").write_text("dummy: true")

    target = tmp_path / "NewWorkspace"

    # Mock Linux behavior to ensure run_env.sh is tested here
    with patch("platform.system", return_value="Linux"):
        init_workspace(str(source), str(target))

    # Verify directory structure
    assert target.exists()
    assert (target / "config").exists()
    assert (target / "data").exists()
    assert (target / "output").exists()
    assert (target / "archive").exists()
    assert (target / "mlruns").exists()

    # Config should be copied
    assert (target / "config" / "model_config.json").exists()
    assert (target / "config" / "workflow.yaml").exists()

    # run_env.sh should be created
    run_env = target / "run_env.sh"
    assert run_env.exists()
    content = run_env.read_text()
    assert "QLIB_WORKSPACE_DIR" in content
    # Should have executable permission (only check on non-windows)
    if os.name != 'nt':
        assert os.stat(run_env).st_mode & stat.S_IXUSR


def test_init_workspace_strategy_yaml_generated(tmp_path):
    """strategy_config.yaml should be auto-generated if missing from source."""
    source = tmp_path / "SourceWorkspace"
    source.mkdir()
    source_config = source / "config"
    source_config.mkdir()
    # No strategy_config.yaml in source config

    target = tmp_path / "NewWorkspace"
    init_workspace(str(source), str(target))

    strategy_yaml = target / "config" / "strategy_config.yaml"
    assert strategy_yaml.exists()
    content = yaml.safe_load(strategy_yaml.read_text())
    assert content["strategy"]["name"] == "topk_dropout"


def test_init_workspace_source_missing(tmp_path, capsys):
    """Should print error and return if source does not exist."""
    target = tmp_path / "NewWorkspace"
    init_workspace(str(tmp_path / "nonexistent_source"), str(target))

    captured = capsys.readouterr()
    assert "Error" in captured.out
    assert not target.exists()


def test_init_workspace_target_exists(tmp_path, capsys):
    """Should print error and return if target already exists."""
    source = tmp_path / "SourceWorkspace"
    source.mkdir()
    source_config = source / "config"
    source_config.mkdir()

    target = tmp_path / "ExistingWorkspace"
    target.mkdir()

    init_workspace(str(source), str(target))

    captured = capsys.readouterr()
    assert "already exists" in captured.out


def test_init_workspace_no_source_config(tmp_path, capsys):
    """If source has no config dir, should create empty config at target."""
    source = tmp_path / "SourceWorkspace"
    source.mkdir()
    # No config directory

    target = tmp_path / "NewWorkspace"
    init_workspace(str(source), str(target))

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert (target / "config").exists()


def test_init_workspace_windows_script(tmp_path):
    """Should generate run_env.ps1 when running on Windows."""
    source = tmp_path / "Source"
    source.mkdir()
    (source / "config").mkdir()
    target = tmp_path / "NewWorkspacePS"

    with patch("platform.system", return_value="Windows"):
        init_workspace(str(source), str(target))

    run_env = target / "run_env.ps1"
    assert run_env.exists()
    content = run_env.read_text(encoding="utf-8")
    assert "$env:QLIB_WORKSPACE_DIR" in content
    assert "run_env.sh" not in [f.name for f in target.iterdir() if f.is_file() and f.suffix == ".sh"]


def test_init_workspace_linux_script(tmp_path):
    """Should generate run_env.sh when running on Linux."""
    source = tmp_path / "Source"
    source.mkdir()
    (source / "config").mkdir()
    target = tmp_path / "NewWorkspaceSH"

    with patch("platform.system", return_value="Linux"):
        init_workspace(str(source), str(target))

    run_env = target / "run_env.sh"
    assert run_env.exists()
    content = run_env.read_text()
    assert "export QLIB_WORKSPACE_DIR" in content
    # Should have executable permission (only check on non-windows)
    if os.name != 'nt':
        assert os.stat(run_env).st_mode & stat.S_IXUSR


def test_main(tmp_path, monkeypatch):
    from quantpits.tools import init_workspace as iw
    import platform
    is_windows = platform.system() == "Windows"
    script_name = "run_env.ps1" if is_windows else "run_env.sh"

    source = tmp_path / "Source"
    source.mkdir()
    (source / "config").mkdir()
    target = tmp_path / "Target"
    
    import sys
    with patch.object(sys, 'argv', ['script.py', '--source', str(source), '--target', str(target)]):
        iw.main()
        
    assert target.exists()
    assert (target / script_name).exists()
