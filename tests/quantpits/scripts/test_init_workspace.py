import os
import stat
import pytest
import yaml

from quantpits.scripts.init_workspace import init_workspace


def test_init_workspace_success(tmp_path):
    """Full lifecycle: config copy, directory creation, run_env.sh generation."""
    source = tmp_path / "SourceWorkspace"
    source.mkdir()
    source_config = source / "config"
    source_config.mkdir()
    (source_config / "model_config.json").write_text('{"market": "csi300"}')
    (source_config / "workflow.yaml").write_text("dummy: true")

    target = tmp_path / "NewWorkspace"

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
    # Should have executable permission
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
