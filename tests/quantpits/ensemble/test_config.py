import json
import os

from quantpits.ensemble.config import load_ensemble_run_config
from quantpits.utils.workspace import WorkspaceContext


def test_load_ensemble_run_config_resolves_relative_record_file(tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    (workspace / "config").mkdir(parents=True)
    (workspace / "latest_train_records.json").write_text(
        json.dumps({"experiment_name": "DemoExp", "models": {"m1": "rid1"}}),
        encoding="utf-8",
    )
    (workspace / "config" / "model_config.json").write_text('{"freq": "week"}', encoding="utf-8")
    (workspace / "config" / "ensemble_config.json").write_text(
        json.dumps({"combos": {"default": {"models": ["m1"], "default": True}}}),
        encoding="utf-8",
    )

    ctx = WorkspaceContext.from_root(workspace)
    cwd = os.getcwd()
    config = load_ensemble_run_config(ctx, record_file="latest_train_records.json")

    assert os.getcwd() == cwd
    assert config.train_records["experiment_name"] == "DemoExp"
    assert config.model_config["freq"] == "week"
    assert "default" in config.ensemble_config["combos"]


def test_load_ensemble_run_config_respects_absolute_record_file(tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    external = tmp_path / "records.json"
    (workspace / "config").mkdir(parents=True)
    external.write_text(json.dumps({"experiment_name": "ExternalExp", "models": {}}), encoding="utf-8")

    ctx = WorkspaceContext.from_root(workspace)
    config = load_ensemble_run_config(ctx, record_file=external.as_posix())

    assert config.train_records["experiment_name"] == "ExternalExp"


def test_load_ensemble_run_config_missing_optional_files(tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    (workspace / "config").mkdir(parents=True)

    ctx = WorkspaceContext.from_root(workspace)
    config = load_ensemble_run_config(ctx, record_file="missing.json")

    assert config.train_records == {"models": {}, "experiment_name": "unknown"}
    assert config.ensemble_config == {}
