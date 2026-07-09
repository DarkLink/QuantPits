import json
import os

import yaml

from quantpits.config_contracts.workspace import validate_workspace
from quantpits.utils.workspace import WorkspaceContext


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def make_workspace(tmp_path, *, include_optional=True):
    workspace = tmp_path / "Workspace"
    config = workspace / "config"
    config.mkdir(parents=True)
    _write_json(
        config / "model_config.json",
        {
            "market": "csi300",
            "benchmark": "SH000300",
            "start_time": "2020-01-01",
            "freq": "week",
        },
    )
    _write_yaml(
        config / "strategy_config.yaml",
        {
            "strategy": {"name": "topk_dropout", "params": {"topk": 20, "n_drop": 3}},
            "backtest": {
                "account": 100000,
                "exchange_kwargs": {
                    "deal_price": "close",
                    "open_cost": 0.001,
                    "close_cost": 0.001,
                    "min_cost": 5,
                },
            },
        },
    )
    _write_json(
        config / "prod_config.json",
        {
            "current_date": "2024-01-01",
            "last_processed_date": "2024-01-01",
            "current_cash": 100000,
            "current_holding": [],
        },
    )
    _write_json(
        config / "ensemble_config.json",
        {
            "combos": {
                "default": {
                    "models": ["demo_linear_Alpha158"],
                    "method": "equal",
                    "default": True,
                }
            }
        },
    )
    _write_yaml(
        config / "model_registry.yaml",
        {
            "models": {
                "demo_linear_Alpha158": {
                    "yaml_file": "config/workflow_config_demo.yaml",
                    "enabled": True,
                    "tags": ["baseline"],
                }
            }
        },
    )
    (config / "workflow_config_demo.yaml").write_text("model: {}\n", encoding="utf-8")
    if include_optional:
        _write_json(config / "ensemble_records.json", {"combos": {"default": "record_1"}, "default_combo": "default"})
        _write_yaml(
            config / "rolling_config.yaml",
            {
                "rolling_start": "2020-01-01",
                "train_years": 3,
                "valid_years": 1,
                "test_step": "3M",
                "training_method": "slide",
            },
        )
        _write_json(workspace / "latest_train_records.json", {"models": {"demo_linear_Alpha158": "record_1"}})
        _write_json(config / "cashflow.json", {"cash_flow_today": 0})
    return workspace


def test_validate_workspace_valid_minimal_with_optional(tmp_path):
    workspace = make_workspace(tmp_path)

    result = validate_workspace(WorkspaceContext.from_root(workspace))

    assert result.ok
    assert {artifact.name for artifact in result.artifacts} >= {"model_config", "ensemble_config"}
    assert all(artifact.fingerprint for artifact in result.artifacts if artifact.exists)


def test_required_config_missing_is_error(tmp_path):
    workspace = make_workspace(tmp_path)
    (workspace / "config" / "model_config.json").unlink()

    result = validate_workspace(WorkspaceContext.from_root(workspace))

    assert not result.ok
    assert any(message.code == "missing-config" and message.severity == "error" for message in result.messages)


def test_optional_config_missing_is_warning_and_strict_error(tmp_path):
    workspace = make_workspace(tmp_path, include_optional=False)

    normal = validate_workspace(WorkspaceContext.from_root(workspace))
    strict = validate_workspace(WorkspaceContext.from_root(workspace), strict=True)

    assert normal.ok
    assert any(message.code == "missing-config" and message.severity == "warning" for message in normal.messages)
    assert not strict.ok
    assert any(message.code == "missing-config" and message.severity == "error" for message in strict.messages)


def test_parse_errors_become_validation_messages(tmp_path):
    workspace = make_workspace(tmp_path)
    (workspace / "config" / "ensemble_config.json").write_text("{not-json", encoding="utf-8")

    result = validate_workspace(WorkspaceContext.from_root(workspace))

    assert not result.ok
    assert any(message.code == "parse-error" for message in result.messages)


def test_validation_does_not_write_files_or_change_cwd(tmp_path):
    workspace = make_workspace(tmp_path)
    before = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()}
    cwd = os.getcwd()

    validate_workspace(WorkspaceContext.from_root(workspace))

    after = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()}
    assert after == before
    assert os.getcwd() == cwd
