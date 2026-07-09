import json

import yaml

from quantpits.utils.config_loader import (
    load_workspace_config,
    load_workspace_config_artifacts,
    load_workspace_config_with_metadata,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def make_workspace(tmp_path):
    workspace = tmp_path / "Workspace"
    config = workspace / "config"
    config.mkdir(parents=True)
    _write_json(config / "model_config.json", {"market": "csi300", "benchmark": "SH000300", "start_time": "2020-01-01", "freq": "week"})
    _write_yaml(
        config / "strategy_config.yaml",
        {
            "strategy": {"name": "topk_dropout", "params": {"topk": 20, "n_drop": 3}},
            "backtest": {"account": 100000, "exchange_kwargs": {"deal_price": "close", "open_cost": 0.001, "close_cost": 0.001, "min_cost": 5}},
        },
    )
    _write_json(config / "prod_config.json", {"current_date": "2024-01-01", "last_processed_date": "2024-01-01", "current_cash": 100000, "current_holding": []})
    _write_json(config / "ensemble_config.json", {"combos": {"default": {"models": ["demo_linear_Alpha158"], "method": "equal", "default": True}}})
    _write_yaml(config / "model_registry.yaml", {"models": {"demo_linear_Alpha158": {"yaml_file": "config/workflow_config_demo.yaml", "enabled": True, "tags": ["baseline"]}}})
    (config / "workflow_config_demo.yaml").write_text("model: {}\n", encoding="utf-8")
    return workspace


def test_config_loader_legacy_interface_unchanged(tmp_path):
    workspace = make_workspace(tmp_path)

    config = load_workspace_config(workspace)

    assert config["market"] == "csi300"
    assert config["topk"] == 20
    assert config["TopK"] == 20
    assert config["current_cash"] == 100000


def test_config_loader_metadata_helpers(tmp_path):
    workspace = make_workspace(tmp_path)

    result = load_workspace_config_artifacts(workspace)
    config, metadata = load_workspace_config_with_metadata(workspace)

    assert result.ok
    assert config["market"] == "csi300"
    assert metadata["ok"] is True
    assert "model_config" in metadata["fingerprints"]
    assert all("raw" not in artifact for artifact in metadata["artifacts"])
