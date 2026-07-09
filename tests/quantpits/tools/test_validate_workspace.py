import json

import yaml

from quantpits.tools.validate_workspace import main


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


def test_cli_json_output_excludes_raw_config(tmp_path, capsys):
    workspace = make_workspace(tmp_path)

    exit_code = main(["--workspace", str(workspace), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert "raw" not in captured.out
    assert "current_holding" not in captured.out
    assert "current_cash" not in captured.out


def test_cli_invalid_workspace_returns_nonzero(tmp_path, capsys):
    missing = tmp_path / "missing"

    exit_code = main(["--workspace", str(missing)])

    assert exit_code == 1
    assert "Status: FAILED" in capsys.readouterr().out
