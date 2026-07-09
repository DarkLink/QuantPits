import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@dataclass(frozen=True)
class _ValidationResult:
    workspace: Path
    ok: bool = True
    artifacts: tuple = ()
    messages: tuple = ()


@pytest.fixture
def ef_module(monkeypatch, tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    (workspace / "config").mkdir(parents=True)
    (workspace / "data").mkdir()
    (workspace / "output").mkdir()
    (workspace / "latest_train_records.json").write_text('{"models": {}}\n')

    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(sys, "argv", ["ensemble_fusion.py"])

    import importlib

    for mod_name in ["quantpits.utils.env", "quantpits.scripts.ensemble_fusion"]:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from quantpits.scripts import ensemble_fusion as ef

    monkeypatch.setattr(ef, "ROOT_DIR", str(workspace))
    yield ef, workspace


def _configs():
    train_records = {
        "anchor_date": "2026-07-09",
        "experiment_name": "DemoExp",
        "models": {"m1@static": "rid1"},
    }
    model_config = {"freq": "week", "TopK": 20}
    ensemble_config = {
        "combos": {
            "default_combo": {
                "models": ["m1"],
                "method": "equal",
                "default": True,
            }
        }
    }
    return train_records, model_config, ensemble_config


def test_explain_plan_does_not_run_heavy_paths(ef_module, capsys):
    ef, workspace = ef_module

    with patch("quantpits.config_contracts.workspace.validate_workspace",
               return_value=_ValidationResult(workspace=workspace)):
        with patch.object(ef, "load_config", return_value=_configs()):
            with patch.object(ef, "init_qlib") as init_qlib:
                with patch.object(ef, "load_selected_predictions") as load_preds:
                    with patch.object(ef, "run_single_combo") as run_combo:
                        with patch("quantpits.utils.env.safeguard") as safeguard:
                            with patch("quantpits.utils.operator_log.OperatorLog") as oplog:
                                ef.main(["--from-config", "--explain-plan", "--run-id", "dry-run"])

    captured = capsys.readouterr().out
    assert "--- Execution Plan (dry run) ---" in captured
    assert "Command: ensemble_fusion" in captured
    assert "Plan fingerprint:" in captured
    init_qlib.assert_not_called()
    load_preds.assert_not_called()
    run_combo.assert_not_called()
    safeguard.assert_not_called()
    oplog.assert_not_called()
    assert not (workspace / "output" / "manifests").exists()


def test_json_plan_outputs_parseable_json_only(ef_module, capsys):
    ef, workspace = ef_module

    with patch("quantpits.config_contracts.workspace.validate_workspace",
               return_value=_ValidationResult(workspace=workspace)):
        with patch.object(ef, "load_config", return_value=_configs()):
            ef.main(["--from-config", "--json-plan", "--run-id", "json-run"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["plan_fingerprint"]
    assert payload["plan"]["run_id"] == "json-run"
    assert payload["plan"]["command"] == "ensemble_fusion"


def test_mocked_run_writes_manifest_and_operator_log_linkage(ef_module):
    ef, workspace = ef_module

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-07-09"), "AAA")],
        names=["datetime", "instrument"],
    )
    norm_df = pd.DataFrame({"m1@static": [0.5]}, index=idx)
    combo_result = {
        "name": "default_combo",
        "models": ["m1@static"],
        "method": "equal",
        "is_default": True,
        "pred_file": "recorder-id",
        "report_df": None,
    }

    with patch("quantpits.config_contracts.workspace.validate_workspace",
               return_value=_ValidationResult(workspace=workspace)):
        with patch.object(ef, "load_config", return_value=_configs()):
            with patch("quantpits.utils.env.safeguard"):
                with patch.object(ef, "init_qlib"):
                    with patch.object(ef, "load_selected_predictions",
                                      return_value=(norm_df, {"m1@static": 0.1}, ["m1@static"])):
                        with patch.object(ef, "filter_norm_df_by_args", return_value=norm_df):
                            with patch.object(ef, "run_single_combo", return_value=combo_result):
                                ef.main([
                                    "--from-config",
                                    "--run-id",
                                    "mock-run",
                                    "--no-backtest",
                                    "--no-charts",
                                ])

    manifest_path = workspace / "output" / "manifests" / "ensemble_fusion" / "mock-run.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "success"
    assert manifest["run_id"] == "mock-run"
    assert manifest["records"]["n_combos"] == 1
    assert manifest["records"]["combos"][0]["name"] == "default_combo"
    assert any(
        output["path"] == "output/manifests/ensemble_fusion/mock-run.json"
        for output in manifest["outputs"]
    )

    log_entries = (workspace / "data" / "operator_log.jsonl").read_text().strip().splitlines()
    entry = json.loads(log_entries[-1])
    assert entry["run_id"] == "mock-run"
    assert entry["manifest_path"] == "output/manifests/ensemble_fusion/mock-run.json"
    assert entry["plan_fingerprint"]


def test_no_manifest_keeps_operator_log_without_manifest_path(ef_module):
    ef, workspace = ef_module

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-07-09"), "AAA")],
        names=["datetime", "instrument"],
    )
    norm_df = pd.DataFrame({"m1@static": [0.5]}, index=idx)
    combo_result = {
        "name": "default_combo",
        "models": ["m1@static"],
        "method": "equal",
        "is_default": True,
        "pred_file": "recorder-id",
        "report_df": None,
    }

    with patch("quantpits.config_contracts.workspace.validate_workspace",
               return_value=_ValidationResult(workspace=workspace)):
        with patch.object(ef, "load_config", return_value=_configs()):
            with patch("quantpits.utils.env.safeguard"):
                with patch.object(ef, "init_qlib"):
                    with patch.object(ef, "load_selected_predictions",
                                      return_value=(norm_df, {"m1@static": 0.1}, ["m1@static"])):
                        with patch.object(ef, "filter_norm_df_by_args", return_value=norm_df):
                            with patch.object(ef, "run_single_combo", return_value=combo_result):
                                ef.main([
                                    "--from-config",
                                    "--run-id",
                                    "no-manifest-run",
                                    "--no-backtest",
                                    "--no-charts",
                                    "--no-manifest",
                                ])

    assert not (workspace / "output" / "manifests").exists()
    entry = json.loads((workspace / "data" / "operator_log.jsonl").read_text().strip())
    assert entry["run_id"] == "no-manifest-run"
    assert entry["manifest_path"] is None
    assert entry["plan_fingerprint"]


def test_execution_failure_writes_failed_manifest_and_reraises(ef_module):
    ef, workspace = ef_module

    with patch("quantpits.config_contracts.workspace.validate_workspace",
               return_value=_ValidationResult(workspace=workspace)):
        with patch.object(ef, "load_config", return_value=_configs()):
            with patch("quantpits.utils.env.safeguard"):
                with patch.object(ef, "init_qlib"):
                    with patch.object(ef, "load_selected_predictions",
                                      side_effect=RuntimeError("recorder missing")):
                        with pytest.raises(RuntimeError, match="recorder missing"):
                            ef.main([
                                "--from-config",
                                "--run-id",
                                "failed-run",
                                "--no-backtest",
                                "--no-charts",
                            ])

    manifest_path = workspace / "output" / "manifests" / "ensemble_fusion" / "failed-run.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["error"]["message"] == "recorder missing"

    entry = json.loads((workspace / "data" / "operator_log.jsonl").read_text().strip())
    assert entry["run_id"] == "failed-run"
    assert entry["manifest_path"] == "output/manifests/ensemble_fusion/failed-run.json"
