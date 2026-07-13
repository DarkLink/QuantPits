import importlib
import json

import pytest
import yaml
from unittest.mock import MagicMock, patch


@pytest.fixture
def script(monkeypatch, tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    (root / "config/model_config.json").write_text(json.dumps({
        "freq": "week", "train_date_mode": "last_trade_date",
        "purged_cv": {"n_groups": 10, "n_test_groups": 2, "n_val_groups": 1},
    }))
    (root / "config/model_registry.yaml").write_text(yaml.safe_dump({
        "models": {"demo": {"enabled": True, "yaml_file": "demo.yaml"}},
    }))
    (root / "config/demo.yaml").write_text("model: {}\n")
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(root))
    from quantpits.utils import env
    env.set_root_dir(str(root))
    import quantpits.scripts.cv_train as module
    module = importlib.reload(module)
    return module, root


def test_parser_exposes_plan_and_manifest_flags(script):
    module, _ = script
    args = module.parse_args([
        "--all-enabled", "--json-plan", "--run-id", "review", "--no-manifest",
    ])
    assert args.all_enabled is True
    assert args.json_plan is True
    assert args.run_id == "review"
    assert args.no_manifest is True


def test_json_plan_returns_before_safeguard_and_execution(script, capsys):
    module, root = script
    with patch("quantpits.utils.env.safeguard") as safeguard, patch(
        "quantpits.training.service.TrainingExecutionService.execute"
    ) as execute:
        assert module.main([
            "--workspace", str(root), "--all-enabled", "--json-plan",
        ]) == 0
    safeguard.assert_not_called()
    execute.assert_not_called()
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1


def test_real_command_routes_through_typed_service(script):
    module, root = script
    with patch("quantpits.utils.env.safeguard") as safeguard, patch(
        "quantpits.training.service.TrainingExecutionService.execute", return_value=MagicMock(),
    ) as execute:
        assert module.main(["--workspace", str(root), "--all-enabled"]) == 0
    safeguard.assert_called_once()
    execute.assert_called_once()


def test_show_state_is_read_only(script, capsys):
    module, root = script
    state = {"schema_version": 2, "status": "failed"}
    (root / "data/run_state.json").write_text(json.dumps(state))
    before = (root / "data/run_state.json").read_bytes()
    assert module.main(["--workspace", str(root), "--show-state"]) is None
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert (root / "data/run_state.json").read_bytes() == before


def test_script_contains_no_legacy_orchestration_entrypoints(script):
    module, _ = script
    assert not hasattr(module, "run_full_train_cpcv")
    assert not hasattr(module, "run_incremental_train_cpcv")
    assert not hasattr(module, "run_predict_only_cpcv")
