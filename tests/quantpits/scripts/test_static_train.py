import importlib
import json
from pathlib import Path

import pytest
import yaml
from unittest.mock import MagicMock, patch

from quantpits.training.state import TrainingRunState, TrainingStateRepository


@pytest.fixture
def script(monkeypatch, tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    (root / "config/model_config.json").write_text(json.dumps({
        "freq": "week", "train_date_mode": "last_trade_date",
    }))
    (root / "config/model_registry.yaml").write_text(yaml.safe_dump({
        "models": {"demo": {"enabled": True, "yaml_file": "demo.yaml"}},
    }))
    (root / "config/demo.yaml").write_text("model: {}\n")
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(root))
    from quantpits.utils import env
    env.set_root_dir(str(root))
    import quantpits.scripts.static_train as module
    module = importlib.reload(module)
    return module, root


def test_parser_exposes_plan_and_manifest_flags(script):
    module, _ = script
    args = module.parse_args([
        "--models", "demo", "--explain-plan", "--run-id", "review", "--no-manifest",
    ])
    assert args.models == "demo"
    assert args.explain_plan is True
    assert args.run_id == "review"
    assert args.no_manifest is True


def test_chinese_and_english_resume_examples_match_the_static_parser(script):
    module, _ = script
    command = (
        "python quantpits/scripts/static_train.py "
        "--models gru,mlp,alstm_Alpha158 --resume"
    )
    repository = Path(__file__).resolve().parents[3]
    assert command in (repository / "docs/01_TRAINING_GUIDE.md").read_text()
    assert command in (repository / "docs/en/01_TRAINING_GUIDE.md").read_text()

    args = module.parse_args(command.split()[2:])
    assert args.models == "gru,mlp,alstm_Alpha158"
    assert args.resume is True


def test_explain_plan_returns_before_safeguard_and_execution(script, capsys):
    module, root = script
    with patch("quantpits.utils.env.safeguard") as safeguard, patch(
        "quantpits.training.service.TrainingExecutionService.execute"
    ) as execute:
        assert module.main([
            "--workspace", str(root), "--full", "--explain-plan",
        ]) == 0
    safeguard.assert_not_called()
    execute.assert_not_called()
    assert "Plan fingerprint:" in capsys.readouterr().out


def test_real_command_delegates_to_typed_service(script):
    module, root = script
    summary = MagicMock(outcomes=())
    with patch("quantpits.utils.env.safeguard") as safeguard, patch(
        "quantpits.training.service.TrainingExecutionService.execute", return_value=summary,
    ) as execute:
        assert module.main([
            "--workspace", str(root), "--full",
        ]) == 0
    safeguard.assert_called_once()
    execute.assert_called_once()


def test_predict_only_missing_source_fails_before_safeguard(script):
    module, root = script
    with patch("quantpits.utils.env.safeguard") as safeguard, patch(
        "quantpits.training.service.TrainingExecutionService.execute"
    ) as execute:
        assert module.main([
            "--workspace", str(root), "--predict-only", "--all-enabled",
        ]) == 1
    safeguard.assert_not_called()
    execute.assert_not_called()


def test_resume_target_mismatch_reaches_no_runtime_hook(script):
    module, root = script
    TrainingStateRepository(root / "data/run_state.json").save(TrainingRunState(
        run_id="persisted-run", family="static", action="incremental",
        plan_fingerprint="plan", execution_fingerprint="execution",
        resume_fingerprint="resume", anchor_date="2026-07-10",
        target_keys=("other@static",), outcomes={}, phase="executing",
    ))

    with patch("quantpits.utils.env.safeguard") as safeguard, patch(
        "quantpits.utils.env.set_root_dir"
    ) as activate_workspace, patch("quantpits.utils.env.init_qlib") as init_qlib, patch(
        "quantpits.training.service.default_execution_hooks"
    ) as default_hooks, patch(
        "quantpits.training.service.TrainingExecutionService.execute"
    ) as execute:
        assert module.main([
            "--workspace", str(root), "--models", "demo", "--resume",
        ]) == 1

    safeguard.assert_not_called()
    activate_workspace.assert_not_called()
    init_qlib.assert_not_called()
    default_hooks.assert_not_called()
    execute.assert_not_called()


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
    assert not hasattr(module, "run_full_train")
    assert not hasattr(module, "run_incremental_train")
    assert not hasattr(module, "run_predict_only")
