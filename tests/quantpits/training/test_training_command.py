import json
import os
import subprocess
import sys

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.errors import TrainingPlanError
from quantpits.utils.workspace import WorkspaceContext


def workspace(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    (root / "config" / "model_registry.yaml").write_text(
        "models:\n  demo:\n    enabled: true\n    algorithm: gru\n    dataset: Alpha158\n    yaml_file: demo.yaml\n"
    )
    (root / "config" / "model_config.json").write_text(
        json.dumps({"freq": "week", "train_date_mode": "last_trade_date"})
    )
    (root / "config" / "demo.yaml").write_text("model: {}\n")
    return root


def test_light_plan_is_deterministic_and_does_not_write(tmp_path):
    root = workspace(tmp_path)
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    options = TrainingRunOptions(family="static", action="full", explain_plan=True)
    first = prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options)
    second = prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options)
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert first.plan_fingerprint == second.plan_fingerprint
    assert first.anchor_resolution == "deferred_to_qlib_calendar"
    assert first.targets[0].key == "demo@static"
    assert before == after


def test_predict_plan_rejects_explicit_v2_downgrade(tmp_path):
    root = workspace(tmp_path)
    (root / "latest_train_records.json").write_text(json.dumps({
        "schema_version": 2, "models": {"demo@static": "r"}, "experiment_name": "stale",
    }))
    options = TrainingRunOptions(family="static", action="predict_only", all_enabled=True)
    with pytest.raises(TrainingPlanError, match="declared schema"):
        prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options)


def test_workflow_change_changes_plan_fingerprint(tmp_path):
    root = workspace(tmp_path)
    options = TrainingRunOptions(family="cpcv", action="full")
    first = prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options)
    (root / "config" / "demo.yaml").write_text("model: {changed: true}\n")
    second = prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options)
    assert first.plan_fingerprint != second.plan_fingerprint


@pytest.mark.parametrize(
    "module",
    ("quantpits.scripts.static_train", "quantpits.scripts.cv_train"),
)
def test_module_entrypoint_propagates_command_exit_code(tmp_path, module):
    root = workspace(tmp_path)
    environment = os.environ.copy()
    environment["QLIB_WORKSPACE_DIR"] = str(root)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--workspace",
            str(root),
            "--models",
            "missing_model",
            "--json-plan",
        ],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
