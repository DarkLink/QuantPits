import json

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.errors import TrainingDatePolicyError, TrainingExecutionError
from quantpits.training.resolved import resolve_training_run
from quantpits.utils.workspace import WorkspaceContext


def make_workspace(tmp_path, config):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir(); (root / "output").mkdir()
    (root / "config/model_registry.yaml").write_text("models:\n  demo:\n    enabled: true\n    yaml_file: demo.yaml\n")
    (root / "config/model_config.json").write_text(json.dumps(config))
    (root / "config/demo.yaml").write_text("model: {}\n")
    return root


def test_configured_date_uses_current_date(tmp_path):
    root = make_workspace(tmp_path, {"train_date_mode": "current_date", "current_date": "2026-07-10"})
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root), options=TrainingRunOptions("static", "full")
    )
    assert prepared.date_policy.configured_anchor == "2026-07-10"
    with pytest.raises(TrainingExecutionError):
        resolve_training_run(prepared, {"anchor_date": "2026-07-09", "freq": "week"})


def test_invalid_configured_date_fails_before_execution(tmp_path):
    root = make_workspace(tmp_path, {"train_date_mode": "current_date", "current_date": "latest"})
    with pytest.raises(TrainingDatePolicyError):
        prepare_training_run(
            ctx=WorkspaceContext.from_root(root), options=TrainingRunOptions("static", "full")
        )


def test_resolved_targets_are_exactly_the_prepared_targets(tmp_path):
    root = make_workspace(tmp_path, {"train_date_mode": "last_trade_date"})
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root), options=TrainingRunOptions("static", "full")
    )
    resolved = resolve_training_run(
        prepared, {"anchor_date": "2026-07-10", "test_end_time": "2026-07-10", "freq": "week"}
    )
    assert tuple(item.key for item in resolved.targets) == tuple(
        item.key for item in prepared.targets
    )
