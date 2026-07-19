import json

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.errors import TrainingDatePolicyError, TrainingExecutionError
from quantpits.training.records import ModelRecordEntry, TrainingRecordSnapshot
from quantpits.training.resolved import resolve_training_run
from quantpits.training.state import TrainingRunState, TrainingStateRepository
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


def test_predict_resume_keeps_original_sources_after_partial_publication(tmp_path):
    root = make_workspace(tmp_path, {"train_date_mode": "last_trade_date"})
    source = {
        "models": {"demo@static": "source-recorder"},
        "experiment_name": "source-experiment",
    }
    (root / "latest_train_records.json").write_text(json.dumps(source))
    options = TrainingRunOptions(
        family="static", action="predict_only", all_enabled=True,
        run_id="predict-resume",
    )
    initial = resolve_training_run(
        prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options),
        {"anchor_date": "2026-07-10", "test_end_time": "2026-07-10", "freq": "week"},
    )
    TrainingStateRepository(root / "data/run_state.json").save(TrainingRunState(
        run_id="predict-resume", family="static", action="predict_only",
        plan_fingerprint=initial.prepared.plan_fingerprint,
        execution_fingerprint=initial.execution_fingerprint,
        resume_fingerprint=initial.resume_fingerprint,
        anchor_date="2026-07-10", target_keys=("demo@static",),
        outcomes={"demo@static": {
            "outcome": "success", "published": True,
            "recorder_id": "output-recorder",
            "source_recorder_id": "source-recorder",
            "source_experiment_name": "source-experiment",
            "source_operation": "legacy_import",
        }}, phase="failed",
    ))
    output = ModelRecordEntry(
        key="demo@static", model_name="demo", training_mode="static",
        operation="predict_only", status="ready",
        recorder_id="output-recorder", experiment_name="output-experiment",
        requested_anchor="2026-07-10", prediction_start="2026-07-01",
        prediction_end="2026-07-10", prediction_rows=2,
        dataset_test_end="2026-07-10",
        source_recorder_id="source-recorder",
        source_experiment_name="source-experiment",
        source_operation="legacy_import",
    )
    (root / "latest_train_records.json").write_text(json.dumps(
        TrainingRecordSnapshot((output,)).to_dict()
    ))

    resumed = resolve_training_run(
        prepare_training_run(
            ctx=WorkspaceContext.from_root(root),
            options=TrainingRunOptions(
                family="static", action="predict_only", all_enabled=True, resume=True,
            ),
        ),
        {"anchor_date": "2026-07-10", "test_end_time": "2026-07-10", "freq": "week"},
    )

    assert resumed.resume_fingerprint == initial.resume_fingerprint
