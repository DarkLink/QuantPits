import json
import os
import subprocess
import sys

import pytest

from quantpits.training.command import (
    TrainingRunOptions, prepare_training_run, render_prepared_plan,
)
from quantpits.training.errors import TrainingPlanError
from quantpits.training.state import TrainingRunState, TrainingStateRepository
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


@pytest.mark.parametrize("family", ["static", "cpcv"])
def test_predict_plan_renders_exact_target_source_and_publication_identity(tmp_path, family):
    root = workspace(tmp_path)
    suffix = "cpcv" if family == "cpcv" else "static"
    (root / "latest_train_records.json").write_text(json.dumps({
        "models": {"demo@%s" % suffix: "source-recorder"},
        "experiment_name": "source-experiment",
    }))

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family=family, action="predict_only", all_enabled=True, explain_plan=True,
        ),
    )
    rendered = render_prepared_plan(prepared)

    assert [item.path for item in prepared.plan.inputs].count(
        "latest_train_records.json"
    ) == 1
    assert "current publication baseline and predict-only source records" in rendered
    assert "Target keys: demo@%s" % suffix in rendered
    assert "Publication policy: merge_successes" in rendered
    assert (
        "Source for demo@%s: experiment=source-experiment recorder=source-recorder "
        "operation=legacy_import status=legacy_unverified" % suffix
    ) in rendered
    assert prepared.plan.metadata["source_identities"] == [{
        "target_key": "demo@%s" % suffix,
        "recorder_id": "source-recorder",
        "experiment_name": "source-experiment",
        "operation": "legacy_import",
        "status": "legacy_unverified",
    }]


def test_workflow_change_changes_plan_fingerprint(tmp_path):
    root = workspace(tmp_path)
    options = TrainingRunOptions(family="cpcv", action="full")
    first = prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options)
    (root / "config" / "demo.yaml").write_text("model: {changed: true}\n")
    second = prepare_training_run(ctx=WorkspaceContext.from_root(root), options=options)
    assert first.plan_fingerprint != second.plan_fingerprint


def test_resume_adopts_persisted_run_id_without_creating_a_lock(tmp_path):
    root = workspace(tmp_path)
    ctx = WorkspaceContext.from_root(root)
    state = TrainingRunState(
        run_id="persisted-run", family="static", action="incremental",
        plan_fingerprint="plan", execution_fingerprint="execution",
        resume_fingerprint="resume", anchor_date="2026-07-10",
        target_keys=("demo@static",), outcomes={}, phase="executing",
    )
    TrainingStateRepository(root / "data/run_state.json").save(state)
    lock = root / "data/run_state.json.lock"
    lock.unlink()
    prepared = prepare_training_run(
        ctx=ctx, options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, resume=True,
        ),
    )
    assert prepared.plan.run_id == "persisted-run"
    assert prepared.resume_state.run_id == "persisted-run"
    assert prepared.plan.metadata["resume_identity_source"] == "persisted_state"
    assert not lock.exists()


def test_predict_only_resume_adopts_persisted_identity(tmp_path):
    root = workspace(tmp_path)
    (root / "latest_train_records.json").write_text(json.dumps({
        "models": {"demo@static": "source-recorder"},
        "experiment_name": "source-experiment",
    }))
    TrainingStateRepository(root / "data/run_state.json").save(TrainingRunState(
        run_id="predict-resume", family="static", action="predict_only",
        plan_fingerprint="plan", execution_fingerprint="execution",
        resume_fingerprint="resume", anchor_date="2026-07-10",
        target_keys=("demo@static",),
        outcomes={"demo@static": {
            "outcome": "failed", "published": False,
            "source_recorder_id": "source-recorder",
            "source_experiment_name": "source-experiment",
            "source_operation": "legacy_import",
        }},
        phase="failed",
    ))

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="predict_only", all_enabled=True, resume=True,
        ),
    )

    assert prepared.plan.run_id == "predict-resume"
    assert prepared.resume_state.source_identities == ((
        "demo@static", "source-recorder", "source-experiment", "legacy_import",
    ),)


def test_full_training_rejects_resume():
    with pytest.raises(TrainingPlanError, match="incremental or predict-only"):
        TrainingRunOptions(family="static", action="full", resume=True)


def test_resume_rejects_explicit_run_id_mismatch(tmp_path):
    root = workspace(tmp_path)
    TrainingStateRepository(root / "data/run_state.json").save(TrainingRunState(
        run_id="persisted-run", family="static", action="incremental",
        plan_fingerprint="plan", execution_fingerprint="execution",
        resume_fingerprint="resume", anchor_date="2026-07-10",
        target_keys=("demo@static",), outcomes={}, phase="executing",
    ))
    with pytest.raises(TrainingPlanError, match="run id"):
        prepare_training_run(
            ctx=WorkspaceContext.from_root(root),
            options=TrainingRunOptions(
                family="static", action="incremental", all_enabled=True,
                resume=True, run_id="different-run",
            ),
        )


@pytest.mark.parametrize(
    "persisted_models",
    [
        ("other",),
        ("demo", "extra"),
        ("extra", "demo"),
        (),
    ],
)
@pytest.mark.parametrize("family", ["static", "cpcv"])
def test_resume_rejects_target_selection_mismatch_during_preparation(
    tmp_path, persisted_models, family,
):
    root = workspace(tmp_path)
    persisted_keys = tuple("%s@%s" % (name, family) for name in persisted_models)
    TrainingStateRepository(root / "data/run_state.json").save(TrainingRunState(
        run_id="persisted-run", family=family, action="incremental",
        plan_fingerprint="plan", execution_fingerprint="execution",
        resume_fingerprint="resume", anchor_date="2026-07-10",
        target_keys=persisted_keys, outcomes={}, phase="executing",
    ))

    with pytest.raises(
        TrainingPlanError, match="resume target selection differs from persisted state",
    ):
        prepare_training_run(
            ctx=WorkspaceContext.from_root(root),
            options=TrainingRunOptions(
                family=family, action="incremental", all_enabled=True, resume=True,
            ),
        )


def test_resume_rejects_legacy_state_before_execution(tmp_path):
    root = workspace(tmp_path)
    (root / "data/run_state.json").write_text(json.dumps({
        "schema_version": 2, "run_id": "legacy-run", "status": "running",
    }))
    with pytest.raises(TrainingPlanError, match="unsupported"):
        prepare_training_run(
            ctx=WorkspaceContext.from_root(root),
            options=TrainingRunOptions(
                family="static", action="incremental", all_enabled=True, resume=True,
            ),
        )


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
