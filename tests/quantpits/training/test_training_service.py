import json

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.errors import TrainingExecutionError
from quantpits.training.records import ModelRecordEntry
from quantpits.training.runners import TrainingTargetResult
from quantpits.training.service import TrainingExecutionHooks, TrainingExecutionService
from quantpits.utils.workspace import WorkspaceContext


def workspace(tmp_path, models=("demo",)):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    registry = ["models:"]
    for name in models:
        registry.extend([
            "  %s:" % name, "    enabled: true", "    yaml_file: %s.yaml" % name,
        ])
        (root / "config" / (name + ".yaml")).write_text("model: {}\n")
    (root / "config" / "model_registry.yaml").write_text("\n".join(registry) + "\n")
    (root / "config" / "model_config.json").write_text(json.dumps({"freq": "week"}))
    return root


def result_for(request, *, outcome="success"):
    target = request.target
    if outcome != "success":
        return TrainingTargetResult(target.key, target.operation, "failed", error_code="fit_failed")
    name, mode = target.key.rsplit("@", 1)
    entry = ModelRecordEntry(
        target.key, name, mode, target.operation, "ready", "rid-" + name,
        request.resolved_run.output_experiment_name,
        requested_anchor=request.resolved_run.anchor_date,
        prediction_start=request.resolved_run.anchor_date,
        prediction_end=request.resolved_run.anchor_date,
        prediction_rows=1,
    )
    return TrainingTargetResult(target.key, target.operation, "success", entry, {"IC_Mean": 0.1})


def hooks(calls, runner=result_for):
    return TrainingExecutionHooks(
        activate_workspace=lambda value: calls.append(("activate", value)),
        init_qlib=lambda: calls.append(("init", None)),
        calculate_dates=lambda: {"anchor_date": "2026-07-10", "test_end_time": "2026-07-10", "freq": "week"},
        run_static_target=lambda request: runner(request),
        run_cpcv_target=lambda request: runner(request),
    )


def test_service_owns_exact_target_loop_and_publication(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(family="static", action="full", run_id="demo-run"),
    )
    calls = []
    summary = TrainingExecutionService(hooks(calls)).execute(prepared)
    assert [item[0] for item in calls] == ["activate", "init"]
    assert summary.manifest_path == "output/manifests/static_train/demo-run.json"
    assert summary.execution_fingerprint
    assert summary.outcomes[0]["published"] is True
    assert json.loads((root / "latest_train_records.json").read_text())["models"] == {"demo@static": "rid-demo"}
    assert not (root / "data" / "run_state.json").exists()


def test_full_failure_preserves_current_record_and_retains_state(tmp_path):
    root = workspace(tmp_path, ("a", "b"))
    original = b'{"experiment_name":"old","models":{}}\n'
    (root / "latest_train_records.json").write_bytes(original)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(family="static", action="full", run_id="failed-run"),
    )
    def runner(request):
        return result_for(request, outcome="failed" if request.target.key == "b@static" else "success")
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(hooks([], runner)).execute(prepared)
    assert (root / "latest_train_records.json").read_bytes() == original
    manifest = json.loads((root / "output/manifests/static_train/failed-run.json").read_text())
    assert manifest["status"] == "failed"
    assert all(item.get("published") is not True for item in manifest["records"]["outcomes"])
    assert (root / "data" / "run_state.json").is_file()


def test_no_manifest_still_writes_operator_log(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(family="static", action="full", no_manifest=True),
    )
    summary = TrainingExecutionService(hooks([])).execute(prepared)
    assert summary.manifest_path is None
    assert (root / "data" / "operator_log.jsonl").is_file()


def test_incremental_partial_failure_publishes_only_successes_and_fails_command(tmp_path):
    root = workspace(tmp_path, ("a", "b"))
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            run_id="partial-run",
        ),
    )

    def runner(request):
        return result_for(
            request, outcome="failed" if request.target.key == "b@static" else "success"
        )

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(hooks([], runner)).execute(prepared)

    record = json.loads((root / "latest_train_records.json").read_text())
    assert record["models"] == {"a@static": "rid-a"}
    manifest = json.loads((root / "output/manifests/static_train/partial-run.json").read_text())
    assert manifest["status"] == "failed"
    outcomes = {item["key"]: item for item in manifest["records"]["outcomes"]}
    assert outcomes["a@static"]["published"] is True
    assert outcomes["b@static"]["outcome"] == "failed"
