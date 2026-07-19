import json
from types import SimpleNamespace

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.errors import TrainingExecutionError, TrainingRecorderIntegrityError
from quantpits.training.records import ModelRecordEntry
from quantpits.training.runners import TrainingTargetResult
from quantpits.training.service import (
    TrainingExecutionHooks, TrainingExecutionService, _verify_source_recorder,
)
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
    import hashlib

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
    receipt_path = next((root / "data/training_transactions").glob("*/receipt.json"))
    closure_path = next((root / "data/training_runs/demo-run").glob("closure-*.json"))
    closure = json.loads(closure_path.read_text())
    assert closure["receipt_fingerprint"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert closure["manifest_fingerprint"]
    assert closure["operator_log_fingerprint"]


def test_predict_only_rejects_source_identity_before_runner_or_publication(tmp_path):
    root = workspace(tmp_path)
    source = {
        "experiment_name": "Declared_Source",
        "models": {"demo@cpcv": "source-recorder"},
    }
    record_path = root / "latest_train_records.json"
    record_path.write_text(json.dumps(source))
    before = record_path.read_bytes()
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="cpcv", action="predict_only", all_enabled=True,
            run_id="source-integrity-failure",
        ),
    )
    base = hooks([])
    verified = []

    def reject_source(run, target):
        verified.append((run.prepared.plan.run_id, target.key))
        raise TrainingRecorderIntegrityError("source_recorder_experiment_mismatch")

    configured = TrainingExecutionHooks(
        activate_workspace=base.activate_workspace,
        init_qlib=base.init_qlib,
        calculate_dates=lambda: {
            "anchor_date": "2026-07-10", "test_end_time": "2026-07-10",
            "freq": "week", "cpcv_folds": [{"fold_idx": 0}],
        },
        run_static_target=base.run_static_target,
        run_cpcv_target=lambda _request: pytest.fail("target runner must not start"),
        verify_source_target=reject_source,
    )

    with pytest.raises(
        TrainingRecorderIntegrityError, match="source_recorder_experiment_mismatch"
    ):
        TrainingExecutionService(configured).execute(prepared)

    assert verified == [("source-integrity-failure", "demo@cpcv")]
    assert record_path.read_bytes() == before
    assert not (root / "data/training_transactions").exists()
    assert not (root / "data/training_runs/source-integrity-failure/targets").exists()
    state = json.loads((root / "data/run_state.json").read_text())
    assert state["phase"] == "failed"
    assert state["aggregate_error_code"] == "training_recorder_integrity_error"
    manifest = json.loads(
        (root / "output/manifests/cv_train/source-integrity-failure.json").read_text()
    )
    assert manifest["status"] == "failed"


def test_failure_diagnostic_does_not_overwrite_foreign_attempt_state(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            run_id="foreign-attempt-state",
        ),
    )
    base = hooks([])
    state_path = root / "data/run_state.json"

    def replace_attempt_then_fail(point, _detail=None):
        if point != "after_target_state":
            return
        state = json.loads(state_path.read_text())
        state["attempt_id"] = "foreign-attempt"
        state["aggregate_error_code"] = "foreign-owner"
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
        raise RuntimeError("injected foreign state replacement")

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace,
            init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates,
            run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target,
            fault_hook=replace_attempt_then_fail,
        )).execute(prepared)

    state = json.loads(state_path.read_text())
    assert state["phase"] == "executing"
    assert state["attempt_id"] == "foreign-attempt"
    assert state["aggregate_error_code"] == "foreign-owner"


@pytest.mark.parametrize(
    "actual_experiment,artifact_relative,error",
    [
        ("Actual_Source", "mlruns/1/rid/artifacts", "source_recorder_experiment_mismatch"),
        ("Declared_Source", None, "source_recorder_artifact_outside_workspace"),
    ],
)
def test_default_source_verifier_rejects_false_or_external_mlflow_identity(
    tmp_path, monkeypatch, actual_experiment, artifact_relative, error,
):
    root = workspace(tmp_path)
    source = ModelRecordEntry(
        "demo@cpcv", "demo", "cpcv", "legacy_import", "legacy_unverified",
        "source-recorder", "Declared_Source",
    )
    artifact = (
        (root / artifact_relative).resolve().as_uri()
        if artifact_relative is not None
        else (tmp_path / "outside" / "artifacts").resolve().as_uri()
    )

    class Client:
        def get_run(self, recorder_id):
            assert recorder_id == "source-recorder"
            return SimpleNamespace(info=SimpleNamespace(
                run_id=recorder_id, experiment_id="1", artifact_uri=artifact,
            ))

        def get_experiment(self, experiment_id):
            assert experiment_id == "1"
            return SimpleNamespace(name=actual_experiment)

    import mlflow.tracking
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Client)
    run = SimpleNamespace(prepared=SimpleNamespace(ctx=WorkspaceContext.from_root(root)))
    target = SimpleNamespace(source_entry=source)

    with pytest.raises(TrainingRecorderIntegrityError, match=error):
        _verify_source_recorder(run, target)


def test_default_source_verifier_accepts_exact_contained_mlflow_identity(
    tmp_path, monkeypatch,
):
    root = workspace(tmp_path)
    source = ModelRecordEntry(
        "demo@static", "demo", "static", "legacy_import", "legacy_unverified",
        "source-recorder", "Declared_Source",
    )

    class Client:
        def get_run(self, recorder_id):
            return SimpleNamespace(info=SimpleNamespace(
                run_id=recorder_id, experiment_id="1",
                artifact_uri=(root / "mlruns/1/rid/artifacts").resolve().as_uri(),
            ))

        def get_experiment(self, experiment_id):
            return SimpleNamespace(name="Declared_Source")

    import mlflow.tracking
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Client)
    run = SimpleNamespace(prepared=SimpleNamespace(ctx=WorkspaceContext.from_root(root)))

    _verify_source_recorder(run, SimpleNamespace(source_entry=source))


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
    assert not (root / "output/manifests").exists()


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


def test_partial_static_publication_promotes_success_before_aggregate_failure(tmp_path):
    root = workspace(tmp_path, ("a", "b"))
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id="promote-run",
        ),
    )
    promoted = []
    base = hooks([], lambda request: result_for(
        request, outcome="failed" if request.target.key == "b@static" else "success"
    ))
    configured = TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
        run_cpcv_target=base.run_cpcv_target,
        promote_static=lambda workspace_root, names: promoted.append((workspace_root, names)),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(configured).execute(prepared)
    assert promoted == [(str(root.resolve()), ("a",))]


def test_resume_preserves_receipt_outputs_and_reruns_only_failed_target(tmp_path):
    root = workspace(tmp_path, ("a", "b"))
    first = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id="resume-run",
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(hooks([], lambda request: result_for(
            request, outcome="failed" if request.target.key == "b@static" else "success"
        ))).execute(first)

    called = []
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id="resume-run",
        ),
    )

    def runner(request):
        called.append(request.target.key)
        return result_for(request)

    summary = TrainingExecutionService(hooks([], runner)).execute(resumed)
    assert called == ["b@static"]
    outcomes = {item["key"]: item for item in summary.outcomes}
    assert outcomes["a@static"]["already_published"] is True
    assert outcomes["b@static"]["published_this_attempt"] is True
    assert not (root / "data/run_state.json").exists()


def test_predict_resume_preserves_success_and_reruns_only_failed_target(tmp_path):
    root = workspace(tmp_path, ("a", "b"))
    (root / "latest_train_records.json").write_text(json.dumps({
        "models": {
            "a@static": "source-a",
            "b@static": "source-b",
        },
        "experiment_name": "source-experiment",
    }))
    first = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="predict_only", all_enabled=True,
            run_id="predict-resume-run",
        ),
    )

    def prediction_result(request, *, outcome="success"):
        target = request.target
        if outcome == "failed":
            return TrainingTargetResult(
                target.key, target.operation, "failed", error_code="predict_failed"
            )
        name, mode = target.key.rsplit("@", 1)
        source = target.source_entry
        entry = ModelRecordEntry(
            target.key, name, mode, target.operation, "ready", "output-" + name,
            request.resolved_run.output_experiment_name,
            requested_anchor=request.resolved_run.anchor_date,
            prediction_start=request.resolved_run.anchor_date,
            prediction_end=request.resolved_run.anchor_date,
            prediction_rows=1,
            source_recorder_id=source.recorder_id,
            source_experiment_name=source.experiment_name,
            source_operation=source.operation,
        )
        return TrainingTargetResult(
            target.key, target.operation, "success", entry, {"IC_Mean": 0.1}
        )

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(hooks([], lambda request: prediction_result(
            request,
            outcome="failed" if request.target.key == "b@static" else "success",
        ))).execute(first)

    called = []
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="predict_only", all_enabled=True,
            resume=True, run_id="predict-resume-run",
        ),
    )

    def retry(request):
        called.append(request.target.key)
        return prediction_result(request)

    summary = TrainingExecutionService(hooks([], retry)).execute(resumed)

    assert called == ["b@static"]
    outcomes = {item["key"]: item for item in summary.outcomes}
    assert outcomes["a@static"]["already_published"] is True
    assert outcomes["b@static"]["published_this_attempt"] is True
    assert not (root / "data/run_state.json").exists()


def test_resume_retries_warning_closure_without_rerunning_target(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            run_id="closure-retry",
        ),
    )
    promotion_calls = []
    base = hooks([])

    def flaky_promotion(_workspace_root, names):
        promotion_calls.append(names)
        if len(promotion_calls) == 1:
            raise RuntimeError("temporary promotion failure")

    first_hooks = TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
        run_cpcv_target=base.run_cpcv_target, promote_static=flaky_promotion,
    )
    first = TrainingExecutionService(first_hooks).execute(prepared)
    assert first.publication_applied is True
    state = json.loads((root / "data/run_state.json").read_text())
    assert state["phase"] == "closing"
    assert state["closure_steps"]["promotion_applied"] == "warning"

    rerun_targets = []
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True,
        ),
    )
    second_hooks = TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        run_static_target=lambda request: rerun_targets.append(request.target.key),
        run_cpcv_target=base.run_cpcv_target, promote_static=flaky_promotion,
    )
    TrainingExecutionService(second_hooks).execute(resumed)
    assert rerun_targets == []
    assert promotion_calls == [("demo",), ("demo",)]
    assert not (root / "data/run_state.json").exists()


@pytest.mark.parametrize("fault_point", [
    "after_target_evidence", "after_target_state",
    "after_intent_write", "before_receipt_write", "after_receipt_write",
    "after_history_closure", "after_promotion_closure",
    "after_manifest_closure", "after_operator_log_append", "after_terminal_state",
])
def test_resume_closes_fault_boundaries_without_rerunning_success(tmp_path, fault_point):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            run_id="fault-" + fault_point,
        ),
    )
    fired = []
    base = hooks([])

    def fault(point, _detail=None):
        if point == fault_point and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    interrupted_hooks = TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
        run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(interrupted_hooks).execute(prepared)

    rerun_targets = []
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, resume=True,
            run_id="fault-" + fault_point,
        ),
    )
    resume_hooks = TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        run_static_target=lambda request: rerun_targets.append(request.target.key),
        run_cpcv_target=base.run_cpcv_target,
    )
    TrainingExecutionService(resume_hooks).execute(resumed)
    assert rerun_targets == []
    assert not (root / "data/run_state.json").exists()


def test_closure_only_resume_preserves_every_receipt_authorized_output_byte(tmp_path):
    root = workspace(tmp_path)
    run_id = "closure-byte-preservation"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_receipt_write" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    receipt_path = next((root / "data/training_transactions").glob("*/receipt.json"))
    receipt = json.loads(receipt_path.read_text())
    before = {
        member["path"]: (root / member["path"]).read_bytes()
        for member in receipt["committed_outputs"]
    }
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )
    TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        run_static_target=lambda _request: pytest.fail("target must not rerun"),
        run_cpcv_target=base.run_cpcv_target,
    )).execute(resumed)

    assert {
        path: (root / path).read_bytes() for path in before
    } == before


def test_resume_revalidates_external_recorder_before_reusing_target_evidence(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            run_id="verify-evidence",
        ),
    )
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_evidence" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    verified = []
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id="verify-evidence",
        ),
    )
    TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        run_static_target=lambda _request: pytest.fail("target must not rerun"),
        run_cpcv_target=base.run_cpcv_target,
        verify_reusable_target=lambda run, evidence: verified.append(
            (run.prepared.plan.run_id, evidence.target_key)
        ),
    )).execute(resumed)
    assert verified == [("verify-evidence", "demo@static")]


def test_targets_complete_resume_prepares_publication_without_runner_or_cache(tmp_path):
    root = workspace(tmp_path)
    run_id = "targets-complete"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_state" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    state_path = root / "data/run_state.json"
    state = json.loads(state_path.read_text())
    state["phase"] = state["status"] = "targets_complete"
    state.pop("aggregate_error_code", None)
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )

    TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        prepare_cache=lambda _run: pytest.fail("cache must not be prepared"),
        run_static_target=lambda _request: pytest.fail("target must not rerun"),
        run_cpcv_target=base.run_cpcv_target,
    )).execute(resumed)
    assert not state_path.exists()


def test_resume_rejects_missing_target_evidence_at_service_boundary(tmp_path):
    root = workspace(tmp_path)
    run_id = "missing-evidence"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_state" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    state_path = root / "data/run_state.json"
    state = json.loads(state_path.read_text())
    evidence_path = root / state["outcomes"]["demo@static"]["target_evidence_path"]
    evidence_path.unlink()
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates,
            run_static_target=lambda _request: pytest.fail("target must not rerun"),
            run_cpcv_target=base.run_cpcv_target,
        )).execute(resumed)
    assert json.loads(state_path.read_text())["aggregate_error_code"] == "target_evidence_mismatch"


@pytest.mark.parametrize(
    "relative_path",
    ["../outside-target-evidence.json", "data/training_runs/other/targets/demo.json"],
)
def test_resume_rejects_target_evidence_path_mismatch(tmp_path, relative_path):
    root = workspace(tmp_path)
    run_id = "evidence-path-mismatch"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_state" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    state_path = root / "data/run_state.json"
    state = json.loads(state_path.read_text())
    state["outcomes"]["demo@static"]["target_evidence_path"] = relative_path
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates,
            run_static_target=lambda _request: pytest.fail("target must not rerun"),
            run_cpcv_target=base.run_cpcv_target,
        )).execute(resumed)
    assert json.loads(state_path.read_text())["aggregate_error_code"] == "target_evidence_mismatch"


def test_resume_rejects_well_formed_evidence_operation_mismatch(tmp_path):
    import hashlib

    root = workspace(tmp_path)
    run_id = "operation-mismatch"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_state" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    state_path = root / "data/run_state.json"
    state = json.loads(state_path.read_text())
    outcome = state["outcomes"]["demo@static"]
    evidence_path = root / outcome["target_evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["operation"] = "legacy_import"
    evidence["entry"]["operation"] = "legacy_import"
    payload = (json.dumps(
        evidence, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")
    evidence_path.write_bytes(payload)
    outcome["target_evidence_fingerprint"] = hashlib.sha256(payload).hexdigest()
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates,
            run_static_target=lambda _request: pytest.fail("target must not rerun"),
            run_cpcv_target=base.run_cpcv_target,
        )).execute(resumed)
    assert json.loads(state_path.read_text())["aggregate_error_code"] == "target_evidence_mismatch"


def test_resume_rejects_well_formed_evidence_source_mismatch(tmp_path):
    import hashlib

    root = workspace(tmp_path)
    run_id = "source-mismatch"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_state" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    state_path = root / "data/run_state.json"
    state = json.loads(state_path.read_text())
    outcome = state["outcomes"]["demo@static"]
    evidence_path = root / outcome["target_evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["source_identity"] = {
        "recorder_id": "unexpected-source",
        "experiment_name": "Unexpected_Experiment", "operation": "train",
    }
    payload = (json.dumps(
        evidence, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")
    evidence_path.write_bytes(payload)
    outcome["target_evidence_fingerprint"] = hashlib.sha256(payload).hexdigest()
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates,
            run_static_target=lambda _request: pytest.fail("target must not rerun"),
            run_cpcv_target=base.run_cpcv_target,
        )).execute(resumed)
    assert json.loads(state_path.read_text())["aggregate_error_code"] == "target_evidence_mismatch"


def test_operator_log_append_is_durable_before_closure_marker_and_is_adopted(tmp_path):
    root = workspace(tmp_path)
    run_id = "operator-boundary"
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_operator_log_append" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)
    state = json.loads((root / "data/run_state.json").read_text())
    assert state["phase"] == "closing"
    assert state["closure_steps"].get("operator_log_linked") is None
    log_path = root / "data/operator_log.jsonl"
    assert len(log_path.read_text().splitlines()) == 1

    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )
    TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        run_static_target=lambda _request: pytest.fail("target must not rerun"),
        run_cpcv_target=base.run_cpcv_target,
    )).execute(resumed)
    assert len(log_path.read_text().splitlines()) == 1
    assert not (root / "data/run_state.json").exists()


def test_operator_log_write_failure_retains_resumable_closing_state(tmp_path, monkeypatch):
    from quantpits.utils.operator_log import OperatorLog

    root = workspace(tmp_path)
    run_id = "operator-write-failure"
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    original = OperatorLog._write_entry
    monkeypatch.setattr(
        OperatorLog, "_write_entry",
        lambda self, entry: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(hooks([])).execute(prepared)
    state = json.loads((root / "data/run_state.json").read_text())
    assert state["phase"] == "closing"
    assert state["aggregate_error_code"] == "operator_log_write_failed"
    assert state["closure_steps"].get("operator_log_linked") is None

    monkeypatch.setattr(OperatorLog, "_write_entry", original)
    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )
    TrainingExecutionService(TrainingExecutionHooks(
        **dict(
            activate_workspace=hooks([]).activate_workspace,
            init_qlib=hooks([]).init_qlib,
            calculate_dates=hooks([]).calculate_dates,
            run_static_target=lambda _request: pytest.fail("target must not rerun"),
            run_cpcv_target=hooks([]).run_cpcv_target,
        )
    )).execute(resumed)
    assert not (root / "data/run_state.json").exists()


def test_manifest_retry_preserves_logical_timing_and_fingerprint(tmp_path):
    import hashlib

    root = workspace(tmp_path)
    run_id = "manifest-timing"
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_manifest_closure" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)
    path = root / "output/manifests/static_train" / (run_id + ".json")
    before = path.read_bytes()
    timing = (json.loads(before)["started_at"], json.loads(before)["finished_at"])

    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )
    TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        run_static_target=lambda _request: pytest.fail("target must not rerun"),
        run_cpcv_target=base.run_cpcv_target,
    )).execute(resumed)
    after = path.read_bytes()
    assert after == before
    assert (json.loads(after)["started_at"], json.loads(after)["finished_at"]) == timing
    assert hashlib.sha256(after).hexdigest() == hashlib.sha256(before).hexdigest()


def test_resume_rejects_state_baseline_race_after_planning(tmp_path):
    root = workspace(tmp_path)
    run_id = "state-race"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_state" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )
    state_path = root / "data/run_state.json"
    state = json.loads(state_path.read_text())
    state["aggregate_error_code"] = "concurrent-owner"
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")
    with pytest.raises(TrainingExecutionError, match="changed after command planning"):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates,
            run_static_target=lambda _request: pytest.fail("target must not rerun"),
            run_cpcv_target=base.run_cpcv_target,
        )).execute(resumed)


def test_resume_safely_backfills_transaction_bound_phase26_v3_state(tmp_path):
    root = workspace(tmp_path)
    run_id = "v3-backfill"
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_receipt_write" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    state_path = root / "data/run_state.json"
    legacy = json.loads(state_path.read_text())
    legacy["phase"] = legacy["status"] = "publication_committed"
    legacy["publication_status"] = "committed"
    legacy.pop("receipt_fingerprint", None)
    legacy.pop("committed_outputs", None)
    legacy.pop("closure_steps", None)
    state_path.write_text(json.dumps(legacy, sort_keys=True) + "\n")

    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )
    TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
        calculate_dates=base.calculate_dates,
        run_static_target=lambda _request: pytest.fail("target must not rerun"),
        run_cpcv_target=base.run_cpcv_target,
    )).execute(resumed)
    assert not state_path.exists()


@pytest.mark.parametrize("field,value", [
    ("run_id", "other-run"),
    ("attempt_id", "other-attempt"),
    ("anchor_date", "2026-07-11"),
    ("output_experiment_name", "other-experiment"),
    ("plan_fingerprint", "other-plan"),
    ("resume_fingerprint", "other-resume"),
    ("execution_fingerprint", "other-execution"),
])
def test_resume_rejects_corrupt_target_evidence_identity(tmp_path, field, value):
    import hashlib

    root = workspace(tmp_path)
    run_id = "corrupt-evidence-" + field
    base = hooks([])
    fired = []

    def fault(point, _detail=None):
        if point == "after_target_state" and not fired:
            fired.append(point)
            raise RuntimeError("injected interruption")

    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates, run_static_target=base.run_static_target,
            run_cpcv_target=base.run_cpcv_target, fault_hook=fault,
        )).execute(prepared)

    state_path = root / "data/run_state.json"
    state = json.loads(state_path.read_text())
    outcome = state["outcomes"]["demo@static"]
    evidence_path = root / outcome["target_evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    evidence[field] = value
    payload = (json.dumps(
        evidence, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")
    evidence_path.write_bytes(payload)
    outcome["target_evidence_fingerprint"] = hashlib.sha256(payload).hexdigest()
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

    resumed = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family="static", action="incremental", all_enabled=True,
            resume=True, run_id=run_id,
        ),
    )
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=base.activate_workspace, init_qlib=base.init_qlib,
            calculate_dates=base.calculate_dates,
            run_static_target=lambda _request: pytest.fail("target must not rerun"),
            run_cpcv_target=base.run_cpcv_target,
        )).execute(resumed)
    failed_state = json.loads(state_path.read_text())
    assert failed_state["aggregate_error_code"] == "target_evidence_mismatch"
