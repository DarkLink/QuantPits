"""Regressions for training dates, output bootstrap, and resolution retries."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.errors import TrainingExecutionError, TrainingManifestConflictError
from quantpits.training.service import TrainingExecutionService, default_execution_hooks
from quantpits.utils.workspace import WorkspaceContext
from tests.quantpits.training.test_training_service import hooks, result_for, workspace


def prepare(root, run_id="boundary", **kwargs):
    return prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(
            family=kwargs.pop("family", "static"), action=kwargs.pop("action", "incremental"),
            all_enabled=True, run_id=run_id, **kwargs
        ),
    )


@pytest.mark.parametrize("family", ["static", "cpcv"])
def test_default_date_resolution_ignores_account_cursor(tmp_path, monkeypatch, family):
    root = workspace(tmp_path)
    model = {"train_date_mode": "fixed", "current_date": "2026-07-10", "freq": "day"}
    if family == "cpcv":
        model["purged_cv"] = {"n_groups": 4, "n_test_groups": 1, "n_val_groups": 1,
                              "purge_steps": 0, "embargo_steps": 0}
    (root / "config/model_config.json").write_text(json.dumps(model))
    account = root / "config/prod_config.json"
    account.write_text(json.dumps({"current_date": "2026-07-03", "current_full_cash": 12345}))
    import qlib.data
    from quantpits.utils import train_utils
    monkeypatch.setattr(train_utils, "ROOT_DIR", str(tmp_path / "unrelated-legacy-workspace"))
    monkeypatch.setattr(qlib.data, "D", SimpleNamespace(
        calendar=lambda **_kw: pd.date_range("2010-01-01", "2026-07-10", freq="B"),
    ))
    # Use the production date hook with deliberately unrelated legacy globals.
    service = TrainingExecutionService(default_execution_hooks(
        activate_workspace=lambda _root: None, init_qlib=lambda: None,
    ))
    before = account.read_bytes()
    first = service._resolve(prepare(root, family=family))
    assert first.anchor_date == "2026-07-10"
    assert first.params["account"] == 12345
    assert account.read_bytes() == before
    account.write_text(json.dumps({"current_date": "2026-06-26", "current_full_cash": 12345}))
    second = service._resolve(prepare(root, family=family))
    assert second.params == first.params
    monkeypatch.setattr(train_utils, "ROOT_DIR", str(root))
    assert train_utils.calculate_dates() == first.params


@pytest.mark.parametrize("date", [None, "", "2026-02-30", "2026-7-1"])
def test_fixed_model_date_cannot_fall_back_to_account(tmp_path, date):
    from quantpits.training.errors import TrainingDatePolicyError
    from quantpits.utils.train_utils import calculate_dates
    root = workspace(tmp_path)
    (root / "config/model_config.json").write_text(json.dumps({"train_date_mode": "fixed", "current_date": date}))
    (root / "config/prod_config.json").write_text('{"current_date":"2026-07-10"}')
    with pytest.raises(TrainingDatePolicyError, match="model_config.current_date"):
        prepare(root)
    with pytest.raises(ValueError, match="model_config.current_date"):
        calculate_dates(ctx=WorkspaceContext.from_root(root))


def test_predict_only_uses_model_date_and_selected_old_source(tmp_path, monkeypatch):
    import qlib.data
    root = workspace(tmp_path)
    (root / "config/model_config.json").write_text(json.dumps({
        "train_date_mode": "fixed", "current_date": "2026-07-10", "freq": "day",
    }))
    account = root / "config/prod_config.json"
    account.write_text('{"current_date":"2026-07-03"}')
    before = account.read_bytes()
    (root / "source.json").write_text(json.dumps({"experiment_name": "Old_Source", "models": {"demo@static": "old-rid"}}))
    monkeypatch.setattr(qlib.data, "D", SimpleNamespace(calendar=lambda **_kw: pytest.fail("fixed static uses no calendar")))
    seen = []

    def runner(request):
        from quantpits.training.records import ModelRecordEntry
        from quantpits.training.runners import TrainingTargetResult
        source = request.target.source_entry
        seen.append((request.target.operation, source.recorder_id, source.experiment_name,
                     request.resolved_run.anchor_date))
        entry = ModelRecordEntry(
            request.target.key, "demo", "static", "predict_only", "ready", "new-prediction",
            request.resolved_run.output_experiment_name, requested_anchor="2026-07-10",
            prediction_start="2026-07-10", prediction_end="2026-07-10", prediction_rows=1,
            source_recorder_id=source.recorder_id, source_experiment_name=source.experiment_name,
            source_operation=source.operation,
        )
        return TrainingTargetResult(request.target.key, "predict_only", "success", entry)

    base = default_execution_hooks(activate_workspace=lambda _root: None, init_qlib=lambda: None)
    TrainingExecutionService(replace(
        base, prepare_cache=lambda _run: None, prepare_output_experiment=lambda _run: None,
        verify_source_target=lambda _run, _target: None, run_static_target=runner,
    )).execute(prepare(root, action="predict_only", source_records="source.json"))
    assert seen == [("predict_only", "old-rid", "Old_Source", "2026-07-10")]
    assert account.read_bytes() == before


def test_last_trade_date_still_uses_provider_calendar(tmp_path, monkeypatch):
    import qlib.data
    root = workspace(tmp_path)
    (root / "config/model_config.json").write_text('{"current_date":"invalid-but-unused"}')
    (root / "config/prod_config.json").write_text('{"current_date":"2026-07-03"}')
    monkeypatch.setattr(qlib.data, "D", SimpleNamespace(calendar=lambda **_kw: [pd.Timestamp("2026-07-17")]))
    base = default_execution_hooks(activate_workspace=lambda _root: None, init_qlib=lambda: None)
    assert TrainingExecutionService(base)._resolve(prepare(root)).anchor_date == "2026-07-17"


@pytest.mark.parametrize("backend", ["sqlite", "file"])
@pytest.mark.parametrize("existing", [False, True])
def test_output_experiment_is_prepared_before_runner(tmp_path, monkeypatch, backend, existing):
    from mlflow.tracking import MlflowClient
    import qlib.workflow

    root = workspace(tmp_path)
    uri = ("sqlite:///" + str(root / "mlflow.db") if backend == "sqlite"
           else (root / "mlruns").as_uri())
    ctx = WorkspaceContext.from_root(root, mlflow_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_name = "Boundary_Output"
    old_location = root / "legacy_artifacts"
    old_id = client.create_experiment(experiment_name, artifact_location=old_location.as_uri()) if existing else None
    monkeypatch.setattr(qlib.workflow, "R", SimpleNamespace(get_uri=lambda: uri))
    monkeypatch.chdir(tmp_path)
    base = default_execution_hooks(
        activate_workspace=lambda _root: None, init_qlib=lambda: None,
        calculate_dates=hooks([]).calculate_dates,
    )

    def runner(request):
        experiment = client.get_experiment_by_name(experiment_name)
        assert experiment is not None
        expected = old_location if existing else root / "output/mlflow_artifacts" / experiment_name
        assert experiment.artifact_location == expected.as_uri()
        if existing:
            assert experiment.experiment_id == old_id
        run = client.create_run(experiment.experiment_id)
        artifact = tmp_path / "tiny.txt"
        artifact.write_text("boundary artifact")
        client.log_artifact(run.info.run_id, str(artifact))
        client.set_terminated(run.info.run_id)
        assert (expected / run.info.run_id / "artifacts/tiny.txt").read_text() == "boundary artifact"
        return result_for(request)

    prepared = prepare_training_run(ctx=ctx, options=TrainingRunOptions(
        family="static", action="incremental", all_enabled=True, run_id="output",
        experiment_name=experiment_name,
    ))
    TrainingExecutionService(replace(base, prepare_cache=lambda _run: None,
                                    promote_static=lambda *_args: None, run_static_target=runner)).execute(prepared)
    assert not (tmp_path / "mlruns").exists()


def test_external_output_fails_before_cache_or_runner(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient
    import qlib.workflow

    root = workspace(tmp_path)
    prepared = prepare(root, experiment_name="External_Output")
    uri = prepared.ctx.mlflow_uri
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("External_Output", artifact_location=(tmp_path / "external").as_uri())
    monkeypatch.setattr(qlib.workflow, "R", SimpleNamespace(get_uri=lambda: uri))
    base = default_execution_hooks(activate_workspace=lambda _root: None, init_qlib=lambda: None,
                                   calculate_dates=hooks([]).calculate_dates)
    with pytest.raises(TrainingExecutionError, match="outside workspace"):
        TrainingExecutionService(replace(
            base, prepare_cache=lambda _run: pytest.fail("cache must not start"),
            run_static_target=lambda _request: pytest.fail("runner must not start"),
        )).execute(prepared)
    assert not (root / "data/training_transactions").exists()
    assert not (tmp_path / "external").exists()


def test_two_workspaces_prepare_independent_output_roots(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient
    import qlib.workflow
    active_uri = [None]
    monkeypatch.setattr(qlib.workflow, "R", SimpleNamespace(get_uri=lambda: active_uri[0]))
    base = default_execution_hooks(activate_workspace=lambda _root: None, init_qlib=lambda: None,
                                   calculate_dates=hooks([]).calculate_dates)
    observed = []

    def runner(request):
        root = request.resolved_run.prepared.ctx.root
        experiment = MlflowClient(tracking_uri=active_uri[0]).get_experiment_by_name("Prod_Train_WEEK")
        assert experiment.artifact_location == (root / "output/mlflow_artifacts/Prod_Train_WEEK").as_uri()
        observed.append(experiment.artifact_location)
        return result_for(request)

    service = TrainingExecutionService(replace(base, prepare_cache=lambda _run: None,
                                               promote_static=lambda *_args: None, run_static_target=runner))
    for name in ("one", "two"):
        root = workspace(tmp_path / name)
        prepared = prepare(root)
        active_uri[0] = prepared.ctx.mlflow_uri
        service.execute(prepared)
    assert len(set(observed)) == 2


@pytest.mark.parametrize("fault", ["tracking", "symlink"])
def test_output_preparation_rejects_foreign_paths_before_model_work(tmp_path, monkeypatch, fault):
    from mlflow.tracking import MlflowClient
    import qlib.workflow
    root = workspace(tmp_path)
    prepared = prepare(root)
    external = tmp_path / "external"
    external.mkdir()
    uri = prepared.ctx.mlflow_uri if fault == "symlink" else "sqlite:///" + str(external / "wrong.db")
    if fault == "symlink":
        (root / "output/mlflow_artifacts").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(qlib.workflow, "R", SimpleNamespace(get_uri=lambda: uri))
    base = default_execution_hooks(activate_workspace=lambda _root: None, init_qlib=lambda: None,
                                   calculate_dates=hooks([]).calculate_dates)
    with pytest.raises(TrainingExecutionError, match="outside workspace|tracking URI differs"):
        TrainingExecutionService(replace(
            base, prepare_cache=lambda _run: pytest.fail("must not prepare cache"),
            run_static_target=lambda _request: pytest.fail("must not run"),
        )).execute(prepared)
    assert list(external.iterdir()) == []
    if fault == "symlink":
        assert MlflowClient(tracking_uri=uri).get_experiment_by_name("Prod_Train_WEEK") is None
    else:
        assert not (root / "mlflow.db").exists()


def resolution_failure():
    raise ValueError("calendar temporarily unavailable")


def test_resolution_failure_can_retry_same_run_without_resume(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare(root)
    base = hooks([])
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(base, calculate_dates=resolution_failure)).execute(prepared)
    path = root / "output/manifests/static_train/boundary.json"
    original = path.read_bytes()
    assert not (root / "data/run_state.json").exists()
    TrainingExecutionService(base).execute(prepare(root))
    assert json.loads(path.read_text())["status"] == "success"
    attempts = list((root / "data/training_attempts/boundary").glob("pre-execution-*.json"))
    assert [item.read_bytes() for item in attempts] == [original]


def test_resolution_retry_then_partial_execution_resumes_without_repeating_success(tmp_path):
    root = workspace(tmp_path, models=("first", "second"))
    base = hooks([])
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(base, calculate_dates=resolution_failure)).execute(prepare(root))
    called = []

    def partial(request):
        called.append(request.target.key)
        return result_for(request, outcome="failed" if request.target.key.startswith("second@") else "success")

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(base, run_static_target=partial)).execute(prepare(root))
    assert called == ["first@static", "second@static"]
    called.clear()
    summary = TrainingExecutionService(replace(
        base, run_static_target=lambda request: (called.append(request.target.key), result_for(request))[1],
    )).execute(prepare(root, resume=True))
    assert called == ["second@static"]
    assert len(summary.outcomes) == 2
    assert not (root / "data/run_state.json").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_residual_resolution_manifest_does_not_block_valid_state_resume(tmp_path, legacy):
    root = workspace(tmp_path, models=("first", "second"))
    base = hooks([])
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(base, calculate_dates=resolution_failure)).execute(prepare(root))
    path = root / "output/manifests/static_train/boundary.json"
    early = json.loads(path.read_text())
    if legacy:
        early["records"] = {key: early["records"][key]
                            for key in ("plan_fingerprint", "outcomes", "publication")}

    def stop_before_publication(point, detail):
        if point == "after_target_state" and detail == "second@static":
            raise RuntimeError("interrupted before publication")

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(
            base, run_static_target=lambda request: result_for(
                request, outcome="failed" if request.target.key == "second@static" else "success"),
            fault_hook=stop_before_publication,
        )).execute(prepare(root))
    # Reproduce the residue produced by the old failed-manifest closure path.
    path.write_text(json.dumps(early))
    calls = []
    TrainingExecutionService(replace(
        base, run_static_target=lambda request: (calls.append(request.target.key), result_for(request))[1],
    )).execute(prepare(root, resume=True))
    assert calls == ["second@static"]
    assert json.loads(path.read_text())["status"] == "success"
    assert not (root / "data/run_state.json").exists()


@pytest.mark.parametrize("mutation", ["request", "source", "success", "unknown", "publication"])
def test_pre_execution_retry_rejects_conflicting_evidence(tmp_path, mutation):
    root = workspace(tmp_path)
    source = root / "source.json"
    source.write_text(json.dumps({"experiment_name": "Old", "models": {"demo@static": "old-rid"}}))
    kwargs = {"action": "predict_only", "source_records": "source.json"} if mutation == "source" else {}
    base = hooks([])
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(base, calculate_dates=resolution_failure)).execute(prepare(root, **kwargs))
    path = root / "output/manifests/static_train/boundary.json"
    if mutation == "request":
        kwargs["experiment_name"] = "Different"
    elif mutation == "source":
        source.write_text(json.dumps({"experiment_name": "Other", "models": {"demo@static": "different-rid"}}))
    elif mutation in ("success", "unknown"):
        value = json.loads(path.read_text())
        if mutation == "success":
            value["status"] = "success"
        else:
            value["records"].pop("pre_execution_failure", None)
        path.write_text(json.dumps(value))
    else:
        (root / "data/training_transactions").mkdir()
        (root / "data/training_transactions/uncertain").mkdir()
        (root / "data/training_transactions/uncertain/intent.json").write_text("{}")
    original = path.read_bytes()
    with pytest.raises(TrainingManifestConflictError):
        TrainingExecutionService(replace(base, run_static_target=lambda _request: pytest.fail("must reject before execution"))).execute(prepare(root, **kwargs))
    assert path.read_bytes() == original


@pytest.mark.parametrize("fault", ["archive", "unlink", "interrupt"])
def test_resolution_retry_preserves_audit_and_checks_postconditions(tmp_path, monkeypatch, fault):
    from pathlib import Path
    root = workspace(tmp_path)
    base = hooks([])
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(base, calculate_dates=resolution_failure)).execute(prepare(root))
    path = root / "output/manifests/static_train/boundary.json"
    original = path.read_bytes()
    if fault == "archive":
        monkeypatch.setattr("quantpits.training.attempts.atomic_write_bytes", lambda *_args: None)
    elif fault == "unlink":
        unlink = Path.unlink
        monkeypatch.setattr(Path, "unlink", lambda item, *args, **kwargs: None if item == path else unlink(item, *args, **kwargs))
    else:
        def interrupt():
            raise KeyboardInterrupt()
        base = replace(base, calculate_dates=interrupt)
    error = KeyboardInterrupt if fault == "interrupt" else TrainingExecutionError
    with pytest.raises(error):
        TrainingExecutionService(replace(base, run_static_target=lambda _request: pytest.fail("must not run"))).execute(prepare(root))
    assert not (root / "data/run_state.json").exists()
    if fault != "interrupt":
        assert path.read_bytes() == original
    if fault != "archive":
        assert [item.read_bytes() for item in (root / "data/training_attempts/boundary").glob("*.json")] == [original]
    if fault == "interrupt":
        with pytest.raises(TrainingManifestConflictError):
            TrainingExecutionService(hooks([])).execute(prepare(root, experiment_name="different-request"))
        # A rejected request must not occupy the freed path with its own failure.
        assert not path.exists()
        TrainingExecutionService(hooks([])).execute(prepare(root))
        assert json.loads(path.read_text())["status"] == "success"


def test_resolution_audit_refuses_symlink_parent_outside_workspace(tmp_path):
    root = workspace(tmp_path)
    base = hooks([])
    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(base, calculate_dates=resolution_failure)).execute(prepare(root))
    path = root / "output/manifests/static_train/boundary.json"
    before = path.read_bytes()
    outside = tmp_path / "outside-audit"
    outside.mkdir()
    (root / "data/training_attempts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(TrainingManifestConflictError, match="leaves workspace"):
        TrainingExecutionService(base).execute(prepare(root))
    assert path.read_bytes() == before
    assert list(outside.iterdir()) == []
