import json

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.service import TrainingExecutionHooks, TrainingExecutionService
from quantpits.utils.workspace import WorkspaceContext


def workspace(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    (root / "config" / "model_registry.yaml").write_text(
        "models:\n  demo:\n    enabled: true\n    yaml_file: demo.yaml\n"
    )
    (root / "config" / "model_config.json").write_text(json.dumps({"freq": "week"}))
    (root / "config" / "demo.yaml").write_text("model: {}\n")
    return root


def test_service_orders_activation_init_and_execution_and_writes_audit(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(family="static", action="full", run_id="demo-run"),
    )
    calls = []
    service = TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=lambda value: calls.append(("activate", value)),
        init_qlib=lambda: calls.append(("init", None)),
        execute_legacy=lambda value: calls.append(("execute", value.plan.run_id)),
    ))
    summary = service.execute(prepared)
    assert [item[0] for item in calls] == ["activate", "init", "execute"]
    assert summary.manifest_path == "output/manifests/static_train/demo-run.json"
    assert summary.execution_fingerprint
    assert summary.outcomes == ({"key": "demo@static", "operation": "full", "outcome": "success"},)
    assert (root / summary.manifest_path).is_file()
    assert (root / "data" / "operator_log.jsonl").is_file()


def test_no_manifest_still_writes_operator_log(tmp_path):
    root = workspace(tmp_path)
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(family="cpcv", action="full", no_manifest=True),
    )
    summary = TrainingExecutionService(TrainingExecutionHooks(
        activate_workspace=lambda value: None, init_qlib=lambda: None,
        execute_legacy=lambda value: None,
    )).execute(prepared)
    assert summary.manifest_path is None
    assert (root / "data" / "operator_log.jsonl").is_file()
