import json
from dataclasses import replace

import pytest

from quantpits.ensemble.command import run_ensemble_command

from .artifact_graph import assert_write_conservation, observe_artifact_graph
from .drivers import ensemble_command_dependencies, execute_order, recorder_inventory
from .expectations import SemanticScenarioExpectation, TerminalMemberExpectation
from .scenario_workspace import ScenarioWorkspace


@pytest.mark.parametrize("family", ("static", "rolling"))
def test_training_record_to_ensemble_to_order_lineage(family, tmp_path, monkeypatch):
    workspace = ScenarioWorkspace.create(tmp_path, family=family)
    inventory = recorder_inventory(workspace)
    baseline = observe_artifact_graph(workspace.root)
    keys = tuple(workspace.read_json("latest_train_records.json")["models"])
    expected = SemanticScenarioExpectation(
        requested_identities=keys,
        terminal_members=tuple(TerminalMemberExpectation(key, "success", True) for key in keys),
        authoritative_inputs=tuple(workspace.read_json("latest_train_records.json")["models"].values()),
        allowed_write_paths=(
            "config/ensemble_records.json", "output/", "data/operator_log.jsonl",
            "mlruns/ensemble-sentinel/",
        ),
    )

    dependencies, _service = ensemble_command_dependencies(workspace, inventory, monkeypatch=monkeypatch)
    command_outcome = {}

    def capture_outcome(request, active_dependencies):
        outcome = run_ensemble_command(request, active_dependencies)
        command_outcome["value"] = outcome
        return outcome

    from quantpits.scripts import ensemble_fusion

    monkeypatch.setattr("quantpits.ensemble.command.run_ensemble_command", capture_outcome)
    monkeypatch.setattr(ensemble_fusion, "default_ensemble_command_dependencies", lambda: dependencies)
    cli_result = ensemble_fusion.main([
        "--from-config", "--run-id", "ensemble-semantic", "--no-backtest", "--no-charts",
    ])
    outcome = command_outcome["value"]
    ensemble_prepared = outcome.prepared
    ensemble = outcome.summary
    order_prepared, order, _generator = execute_order(workspace)
    observed = observe_artifact_graph(workspace.root)

    assert tuple(item["resolved_key"] for item in ensemble.input_evidence) == expected.requested_identities
    assert tuple(item["status"] for item in ensemble.input_evidence) == ("ready", "ready")
    records = observed.json("config/ensemble_records.json")
    meta = records["combo_meta"]["sentinel"]
    assert tuple(meta["models"]) == expected.requested_identities
    assert tuple(meta["source_recorders"][key] for key in expected.requested_identities) == expected.authoritative_inputs
    assert order_prepared.source.record_id == ensemble.combo_results[0]["recorder_id"] == "ensemble-sentinel"
    assert order.source_description == "sanitized ensemble source"

    ensemble_manifest = observed.json(ensemble.manifest_path)
    order_manifest = observed.json(order.manifest_path)
    assert ensemble_manifest["status"] == order_manifest["status"] == expected.aggregate_status
    assert cli_result is None and outcome.mode == "execute" and outcome.rendered_output is None
    assert ensemble.run_id == ensemble_manifest["run_id"] == ensemble_prepared.plan.run_id
    assert tuple(ensemble.input_evidence) == tuple(ensemble_manifest["records"]["input_models"])
    assert ensemble_manifest["records"]["combos"][0]["recorder_id"] == ensemble.combo_results[0]["recorder_id"]
    assert order.run_id == order_manifest["run_id"] == order_prepared.plan.run_id
    assert order.manifest_path == "output/manifests/order_gen/order-semantic.json"
    assert order.source_label == order_prepared.source.source_label
    assert tuple(order_manifest["args"]) == order_prepared.plan.args
    assert order_manifest["config_fingerprints"] == order_prepared.plan.config_fingerprints
    assert order_manifest["records"]["source"] == {
        "mode": order_prepared.source.mode,
        "requested_name": order_prepared.source.requested_name,
        "resolved_name": order_prepared.source.resolved_name,
        "record_id": order_prepared.source.record_id,
        "experiment_name": order_prepared.source.experiment_name,
    }
    logs = observed.jsonl("data/operator_log.jsonl")
    assert [(item["script"], item["exception"]) for item in logs] == [
        ("ensemble_fusion", None), ("order_gen", None),
    ]
    ensemble_log = logs[0]
    assert ensemble_log["run_id"] == ensemble.run_id
    assert ensemble_log["plan_fingerprint"] == ensemble_prepared.plan_fingerprint
    assert ensemble_log["manifest_path"] == ensemble.manifest_path
    assert ensemble_log["result_summary"] == {
        "anchor_date": ensemble.anchor_date,
        "n_combos": len(ensemble.combo_results),
        "experiment_name": ensemble.experiment_name,
        "manifest_path": ensemble.manifest_path,
    }
    order_log = logs[1]
    assert tuple(order_log["args"]) == order_prepared.cli_args
    assert order_log["run_id"] == order.run_id
    assert order_log["plan_fingerprint"] == order_prepared.plan_fingerprint
    assert order_log["manifest_path"] == order.manifest_path
    assert order_log["result_summary"] == order_manifest["records"]
    assert order_log["result_summary"]["source"] == order_manifest["records"]["source"]
    assert_write_conservation(observed, baseline, expected.allowed_write_paths)
    assert "Demo_Workspace" in json.dumps(ensemble_manifest)


def test_predict_resolution_retry_preserves_account_and_downstream_sources(tmp_path, monkeypatch):
    from quantpits.training.command import TrainingRunOptions, prepare_training_run
    from quantpits.training.errors import TrainingExecutionError
    from quantpits.training.records import ModelRecordEntry
    from quantpits.training.runners import TrainingTargetResult
    from quantpits.training.service import TrainingExecutionHooks, TrainingExecutionService
    from quantpits.utils.config_loader import load_training_config
    from .drivers import FakeRecorder, prediction_frame

    workspace = ScenarioWorkspace.create(tmp_path)
    workspace.write_json("config/model_config.json", {
        "train_date_mode": "fixed", "current_date": "2026-07-16", "freq": "week",
    })
    workspace.write_json("config/prod_config.json", {
        "current_date": "2026-07-09", "last_processed_date": "2026-07-09",
        "current_cash": 1000.0, "current_holding": [],
    })
    (workspace.root / "config/model_registry.yaml").write_text(
        "models:\n  m1: {enabled: true, yaml_file: config/demo.yaml}\n"
        "  m2: {enabled: true, yaml_file: config/demo.yaml}\n", encoding="utf-8",
    )
    baseline = observe_artifact_graph(workspace.root)
    account_before = workspace.ctx.config_path("prod_config.json").read_bytes()
    inventory, calls = {}, []

    def dates():
        anchor = load_training_config(workspace.root)["current_date"]
        return {"anchor_date": anchor, "test_end_time": anchor, "freq": "week"}

    def predict(request):
        source = request.target.source_entry
        key = request.target.key
        calls.append((key, request.target.operation, source.recorder_id))
        name = request.target.target.model_name
        recorder_id = "prediction-" + name
        relative = "mlruns/%s/artifacts" % recorder_id
        artifact = workspace.root / relative
        artifact.mkdir(parents=True)
        frame = prediction_frame()
        (artifact / "pred.pkl").write_bytes(frame.to_csv().encode("utf-8"))
        inventory[recorder_id] = FakeRecorder(recorder_id, artifact.as_uri(), frame)
        entry = ModelRecordEntry(
            key, name, "static", "predict_only", "ready", recorder_id,
            request.resolved_run.output_experiment_name, artifact_path=relative,
            requested_anchor="2026-07-16", prediction_start="2026-07-16",
            prediction_end="2026-07-16", prediction_rows=2,
            source_recorder_id=source.recorder_id, source_experiment_name=source.experiment_name,
            source_operation=source.operation,
        )
        return TrainingTargetResult(key, "predict_only", "success", entry)

    hooks = TrainingExecutionHooks(activate_workspace=lambda _root: None, init_qlib=lambda: None,
                                   calculate_dates=dates, run_static_target=predict)

    def prepared():
        return prepare_training_run(ctx=workspace.ctx, options=TrainingRunOptions(
            family="static", action="predict_only", all_enabled=True, run_id="predict-retry",
        ))

    def unavailable_dates():
        raise ValueError("synthetic calendar unavailable")

    with pytest.raises(TrainingExecutionError):
        TrainingExecutionService(replace(hooks, calculate_dates=unavailable_dates)).execute(prepared())
    path = workspace.root / "output/manifests/static_train/predict-retry.json"
    failure = path.read_bytes()
    assert calls == [] and not workspace.ctx.data_path("run_state.json").exists()
    summary = TrainingExecutionService(hooks).execute(prepared())
    assert calls == [("m1@static", "predict_only", "source-m1"), ("m2@static", "predict_only", "source-m2")]
    assert json.loads(path.read_text())["status"] == "success"
    assert [p.read_bytes() for p in workspace.ctx.data_path("training_attempts/predict-retry").glob("*.json")] == [failure]
    dependencies, _service = ensemble_command_dependencies(workspace, inventory, monkeypatch=monkeypatch)
    from quantpits.ensemble.command import EnsembleCommandRequest, build_ensemble_arg_parser
    args = build_ensemble_arg_parser().parse_args([
        "--from-config", "--run-id", "ensemble-retry", "--no-backtest", "--no-charts",
    ])
    ensemble = run_ensemble_command(EnsembleCommandRequest(args, ()), dependencies).summary
    order_prepared, _order, _generator = execute_order(workspace)
    assert {item["key"] for item in summary.outcomes} == {"m1@static", "m2@static"}
    meta = workspace.read_json("config/ensemble_records.json")["combo_meta"]["sentinel"]
    assert meta["source_recorders"] == {"m1@static": "prediction-m1", "m2@static": "prediction-m2"}
    assert order_prepared.source.record_id == ensemble.combo_results[0]["recorder_id"]
    assert workspace.ctx.config_path("prod_config.json").read_bytes() == account_before
    assert not workspace.ctx.path("mlflow.db").exists()
    assert_write_conservation(observe_artifact_graph(workspace.root), baseline, (
        "latest_train_records.json", "latest_train_records.json.lock", "config/ensemble_records.json",
        "data/", "output/", "mlruns/",
    ))
