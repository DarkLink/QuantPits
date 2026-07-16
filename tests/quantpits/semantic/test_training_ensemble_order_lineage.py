import json

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
    assert order_manifest["records"]["source"]["record_id"] == order_prepared.source.record_id
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
    assert_write_conservation(observed, baseline, expected.allowed_write_paths)
    assert "Demo_Workspace" in json.dumps(ensemble_manifest)
