import json
from types import SimpleNamespace

import pytest

from .artifact_graph import assert_declared_writes, observe_artifact_graph
from .drivers import execute_ensemble, execute_order, recorder_inventory
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
        allowed_write_prefixes=(
            "config/ensemble_records.json", "output/", "data/operator_log.jsonl",
            "mlruns/ensemble-sentinel/",
        ),
    )

    ensemble_prepared, ensemble = execute_ensemble(workspace, inventory, monkeypatch=monkeypatch)
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
    assert ensemble_manifest["run_id"] == ensemble_prepared.plan.run_id
    assert order_manifest["records"]["source"]["record_id"] == order_prepared.source.record_id
    logs = observed.jsonl("data/operator_log.jsonl")
    assert [(item["script"], item["exception"]) for item in logs] == [
        ("ensemble_fusion", None), ("order_gen", None),
    ]
    assert not observed.physical_escapes
    assert_declared_writes(observed.changed_paths(baseline), expected.allowed_write_prefixes)
    assert "Demo_Workspace" in json.dumps(ensemble_manifest)

    from quantpits.scripts import ensemble_fusion

    monkeypatch.setattr("quantpits.ensemble.command.run_ensemble_command", lambda *args, **kwargs: SimpleNamespace(rendered_output=None))
    monkeypatch.setattr(ensemble_fusion, "default_ensemble_command_dependencies", lambda: None)
    assert ensemble_fusion.main(["--from-config"]) is None
