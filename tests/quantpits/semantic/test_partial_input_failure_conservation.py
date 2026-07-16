import json

import pytest

from quantpits.ensemble.execution import EnsembleExecutionError
from quantpits.ensemble.input_integrity import PredictionLoadIntegrityError

from .artifact_graph import assert_write_conservation, observe_artifact_graph
from .drivers import ensemble_command_dependencies, recorder_inventory
from .scenario_workspace import ScenarioWorkspace


@pytest.mark.parametrize("failure", ("missing", "foreign"))
def test_partial_input_fails_closed_without_requested_set_shrink(failure, tmp_path, monkeypatch):
    workspace = ScenarioWorkspace.create(tmp_path)
    keys = tuple(workspace.read_json("latest_train_records.json")["models"])
    kwargs = {"missing_key": keys[1]} if failure == "missing" else {"foreign_key": keys[1]}
    inventory = recorder_inventory(workspace, **kwargs)
    baseline = observe_artifact_graph(workspace.root)

    run_id = "ensemble-%s" % failure
    dependencies, service = ensemble_command_dependencies(workspace, inventory, monkeypatch=monkeypatch)
    service_execute = service.execute
    command_evidence = {}

    def capture_failure(prepared):
        command_evidence["prepared"] = prepared
        try:
            return service_execute(prepared)
        except PredictionLoadIntegrityError as exc:
            command_evidence["exception"] = exc
            raise

    service.execute = capture_failure
    from quantpits.scripts import ensemble_fusion

    monkeypatch.setattr(ensemble_fusion, "default_ensemble_command_dependencies", lambda: dependencies)
    with pytest.raises(SystemExit) as cli:
        ensemble_fusion.main([
            "--from-config", "--run-id", run_id, "--no-backtest", "--no-charts",
        ])
    assert cli.value.code == 1

    raised = command_evidence["exception"]
    prepared = command_evidence["prepared"]
    evidence = raised.evidence
    assert tuple(item.resolved_key for item in evidence) == keys
    assert len(evidence) == len(keys)
    assert evidence[0].status == "ready"
    assert evidence[1].status in {"missing_record", "external_artifact"}
    observed = observe_artifact_graph(workspace.root)
    manifest_path = "output/manifests/ensemble_fusion/%s.json" % run_id
    manifest = observed.json(manifest_path)
    log = observed.jsonl("data/operator_log.jsonl")[-1]
    assert manifest["status"] == "failed"
    assert manifest["run_id"] == log["run_id"] == prepared.plan.run_id == run_id
    assert log["plan_fingerprint"] == prepared.plan_fingerprint
    assert log["manifest_path"] == manifest_path
    assert tuple(item["resolved_key"] for item in manifest["records"]["input_models"]) == keys
    assert tuple(manifest["records"]["input_models"]) == tuple(item.to_public_dict() for item in evidence)
    assert manifest["error"] == {"type": type(raised).__name__, "message": str(raised)}
    assert log["exception"]["type"] == "PredictionLoadIntegrityError"
    assert log["exception"]["value"] == str(raised)
    assert log["result_summary"] == {}
    assert not (workspace.root / "config/ensemble_records.json").exists()
    assert not any(path.startswith("output/buy_suggestion") for path in observed.files)
    assert_write_conservation(
        observed, baseline,
        ("output/manifests", "output/manifests/ensemble_fusion/", "data/operator_log.jsonl"),
    )


def test_publication_failure_preserves_effects_and_maps_same_outcome_to_cli(tmp_path, monkeypatch):
    workspace = ScenarioWorkspace.create(tmp_path)
    inventory = recorder_inventory(workspace)
    baseline = observe_artifact_graph(workspace.root)
    run_id = "ensemble-publication-failure"
    dependencies, service = ensemble_command_dependencies(
        workspace, inventory, monkeypatch=monkeypatch, publication_error=True,
    )
    service_execute = service.execute
    command_evidence = {}

    def capture_failure(prepared):
        command_evidence["prepared"] = prepared
        try:
            return service_execute(prepared)
        except EnsembleExecutionError as exc:
            command_evidence["exception"] = exc
            raise

    service.execute = capture_failure
    from quantpits.scripts import ensemble_fusion

    monkeypatch.setattr(ensemble_fusion, "default_ensemble_command_dependencies", lambda: dependencies)
    with pytest.raises(SystemExit) as cli:
        ensemble_fusion.main([
            "--from-config", "--run-id", run_id, "--no-backtest", "--no-charts",
        ])

    assert cli.value.code == 1
    raised = command_evidence["exception"]
    prepared = command_evidence["prepared"]
    observed = observe_artifact_graph(workspace.root)
    manifest_path = "output/manifests/ensemble_fusion/%s.json" % run_id
    manifest = observed.json(manifest_path)
    log = observed.jsonl("data/operator_log.jsonl")[-1]
    recorder_bytes = workspace.root / "mlruns/ensemble-sentinel/artifacts/pred.pkl"

    assert recorder_bytes.is_file() and recorder_bytes.read_bytes()
    assert not (workspace.root / "config/ensemble_records.json").exists()
    assert manifest["status"] == "failed" and manifest["outputs"] == []
    assert manifest["records"]["n_combos"] == 0
    assert manifest["run_id"] == log["run_id"] == prepared.plan.run_id == run_id
    assert manifest["error"] == {"type": type(raised).__name__, "message": str(raised)}
    assert log["exception"] == {"type": type(raised).__name__, "value": str(raised)}
    assert log["plan_fingerprint"] == prepared.plan_fingerprint
    assert log["manifest_path"] == manifest_path
    assert log["result_summary"] == {}
    assert_write_conservation(
        observed, baseline,
        (
            "mlruns/ensemble-sentinel/", "output/manifests",
            "output/manifests/ensemble_fusion/", "data/operator_log.jsonl",
        ),
    )
