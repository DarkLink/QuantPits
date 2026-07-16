import json

import pytest

from quantpits.ensemble.input_integrity import PredictionLoadIntegrityError

from .artifact_graph import assert_declared_writes, observe_artifact_graph
from .drivers import execute_ensemble, recorder_inventory
from .scenario_workspace import ScenarioWorkspace


@pytest.mark.parametrize("failure", ("missing", "foreign"))
def test_partial_input_fails_closed_without_requested_set_shrink(failure, tmp_path, monkeypatch):
    workspace = ScenarioWorkspace.create(tmp_path)
    keys = tuple(workspace.read_json("latest_train_records.json")["models"])
    kwargs = {"missing_key": keys[1]} if failure == "missing" else {"foreign_key": keys[1]}
    inventory = recorder_inventory(workspace, **kwargs)
    baseline = observe_artifact_graph(workspace.root)

    with pytest.raises(PredictionLoadIntegrityError) as raised:
        execute_ensemble(workspace, inventory, run_id="ensemble-%s" % failure, monkeypatch=monkeypatch)

    evidence = raised.value.evidence
    assert tuple(item.resolved_key for item in evidence) == keys
    assert len(evidence) == len(keys)
    assert evidence[0].status == "ready"
    assert evidence[1].status in {"missing_record", "external_artifact"}
    observed = observe_artifact_graph(workspace.root)
    manifest = observed.json("output/manifests/ensemble_fusion/ensemble-%s.json" % failure)
    log = observed.jsonl("data/operator_log.jsonl")[-1]
    assert manifest["status"] == "failed"
    assert tuple(item["resolved_key"] for item in manifest["records"]["input_models"]) == keys
    assert log["exception"]["type"] == "PredictionLoadIntegrityError"
    assert not (workspace.root / "config/ensemble_records.json").exists()
    assert not any(path.startswith("output/buy_suggestion") for path in observed.files)
    assert_declared_writes(
        observed.changed_paths(baseline),
        ("output/manifests/ensemble_fusion/", "data/operator_log.jsonl"),
    )

    from quantpits.scripts import ensemble_fusion

    monkeypatch.setattr(
        "quantpits.ensemble.command.run_ensemble_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(raised.value),
    )
    monkeypatch.setattr(ensemble_fusion, "default_ensemble_command_dependencies", lambda: None)
    with pytest.raises(SystemExit) as cli:
        ensemble_fusion.main(["--from-config"])
    assert cli.value.code == 1
