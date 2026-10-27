"""Explicit export feeds the existing predict-only plan without mutable reselect."""

import builtins

import pytest

from quantpits.scripts.model_source import main
from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.records import ModelRecordEntry, TrainingRecordSnapshot
from .artifact_graph import observe_artifact_graph, assert_write_conservation
from .scenario_workspace import ScenarioWorkspace


@pytest.mark.parametrize("version", [1, 2])
def test_exported_source_is_the_existing_predict_only_plan_input(tmp_path, monkeypatch, version):
    workspace = ScenarioWorkspace.create(tmp_path)
    entry = ModelRecordEntry("demo@static", "demo", "static", "train", "ready", "original-fit", "Fit",
                             requested_anchor="2026-07-16", prediction_start="2026-07-16",
                             prediction_end="2026-07-16", prediction_rows=1)
    document = (TrainingRecordSnapshot((entry,)).to_dict() if version == 2 else
                {"models": {"demo": "original-fit"}, "experiment_name": "Fit"})
    workspace.write_json("data/explicit.json", document)
    workspace.write_json("latest_train_records.json", {"models": {"demo": "later-copy"}, "experiment_name": "Predict"})
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", "/must/not/use/inherited")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////must/not/create.db")
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("qlib", "mlflow"):
            raise AssertionError("source preview/export/plan must not initialize a backend")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    before = observe_artifact_graph(workspace.root)
    arguments = ["--workspace", str(workspace.root), "--record-file", "data/explicit.json", "--models", "demo"]
    assert main(arguments) == 0
    assert observe_artifact_graph(workspace.root).changed_paths(before) == ()
    assert main(arguments + ["--output", "data/chosen.json"]) == 0
    exported = observe_artifact_graph(workspace.root)
    assert_write_conservation(exported, before, ("data/chosen.json",))
    prepared = prepare_training_run(ctx=workspace.ctx, options=TrainingRunOptions(
        family="static", action="predict_only", models=("demo",), source_records="data/chosen.json", json_plan=True))
    assert prepared.source_snapshot.entry_map["demo@static"].recorder_id == "original-fit"
    assert prepared.current_snapshot.entry_map["demo@static"].recorder_id == "later-copy"
    assert observe_artifact_graph(workspace.root).changed_paths(exported) == ()
