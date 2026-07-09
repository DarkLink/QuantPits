import json

from quantpits.runtime import (
    CommandPlan,
    CommandResult,
    CommandStep,
    InputRef,
    OutputRef,
    StateRef,
    manifest_from_result,
    manifest_path,
    write_run_manifest,
)
from quantpits.utils.workspace import WorkspaceContext


def _sample_result(tmp_path):
    plan = CommandPlan(
        command="ensemble_fusion",
        workspace=str(tmp_path),
        run_id="run-1",
        args=("--all",),
        inputs=(InputRef("config/ensemble_config.json", kind="config", fingerprint="abc"),),
        outputs=(OutputRef("output/ensemble/planned.pkl", kind="prediction"),),
        states=(StateRef("config/ensemble_records.json", action="read_write"),),
        steps=(CommandStep("fuse", "fuse combos"),),
        config_fingerprints={"ensemble_config": "abc"},
        warnings=("plan warning",),
    )
    return CommandResult(
        plan=plan,
        status="success",
        started_at="2026-07-09T14:30:12",
        finished_at="2026-07-09T14:31:12",
        outputs=(OutputRef("output/ensemble/actual.pkl", kind="prediction"),),
        records={"record_id": "rid", "raw_config": {"secret": "do-not-render"}},
        warnings=("runtime warning",),
    )


def test_manifest_from_result_copies_plan_and_result_fields(tmp_path):
    manifest = manifest_from_result(_sample_result(tmp_path))

    assert manifest.schema_version == 1
    assert manifest.command == "ensemble_fusion"
    assert manifest.status == "success"
    assert manifest.outputs[0].path == "output/ensemble/actual.pkl"
    assert manifest.records == {"record_id": "rid", "raw_config": {"secret": "do-not-render"}}
    assert manifest.warnings == ("plan warning", "runtime warning")


def test_write_run_manifest_writes_json_under_workspace_output(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    manifest = manifest_from_result(_sample_result(tmp_path))

    path = write_run_manifest(ctx, manifest)

    assert path == manifest_path(ctx, "ensemble_fusion", "run-1")
    assert path == tmp_path / "output" / "manifests" / "ensemble_fusion" / "run-1.json"

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["command"] == "ensemble_fusion"
    assert payload["inputs"][0]["path"] == "config/ensemble_config.json"
    assert payload["outputs"][0]["path"] == "output/ensemble/actual.pkl"
    assert "raw_config" not in json.dumps(payload)


def test_write_run_manifest_replaces_existing_file(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    manifest = manifest_from_result(_sample_result(tmp_path))
    path = write_run_manifest(ctx, manifest)
    path.write_text("{\"stale\": true}\n")

    rewritten = write_run_manifest(ctx, manifest)

    assert rewritten == path
    assert json.loads(path.read_text())["run_id"] == "run-1"
