import json
import hashlib
from dataclasses import replace

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
    write_or_adopt_run_manifest,
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


def test_write_or_adopt_manifest_rejects_conflicting_durable_evidence(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    manifest = manifest_from_result(_sample_result(tmp_path))
    path, fingerprint = write_or_adopt_run_manifest(ctx, manifest)
    assert write_or_adopt_run_manifest(ctx, manifest) == (path, fingerprint)
    path.write_text('{"status":"success","different":true}\n')
    import pytest
    with pytest.raises(RuntimeError, match="conflicts"):
        write_or_adopt_run_manifest(ctx, manifest)


def test_failed_manifest_supersession_requires_same_logical_run(tmp_path):
    import pytest
    from dataclasses import replace

    ctx = WorkspaceContext.from_root(tmp_path)
    manifest = manifest_from_result(_sample_result(tmp_path))
    failed = replace(manifest, status="failed")
    path, _ = write_or_adopt_run_manifest(ctx, failed)
    path.write_text(json.dumps(dict(failed.to_public_dict(), command="other_command")) + "\n")
    with pytest.raises(RuntimeError, match="conflicts"):
        write_or_adopt_run_manifest(ctx, manifest, allow_failed_supersession=True)


def test_failed_manifest_supersession_accepts_same_plan_identity(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    manifest = manifest_from_result(_sample_result(tmp_path))
    records = dict(manifest.records, plan_fingerprint="plan-fp")
    failed = replace(manifest, status="failed", records=records)
    write_or_adopt_run_manifest(ctx, failed)
    succeeded = replace(manifest, records=records, started_at="2026-07-09T15:00:00")
    path, _ = write_or_adopt_run_manifest(
        ctx, succeeded, allow_failed_supersession=True
    )
    payload = json.loads(path.read_text())
    assert payload["status"] == "success"
    assert payload["started_at"] == failed.started_at


def test_different_success_manifest_is_never_subset_adopted(tmp_path):
    import pytest

    ctx = WorkspaceContext.from_root(tmp_path)
    manifest = manifest_from_result(_sample_result(tmp_path))
    path, _ = write_or_adopt_run_manifest(ctx, manifest)
    changed = replace(manifest, warnings=manifest.warnings + ("different closure",))
    with pytest.raises(RuntimeError, match="conflicts"):
        write_or_adopt_run_manifest(
            ctx, changed, allow_verified_success_adoption=True,
        )
    assert json.loads(path.read_text())["warnings"] == list(manifest.warnings)


def test_failed_supersession_requires_exact_receipt_ledger(tmp_path):
    import pytest

    ctx = WorkspaceContext.from_root(tmp_path)
    manifest = manifest_from_result(_sample_result(tmp_path))
    failed_records = dict(manifest.records, plan_fingerprint="plan-fp")
    failed = replace(manifest, status="failed", records=failed_records)
    write_or_adopt_run_manifest(ctx, failed)
    publication = {
        "committed_outputs": [{
            "path": "latest_train_records.json", "kind": "record", "fingerprint": "actual",
        }],
    }
    succeeded = replace(
        manifest,
        records=dict(failed_records, publication=publication),
    )
    with pytest.raises(RuntimeError, match="ledger"):
        write_or_adopt_run_manifest(
            ctx, succeeded, allow_failed_supersession=True,
            expected_receipt_ledger=({
                "path": "latest_train_records.json", "kind": "record",
                "fingerprint": "expected",
            },),
        )


def test_manifest_fingerprint_matches_durable_bytes_and_no_manifest_is_pure(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "workspace")
    manifest = manifest_from_result(_sample_result(ctx.root))
    assert not ctx.root.exists()
    assert not manifest_path(ctx, manifest.command, manifest.run_id).parent.exists()
    path, fingerprint = write_or_adopt_run_manifest(ctx, manifest)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint
