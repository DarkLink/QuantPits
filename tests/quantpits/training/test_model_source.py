import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.model_source import model_source
from quantpits.training.publication import TrainingPublicationCoordinator
from quantpits.training.record_repository import TrainingRecordRepository
from quantpits.training.records import ModelRecordEntry, TrainingRecordSnapshot, snapshot_from_dict
from quantpits.training.resolved import resolve_training_run
from quantpits.training.runners import TrainingTargetResult
from quantpits.training.service import TrainingExecutionService, _public_outcomes
from quantpits.utils.workspace import WorkspaceContext


def _entry(name="alpha", recorder="old-fit", mode="static", **kwargs):
    return ModelRecordEntry(
        key=name + "@" + mode, model_name=name, training_mode=mode,
        operation="train", status="ready", recorder_id=recorder, experiment_name="Train",
        requested_anchor="2026-10-23", prediction_start="2026-10-19",
        prediction_end="2026-10-23", prediction_rows=5, **kwargs)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


def _tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def record_source(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "account")
    path = ctx.data_path("sources.json")
    doc = TrainingRecordSnapshot((_entry(), _entry("beta", "other-fit"))).to_dict()
    _write(path, doc)
    return ctx, path, doc


def test_preview_is_zero_write_and_keeps_requested_order(record_source):
    ctx, path, doc = record_source
    before = _tree(ctx.root)
    result = model_source(ctx, record_file="data/sources.json", models="beta,alpha@static")
    assert result.complete
    assert [m.key for m in result.members] == ["beta@static", "alpha@static"]
    assert [m.record["recorder_id"] for m in result.members] == ["other-fit", "old-fit"]
    assert result.output_status == "not_requested"
    assert _tree(ctx.root) == before
    assert not ctx.path("mlflow.db").exists()


def test_v2_export_preserves_direct_prediction_source_and_status(record_source):
    ctx, path, doc = record_source
    entry = replace(_entry(recorder="prediction-copy"), operation="predict_only",
                    source_recorder_id="direct-copy", source_experiment_name="Predict",
                    source_operation="predict_only", fit_start="2024-01-01", fit_end="2024-12-31")
    doc = TrainingRecordSnapshot((entry, _entry("beta"))).to_dict()
    _write(path, doc)
    before = _tree(ctx.root)
    result = model_source(ctx, record_file=str(path), models=["alpha"], output="data/selected.json")
    assert result.complete and result.output_status == "created"
    selected = json.loads(ctx.data_path("selected.json").read_text())
    assert selected["model_records"] == {entry.key: entry.to_dict()}
    assert result.members[0].record["operation"] == "predict_only"
    assert result.members[0].record["source_recorder_id"] == "direct-copy"
    assert _tree(ctx.root) == dict(before, **{"data/selected.json": ctx.data_path("selected.json").read_bytes()})


def test_v1_subset_retains_original_keys_mode_experiments_and_unknown_metadata(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    doc = {"models": {"beta@rolling": "roll-id", "alpha": "fit-id", "gamma": "third"},
           "experiment_name": "Train", "rolling_experiment_name": "Rolling",
           "anchor_date": "2026-10-23", "legacy_note": "keep"}
    _write(ctx.path("old.json"), doc)
    result = model_source(ctx, record_file="old.json", models="alpha@static,beta@rolling", output="picked.json")
    assert result.complete
    subset = json.loads(ctx.path("picked.json").read_text())
    assert "schema_version" not in subset
    assert list(subset["models"]) == ["beta@rolling", "alpha"]
    assert subset == dict(doc, models={"beta@rolling": "roll-id", "alpha": "fit-id"})
    assert all(m.record["status"] == "legacy_unverified" for m in result.members)
    assert all(m.record["operation"] == "legacy_import" for m in result.members)
    assert snapshot_from_dict(subset).entry_map["beta@rolling"].experiment_name == "Rolling"


@pytest.mark.parametrize("models,statuses", [
    ("alpha,,beta", ["selected", "unavailable", "selected"]),
    ("alpha,missing,beta", ["selected", "missing", "selected"]),
    ("alpha,alpha@static,beta", ["selected", "duplicate", "selected"]),
    ("alpha@unknown,beta", ["unavailable", "selected"]),
    ("", ["unavailable"]),
    ([], []),
])
def test_invalid_requests_never_write_a_reduced_success_set(record_source, models, statuses):
    ctx, path, doc = record_source
    before = _tree(ctx.root)
    result = model_source(ctx, record_file=str(path), models=models, output="must-not-create.json")
    assert not result.complete
    assert [m.status for m in result.members] == statuses
    assert _tree(ctx.root) == before


def test_ambiguous_bare_key_is_reported_and_explicit_mode_is_supported(record_source):
    ctx, path, doc = record_source
    _write(path, TrainingRecordSnapshot((_entry(), _entry(mode="rolling", recorder="roll"))).to_dict())
    ambiguous = model_source(ctx, record_file=str(path), models="alpha")
    assert ambiguous.members[0].status == "ambiguous"
    explicit = model_source(ctx, record_file=str(path), models="alpha@rolling", output="rolling.json")
    assert explicit.complete
    assert snapshot_from_dict(json.loads(ctx.path("rolling.json").read_text())).entry_map["alpha@rolling"].recorder_id == "roll"


def test_unselected_broken_entry_does_not_hide_explicit_valid_subset(record_source):
    ctx, path, doc = record_source
    doc["model_records"]["beta@static"]["recorder_id"] = "mismatch"
    _write(path, doc)
    result = model_source(ctx, record_file=str(path), models="alpha", output="subset.json")
    assert result.complete
    failed = model_source(ctx, record_file=str(path), models="beta,alpha", output="all.json")
    assert [m.status for m in failed.members] == ["unavailable", "selected"]
    assert not ctx.path("all.json").exists()


@pytest.fixture
def native_run(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "native")
    _write(ctx.config_path("model_config.json"), {"freq": "week"})
    ctx.config_path("model_registry.yaml").write_text(
        "models:\n  alpha:\n    enabled: true\n    yaml_file: alpha.yaml\n  beta:\n    enabled: true\n    yaml_file: beta.yaml\n")
    for name in ("alpha", "beta"):
        ctx.config_path(name + ".yaml").write_text("model: {}\n")
    # Preserve beta's old current pointer even though this run fails beta.
    _write(ctx.path("latest_train_records.json"), TrainingRecordSnapshot((_entry("beta", "older-beta"),)).to_dict())
    prepared = prepare_training_run(ctx=ctx, options=TrainingRunOptions(
        family="static", action="incremental", all_enabled=True, run_id="chosen-run"))
    run = resolve_training_run(prepared, {"anchor_date": "2026-10-23", "test_end_time": "2026-10-23"})
    entry = replace(_entry(), experiment_name=run.output_experiment_name)
    results = (TrainingTargetResult(entry.key, "train", "success", entry),
               TrainingTargetResult("beta@static", "train", "failed", error_code="fixture_failure"))
    coordinator = TrainingPublicationCoordinator(run, clock=datetime.now)
    intent = coordinator.prepare(results)
    receipt = coordinator.commit(intent)
    records = TrainingExecutionService._records(run, _public_outcomes(results, frozenset(receipt.published_keys)), receipt)
    manifest = {"schema_version": 1, "command": "static_train", "run_id": "chosen-run",
                "status": "failed", "records": records}
    path = _write(ctx.output_path("manifests/static_train/chosen-run.json"), manifest)
    return ctx, path, coordinator, intent, entry


def test_original_run_export_is_independent_of_later_latest(native_run):
    ctx, path, coordinator, intent, entry = native_run
    ctx.path("latest_train_records.json").write_text('{"models":{"alpha":"new-later-id"}}')
    before = _tree(ctx.root)
    result = model_source(ctx, run_id="chosen-run", models="alpha", output="data/original.json")
    assert result.complete and result.run_status == "failed"
    assert result.members[0].record == entry.to_dict()
    selected = snapshot_from_dict(json.loads(ctx.data_path("original.json").read_text()))
    assert selected.entry_map["alpha@static"] == entry
    assert _tree(ctx.root) == dict(before, **{"data/original.json": ctx.data_path("original.json").read_bytes()})


def test_run_members_never_fall_back_to_old_merged_current_pointer(native_run):
    ctx, path, coordinator, intent, entry = native_run
    result = model_source(ctx, run_id="chosen-run", models="beta,alpha,missing", output="all.json")
    assert [m.status for m in result.members] == ["failed", "selected", "missing"]
    assert result.members[1].record["recorder_id"] == "old-fit"
    assert not result.complete and not ctx.path("all.json").exists()


@pytest.mark.parametrize("asset", ["manifest", "intent", "receipt", "postimage"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_original_run_evidence_is_not_exportable(native_run, asset, damage):
    ctx, path, coordinator, intent, entry = native_run
    record = next(m for m in intent.members if m.kind == "record")
    target = {"manifest": path, "intent": coordinator.intent_path, "receipt": coordinator.receipt_path,
              "postimage": coordinator.directory / record.staged_relative_path}[asset]
    if damage == "missing":
        target.unlink()
    else:
        target.write_text("{}")
    result = model_source(ctx, run_id="chosen-run", models="alpha,beta", output="bad.json")
    assert not result.complete and len(result.members) == 2
    assert not ctx.path("bad.json").exists()


@pytest.mark.parametrize("field", ["run_id", "command", "schema_version", "family", "plan_fingerprint", "outcome_recorder", "outcome_operation"])
def test_run_identity_mismatch_is_reported(native_run, field):
    ctx, path, coordinator, intent, entry = native_run
    doc = json.loads(path.read_text())
    if field in ("run_id", "command"):
        doc[field] = "wrong"
    elif field == "schema_version":
        doc[field] = True
    elif field in ("family", "plan_fingerprint"):
        doc["records"][field] = "wrong"
    else:
        doc["records"]["outcomes"][0]["recorder_id" if field == "outcome_recorder" else "operation"] = "wrong"
    _write(path, doc)
    result = model_source(ctx, run_id="chosen-run", models="alpha", output="bad.json")
    assert not result.complete
    assert not ctx.path("bad.json").exists()


def test_no_publication_reports_failed_and_uncertain_members(native_run):
    ctx, path, coordinator, intent, entry = native_run
    doc = json.loads(path.read_text())
    doc["records"]["publication"] = {"applied": False}
    _write(path, doc)
    result = model_source(ctx, run_id="chosen-run", models="beta,alpha")
    assert [m.status for m in result.members] == ["failed", "unavailable"]


def test_output_reuses_identical_content_without_rewrite_and_rejects_conflicts(record_source):
    ctx, path, doc = record_source
    result = model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    target = ctx.path("picked.json")
    # Existing equivalent JSON with a different layout is retained byte for byte.
    target.write_text(json.dumps(json.loads(target.read_text()), separators=(",", ":")))
    prior = target.read_bytes(), target.stat().st_ino
    reused = model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    assert reused.output_status == "reused" and reused.complete
    assert (target.read_bytes(), target.stat().st_ino) == prior
    conflict = model_source(ctx, record_file=str(path), models="beta", output="picked.json")
    assert not conflict.complete and conflict.output_status == "failed"
    assert conflict.members[0].status == "selected"
    assert (target.read_bytes(), target.stat().st_ino) == prior


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_source_cannot_be_its_own_output(record_source, alias):
    ctx, path, doc = record_source
    target = path
    if alias != "same":
        target = ctx.path("alias.json")
        target.symlink_to(path) if alias == "symlink" else os.link(path, target)
    before = path.read_bytes()
    result = model_source(ctx, record_file=str(path), models="alpha,beta", output=str(target))
    assert not result.complete and "different files" in result.errors[0]
    assert path.read_bytes() == before


@pytest.mark.parametrize("selector", ["record_file", "output", "run_id"])
def test_paths_do_not_escape_workspace(record_source, tmp_path, selector):
    ctx, path, doc = record_source
    outside = tmp_path / "outside"
    outside.mkdir()
    ctx.path("link").symlink_to(outside, target_is_directory=True)
    kwargs = dict(record_file=str(path), models="alpha")
    if selector == "run_id":
        kwargs.pop("record_file")
        kwargs[selector] = "../elsewhere"
    else:
        kwargs[selector] = "link/target.json"
    result = model_source(ctx, **kwargs)
    assert not result.complete
    assert not list(outside.iterdir())


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_write_interrupts_propagate_and_leave_no_partial_target(record_source, monkeypatch, control):
    ctx, path, doc = record_source
    monkeypatch.setattr("quantpits.training.model_source.os.link", lambda *a, **k: (_ for _ in ()).throw(control()))
    with pytest.raises(control):
        model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    assert not ctx.path("picked.json").exists()
    assert not list(ctx.root.glob(".picked.json.*"))


def test_racing_existing_target_is_not_overwritten(record_source, monkeypatch):
    ctx, path, doc = record_source
    original = os.link
    def racing(source, target):
        Path(target).write_text("another writer")
        return original(source, target)
    monkeypatch.setattr("quantpits.training.model_source.os.link", racing)
    result = model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    assert not result.complete
    assert ctx.path("picked.json").read_text() == "another writer"


def test_concurrent_symlink_target_cannot_escape_workspace(record_source, tmp_path, monkeypatch):
    ctx, path, doc = record_source
    outside = tmp_path / "private.json"
    outside.write_text('{"private":"not a source"}')
    original = os.link
    def racing(source, target):
        Path(target).symlink_to(outside)
        return original(source, target)
    monkeypatch.setattr("quantpits.training.model_source.os.link", racing)
    result = model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    assert not result.complete
    assert "inside workspace" in result.errors[0]
    assert outside.read_text() == '{"private":"not a source"}'


def test_canonical_path_is_verified_after_write(record_source, monkeypatch):
    ctx, path, doc = record_source
    original = os.link
    def disappearing(source, target):
        original(source, target)
        Path(target).unlink()
    monkeypatch.setattr("quantpits.training.model_source.os.link", disappearing)
    result = model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    assert not result.complete and result.output_status == "failed"
    assert not ctx.path("picked.json").exists()


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_interruption_after_atomic_creation_leaves_a_reusable_complete_file(record_source, monkeypatch, control):
    ctx, path, doc = record_source
    original = os.link
    def interrupted(source, target):
        original(source, target)
        raise control()
    with monkeypatch.context() as patch:
        patch.setattr("quantpits.training.model_source.os.link", interrupted)
        with pytest.raises(control):
            model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    assert snapshot_from_dict(json.loads(ctx.path("picked.json").read_text())).entry_map["alpha@static"].recorder_id == "old-fit"
    resumed = model_source(ctx, record_file=str(path), models="alpha", output="picked.json")
    assert resumed.complete and resumed.output_status == "reused"


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_read_controls_propagate_and_missing_workspace_stays_missing(tmp_path, monkeypatch, control):
    ctx = WorkspaceContext.from_root(tmp_path / "missing")
    absent = model_source(ctx, record_file="missing.json", models="alpha,beta")
    assert len(absent.members) == 2 and not absent.complete
    assert not ctx.root.exists()
    monkeypatch.setattr("quantpits.training.model_source._json", lambda *a: (_ for _ in ()).throw(control()))
    with pytest.raises(control):
        model_source(ctx, record_file="missing.json", models="alpha,beta")
    assert not ctx.root.exists()


def test_cli_preview_has_no_workspace_env_or_backend_dependency(record_source):
    ctx, path, doc = record_source
    before = _tree(ctx.root)
    env = {k: v for k, v in os.environ.items() if k not in ("QLIB_WORKSPACE_DIR", "MLFLOW_TRACKING_URI")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = """
import sys
from quantpits.scripts.model_source import main
assert main(sys.argv[1:]) == 0
assert not any(m == 'qlib' or m.startswith('qlib.') or m == 'mlflow' or m.startswith('mlflow.') for m in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code, "--workspace", str(ctx.root),
                             "--record-file", str(path), "--models", "beta,alpha"],
                            cwd=str(Path(__file__).resolve().parents[3]), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert [m["key"] for m in json.loads(result.stdout)["members"]] == ["beta@static", "alpha@static"]
    assert _tree(ctx.root) == before
