"""Audit and retire proven resolution failures without relaxing manifest closure."""

import json

from quantpits.runtime import manifest_path
from quantpits.training.errors import TrainingManifestConflictError
from quantpits.training.persistence import atomic_write_bytes, fsync_directory, sha256_bytes
from quantpits.utils.workspace import fingerprint_value


def _contained(prepared, path):
    try:
        path.resolve().relative_to(prepared.ctx.root.resolve())
    except ValueError as exc:
        raise TrainingManifestConflictError("training attempt path leaves workspace") from exc
    return path


def _request_fingerprint(prepared):
    return fingerprint_value({
        "plan_fingerprint": prepared.plan_fingerprint,
        "workspace": str(prepared.ctx.root.resolve()),
        "tracking_uri": prepared.ctx.mlflow_uri,
    })


def _execution_absent(prepared):
    """Observe the existing state/evidence/publication boundaries under the lease."""
    for path in (
        prepared.ctx.data_path("run_state.json"),
        prepared.ctx.data_path("training_runs", prepared.plan.run_id),
    ):
        _contained(prepared, path)
        if path.exists() or path.is_symlink():
            return False
    parent = _contained(prepared, prepared.ctx.data_path("training_transactions"))
    if parent.exists():
        for directory in parent.iterdir():
            _contained(prepared, directory)
            # An unidentified journal may describe an interrupted publication.
            intent = _contained(prepared, directory / "intent.json")
            try:
                value = json.loads(intent.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if not isinstance(value, dict) or not isinstance(value.get("run_id"), str) or not value["run_id"]:
                return False
            if value["run_id"] == prepared.plan.run_id or directory.name.startswith(prepared.plan.run_id + "-"):
                return False
    return True


def observe_resolution_failure(prepared):
    """Only a service-observed failure before resolution may receive this marker."""
    try:
        if _execution_absent(prepared):
            return {"schema_version": 1, "stage": "resolution",
                    "request_fingerprint": _request_fingerprint(prepared)}
    except (OSError, TrainingManifestConflictError):
        pass
    return None


def _read_manifest(raw):
    try:
        value = json.loads(raw.decode("utf-8"))
        records = value["records"]
        if not isinstance(records, dict):
            raise ValueError("invalid manifest records")
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise TrainingManifestConflictError("training manifest is not identifiable") from exc
    return value


def _validate_resolution_manifest(prepared, value, *, resumed):
    records = value["records"]
    marker = records.get("pre_execution_failure")
    legacy_resume = resumed and marker is None and set(records) == {
        "plan_fingerprint", "outcomes", "publication",
    }
    expected = {"schema_version": 1, "stage": "resolution",
                "request_fingerprint": _request_fingerprint(prepared)}
    if (
        type(value.get("schema_version")) is not int or value["schema_version"] != 1
        or value.get("status") != "failed"
        or value.get("run_id") != prepared.plan.run_id
        or value.get("command") != prepared.plan.command
        or value.get("workspace") != prepared.plan.workspace
        or records.get("plan_fingerprint") != prepared.plan_fingerprint
        or (not legacy_resume and (
            set(records) != {"family", "action", "target_keys", "plan_fingerprint",
                             "outcomes", "publication", "pre_execution_failure"}
            or not isinstance(marker, dict) or type(marker.get("schema_version")) is not int
            or records.get("family") != prepared.options.family
            or records.get("action") != prepared.options.action
            or records.get("target_keys") != [item.key for item in prepared.targets]
            or marker != expected
        ))
        or records.get("outcomes") != []
        or records.get("publication") != {"applied": False}
        or any(name in records for name in ("execution_fingerprint", "resume_fingerprint", "anchor_date"))
        or value.get("outputs") != []
        or not (resumed or _execution_absent(prepared))
    ):
        raise TrainingManifestConflictError("resolution failure lacks matching request or absence of execution evidence")


def retire_resolution_failure(prepared, *, validated_run=None, validated_state=None):
    """Keep exact failed bytes before freeing the canonical manifest for a retry.

    Without state, old failures lacking an observation marker are not inferred
    to be unexecuted. A validated resume state can establish the original request
    for a legacy resolution failure left behind before execution began.
    """
    path = _contained(prepared, manifest_path(prepared.ctx, prepared.plan.command, prepared.plan.run_id))
    resumed = validated_state is not None and validated_run is not None and (
        validated_state.run_id == prepared.plan.run_id
        and validated_state.plan_fingerprint == prepared.plan_fingerprint
        and validated_state.resume_fingerprint == validated_run.resume_fingerprint
        and validated_state.execution_fingerprint == validated_run.execution_fingerprint
        and validated_state.family == prepared.options.family
        and validated_state.action == prepared.options.action
        and validated_state.anchor_date == validated_run.anchor_date
        and validated_state.target_keys == tuple(item.key for item in validated_run.targets)
    )
    defer_to_state = (
        prepared.options.resume and validated_state is None
        and prepared.ctx.data_path("run_state.json").is_file()
    )
    if not path.exists():
        if defer_to_state:
            return
        # An interrupt after unlink but before state creation must retain the
        # same request identity through its exact-byte attempt audit.
        directory = _contained(prepared, prepared.ctx.data_path("training_attempts", prepared.plan.run_id))
        if directory.exists():
            for audit in directory.iterdir():
                raw = _contained(prepared, audit).read_bytes()
                if audit.name != "pre-execution-%s.json" % sha256_bytes(raw):
                    raise TrainingManifestConflictError("resolution failure audit fingerprint differs")
                _validate_resolution_manifest(prepared, _read_manifest(raw), resumed=resumed)
        return
    raw = path.read_bytes()
    value = _read_manifest(raw)
    records = value["records"]
    if records.get("pre_execution_failure") is None and records.get("execution_fingerprint"):
        # Normal executed-run closure remains governed by state and receipts.
        if prepared.options.resume:
            return
        raise TrainingManifestConflictError("existing executed manifest requires compatible --resume state")
    if value.get("status") == "failed" and defer_to_state:
        return
    _validate_resolution_manifest(prepared, value, resumed=resumed)
    fingerprint = sha256_bytes(raw)
    audit = _contained(prepared, prepared.ctx.data_path(
        "training_attempts", prepared.plan.run_id, "pre-execution-%s.json" % fingerprint,
    ))
    if audit.exists() and audit.read_bytes() != raw:
        raise TrainingManifestConflictError("resolution failure audit conflicts")
    if not audit.exists():
        atomic_write_bytes(audit, raw)
    # Verify through canonical paths, including after replace/unlink.
    if audit.read_bytes() != raw or path.read_bytes() != raw:
        raise TrainingManifestConflictError("resolution failure changed during audit")
    path.unlink()
    fsync_directory(path.parent)
    if path.exists() or path.is_symlink():
        raise TrainingManifestConflictError("resolution failure manifest removal is uncertain")
