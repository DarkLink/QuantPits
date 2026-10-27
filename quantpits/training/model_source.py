"""Explicit, read-only model reference selection and optional single-file export.

This selects existing record identities. It does not inspect model artifacts,
certify a model, follow prediction ancestry, or publish current records.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from quantpits.training.persistence import fsync_directory, sha256_bytes
from quantpits.training.publication import (
    TrainingPublicationCoordinator, TrainingPublicationIntent, TrainingPublicationReceipt,
)
from quantpits.training.record_repository import TrainingRecordRepository
from quantpits.training.records import (
    TrainingRecordSnapshot, parse_training_record_document, resolve_model_record, split_model_key,
)
from quantpits.utils.model_keys import make_model_key, resolve_model_key
from quantpits.utils.workspace import WorkspaceContext


@dataclass(frozen=True)
class ModelSourceMember:
    requested: str
    key: Optional[str]
    status: str
    record: Optional[dict] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ModelSourceResult:
    source: str
    source_kind: str
    run_id: Optional[str]
    run_status: Optional[str]
    members: Tuple[ModelSourceMember, ...]
    errors: Tuple[str, ...] = ()
    output: Optional[str] = None
    output_status: str = "not_requested"

    @property
    def complete(self):
        return bool(self.members) and not self.errors and all(m.status == "selected" for m in self.members)

    def to_dict(self):
        from dataclasses import asdict
        return dict(asdict(self), complete=self.complete,
                    scope="existing record references; model artifacts not checked")


def _contained(ctx, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ctx.root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(ctx.root.resolve())
    except ValueError as exc:
        raise ValueError("model source path must stay inside workspace: %s" % path) from exc
    return resolved


def _json(path):
    value = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model source JSON must be an object: %s" % path)
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not value or value in (".", "..") or Path(value).name != value:
        raise ValueError("%s must be a single nonempty identifier" % label)
    return value


def _run_records(ctx, run_id, manifest):
    """Read the original published record postimage, never mutable latest."""
    _identifier(run_id, "run-id")
    path = _contained(ctx, "output/manifests/static_train/%s.json" % run_id)
    if (manifest.get("command") != "static_train" or manifest.get("run_id") != run_id
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1):
        raise ValueError("static run manifest identity/schema mismatch")
    if not isinstance(manifest.get("status"), str) or not manifest["status"]:
        raise ValueError("static run manifest status is missing")
    records = manifest.get("records", {})
    if not isinstance(records, dict) or records.get("family") != "static":
        raise ValueError("run-id export supports native static runs only")
    action = records.get("action")
    if action not in ("full", "incremental", "predict_only"):
        raise ValueError("unsupported static run action")
    keys = records.get("target_keys")
    if not isinstance(keys, list) or not keys or len(set(keys)) != len(keys):
        raise ValueError("static run target membership is missing or ambiguous")
    for key in keys:
        if split_model_key(key)[1] != "static":
            raise ValueError("static run contains a non-static target")
    outcomes = records.get("outcomes")
    if not isinstance(outcomes, list) or any(not isinstance(o, dict) for o in outcomes):
        raise ValueError("static run outcomes are malformed")
    # Failed/unpublished members are reported independently. Their old pointers
    # may still occur in a merged postimage and must never masquerade as this run.
    publication = records.get("publication", {})
    if not isinstance(publication, dict):
        raise ValueError("static run publication metadata is malformed")
    transaction = publication.get("transaction_id")
    if not transaction:
        return path, manifest, records, None, (path,)
    _identifier(transaction, "transaction-id")
    folder = _contained(ctx, "data/training_transactions/" + transaction)
    intent_path = _contained(ctx, folder / "intent.json")
    receipt_path = _contained(ctx, folder / "receipt.json")
    intent = TrainingPublicationIntent.from_dict(_json(intent_path))
    receipt = TrainingPublicationReceipt.from_dict(_json(receipt_path))
    validator = object.__new__(TrainingPublicationCoordinator)
    validator.verify_receipt(intent, receipt, verify_current=False)
    if (intent.transaction_id, intent.run_id, intent.family, intent.action) != (transaction, run_id, "static", action):
        raise ValueError("original publication belongs to another run")
    for name in ("plan_fingerprint", "execution_fingerprint", "resume_fingerprint"):
        if getattr(intent, name) != records.get(name):
            raise ValueError("original publication %s differs from manifest" % name)
    if (publication.get("status") != "committed" or publication.get("applied") is not True
            or publication.get("committed_outputs") != list(receipt.committed_outputs)):
        raise ValueError("manifest does not describe the original committed publication")
    published = {o.get("key") for o in outcomes if o.get("outcome") == "success" and o.get("published") is True}
    if set(receipt.published_keys) != published or not published.issubset(set(keys)):
        raise ValueError("original publication membership differs from manifest outcomes")
    member = next(m for m in intent.members if m.kind == "record")
    postimage = _contained(ctx, folder / member.staged_relative_path)
    # Stage paths must stay in this original transaction, not another run.
    postimage.relative_to(folder)
    raw = postimage.read_bytes()
    if sha256_bytes(raw) != member.postimage_fingerprint:
        raise ValueError("original record postimage fingerprint mismatch")
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("original record postimage must be an object")
    return path, manifest, records, document, (path, intent_path, receipt_path, postimage)


def _available(document):
    if not isinstance(document, dict):
        raise ValueError("source record must be an object")
    # Validate the declared format without letting an unrelated broken entry
    # hide the explicitly requested members.
    probe = dict(document, models={})
    if document.get("schema_version") == 2:
        probe["model_records"] = {}
    parse_training_record_document(probe)
    models = document.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("source models must be a mapping")
    names = list(models)
    if document.get("schema_version") == 2:
        if not isinstance(document.get("model_records"), dict):
            raise ValueError("V2 requires model_records")
        names.extend(key for key in document["model_records"] if key not in models)
    result = {}
    for name in names:
        canonical = make_model_key(name)
        result.setdefault(canonical, []).append(name)
    return result


def _subset(document, raw_keys):
    chosen = set(raw_keys)
    subset = copy.deepcopy(document)
    subset["models"] = {k: v for k, v in document.get("models", {}).items() if k in chosen}
    if document.get("schema_version") == 2:
        subset["model_records"] = {k: v for k, v in document["model_records"].items() if k in chosen}
    return subset


def _select(document, available, requested, run_records=None):
    members, raw_keys, used = [], [], set()
    for name in requested:
        key = None
        try:
            if not isinstance(name, str) or not name:
                raise ValueError("model request must be nonempty")
            if "@" in name:
                split_model_key(name)
            matches = [k for k in available if k == name or ("@" not in name and k.rsplit("@", 1)[0] == name)]
            if len(matches) > 1:
                members.append(ModelSourceMember(name, None, "ambiguous", error="specify model@mode: %s" % ", ".join(matches)))
                continue
            key = resolve_model_key(name, available)
            if key is None:
                members.append(ModelSourceMember(name, None, "missing", error="model is not in the explicit source"))
                continue
            if key in used:
                members.append(ModelSourceMember(name, key, "duplicate", error="model was already requested"))
                continue
            used.add(key)
            if len(available[key]) != 1:
                raise ValueError("source contains multiple aliases for this model key")
            raw_key = available[key][0]
            if run_records is not None:
                outcomes = [o for o in run_records["outcomes"] if o.get("key") == key]
                if len(outcomes) != 1:
                    raise ValueError("run outcome is missing or duplicated")
                outcome = outcomes[0]
                if outcome.get("outcome") != "success":
                    members.append(ModelSourceMember(name, key, "failed", error="run member outcome: %s" % outcome.get("outcome")))
                    continue
                if outcome.get("published") is not True:
                    raise ValueError("run member has no committed published record")
                expected = "predict_only" if run_records["action"] == "predict_only" else "train"
                if outcome.get("operation") != expected:
                    raise ValueError("run member operation differs from run action")
            recorder_id = document.get("models", {}).get(raw_key)
            if not isinstance(recorder_id, str) or not recorder_id.strip():
                raise ValueError("model has no recorder identity")
            resolved = resolve_model_record(_subset(document, [raw_key]), key)
            entry = resolved.entry
            if run_records is not None:
                if (entry.recorder_id, entry.experiment_name, entry.operation, entry.requested_anchor) != (
                    outcome.get("recorder_id"), outcome.get("experiment_name"), outcome.get("operation"), run_records.get("anchor_date")):
                    raise ValueError("original record differs from the requested run member")
            members.append(ModelSourceMember(name, key, "selected", entry.to_dict()))
            raw_keys.append(raw_key)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            members.append(ModelSourceMember(name, key, "unavailable", error=str(exc)))
    return tuple(members), raw_keys


def _same_json(raw, payload):
    try:
        return json.dumps(json.loads(raw), sort_keys=True, allow_nan=False) == json.dumps(json.loads(payload), sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, UnicodeError):
        return False


def _write_source(ctx, output, sources, payload):
    target = _contained(ctx, output)
    if target in sources or (target.exists() and any(p.exists() and target.samefile(p) for p in sources)):
        raise ValueError("source and output must be different files")
    if target.exists():
        if _same_json(target.read_bytes(), payload):
            if _contained(ctx, output) != target or not _same_json(target.read_bytes(), payload):
                raise ValueError("existing output changed during reuse")
            return target, "reused"
        raise FileExistsError("output already exists with different content; choose a new path: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _contained(ctx, output) != target:
        raise ValueError("output path changed before writing")
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if _contained(ctx, output) != target:
            raise ValueError("output path changed before commit")
        try:
            # Same-directory hard link publishes the complete fsynced file
            # atomically, with no replace/overwrite window.
            os.link(temporary, str(target))
            status = "created"
        except FileExistsError:
            if _contained(ctx, output) != target:
                raise ValueError("output path changed during concurrent creation")
            if not _same_json(target.read_bytes(), payload):
                raise FileExistsError("output appeared with different content; choose a new path")
            status = "reused"
        fsync_directory(target.parent)
        if _contained(ctx, output) != target or not _same_json(target.read_bytes(), payload):
            raise ValueError("output postcondition is uncertain at the requested path")
        return target, status
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_source(ctx: WorkspaceContext, *, models, run_id=None, record_file=None, output=None):
    """Preview explicit references; only output=... permits a single-file write.

    Ordinary source/member failures are reported, with requested order/count
    retained. Process-control exceptions propagate. Consumers still verify the
    referenced recorder and model; this function never upgrades their status.
    """
    requested = tuple(m.strip() if isinstance(m, str) else m for m in
                      (models.split(",") if isinstance(models, str) else models))
    source, run_status = str(run_id if run_id is not None else record_file), None
    kind = "run" if run_id is not None else "record_file"
    try:
        if (run_id is None) == (record_file is None):
            raise ValueError("choose exactly one run-id or record-file")
        records = None
        if run_id is not None:
            # Preserve the overall status even when original evidence is missing.
            path = _contained(ctx, "output/manifests/static_train/%s.json" % _identifier(run_id, "run-id"))
            header = _json(path)
            run_status = header.get("status")
            source = str(path)
            path, manifest, records, document, source_paths = _run_records(ctx, run_id, header)
            available = {k: [k] for k in records["target_keys"]}
            if document is None:
                document = {"schema_version": 2, "models": {}, "model_records": {}}
        else:
            path = _contained(ctx, record_file)
            source = str(path)
            document, source_paths = _json(path), (path,)
            available = _available(document)
        members, raw_keys = _select(document, available, requested, records)
        errors = () if requested else ("models request must not be empty",)
        output_path, output_status = None, "not_requested"
        if members and all(m.status == "selected" for m in members):
            subset = _subset(document, raw_keys)
            snapshot = parse_training_record_document(subset)
            payload = (TrainingRecordRepository.serialize(TrainingRecordSnapshot(snapshot.entries, snapshot.updated_at))
                       if subset.get("schema_version") == 2 else
                       (json.dumps(subset, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            if output is not None:
                try:
                    output_path, output_status = _write_source(ctx, output, source_paths, payload)
                    output_path = str(output_path)
                except (OSError, ValueError) as exc:
                    return ModelSourceResult(source, kind, run_id, run_status, members,
                                             (str(exc),), str(output), "failed")
        elif output is not None:
            output_status = "not_written"
        return ModelSourceResult(source, kind, run_id, run_status, members, errors, output_path, output_status)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return ModelSourceResult(source, kind, run_id, run_status,
                                 tuple(ModelSourceMember(m, None, "unavailable", error=str(exc)) for m in requested),
                                 (str(exc),), output_status="not_written" if output is not None else "not_requested")
