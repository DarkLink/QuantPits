"""Definition-bound, create-only retention of sealed model artifact bytes."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import sys
import weakref
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from quantpits.evidence.inspection import SourceMutationObserver
from quantpits.research import decision_surface as _phase37
from quantpits.research.forward_bootstrap import _accept_exact_create


SCHEMA_VERSION = 1
CAPSULE_KIND = "DEFINITION_BOUND_MODEL_ARTIFACT_CAPSULE_V1"
STORAGE_CLAIM = "CREATE_ONLY_EXACT_BYTES"
AUTHORIZATION_ACTION = "PUBLISH_ONE_DEFINITION_BOUND_MODEL_ARTIFACT_CAPSULE_V1"
MODEL_ARTIFACT_CAPSULE_AUTHORIZATION_ACTION = AUTHORIZATION_ACTION
SOURCE_RECEIPT_NAME = "source_receipt.json"
PAYLOAD_NAME = "artifact_payload.json"
MANIFEST_NAME = "capsule_manifest.json"
FINAL_NAMES = tuple(sorted((SOURCE_RECEIPT_NAME, PAYLOAD_NAME, MANIFEST_NAME)))
MAX_RAW_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_RAW_BYTES = 512 * 1024 * 1024
MAX_PAYLOAD_BYTES = 768 * 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID_RE = re.compile(r"^modelcapsule\.[0-9a-f]{64}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_REQUEST_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()


class ModelArtifactCapsuleContractError(ValueError):
    reason_code = "CONTRACT_INVALID"


class ModelArtifactCapsuleInputError(ModelArtifactCapsuleContractError):
    reason_code = "PRECONDITION_BLOCKED"


class _Blocked(ModelArtifactCapsuleInputError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class _Uncertain(Exception):
    pass


class _Conflict(Exception):
    pass


def _canonical(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ModelArtifactCapsuleContractError(
            "value is not canonical JSON",
        ) from exc


def _raw_digest(data: bytes) -> Dict[str, Any]:
    return TypedDigest.raw(data).to_dict()


def _canonical_digest(value: Any) -> Dict[str, Any]:
    return TypedDigest.canonical(value).to_dict()


def _typed_digest(value: Any, name: str, domain: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        raise ModelArtifactCapsuleContractError("%s digest fields are invalid" % name)
    try:
        result = TypedDigest(**value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ModelArtifactCapsuleContractError("%s digest is invalid" % name) from exc
    if result.domain != domain:
        raise ModelArtifactCapsuleContractError("%s digest domain is invalid" % name)
    return result.to_dict()


def _cycle(value: Any) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise ModelArtifactCapsuleContractError("evidence_cycle_id is invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ModelArtifactCapsuleContractError("evidence_cycle_id is invalid") from exc
    if parsed.isoformat() != value:
        raise ModelArtifactCapsuleContractError("evidence_cycle_id is invalid")
    return value


def _comparable_time(value: Any, name: str) -> datetime:
    if type(value) is not str or not value or value != value.strip():
        raise ModelArtifactCapsuleContractError("%s is invalid" % name)
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        if "T" in raw or " " in raw:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        parsed_date = date.fromisoformat(raw)
        return datetime(
            parsed_date.year, parsed_date.month, parsed_date.day,
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise ModelArtifactCapsuleContractError("%s is invalid" % name) from exc


def _capsule_id(value: Any) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ModelArtifactCapsuleContractError("capsule_id is invalid")
    return value


def _physical_directory(value: Any, name: str, *, private: bool = False) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise ModelArtifactCapsuleContractError("%s must be a path" % name)
    path = Path(value)
    if not path.is_absolute():
        raise ModelArtifactCapsuleContractError("%s must be absolute" % name)
    try:
        resolved = path.resolve(strict=True)
        info = os.lstat(str(path))
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise ModelArtifactCapsuleInputError("%s is unavailable" % name) from exc
    if (
        resolved != path or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or private and stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ModelArtifactCapsuleInputError("%s is not an admitted directory" % name)
    return path


def _identity(path: Path, *, private: bool = False) -> Tuple[int, int, int, int, int]:
    info = os.lstat(str(path))
    if (
        stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
        or private and stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ModelArtifactCapsuleInputError("directory identity is invalid")
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_ctime_ns


def _protected_inventory(
    root: Path, excluded: Optional[Path],
) -> Tuple[Tuple[Any, ...], ...]:
    """Observe whole-workspace namespace metadata, excluding one write target."""
    rows = []
    def failed(exc: OSError) -> None:
        raise ModelArtifactCapsuleInputError(
            "protected workspace inventory is incomparable",
        ) from exc
    for current, directories, files in os.walk(
        str(root), topdown=True, onerror=failed, followlinks=False,
    ):
        current_path = Path(current)
        kept = []
        for name in sorted(directories):
            path = current_path / name
            if excluded is not None and path == excluded:
                continue
            info = os.lstat(str(path))
            if excluded is None or path != excluded.parent:
                rows.append((
                    path.relative_to(root).as_posix(), "D", info.st_dev,
                    info.st_ino, info.st_mode, info.st_nlink, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns,
                ))
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            path = current_path / name
            if excluded is not None and (path == excluded or excluded in path.parents):
                continue
            info = os.lstat(str(path))
            rows.append((
                path.relative_to(root).as_posix(), "F", info.st_dev,
                info.st_ino, info.st_mode, info.st_nlink, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns,
            ))
    return tuple(rows)


def _roots(
    production_workspace_root: Any, research_workspace_root: Any,
    activation_path: Any, definition_store_root: Any,
    evidence_store_root: Any, capsule_store_root: Any,
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    production = _physical_directory(
        production_workspace_root, "production_workspace_root",
    )
    research = _physical_directory(
        research_workspace_root, "research_workspace_root",
    )
    activation = Path(activation_path)
    definitions = Path(definition_store_root)
    evidence = Path(evidence_store_root)
    capsules = _physical_directory(
        capsule_store_root, "capsule_store_root", private=True,
    )
    if any(not value.is_absolute() for value in (activation, definitions, evidence)):
        raise ModelArtifactCapsuleContractError("Research paths must be absolute")
    shadow = research / "research" / "shadow_v1"
    if (
        activation.parent != shadow / "activations"
        or definitions != shadow / "definitions"
        or evidence != shadow / "definition_evidence"
        or capsules != shadow / "model_artifact_capsules"
    ):
        raise ModelArtifactCapsuleContractError(
            "paths do not match the fixed Research layout",
        )
    for value, name, private in (
        (activation.parent, "activation_root", True),
        (definitions, "definition_store_root", True),
        (evidence, "evidence_store_root", True),
    ):
        _physical_directory(value, name, private=private)
    try:
        activation_resolved = activation.resolve(strict=True)
        activation_info = os.lstat(str(activation))
    except OSError as exc:
        raise ModelArtifactCapsuleInputError("activation is unavailable") from exc
    if (
        activation_resolved != activation or stat.S_ISLNK(activation_info.st_mode)
        or not stat.S_ISREG(activation_info.st_mode) or activation_info.st_nlink != 1
        or stat.S_IMODE(activation_info.st_mode) != 0o600
    ):
        raise ModelArtifactCapsuleInputError("activation is not an admitted file")
    if (
        production == research or production in research.parents
        or research in production.parents or production in capsules.parents
    ):
        raise ModelArtifactCapsuleContractError(
            "Production and Research must be physically separated",
        )
    return production, research, activation, definitions, evidence, capsules


def _close_guard(guard: Optional[SourceMutationObserver]) -> None:
    if guard is None:
        return
    active = sys.exc_info()[1]
    try:
        guard.close()
    except _PROCESS_CONTROL:
        if not isinstance(active, _PROCESS_CONTROL):
            raise
    except OSError:
        if active is None:
            raise


def _close_fd(fd: int, *, suppress: bool = False) -> None:
    active = sys.exc_info()[1]
    try:
        os.close(fd)
    except _PROCESS_CONTROL:
        if not isinstance(active, _PROCESS_CONTROL):
            raise
    except OSError:
        if active is None and not suppress:
            raise


def _relative(value: Any, name: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\0" in value:
        raise _Blocked(name + "_INVALID")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise _Blocked(name + "_INVALID")
    return value


def _read_regular(
    path: Path, maximum: int, *, expected: Optional[Mapping[str, Any]] = None,
    private: Optional[bool] = None,
) -> Tuple[bytes, Tuple[int, ...]]:
    try:
        before = os.lstat(str(path))
        if (
            stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1 or before.st_size > maximum
            or private is True and stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise _Blocked("SOURCE_MEMBER_INVALID")
        fd = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            opened = os.fstat(fd)
            identity = (
                before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
                before.st_size, before.st_mtime_ns, before.st_ctime_ns,
            )
            opened_identity = (
                opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink,
                opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns,
            )
            if opened_identity != identity:
                raise _Blocked("SOURCE_MEMBER_CONTINUITY_LOST")
            remaining = maximum + 1
            chunks = []
            while remaining:
                chunk = os.read(fd, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            _close_fd(fd)
        after = os.lstat(str(path))
        after_identity = (
            after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
            after.st_size, after.st_mtime_ns, after.st_ctime_ns,
        )
        if after_identity != identity or len(data) != before.st_size:
            raise _Blocked("SOURCE_MEMBER_CONTINUITY_LOST")
        if expected is not None and _raw_digest(data) != dict(expected):
            raise _Blocked("SOURCE_MEMBER_DIGEST_MISMATCH")
        return data, identity
    except _PROCESS_CONTROL:
        raise
    except _Blocked:
        raise
    except OSError as exc:
        raise _Blocked("SOURCE_MEMBER_READ_FAILED") from exc


def _definition_authority(paths: Tuple[Path, ...], cycle_id: str) -> Tuple[Any, Dict[str, Any]]:
    production, research, activation, definitions, evidence, _capsules = paths
    from quantpits.research.forward_definition_evidence import (
        adopt_fresh_champion_segment_definition_evidence,
    )
    from quantpits.research.forward_observation import (
        FreshChampionSegmentCandidate,
        observe_fresh_champion_segment_candidate,
    )
    adopted = adopt_fresh_champion_segment_definition_evidence(
        production, research, cycle_id, activation, definitions, evidence,
    )
    if (
        adopted.status != "ADOPTED" or adopted.evidence_receipt.did_write
        or not adopted.definition_evidence_complete
    ):
        raise _Blocked("DEFINITION_EVIDENCE_NOT_ADOPTED")
    candidate = observe_fresh_champion_segment_candidate(
        production, research, cycle_id, activation,
    )
    if type(candidate) is not FreshChampionSegmentCandidate:
        raise ModelArtifactCapsuleContractError(
            "definition observation returned foreign authority",
        )
    compiled_request = candidate.compiled_definitions.to_store_request()
    if (
        candidate.definition_set_id != adopted.definition_receipt.definition_set_id
        or candidate.evidence_cycle_id != cycle_id
        or dict(candidate.compiled_request_digest)
        != dict(adopted.definition_receipt.request_digest)
        or dict(compiled_request.request_digest)
        != dict(adopted.definition_receipt.request_digest)
    ):
        raise _Blocked("DEFINITION_EVIDENCE_JOIN_FAILED")
    champion = candidate.compiled_definitions.champion.to_dict()
    challenger = candidate.compiled_definitions.challenger.to_dict()
    champion_ids = tuple(row.get("source_id") for row in champion.get("source_members", ()))
    challenger_ids = tuple(row.get("source_id") for row in challenger.get("source_members", ()))
    if (
        len(champion_ids) != 4 or len(set(champion_ids)) != 4
        or len(challenger_ids) != 3 or len(set(challenger_ids)) != 3
        or not all(type(value) is str and value for value in champion_ids + challenger_ids)
    ):
        raise _Blocked("DEFINITION_SOURCE_SET_INVALID")
    cursor = 0
    positions = []
    for source_id in challenger_ids:
        try:
            position = champion_ids.index(source_id, cursor)
        except ValueError as exc:
            raise _Blocked("CHALLENGER_SUBSET_INVALID") from exc
        positions.append(position)
        cursor = position + 1
    if len(set(positions)) != 3:
        raise _Blocked("CHALLENGER_SUBSET_INVALID")
    authority = {
        "definition_set_id": candidate.definition_set_id,
        "definition_request_digest": dict(adopted.definition_receipt.request_digest),
        "definition_evidence_request_digest": dict(adopted.evidence_receipt.request_digest),
        "definition_evidence_manifest_digest": dict(adopted.evidence_receipt.manifest_digest),
        "definition_evidence_operation_id": adopted.evidence_receipt.operation_id,
        "champion_source_ids": list(champion_ids),
        "challenger_source_ids": list(challenger_ids),
        "omitted_champion_position": next(iter(set(range(4)) - set(positions))),
    }
    return adopted, authority


def _partition(manifest: Mapping[str, Any], authority: Mapping[str, Any]) -> Dict[str, Any]:
    # Reuse the sealed-cycle projection as aggregate schema/partition truth owner.
    projection = _phase37._source_projection(manifest)
    lineage = manifest.get("model_and_ensemble_lineage")
    artifacts = lineage.get("source_artifacts") if type(lineage) is dict else None
    combo = lineage.get("combo") if type(lineage) is dict else None
    models = lineage.get("source_models") if type(lineage) is dict else None
    resolved = combo.get("resolved_members") if type(combo) is dict else None
    if (
        type(artifacts) is not list or type(models) is not list
        or resolved != authority["champion_source_ids"]
        or len(artifacts) != 9 or len(models) != 4
        or [row.get("source_id") for row in projection.get("members", ())]
        != authority["champion_source_ids"]
    ):
        raise _Blocked("PHASE37A_SOURCE_PARTITION_INVALID")
    training: Dict[int, Mapping[str, Any]] = {}
    predictions: Dict[int, Mapping[str, Any]] = {}
    ensemble = []
    for row in artifacts:
        position = row.get("position") if type(row) is dict else None
        if type(row) is dict and row.get("role") == "source_training" and type(position) is int:
            training[position] = row
        elif type(row) is dict and "role" not in row and type(position) is int:
            predictions[position] = row
        elif type(row) is dict and "role" not in row and position == "ensemble":
            ensemble.append(row)
        else:
            raise _Blocked("PHASE37A_SOURCE_REMAINDER")
    if set(training) != set(range(4)) or set(predictions) != set(range(4)) or len(ensemble) != 1:
        raise _Blocked("PHASE37A_SOURCE_PARTITION_INVALID")
    for position in range(4):
        model = models[position]
        if (
            model.get("resolved_key") != authority["champion_source_ids"][position]
            or training[position].get("recorder_id") != model.get("source_recorder_id")
            or predictions[position].get("recorder_id") != model.get("recorder_id")
            or predictions[position].get("source_recorder_id") != model.get("source_recorder_id")
        ):
            raise _Blocked("PHASE37A_SOURCE_JOIN_FAILED")
    return {
        "training": tuple(training[position] for position in range(4)),
        "predictions": tuple(predictions[position] for position in range(4)),
        "ensemble": ensemble[0],
    }


def _selected_references(manifest: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    rows = manifest.get("referenced_evidence")
    if type(rows) is not list:
        raise _Blocked("REFERENCED_EVIDENCE_INVALID")
    selected = []
    seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if type(row) is not dict:
            raise _Blocked("REFERENCED_EVIDENCE_REMAINDER")
        locator = row.get("locator")
        is_prediction = (
            row.get("collection") == "inputs"
            and row.get("evidence_class") == "prediction"
            and row.get("kind") in {"config", "record"}
        )
        is_ensemble = (
            row.get("collection") == "inputs"
            and row.get("kind") in {"config", "record"}
            and (
                row.get("evidence_class") == "ensemble"
                or PurePosixPath(str(locator)).name
                in {"ensemble_config.json", "ensemble_records.json"}
            )
        )
        if not (is_prediction or is_ensemble):
            continue
        logical = _relative(locator, "REFERENCE_LOCATOR")
        observation = row.get("observation")
        if (
            row.get("locator_type") != "workspace_file"
            or type(observation) is not dict
            or observation.get("status") != "observed"
            or observation.get("preservation_status") != "embedded"
        ):
            raise _Blocked("SELECTED_REFERENCE_NOT_EMBEDDED")
        normalized = _typed_digest(
            observation.get("digest"), "reference", "raw_bytes",
        )
        previous = seen.get(logical)
        if previous is not None and previous != normalized:
            raise _Blocked("SELECTED_REFERENCE_DUPLICATE_CONFLICT")
        if previous is None:
            seen[logical] = normalized
            selected.append({"locator": logical, "digest": normalized})
    if (
        not selected
        or not {"ensemble_config.json", "ensemble_records.json"}.issubset({
            PurePosixPath(row["locator"]).name for row in selected
        })
    ):
        raise _Blocked("SELECTED_REFERENCE_SET_EMPTY")
    return tuple(sorted(selected, key=lambda row: row["locator"]))


def _metadata_paths(
    production: Path, row: Mapping[str, Any], position: int,
) -> Tuple[Path, Tuple[Path, ...]]:
    artifact_locator = _relative(row.get("artifact_locator"), "ARTIFACT_LOCATOR")
    artifact_root = production / artifact_locator
    try:
        resolved = artifact_root.resolve(strict=True)
        resolved.relative_to(production)
    except Exception as exc:
        raise _Blocked("ARTIFACT_PATH_ESCAPE") from exc
    if resolved != artifact_root or artifact_root.name != "artifacts":
        raise _Blocked("SOURCE_RUN_LAYOUT_INVALID")
    run_root = artifact_root.parent
    if run_root.name != row.get("recorder_id"):
        raise _Blocked("SOURCE_RUN_IDENTITY_MISMATCH")
    paths = [run_root / "meta.yaml"]
    for directory_name in ("params", "tags"):
        directory = run_root / directory_name
        try:
            info = os.lstat(str(directory))
        except OSError as exc:
            raise _Blocked("SOURCE_RUN_METADATA_MISSING") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise _Blocked("SOURCE_RUN_METADATA_INVALID")
        for parent, directories, files in os.walk(str(directory), followlinks=False):
            directories.sort()
            files.sort()
            for name in directories:
                info = os.lstat(os.path.join(parent, name))
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise _Blocked("SOURCE_RUN_METADATA_INVALID")
            paths.extend(Path(parent) / name for name in files)
    if len(paths) <= 1:
        raise _Blocked("SOURCE_RUN_METADATA_EMPTY")
    return run_root, tuple(paths)


def _run_metadata_claim(
    run_root: Path, paths: Sequence[Path], recorder_id: str, source_id: str,
    data_by_path: Mapping[Path, bytes],
) -> Dict[str, Any]:
    try:
        import yaml
        meta = yaml.safe_load(data_by_path[run_root / "meta.yaml"].decode("utf-8"))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _Blocked("SOURCE_RUN_META_INVALID") from exc
    if (
        type(meta) is not dict or meta.get("run_id") != recorder_id
        or str(meta.get("experiment_id")) != run_root.parent.name
    ):
        raise _Blocked("SOURCE_RUN_META_IDENTITY_MISMATCH")
    params = {}
    for path in paths:
        if path == run_root / "meta.yaml" or run_root / "params" not in path.parents:
            continue
        key = path.relative_to(run_root / "params").as_posix()
        try:
            params[key] = data_by_path[path].decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise _Blocked("SOURCE_RUN_PARAM_INVALID") from exc
    def find(name: str) -> Optional[str]:
        values = [value for key, value in params.items() if PurePosixPath(key).name == name]
        return values[0] if len(values) == 1 else None
    fit_start = find("fit_start_time")
    fit_end = find("fit_end_time")
    if not fit_start or not fit_end:
        raise _Blocked("SOURCE_RUN_TRAINING_WINDOW_MISSING")
    try:
        if _comparable_time(fit_start, "fit_start_time") > _comparable_time(
            fit_end, "fit_end_time",
        ):
            raise _Blocked("SOURCE_RUN_TRAINING_WINDOW_INVALID")
    except ModelArtifactCapsuleContractError as exc:
        raise _Blocked("SOURCE_RUN_TRAINING_WINDOW_INVALID") from exc
    mode = source_id.rsplit("@", 1)[1] if "@" in source_id else "LEGACY_UNQUALIFIED"
    observed_modes = [
        value for key, value in params.items()
        if PurePosixPath(key).name in {"mode", "training_mode"}
    ]
    if mode == "LEGACY_UNQUALIFIED":
        mode_consistency = "NOT_APPLICABLE"
    elif not observed_modes:
        mode_consistency = "NOT_DECLARED"
    elif len(observed_modes) == 1 and observed_modes[0] == mode:
        mode_consistency = "CONSISTENT"
    else:
        raise _Blocked("SOURCE_RUN_MODE_MISMATCH")
    return {
        "recorder_join_verified": True,
        "source_member_mode": mode,
        "source_member_mode_authority": "SEALED_SOURCE_MEMBER_ID",
        "source_member_mode_bound": True,
        "run_metadata_mode_consistency": mode_consistency,
        "fit_start_time": fit_start,
        "fit_end_time": fit_end,
        "training_window_verified": True,
    }


def _payload_member(
    category: str, position: Optional[int], slot: str, locator: str, data: bytes,
    preservation: str,
) -> Dict[str, Any]:
    return {
        "category": category,
        "source_position": position,
        "logical_slot": slot,
        "original_logical_locator": locator,
        "original_preservation": preservation,
        "raw_digest": _raw_digest(data),
        "size_bytes": len(data),
        "content_base64": base64.b64encode(data).decode("ascii"),
    }


def _payload_bytes(members: Sequence[Mapping[str, Any]]) -> Tuple[bytes, Dict[str, Any]]:
    order = {"SOURCE_TRAINING_ARTIFACT": 0, "MODEL_CONFIG": 1, "SOURCE_RUN_METADATA": 2}
    sorted_members = sorted(
        (dict(row) for row in members),
        key=lambda row: (
            order[row["category"]],
            -1 if row["source_position"] is None else row["source_position"],
            row["logical_slot"],
        ),
    )
    slots = [row["logical_slot"] for row in sorted_members]
    locators = [(row["category"], row["source_position"], row["original_logical_locator"]) for row in sorted_members]
    if len(slots) != len(set(slots)) or len(locators) != len(set(locators)):
        raise _Blocked("PAYLOAD_MEMBER_DUPLICATE")
    core = {
        "schema_version": SCHEMA_VERSION,
        "payload_kind": "DEFINITION_BOUND_RAW_MODEL_PAYLOAD_V1",
        "encoding": "STANDARD_BASE64",
        "members": sorted_members,
    }
    semantic = _canonical_digest(core)
    payload = {**core, "payload_digest": semantic}
    data = _canonical(payload)
    if len(data) > MAX_PAYLOAD_BYTES:
        raise _Blocked("PAYLOAD_BUDGET_EXCEEDED")
    return data, payload


class _Request:
    def __init__(
        self, cycle_id: str, authority: Mapping[str, Any], receipt: Mapping[str, Any],
        payload_data: bytes, payload: Mapping[str, Any], *, _authority: Any,
    ) -> None:
        if _authority is not _AUTHORITY:
            raise ModelArtifactCapsuleContractError("requests are inspector-owned")
        self.cycle_id = cycle_id
        self.authority = dict(authority)
        self.receipt = dict(receipt)
        self.payload_data = payload_data
        self.payload = dict(payload)
        identity = {
            "protocol": CAPSULE_KIND,
            "cycle_id": cycle_id,
            "definition_set_id": authority["definition_set_id"],
            "definition_request_digest": authority["definition_request_digest"],
            "definition_evidence_request_digest": authority["definition_evidence_request_digest"],
            "definition_evidence_manifest_digest": authority["definition_evidence_manifest_digest"],
            "definition_evidence_operation_id": authority["definition_evidence_operation_id"],
            "phase37a_seal_digest": receipt["phase37a_seal_digest"],
            "phase37a_manifest_digest": receipt["phase37a_manifest_digest"],
            "champion_source_ids": authority["champion_source_ids"],
            "challenger_source_ids": authority["challenger_source_ids"],
            "payload_digest": _raw_digest(payload_data),
            "source_receipt_digest": _raw_digest(_canonical(receipt)),
        }
        self.capsule_id = "modelcapsule." + hashlib.sha256(_canonical(identity)).hexdigest()
        self.request_digest = _canonical_digest(identity)
        self._validate(False)
        _REQUEST_BINDINGS[self] = self._binding()

    def _binding(self) -> bytes:
        return _canonical({
            "cycle_id": self.cycle_id, "authority": self.authority,
            "receipt": self.receipt, "payload_raw_digest": _raw_digest(self.payload_data),
            "payload": self.payload, "capsule_id": self.capsule_id,
            "request_digest": self.request_digest,
        })

    def _validate(self, binding: bool = True) -> None:
        _cycle(self.cycle_id)
        _capsule_id(self.capsule_id)
        if self.payload_data != _canonical(self.payload):
            raise ModelArtifactCapsuleContractError("payload bytes are not canonical")
        _validate_payload(self.payload)
        _validate_receipt(
            self.receipt, self.authority, self.payload_data, self.payload,
            self.cycle_id,
        )
        if binding and _REQUEST_BINDINGS.get(self) != self._binding():
            raise ModelArtifactCapsuleContractError("request authority is absent")

    @property
    def payload_digest(self) -> Mapping[str, Any]:
        self._validate()
        return MappingProxyType(_raw_digest(self.payload_data))

    def member_bytes(self) -> Tuple[Tuple[str, bytes], ...]:
        self._validate()
        return (
            (SOURCE_RECEIPT_NAME, _canonical(self.receipt)),
            (PAYLOAD_NAME, self.payload_data),
        )


def _validate_payload(payload: Mapping[str, Any]) -> Tuple[bytes, ...]:
    if type(payload) is not dict or set(payload) != {
        "schema_version", "payload_kind", "encoding", "members", "payload_digest",
    }:
        raise ModelArtifactCapsuleContractError("payload fields are invalid")
    if (
        type(payload["schema_version"]) is not int or payload["schema_version"] != 1
        or payload["payload_kind"] != "DEFINITION_BOUND_RAW_MODEL_PAYLOAD_V1"
        or payload["encoding"] != "STANDARD_BASE64" or type(payload["members"]) is not list
    ):
        raise ModelArtifactCapsuleContractError("payload envelope is invalid")
    core = {key: value for key, value in payload.items() if key != "payload_digest"}
    if _typed_digest(payload["payload_digest"], "payload", "canonical_json") != _canonical_digest(core):
        raise ModelArtifactCapsuleContractError("payload semantic digest is invalid")
    decoded = []
    slots = set()
    identities = set()
    total = 0
    previous = None
    order = {"SOURCE_TRAINING_ARTIFACT": 0, "MODEL_CONFIG": 1, "SOURCE_RUN_METADATA": 2}
    for row in payload["members"]:
        if type(row) is not dict or set(row) != {
            "category", "source_position", "logical_slot", "original_logical_locator",
            "original_preservation", "raw_digest", "size_bytes", "content_base64",
        }:
            raise ModelArtifactCapsuleContractError("payload member fields are invalid")
        category = row["category"]
        position = row["source_position"]
        if (
            category not in order
            or position is not None and (type(position) is not int or position not in range(4))
            or (category == "MODEL_CONFIG") != (position is None)
            or type(row["logical_slot"]) is not str or not row["logical_slot"]
            or row["logical_slot"] in slots
            or row["original_preservation"] not in {"embedded", "workspace_file", "source_run"}
        ):
            raise ModelArtifactCapsuleContractError("payload member identity is invalid")
        locator = _relative(row["original_logical_locator"], "PAYLOAD_LOCATOR")
        identity = category, position, locator
        if identity in identities:
            raise ModelArtifactCapsuleContractError("payload member is duplicated")
        key = (
            order[category], -1 if position is None else position,
            row["logical_slot"],
        )
        if previous is not None and key <= previous:
            raise ModelArtifactCapsuleContractError("payload member order is invalid")
        previous = key
        slots.add(row["logical_slot"])
        identities.add(identity)
        digest = _typed_digest(row["raw_digest"], "payload member", "raw_bytes")
        if type(row["size_bytes"]) is not int or row["size_bytes"] != digest["size_bytes"]:
            raise ModelArtifactCapsuleContractError("payload member size is invalid")
        try:
            raw = base64.b64decode(row["content_base64"], validate=True)
        except _PROCESS_CONTROL:
            raise
        except Exception as exc:
            raise ModelArtifactCapsuleContractError("payload Base64 is invalid") from exc
        if base64.b64encode(raw).decode("ascii") != row["content_base64"] or _raw_digest(raw) != digest:
            raise ModelArtifactCapsuleContractError("payload bytes are invalid")
        if len(raw) > MAX_RAW_MEMBER_BYTES:
            raise ModelArtifactCapsuleContractError("payload member exceeds budget")
        total += len(raw)
        decoded.append(raw)
    category_positions = {
        category: {
            row["source_position"] for row in payload["members"]
            if row["category"] == category
        }
        for category in order
    }
    groups: Dict[Tuple[str, Optional[int]], list] = {}
    for row in payload["members"]:
        groups.setdefault(
            (row["category"], row["source_position"]), [],
        ).append(row["logical_slot"])
    slot_valid = True
    for (category, position), values in groups.items():
        prefix = (
            "config-" if category == "MODEL_CONFIG"
            else "source-%d-artifact-" % position
            if category == "SOURCE_TRAINING_ARTIFACT"
            else "source-%d-metadata-" % position
        )
        if values != [prefix + "%06d" % index for index in range(len(values))]:
            slot_valid = False
    if (
        not decoded or len(decoded) > 200000 or total > MAX_TOTAL_RAW_BYTES
        or not slot_valid
        or category_positions["SOURCE_TRAINING_ARTIFACT"] != set(range(4))
        or category_positions["SOURCE_RUN_METADATA"] != set(range(4))
        or category_positions["MODEL_CONFIG"] != {None}
        or any(
            row["category"] == "SOURCE_TRAINING_ARTIFACT"
            and row["original_preservation"] not in {"embedded", "workspace_file"}
            or row["category"] == "MODEL_CONFIG"
            and row["original_preservation"] != "embedded"
            or row["category"] == "SOURCE_RUN_METADATA"
            and row["original_preservation"] != "source_run"
            for row in payload["members"]
        )
    ):
        raise ModelArtifactCapsuleContractError("payload total exceeds budget")
    return tuple(decoded)


_RECEIPT_FIELDS = {
    "schema_version", "source_claim", "evidence_cycle_id", "definition_set_id",
    "definition_request_digest", "definition_evidence_request_digest",
    "definition_evidence_manifest_digest", "definition_evidence_operation_id",
    "champion_source_ids", "challenger_source_ids", "omitted_champion_position",
    "phase37a_status", "phase37a_seal_digest", "phase37a_manifest_digest",
    "source_partition", "source_training_tree_digests", "prediction_tree_digests",
    "ensemble_tree_digest", "source_run_metadata_inventory_digests",
    "run_metadata_claims", "raw_member_count",
    "total_raw_bytes", "embedded_source_count", "live_source_count",
    "payload_digest", "definition_bound", "source_observation_complete", "did_write",
}


def _validate_receipt(
    receipt: Mapping[str, Any], authority: Mapping[str, Any], payload_data: bytes,
    payload: Mapping[str, Any], cycle_id: str,
) -> None:
    if type(receipt) is not dict or set(receipt) != _RECEIPT_FIELDS:
        raise ModelArtifactCapsuleContractError("source receipt fields are invalid")
    if (
        type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
        or receipt["source_claim"] != "EXACT_DEFINITION_BOUND_PHASE37A_MODEL_SOURCES_V1"
        or receipt["evidence_cycle_id"] != cycle_id
        or receipt["phase37a_status"] != "sealed_complete"
        or receipt["definition_bound"] is not True
        or receipt["source_observation_complete"] is not True
        or receipt["did_write"] is not False
        or any(receipt.get(key) != value for key, value in authority.items())
        or receipt["source_partition"] != {
            "source_training_count": 4, "prediction_join_count": 4,
            "ensemble_join_count": 1, "unassigned_count": 0,
        }
        or type(receipt["raw_member_count"]) is not int
        or receipt["raw_member_count"] != len(payload["members"])
        or type(receipt["total_raw_bytes"]) is not int
        or receipt["total_raw_bytes"] != sum(row["size_bytes"] for row in payload["members"])
        or type(receipt["embedded_source_count"]) is not int
        or type(receipt["live_source_count"]) is not int
        or receipt["embedded_source_count"] != sum(
            row["original_preservation"] == "embedded" for row in payload["members"]
        )
        or receipt["live_source_count"] != sum(
            row["original_preservation"] != "embedded" for row in payload["members"]
        )
        or receipt["payload_digest"] != _raw_digest(payload_data)
    ):
        raise ModelArtifactCapsuleContractError("source receipt is inconsistent")
    _typed_digest(receipt["phase37a_seal_digest"], "phase37a seal", "raw_bytes")
    _typed_digest(receipt["phase37a_manifest_digest"], "phase37a manifest", "raw_bytes")
    _typed_digest(receipt["definition_request_digest"], "definition request", "canonical_json")
    _typed_digest(receipt["definition_evidence_request_digest"], "evidence request", "canonical_json")
    _typed_digest(receipt["definition_evidence_manifest_digest"], "evidence manifest", "raw_bytes")
    if (
        type(receipt["definition_evidence_operation_id"]) is not str
        or _HEX_RE.fullmatch(receipt["definition_evidence_operation_id"]) is None
        or type(receipt["source_training_tree_digests"]) is not list
        or len(receipt["source_training_tree_digests"]) != 4
        or type(receipt["prediction_tree_digests"]) is not list
        or len(receipt["prediction_tree_digests"]) != 4
        or type(receipt["source_run_metadata_inventory_digests"]) is not list
        or len(receipt["source_run_metadata_inventory_digests"]) != 4
    ):
        raise ModelArtifactCapsuleContractError("source provenance is invalid")
    for digest in (
        receipt["source_training_tree_digests"]
        + receipt["prediction_tree_digests"]
        + receipt["source_run_metadata_inventory_digests"]
        + [receipt["ensemble_tree_digest"]]
    ):
        _typed_digest(digest, "artifact tree", "file_inventory")
    expected_metadata = [
        TypedDigest.canonical([{
            "path": member["original_logical_locator"],
            "digest": member["raw_digest"],
        } for member in payload["members"] if (
            member["category"] == "SOURCE_RUN_METADATA"
            and member["source_position"] == position
        )], "file_inventory").to_dict()
        for position in range(4)
    ]
    if receipt["source_run_metadata_inventory_digests"] != expected_metadata:
        raise ModelArtifactCapsuleContractError(
            "run metadata inventory digest is invalid",
        )
    claims = receipt["run_metadata_claims"]
    if type(claims) is not list or len(claims) != 4:
        raise ModelArtifactCapsuleContractError("run metadata claims are invalid")
    for position, claim in enumerate(claims):
        expected_mode = (
            authority["champion_source_ids"][position].rsplit("@", 1)[1]
            if "@" in authority["champion_source_ids"][position]
            else "LEGACY_UNQUALIFIED"
        )
        if (
            type(claim) is not dict or set(claim) != {
                "recorder_join_verified", "source_member_mode",
                "source_member_mode_authority", "source_member_mode_bound",
                "run_metadata_mode_consistency",
                "fit_start_time", "fit_end_time", "training_window_verified",
            }
            or claim["recorder_join_verified"] is not True
            or claim["source_member_mode_authority"] != "SEALED_SOURCE_MEMBER_ID"
            or claim["source_member_mode_bound"] is not True
            or claim["training_window_verified"] is not True
            or claim["source_member_mode"] != expected_mode
            or expected_mode == "LEGACY_UNQUALIFIED"
            and claim["run_metadata_mode_consistency"] != "NOT_APPLICABLE"
            or expected_mode != "LEGACY_UNQUALIFIED"
            and claim["run_metadata_mode_consistency"] not in {
                "NOT_DECLARED", "CONSISTENT",
            }
            or type(claim["fit_start_time"]) is not str or not claim["fit_start_time"]
            or type(claim["fit_end_time"]) is not str or not claim["fit_end_time"]
        ):
            raise ModelArtifactCapsuleContractError("run metadata claim is invalid")
        if _comparable_time(
            claim["fit_start_time"], "fit_start_time",
        ) > _comparable_time(claim["fit_end_time"], "fit_end_time"):
            raise ModelArtifactCapsuleContractError(
                "run metadata training window is invalid",
            )


def _assemble_live(
    paths: Tuple[Path, ...], cycle_id: str, production_guard: SourceMutationObserver,
) -> Tuple[_Request, Any]:
    production = paths[0]
    adopted, authority = _definition_authority(paths, cycle_id)
    try:
        cycle_path, manifest, _seal = _phase37._cycle_authority(production, cycle_id)
        manifest_data, _ = _phase37._read_regular(cycle_path / "manifest.json", private=True)
        seal_data, _ = _phase37._read_regular(cycle_path / "seal.json", private=True)
        partition = _partition(manifest, authority)
        references = _selected_references(manifest)
    except _PROCESS_CONTROL:
        raise
    except _Blocked:
        raise
    except Exception as exc:
        reason = getattr(exc, "reason_code", "PHASE37A_ADMISSION_FAILED")
        raise _Blocked(reason) from exc
    live_paths = []
    metadata_sets = []
    failures = []
    for position, row in enumerate(partition["training"]):
        live_paths.append(row["artifact_locator"])
        live_paths.extend(
            member["path"] for member in row["members"]
            if member["preservation_status"] == "workspace_file"
        )
        try:
            run_root, metadata_paths = _metadata_paths(production, row, position)
            metadata_sets.append((run_root, metadata_paths))
            live_paths.extend((
                (run_root / "params").relative_to(production).as_posix(),
                (run_root / "tags").relative_to(production).as_posix(),
            ))
            live_paths.extend(
                path.relative_to(production).as_posix() for path in metadata_paths
            )
        except _PROCESS_CONTROL:
            raise
        except Exception as exc:
            failures.append(getattr(
                exc, "reason_code", "SOURCE_RUN_METADATA_ADMISSION_FAILED",
            ))
            metadata_sets.append(None)
    production_guard.add_paths(tuple(live_paths))
    if not production_guard.supported or production_guard.mutated():
        raise _Blocked("SOURCE_CONTINUITY_LOST")
    members = []
    embedded_count = 0
    live_count = 0
    for position, row in enumerate(partition["training"]):
        for member_index, member in enumerate(row["members"]):
            try:
                locator = _relative(member.get("path"), "ARTIFACT_MEMBER_PATH")
                expected = _typed_digest(member.get("digest"), "artifact member", "raw_bytes")
                if expected["size_bytes"] > MAX_RAW_MEMBER_BYTES:
                    raise _Blocked("RAW_MEMBER_BUDGET_EXCEEDED")
                if member.get("status") != "observed" or member.get("preservation_status") not in {"embedded", "workspace_file"}:
                    raise _Blocked("ARTIFACT_MEMBER_INCOMPARABLE")
                if member["preservation_status"] == "embedded":
                    source = cycle_path / "objects" / expected["value"][:2] / expected["value"]
                    embedded_count += 1
                    private = True
                else:
                    source = production / locator
                    try:
                        resolved = source.resolve(strict=True)
                        resolved.relative_to(production)
                    except Exception as exc:
                        raise _Blocked("ARTIFACT_PATH_ESCAPE") from exc
                    if resolved != source:
                        raise _Blocked("ARTIFACT_PATH_ESCAPE")
                    live_count += 1
                    private = None
                data, _ = _read_regular(source, MAX_RAW_MEMBER_BYTES, expected=expected, private=private)
                members.append(_payload_member(
                    "SOURCE_TRAINING_ARTIFACT", position,
                    "source-%d-artifact-%06d" % (position, member_index),
                    locator, data, member["preservation_status"],
                ))
            except _PROCESS_CONTROL:
                raise
            except Exception as exc:
                failures.append(getattr(exc, "reason_code", "ARTIFACT_MEMBER_READ_FAILED"))
    for index, row in enumerate(references):
        try:
            expected = row["digest"]
            if expected["size_bytes"] > MAX_RAW_MEMBER_BYTES:
                raise _Blocked("RAW_MEMBER_BUDGET_EXCEEDED")
            source = cycle_path / "objects" / expected["value"][:2] / expected["value"]
            data, _ = _read_regular(source, MAX_RAW_MEMBER_BYTES, expected=expected, private=True)
            members.append(_payload_member(
                "MODEL_CONFIG", None, "config-%06d" % index,
                row["locator"], data, "embedded",
            ))
            embedded_count += 1
        except _PROCESS_CONTROL:
            raise
        except Exception as exc:
            failures.append(getattr(exc, "reason_code", "REFERENCE_READ_FAILED"))
    run_claims = []
    for position, metadata_set in enumerate(metadata_sets):
        if metadata_set is None:
            continue
        run_root, metadata_paths = metadata_set
        values = {}
        for member_index, path in enumerate(metadata_paths):
            try:
                data, _ = _read_regular(path, MAX_RAW_MEMBER_BYTES)
                values[path] = data
                members.append(_payload_member(
                    "SOURCE_RUN_METADATA", position,
                    "source-%d-metadata-%06d" % (position, member_index),
                    path.relative_to(production).as_posix(), data, "source_run",
                ))
                live_count += 1
            except _PROCESS_CONTROL:
                raise
            except Exception as exc:
                failures.append(getattr(exc, "reason_code", "SOURCE_RUN_METADATA_READ_FAILED"))
        if len(values) == len(metadata_paths):
            try:
                run_claims.append(_run_metadata_claim(
                    run_root, metadata_paths, partition["training"][position]["recorder_id"],
                    authority["champion_source_ids"][position], values,
                ))
            except _PROCESS_CONTROL:
                raise
            except Exception as exc:
                failures.append(getattr(exc, "reason_code", "SOURCE_RUN_METADATA_INVALID"))
    if failures:
        raise _Blocked(failures[0])
    total = sum(row["size_bytes"] for row in members)
    if any(row["size_bytes"] > MAX_RAW_MEMBER_BYTES for row in members):
        raise _Blocked("RAW_MEMBER_BUDGET_EXCEEDED")
    if total > MAX_TOTAL_RAW_BYTES:
        raise _Blocked("TOTAL_RAW_BUDGET_EXCEEDED")
    payload_data, payload = _payload_bytes(members)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "source_claim": "EXACT_DEFINITION_BOUND_PHASE37A_MODEL_SOURCES_V1",
        "evidence_cycle_id": cycle_id,
        **authority,
        "phase37a_status": manifest.get("status"),
        "phase37a_seal_digest": _raw_digest(seal_data),
        "phase37a_manifest_digest": _raw_digest(manifest_data),
        "source_partition": {
            "source_training_count": 4, "prediction_join_count": 4,
            "ensemble_join_count": 1, "unassigned_count": 0,
        },
        "source_training_tree_digests": [
            dict(row["artifact_tree_digest"]) for row in partition["training"]
        ],
        "prediction_tree_digests": [
            dict(row["artifact_tree_digest"]) for row in partition["predictions"]
        ],
        "ensemble_tree_digest": dict(partition["ensemble"]["artifact_tree_digest"]),
        "source_run_metadata_inventory_digests": [
            TypedDigest.canonical([{
                "path": member["original_logical_locator"],
                "digest": member["raw_digest"],
            } for member in payload["members"] if (
                member["category"] == "SOURCE_RUN_METADATA"
                and member["source_position"] == position
            )], "file_inventory").to_dict()
            for position in range(4)
        ],
        "run_metadata_claims": run_claims,
        "raw_member_count": len(payload["members"]),
        "total_raw_bytes": total,
        "embedded_source_count": embedded_count,
        "live_source_count": live_count,
        "payload_digest": _raw_digest(payload_data),
        "definition_bound": True,
        "source_observation_complete": True,
        "did_write": False,
    }
    request = _Request(
        cycle_id, authority, receipt, payload_data, payload, _authority=_AUTHORITY,
    )
    if production_guard.mutated():
        raise _Blocked("SOURCE_CONTINUITY_LOST")
    return request, adopted


def _manifest(request: _Request, members: Sequence[Tuple[str, bytes]]) -> bytes:
    rows = [{
        "logical_path": name, "raw_digest": _raw_digest(data),
        "size_bytes": len(data),
    } for name, data in members]
    return _canonical({
        "schema_version": SCHEMA_VERSION,
        "capsule_kind": CAPSULE_KIND,
        "storage_claim": STORAGE_CLAIM,
        "capsule_id": request.capsule_id,
        "evidence_cycle_id": request.cycle_id,
        "definition_set_id": request.authority["definition_set_id"],
        "request_digest": request.request_digest,
        "payload_digest": _raw_digest(request.payload_data),
        "source_receipt_digest": _raw_digest(_canonical(request.receipt)),
        "member_count": 2,
        "members": rows,
        "definition_bound": True,
        "self_contained_raw_retention": True,
        "model_artifact_retention_complete": True,
        "model_executable": False,
        "inference_parity_tested": False,
        "intent_capability": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
    })


def _read_bundle_member(fd: int, name: str, maximum: int) -> Tuple[bytes, Tuple[int, ...]]:
    before = os.stat(name, dir_fd=fd, follow_symlinks=False)
    if (
        stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size > maximum
    ):
        raise _Conflict()
    member_fd = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0), dir_fd=fd,
    )
    try:
        opened = os.fstat(member_fd)
        identity = (
            before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
            before.st_size, before.st_mtime_ns, before.st_ctime_ns,
        )
        if identity != (
            opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink,
            opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns,
        ):
            raise _Uncertain()
        remaining = maximum + 1
        chunks = []
        while remaining:
            chunk = os.read(member_fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    finally:
        _close_fd(member_fd)
    after = os.stat(name, dir_fd=fd, follow_symlinks=False)
    if identity != (
        after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
        after.st_size, after.st_mtime_ns, after.st_ctime_ns,
    ) or len(data) != before.st_size:
        raise _Uncertain()
    return data, identity


def _strict_json(data: bytes, name: str) -> Dict[str, Any]:
    def pairs(values: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _Conflict() from exc
    if type(value) is not dict or data != _canonical(value):
        raise _Conflict()
    return value


def _verify_bundle(target: Path, request: _Request) -> bytes:
    info = os.lstat(str(target))
    if (
        stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise _Conflict()
    fd = os.open(
        str(target), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if tuple(sorted(os.listdir(fd))) != FINAL_NAMES:
            raise _Conflict()
        receipt_data, _ = _read_bundle_member(fd, SOURCE_RECEIPT_NAME, MAX_RECEIPT_BYTES)
        payload_data, _ = _read_bundle_member(fd, PAYLOAD_NAME, MAX_PAYLOAD_BYTES)
        if receipt_data != _canonical(request.receipt) or payload_data != request.payload_data:
            raise _Conflict()
        expected = _manifest(request, ((SOURCE_RECEIPT_NAME, receipt_data), (PAYLOAD_NAME, payload_data)))
        manifest_data, _ = _read_bundle_member(fd, MANIFEST_NAME, MAX_RECEIPT_BYTES)
        if manifest_data != expected:
            raise _Conflict()
        return manifest_data
    finally:
        _close_fd(fd)


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise _Uncertain()
        offset += written


def _result(
    request: _Request, status: str, did_write: bool, verified: int,
    manifest: Optional[bytes] = None, reason: Optional[str] = None,
) -> "ModelArtifactCapsuleResult":
    reasons = {
        "COMMITTED": "COMMITTED", "ADOPTED": "EXACT_CAPSULE_ALREADY_PRESENT",
        "CONFLICT": "CAPSULE_CONFLICT", "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN",
    }
    return ModelArtifactCapsuleResult(
        _authority=_AUTHORITY, request=request, status=status,
        reason_code=reason or reasons[status], did_write=did_write,
        verified_file_count=verified,
        manifest_digest=None if manifest is None else _raw_digest(manifest),
        cycle_id=request.cycle_id, requested_capsule_id=request.capsule_id,
    )


def _blocked_result(
    cycle_id: str, reason: str, *, status: str = "PRECONDITION_BLOCKED",
    capsule_id: Optional[str] = None,
) -> "ModelArtifactCapsuleResult":
    return ModelArtifactCapsuleResult(
        _authority=_AUTHORITY, request=None, status=status, reason_code=reason,
        did_write=False, verified_file_count=0, manifest_digest=None,
        cycle_id=cycle_id, requested_capsule_id=capsule_id,
    )


def _publish_store(
    root: Path, request: _Request,
    handoff: Optional[list] = None,
) -> "ModelArtifactCapsuleResult":
    request._validate()
    target = root / request.capsule_id
    entry_guard = SourceMutationObserver(root, (request.capsule_id,))
    root_guard = SourceMutationObserver(root, ())
    parent_guard = SourceMutationObserver(root.parent, ())
    child_guard = None
    root_before = _identity(root, private=True)
    parent_before = _identity(root.parent)
    root_fd = target_fd = -1
    created_target = False
    prefix = 0
    created = {}
    try:
        if not entry_guard.supported or not root_guard.supported or not parent_guard.supported:
            raise ModelArtifactCapsuleInputError("publication continuity is unsupported")
        root_fd = os.open(
            str(root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.mkdir(request.capsule_id, 0o700, dir_fd=root_fd)
            created_target = True
            if not _accept_exact_create(entry_guard, request.capsule_id, directory=True):
                raise _Uncertain()
        except FileExistsError:
            child_guard = SourceMutationObserver(target, FINAL_NAMES)
            if not child_guard.supported:
                return _result(request, "UNCERTAIN", False, 0)
            target_before = _identity(target, private=True)
            try:
                manifest = _verify_bundle(target, request)
            except _Conflict:
                return _result(request, "CONFLICT", False, 0)
            except _PROCESS_CONTROL:
                raise
            except Exception:
                return _result(request, "UNCERTAIN", False, 0)
            if (
                entry_guard.mutated() or root_guard.mutated() or parent_guard.mutated()
                or child_guard.mutated() or _identity(root, private=True) != root_before
                or _identity(root.parent) != parent_before
                or _identity(target, private=True) != target_before
            ):
                return _result(request, "UNCERTAIN", False, 0)
            if handoff is not None:
                handoff.append(child_guard)
                child_guard = None
            return _result(request, "ADOPTED", False, 3, manifest)
        target_fd = os.open(
            request.capsule_id, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd,
        )
        child_guard = SourceMutationObserver(target, FINAL_NAMES)
        if not child_guard.supported or os.listdir(target_fd):
            raise _Uncertain()
        members = request.member_bytes()
        for name, data in members:
            fd = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=target_fd,
            )
            try:
                info = os.fstat(fd)
                identity = info.st_dev, info.st_ino, info.st_mode, info.st_nlink
                _write_all(fd, data)
                os.fsync(fd)
            finally:
                _close_fd(fd)
            if not _accept_exact_create(child_guard, name, directory=False):
                raise _Uncertain()
            observed, observed_identity = _read_bundle_member(target_fd, name, max(len(data), 1))
            if observed != data or observed_identity[:4] != identity:
                raise _Uncertain()
            created[name] = identity
            prefix += 1
        manifest = _manifest(request, members)
        fd = os.open(
            MANIFEST_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=target_fd,
        )
        try:
            info = os.fstat(fd)
            identity = info.st_dev, info.st_ino, info.st_mode, info.st_nlink
            _write_all(fd, manifest)
            os.fsync(fd)
        finally:
            _close_fd(fd)
        if not _accept_exact_create(child_guard, MANIFEST_NAME, directory=False):
            raise _Uncertain()
        observed, observed_identity = _read_bundle_member(target_fd, MANIFEST_NAME, max(len(manifest), 1))
        if observed != manifest or observed_identity[:4] != identity:
            raise _Uncertain()
        created[MANIFEST_NAME] = identity
        prefix += 1
        os.fsync(target_fd)
        os.fsync(root_fd)
        target_before = _identity(target, private=True)
        for name in FINAL_NAMES:
            info = os.stat(name, dir_fd=target_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino, info.st_mode, info.st_nlink) != created[name]:
                raise _Uncertain()
        _close_fd(target_fd)
        target_fd = -1
        _close_fd(root_fd)
        root_fd = -1
        verified = _verify_bundle(target, request)
        if (
            verified != manifest or entry_guard.mutated() or root_guard.mutated()
            or parent_guard.mutated() or child_guard.mutated()
            or _identity(root, private=True)[:3] != root_before[:3]
            or _identity(root.parent) != parent_before
            or _identity(target, private=True) != target_before
        ):
            raise _Uncertain()
        if handoff is not None:
            handoff.append(child_guard)
            child_guard = None
        return _result(request, "COMMITTED", True, 3, manifest)
    except _PROCESS_CONTROL:
        raise
    except _Conflict:
        return _result(
            request, "UNCERTAIN" if created_target else "CONFLICT",
            created_target, prefix,
        )
    except Exception as exc:
        if not created_target:
            if isinstance(exc, ModelArtifactCapsuleContractError):
                raise
            raise ModelArtifactCapsuleInputError(
                "publication failed before target creation",
            ) from exc
        return _result(request, "UNCERTAIN", True, prefix)
    finally:
        if target_fd >= 0:
            _close_fd(target_fd, suppress=True)
        if root_fd >= 0:
            _close_fd(root_fd, suppress=True)
        _close_guard(child_guard)
        _close_guard(parent_guard)
        _close_guard(root_guard)
        _close_guard(entry_guard)


def _adopt_store(root: Path, request: _Request) -> "ModelArtifactCapsuleResult":
    target = root / request.capsule_id
    entry_guard = SourceMutationObserver(root, (request.capsule_id,))
    parent_guard = SourceMutationObserver(root.parent, ())
    child_guard = None
    try:
        if not entry_guard.supported or not parent_guard.supported:
            return _result(request, "UNCERTAIN", False, 0)
        if not os.path.lexists(str(target)):
            return _result(request, "CONFLICT", False, 0)
        child_guard = SourceMutationObserver(target, FINAL_NAMES)
        if not child_guard.supported:
            return _result(request, "UNCERTAIN", False, 0)
        root_before = _identity(root, private=True)
        parent_before = _identity(root.parent)
        target_before = _identity(target, private=True)
        try:
            manifest = _verify_bundle(target, request)
        except _Conflict:
            return _result(request, "CONFLICT", False, 0)
        except _PROCESS_CONTROL:
            raise
        except Exception:
            return _result(request, "UNCERTAIN", False, 0)
        if (
            entry_guard.mutated() or parent_guard.mutated() or child_guard.mutated()
            or _identity(root, private=True) != root_before
            or _identity(root.parent) != parent_before
            or _identity(target, private=True) != target_before
        ):
            return _result(request, "UNCERTAIN", False, 0)
        return _result(request, "ADOPTED", False, 3, manifest)
    finally:
        _close_guard(child_guard)
        _close_guard(parent_guard)
        _close_guard(entry_guard)


def _request_from_target(
    paths: Tuple[Path, ...], cycle_id: str, capsule_id: str,
    target_guard: Optional[SourceMutationObserver] = None,
) -> Tuple[_Request, Any]:
    production, _research, _activation, _definitions, _evidence, capsules = paths
    adopted, authority = _definition_authority(paths, cycle_id)
    cycle_path, manifest, _seal = _phase37._cycle_authority(production, cycle_id)
    partition = _partition(manifest, authority)
    _selected_references(manifest)
    manifest_data, _ = _phase37._read_regular(cycle_path / "manifest.json", private=True)
    seal_data, _ = _phase37._read_regular(cycle_path / "seal.json", private=True)
    target = capsules / capsule_id
    if not os.path.lexists(str(target)):
        raise _Conflict()
    owns_guard = target_guard is None
    if target_guard is None:
        target_guard = SourceMutationObserver(target, FINAL_NAMES)
    try:
        if not target_guard.supported:
            raise _Uncertain()
        info = os.lstat(str(target))
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
            raise _Conflict()
        fd = os.open(str(target), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            if tuple(sorted(os.listdir(fd))) != FINAL_NAMES:
                raise _Conflict()
            receipt_data, _ = _read_bundle_member(fd, SOURCE_RECEIPT_NAME, MAX_RECEIPT_BYTES)
            payload_data, _ = _read_bundle_member(fd, PAYLOAD_NAME, MAX_PAYLOAD_BYTES)
        finally:
            _close_fd(fd)
        receipt = _strict_json(receipt_data, "source receipt")
        payload = _strict_json(payload_data, "artifact payload")
        _validate_payload(payload)
        required = {
            "schema_version", "source_claim", "evidence_cycle_id", "definition_set_id",
            "definition_request_digest", "definition_evidence_request_digest",
            "definition_evidence_manifest_digest", "definition_evidence_operation_id",
            "champion_source_ids", "challenger_source_ids", "omitted_champion_position",
            "phase37a_status", "phase37a_seal_digest", "phase37a_manifest_digest",
            "source_partition", "source_training_tree_digests", "prediction_tree_digests",
            "ensemble_tree_digest", "source_run_metadata_inventory_digests",
            "run_metadata_claims", "raw_member_count",
            "total_raw_bytes", "embedded_source_count", "live_source_count",
            "payload_digest", "definition_bound", "source_observation_complete", "did_write",
        }
        if set(receipt) != required:
            raise _Conflict()
        expected_authority = {key: authority[key] for key in authority}
        if any(receipt.get(key) != value for key, value in expected_authority.items()):
            raise _Conflict()
        if (
            receipt["evidence_cycle_id"] != cycle_id
            or receipt["phase37a_status"] != manifest.get("status")
            or receipt["phase37a_seal_digest"] != _raw_digest(seal_data)
            or receipt["phase37a_manifest_digest"] != _raw_digest(manifest_data)
            or receipt["source_training_tree_digests"]
            != [row["artifact_tree_digest"] for row in partition["training"]]
            or receipt["prediction_tree_digests"]
            != [row["artifact_tree_digest"] for row in partition["predictions"]]
            or receipt["ensemble_tree_digest"] != partition["ensemble"]["artifact_tree_digest"]
            or receipt["source_partition"] != {
                "source_training_count": 4, "prediction_join_count": 4,
                "ensemble_join_count": 1, "unassigned_count": 0,
            }
            or receipt["payload_digest"] != _raw_digest(payload_data)
            or receipt["definition_bound"] is not True
            or receipt["source_observation_complete"] is not True
            or receipt["did_write"] is not False
        ):
            raise _Conflict()
        claims = receipt["run_metadata_claims"]
        if type(claims) is not list or len(claims) != 4 or any(
            type(row) is not dict
            or row.get("recorder_join_verified") is not True
            or row.get("training_window_verified") is not True
            or not row.get("fit_start_time") or not row.get("fit_end_time")
            for row in claims
        ):
            raise _Conflict()
        request = _Request(
            cycle_id, authority, receipt, payload_data, payload, _authority=_AUTHORITY,
        )
        if request.capsule_id != capsule_id or target_guard.mutated():
            raise _Conflict()
        return request, adopted
    finally:
        if owns_guard:
            _close_guard(target_guard)


def _safe(
    request: _Request, status: str, reason: str, did_write: bool,
    complete: bool, verified: int, manifest_digest: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    receipt = request.receipt
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_kind": CAPSULE_KIND,
        "status": status,
        "reason_code": reason,
        "evidence_cycle_id": request.cycle_id,
        "definition_set_id": request.authority["definition_set_id"],
        "capsule_id": request.capsule_id,
        "request_digest": dict(request.request_digest),
        "manifest_digest": None if manifest_digest is None else dict(manifest_digest),
        "payload_digest": _raw_digest(request.payload_data),
        "champion_source_count": 4,
        "challenger_source_count": 3,
        "raw_member_count": receipt["raw_member_count"],
        "verified_file_count": verified,
        "total_raw_bytes": receipt["total_raw_bytes"],
        "embedded_source_count": receipt["embedded_source_count"],
        "live_source_count": receipt["live_source_count"],
        "definition_bound": complete,
        "self_contained_raw_retention": complete,
        "model_artifact_retention_complete": complete,
        "model_executable": False,
        "inference_parity_tested": False,
        "intent_capability": False,
        "state_chain_advanced": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
        "did_write": did_write,
    }


def _blocked_safe(
    cycle_id: str, status: str, reason: str, capsule_id: Optional[str],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_kind": CAPSULE_KIND,
        "status": status,
        "reason_code": reason,
        "evidence_cycle_id": cycle_id,
        "definition_set_id": None,
        "capsule_id": capsule_id,
        "request_digest": None,
        "manifest_digest": None,
        "payload_digest": None,
        "champion_source_count": 4,
        "challenger_source_count": 3,
        "raw_member_count": None,
        "verified_file_count": 0,
        "total_raw_bytes": None,
        "embedded_source_count": None,
        "live_source_count": None,
        "definition_bound": False,
        "self_contained_raw_retention": False,
        "model_artifact_retention_complete": False,
        "model_executable": False,
        "inference_parity_tested": False,
        "intent_capability": False,
        "state_chain_advanced": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
        "did_write": False,
    }


class ModelArtifactCapsulePlan:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY or set(kwargs) != {
            "request", "target_state", "status", "reason_code", "cycle_id",
        }:
            raise ModelArtifactCapsuleContractError("plans are inspector-owned")
        self._request = kwargs["request"]
        self.target_state = kwargs["target_state"]
        self.status = kwargs["status"]
        self.reason_code = kwargs["reason_code"]
        self.cycle_id = kwargs["cycle_id"]
        prepared = self.status == "PREPARED"
        if (
            self.status not in {"PREPARED", "PRECONDITION_BLOCKED"}
            or prepared != (type(self._request) is _Request)
            or prepared and self.target_state not in {"ABSENT", "PRESENT"}
            or not prepared and self.target_state != "UNOBSERVED"
            or prepared and self.reason_code != "PREPARED"
            or not prepared and (type(self.reason_code) is not str or not self.reason_code)
            or _cycle(self.cycle_id) != self.cycle_id
        ):
            raise ModelArtifactCapsuleContractError("plan fields are invalid")
        if self._request is not None:
            self._request._validate()
        _BINDINGS[self] = _canonical(self.to_safe_summary_dict(_binding=False))

    @property
    def capsule_id(self) -> Optional[str]:
        self._validate()
        return None if self._request is None else self._request.capsule_id

    @property
    def request_digest(self) -> Optional[Mapping[str, Any]]:
        self._validate()
        return None if self._request is None else MappingProxyType(dict(self._request.request_digest))

    def _validate(self) -> None:
        if self._request is not None:
            self._request._validate()
        if _BINDINGS.get(self) != _canonical(self.to_safe_summary_dict(_binding=False)):
            raise ModelArtifactCapsuleContractError("plan authority is absent")

    def to_safe_summary_dict(self, _binding: bool = True) -> Dict[str, Any]:
        if _binding:
            self._validate()
        result = (
            _safe(self._request, "PREPARED", "PREPARED", False, False, 0, None)
            if self._request is not None
            else _blocked_safe(self.cycle_id, self.status, self.reason_code, None)
        )
        result["target_state"] = self.target_state
        result["publication_capability"] = False
        return result


class ModelArtifactCapsuleResult:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY or set(kwargs) != {
            "request", "status", "reason_code", "did_write",
            "verified_file_count", "manifest_digest", "cycle_id",
            "requested_capsule_id",
        }:
            raise ModelArtifactCapsuleContractError("results are writer-owned")
        self._request = kwargs["request"]
        self.status = kwargs["status"]
        self.reason_code = kwargs["reason_code"]
        self.did_write = kwargs["did_write"]
        self.verified_file_count = kwargs["verified_file_count"]
        self.manifest_digest = kwargs["manifest_digest"]
        self.cycle_id = kwargs["cycle_id"]
        self.requested_capsule_id = kwargs["requested_capsule_id"]
        self._validate(False)
        _BINDINGS[self] = _canonical(self.to_safe_summary_dict(_binding=False))

    def _validate(self, binding: bool = True) -> None:
        if self._request is not None:
            self._request._validate()
        positive = self.status in {"COMMITTED", "ADOPTED"}
        expected_reason = {
            "COMMITTED": "COMMITTED", "ADOPTED": "EXACT_CAPSULE_ALREADY_PRESENT",
            "CONFLICT": "CAPSULE_CONFLICT", "UNCERTAIN": self.reason_code,
        }
        if (
            self.status not in {*expected_reason, "PRECONDITION_BLOCKED"}
            or positive and type(self._request) is not _Request
            or self.status == "PRECONDITION_BLOCKED" and self._request is not None
            or self._request is None and self.status not in {
                "PRECONDITION_BLOCKED", "CONFLICT", "UNCERTAIN",
            }
            or type(self.did_write) is not bool
            or type(self.verified_file_count) is not int
            or not 0 <= self.verified_file_count <= 3
            or positive != (self.verified_file_count == 3 and self.manifest_digest is not None)
            or self.status == "COMMITTED" and not self.did_write
            or self.status in {"ADOPTED", "CONFLICT"} and self.did_write
            or positive and self.reason_code != expected_reason[self.status]
            or self._request is None and (
                self.did_write or self.verified_file_count
                or self.manifest_digest is not None
            )
            or _cycle(self.cycle_id) != self.cycle_id
            or self.requested_capsule_id is not None
            and _capsule_id(self.requested_capsule_id) != self.requested_capsule_id
            or self._request is not None
            and self.requested_capsule_id != self._request.capsule_id
        ):
            raise ModelArtifactCapsuleContractError("result fields are inconsistent")
        if self.manifest_digest is not None:
            _typed_digest(self.manifest_digest, "manifest", "raw_bytes")
        if binding and _BINDINGS.get(self) != _canonical(self.to_safe_summary_dict(_binding=False)):
            raise ModelArtifactCapsuleContractError("result authority is absent")

    @property
    def model_artifact_retention_complete(self) -> bool:
        self._validate()
        return self.status in {"COMMITTED", "ADOPTED"}

    def to_safe_summary_dict(self, _binding: bool = True) -> Dict[str, Any]:
        if _binding:
            self._validate()
        if self._request is None:
            return _blocked_safe(
                self.cycle_id, self.status, self.reason_code,
                self.requested_capsule_id,
            )
        return _safe(
            self._request, self.status, self.reason_code, self.did_write,
            self.status in {"COMMITTED", "ADOPTED"}, self.verified_file_count,
            self.manifest_digest,
        )


def _guards(paths: Tuple[Path, ...], cycle_id: str) -> Tuple[SourceMutationObserver, SourceMutationObserver, Dict[str, Tuple[int, ...]]]:
    production, research, activation, definitions, evidence, capsules = paths
    production_guard = SourceMutationObserver(
        production, ("data/evidence/v1/cycles/%s" % cycle_id,),
    )
    research_guard = SourceMutationObserver(
        research, tuple(path.relative_to(research).as_posix() for path in (
            activation, definitions / activation.stem,
            evidence / activation.stem,
        )),
    )
    identities = {
        "production": _identity(production),
        "production_parent": _identity(production.parent),
        "research": _identity(research),
        "research_parent": _identity(research.parent),
        "capsules": _identity(capsules, private=True),
    }
    return production_guard, research_guard, identities


def _stable(
    paths: Tuple[Path, ...], guards: Tuple[SourceMutationObserver, SourceMutationObserver],
    identities: Mapping[str, Tuple[int, ...]], *, allow_capsule_change: bool = False,
) -> bool:
    production, research, _activation, _definitions, _evidence, capsules = paths
    return (
        all(guard.supported and not guard.mutated() for guard in guards)
        and _identity(production) == identities["production"]
        and _identity(production.parent) == identities["production_parent"]
        and _identity(research) == identities["research"]
        and _identity(research.parent) == identities["research_parent"]
        and (
            _identity(capsules, private=True) == identities["capsules"]
            if not allow_capsule_change
            else _identity(capsules, private=True)[:3] == identities["capsules"][:3]
        )
    )


def prepare_definition_bound_model_artifact_capsule(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any, definition_store_root: Any,
    evidence_store_root: Any, capsule_store_root: Any,
) -> ModelArtifactCapsulePlan:
    cycle_id = _cycle(evidence_cycle_id)
    paths = _roots(
        production_workspace_root, research_workspace_root, activation_path,
        definition_store_root, evidence_store_root, capsule_store_root,
    )
    production_guard, research_guard, identities = _guards(paths, cycle_id)
    try:
        production_before = _protected_inventory(paths[0], None)
        research_before = _protected_inventory(paths[1], None)
        if not _stable(paths, (production_guard, research_guard), identities):
            raise ModelArtifactCapsuleInputError("preflight continuity is unavailable")
        first, _ = _assemble_live(paths, cycle_id, production_guard)
        second, _ = _assemble_live(paths, cycle_id, production_guard)
        if first.capsule_id != second.capsule_id or first.request_digest != second.request_digest:
            raise ModelArtifactCapsuleInputError("repeated source observation differs")
        target = paths[-1] / second.capsule_id
        state = "PRESENT" if os.path.lexists(str(target)) else "ABSENT"
        if (
            not _stable(paths, (production_guard, research_guard), identities)
            or _protected_inventory(paths[0], None) != production_before
            or _protected_inventory(paths[1], None) != research_before
        ):
            raise ModelArtifactCapsuleInputError("preflight continuity is uncertain")
        return ModelArtifactCapsulePlan(
            _authority=_AUTHORITY, request=second, target_state=state,
            status="PREPARED", reason_code="PREPARED", cycle_id=cycle_id,
        )
    except _PROCESS_CONTROL:
        raise
    except ModelArtifactCapsuleContractError as exc:
        reason = getattr(exc, "reason_code", "PRECONDITION_BLOCKED")
        return ModelArtifactCapsulePlan(
            _authority=_AUTHORITY, request=None, target_state="UNOBSERVED",
            status="PRECONDITION_BLOCKED", reason_code=reason,
            cycle_id=cycle_id,
        )
    except Exception:
        return ModelArtifactCapsulePlan(
            _authority=_AUTHORITY, request=None, target_state="UNOBSERVED",
            status="PRECONDITION_BLOCKED", reason_code="SOURCE_ADMISSION_FAILED",
            cycle_id=cycle_id,
        )
    finally:
        _close_guard(research_guard)
        _close_guard(production_guard)


def publish_definition_bound_model_artifact_capsule(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any, definition_store_root: Any,
    evidence_store_root: Any, capsule_store_root: Any,
    expected_capsule_id: Any, expected_request_digest: Any,
    authorization_action: Any,
) -> ModelArtifactCapsuleResult:
    cycle_id = _cycle(evidence_cycle_id)
    expected_id = _capsule_id(expected_capsule_id)
    expected_digest = _typed_digest(
        expected_request_digest, "expected request", "canonical_json",
    )
    if type(authorization_action) is not str or authorization_action != AUTHORIZATION_ACTION:
        raise ModelArtifactCapsuleContractError("authorization_action is invalid")
    paths = _roots(
        production_workspace_root, research_workspace_root, activation_path,
        definition_store_root, evidence_store_root, capsule_store_root,
    )
    production_guard, research_guard, identities = _guards(paths, cycle_id)
    result = None
    final_result = None
    close_stable = True
    target_handoff = []
    try:
        target = paths[-1] / expected_id
        production_before = _protected_inventory(paths[0], None)
        research_before = _protected_inventory(paths[1], target)
        try:
            request, adopted = _assemble_live(paths, cycle_id, production_guard)
        except _PROCESS_CONTROL:
            raise
        except ModelArtifactCapsuleContractError as exc:
            final_result = _blocked_result(
                cycle_id, getattr(exc, "reason_code", "PRECONDITION_BLOCKED"),
            )
            request = None
            adopted = None
        except Exception:
            final_result = _blocked_result(cycle_id, "SOURCE_ADMISSION_FAILED")
            request = None
            adopted = None
        if request is None:
            pass
        elif request.capsule_id != expected_id or request.request_digest != expected_digest:
            raise ModelArtifactCapsuleContractError(
                "prepared request no longer matches owner authorization",
            )
        elif (
            not _stable(paths, (production_guard, research_guard), identities)
            or _protected_inventory(paths[0], None) != production_before
            or _protected_inventory(paths[1], target) != research_before
        ):
            raise ModelArtifactCapsuleInputError("sources changed before publication")
        else:
            result = _publish_store(paths[-1], request, target_handoff)
            if result.status not in {"COMMITTED", "ADOPTED"}:
                final_result = result
            else:
                try:
                    after, after_evidence = _assemble_live(paths, cycle_id, production_guard)
                    public_manifest = _verify_bundle(
                        paths[-1] / request.capsule_id, request,
                    )
                    stable = (
                        after.capsule_id == request.capsule_id
                        and after.request_digest == request.request_digest
                        and after_evidence.status == "ADOPTED"
                        and adopted.status == "ADOPTED"
                        and _stable(
                            paths, (production_guard, research_guard), identities,
                            allow_capsule_change=result.did_write,
                        )
                        and _protected_inventory(paths[0], None) == production_before
                        and _protected_inventory(paths[1], target) == research_before
                        and len(target_handoff) == 1
                        and target_handoff[0].supported
                        and not target_handoff[0].mutated()
                        and result.manifest_digest == _raw_digest(public_manifest)
                    )
                except _PROCESS_CONTROL:
                    raise
                except Exception:
                    stable = False
                final_result = result if stable else _result(
                    request, "UNCERTAIN", result.did_write,
                    result.verified_file_count,
                    reason="SOURCE_CONTINUITY_UNCERTAIN",
                )
    finally:
        active = sys.exc_info()[1]
        for guard in tuple(target_handoff) + (research_guard, production_guard):
            try:
                close_stable = close_stable and guard.supported and not guard.mutated()
                _close_guard(guard)
            except _PROCESS_CONTROL:
                if not isinstance(active, _PROCESS_CONTROL):
                    raise
            except OSError:
                close_stable = False
    if final_result is None:
        raise ModelArtifactCapsuleInputError("publication produced no result")
    if final_result.status in {"COMMITTED", "ADOPTED"} and not close_stable:
        return _result(
            final_result._request, "UNCERTAIN", final_result.did_write,
            final_result.verified_file_count,
            reason="SOURCE_CONTINUITY_UNCERTAIN",
        )
    return final_result


def adopt_definition_bound_model_artifact_capsule(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any, definition_store_root: Any,
    evidence_store_root: Any, capsule_store_root: Any, capsule_id: Any,
) -> ModelArtifactCapsuleResult:
    cycle_id = _cycle(evidence_cycle_id)
    identifier = _capsule_id(capsule_id)
    paths = _roots(
        production_workspace_root, research_workspace_root, activation_path,
        definition_store_root, evidence_store_root, capsule_store_root,
    )
    production_guard, research_guard, identities = _guards(paths, cycle_id)
    target_guard = None
    final_result = None
    close_stable = True
    try:
        production_before = _protected_inventory(paths[0], None)
        research_before = _protected_inventory(paths[1], None)
        target = paths[-1] / identifier
        if not os.path.lexists(str(target)):
            final_result = _blocked_result(
                cycle_id, "CAPSULE_CONFLICT", status="CONFLICT",
                capsule_id=identifier,
            )
        else:
            try:
                target_guard = SourceMutationObserver(target, FINAL_NAMES)
                request, _adopted = _request_from_target(
                    paths, cycle_id, identifier, target_guard,
                )
            except _Conflict:
                final_result = _blocked_result(
                    cycle_id, "CAPSULE_CONFLICT", status="CONFLICT",
                    capsule_id=identifier,
                )
            except _Uncertain:
                final_result = _blocked_result(
                    cycle_id, "CAPSULE_OBSERVATION_UNCERTAIN", status="UNCERTAIN",
                    capsule_id=identifier,
                )
            except _PROCESS_CONTROL:
                raise
            except ModelArtifactCapsuleContractError as exc:
                final_result = _blocked_result(
                    cycle_id, getattr(exc, "reason_code", "PRECONDITION_BLOCKED"),
                    capsule_id=identifier,
                )
            if final_result is None:
                if (
                    target_guard is None or not target_guard.supported
                    or target_guard.mutated()
                    or not _stable(paths, (production_guard, research_guard), identities)
                ):
                    final_result = _result(request, "UNCERTAIN", False, 0)
                else:
                    result = _adopt_store(paths[-1], request)
                    if result.status == "ADOPTED" and (
                        target_guard.mutated()
                        or not _stable(paths, (production_guard, research_guard), identities)
                        or _protected_inventory(paths[0], None) != production_before
                        or _protected_inventory(paths[1], None) != research_before
                    ):
                        final_result = _result(request, "UNCERTAIN", False, 0)
                    else:
                        final_result = result
    finally:
        active = sys.exc_info()[1]
        for guard in (target_guard, research_guard, production_guard):
            if guard is None:
                continue
            try:
                close_stable = close_stable and guard.supported and not guard.mutated()
                _close_guard(guard)
            except _PROCESS_CONTROL:
                if not isinstance(active, _PROCESS_CONTROL):
                    raise
            except OSError:
                close_stable = False
    if final_result is None:
        raise ModelArtifactCapsuleInputError("adoption produced no result")
    if final_result.status == "ADOPTED" and not close_stable:
        return _result(final_result._request, "UNCERTAIN", False, 0)
    return final_result


__all__ = [
    "AUTHORIZATION_ACTION", "MODEL_ARTIFACT_CAPSULE_AUTHORIZATION_ACTION",
    "CAPSULE_KIND", "ModelArtifactCapsuleContractError",
    "ModelArtifactCapsuleInputError", "ModelArtifactCapsulePlan",
    "ModelArtifactCapsuleResult",
    "prepare_definition_bound_model_artifact_capsule",
    "publish_definition_bound_model_artifact_capsule",
    "adopt_definition_bound_model_artifact_capsule",
]
