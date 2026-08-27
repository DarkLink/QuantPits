"""Read-only sealed-reference observation for shadow-forward definitions."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import re
import stat
import sys
import weakref
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import (
    DecisionEvent,
    TypedDigest,
    canonical_json_bytes,
)
from quantpits.evidence.inspection import SourceMutationObserver


_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_OBSERVED_BINDINGS: "weakref.WeakKeyDictionary[Any, Tuple[Any, ...]]" = weakref.WeakKeyDictionary()
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION_LIMIT = 128 * 1024
_CONFIG_LIMIT = 256 * 1024
_SEAL_LIMIT = 128 * 1024
_MANIFEST_LIMIT = 16 * 1024 * 1024
_ACTIVATION_FIELDS = (
    "schema_version", "purpose", "definition_set_id", "protocol_id",
    "execution_assumption_id", "intent_definition_id", "champion_strategy_id",
    "challenger_strategy_id", "created_at", "effective_cycle", "evidence_cycle_id",
    "omitted_position", "selection_decision", "hypothesis", "execution_assumption",
)
_MANIFEST_FIELDS = frozenset({
    "schema_version", "cycle_identity", "engine_identity", "workspace_identity",
    "data_identity", "run_evidence", "model_and_ensemble_lineage", "ranking",
    "portfolio_state", "decision_state", "referenced_evidence", "preservation",
    "problems", "capture_time", "status", "request_content_digest",
})
_SEAL_FIELDS = frozenset({
    "schema_version", "cycle_id", "status", "manifest_digest",
    "artifact_root_digest", "object_digests", "named_file_digests",
})
_RECEIPT_FIELDS = (
    "schema_version", "observation_claim", "purpose", "definition_set_id",
    "evidence_cycle_id", "evidence_seal_digest", "evidence_manifest_digest",
    "strategy_config_digest", "activation_input_digest", "source_reference_count",
    "source_reference_inventory_digest", "selected_omitted_position",
    "compiled_request_digest", "sealed_reference_join_verified",
    "sealed_artifact_inventory_verified", "all_artifact_member_bytes_embedded",
    "live_artifact_member_bytes_reverified", "external_definition_claim",
    "publication_capability", "epoch_started", "prospective_claim",
    "promotion_capability",
)


class ForwardObservationContractError(ValueError):
    """The observation request or aggregate violates the C1B0 contract."""

    reason_code = "CONTRACT_INVALID"


class ForwardObservationInputError(ForwardObservationContractError):
    """A private input cannot support a sealed-reference join."""

    reason_code = "INPUT_INCOMPARABLE"

    def __init__(self, message: str, *, requested_sources: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.requested_source_count = len(tuple(requested_sources))
        self.requested_source_positions = tuple(range(self.requested_source_count))


def _input(message: str, *, sources: Sequence[str] = ()) -> ForwardObservationInputError:
    return ForwardObservationInputError(message, requested_sources=sources)


def _directory_identity(path: Path, *, private: bool = False) -> Tuple[int, int, int, int]:
    info = os.lstat(str(path))
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _input("authority directory is not physical")
    if private and stat.S_IMODE(info.st_mode) != 0o700:
        raise _input("sealed directory mode must be 0700")
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink


def _file_identity(info: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns,
    )


def _close_descriptor(descriptor: int) -> None:
    active = sys.exc_info()[1]
    try:
        os.close(descriptor)
    except _PROCESS_CONTROL:
        if isinstance(active, _PROCESS_CONTROL):
            return
        raise
    except OSError as exc:
        if isinstance(active, _PROCESS_CONTROL):
            return
        raise _input("descriptor cleanup failed") from exc


@contextmanager
def _mutation_guard(root: Path, watched: Sequence[str]):
    observer = SourceMutationObserver(root, watched)
    try:
        yield observer
    finally:
        active = sys.exc_info()[1]
        try:
            observer.close()
        except _PROCESS_CONTROL:
            if not isinstance(active, _PROCESS_CONTROL):
                raise
        except OSError as exc:
            if not isinstance(active, _PROCESS_CONTROL):
                raise _input("mutation observer cleanup failed") from exc


def _physical_root(value: Any) -> Path:
    if type(value) is not Path:
        try:
            value = Path(value)
        except Exception as exc:
            raise ForwardObservationContractError("workspace_root must be a path") from exc
    if not value.is_absolute():
        raise ForwardObservationContractError("workspace_root must be absolute")
    try:
        resolved = value.resolve(strict=True)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("workspace_root is unavailable") from exc
    if resolved != value or _directory_identity(value) != _directory_identity(resolved):
        raise _input("workspace_root must be its physical canonical path")
    return resolved


def _contained(root: Path, path: Path, *, regular: bool = False) -> Path:
    if not path.is_absolute():
        raise ForwardObservationContractError("private input path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("private input is unavailable or outside workspace") from exc
    if resolved != path:
        raise _input("private input must use its physical canonical path")
    info = os.lstat(str(path))
    if stat.S_ISLNK(info.st_mode):
        raise _input("private input cannot be a symlink")
    if regular and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise _input("private input must be a single-link regular file")
    return resolved


def _read_regular(
    path: Path, *, maximum: Optional[int] = None, expected_size: Optional[int] = None,
    private: bool = False, sealed: bool = False,
) -> Tuple[bytes, Tuple[int, int, int, int, int, int]]:
    """Read at most the accepted size plus one byte and prove name continuity."""
    try:
        before = os.lstat(str(path))
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise _input("observed member is not a single-link regular file")
        if private and stat.S_IMODE(before.st_mode) != 0o600:
            raise _input("activation mode must be 0600")
        if sealed and stat.S_IMODE(before.st_mode) != 0o600:
            raise _input("sealed member mode must be 0600")
        if expected_size is not None:
            if type(expected_size) is not int or expected_size < 0 or expected_size > _MANIFEST_LIMIT:
                raise _input("declared member size is invalid")
            limit = expected_size
            if before.st_size != expected_size:
                raise _input("observed member size differs from its sealed digest")
        else:
            if type(maximum) is not int or maximum < 0:
                raise ForwardObservationContractError("read maximum is invalid")
            limit = maximum
            if before.st_size > limit:
                raise _input("observed member exceeds its fixed size limit")
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                _file_identity(opened) != _file_identity(before)
                or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or opened.st_size > limit
            ):
                raise _input("observed member changed while opening")
            remaining = limit + 1
            chunks = []
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) != opened.st_size or len(data) > limit:
                raise _input("observed member changed while reading")
        finally:
            _close_descriptor(descriptor)
        after = os.lstat(str(path))
        if _file_identity(before) != _file_identity(after):
            raise _input("observed member public identity changed")
        return data, _file_identity(before)
    except _PROCESS_CONTROL:
        raise
    except ForwardObservationContractError:
        raise
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise _input("observed member path became noncanonical") from exc
        raise _input("observed member could not be read") from exc


def _strict_json(data: bytes, name: str) -> Dict[str, Any]:
    def pairs(values: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject(_value: str) -> None:
        raise ValueError("non-finite number")

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("%s is not strict JSON" % name) from exc
    if type(value) is not dict or data != canonical_json_bytes(value):
        raise _input("%s is not a canonical JSON object" % name)
    return value


def _typed_digest(value: Any, name: str, domain: Optional[str] = None) -> TypedDigest:
    if type(value) is not dict or set(value) != {"algorithm", "domain", "value", "size_bytes"}:
        raise _input("%s digest fields are invalid" % name)
    try:
        digest = TypedDigest(**value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("%s digest is invalid" % name) from exc
    if domain is not None and digest.domain != domain:
        raise _input("%s digest domain is invalid" % name)
    return digest


def _calendar_date(value: Any, name: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise ForwardObservationContractError("%s must be YYYY-MM-DD" % name)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ForwardObservationContractError("%s is not a real date" % name) from exc
    if parsed.isoformat() != value:
        raise ForwardObservationContractError("%s is not canonical" % name)
    return value


def _embedded_rows(value: Any) -> Tuple[Mapping[str, Any], ...]:
    found = []

    def visit(item: Any) -> None:
        if type(item) is dict:
            fields = {"path", "status", "digest", "preservation_status", "detail"}
            if fields.issubset(item) and item.get("preservation_status") == "embedded":
                found.append(item)
            for child in item.values():
                visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)

    visit(value)
    return tuple(found)


def _named_manifest_digests(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}
    ranking = manifest.get("ranking")
    if type(ranking) is dict and ranking.get("ranking_digest") is not None:
        result["ranking.csv"] = _typed_digest(
            ranking["ranking_digest"], "ranking", "raw_bytes",
        ).to_dict()
    portfolio = manifest.get("portfolio_state")
    if type(portfolio) is dict and portfolio.get("canonical_digest") is not None:
        digest = _typed_digest(
            portfolio["canonical_digest"], "portfolio", "semantic_config",
        )
        result["portfolio_state.json"] = TypedDigest(
            digest.algorithm, "raw_bytes", digest.value, digest.size_bytes,
        ).to_dict()
    return result


def _inventory(path: Path) -> Tuple[str, ...]:
    try:
        return tuple(sorted(item.name for item in os.scandir(str(path))))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("sealed directory inventory is unavailable") from exc


def _attach_requested_sources(
    exc: ForwardObservationInputError, sources: Sequence[str],
) -> ForwardObservationInputError:
    if sources:
        exc.requested_source_count = len(tuple(sources))
        exc.requested_source_positions = tuple(range(exc.requested_source_count))
    return exc


def _authority_identity(value: Any) -> Tuple[Any, ...]:
    """Freeze object identity recursively without trusting rendered equality."""
    if isinstance(value, Mapping):
        return (
            "mapping", type(value), id(value),
            tuple((id(key), key, _authority_identity(item)) for key, item in value.items()),
        )
    if isinstance(value, (tuple, list)):
        return (
            "sequence", type(value), id(value),
            tuple(_authority_identity(item) for item in value),
        )
    try:
        attributes = vars(value)
    except TypeError:
        return "atom", type(value), id(value)
    return "object", type(value), id(value), _authority_identity(attributes)


def _verify_bundle(
    root: Path, cycle_id: str,
) -> Tuple[Dict[str, Any], bytes, bytes, bool, Tuple[str, ...], Tuple[Dict[str, Any], ...], Tuple[int, ...]]:
    cycle = root / "data" / "evidence" / "v1" / "cycles" / cycle_id
    try:
        cycle = _contained(root, cycle)
    except ForwardObservationContractError:
        raise
    cycle_before = _directory_identity(cycle, private=True)
    seal_data, _ = _read_regular(cycle / "seal.json", maximum=_SEAL_LIMIT, sealed=True)
    seal = _strict_json(seal_data, "seal")
    if (
        set(seal) != _SEAL_FIELDS
        or type(seal.get("schema_version")) is not int
        or seal["schema_version"] != 1
    ):
        raise _input("seal schema is invalid")
    manifest_digest = _typed_digest(seal.get("manifest_digest"), "manifest", "raw_bytes")
    manifest_data, _ = _read_regular(
        cycle / "manifest.json", expected_size=manifest_digest.size_bytes, sealed=True,
    )
    if TypedDigest.raw(manifest_data) != manifest_digest:
        raise _input("manifest bytes do not match the seal")
    manifest = _strict_json(manifest_data, "manifest")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise _input("manifest schema is invalid")
    cycle_identity = manifest.get("cycle_identity")
    problems = manifest.get("problems")
    if (
        type(cycle_identity) is not dict
        or cycle_identity.get("cycle_id") != cycle_id
        or seal.get("cycle_id") != cycle_id
        or type(problems) is not list
        or any(
            type(item) is not dict
            or set(item) != {"code", "evidence_class", "detail", "blocks_complete"}
            or type(item.get("code")) is not str or not item.get("code")
            or type(item.get("evidence_class")) is not str or not item.get("evidence_class")
            or type(item.get("detail")) is not str
            or type(item.get("blocks_complete")) is not bool
            for item in problems
        )
    ):
        raise _input("sealed cycle identity or problem inventory is invalid")
    derived = "sealed_partial" if any(item["blocks_complete"] for item in problems) else "sealed_complete"
    if manifest.get("status") != derived or seal.get("status") != derived or derived != "sealed_complete":
        raise _input("cycle is not sealed_complete")
    replay_core = {
        key: value for key, value in manifest.items()
        if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
    }
    if _typed_digest(
        manifest.get("request_content_digest"), "request content", "canonical_json",
    ) != TypedDigest.canonical(replay_core):
        raise _input("manifest request content digest is invalid")
    lineage_hint = manifest.get("model_and_ensemble_lineage")
    combo_hint = lineage_hint.get("combo") if type(lineage_hint) is dict else None
    resolved_hint = combo_hint.get("resolved_members") if type(combo_hint) is dict else None
    requested_hint = tuple(resolved_hint) if (
        type(resolved_hint) is list and len(resolved_hint) == 4
        and all(type(value) is str and value for value in resolved_hint)
    ) else ()
    object_digests = seal.get("object_digests")
    named_digests = seal.get("named_file_digests")
    if (
        type(object_digests) is not list
        or object_digests != sorted(set(object_digests))
        or any(type(value) is not str or _HEX_RE.fullmatch(value) is None for value in object_digests)
        or type(named_digests) is not dict
        or set(named_digests) - {"ranking.csv", "portfolio_state.json"}
    ):
        raise _input("sealed artifact inventory is invalid", sources=requested_hint)
    embedded_rows = _embedded_rows(manifest)
    sizes: Dict[str, set] = {}
    for row in embedded_rows:
        # The sealed object public name always identifies its raw bytes, while
        # the manifest leaf may legitimately describe those canonical bytes
        # with a semantic_config/file_inventory domain.
        digest = _typed_digest(row.get("digest"), "embedded member")
        sizes.setdefault(digest.value, set()).add(digest.size_bytes)
    if any(len(values) != 1 for values in sizes.values()):
        raise _input("embedded member size observations conflict", sources=requested_hint)
    expected_objects = sorted(sizes)
    expected_named = _named_manifest_digests(manifest)
    preservation = manifest.get("preservation")
    if (
        expected_objects != object_digests or expected_named != named_digests
        or set(expected_named) != {"ranking.csv", "portfolio_state.json"}
        or type(preservation) is not dict
        or set(preservation) != {"embedded_object_count", "named_file_count"}
        or type(preservation.get("embedded_object_count")) is not int
        or type(preservation.get("named_file_count")) is not int
        or preservation["embedded_object_count"] != len(object_digests)
        or preservation["named_file_count"] != len(named_digests)
    ):
        raise _input(
            "manifest and seal artifact inventories do not join", sources=requested_hint,
        )
    expected_top = {"manifest.json", "seal.json", *named_digests}
    if object_digests:
        expected_top.add("objects")
    if set(_inventory(cycle)) != expected_top:
        raise _input("sealed cycle contains an unsealed top-level member", sources=requested_hint)
    if object_digests:
        objects = cycle / "objects"
        _directory_identity(objects, private=True)
        prefixes = sorted(set(value[:2] for value in object_digests))
        if _inventory(objects) != tuple(prefixes):
            raise _input("object prefix inventory is invalid")
        for prefix in prefixes:
            directory = objects / prefix
            _directory_identity(directory, private=True)
            expected = tuple(value for value in object_digests if value[:2] == prefix)
            if _inventory(directory) != expected:
                raise _input("object member inventory is invalid")
        for value in object_digests:
            expected_size = next(iter(sizes[value]))
            try:
                data, _ = _read_regular(
                    objects / value[:2] / value, expected_size=expected_size, sealed=True,
                )
            except ForwardObservationInputError as exc:
                raise _attach_requested_sources(exc, requested_hint)
            if hashlib.sha256(data).hexdigest() != value:
                raise _input(
                    "embedded object bytes do not match public name", sources=requested_hint,
                )
    for name, raw_digest in named_digests.items():
        digest = _typed_digest(raw_digest, "named member", "raw_bytes")
        try:
            data, _ = _read_regular(
                cycle / name, expected_size=digest.size_bytes, sealed=True,
            )
        except ForwardObservationInputError as exc:
            raise _attach_requested_sources(exc, requested_hint)
        if TypedDigest.raw(data) != digest:
            raise _input("named member bytes do not match the seal", sources=requested_hint)
    artifact_root = TypedDigest.canonical({
        "objects": object_digests, "named_files": named_digests,
    })
    if _typed_digest(
        seal.get("artifact_root_digest"), "artifact root", "canonical_json",
    ) != artifact_root:
        raise _input("artifact root digest is invalid")
    data_identity = manifest.get("data_identity")
    materialization = data_identity.get("qlib_materialization_identity") if type(data_identity) is dict else None
    cutoff = materialization.get("calendar_cutoff") if type(materialization) is dict else None
    if cutoff != cycle_id:
        raise _input("Qlib calendar cutoff does not match evidence cycle")
    lineage = manifest.get("model_and_ensemble_lineage")
    if type(lineage) is not dict or lineage.get("status") != "complete":
        raise _input("model and ensemble lineage is not complete")
    combo = lineage.get("combo")
    sources = lineage.get("source_models")
    artifacts = lineage.get("source_artifacts")
    resolved = combo.get("resolved_members") if type(combo) is dict else None
    if (
        type(resolved) is not list or len(resolved) != 4
        or any(type(value) is not str or not value for value in resolved)
        or len(set(resolved)) != 4
        or type(sources) is not list or len(sources) != 4
        or type(artifacts) is not list
    ):
        raise _input("Champion source inventory is not exact four")
    if any(
        type(row) is not dict
        or row.get("resolved_key") != resolved[position]
        or row.get("status") != "ready"
        or type(row.get("recorder_id")) is not str or not row.get("recorder_id")
        or type(row.get("source_recorder_id")) is not str or not row.get("source_recorder_id")
        for position, row in enumerate(sources)
    ):
        raise _input("Champion source model order or readiness is invalid", sources=resolved)
    training_rows: Dict[int, Dict[str, Any]] = {}
    prediction_rows: Dict[int, Dict[str, Any]] = {}
    ensemble_rows = []
    for artifact in artifacts:
        if type(artifact) is not dict:
            raise _input("source artifact remainder is malformed", sources=resolved)
        position = artifact.get("position")
        role_present = "role" in artifact
        role = artifact.get("role")
        if role_present and role == "source_training" and type(position) is int and position in range(4):
            expected_fields = {
                "position", "role", "recorder_id", "experiment_name",
                "artifact_locator", "members", "artifact_tree_digest",
            }
            destination = training_rows
        elif not role_present and type(position) is int and position in range(4):
            expected_fields = {
                "position", "recorder_id", "artifact_locator", "members",
                "artifact_tree_digest",
            }
            destination = prediction_rows
        elif not role_present and position == "ensemble":
            expected_fields = {
                "position", "recorder_id", "artifact_locator", "members",
                "artifact_tree_digest",
            }
            destination = None
        else:
            raise _input("source artifact remainder is unassigned", sources=resolved)
        if set(artifact) != expected_fields:
            raise _input("source artifact fields are not exact", sources=resolved)
        recorder_id = artifact.get("recorder_id")
        locator = artifact.get("artifact_locator")
        locator_path = PurePosixPath(locator) if type(locator) is str else None
        if (
            type(recorder_id) is not str or not recorder_id
            or type(locator) is not str or not locator
            or "\\" in locator or "\0" in locator or locator_path is None
            or locator_path.is_absolute() or locator_path.as_posix() != locator
            or any(part in {"", ".", ".."} for part in locator_path.parts)
            or (
                role_present
                and (type(artifact.get("experiment_name")) is not str or not artifact["experiment_name"])
            )
        ):
            raise _input("source artifact aggregate identity is invalid", sources=resolved)
        members = artifact.get("members")
        if type(members) is not list or not members:
            raise _input("source artifact inventory is empty", sources=resolved)
        inventory = []
        member_paths = set()
        for member in members:
            if type(member) is not dict or set(member) != {
                "path", "status", "digest", "preservation_status", "detail",
            }:
                raise _input("source artifact row is invalid", sources=resolved)
            path = member.get("path")
            pure_path = PurePosixPath(path) if type(path) is str else None
            if (
                type(path) is not str or not path or path in member_paths
                or "\\" in path or "\0" in path or pure_path is None
                or pure_path.is_absolute() or pure_path.as_posix() != path
                or any(part in {"", ".", ".."} for part in pure_path.parts)
                or member.get("status") != "observed"
                or member.get("preservation_status") not in {"embedded", "workspace_file"}
                or type(member.get("detail")) is not str
            ):
                raise _input("source artifact identity is invalid", sources=resolved)
            member_paths.add(path)
            digest = _typed_digest(member.get("digest"), "source artifact", "raw_bytes")
            inventory.append({"path": path, "digest": digest.to_dict()})
        inventory_bytes = canonical_json_bytes(inventory)
        tree = _typed_digest(
            artifact.get("artifact_tree_digest"), "source artifact tree", "file_inventory",
        )
        rebuilt = TypedDigest.canonical(inventory, "file_inventory")
        if tree != rebuilt:
            raise _input("source artifact inventory digest is invalid", sources=resolved)
        validated = {
            "row": artifact,
            "inventory_bytes": inventory_bytes,
            "all_embedded": all(
                member["preservation_status"] == "embedded" for member in members
            ),
        }
        if destination is None:
            ensemble_rows.append(validated)
            if len(ensemble_rows) != 1:
                raise _input("ensemble artifact relation is not unique", sources=resolved)
        else:
            if position in destination:
                raise _input("source artifact relation is not unique", sources=resolved)
            destination[position] = validated

    if set(training_rows) != set(range(4)):
        raise _input("source training relation is not one-to-one", sources=resolved)
    if any(
        row["row"]["recorder_id"] != sources[position]["recorder_id"]
        for position, row in prediction_rows.items()
    ):
        raise _input("prediction artifact recorder identity does not join", sources=resolved)
    combo_recorder = combo.get("recorder_id") if type(combo) is dict else None
    if ensemble_rows and (
        type(combo_recorder) is not str or not combo_recorder
        or ensemble_rows[0]["row"]["recorder_id"] != combo_recorder
    ):
        raise _input("ensemble artifact recorder identity does not join", sources=resolved)

    reference_rows = []
    all_embedded = True
    for position, source_id in enumerate(resolved):
        validated = training_rows[position]
        artifact = validated["row"]
        if artifact["recorder_id"] != sources[position].get("source_recorder_id"):
            raise _input("source training recorder identity does not join", sources=resolved)
        inventory_bytes = validated["inventory_bytes"]
        all_embedded = all_embedded and validated["all_embedded"]
        reference_rows.append({
            "position": position,
            "source_id": source_id,
            "model_artifact_digest": {
                "algorithm": "sha256", "domain": "raw_bytes",
                "value": hashlib.sha256(inventory_bytes).hexdigest(),
                "size_bytes": len(inventory_bytes),
            },
        })
    if _directory_identity(cycle, private=True) != cycle_before:
        raise _input("cycle directory identity changed during observation", sources=resolved)
    return (
        manifest, manifest_data, seal_data, all_embedded, tuple(resolved),
        tuple(reference_rows), tuple(range(4)),
    )


def _activation(data: bytes, cycle_id: str) -> Tuple[Dict[str, Any], DecisionEvent, Any]:
    raw = _strict_json(data, "activation")
    if set(raw) != set(_ACTIVATION_FIELDS):
        raise _input("activation fields are not exact")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise _input("activation schema_version is invalid")
    if raw.get("purpose") != "ENGINEERING_VALIDATION" or type(raw.get("purpose")) is not str:
        raise _input("activation purpose is invalid")
    if raw.get("evidence_cycle_id") != cycle_id:
        raise _input("activation evidence cycle does not match")
    omitted = raw.get("omitted_position")
    if type(omitted) is not int or omitted not in range(4):
        raise _input("activation omitted_position is invalid")
    if type(raw.get("hypothesis")) is not str or not raw["hypothesis"].strip():
        raise _input("activation hypothesis is invalid")
    try:
        decision = DecisionEvent.from_mapping(raw.get("selection_decision"))
        from quantpits.research.accounting import ExecutionAssumption
        assumption = ExecutionAssumption.from_dict(raw.get("execution_assumption"))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("activation typed leaves are invalid") from exc
    if (
        decision.decision != "APPROVE"
        or decision.reason_code != "ENGINEERING_ONLY"
        or decision.target != raw.get("challenger_strategy_id")
        or decision.evidence_cycle_id != cycle_id
        or assumption.assumption_id != raw.get("execution_assumption_id")
    ):
        raise _input("activation engineering decision or assumption join is invalid")
    return raw, decision, assumption


def _strategy_config(data: bytes, definition_id: str) -> Dict[str, Any]:
    try:
        import yaml

        class ExactLoader(yaml.SafeLoader):
            pass

        def mapping(loader: Any, node: Any, deep: bool = False) -> Dict[str, Any]:
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if type(key) is not str or key in result:
                    raise ValueError("duplicate or non-text YAML key")
                result[key] = loader.construct_object(value_node, deep=deep)
            return result

        ExactLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping,
        )
        text = data.decode("utf-8")
        documents = list(yaml.load_all(text, Loader=ExactLoader))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("strategy config is not strict safe YAML") from exc
    if len(documents) != 1 or type(documents[0]) is not dict:
        raise _input("strategy config must contain one mapping document")
    top = documents[0]
    strategy = top.get("strategy")
    if type(strategy) is not dict or set(strategy) != {"name", "params"}:
        raise _input("strategy config strategy fields are not exact")
    params = strategy.get("params")
    fields = {"topk", "n_drop", "buy_suggestion_factor", "sell_out_of_universe"}
    if type(params) is not dict or not fields.issubset(params):
        raise _input("strategy config relevant fields are incomplete")
    relevant = {field: params[field] for field in fields}
    if strategy.get("name") != "topk_dropout" or type(strategy.get("name")) is not str:
        raise _input("strategy name is not current topk_dropout")
    if (
        type(relevant["topk"]) is not int or relevant["topk"] <= 0
        or type(relevant["n_drop"]) is not int or relevant["n_drop"] < 0
        or type(relevant["buy_suggestion_factor"]) is not int
        or relevant["buy_suggestion_factor"] <= 0
        or type(relevant["sell_out_of_universe"]) is not bool
    ):
        raise _input("strategy config relevant value types are invalid")
    payload = {
        "schema_version": 1,
        "definition_id": definition_id,
        "strategy_name": "topk_dropout",
        **relevant,
        "production_buy_lot_size": 100,
        "cashflow_mode": "ZERO_EXTERNAL_CASHFLOW",
        "planning_price_rule": "CASH_CLOSE_WITH_0_9_1_1_ESTIMATES_V1",
        "buy_selection_rule": "FIRST_N_GENERATED_SUGGESTIONS_V1",
        "production_primitive_id": "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1",
    }
    try:
        from quantpits.research.intents import CurrentRuleIntentDefinition
        return CurrentRuleIntentDefinition.from_dict(payload)._payload()
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _input("strategy config cannot form the current intent definition") from exc


def _forward_raw(
    activation: Mapping[str, Any], decision: DecisionEvent, assumption: Any,
    intent: Mapping[str, Any], cutoff: str, seal_digest: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    false_claims = {
        "external_references_verified": False, "epoch_started": False,
        "prospective_claim": False, "promotion_capability": False,
    }
    common = {
        "schema_version": 1,
        "definition_kind": "SHADOW_FORWARD_STRATEGY_V1",
        "definition_claim": "TYPED_OWNER_DECLARATION_V1",
        "created_at": activation["created_at"],
        "effective_cycle": activation["effective_cycle"],
        "data_cutoff": cutoff,
        "evidence_cycle_id": activation["evidence_cycle_id"],
        "evidence_seal_digest": dict(seal_digest),
        "intent_definition_id": activation["intent_definition_id"],
        "fusion_definition": "PER_MODEL_CROSS_SECTIONAL_PERCENTILE_RANK_EQUAL_MEAN_V1",
        **false_claims,
    }
    omitted = activation["omitted_position"]
    champion_members = [copy.deepcopy(dict(row)) for row in references]
    challenger_members = []
    for row in references:
        if row["position"] != omitted:
            rebuilt = copy.deepcopy(dict(row))
            rebuilt["position"] = len(challenger_members)
            challenger_members.append(rebuilt)
    execution = assumption._payload()
    execution.update({
        "definition_kind": "SHADOW_FORWARD_EXECUTION_ASSUMPTION_V1",
        "definition_claim": "TYPED_OWNER_DECLARATION_V1",
        "created_at": activation["created_at"],
        "effective_cycle": activation["effective_cycle"],
        **false_claims,
    })
    champion = {
        **common, "strategy_id": activation["champion_strategy_id"], "role": "CHAMPION",
        "parent_strategy_id": None, "mutation_level": "CHAMPION_BASELINE",
        "selection_decision_id": None,
        "ranking_rule": "SEALED_PRODUCTION_FULL_RANKING_V1",
        "source_members": champion_members, "hypothesis": None,
    }
    challenger = {
        **common, "strategy_id": activation["challenger_strategy_id"], "role": "CHALLENGER",
        "parent_strategy_id": activation["champion_strategy_id"],
        "mutation_level": "L2_ENSEMBLE_COMPOSITION",
        "selection_decision_id": decision.decision_id,
        "ranking_rule": "FROZEN_MEMBER_EQUAL_FUSION_V1",
        "source_members": challenger_members,
        "hypothesis": activation["hypothesis"],
    }
    protocol = {
        "schema_version": 1, "definition_kind": "SHADOW_FORWARD_PROTOCOL_V1",
        "definition_claim": "TYPED_OWNER_DECLARATION_V1",
        "protocol_id": activation["protocol_id"],
        "definition_set_id": activation["definition_set_id"],
        "research_epoch_id": "PROSPECTIVE_SHADOW_V1",
        "created_at": activation["created_at"],
        "effective_cycle": activation["effective_cycle"],
        "champion_strategy_id": activation["champion_strategy_id"],
        "challenger_strategy_id": activation["challenger_strategy_id"],
        "execution_assumption_id": activation["execution_assumption_id"],
        "selection_decision": decision.to_dict(), "intent_definition": dict(intent),
        "bootstrap_rule": "MATCHED_CHAMPION_SNAPSHOT_ONCE_V1",
        "champion_ranking_rule": "SEALED_PRODUCTION_FULL_RANKING_V1",
        "challenger_ranking_rule": "FROZEN_MEMBER_EQUAL_FUSION_V1",
        "settlement_rule": "NEXT_OPEN_SELL_BEFORE_BUY_V1",
        "promotion_rule": "EXPLICIT_OWNER_DECISION_ONLY_V1", **false_claims,
    }
    return {
        "definition_set_id": activation["definition_set_id"], "protocol": protocol,
        "execution_assumption": execution, "champion_strategy": champion,
        "challenger_strategy": challenger,
    }


class ObservedForwardDefinitionCandidate:
    """Inspector-owned aggregate that revalidates before granting capability."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _AUTHORITY or args:
            raise ForwardObservationContractError("observed candidates are inspector-owned")
        expected = {
            "compiled_definitions", "reference_receipt", "reference_receipt_bytes",
            "reference_receipt_digest",
        }
        if set(kwargs) != expected:
            raise ForwardObservationContractError("candidate fields are not exact")
        object.__setattr__(self, "compiled_definitions", kwargs["compiled_definitions"])
        object.__setattr__(self, "_reference_receipt", MappingProxyType(copy.deepcopy(dict(kwargs["reference_receipt"]))))
        object.__setattr__(self, "_reference_receipt_bytes", kwargs["reference_receipt_bytes"])
        object.__setattr__(self, "_reference_receipt_digest", MappingProxyType(dict(kwargs["reference_receipt_digest"])))
        object.__setattr__(self, "_authority", _AUTHORITY)

    def _bind_original_observation(self) -> None:
        if self in _OBSERVED_BINDINGS:
            raise ForwardObservationContractError("candidate observation is already bound")
        compiled_raw = self._compiled_raw()
        _OBSERVED_BINDINGS[self] = (
            _authority_identity(self.compiled_definitions),
            canonical_json_bytes(compiled_raw),
            _authority_identity(self._reference_receipt),
            _authority_identity(self._reference_receipt_bytes),
            _authority_identity(self._reference_receipt_digest),
            self._reference_receipt_bytes,
            canonical_json_bytes(dict(self._reference_receipt_digest)),
        )

    def _compiled_raw(self) -> Dict[str, Any]:
        return {
            "definition_set_id": self.compiled_definitions.definition_set_id,
            "protocol": self.compiled_definitions.protocol.to_dict(),
            "execution_assumption": self.compiled_definitions.execution_assumption.to_dict(),
            "champion_strategy": self.compiled_definitions.champion.to_dict(),
            "challenger_strategy": self.compiled_definitions.challenger.to_dict(),
        }

    def _validated(self) -> Tuple[Any, Dict[str, Any]]:
        if type(self) is not ObservedForwardDefinitionCandidate or getattr(self, "_authority", None) is not _AUTHORITY:
            raise ForwardObservationContractError("candidate authority is absent")
        from quantpits.research.forward_definitions import (
            CompiledShadowForwardDefinitions, compile_shadow_forward_definitions,
        )
        if type(self.compiled_definitions) is not CompiledShadowForwardDefinitions:
            raise ForwardObservationContractError("candidate contains foreign compiled definitions")
        binding = _OBSERVED_BINDINGS.get(self)
        current_raw = self._compiled_raw()
        if binding is None or (
            _authority_identity(self.compiled_definitions) != binding[0]
            or canonical_json_bytes(current_raw) != binding[1]
            or _authority_identity(self._reference_receipt) != binding[2]
            or _authority_identity(self._reference_receipt_bytes) != binding[3]
            or _authority_identity(self._reference_receipt_digest) != binding[4]
            or self._reference_receipt_bytes != binding[5]
            or canonical_json_bytes(dict(self._reference_receipt_digest)) != binding[6]
        ):
            raise ForwardObservationContractError(
                "candidate differs from its original inspector observation",
            )
        try:
            fresh = compile_shadow_forward_definitions(current_raw)
        except _PROCESS_CONTROL:
            raise
        except Exception as exc:
            raise ForwardObservationContractError(
                "candidate compiled payload no longer has authority",
            ) from exc
        receipt = copy.deepcopy(dict(self._reference_receipt))
        if set(receipt) != set(_RECEIPT_FIELDS):
            raise ForwardObservationContractError("reference receipt fields are invalid")
        receipt_bytes = canonical_json_bytes(receipt)
        if type(self._reference_receipt_bytes) is not bytes or receipt_bytes != self._reference_receipt_bytes:
            raise ForwardObservationContractError("reference receipt bytes changed")
        digest = TypedDigest.canonical(receipt)
        if digest.to_dict() != dict(self._reference_receipt_digest):
            raise ForwardObservationContractError("reference receipt digest changed")
        champion = fresh.champion.to_dict()
        challenger = fresh.challenger.to_dict()
        champion_members = champion["source_members"]
        challenger_keys = tuple(
            (row["source_id"], row["model_artifact_digest"])
            for row in challenger["source_members"]
        )
        omitted = tuple(
            index for index, row in enumerate(champion_members)
            if (row["source_id"], row["model_artifact_digest"]) not in challenger_keys
        )
        for field in (
            "evidence_seal_digest", "evidence_manifest_digest",
            "strategy_config_digest", "activation_input_digest",
        ):
            expected_domain = "raw_bytes"
            try:
                _typed_digest(receipt[field], "receipt.%s" % field, expected_domain)
            except ForwardObservationInputError as exc:
                raise ForwardObservationContractError("reference receipt digest is invalid") from exc
        try:
            _typed_digest(
                receipt["source_reference_inventory_digest"],
                "receipt.source_reference_inventory_digest", "file_inventory",
            )
            _typed_digest(
                receipt["compiled_request_digest"],
                "receipt.compiled_request_digest", "canonical_json",
            )
        except ForwardObservationInputError as exc:
            raise ForwardObservationContractError("reference receipt aggregate digest is invalid") from exc
        if (
            receipt["schema_version"] != 1 or type(receipt["schema_version"]) is not int
            or receipt["observation_claim"] != "SEALED_REFERENCE_JOIN_VERIFIED_V1"
            or receipt["purpose"] != "ENGINEERING_VALIDATION"
            or receipt["definition_set_id"] != fresh.definition_set_id
            or receipt["evidence_cycle_id"] != champion["evidence_cycle_id"]
            or receipt["evidence_seal_digest"] != champion["evidence_seal_digest"]
            or receipt["source_reference_inventory_digest"] != TypedDigest.canonical(
                champion_members, "file_inventory",
            ).to_dict()
            or type(receipt["selected_omitted_position"]) is not int
            or omitted != (receipt["selected_omitted_position"],)
            or receipt["compiled_request_digest"] != dict(fresh.request_digest)
        ):
            raise ForwardObservationContractError("compiled request no longer matches receipt")
        if (
            receipt["sealed_reference_join_verified"] is not True
            or receipt["sealed_artifact_inventory_verified"] is not True
            or type(receipt["source_reference_count"]) is not int
            or receipt["source_reference_count"] != 4
            or type(receipt["all_artifact_member_bytes_embedded"]) is not bool
            or any(receipt[field] is not False for field in (
                "live_artifact_member_bytes_reverified", "external_definition_claim",
                "publication_capability", "epoch_started", "prospective_claim",
                "promotion_capability",
            ))
        ):
            raise ForwardObservationContractError("reference receipt authority claims are invalid")
        return fresh, receipt

    def __copy__(self) -> Any:
        raise ForwardObservationContractError("candidate replay is not authoritative")

    def __deepcopy__(self, _memo: Any) -> Any:
        raise ForwardObservationContractError("candidate replay is not authoritative")

    def __reduce__(self) -> Any:
        raise ForwardObservationContractError("candidate replay is not authoritative")

    @property
    def definition_set_id(self) -> str:
        fresh, _ = self._validated()
        return fresh.definition_set_id

    @property
    def evidence_cycle_id(self) -> str:
        _, receipt = self._validated()
        return receipt["evidence_cycle_id"]

    @property
    def selected_omitted_position(self) -> int:
        _, receipt = self._validated()
        return receipt["selected_omitted_position"]

    @property
    def reference_receipt(self) -> Mapping[str, Any]:
        _, receipt = self._validated()
        return MappingProxyType(receipt)

    @property
    def reference_receipt_digest(self) -> Mapping[str, Any]:
        self._validated()
        return MappingProxyType(dict(self._reference_receipt_digest))

    @property
    def sealed_reference_join_verified(self) -> bool:
        self._validated()
        return True

    @property
    def compiled_capability(self) -> bool:
        fresh, _ = self._validated()
        return fresh.compiled_capability

    @property
    def publication_capability(self) -> bool:
        self._validated()
        return False

    @property
    def epoch_started(self) -> bool:
        self._validated()
        return False

    @property
    def prospective_claim(self) -> bool:
        self._validated()
        return False

    @property
    def promotion_capability(self) -> bool:
        self._validated()
        return False

    def to_store_request(self) -> Any:
        fresh, _ = self._validated()
        return fresh.to_store_request()

    def to_safe_summary_dict(self) -> Dict[str, Any]:
        fresh, receipt = self._validated()
        compiled = fresh.to_summary_dict()
        return {
            "status": "complete", "schema_version": 1,
            "definition_set_id": fresh.definition_set_id,
            "protocol_id": compiled["protocol_id"],
            "execution_assumption_id": compiled["execution_assumption_id"],
            "champion_strategy_id": compiled["champion_strategy_id"],
            "challenger_strategy_id": compiled["challenger_strategy_id"],
            "evidence_cycle_id": receipt["evidence_cycle_id"],
            "selected_omitted_position": receipt["selected_omitted_position"],
            "evidence_seal_digest": receipt["evidence_seal_digest"],
            "evidence_manifest_digest": receipt["evidence_manifest_digest"],
            "source_reference_inventory_digest": receipt["source_reference_inventory_digest"],
            "compiled_request_digest": receipt["compiled_request_digest"],
            "reference_receipt_digest": dict(self._reference_receipt_digest),
            "sealed_reference_join_verified": True, "compiled_capability": True,
            "publication_capability": False, "epoch_started": False,
            "prospective_claim": False, "promotion_capability": False,
        }


def _observe_shadow_forward_definition_candidate(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
) -> ObservedForwardDefinitionCandidate:
    """Observe, join, and compile one engineering-only definition candidate."""
    cycle_id = _calendar_date(evidence_cycle_id, "evidence_cycle_id")
    root = _physical_root(workspace_root)
    if type(activation_path) is not Path:
        try:
            activation_path = Path(activation_path)
        except Exception as exc:
            raise ForwardObservationContractError("activation_path must be a path") from exc
    try:
        activation_logical = activation_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise _input("activation is outside workspace") from exc
    cycle_path = root / "data" / "evidence" / "v1" / "cycles" / cycle_id
    cycles_path = cycle_path.parent
    config_raw_path = root / "config" / "strategy_config.yaml"
    watched = tuple(value for value in (
        activation_logical,
        Path(activation_logical).parent.as_posix(),
        "config",
        config_raw_path.relative_to(root).as_posix(),
        cycles_path.relative_to(root).as_posix(),
        cycle_path.relative_to(root).as_posix(),
    ) if value not in {"", "."})
    try:
        with _mutation_guard(root, watched) as observer:
            if not observer.supported:
                raise _input("source mutation observation is unavailable")
            activation_file = _contained(root, activation_path, regular=True)
            config_path = _contained(root, config_raw_path, regular=True)
            cycles_physical = _contained(root, cycles_path)
            cycle_physical = _contained(root, cycle_path)
            authority_directories = tuple(dict.fromkeys((
                root, activation_file.parent, config_path.parent,
                cycles_physical, cycle_physical,
            )))
            authority_before = {
                path: _directory_identity(path) for path in authority_directories
            }
            activation_data, activation_identity = _read_regular(
                activation_file, maximum=_ACTIVATION_LIMIT, private=True,
            )
            activation, decision, assumption = _activation(activation_data, cycle_id)
            (
                manifest, manifest_data, seal_data, all_embedded, source_ids,
                references, _positions,
            ) = _verify_bundle(root, cycle_id)
            config_data, config_identity = _read_regular(config_path, maximum=_CONFIG_LIMIT)
            intent = _strategy_config(config_data, activation["intent_definition_id"])
            cutoff = manifest["data_identity"]["qlib_materialization_identity"]["calendar_cutoff"]
            seal_raw = TypedDigest.raw(seal_data).to_dict()
            raw = _forward_raw(
                activation, decision, assumption, intent, cutoff, seal_raw, references,
            )
            try:
                from quantpits.research.forward_definitions import compile_shadow_forward_definitions
                compiled = compile_shadow_forward_definitions(raw)
            except _PROCESS_CONTROL:
                raise
            except Exception as exc:
                raise ForwardObservationContractError("C1A compilation rejected the observed join") from exc
            source_inventory = [
                {
                    "position": row["position"], "source_id": row["source_id"],
                    "model_artifact_digest": row["model_artifact_digest"],
                }
                for row in references
            ]
            receipt = {
                "schema_version": 1,
                "observation_claim": "SEALED_REFERENCE_JOIN_VERIFIED_V1",
                "purpose": "ENGINEERING_VALIDATION",
                "definition_set_id": activation["definition_set_id"],
                "evidence_cycle_id": cycle_id,
                "evidence_seal_digest": seal_raw,
                "evidence_manifest_digest": TypedDigest.raw(manifest_data).to_dict(),
                "strategy_config_digest": TypedDigest.raw(config_data).to_dict(),
                "activation_input_digest": TypedDigest.raw(activation_data).to_dict(),
                "source_reference_count": len(source_ids),
                "source_reference_inventory_digest": TypedDigest.canonical(
                    source_inventory, "file_inventory",
                ).to_dict(),
                "selected_omitted_position": activation["omitted_position"],
                "compiled_request_digest": dict(compiled.request_digest),
                "sealed_reference_join_verified": True,
                "sealed_artifact_inventory_verified": True,
                "all_artifact_member_bytes_embedded": all_embedded,
                "live_artifact_member_bytes_reverified": False,
                "external_definition_claim": False,
                "publication_capability": False,
                "epoch_started": False,
                "prospective_claim": False,
                "promotion_capability": False,
            }
            # Re-observe the two caller-controlled files and every enclosing authority.
            _, activation_after = _read_regular(
                activation_file, expected_size=len(activation_data), private=True,
            )
            _, config_after = _read_regular(config_path, expected_size=len(config_data))
            if (
                activation_after != activation_identity
                or config_after != config_identity
                or any(
                    _directory_identity(path) != identity
                    for path, identity in authority_before.items()
                )
                or observer.mutated()
            ):
                raise _input("observation source continuity was lost", sources=source_ids)
    except _PROCESS_CONTROL:
        raise
    receipt_bytes = canonical_json_bytes(receipt)
    digest = TypedDigest.canonical(receipt).to_dict()
    candidate = ObservedForwardDefinitionCandidate(
        _authority=_AUTHORITY, compiled_definitions=compiled,
        reference_receipt=receipt, reference_receipt_bytes=receipt_bytes,
        reference_receipt_digest=digest,
    )
    candidate._bind_original_observation()
    return candidate


def observe_shadow_forward_definition_candidate(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
) -> ObservedForwardDefinitionCandidate:
    """Fail closed with typed errors while preserving process-control exceptions."""
    try:
        return _observe_shadow_forward_definition_candidate(
            workspace_root, evidence_cycle_id, activation_path,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardObservationContractError:
        raise
    except Exception as exc:
        raise _input("observation failed closed") from exc


__all__ = [
    "ForwardObservationContractError", "ForwardObservationInputError",
    "ObservedForwardDefinitionCandidate", "observe_shadow_forward_definition_candidate",
]
