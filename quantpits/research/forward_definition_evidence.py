"""Create-only durable evidence for one already-adopted forward definition."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import weakref
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes


AUTHORIZATION_ACTION = "PUBLISH_ONE_FORWARD_DEFINITION_EVIDENCE_RECORD_V1"
FORWARD_EVIDENCE_AUTHORIZATION_ACTION = AUTHORIZATION_ACTION
FRESH_EVIDENCE_AUTHORIZATION_ACTION = (
    "PUBLISH_ONE_FRESH_CHAMPION_SEGMENT_DEFINITION_EVIDENCE_RECORD_V1"
)
EVIDENCE_KIND = "FORWARD_DEFINITION_EVIDENCE_V1"
STORAGE_CLAIM = "CREATE_ONLY_EXACT_BYTES"
MEMBER_PATHS = ("reference_receipt.json", "definition_store_receipt.json")
MANIFEST_NAME = "evidence_manifest.json"
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_RECEIPT_AUTHORITY = object()
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLAN_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_RESULT_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_RECEIPT_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_OUTER_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_MAX_MEMBER_BYTES = 1024 * 1024
_MAX_INVENTORY_MEMBERS = 200000
_MAX_INVENTORY_BYTES = 256 * 1024 * 1024
_FRESH_MAX_INVENTORY_BYTES = 64 * 1024 * 1024


class ForwardDefinitionEvidenceContractError(ValueError):
    """A caller-controlled evidence value violates the frozen contract."""

    reason_code = "CONTRACT_INVALID"


class ForwardDefinitionEvidenceInputError(ForwardDefinitionEvidenceContractError):
    """The requested evidence inputs cannot be compared safely."""

    reason_code = "INPUT_INCOMPARABLE"


class _ObservationUncertain(Exception):
    pass


class _ExistingConflict(Exception):
    def __init__(self, fingerprint: str) -> None:
        super().__init__(fingerprint)
        self.fingerprint = fingerprint


def _input(message: str) -> ForwardDefinitionEvidenceInputError:
    return ForwardDefinitionEvidenceInputError(message)


def _canonical(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ForwardDefinitionEvidenceContractError("value is not canonical JSON") from exc


def _raw_digest(data: bytes) -> Dict[str, Any]:
    return TypedDigest.raw(data).to_dict()


def _typed_digest(value: Any, name: str, domain: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        raise ForwardDefinitionEvidenceContractError("%s fields are invalid" % name)
    try:
        digest = TypedDigest(**value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ForwardDefinitionEvidenceContractError("%s is invalid" % name) from exc
    if digest.domain != domain:
        raise ForwardDefinitionEvidenceContractError("%s domain is invalid" % name)
    return digest.to_dict()


def _definition_id(value: Any) -> str:
    if (
        type(value) is not str or value != value.strip() or value in {".", ".."}
        or _ID_RE.fullmatch(value) is None or "/" in value or "\\" in value
    ):
        raise ForwardDefinitionEvidenceContractError("definition_set_id is invalid")
    return value


def _cycle(value: Any) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise ForwardDefinitionEvidenceContractError("evidence_cycle_id is invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ForwardDefinitionEvidenceContractError("evidence_cycle_id is invalid") from exc
    if parsed.isoformat() != value:
        raise ForwardDefinitionEvidenceContractError("evidence_cycle_id is invalid")
    return value


def _path(value: Any, name: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise ForwardDefinitionEvidenceContractError("%s must be a path" % name)
    path = Path(value)
    if not path.is_absolute():
        raise ForwardDefinitionEvidenceContractError("%s must be absolute" % name)
    try:
        resolved = path.resolve(strict=True)
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _input("%s is unavailable" % name) from exc
    if resolved != path:
        raise _input("%s must be a physical canonical path" % name)
    return path


def _directory_identity(path: Path, *, private: bool = False) -> Tuple[int, int, int, int, int]:
    try:
        info = os.lstat(str(path))
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _input("evidence directory is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _input("evidence directory is not physical")
    if private and stat.S_IMODE(info.st_mode) != 0o700:
        raise _input("private evidence directory mode must be 0700")
    return info.st_dev, info.st_ino, info.st_mode, 0, 0


def _continuity(path: Path) -> Tuple[int, int, int, int, int]:
    info = os.lstat(str(path))
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _input("evidence directory is not physical")
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_ctime_ns


def _file_identity(path: Path) -> Tuple[int, int, int, int, int, int, int]:
    info = os.lstat(str(path))
    if (
        stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise _input("activation must be a private single-link regular file")
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _formal_inputs(
    workspace_root: Any, activation_path: Any, definition_store_root: Any,
    evidence_store_root: Any,
) -> Tuple[Path, Path, Path, Path, Dict[str, Tuple[int, ...]]]:
    root = _path(workspace_root, "workspace_root")
    activation = _path(activation_path, "activation_path")
    definitions = _path(definition_store_root, "definition_store_root")
    evidence = _path(evidence_store_root, "evidence_store_root")
    research = root / "research"
    shadow = research / "shadow_v1"
    if (
        activation.parent != shadow / "activations"
        or definitions != shadow / "definitions"
        or evidence != shadow / "definition_evidence"
    ):
        raise _input("evidence paths do not match the formal private layout")
    identities: Dict[str, Tuple[int, ...]] = {
        "workspace": _directory_identity(root),
        "workspace_continuity": _continuity(root),
        "workspace_parent_continuity": _continuity(root.parent),
        "research": _directory_identity(research, private=True),
        "research_continuity": _continuity(research),
        "shadow": _directory_identity(shadow, private=True),
        "shadow_continuity": _continuity(shadow),
        "activations": _directory_identity(activation.parent, private=True),
        "activations_continuity": _continuity(activation.parent),
        "definitions": _directory_identity(definitions, private=True),
        "definitions_continuity": _continuity(definitions),
        "evidence": _directory_identity(evidence, private=True),
        "evidence_continuity": _continuity(evidence),
        "activation": _file_identity(activation),
    }
    return root, activation, definitions, evidence, identities


def _same_identities(
    root: Path, activation: Path, definitions: Path, evidence: Path,
    expected: Mapping[str, Tuple[int, ...]], *, evidence_did_write: bool = False,
) -> bool:
    try:
        _, _, _, _, current = _formal_inputs(root, activation, definitions, evidence)
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        return False
    before = dict(expected)
    if evidence_did_write:
        before.pop("evidence_continuity", None)
        current.pop("evidence_continuity", None)
    return before == current


def _workspace_inventory(
    root: Path, excluded: Path, *, byte_budget: Optional[int] = None,
) -> Tuple[Tuple[Any, ...], ...]:
    rows = []
    remaining = _MAX_INVENTORY_BYTES if byte_budget is None else byte_budget
    for path in sorted(root.rglob("*")):
        if path == excluded or excluded in path.parents:
            continue
        if len(rows) >= _MAX_INVENTORY_MEMBERS:
            raise _input("protected workspace inventory exceeds its member budget")
        info = os.lstat(str(path))
        logical = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            rows.append((logical, "symlink", info.st_mode, info.st_ctime_ns))
        elif stat.S_ISDIR(info.st_mode):
            if path.name == "definition_evidence" and path.parent.name == "shadow_v1":
                rows.append((logical, "directory", info.st_dev, info.st_ino, info.st_mode))
            else:
                rows.append((
                    logical, "directory", info.st_dev, info.st_ino, info.st_mode,
                    info.st_nlink, info.st_ctime_ns,
                ))
        elif stat.S_ISREG(info.st_mode):
            if info.st_size > remaining:
                raise _input("protected workspace inventory exceeds its byte budget")
            descriptor = os.open(
                str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            try:
                opened = os.fstat(descriptor)
                before_identity = (
                    info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                    info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                )
                opened_identity = (
                    opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink,
                    opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns,
                )
                if before_identity != opened_identity or not stat.S_ISREG(opened.st_mode):
                    raise _input("protected workspace member changed while opening")
                chunks = []
                remaining_member = opened.st_size + 1
                while remaining_member:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining_member))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining_member -= len(chunk)
                opened_after = os.fstat(descriptor)
            finally:
                _close(descriptor, suppress=False)
            data = b"".join(chunks)
            remaining -= len(data)
            after = os.lstat(str(path))
            after_identity = (
                after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
                after.st_size, after.st_mtime_ns, after.st_ctime_ns,
            )
            if (
                len(data) != info.st_size
                or before_identity != (
                    opened_after.st_dev, opened_after.st_ino, opened_after.st_mode,
                    opened_after.st_nlink, opened_after.st_size,
                    opened_after.st_mtime_ns, opened_after.st_ctime_ns,
                )
                or before_identity != after_identity
            ):
                raise _input("protected workspace member changed while reading")
            rows.append((
                logical, "regular", info.st_dev, info.st_ino, info.st_mode,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                hashlib.sha256(data).hexdigest(),
            ))
        else:
            rows.append((logical, "special", info.st_dev, info.st_ino, info.st_mode))
    return tuple(rows)


def _split_formal_inputs(
    production_workspace_root: Any, research_workspace_root: Any,
    activation_path: Any, definition_store_root: Any, evidence_store_root: Any,
) -> Tuple[Path, Path, Path, Path, Path, Dict[str, Tuple[int, ...]]]:
    """Freeze physically separated Production and formal Research roots."""
    production = _path(production_workspace_root, "production_workspace_root")
    research, activation, definitions, evidence, research_identities = _formal_inputs(
        research_workspace_root, activation_path, definition_store_root,
        evidence_store_root,
    )
    if (
        production == research
        or production in research.parents
        or research in production.parents
    ):
        raise _input("Production and Research roots must be physically separate")
    identities = {
        "production_workspace": _directory_identity(production),
        "production_workspace_continuity": _continuity(production),
        "production_parent_continuity": _continuity(production.parent),
    }
    identities.update({
        "research_" + name: identity
        for name, identity in research_identities.items()
    })
    return production, research, activation, definitions, evidence, identities


def _same_split_identities(
    production: Path, research: Path, activation: Path, definitions: Path,
    evidence: Path, expected: Mapping[str, Tuple[int, ...]],
    *, evidence_did_write: bool = False,
) -> bool:
    try:
        production_current = {
            "production_workspace": _directory_identity(production),
            "production_workspace_continuity": _continuity(production),
            "production_parent_continuity": _continuity(production.parent),
        }
    except _PROCESS_CONTROL:
        raise
    except (OSError, ForwardDefinitionEvidenceContractError):
        return False
    production_expected = {
        name: identity for name, identity in expected.items()
        if name.startswith("production_")
    }
    research_expected = {
        name[len("research_"):]: identity
        for name, identity in expected.items()
        if name.startswith("research_")
    }
    return (
        production_current == production_expected
        and _same_identities(
            research, activation, definitions, evidence, research_expected,
            evidence_did_write=evidence_did_write,
        )
    )


def _close(descriptor: int, *, suppress: bool) -> None:
    active = sys.exc_info()[1]
    try:
        os.close(descriptor)
    except _PROCESS_CONTROL:
        if isinstance(active, _PROCESS_CONTROL):
            return
        raise
    except OSError as exc:
        if isinstance(active, _PROCESS_CONTROL) or suppress:
            return
        raise _ObservationUncertain("descriptor cleanup failed") from exc


def _open_root(path: Path) -> Tuple[int, Tuple[int, int, int, int, int]]:
    before = _directory_identity(path, private=True)
    try:
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise _input("evidence store cannot be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        opened = (info.st_dev, info.st_ino, info.st_mode, 0, 0)
        if opened != before or _directory_identity(path, private=True) != before:
            raise _input("evidence store identity changed while opening")
    except BaseException:
        _close(descriptor, suppress=True)
        raise
    return descriptor, before


def _write_exclusive(descriptor: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    member = os.open(name, flags, 0o600, dir_fd=descriptor)
    try:
        os.fchmod(member, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(member, view)
            if written <= 0:
                raise OSError("short evidence write")
            view = view[written:]
        os.fsync(member)
    finally:
        _close(member, suppress=False)


def _read_exact(descriptor: int, name: str, expected: bytes, *, conflict: bool) -> bytes:
    try:
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        metadata = (
            stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
            and before.st_nlink == 1 and stat.S_IMODE(before.st_mode) == 0o600
            and before.st_size == len(expected)
        )
        if not metadata:
            if conflict:
                raise _ExistingConflict(hashlib.sha256(_canonical({
                    "class": "MEMBER_METADATA", "name": name,
                    "mode": before.st_mode, "nlink": before.st_nlink,
                    "size": before.st_size,
                })).hexdigest())
            raise _ObservationUncertain("evidence member metadata differs")
        member = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
        try:
            opened = os.fstat(member)
            chunks = []
            remaining = len(expected) + 1
            while remaining:
                chunk = os.read(member, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            opened_after = os.fstat(member)
        finally:
            _close(member, suppress=False)
        after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except _PROCESS_CONTROL:
        raise
    except (_ExistingConflict, _ObservationUncertain):
        raise
    except OSError as exc:
        raise _ObservationUncertain("evidence member observation failed") from exc
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )
    if len({identity(before), identity(opened), identity(opened_after), identity(after)}) != 1:
        raise _ObservationUncertain("evidence member identity changed")
    data = b"".join(chunks)
    if data != expected:
        if conflict:
            raise _ExistingConflict(hashlib.sha256(_canonical({
                "class": "MEMBER_BYTES", "name": name,
                "expected": _raw_digest(expected), "observed": _raw_digest(data),
            })).hexdigest())
        raise _ObservationUncertain("evidence member bytes differ")
    return data


class _EvidenceRequest:
    def __init__(
        self, definition_set_id: str, evidence_cycle_id: str,
        reference_bytes: bytes, definition_receipt_bytes: bytes,
        reference_receipt_digest: Mapping[str, Any],
        definition_request_digest: Mapping[str, Any],
        definition_manifest_digest: Mapping[str, Any],
        definition_operation_id: str,
    ) -> None:
        self.definition_set_id = _definition_id(definition_set_id)
        self.evidence_cycle_id = _cycle(evidence_cycle_id)
        if any(type(value) is not bytes or not value or len(value) > _MAX_MEMBER_BYTES for value in (
            reference_bytes, definition_receipt_bytes,
        )):
            raise ForwardDefinitionEvidenceContractError("evidence members are invalid")
        self.members = (
            (MEMBER_PATHS[0], reference_bytes),
            (MEMBER_PATHS[1], definition_receipt_bytes),
        )
        self.reference_receipt_digest = MappingProxyType(_typed_digest(
            dict(reference_receipt_digest), "reference_receipt_digest", "canonical_json",
        ))
        try:
            reference_value = json.loads(reference_bytes.decode("utf-8"))
        except Exception as exc:
            raise ForwardDefinitionEvidenceContractError("reference receipt bytes are invalid") from exc
        if (
            type(reference_value) is not dict
            or _canonical(reference_value) != reference_bytes
            or TypedDigest.canonical(reference_value).to_dict()
            != dict(self.reference_receipt_digest)
        ):
            raise ForwardDefinitionEvidenceContractError("reference receipt digest is inconsistent")
        self.definition_request_digest = MappingProxyType(_typed_digest(
            dict(definition_request_digest), "definition_request_digest", "canonical_json",
        ))
        self.definition_manifest_digest = MappingProxyType(_typed_digest(
            dict(definition_manifest_digest), "definition_manifest_digest", "raw_bytes",
        ))
        if (
            type(definition_operation_id) is not str or len(definition_operation_id) != 64
            or any(character not in "0123456789abcdef" for character in definition_operation_id)
        ):
            raise ForwardDefinitionEvidenceContractError("definition operation ID is invalid")
        self.definition_operation_id = definition_operation_id
        payload = {
            "schema_version": 1, "evidence_kind": EVIDENCE_KIND,
            "definition_set_id": self.definition_set_id,
            "evidence_cycle_id": self.evidence_cycle_id,
            "definition_request_digest": dict(self.definition_request_digest),
            "members": [
                {"logical_path": name, "size_bytes": len(data), "digest": _raw_digest(data)}
                for name, data in self.members
            ],
        }
        self.request_digest = MappingProxyType(TypedDigest.canonical(payload).to_dict())


def _revalidate_evidence_request(value: Any) -> _EvidenceRequest:
    if type(value) is not _EvidenceRequest:
        raise ForwardDefinitionEvidenceContractError("evidence request is foreign")
    try:
        rebuilt = _EvidenceRequest(
            value.definition_set_id, value.evidence_cycle_id,
            value.members[0][1], value.members[1][1],
            value.reference_receipt_digest, value.definition_request_digest,
            value.definition_manifest_digest, value.definition_operation_id,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        raise
    except Exception as exc:
        raise ForwardDefinitionEvidenceContractError("evidence request cannot be revalidated") from exc
    if (
        tuple(name for name, _data in value.members) != MEMBER_PATHS
        or dict(value.request_digest) != dict(rebuilt.request_digest)
    ):
        raise ForwardDefinitionEvidenceContractError("evidence request authority is inconsistent")
    return rebuilt


def _manifest_bytes(request: _EvidenceRequest) -> bytes:
    return _canonical({
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "storage_claim": STORAGE_CLAIM,
        "definition_set_id": request.definition_set_id,
        "evidence_cycle_id": request.evidence_cycle_id,
        "evidence_request_digest": dict(request.request_digest),
        "definition_request_digest": dict(request.definition_request_digest),
        "reference_receipt_digest": dict(request.reference_receipt_digest),
        "definition_store_receipt_digest": _raw_digest(request.members[1][1]),
        "definition_manifest_digest": dict(request.definition_manifest_digest),
        "definition_store_operation_id": request.definition_operation_id,
        "definition_store_status": "ADOPTED",
        "definition_store_did_write": False,
        "member_count": len(MEMBER_PATHS),
        "members": [
            {"logical_path": name, "size_bytes": len(data), "digest": _raw_digest(data)}
            for name, data in request.members
        ],
        "original_commit_receipt_recorded": False,
        "portfolio_bootstrap_claim": False,
        "prospective_claim": False,
        "promotion_capability": False,
    })


def _operation_id(
    request: _EvidenceRequest, identity: Tuple[int, int, int, int, int],
) -> str:
    return hashlib.sha256(_canonical({
        "domain": "FORWARD_DEFINITION_EVIDENCE_PUBLICATION_V1",
        "definition_set_id": request.definition_set_id,
        "evidence_request_digest": dict(request.request_digest),
        "evidence_store_root_identity_before": list(identity),
    })).hexdigest()


class ForwardDefinitionEvidenceStoreReceipt:
    """Writer-owned receipt for the create-only evidence store."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _RECEIPT_AUTHORITY:
            raise ForwardDefinitionEvidenceContractError("evidence store receipts are writer-owned")
        expected = {
            "operation_id", "status", "reason_code", "did_write", "definition_set_id",
            "request_digest", "manifest_digest", "member_count", "store_root_identity_before",
            "store_root_identity_after", "bundle_root_identity",
        }
        if set(kwargs) != expected:
            raise ForwardDefinitionEvidenceContractError("evidence receipt fields are not exact")
        for name, value in kwargs.items():
            if name in {"request_digest", "manifest_digest"} and value is not None:
                value = MappingProxyType(dict(value))
            object.__setattr__(self, name, value)
        reasons = {
            "COMMITTED": "COMMITTED", "ADOPTED": "EXACT_EVIDENCE_ALREADY_PRESENT",
            "CONFLICT": "EVIDENCE_SET_ID_CONFLICT",
            "UNCERTAIN": "EVIDENCE_POSTCONDITION_UNCERTAIN",
        }
        valid_identity = lambda value: (
            type(value) is tuple and len(value) == 5 and all(type(item) is int for item in value)
        )
        if (
            type(self.operation_id) is not str or len(self.operation_id) != 64
            or any(character not in "0123456789abcdef" for character in self.operation_id)
            or self.status not in reasons or self.reason_code != reasons[self.status]
            or type(self.did_write) is not bool or _definition_id(self.definition_set_id) != self.definition_set_id
            or _typed_digest(dict(self.request_digest), "evidence_request_digest", "canonical_json")
            != dict(self.request_digest)
            or type(self.member_count) is not int or not valid_identity(self.store_root_identity_before)
        ):
            raise ForwardDefinitionEvidenceContractError("evidence receipt identity is invalid")
        expected_operation = hashlib.sha256(_canonical({
            "domain": "FORWARD_DEFINITION_EVIDENCE_PUBLICATION_V1",
            "definition_set_id": self.definition_set_id,
            "evidence_request_digest": dict(self.request_digest),
            "evidence_store_root_identity_before": list(self.store_root_identity_before),
        })).hexdigest()
        if self.operation_id != expected_operation:
            raise ForwardDefinitionEvidenceContractError("evidence operation identity is inconsistent")
        manifest = None if self.manifest_digest is None else dict(self.manifest_digest)
        if manifest is not None:
            _typed_digest(manifest, "evidence_manifest_digest", "raw_bytes")
        after_valid = self.store_root_identity_after is None or valid_identity(self.store_root_identity_after)
        bundle_valid = self.bundle_root_identity is None or valid_identity(self.bundle_root_identity)
        positive = self.status in {"COMMITTED", "ADOPTED"}
        if positive:
            valid = (
                self.did_write is (self.status == "COMMITTED") and manifest is not None
                and self.member_count == len(MEMBER_PATHS) and bundle_valid
                and self.bundle_root_identity is not None and self.store_root_identity_after == self.store_root_identity_before
            )
        elif self.status == "CONFLICT":
            valid = (
                self.did_write is False and manifest is None and self.member_count == 0
                and self.bundle_root_identity is None and self.store_root_identity_after == self.store_root_identity_before
            )
        else:
            valid = (
                0 <= self.member_count <= len(MEMBER_PATHS) and manifest is None
                and self.bundle_root_identity is None and after_valid
                and (self.did_write or self.member_count == 0)
            )
        if not valid:
            raise ForwardDefinitionEvidenceContractError("evidence receipt cross-fields are inconsistent")
        _RECEIPT_BINDINGS[self] = _canonical(self._render())

    def _validate(self) -> None:
        if (
            type(self) is not ForwardDefinitionEvidenceStoreReceipt
            or _RECEIPT_BINDINGS.get(self) != _canonical(self._render())
        ):
            raise ForwardDefinitionEvidenceContractError("evidence receipt authority is absent")

    @property
    def durable_record_capability(self) -> bool:
        self._validate()
        return self.status in {"COMMITTED", "ADOPTED"}

    def _render(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id, "status": self.status,
            "reason_code": self.reason_code, "did_write": self.did_write,
            "definition_set_id": self.definition_set_id,
            "evidence_request_digest": dict(self.request_digest),
            "manifest_digest": None if self.manifest_digest is None else dict(self.manifest_digest),
            "member_count": self.member_count,
            "evidence_store_root_identity_before": list(self.store_root_identity_before),
            "evidence_store_root_identity_after": (
                None if self.store_root_identity_after is None else list(self.store_root_identity_after)
            ),
            "evidence_bundle_root_identity": (
                None if self.bundle_root_identity is None else list(self.bundle_root_identity)
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return self._render()


def _store_receipt(
    request: _EvidenceRequest, operation: str, status: str, did_write: bool,
    count: int, before: Tuple[int, int, int, int, int],
    after: Optional[Tuple[int, int, int, int, int]],
    bundle: Optional[Tuple[int, int, int, int, int]] = None,
    manifest: Optional[bytes] = None,
) -> ForwardDefinitionEvidenceStoreReceipt:
    reasons = {
        "COMMITTED": "COMMITTED", "ADOPTED": "EXACT_EVIDENCE_ALREADY_PRESENT",
        "CONFLICT": "EVIDENCE_SET_ID_CONFLICT",
        "UNCERTAIN": "EVIDENCE_POSTCONDITION_UNCERTAIN",
    }
    return ForwardDefinitionEvidenceStoreReceipt(
        _authority=_RECEIPT_AUTHORITY, operation_id=operation, status=status,
        reason_code=reasons[status], did_write=did_write,
        definition_set_id=request.definition_set_id, request_digest=request.request_digest,
        manifest_digest=None if manifest is None else _raw_digest(manifest),
        member_count=count, store_root_identity_before=before,
        store_root_identity_after=after, bundle_root_identity=bundle,
    )


def _try_root_identity(path: Path) -> Optional[Tuple[int, int, int, int, int]]:
    try:
        return _directory_identity(path, private=True)
    except (OSError, ForwardDefinitionEvidenceContractError):
        return None


class _CreateOnlyEvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def publish(self, request: _EvidenceRequest) -> ForwardDefinitionEvidenceStoreReceipt:
        request = _revalidate_evidence_request(request)
        root_fd, before = _open_root(self.root)
        operation = _operation_id(request, before)
        target = self.root / request.definition_set_id
        bundle_fd: Optional[int] = None
        prefix = 0
        try:
            try:
                os.mkdir(request.definition_set_id, 0o700, dir_fd=root_fd)
            except FileExistsError:
                return self._classify_existing(request, root_fd, before, operation)
            except _PROCESS_CONTROL:
                raise
            except OSError as exc:
                raise _input("evidence target could not be created") from exc
            try:
                if _directory_identity(self.root, private=True) != before:
                    raise _ObservationUncertain("evidence root changed after create")
                target_info = os.stat(request.definition_set_id, dir_fd=root_fd, follow_symlinks=False)
                target_identity = (
                    target_info.st_dev, target_info.st_ino, target_info.st_mode, 0, 0,
                )
                if not stat.S_ISDIR(target_info.st_mode) or stat.S_IMODE(target_info.st_mode) != 0o700:
                    raise _ObservationUncertain("created evidence target is invalid")
                root_continuity_after_create = _continuity(self.root)
                bundle_fd = os.open(
                    request.definition_set_id,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_fd,
                )
                for name, data in request.members:
                    _write_exclusive(bundle_fd, name, data)
                    _read_exact(bundle_fd, name, data, conflict=False)
                    prefix += 1
                os.fsync(bundle_fd)
                manifest = _manifest_bytes(request)
                _write_exclusive(bundle_fd, MANIFEST_NAME, manifest)
                _read_exact(bundle_fd, MANIFEST_NAME, manifest, conflict=False)
                target_continuity = _continuity(target)
                self._verify_complete(request, bundle_fd, conflict=False)
                os.fsync(bundle_fd)
                os.fsync(root_fd)
                if (
                    _directory_identity(self.root, private=True) != before
                    or _continuity(self.root) != root_continuity_after_create
                    or _directory_identity(target, private=True) != target_identity
                    or _continuity(target) != target_continuity
                ):
                    raise _ObservationUncertain("final evidence namespace changed")
                return _store_receipt(
                    request, operation, "COMMITTED", True, len(MEMBER_PATHS),
                    before, before, target_identity, manifest,
                )
            except _PROCESS_CONTROL:
                raise
            except Exception:
                return _store_receipt(
                    request, operation, "UNCERTAIN", True, prefix,
                    before, _try_root_identity(self.root),
                )
        finally:
            if bundle_fd is not None:
                _close(bundle_fd, suppress=True)
            _close(root_fd, suppress=True)

    def _classify_existing(
        self, request: _EvidenceRequest, root_fd: int,
        before: Tuple[int, int, int, int, int], operation: str,
        *, strict_close: bool = False,
    ) -> ForwardDefinitionEvidenceStoreReceipt:
        try:
            manifest, target_identity = self._observe_existing(
                request, root_fd, before, strict_close=strict_close,
            )
            return _store_receipt(
                request, operation, "ADOPTED", False, len(MEMBER_PATHS),
                before, before, target_identity, manifest,
            )
        except _PROCESS_CONTROL:
            raise
        except _ExistingConflict as first:
            try:
                self._observe_existing(
                    request, root_fd, before, strict_close=strict_close,
                )
            except _PROCESS_CONTROL:
                raise
            except _ExistingConflict as second:
                stable = first.fingerprint == second.fingerprint
            except Exception:
                stable = False
            else:
                stable = False
            if not stable or _directory_identity(self.root, private=True) != before:
                return _store_receipt(
                    request, operation, "UNCERTAIN", False, 0,
                    before, _try_root_identity(self.root),
                )
            return _store_receipt(request, operation, "CONFLICT", False, 0, before, before)
        except Exception:
            return _store_receipt(
                request, operation, "UNCERTAIN", False, 0,
                before, _try_root_identity(self.root),
            )

    def _observe_existing(
        self, request: _EvidenceRequest, root_fd: int,
        before: Tuple[int, int, int, int, int],
        *, strict_close: bool = False,
    ) -> Tuple[bytes, Tuple[int, int, int, int, int]]:
        bundle_fd: Optional[int] = None
        target = self.root / request.definition_set_id
        if _directory_identity(self.root, private=True) != before:
            raise _ObservationUncertain("evidence root changed before inspection")
        try:
            info = os.stat(request.definition_set_id, dir_fd=root_fd, follow_symlinks=False)
            entry = (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_ctime_ns)
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
                raise _ExistingConflict(hashlib.sha256(_canonical({
                    "class": "TARGET_METADATA", "entry": list(entry),
                })).hexdigest())
            identity = (info.st_dev, info.st_ino, info.st_mode, 0, 0)
            continuity = _continuity(target)
            bundle_fd = os.open(
                request.definition_set_id,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            manifest = self._verify_complete(request, bundle_fd, conflict=True)
            if (
                _directory_identity(self.root, private=True) != before
                or _directory_identity(target, private=True) != identity
                or _continuity(target) != continuity
            ):
                raise _ObservationUncertain("existing evidence namespace changed")
            return manifest, identity
        finally:
            if bundle_fd is not None:
                _close(bundle_fd, suppress=not strict_close)

    @staticmethod
    def _verify_complete(request: _EvidenceRequest, descriptor: int, *, conflict: bool) -> bytes:
        expected = tuple(sorted(MEMBER_PATHS + (MANIFEST_NAME,)))
        try:
            inventory = tuple(sorted(os.listdir(descriptor)))
        except OSError as exc:
            raise _ObservationUncertain("evidence inventory is unavailable") from exc
        if inventory != expected:
            if conflict:
                raise _ExistingConflict(hashlib.sha256(_canonical({
                    "class": "INVENTORY", "inventory": list(inventory),
                })).hexdigest())
            raise _ObservationUncertain("created evidence inventory differs")
        for name, data in request.members:
            _read_exact(descriptor, name, data, conflict=conflict)
        manifest = _manifest_bytes(request)
        _read_exact(descriptor, MANIFEST_NAME, manifest, conflict=conflict)
        return manifest


def _definition_receipt_bytes(receipt: Any) -> bytes:
    from quantpits.research.definition_store import DefinitionStoreReceipt
    if type(receipt) is not DefinitionStoreReceipt:
        raise ForwardDefinitionEvidenceContractError("definition receipt is foreign")
    rendered = receipt.to_dict()
    expected_fields = {
        "operation_id", "status", "reason_code", "did_write", "definition_set_id",
        "request_digest", "manifest_digest", "member_count",
        "store_root_identity_before", "store_root_identity_after", "bundle_root_identity",
    }
    valid_identity = lambda value: (
        type(value) is tuple and len(value) == 5 and all(type(item) is int for item in value)
    )
    if set(rendered) != expected_fields:
        raise ForwardDefinitionEvidenceContractError("definition receipt fields are invalid")
    before = receipt.store_root_identity_before
    expected_operation = hashlib.sha256(json.dumps({
        "domain": "C0_DEFINITION_PUBLICATION_V1",
        "definition_set_id": receipt.definition_set_id,
        "request_digest": dict(receipt.request_digest),
        "store_root_identity_before": list(before),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8",
    )).hexdigest() if valid_identity(before) else None
    if (
        rendered["operation_id"] != receipt.operation_id
        or receipt.operation_id != expected_operation
        or rendered["status"] != receipt.status or receipt.status != "ADOPTED"
        or rendered["reason_code"] != receipt.reason_code
        or receipt.reason_code != "EXACT_BUNDLE_ALREADY_PRESENT"
        or rendered["did_write"] is not receipt.did_write or receipt.did_write is not False
        or rendered["definition_set_id"] != receipt.definition_set_id
        or _definition_id(receipt.definition_set_id) != receipt.definition_set_id
        or rendered["request_digest"] != dict(receipt.request_digest)
        or _typed_digest(
            dict(receipt.request_digest), "definition_request_digest", "canonical_json",
        ) != dict(receipt.request_digest)
        or rendered["manifest_digest"] != dict(receipt.manifest_digest)
        or _typed_digest(
            dict(receipt.manifest_digest), "definition_manifest_digest", "raw_bytes",
        ) != dict(receipt.manifest_digest)
        or rendered["member_count"] != receipt.member_count or receipt.member_count != 4
        or not valid_identity(before)
        or not valid_identity(receipt.store_root_identity_after)
        or receipt.store_root_identity_after != before
        or rendered["store_root_identity_before"] != list(before)
        or rendered["store_root_identity_after"] != list(before)
        or not valid_identity(receipt.bundle_root_identity)
        or rendered["bundle_root_identity"] != list(receipt.bundle_root_identity)
    ):
        raise ForwardDefinitionEvidenceContractError("definition bundle was not adopted exactly")
    return _canonical(rendered)


def _verify_public_evidence(
    root: Path, request: _EvidenceRequest,
    receipt: ForwardDefinitionEvidenceStoreReceipt,
) -> bool:
    if receipt.status not in {"COMMITTED", "ADOPTED"}:
        return True
    root_fd, identity = _open_root(root)
    try:
        if identity != receipt.store_root_identity_before:
            return False
        manifest, bundle = _CreateOnlyEvidenceStore(root)._observe_existing(
            request, root_fd, identity,
        )
        return (
            bundle == receipt.bundle_root_identity
            and _raw_digest(manifest) == dict(receipt.manifest_digest)
            and _directory_identity(root, private=True) == receipt.store_root_identity_after
        )
    except _PROCESS_CONTROL:
        raise
    except Exception:
        return False
    finally:
        _close(root_fd, suppress=True)


def _fresh_join(
    root: Path, cycle_id: str, activation: Path, definitions: Path,
) -> Tuple[Any, Any, _EvidenceRequest]:
    from quantpits.research.forward_observation import (
        observe_frozen_shadow_forward_definition_candidate,
    )
    candidate = observe_frozen_shadow_forward_definition_candidate(root, cycle_id, activation)
    request = candidate.to_store_request()
    if activation.name != request.definition_set_id + ".json":
        raise _input("activation public name does not match the definition set")
    from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
    receipt = CreateOnlyDefinitionBundleStore(definitions).adopt_existing(request)
    receipt_bytes = _definition_receipt_bytes(receipt)
    if (
        receipt.definition_set_id != request.definition_set_id
        or dict(receipt.request_digest) != dict(request.request_digest)
    ):
        raise ForwardDefinitionEvidenceContractError("definition receipt does not join fresh request")
    reference_bytes = _canonical(dict(candidate.reference_receipt))
    if TypedDigest.canonical(dict(candidate.reference_receipt)).to_dict() != dict(
        candidate.reference_receipt_digest
    ):
        raise ForwardDefinitionEvidenceContractError("reference receipt is not stable")
    evidence_request = _EvidenceRequest(
        request.definition_set_id, candidate.evidence_cycle_id,
        reference_bytes, receipt_bytes, candidate.reference_receipt_digest,
        request.request_digest,
        receipt.manifest_digest, receipt.operation_id,
    )
    return candidate, receipt, evidence_request


def _fresh_segment_join(
    production: Path, research: Path, cycle_id: str, activation: Path,
    definitions: Path,
) -> Tuple[Any, Any, _EvidenceRequest]:
    """Join one inspector-owned split-root candidate to an actual C0 adoption."""
    from quantpits.research.forward_observation import (
        FreshChampionSegmentCandidate,
        observe_fresh_champion_segment_candidate,
    )
    candidate = observe_fresh_champion_segment_candidate(
        production, research, cycle_id, activation,
    )
    if type(candidate) is not FreshChampionSegmentCandidate:
        raise ForwardDefinitionEvidenceContractError(
            "fresh observation returned foreign candidate authority",
        )
    if (
        candidate.fresh_segment_candidate_ready is not True
        or candidate.sealed_reference_join_verified is not True
        or candidate.publication_capability is not False
        or candidate.definition_bound is not False
        or candidate.intent_capability is not False
        or candidate.epoch_started is not False
        or candidate.prospective_claim is not False
        or candidate.promotion_capability is not False
        or candidate.did_write is not False
    ):
        raise ForwardDefinitionEvidenceContractError(
            "fresh candidate authority claims are invalid",
        )
    request = candidate.compiled_definitions.to_store_request()
    if (
        activation.name != request.definition_set_id + ".json"
        or request.definition_set_id != candidate.definition_set_id
        or dict(request.request_digest) != dict(candidate.compiled_request_digest)
    ):
        raise ForwardDefinitionEvidenceContractError(
            "fresh candidate and C0 request do not join",
        )
    from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
    receipt = CreateOnlyDefinitionBundleStore(definitions).adopt_existing(request)
    receipt_bytes = _definition_receipt_bytes(receipt)
    if (
        receipt.definition_set_id != request.definition_set_id
        or dict(receipt.request_digest) != dict(request.request_digest)
    ):
        raise ForwardDefinitionEvidenceContractError(
            "definition receipt does not join fresh split-root request",
        )
    reference = dict(candidate.reference_receipt)
    reference_bytes = _canonical(reference)
    if TypedDigest.canonical(reference).to_dict() != dict(
        candidate.reference_receipt_digest
    ):
        raise ForwardDefinitionEvidenceContractError(
            "fresh reference receipt is not stable",
        )
    evidence_request = _EvidenceRequest(
        request.definition_set_id, candidate.evidence_cycle_id,
        reference_bytes, receipt_bytes, candidate.reference_receipt_digest,
        request.request_digest, receipt.manifest_digest, receipt.operation_id,
    )
    return candidate, receipt, evidence_request


class ForwardDefinitionEvidencePlan:
    """Inspector-owned zero-capability preflight result."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise ForwardDefinitionEvidenceContractError("evidence plans are inspector-owned")
        expected = {
            "definition_set_id", "evidence_cycle_id", "definition_request_digest",
            "reference_receipt_digest", "definition_store_receipt_digest",
            "definition_manifest_digest", "definition_store_operation_id",
            "evidence_request_digest", "target_state",
        }
        if set(kwargs) != expected:
            raise ForwardDefinitionEvidenceContractError("evidence plan fields are not exact")
        for name, value in kwargs.items():
            if name.endswith("_digest"):
                value = MappingProxyType(dict(value))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_authority", _AUTHORITY)
        _PLAN_BINDINGS[self] = _canonical(self._summary())

    def _validate(self) -> None:
        if type(self) is not ForwardDefinitionEvidencePlan or getattr(self, "_authority", None) is not _AUTHORITY:
            raise ForwardDefinitionEvidenceContractError("evidence plan authority is absent")
        if self.target_state not in {"ABSENT", "PRESENT"}:
            raise ForwardDefinitionEvidenceContractError("evidence target state is invalid")
        _definition_id(self.definition_set_id)
        _cycle(self.evidence_cycle_id)
        if (
            type(self.definition_store_operation_id) is not str
            or len(self.definition_store_operation_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.definition_store_operation_id
            )
        ):
            raise ForwardDefinitionEvidenceContractError("definition operation ID is invalid")
        for name in (
            "definition_request_digest", "reference_receipt_digest",
            "definition_store_receipt_digest", "definition_manifest_digest",
            "evidence_request_digest",
        ):
            domain = {
                "definition_request_digest": "canonical_json",
                "reference_receipt_digest": "canonical_json",
                "definition_store_receipt_digest": "raw_bytes",
                "definition_manifest_digest": "raw_bytes",
                "evidence_request_digest": "canonical_json",
            }[name]
            _typed_digest(dict(getattr(self, name)), name, domain)
        expected = _PLAN_BINDINGS.get(self)
        if expected is None or expected != _canonical(self._summary()):
            raise ForwardDefinitionEvidenceContractError("evidence plan differs from its observation")

    def __copy__(self) -> Any:
        raise ForwardDefinitionEvidenceContractError("evidence plan replay is not authoritative")

    __deepcopy__ = lambda self, memo: self.__copy__()

    @property
    def publication_capability(self) -> bool:
        self._validate()
        return False

    @property
    def definition_evidence_complete(self) -> bool:
        self._validate()
        return False

    def _summary(self) -> Dict[str, Any]:
        return {
            "status": "PREPARED", "schema_version": 1,
            "definition_set_id": self.definition_set_id,
            "evidence_cycle_id": self.evidence_cycle_id,
            "definition_request_digest": dict(self.definition_request_digest),
            "reference_receipt_digest": dict(self.reference_receipt_digest),
            "definition_store_receipt_digest": dict(self.definition_store_receipt_digest),
            "definition_manifest_digest": dict(self.definition_manifest_digest),
            "definition_store_operation_id": self.definition_store_operation_id,
            "definition_store_status": "ADOPTED",
            "definition_store_did_write": False,
            "evidence_request_digest": dict(self.evidence_request_digest),
            "target_state": self.target_state,
            "publication_capability": False,
            "durable_reference_recorded": False,
            "definition_evidence_complete": False,
            "portfolio_bootstrap_complete": False,
            "epoch_started": False, "prospective_claim": False,
            "promotion_capability": False,
        }

    def to_safe_summary_dict(self) -> Dict[str, Any]:
        self._validate()
        return self._summary()


class _OuterObservation:
    """Inspector-owned final comparison facts for the outer capability join."""

    def __init__(
        self, *, _authority: Any, identities_before: Mapping[str, Tuple[int, ...]],
        identities_after: Optional[Mapping[str, Tuple[int, ...]]],
        inventory_before: Sequence[Tuple[Any, ...]],
        inventory_after: Optional[Sequence[Tuple[Any, ...]]],
        evidence_did_write: bool, requires_public_exact: bool, public_exact: bool,
    ) -> None:
        if _authority is not _AUTHORITY:
            raise ForwardDefinitionEvidenceContractError("outer observations are inspector-owned")
        if any(type(value) is not bool for value in (
            evidence_did_write, requires_public_exact, public_exact,
        )):
            raise ForwardDefinitionEvidenceContractError("outer observation flags are invalid")
        object.__setattr__(self, "identities_before", MappingProxyType(dict(identities_before)))
        object.__setattr__(
            self, "identities_after",
            None if identities_after is None else MappingProxyType(dict(identities_after)),
        )
        object.__setattr__(
            self, "inventory_before_digest",
            MappingProxyType(TypedDigest.canonical(tuple(inventory_before), "file_inventory").to_dict()),
        )
        object.__setattr__(
            self, "inventory_after_digest",
            None if inventory_after is None else MappingProxyType(
                TypedDigest.canonical(tuple(inventory_after), "file_inventory").to_dict()
            ),
        )
        object.__setattr__(self, "evidence_did_write", evidence_did_write)
        object.__setattr__(self, "requires_public_exact", requires_public_exact)
        object.__setattr__(self, "public_exact", public_exact)
        object.__setattr__(self, "_authority", _AUTHORITY)
        self._validate(check_binding=False)
        _OUTER_BINDINGS[self] = _canonical(self._render())

    def _validate(self, check_binding: bool = True) -> None:
        if type(self) is not _OuterObservation or getattr(self, "_authority", None) is not _AUTHORITY:
            raise ForwardDefinitionEvidenceContractError("outer observation authority is absent")
        if not self.identities_before or any(
            type(name) is not str or type(identity) is not tuple
            or not identity or any(type(item) is not int for item in identity)
            for name, identity in self.identities_before.items()
        ):
            raise ForwardDefinitionEvidenceContractError("outer original identities are invalid")
        if self.identities_after is not None and set(self.identities_after) != set(self.identities_before):
            raise ForwardDefinitionEvidenceContractError("outer final identities are incomplete")
        _typed_digest(
            dict(self.inventory_before_digest), "inventory_before_digest", "file_inventory",
        )
        if self.inventory_after_digest is not None:
            _typed_digest(
                dict(self.inventory_after_digest), "inventory_after_digest", "file_inventory",
            )
        if check_binding and _OUTER_BINDINGS.get(self) != _canonical(self._render()):
            raise ForwardDefinitionEvidenceContractError("outer observation differs from inspector facts")

    @property
    def stable(self) -> bool:
        self._validate()
        if self.identities_after is None or self.inventory_after_digest is None:
            return False
        before = dict(self.identities_before)
        after = dict(self.identities_after)
        if self.evidence_did_write:
            before.pop("evidence_continuity", None)
            after.pop("evidence_continuity", None)
            before.pop("research_evidence_continuity", None)
            after.pop("research_evidence_continuity", None)
        return (
            before == after
            and dict(self.inventory_before_digest) == dict(self.inventory_after_digest)
            and (not self.requires_public_exact or self.public_exact)
        )

    def _render(self) -> Dict[str, Any]:
        render_identities = lambda values: None if values is None else {
            name: list(identity) for name, identity in sorted(values.items())
        }
        return {
            "identities_before": render_identities(self.identities_before),
            "identities_after": render_identities(self.identities_after),
            "inventory_before_digest": dict(self.inventory_before_digest),
            "inventory_after_digest": (
                None if self.inventory_after_digest is None
                else dict(self.inventory_after_digest)
            ),
            "evidence_did_write": self.evidence_did_write,
            "requires_public_exact": self.requires_public_exact,
            "public_exact": self.public_exact,
        }


def _observe_outer(
    root: Path, activation: Path, definitions: Path, evidence: Path,
    identities_before: Mapping[str, Tuple[int, ...]],
    inventory_before: Sequence[Tuple[Any, ...]], excluded: Path,
    request: _EvidenceRequest, receipt: ForwardDefinitionEvidenceStoreReceipt,
) -> _OuterObservation:
    identities_after = None
    inventory_after = None
    public_exact = False
    try:
        _, _, _, _, identities_after = _formal_inputs(
            root, activation, definitions, evidence,
        )
        inventory_after = _workspace_inventory(root, excluded)
        public_exact = _verify_public_evidence(evidence, request, receipt)
    except _PROCESS_CONTROL:
        raise
    except Exception:
        identities_after = None
        inventory_after = None
        public_exact = False
    return _OuterObservation(
        _authority=_AUTHORITY, identities_before=identities_before,
        identities_after=identities_after, inventory_before=inventory_before,
        inventory_after=inventory_after, evidence_did_write=receipt.did_write,
        requires_public_exact=receipt.status in {"COMMITTED", "ADOPTED"},
        public_exact=public_exact,
    )


def _observe_fresh_outer(
    production: Path, research: Path, activation: Path, definitions: Path,
    evidence: Path, identities_before: Mapping[str, Tuple[int, ...]],
    inventory_before: Sequence[Tuple[Any, ...]], excluded: Path,
    request: _EvidenceRequest, receipt: ForwardDefinitionEvidenceStoreReceipt,
) -> _OuterObservation:
    identities_after = None
    inventory_after = None
    public_exact = False
    try:
        (
            _production, _research, _activation, _definitions, _evidence,
            identities_after,
        ) = _split_formal_inputs(
            production, research, activation, definitions, evidence,
        )
        inventory_after = _workspace_inventory(
            research, excluded, byte_budget=_FRESH_MAX_INVENTORY_BYTES,
        )
        public_exact = _verify_public_evidence(evidence, request, receipt)
        inventory_after = _workspace_inventory(
            research, excluded, byte_budget=_FRESH_MAX_INVENTORY_BYTES,
        )
        if not _same_split_identities(
            production, research, activation, definitions, evidence,
            identities_before, evidence_did_write=receipt.did_write,
        ):
            identities_after = None
    except _PROCESS_CONTROL:
        raise
    except Exception:
        identities_after = None
        inventory_after = None
        public_exact = False
    return _OuterObservation(
        _authority=_AUTHORITY, identities_before=identities_before,
        identities_after=identities_after, inventory_before=inventory_before,
        inventory_after=inventory_after, evidence_did_write=receipt.did_write,
        requires_public_exact=receipt.status in {"COMMITTED", "ADOPTED"},
        public_exact=public_exact,
    )


class ForwardDefinitionEvidenceResult:
    """Writer-owned result joining fresh definition and evidence receipts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise ForwardDefinitionEvidenceContractError("evidence results are writer-owned")
        expected = {
            "definition_receipt", "evidence_receipt", "evidence_cycle_id",
            "reference_receipt_digest", "definition_store_receipt_digest",
            "definition_manifest_digest", "evidence_request", "outer_observation",
        }
        if set(kwargs) != expected:
            raise ForwardDefinitionEvidenceContractError("evidence result fields are invalid")
        for name, value in kwargs.items():
            if name.endswith("_digest"):
                value = MappingProxyType(dict(value))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_authority", _AUTHORITY)
        self._validate(check_binding=False)
        _RESULT_BINDINGS[self] = _canonical(self._summary())

    def _validate(self, check_binding: bool = True) -> None:
        from quantpits.research.definition_store import DefinitionStoreReceipt
        if (
            type(self) is not ForwardDefinitionEvidenceResult
            or getattr(self, "_authority", None) is not _AUTHORITY
            or type(self.definition_receipt) is not DefinitionStoreReceipt
            or type(self.evidence_receipt) is not ForwardDefinitionEvidenceStoreReceipt
            or type(self.outer_observation) is not _OuterObservation
        ):
            raise ForwardDefinitionEvidenceContractError("evidence result authority is absent")
        definition_bytes = _definition_receipt_bytes(self.definition_receipt)
        definition_digest = _raw_digest(definition_bytes)
        request = _revalidate_evidence_request(self.evidence_request)
        if (
            definition_digest != dict(self.definition_store_receipt_digest)
            or dict(self.definition_receipt.manifest_digest) != dict(self.definition_manifest_digest)
            or self.definition_receipt.definition_set_id != self.evidence_receipt.definition_set_id
            or request.definition_set_id != self.definition_receipt.definition_set_id
            or request.evidence_cycle_id != self.evidence_cycle_id
            or request.members[1][1] != definition_bytes
            or dict(request.reference_receipt_digest) != dict(self.reference_receipt_digest)
            or dict(request.definition_request_digest) != dict(self.definition_receipt.request_digest)
            or dict(request.definition_manifest_digest) != dict(self.definition_receipt.manifest_digest)
            or request.definition_operation_id != self.definition_receipt.operation_id
            or dict(request.request_digest) != dict(self.evidence_receipt.request_digest)
            or self.outer_observation.evidence_did_write is not self.evidence_receipt.did_write
        ):
            raise ForwardDefinitionEvidenceContractError("evidence result receipt join is invalid")
        _cycle(self.evidence_cycle_id)
        _typed_digest(dict(self.reference_receipt_digest), "reference_receipt_digest", "canonical_json")
        self.outer_observation._validate()
        if check_binding:
            expected = _RESULT_BINDINGS.get(self)
            if expected is None or expected != _canonical(self._summary()):
                raise ForwardDefinitionEvidenceContractError("evidence result differs from writer authority")

    @property
    def status(self) -> str:
        self._validate()
        if not self.outer_observation.stable and self.evidence_receipt.status != "UNCERTAIN":
            return "UNCERTAIN"
        return self.evidence_receipt.status

    @property
    def definition_evidence_complete(self) -> bool:
        self._validate()
        return self.outer_observation.stable and self.evidence_receipt.durable_record_capability

    @property
    def definition_bundle_adopted_verified(self) -> bool:
        return self.definition_evidence_complete

    @property
    def durable_reference_recorded(self) -> bool:
        return self.definition_evidence_complete

    @property
    def portfolio_bootstrap_complete(self) -> bool:
        self._validate()
        return False

    @property
    def epoch_started(self) -> bool:
        self._validate()
        return False

    @property
    def prospective_claim(self) -> bool:
        self._validate()
        return False

    @property
    def promotion_capability(self) -> bool:
        self._validate()
        return False

    def __copy__(self) -> Any:
        raise ForwardDefinitionEvidenceContractError("evidence result replay is not authoritative")

    __deepcopy__ = lambda self, memo: self.__copy__()

    def _summary(self) -> Dict[str, Any]:
        stable = self.outer_observation.stable
        durable = stable and self.evidence_receipt.durable_record_capability
        outer_uncertain = not stable and self.evidence_receipt.status != "UNCERTAIN"
        return {
            "status": "UNCERTAIN" if outer_uncertain else self.evidence_receipt.status,
            "schema_version": 1,
            "reason_code": (
                "OUTER_EVIDENCE_IDENTITY_UNCERTAIN"
                if outer_uncertain else self.evidence_receipt.reason_code
            ),
            "did_write": self.evidence_receipt.did_write,
            "definition_set_id": self.definition_receipt.definition_set_id,
            "evidence_cycle_id": self.evidence_cycle_id,
            "definition_request_digest": dict(self.definition_receipt.request_digest),
            "reference_receipt_digest": dict(self.reference_receipt_digest),
            "definition_store_receipt_digest": dict(self.definition_store_receipt_digest),
            "definition_manifest_digest": dict(self.definition_manifest_digest),
            "definition_store_operation_id": self.definition_receipt.operation_id,
            "definition_store_status": "ADOPTED",
            "definition_store_did_write": False,
            "evidence_request_digest": dict(self.evidence_receipt.request_digest),
            "evidence_store_operation_id": self.evidence_receipt.operation_id,
            "evidence_manifest_digest": (
                None if self.evidence_receipt.manifest_digest is None
                else dict(self.evidence_receipt.manifest_digest)
            ),
            "evidence_member_count": self.evidence_receipt.member_count,
            "definition_bundle_adopted_verified": durable,
            "durable_reference_recorded": durable,
            "definition_evidence_complete": durable,
            "original_commit_receipt_recorded": False,
            "portfolio_bootstrap_complete": False,
            "epoch_started": False, "prospective_claim": False,
            "promotion_capability": False,
        }

    def to_safe_summary_dict(self) -> Dict[str, Any]:
        self._validate()
        return self._summary()


def _target_state(evidence: Path, identifier: str) -> str:
    try:
        os.lstat(str(evidence / identifier))
    except FileNotFoundError:
        return "ABSENT"
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _input("evidence target cannot be observed") from exc
    return "PRESENT"


def _prepare(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidencePlan:
    cycle_id = _cycle(evidence_cycle_id)
    root, activation, definitions, evidence, identities = _formal_inputs(
        workspace_root, activation_path, definition_store_root, evidence_store_root,
    )
    excluded = evidence / activation.stem
    before = _workspace_inventory(root, excluded)
    candidate, definition_receipt, request = _fresh_join(root, cycle_id, activation, definitions)
    if request.definition_set_id != activation.stem:
        raise _input("activation and evidence target identities differ")
    state = _target_state(evidence, request.definition_set_id)
    if (
        not _same_identities(root, activation, definitions, evidence, identities)
        or _workspace_inventory(root, excluded) != before
    ):
        raise _input("evidence inputs changed during preflight")
    return ForwardDefinitionEvidencePlan(
        _authority=_AUTHORITY, definition_set_id=request.definition_set_id,
        evidence_cycle_id=request.evidence_cycle_id,
        definition_request_digest=definition_receipt.request_digest,
        reference_receipt_digest=request.reference_receipt_digest,
        definition_store_receipt_digest=_raw_digest(request.members[1][1]),
        definition_manifest_digest=definition_receipt.manifest_digest,
        definition_store_operation_id=definition_receipt.operation_id,
        evidence_request_digest=request.request_digest, target_state=state,
    )


def _publish(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
    expected_definition_set_id: Any, expected_definition_request_digest: Any,
    expected_evidence_request_digest: Any, authorization_action: Any,
) -> ForwardDefinitionEvidenceResult:
    if type(authorization_action) is not str or authorization_action != AUTHORIZATION_ACTION:
        raise ForwardDefinitionEvidenceContractError("evidence authorization is invalid")
    expected_id = _definition_id(expected_definition_set_id)
    expected_definition = _typed_digest(
        expected_definition_request_digest, "expected_definition_request_digest", "canonical_json",
    )
    expected_evidence = _typed_digest(
        expected_evidence_request_digest, "expected_evidence_request_digest", "canonical_json",
    )
    cycle_id = _cycle(evidence_cycle_id)
    root, activation, definitions, evidence, identities = _formal_inputs(
        workspace_root, activation_path, definition_store_root, evidence_store_root,
    )
    excluded = evidence / expected_id
    before = _workspace_inventory(root, excluded)
    candidate, definition_receipt, request = _fresh_join(root, cycle_id, activation, definitions)
    if (
        request.definition_set_id != expected_id or activation.stem != expected_id
        or dict(definition_receipt.request_digest) != expected_definition
        or dict(request.request_digest) != expected_evidence
    ):
        raise ForwardDefinitionEvidenceContractError(
            "fresh evidence request does not match owner authorization",
        )
    if (
        not _same_identities(root, activation, definitions, evidence, identities)
        or _workspace_inventory(root, excluded) != before
    ):
        raise _input("evidence inputs changed before publication")
    evidence_receipt = _CreateOnlyEvidenceStore(evidence).publish(request)
    outer_observation = _observe_outer(
        root, activation, definitions, evidence, identities, before,
        excluded, request, evidence_receipt,
    )
    return ForwardDefinitionEvidenceResult(
        _authority=_AUTHORITY, definition_receipt=definition_receipt,
        evidence_receipt=evidence_receipt, evidence_cycle_id=request.evidence_cycle_id,
        reference_receipt_digest=request.reference_receipt_digest,
        definition_store_receipt_digest=_raw_digest(request.members[1][1]),
        definition_manifest_digest=definition_receipt.manifest_digest,
        evidence_request=request,
        outer_observation=outer_observation,
    )


def _prepare_fresh_champion_segment_definition_evidence(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidencePlan:
    cycle_id = _cycle(evidence_cycle_id)
    (
        production, research, activation, definitions, evidence, identities,
    ) = _split_formal_inputs(
        production_workspace_root, research_workspace_root, activation_path,
        definition_store_root, evidence_store_root,
    )
    excluded = evidence / activation.stem
    before = _workspace_inventory(
        research, excluded, byte_budget=_FRESH_MAX_INVENTORY_BYTES,
    )
    _candidate, definition_receipt, request = _fresh_segment_join(
        production, research, cycle_id, activation, definitions,
    )
    if request.definition_set_id != activation.stem:
        raise _input("activation and evidence target identities differ")
    state = _target_state(evidence, request.definition_set_id)
    if (
        not _same_split_identities(
            production, research, activation, definitions, evidence, identities,
        )
        or _workspace_inventory(
            research, excluded, byte_budget=_FRESH_MAX_INVENTORY_BYTES,
        ) != before
    ):
        raise _input("fresh evidence inputs changed during preflight")
    return ForwardDefinitionEvidencePlan(
        _authority=_AUTHORITY, definition_set_id=request.definition_set_id,
        evidence_cycle_id=request.evidence_cycle_id,
        definition_request_digest=definition_receipt.request_digest,
        reference_receipt_digest=request.reference_receipt_digest,
        definition_store_receipt_digest=_raw_digest(request.members[1][1]),
        definition_manifest_digest=definition_receipt.manifest_digest,
        definition_store_operation_id=definition_receipt.operation_id,
        evidence_request_digest=request.request_digest, target_state=state,
    )


def _publish_fresh_champion_segment_definition_evidence(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
    expected_definition_set_id: Any, expected_definition_request_digest: Any,
    expected_evidence_request_digest: Any, authorization_action: Any,
) -> ForwardDefinitionEvidenceResult:
    if (
        type(authorization_action) is not str
        or authorization_action != FRESH_EVIDENCE_AUTHORIZATION_ACTION
    ):
        raise ForwardDefinitionEvidenceContractError(
            "fresh evidence authorization is invalid",
        )
    expected_id = _definition_id(expected_definition_set_id)
    expected_definition = _typed_digest(
        expected_definition_request_digest,
        "expected_definition_request_digest", "canonical_json",
    )
    expected_evidence = _typed_digest(
        expected_evidence_request_digest,
        "expected_evidence_request_digest", "canonical_json",
    )
    cycle_id = _cycle(evidence_cycle_id)
    (
        production, research, activation, definitions, evidence, identities,
    ) = _split_formal_inputs(
        production_workspace_root, research_workspace_root, activation_path,
        definition_store_root, evidence_store_root,
    )
    excluded = evidence / expected_id
    before = _workspace_inventory(
        research, excluded, byte_budget=_FRESH_MAX_INVENTORY_BYTES,
    )
    candidate, definition_receipt, request = _fresh_segment_join(
        production, research, cycle_id, activation, definitions,
    )
    if (
        request.definition_set_id != expected_id
        or candidate.definition_set_id != expected_id
        or activation.stem != expected_id
        or dict(candidate.compiled_request_digest) != expected_definition
        or dict(definition_receipt.request_digest) != expected_definition
        or dict(request.request_digest) != expected_evidence
    ):
        raise ForwardDefinitionEvidenceContractError(
            "fresh evidence request does not match owner authorization",
        )
    if (
        not _same_split_identities(
            production, research, activation, definitions, evidence, identities,
        )
        or _workspace_inventory(
            research, excluded, byte_budget=_FRESH_MAX_INVENTORY_BYTES,
        ) != before
    ):
        raise _input("fresh evidence inputs changed before publication")
    evidence_receipt = _CreateOnlyEvidenceStore(evidence).publish(request)
    outer_observation = _observe_fresh_outer(
        production, research, activation, definitions, evidence, identities,
        before, excluded, request, evidence_receipt,
    )
    return ForwardDefinitionEvidenceResult(
        _authority=_AUTHORITY, definition_receipt=definition_receipt,
        evidence_receipt=evidence_receipt,
        evidence_cycle_id=request.evidence_cycle_id,
        reference_receipt_digest=request.reference_receipt_digest,
        definition_store_receipt_digest=_raw_digest(request.members[1][1]),
        definition_manifest_digest=definition_receipt.manifest_digest,
        evidence_request=request, outer_observation=outer_observation,
    )


def _adopt(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidenceResult:
    """Observe an already-published evidence record without crossing a write edge."""
    cycle_id = _cycle(evidence_cycle_id)
    root, activation, definitions, evidence, identities = _formal_inputs(
        workspace_root, activation_path, definition_store_root, evidence_store_root,
    )
    excluded = evidence / activation.stem
    before = _workspace_inventory(root, excluded)
    from quantpits.evidence.inspection import SourceMutationObserver
    guard = SourceMutationObserver(root, tuple(path.relative_to(root).as_posix() for path in (
        activation, definitions / activation.stem, evidence / activation.stem,
    )))
    try:
        candidate, definition_receipt, request = _fresh_join(root, cycle_id, activation, definitions)
        if request.definition_set_id != activation.stem:
            raise _input("activation and evidence target identities differ")
        root_fd, root_identity = _open_root(evidence)
        close_uncertain = False
        try:
            try:
                os.stat(request.definition_set_id, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise _input("evidence target is absent") from exc
            receipt = _CreateOnlyEvidenceStore(evidence)._classify_existing(
                request, root_fd, root_identity, _operation_id(request, root_identity),
                strict_close=True,
            )
        finally:
            try:
                _close(root_fd, suppress=False)
            except _PROCESS_CONTROL:
                raise
            except Exception:
                close_uncertain = True
        if close_uncertain:
            receipt = _store_receipt(
                request, _operation_id(request, root_identity), "UNCERTAIN", False, 0,
                root_identity, _try_root_identity(evidence),
            )
        outer = _observe_outer(
            root, activation, definitions, evidence, identities, before,
            excluded, request, receipt,
        )
        if not guard.supported or guard.mutated():
            receipt = _store_receipt(
                request, _operation_id(request, root_identity), "UNCERTAIN", False, 0,
                root_identity, _try_root_identity(evidence),
            )
            outer = _observe_outer(
                root, activation, definitions, evidence, identities, before,
                excluded, request, receipt,
            )
    finally:
        active = sys.exc_info()[1]
        try:
            guard.close()
        except _PROCESS_CONTROL:
            if not isinstance(active, _PROCESS_CONTROL):
                raise
        except OSError:
            if active is None:
                raise
    return ForwardDefinitionEvidenceResult(
        _authority=_AUTHORITY, definition_receipt=definition_receipt,
        evidence_receipt=receipt, evidence_cycle_id=request.evidence_cycle_id,
        reference_receipt_digest=request.reference_receipt_digest,
        definition_store_receipt_digest=_raw_digest(request.members[1][1]),
        definition_manifest_digest=definition_receipt.manifest_digest,
        evidence_request=request, outer_observation=outer,
    )


def _adopt_fresh_champion_segment_definition_evidence(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidenceResult:
    """Observe an existing split-root evidence record without a write path."""
    cycle_id = _cycle(evidence_cycle_id)
    (
        production, research, activation, definitions, evidence, identities,
    ) = _split_formal_inputs(
        production_workspace_root, research_workspace_root, activation_path,
        definition_store_root, evidence_store_root,
    )
    excluded = evidence / activation.stem
    from quantpits.evidence.inspection import SourceMutationObserver
    production_guard = SourceMutationObserver(
        production, (
            "config/strategy_config.yaml",
            "data/evidence/v1/cycles/%s" % cycle_id,
        ),
    )
    research_guard = SourceMutationObserver(
        research, tuple(path.relative_to(research).as_posix() for path in (
            activation, definitions / activation.stem,
            evidence / activation.stem,
        )),
    )
    try:
        before = _workspace_inventory(
            research, excluded, byte_budget=_FRESH_MAX_INVENTORY_BYTES,
        )
        candidate, definition_receipt, request = _fresh_segment_join(
            production, research, cycle_id, activation, definitions,
        )
        if request.definition_set_id != activation.stem:
            raise _input("activation and evidence target identities differ")
        root_fd, root_identity = _open_root(evidence)
        close_uncertain = False
        try:
            try:
                os.stat(request.definition_set_id, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise _input("evidence target is absent") from exc
            receipt = _CreateOnlyEvidenceStore(evidence)._classify_existing(
                request, root_fd, root_identity,
                _operation_id(request, root_identity), strict_close=True,
            )
        finally:
            try:
                _close(root_fd, suppress=False)
            except _PROCESS_CONTROL:
                raise
            except Exception:
                close_uncertain = True
        if close_uncertain:
            receipt = _store_receipt(
                request, _operation_id(request, root_identity),
                "UNCERTAIN", False, 0, root_identity,
                _try_root_identity(evidence),
            )
        outer = _observe_fresh_outer(
            production, research, activation, definitions, evidence,
            identities, before, excluded, request, receipt,
        )
        if (
            not production_guard.supported or production_guard.mutated()
            or not research_guard.supported or research_guard.mutated()
        ):
            receipt = _store_receipt(
                request, _operation_id(request, root_identity),
                "UNCERTAIN", False, 0, root_identity,
                _try_root_identity(evidence),
            )
            outer = _observe_fresh_outer(
                production, research, activation, definitions, evidence,
                identities, before, excluded, request, receipt,
            )
    finally:
        active = sys.exc_info()[1]
        for guard in (research_guard, production_guard):
            try:
                guard.close()
            except _PROCESS_CONTROL:
                if not isinstance(active, _PROCESS_CONTROL):
                    raise
            except OSError:
                if active is None:
                    raise
    result = ForwardDefinitionEvidenceResult(
        _authority=_AUTHORITY, definition_receipt=definition_receipt,
        evidence_receipt=receipt, evidence_cycle_id=request.evidence_cycle_id,
        reference_receipt_digest=request.reference_receipt_digest,
        definition_store_receipt_digest=_raw_digest(request.members[1][1]),
        definition_manifest_digest=definition_receipt.manifest_digest,
        evidence_request=request, outer_observation=outer,
    )
    if (
        result.status != "ADOPTED" or result.evidence_receipt.did_write
        or not result.definition_evidence_complete
    ):
        raise _input("fresh definition evidence was not adopted exactly")
    return result


def prepare_forward_definition_evidence(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidencePlan:
    """Freshly prepare a zero-write evidence request."""
    try:
        return _prepare(
            workspace_root, evidence_cycle_id, activation_path,
            definition_store_root, evidence_store_root,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        raise
    except Exception as exc:
        raise _input("evidence preflight failed closed") from exc


def publish_forward_definition_evidence(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
    expected_definition_set_id: Any, expected_definition_request_digest: Any,
    expected_evidence_request_digest: Any, authorization_action: Any,
) -> ForwardDefinitionEvidenceResult:
    """Freshly verify definitions and create or adopt one evidence record."""
    try:
        return _publish(
            workspace_root, evidence_cycle_id, activation_path,
            definition_store_root, evidence_store_root,
            expected_definition_set_id, expected_definition_request_digest,
            expected_evidence_request_digest, authorization_action,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        raise
    except Exception as exc:
        raise _input("evidence publication failed closed") from exc


def adopt_forward_definition_evidence(
    workspace_root: Any, definition_evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidenceResult:
    """Freshly adopt one exact existing evidence record, strictly without writes."""
    try:
        return _adopt(
            workspace_root, definition_evidence_cycle_id, activation_path,
            definition_store_root, evidence_store_root,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        raise
    except Exception as exc:
        raise _input("evidence adoption failed closed") from exc


def adopt_fresh_champion_segment_definition_evidence(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidenceResult:
    """Freshly adopt exact split-root evidence, strictly without writes."""
    try:
        return _adopt_fresh_champion_segment_definition_evidence(
            production_workspace_root, research_workspace_root,
            evidence_cycle_id, activation_path, definition_store_root,
            evidence_store_root,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        raise
    except Exception as exc:
        raise _input("fresh evidence adoption failed closed") from exc


def prepare_fresh_champion_segment_definition_evidence(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
) -> ForwardDefinitionEvidencePlan:
    """Prepare split-root fresh evidence without writing or granting capability."""
    try:
        return _prepare_fresh_champion_segment_definition_evidence(
            production_workspace_root, research_workspace_root,
            evidence_cycle_id, activation_path, definition_store_root,
            evidence_store_root,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        raise
    except Exception as exc:
        raise _input("fresh evidence preflight failed closed") from exc


def publish_fresh_champion_segment_definition_evidence(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
    expected_definition_set_id: Any, expected_definition_request_digest: Any,
    expected_evidence_request_digest: Any, authorization_action: Any,
) -> ForwardDefinitionEvidenceResult:
    """Freshly verify and create/adopt one split-root evidence record."""
    try:
        return _publish_fresh_champion_segment_definition_evidence(
            production_workspace_root, research_workspace_root,
            evidence_cycle_id, activation_path, definition_store_root,
            evidence_store_root, expected_definition_set_id,
            expected_definition_request_digest, expected_evidence_request_digest,
            authorization_action,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionEvidenceContractError:
        raise
    except Exception as exc:
        raise _input("fresh evidence publication failed closed") from exc


__all__ = [
    "AUTHORIZATION_ACTION", "FORWARD_EVIDENCE_AUTHORIZATION_ACTION",
    "FRESH_EVIDENCE_AUTHORIZATION_ACTION",
    "EVIDENCE_KIND", "STORAGE_CLAIM", "MEMBER_PATHS",
    "MANIFEST_NAME", "ForwardDefinitionEvidenceContractError",
    "ForwardDefinitionEvidenceInputError", "ForwardDefinitionEvidenceStoreReceipt",
    "ForwardDefinitionEvidencePlan", "ForwardDefinitionEvidenceResult",
    "prepare_forward_definition_evidence", "publish_forward_definition_evidence",
    "adopt_forward_definition_evidence",
    "adopt_fresh_champion_segment_definition_evidence",
    "prepare_fresh_champion_segment_definition_evidence",
    "publish_fresh_champion_segment_definition_evidence",
]
