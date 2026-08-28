"""Synthetic create-only storage for exact shadow-forward definition bytes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


BUNDLE_KIND = "SHADOW_FORWARD_DEFINITIONS_V1"
STORAGE_CLAIM = "CREATE_ONLY_EXACT_BYTES"
DEFINITION_PATHS = (
    "protocol.json",
    "execution_assumption.json",
    "champion_strategy.json",
    "challenger_strategy.json",
)
MANIFEST_NAME = "definition_manifest.json"
MAX_MEMBER_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 4 * 1024 * 1024

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_RECEIPT_AUTHORITY = object()
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class DefinitionStoreContractError(ValueError):
    """A caller-controlled definition-store value violates the contract."""


class DefinitionStoreInputError(RuntimeError):
    """The requested physical store root cannot be observed safely."""


@dataclass(frozen=True)
class _ConflictObservation:
    fingerprint: str


class _ExistingConflict(Exception):
    def __init__(self, observation: _ConflictObservation) -> None:
        super().__init__(observation.fingerprint)
        self.observation = observation


class _ObservationUncertain(Exception):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _conflict_observation(conflict_class: str, **facts: Any) -> _ConflictObservation:
    payload = {
        "domain": "C0_EXISTING_CONFLICT_OBSERVATION_V1",
        "conflict_class": conflict_class,
    }
    payload.update(facts)
    return _ConflictObservation(hashlib.sha256(_canonical_json(payload)).hexdigest())


def _raise_conflict(conflict_class: str, **facts: Any) -> None:
    raise _ExistingConflict(_conflict_observation(conflict_class, **facts))


def _close_descriptor(descriptor: int, *, suppress_ordinary: bool) -> None:
    """Close without allowing a secondary failure to mask active process control."""
    active = sys.exc_info()[1]
    try:
        os.close(descriptor)
    except _PROCESS_CONTROL:
        if isinstance(active, _PROCESS_CONTROL):
            return
        raise
    except OSError as exc:
        if isinstance(active, _PROCESS_CONTROL) or suppress_ordinary:
            return
        raise _ObservationUncertain("descriptor cleanup failed") from exc


def _digest(data: bytes, domain: str = "raw_bytes") -> Dict[str, Any]:
    return {
        "algorithm": "sha256",
        "domain": domain,
        "value": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _manifest_digest(data: bytes) -> Dict[str, Any]:
    return _digest(data, "raw_bytes")


def _manifest_row_digest(data: bytes) -> Dict[str, str]:
    return {"algorithm": "sha256", "value": hashlib.sha256(data).hexdigest()}


def _strict_json_object(data: Any, field: str) -> bytes:
    if not isinstance(data, bytes) or not data:
        raise DefinitionStoreContractError("%s must be non-empty exact bytes" % field)
    if len(data) > MAX_MEMBER_BYTES:
        raise DefinitionStoreContractError("%s exceeds the member size limit" % field)

    def pairs(items: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise DefinitionStoreContractError("%s must be strict canonical JSON" % field) from exc
    if not isinstance(value, dict):
        raise DefinitionStoreContractError("%s must be a JSON object" % field)
    try:
        rendered = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise DefinitionStoreContractError("%s must be strict canonical JSON" % field) from exc
    if rendered != data:
        raise DefinitionStoreContractError("%s must be byte-exact canonical JSON" % field)
    return data


def _definition_set_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or value in (".", "..")
        or value != value.strip()
        or _ID_PATTERN.fullmatch(value) is None
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise DefinitionStoreContractError("definition_set_id is not canonical")
    return value


@dataclass(frozen=True)
class DefinitionBundleMember:
    logical_path: str
    canonical_json_bytes: bytes

    def __post_init__(self) -> None:
        if self.logical_path not in DEFINITION_PATHS:
            raise DefinitionStoreContractError("definition member logical_path is invalid")
        _strict_json_object(self.canonical_json_bytes, self.logical_path)


@dataclass(frozen=True)
class DefinitionBundleRequest:
    definition_set_id: str
    members: Tuple[DefinitionBundleMember, ...]
    request_digest: Mapping[str, Any]

    def __init__(self, definition_set_id: Any, members: Any) -> None:
        identifier = _definition_set_id(definition_set_id)
        if not isinstance(members, (tuple, list)):
            raise DefinitionStoreContractError("members must be an ordered sequence")
        frozen = tuple(members)
        if len(frozen) != len(DEFINITION_PATHS) or any(
            type(member) is not DefinitionBundleMember for member in frozen
        ):
            raise DefinitionStoreContractError("members must contain exact definition leaves")
        paths = tuple(member.logical_path for member in frozen)
        if paths != DEFINITION_PATHS:
            raise DefinitionStoreContractError("definition member set and order are not exact")
        for member in frozen:
            _strict_json_object(member.canonical_json_bytes, member.logical_path)
        if sum(len(member.canonical_json_bytes) for member in frozen) > MAX_BUNDLE_BYTES:
            raise DefinitionStoreContractError("definition bundle exceeds the aggregate size limit")
        payload = _request_digest_payload(identifier, frozen)
        object.__setattr__(self, "definition_set_id", identifier)
        object.__setattr__(self, "members", frozen)
        object.__setattr__(
            self, "request_digest",
            MappingProxyType(_digest(_canonical_json(payload), "canonical_json")),
        )


def _request_digest_payload(
    definition_set_id: str, members: Sequence[DefinitionBundleMember],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "bundle_kind": BUNDLE_KIND,
        "definition_set_id": definition_set_id,
        "member_count": len(DEFINITION_PATHS),
        "members": [
            {
                "logical_path": member.logical_path,
                "size_bytes": len(member.canonical_json_bytes),
                "digest": _digest(member.canonical_json_bytes),
            }
            for member in members
        ],
    }


def revalidate_definition_bundle_request(value: Any) -> DefinitionBundleRequest:
    """Rebuild all caller-controlled request facts from the current payload."""
    if type(value) is not DefinitionBundleRequest:
        raise DefinitionStoreContractError("definition request is foreign")
    try:
        members = tuple(
            DefinitionBundleMember(member.logical_path, member.canonical_json_bytes)
            for member in value.members
        )
        return DefinitionBundleRequest(value.definition_set_id, members)
    except _PROCESS_CONTROL:
        raise
    except DefinitionStoreContractError:
        raise
    except Exception as exc:
        raise DefinitionStoreContractError("definition request cannot be revalidated") from exc


def _valid_digest(value: Any, domain: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"algorithm", "domain", "value", "size_bytes"}
        and value.get("algorithm") == "sha256"
        and value.get("domain") == domain
        and isinstance(value.get("value"), str)
        and len(value["value"]) == 64
        and all(character in "0123456789abcdef" for character in value["value"])
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and value["size_bytes"] >= 0
    )


def _valid_identity(value: Any) -> bool:
    return bool(
        isinstance(value, tuple)
        and len(value) == 5
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _expected_operation_id(
    definition_set_id: str,
    request_digest: Mapping[str, Any],
    identity: Tuple[int, int, int, int, int],
) -> str:
    payload = {
        "domain": "C0_DEFINITION_PUBLICATION_V1",
        "definition_set_id": definition_set_id,
        "request_digest": dict(request_digest),
        "store_root_identity_before": list(identity),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, init=False)
class DefinitionStoreReceipt:
    operation_id: str
    status: str
    reason_code: str
    did_write: bool
    definition_set_id: str
    request_digest: Mapping[str, Any]
    manifest_digest: Optional[Mapping[str, Any]]
    member_count: int
    store_root_identity_before: Tuple[int, int, int, int, int]
    store_root_identity_after: Optional[Tuple[int, int, int, int, int]]
    bundle_root_identity: Optional[Tuple[int, int, int, int, int]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _RECEIPT_AUTHORITY:
            raise DefinitionStoreContractError("definition store receipts are writer-owned")
        expected = {
            "operation_id", "status", "reason_code", "did_write", "definition_set_id",
            "request_digest", "manifest_digest", "member_count",
            "store_root_identity_before", "store_root_identity_after", "bundle_root_identity",
        }
        if set(kwargs) != expected:
            raise DefinitionStoreContractError("receipt fields are not exact")
        try:
            kwargs["request_digest"] = MappingProxyType(dict(kwargs["request_digest"]))
            if kwargs["manifest_digest"] is not None:
                kwargs["manifest_digest"] = MappingProxyType(dict(kwargs["manifest_digest"]))
        except (TypeError, ValueError) as exc:
            raise DefinitionStoreContractError("receipt digest is invalid") from exc
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        if (
            not isinstance(self.operation_id, str)
            or len(self.operation_id) != 64
            or any(character not in "0123456789abcdef" for character in self.operation_id)
            or not isinstance(self.did_write, bool)
            or _definition_set_id(self.definition_set_id) != self.definition_set_id
            or not _valid_digest(self.request_digest, "canonical_json")
            or not _valid_identity(self.store_root_identity_before)
        ):
            raise DefinitionStoreContractError("receipt identity is invalid")
        if self.operation_id != _expected_operation_id(
            self.definition_set_id, self.request_digest, self.store_root_identity_before,
        ):
            raise DefinitionStoreContractError("receipt operation identity is inconsistent")
        if self.manifest_digest is not None and not _valid_digest(self.manifest_digest, "raw_bytes"):
            raise DefinitionStoreContractError("receipt manifest digest is invalid")
        if self.store_root_identity_after is not None and not _valid_identity(self.store_root_identity_after):
            raise DefinitionStoreContractError("receipt store identity is invalid")
        if self.bundle_root_identity is not None and not _valid_identity(self.bundle_root_identity):
            raise DefinitionStoreContractError("receipt bundle identity is invalid")
        if isinstance(self.member_count, bool) or not isinstance(self.member_count, int):
            raise DefinitionStoreContractError("receipt member_count is invalid")
        reasons = {
            "COMMITTED": "COMMITTED",
            "ADOPTED": "EXACT_BUNDLE_ALREADY_PRESENT",
            "CONFLICT": "DEFINITION_SET_ID_CONFLICT",
            "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN",
        }
        if self.status not in reasons or self.reason_code != reasons[self.status]:
            raise DefinitionStoreContractError("receipt status and reason are inconsistent")
        positive = self.status in ("COMMITTED", "ADOPTED")
        if positive:
            valid = (
                self.did_write is (self.status == "COMMITTED")
                and self.manifest_digest is not None
                and self.member_count == len(DEFINITION_PATHS)
                and self.bundle_root_identity is not None
                and self.store_root_identity_after == self.store_root_identity_before
            )
        elif self.status == "CONFLICT":
            valid = (
                self.did_write is False
                and self.manifest_digest is None
                and self.member_count == 0
                and self.bundle_root_identity is None
                and self.store_root_identity_after == self.store_root_identity_before
            )
        else:
            valid = (
                0 <= self.member_count <= len(DEFINITION_PATHS)
                and self.manifest_digest is None
                and self.bundle_root_identity is None
                and (self.did_write or self.member_count == 0)
            )
        if not valid:
            raise DefinitionStoreContractError("receipt cross-fields are inconsistent")

    @property
    def publication_capability(self) -> bool:
        return self.status in ("COMMITTED", "ADOPTED")

    @property
    def semantic_claim(self) -> bool:
        return False

    @property
    def prospective_claim(self) -> bool:
        return False

    @property
    def promotion_capability(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "did_write": self.did_write,
            "definition_set_id": self.definition_set_id,
            "request_digest": dict(self.request_digest),
            "manifest_digest": None if self.manifest_digest is None else dict(self.manifest_digest),
            "member_count": self.member_count,
            "store_root_identity_before": list(self.store_root_identity_before),
            "store_root_identity_after": (
                None if self.store_root_identity_after is None else list(self.store_root_identity_after)
            ),
            "bundle_root_identity": (
                None if self.bundle_root_identity is None else list(self.bundle_root_identity)
            ),
        }


def _directory_identity(info: os.stat_result) -> Tuple[int, int, int, int, int]:
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DefinitionStoreInputError("expected a physical directory")
    return (info.st_dev, info.st_ino, info.st_mode, 0, 0)


def _fd_directory_identity(descriptor: int) -> Tuple[int, int, int, int, int]:
    return _directory_identity(os.fstat(descriptor))


def _path_directory_identity(path: Path) -> Tuple[int, int, int, int, int]:
    return _directory_identity(os.lstat(str(path)))


def _try_path_directory_identity(path: Path) -> Optional[Tuple[int, int, int, int, int]]:
    try:
        return _path_directory_identity(path)
    except (OSError, DefinitionStoreInputError):
        return None


def _continuity_snapshot(path: Path) -> Tuple[Any, ...]:
    """Observe immediate namespace continuity without opening or mutating it."""
    info = os.lstat(str(path))
    entry = (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        return (entry,)
    rows = []
    for name in sorted(os.listdir(str(path))):
        member = os.lstat(str(path / name))
        rows.append((
            name, member.st_dev, member.st_ino, member.st_mode, member.st_nlink,
            member.st_size, member.st_mtime_ns, member.st_ctime_ns,
        ))
    return entry, tuple(rows)


def _require_root(root: Any) -> Tuple[Path, int, Tuple[int, int, int, int, int]]:
    if isinstance(root, bool) or not isinstance(root, (str, os.PathLike)):
        raise DefinitionStoreContractError("store root must be an absolute physical path")
    path = Path(root)
    if not path.is_absolute():
        raise DefinitionStoreContractError("store root must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DefinitionStoreInputError("store root must already exist") from exc
    if resolved != path:
        raise DefinitionStoreInputError("store root must not use aliases or symlinks")
    try:
        before = _path_directory_identity(path)
        if stat.S_IMODE(before[2]) != 0o700:
            raise DefinitionStoreInputError("store root mode must be 0700")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(path), flags)
    except DefinitionStoreInputError:
        raise
    except OSError as exc:
        raise DefinitionStoreInputError("store root cannot be opened safely") from exc
    try:
        if _fd_directory_identity(descriptor) != before or _path_directory_identity(path) != before:
            raise DefinitionStoreInputError("store root identity changed while opening")
    except BaseException:
        _close_descriptor(descriptor, suppress_ordinary=True)
        raise
    return path, descriptor, before


def _file_identity(info: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _read_regular_at(
    descriptor: int, name: str, expected: bytes, *, conflict: bool,
) -> bytes:
    expected_size = len(expected)
    expected_digest = hashlib.sha256(expected).hexdigest()
    try:
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            if conflict:
                _raise_conflict(
                    "MEMBER_METADATA_MISMATCH",
                    decisive_logical_path=name,
                    entry_identity=list(_file_identity(before)),
                    entry_nlink=before.st_nlink,
                    expected_size=expected_size,
                    expected_digest=expected_digest,
                )
            raise _ObservationUncertain("definition member type or mode is invalid")
        if before.st_size != expected_size:
            if conflict:
                _raise_conflict(
                    "MEMBER_SIZE_MISMATCH",
                    decisive_logical_path=name,
                    entry_identity=list(_file_identity(before)),
                    entry_nlink=before.st_nlink,
                    expected_size=expected_size,
                    expected_digest=expected_digest,
                    observed_size=before.st_size,
                )
            raise _ObservationUncertain("definition member size differs")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(name, flags, dir_fd=descriptor)
        try:
            opened = os.fstat(file_fd)
            if opened.st_size != expected_size:
                raise _ObservationUncertain("definition member size changed while opening")
            chunks = []
            remaining = expected_size
            while remaining:
                chunk = os.read(file_fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            trailing = os.read(file_fd, 1)
            if trailing:
                chunks.append(trailing)
            opened_after = os.fstat(file_fd)
        finally:
            _close_descriptor(file_fd, suppress_ordinary=False)
        after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except _PROCESS_CONTROL:
        raise
    except (_ExistingConflict, _ObservationUncertain):
        raise
    except OSError as exc:
        raise _ObservationUncertain("definition member observation failed") from exc
    identities = tuple(_file_identity(item) for item in (before, opened, opened_after, after))
    if len(set(identities)) != 1:
        raise _ObservationUncertain("definition member identity changed")
    observed = b"".join(chunks)
    if len(observed) != expected_size:
        raise _ObservationUncertain("definition member bounded read was incomplete")
    return observed


def _write_exclusive(descriptor: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    file_fd = os.open(name, flags, 0o600, dir_fd=descriptor)
    try:
        os.fchmod(file_fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("short definition write")
            view = view[written:]
        os.fsync(file_fd)
    finally:
        _close_descriptor(file_fd, suppress_ordinary=False)


def _manifest_bytes(request: DefinitionBundleRequest) -> bytes:
    by_path = {member.logical_path: member.canonical_json_bytes for member in request.members}
    payload = {
        "schema_version": 1,
        "bundle_kind": BUNDLE_KIND,
        "storage_claim": STORAGE_CLAIM,
        "semantic_claim": False,
        "prospective_claim": False,
        "promotion_capability": False,
        "definition_set_id": request.definition_set_id,
        "request_digest": dict(request.request_digest),
        "member_count": len(DEFINITION_PATHS),
        "members": [
            {
                "logical_path": path,
                "size_bytes": len(by_path[path]),
                "digest": _manifest_row_digest(by_path[path]),
            }
            for path in sorted(DEFINITION_PATHS)
        ],
    }
    return _canonical_json(payload)


def _operation_id(
    request: DefinitionBundleRequest, identity: Tuple[int, int, int, int, int],
) -> str:
    return _expected_operation_id(request.definition_set_id, request.request_digest, identity)


def _receipt(
    request: DefinitionBundleRequest,
    operation_id: str,
    status: str,
    did_write: bool,
    member_count: int,
    before: Tuple[int, int, int, int, int],
    after: Optional[Tuple[int, int, int, int, int]],
    bundle_identity: Optional[Tuple[int, int, int, int, int]] = None,
    manifest: Optional[bytes] = None,
) -> DefinitionStoreReceipt:
    reasons = {
        "COMMITTED": "COMMITTED",
        "ADOPTED": "EXACT_BUNDLE_ALREADY_PRESENT",
        "CONFLICT": "DEFINITION_SET_ID_CONFLICT",
        "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN",
    }
    return DefinitionStoreReceipt(
        _authority=_RECEIPT_AUTHORITY,
        operation_id=operation_id,
        status=status,
        reason_code=reasons[status],
        did_write=did_write,
        definition_set_id=request.definition_set_id,
        request_digest=request.request_digest,
        manifest_digest=None if manifest is None else _manifest_digest(manifest),
        member_count=member_count,
        store_root_identity_before=before,
        store_root_identity_after=after,
        bundle_root_identity=bundle_identity,
    )


class CreateOnlyDefinitionBundleStore:
    """Publish or adopt one exact definition bundle under a physical store root."""

    def __init__(self, store_root: Any) -> None:
        self._store_root = store_root

    def publish(self, request: Any) -> DefinitionStoreReceipt:
        snapshot = revalidate_definition_bundle_request(request)
        root, store_fd, root_before = _require_root(self._store_root)
        operation_id = _operation_id(snapshot, root_before)
        target = root / snapshot.definition_set_id
        bundle_fd: Optional[int] = None
        verified_prefix = 0
        try:
            try:
                os.mkdir(snapshot.definition_set_id, 0o700, dir_fd=store_fd)
            except FileExistsError:
                return self._classify_existing(
                    snapshot, root, target, store_fd, root_before, operation_id,
                )
            except _PROCESS_CONTROL:
                raise
            except OSError as exc:
                raise DefinitionStoreInputError("bundle child could not be created") from exc
            try:
                if _fd_directory_identity(store_fd) != root_before or _path_directory_identity(root) != root_before:
                    raise _ObservationUncertain("store identity changed after create")
                target_info = os.stat(
                    snapshot.definition_set_id, dir_fd=store_fd, follow_symlinks=False,
                )
                target_identity = _directory_identity(target_info)
                if stat.S_IMODE(target_identity[2]) != 0o700:
                    raise _ObservationUncertain("created bundle mode is invalid")
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
                bundle_fd = os.open(snapshot.definition_set_id, flags, dir_fd=store_fd)
                if (
                    _fd_directory_identity(bundle_fd) != target_identity
                    or _path_directory_identity(target) != target_identity
                ):
                    raise _ObservationUncertain("created bundle identity changed")
                by_path = {
                    member.logical_path: member.canonical_json_bytes for member in snapshot.members
                }
                for logical_path in sorted(DEFINITION_PATHS):
                    _write_exclusive(bundle_fd, logical_path, by_path[logical_path])
                    if _read_regular_at(
                        bundle_fd, logical_path, by_path[logical_path], conflict=False,
                    ) != by_path[logical_path]:
                        raise _ObservationUncertain("created member bytes changed")
                    verified_prefix += 1
                os.fsync(bundle_fd)
                manifest = _manifest_bytes(snapshot)
                _write_exclusive(bundle_fd, MANIFEST_NAME, manifest)
                if _read_regular_at(
                    bundle_fd, MANIFEST_NAME, manifest, conflict=False,
                ) != manifest:
                    raise _ObservationUncertain("created manifest bytes changed")
                self._verify_complete(
                    snapshot, root, target, store_fd, bundle_fd, root_before,
                    target_identity, conflict=False,
                )
                os.fsync(bundle_fd)
                os.fsync(store_fd)
                if (
                    _fd_directory_identity(store_fd) != root_before
                    or _path_directory_identity(root) != root_before
                    or _fd_directory_identity(bundle_fd) != target_identity
                    or _path_directory_identity(target) != target_identity
                ):
                    raise _ObservationUncertain("final publication namespace changed")
                return _receipt(
                    snapshot, operation_id, "COMMITTED", True, len(DEFINITION_PATHS),
                    root_before, root_before, target_identity, manifest,
                )
            except _PROCESS_CONTROL:
                raise
            except Exception:
                return _receipt(
                    snapshot, operation_id, "UNCERTAIN", True, verified_prefix,
                    root_before, _try_path_directory_identity(root),
                )
        finally:
            if bundle_fd is not None:
                _close_descriptor(bundle_fd, suppress_ordinary=True)
            _close_descriptor(store_fd, suppress_ordinary=True)

    def adopt_existing(self, request: Any) -> DefinitionStoreReceipt:
        """Verify one already-public bundle without crossing a write boundary."""
        snapshot = revalidate_definition_bundle_request(request)
        if isinstance(self._store_root, bool) or not isinstance(
            self._store_root, (str, os.PathLike),
        ):
            raise DefinitionStoreContractError("store root must be an absolute physical path")
        root_input = Path(self._store_root)
        if not root_input.is_absolute():
            raise DefinitionStoreContractError("store root must be absolute")
        try:
            if root_input.resolve(strict=True) != root_input:
                raise DefinitionStoreInputError("store root must not use aliases or symlinks")
            root_continuity = _continuity_snapshot(root_input)
        except DefinitionStoreInputError:
            raise
        except _PROCESS_CONTROL:
            raise
        except OSError as exc:
            raise DefinitionStoreInputError("store root must already exist") from exc
        root, store_fd, root_before = _require_root(self._store_root)
        operation_id = _operation_id(snapshot, root_before)
        target = root / snapshot.definition_set_id
        try:
            try:
                os.stat(
                    snapshot.definition_set_id,
                    dir_fd=store_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError as exc:
                raise DefinitionStoreInputError(
                    "definition bundle does not already exist",
                ) from exc
            except _PROCESS_CONTROL:
                raise
            except OSError:
                return _receipt(
                    snapshot, operation_id, "UNCERTAIN", False, 0,
                    root_before, _try_path_directory_identity(root),
                )
            try:
                if _continuity_snapshot(root) != root_continuity:
                    raise _ObservationUncertain("store continuity changed while opening")
                target_continuity = _continuity_snapshot(target)
            except _PROCESS_CONTROL:
                raise
            except (OSError, _ObservationUncertain):
                return _receipt(
                    snapshot, operation_id, "UNCERTAIN", False, 0,
                    root_before, _try_path_directory_identity(root),
                )
            receipt = self._classify_existing(
                snapshot, root, target, store_fd, root_before, operation_id,
            )
            if receipt.status == "UNCERTAIN":
                return receipt
            try:
                continuous = (
                    _continuity_snapshot(root) == root_continuity
                    and _continuity_snapshot(target) == target_continuity
                )
            except _PROCESS_CONTROL:
                raise
            except OSError:
                continuous = False
            if not continuous:
                return _receipt(
                    snapshot, operation_id, "UNCERTAIN", False, 0,
                    root_before, _try_path_directory_identity(root),
                )
            return receipt
        finally:
            _close_descriptor(store_fd, suppress_ordinary=True)

    def _classify_existing(
        self,
        request: DefinitionBundleRequest,
        root: Path,
        target: Path,
        store_fd: int,
        root_before: Tuple[int, int, int, int, int],
        operation_id: str,
    ) -> DefinitionStoreReceipt:
        try:
            manifest, target_identity = self._observe_existing(
                request, root, target, store_fd, root_before,
            )
            return _receipt(
                request, operation_id, "ADOPTED", False, len(DEFINITION_PATHS),
                root_before, root_before, target_identity, manifest,
            )
        except _PROCESS_CONTROL:
            raise
        except _ExistingConflict as first:
            try:
                self._observe_existing(request, root, target, store_fd, root_before)
            except _PROCESS_CONTROL:
                raise
            except _ExistingConflict as second:
                if first.observation.fingerprint != second.observation.fingerprint:
                    return _receipt(
                        request, operation_id, "UNCERTAIN", False, 0,
                        root_before, _try_path_directory_identity(root),
                    )
            except Exception:
                return _receipt(
                    request, operation_id, "UNCERTAIN", False, 0,
                    root_before, _try_path_directory_identity(root),
                )
            else:
                return _receipt(
                    request, operation_id, "UNCERTAIN", False, 0,
                    root_before, _try_path_directory_identity(root),
                )
            if (
                _fd_directory_identity(store_fd) != root_before
                or _try_path_directory_identity(root) != root_before
            ):
                return _receipt(
                    request, operation_id, "UNCERTAIN", False, 0,
                    root_before, _try_path_directory_identity(root),
                )
            return _receipt(
                request, operation_id, "CONFLICT", False, 0,
                root_before, root_before,
            )
        except Exception:
            return _receipt(
                request, operation_id, "UNCERTAIN", False, 0,
                root_before, _try_path_directory_identity(root),
            )

    @staticmethod
    def _observe_existing(
        request: DefinitionBundleRequest,
        root: Path,
        target: Path,
        store_fd: int,
        root_identity: Tuple[int, int, int, int, int],
    ) -> Tuple[bytes, Tuple[int, int, int, int, int]]:
        bundle_fd: Optional[int] = None
        if _fd_directory_identity(store_fd) != root_identity or _path_directory_identity(root) != root_identity:
            raise _ObservationUncertain("store identity changed before existing inspection")
        target_info = os.stat(
            request.definition_set_id, dir_fd=store_fd, follow_symlinks=False,
        )
        target_entry_identity = _file_identity(target_info)
        if not stat.S_ISDIR(target_info.st_mode) or stat.S_ISLNK(target_info.st_mode):
            _raise_conflict(
                "TARGET_TYPE_MISMATCH",
                decisive_logical_path=request.definition_set_id,
                root_identity=list(root_identity),
                target_identity=list(target_entry_identity),
                target_nlink=target_info.st_nlink,
            )
        target_identity = _directory_identity(target_info)
        if stat.S_IMODE(target_identity[2]) != 0o700:
            _raise_conflict(
                "TARGET_MODE_MISMATCH",
                decisive_logical_path=request.definition_set_id,
                root_identity=list(root_identity),
                target_identity=list(target_entry_identity),
                target_nlink=target_info.st_nlink,
            )
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            bundle_fd = os.open(request.definition_set_id, flags, dir_fd=store_fd)
            if (
                _fd_directory_identity(bundle_fd) != target_identity
                or _path_directory_identity(target) != target_identity
            ):
                raise _ObservationUncertain("existing bundle identity changed")
            manifest = CreateOnlyDefinitionBundleStore._verify_complete(
                request, root, target, store_fd, bundle_fd, root_identity,
                target_identity, conflict=True,
            )
            if (
                _fd_directory_identity(store_fd) != root_identity
                or _path_directory_identity(root) != root_identity
                or _fd_directory_identity(bundle_fd) != target_identity
                or _path_directory_identity(target) != target_identity
            ):
                raise _ObservationUncertain("existing final namespace changed")
            return manifest, target_identity
        finally:
            if bundle_fd is not None:
                _close_descriptor(bundle_fd, suppress_ordinary=True)

    @staticmethod
    def _verify_complete(
        request: DefinitionBundleRequest,
        root: Path,
        target: Path,
        store_fd: int,
        bundle_fd: int,
        root_identity: Tuple[int, int, int, int, int],
        target_identity: Tuple[int, int, int, int, int],
        *,
        conflict: bool,
    ) -> bytes:
        expected_inventory = tuple(sorted(DEFINITION_PATHS + (MANIFEST_NAME,)))
        try:
            inventory = tuple(sorted(os.listdir(bundle_fd)))
        except OSError as exc:
            raise _ObservationUncertain("bundle inventory cannot be observed") from exc
        if inventory != expected_inventory:
            if (
                _fd_directory_identity(store_fd) != root_identity
                or _path_directory_identity(root) != root_identity
                or _fd_directory_identity(bundle_fd) != target_identity
                or _try_path_directory_identity(target) != target_identity
            ):
                raise _ObservationUncertain("namespace changed during inventory observation")
            if conflict:
                _raise_conflict(
                    "INVENTORY_MISMATCH",
                    decisive_logical_path=".",
                    raw_inventory=list(inventory),
                    root_identity=list(root_identity),
                    target_identity=list(target_identity),
                )
            raise _ObservationUncertain("created bundle inventory differs")
        by_path = {member.logical_path: member.canonical_json_bytes for member in request.members}
        for logical_path in sorted(DEFINITION_PATHS):
            try:
                observed = _read_regular_at(
                    bundle_fd, logical_path, by_path[logical_path], conflict=conflict,
                )
            except _ExistingConflict as exc:
                _raise_conflict(
                    "MEMBER_OBSERVATION_MISMATCH",
                    decisive_logical_path=logical_path,
                    raw_inventory=list(inventory),
                    root_identity=list(root_identity),
                    target_identity=list(target_identity),
                    member_observation=exc.observation.fingerprint,
                )
            if observed != by_path[logical_path]:
                if conflict:
                    _raise_conflict(
                        "MEMBER_BYTES_MISMATCH",
                        decisive_logical_path=logical_path,
                        raw_inventory=list(inventory),
                        root_identity=list(root_identity),
                        target_identity=list(target_identity),
                        expected_size=len(by_path[logical_path]),
                        expected_digest=hashlib.sha256(by_path[logical_path]).hexdigest(),
                        observed_size=len(observed),
                        observed_digest=hashlib.sha256(observed).hexdigest(),
                    )
                raise _ObservationUncertain("created definition bytes differ")
        expected_manifest = _manifest_bytes(request)
        try:
            manifest = _read_regular_at(
                bundle_fd, MANIFEST_NAME, expected_manifest, conflict=conflict,
            )
        except _ExistingConflict as exc:
            _raise_conflict(
                "MEMBER_OBSERVATION_MISMATCH",
                decisive_logical_path=MANIFEST_NAME,
                raw_inventory=list(inventory),
                root_identity=list(root_identity),
                target_identity=list(target_identity),
                member_observation=exc.observation.fingerprint,
            )
        if manifest != expected_manifest:
            if conflict:
                _raise_conflict(
                    "MEMBER_BYTES_MISMATCH",
                    decisive_logical_path=MANIFEST_NAME,
                    raw_inventory=list(inventory),
                    root_identity=list(root_identity),
                    target_identity=list(target_identity),
                    expected_size=len(expected_manifest),
                    expected_digest=hashlib.sha256(expected_manifest).hexdigest(),
                    observed_size=len(manifest),
                    observed_digest=hashlib.sha256(manifest).hexdigest(),
                )
            raise _ObservationUncertain("created manifest differs")
        if (
            _fd_directory_identity(store_fd) != root_identity
            or _path_directory_identity(root) != root_identity
            or _fd_directory_identity(bundle_fd) != target_identity
            or _path_directory_identity(target) != target_identity
        ):
            raise _ObservationUncertain("publication identity changed during verification")
        return manifest


__all__ = [
    "BUNDLE_KIND", "STORAGE_CLAIM", "DEFINITION_PATHS", "MANIFEST_NAME",
    "DefinitionStoreContractError", "DefinitionStoreInputError",
    "DefinitionBundleMember", "DefinitionBundleRequest", "DefinitionStoreReceipt",
    "CreateOnlyDefinitionBundleStore", "revalidate_definition_bundle_request",
]
