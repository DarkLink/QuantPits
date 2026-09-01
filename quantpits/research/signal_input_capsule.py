"""Create-only retention of one sealed cycle's critical signal bytes."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import weakref
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from quantpits.evidence.inspection import SourceMutationObserver, contained_path
from quantpits.research import decision_surface as _phase37
from quantpits.research.forward_bootstrap import _accept_exact_create


SCHEMA_VERSION = 1
CAPSULE_KIND = "WEEKLY_CRITICAL_SIGNAL_INPUT_CAPSULE_V1"
STORAGE_CLAIM = "CREATE_ONLY_EXACT_BYTES"
AUTHORIZATION_ACTION = "PUBLISH_ONE_WEEKLY_CRITICAL_SIGNAL_INPUT_CAPSULE_V1"
ROLES = (
    "source_prediction_0", "source_prediction_1", "source_prediction_2",
    "source_prediction_3", "ensemble_prediction",
)
MEMBER_NAMES = tuple("%s.bin" % role for role in ROLES)
RECEIPT_NAME = "source_receipt.json"
MANIFEST_NAME = "capsule_manifest.json"
FINAL_NAMES = tuple(sorted((RECEIPT_NAME,) + MEMBER_NAMES + (MANIFEST_NAME,)))
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID_RE = re.compile(r"^signalcapsule\.[0-9a-f]{64}$")
_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_REQUEST_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()


class SignalInputCapsuleContractError(ValueError):
    reason_code = "CONTRACT_INVALID"


class SignalInputCapsuleInputError(SignalInputCapsuleContractError):
    reason_code = "PRECONDITION_BLOCKED"


class _Blocked(SignalInputCapsuleInputError):
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
        raise SignalInputCapsuleContractError("value is not canonical JSON") from exc


def _digest(data: bytes, domain: str = "raw_bytes") -> Dict[str, Any]:
    if domain == "raw_bytes":
        return TypedDigest.raw(data).to_dict()
    return TypedDigest.canonical(data, domain).to_dict()


def _typed_digest(value: Any, name: str, domain: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        raise _Blocked(name.upper() + "_DIGEST_INVALID")
    try:
        result = TypedDigest(**value)
    except Exception as exc:
        raise _Blocked(name.upper() + "_DIGEST_INVALID") from exc
    if result.domain != domain:
        raise _Blocked(name.upper() + "_DIGEST_INVALID")
    return result.to_dict()


def _cycle(value: Any) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise SignalInputCapsuleContractError("cycle_id is invalid")
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise SignalInputCapsuleContractError("cycle_id is invalid") from exc
    return value


def _physical_directory(value: Any, name: str, *, private: bool = False) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise SignalInputCapsuleContractError("%s must be an absolute path" % name)
    path = Path(value)
    if not path.is_absolute():
        raise SignalInputCapsuleContractError("%s must be an absolute path" % name)
    try:
        info = os.lstat(str(path))
        resolved = path.resolve(strict=True)
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise SignalInputCapsuleInputError("%s is unavailable" % name) from exc
    if (
        resolved != path or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
        or private and stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise SignalInputCapsuleInputError("%s is not a physical directory" % name)
    return path


def _identity(path: Path, *, private: bool = False) -> Tuple[int, int, int, int, int]:
    info = os.lstat(str(path))
    if (
        stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
        or private and stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise _Uncertain("directory identity is invalid")
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _close_fd(fd: int, *, suppress: bool = False) -> None:
    active = sys.exc_info()[1]
    try:
        os.close(fd)
    except _PROCESS_CONTROL:
        if isinstance(active, _PROCESS_CONTROL):
            return
        raise
    except OSError:
        if not suppress and active is None:
            raise


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


def _entry_observer(root: Path, name: str) -> SourceMutationObserver:
    """Watch one public child name without resolving the child itself."""
    guard = SourceMutationObserver(root, ())
    try:
        if guard.supported:
            guard._watch(root, guard._PARENT_MASK, {name}, False)
    except _PROCESS_CONTROL:
        _close_guard(guard)
        raise
    except Exception:
        guard.supported = False
    return guard


def _accept_created_file(
    guard: SourceMutationObserver, name: str, data: bytes,
) -> bool:
    if data:
        return _accept_exact_create(guard, name, directory=False)
    descriptor = getattr(guard, "fd", -1)
    event_type = getattr(guard, "_EVENT", None)
    watches = getattr(guard, "_watches", None)
    bad_global = getattr(guard, "_BAD_GLOBAL", 0)
    if not guard.supported or descriptor < 0 or event_type is None or type(watches) is not dict:
        return False
    state = "EXPECT_CREATE"
    while True:
        try:
            events = os.read(descriptor, 64 * 1024)
        except BlockingIOError:
            break
        except OSError:
            return False
        if not events:
            break
        offset = 0
        while offset + event_type.size <= len(events):
            watch, mask, _cookie, length = event_type.unpack_from(events, offset)
            offset += event_type.size
            if offset + length > len(events):
                return False
            observed_name = events[offset:offset + length].split(b"\0", 1)[0].decode(
                "utf-8", "surrogateescape",
            )
            offset += length
            rule = watches.get(watch)
            relevant = bool(mask & bad_global or rule and (rule["any"] or observed_name in rule["names"]))
            if not relevant:
                continue
            if observed_name != name:
                return False
            if state == "EXPECT_CREATE" and mask == 0x00000100:
                state = "EXPECT_CLOSE"
            elif state == "EXPECT_CLOSE" and mask == 0x00000008:
                state = "CLOSED"
            else:
                return False
        if offset != len(events):
            return False
    return state == "CLOSED"


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        count = os.write(fd, data[offset:])
        if count <= 0:
            raise _Uncertain("write made no progress")
        offset += count


def _read_member_at(
    root_fd: int, name: str, maximum: int, *, private: bool = True,
) -> Tuple[bytes, Tuple[int, ...]]:
    fd = -1
    try:
        before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or private and stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > maximum
        ):
            raise _Conflict("member metadata differs")
        fd = os.open(
            name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
        opened = os.fstat(fd)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
            row.st_size, row.st_mtime_ns,
        )
        if identity(opened) != identity(before):
            raise _Uncertain("member open drift")
        chunks = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if len(data) != before.st_size or identity(after) != identity(before):
            raise _Uncertain("member read drift")
        return data, identity(before)
    finally:
        if fd >= 0:
            _close_fd(fd, suppress=False)


def _read_live(root: Path, logical: str, expected: Mapping[str, Any]) -> bytes:
    try:
        path = contained_path(root, logical)
        if path == root or path.is_symlink():
            raise _Blocked("SIGNAL_LOCATOR_INVALID")
        parent = path.parent
        parent_fd = os.open(
            str(parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            data, _ = _read_member_at(
                parent_fd, path.name, expected["size_bytes"], private=False,
            )
        finally:
            _close_fd(parent_fd, suppress=False)
    except _PROCESS_CONTROL:
        raise
    except _Blocked:
        raise
    except Exception as exc:
        raise _Blocked("SIGNAL_SOURCE_READ_FAILED") from exc
    if TypedDigest.raw(data).to_dict() != dict(expected):
        raise _Blocked("SIGNAL_SOURCE_DIGEST_MISMATCH")
    return data


def _run_context(cycle_path: Path, manifest: Mapping[str, Any]) -> None:
    commands = {
        "prediction": {"static_train", "static-train"},
        "post_trade": {"prod_post_trade", "post-trade"},
        "ensemble": {"ensemble_fusion", "ensemble-fusion"},
        "order": {"order_gen", "order-gen"},
    }
    for evidence_class in ("prediction", "post_trade", "ensemble", "order"):
        raw = _phase37._run_manifest(cycle_path, manifest, evidence_class)
        if (
            type(raw.get("schema_version")) is not int or raw["schema_version"] != 1
            or raw.get("status") != "success" or raw.get("command") not in commands[evidence_class]
            or type(raw.get("run_id")) is not str or not raw["run_id"]
        ):
            raise _Blocked("RUN_CONTEXT_INVALID")
    _phase37._source_projection(manifest)
    _phase37._prediction_projection(manifest)
    _phase37._ensemble_projection(cycle_path, manifest)
    _phase37._market_projection(manifest)
    ranking = manifest.get("ranking")
    if (
        type(ranking) is not dict or ranking.get("status") != "complete"
        or ranking.get("ranking_digest") is None
    ):
        raise _Blocked("RANKING_CONTEXT_INVALID")
    material = manifest.get("data_identity", {}).get("qlib_materialization_identity")
    if type(material) is not dict or material.get("status") != "observed":
        raise _Blocked("UNIVERSE_CONTEXT_INVALID")
    referenced = manifest.get("referenced_evidence")
    if type(referenced) is not list:
        raise _Blocked("CONFIG_CONTEXT_INVALID")
    for row in referenced:
        if type(row) is not dict:
            raise _Blocked("CONFIG_CONTEXT_INVALID")
        if row.get("kind") in {"config", "reference"}:
            observation = row.get("observation")
            if (
                type(observation) is not dict or observation.get("status") != "observed"
                or observation.get("preservation_status") != "embedded"
            ):
                raise _Blocked("CONFIG_CONTEXT_INVALID")


def _artifact_rows(manifest: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    lineage = manifest.get("model_and_ensemble_lineage")
    artifacts = lineage.get("source_artifacts") if type(lineage) is dict else None
    if type(artifacts) is not list:
        raise _Blocked("SIGNAL_ARTIFACT_PARTITION_INVALID")
    predictions: Dict[Any, Mapping[str, Any]] = {}
    for row in artifacts:
        if type(row) is not dict:
            raise _Blocked("SIGNAL_ARTIFACT_REMAINDER_INVALID")
        position = row.get("position")
        if "role" not in row and (type(position) is int and position in range(4) or position == "ensemble"):
            if position in predictions:
                raise _Blocked("SIGNAL_ARTIFACT_DUPLICATE")
            predictions[position] = row
    if set(predictions) != {0, 1, 2, 3, "ensemble"}:
        raise _Blocked("SIGNAL_ARTIFACT_PARTITION_INVALID")
    return tuple(predictions[position] for position in (0, 1, 2, 3, "ensemble"))


def _member_metadata(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    result = []
    for role, row in zip(ROLES, rows):
        members = row.get("members")
        if type(members) is not list:
            raise _Blocked("SIGNAL_MEMBER_INVENTORY_INVALID")
        matches = [
            member for member in members
            if type(member) is dict and Path(str(member.get("path", ""))).name == "pred.pkl"
        ]
        if len(matches) != 1:
            raise _Blocked("SIGNAL_PRED_MEMBER_AMBIGUOUS")
        member = matches[0]
        digest = _typed_digest(member.get("digest"), role, "raw_bytes")
        if (
            member.get("status") != "observed"
            or member.get("preservation_status") not in {"embedded", "workspace_file"}
            or type(member.get("path")) is not str
        ):
            raise _Blocked("SIGNAL_MEMBER_INCOMPARABLE")
        result.append({
            "role": role, "preservation_status": member["preservation_status"],
            "logical_locator": member["path"], "raw_digest": digest,
            "size_bytes": digest["size_bytes"],
        })
    if any(row["size_bytes"] > MAX_MEMBER_BYTES for row in result):
        raise _Blocked("SIGNAL_MEMBER_BUDGET_EXCEEDED")
    if sum(row["size_bytes"] for row in result) > MAX_TOTAL_BYTES:
        raise _Blocked("SIGNAL_TOTAL_BUDGET_EXCEEDED")
    return tuple(result)


def _source_receipt(
    cycle_id: str, manifest: Mapping[str, Any], manifest_digest: Mapping[str, Any],
    seal_digest: Mapping[str, Any], members: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    problems = manifest.get("problems")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_claim": "EXACT_PHASE37A_SIGNAL_INPUTS_V1",
        "cycle_id": cycle_id,
        "phase37a_status": manifest["status"],
        "problem_inventory_digest": TypedDigest.canonical(problems).to_dict(),
        "phase37a_seal_digest": dict(seal_digest),
        "phase37a_manifest_digest": dict(manifest_digest),
        "run_context_verified": True,
        "config_context_verified": True,
        "ranking_context_verified": True,
        "universe_identity_verified": True,
        "requested_member_count": 5,
        "members": [{
            "role": row["role"],
            "original_preservation": row["preservation_status"],
            "raw_digest": dict(row["raw_digest"]),
            "size_bytes": row["size_bytes"],
        } for row in members],
        "source_observation_complete": True,
        "did_write": False,
        "definition_bound": False,
        "same_champion_segment": False,
    }


class _Request:
    def __init__(
        self, cycle_id: str, metadata: Sequence[Mapping[str, Any]],
        data: Optional[Sequence[bytes]], receipt: Mapping[str, Any], *, _authority: Any,
    ) -> None:
        if _authority is not _AUTHORITY:
            raise SignalInputCapsuleContractError("requests are inspector-owned")
        self.cycle_id = cycle_id
        self.metadata = tuple(dict(row) for row in metadata)
        self.data = None if data is None else tuple(data)
        self.receipt = dict(receipt)
        identity = {
            "protocol": CAPSULE_KIND,
            "cycle_id": cycle_id,
            "phase37a_seal_digest": receipt["phase37a_seal_digest"],
            "phase37a_manifest_digest": receipt["phase37a_manifest_digest"],
            "members": [{
                "role": row["role"], "raw_digest": row["raw_digest"],
                "size_bytes": row["size_bytes"],
            } for row in self.metadata],
        }
        self.capsule_id = "signalcapsule." + hashlib.sha256(_canonical(identity)).hexdigest()
        request = {**identity, "source_receipt_digest": TypedDigest.raw(
            _canonical(self.receipt),
        ).to_dict()}
        self.request_digest = TypedDigest.canonical(request).to_dict()
        self._validate(False)
        _REQUEST_BINDINGS[self] = self._binding()

    def _binding(self) -> bytes:
        return _canonical({
            "cycle_id": self.cycle_id,
            "metadata": self.metadata,
            "data_digests": None if self.data is None else [
                TypedDigest.raw(value).to_dict() for value in self.data
            ],
            "receipt": self.receipt,
            "capsule_id": self.capsule_id,
            "request_digest": self.request_digest,
        })

    def _validate(self, binding: bool = True) -> None:
        _cycle(self.cycle_id)
        if (
            len(self.metadata) != 5
            or tuple(row.get("role") for row in self.metadata) != ROLES
            or self.data is not None and len(self.data) != 5
            or not _ID_RE.fullmatch(self.capsule_id)
        ):
            raise SignalInputCapsuleContractError("request member set is invalid")
        for index, row in enumerate(self.metadata):
            if set(row) != {
                "role", "preservation_status", "logical_locator", "raw_digest", "size_bytes",
            } or row["preservation_status"] not in {"embedded", "workspace_file"}:
                raise SignalInputCapsuleContractError("request member is invalid")
            digest = _typed_digest(row["raw_digest"], row["role"], "raw_bytes")
            if row["size_bytes"] != digest["size_bytes"]:
                raise SignalInputCapsuleContractError("request member size is invalid")
            if self.data is not None and TypedDigest.raw(self.data[index]).to_dict() != digest:
                raise SignalInputCapsuleContractError("request member bytes are invalid")
        if binding and _REQUEST_BINDINGS.get(self) != self._binding():
            raise SignalInputCapsuleContractError("request authority is absent")

    def member_bytes(self) -> Tuple[Tuple[str, bytes], ...]:
        self._validate()
        if self.data is None:
            raise SignalInputCapsuleInputError("live signal bytes were not admitted")
        return ((RECEIPT_NAME, _canonical(self.receipt)),) + tuple(
            (name, data) for name, data in zip(MEMBER_NAMES, self.data)
        )


def _derive_request(
    production: Path, cycle_id: str, *, read_live: bool,
    existing_target: Optional[Path] = None,
) -> _Request:
    cycle_relative = "data/evidence/v1/cycles/%s" % cycle_id
    cycle_guard = SourceMutationObserver(production, (cycle_relative,))
    live_guard: Optional[SourceMutationObserver] = None
    try:
        if not cycle_guard.supported:
            raise _Blocked("SOURCE_CONTINUITY_UNSUPPORTED")
        cycle_path, manifest, _seal = _phase37._cycle_authority(production, cycle_id)
        _run_context(cycle_path, manifest)
        rows = _artifact_rows(manifest)
        metadata = _member_metadata(rows)
        live_paths = tuple(
            row["logical_locator"] for row in metadata
            if row["preservation_status"] == "workspace_file"
        )
        if read_live and live_paths:
            live_guard = SourceMutationObserver(production, live_paths)
            if not live_guard.supported:
                raise _Blocked("SOURCE_CONTINUITY_UNSUPPORTED")
        manifest_data, _ = _phase37._read_regular(cycle_path / "manifest.json", private=True)
        seal_data, _ = _phase37._read_regular(cycle_path / "seal.json", private=True)
        receipt = _source_receipt(
            cycle_id, manifest, TypedDigest.raw(manifest_data).to_dict(),
            TypedDigest.raw(seal_data).to_dict(), metadata,
        )
        data = None
        if read_live:
            values = []
            failures = []
            for row in metadata:
                try:
                    expected = row["raw_digest"]
                    if row["preservation_status"] == "embedded":
                        path = cycle_path / "objects" / expected["value"][:2] / expected["value"]
                        value, _ = _phase37._read_regular(path, maximum=expected["size_bytes"], private=True)
                        if TypedDigest.raw(value).to_dict() != expected:
                            raise _Blocked("SIGNAL_SOURCE_DIGEST_MISMATCH")
                    else:
                        value = _read_live(production, row["logical_locator"], expected)
                    values.append(value)
                except _PROCESS_CONTROL:
                    raise
                except _Blocked as exc:
                    failures.append(exc.reason_code)
                    values.append(b"")
                except Exception:
                    failures.append("SIGNAL_SOURCE_READ_FAILED")
                    values.append(b"")
            if failures:
                raise _Blocked(failures[0])
            data = tuple(values)
        request = _Request(cycle_id, metadata, data, receipt, _authority=_AUTHORITY)
        if existing_target is not None and existing_target.name != request.capsule_id:
            raise _Conflict("capsule identifier differs")
        if (
            cycle_guard.mutated() or live_guard is not None and live_guard.mutated()
        ):
            raise _Blocked("SOURCE_CONTINUITY_LOST")
        return request
    except _PROCESS_CONTROL:
        raise
    except (_Blocked, _Conflict):
        raise
    except _phase37._ComponentIncomparable as exc:
        raise _Blocked(exc.reason_code) from exc
    except Exception as exc:
        raise _Blocked("SOURCE_ADMISSION_FAILED") from exc
    finally:
        _close_guard(live_guard)
        _close_guard(cycle_guard)


def _manifest(request: _Request, members: Sequence[Tuple[str, bytes]]) -> bytes:
    rows = [{
        "logical_path": name, "raw_digest": TypedDigest.raw(data).to_dict(),
        "size_bytes": len(data),
    } for name, data in members]
    return _canonical({
        "schema_version": SCHEMA_VERSION,
        "capsule_kind": CAPSULE_KIND,
        "storage_claim": STORAGE_CLAIM,
        "capsule_id": request.capsule_id,
        "cycle_id": request.cycle_id,
        "source_receipt_digest": TypedDigest.raw(_canonical(request.receipt)).to_dict(),
        "member_count": 6,
        "members": rows,
        "critical_signal_retention_complete": True,
        "definition_bound": False,
        "intent_capability": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
    })


class SignalInputCapsulePlan:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise SignalInputCapsuleContractError("plans are inspector-owned")
        if set(kwargs) != {"request", "status", "reason_code", "cycle_id"}:
            raise SignalInputCapsuleContractError("plan fields are invalid")
        self._request = kwargs["request"]
        self.status = kwargs["status"]
        self.reason_code = kwargs["reason_code"]
        self.cycle_id = kwargs["cycle_id"]
        self._validate(False)
        _BINDINGS[self] = _canonical(self.to_safe_dict(_binding=False))

    def _validate(self, binding: bool = True) -> None:
        prepared = self.status == "PREPARED"
        if (
            self.status not in {"PREPARED", "PRECONDITION_BLOCKED"}
            or prepared != (type(self._request) is _Request)
            or prepared != (self.reason_code == "PREPARED")
            or type(self.reason_code) is not str or not self.reason_code
            or _cycle(self.cycle_id) != self.cycle_id
        ):
            raise SignalInputCapsuleContractError("plan is inconsistent")
        if prepared:
            self._request._validate()
            if self._request.cycle_id != self.cycle_id:
                raise SignalInputCapsuleContractError("plan cycle is inconsistent")
        if binding and _BINDINGS.get(self) != _canonical(self.to_safe_dict(_binding=False)):
            raise SignalInputCapsuleContractError("plan authority is absent")

    @property
    def capsule_id(self) -> Optional[str]:
        self._validate()
        return None if self._request is None else self._request.capsule_id

    @property
    def request_digest(self) -> Optional[Mapping[str, Any]]:
        self._validate()
        return None if self._request is None else MappingProxyType(dict(self._request.request_digest))

    def to_safe_dict(self, _binding: bool = True) -> Dict[str, Any]:
        if _binding:
            self._validate()
        if self._request is None:
            return _blocked_payload(self.cycle_id, self.reason_code)
        return _safe_payload(self.status, self.reason_code, self._request, False, False)


class SignalInputCapsuleResult:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise SignalInputCapsuleContractError("results are inspector-owned")
        expected = {
            "request", "status", "reason_code", "did_write", "verified_count",
            "manifest_digest", "cycle_id",
        }
        if set(kwargs) != expected:
            raise SignalInputCapsuleContractError("result fields are invalid")
        for key, value in kwargs.items():
            setattr(self, "_request" if key == "request" else key, value)
        self._validate(False)
        _BINDINGS[self] = _canonical(self.to_safe_dict(_binding=False))

    def _validate(self, binding: bool = True) -> None:
        positive = self.status in {"COMMITTED", "ADOPTED"}
        reasons = {
            "COMMITTED": "COMMITTED", "ADOPTED": "EXACT_CAPSULE_ALREADY_PRESENT",
            "CONFLICT": "CAPSULE_CONFLICT", "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN",
        }
        blocked = self.status == "PRECONDITION_BLOCKED"
        if (
            self.status not in {*reasons, "PRECONDITION_BLOCKED"}
            or blocked != (self._request is None)
            or not blocked and type(self._request) is not _Request
            or not blocked and self.reason_code != reasons[self.status]
            or blocked and (type(self.reason_code) is not str or not self.reason_code)
            or type(self.did_write) is not bool
            or type(self.verified_count) is not int or not 0 <= self.verified_count <= 7
            or positive != (self.verified_count == 7 and self.manifest_digest is not None)
            or self.did_write is not (self.status == "COMMITTED") and self.status != "UNCERTAIN"
            or blocked and (self.did_write or self.verified_count or self.manifest_digest is not None)
            or _cycle(self.cycle_id) != self.cycle_id
        ):
            raise SignalInputCapsuleContractError("result is inconsistent")
        if self._request is not None:
            self._request._validate()
            if self._request.cycle_id != self.cycle_id:
                raise SignalInputCapsuleContractError("result cycle is inconsistent")
        if self.manifest_digest is not None:
            _typed_digest(dict(self.manifest_digest), "manifest", "raw_bytes")
        if binding and _BINDINGS.get(self) != _canonical(self.to_safe_dict(_binding=False)):
            raise SignalInputCapsuleContractError("result authority is absent")

    @property
    def critical_signal_retention_complete(self) -> bool:
        self._validate()
        return self.status in {"COMMITTED", "ADOPTED"}

    def to_safe_dict(self, _binding: bool = True) -> Dict[str, Any]:
        if _binding:
            self._validate()
        if self._request is None:
            return _blocked_payload(self.cycle_id, self.reason_code)
        return _safe_payload(
            self.status, self.reason_code, self._request, self.did_write,
            self.status in {"COMMITTED", "ADOPTED"},
            verified_count=self.verified_count,
            manifest_digest=self.manifest_digest,
        )


def _safe_payload(
    status: str, reason: str, request: _Request, did_write: bool, complete: bool,
    *, verified_count: int = 0, manifest_digest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    embedded = sum(row["preservation_status"] == "embedded" for row in request.metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_kind": CAPSULE_KIND,
        "status": status,
        "reason_code": reason,
        "cycle_id": request.cycle_id,
        "capsule_id": request.capsule_id,
        "request_digest": dict(request.request_digest),
        "requested_member_count": 5,
        "verified_file_count": verified_count,
        "total_signal_bytes": sum(row["size_bytes"] for row in request.metadata),
        "embedded_source_count": embedded,
        "workspace_source_count": 5 - embedded,
        "manifest_digest": None if manifest_digest is None else dict(manifest_digest),
        "critical_signal_retention_complete": complete,
        "same_champion_segment": False,
        "definition_bound": False,
        "intent_capability": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
        "did_write": did_write,
    }


def _blocked_payload(cycle_id: str, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_kind": CAPSULE_KIND,
        "status": "PRECONDITION_BLOCKED",
        "reason_code": reason,
        "cycle_id": cycle_id,
        "capsule_id": None,
        "request_digest": None,
        "requested_member_count": 5,
        "requested_members": [
            {"role": role, "status": "blocked"} for role in ROLES
        ],
        "verified_file_count": 0,
        "total_signal_bytes": None,
        "embedded_source_count": None,
        "workspace_source_count": None,
        "manifest_digest": None,
        "critical_signal_retention_complete": False,
        "same_champion_segment": False,
        "definition_bound": False,
        "intent_capability": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
        "did_write": False,
    }


def _result(request: _Request, status: str, did_write: bool, count: int, manifest: Optional[bytes] = None) -> SignalInputCapsuleResult:
    reasons = {
        "COMMITTED": "COMMITTED", "ADOPTED": "EXACT_CAPSULE_ALREADY_PRESENT",
        "CONFLICT": "CAPSULE_CONFLICT", "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN",
    }
    return SignalInputCapsuleResult(
        _authority=_AUTHORITY, request=request, status=status, reason_code=reasons[status],
        did_write=did_write, verified_count=count,
        manifest_digest=None if manifest is None else TypedDigest.raw(manifest).to_dict(),
        cycle_id=request.cycle_id,
    )


def _blocked_result(cycle_id: str, reason: str) -> SignalInputCapsuleResult:
    return SignalInputCapsuleResult(
        _authority=_AUTHORITY, request=None, status="PRECONDITION_BLOCKED",
        reason_code=reason, did_write=False, verified_count=0,
        manifest_digest=None, cycle_id=cycle_id,
    )


def _verify_bundle(target: Path, request: _Request, *, use_request_data: bool) -> bytes:
    fd = -1
    try:
        info = os.lstat(str(target))
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
            raise _Conflict("target metadata differs")
        fd = os.open(str(target), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        if tuple(sorted(os.listdir(fd))) != FINAL_NAMES:
            raise _Conflict("target inventory differs")
        receipt, _ = _read_member_at(fd, RECEIPT_NAME, 16 * 1024 * 1024)
        if receipt != _canonical(request.receipt):
            raise _Conflict("receipt differs")
        members = [(RECEIPT_NAME, receipt)]
        for index, (name, metadata) in enumerate(zip(MEMBER_NAMES, request.metadata)):
            data, _ = _read_member_at(fd, name, MAX_MEMBER_BYTES)
            if TypedDigest.raw(data).to_dict() != metadata["raw_digest"]:
                raise _Conflict("signal differs")
            if use_request_data and request.data is not None and data != request.data[index]:
                raise _Conflict("signal differs")
            members.append((name, data))
        expected_manifest = _manifest(request, members)
        manifest, _ = _read_member_at(fd, MANIFEST_NAME, 16 * 1024 * 1024)
        if manifest != expected_manifest:
            raise _Conflict("manifest differs")
        return manifest
    finally:
        if fd >= 0:
            _close_fd(fd, suppress=False)


def _publish_store(root: Path, request: _Request) -> SignalInputCapsuleResult:
    request._validate()
    target = root / request.capsule_id
    entry_guard = _entry_observer(root, request.capsule_id)
    root_guard = SourceMutationObserver(root, ())
    parent_guard = SourceMutationObserver(root.parent, ())
    root_before = _identity(root, private=True)
    parent_before = _identity(root.parent)
    did_create = False
    prefix = 0
    root_fd = bundle_fd = -1
    child_guard: Optional[SourceMutationObserver] = None
    created = {}
    try:
        if not entry_guard.supported or not root_guard.supported or not parent_guard.supported:
            raise SignalInputCapsuleInputError("publication continuity is unsupported")
        if os.path.lexists(str(target)):
            try:
                target_info = os.lstat(str(target))
            except OSError:
                return _result(request, "UNCERTAIN", False, 0)
            if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISDIR(target_info.st_mode):
                return _result(request, "CONFLICT", False, 0)
        root_fd = os.open(str(root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.mkdir(request.capsule_id, 0o700, dir_fd=root_fd)
            did_create = True
            if not _accept_exact_create(entry_guard, request.capsule_id, directory=True):
                raise _Uncertain("target create continuity failed")
        except FileExistsError:
            try:
                child_guard = SourceMutationObserver(target, FINAL_NAMES)
                if not child_guard.supported:
                    raise _Uncertain("existing target continuity is unsupported")
                target_before = _identity(target, private=True)
                manifest = _verify_bundle(target, request, use_request_data=True)
                if (
                    _identity(root, private=True) != root_before
                    or _identity(root.parent) != parent_before
                    or _identity(target, private=True) != target_before
                    or entry_guard.mutated() or root_guard.mutated()
                    or parent_guard.mutated() or child_guard.mutated()
                ):
                    raise _Uncertain("existing target continuity failed")
                return _result(request, "ADOPTED", False, 7, manifest)
            except _Conflict:
                return _result(request, "CONFLICT", False, 0)
            except _PROCESS_CONTROL:
                raise
            except Exception:
                return _result(request, "UNCERTAIN", False, 0)
        bundle_fd = os.open(request.capsule_id, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
        child_guard = SourceMutationObserver(target, (RECEIPT_NAME,) + MEMBER_NAMES + (MANIFEST_NAME,))
        if not child_guard.supported or os.listdir(bundle_fd):
            raise _Uncertain("target child continuity failed")
        members = request.member_bytes()
        for name, data in members:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=bundle_fd)
            try:
                info = os.fstat(fd)
                identity = (info.st_dev, info.st_ino, info.st_mode, info.st_nlink)
                _write_all(fd, data)
                os.fsync(fd)
            finally:
                _close_fd(fd, suppress=False)
            if not _accept_created_file(child_guard, name, data):
                raise _Uncertain("member create continuity failed")
            observed, observed_identity = _read_member_at(bundle_fd, name, max(len(data), 1))
            if observed != data or observed_identity[:4] != identity:
                raise _Uncertain("created member differs")
            created[name] = identity
            prefix += 1
        manifest = _manifest(request, members)
        fd = os.open(MANIFEST_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=bundle_fd)
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino, info.st_mode, info.st_nlink)
            _write_all(fd, manifest)
            os.fsync(fd)
        finally:
            _close_fd(fd, suppress=False)
        if not _accept_exact_create(child_guard, MANIFEST_NAME, directory=False):
            raise _Uncertain("manifest create continuity failed")
        observed, observed_identity = _read_member_at(bundle_fd, MANIFEST_NAME, max(len(manifest), 1))
        if observed != manifest or observed_identity[:4] != identity:
            raise _Uncertain("created manifest differs")
        created[MANIFEST_NAME] = identity
        prefix += 1
        os.fsync(bundle_fd)
        os.fsync(root_fd)
        target_identity = _identity(target, private=True)
        for name in (RECEIPT_NAME,) + MEMBER_NAMES + (MANIFEST_NAME,):
            info = os.stat(name, dir_fd=bundle_fd, follow_symlinks=False)
            identity = (info.st_dev, info.st_ino, info.st_mode, info.st_nlink)
            if identity != created[name]:
                raise _Uncertain("created member identity drift")
        _close_fd(bundle_fd, suppress=False)
        bundle_fd = -1
        _close_fd(root_fd, suppress=False)
        root_fd = -1
        verified_manifest = _verify_bundle(target, request, use_request_data=True)
        if verified_manifest != manifest:
            raise _Uncertain("final manifest verification differs")
        if (
            _identity(root, private=True) != root_before or _identity(target, private=True) != target_identity
            or _identity(root.parent) != parent_before
            or entry_guard.mutated() or root_guard.mutated()
            or parent_guard.mutated() or child_guard.mutated()
        ):
            raise _Uncertain("final namespace continuity failed")
        return _result(request, "COMMITTED", True, 7, manifest)
    except _PROCESS_CONTROL:
        raise
    except _Conflict:
        if did_create:
            return _result(request, "UNCERTAIN", True, prefix)
        return _result(request, "CONFLICT", False, 0)
    except Exception:
        if not did_create:
            raise SignalInputCapsuleInputError("publication failed before target creation")
        return _result(request, "UNCERTAIN", True, prefix)
    finally:
        if bundle_fd >= 0:
            _close_fd(bundle_fd, suppress=True)
        if root_fd >= 0:
            _close_fd(root_fd, suppress=True)
        _close_guard(child_guard)
        _close_guard(parent_guard)
        _close_guard(root_guard)
        _close_guard(entry_guard)


def _adopt_store(root: Path, request: _Request) -> SignalInputCapsuleResult:
    request._validate()
    target = root / request.capsule_id
    root_guard = _entry_observer(root, request.capsule_id)
    parent_guard = SourceMutationObserver(root.parent, ())
    target_guard: Optional[SourceMutationObserver] = None
    try:
        if not root_guard.supported or not parent_guard.supported:
            return _result(request, "UNCERTAIN", False, 0)
        if not os.path.lexists(str(target)):
            return _result(request, "CONFLICT", False, 0)
        try:
            info = os.lstat(str(target))
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                return _result(request, "CONFLICT", False, 0)
        except OSError:
            return _result(request, "UNCERTAIN", False, 0)
        target_guard = SourceMutationObserver(target, FINAL_NAMES)
        if not target_guard.supported:
            return _result(request, "UNCERTAIN", False, 0)
        root_before = _identity(root, private=True)
        parent_before = _identity(root.parent)
        target_before = _identity(target, private=True)
        try:
            manifest = _verify_bundle(target, request, use_request_data=False)
        except _PROCESS_CONTROL:
            raise
        except _Conflict:
            return _result(request, "CONFLICT", False, 0)
        except Exception:
            return _result(request, "UNCERTAIN", False, 0)
        if (
            root_guard.mutated() or parent_guard.mutated() or target_guard.mutated()
            or _identity(root, private=True) != root_before
            or _identity(root.parent) != parent_before
            or _identity(target, private=True) != target_before
        ):
            return _result(request, "UNCERTAIN", False, 0)
        return _result(request, "ADOPTED", False, 7, manifest)
    finally:
        _close_guard(target_guard)
        _close_guard(parent_guard)
        _close_guard(root_guard)


def _roots(production_root: Any, research_root: Any, store_root: Any) -> Tuple[Path, Path, Path]:
    production = _physical_directory(production_root, "production_workspace_root")
    research = _physical_directory(research_root, "research_workspace_root")
    store = _physical_directory(store_root, "capsule_store_root", private=True)
    expected = research / "research" / "shadow_v1" / "signal_input_capsules"
    if store != expected or store.parent.resolve(strict=True) != expected.parent:
        raise SignalInputCapsuleContractError("capsule_store_root does not match the fixed Research layout")
    if store == production or production in store.parents:
        raise SignalInputCapsuleContractError(
            "capsule_store_root must be physically isolated from Production",
        )
    return production, research, store


def prepare_signal_input_capsule(
    production_workspace_root: Any, research_workspace_root: Any,
    cycle_id: Any, capsule_store_root: Any,
) -> SignalInputCapsulePlan:
    production, _research, _store = _roots(
        production_workspace_root, research_workspace_root, capsule_store_root,
    )
    canonical_cycle = _cycle(cycle_id)
    try:
        request = _derive_request(production, canonical_cycle, read_live=True)
    except _PROCESS_CONTROL:
        raise
    except _Blocked as exc:
        return SignalInputCapsulePlan(
            _authority=_AUTHORITY, request=None, status="PRECONDITION_BLOCKED",
            reason_code=exc.reason_code, cycle_id=canonical_cycle,
        )
    return SignalInputCapsulePlan(
        _authority=_AUTHORITY, request=request, status="PREPARED", reason_code="PREPARED",
        cycle_id=canonical_cycle,
    )


def publish_signal_input_capsule(
    production_workspace_root: Any, research_workspace_root: Any,
    cycle_id: Any, capsule_store_root: Any, expected_capsule_id: Any,
    expected_request_digest: Any, authorization_action: Any,
) -> SignalInputCapsuleResult:
    production, _research, store = _roots(
        production_workspace_root, research_workspace_root, capsule_store_root,
    )
    if type(expected_capsule_id) is not str or _ID_RE.fullmatch(expected_capsule_id) is None:
        raise SignalInputCapsuleContractError("expected_capsule_id is invalid")
    expected_digest = _typed_digest(expected_request_digest, "expected_request", "canonical_json")
    if authorization_action != AUTHORIZATION_ACTION:
        raise SignalInputCapsuleContractError("authorization_action is invalid")
    canonical_cycle = _cycle(cycle_id)
    try:
        request = _derive_request(production, canonical_cycle, read_live=True)
    except _PROCESS_CONTROL:
        raise
    except _Blocked as exc:
        return _blocked_result(canonical_cycle, exc.reason_code)
    if request.capsule_id != expected_capsule_id or request.request_digest != expected_digest:
        raise SignalInputCapsuleInputError("prepared request no longer matches")
    return _publish_store(store, request)


def adopt_signal_input_capsule(
    production_workspace_root: Any, research_workspace_root: Any,
    cycle_id: Any, capsule_store_root: Any, capsule_id: Any,
) -> SignalInputCapsuleResult:
    production, _research, store = _roots(
        production_workspace_root, research_workspace_root, capsule_store_root,
    )
    if type(capsule_id) is not str or _ID_RE.fullmatch(capsule_id) is None:
        raise SignalInputCapsuleContractError("capsule_id is invalid")
    target = store / capsule_id
    canonical_cycle = _cycle(cycle_id)
    try:
        request = _derive_request(
            production, canonical_cycle, read_live=False, existing_target=target,
        )
    except _PROCESS_CONTROL:
        raise
    except _Blocked as exc:
        return _blocked_result(canonical_cycle, exc.reason_code)
    except _Conflict:
        metadata_request = _derive_request(production, canonical_cycle, read_live=False)
        return _result(metadata_request, "CONFLICT", False, 0)
    return _adopt_store(store, request)


__all__ = [
    "AUTHORIZATION_ACTION", "SignalInputCapsuleContractError",
    "SignalInputCapsuleInputError", "SignalInputCapsulePlan",
    "SignalInputCapsuleResult", "prepare_signal_input_capsule",
    "publish_signal_input_capsule", "adopt_signal_input_capsule",
]
