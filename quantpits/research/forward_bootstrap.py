"""Create-only matched Champion--Challenger portfolio bootstrap bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import weakref
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from quantpits.evidence.inspection import SourceMutationObserver


AUTHORIZATION_ACTION = "PUBLISH_ONE_MATCHED_FORWARD_BOOTSTRAP_V1"
BOOTSTRAP_KIND = "MATCHED_FORWARD_BOOTSTRAP_V1"
STORAGE_CLAIM = "CREATE_ONLY_EXACT_BYTES"
MEMBER_PATHS = ("source_receipt.json", "champion_state.json", "challenger_state.json")
MANIFEST_NAME = "bootstrap_manifest.json"
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_MEMBER = 4 * 1024 * 1024
_PLAN_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_RECEIPT_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_RESULT_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()
_REQUEST_BINDINGS: "weakref.WeakKeyDictionary[Any, bytes]" = weakref.WeakKeyDictionary()


class ForwardBootstrapContractError(ValueError):
    """A caller-controlled bootstrap value violates the frozen contract."""


class ForwardBootstrapInputError(ForwardBootstrapContractError):
    """A physical input cannot be compared safely."""


class _Uncertain(Exception):
    pass


class _Conflict(Exception):
    def __init__(self, fingerprint: str) -> None:
        super().__init__(fingerprint)
        self.fingerprint = fingerprint


def _canonical(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ForwardBootstrapContractError("value is not canonical JSON") from exc


def _close_fd(descriptor: int, *, suppress: bool) -> None:
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
        raise _Uncertain("descriptor close failed") from exc


def _close_guard(guard: SourceMutationObserver) -> None:
    active = sys.exc_info()[1]
    try:
        guard.close()
    except _PROCESS_CONTROL:
        if isinstance(active, _PROCESS_CONTROL):
            return
        raise
    except OSError:
        if isinstance(active, _PROCESS_CONTROL):
            return
        raise


def _accept_exact_target_create(
    guard: SourceMutationObserver, identifier: str,
) -> bool:
    """Consume only the one CREATE event authorized for an absent target.

    The guard starts before ``mkdir``.  Reading its event stream directly here
    lets the store distinguish that authorized namespace edge from a create
    followed by rename/delete/recreate before ``mkdir`` returns.  The same
    guard remains live afterwards, so this does not replace continuity with a
    fresh observation.
    """
    descriptor = getattr(guard, "fd", -1)
    event_type = getattr(guard, "_EVENT", None)
    watches = getattr(guard, "_watches", None)
    bad_global = getattr(guard, "_BAD_GLOBAL", 0)
    if (
        not guard.supported or descriptor < 0
        or event_type is None or type(watches) is not dict
    ):
        return False
    accepted = 0
    while True:
        try:
            data = os.read(descriptor, 64 * 1024)
        except BlockingIOError:
            break
        except OSError:
            return False
        if not data:
            break
        offset = 0
        while offset + event_type.size <= len(data):
            watch, mask, _cookie, length = event_type.unpack_from(data, offset)
            offset += event_type.size
            if offset + length > len(data):
                return False
            name = data[offset:offset + length].split(b"\0", 1)[0].decode(
                "utf-8", "surrogateescape",
            )
            offset += length
            rule = watches.get(watch)
            relevant = bool(
                mask & bad_global
                or rule and (rule["any"] or name in rule["names"])
            )
            if not relevant:
                continue
            # Linux reports mkdir in a watched parent as IN_CREATE|IN_ISDIR.
            if name != identifier or mask != 0x40000100:
                return False
            accepted += 1
        if offset != len(data):
            return False
    return accepted == 1


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise _Uncertain("member write made no progress")
        offset += written


def _digest(value: Any, domain: str = "canonical_json") -> Dict[str, Any]:
    if domain == "raw_bytes":
        return TypedDigest.raw(value).to_dict()
    return TypedDigest.canonical(value, domain).to_dict()


def _typed_digest(value: Any, name: str, domain: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {"algorithm", "domain", "value", "size_bytes"}:
        raise ForwardBootstrapContractError("%s fields are invalid" % name)
    try:
        result = TypedDigest(**value)
    except Exception as exc:
        raise ForwardBootstrapContractError("%s is invalid" % name) from exc
    if result.domain != domain:
        raise ForwardBootstrapContractError("%s domain is invalid" % name)
    return result.to_dict()


def _id(value: Any, name: str) -> str:
    if (type(value) is not str or value in {".", ".."} or value != value.strip()
            or _ID_RE.fullmatch(value) is None):
        raise ForwardBootstrapContractError("%s is invalid" % name)
    return value


def _cycle(value: Any, name: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise ForwardBootstrapContractError("%s is invalid" % name)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ForwardBootstrapContractError("%s is invalid" % name) from exc
    if parsed.isoformat() != value:
        raise ForwardBootstrapContractError("%s is invalid" % name)
    return value


def _path(value: Any, name: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise ForwardBootstrapContractError("%s must be a path" % name)
    path = Path(value)
    if not path.is_absolute():
        raise ForwardBootstrapContractError("%s must be absolute" % name)
    try:
        resolved = path.resolve(strict=True)
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise ForwardBootstrapInputError("%s is unavailable" % name) from exc
    if resolved != path:
        raise ForwardBootstrapInputError("%s must be a physical canonical path" % name)
    return path


def _dir_identity(path: Path, *, private: bool = False) -> Tuple[int, int, int, int, int]:
    try:
        info = os.lstat(str(path))
    except OSError as exc:
        raise ForwardBootstrapInputError("directory is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ForwardBootstrapInputError("directory must be physical")
    if private and stat.S_IMODE(info.st_mode) != 0o700:
        raise ForwardBootstrapInputError("private directory mode must be 0700")
    # Namespace continuity is observed separately by SourceMutationObserver.  A
    # directory's ctime legitimately changes when this writer creates children.
    return info.st_dev, info.st_ino, info.st_mode, 0, 0


def _namespace_identity(path: Path) -> Tuple[int, int, int, int, int]:
    info = os.lstat(str(path))
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _Uncertain("namespace is not a physical directory")
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_ctime_ns


def _formal_paths(workspace_root: Any, activation_path: Any, definition_store_root: Any,
                  evidence_store_root: Any, bootstrap_store_root: Any) -> Tuple[Path, ...]:
    root = _path(workspace_root, "workspace_root")
    activation = _path(activation_path, "activation_path")
    definitions = _path(definition_store_root, "definition_store_root")
    evidence = _path(evidence_store_root, "evidence_store_root")
    bootstraps = _path(bootstrap_store_root, "bootstrap_store_root")
    shadow = root / "research" / "shadow_v1"
    if (activation.parent != shadow / "activations" or definitions != shadow / "definitions"
            or evidence != shadow / "definition_evidence"
            or bootstraps != shadow / "bootstraps"):
        raise ForwardBootstrapInputError("inputs do not match the formal private layout")
    for path in (root, shadow.parent, shadow, activation.parent, definitions, evidence, bootstraps):
        _dir_identity(path, private=path != root)
    info = os.lstat(str(activation))
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise ForwardBootstrapInputError("activation must be a private single-link file")
    return root, activation, definitions, evidence, bootstraps


def _source_paths(root: Path, activation: Path, definitions: Path, evidence: Path,
                  definition_id: str, source_cycle: str) -> Tuple[str, ...]:
    return tuple(path.relative_to(root).as_posix() for path in (
        activation, definitions / definition_id, evidence / definition_id,
        root / "data" / "evidence" / "v1" / "cycles" / source_cycle,
    ))


def _protected_inventory(root: Path, excluded: Optional[Path]) -> Tuple[Tuple[Any, ...], ...]:
    """Cheap whole-workspace delta guard; source bytes have dedicated observers."""
    rows = []
    def failed(exc: OSError) -> None:
        raise ForwardBootstrapInputError("workspace inventory is incomparable") from exc

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
                rows.append((path.relative_to(root).as_posix(), "D", info.st_dev, info.st_ino,
                             info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            path = current_path / name
            if excluded is not None and (path == excluded or excluded in path.parents):
                continue
            info = os.lstat(str(path))
            rows.append((path.relative_to(root).as_posix(), "F", info.st_dev, info.st_ino,
                         info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(rows)


def _state_payload(state: Any) -> Dict[str, Any]:
    raw = state.to_dict()
    if set(raw) != {"portfolio_id", "as_of_date", "cash", "positions", "digest"}:
        raise ForwardBootstrapContractError("portfolio state fields are invalid")
    return {key: raw[key] for key in ("portfolio_id", "as_of_date", "cash", "positions")}


def _economic_payload(state: Any) -> Dict[str, Any]:
    raw = _state_payload(state)
    return {key: raw[key] for key in ("as_of_date", "cash", "positions")}


def _portfolio_id(role: str, source: Any, candidate: Any) -> str:
    strategy = candidate.compiled_definitions.champion if role == "CHAMPION" else candidate.compiled_definitions.challenger
    payload = {
        "domain": "MATCHED_FORWARD_PORTFOLIO_ID_V1", "role": role,
        "bootstrap_source_cycle_id": source.cycle_id,
        "source_portfolio_raw_digest": dict(source.portfolio_raw_digest),
        "definition_set_id": candidate.definition_set_id,
        "strategy_id": strategy.to_dict()["strategy_id"],
    }
    return "portfolio." + hashlib.sha256(_canonical(payload)).hexdigest()


def _build_states(source: Any, candidate: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    from quantpits.research.accounting import ExecutionAssumption, ShadowPortfolioState
    portfolio = source.verified_portfolio()
    execution = candidate.compiled_definitions.execution_assumption.to_dict()
    assumption = ExecutionAssumption.from_dict({key: execution[key] for key in ExecutionAssumption._FIELDS})
    cash = Decimal(portfolio["current_cash"])
    if cash % assumption.money_quantum:
        raise ForwardBootstrapContractError("source cash is not aligned to money_quantum")
    positions = []
    for row in sorted(portfolio["current_holding"], key=lambda item: item["instrument"]):
        cost = Decimal(row["amount"])
        if cost % assumption.money_quantum:
            raise ForwardBootstrapContractError("source book_cost is not aligned to money_quantum")
        positions.append({
            "instrument": row["instrument"], "quantity": int(row["value"]),
            "book_cost": row["amount"],
        })
    states = []
    for role in ("CHAMPION", "CHALLENGER"):
        states.append(ShadowPortfolioState.from_dict({
            "portfolio_id": _portfolio_id(role, source, candidate),
            "as_of_date": source.cycle_id, "cash": portfolio["current_cash"],
            "positions": positions,
        }))
    if states[0].portfolio_id == states[1].portfolio_id or _economic_payload(states[0]) != _economic_payload(states[1]):
        raise ForwardBootstrapContractError("matched portfolio predicate failed")
    return states[0], states[1], assumption.to_dict()


class _Request:
    def __init__(self, *, _authority: Any, source_receipt: Mapping[str, Any], champion: Any,
                 challenger: Any, definition: Any, evidence: Any) -> None:
        if _authority is not _AUTHORITY:
            raise ForwardBootstrapContractError("bootstrap requests are inspector-owned")
        source = dict(source_receipt)
        self.source_bytes = _canonical(source)
        self.champion_bytes = _canonical(champion.to_dict())
        self.challenger_bytes = _canonical(challenger.to_dict())
        economic = _digest(_economic_payload(champion))
        if economic != _digest(_economic_payload(challenger)):
            raise ForwardBootstrapContractError("economic states do not match")
        self.definition_set_id = definition.definition_set_id
        self.definition_evidence_cycle_id = evidence.evidence_cycle_id
        self.bootstrap_source_cycle_id = source["bootstrap_source_cycle_id"]
        self.economic_state_digest = MappingProxyType(economic)
        self.roles = (
            MappingProxyType({"role": "CHAMPION", "strategy_id": definition.compiled_definitions.champion.to_dict()["strategy_id"],
                              "portfolio_id": champion.portfolio_id, "state_digest": _digest(champion.to_dict())}),
            MappingProxyType({"role": "CHALLENGER", "strategy_id": definition.compiled_definitions.challenger.to_dict()["strategy_id"],
                              "portfolio_id": challenger.portfolio_id, "state_digest": _digest(challenger.to_dict())}),
        )
        identity = {
            "domain": "MATCHED_FORWARD_BOOTSTRAP_SET_V1",
            "definition_set_id": self.definition_set_id,
            "definition_evidence_cycle_id": self.definition_evidence_cycle_id,
            "bootstrap_source_cycle_id": self.bootstrap_source_cycle_id,
            "definition_request_digest": dict(evidence.definition_receipt.request_digest),
            "definition_evidence_request_digest": dict(evidence.evidence_receipt.request_digest),
            "definition_evidence_manifest_digest": dict(evidence.evidence_receipt.manifest_digest),
            "definition_evidence_operation_id": source["definition_evidence_operation_id"],
            "phase37a_seal_digest": source["phase37a_seal_digest"],
            "phase37a_manifest_digest": source["phase37a_manifest_digest"],
            "source_portfolio_raw_digest": source["source_portfolio_raw_digest"],
            "source_portfolio_semantic_digest": source["source_portfolio_semantic_digest"],
            "source_observation_status": source["source_observation_status"],
            "source_cycle_status": source["source_cycle_status"],
            "source_problem_inventory_digest": source["source_problem_inventory_digest"],
            "source_portfolio_holding_count": source["source_portfolio_holding_count"],
            "portfolio_member_verified": source["portfolio_member_verified"],
            "roles": [dict(row) for row in self.roles],
            "economic_state_digest": dict(self.economic_state_digest),
        }
        self.bootstrap_set_id = "bootstrap." + hashlib.sha256(_canonical(identity)).hexdigest()
        self.members = (
            (MEMBER_PATHS[0], self.source_bytes), (MEMBER_PATHS[1], self.champion_bytes),
            (MEMBER_PATHS[2], self.challenger_bytes),
        )
        if any(len(data) > _MAX_MEMBER for _name, data in self.members):
            raise ForwardBootstrapContractError("bootstrap member exceeds the size limit")
        payload = dict(identity)
        payload.update({"bootstrap_set_id": self.bootstrap_set_id,
                        "members": [_member_row(name, data) for name, data in self.members]})
        self.request_digest = MappingProxyType(_digest(payload))
        self.definition_request_digest = MappingProxyType(dict(evidence.definition_receipt.request_digest))
        self.evidence_request_digest = MappingProxyType(dict(evidence.evidence_receipt.request_digest))
        self.evidence_manifest_digest = MappingProxyType(dict(evidence.evidence_receipt.manifest_digest))
        self.evidence_operation_id = evidence.evidence_receipt.operation_id
        self.source = MappingProxyType(source)
        _REQUEST_BINDINGS[self] = self._signature()

    def _signature(self) -> bytes:
        return _canonical({
            "bootstrap_set_id": self.bootstrap_set_id,
            "request_digest": dict(self.request_digest),
            "members": [_member_row(name, data) for name, data in self.members],
            "manifest_digest": _digest(_manifest(self), "raw_bytes"),
        })

    def validate(self) -> None:
        if type(self) is not _Request or _REQUEST_BINDINGS.get(self) != self._signature():
            raise ForwardBootstrapContractError("bootstrap request authority is absent")


def _member_row(name: str, data: bytes) -> Dict[str, Any]:
    return {"logical_path": name, "size_bytes": len(data), "digest": _digest(data, "raw_bytes")}


def _manifest(request: _Request) -> bytes:
    source = dict(request.source)
    value = {
        "schema_version": 1, "bootstrap_kind": BOOTSTRAP_KIND,
        "storage_claim": STORAGE_CLAIM, "bootstrap_set_id": request.bootstrap_set_id,
        "definition_set_id": request.definition_set_id,
        "definition_evidence_cycle_id": request.definition_evidence_cycle_id,
        "bootstrap_source_cycle_id": request.bootstrap_source_cycle_id,
        "definition_request_digest": dict(request.definition_request_digest),
        "definition_evidence_request_digest": dict(request.evidence_request_digest),
        "definition_evidence_manifest_digest": dict(request.evidence_manifest_digest),
        "definition_evidence_operation_id": request.evidence_operation_id,
        "phase37a_seal_digest": source["phase37a_seal_digest"],
        "phase37a_manifest_digest": source["phase37a_manifest_digest"],
        "source_portfolio_raw_digest": source["source_portfolio_raw_digest"],
        "source_portfolio_semantic_digest": source["source_portfolio_semantic_digest"],
        "source_state_interpretation": "SEALED_PRODUCTION_PORTFOLIO_AT_BOOTSTRAP_SOURCE_CYCLE_V1",
        "economic_state_digest": dict(request.economic_state_digest),
        "roles": [dict(row) for row in request.roles], "member_count": 3,
        "members": [_member_row(name, data) for name, data in request.members],
        "matched_economics_checked": True, "portfolio_bootstrap_claim": True,
        "intent_claim": False, "epoch_started": False, "prospective_claim": False,
        "promotion_capability": False,
    }
    return _canonical(value)


def _read_exact(root_fd: int, name: str, expected: bytes, *, conflict: bool) -> None:
    def fingerprint(kind: str, facts: Mapping[str, Any]) -> str:
        return hashlib.sha256(_canonical({"member": name, "kind": kind, "facts": dict(facts)})).hexdigest()

    try:
        before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                                  value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_size != len(expected)
                or before.st_size > _MAX_MEMBER):
            facts = {"identity": list(identity(before))}
            if conflict:
                raise _Conflict(fingerprint("METADATA_MISMATCH", facts))
            raise _Uncertain("member metadata mismatch")
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=root_fd,
        )
        try:
            info = os.fstat(fd)
            chunks = []
            remaining = len(expected) + 1
            while remaining:
                chunk = os.read(fd, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            _close_fd(fd, suppress=False)
        after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (identity(info) != identity(before) or data != expected
                or identity(after) != identity(before)):
            facts = {"identity": list(identity(after)), "raw_digest": _digest(data, "raw_bytes")}
            if conflict:
                raise _Conflict(fingerprint("MISMATCH", facts))
            raise _Uncertain("member mismatch")
    except _PROCESS_CONTROL:
        raise
    except _Conflict:
        raise
    except Exception as exc:
        if conflict:
            raise _Conflict(fingerprint("ERROR", {
                "type": type(exc).__name__, "errno": getattr(exc, "errno", None),
            }))
        raise _Uncertain("member postcondition uncertain") from exc


class MatchedForwardBootstrapStoreReceipt:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise ForwardBootstrapContractError("store receipts are writer-owned")
        expected = {"operation_id", "status", "reason_code", "did_write", "bootstrap_set_id", "request_digest",
                    "manifest_digest", "member_count", "root_before", "root_after", "bundle_identity"}
        if set(kwargs) != expected:
            raise ForwardBootstrapContractError("store receipt fields are invalid")
        for key, value in kwargs.items():
            if key.endswith("digest") and value is not None:
                value = MappingProxyType(dict(value))
            object.__setattr__(self, key, value)
        self._validate(False)
        _RECEIPT_BINDINGS[self] = _canonical(self.to_dict())

    def _validate(self, binding: bool = True) -> None:
        reasons = {"COMMITTED": "COMMITTED", "ADOPTED": "EXACT_BUNDLE_ALREADY_PRESENT",
                   "CONFLICT": "BOOTSTRAP_SET_ID_CONFLICT", "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN"}
        valid_identity = lambda value: (type(value) is tuple and len(value) == 5
                                        and all(type(item) is int for item in value))
        expected_operation = hashlib.sha256(_canonical({
            "domain": "C2_MATCHED_BOOTSTRAP_PUBLICATION_V1",
            "bootstrap_set_id": self.bootstrap_set_id,
            "bootstrap_request_digest": dict(self.request_digest),
            "bootstrap_store_root_identity_before": list(self.root_before),
        })).hexdigest() if valid_identity(self.root_before) else None
        if (self.status not in reasons or self.reason_code != reasons[self.status]
                or type(self.did_write) is not bool or self.operation_id != expected_operation):
            raise ForwardBootstrapContractError("store receipt status is inconsistent")
        _id(self.bootstrap_set_id, "bootstrap_set_id")
        _typed_digest(dict(self.request_digest), "bootstrap_request_digest", "canonical_json")
        if (not valid_identity(self.root_before)
                or (self.root_after is not None and not valid_identity(self.root_after))
                or (self.bundle_identity is not None and not valid_identity(self.bundle_identity))
                or type(self.member_count) is not int):
            raise ForwardBootstrapContractError("store receipt identity fields are invalid")
        if self.manifest_digest is not None:
            _typed_digest(dict(self.manifest_digest), "bootstrap_manifest_digest", "raw_bytes")
        positive = self.status in {"COMMITTED", "ADOPTED"}
        if positive != (self.manifest_digest is not None and self.member_count == 3 and self.bundle_identity is not None):
            raise ForwardBootstrapContractError("store receipt capability fields are inconsistent")
        if not positive and (self.manifest_digest is not None or self.bundle_identity is not None):
            raise ForwardBootstrapContractError("nonpositive receipt exposes durable evidence")
        if self.did_write is not (self.status == "COMMITTED") and self.status != "UNCERTAIN":
            raise ForwardBootstrapContractError("store receipt write fact is inconsistent")
        if self.status == "CONFLICT" and self.member_count != 0:
            raise ForwardBootstrapContractError("conflict member count is invalid")
        if (positive or self.status == "CONFLICT") and self.root_after != self.root_before:
            raise ForwardBootstrapContractError("terminal root identity is inconsistent")
        if self.status == "UNCERTAIN" and not 0 <= self.member_count <= 3:
            raise ForwardBootstrapContractError("uncertain member count is invalid")
        if self.status == "UNCERTAIN" and not self.did_write and self.member_count != 0:
            raise ForwardBootstrapContractError("zero-write uncertainty cannot report a prefix")
        if binding and _RECEIPT_BINDINGS.get(self) != _canonical(self.to_dict()):
            raise ForwardBootstrapContractError("store receipt authority is absent")

    @property
    def durable_capability(self) -> bool:
        self._validate()
        return self.status in {"COMMITTED", "ADOPTED"}

    def to_dict(self) -> Dict[str, Any]:
        return {"operation_id": self.operation_id, "status": self.status,
                "reason_code": self.reason_code, "did_write": self.did_write,
                "bootstrap_set_id": self.bootstrap_set_id, "bootstrap_request_digest": dict(self.request_digest),
                "manifest_digest": None if self.manifest_digest is None else dict(self.manifest_digest),
                "member_count": self.member_count, "bootstrap_store_root_identity_before": list(self.root_before),
                "bootstrap_store_root_identity_after": None if self.root_after is None else list(self.root_after),
                "bootstrap_bundle_root_identity": None if self.bundle_identity is None else list(self.bundle_identity)}


def _receipt(request: _Request, status: str, did_write: bool, count: int, before: Tuple[int, ...],
             after: Optional[Tuple[int, ...]], bundle: Optional[Tuple[int, ...]] = None,
             manifest: Optional[bytes] = None) -> MatchedForwardBootstrapStoreReceipt:
    reasons = {"COMMITTED": "COMMITTED", "ADOPTED": "EXACT_BUNDLE_ALREADY_PRESENT",
               "CONFLICT": "BOOTSTRAP_SET_ID_CONFLICT", "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN"}
    return MatchedForwardBootstrapStoreReceipt(
        _authority=_AUTHORITY,
        operation_id=hashlib.sha256(_canonical({
            "domain": "C2_MATCHED_BOOTSTRAP_PUBLICATION_V1",
            "bootstrap_set_id": request.bootstrap_set_id,
            "bootstrap_request_digest": dict(request.request_digest),
            "bootstrap_store_root_identity_before": list(before),
        })).hexdigest(), status=status, reason_code=reasons[status], did_write=did_write,
        bootstrap_set_id=request.bootstrap_set_id, request_digest=request.request_digest,
        manifest_digest=None if manifest is None else _digest(manifest, "raw_bytes"), member_count=count,
        root_before=before, root_after=after, bundle_identity=bundle,
    )


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._did_create = False
        self._prefix = 0
        self._entry_guard: Optional[SourceMutationObserver] = None
        self._terminal_guard: Optional[SourceMutationObserver] = None

    def begin_handoff(self, identifier: str) -> None:
        if self._entry_guard is not None:
            raise ForwardBootstrapContractError("store handoff is already active")
        self._entry_guard = SourceMutationObserver(self.root, (identifier,))

    def terminal_continuity(self) -> bool:
        guard = self._terminal_guard
        self._terminal_guard = None
        if guard is None:
            return False
        try:
            return guard.supported and not guard.mutated()
        finally:
            _close_guard(guard)

    def _verify(self, request: _Request, fd: int, *, conflict: bool) -> bytes:
        expected_names = tuple(sorted(MEMBER_PATHS + (MANIFEST_NAME,)))
        try:
            inventory = tuple(sorted(os.listdir(fd)))
        except OSError as exc:
            raise _Uncertain("inventory unavailable") from exc
        if inventory != expected_names:
            if conflict:
                raise _Conflict(hashlib.sha256(_canonical({"inventory": inventory})).hexdigest())
            raise _Uncertain("inventory mismatch")
        for name, data in request.members:
            _read_exact(fd, name, data, conflict=conflict)
        manifest = _manifest(request)
        _read_exact(fd, MANIFEST_NAME, manifest, conflict=conflict)
        return manifest

    def _existing(self, request: _Request, root_fd: int, before: Tuple[int, ...]) -> MatchedForwardBootstrapStoreReceipt:
        fingerprints = []
        for _ in range(2):
            bundle_fd = None
            try:
                if _dir_identity(self.root, private=True) != before:
                    raise _Uncertain("bootstrap root drift")
                root_continuity = _namespace_identity(self.root)
                info = os.stat(request.bootstrap_set_id, dir_fd=root_fd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
                    raise _Conflict(hashlib.sha256(_canonical({"metadata": list((info.st_mode, info.st_nlink))})).hexdigest())
                target_continuity = _namespace_identity(self.root / request.bootstrap_set_id)
                identity = target_continuity
                bundle_fd = os.open(request.bootstrap_set_id, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
                manifest = self._verify(request, bundle_fd, conflict=True)
                _close_fd(bundle_fd, suppress=False)
                bundle_fd = None
                if (_namespace_identity(self.root) != root_continuity
                        or _namespace_identity(self.root / request.bootstrap_set_id) != identity):
                    raise _Uncertain("bundle namespace drift")
                return _receipt(request, "ADOPTED", False, 3, before, before, identity, manifest)
            except _PROCESS_CONTROL:
                raise
            except _Conflict as exc:
                fingerprints.append(exc.fingerprint)
            except Exception:
                return _receipt(request, "UNCERTAIN", False, 0, before, None)
            finally:
                if bundle_fd is not None:
                    _close_fd(bundle_fd, suppress=True)
        if len(fingerprints) == 2 and fingerprints[0] == fingerprints[1] and _dir_identity(self.root, private=True) == before:
            return _receipt(request, "CONFLICT", False, 0, before, before)
        return _receipt(request, "UNCERTAIN", False, 0, before, None)

    def publish(self, request: _Request) -> MatchedForwardBootstrapStoreReceipt:
        request.validate()
        self._did_create = False
        self._prefix = 0
        before = _dir_identity(self.root, private=True)
        target = self.root / request.bootstrap_set_id
        existed = os.path.lexists(str(target))
        try:
            if self._entry_guard is not None:
                if not self._entry_guard.supported or self._entry_guard.mutated():
                    raise ForwardBootstrapInputError("bootstrap store handoff is uncertain")
            return self._publish_impl(request, existed=existed)
        except _PROCESS_CONTROL:
            raise
        except Exception as exc:
            if not existed and not self._did_create:
                raise ForwardBootstrapInputError(
                    "bootstrap publication failed before target creation",
                ) from exc
            did_write = self._did_create
            count = self._prefix if did_write else 0
            try:
                after = _dir_identity(self.root, private=True)
            except Exception:
                after = None
            return _receipt(request, "UNCERTAIN", did_write, count, before, after)
        finally:
            if self._entry_guard is not None:
                _close_guard(self._entry_guard)
                self._entry_guard = None

    def _publish_impl(
        self, request: _Request, *, existed: bool,
    ) -> MatchedForwardBootstrapStoreReceipt:
        request.validate()
        continuity = SourceMutationObserver(self.root, ())
        root_fd: Optional[int] = os.open(str(self.root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        before = _dir_identity(self.root, private=True)
        bundle_fd = None
        final_guard = None
        prefix = 0
        try:
            try:
                os.mkdir(request.bootstrap_set_id, 0o700, dir_fd=root_fd)
                self._did_create = True
                if (self._entry_guard is None
                        or not _accept_exact_target_create(
                            self._entry_guard, request.bootstrap_set_id,
                        )):
                    raise _Uncertain("bootstrap target create continuity uncertain")
            except FileExistsError:
                existing_guard = self._entry_guard
                self._entry_guard = None
                try:
                    if existing_guard is None or not existed:
                        result = _receipt(request, "UNCERTAIN", False, 0, before, None)
                    else:
                        result = self._existing(request, root_fd, before)
                        if not existing_guard.supported or existing_guard.mutated():
                            result = _receipt(request, "UNCERTAIN", False, 0, before, None)
                        else:
                            self._terminal_guard = existing_guard
                            existing_guard = None
                finally:
                    if existing_guard is not None:
                        _close_guard(existing_guard)
                _close_fd(root_fd, suppress=False)
                root_fd = None
                if not continuity.supported or continuity.mutated():
                    return _receipt(request, "UNCERTAIN", False, 0, before, None)
                return result
            except _PROCESS_CONTROL:
                raise
            except OSError as exc:
                raise ForwardBootstrapInputError("bootstrap target could not be created") from exc
            try:
                root_continuity = _namespace_identity(self.root)
                bundle_fd = os.open(request.bootstrap_set_id, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
                for name, data in request.members:
                    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=bundle_fd)
                    try:
                        _write_all(fd, data); os.fsync(fd)
                        prefix += 1
                        self._prefix = prefix
                    finally:
                        _close_fd(fd, suppress=False)
                manifest = _manifest(request)
                fd = os.open(MANIFEST_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=bundle_fd)
                try:
                    _write_all(fd, manifest); os.fsync(fd)
                finally:
                    _close_fd(fd, suppress=False)
                final_guard = SourceMutationObserver(
                    self.root, (request.bootstrap_set_id,),
                )
                target_continuity = _namespace_identity(self.root / request.bootstrap_set_id)
                identity = target_continuity
                self._verify(request, bundle_fd, conflict=False)
                os.fsync(bundle_fd); os.fsync(root_fd)
                _close_fd(bundle_fd, suppress=False)
                bundle_fd = None
                _close_fd(root_fd, suppress=False)
                root_fd = None
                if (_dir_identity(self.root, private=True) != before
                        or _namespace_identity(self.root) != root_continuity
                        or _namespace_identity(self.root / request.bootstrap_set_id) != identity):
                    raise _Uncertain("final namespace drift")
                if (not continuity.supported or continuity.mutated()
                        or self._entry_guard is None
                        or not self._entry_guard.supported
                        or self._entry_guard.mutated()
                        or not final_guard.supported or final_guard.mutated()):
                    raise _Uncertain("bootstrap root continuity uncertain")
                _close_guard(self._entry_guard)
                self._entry_guard = None
                self._terminal_guard = final_guard
                final_guard = None
                return _receipt(request, "COMMITTED", True, 3, before, before, identity, manifest)
            except _PROCESS_CONTROL:
                raise
            except Exception:
                try: after = _dir_identity(self.root, private=True)
                except Exception: after = None
                return _receipt(request, "UNCERTAIN", True, prefix, before, after)
        finally:
            if bundle_fd is not None:
                _close_fd(bundle_fd, suppress=True)
            if root_fd is not None:
                _close_fd(root_fd, suppress=True)
            if final_guard is not None:
                try:
                    _close_guard(final_guard)
                except OSError:
                    pass
            _close_guard(continuity)


def _verify_public_bootstrap(
    root: Path, request: _Request, receipt: MatchedForwardBootstrapStoreReceipt,
) -> bool:
    if receipt.status not in {"COMMITTED", "ADOPTED"}:
        return True
    descriptor = None
    result = False
    try:
        before = _dir_identity(root, private=True)
        if before != receipt.root_before or receipt.root_after != before:
            return False
        descriptor = os.open(
            str(root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        observed = _Store(root)._existing(request, descriptor, before)
        result = (
            observed.status == "ADOPTED"
            and observed.bundle_identity == receipt.bundle_identity
            and dict(observed.manifest_digest) == dict(receipt.manifest_digest)
            and _dir_identity(root, private=True) == receipt.root_after
        )
    except _PROCESS_CONTROL:
        raise
    except Exception:
        return False
    finally:
        if descriptor is not None:
            try:
                _close_fd(descriptor, suppress=False)
            except _PROCESS_CONTROL:
                raise
            except Exception:
                result = False
    return result


def _source_receipt(evidence: Any, source: Any, definition_cycle: str,
                    source_cycle: str) -> Dict[str, Any]:
    problems = [dict(row) for row in source.problem_inventory()]
    cash_text = source.verified_portfolio()["current_cash"]
    cash = Decimal(cash_text)
    deficit = min(cash, Decimal("0"))
    return {
        "schema_version": 1, "source_claim": "C2_MATCHED_BOOTSTRAP_SOURCE_VERIFIED_V1",
        "definition_set_id": evidence.definition_receipt.definition_set_id,
        "definition_evidence_cycle_id": definition_cycle,
        "bootstrap_source_cycle_id": source_cycle,
        "definition_evidence_status": evidence.status,
        "definition_evidence_request_digest": dict(evidence.evidence_receipt.request_digest),
        "definition_evidence_manifest_digest": dict(evidence.evidence_receipt.manifest_digest),
        "definition_evidence_operation_id": evidence.evidence_receipt.operation_id,
        "source_observation_status": source.status, "source_cycle_status": source.source_cycle_status,
        "source_problem_inventory": problems,
        "source_problem_inventory_digest": dict(source.problem_inventory_digest),
        "phase37a_seal_digest": dict(source.seal_digest),
        "phase37a_manifest_digest": dict(source.manifest_digest),
        "source_portfolio_raw_digest": dict(source.portfolio_raw_digest),
        "source_portfolio_semantic_digest": dict(source.portfolio_semantic_digest),
        "source_portfolio_holding_count": source.portfolio_holding_count,
        "portfolio_member_verified": True, "chronology_checked": True,
        "cash_sign_and_deficit_digest": _digest({
            "cash": cash_text, "deficit": format(deficit, "f"),
        }),
        "did_write": False, "prospective_claim": False, "promotion_capability": False,
    }


def _fresh_request(paths: Tuple[Path, ...], definition_cycle: str, source_cycle: str) -> Tuple[_Request, Any, Any]:
    root, activation, definitions, evidence_root, _bootstraps = paths
    from quantpits.research.forward_definition_evidence import adopt_forward_definition_evidence
    evidence = adopt_forward_definition_evidence(root, definition_cycle, activation, definitions, evidence_root)
    if evidence.status != "ADOPTED" or evidence.evidence_receipt.did_write or not evidence.definition_evidence_complete:
        raise ForwardBootstrapInputError("definition evidence was not adopted exactly")
    from quantpits.research.forward_observation import observe_frozen_shadow_forward_definition_candidate
    definition = observe_frozen_shadow_forward_definition_candidate(root, definition_cycle, activation)
    if dict(definition.to_store_request().request_digest) != dict(evidence.definition_receipt.request_digest):
        raise ForwardBootstrapInputError("definition observation does not join evidence")
    champion = definition.compiled_definitions.champion.to_dict()
    challenger = definition.compiled_definitions.challenger.to_dict()
    cutoff = max(champion["data_cutoff"], challenger["data_cutoff"])
    if max(date.fromisoformat(cutoff), date.fromisoformat(definition_cycle)) > date.fromisoformat(source_cycle):
        raise ForwardBootstrapInputError("bootstrap source predates definition cutoff")
    from quantpits.research.forward_portfolio_source import observe_forward_portfolio_source
    source = observe_forward_portfolio_source(root, source_cycle)
    if not source.bootstrap_source_capability:
        raise ForwardBootstrapInputError("portfolio source has no bootstrap capability")
    champion_state, challenger_state, _assumption = _build_states(source, definition)
    receipt = _source_receipt(evidence, source, definition_cycle, source_cycle)
    return _Request(_authority=_AUTHORITY, source_receipt=receipt,
                    champion=champion_state, challenger=challenger_state,
                    definition=definition, evidence=evidence), definition, source


class MatchedForwardBootstrapPlan:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY or set(kwargs) != {"request", "target_state"}:
            raise ForwardBootstrapContractError("bootstrap plans are inspector-owned")
        self._request = kwargs["request"]
        self._request.validate()
        self.target_state = kwargs["target_state"]
        _PLAN_BINDINGS[self] = _canonical(self._summary())

    def _validate(self) -> None:
        if self.target_state not in {"ABSENT", "PRESENT"} or _PLAN_BINDINGS.get(self) != _canonical(self._summary()):
            raise ForwardBootstrapContractError("bootstrap plan authority is absent")

    @property
    def bootstrap_set_id(self) -> str: return self._request.bootstrap_set_id
    @property
    def bootstrap_request_digest(self) -> Mapping[str, Any]: return MappingProxyType(dict(self._request.request_digest))

    def _summary(self) -> Dict[str, Any]:
        r = self._request; source = dict(r.source)
        return {"status": "PREPARED", "schema_version": 1, "bootstrap_set_id": r.bootstrap_set_id,
                "definition_set_id": r.definition_set_id, "definition_evidence_cycle_id": r.definition_evidence_cycle_id,
                "bootstrap_source_cycle_id": r.bootstrap_source_cycle_id,
                "definition_request_digest": dict(r.definition_request_digest),
                "definition_evidence_request_digest": dict(r.evidence_request_digest),
                "definition_evidence_manifest_digest": dict(r.evidence_manifest_digest),
                "source_portfolio_raw_digest": source["source_portfolio_raw_digest"],
                "source_portfolio_semantic_digest": source["source_portfolio_semantic_digest"],
                "economic_state_digest": dict(r.economic_state_digest),
                "champion_state_digest": dict(r.roles[0]["state_digest"]),
                "challenger_state_digest": dict(r.roles[1]["state_digest"]),
                "bootstrap_request_digest": dict(r.request_digest), "target_state": self.target_state,
                "definition_evidence_status": "ADOPTED", "matched_economics_checked": True,
                "publication_capability": False, "portfolio_bootstrap_complete": False,
                "state_chain_advanced": False, "intent_sealed": False, "epoch_started": False,
                "prospective_claim": False, "promotion_capability": False}

    def to_safe_summary_dict(self) -> Dict[str, Any]: self._validate(); return self._summary()


class MatchedForwardBootstrapResult:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY or set(kwargs) != {"request", "receipt", "sources_stable"}:
            raise ForwardBootstrapContractError("bootstrap results are writer-owned")
        self._request = kwargs["request"]; self.receipt = kwargs["receipt"]; self.sources_stable = kwargs["sources_stable"]
        if type(self.sources_stable) is not bool:
            raise ForwardBootstrapContractError("source stability fact is invalid")
        _RESULT_BINDINGS[self] = _canonical(self._summary())

    def _validate(self) -> None:
        self.receipt._validate()
        self._request.validate()
        if (_RESULT_BINDINGS.get(self) != _canonical(self._summary())
                or self.receipt.bootstrap_set_id != self._request.bootstrap_set_id
                or dict(self.receipt.request_digest) != dict(self._request.request_digest)
                or (self.receipt.manifest_digest is not None and
                    dict(self.receipt.manifest_digest) != _digest(_manifest(self._request), "raw_bytes"))):
            raise ForwardBootstrapContractError("bootstrap result authority is absent")

    @property
    def status(self) -> str:
        self._validate(); return self.receipt.status if self.sources_stable else "UNCERTAIN"
    @property
    def portfolio_bootstrap_complete(self) -> bool:
        self._validate(); return self.sources_stable and self.receipt.durable_capability

    def _summary(self) -> Dict[str, Any]:
        complete = self.sources_stable and self.receipt.durable_capability
        status = self.receipt.status if self.sources_stable else "UNCERTAIN"
        source = dict(self._request.source)
        return {"status": status, "schema_version": 1,
                "reason_code": self.receipt.reason_code if self.sources_stable else "SOURCE_CONTINUITY_UNCERTAIN",
                "did_write": self.receipt.did_write, "bootstrap_set_id": self._request.bootstrap_set_id,
                "operation_id": self.receipt.operation_id,
                "definition_set_id": self._request.definition_set_id,
                "definition_evidence_cycle_id": self._request.definition_evidence_cycle_id,
                "bootstrap_source_cycle_id": self._request.bootstrap_source_cycle_id,
                "definition_request_digest": dict(self._request.definition_request_digest),
                "definition_evidence_request_digest": dict(self._request.evidence_request_digest),
                "definition_evidence_manifest_digest": dict(self._request.evidence_manifest_digest),
                "source_portfolio_raw_digest": source["source_portfolio_raw_digest"],
                "source_portfolio_semantic_digest": source["source_portfolio_semantic_digest"],
                "economic_state_digest": dict(self._request.economic_state_digest),
                "champion_state_digest": dict(self._request.roles[0]["state_digest"]),
                "challenger_state_digest": dict(self._request.roles[1]["state_digest"]),
                "bootstrap_request_digest": dict(self._request.request_digest),
                "manifest_digest": None if self.receipt.manifest_digest is None else dict(self.receipt.manifest_digest),
                "member_count": self.receipt.member_count, "definition_evidence_status": "ADOPTED",
                "matched_economics_checked": True, "portfolio_bootstrap_complete": complete,
                "state_chain_advanced": False, "intent_sealed": False, "epoch_started": False,
                "prospective_claim": False, "promotion_capability": False}

    def to_safe_summary_dict(self) -> Dict[str, Any]: self._validate(); return self._summary()


def _target_state(root: Path, identifier: str) -> str:
    try: os.lstat(str(root / identifier))
    except FileNotFoundError: return "ABSENT"
    except OSError as exc: raise ForwardBootstrapInputError("bootstrap target is incomparable") from exc
    return "PRESENT"


def _prepare_matched_forward_bootstrap(workspace_root: Any, definition_evidence_cycle_id: Any,
                                      bootstrap_source_cycle_id: Any, activation_path: Any,
                                      definition_store_root: Any, evidence_store_root: Any,
                                      bootstrap_store_root: Any) -> MatchedForwardBootstrapPlan:
    definition_cycle = _cycle(definition_evidence_cycle_id, "definition_evidence_cycle_id")
    source_cycle = _cycle(bootstrap_source_cycle_id, "bootstrap_source_cycle_id")
    paths = _formal_paths(workspace_root, activation_path, definition_store_root, evidence_store_root, bootstrap_store_root)
    root, activation, definitions, evidence, bootstraps = paths
    definition_id = activation.stem
    source_guard = SourceMutationObserver(
        root, _source_paths(root, activation, definitions, evidence, definition_id, source_cycle),
    )
    bootstrap_guard = SourceMutationObserver(bootstraps, ())
    try:
        before = _protected_inventory(root, None)
        first, _definition, _source = _fresh_request(paths, definition_cycle, source_cycle)
        bootstrap_guard.add_paths((first.bootstrap_set_id,))
        request, definition, source = _fresh_request(paths, definition_cycle, source_cycle)
        if (request.bootstrap_set_id != first.bootstrap_set_id
                or dict(request.request_digest) != dict(first.request_digest)):
            raise ForwardBootstrapInputError("preflight source observations are inconsistent")
        state = _target_state(bootstraps, request.bootstrap_set_id)
        definition.to_store_request(); source.verified_portfolio(); request.validate()
        if (not source_guard.supported or source_guard.mutated()
                or not bootstrap_guard.supported or bootstrap_guard.mutated()
                or _protected_inventory(root, None) != before):
            raise ForwardBootstrapInputError("preflight continuity is uncertain")
        return MatchedForwardBootstrapPlan(_authority=_AUTHORITY, request=request, target_state=state)
    finally:
        _close_guard(source_guard); _close_guard(bootstrap_guard)


def _publish_matched_forward_bootstrap(workspace_root: Any, definition_evidence_cycle_id: Any,
                                      bootstrap_source_cycle_id: Any, activation_path: Any,
                                      definition_store_root: Any, evidence_store_root: Any,
                                      bootstrap_store_root: Any, expected_bootstrap_set_id: Any,
                                      expected_bootstrap_request_digest: Any,
                                      authorization_action: Any) -> MatchedForwardBootstrapResult:
    expected_id = _id(expected_bootstrap_set_id, "expected_bootstrap_set_id")
    expected_digest = _typed_digest(expected_bootstrap_request_digest, "expected_bootstrap_request_digest", "canonical_json")
    if type(authorization_action) is not str or authorization_action != AUTHORIZATION_ACTION:
        raise ForwardBootstrapContractError("bootstrap authorization is invalid")
    definition_cycle = _cycle(definition_evidence_cycle_id, "definition_evidence_cycle_id")
    source_cycle = _cycle(bootstrap_source_cycle_id, "bootstrap_source_cycle_id")
    paths = _formal_paths(workspace_root, activation_path, definition_store_root, evidence_store_root, bootstrap_store_root)
    root, activation, definitions, evidence, bootstraps = paths
    guard = SourceMutationObserver(
        root, _source_paths(root, activation, definitions, evidence, activation.stem, source_cycle),
    )
    target_guard = SourceMutationObserver(bootstraps, (expected_id,))
    post_guard = None
    store = None
    result = None
    stable = False
    try:
        before = _protected_inventory(root, bootstraps / expected_id)
        request, definition, source = _fresh_request(paths, definition_cycle, source_cycle)
        if request.bootstrap_set_id != expected_id or dict(request.request_digest) != expected_digest:
            raise ForwardBootstrapContractError("fresh bootstrap request does not match owner authorization")
        if not target_guard.supported or target_guard.mutated():
            raise ForwardBootstrapInputError("bootstrap target changed before publication")
        store = _Store(bootstraps)
        store.begin_handoff(request.bootstrap_set_id)
        _close_guard(target_guard)
        target_guard = None
        receipt = store.publish(request)
        post_guard = SourceMutationObserver(bootstraps, (request.bootstrap_set_id,))
        # Revalidate both inspector-owned aggregates and require no protected namespace event.
        try:
            definition.to_store_request(); source.verified_portfolio()
            stable = (guard.supported and not guard.mutated()
                      and post_guard.supported and not post_guard.mutated()
                      and store.terminal_continuity()
                      and _verify_public_bootstrap(bootstraps, request, receipt)
                      and _protected_inventory(root, bootstraps / request.bootstrap_set_id) == before)
        except _PROCESS_CONTROL:
            raise
        except Exception:
            stable = False
        result = (request, receipt)
    finally:
        if target_guard is not None:
            _close_guard(target_guard)
        if post_guard is not None:
            try:
                stable = stable and not post_guard.mutated()
                _close_guard(post_guard)
            except _PROCESS_CONTROL:
                raise
            except OSError:
                stable = False
        if store is not None and store._terminal_guard is not None:
            try:
                stable = stable and store.terminal_continuity()
            except _PROCESS_CONTROL:
                raise
            except OSError:
                stable = False
        try:
            stable = stable and not guard.mutated()
            _close_guard(guard)
        except _PROCESS_CONTROL:
            raise
        except OSError:
            stable = False
    if result is None:
        raise ForwardBootstrapInputError("bootstrap publication did not produce a result")
    return MatchedForwardBootstrapResult(
        _authority=_AUTHORITY, request=result[0], receipt=result[1], sources_stable=stable,
    )


def prepare_matched_forward_bootstrap(
    workspace_root: Any, definition_evidence_cycle_id: Any,
    bootstrap_source_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
    bootstrap_store_root: Any,
) -> MatchedForwardBootstrapPlan:
    try:
        return _prepare_matched_forward_bootstrap(
            workspace_root, definition_evidence_cycle_id, bootstrap_source_cycle_id,
            activation_path, definition_store_root, evidence_store_root,
            bootstrap_store_root,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardBootstrapContractError:
        raise
    except Exception as exc:
        raise ForwardBootstrapInputError("bootstrap preflight failed closed") from exc


def publish_matched_forward_bootstrap(
    workspace_root: Any, definition_evidence_cycle_id: Any,
    bootstrap_source_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, evidence_store_root: Any,
    bootstrap_store_root: Any, expected_bootstrap_set_id: Any,
    expected_bootstrap_request_digest: Any, authorization_action: Any,
) -> MatchedForwardBootstrapResult:
    try:
        return _publish_matched_forward_bootstrap(
            workspace_root, definition_evidence_cycle_id, bootstrap_source_cycle_id,
            activation_path, definition_store_root, evidence_store_root,
            bootstrap_store_root, expected_bootstrap_set_id,
            expected_bootstrap_request_digest, authorization_action,
        )
    except _PROCESS_CONTROL:
        raise
    except ForwardBootstrapContractError:
        raise
    except Exception as exc:
        raise ForwardBootstrapInputError("bootstrap publication failed closed") from exc


__all__ = [
    "AUTHORIZATION_ACTION", "BOOTSTRAP_KIND", "STORAGE_CLAIM", "MEMBER_PATHS", "MANIFEST_NAME",
    "ForwardBootstrapContractError", "ForwardBootstrapInputError", "MatchedForwardBootstrapStoreReceipt",
    "MatchedForwardBootstrapPlan", "MatchedForwardBootstrapResult", "prepare_matched_forward_bootstrap",
    "publish_matched_forward_bootstrap",
]
