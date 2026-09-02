"""Explicit create-only publication of one formally frozen definition bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import weakref
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from quantpits.evidence.contracts import TypedDigest


AUTHORIZATION_ACTION = "PUBLISH_ONE_FROZEN_DEFINITION_BUNDLE_V1"
FRESH_AUTHORIZATION_ACTION = (
    "PUBLISH_ONE_FRESH_CHAMPION_SEGMENT_DEFINITION_BUNDLE_V1"
)
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRESH_PROTECTED_BYTE_BUDGET = 64 * 1024 * 1024
_PLAN_BINDINGS: "weakref.WeakKeyDictionary[Any, Tuple[Any, ...]]" = weakref.WeakKeyDictionary()
_RESULT_BINDINGS: "weakref.WeakKeyDictionary[Any, Tuple[Any, ...]]" = weakref.WeakKeyDictionary()


class FrozenDefinitionPublicationContractError(ValueError):
    """A caller-controlled publication value violates the frozen contract."""

    reason_code = "CONTRACT_INVALID"


class FrozenDefinitionPublicationInputError(FrozenDefinitionPublicationContractError):
    """The formal publication inputs cannot be compared safely."""

    reason_code = "INPUT_INCOMPARABLE"


def _input(message: str) -> FrozenDefinitionPublicationInputError:
    return FrozenDefinitionPublicationInputError(message)


def _path(value: Any, name: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise FrozenDefinitionPublicationContractError("%s must be a path" % name)
    path = Path(value)
    if not path.is_absolute():
        raise FrozenDefinitionPublicationContractError("%s must be absolute" % name)
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
        raise _input("publication directory is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _input("publication directory is not physical")
    if private and stat.S_IMODE(info.st_mode) != 0o700:
        raise _input("private publication directory mode must be 0700")
    return info.st_dev, info.st_ino, info.st_mode, 0, 0


def _directory_continuity(path: Path) -> Tuple[int, int, int, int, int]:
    info = os.lstat(str(path))
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _input("publication directory is not physical")
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_ctime_ns


def _file_identity(path: Path) -> Tuple[int, int, int, int, int, int, int]:
    try:
        info = os.lstat(str(path))
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _input("activation is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise _input("activation must be a private single-link regular file")
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _definition_id(value: Any) -> str:
    if (
        type(value) is not str or value != value.strip() or value in {".", ".."}
        or _ID_RE.fullmatch(value) is None
    ):
        raise FrozenDefinitionPublicationContractError(
            "expected_definition_set_id is invalid",
        )
    return value


def _typed_request_digest(value: Any) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        raise FrozenDefinitionPublicationContractError(
            "expected_request_digest fields are invalid",
        )
    try:
        digest = TypedDigest(**value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise FrozenDefinitionPublicationContractError(
            "expected_request_digest is invalid",
        ) from exc
    if digest.domain != "canonical_json":
        raise FrozenDefinitionPublicationContractError(
            "expected_request_digest domain is invalid",
        )
    return digest.to_dict()


def _typed_reference_digest(value: Any) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        raise FrozenDefinitionPublicationContractError(
            "reference_receipt_digest fields are invalid",
        )
    try:
        digest = TypedDigest(**value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise FrozenDefinitionPublicationContractError(
            "reference_receipt_digest is invalid",
        ) from exc
    if digest.domain != "canonical_json":
        raise FrozenDefinitionPublicationContractError(
            "reference_receipt_digest domain is invalid",
        )
    return digest.to_dict()


def _cycle_id(value: Any) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise FrozenDefinitionPublicationContractError("evidence_cycle_id is invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise FrozenDefinitionPublicationContractError("evidence_cycle_id is invalid") from exc
    if parsed.isoformat() != value:
        raise FrozenDefinitionPublicationContractError("evidence_cycle_id is invalid")
    return value


def _valid_digest(value: Any, domain: str) -> bool:
    if type(value) is not dict or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        return False
    try:
        digest = TypedDigest(**value)
    except Exception:
        return False
    return digest.domain == domain and digest.to_dict() == value


def _valid_store_identity(value: Any) -> bool:
    return bool(
        type(value) is tuple and len(value) == 5
        and all(type(item) is int for item in value)
    )


def _revalidate_store_receipt(receipt: Any) -> Dict[str, Any]:
    rendered = receipt.to_dict()
    expected_fields = {
        "operation_id", "status", "reason_code", "did_write",
        "definition_set_id", "request_digest", "manifest_digest",
        "member_count", "store_root_identity_before",
        "store_root_identity_after", "bundle_root_identity",
    }
    if set(rendered) != expected_fields:
        raise FrozenDefinitionPublicationContractError("C0 receipt fields are invalid")
    reasons = {
        "COMMITTED": "COMMITTED",
        "ADOPTED": "EXACT_BUNDLE_ALREADY_PRESENT",
        "CONFLICT": "DEFINITION_SET_ID_CONFLICT",
        "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN",
    }
    before = receipt.store_root_identity_before
    after = receipt.store_root_identity_after
    bundle = receipt.bundle_root_identity
    expected_operation = hashlib.sha256(json.dumps({
        "domain": "C0_DEFINITION_PUBLICATION_V1",
        "definition_set_id": receipt.definition_set_id,
        "request_digest": dict(receipt.request_digest),
        "store_root_identity_before": list(before),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8",
    )).hexdigest() if _valid_store_identity(before) else None
    common = (
        rendered["operation_id"] == receipt.operation_id == expected_operation
        and rendered["status"] == receipt.status
        and rendered["reason_code"] == receipt.reason_code == reasons.get(receipt.status)
        and rendered["did_write"] is receipt.did_write
        and type(receipt.did_write) is bool
        and rendered["definition_set_id"] == receipt.definition_set_id
        and _definition_id(receipt.definition_set_id) == receipt.definition_set_id
        and rendered["request_digest"] == dict(receipt.request_digest)
        and _valid_digest(dict(receipt.request_digest), "canonical_json")
        and type(receipt.member_count) is int
        and rendered["member_count"] == receipt.member_count
        and _valid_store_identity(before)
        and rendered["store_root_identity_before"] == list(before)
    )
    manifest = None if receipt.manifest_digest is None else dict(receipt.manifest_digest)
    if receipt.status in {"COMMITTED", "ADOPTED"}:
        status_valid = (
            receipt.did_write is (receipt.status == "COMMITTED")
            and _valid_digest(manifest, "raw_bytes")
            and receipt.member_count == 4
            and _valid_store_identity(after) and after == before
            and _valid_store_identity(bundle)
        )
    elif receipt.status == "CONFLICT":
        status_valid = (
            receipt.did_write is False and manifest is None
            and receipt.member_count == 0 and after == before and bundle is None
        )
    else:
        status_valid = (
            receipt.status == "UNCERTAIN" and 0 <= receipt.member_count <= 4
            and manifest is None and bundle is None
            and (receipt.did_write or receipt.member_count == 0)
            and (after is None or _valid_store_identity(after))
        )
    if not common or not status_valid:
        raise FrozenDefinitionPublicationContractError("C0 receipt cross-fields are invalid")
    expected_after = None if after is None else list(after)
    expected_bundle = None if bundle is None else list(bundle)
    if (
        rendered["manifest_digest"] != manifest
        or rendered["store_root_identity_after"] != expected_after
        or rendered["bundle_root_identity"] != expected_bundle
    ):
        raise FrozenDefinitionPublicationContractError("C0 receipt rendering is invalid")
    return rendered


def _formal_inputs(
    workspace_root: Any, activation_path: Any, definition_store_root: Any,
) -> Tuple[Path, Path, Path, Dict[str, Tuple[int, ...]]]:
    root = _path(workspace_root, "workspace_root")
    activation = _path(activation_path, "activation_path")
    store = _path(definition_store_root, "definition_store_root")
    research = root / "research"
    shadow = research / "shadow_v1"
    activations = shadow / "activations"
    expected_store = shadow / "definitions"
    if activation.parent != activations or store != expected_store:
        raise _input("publication paths do not match the formal private layout")
    identities = {
        "workspace": _directory_identity(root),
        "workspace_continuity": _directory_continuity(root),
        "research": _directory_identity(research, private=True),
        "shadow": _directory_identity(shadow, private=True),
        "activations": _directory_identity(activations, private=True),
        "store": _directory_identity(store, private=True),
        "activation": _file_identity(activation),
        "research_continuity": _directory_continuity(research),
        "shadow_continuity": _directory_continuity(shadow),
        "activations_continuity": _directory_continuity(activations),
        "store_continuity": _directory_continuity(store),
    }
    return root, activation, store, identities


def _same_identities(
    root: Path, activation: Path, store: Path,
    expected: Mapping[str, Tuple[int, ...]], *, after_write: bool = False,
) -> bool:
    try:
        current = {
            "workspace": _directory_identity(root),
            "workspace_continuity": _directory_continuity(root),
            "research": _directory_identity(root / "research", private=True),
            "shadow": _directory_identity(root / "research" / "shadow_v1", private=True),
            "activations": _directory_identity(activation.parent, private=True),
            "store": _directory_identity(store, private=True),
            "activation": _file_identity(activation),
            "research_continuity": _directory_continuity(root / "research"),
            "shadow_continuity": _directory_continuity(root / "research" / "shadow_v1"),
            "activations_continuity": _directory_continuity(activation.parent),
            "store_continuity": _directory_continuity(store),
        }
    except _PROCESS_CONTROL:
        raise
    except FrozenDefinitionPublicationContractError:
        return False
    comparable = dict(expected)
    if after_write:
        current.pop("store_continuity", None)
        comparable.pop("store_continuity", None)
    return current == comparable


def _split_formal_inputs(
    production_workspace_root: Any, research_workspace_root: Any,
    activation_path: Any, definition_store_root: Any,
) -> Tuple[Path, Path, Path, Path, Dict[str, Tuple[int, ...]]]:
    """Freeze two physically separate roots and the formal Research layout."""
    production = _path(production_workspace_root, "production_workspace_root")
    research, activation, store, research_identities = _formal_inputs(
        research_workspace_root, activation_path, definition_store_root,
    )
    if (
        production == research
        or production in research.parents
        or research in production.parents
    ):
        raise _input("Production and Research roots must be physically separate")
    identities = {
        "production_workspace": _directory_identity(production),
        "production_workspace_continuity": _directory_continuity(production),
    }
    identities.update({"research_" + key: value for key, value in research_identities.items()})
    return production, research, activation, store, identities


def _same_split_identities(
    production: Path, research: Path, activation: Path, store: Path,
    expected: Mapping[str, Tuple[int, ...]], *, after_write: bool = False,
) -> bool:
    try:
        production_current = {
            "production_workspace": _directory_identity(production),
            "production_workspace_continuity": _directory_continuity(production),
        }
    except _PROCESS_CONTROL:
        raise
    except (FrozenDefinitionPublicationContractError, OSError):
        return False
    research_expected = {
        key[len("research_"):]: value
        for key, value in expected.items()
        if key.startswith("research_")
    }
    return (
        production_current == {
            key: value for key, value in expected.items()
            if key.startswith("production_")
        }
        and _same_identities(
            research, activation, store, research_expected,
            after_write=after_write,
        )
    )


def _protected_inventory(
    shadow: Path, target: Path, *, mutable_parent: Optional[Path] = None,
    byte_budget: int = 16 * 1024 * 1024,
) -> Tuple[Tuple[Any, ...], ...]:
    """Observe every non-target Research member with bounded byte fingerprints."""
    rows = []
    member_budget = 4096
    definitions = shadow / "definitions" if mutable_parent is None else mutable_parent
    for path in sorted(shadow.rglob("*")):
        if path == target or target in path.parents:
            continue
        if len(rows) >= member_budget:
            raise _input("protected publication inventory exceeds its member budget")
        info = os.lstat(str(path))
        relative = path.relative_to(shadow).as_posix()
        if stat.S_ISLNK(info.st_mode):
            rows.append((relative, "symlink", info.st_mode, info.st_ctime_ns))
        elif stat.S_ISDIR(info.st_mode):
            # Creating the authorized target legitimately changes the store root.
            if path != definitions:
                rows.append((
                    relative, "directory", info.st_dev, info.st_ino,
                    info.st_mode, info.st_nlink, info.st_ctime_ns,
                ))
        elif stat.S_ISREG(info.st_mode):
            if info.st_size > byte_budget:
                raise _input("protected publication inventory exceeds its byte budget")
            data = path.read_bytes()
            byte_budget -= len(data)
            rows.append((
                relative, "regular", info.st_dev, info.st_ino, info.st_mode,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                hashlib.sha256(data).hexdigest(),
            ))
        else:
            rows.append((relative, "special", info.st_dev, info.st_ino, info.st_mode))
    return tuple(rows)


def _binding_identity(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, Mapping):
        return (
            "mapping", type(value), id(value),
            tuple((id(key), key, _binding_identity(item)) for key, item in value.items()),
        )
    if isinstance(value, (tuple, list)):
        return "sequence", type(value), id(value), tuple(_binding_identity(item) for item in value)
    try:
        attributes = vars(value)
    except TypeError:
        return "atom", type(value), id(value)
    return "object", type(value), id(value), _binding_identity(attributes)


class FrozenDefinitionPublicationPlan:
    """Inspector-owned, zero-capability safe preflight result."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise FrozenDefinitionPublicationContractError("publication plans are inspector-owned")
        expected = {
            "definition_set_id", "evidence_cycle_id", "request_digest",
            "reference_receipt_digest", "target_state",
        }
        if set(kwargs) != expected:
            raise FrozenDefinitionPublicationContractError("publication plan fields are not exact")
        object.__setattr__(self, "definition_set_id", kwargs["definition_set_id"])
        object.__setattr__(self, "evidence_cycle_id", kwargs["evidence_cycle_id"])
        object.__setattr__(self, "_request_digest", MappingProxyType(dict(kwargs["request_digest"])))
        object.__setattr__(self, "_reference_receipt_digest", MappingProxyType(dict(kwargs["reference_receipt_digest"])))
        object.__setattr__(self, "target_state", kwargs["target_state"])
        object.__setattr__(self, "_authority", _AUTHORITY)
        _PLAN_BINDINGS[self] = _binding_identity(vars(self))

    def _validate(self) -> None:
        if (
            type(self) is not FrozenDefinitionPublicationPlan
            or getattr(self, "_authority", None) is not _AUTHORITY
            or _PLAN_BINDINGS.get(self) != _binding_identity(vars(self))
            or self.target_state not in {"ABSENT", "PRESENT"}
        ):
            raise FrozenDefinitionPublicationContractError("publication plan authority is absent")
        _definition_id(self.definition_set_id)
        _cycle_id(self.evidence_cycle_id)
        _typed_request_digest(dict(self._request_digest))
        _typed_reference_digest(dict(self._reference_receipt_digest))

    @property
    def request_digest(self) -> Mapping[str, Any]:
        self._validate()
        return MappingProxyType(dict(self._request_digest))

    @property
    def publication_capability(self) -> bool:
        self._validate()
        return False

    def __copy__(self) -> Any:
        raise FrozenDefinitionPublicationContractError("publication plan replay is not authoritative")

    __deepcopy__ = lambda self, memo: self.__copy__()

    def to_safe_summary_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "status": "PREPARED", "schema_version": 1,
            "definition_set_id": self.definition_set_id,
            "evidence_cycle_id": self.evidence_cycle_id,
            "request_digest": dict(self._request_digest),
            "reference_receipt_digest": dict(self._reference_receipt_digest),
            "target_state": self.target_state,
            "publication_capability": False,
            "definition_bytes_durable": False,
            "durable_reference_recorded": False, "c1b1_complete": False,
            "epoch_started": False, "prospective_claim": False,
            "promotion_capability": False,
        }


class FrozenDefinitionBytePublicationResult:
    """Writer-owned outer result joined to one C0 receipt."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise FrozenDefinitionPublicationContractError("publication results are writer-owned")
        expected = {
            "receipt", "evidence_cycle_id", "reference_receipt_digest",
            "outer_uncertain",
        }
        if set(kwargs) != expected:
            raise FrozenDefinitionPublicationContractError("publication result fields are not exact")
        from quantpits.research.definition_store import DefinitionStoreReceipt
        if type(kwargs["receipt"]) is not DefinitionStoreReceipt or type(kwargs["outer_uncertain"]) is not bool:
            raise FrozenDefinitionPublicationContractError("publication result authority is invalid")
        object.__setattr__(self, "_receipt", kwargs["receipt"])
        object.__setattr__(self, "evidence_cycle_id", kwargs["evidence_cycle_id"])
        object.__setattr__(self, "_reference_receipt_digest", MappingProxyType(dict(kwargs["reference_receipt_digest"])))
        object.__setattr__(self, "_outer_uncertain", kwargs["outer_uncertain"])
        object.__setattr__(self, "_authority", _AUTHORITY)
        _RESULT_BINDINGS[self] = _binding_identity(vars(self))
        self._validate()

    def _validate(self) -> None:
        from quantpits.research.definition_store import DefinitionStoreReceipt
        if (
            type(self) is not FrozenDefinitionBytePublicationResult
            or getattr(self, "_authority", None) is not _AUTHORITY
            or _RESULT_BINDINGS.get(self) != _binding_identity(vars(self))
            or type(self._receipt) is not DefinitionStoreReceipt
        ):
            raise FrozenDefinitionPublicationContractError("publication result authority is absent")
        receipt = self._receipt
        _cycle_id(self.evidence_cycle_id)
        _typed_reference_digest(dict(self._reference_receipt_digest))
        # Rebuild the public receipt representation and correlate all capability fields.
        rendered = _revalidate_store_receipt(receipt)
        if (
            rendered["definition_set_id"] != receipt.definition_set_id
            or rendered["request_digest"] != dict(receipt.request_digest)
            or rendered["status"] != receipt.status
            or rendered["did_write"] is not receipt.did_write
            or receipt.status not in {"COMMITTED", "ADOPTED", "CONFLICT", "UNCERTAIN"}
        ):
            raise FrozenDefinitionPublicationContractError("C0 receipt join is invalid")
        if self._outer_uncertain and receipt.status == "UNCERTAIN":
            raise FrozenDefinitionPublicationContractError("outer uncertainty combination is invalid")

    @property
    def status(self) -> str:
        self._validate()
        return "UNCERTAIN" if self._outer_uncertain else self._receipt.status

    @property
    def definition_bytes_durable(self) -> bool:
        self._validate()
        return not self._outer_uncertain and self._receipt.status in {"COMMITTED", "ADOPTED"}

    def __copy__(self) -> Any:
        raise FrozenDefinitionPublicationContractError("publication result replay is not authoritative")

    __deepcopy__ = lambda self, memo: self.__copy__()

    def to_safe_summary_dict(self) -> Dict[str, Any]:
        self._validate()
        receipt = self._receipt
        return {
            "status": self.status, "schema_version": 1,
            "reason_code": (
                "OUTER_PUBLICATION_IDENTITY_UNCERTAIN"
                if self._outer_uncertain else receipt.reason_code
            ),
            "did_write": receipt.did_write,
            "definition_set_id": receipt.definition_set_id,
            "evidence_cycle_id": self.evidence_cycle_id,
            "request_digest": dict(receipt.request_digest),
            "reference_receipt_digest": dict(self._reference_receipt_digest),
            "store_operation_id": receipt.operation_id,
            "store_manifest_digest": (
                None if self._outer_uncertain or receipt.manifest_digest is None
                else dict(receipt.manifest_digest)
            ),
            "store_member_count": receipt.member_count,
            "definition_bytes_durable": self.definition_bytes_durable,
            "durable_reference_recorded": False, "c1b1_complete": False,
            "epoch_started": False, "prospective_claim": False,
            "promotion_capability": False,
        }


def _fresh_candidate(
    root: Path, evidence_cycle_id: Any, activation: Path,
) -> Any:
    from quantpits.research.forward_observation import (
        observe_frozen_shadow_forward_definition_candidate,
    )
    return observe_frozen_shadow_forward_definition_candidate(
        root, evidence_cycle_id, activation,
    )


def _fresh_segment_candidate(
    production: Path, research: Path, evidence_cycle_id: Any, activation: Path,
) -> Any:
    from quantpits.research.forward_observation import (
        observe_fresh_champion_segment_candidate,
    )
    return observe_fresh_champion_segment_candidate(
        production, research, evidence_cycle_id, activation,
    )


def _fresh_segment_request(candidate: Any) -> Any:
    """Rebuild and join the inspector-owned candidate to one exact C0 request."""
    from quantpits.research.forward_observation import FreshChampionSegmentCandidate
    if type(candidate) is not FreshChampionSegmentCandidate:
        raise FrozenDefinitionPublicationContractError(
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
        raise FrozenDefinitionPublicationContractError(
            "fresh candidate authority claims are invalid",
        )
    request = candidate.compiled_definitions.to_store_request()
    if (
        request.definition_set_id != candidate.definition_set_id
        or dict(request.request_digest) != dict(candidate.compiled_request_digest)
    ):
        raise FrozenDefinitionPublicationContractError(
            "fresh candidate and C0 request do not join",
        )
    return request


def _public_target_exact(store: Path, request: Any, receipt: Any) -> bool:
    """Re-establish positive C0 bytes through the canonical public target name."""
    from quantpits.research.definition_store import (
        DEFINITION_PATHS, MANIFEST_NAME, _manifest_bytes,
    )
    target = store / request.definition_set_id
    target_info = os.lstat(str(target))
    target_continuity = (
        target_info.st_dev, target_info.st_ino, target_info.st_mode,
        target_info.st_nlink, target_info.st_ctime_ns,
    )
    target_identity = (
        target_info.st_dev, target_info.st_ino, target_info.st_mode, 0, 0,
    )
    if (
        stat.S_ISLNK(target_info.st_mode)
        or not stat.S_ISDIR(target_info.st_mode)
        or stat.S_IMODE(target_info.st_mode) != 0o700
        or target_identity != receipt.bundle_root_identity
    ):
        return False
    expected = {
        member.logical_path: member.canonical_json_bytes
        for member in request.members
    }
    expected[MANIFEST_NAME] = _manifest_bytes(request)
    if tuple(sorted(os.listdir(str(target)))) != tuple(sorted(expected)):
        return False
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    for name in tuple(DEFINITION_PATHS) + (MANIFEST_NAME,):
        path = target / name
        before = os.lstat(str(path))
        data = expected[name]
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != len(data)
        ):
            return False
        descriptor = os.open(str(path), flags)
        try:
            opened = os.fstat(descriptor)
            chunks = []
            remaining = len(data)
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            trailing = os.read(descriptor, 1)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = os.lstat(str(path))
        identity = lambda info: (
            info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns,
        )
        if (
            identity(before) != identity(opened)
            or identity(opened) != identity(opened_after)
            or identity(opened_after) != identity(after)
            or b"".join(chunks) != data
            or trailing
        ):
            return False
    target_after = os.lstat(str(target))
    manifest = expected[MANIFEST_NAME]
    return (
        target_continuity == (
            target_after.st_dev, target_after.st_ino, target_after.st_mode,
            target_after.st_nlink, target_after.st_ctime_ns,
        )
        and TypedDigest.raw(manifest).to_dict() == dict(receipt.manifest_digest)
        and _directory_identity(store, private=True) == receipt.store_root_identity_after
    )


def _target_state(store: Path, definition_set_id: str) -> str:
    target = store / definition_set_id
    try:
        os.lstat(str(target))
    except FileNotFoundError:
        return "ABSENT"
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _input("definition target cannot be observed") from exc
    return "PRESENT"


def _prepare_frozen_shadow_forward_definition_publication(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any,
) -> FrozenDefinitionPublicationPlan:
    """Freshly compile one formal candidate without writing or granting authority."""
    root, activation, store, identities = _formal_inputs(
        workspace_root, activation_path, definition_store_root,
    )
    target = store / activation.stem
    protected_before = _protected_inventory(store.parent, target)
    candidate = _fresh_candidate(root, evidence_cycle_id, activation)
    request = candidate.to_store_request()
    if activation.name != request.definition_set_id + ".json":
        raise _input("activation public name does not match the definition set")
    state = _target_state(store, request.definition_set_id)
    if (
        not _same_identities(root, activation, store, identities)
        or _protected_inventory(store.parent, target) != protected_before
    ):
        raise _input("publication inputs changed during preflight")
    return FrozenDefinitionPublicationPlan(
        _authority=_AUTHORITY,
        definition_set_id=request.definition_set_id,
        evidence_cycle_id=candidate.evidence_cycle_id,
        request_digest=request.request_digest,
        reference_receipt_digest=candidate.reference_receipt_digest,
        target_state=state,
    )


def _publish_frozen_shadow_forward_definition_bundle(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, expected_definition_set_id: Any,
    expected_request_digest: Any, authorization_action: Any,
) -> FrozenDefinitionBytePublicationResult:
    """Freshly observe and explicitly publish/adopt exactly one C0 bundle."""
    if type(authorization_action) is not str or authorization_action != AUTHORIZATION_ACTION:
        raise FrozenDefinitionPublicationContractError("publication authorization is invalid")
    expected_id = _definition_id(expected_definition_set_id)
    expected_digest = _typed_request_digest(expected_request_digest)
    root, activation, store, identities = _formal_inputs(
        workspace_root, activation_path, definition_store_root,
    )
    target = store / expected_id
    protected_before = _protected_inventory(store.parent, target)
    candidate = _fresh_candidate(root, evidence_cycle_id, activation)
    request = candidate.to_store_request()
    if (
        request.definition_set_id != expected_id
        or activation.name != expected_id + ".json"
        or dict(request.request_digest) != expected_digest
    ):
        raise FrozenDefinitionPublicationContractError(
            "fresh publication request does not match owner authorization",
        )
    if (
        not _same_identities(root, activation, store, identities)
        or _protected_inventory(store.parent, target) != protected_before
    ):
        raise _input("publication inputs changed before C0")
    from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
    receipt = CreateOnlyDefinitionBundleStore(store).publish(request)
    joined = (
        receipt.definition_set_id == request.definition_set_id
        and dict(receipt.request_digest) == dict(request.request_digest)
        and receipt.store_root_identity_before == identities["store"]
    )
    if not joined:
        raise FrozenDefinitionPublicationContractError("C0 receipt does not join the fresh request")
    try:
        stable = (
            _same_identities(root, activation, store, identities, after_write=True)
            and _protected_inventory(store.parent, target) == protected_before
        )
    except _PROCESS_CONTROL:
        raise
    except Exception:
        # C0 has crossed the irreversible namespace boundary.  A failed outer
        # observation must retain its writer-owned receipt and write fact.
        stable = False
    outer_uncertain = not stable and receipt.status != "UNCERTAIN"
    return FrozenDefinitionBytePublicationResult(
        _authority=_AUTHORITY, receipt=receipt,
        evidence_cycle_id=candidate.evidence_cycle_id,
        reference_receipt_digest=candidate.reference_receipt_digest,
        outer_uncertain=outer_uncertain,
    )


def prepare_frozen_shadow_forward_definition_publication(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any,
) -> FrozenDefinitionPublicationPlan:
    """Fail closed with typed errors while preserving process control."""
    try:
        return _prepare_frozen_shadow_forward_definition_publication(
            workspace_root, evidence_cycle_id, activation_path,
            definition_store_root,
        )
    except _PROCESS_CONTROL:
        raise
    except FrozenDefinitionPublicationContractError:
        raise
    except Exception as exc:
        raise _input("publication preflight failed closed") from exc


def publish_frozen_shadow_forward_definition_bundle(
    workspace_root: Any, evidence_cycle_id: Any, activation_path: Any,
    definition_store_root: Any, expected_definition_set_id: Any,
    expected_request_digest: Any, authorization_action: Any,
) -> FrozenDefinitionBytePublicationResult:
    """Fail closed with typed errors while preserving process control."""
    try:
        return _publish_frozen_shadow_forward_definition_bundle(
            workspace_root, evidence_cycle_id, activation_path,
            definition_store_root, expected_definition_set_id,
            expected_request_digest, authorization_action,
        )
    except _PROCESS_CONTROL:
        raise
    except FrozenDefinitionPublicationContractError:
        raise
    except Exception as exc:
        raise _input("definition publication failed closed") from exc


def _prepare_fresh_champion_segment_definition_publication(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any, definition_store_root: Any,
) -> FrozenDefinitionPublicationPlan:
    production, research, activation, store, identities = _split_formal_inputs(
        production_workspace_root, research_workspace_root,
        activation_path, definition_store_root,
    )
    target = store / activation.stem
    protected_before = _protected_inventory(
        research, target, mutable_parent=store,
        byte_budget=_FRESH_PROTECTED_BYTE_BUDGET,
    )
    candidate = _fresh_segment_candidate(
        production, research, evidence_cycle_id, activation,
    )
    request = _fresh_segment_request(candidate)
    if activation.name != request.definition_set_id + ".json":
        raise _input("activation public name does not match the fresh definition set")
    state = _target_state(store, request.definition_set_id)
    if (
        not _same_split_identities(
            production, research, activation, store, identities,
        )
        or _protected_inventory(
            research, target, mutable_parent=store,
            byte_budget=_FRESH_PROTECTED_BYTE_BUDGET,
        ) != protected_before
    ):
        raise _input("fresh publication inputs changed during preflight")
    return FrozenDefinitionPublicationPlan(
        _authority=_AUTHORITY,
        definition_set_id=request.definition_set_id,
        evidence_cycle_id=candidate.evidence_cycle_id,
        request_digest=request.request_digest,
        reference_receipt_digest=candidate.reference_receipt_digest,
        target_state=state,
    )


def _publish_fresh_champion_segment_definition_bundle(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any, definition_store_root: Any,
    expected_definition_set_id: Any, expected_request_digest: Any,
    authorization_action: Any,
) -> FrozenDefinitionBytePublicationResult:
    if (
        type(authorization_action) is not str
        or authorization_action != FRESH_AUTHORIZATION_ACTION
    ):
        raise FrozenDefinitionPublicationContractError(
            "fresh publication authorization is invalid",
        )
    expected_id = _definition_id(expected_definition_set_id)
    expected_digest = _typed_request_digest(expected_request_digest)
    production, research, activation, store, identities = _split_formal_inputs(
        production_workspace_root, research_workspace_root,
        activation_path, definition_store_root,
    )
    target = store / expected_id
    protected_before = _protected_inventory(
        research, target, mutable_parent=store,
        byte_budget=_FRESH_PROTECTED_BYTE_BUDGET,
    )
    candidate = _fresh_segment_candidate(
        production, research, evidence_cycle_id, activation,
    )
    request = _fresh_segment_request(candidate)
    if (
        request.definition_set_id != expected_id
        or candidate.definition_set_id != expected_id
        or activation.name != expected_id + ".json"
        or dict(candidate.compiled_request_digest) != expected_digest
        or dict(request.request_digest) != expected_digest
    ):
        raise FrozenDefinitionPublicationContractError(
            "fresh publication request does not match owner authorization",
        )
    if (
        not _same_split_identities(
            production, research, activation, store, identities,
        )
        or _protected_inventory(
            research, target, mutable_parent=store,
            byte_budget=_FRESH_PROTECTED_BYTE_BUDGET,
        ) != protected_before
    ):
        raise _input("fresh publication inputs changed before C0")
    from quantpits.research.definition_store import CreateOnlyDefinitionBundleStore
    receipt = CreateOnlyDefinitionBundleStore(store).publish(request)
    joined = (
        receipt.definition_set_id == candidate.definition_set_id == request.definition_set_id
        and dict(candidate.compiled_request_digest) == dict(request.request_digest)
        and dict(receipt.request_digest) == dict(request.request_digest)
        and receipt.store_root_identity_before == identities["research_store"]
    )
    if not joined:
        raise FrozenDefinitionPublicationContractError(
            "C0 receipt does not join the fresh split-root request",
        )
    try:
        _revalidate_store_receipt(receipt)
        stable = (
            _same_split_identities(
                production, research, activation, store, identities,
                after_write=receipt.did_write,
            )
            and _protected_inventory(
                research, target, mutable_parent=store,
                byte_budget=_FRESH_PROTECTED_BYTE_BUDGET,
            ) == protected_before
            and (
                receipt.status not in {"COMMITTED", "ADOPTED"}
                or _public_target_exact(store, request, receipt)
            )
            and _same_split_identities(
                production, research, activation, store, identities,
                after_write=receipt.did_write,
            )
            and _protected_inventory(
                research, target, mutable_parent=store,
                byte_budget=_FRESH_PROTECTED_BYTE_BUDGET,
            ) == protected_before
        )
    except _PROCESS_CONTROL:
        raise
    except Exception:
        # C0 may already have crossed its irreversible namespace edge.  Keep
        # the writer-owned receipt and expose only an outer UNCERTAIN result.
        stable = False
    outer_uncertain = not stable and receipt.status != "UNCERTAIN"
    return FrozenDefinitionBytePublicationResult(
        _authority=_AUTHORITY, receipt=receipt,
        evidence_cycle_id=candidate.evidence_cycle_id,
        reference_receipt_digest=candidate.reference_receipt_digest,
        outer_uncertain=outer_uncertain,
    )


def prepare_fresh_champion_segment_definition_publication(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any, definition_store_root: Any,
) -> FrozenDefinitionPublicationPlan:
    """Prepare one split-root fresh publication without any write capability."""
    try:
        return _prepare_fresh_champion_segment_definition_publication(
            production_workspace_root, research_workspace_root,
            evidence_cycle_id, activation_path, definition_store_root,
        )
    except _PROCESS_CONTROL:
        raise
    except FrozenDefinitionPublicationContractError:
        raise
    except Exception as exc:
        raise _input("fresh publication preflight failed closed") from exc


def publish_fresh_champion_segment_definition_bundle(
    production_workspace_root: Any, research_workspace_root: Any,
    evidence_cycle_id: Any, activation_path: Any, definition_store_root: Any,
    expected_definition_set_id: Any, expected_request_digest: Any,
    authorization_action: Any,
) -> FrozenDefinitionBytePublicationResult:
    """Freshly observe and create/adopt one authorized split-root C0 bundle."""
    try:
        return _publish_fresh_champion_segment_definition_bundle(
            production_workspace_root, research_workspace_root,
            evidence_cycle_id, activation_path, definition_store_root,
            expected_definition_set_id, expected_request_digest,
            authorization_action,
        )
    except _PROCESS_CONTROL:
        raise
    except FrozenDefinitionPublicationContractError:
        raise
    except Exception as exc:
        raise _input("fresh definition publication failed closed") from exc


__all__ = [
    "AUTHORIZATION_ACTION", "FRESH_AUTHORIZATION_ACTION",
    "FrozenDefinitionPublicationContractError",
    "FrozenDefinitionPublicationInputError", "FrozenDefinitionPublicationPlan",
    "FrozenDefinitionBytePublicationResult",
    "prepare_frozen_shadow_forward_definition_publication",
    "publish_frozen_shadow_forward_definition_bundle",
    "prepare_fresh_champion_segment_definition_publication",
    "publish_fresh_champion_segment_definition_bundle",
]
