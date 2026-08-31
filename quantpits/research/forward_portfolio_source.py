"""Read-only, member-scoped admission of a sealed Production portfolio source."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import sys
import weakref
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from quantpits.evidence.inspection import SourceMutationObserver


_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_BINDINGS: "weakref.WeakKeyDictionary[Any, Tuple[Any, ...]]" = weakref.WeakKeyDictionary()
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_INSTRUMENT_RE = re.compile(r"^(?:SH|SZ)[0-9]{6}$")
_DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SEAL_LIMIT = 128 * 1024
_MANIFEST_LIMIT = 16 * 1024 * 1024
_PORTFOLIO_LIMIT = 4 * 1024 * 1024
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
_PORTFOLIO_FIELDS = frozenset({
    "path", "status", "digest", "preservation_status", "detail",
    "canonical_digest", "holding_count",
})
_PROBLEM_FIELDS = frozenset({"code", "evidence_class", "detail", "blocks_complete"})
_ALLOWED_PARTIAL = frozenset({
    ("deep_analysis_missing", "deep_analysis", True),
    ("deep_analysis_incomplete", "deep_analysis", True),
})
_POSITIVE_STATUSES = frozenset({"VERIFIED", "VERIFIED_WITH_UNRELATED_CYCLE_DEBT"})


class ForwardPortfolioSourceContractError(ValueError):
    """The observer request or a public replay violates the R2 contract."""


class ForwardPortfolioSourceInputError(ForwardPortfolioSourceContractError):
    """The selected source cannot be compared to the R2 admission contract."""


class _Blocked(Exception):
    def __init__(self, reason_code: str, *, source_status: Optional[str] = None,
                 problems: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.source_status = source_status
        self.problems = tuple(dict(item) for item in problems)


class _Uncertain(Exception):
    def __init__(self, reason_code: str, *, source_status: Optional[str] = None,
                 problems: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.source_status = source_status
        self.problems = tuple(dict(item) for item in problems)


def _identity(info: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns,
    )


def _directory_identity(path: Path, *, private: bool = False) -> Tuple[int, int, int, int]:
    try:
        info = os.lstat(str(path))
    except FileNotFoundError as exc:
        raise _Blocked("SOURCE_DIRECTORY_MISSING") from exc
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _Uncertain("SOURCE_DIRECTORY_INCOMPARABLE") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _Blocked("SOURCE_DIRECTORY_NOT_PHYSICAL")
    if private and stat.S_IMODE(info.st_mode) != 0o700:
        raise _Blocked("SOURCE_DIRECTORY_MODE_INVALID")
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink


def _ancestor_identities(
    root: Path, relative: str, *, private_final: bool = False,
) -> Tuple[Tuple[str, Tuple[int, int, int, int]], ...]:
    current = root
    result = [(".", _directory_identity(root))]
    parts = Path(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        identity = _directory_identity(
            current, private=private_final and index == len(parts) - 1,
        )
        result.append((Path(*parts[:index + 1]).as_posix(), identity))
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _Blocked("SOURCE_ANCESTOR_CONTAINMENT_INVALID") from exc
    if resolved != current:
        raise _Blocked("SOURCE_ANCESTOR_NOT_PHYSICAL")
    return tuple(result)


class _ContinuityGuard:
    """Watch the workspace, every source ancestor, and the full cycle tree."""

    def __init__(self, root: Path, relative: str) -> None:
        observers = []
        try:
            observers.append(SourceMutationObserver(root, (relative,)))
            current = root
            for part in Path(relative).parts:
                candidate = current / part
                try:
                    info = os.lstat(str(candidate))
                except FileNotFoundError:
                    break
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    break
                current = candidate
                observers.append(SourceMutationObserver(current, ()))
        except BaseException:
            for observer in reversed(observers):
                try:
                    observer.close()
                except BaseException:
                    pass
            raise
        self._observers = tuple(observers)
        self.supported = all(item.supported for item in self._observers)

    def mutated(self) -> bool:
        return any(item.mutated() for item in self._observers)

    def close(self) -> None:
        failure = None
        for observer in reversed(self._observers):
            try:
                observer.close()
            except _PROCESS_CONTROL:
                raise
            except OSError as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure


def _physical_root(value: Any) -> Path:
    if isinstance(value, bool):
        raise ForwardPortfolioSourceContractError("workspace_root must be an absolute path")
    try:
        root = Path(value)
    except (TypeError, ValueError) as exc:
        raise ForwardPortfolioSourceContractError("workspace_root must be an absolute path") from exc
    if not root.is_absolute():
        raise ForwardPortfolioSourceContractError("workspace_root must be absolute")
    try:
        resolved = root.resolve(strict=True)
        direct = os.lstat(str(root))
        physical = os.lstat(str(resolved))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ForwardPortfolioSourceInputError("workspace_root is unavailable") from exc
    if (
        resolved != root or stat.S_ISLNK(direct.st_mode) or not stat.S_ISDIR(direct.st_mode)
        or _identity(direct) != _identity(physical)
    ):
        raise ForwardPortfolioSourceInputError(
            "workspace_root must be its physical canonical directory"
        )
    return resolved


def _cycle_id(value: Any) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise ForwardPortfolioSourceContractError("cycle_id must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ForwardPortfolioSourceContractError("cycle_id must be a real date") from exc
    if parsed.isoformat() != value:
        raise ForwardPortfolioSourceContractError("cycle_id must be canonical YYYY-MM-DD")
    return value


def _read_regular(
    root: Path, path: Path, *, maximum: int, expected_size: Optional[int] = None,
) -> Tuple[bytes, Tuple[int, int, int, int, int, int]]:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise _Blocked("SOURCE_MEMBER_OUTSIDE_WORKSPACE") from exc
    if not relative.parts:
        raise _Blocked("SOURCE_MEMBER_PATH_INVALID")
    directory_descriptors = []
    descriptor = -1
    try:
        before = os.lstat(str(path))
    except FileNotFoundError as exc:
        raise _Blocked("SOURCE_MEMBER_MISSING") from exc
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _Uncertain("SOURCE_MEMBER_INCOMPARABLE") from exc
    if (
        stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise _Blocked("SOURCE_MEMBER_NOT_SINGLE_LINK_REGULAR")
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise _Blocked("SOURCE_MEMBER_MODE_INVALID")
    if expected_size is not None:
        if type(expected_size) is not int or expected_size < 0 or expected_size > maximum:
            raise _Blocked("SOURCE_MEMBER_DECLARED_SIZE_INVALID")
        if before.st_size != expected_size:
            raise _Blocked("SOURCE_MEMBER_SIZE_MISMATCH")
        limit = expected_size
    else:
        limit = maximum
        if before.st_size > limit:
            raise _Blocked("SOURCE_MEMBER_TOO_LARGE")
    try:
        current_descriptor = os.open(
            str(root), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        directory_descriptors.append(current_descriptor)
        for part in relative.parts[:-1]:
            current_descriptor = os.open(
                part, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_descriptor,
            )
            directory_descriptors.append(current_descriptor)
        descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptors[-1],
        )
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before) or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1 or opened.st_size > limit
        ):
            raise _Uncertain("SOURCE_MEMBER_CHANGED_WHILE_OPENING")
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
            raise _Uncertain("SOURCE_MEMBER_CHANGED_WHILE_READING")
    except _PROCESS_CONTROL:
        raise
    except (_Blocked, _Uncertain):
        raise
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise _Uncertain("SOURCE_MEMBER_PATH_CHANGED") from exc
        raise _Uncertain("SOURCE_MEMBER_READ_FAILED") from exc
    finally:
        active = sys.exc_info()[1]
        descriptors = ([descriptor] if descriptor >= 0 else []) + list(
            reversed(directory_descriptors)
        )
        close_failure = None
        for current in descriptors:
            try:
                os.close(current)
            except _PROCESS_CONTROL:
                if not isinstance(active, _PROCESS_CONTROL):
                    raise
            except OSError as exc:
                if active is None and close_failure is None:
                    close_failure = exc
        if close_failure is not None:
            raise _Uncertain("SOURCE_DESCRIPTOR_CLOSE_FAILED") from close_failure
    try:
        after = os.lstat(str(path))
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _Uncertain("SOURCE_MEMBER_FINAL_IDENTITY_UNAVAILABLE") from exc
    if _identity(after) != _identity(before):
        raise _Uncertain("SOURCE_MEMBER_PUBLIC_IDENTITY_CHANGED")
    return data, _identity(before)


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
            data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject,
        )
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _Blocked("%s_JSON_INVALID" % name.upper()) from exc
    if type(value) is not dict or canonical_json_bytes(value) != data:
        raise _Blocked("%s_NOT_CANONICAL" % name.upper())
    return value


def _typed_digest(value: Any, name: str, domain: str) -> TypedDigest:
    if type(value) is not dict or set(value) != {"algorithm", "domain", "value", "size_bytes"}:
        raise _Blocked("%s_DIGEST_INVALID" % name.upper())
    try:
        digest = TypedDigest(**value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _Blocked("%s_DIGEST_INVALID" % name.upper()) from exc
    if digest.domain != domain:
        raise _Blocked("%s_DIGEST_DOMAIN_INVALID" % name.upper())
    return digest


def _inventory(path: Path) -> Tuple[str, ...]:
    try:
        return tuple(sorted(item.name for item in os.scandir(str(path))))
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise _Uncertain("SOURCE_INVENTORY_INCOMPARABLE") from exc


def _embedded_digests(value: Any) -> Tuple[Tuple[str, int], ...]:
    found = []

    def visit(item: Any) -> None:
        if type(item) is dict:
            if item.get("preservation_status") == "embedded":
                required = {"path", "status", "digest", "preservation_status", "detail"}
                if not required.issubset(item):
                    raise _Blocked("EMBEDDED_MEMBER_INVALID")
                digest_value = item.get("digest")
                if type(digest_value) is not dict:
                    raise _Blocked("EMBEDDED_MEMBER_DIGEST_INVALID")
                domain = digest_value.get("domain")
                if domain not in {"raw_bytes", "semantic_config", "file_inventory"}:
                    raise _Blocked("EMBEDDED_MEMBER_DIGEST_INVALID")
                digest = _typed_digest(digest_value, "embedded_member", domain)
                found.append((digest.value, digest.size_bytes))
            for child in item.values():
                visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)

    visit(value)
    sizes: Dict[str, set] = {}
    for digest, size in found:
        sizes.setdefault(digest, set()).add(size)
    if any(len(values) != 1 for values in sizes.values()):
        raise _Blocked("EMBEDDED_MEMBER_SIZE_CONFLICT")
    return tuple(sorted((digest, next(iter(sizes[digest]))) for digest in sizes))


def _problem_inventory(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if type(value) is not list:
        raise _Blocked("PROBLEM_INVENTORY_INVALID")
    result = []
    identities = set()
    for item in value:
        if (
            type(item) is not dict or set(item) != _PROBLEM_FIELDS
            or type(item.get("code")) is not str or not item["code"]
            or type(item.get("evidence_class")) is not str or not item["evidence_class"]
            or type(item.get("detail")) is not str
            or type(item.get("blocks_complete")) is not bool
        ):
            raise _Blocked("PROBLEM_INVENTORY_INVALID")
        identity = (item["code"], item["evidence_class"], item["blocks_complete"])
        if identity in identities:
            raise _Blocked("PROBLEM_INVENTORY_DUPLICATE")
        identities.add(identity)
        result.append(MappingProxyType(dict(item)))
    return tuple(result)


def _canonical_decimal(value: Any, name: str, *, non_negative: bool) -> Decimal:
    if type(value) is not str or _DECIMAL_RE.fullmatch(value) is None:
        raise _Blocked("PORTFOLIO_%s_INVALID" % name.upper())
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise _Blocked("PORTFOLIO_%s_INVALID" % name.upper()) from exc
    if not number.is_finite() or (non_negative and (number < 0 or value.startswith("-"))):
        raise _Blocked("PORTFOLIO_%s_INVALID" % name.upper())
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if (number == 0 and value != "0") or (rendered or "0") != value:
        raise _Blocked("PORTFOLIO_%s_NOT_CANONICAL" % name.upper())
    return number


def _validate_portfolio(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"current_cash", "current_holding"}:
        raise _Blocked("PORTFOLIO_SCHEMA_INVALID")
    _canonical_decimal(payload["current_cash"], "cash", non_negative=False)
    holdings = payload["current_holding"]
    if type(holdings) is not list:
        raise _Blocked("PORTFOLIO_HOLDINGS_INVALID")
    instruments = []
    for holding in holdings:
        if type(holding) is not dict or set(holding) != {"instrument", "value", "amount"}:
            raise _Blocked("PORTFOLIO_HOLDING_SCHEMA_INVALID")
        instrument = holding["instrument"]
        if type(instrument) is not str or _INSTRUMENT_RE.fullmatch(instrument) is None:
            raise _Blocked("PORTFOLIO_INSTRUMENT_INVALID")
        quantity = _canonical_decimal(holding["value"], "quantity", non_negative=True)
        if quantity <= 0 or quantity != quantity.to_integral_value():
            raise _Blocked("PORTFOLIO_QUANTITY_INVALID")
        _canonical_decimal(holding["amount"], "book_cost", non_negative=True)
        instruments.append(instrument)
    if len(set(instruments)) != len(instruments):
        raise _Blocked("PORTFOLIO_INSTRUMENT_DUPLICATE")


def _problem_bytes(problems: Sequence[Mapping[str, Any]]) -> bytes:
    return canonical_json_bytes([dict(item) for item in problems])


@dataclass(frozen=True, init=False, eq=False)
class ForwardPortfolioSourceObservation:
    """Inspector-owned R2 result; only positive instances expose source bytes."""

    cycle_id: str
    status: str
    reason_code: str
    source_cycle_status: Optional[str]
    problem_count: int
    problem_inventory_digest: Mapping[str, Any]
    seal_digest: Optional[Mapping[str, Any]]
    manifest_digest: Optional[Mapping[str, Any]]
    portfolio_raw_digest: Optional[Mapping[str, Any]]
    portfolio_semantic_digest: Optional[Mapping[str, Any]]
    portfolio_holding_count: Optional[int]
    portfolio_member_verified: bool
    bootstrap_source_capability: bool
    did_write: bool
    _problem_inventory_bytes: bytes
    _portfolio_bytes: Optional[bytes]
    _observation_binding: Tuple[Any, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ForwardPortfolioSourceContractError("observations are inspector-owned")

    @classmethod
    def _create(cls, token: object, **values: Any) -> "ForwardPortfolioSourceObservation":
        if token is not _AUTHORITY:
            raise ForwardPortfolioSourceContractError("observations are inspector-owned")
        obj = object.__new__(cls)
        for field, value in values.items():
            object.__setattr__(obj, field, value)
        obj._validate()
        _BINDINGS[obj] = obj._authority_signature()
        return obj

    def _validate(self) -> None:
        if self.status not in _POSITIVE_STATUSES | {"BLOCKED", "UNCERTAIN"}:
            raise ForwardPortfolioSourceContractError("observation status is invalid")
        if type(self.cycle_id) is not str or _DATE_RE.fullmatch(self.cycle_id) is None:
            raise ForwardPortfolioSourceContractError("observation cycle identity is invalid")
        if type(self.reason_code) is not str or not self.reason_code:
            raise ForwardPortfolioSourceContractError("observation reason is invalid")
        if self.source_cycle_status not in {None, "sealed_complete", "sealed_partial"}:
            raise ForwardPortfolioSourceContractError("source cycle status is invalid")
        if self.did_write is not False:
            raise ForwardPortfolioSourceContractError("portfolio observation must be zero-write")
        positive = self.status in _POSITIVE_STATUSES
        if (
            self.portfolio_member_verified is not positive
            or self.bootstrap_source_capability is not positive
            or (positive and self.reason_code != "PORTFOLIO_MEMBER_VERIFIED")
            or (not positive and (self._portfolio_bytes is not None
                                  or self.portfolio_holding_count is not None))
        ):
            raise ForwardPortfolioSourceContractError("observation capability fields are inconsistent")
        if positive:
            required = (
                self.source_cycle_status, self.seal_digest, self.manifest_digest,
                self.portfolio_raw_digest, self.portfolio_semantic_digest,
                self.portfolio_holding_count, self._portfolio_bytes,
            )
            if any(value is None for value in required):
                raise ForwardPortfolioSourceContractError("verified observation evidence is incomplete")
            if self.status == "VERIFIED" and (
                self.source_cycle_status != "sealed_complete" or self.problem_count != 0
            ):
                raise ForwardPortfolioSourceContractError("complete observation is inconsistent")
            if self.status == "VERIFIED_WITH_UNRELATED_CYCLE_DEBT" and (
                self.source_cycle_status != "sealed_partial" or self.problem_count <= 0
            ):
                raise ForwardPortfolioSourceContractError("partial observation is inconsistent")
        if self.problem_count < 0:
            raise ForwardPortfolioSourceContractError("problem count is invalid")
        try:
            inventory = json.loads(self._problem_inventory_bytes.decode("utf-8"))
        except Exception as exc:
            raise ForwardPortfolioSourceContractError("problem inventory is invalid") from exc
        if (
            type(inventory) is not list
            or canonical_json_bytes(inventory) != self._problem_inventory_bytes
            or len(inventory) != self.problem_count
        ):
            raise ForwardPortfolioSourceContractError("problem inventory count is inconsistent")
        if TypedDigest.canonical(inventory).to_dict() != dict(self.problem_inventory_digest):
            raise ForwardPortfolioSourceContractError("problem inventory digest is inconsistent")
        try:
            canonical_problems = _problem_inventory(inventory)
        except _Blocked as exc:
            raise ForwardPortfolioSourceContractError("problem inventory is invalid") from exc
        if positive:
            if self.status == "VERIFIED" and canonical_problems:
                raise ForwardPortfolioSourceContractError("complete observation cannot carry problems")
            if self.status == "VERIFIED_WITH_UNRELATED_CYCLE_DEBT":
                identities = {
                    (item["code"], item["evidence_class"], item["blocks_complete"])
                    for item in canonical_problems
                }
                if not identities or not identities.issubset(_ALLOWED_PARTIAL):
                    raise ForwardPortfolioSourceContractError(
                        "partial observation problem inventory is not admissible"
                    )
            if not self._observation_binding:
                raise ForwardPortfolioSourceContractError("verified observation has no identity binding")
            try:
                seal_digest = _typed_digest(dict(self.seal_digest), "seal", "raw_bytes")
                manifest_digest = _typed_digest(
                    dict(self.manifest_digest), "manifest", "raw_bytes",
                )
                raw_digest = _typed_digest(
                    dict(self.portfolio_raw_digest), "portfolio_raw", "raw_bytes",
                )
                semantic_digest = _typed_digest(
                    dict(self.portfolio_semantic_digest),
                    "portfolio_semantic", "semantic_config",
                )
                portfolio = _strict_json(self._portfolio_bytes, "portfolio")
                _validate_portfolio(portfolio)
            except _Blocked as exc:
                raise ForwardPortfolioSourceContractError(
                    "verified observation member evidence is invalid"
                ) from exc
            if (
                seal_digest.size_bytes <= 0 or manifest_digest.size_bytes <= 0
                or TypedDigest.raw(self._portfolio_bytes) != raw_digest
                or TypedDigest.canonical(portfolio, "semantic_config") != semantic_digest
                or self.portfolio_holding_count != len(portfolio["current_holding"])
            ):
                raise ForwardPortfolioSourceContractError(
                    "verified observation member evidence is inconsistent"
                )
            expected_fact_binding = (
                "R2_OBSERVED_FACTS_V1",
                self.source_cycle_status,
                tuple(sorted(self.problem_inventory_digest.items())),
                tuple(sorted(seal_digest.to_dict().items())),
                tuple(sorted(manifest_digest.to_dict().items())),
                tuple(sorted(raw_digest.to_dict().items())),
                tuple(sorted(semantic_digest.to_dict().items())),
                self.portfolio_holding_count,
            )
            if self._observation_binding[-1] != expected_fact_binding:
                raise ForwardPortfolioSourceContractError(
                    "verified observation identity binding is inconsistent"
                )

    def _authority_signature(self) -> Tuple[Any, ...]:
        return (
            self.cycle_id, self.status, self.reason_code, self.source_cycle_status,
            self.problem_count, tuple(sorted(self.problem_inventory_digest.items())),
            None if self.seal_digest is None else tuple(sorted(self.seal_digest.items())),
            None if self.manifest_digest is None else tuple(sorted(self.manifest_digest.items())),
            None if self.portfolio_raw_digest is None else tuple(sorted(self.portfolio_raw_digest.items())),
            None if self.portfolio_semantic_digest is None else tuple(sorted(self.portfolio_semantic_digest.items())),
            self.portfolio_holding_count, self.portfolio_member_verified,
            self.bootstrap_source_capability, self.did_write,
            hashlib.sha256(self._problem_inventory_bytes).hexdigest(),
            None if self._portfolio_bytes is None else hashlib.sha256(self._portfolio_bytes).hexdigest(),
            self._observation_binding,
        )

    def _require_authority(self) -> None:
        if _BINDINGS.get(self) != self._authority_signature():
            raise ForwardPortfolioSourceContractError("observation authority is unavailable")

    def problem_inventory(self) -> Tuple[Mapping[str, Any], ...]:
        self._require_authority()
        values = json.loads(self._problem_inventory_bytes.decode("utf-8"))
        return tuple(MappingProxyType(dict(item)) for item in values)

    def verified_portfolio(self) -> Dict[str, Any]:
        self._require_authority()
        if not self.bootstrap_source_capability or self._portfolio_bytes is None:
            raise ForwardPortfolioSourceContractError("observation has no bootstrap source capability")
        return json.loads(self._portfolio_bytes.decode("utf-8"))

    def to_safe_dict(self) -> Dict[str, Any]:
        self._require_authority()
        payload = {
            "schema_version": 1,
            "cycle_id": self.cycle_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "source_cycle_status": self.source_cycle_status,
            "problem_count": self.problem_count,
            "problem_inventory_digest": dict(self.problem_inventory_digest),
            "seal_digest": None if self.seal_digest is None else dict(self.seal_digest),
            "manifest_digest": None if self.manifest_digest is None else dict(self.manifest_digest),
            "portfolio_raw_digest": (
                None if self.portfolio_raw_digest is None else dict(self.portfolio_raw_digest)
            ),
            "portfolio_semantic_digest": (
                None if self.portfolio_semantic_digest is None
                else dict(self.portfolio_semantic_digest)
            ),
            "portfolio_holding_count": self.portfolio_holding_count,
            "portfolio_member_verified": self.portfolio_member_verified,
            "bootstrap_source_capability": self.bootstrap_source_capability,
            "did_write": self.did_write,
            "prospective_claim": False,
            "promotion_capability": False,
        }
        payload["result_digest"] = TypedDigest.canonical(payload).to_dict()
        return payload


def _result(
    cycle_id: str, status: str, reason_code: str, *,
    source_status: Optional[str] = None,
    problems: Sequence[Mapping[str, Any]] = (),
    seal_digest: Optional[TypedDigest] = None,
    manifest_digest: Optional[TypedDigest] = None,
    portfolio_raw_digest: Optional[TypedDigest] = None,
    portfolio_semantic_digest: Optional[TypedDigest] = None,
    portfolio_bytes: Optional[bytes] = None,
    holding_count: Optional[int] = None,
    observation_binding: Tuple[Any, ...] = (),
) -> ForwardPortfolioSourceObservation:
    problem_data = _problem_bytes(problems)
    positive = status in _POSITIVE_STATUSES
    return ForwardPortfolioSourceObservation._create(
        _AUTHORITY,
        cycle_id=cycle_id,
        status=status,
        reason_code=reason_code,
        source_cycle_status=source_status,
        problem_count=len(tuple(problems)),
        problem_inventory_digest=MappingProxyType(
            TypedDigest.canonical(json.loads(problem_data.decode("utf-8"))).to_dict()
        ),
        seal_digest=None if seal_digest is None else MappingProxyType(seal_digest.to_dict()),
        manifest_digest=(
            None if manifest_digest is None else MappingProxyType(manifest_digest.to_dict())
        ),
        portfolio_raw_digest=(
            None if portfolio_raw_digest is None
            else MappingProxyType(portfolio_raw_digest.to_dict())
        ),
        portfolio_semantic_digest=(
            None if portfolio_semantic_digest is None
            else MappingProxyType(portfolio_semantic_digest.to_dict())
        ),
        portfolio_holding_count=holding_count if positive else None,
        portfolio_member_verified=positive,
        bootstrap_source_capability=positive,
        did_write=False,
        _problem_inventory_bytes=problem_data,
        _portfolio_bytes=portfolio_bytes if positive else None,
        _observation_binding=observation_binding,
    )


def _observe_bundle_impl(
    root: Path, cycle_id: str, context: Dict[str, Any],
) -> Tuple[
    str, Tuple[Mapping[str, Any], ...], TypedDigest, TypedDigest,
    TypedDigest, TypedDigest, bytes, int, Tuple[Any, ...],
]:
    relative = "data/evidence/v1/cycles/%s" % cycle_id
    cycle = root / relative
    ancestor_identities = _ancestor_identities(root, relative, private_final=True)
    root_identity = ancestor_identities[0][1]
    cycle_identity = ancestor_identities[-1][1]

    seal_data, seal_identity = _read_regular(root, cycle / "seal.json", maximum=_SEAL_LIMIT)
    seal = _strict_json(seal_data, "seal")
    if (
        set(seal) != _SEAL_FIELDS or type(seal.get("schema_version")) is not int
        or seal["schema_version"] != 1 or seal.get("cycle_id") != cycle_id
    ):
        raise _Blocked("SEAL_SCHEMA_INVALID")
    declared_manifest = _typed_digest(seal.get("manifest_digest"), "manifest", "raw_bytes")
    manifest_data, manifest_identity = _read_regular(
        root, cycle / "manifest.json", maximum=_MANIFEST_LIMIT,
        expected_size=declared_manifest.size_bytes,
    )
    manifest_digest = TypedDigest.raw(manifest_data)
    if manifest_digest != declared_manifest:
        raise _Blocked("MANIFEST_SEAL_DIGEST_MISMATCH")
    manifest = _strict_json(manifest_data, "manifest")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise _Blocked("MANIFEST_SCHEMA_INVALID")
    cycle_identity_value = manifest.get("cycle_identity")
    if type(cycle_identity_value) is not dict or cycle_identity_value.get("cycle_id") != cycle_id:
        raise _Blocked("MANIFEST_CYCLE_IDENTITY_INVALID")

    problems = _problem_inventory(manifest.get("problems"))
    derived_status = (
        "sealed_partial" if any(item["blocks_complete"] for item in problems)
        else "sealed_complete"
    )
    context["source_status"] = derived_status
    context["problems"] = problems
    if manifest.get("status") != derived_status or seal.get("status") != derived_status:
        raise _Blocked("SOURCE_STATUS_NOT_DERIVED", source_status=derived_status, problems=problems)
    replay_core = {
        key: value for key, value in manifest.items()
        if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
    }
    request_digest = _typed_digest(
        manifest.get("request_content_digest"), "request_content", "canonical_json",
    )
    if request_digest != TypedDigest.canonical(replay_core):
        raise _Blocked("REQUEST_CONTENT_DIGEST_MISMATCH", source_status=derived_status,
                       problems=problems)

    if derived_status == "sealed_complete":
        if problems:
            raise _Blocked("COMPLETE_SOURCE_HAS_PROBLEMS", source_status=derived_status,
                           problems=problems)
        result_status = "VERIFIED"
    else:
        identities = {
            (item["code"], item["evidence_class"], item["blocks_complete"])
            for item in problems
        }
        if not identities or not identities.issubset(_ALLOWED_PARTIAL):
            raise _Blocked("SOURCE_PROBLEM_NOT_ALLOWLISTED", source_status=derived_status,
                           problems=problems)
        result_status = "VERIFIED_WITH_UNRELATED_CYCLE_DEBT"

    object_digests = seal.get("object_digests")
    named_digests = seal.get("named_file_digests")
    if (
        type(object_digests) is not list or object_digests != sorted(set(object_digests))
        or any(type(value) is not str or _HEX_RE.fullmatch(value) is None
               for value in object_digests)
        or type(named_digests) is not dict
        or set(named_digests) != {"ranking.csv", "portfolio_state.json"}
    ):
        raise _Blocked("SEALED_ARTIFACT_INVENTORY_INVALID", source_status=derived_status,
                       problems=problems)
    ranking = manifest.get("ranking")
    portfolio_leaf = manifest.get("portfolio_state")
    if type(ranking) is not dict or ranking.get("ranking_digest") is None:
        raise _Blocked("RANKING_NAMED_DIGEST_MISSING", source_status=derived_status,
                       problems=problems)
    expected_ranking = _typed_digest(ranking["ranking_digest"], "ranking", "raw_bytes")
    actual_ranking = _typed_digest(named_digests["ranking.csv"], "named_ranking", "raw_bytes")
    if expected_ranking != actual_ranking:
        raise _Blocked("RANKING_NAMED_DIGEST_MISMATCH", source_status=derived_status,
                       problems=problems)
    if type(portfolio_leaf) is not dict or set(portfolio_leaf) != _PORTFOLIO_FIELDS:
        raise _Blocked("PORTFOLIO_MANIFEST_LEAF_INVALID", source_status=derived_status,
                       problems=problems)
    if (
        portfolio_leaf.get("path") != "config/prod_config.json"
        or portfolio_leaf.get("status") != "observed"
        or portfolio_leaf.get("preservation_status") != "embedded"
        or type(portfolio_leaf.get("detail")) is not str
        or type(portfolio_leaf.get("holding_count")) is not int
        or portfolio_leaf["holding_count"] < 0
    ):
        raise _Blocked("PORTFOLIO_MANIFEST_LEAF_INVALID", source_status=derived_status,
                       problems=problems)
    _typed_digest(portfolio_leaf.get("digest"), "portfolio_source", "raw_bytes")
    semantic_digest = _typed_digest(
        portfolio_leaf.get("canonical_digest"), "portfolio_semantic", "semantic_config",
    )
    named_portfolio = _typed_digest(
        named_digests["portfolio_state.json"], "named_portfolio", "raw_bytes",
    )
    if (
        semantic_digest.algorithm != named_portfolio.algorithm
        or semantic_digest.value != named_portfolio.value
        or semantic_digest.size_bytes != named_portfolio.size_bytes
    ):
        raise _Blocked("PORTFOLIO_NAMED_DIGEST_MISMATCH", source_status=derived_status,
                       problems=problems)

    embedded = _embedded_digests(manifest)
    if [value for value, _size in embedded] != object_digests:
        raise _Blocked("OBJECT_INVENTORY_MANIFEST_MISMATCH", source_status=derived_status,
                       problems=problems)
    preservation = manifest.get("preservation")
    if (
        type(preservation) is not dict
        or set(preservation) != {"embedded_object_count", "named_file_count"}
        or type(preservation.get("embedded_object_count")) is not int
        or type(preservation.get("named_file_count")) is not int
        or preservation["embedded_object_count"] != len(object_digests)
        or preservation["named_file_count"] != len(named_digests)
    ):
        raise _Blocked("PRESERVATION_COUNTS_INVALID", source_status=derived_status,
                       problems=problems)
    artifact_root = _typed_digest(
        seal.get("artifact_root_digest"), "artifact_root", "canonical_json",
    )
    if artifact_root != TypedDigest.canonical({
        "objects": object_digests, "named_files": named_digests,
    }):
        raise _Blocked("ARTIFACT_ROOT_DIGEST_MISMATCH", source_status=derived_status,
                       problems=problems)

    expected_top = {"manifest.json", "seal.json", "ranking.csv", "portfolio_state.json"}
    if object_digests:
        expected_top.add("objects")
    if set(_inventory(cycle)) != expected_top:
        raise _Blocked("SOURCE_TOP_INVENTORY_INVALID", source_status=derived_status,
                       problems=problems)
    object_binding: Tuple[Any, ...] = ()
    if object_digests:
        objects_root = cycle / "objects"
        objects_identity = _directory_identity(objects_root, private=True)
        prefixes = tuple(sorted(set(value[:2] for value in object_digests)))
        if _inventory(objects_root) != prefixes:
            raise _Blocked("OBJECT_PREFIX_INVENTORY_INVALID", source_status=derived_status,
                           problems=problems)
        prefix_bindings = []
        for prefix in prefixes:
            prefix_root = objects_root / prefix
            prefix_identity = _directory_identity(prefix_root, private=True)
            expected = tuple(value for value in object_digests if value[:2] == prefix)
            if _inventory(prefix_root) != expected:
                raise _Blocked("OBJECT_MEMBER_INVENTORY_INVALID", source_status=derived_status,
                               problems=problems)
            prefix_bindings.append((prefix, prefix_identity))
        object_binding = (objects_identity, tuple(prefix_bindings))

    portfolio_data, portfolio_identity = _read_regular(
        root, cycle / "portfolio_state.json", maximum=_PORTFOLIO_LIMIT,
        expected_size=named_portfolio.size_bytes,
    )
    portfolio_raw = TypedDigest.raw(portfolio_data)
    if portfolio_raw != named_portfolio:
        raise _Blocked("PORTFOLIO_RAW_DIGEST_MISMATCH", source_status=derived_status,
                       problems=problems)
    portfolio_payload = _strict_json(portfolio_data, "portfolio")
    _validate_portfolio(portfolio_payload)
    observed_semantic = TypedDigest.canonical(portfolio_payload, "semantic_config")
    if observed_semantic != semantic_digest:
        raise _Blocked("PORTFOLIO_SEMANTIC_DIGEST_MISMATCH", source_status=derived_status,
                       problems=problems)
    holdings = portfolio_payload["current_holding"]
    if portfolio_leaf["holding_count"] != len(holdings):
        raise _Blocked("PORTFOLIO_HOLDING_COUNT_MISMATCH", source_status=derived_status,
                       problems=problems)

    # Re-establish every capability-bearing public name and exact byte fact.
    if _ancestor_identities(root, relative, private_final=True) != ancestor_identities:
        raise _Uncertain("SOURCE_DIRECTORY_IDENTITY_DRIFT")
    final_seal, final_seal_identity = _read_regular(
        root, cycle / "seal.json", maximum=_SEAL_LIMIT,
    )
    final_manifest, final_manifest_identity = _read_regular(
        root, cycle / "manifest.json", maximum=_MANIFEST_LIMIT,
        expected_size=manifest_digest.size_bytes,
    )
    final_portfolio, final_portfolio_identity = _read_regular(
        root, cycle / "portfolio_state.json", maximum=_PORTFOLIO_LIMIT,
        expected_size=portfolio_raw.size_bytes,
    )
    if (
        final_seal != seal_data or final_manifest != manifest_data
        or final_portfolio != portfolio_data
        or final_seal_identity != seal_identity
        or final_manifest_identity != manifest_identity
        or final_portfolio_identity != portfolio_identity
    ):
        raise _Uncertain("SOURCE_FINAL_PUBLIC_VERIFICATION_FAILED")
    observed_seal_digest = TypedDigest.raw(seal_data)
    problem_digest = TypedDigest.canonical(
        [dict(item) for item in problems],
    )
    fact_binding = (
        "R2_OBSERVED_FACTS_V1",
        derived_status,
        tuple(sorted(problem_digest.to_dict().items())),
        tuple(sorted(observed_seal_digest.to_dict().items())),
        tuple(sorted(manifest_digest.to_dict().items())),
        tuple(sorted(portfolio_raw.to_dict().items())),
        tuple(sorted(observed_semantic.to_dict().items())),
        len(holdings),
    )
    binding = (
        ancestor_identities, root_identity, cycle_identity, seal_identity, manifest_identity,
        portfolio_identity, object_binding, fact_binding,
    )
    return (
        result_status, problems, observed_seal_digest, manifest_digest,
        portfolio_raw, observed_semantic, portfolio_data, len(holdings), binding,
    )


def _observe_bundle(
    root: Path, cycle_id: str,
) -> Tuple[
    str, Tuple[Mapping[str, Any], ...], TypedDigest, TypedDigest,
    TypedDigest, TypedDigest, bytes, int, Tuple[Any, ...],
]:
    context: Dict[str, Any] = {}
    try:
        return _observe_bundle_impl(root, cycle_id, context)
    except _Blocked as exc:
        if "source_status" in context:
            exc.source_status = context["source_status"]
            exc.problems = tuple(dict(item) for item in context["problems"])
        raise
    except _Uncertain as exc:
        if "source_status" in context:
            exc.source_status = context["source_status"]
            exc.problems = tuple(dict(item) for item in context["problems"])
        raise


def observe_forward_portfolio_source(
    workspace_root: Any, cycle_id: Any,
) -> ForwardPortfolioSourceObservation:
    """Freshly verify one sealed portfolio member without modifying its workspace."""
    root = _physical_root(workspace_root)
    canonical_cycle = _cycle_id(cycle_id)
    relative = "data/evidence/v1/cycles/%s" % canonical_cycle
    try:
        observer = _ContinuityGuard(root, relative)
    except _PROCESS_CONTROL:
        raise
    except Exception:
        return _result(canonical_cycle, "UNCERTAIN", "MUTATION_OBSERVER_UNAVAILABLE")
    pending: Optional[ForwardPortfolioSourceObservation] = None
    close_failed = False
    try:
        if not observer.supported:
            pending = _result(canonical_cycle, "UNCERTAIN", "MUTATION_OBSERVER_UNAVAILABLE")
        else:
            try:
                observed = _observe_bundle(root, canonical_cycle)
                status, problems, seal_digest, manifest_digest = observed[:4]
                portfolio_raw, portfolio_semantic, portfolio_data = observed[4:7]
                holding_count, binding = observed[7:9]
                pending = _result(
                    canonical_cycle, status, "PORTFOLIO_MEMBER_VERIFIED",
                    source_status=(
                        "sealed_complete" if status == "VERIFIED" else "sealed_partial"
                    ),
                    problems=problems,
                    seal_digest=seal_digest,
                    manifest_digest=manifest_digest,
                    portfolio_raw_digest=portfolio_raw,
                    portfolio_semantic_digest=portfolio_semantic,
                    portfolio_bytes=portfolio_data,
                    holding_count=holding_count,
                    observation_binding=binding,
                )
            except _Blocked as exc:
                pending = _result(
                    canonical_cycle, "BLOCKED", exc.reason_code,
                    source_status=exc.source_status, problems=exc.problems,
                )
            except _Uncertain as exc:
                pending = _result(
                    canonical_cycle, "UNCERTAIN", exc.reason_code,
                    source_status=exc.source_status, problems=exc.problems,
                )
            except _PROCESS_CONTROL:
                raise
            except Exception:
                pending = _result(canonical_cycle, "UNCERTAIN", "SOURCE_OBSERVATION_FAILED")
        try:
            mutated = observer.mutated()
        except _PROCESS_CONTROL:
            raise
        except Exception:
            mutated = True
        if mutated:
            pending = _result(
                canonical_cycle, "UNCERTAIN", "SOURCE_MUTATION_OBSERVED",
                source_status=None if pending is None else pending.source_cycle_status,
                problems=() if pending is None else pending.problem_inventory(),
            )
    finally:
        active = sys.exc_info()[1]
        try:
            observer.close()
        except _PROCESS_CONTROL:
            if not isinstance(active, _PROCESS_CONTROL):
                raise
        except OSError:
            if active is None:
                close_failed = True
    if close_failed:
        return _result(
            canonical_cycle, "UNCERTAIN", "MUTATION_OBSERVER_CLOSE_FAILED",
            source_status=None if pending is None else pending.source_cycle_status,
            problems=() if pending is None else pending.problem_inventory(),
        )
    if pending is None:
        return _result(canonical_cycle, "UNCERTAIN", "SOURCE_OBSERVATION_FAILED")
    return pending


__all__ = [
    "ForwardPortfolioSourceContractError",
    "ForwardPortfolioSourceInputError",
    "ForwardPortfolioSourceObservation",
    "observe_forward_portfolio_source",
]
