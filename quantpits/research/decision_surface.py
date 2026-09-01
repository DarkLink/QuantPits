"""Read-only Production decision-surface continuity observation.

The observer deliberately produces no durable receipt.  Positive authority is
owned by one live call and cannot be reconstructed from the privacy-safe JSON
renderer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import weakref
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from quantpits.evidence.inspection import SourceMutationObserver


SCHEMA_VERSION = 1
OBSERVATION_KIND = "PRODUCTION_DECISION_SURFACE_CONTINUITY_V1"
SURFACE_PROTOCOL = "PRODUCTION_DECISION_SURFACE_V1"
COMPONENT_NAMES = (
    "economic_code_surface",
    "champion_source_training_set",
    "ensemble_semantics",
    "prediction_resolution_protocol",
    "market_universe_policy",
    "portfolio_intent_policy",
)
CURATED_CODE_PATHS = (
    "quantpits/scripts/ensemble_fusion.py",
    "quantpits/ensemble/command.py",
    "quantpits/ensemble/config.py",
    "quantpits/ensemble/execution.py",
    "quantpits/ensemble/input_integrity.py",
    "quantpits/ensemble/persistence.py",
    "quantpits/ensemble/pipeline.py",
    "quantpits/ensemble/service.py",
    "quantpits/ensemble/types.py",
    "quantpits/utils/ensemble_plan.py",
    "quantpits/utils/ensemble_utils.py",
    "quantpits/utils/fusion_engine.py",
    "quantpits/utils/predict_utils.py",
    "quantpits/utils/train_utils.py",
    "quantpits/training/records.py",
    "quantpits/runtime/mlflow_integrity.py",
    "quantpits/scripts/order_gen.py",
    "quantpits/order/command.py",
    "quantpits/order/execution.py",
    "quantpits/order/service.py",
    "quantpits/utils/config_loader.py",
    "quantpits/utils/strategy.py",
    "quantpits/evidence/ranking.py",
)

_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_AUTHORITY = object()
_RESULT_BINDINGS = weakref.WeakKeyDictionary()  # type: ignore[var-annotated]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_MAX_JSON = 16 * 1024 * 1024
_MAX_GIT_BLOB = 8 * 1024 * 1024
_ALLOWED_PARTIAL = frozenset({
    ("deep_analysis_missing", "deep_analysis", True),
    ("deep_analysis_incomplete", "deep_analysis", True),
})


class DecisionSurfaceContractError(ValueError):
    """Caller-controlled request or public aggregate is invalid."""

    reason_code = "CONTRACT_INVALID"


class DecisionSurfaceInputError(DecisionSurfaceContractError):
    """One or more requested authorities cannot be compared safely."""

    reason_code = "INPUT_INCOMPARABLE"


class _ComponentIncomparable(DecisionSurfaceInputError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _canonical(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise DecisionSurfaceContractError("value is not canonical JSON") from exc


def _digest(value: Any, domain: str = "canonical_json") -> Dict[str, Any]:
    if domain == "raw_bytes":
        return TypedDigest.raw(value).to_dict()
    return TypedDigest.canonical(value, domain).to_dict()


def _typed_digest(value: Any, name: str, domain: Optional[str] = None) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        raise _ComponentIncomparable(name.upper() + "_DIGEST_INVALID")
    try:
        result = TypedDigest(**value)
    except Exception as exc:
        raise _ComponentIncomparable(name.upper() + "_DIGEST_INVALID") from exc
    if domain is not None and result.domain != domain:
        raise _ComponentIncomparable(name.upper() + "_DIGEST_DOMAIN_INVALID")
    return result.to_dict()


def _date(value: Any, name: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise DecisionSurfaceContractError("%s must be canonical YYYY-MM-DD" % name)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise DecisionSurfaceContractError("%s must be a real date" % name) from exc
    if parsed.isoformat() != value:
        raise DecisionSurfaceContractError("%s must be canonical YYYY-MM-DD" % name)
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        type(value) is not str or value in {".", ".."}
        or value != value.strip() or _ID_RE.fullmatch(value) is None
    ):
        raise DecisionSurfaceContractError("%s is invalid" % name)
    return value


def _physical_path(value: Any, name: str, *, directory: bool) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise DecisionSurfaceContractError("%s must be an absolute path" % name)
    path = Path(value)
    if not path.is_absolute():
        raise DecisionSurfaceContractError("%s must be absolute" % name)
    try:
        resolved = path.resolve(strict=True)
        info = os.lstat(str(path))
    except _PROCESS_CONTROL:
        raise
    except OSError as exc:
        raise DecisionSurfaceInputError("%s is unavailable" % name) from exc
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if resolved != path or stat.S_ISLNK(info.st_mode) or not expected:
        raise DecisionSurfaceInputError("%s must be a physical canonical path" % name)
    if not directory and info.st_nlink != 1:
        raise DecisionSurfaceInputError("%s must be a single-link regular file" % name)
    return path


def _strict_json(data: bytes, name: str) -> Dict[str, Any]:
    def pairs(rows: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject(_value: str) -> None:
        raise ValueError("non-finite value")

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject,
        )
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _ComponentIncomparable(name.upper() + "_JSON_INVALID") from exc
    if type(value) is not dict or _canonical(value) != data:
        raise _ComponentIncomparable(name.upper() + "_NOT_CANONICAL")
    return value


def _json_object_bytes(data: bytes, name: str) -> Dict[str, Any]:
    """Decode strict JSON while preserving a sealed noncanonical representation."""
    def pairs(rows: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject(_value: str) -> None:
        raise ValueError("non-finite value")

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject,
        )
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _ComponentIncomparable(name.upper() + "_JSON_INVALID") from exc
    if type(value) is not dict:
        raise _ComponentIncomparable(name.upper() + "_JSON_INVALID")
    return value


def _read_regular(
    path: Path, *, maximum: int = _MAX_JSON, private: bool = False,
) -> Tuple[bytes, Tuple[int, ...]]:
    descriptor = -1
    try:
        before = os.lstat(str(path))
        if (
            stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1 or before.st_size > maximum
            or (private and stat.S_IMODE(before.st_mode) != 0o600)
        ):
            raise _ComponentIncomparable("SOURCE_MEMBER_INVALID")
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
            item.st_size, item.st_mtime_ns,
        )
        if identity(opened) != identity(before):
            raise _ComponentIncomparable("SOURCE_MEMBER_OPEN_DRIFT")
        chunks = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.lstat(str(path))
        if len(data) != before.st_size or len(data) > maximum or identity(after) != identity(before):
            raise _ComponentIncomparable("SOURCE_MEMBER_READ_DRIFT")
        return data, identity(before)
    except _PROCESS_CONTROL:
        raise
    except _ComponentIncomparable:
        raise
    except OSError as exc:
        raise _ComponentIncomparable("SOURCE_MEMBER_READ_FAILED") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _selected_fingerprint(paths: Sequence[Path]) -> Tuple[Tuple[Any, ...], ...]:
    """Fingerprint only the predeclared read authorities, never a whole workspace."""
    rows = []
    seen = set()
    for selected in paths:
        key = str(selected)
        if key in seen:
            continue
        seen.add(key)
        try:
            info = os.lstat(key)
        except FileNotFoundError:
            rows.append((key, "ABSENT"))
            continue
        if stat.S_ISLNK(info.st_mode):
            rows.append((key, "SYMLINK", info.st_mode, info.st_ctime_ns))
            continue
        if stat.S_ISREG(info.st_mode):
            data, identity = _read_regular(selected, maximum=32 * 1024 * 1024)
            rows.append((key, "FILE", *identity, hashlib.sha256(data).hexdigest()))
            continue
        if not stat.S_ISDIR(info.st_mode):
            rows.append((key, "SPECIAL", info.st_mode, info.st_rdev, info.st_ctime_ns))
            continue
        rows.append((key, "DIRECTORY", info.st_dev, info.st_ino, info.st_mode, info.st_nlink))
        for current, directories, files in os.walk(key, topdown=True, followlinks=False):
            base = Path(current)
            directories[:] = sorted(directories)
            for name in directories + sorted(files):
                child = base / name
                child_key = str(child)
                child_info = os.lstat(child_key)
                if stat.S_ISREG(child_info.st_mode) and not stat.S_ISLNK(child_info.st_mode):
                    data, identity = _read_regular(child, maximum=32 * 1024 * 1024)
                    rows.append((
                        child_key, "FILE", *identity, hashlib.sha256(data).hexdigest(),
                    ))
                elif stat.S_ISDIR(child_info.st_mode) and not stat.S_ISLNK(child_info.st_mode):
                    rows.append((
                        child_key, "DIRECTORY", child_info.st_dev, child_info.st_ino,
                        child_info.st_mode, child_info.st_nlink,
                    ))
                elif stat.S_ISLNK(child_info.st_mode):
                    rows.append((child_key, "SYMLINK", child_info.st_mode, child_info.st_ctime_ns))
                else:
                    rows.append((
                        child_key, "SPECIAL", child_info.st_mode,
                        child_info.st_rdev, child_info.st_ctime_ns,
                    ))
    return tuple(rows)


def _directory_identities(paths: Sequence[Path]) -> Tuple[Tuple[Any, ...], ...]:
    rows = []
    for path in paths:
        try:
            info = os.lstat(str(path))
        except FileNotFoundError:
            rows.append((str(path), "ABSENT"))
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            rows.append((str(path), "NONPHYSICAL", info.st_mode, info.st_ctime_ns))
        else:
            rows.append((
                str(path), "DIRECTORY", info.st_dev, info.st_ino,
                info.st_mode, info.st_nlink,
            ))
    return tuple(rows)


class DecisionSurfaceComponent:
    """One terminal comparison row; construction is observer-owned."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise DecisionSurfaceContractError("component rows are observer-owned")
        if set(kwargs) != {
            "name", "comparison", "reference_digest", "current_digest", "reason_code",
        }:
            raise DecisionSurfaceContractError("component fields are invalid")
        self.name = kwargs["name"]
        self.comparison = kwargs["comparison"]
        self.reference_digest = MappingProxyType(dict(kwargs["reference_digest"]))
        current = kwargs["current_digest"]
        self.current_digest = None if current is None else MappingProxyType(dict(current))
        self.reason_code = kwargs["reason_code"]
        self._validate()

    def _validate(self) -> None:
        if self.name not in COMPONENT_NAMES or self.comparison not in {
            "EQUAL", "DIFFERENT", "INCOMPARABLE",
        }:
            raise DecisionSurfaceContractError("component classification is invalid")
        _typed_digest(dict(self.reference_digest), "reference", "canonical_json")
        if self.current_digest is not None:
            _typed_digest(dict(self.current_digest), "current", "canonical_json")
        if (
            type(self.reason_code) is not str or not self.reason_code
            or (self.comparison == "INCOMPARABLE") != (self.current_digest is None)
            or (self.comparison == "EQUAL") != (
                self.current_digest is not None
                and dict(self.reference_digest) == dict(self.current_digest)
            )
        ):
            raise DecisionSurfaceContractError("component fields are inconsistent")

    def to_safe_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "name": self.name,
            "comparison": self.comparison,
            "reference_digest": dict(self.reference_digest),
            "current_digest": None if self.current_digest is None else dict(self.current_digest),
            "reason_code": self.reason_code,
        }


def _component(
    name: str, reference: Any, current: Optional[Any], reason: str,
) -> DecisionSurfaceComponent:
    reference_digest = _digest(reference)
    current_digest = None if current is None else _digest(current)
    comparison = "INCOMPARABLE" if current is None else (
        "EQUAL" if reference_digest == current_digest else "DIFFERENT"
    )
    if comparison == "EQUAL":
        reason = "EXACT_EQUAL"
    elif comparison == "DIFFERENT":
        reason = "STABLE_COMPONENT_DIFFERENT"
    return DecisionSurfaceComponent(
        _authority=_AUTHORITY, name=name, comparison=comparison,
        reference_digest=reference_digest, current_digest=current_digest,
        reason_code=reason,
    )


class ProductionDecisionSurfaceResult:
    """Inspector-owned aggregate; safe JSON replay carries no authority."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs.pop("_authority", None) is not _AUTHORITY:
            raise DecisionSurfaceContractError("results are observer-owned")
        if set(kwargs) != {"components", "reference_cycle_id", "current_cycle_id"}:
            raise DecisionSurfaceContractError("result fields are invalid")
        self._components = tuple(kwargs["components"])
        self.reference_cycle_id = kwargs["reference_cycle_id"]
        self.current_cycle_id = kwargs["current_cycle_id"]
        self._validate(False)
        _RESULT_BINDINGS[self] = _canonical(self.to_safe_summary_dict(_binding=False))

    def _validate(self, binding: bool = True) -> None:
        if (
            len(self._components) != len(COMPONENT_NAMES)
            or tuple(item.name for item in self._components) != COMPONENT_NAMES
            or any(type(item) is not DecisionSurfaceComponent for item in self._components)
        ):
            raise DecisionSurfaceContractError("result must contain exact ordered components")
        for item in self._components:
            item._validate()
        _date(self.reference_cycle_id, "reference_cycle_id")
        _date(self.current_cycle_id, "current_cycle_id")
        if date.fromisoformat(self.current_cycle_id) <= date.fromisoformat(self.reference_cycle_id):
            raise DecisionSurfaceContractError("current cycle must be later than reference cycle")
        if binding and _RESULT_BINDINGS.get(self) != _canonical(
            self.to_safe_summary_dict(_binding=False),
        ):
            raise DecisionSurfaceContractError("result authority is absent")

    @property
    def components(self) -> Tuple[DecisionSurfaceComponent, ...]:
        self._validate()
        return self._components

    @property
    def status(self) -> str:
        comparisons = tuple(item.comparison for item in self._components)
        if "INCOMPARABLE" in comparisons:
            return "INCOMPARABLE"
        if "DIFFERENT" in comparisons:
            return "VERSION_BREAK"
        return "SAME_CHAMPION_SEGMENT"

    @property
    def same_champion_segment(self) -> bool:
        self._validate()
        return self.status == "SAME_CHAMPION_SEGMENT"

    @property
    def decision_surface_id(self) -> Optional[str]:
        self._validate()
        if not self.same_champion_segment:
            return None
        payload = {
            "protocol": SURFACE_PROTOCOL,
            "components": [dict(item.reference_digest) for item in self._components],
        }
        return "surface." + hashlib.sha256(_canonical(payload)).hexdigest()

    def to_safe_summary_dict(self, _binding: bool = True) -> Dict[str, Any]:
        if _binding:
            self._validate()
        status = self.status
        reasons = tuple(dict.fromkeys(
            item.reason_code for item in self._components if item.reason_code != "EXACT_EQUAL"
        ))
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_kind": OBSERVATION_KIND,
            "status": status,
            "reason_codes": list(reasons),
            "component_count": len(self._components),
            "components": [item.to_safe_dict() for item in self._components],
            "decision_surface_id": self.decision_surface_id if _binding else (
                "surface." + hashlib.sha256(_canonical({
                    "protocol": SURFACE_PROTOCOL,
                    "components": [dict(item.reference_digest) for item in self._components],
                })).hexdigest() if status == "SAME_CHAMPION_SEGMENT" else None
            ),
            "same_champion_segment": status == "SAME_CHAMPION_SEGMENT",
            "may_continue_to_retention_gates": status == "SAME_CHAMPION_SEGMENT",
            "intent_capability": False,
            "epoch_started": False,
            "prospective_claim": False,
            "promotion_capability": False,
            "did_write": False,
        }


def _make_result(
    components: Sequence[DecisionSurfaceComponent], reference_cycle: str,
    current_cycle: str,
) -> ProductionDecisionSurfaceResult:
    return ProductionDecisionSurfaceResult(
        _authority=_AUTHORITY, components=tuple(components),
        reference_cycle_id=reference_cycle, current_cycle_id=current_cycle,
    )


def _manifest_object(cycle: Path, observation: Mapping[str, Any], name: str) -> Dict[str, Any]:
    if type(observation) is not dict or observation.get("status") != "observed":
        raise _ComponentIncomparable(name.upper() + "_OBSERVATION_INVALID")
    digest = _typed_digest(observation.get("digest"), name, "raw_bytes")
    if observation.get("preservation_status") != "embedded":
        raise _ComponentIncomparable(name.upper() + "_NOT_EMBEDDED")
    path = cycle / "objects" / digest["value"][:2] / digest["value"]
    data, _identity = _read_regular(path, maximum=digest["size_bytes"])
    if _digest(data, "raw_bytes") != digest:
        raise _ComponentIncomparable(name.upper() + "_OBJECT_DIGEST_MISMATCH")
    return _strict_json(data, name)


def _run_manifest(cycle: Path, manifest: Mapping[str, Any], evidence_class: str) -> Dict[str, Any]:
    rows = manifest.get("run_evidence")
    if type(rows) is not list:
        raise _ComponentIncomparable("RUN_EVIDENCE_INVALID")
    matches = [row for row in rows if type(row) is dict and row.get("class") == evidence_class]
    if len(matches) != 1:
        raise _ComponentIncomparable(evidence_class.upper() + "_RUN_EVIDENCE_AMBIGUOUS")
    return _manifest_object(cycle, matches[0], evidence_class + "_manifest")


def _embedded_inventory(value: Any) -> Tuple[Tuple[str, int], ...]:
    found = []

    def visit(item: Any) -> None:
        if type(item) is dict:
            if item.get("preservation_status") == "embedded":
                digest_value = item.get("digest")
                if type(digest_value) is not dict:
                    raise _ComponentIncomparable("EMBEDDED_MEMBER_DIGEST_INVALID")
                domain = digest_value.get("domain")
                if domain not in {"raw_bytes", "semantic_config", "file_inventory"}:
                    raise _ComponentIncomparable("EMBEDDED_MEMBER_DIGEST_INVALID")
                digest = _typed_digest(digest_value, "embedded_member", domain)
                found.append((digest["value"], digest["size_bytes"]))
            for child in item.values():
                visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)

    visit(value)
    sizes = {}
    for digest, size in found:
        sizes.setdefault(digest, set()).add(size)
    if any(len(values) != 1 for values in sizes.values()):
        raise _ComponentIncomparable("EMBEDDED_MEMBER_SIZE_CONFLICT")
    return tuple(sorted((digest, next(iter(sizes[digest]))) for digest in sizes))


def _expected_named_digests(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    ranking = manifest.get("ranking")
    if type(ranking) is dict and ranking.get("ranking_digest") is not None:
        result["ranking.csv"] = _typed_digest(
            ranking["ranking_digest"], "ranking", "raw_bytes",
        )
    portfolio = manifest.get("portfolio_state")
    if type(portfolio) is dict and portfolio.get("canonical_digest") is not None:
        semantic = _typed_digest(
            portfolio["canonical_digest"], "portfolio", "semantic_config",
        )
        result["portfolio_state.json"] = {
            **semantic, "domain": "raw_bytes",
        }
    return result


def _referenced_bytes(
    cycle: Path, manifest: Mapping[str, Any], evidence_class: str, locator: str,
) -> bytes:
    rows = manifest.get("referenced_evidence")
    if type(rows) is not list:
        raise _ComponentIncomparable("REFERENCED_EVIDENCE_INVALID")
    matches = [
        row for row in rows
        if type(row) is dict and row.get("evidence_class") == evidence_class
        and row.get("collection") == "inputs" and row.get("locator") == locator
        and row.get("locator_type") == "workspace_file"
    ]
    if len(matches) != 1 or type(matches[0].get("observation")) is not dict:
        raise _ComponentIncomparable("SEALED_CONFIG_REFERENCE_AMBIGUOUS")
    observation = matches[0]["observation"]
    digest = _typed_digest(observation.get("digest"), "sealed_config", "raw_bytes")
    if observation.get("status") != "observed" or observation.get("preservation_status") != "embedded":
        raise _ComponentIncomparable("SEALED_CONFIG_NOT_EMBEDDED")
    data, _ = _read_regular(
        cycle / "objects" / digest["value"][:2] / digest["value"],
        maximum=digest["size_bytes"], private=True,
    )
    if _digest(data, "raw_bytes") != digest:
        raise _ComponentIncomparable("SEALED_CONFIG_DIGEST_MISMATCH")
    return data


def _cycle_authority(root: Path, cycle_id: str) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    cycle = root / "data" / "evidence" / "v1" / "cycles" / cycle_id
    try:
        cycle_info = os.lstat(str(cycle))
    except OSError as exc:
        raise _ComponentIncomparable("CYCLE_DIRECTORY_UNAVAILABLE") from exc
    if (
        not stat.S_ISDIR(cycle_info.st_mode) or stat.S_ISLNK(cycle_info.st_mode)
        or stat.S_IMODE(cycle_info.st_mode) != 0o700
    ):
        raise _ComponentIncomparable("CYCLE_DIRECTORY_INVALID")
    manifest_data, _ = _read_regular(cycle / "manifest.json", private=True)
    seal_data, _ = _read_regular(cycle / "seal.json", private=True)
    manifest = _strict_json(manifest_data, "cycle_manifest")
    seal = _strict_json(seal_data, "cycle_seal")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise _ComponentIncomparable("CYCLE_MANIFEST_SCHEMA_INVALID")
    if type(seal.get("schema_version")) is not int or seal["schema_version"] != 1:
        raise _ComponentIncomparable("CYCLE_SEAL_SCHEMA_INVALID")
    if (
        seal.get("cycle_id") != cycle_id
        or manifest.get("cycle_identity", {}).get("cycle_id") != cycle_id
        or seal.get("status") != manifest.get("status")
        or manifest.get("status") not in {"sealed_complete", "sealed_partial"}
        or _typed_digest(seal.get("manifest_digest"), "manifest", "raw_bytes")
        != _digest(manifest_data, "raw_bytes")
    ):
        raise _ComponentIncomparable("CYCLE_SEAL_JOIN_INVALID")
    problems = manifest.get("problems")
    if type(problems) is not list:
        raise _ComponentIncomparable("CYCLE_PROBLEM_INVENTORY_INVALID")
    if manifest["status"] == "sealed_partial":
        identities = set()
        for row in problems:
            if (
                type(row) is not dict
                or set(row) != {"code", "evidence_class", "detail", "blocks_complete"}
                or type(row.get("code")) is not str
                or type(row.get("evidence_class")) is not str
                or type(row.get("detail")) is not str
                or type(row.get("blocks_complete")) is not bool
            ):
                raise _ComponentIncomparable("CYCLE_PROBLEM_INVENTORY_INVALID")
            identity = (row["code"], row["evidence_class"], row["blocks_complete"])
            if identity in identities:
                raise _ComponentIncomparable("CYCLE_PROBLEM_INVENTORY_DUPLICATE")
            identities.add(identity)
        blocking = {item for item in identities if item[2]}
        if not blocking.issubset(_ALLOWED_PARTIAL):
            raise _ComponentIncomparable("CYCLE_BLOCKING_PROBLEM_NOT_SCOPED")
    replay_core = {
        key: value for key, value in manifest.items()
        if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
    }
    if _typed_digest(
        manifest.get("request_content_digest"), "request_content", "canonical_json",
    ) != _digest(replay_core):
        raise _ComponentIncomparable("CYCLE_REQUEST_CONTENT_DIGEST_MISMATCH")
    named = seal.get("named_file_digests")
    objects = seal.get("object_digests")
    if type(named) is not dict or type(objects) is not list or objects != sorted(set(objects)):
        raise _ComponentIncomparable("CYCLE_SEAL_INVENTORY_INVALID")
    embedded = _embedded_inventory(manifest)
    if objects != [digest for digest, _size in embedded]:
        raise _ComponentIncomparable("CYCLE_EMBEDDED_OBJECT_INVENTORY_MISMATCH")
    if named != _expected_named_digests(manifest):
        raise _ComponentIncomparable("CYCLE_NAMED_MANIFEST_JOIN_INVALID")
    expected_top = {"manifest.json", "seal.json", *named}
    if objects:
        expected_top.add("objects")
    if {item.name for item in cycle.iterdir()} != expected_top:
        raise _ComponentIncomparable("CYCLE_TOP_INVENTORY_INVALID")
    for name, raw_digest in named.items():
        if type(name) is not str or Path(name).name != name:
            raise _ComponentIncomparable("CYCLE_NAMED_MEMBER_INVALID")
        expected = _typed_digest(raw_digest, "named_member", "raw_bytes")
        data, _ = _read_regular(
            cycle / name, maximum=expected["size_bytes"], private=True,
        )
        if _digest(data, "raw_bytes") != expected:
            raise _ComponentIncomparable("CYCLE_NAMED_MEMBER_DIGEST_MISMATCH")
    for value in objects:
        if type(value) is not str or _HEX_RE.fullmatch(value) is None:
            raise _ComponentIncomparable("CYCLE_OBJECT_ID_INVALID")
        expected_size = dict(embedded)[value]
        data, _ = _read_regular(
            cycle / "objects" / value[:2] / value,
            maximum=expected_size, private=True,
        )
        if len(data) != expected_size or hashlib.sha256(data).hexdigest() != value:
            raise _ComponentIncomparable("CYCLE_OBJECT_DIGEST_MISMATCH")
    if objects:
        object_root = cycle / "objects"
        object_info = os.lstat(str(object_root))
        if (
            not stat.S_ISDIR(object_info.st_mode) or stat.S_ISLNK(object_info.st_mode)
            or stat.S_IMODE(object_info.st_mode) != 0o700
        ):
            raise _ComponentIncomparable("CYCLE_OBJECT_ROOT_INVALID")
        prefixes = sorted(set(value[:2] for value in objects))
        if tuple(sorted(item.name for item in object_root.iterdir())) != tuple(prefixes):
            raise _ComponentIncomparable("CYCLE_OBJECT_PREFIX_INVENTORY_INVALID")
        for prefix in prefixes:
            prefix_info = os.lstat(str(object_root / prefix))
            if (
                not stat.S_ISDIR(prefix_info.st_mode)
                or stat.S_ISLNK(prefix_info.st_mode)
                or stat.S_IMODE(prefix_info.st_mode) != 0o700
            ):
                raise _ComponentIncomparable("CYCLE_OBJECT_PREFIX_INVALID")
            expected = tuple(value for value in objects if value[:2] == prefix)
            observed = tuple(sorted(item.name for item in (object_root / prefix).iterdir()))
            if observed != expected:
                raise _ComponentIncomparable("CYCLE_OBJECT_MEMBER_INVENTORY_INVALID")
    artifact = _digest({"objects": objects, "named_files": named})
    if _typed_digest(seal.get("artifact_root_digest"), "artifact_root") != artifact:
        raise _ComponentIncomparable("CYCLE_ARTIFACT_ROOT_DIGEST_MISMATCH")
    return cycle, manifest, seal


def _bootstrap_authority(
    root: Path, store: Path, identifier: str, definition: Any, evidence: Any,
) -> Tuple[str, Dict[str, Any]]:
    target = store / identifier
    try:
        target_info = os.lstat(str(target))
    except OSError as exc:
        raise _ComponentIncomparable("BOOTSTRAP_TARGET_UNAVAILABLE") from exc
    if (
        target.parent != store or not stat.S_ISDIR(target_info.st_mode)
        or stat.S_ISLNK(target_info.st_mode)
        or stat.S_IMODE(target_info.st_mode) != 0o700
    ):
        raise _ComponentIncomparable("BOOTSTRAP_TARGET_INVALID")
    if tuple(sorted(item.name for item in target.iterdir())) != (
        "bootstrap_manifest.json", "challenger_state.json", "champion_state.json",
        "source_receipt.json",
    ):
        raise _ComponentIncomparable("BOOTSTRAP_INVENTORY_INVALID")
    manifest_data, _ = _read_regular(target / "bootstrap_manifest.json", private=True)
    manifest = _strict_json(manifest_data, "bootstrap_manifest")
    source_data, _ = _read_regular(target / "source_receipt.json", private=True)
    source = _strict_json(source_data, "bootstrap_source_receipt")
    manifest_fields = {
        "schema_version", "bootstrap_kind", "storage_claim", "bootstrap_set_id",
        "definition_set_id", "definition_evidence_cycle_id",
        "bootstrap_source_cycle_id", "definition_request_digest",
        "definition_evidence_request_digest", "definition_evidence_manifest_digest",
        "definition_evidence_operation_id", "phase37a_seal_digest",
        "phase37a_manifest_digest", "source_portfolio_raw_digest",
        "source_portfolio_semantic_digest", "source_state_interpretation",
        "economic_state_digest", "roles", "member_count", "members",
        "matched_economics_checked", "portfolio_bootstrap_claim", "intent_claim",
        "epoch_started", "prospective_claim", "promotion_capability",
    }
    source_fields = {
        "schema_version", "source_claim", "definition_set_id",
        "definition_evidence_cycle_id", "bootstrap_source_cycle_id",
        "definition_evidence_status", "definition_evidence_request_digest",
        "definition_evidence_manifest_digest", "definition_evidence_operation_id",
        "source_observation_status", "source_cycle_status", "source_problem_inventory",
        "source_problem_inventory_digest", "phase37a_seal_digest",
        "phase37a_manifest_digest", "source_portfolio_raw_digest",
        "source_portfolio_semantic_digest", "source_portfolio_holding_count",
        "portfolio_member_verified", "chronology_checked",
        "cash_sign_and_deficit_digest", "did_write", "prospective_claim",
        "promotion_capability",
    }
    if set(manifest) != manifest_fields or set(source) != source_fields:
        raise _ComponentIncomparable("BOOTSTRAP_SCHEMA_INVALID")
    if (
        type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
        or manifest.get("bootstrap_kind") != "MATCHED_FORWARD_BOOTSTRAP_V1"
        or manifest.get("storage_claim") != "CREATE_ONLY_EXACT_BYTES"
        or type(manifest.get("member_count")) is not int or manifest["member_count"] != 3
        or manifest.get("matched_economics_checked") is not True
        or manifest.get("portfolio_bootstrap_claim") is not True
        or any(manifest.get(field) is not False for field in (
            "intent_claim", "epoch_started", "prospective_claim", "promotion_capability",
        ))
        or type(source.get("schema_version")) is not int or source["schema_version"] != 1
        or source.get("source_claim") != "C2_MATCHED_BOOTSTRAP_SOURCE_VERIFIED_V1"
        or source.get("chronology_checked") is not True
        or source.get("portfolio_member_verified") is not True
        or source.get("did_write") is not False
        or source.get("prospective_claim") is not False
        or source.get("promotion_capability") is not False
    ):
        raise _ComponentIncomparable("BOOTSTRAP_FIXED_CLAIMS_INVALID")
    members = manifest.get("members")
    if type(members) is not list or len(members) != 3 or manifest.get("member_count") != 3:
        raise _ComponentIncomparable("BOOTSTRAP_MEMBER_SET_INVALID")
    expected_names = ("source_receipt.json", "champion_state.json", "challenger_state.json")
    if tuple(row.get("logical_path") for row in members if type(row) is dict) != expected_names:
        raise _ComponentIncomparable("BOOTSTRAP_MEMBER_ORDER_INVALID")
    for row in members:
        expected = _typed_digest(row.get("digest"), "bootstrap_member", "raw_bytes")
        data, _ = _read_regular(
            target / row["logical_path"], maximum=expected["size_bytes"], private=True,
        )
        if row.get("size_bytes") != len(data) or _digest(data, "raw_bytes") != expected:
            raise _ComponentIncomparable("BOOTSTRAP_MEMBER_DIGEST_MISMATCH")
    definition_request = dict(definition.to_store_request().request_digest)
    if (
        manifest.get("bootstrap_set_id") != identifier
        or manifest.get("definition_set_id") != definition.definition_set_id
        or manifest.get("definition_request_digest") != definition_request
        or manifest.get("definition_evidence_request_digest")
        != dict(evidence.evidence_receipt.request_digest)
        or manifest.get("definition_evidence_manifest_digest")
        != dict(evidence.evidence_receipt.manifest_digest)
        or source.get("definition_set_id") != definition.definition_set_id
        or source.get("definition_evidence_cycle_id") != evidence.evidence_cycle_id
        or source.get("phase37a_seal_digest") != manifest.get("phase37a_seal_digest")
        or source.get("phase37a_manifest_digest") != manifest.get("phase37a_manifest_digest")
        or source.get("portfolio_member_verified") is not True
        or source.get("did_write") is not False
    ):
        raise _ComponentIncomparable("BOOTSTRAP_PROVENANCE_JOIN_INVALID")
    try:
        source_cycle = _date(
            source.get("bootstrap_source_cycle_id"), "bootstrap_source_cycle_id",
        )
    except DecisionSurfaceContractError as exc:
        raise _ComponentIncomparable("BOOTSTRAP_SOURCE_CYCLE_INVALID") from exc
    identity = {
        "domain": "MATCHED_FORWARD_BOOTSTRAP_SET_V1",
        "definition_set_id": manifest["definition_set_id"],
        "definition_evidence_cycle_id": manifest["definition_evidence_cycle_id"],
        "bootstrap_source_cycle_id": source_cycle,
        "definition_request_digest": manifest["definition_request_digest"],
        "definition_evidence_request_digest": manifest["definition_evidence_request_digest"],
        "definition_evidence_manifest_digest": manifest["definition_evidence_manifest_digest"],
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
        "roles": manifest["roles"],
        "economic_state_digest": manifest["economic_state_digest"],
    }
    rebuilt_id = "bootstrap." + hashlib.sha256(_canonical(identity)).hexdigest()
    if rebuilt_id != identifier:
        raise _ComponentIncomparable("BOOTSTRAP_SET_ID_PROVENANCE_MISMATCH")
    return source_cycle, source


def _git_blob_projection(engine: Path, commit: Any) -> Dict[str, Any]:
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise _ComponentIncomparable("ENGINE_COMMIT_INVALID")
    rows = []
    for logical in CURATED_CODE_PATHS:
        try:
            completed = subprocess.run(
                ["git", "--no-replace-objects", "cat-file", "blob", "%s:%s" % (commit, logical)],
                cwd=str(engine), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                check=False, timeout=10,
            )
        except _PROCESS_CONTROL:
            raise
        except Exception as exc:
            raise _ComponentIncomparable("ENGINE_GIT_READ_FAILED") from exc
        if completed.returncode != 0 or len(completed.stdout) > _MAX_GIT_BLOB:
            raise _ComponentIncomparable("ENGINE_CURATED_BLOB_INCOMPARABLE")
        rows.append({"path": logical, "digest": _digest(completed.stdout, "raw_bytes")})
    return {"protocol": "CURATED_ECONOMIC_CODE_SURFACE_V1", "members": rows}


def _engine_commit(manifest: Mapping[str, Any]) -> str:
    engine = manifest.get("engine_identity")
    if type(engine) is not dict:
        raise _ComponentIncomparable("ENGINE_IDENTITY_INVALID")
    dirty = engine.get("status_inventory", [])
    if type(dirty) is not list or any(type(item) is not str for item in dirty):
        raise _ComponentIncomparable("ENGINE_DIRTY_INVENTORY_INVALID")
    if set(dirty).intersection(CURATED_CODE_PATHS):
        raise _ComponentIncomparable("ENGINE_CURATED_SURFACE_DIRTY")
    commit = engine.get("commit")
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise _ComponentIncomparable("ENGINE_COMMIT_INVALID")
    return commit


def _artifact_inventory(artifact: Mapping[str, Any]) -> Tuple[list, Dict[str, Any]]:
    members_raw = artifact.get("members")
    if type(members_raw) is not list or not members_raw:
        raise _ComponentIncomparable("SOURCE_ARTIFACT_INVENTORY_INVALID")
    inventory = []
    paths = set()
    for member in members_raw:
        if type(member) is not dict or set(member) != {
            "path", "digest", "status", "preservation_status", "detail",
        }:
            raise _ComponentIncomparable("SOURCE_ARTIFACT_MEMBER_INVALID")
        path = member.get("path")
        if (
            type(path) is not str or not path or path in paths
            or "\\" in path or "\0" in path or Path(path).is_absolute()
            or Path(path).as_posix() != path
            or any(part in {"", ".", ".."} for part in Path(path).parts)
            or member.get("status") != "observed"
            or member.get("preservation_status") not in {"embedded", "workspace_file"}
            or type(member.get("detail")) is not str
        ):
            raise _ComponentIncomparable("SOURCE_ARTIFACT_MEMBER_INCOMPARABLE")
        paths.add(path)
        inventory.append({
            "path": path,
            "digest": _typed_digest(member.get("digest"), "artifact_member", "raw_bytes"),
        })
    tree = _typed_digest(
        artifact.get("artifact_tree_digest"), "artifact_tree", "file_inventory",
    )
    if tree != _digest(inventory, "file_inventory"):
        raise _ComponentIncomparable("SOURCE_ARTIFACT_TREE_DIGEST_MISMATCH")
    return inventory, tree


def _source_projection(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    lineage = manifest.get("model_and_ensemble_lineage")
    if type(lineage) is not dict:
        raise _ComponentIncomparable("MODEL_LINEAGE_INVALID")
    combo = lineage.get("combo")
    models = lineage.get("source_models")
    artifacts = lineage.get("source_artifacts")
    if type(combo) is not dict or type(models) is not list or type(artifacts) is not list:
        raise _ComponentIncomparable("MODEL_LINEAGE_INVALID")
    members = combo.get("resolved_members")
    if (
        type(members) is not list or len(members) != 4
        or any(type(item) is not str or not item for item in members)
        or len(set(members)) != 4
    ):
        raise _ComponentIncomparable("CHAMPION_SOURCE_SET_INVALID")
    by_key = {}
    for row in models:
        if (
            type(row) is not dict or type(row.get("resolved_key")) is not str
            or not row["resolved_key"] or row.get("status") != "ready"
            or type(row.get("recorder_id")) is not str or not row["recorder_id"]
            or type(row.get("source_recorder_id")) is not str
            or not row["source_recorder_id"]
        ):
            raise _ComponentIncomparable("SOURCE_MODEL_ROW_INVALID")
        if row["resolved_key"] in by_key:
            raise _ComponentIncomparable("SOURCE_MODEL_DUPLICATE")
        by_key[row["resolved_key"]] = row
    training_by_position = {}
    prediction_by_position = {}
    ensemble_rows = []
    inventories = {}
    for artifact in artifacts:
        if type(artifact) is not dict:
            raise _ComponentIncomparable("SOURCE_ARTIFACT_REMAINDER_INVALID")
        position = artifact.get("position")
        if artifact.get("role") == "source_training" and type(position) is int and position in range(4):
            expected = {
                "position", "role", "recorder_id", "experiment_name",
                "artifact_locator", "members", "artifact_tree_digest",
            }
            target = training_by_position
        elif "role" not in artifact and type(position) is int and position in range(4):
            expected = {
                "position", "recorder_id", "source_recorder_id",
                "artifact_locator", "members", "artifact_tree_digest",
            }
            target = prediction_by_position
        elif "role" not in artifact and position == "ensemble":
            expected = {
                "position", "recorder_id", "artifact_locator", "members",
                "artifact_tree_digest",
            }
            target = None
        else:
            raise _ComponentIncomparable("SOURCE_ARTIFACT_REMAINDER_UNASSIGNED")
        if set(artifact) != expected:
            raise _ComponentIncomparable("SOURCE_ARTIFACT_ROW_SCHEMA_INVALID")
        if (
            type(artifact.get("recorder_id")) is not str or not artifact["recorder_id"]
            or type(artifact.get("artifact_locator")) is not str
            or not artifact["artifact_locator"]
        ):
            raise _ComponentIncomparable("SOURCE_ARTIFACT_IDENTITY_INVALID")
        inventory, _tree = _artifact_inventory(artifact)
        inventories[id(artifact)] = inventory
        if target is None:
            ensemble_rows.append(artifact)
        else:
            if position in target:
                raise _ComponentIncomparable("SOURCE_ARTIFACT_RELATION_DUPLICATE")
            target[position] = artifact
    if set(training_by_position) != set(range(4)):
        raise _ComponentIncomparable("SOURCE_TRAINING_PARTITION_INVALID")
    if set(prediction_by_position) != set(range(4)) or len(ensemble_rows) != 1:
        raise _ComponentIncomparable("PREDICTION_ARTIFACT_PARTITION_INVALID")
    combo_recorder = combo.get("ensemble_recorder_id")
    if type(combo_recorder) is not str or ensemble_rows[0].get("recorder_id") != combo_recorder:
        raise _ComponentIncomparable("ENSEMBLE_ARTIFACT_RELATION_INVALID")
    rows = []
    for position, key in enumerate(members):
        model = by_key.get(key)
        artifact = training_by_position[position]
        prediction = prediction_by_position[position]
        if (
            type(model) is not dict or model.get("source_recorder_id") != artifact.get("recorder_id")
            or artifact.get("position") != position
            or prediction.get("recorder_id") != model.get("recorder_id")
            or prediction.get("source_recorder_id") != model.get("source_recorder_id")
            or model.get("operation") not in {"predict_only", "cpcv_predict"}
        ):
            raise _ComponentIncomparable("PREDICTION_PARENT_JOIN_INVALID")
        if "@" in key:
            family, mode = key.rsplit("@", 1)
            if not family or not mode:
                raise _ComponentIncomparable("SOURCE_MODEL_KEY_INVALID")
        else:
            family, mode = key, "LEGACY_UNQUALIFIED"
        inventory = inventories[id(artifact)]
        inventory_bytes = _canonical(inventory)
        raw_inventory = _digest(inventory_bytes, "raw_bytes")
        rows.append({
            "position": position,
            "resolved_key": key,
            "source_id": key,
            "source_parent_id": artifact.get("recorder_id"),
            "artifact_inventory_digest": raw_inventory,
            "family": family,
            "mode": mode,
            "training_cutoff_identity": raw_inventory,
        })
    return {"protocol": "CHAMPION_SOURCE_TRAINING_SET_V1", "members": rows}


def _definition_source_projection(definition: Any) -> Dict[str, Any]:
    champion = definition.compiled_definitions.champion.to_dict()
    rows = []
    for item in champion["source_members"]:
        rows.append({
            "position": item["position"], "resolved_key": item["source_id"],
            "source_id": item["source_id"],
            "source_parent_id": None,
            "artifact_inventory_digest": item["model_artifact_digest"],
            "family": item["source_id"].rsplit("@", 1)[0],
            "mode": (
                item["source_id"].rsplit("@", 1)[1]
                if "@" in item["source_id"] else "LEGACY_UNQUALIFIED"
            ),
            "training_cutoff_identity": item["model_artifact_digest"],
        })
    return {"protocol": "CHAMPION_SOURCE_TRAINING_SET_V1", "members": rows}


def _source_matches_definition(observed: Mapping[str, Any], definition: Any) -> bool:
    expected = _definition_source_projection(definition)["members"]
    actual = observed.get("members")
    if type(actual) is not list or len(actual) != len(expected):
        return False
    return all(
        row["position"] == wanted["position"]
        and row["source_id"] == wanted["source_id"]
        and row["artifact_inventory_digest"] == wanted["artifact_inventory_digest"]
        and row["family"] == wanted["family"]
        and row["mode"] == wanted["mode"]
        and row["training_cutoff_identity"] == wanted["training_cutoff_identity"]
        for row, wanted in zip(actual, expected)
    )


def _ensemble_projection(cycle: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    lineage = manifest["model_and_ensemble_lineage"]
    combo = lineage.get("combo")
    raw = _run_manifest(cycle, manifest, "ensemble")
    records = raw.get("records")
    combos = records.get("combos") if type(records) is dict else None
    if type(combo) is not dict or type(combos) is not list or not combos:
        raise _ComponentIncomparable("ENSEMBLE_SELECTION_INVALID")
    defaults = [row for row in combos if type(row) is dict and row.get("is_default") is True]
    selected = defaults if defaults else [row for row in combos if type(row) is dict]
    if len(selected) != 1:
        raise _ComponentIncomparable("ENSEMBLE_SELECTION_AMBIGUOUS")
    row = selected[0]
    resolved = row.get("resolved_models", row.get("models"))
    if (
        row.get("method") != "equal" or combo.get("method") != "equal"
        or resolved != combo.get("resolved_members")
        or type(resolved) is not list or len(resolved) != 4
    ):
        raise _ComponentIncomparable("ENSEMBLE_ACTUAL_SELECTION_JOIN_INVALID")
    args = raw.get("args")
    if type(args) is not list or any(type(item) is not str for item in args):
        raise _ComponentIncomparable("ENSEMBLE_COMMAND_ARGUMENTS_INVALID")
    normalization = "rank"
    if "--norm-method" in args:
        index = args.index("--norm-method")
        if index + 1 >= len(args):
            raise _ComponentIncomparable("ENSEMBLE_NORMALIZATION_INVALID")
        normalization = args[index + 1]
    if normalization not in {"rank", "percentile_rank"}:
        raise _ComponentIncomparable("ENSEMBLE_NORMALIZATION_INVALID")
    selected_name = row.get("name")
    if type(selected_name) is not str or not selected_name:
        raise _ComponentIncomparable("ENSEMBLE_SELECTED_NAME_INVALID")
    if "--combo" in args:
        index = args.index("--combo")
        if index + 1 >= len(args) or args[index + 1] != selected_name:
            raise _ComponentIncomparable("ENSEMBLE_SELECTOR_JOIN_INVALID")
        selection_mode = "EXPLICIT_COMBO"
    elif "--from-config-all" in args:
        if not defaults:
            raise _ComponentIncomparable("ENSEMBLE_DEFAULT_RELATION_INVALID")
        selection_mode = "DEFAULT_FROM_ALL"
    elif "--from-config" in args:
        if not defaults:
            raise _ComponentIncomparable("ENSEMBLE_DEFAULT_RELATION_INVALID")
        selection_mode = "DEFAULT"
    elif "--models" in args:
        index = args.index("--models")
        if index + 1 >= len(args):
            raise _ComponentIncomparable("ENSEMBLE_SELECTOR_JOIN_INVALID")
        requested = [item.strip() for item in args[index + 1].split(",") if item.strip()]
        if requested != resolved:
            raise _ComponentIncomparable("ENSEMBLE_SELECTOR_JOIN_INVALID")
        selection_mode = "EXPLICIT_MODELS"
    else:
        raise _ComponentIncomparable("ENSEMBLE_SELECTOR_UNPROVEN")
    return {
        "protocol": "PER_MODEL_CROSS_SECTIONAL_PERCENTILE_RANK_EQUAL_MEAN_V1",
        "method": "equal", "normalization": "rank",
        "members": resolved, "selection_mode": selection_mode,
        "selected_name": selected_name,
    }


def _prediction_projection(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    source = _source_projection(manifest)
    rows = []
    for row in source["members"]:
        rows.append({
            "position": row["position"], "resolved_key": row["resolved_key"],
            "source_parent_id": row["source_parent_id"],
            "source_artifact_inventory_digest": row["artifact_inventory_digest"],
            "dynamic_child_id_compared": False,
        })
    return {"protocol": "EXACT_FROZEN_TRAINING_PARENT_JOIN_V1", "members": rows}


def _market_projection(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    data = manifest.get("data_identity")
    if type(data) is not dict:
        raise _ComponentIncomparable("MARKET_POLICY_INVALID")
    material = data.get("qlib_materialization_identity")
    if type(material) is not dict or material.get("status") != "observed":
        raise _ComponentIncomparable("MARKET_POLICY_INCOMPARABLE")
    name = material.get("universe_name")
    if type(name) is not str or not name:
        raise _ComponentIncomparable("UNIVERSE_POLICY_INVALID")
    return {
        "protocol": "QLIB_MARKET_MEMBERSHIP_AT_ANCHOR_V1",
        "market": name.lower(), "instrument_selector": "qlib_market",
        "membership_policy": "ANCHOR_DATE_INCLUSIVE_INTERVAL_V1",
    }


def _intent_projection(cycle: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _run_manifest(cycle, manifest, "order")
    records = raw.get("records")
    source = records.get("source") if type(records) is dict else None
    if type(source) is not dict or source.get("mode") != "ensemble":
        raise _ComponentIncomparable("ORDER_SOURCE_NOT_EXACT_ENSEMBLE")
    resolved_combo = source.get("resolved_name")
    requested_combo = source.get("requested_name")
    if (
        type(resolved_combo) is not str or not resolved_combo
        or requested_combo is not None and (
            type(requested_combo) is not str or requested_combo != resolved_combo
        )
    ):
        raise _ComponentIncomparable("ORDER_ENSEMBLE_SOURCE_FALLBACK")
    ensemble_records = _json_object_bytes(
        _referenced_bytes(
            cycle, manifest, "order", "config/ensemble_records.json",
        ),
        "ensemble_records",
    )
    recorded_combos = ensemble_records.get("combos")
    if (
        type(recorded_combos) is not dict or resolved_combo not in recorded_combos
        or recorded_combos[resolved_combo] != source.get("record_id")
        or (
            requested_combo is None
            and ensemble_records.get("default_combo") != resolved_combo
        )
    ):
        raise _ComponentIncomparable("ORDER_ENSEMBLE_SOURCE_FALLBACK")
    config_data = _referenced_bytes(
        cycle, manifest, "order", "config/strategy_config.yaml",
    )
    try:
        import yaml
        config = yaml.safe_load(config_data.decode("utf-8"))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _ComponentIncomparable("STRATEGY_CONFIG_INVALID") from exc
    if type(config) is not dict or type(config.get("strategy")) is not dict:
        raise _ComponentIncomparable("STRATEGY_CONFIG_INVALID")
    strategy = config["strategy"]
    params = strategy.get("params")
    if type(params) is not dict:
        raise _ComponentIncomparable("STRATEGY_CONFIG_INVALID")
    required = ("topk", "n_drop", "buy_suggestion_factor", "sell_out_of_universe")
    if any(field not in params for field in required):
        raise _ComponentIncomparable("STRATEGY_CONFIG_FALLBACK_REQUIRED")
    if (
        type(strategy.get("name")) is not str
        or type(params["topk"]) is not int or type(params["n_drop"]) is not int
        or type(params["buy_suggestion_factor"]) is not int
        or type(params["sell_out_of_universe"]) is not bool
    ):
        raise _ComponentIncomparable("STRATEGY_CONFIG_PROJECTION_INVALID")
    fingerprints = raw.get("config_fingerprints")
    expected_fingerprint = fingerprints.get("strategy_config") if type(fingerprints) is dict else None
    try:
        from quantpits.config_contracts.normalizers import normalize_strategy_config
        from quantpits.utils.workspace import fingerprint_value
        actual_fingerprint = fingerprint_value(normalize_strategy_config(config))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _ComponentIncomparable("STRATEGY_CONFIG_NORMALIZATION_INVALID") from exc
    if type(expected_fingerprint) is not str or expected_fingerprint != actual_fingerprint:
        raise _ComponentIncomparable("STRATEGY_CONFIG_FINGERPRINT_MISMATCH")
    return {
        "schema_version": 1,
        "strategy_name": strategy["name"],
        "topk": params["topk"], "n_drop": params["n_drop"],
        "buy_suggestion_factor": params["buy_suggestion_factor"],
        "sell_out_of_universe": params["sell_out_of_universe"],
        "production_buy_lot_size": 100,
        "cashflow_mode": "ZERO_EXTERNAL_CASHFLOW",
        "planning_price_rule": "CASH_CLOSE_WITH_0_9_1_1_ESTIMATES_V1",
        "buy_selection_rule": "FIRST_N_GENERATED_SUGGESTIONS_V1",
        "production_primitive_id": "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1",
        "ensemble_source_mode": "EXACT_SELECTED_ENSEMBLE_V1",
        "resolved_combo": resolved_combo,
        "requested_combo_relation": "EXACT" if requested_combo is not None else "DEFAULT",
    }


def _definition_intent_projection(definition: Any) -> Dict[str, Any]:
    value = definition.compiled_definitions.protocol.to_dict()["intent_definition"]
    return {key: value[key] for key in (
        "schema_version", "strategy_name", "topk", "n_drop",
        "buy_suggestion_factor", "sell_out_of_universe",
        "production_buy_lot_size", "cashflow_mode", "planning_price_rule",
        "buy_selection_rule", "production_primitive_id",
    )}


def _intent_matches_definition(observed: Mapping[str, Any], definition: Any) -> bool:
    expected = _definition_intent_projection(definition)
    return all(observed.get(key) == value for key, value in expected.items())


def _observe_component(name: str, reference: Any, factory: Any) -> DecisionSurfaceComponent:
    try:
        current = factory()
        return _component(name, reference, current, "EXACT_EQUAL")
    except _PROCESS_CONTROL:
        raise
    except _ComponentIncomparable as exc:
        return _component(name, reference, None, exc.reason_code)
    except Exception:
        return _component(name, reference, None, name.upper() + "_OBSERVATION_FAILED")


def _observe_production_decision_surface(
    research_workspace_root: Any,
    production_workspace_root: Any,
    engine_root: Any,
    current_cycle_id: Any,
    activation_path: Any,
    definition_store_root: Any,
    evidence_store_root: Any,
    bootstrap_store_root: Any,
    bootstrap_set_id: Any,
) -> ProductionDecisionSurfaceResult:
    """Freshly classify one current Production cycle, strictly without writes."""
    current_cycle = _date(current_cycle_id, "current_cycle_id")
    identifier = _identifier(bootstrap_set_id, "bootstrap_set_id")
    research = _physical_path(research_workspace_root, "research_workspace_root", directory=True)
    production = _physical_path(production_workspace_root, "production_workspace_root", directory=True)
    engine = _physical_path(engine_root, "engine_root", directory=True)
    activation = _physical_path(activation_path, "activation_path", directory=False)
    definitions = _physical_path(definition_store_root, "definition_store_root", directory=True)
    evidence_root = _physical_path(evidence_store_root, "evidence_store_root", directory=True)
    bootstraps = _physical_path(bootstrap_store_root, "bootstrap_store_root", directory=True)
    shadow = research / "research" / "shadow_v1"
    if (
        activation.parent != shadow / "activations"
        or definitions != shadow / "definitions"
        or evidence_root != shadow / "definition_evidence"
        or bootstraps != shadow / "bootstraps"
    ):
        raise DecisionSurfaceInputError("formal Research paths do not match the private layout")
    for path in (definitions, evidence_root, bootstraps):
        if stat.S_IMODE(os.lstat(str(path)).st_mode) != 0o700:
            raise DecisionSurfaceInputError("formal Research stores must use mode 0700")
    if stat.S_IMODE(os.lstat(str(activation)).st_mode) != 0o600:
        raise DecisionSurfaceInputError("formal activation must use mode 0600")
    git_control = engine / ".git"
    try:
        git_info = os.lstat(str(git_control))
    except OSError as exc:
        raise DecisionSurfaceInputError("engine Git control directory is unavailable") from exc
    if stat.S_ISLNK(git_info.st_mode) or not stat.S_ISDIR(git_info.st_mode):
        raise DecisionSurfaceInputError("engine root must be an explicit physical Git repository")
    try:
        activation_raw = _strict_json(_read_regular(activation)[0], "activation")
        evidence_cycle = _date(activation_raw.get("evidence_cycle_id"), "evidence_cycle_id")
    except _ComponentIncomparable as exc:
        raise DecisionSurfaceInputError("formal activation is incomparable") from exc
    watched_research = (
        activation.relative_to(research).as_posix(),
        definitions.relative_to(research).as_posix(),
        evidence_root.relative_to(research).as_posix(),
        bootstraps.relative_to(research).as_posix(),
        (research / "data" / "evidence" / "v1" / "cycles").relative_to(research).as_posix(),
    )
    current_relative = "data/evidence/v1/cycles/%s" % current_cycle
    research_guard = SourceMutationObserver(research, watched_research)
    production_guard = SourceMutationObserver(production, (current_relative,))
    engine_guard = SourceMutationObserver(engine, CURATED_CODE_PATHS)
    git_control_paths = tuple(
        engine / ".git" / name for name in ("HEAD", "index", "packed-refs", "refs")
    )
    git_guards = tuple(
        SourceMutationObserver(path, ())
        for path in (engine / ".git", engine / ".git" / "refs")
        if path.is_dir() and not path.is_symlink()
    )
    selected_paths = (
        activation, definitions / activation.stem, evidence_root / activation.stem,
        bootstraps / identifier,
        production / "data" / "evidence" / "v1" / "cycles" / current_cycle,
        *(engine / logical for logical in CURATED_CODE_PATHS),
        *git_control_paths,
    )
    authority_directories = (
        research, research / "research", shadow, activation.parent,
        definitions, evidence_root, bootstraps,
        production, production / "data", production / "data" / "evidence",
        production / "data" / "evidence" / "v1",
        production / "data" / "evidence" / "v1" / "cycles",
        engine, engine / ".git",
    )
    before = (
        _directory_identities(authority_directories),
        _selected_fingerprint(selected_paths),
    )
    try:
        if not all(
            item.supported
            for item in (research_guard, production_guard, engine_guard, *git_guards)
        ):
            raise DecisionSurfaceInputError("source mutation observation is unavailable")
        from quantpits.research.forward_definition_evidence import adopt_forward_definition_evidence
        evidence = adopt_forward_definition_evidence(
            research, evidence_cycle, activation, definitions, evidence_root,
        )
        if evidence.status != "ADOPTED" or not evidence.definition_evidence_complete:
            raise DecisionSurfaceInputError("formal definition evidence was not adopted exactly")
        from quantpits.research.forward_observation import observe_frozen_shadow_forward_definition_candidate
        definition = observe_frozen_shadow_forward_definition_candidate(
            research, evidence_cycle, activation,
        )
        reference_cycle, source_receipt = _bootstrap_authority(
            research, bootstraps, identifier, definition, evidence,
        )
        if not (
            date.fromisoformat(current_cycle) > date.fromisoformat(reference_cycle)
            and date.fromisoformat(current_cycle)
            >= date.fromisoformat(activation_raw["effective_cycle"])
        ):
            raise DecisionSurfaceContractError("current cycle chronology is invalid")
        reference_path, reference_manifest, _reference_seal = _cycle_authority(
            research, reference_cycle,
        )
        reference_before = _selected_fingerprint((reference_path,))
        if (
            source_receipt.get("phase37a_seal_digest")
            != _digest(_read_regular(reference_path / "seal.json", private=True)[0], "raw_bytes")
            or source_receipt.get("phase37a_manifest_digest")
            != _digest(_read_regular(reference_path / "manifest.json", private=True)[0], "raw_bytes")
        ):
            raise DecisionSurfaceInputError("bootstrap source cycle provenance does not join")
        reference_sources = _source_projection(reference_manifest)
        source_reference_valid = _source_matches_definition(reference_sources, definition)
        reference_ensemble = _ensemble_projection(reference_path, reference_manifest)
        ensemble_reference_valid = (
            reference_ensemble["protocol"]
            == definition.compiled_definitions.champion.to_dict()["fusion_definition"]
        )
        try:
            reference_intent = _intent_projection(reference_path, reference_manifest)
            intent_reference_valid = _intent_matches_definition(reference_intent, definition)
        except _PROCESS_CONTROL:
            raise
        except Exception:
            intent_reference_valid = False
            reference_intent = _definition_intent_projection(definition)
        reference_commit = _engine_commit(reference_manifest)
        reference_code = _git_blob_projection(engine, reference_commit)
        reference_prediction = _prediction_projection(reference_manifest)
        reference_market = _market_projection(reference_manifest)
        reference_values = (
            reference_code, reference_sources, reference_ensemble,
            reference_prediction, reference_market, reference_intent,
        )
        try:
            current_path, current_manifest, _current_seal = _cycle_authority(
                production, current_cycle,
            )
        except _PROCESS_CONTROL:
            raise
        except _ComponentIncomparable as exc:
            components = tuple(
                _component(name, reference, None, exc.reason_code)
                for name, reference in zip(COMPONENT_NAMES, reference_values)
            )
            after = (
                _directory_identities(authority_directories),
                _selected_fingerprint(selected_paths),
            )
            if (
                before != after or research_guard.mutated()
                or production_guard.mutated() or engine_guard.mutated()
                or any(item.mutated() for item in git_guards)
                or reference_before != _selected_fingerprint((reference_path,))
            ):
                raise DecisionSurfaceInputError("authority continuity was lost")
            return _make_result(components, reference_cycle, current_cycle)
        current_commit = _engine_commit(current_manifest)
        components = (
            _observe_component(
                COMPONENT_NAMES[0], reference_code,
                lambda: _git_blob_projection(engine, current_commit),
            ),
            _component(
                COMPONENT_NAMES[1], reference_sources, None,
                "REFERENCE_SOURCE_DEFINITION_MISMATCH",
            ) if not source_reference_valid else _observe_component(
                COMPONENT_NAMES[1], reference_sources,
                lambda: _source_projection(current_manifest),
            ),
            _component(
                COMPONENT_NAMES[2], reference_ensemble, None,
                "REFERENCE_ENSEMBLE_DEFINITION_MISMATCH",
            ) if not ensemble_reference_valid else _observe_component(
                COMPONENT_NAMES[2], reference_ensemble,
                lambda: _ensemble_projection(current_path, current_manifest),
            ),
            _component(
                COMPONENT_NAMES[3], reference_prediction, None,
                "REFERENCE_SOURCE_DEFINITION_MISMATCH",
            ) if not source_reference_valid else _observe_component(
                COMPONENT_NAMES[3], reference_prediction,
                lambda: _prediction_projection(current_manifest),
            ),
            _observe_component(
                COMPONENT_NAMES[4], reference_market,
                lambda: _market_projection(current_manifest),
            ),
            _component(
                COMPONENT_NAMES[5], reference_intent, None,
                "REFERENCE_INTENT_DEFINITION_MISMATCH",
            ) if not intent_reference_valid else _observe_component(
                COMPONENT_NAMES[5], reference_intent,
                lambda: _intent_projection(current_path, current_manifest),
            ),
        )
        after = (
            _directory_identities(authority_directories),
            _selected_fingerprint(selected_paths),
        )
        if (
            before != after or research_guard.mutated() or production_guard.mutated()
            or engine_guard.mutated() or any(item.mutated() for item in git_guards)
            or reference_before != _selected_fingerprint((reference_path,))
        ):
            raise DecisionSurfaceInputError("authority continuity was lost")
        return _make_result(components, reference_cycle, current_cycle)
    except _PROCESS_CONTROL:
        raise
    except (DecisionSurfaceContractError, DecisionSurfaceInputError):
        raise
    except Exception as exc:
        raise DecisionSurfaceInputError("decision-surface observation failed closed") from exc
    finally:
        active = __import__("sys").exc_info()[1]
        for guard in (*reversed(git_guards), engine_guard, production_guard, research_guard):
            try:
                guard.close()
            except _PROCESS_CONTROL:
                if not isinstance(active, _PROCESS_CONTROL):
                    raise
            except OSError:
                if active is None:
                    raise DecisionSurfaceInputError("mutation observer cleanup failed")


def observe_production_decision_surface(
    research_workspace_root: Any,
    production_workspace_root: Any,
    engine_root: Any,
    current_cycle_id: Any,
    activation_path: Any,
    definition_store_root: Any,
    evidence_store_root: Any,
    bootstrap_store_root: Any,
    bootstrap_set_id: Any,
) -> ProductionDecisionSurfaceResult:
    """Fail closed with typed errors while preserving process-control."""
    try:
        return _observe_production_decision_surface(
            research_workspace_root, production_workspace_root, engine_root,
            current_cycle_id, activation_path, definition_store_root,
            evidence_store_root, bootstrap_store_root, bootstrap_set_id,
        )
    except _PROCESS_CONTROL:
        raise
    except DecisionSurfaceContractError:
        raise
    except Exception as exc:
        raise DecisionSurfaceInputError(
            "decision-surface observation failed closed",
        ) from exc


__all__ = [
    "SCHEMA_VERSION", "OBSERVATION_KIND", "SURFACE_PROTOCOL",
    "COMPONENT_NAMES", "CURATED_CODE_PATHS", "DecisionSurfaceContractError",
    "DecisionSurfaceInputError", "DecisionSurfaceComponent",
    "ProductionDecisionSurfaceResult", "observe_production_decision_surface",
]
