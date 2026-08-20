"""Strict public contracts for production-cycle evidence.

Authority-bearing facts are deliberately absent from :class:`CaptureRequest`.
They are observed and assembled only by ``ProductionCycleEvidenceSealer``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import InitVar, dataclass
from datetime import datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


class ContractError(ValueError):
    """A representation cannot enter the evidence contract."""


STRICT_ID = re.compile(r"^[A-Z0-9][A-Z0-9_.-]{0,127}$")
DIGEST_DOMAINS = frozenset({
    "raw_bytes", "canonical_json", "file_inventory", "semantic_config",
})
DECISIONS = frozenset({"NO_ACTION", "APPROVE", "REJECT", "DEFER"})
REASON_CODES = frozenset({
    "FROZEN_OBSERVATION", "INSUFFICIENT_EVIDENCE", "RISK_BOUNDARY",
    "ENGINEERING_ONLY", "OWNER_OVERRIDE",
})
PRESERVATION = frozenset({
    "embedded", "workspace_file", "local_artifact", "missing", "incomparable",
})
STATUSES = frozenset({
    "sealed_complete", "sealed_partial", "adopted", "conflict", "blocked",
    "failed_no_final", "uncertain", "preview_complete", "preview_partial",
})
_RESULT_AUTHORITY = object()


def strict_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not STRICT_ID.fullmatch(value):
        raise ContractError("%s must be a strict uppercase identifier" % name)
    return value


def workspace_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ContractError("%s must be a canonical workspace-relative path" % name)
    path = PurePosixPath(value)
    if (
        path.is_absolute() or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractError("%s must be a canonical workspace-relative path" % name)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON and reject non-finite numbers."""
    try:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("value is not canonical JSON: %s" % exc) from exc
    return (text + "\n").encode("utf-8")


@dataclass(frozen=True)
class TypedDigest:
    algorithm: str
    domain: str
    value: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ContractError("only sha256 digests are supported")
        if not isinstance(self.domain, str) or self.domain not in DIGEST_DOMAINS:
            raise ContractError("unknown digest domain")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ContractError("digest size_bytes must be a non-negative integer")
        if not isinstance(self.value, str) or not re.fullmatch(r"[0-9a-f]{64}", self.value):
            raise ContractError("digest value must be lowercase sha256 hex")

    @classmethod
    def raw(cls, data: bytes) -> "TypedDigest":
        if not isinstance(data, bytes):
            raise ContractError("raw digest input must be bytes")
        return cls("sha256", "raw_bytes", hashlib.sha256(data).hexdigest(), len(data))

    @classmethod
    def canonical(cls, value: Any, domain: str = "canonical_json") -> "TypedDigest":
        if not isinstance(domain, str) or domain not in {"canonical_json", "file_inventory", "semantic_config"}:
            raise ContractError("invalid canonical digest domain")
        data = canonical_json_bytes(value)
        return cls("sha256", domain, hashlib.sha256(data).hexdigest(), len(data))

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm, "domain": self.domain,
            "value": self.value, "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class DecisionEvent:
    decision_id: str
    decision_time: str
    evidence_cycle_id: str
    actor: str
    decision: str
    target: str
    reason_code: str
    optional_note: Optional[str] = None

    def __post_init__(self) -> None:
        strict_id(self.decision_id, "decision_id")
        strict_id(self.evidence_cycle_id, "evidence_cycle_id")
        for name in ("decision_time", "actor", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError("decision %s must be non-empty text" % name)
        try:
            observed_time = datetime.fromisoformat(self.decision_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError("decision_time must be an ISO-8601 timestamp") from exc
        if observed_time.tzinfo is None:
            raise ContractError("decision_time must include an explicit timezone")
        if self.decision not in DECISIONS:
            raise ContractError("decision enum is invalid")
        if self.reason_code not in REASON_CODES:
            raise ContractError("reason_code enum is invalid")
        if self.optional_note is not None and not isinstance(self.optional_note, str):
            raise ContractError("optional_note must be text or null")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionEvent":
        if not isinstance(value, Mapping):
            raise ContractError("decision event must be an object")
        allowed = {
            "decision_id", "decision_time", "evidence_cycle_id", "actor",
            "decision", "target", "reason_code", "optional_note",
        }
        required = allowed - {"optional_note"}
        if set(value) - allowed or not required.issubset(value):
            raise ContractError("decision event fields are not exact")
        return cls(**{name: value.get(name) for name in allowed})

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_time": self.decision_time,
            "evidence_cycle_id": self.evidence_cycle_id,
            "actor": self.actor,
            "decision": self.decision,
            "target": self.target,
            "reason_code": self.reason_code,
            "optional_note": self.optional_note,
        }


@dataclass(frozen=True)
class CaptureRequest:
    cycle_id: str
    research_epoch_id: str
    post_trade_manifest: str
    prediction_manifest: str
    ensemble_manifest: str
    order_manifest: str
    deep_analysis_run: Optional[str] = None
    decision_event: Optional[str] = None

    def __post_init__(self) -> None:
        strict_id(self.cycle_id, "cycle_id")
        strict_id(self.research_epoch_id, "research_epoch_id")
        for name in (
            "post_trade_manifest", "prediction_manifest", "ensemble_manifest",
            "order_manifest",
        ):
            workspace_relative_path(getattr(self, name), name)
        for name in ("deep_analysis_run", "decision_event"):
            value = getattr(self, name)
            if value is not None:
                workspace_relative_path(value, name)

    def source_paths(self) -> Tuple[Tuple[str, str, bool], ...]:
        return (
            ("post_trade", self.post_trade_manifest, True),
            ("prediction", self.prediction_manifest, True),
            ("ensemble", self.ensemble_manifest, True),
            ("order", self.order_manifest, True),
            ("deep_analysis", self.deep_analysis_run or "", False),
            ("decision", self.decision_event or "", False),
        )


@dataclass(frozen=True)
class CaptureResult:
    cycle_id: str
    status: str
    did_write: bool
    bundle_path: Optional[str]
    seal_digest: Optional[TypedDigest]
    problems: Tuple[dict, ...] = ()
    sealed_status: Optional[str] = None
    _authority: InitVar[object] = None

    def __post_init__(self, _authority: object) -> None:
        if _authority is not _RESULT_AUTHORITY:
            raise ContractError("capture results are inspector-owned")
        strict_id(self.cycle_id, "cycle_id")
        if self.status not in STATUSES:
            raise ContractError("invalid capture result status")
        if not isinstance(self.did_write, bool):
            raise ContractError("did_write must be boolean")
        if self.seal_digest is not None and not isinstance(self.seal_digest, TypedDigest):
            raise ContractError("seal_digest must be a typed digest or null")
        if not isinstance(self.problems, tuple) or any(
            not isinstance(item, Mapping)
            or set(item) != {"code", "evidence_class", "detail", "blocks_complete"}
            or not isinstance(item.get("code"), str)
            or not item.get("code")
            or not isinstance(item.get("evidence_class"), str)
            or not item.get("evidence_class")
            or not isinstance(item.get("detail"), str)
            or not isinstance(item.get("blocks_complete"), bool)
            for item in self.problems
        ):
            raise ContractError("problems must be an exact inspector-owned inventory")
        object.__setattr__(self, "problems", tuple(
            MappingProxyType(dict(item)) for item in self.problems
        ))
        if self.status in {"conflict", "blocked", "failed_no_final", "uncertain"} and self.seal_digest is not None:
            raise ContractError("failed outcomes cannot grant a seal capability")
        if self.status in {"conflict", "blocked"} and self.did_write:
            raise ContractError("conflict and blocked outcomes cannot claim a write")
        if self.status in {"failed_no_final", "uncertain"} and not self.did_write:
            raise ContractError("post-write failures must retain their write fact")
        if self.status in {"conflict", "blocked", "failed_no_final", "uncertain", "preview_complete", "preview_partial"} and self.bundle_path is not None:
            raise ContractError("failed outcomes cannot expose a bundle capability path")
        if self.status in {"sealed_complete", "sealed_partial", "adopted"}:
            if not isinstance(self.bundle_path, str) or not self.bundle_path or self.seal_digest is None:
                raise ContractError("successful outcome requires bundle and seal")
            path = PurePosixPath(self.bundle_path)
            if (
                path.is_absolute() or path.as_posix() != self.bundle_path
                or "\\" in self.bundle_path
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ContractError("bundle capability path must be canonical and relative")
        if self.status in {"sealed_complete", "sealed_partial"} and not self.did_write:
            raise ContractError("new sealed evidence must retain its final write fact")
        if self.seal_digest is not None and self.seal_digest.domain != "raw_bytes":
            raise ContractError("seal digest must identify raw canonical seal bytes")
        if self.status == "adopted" and self.did_write:
            raise ContractError("adopted cannot claim a write")
        if self.status in {"preview_complete", "preview_partial"}:
            if self.did_write or self.seal_digest is None:
                raise ContractError("preview requires a non-authoritative digest and zero writes")
        expected_sealed = self.status if self.status in {"sealed_complete", "sealed_partial"} else None
        if self.status == "adopted":
            if self.sealed_status not in {"sealed_complete", "sealed_partial"}:
                raise ContractError("adopted requires its verified existing seal status")
        elif self.sealed_status != expected_sealed:
            raise ContractError("sealed_status must derive from terminal status")
        effective_status = self.sealed_status if self.status == "adopted" else {
            "preview_complete": "sealed_complete",
            "preview_partial": "sealed_partial",
        }.get(self.status, self.status)
        has_blocking_problem = any(item["blocks_complete"] for item in self.problems)
        if effective_status == "sealed_complete" and has_blocking_problem:
            raise ContractError("complete evidence cannot retain a blocking problem")
        if effective_status == "sealed_partial" and not has_blocking_problem:
            raise ContractError("partial evidence requires an exact blocking problem")

    @property
    def capability(self) -> str:
        if self.status.startswith("preview_") or (self.status != "adopted" and not self.did_write):
            return "none"
        status = self.sealed_status
        if status == "sealed_complete":
            return "local_evidence_replay"
        if status == "sealed_partial":
            return "partial_replay_only"
        return "none"

    @property
    def write_scope(self) -> str:
        if not self.did_write:
            return "none"
        if self.status == "failed_no_final":
            return "staging_only"
        if self.status == "uncertain":
            return "final_uncertain"
        return "final_bundle"

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id, "status": self.status,
            "did_write": self.did_write, "bundle_path": self.bundle_path,
            "seal_digest": self.seal_digest.to_dict() if self.seal_digest else None,
            "problems": [dict(item) for item in self.problems],
            "sealed_status": self.sealed_status,
            "capability": self.capability,
            "write_scope": self.write_scope,
        }


def finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False
