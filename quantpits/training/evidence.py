"""Immutable evidence for completed training targets."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from quantpits.training.errors import TrainingEvidenceConflictError
from quantpits.training.persistence import atomic_write_json_bytes, read_with_baseline, sha256_bytes
from quantpits.training.records import ModelRecordEntry
from quantpits.training.runners import TrainingTargetResult


def _safe_key(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return "%s-%s" % (safe or "target", hashlib.sha256(value.encode("utf-8")).hexdigest()[:12])


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TrainingEvidenceConflictError("training target evidence contains an unsupported value")


@dataclass(frozen=True)
class TrainingTargetEvidence:
    run_id: str
    attempt_id: str
    target_key: str
    operation: str
    outcome: str
    entry: Optional[Mapping[str, Any]]
    performance: Optional[Mapping[str, Any]]
    history_payload: Optional[Mapping[str, Any]]
    source_identity: Optional[Mapping[str, Any]]
    anchor_date: str
    output_experiment_name: str
    plan_fingerprint: str
    resume_fingerprint: str
    execution_fingerprint: str
    observed_at: str
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.outcome not in ("success", "failed", "skipped"):
            raise TrainingEvidenceConflictError("unsupported training target evidence")
        if self.outcome == "success" and self.entry is None:
            raise TrainingEvidenceConflictError("successful target evidence requires a record entry")
        if self.entry is not None:
            entry = ModelRecordEntry.from_dict(self.entry)
            if entry.key != self.target_key or entry.operation != self.operation:
                raise TrainingEvidenceConflictError("target evidence identity differs from record entry")

    def to_dict(self):
        return {
            "schema_version": self.schema_version, "run_id": self.run_id,
            "attempt_id": self.attempt_id, "target_key": self.target_key,
            "operation": self.operation, "outcome": self.outcome,
            "entry": dict(self.entry) if self.entry is not None else None,
            "performance": dict(self.performance) if self.performance is not None else None,
            "history_payload": dict(self.history_payload) if self.history_payload is not None else None,
            "source_identity": dict(self.source_identity) if self.source_identity is not None else None,
            "anchor_date": self.anchor_date,
            "output_experiment_name": self.output_experiment_name,
            "plan_fingerprint": self.plan_fingerprint,
            "resume_fingerprint": self.resume_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))

    def to_result(self) -> TrainingTargetResult:
        entry = ModelRecordEntry.from_dict(self.entry) if self.entry is not None else None
        return TrainingTargetResult(
            self.target_key, self.operation, self.outcome, entry=entry,
            performance=self.performance, history_event=self.history_payload,
        )


class TrainingTargetEvidenceRepository:
    def __init__(self, ctx, run_id: str):
        self.ctx = ctx
        self.run_id = run_id
        self.root = ctx.data_path("training_runs", run_id, "targets")

    def path_for(self, target_key: str) -> Path:
        return self.root / ("%s.json" % _safe_key(target_key))

    def relative_path(self, target_key: str) -> str:
        return self.path_for(target_key).relative_to(self.ctx.root).as_posix()

    @staticmethod
    def _payload(evidence: TrainingTargetEvidence) -> bytes:
        try:
            return (json.dumps(
                _json_safe(evidence.to_dict()), sort_keys=True, ensure_ascii=False, indent=2
            ) + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TrainingEvidenceConflictError("training target evidence is not JSON serializable") from exc

    def write(self, evidence: TrainingTargetEvidence) -> Tuple[str, str]:
        if evidence.run_id != self.run_id:
            raise TrainingEvidenceConflictError("target evidence belongs to another run")
        path = self.path_for(evidence.target_key)
        payload = self._payload(evidence)
        fingerprint = sha256_bytes(payload)
        current, _baseline = read_with_baseline(path, display_path=self.relative_path(evidence.target_key))
        if current is not None:
            if current != payload:
                raise TrainingEvidenceConflictError("immutable target evidence already exists with different bytes")
            return self.relative_path(evidence.target_key), fingerprint
        atomic_write_json_bytes(path, payload)
        return self.relative_path(evidence.target_key), fingerprint

    def load(self, target_key: str, *, expected_fingerprint: Optional[str] = None) -> TrainingTargetEvidence:
        path = self.path_for(target_key)
        raw, baseline = read_with_baseline(path, display_path=self.relative_path(target_key))
        if raw is None:
            raise TrainingEvidenceConflictError("training target evidence is missing")
        if expected_fingerprint is not None and baseline.fingerprint != expected_fingerprint:
            raise TrainingEvidenceConflictError("training target evidence fingerprint differs from state")
        try:
            evidence = TrainingTargetEvidence.from_dict(json.loads(raw.decode("utf-8")))
        except (TypeError, ValueError, KeyError) as exc:
            raise TrainingEvidenceConflictError("training target evidence is invalid") from exc
        if evidence.run_id != self.run_id or evidence.target_key != target_key:
            raise TrainingEvidenceConflictError("training target evidence identity mismatch")
        return evidence
