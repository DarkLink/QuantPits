"""Versioned, atomic training resume state."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from quantpits.training.errors import TrainingStateConflictError


@dataclass(frozen=True)
class TrainingRunState:
    run_id: str
    family: str
    action: str
    plan_fingerprint: str
    execution_fingerprint: str
    resume_fingerprint: str
    anchor_date: str
    target_keys: Tuple[str, ...]
    outcomes: Mapping[str, Mapping[str, Any]]
    status: str = "running"
    schema_version: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "family": self.family,
            "action": self.action,
            "plan_fingerprint": self.plan_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "resume_fingerprint": self.resume_fingerprint,
            "anchor_date": self.anchor_date,
            "target_keys": list(self.target_keys),
            "outcomes": dict(self.outcomes),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRunState":
        if value.get("schema_version") != 2:
            raise TrainingStateConflictError("unsupported training run-state schema")
        return cls(
            run_id=str(value["run_id"]), family=str(value["family"]), action=str(value["action"]),
            plan_fingerprint=str(value["plan_fingerprint"]),
            execution_fingerprint=str(value["execution_fingerprint"]),
            resume_fingerprint=str(value.get("resume_fingerprint") or value["execution_fingerprint"]),
            anchor_date=str(value["anchor_date"]), target_keys=tuple(value["target_keys"]),
            outcomes=dict(value.get("outcomes", {})), status=str(value.get("status", "running")),
        )


class TrainingStateRepository:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def load(self) -> Optional[TrainingRunState]:
        if not self.path.is_file():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            return TrainingRunState.from_dict(json.load(handle))

    def save(self, state: TrainingRunState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".%s." % self.path.name, dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            os.replace(name, str(self.path))
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
