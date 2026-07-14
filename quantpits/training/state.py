"""Locked, compare-and-swap Training Run State V3."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from quantpits.training.errors import TrainingStateConflictError
from quantpits.training.persistence import FileBaseline, atomic_write_json_bytes, read_with_baseline

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


PHASES = (
    "prepared", "executing", "targets_complete", "publication_prepared",
    "publication_committed", "failed", "completed",
)
TERMINAL_PHASES = ("failed", "completed")
TRANSITIONS = {
    "prepared": ("executing", "failed"),
    "executing": ("executing", "targets_complete", "failed"),
    "targets_complete": ("publication_prepared", "completed", "failed"),
    "publication_prepared": ("publication_committed", "failed"),
    "publication_committed": ("executing", "completed", "failed"),
    "failed": ("executing", "publication_prepared", "publication_committed", "completed"),
    "completed": (),
}


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
    phase: str = "executing"
    publication_transaction_id: Optional[str] = None
    publication_status: Optional[str] = None
    postprocess_status: Optional[str] = None
    manifest_path: Optional[str] = None
    manifest_fingerprint: Optional[str] = None
    aggregate_error_code: Optional[str] = None
    schema_version: int = 3

    @property
    def status(self):
        return self.phase

    def __post_init__(self):
        if self.phase not in PHASES:
            raise TrainingStateConflictError("unsupported training run-state phase")
        if len(self.target_keys) != len(set(self.target_keys)):
            raise TrainingStateConflictError("training run-state has duplicate target keys")
        if not set(self.outcomes).issubset(self.target_keys):
            raise TrainingStateConflictError("training run-state has an unknown target outcome")
        if self.phase in ("publication_prepared", "publication_committed") and not self.publication_transaction_id:
            raise TrainingStateConflictError("publication phase requires transaction identity")
        for key, outcome in self.outcomes.items():
            if outcome.get("published") and not outcome.get("recorder_id"):
                raise TrainingStateConflictError("published target requires recorder evidence: %s" % key)

    def to_dict(self) -> Dict[str, Any]:
        value = {
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
            "phase": self.phase,
            "status": self.phase,
        }
        optional = {
            "publication_transaction_id": self.publication_transaction_id,
            "publication_status": self.publication_status,
            "postprocess_status": self.postprocess_status,
            "manifest_path": self.manifest_path,
            "manifest_fingerprint": self.manifest_fingerprint,
            "aggregate_error_code": self.aggregate_error_code,
        }
        value.update({key: item for key, item in optional.items() if item is not None})
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRunState":
        if value.get("schema_version") != 3:
            raise TrainingStateConflictError(
                "legacy training run-state is display-only; clear or migrate it before resume"
            )
        return cls(
            run_id=str(value["run_id"]), family=str(value["family"]), action=str(value["action"]),
            plan_fingerprint=str(value["plan_fingerprint"]),
            execution_fingerprint=str(value["execution_fingerprint"]),
            resume_fingerprint=str(value.get("resume_fingerprint") or value["execution_fingerprint"]),
            anchor_date=str(value["anchor_date"]), target_keys=tuple(value["target_keys"]),
            outcomes=dict(value.get("outcomes", {})), phase=str(value.get("phase") or value.get("status")),
            publication_transaction_id=value.get("publication_transaction_id"),
            publication_status=value.get("publication_status"), manifest_path=value.get("manifest_path"),
            postprocess_status=value.get("postprocess_status"),
            manifest_fingerprint=value.get("manifest_fingerprint"),
            aggregate_error_code=value.get("aggregate_error_code"),
        )


class TrainingStateRepository:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _lock(self, *, shared=False, create=True):
        if not create and not self.path.parent.exists():
            yield
            return
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self):
        raw, baseline = read_with_baseline(self.path, display_path="data/run_state.json")
        state = None if raw is None else TrainingRunState.from_dict(json.loads(raw.decode("utf-8")))
        return state, baseline

    def inspect_readonly(self):
        return self._read_unlocked()

    def load_locked(self):
        with self._lock(shared=True):
            return self._read_unlocked()

    def load(self) -> Optional[TrainingRunState]:
        return self.load_locked()[0]

    @staticmethod
    def _same_baseline(left, right):
        return (
            left.existed == right.existed
            and left.fingerprint == right.fingerprint
            and left.size_bytes == right.size_bytes
        )

    @staticmethod
    def _validate_transition(previous, current):
        if previous is None:
            return
        if previous.run_id != current.run_id:
            raise TrainingStateConflictError("another run owns the training state")
        identity = ("family", "action", "plan_fingerprint", "execution_fingerprint", "resume_fingerprint", "anchor_date", "target_keys")
        if any(getattr(previous, name) != getattr(current, name) for name in identity):
            raise TrainingStateConflictError("training state identity changed during execution")
        if current.phase != previous.phase and current.phase not in TRANSITIONS[previous.phase]:
            raise TrainingStateConflictError("invalid training state phase transition")

    def save(self, state: TrainingRunState, *, expected: Optional[FileBaseline] = None) -> FileBaseline:
        with self._lock():
            previous, observed = self._read_unlocked()
            if expected is not None and not self._same_baseline(observed, expected):
                raise TrainingStateConflictError("training state changed since it was inspected")
            self._validate_transition(previous, state)
            payload = (json.dumps(state.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            atomic_write_json_bytes(self.path, payload)
            return read_with_baseline(self.path, display_path="data/run_state.json")[1]

    def clear(self, *, expected: Optional[FileBaseline] = None, require_terminal=False) -> None:
        with self._lock():
            state, observed = self._read_unlocked()
            if expected is not None and not self._same_baseline(observed, expected):
                raise TrainingStateConflictError("training state changed before clear")
            if require_terminal and state is not None and state.phase not in TERMINAL_PHASES:
                raise TrainingStateConflictError("cannot clear a nonterminal training state")
            if self.path.exists():
                self.path.unlink()
