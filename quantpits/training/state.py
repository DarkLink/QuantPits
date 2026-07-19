"""Locked, compare-and-swap Training Run State V3."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from quantpits.training.errors import TrainingStateConflictError
from quantpits.training.closure import CLOSURE_STEPS, RETRYABLE_WARNING_STEPS
from quantpits.training.persistence import FileBaseline, atomic_write_json_bytes, read_with_baseline

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


PHASES = (
    "prepared", "executing", "targets_complete", "publication_prepared",
    "publication_committed", "closing", "failed", "completed",
)
TERMINAL_PHASES = ("failed", "completed")
TRANSITIONS = {
    "prepared": ("executing", "failed"),
    "executing": ("executing", "targets_complete", "failed"),
    "targets_complete": ("publication_prepared", "completed", "failed"),
    "publication_prepared": ("publication_committed", "failed"),
    "publication_committed": ("executing", "closing", "completed", "failed"),
    "closing": ("closing", "completed", "failed"),
    "failed": ("executing", "publication_prepared", "publication_committed", "closing", "completed"),
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
    attempt_id: Optional[str] = None
    receipt_fingerprint: Optional[str] = None
    committed_outputs: Tuple[Mapping[str, Any], ...] = ()
    closure_steps: Mapping[str, str] = None
    logical_started_at: Optional[str] = None
    logical_finished_at: Optional[str] = None
    recovery_backfill_required: bool = False
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
        if self.phase in ("publication_prepared", "publication_committed", "closing") and not self.publication_transaction_id:
            raise TrainingStateConflictError("publication phase requires transaction identity")
        publication_backed = bool(
            self.publication_transaction_id
            or self.publication_status == "committed"
            or self.receipt_fingerprint
            or self.committed_outputs
        )
        if not self.recovery_backfill_required and (self.phase in ("publication_committed", "closing") or (
            self.phase == "completed" and publication_backed
        ) or self.publication_status == "committed"):
            if self.publication_status != "committed":
                raise TrainingStateConflictError("publication-backed state requires committed status")
            if not self.receipt_fingerprint or not self.committed_outputs:
                raise TrainingStateConflictError("publication-backed state requires receipt evidence")
        paths = []
        for item in self.committed_outputs:
            if not isinstance(item, Mapping) or set(item) != {"path", "kind", "fingerprint"}:
                raise TrainingStateConflictError("training state committed-output ledger is malformed")
            if any(not isinstance(item.get(name), str) or not item.get(name) for name in item):
                raise TrainingStateConflictError("training state committed-output ledger is malformed")
            paths.append(item["path"])
        if len(paths) != len(set(paths)):
            raise TrainingStateConflictError("training state committed-output ledger has duplicates")
        for key, outcome in self.outcomes.items():
            if outcome.get("published") and not outcome.get("recorder_id"):
                raise TrainingStateConflictError("published target requires recorder evidence: %s" % key)
            if outcome.get("published_this_attempt") and outcome.get("already_published"):
                raise TrainingStateConflictError("target cannot be newly and previously published")
            if (outcome.get("published_this_attempt") or outcome.get("already_published")) and not outcome.get("published"):
                raise TrainingStateConflictError("publication provenance requires published target")
            if outcome.get("outcome") == "success" and not outcome.get("published") and (
                not outcome.get("target_evidence_path")
                or not outcome.get("target_evidence_fingerprint")
            ):
                raise TrainingStateConflictError("successful unpublished target requires durable evidence: %s" % key)
        closure_steps = dict(self.closure_steps or {})
        if set(closure_steps) - set(CLOSURE_STEPS):
            raise TrainingStateConflictError("training state contains an unknown closure step")
        for name, status in closure_steps.items():
            if status not in ("warning", "completed"):
                raise TrainingStateConflictError("training state contains an invalid closure status")
            if status == "warning" and name not in RETRYABLE_WARNING_STEPS:
                raise TrainingStateConflictError("training state contains a non-retryable warning")
        if (
            not self.recovery_backfill_required
            and self.phase in ("closing", "completed")
            and publication_backed
        ):
            if closure_steps.get("receipt_verified") != "completed":
                raise TrainingStateConflictError("closing state requires verified receipt")
            if closure_steps.get("state_publication_bound") != "completed":
                raise TrainingStateConflictError("closing state requires publication binding")
        if not self.recovery_backfill_required and closure_steps and (
            not self.publication_transaction_id
            or not self.receipt_fingerprint
            or not self.committed_outputs
        ):
            raise TrainingStateConflictError("closure steps require publication evidence")
        if not self.recovery_backfill_required and closure_steps.get("manifest_verified") == "completed" and (
            not self.manifest_path or not self.manifest_fingerprint
        ):
            raise TrainingStateConflictError("verified manifest requires path and fingerprint")
        if self.logical_finished_at and not self.logical_started_at:
            raise TrainingStateConflictError("logical finish time requires logical start time")
        object.__setattr__(self, "closure_steps", closure_steps)
        object.__setattr__(self, "committed_outputs", tuple(dict(item) for item in self.committed_outputs))

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
            "attempt_id": self.attempt_id,
            "receipt_fingerprint": self.receipt_fingerprint,
            "logical_started_at": self.logical_started_at,
            "logical_finished_at": self.logical_finished_at,
        }
        value.update({key: item for key, item in optional.items() if item is not None})
        if self.committed_outputs:
            value["committed_outputs"] = list(self.committed_outputs)
        if self.closure_steps:
            value["closure_steps"] = dict(self.closure_steps)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRunState":
        if value.get("schema_version") != 3:
            raise TrainingStateConflictError(
                "legacy training run-state is display-only; clear or migrate it before resume"
            )
        phase = str(value.get("phase") or value.get("status"))
        transaction_id = value.get("publication_transaction_id")
        closure_steps = dict(value.get("closure_steps", {}))
        recovery_backfill_required = bool(
            transaction_id
            and (
                phase in ("publication_committed", "closing", "completed")
                or value.get("publication_status") == "committed"
            )
            and (
                not value.get("receipt_fingerprint")
                or not value.get("committed_outputs")
                or (
                    phase in ("closing", "completed")
                    and closure_steps.get("state_publication_bound") != "completed"
                )
                or (
                    closure_steps.get("manifest_verified") == "completed"
                    and (not value.get("manifest_path") or not value.get("manifest_fingerprint"))
                )
            )
        )
        return cls(
            run_id=str(value["run_id"]), family=str(value["family"]), action=str(value["action"]),
            plan_fingerprint=str(value["plan_fingerprint"]),
            execution_fingerprint=str(value["execution_fingerprint"]),
            resume_fingerprint=str(value.get("resume_fingerprint") or value["execution_fingerprint"]),
            anchor_date=str(value["anchor_date"]), target_keys=tuple(value["target_keys"]),
            outcomes=dict(value.get("outcomes", {})), phase=phase,
            publication_transaction_id=transaction_id,
            publication_status=value.get("publication_status"), manifest_path=value.get("manifest_path"),
            postprocess_status=value.get("postprocess_status"),
            manifest_fingerprint=value.get("manifest_fingerprint"),
            aggregate_error_code=value.get("aggregate_error_code"),
            attempt_id=value.get("attempt_id"),
            receipt_fingerprint=value.get("receipt_fingerprint"),
            committed_outputs=tuple(value.get("committed_outputs", ())),
            closure_steps=closure_steps,
            logical_started_at=value.get("logical_started_at"),
            logical_finished_at=value.get("logical_finished_at"),
            recovery_backfill_required=recovery_backfill_required,
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
        if previous.logical_started_at and current.logical_started_at != previous.logical_started_at:
            raise TrainingStateConflictError("logical training start time changed")
        if previous.logical_finished_at and current.logical_finished_at != previous.logical_finished_at:
            raise TrainingStateConflictError("logical training finish time changed")
        resets_publication = (
            previous.phase == "failed"
            and current.phase == "executing"
            and current.publication_transaction_id is None
        )
        if not resets_publication:
            for name, status in previous.closure_steps.items():
                next_status = current.closure_steps.get(name)
                if status == "completed" and next_status != "completed":
                    raise TrainingStateConflictError("completed training closure step regressed")
                if status == "warning" and next_status not in ("warning", "completed"):
                    raise TrainingStateConflictError("warning training closure step regressed")

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
            raw, observed = read_with_baseline(
                self.path, display_path="data/run_state.json",
            )
            if expected is not None and not self._same_baseline(observed, expected):
                raise TrainingStateConflictError("training state changed before clear")
            if require_terminal and raw is not None:
                state = TrainingRunState.from_dict(json.loads(raw.decode("utf-8")))
                if state.phase not in TERMINAL_PHASES:
                    raise TrainingStateConflictError("cannot clear a nonterminal training state")
            if self.path.exists():
                self.path.unlink()
