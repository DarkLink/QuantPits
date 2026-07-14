"""Durable, idempotent post-publication closure state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from quantpits.training.persistence import atomic_write_json_bytes, read_with_baseline
from quantpits.training.errors import TrainingStateConflictError


CLOSURE_STEPS = (
    "receipt_verified", "state_publication_bound", "history_appended", "promotion_applied",
    "manifest_verified", "operator_log_linked", "terminal_state_saved", "state_cleared",
)
RETRYABLE_WARNING_STEPS = frozenset(("history_appended", "promotion_applied"))
STEP_STATUSES = frozenset(("warning", "completed"))


def required_closure_steps(*, no_manifest, include_terminal=True, include_clear=False):
    steps = [
        "receipt_verified", "state_publication_bound", "history_appended",
        "promotion_applied", "operator_log_linked",
    ]
    if not no_manifest:
        steps.append("manifest_verified")
    if include_terminal:
        steps.append("terminal_state_saved")
    if include_clear:
        steps.append("state_cleared")
    return tuple(steps)


def closure_complete(steps, *, no_manifest, include_terminal=True, include_clear=False):
    return all(
        steps.get(name) == "completed"
        for name in required_closure_steps(
            no_manifest=no_manifest,
            include_terminal=include_terminal,
            include_clear=include_clear,
        )
    )


@dataclass(frozen=True)
class TrainingClosureState:
    run_id: str
    transaction_id: str
    receipt_fingerprint: str
    steps: Mapping[str, str]
    warnings: Tuple[str, ...] = ()
    committed_outputs: Tuple[Mapping[str, str], ...] = ()
    manifest_fingerprint: Optional[str] = None
    operator_log_fingerprint: Optional[str] = None
    step_errors: Mapping[str, str] = None
    schema_version: int = 1

    def __post_init__(self):
        if not self.run_id or not self.transaction_id or not self.receipt_fingerprint:
            raise TrainingStateConflictError("training closure identity is incomplete")
        steps = dict(self.steps)
        unknown = set(steps) - set(CLOSURE_STEPS)
        if unknown or any(status not in STEP_STATUSES for status in steps.values()):
            raise TrainingStateConflictError("training closure contains an invalid step")
        for name, status in steps.items():
            if status == "warning" and name not in RETRYABLE_WARNING_STEPS:
                raise TrainingStateConflictError("training closure warning is not retryable")
        errors = dict(self.step_errors or {})
        if set(errors) - set(CLOSURE_STEPS):
            raise TrainingStateConflictError("training closure contains an unknown step error")
        if any(steps.get(name) != "warning" for name in errors):
            raise TrainingStateConflictError("training closure step error lacks warning state")
        if steps.get("manifest_verified") == "completed" and not self.manifest_fingerprint:
            raise TrainingStateConflictError("verified manifest requires a durable fingerprint")
        if steps.get("operator_log_linked") == "completed" and not self.operator_log_fingerprint:
            raise TrainingStateConflictError("operator-log closure requires durable evidence")
        if (
            steps.get("receipt_verified") == "completed"
            or steps.get("state_publication_bound") == "completed"
        ) and not self.committed_outputs:
            raise TrainingStateConflictError("verified receipt requires a committed-output ledger")
        paths = []
        for item in self.committed_outputs:
            if not isinstance(item, Mapping) or set(item) != {"path", "kind", "fingerprint"}:
                raise TrainingStateConflictError("training closure output ledger is malformed")
            if any(not isinstance(item.get(name), str) or not item.get(name) for name in item):
                raise TrainingStateConflictError("training closure output ledger is malformed")
            paths.append(item["path"])
        if len(paths) != len(set(paths)):
            raise TrainingStateConflictError("training closure output ledger has duplicates")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "step_errors", errors)
        object.__setattr__(self, "committed_outputs", tuple(dict(item) for item in self.committed_outputs))

    def to_dict(self):
        value = {"schema_version": self.schema_version, "run_id": self.run_id,
                "transaction_id": self.transaction_id, "receipt_fingerprint": self.receipt_fingerprint,
                "steps": dict(self.steps), "warnings": list(self.warnings),
                "committed_outputs": [dict(item) for item in self.committed_outputs],
                "step_errors": dict(self.step_errors)}
        if self.manifest_fingerprint:
            value["manifest_fingerprint"] = self.manifest_fingerprint
        if self.operator_log_fingerprint:
            value["operator_log_fingerprint"] = self.operator_log_fingerprint
        return value


class TrainingClosureRepository:
    def __init__(self, ctx, run_id, transaction_id=None):
        filename = "closure-%s.json" % transaction_id if transaction_id else "closure.json"
        self.path = ctx.data_path("training_runs", run_id, filename)

    def load(self):
        raw, baseline = read_with_baseline(self.path, display_path="data/training_runs/<run_id>/closure.json")
        if raw is None:
            return None, baseline
        value = json.loads(raw.decode("utf-8"))
        if value.get("schema_version") != 1:
            raise TrainingStateConflictError("unsupported training closure state")
        return TrainingClosureState(
            str(value["run_id"]), str(value["transaction_id"]), str(value["receipt_fingerprint"]),
            dict(value.get("steps", {})), tuple(value.get("warnings", ())),
            tuple(value.get("committed_outputs", ())), value.get("manifest_fingerprint"),
            value.get("operator_log_fingerprint"), dict(value.get("step_errors", {})),
        ), baseline

    def save(self, state):
        existing, _baseline = self.load()
        if existing is not None and (
            existing.run_id != state.run_id
            or existing.transaction_id != state.transaction_id
            or existing.receipt_fingerprint != state.receipt_fingerprint
        ):
            raise TrainingStateConflictError("training closure identity changed")
        if existing is not None:
            if existing.committed_outputs and existing.committed_outputs != state.committed_outputs:
                raise TrainingStateConflictError("training closure output ledger changed")
            if existing.manifest_fingerprint and (
                state.manifest_fingerprint != existing.manifest_fingerprint
            ):
                raise TrainingStateConflictError("training closure manifest evidence changed")
            if existing.operator_log_fingerprint and (
                state.operator_log_fingerprint != existing.operator_log_fingerprint
            ):
                raise TrainingStateConflictError("training closure operator evidence changed")
            for name, status in existing.steps.items():
                next_status = state.steps.get(name)
                if status == "completed" and next_status != "completed":
                    raise TrainingStateConflictError("completed training closure step regressed")
                if status == "warning" and next_status not in ("warning", "completed"):
                    raise TrainingStateConflictError("warning training closure step regressed")
        payload = (json.dumps(state.to_dict(), sort_keys=True, indent=2) + "\n").encode("utf-8")
        atomic_write_json_bytes(self.path, payload)
