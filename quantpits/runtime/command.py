"""Shared runtime command plan primitives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

from quantpits.utils.workspace import short_fingerprint


RefKind = Literal[
    "config",
    "data",
    "state",
    "prediction",
    "report",
    "record",
    "checkpoint",
    "manifest",
    "other",
]

StateAction = Literal["read", "write", "read_write"]
CommandStatus = Literal["success", "failed", "skipped"]


def to_public_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_public_value(asdict(value))
    if isinstance(value, tuple):
        return [to_public_value(item) for item in value]
    if isinstance(value, list):
        return [to_public_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): to_public_value(item)
            for key, item in value.items()
            if str(key) not in {"raw", "raw_config"}
        }
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        normalized = [to_public_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), default=str),
        )
    return value


@dataclass(frozen=True)
class InputRef:
    path: str
    kind: RefKind = "other"
    fingerprint: str | None = None
    required: bool = True
    description: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "required": self.required,
        }
        if self.fingerprint:
            payload["fingerprint"] = self.fingerprint
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class OutputRef:
    path: str
    kind: RefKind = "other"
    description: str = ""
    overwrite: bool = False

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "overwrite": self.overwrite,
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class StateRef:
    path: str
    action: StateAction
    description: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"path": self.path, "action": self.action}
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class CommandStep:
    name: str
    description: str
    expensive: bool = False
    can_skip: bool = False
    skip_reason: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "expensive": self.expensive,
            "can_skip": self.can_skip,
        }
        if self.skip_reason:
            payload["skip_reason"] = self.skip_reason
        return payload


@dataclass(frozen=True)
class CommandPlan:
    command: str
    workspace: str
    run_id: str
    mode: str = ""
    args: Tuple[str, ...] = ()
    inputs: Tuple[InputRef, ...] = ()
    outputs: Tuple[OutputRef, ...] = ()
    states: Tuple[StateRef, ...] = ()
    steps: Tuple[CommandStep, ...] = ()
    config_fingerprints: Dict[str, str] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "command": self.command,
            "workspace": self.workspace,
            "run_id": self.run_id,
            "mode": self.mode,
            "args": list(self.args),
            "inputs": [item.to_public_dict() for item in self.inputs],
            "outputs": [item.to_public_dict() for item in self.outputs],
            "states": [item.to_public_dict() for item in self.states],
            "steps": [item.to_public_dict() for item in self.steps],
            "config_fingerprints": dict(self.config_fingerprints),
            "warnings": list(self.warnings),
            "metadata": to_public_value(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class CommandResult:
    plan: CommandPlan
    status: CommandStatus
    started_at: str
    finished_at: str
    outputs: Tuple[OutputRef, ...] = ()
    records: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    error: Dict[str, str] | None = None

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "plan": self.plan.to_public_dict(),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outputs": [item.to_public_dict() for item in self.outputs],
            "records": to_public_value(self.records),
            "warnings": list(self.warnings),
            "error": self.error,
        }
        return payload


def _strip_plan_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("plan_fingerprint", None)
        normalized["metadata"] = metadata
    return normalized


def fingerprint_command_plan(
    plan: CommandPlan,
    *,
    length: int = 12,
    include_run_id: bool = False,
) -> str:
    """Return a stable short fingerprint for a command plan.

    ``run_id`` is volatile and excluded by default so repeated dry-runs of the
    same semantic plan are comparable.
    """

    payload = _strip_plan_fingerprint(plan.to_public_dict())
    if not include_run_id:
        payload.pop("run_id", None)
    return short_fingerprint(payload, length=length)
