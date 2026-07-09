"""Shared types for workspace configuration validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple


Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class ValidationMessage:
    severity: Severity
    code: str
    path: str
    message: str
    hint: str = ""

    def to_dict(self) -> Dict[str, str]:
        payload = {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.hint:
            payload["hint"] = self.hint
        return payload


@dataclass(frozen=True)
class ConfigArtifact:
    name: str
    path: Path
    exists: bool
    raw: Optional[Any]
    normalized: Optional[Any]
    fingerprint: Optional[str]
    schema_version: Optional[int] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self, *, workspace: Optional[Path] = None) -> Dict[str, Any]:
        display_path = self.path
        if workspace is not None:
            try:
                display_path = self.path.relative_to(workspace)
            except ValueError:
                display_path = self.path

        payload: Dict[str, Any] = {
            "name": self.name,
            "path": display_path.as_posix(),
            "exists": self.exists,
            "fingerprint": self.fingerprint,
        }
        if self.schema_version is not None:
            payload["schema_version"] = self.schema_version
        if self.summary:
            payload["summary"] = self.summary
        return payload


@dataclass(frozen=True)
class WorkspaceValidationResult:
    workspace: Path
    ok: bool
    artifacts: Tuple[ConfigArtifact, ...] = field(default_factory=tuple)
    messages: Tuple[ValidationMessage, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> Tuple[ValidationMessage, ...]:
        return tuple(msg for msg in self.messages if msg.severity == "error")

    @property
    def warnings(self) -> Tuple[ValidationMessage, ...]:
        return tuple(msg for msg in self.messages if msg.severity == "warning")

    @property
    def infos(self) -> Tuple[ValidationMessage, ...]:
        return tuple(msg for msg in self.messages if msg.severity == "info")

    def to_public_dict(self, *, workspace_label: Optional[str] = None) -> Dict[str, Any]:
        return {
            "workspace": workspace_label or self.workspace.as_posix(),
            "ok": self.ok,
            "artifacts": [
                artifact.to_public_dict(workspace=self.workspace)
                for artifact in self.artifacts
            ],
            "messages": [message.to_dict() for message in self.messages],
        }


def has_errors(messages: Iterable[ValidationMessage]) -> bool:
    return any(message.severity == "error" for message in messages)
