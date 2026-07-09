"""Safe config readers for workspace validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from quantpits.config_contracts.core import ConfigArtifact, ValidationMessage
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    relative_path: str
    required: bool
    loader: str
    normalizer: Callable[[Dict[str, Any]], Dict[str, Any]]
    validator: Callable[[Dict[str, Any]], List[ValidationMessage]]


def read_json_config(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml_config(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config_artifact(
    ctx: WorkspaceContext,
    spec: ConfigSpec,
    *,
    strict: bool = False,
) -> tuple[ConfigArtifact, List[ValidationMessage]]:
    path = ctx.path(*spec.relative_path.split("/"))
    messages: List[ValidationMessage] = []
    if not path.exists():
        severity = "error" if spec.required or strict else "warning"
        messages.append(
            ValidationMessage(
                severity=severity,
                code="missing-config",
                path=spec.relative_path,
                message=f"Config file not found: {spec.relative_path}",
            )
        )
        return ConfigArtifact(spec.name, path, False, None, None, None), messages

    try:
        raw = read_json_config(path) if spec.loader == "json" else read_yaml_config(path)
    except Exception as exc:
        messages.append(
            ValidationMessage(
                severity="error",
                code="parse-error",
                path=spec.relative_path,
                message=f"Failed to parse config: {exc}",
            )
        )
        return ConfigArtifact(spec.name, path, True, None, None, None), messages

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        messages.append(
            ValidationMessage(
                severity="error",
                code="invalid-root-type",
                path=spec.relative_path,
                message="Config root must be a mapping object.",
            )
        )
        return ConfigArtifact(spec.name, path, True, raw, None, None), messages

    normalized: Optional[Dict[str, Any]]
    try:
        normalized = spec.normalizer(raw)
    except Exception as exc:
        messages.append(
            ValidationMessage(
                severity="error",
                code="normalization-error",
                path=spec.relative_path,
                message=f"Failed to normalize config: {exc}",
            )
        )
        normalized = None

    if normalized is not None:
        messages.extend(spec.validator(normalized))
        fingerprint = fingerprint_value(normalized)
        schema_version = normalized.get("_schema_version")
    else:
        fingerprint = None
        schema_version = None

    from quantpits.config_contracts.normalizers import summarize_config

    artifact = ConfigArtifact(
        spec.name,
        path,
        True,
        raw,
        normalized,
        fingerprint,
        schema_version=schema_version if isinstance(schema_version, int) else None,
        summary=summarize_config(spec.name, normalized) if normalized is not None else {},
    )
    return artifact, messages
