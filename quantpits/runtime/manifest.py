"""Run manifest dataclass and writer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from quantpits.runtime.command import (
    CommandResult,
    CommandStep,
    InputRef,
    OutputRef,
    StateRef,
    to_public_value,
)
from quantpits.utils.workspace import WorkspaceContext


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    command: str
    workspace: str
    started_at: str
    finished_at: str
    status: str
    args: Tuple[str, ...]
    inputs: Tuple[InputRef, ...]
    outputs: Tuple[OutputRef, ...]
    states: Tuple[StateRef, ...]
    steps: Tuple[CommandStep, ...]
    config_fingerprints: Dict[str, str]
    records: Dict[str, Any]
    warnings: Tuple[str, ...]
    error: Dict[str, str] | None = None
    schema_version: int = 1

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "command": self.command,
            "workspace": self.workspace,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "args": list(self.args),
            "inputs": [item.to_public_dict() for item in self.inputs],
            "outputs": [item.to_public_dict() for item in self.outputs],
            "states": [item.to_public_dict() for item in self.states],
            "steps": [item.to_public_dict() for item in self.steps],
            "config_fingerprints": dict(self.config_fingerprints),
            "records": to_public_value(self.records),
            "warnings": list(self.warnings),
            "error": self.error,
        }


def run_manifest_to_public_dict(manifest: RunManifest) -> Dict[str, Any]:
    return manifest.to_public_dict()


def manifest_from_result(result: CommandResult) -> RunManifest:
    plan = result.plan
    return RunManifest(
        run_id=plan.run_id,
        command=plan.command,
        workspace=plan.workspace,
        started_at=result.started_at,
        finished_at=result.finished_at,
        status=result.status,
        args=plan.args,
        inputs=plan.inputs,
        outputs=result.outputs or plan.outputs,
        states=plan.states,
        steps=plan.steps,
        config_fingerprints=plan.config_fingerprints,
        records=result.records,
        warnings=tuple(plan.warnings) + tuple(result.warnings),
        error=result.error,
    )


def _safe_command_dir(command: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in command).strip("_") or "command"


def manifest_path(ctx: WorkspaceContext, command: str, run_id: str) -> Path:
    return ctx.output_path("manifests", _safe_command_dir(command), f"{run_id}.json")


def write_run_manifest(ctx: WorkspaceContext, manifest: RunManifest) -> Path:
    path = manifest_path(ctx, manifest.command, manifest.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = manifest.to_public_dict()
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path
