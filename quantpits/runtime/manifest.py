"""Run manifest dataclass and writer."""

from __future__ import annotations

import json
import hashlib
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


class RunManifestConflictError(RuntimeError):
    code = "manifest_conflict"


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
        # ``CommandResult.outputs`` is the committed-artifact ledger.  An
        # empty tuple is meaningful (for example, a failure before the first
        # write) and must never fall back to planned outputs.
        outputs=result.outputs,
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
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def write_or_adopt_run_manifest(
    ctx: WorkspaceContext, manifest: RunManifest, *, allow_failed_supersession=False,
    allow_verified_success_adoption=False, expected_receipt_ledger=None,
):
    """Write a manifest once, or adopt byte-identical durable evidence.

    Returns ``(path, sha256)``.  A different manifest at the same logical
    run path is a recovery conflict. Callers may explicitly supersede a
    prior failed-attempt manifest while closing the same logical run.
    """
    path = manifest_path(ctx, manifest.command, manifest.run_id)
    payload = (json.dumps(
        manifest.to_public_dict(), indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n").encode("utf-8")
    fingerprint = hashlib.sha256(payload).hexdigest()
    if path.is_file():
        current = path.read_bytes()
        if current != payload:
            try:
                previous = json.loads(current.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                previous = {}
            proposed = manifest.to_public_dict()
            next_publication = proposed.get("records", {}).get("publication", {})
            if previous.get("status") == "success":
                canonical = dict(proposed)
                for name in (
                    "args", "inputs", "states", "steps", "config_fingerprints",
                ):
                    if name in previous:
                        canonical[name] = previous[name]
                canonical_payload = (json.dumps(
                    canonical, indent=2, sort_keys=True, ensure_ascii=False
                ) + "\n").encode("utf-8")
                prior_ledger = previous.get("records", {}).get(
                    "publication", {}
                ).get("committed_outputs")
                if (
                    allow_verified_success_adoption
                    and expected_receipt_ledger is not None
                    and prior_ledger == list(expected_receipt_ledger)
                    and canonical_payload == current
                ):
                    return path, hashlib.sha256(current).hexdigest()
                raise RunManifestConflictError("run manifest conflicts with durable evidence")
            failed_supersession = (
                allow_failed_supersession
                and previous.get("status") == "failed"
                and previous.get("run_id") == proposed.get("run_id")
                and previous.get("command") == proposed.get("command")
                and previous.get("workspace") == proposed.get("workspace")
                and previous.get("records", {}).get("plan_fingerprint") is not None
                and previous.get("records", {}).get("plan_fingerprint")
                == proposed.get("records", {}).get("plan_fingerprint")
                and all(
                    previous.get("records", {}).get(name)
                    == proposed.get("records", {}).get(name)
                    for name in (
                        "family", "action", "target_keys", "execution_fingerprint",
                        "resume_fingerprint", "anchor_date",
                    )
                )
            )
            if not failed_supersession:
                raise RunManifestConflictError("run manifest conflicts with durable evidence")
            if expected_receipt_ledger is not None:
                if next_publication.get("committed_outputs") != list(expected_receipt_ledger):
                    raise RunManifestConflictError(
                        "run manifest receipt ledger conflicts with durable evidence"
                    )
            # A resume may close a failed attempt, but it must not rewrite
            # the logical run's original start time.
            if previous.get("started_at"):
                proposed["started_at"] = previous["started_at"]
            # The logical run keeps the original prepared command evidence;
            # --resume arguments and the transient state input describe the
            # closure attempt, not a new execution identity.
            for name in ("args", "inputs", "states", "steps", "config_fingerprints"):
                if name in previous:
                    proposed[name] = previous[name]
            payload = (json.dumps(
                proposed, indent=2, sort_keys=True, ensure_ascii=False
            ) + "\n").encode("utf-8")
            fingerprint = hashlib.sha256(payload).hexdigest()
        else:
            return path, fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path, fingerprint
