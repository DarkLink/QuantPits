"""Execution lifecycle for plan-first static and CPCV commands.

The existing model runners remain injected compatibility hooks.  This module
owns the safeguard-adjacent runtime lifecycle, manifest truth, and operator-log
linkage without importing either CLI script.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from quantpits.runtime import CommandResult, OutputRef, manifest_from_result, write_run_manifest
from quantpits.training.command import PreparedTrainingRun
from quantpits.training.errors import TrainingExecutionError
from quantpits.utils.operator_log import OperatorLog
from quantpits.utils.workspace import fingerprint_file, short_fingerprint


@dataclass(frozen=True)
class TrainingExecutionHooks:
    activate_workspace: Callable[[str], None]
    init_qlib: Callable[[], None]
    execute_legacy: Callable[[PreparedTrainingRun], object]
    clock: Callable[[], datetime] = datetime.now


@dataclass(frozen=True)
class TrainingRunSummary:
    run_id: str
    family: str
    action: str
    target_keys: tuple[str, ...]
    plan_fingerprint: str
    execution_fingerprint: str
    outcomes: tuple[dict, ...]
    manifest_path: Optional[str]


def _relative(prepared: PreparedTrainingRun, path: Path) -> str:
    return path.resolve().relative_to(prepared.ctx.root.resolve()).as_posix()


def _normalize_outcomes(prepared: PreparedTrainingRun, result) -> tuple[dict, ...]:
    if not isinstance(result, dict):
        succeeded = {item.key for item in prepared.targets}
        failed = set()
    else:
        succeeded = set(result.get("succeeded", ()))
        failed = set(result.get("failed", ()))
    outcomes = []
    for target in prepared.targets:
        if target.key in succeeded or target.model_name in succeeded:
            status, code = "success", None
        elif target.key in failed or target.model_name in failed:
            status, code = "failed", "target_execution_failed"
        else:
            status, code = "preserved", "no_verified_output"
        item = {"key": target.key, "operation": prepared.options.action, "outcome": status}
        if code:
            item["error_code"] = code
        outcomes.append(item)
    return tuple(outcomes)


class TrainingExecutionService:
    def __init__(self, hooks: TrainingExecutionHooks):
        self.hooks = hooks

    def execute(self, prepared: PreparedTrainingRun) -> TrainingRunSummary:
        started = self.hooks.clock().replace(microsecond=0).isoformat()
        manifest_rel = None
        baseline = {}
        execution_result = None
        outcomes = ()
        execution_fingerprint = ""
        for item in prepared.plan.outputs:
            if item.kind == "manifest":
                continue
            path = prepared.ctx.path(item.path)
            baseline[item.path] = fingerprint_file(path) if path.is_file() else None
        log_file = prepared.ctx.data_path("operator_log.jsonl")
        with OperatorLog(
            prepared.plan.command,
            args=list(prepared.cli_args),
            log_file=str(log_file),
            run_id=prepared.plan.run_id,
            plan_fingerprint=prepared.plan_fingerprint,
        ) as oplog:
            try:
                self.hooks.activate_workspace(str(prepared.ctx.root))
                self.hooks.init_qlib()
                execution_result = self.hooks.execute_legacy(prepared)
                outcomes = _normalize_outcomes(prepared, execution_result)
                execution_fingerprint = short_fingerprint({
                    "plan_fingerprint": prepared.plan_fingerprint,
                    "anchor_date": execution_result.get("anchor_date") if isinstance(execution_result, dict) else None,
                    "experiment_name": execution_result.get("experiment_name") if isinstance(execution_result, dict) else None,
                    "outcomes": outcomes,
                })
                if isinstance(execution_result, dict) and execution_result.get("success") is False:
                    raise TrainingExecutionError("one or more training targets failed")
                finished = self.hooks.clock().replace(microsecond=0).isoformat()
                outputs = tuple(
                    item for item in prepared.plan.outputs
                    if item.kind != "manifest" and prepared.ctx.path(item.path).is_file()
                    and fingerprint_file(prepared.ctx.path(item.path)) != baseline.get(item.path)
                )
                result = CommandResult(
                    plan=prepared.plan, status="success", started_at=started, finished_at=finished,
                    outputs=outputs, records={
                        "family": prepared.options.family,
                        "action": prepared.options.action,
                        "target_keys": [item.key for item in prepared.targets],
                        "plan_fingerprint": prepared.plan_fingerprint,
                        "execution_fingerprint": execution_fingerprint,
                        "outcomes": list(outcomes),
                        "publication": {
                            "policy": prepared.plan.metadata.get("publication_policy"),
                            "applied": bool(execution_result.get("published")) if isinstance(execution_result, dict) else None,
                        },
                    },
                )
                if not prepared.options.no_manifest:
                    path = write_run_manifest(prepared.ctx, manifest_from_result(result))
                    manifest_rel = _relative(prepared, path)
                    oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
                oplog.set_result({"status": "success", "target_count": len(prepared.targets)})
            except Exception as exc:
                finished = self.hooks.clock().replace(microsecond=0).isoformat()
                if not prepared.options.no_manifest:
                    committed_outputs = tuple(
                        item for item in prepared.plan.outputs
                        if item.kind != "manifest" and prepared.ctx.path(item.path).is_file()
                        and fingerprint_file(prepared.ctx.path(item.path)) != baseline.get(item.path)
                    )
                    result = CommandResult(
                        plan=prepared.plan, status="failed", started_at=started, finished_at=finished,
                        outputs=committed_outputs, records={
                            "family": prepared.options.family,
                            "action": prepared.options.action,
                            "target_keys": [item.key for item in prepared.targets],
                            "plan_fingerprint": prepared.plan_fingerprint,
                            "execution_fingerprint": execution_fingerprint,
                            "outcomes": list(outcomes),
                            "publication": {
                                "policy": prepared.plan.metadata.get("publication_policy"),
                                "applied": bool(execution_result.get("published")) if isinstance(execution_result, dict) else False,
                            },
                        }, error={"type": type(exc).__name__, "message": "training execution failed"},
                    )
                    path = write_run_manifest(prepared.ctx, manifest_from_result(result))
                    manifest_rel = _relative(prepared, path)
                    oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
                oplog.set_result({"status": "failed", "error_type": type(exc).__name__})
                raise TrainingExecutionError("training execution failed") from exc
        return TrainingRunSummary(
            run_id=prepared.plan.run_id, family=prepared.options.family,
            action=prepared.options.action, target_keys=tuple(item.key for item in prepared.targets),
            plan_fingerprint=prepared.plan_fingerprint,
            execution_fingerprint=execution_fingerprint, outcomes=outcomes,
            manifest_path=manifest_rel,
        )
