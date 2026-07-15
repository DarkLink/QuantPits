"""Authoritative adapter around the legacy Rolling execution functions."""

import copy
import os
from dataclasses import dataclass
from pathlib import Path

from quantpits.rolling.errors import RollingExecutionError, RollingInputChangedError
from quantpits.utils.workspace import fingerprint_file


def resolve_legacy_workflow_path(value, workspace_root=None):
    """Resolve a legacy workflow path against the explicit workspace root.

    Rolling historically achieved this by changing the process cwd at module
    import.  Phase 28A removes that global side effect, so legacy entry points
    resolve the same relative paths locally instead.
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    root = workspace_root or os.environ.get("QLIB_WORKSPACE_DIR")
    if root:
        return str((Path(root).expanduser().resolve() / path).resolve())
    return str(path)


def recheck_prepared_inputs(prepared):
    """Fail closed when any declared filesystem baseline has drifted."""

    changed = []
    for ref in prepared.plan.inputs:
        if (prepared.options.action == "clear_state"
                and ref.path != prepared.state.path):
            continue
        path = prepared.ctx.path(ref.path)
        actual = fingerprint_file(path) if path.is_file() else None
        if actual != ref.fingerprint:
            changed.append(ref.path)
    if changed:
        raise RollingInputChangedError(
            "prepared input changed: %s" % ", ".join(sorted(changed))
        )


@dataclass(frozen=True)
class RollingExecutionOutcome:
    status: str
    run_id: str
    plan_fingerprint: str
    action: str = None
    reason_code: str = None
    message: str = None
    did_execute: bool = False
    execution_fingerprint: str = None
    target_keys: tuple = ()


class LegacyRollingExecutionAdapter:
    """Execute only the targets, config and windows frozen by Phase 28."""

    def __init__(self, facade):
        self.facade = facade

    @staticmethod
    def _targets(prepared):
        targets = {}
        for target in prepared.targets:
            info = dict(target.legacy_info)
            info["yaml_file"] = str(prepared.ctx.path(target.workflow_path))
            targets[target.model_name] = info
        return targets

    @staticmethod
    def _edit_retrain_models(prepared, state):
        for target in prepared.targets:
            removed = state.remove_model(target.model_name)
            if removed:
                print("🔄 %s: cleared %s window record(s)" % (
                    target.model_name, removed,
                ))

    @staticmethod
    def _edit_retrain_last(prepared, state):
        if not state.anchor_date:
            raise RollingExecutionError("no Rolling state; run --cold-start first")
        last_idx = state.get_last_completed_window_idx()
        if last_idx is None:
            print("ℹ️  No completed windows to retrain")
            return False
        completed = state.get_all_completed_windows()
        key = str(last_idx)
        removed = 0
        for target in prepared.targets:
            if key in completed and target.model_name in completed[key]:
                del completed[key][target.model_name]
                removed += 1
        if removed:
            state.save()
        print("🔄 Window %s: cleared %s selected model(s)" % (last_idx, removed))
        return removed > 0

    def _run_action(self, action, effective_args, prepared, resolved, targets,
                    state_path):
        """Return status metadata for command-visible legacy action results."""

        from quantpits.scripts.rolling.state import RollingState

        if action == "clear_state" and prepared.state.status == "missing":
            return ("skipped", "rolling_action_skipped",
                    "legacy state is already missing", False)
        facade_result = None
        if action == "clear_state":
            RollingState(state_file=state_path).clear()
            return (
                "success", "rolling_action_completed",
                "legacy state was cleared", True,
            )
        elif action == "retrain_models":
            self._edit_retrain_models(
                prepared, RollingState(state_file=state_path),
            )
            effective_args.merge = True
            effective_args.resume = False
            facade_result = self.facade.run_cold_start(
                effective_args, targets, prepared.effective_config,
                resolved=resolved,
            )
        elif action == "retrain_last":
            should_run = self._edit_retrain_last(
                prepared, RollingState(state_file=state_path),
            )
            if not should_run:
                return (
                    "skipped", "rolling_action_skipped",
                    "no selected model record exists in the last window", False,
                )
            facade_result = self.facade.run_daily(
                effective_args, targets, prepared.effective_config,
                resolved=resolved,
            )
        elif action == "backtest_only":
            facade_result = self.facade.run_backtest_only(
                effective_args, targets, dict(resolved.params),
                mode=prepared.plan.metadata["family"],
            )
        elif action == "predict_only":
            facade_result = self.facade.run_predict_only(
                effective_args, targets, prepared.effective_config,
                resolved=resolved,
            )
        elif action in ("cold_start", "merge", "resume"):
            facade_result = self.facade.run_cold_start(
                effective_args, targets, prepared.effective_config,
                resolved=resolved,
            )
        elif action == "daily":
            facade_result = self.facade.run_daily(
                effective_args, targets, prepared.effective_config,
                resolved=resolved,
            )
        else:
            raise RollingExecutionError("unsupported Rolling action: %s" % action)
        if isinstance(facade_result, dict):
            result_status = facade_result.get("status")
            if result_status in ("success", "skipped", "failed"):
                return (
                    result_status,
                    facade_result.get("reason_code") or (
                        "rolling_action_skipped" if result_status == "skipped"
                        else (
                            "rolling_execution_failed" if result_status == "failed"
                            else "legacy_partial_visibility"
                        )
                    ),
                    facade_result.get("message") or (
                        "legacy action was skipped" if result_status == "skipped"
                        else (
                            "legacy action reported failure"
                            if result_status == "failed"
                            else "legacy action completed"
                        )
                    ),
                    bool(facade_result.get("did_execute", False)),
                )
            raise RollingExecutionError(
                "legacy action returned invalid status: %r" % result_status
            )
        return (
            "success", "legacy_partial_visibility",
            "legacy action returned without a command-level failure; "
            "per-window completion remains legacy-managed", True,
        )

    def execute(self, args, prepared, resolved, run_id):
        """Run the canonical action without registry or window rediscovery."""

        from quantpits.utils.operator_log import OperatorLog

        action = prepared.options.action
        if action == "clear_state":
            if resolved is not None:
                raise RollingExecutionError("clear-state must not carry runtime windows")
        elif resolved is None or resolved.prepared is not prepared:
            raise RollingExecutionError(
                "legacy adapter requires the matching ResolvedRollingRun"
            )
        execution_fingerprint = (
            resolved.execution_fingerprint if resolved is not None else None
        )
        targets = self._targets(prepared)
        state_path = str(prepared.ctx.path(prepared.state.path))
        effective_args = copy.copy(args)
        log = OperatorLog(
            "rolling_train",
            args=list(prepared.cli_args),
            log_file=str(prepared.ctx.data_path("operator_log.jsonl")),
            run_id=run_id,
            plan_fingerprint=prepared.plan_fingerprint,
        )
        with log:
            try:
                status, reason_code, message, did_execute = self._run_action(
                    action, effective_args, prepared, resolved, targets, state_path,
                )
                if (status == "success" and action not in
                        ("predict_only", "backtest_only", "clear_state")):
                    try:
                        from quantpits.scripts.deep_analysis.promote_config import (
                            update_promote_status,
                        )
                        update_promote_status(
                            str(prepared.ctx.root), model_names=list(targets),
                        )
                    except Exception:
                        pass
                log.set_result({
                    "status": status,
                    "action": action,
                    "reason_code": reason_code,
                    "message": message,
                    "did_execute": did_execute,
                    "n_targets": len(targets),
                    "target_keys": [item.target_key for item in prepared.targets],
                    "execution_fingerprint": execution_fingerprint,
                    "training_method": prepared.effective_config["training_method"],
                })
            except RollingExecutionError as exc:
                log.set_result({
                    "status": "failed", "action": action,
                    "reason_code": exc.code, "message": str(exc),
                    "did_execute": not (
                        action == "clear_state"
                        and prepared.state.status == "missing"
                    ),
                })
                raise
            except Exception as exc:
                wrapped = RollingExecutionError(
                    "legacy Rolling execution failed: %s" % exc
                )
                log.set_result({
                    "status": "failed", "action": action,
                    "reason_code": wrapped.code, "message": str(wrapped),
                    "did_execute": not (
                        action == "clear_state"
                        and prepared.state.status == "missing"
                    ),
                })
                raise wrapped from exc
        return RollingExecutionOutcome(
            status=status,
            run_id=run_id,
            plan_fingerprint=prepared.plan_fingerprint,
            action=action,
            reason_code=reason_code,
            message=message,
            did_execute=did_execute,
            execution_fingerprint=execution_fingerprint,
            target_keys=tuple(item.target_key for item in prepared.targets),
        )
