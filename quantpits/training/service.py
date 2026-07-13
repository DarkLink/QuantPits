"""Service-owned static/CPCV execution, publication, state, and audit lifecycle."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

from quantpits.runtime import CommandResult, OutputRef, manifest_from_result, write_run_manifest
from quantpits.training.command import PreparedTrainingRun
from quantpits.training.errors import (
    TrainingCommandError,
    TrainingExecutionError,
    TrainingPublicationError,
    TrainingRunnerContractError,
    TrainingStateConflictError,
)
from quantpits.training.record_repository import TrainingRecordConflictError, TrainingRecordRepository
from quantpits.training.records import ModelRecordOutcome
from quantpits.training.resolved import ResolvedTrainingRun, resolve_training_run
from quantpits.training.runners import (
    TrainingTargetRequest,
    TrainingTargetResult,
    run_cpcv_target,
    run_static_target,
)
from quantpits.training.state import TrainingRunState, TrainingStateRepository
from quantpits.utils.operator_log import OperatorLog
from quantpits.utils.workspace import fingerprint_file


def _noop_cache(_run):
    return None


@dataclass(frozen=True)
class TrainingExecutionHooks:
    activate_workspace: Callable[[str], None]
    init_qlib: Callable[[], None]
    calculate_dates: Callable[[], Mapping[str, Any]]
    prepare_cache: Callable[[ResolvedTrainingRun], Any] = _noop_cache
    run_static_target: Callable[[TrainingTargetRequest], TrainingTargetResult] = run_static_target
    run_cpcv_target: Callable[[TrainingTargetRequest], TrainingTargetResult] = run_cpcv_target
    clock: Callable[[], datetime] = datetime.now


@dataclass(frozen=True)
class TrainingRunSummary:
    run_id: str
    family: str
    action: str
    target_keys: Tuple[str, ...]
    plan_fingerprint: str
    execution_fingerprint: str
    outcomes: Tuple[dict, ...]
    manifest_path: Optional[str]
    publication_applied: bool


def prepare_default_cache(run: ResolvedTrainingRun):
    if run.prepared.options.cache_size_mb == 0:
        return None
    from quantpits.utils.handler_cache import (
        HandlerCacheManager,
        enumerate_tasks_cpcv,
        enumerate_tasks_static,
        pre_analyze,
    )

    manager = HandlerCacheManager(max_size_mb=run.prepared.options.cache_size_mb)
    names = [item.target.model_name for item in run.targets]
    paths = {
        item.target.model_name: str(run.prepared.ctx.path(item.target.workflow_path))
        for item in run.targets
    }
    enumerate_tasks = enumerate_tasks_cpcv if run.prepared.options.family == "cpcv" else enumerate_tasks_static
    pre_analyze(enumerate_tasks(names, paths, dict(run.params)), manager)
    return manager


def default_execution_hooks(*, activate_workspace, init_qlib, calculate_dates):
    return TrainingExecutionHooks(
        activate_workspace=activate_workspace,
        init_qlib=init_qlib,
        calculate_dates=calculate_dates,
        prepare_cache=prepare_default_cache,
    )


def _relative(prepared: PreparedTrainingRun, path: Path) -> str:
    return path.resolve().relative_to(prepared.ctx.root.resolve()).as_posix()


def _operation_results(results: Tuple[TrainingTargetResult, ...]) -> Tuple[ModelRecordOutcome, ...]:
    return tuple(ModelRecordOutcome(
        key=item.key,
        requested_operation=item.operation,
        outcome=item.outcome,
        entry=item.entry,
        error_type=item.error_type,
        error_code=item.error_code,
    ) for item in results)


def _public_outcomes(results: Tuple[TrainingTargetResult, ...], published_keys=frozenset()) -> Tuple[dict, ...]:
    values = []
    for item in results:
        value = {"key": item.key, "operation": item.operation, "outcome": item.outcome}
        if item.entry is not None:
            value.update({
                "recorder_id": item.entry.recorder_id,
                "experiment_name": item.entry.experiment_name,
                "published": item.key in published_keys,
            })
        if item.error_code:
            value["error_code"] = item.error_code
        if item.error_type:
            value["error_type"] = item.error_type
        values.append(value)
    return tuple(values)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


class TrainingExecutionService:
    def __init__(self, hooks: TrainingExecutionHooks):
        self.hooks = hooks

    def _resolve(self, prepared: PreparedTrainingRun) -> ResolvedTrainingRun:
        self.hooks.activate_workspace(str(prepared.ctx.root))
        self.hooks.init_qlib()
        run = resolve_training_run(prepared, self.hooks.calculate_dates())
        self._verify_input_baselines(prepared)
        return run

    @staticmethod
    def _verify_input_baselines(prepared: PreparedTrainingRun) -> None:
        for relative, expected in prepared.input_fingerprints.items():
            path = prepared.ctx.path(relative)
            if expected is None:
                if path.exists():
                    raise TrainingStateConflictError("a prepared input appeared before execution")
                continue
            if not path.is_file() or fingerprint_file(path) != expected:
                raise TrainingStateConflictError("a prepared input changed before execution")

    @staticmethod
    def _state_for(run: ResolvedTrainingRun, outcomes, status="running") -> TrainingRunState:
        normalized = {}
        for item in outcomes:
            if isinstance(item, Mapping):
                key = str(item.get("key") or "")
                if not key:
                    raise TrainingRunnerContractError("run-state outcome has no target key")
                normalized[key] = {name: value for name, value in item.items() if name != "key"}
            else:
                key = str(item.key)
                normalized[key] = item
        return TrainingRunState(
            run_id=run.prepared.plan.run_id,
            family=run.prepared.options.family,
            action=run.prepared.options.action,
            plan_fingerprint=run.prepared.plan_fingerprint,
            execution_fingerprint=run.execution_fingerprint,
            resume_fingerprint=run.resume_fingerprint,
            anchor_date=run.anchor_date,
            target_keys=tuple(item.key for item in run.targets),
            outcomes=normalized,
            status=status,
        )

    @staticmethod
    def _validate_resume(run: ResolvedTrainingRun, state: TrainingRunState) -> None:
        expected = (
            run.prepared.options.family,
            run.prepared.options.action,
            run.resume_fingerprint,
            run.anchor_date,
            tuple(item.key for item in run.targets),
        )
        actual = (
            state.family, state.action, state.resume_fingerprint,
            state.anchor_date, state.target_keys,
        )
        if actual != expected:
            raise TrainingStateConflictError("resume state differs from resolved execution")

    def _run_targets(self, run: ResolvedTrainingRun, state_repo: TrainingStateRepository):
        cache = self.hooks.prepare_cache(run)
        prior = state_repo.load() if run.prepared.options.resume else None
        if prior is not None:
            self._validate_resume(run, prior)
        outcome_map = dict(prior.outcomes) if prior else {}
        for target in run.targets:
            existing = dict(outcome_map.get(target.key, {}))
            source = target.source_entry
            if source is not None:
                expected_source = (
                    existing.get("source_recorder_id"),
                    existing.get("source_experiment_name"),
                    existing.get("source_operation"),
                )
                actual_source = (source.recorder_id, source.experiment_name, source.operation)
                if prior is not None and existing.get("published") is not True and expected_source != actual_source:
                    raise TrainingStateConflictError("resume source recorder differs from original execution")
                existing.update({
                    "source_recorder_id": source.recorder_id,
                    "source_experiment_name": source.experiment_name,
                    "source_operation": source.operation,
                })
            outcome_map[target.key] = existing
        current = run.prepared.current_snapshot.entry_map
        results = []
        for target in run.targets:
            saved = outcome_map.get(target.key)
            if saved and saved.get("outcome") == "success" and saved.get("published") is True:
                entry = current.get(target.key)
                if entry is None or entry.recorder_id != saved.get("recorder_id"):
                    raise TrainingStateConflictError("completed resume target no longer matches current record")
                results.append(TrainingTargetResult(
                    target.key, target.operation, "preserved",
                    error_code="resume_already_published",
                ))
                continue
            request = TrainingTargetRequest(run, target, cache)
            try:
                runner = self.hooks.run_cpcv_target if run.prepared.options.family == "cpcv" else self.hooks.run_static_target
                result = runner(request)
                if not isinstance(result, TrainingTargetResult):
                    raise TrainingRunnerContractError("runner did not return TrainingTargetResult")
                if result.key != target.key or result.operation != target.operation:
                    raise TrainingRunnerContractError("runner result differs from requested target")
            except TrainingCommandError as exc:
                result = TrainingTargetResult(
                    target.key, target.operation, "failed",
                    error_code=exc.code, error_type=type(exc).__name__,
                )
            except Exception as exc:
                result = TrainingTargetResult(
                    target.key, target.operation, "failed",
                    error_code="target_execution_failed", error_type=type(exc).__name__,
                )
            results.append(result)
            source_fields = {
                key: value for key, value in outcome_map.get(target.key, {}).items()
                if key.startswith("source_")
            }
            outcome_map[target.key] = {
                **source_fields,
                "outcome": result.outcome,
                "operation": result.operation,
                "recorder_id": result.entry.recorder_id if result.entry else None,
                "error_code": result.error_code,
                "published": False,
            }
            state_repo.save(self._state_for(
                run, ({"key": key, **value} for key, value in outcome_map.items())
            ))
        if tuple(item.key for item in results) != tuple(item.key for item in run.targets):
            raise TrainingRunnerContractError("target result cardinality differs from resolved run")
        return tuple(results)

    def _publish(self, run: ResolvedTrainingRun, results: Tuple[TrainingTargetResult, ...]):
        repo = TrainingRecordRepository.for_workspace(run.prepared.ctx, clock=self.hooks.clock)
        outcomes = _operation_results(results)
        successes = tuple(item for item in results if item.outcome == "success")
        failures = tuple(item for item in results if item.outcome == "failed")
        published = frozenset()
        if run.prepared.options.action == "full":
            if failures or len(successes) != len(run.targets):
                return False, published
            repo.overwrite(outcomes, baseline=run.prepared.current_record_baseline)
            published = frozenset(item.key for item in successes)
        elif successes:
            repo.merge(outcomes, baseline=run.prepared.current_record_baseline)
            published = frozenset(item.key for item in successes)
        return bool(published), published

    def _publish_performance(self, run: ResolvedTrainingRun, results, published_keys):
        values = {item.key.rsplit("@", 1)[0]: dict(item.performance) for item in results
                  if item.key in published_keys and item.performance is not None}
        if not values:
            return None
        suffix = "_cpcv" if run.prepared.options.family == "cpcv" else ""
        path = run.prepared.ctx.output_path("model_performance%s_%s.json" % (suffix, run.anchor_date))
        if run.prepared.options.action != "full" and path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise TrainingPublicationError("existing performance output is invalid") from exc
            for model_name, new_value in values.items():
                old_value = existing.get(model_name)
                if isinstance(old_value, Mapping) and "convergence" in old_value and "convergence" not in new_value:
                    new_value["convergence"] = old_value["convergence"]
                existing[model_name] = new_value
            values = existing
        _atomic_json(path, values)
        return path

    def execute(self, prepared: PreparedTrainingRun) -> TrainingRunSummary:
        started = self.hooks.clock().replace(microsecond=0).isoformat()
        manifest_rel = None
        run = None
        results = ()
        published_keys = frozenset()
        publication_applied = False
        committed = []
        state_repo = TrainingStateRepository(prepared.ctx.data_path("run_state.json"))
        log_file = prepared.ctx.data_path("operator_log.jsonl")
        with OperatorLog(
            prepared.plan.command, args=list(prepared.cli_args), log_file=str(log_file),
            run_id=prepared.plan.run_id, plan_fingerprint=prepared.plan_fingerprint,
        ) as oplog:
            try:
                run = self._resolve(prepared)
                results = self._run_targets(run, state_repo)
                publication_applied, published_keys = self._publish(run, results)
                if publication_applied:
                    record_path = prepared.ctx.root / "latest_train_records.json"
                    committed.append(OutputRef(
                        "latest_train_records.json", kind="record", overwrite=True,
                        description="verified current recorder registry",
                    ))
                    performance_path = self._publish_performance(run, results, published_keys)
                    if performance_path is not None:
                        committed.append(OutputRef(_relative(prepared, performance_path), kind="report", overwrite=True))
                failed = [item for item in results if item.outcome == "failed"]
                state = self._state_for(
                    run,
                    ({"key": item.key, "outcome": item.outcome,
                      "recorder_id": item.entry.recorder_id if item.entry else None,
                      "error_code": item.error_code,
                      "published": item.key in published_keys} for item in results),
                    status="failed" if failed else "completed",
                )
                if failed:
                    state_repo.save(state)
                    committed.append(OutputRef("data/run_state.json", kind="state", overwrite=True))
                else:
                    state_repo.clear()
                public = _public_outcomes(results, published_keys)
                if failed:
                    raise TrainingExecutionError("one or more training targets failed")
                finished = self.hooks.clock().replace(microsecond=0).isoformat()
                result = CommandResult(
                    plan=prepared.plan, status="success", started_at=started, finished_at=finished,
                    outputs=tuple(committed), records=self._records(run, public, publication_applied),
                )
                if not prepared.options.no_manifest:
                    path = write_run_manifest(prepared.ctx, manifest_from_result(result))
                    manifest_rel = _relative(prepared, path)
                    oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
                oplog.set_result({
                    "status": "success", "target_count": len(results),
                    "execution_fingerprint": run.execution_fingerprint,
                    "publication_applied": publication_applied,
                })
            except Exception as exc:
                finished = self.hooks.clock().replace(microsecond=0).isoformat()
                execution_fp = run.execution_fingerprint if run else ""
                public = _public_outcomes(results, published_keys)
                if not prepared.options.no_manifest:
                    result = CommandResult(
                        plan=prepared.plan, status="failed", started_at=started, finished_at=finished,
                        outputs=tuple(committed),
                        records=self._records(run, public, publication_applied),
                        error={"type": type(exc).__name__, "message": "training execution failed"},
                    )
                    path = write_run_manifest(prepared.ctx, manifest_from_result(result))
                    manifest_rel = _relative(prepared, path)
                    oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
                oplog.set_result({
                    "status": "failed", "error_type": type(exc).__name__,
                    "execution_fingerprint": execution_fp,
                    "publication_applied": publication_applied,
                })
                if isinstance(exc, TrainingExecutionError):
                    raise
                if isinstance(exc, TrainingRecordConflictError):
                    raise TrainingPublicationError("training record publication conflicted") from exc
                raise TrainingExecutionError("training execution failed") from exc
        return TrainingRunSummary(
            run_id=prepared.plan.run_id, family=prepared.options.family,
            action=prepared.options.action, target_keys=tuple(item.key for item in prepared.targets),
            plan_fingerprint=prepared.plan_fingerprint,
            execution_fingerprint=run.execution_fingerprint,
            outcomes=_public_outcomes(results, published_keys), manifest_path=manifest_rel,
            publication_applied=publication_applied,
        )

    @staticmethod
    def _records(run, outcomes, publication_applied):
        if run is None:
            return {"outcomes": list(outcomes), "publication": {"applied": False}}
        return {
            "family": run.prepared.options.family,
            "action": run.prepared.options.action,
            "target_keys": [item.key for item in run.targets],
            "plan_fingerprint": run.prepared.plan_fingerprint,
            "execution_fingerprint": run.execution_fingerprint,
            "resume_fingerprint": run.resume_fingerprint,
            "anchor_date": run.anchor_date,
            "output_experiment_name": run.output_experiment_name,
            "outcomes": list(outcomes),
            "publication": {
                "policy": run.prepared.plan.metadata.get("publication_policy"),
                "applied": publication_applied,
                "record_baseline": run.prepared.current_record_baseline.file_fingerprint,
            },
        }
