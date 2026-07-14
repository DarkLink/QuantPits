"""Service-owned static/CPCV execution and recoverable publication lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

from quantpits.runtime import CommandResult, OutputRef, manifest_from_result, write_run_manifest
from quantpits.training.command import PreparedTrainingRun
from quantpits.training.errors import (
    TrainingCommandError, TrainingExecutionError, TrainingPublicationError,
    TrainingRunnerContractError, TrainingStateConflictError,
)
from quantpits.training.lease import TrainingExecutionLease
from quantpits.training.history import TrainingHistoryRepository
from quantpits.training.persistence import FileBaseline, read_with_baseline
from quantpits.training.publication import TrainingPublicationCoordinator
from quantpits.training.record_repository import TrainingRecordConflictError, TrainingRecordRepository
from quantpits.training.resolved import ResolvedTrainingRun, resolve_training_run
from quantpits.training.runners import (
    TrainingTargetRequest, TrainingTargetResult, run_cpcv_target, run_static_target,
)
from quantpits.training.state import TrainingRunState, TrainingStateRepository
from quantpits.utils.operator_log import OperatorLog
from quantpits.utils.workspace import fingerprint_file


def _noop_cache(_run):
    return None


def _noop_promote(_workspace, _models):
    return None


@dataclass(frozen=True)
class TrainingExecutionHooks:
    activate_workspace: Callable[[str], None]
    init_qlib: Callable[[], None]
    calculate_dates: Callable[[], Mapping[str, Any]]
    prepare_cache: Callable[[ResolvedTrainingRun], Any] = _noop_cache
    run_static_target: Callable[[TrainingTargetRequest], TrainingTargetResult] = run_static_target
    run_cpcv_target: Callable[[TrainingTargetRequest], TrainingTargetResult] = run_cpcv_target
    promote_static: Callable[[str, Tuple[str, ...]], None] = _noop_promote
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
    warnings: Tuple[str, ...] = ()


def prepare_default_cache(run: ResolvedTrainingRun):
    if run.prepared.options.cache_size_mb == 0:
        return None
    from quantpits.utils.handler_cache import (
        HandlerCacheManager, enumerate_tasks_cpcv, enumerate_tasks_static, pre_analyze,
    )
    manager = HandlerCacheManager(max_size_mb=run.prepared.options.cache_size_mb)
    names = [item.target.model_name for item in run.targets]
    paths = {item.target.model_name: str(run.prepared.ctx.path(item.target.workflow_path)) for item in run.targets}
    enumerate_tasks = enumerate_tasks_cpcv if run.prepared.options.family == "cpcv" else enumerate_tasks_static
    pre_analyze(enumerate_tasks(names, paths, dict(run.params)), manager)
    return manager


def _lazy_calculate_dates():
    from quantpits.utils.train_utils import calculate_dates
    return calculate_dates()


def _default_promote(workspace, models):
    from quantpits.scripts.deep_analysis.promote_config import update_promote_status
    update_promote_status(workspace, model_names=list(models))


def default_execution_hooks(*, activate_workspace, init_qlib, calculate_dates=None):
    return TrainingExecutionHooks(
        activate_workspace=activate_workspace,
        init_qlib=init_qlib,
        calculate_dates=calculate_dates or _lazy_calculate_dates,
        prepare_cache=prepare_default_cache,
        promote_static=_default_promote,
    )


def _relative(prepared: PreparedTrainingRun, path: Path) -> str:
    return path.resolve().relative_to(prepared.ctx.root.resolve()).as_posix()


def _public_outcomes(results, published_keys=frozenset(), already_published=frozenset()):
    values = []
    for item in results:
        value = {"key": item.key, "operation": item.operation, "outcome": item.outcome}
        if item.entry is not None:
            value.update({
                "recorder_id": item.entry.recorder_id,
                "experiment_name": item.entry.experiment_name,
                "published": item.key in published_keys or item.key in already_published,
                "published_this_attempt": item.key in published_keys,
                "already_published": item.key in already_published,
            })
        if item.error_code:
            value["error_code"] = item.error_code
        if item.error_type:
            value["error_type"] = item.error_type
        values.append(value)
    return tuple(values)


class TrainingExecutionService:
    def __init__(self, hooks: TrainingExecutionHooks):
        self.hooks = hooks

    def _resolve(self, prepared):
        self.hooks.activate_workspace(str(prepared.ctx.root))
        self.hooks.init_qlib()
        run = resolve_training_run(prepared, self.hooks.calculate_dates())
        self._verify_input_baselines(prepared)
        return run

    @staticmethod
    def _verify_input_baselines(prepared):
        for relative, expected in prepared.input_fingerprints.items():
            if relative == "data/run_state.json":
                continue
            path = prepared.ctx.path(relative)
            if expected is None:
                if path.exists():
                    raise TrainingStateConflictError("a prepared input appeared before execution")
            elif not path.is_file() or fingerprint_file(path) != expected:
                raise TrainingStateConflictError("a prepared input changed before execution")

    @staticmethod
    def _state_for(
        run, outcome_map, phase="executing", receipt=None, error_code=None,
        manifest_path=None, manifest_fingerprint=None, transaction_id=None,
        publication_status=None, postprocess_status=None,
    ):
        return TrainingRunState(
            run_id=run.prepared.plan.run_id,
            family=run.prepared.options.family,
            action=run.prepared.options.action,
            plan_fingerprint=run.prepared.plan_fingerprint,
            execution_fingerprint=run.execution_fingerprint,
            resume_fingerprint=run.resume_fingerprint,
            anchor_date=run.anchor_date,
            target_keys=tuple(item.key for item in run.targets),
            outcomes={key: dict(value) for key, value in outcome_map.items()},
            phase=phase,
            publication_transaction_id=receipt.transaction_id if receipt else transaction_id,
            publication_status=receipt.status if receipt else publication_status,
            postprocess_status=postprocess_status,
            manifest_path=manifest_path,
            manifest_fingerprint=manifest_fingerprint,
            aggregate_error_code=error_code,
        )

    @staticmethod
    def _validate_resume(run, state):
        expected = (run.prepared.options.family, run.prepared.options.action, run.resume_fingerprint, run.anchor_date, tuple(item.key for item in run.targets))
        actual = (state.family, state.action, state.resume_fingerprint, state.anchor_date, state.target_keys)
        if actual != expected:
            raise TrainingStateConflictError("resume state differs from resolved execution")

    def _run_targets(self, run, state_repo, prior, state_baseline):
        if prior is None and run.prepared.options.resume:
            raise TrainingStateConflictError("resume requested but no compatible training state exists")
        outcome_map = {key: dict(value) for key, value in (prior.outcomes.items() if prior else ())}
        if prior is None:
            prepared_state = self._state_for(run, outcome_map, phase="prepared")
            state_baseline = state_repo.save(prepared_state, expected=state_baseline)
            state_baseline = state_repo.save(replace(prepared_state, phase="executing"), expected=state_baseline)
        elif prior.phase != "executing":
            state_baseline = state_repo.save(replace(
                prior, phase="executing", publication_transaction_id=None,
                publication_status=None, postprocess_status=None,
                manifest_path=None, manifest_fingerprint=None,
                aggregate_error_code=None,
            ), expected=state_baseline)

        cache = self.hooks.prepare_cache(run)
        current = TrainingRecordRepository.for_workspace(run.prepared.ctx).load().entry_map
        results = []
        for target in run.targets:
            saved = outcome_map.get(target.key, {})
            if saved.get("outcome") in ("success", "preserved") and saved.get("published") is True:
                entry = current.get(target.key)
                if entry is None or entry.recorder_id != saved.get("recorder_id"):
                    raise TrainingStateConflictError("completed resume target no longer matches current record")
                results.append(TrainingTargetResult(
                    target.key, target.operation, "preserved", entry=entry,
                    error_code="resume_already_published",
                    history_event=saved.get("history_event"),
                ))
                continue
            source = target.source_entry
            if source is not None and saved:
                expected_source = (saved.get("source_recorder_id"), saved.get("source_experiment_name"), saved.get("source_operation"))
                actual_source = (source.recorder_id, source.experiment_name, source.operation)
                if expected_source != (None, None, None) and expected_source != actual_source:
                    raise TrainingStateConflictError("resume source recorder differs from original execution")
            try:
                runner = self.hooks.run_cpcv_target if run.prepared.options.family == "cpcv" else self.hooks.run_static_target
                result = runner(TrainingTargetRequest(run, target, cache))
                if not isinstance(result, TrainingTargetResult):
                    raise TrainingRunnerContractError("runner did not return TrainingTargetResult")
                if result.key != target.key or result.operation != target.operation:
                    raise TrainingRunnerContractError("runner result differs from requested target")
            except TrainingCommandError as exc:
                result = TrainingTargetResult(target.key, target.operation, "failed", error_code=exc.code, error_type=type(exc).__name__)
            except Exception as exc:
                result = TrainingTargetResult(target.key, target.operation, "failed", error_code="target_execution_failed", error_type=type(exc).__name__)
            results.append(result)
            outcome = {
                "outcome": result.outcome, "operation": result.operation,
                "recorder_id": result.entry.recorder_id if result.entry else None,
                "error_code": result.error_code, "published": False,
            }
            if source is not None:
                outcome.update({
                    "source_recorder_id": source.recorder_id,
                    "source_experiment_name": source.experiment_name,
                    "source_operation": source.operation,
                })
            if result.history_event is not None:
                outcome["history_event"] = dict(result.history_event)
            outcome_map[target.key] = outcome
            state_baseline = state_repo.save(
                self._state_for(run, outcome_map, phase="executing"), expected=state_baseline
            )
        if tuple(item.key for item in results) != tuple(item.key for item in run.targets):
            raise TrainingRunnerContractError("target result cardinality differs from resolved run")
        state_baseline = state_repo.save(
            self._state_for(run, outcome_map, phase="targets_complete"), expected=state_baseline
        )
        return tuple(results), outcome_map, state_baseline

    def _promote(self, run, receipt):
        if run.prepared.options.family != "static" or run.prepared.options.action == "predict_only":
            return ()
        names = tuple(key.rsplit("@", 1)[0] for key in receipt.published_keys)
        if not names:
            return ()
        try:
            self.hooks.promote_static(str(run.prepared.ctx.root), names)
            return ()
        except Exception as exc:
            return ("promotion_status_update_failed:%s" % type(exc).__name__,)

    @staticmethod
    def _results_from_receipt(run, prior, receipt):
        current = TrainingRecordRepository.for_workspace(run.prepared.ctx).load().entry_map
        results = []
        published = frozenset(receipt.published_keys)
        for target in run.targets:
            saved = dict(prior.outcomes.get(target.key, {}))
            if target.key in published or saved.get("published") is True:
                entry = current.get(target.key)
                if entry is None or entry.recorder_id != saved.get("recorder_id"):
                    raise TrainingStateConflictError("committed publication no longer matches current record")
                results.append(TrainingTargetResult(
                    target.key, target.operation, "preserved", entry=entry,
                    error_code="resume_already_published",
                    history_event=saved.get("history_event"),
                ))
            elif saved.get("outcome") == "failed":
                results.append(TrainingTargetResult(
                    target.key, target.operation, "failed",
                    error_code=saved.get("error_code") or "target_execution_failed",
                ))
            else:
                raise TrainingStateConflictError("publication receipt lacks a terminal target outcome")
        return tuple(results)

    @staticmethod
    def _output_refs(receipt):
        refs = []
        for item in receipt.committed_outputs:
            refs.append(OutputRef(
                item["path"], kind="record" if item["kind"] == "record" else "report",
                overwrite=True,
            ))
        return tuple(refs)

    @staticmethod
    def _append_history(run, results, receipt):
        repository = TrainingHistoryRepository.for_run(run)
        warnings = []
        wrote = False
        by_key = {item.key: item for item in results}
        for key in receipt.published_keys:
            item = by_key.get(key)
            if item is None or item.history_event is None:
                continue
            try:
                wrote = repository.append(item.history_event) or wrote
            except Exception as exc:
                warnings.append("training_history_append_failed:%s" % type(exc).__name__)
        return tuple(warnings), wrote

    @staticmethod
    def _history_output(run):
        filename = "prediction_history.jsonl" if run.prepared.options.action == "predict_only" else "training_history.jsonl"
        return OutputRef("data/%s" % filename, kind="state", overwrite=True)

    def execute(self, prepared: PreparedTrainingRun) -> TrainingRunSummary:
        started = self.hooks.clock().replace(microsecond=0).isoformat()
        manifest_rel = None
        manifest_fp = None
        run = None
        results = ()
        published_keys = frozenset()
        already_published = frozenset()
        receipt = None
        warnings = ()
        committed = []
        state_repo = TrainingStateRepository(prepared.ctx.data_path("run_state.json"))
        lease = TrainingExecutionLease.for_workspace(prepared.ctx)
        lease.acquire(run_id=prepared.plan.run_id)
        try:
            log_file = prepared.ctx.data_path("operator_log.jsonl")
            with OperatorLog(
                prepared.plan.command, args=list(prepared.cli_args), log_file=str(log_file),
                run_id=prepared.plan.run_id, plan_fingerprint=prepared.plan_fingerprint,
            ) as oplog:
                state_baseline = None
                outcome_map = {}
                recovered_prior = None
                try:
                    run = self._resolve(prepared)
                    prior, state_baseline = state_repo.load_locked()
                    prior_phase_at_load = prior.phase if prior is not None else None
                    if prior is not None and not prepared.options.resume:
                        raise TrainingStateConflictError("existing training state requires --resume or explicit clear")
                    if prior is not None:
                        recoverable_publication = (
                            prior.publication_transaction_id
                            and prior.phase in (
                                "publication_prepared", "publication_committed", "failed", "completed"
                            )
                        )
                        if recoverable_publication:
                            expected = (
                                run.prepared.options.family, run.prepared.options.action,
                                run.anchor_date, tuple(item.key for item in run.targets),
                            )
                            actual = (prior.family, prior.action, prior.anchor_date, prior.target_keys)
                            if actual != expected:
                                raise TrainingStateConflictError("publication recovery target identity changed")
                        else:
                            self._validate_resume(run, prior)
                        run = replace(
                            run,
                            prepared=replace(run.prepared, plan_fingerprint=prior.plan_fingerprint),
                            execution_fingerprint=prior.execution_fingerprint,
                        )
                    coordinator = TrainingPublicationCoordinator.find_for_run(
                        run, clock=self.hooks.clock,
                        expected_resume_fingerprint=prior.resume_fingerprint if prior is not None else None,
                        transaction_id=prior.publication_transaction_id if prior is not None else None,
                    ) if prior is not None and prepared.options.resume else None
                    recovered = coordinator.recover() if coordinator is not None else None
                    if recovered is not None:
                        if prior.phase == "publication_prepared":
                            prior = replace(
                                prior, phase="publication_committed",
                                publication_transaction_id=recovered.transaction_id,
                                publication_status=recovered.status,
                            )
                            state_baseline = state_repo.save(prior, expected=state_baseline)
                        outcome_map = {key: dict(value) for key, value in prior.outcomes.items()}
                        for key in recovered.published_keys:
                            outcome_map[key]["published"] = True
                        already_published = frozenset(
                            key for key, value in outcome_map.items() if value.get("published") is True
                        )
                        has_failed_targets = any(
                            value.get("outcome") == "failed" for value in outcome_map.values()
                        )
                        recovered_results = self._results_from_receipt(run, prior, recovered)
                        if (
                            prior_phase_at_load in ("publication_prepared", "publication_committed")
                            and prior.postprocess_status is None
                        ):
                            history_warnings, history_written = self._append_history(
                                run, recovered_results, recovered
                            )
                            warnings += history_warnings
                            if history_written:
                                committed.append(self._history_output(run))
                            warnings += self._promote(run, recovered)
                            prior = replace(
                                prior,
                                postprocess_status="completed_with_warnings" if warnings else "completed",
                            )
                            state_baseline = state_repo.save(prior, expected=state_baseline)
                        if has_failed_targets:
                            results, outcome_map, state_baseline = self._run_targets(
                                run, state_repo, prior, state_baseline
                            )
                            coordinator = TrainingPublicationCoordinator.fresh_for_run(
                                run, clock=self.hooks.clock
                            )
                        else:
                            receipt = recovered
                            recovered_prior = prior
                            results = recovered_results
                            committed.extend(self._output_refs(receipt))
                            oplog.transaction_id = receipt.transaction_id
                    else:
                        results, outcome_map, state_baseline = self._run_targets(
                            run, state_repo, prior, state_baseline
                        )
                    coordinator = coordinator or TrainingPublicationCoordinator.fresh_for_run(
                        run, clock=self.hooks.clock
                    )
                    intent = None if receipt is not None else coordinator.prepare(results)
                    if intent is not None:
                        oplog.transaction_id = intent.transaction_id
                        prepared_state = self._state_for(
                            run, outcome_map, phase="publication_prepared",
                            transaction_id=intent.transaction_id,
                            publication_status="prepared",
                        )
                        state_baseline = state_repo.save(prepared_state, expected=state_baseline)
                        receipt = coordinator.commit(intent)
                        published_keys = frozenset(receipt.published_keys)
                        for key in published_keys:
                            outcome_map[key]["published"] = True
                        state_baseline = state_repo.save(
                            self._state_for(run, outcome_map, phase="publication_committed", receipt=receipt),
                            expected=state_baseline,
                        )
                        committed.extend(self._output_refs(receipt))
                        oplog.transaction_id = receipt.transaction_id
                        history_warnings, history_written = self._append_history(run, results, receipt)
                        warnings += history_warnings
                        if history_written:
                            committed.append(self._history_output(run))
                        warnings += self._promote(run, receipt)
                        state_baseline = state_repo.save(
                            self._state_for(
                                run, outcome_map, phase="publication_committed", receipt=receipt,
                                postprocess_status="completed_with_warnings" if warnings else "completed",
                            ),
                            expected=state_baseline,
                        )
                    failed = tuple(item for item in results if item.outcome == "failed")
                    public = _public_outcomes(results, published_keys, already_published)
                    status = "failed" if failed else "success"
                    finished = self.hooks.clock().replace(microsecond=0).isoformat()
                    result = CommandResult(
                        plan=prepared.plan, status=status, started_at=started, finished_at=finished,
                        outputs=tuple(committed), records=self._records(run, public, receipt), warnings=warnings,
                        error={"type": "TrainingExecutionError", "message": "training execution failed"} if failed else None,
                    )
                    if not prepared.options.no_manifest:
                        path = write_run_manifest(prepared.ctx, manifest_from_result(result))
                        manifest_rel = _relative(prepared, path)
                        manifest_fp = fingerprint_file(path)
                        oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
                    final_phase = "failed" if failed else "completed"
                    if recovered_prior is not None:
                        final_state = replace(
                            recovered_prior, phase=final_phase,
                            publication_transaction_id=receipt.transaction_id,
                            publication_status=receipt.status,
                            postprocess_status=recovered_prior.postprocess_status,
                            manifest_path=manifest_rel, manifest_fingerprint=manifest_fp,
                            aggregate_error_code="training_execution_error" if failed else None,
                        )
                    else:
                        final_state = self._state_for(
                            run, outcome_map, phase=final_phase, receipt=receipt,
                            error_code="training_execution_error" if failed else None,
                            manifest_path=manifest_rel, manifest_fingerprint=manifest_fp,
                            postprocess_status=(
                                "completed_with_warnings" if warnings else "completed"
                            ) if receipt is not None else None,
                        )
                    state_baseline = state_repo.save(final_state, expected=state_baseline)
                    if not failed:
                        state_repo.clear(expected=state_baseline, require_terminal=True)
                    oplog.set_result({
                        "status": status, "target_count": len(results),
                        "execution_fingerprint": run.execution_fingerprint,
                        "publication_applied": receipt is not None,
                        "warnings": list(warnings),
                    })
                    if failed:
                        raise TrainingExecutionError("one or more training targets failed")
                except Exception as exc:
                    finished = self.hooks.clock().replace(microsecond=0).isoformat()
                    public = _public_outcomes(results, published_keys, already_published)
                    if not prepared.options.no_manifest and manifest_rel is None:
                        result = CommandResult(
                            plan=prepared.plan, status="failed", started_at=started, finished_at=finished,
                            outputs=tuple(committed), records=self._records(run, public, receipt), warnings=warnings,
                            error={"type": type(exc).__name__, "message": "training execution failed"},
                        )
                        path = write_run_manifest(prepared.ctx, manifest_from_result(result))
                        manifest_rel = _relative(prepared, path)
                        oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
                    oplog.set_result({
                        "status": "failed", "error_type": type(exc).__name__,
                        "execution_fingerprint": run.execution_fingerprint if run else "",
                        "publication_applied": receipt is not None,
                    })
                    if isinstance(exc, TrainingExecutionError):
                        raise
                    if isinstance(exc, TrainingRecordConflictError):
                        raise TrainingPublicationError("training record publication conflicted") from exc
                    raise TrainingExecutionError("training execution failed") from exc
        finally:
            lease.release()
        return TrainingRunSummary(
            run_id=prepared.plan.run_id, family=prepared.options.family, action=prepared.options.action,
            target_keys=tuple(item.key for item in prepared.targets), plan_fingerprint=prepared.plan_fingerprint,
            execution_fingerprint=run.execution_fingerprint,
            outcomes=_public_outcomes(results, published_keys, already_published), manifest_path=manifest_rel,
            publication_applied=receipt is not None, warnings=warnings,
        )

    @staticmethod
    def _records(run, outcomes, receipt):
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
                "applied": receipt is not None,
                "transaction_id": receipt.transaction_id if receipt else None,
                "status": receipt.status if receipt else None,
                "recovery_action": receipt.recovery_action if receipt else None,
                "committed_outputs": list(receipt.committed_outputs) if receipt else [],
            },
        }
