"""Service-owned static/CPCV execution and recoverable publication lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple
import json
import uuid

from quantpits.runtime import (
    CommandResult, OutputRef, RunManifestConflictError, manifest_from_result,
    manifest_path, write_or_adopt_run_manifest,
)
from quantpits.training.command import PreparedTrainingRun
from quantpits.training.errors import (
    TrainingCommandError, TrainingExecutionError, TrainingPublicationError,
    TrainingEvidenceConflictError, TrainingManifestConflictError, TrainingOperatorLogError,
    TrainingRunnerContractError, TrainingStateConflictError,
)
from quantpits.training.lease import TrainingExecutionLease
from quantpits.training.history import TrainingHistoryRepository
from quantpits.training.evidence import TrainingTargetEvidence, TrainingTargetEvidenceRepository
from quantpits.training.closure import (
    TrainingClosureRepository, TrainingClosureState, closure_complete,
)
from quantpits.training.recovery import (
    TrainingRecoveryObservation, classify_training_recovery,
)
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
from quantpits.training.persistence import sha256_bytes


def _noop_cache(_run):
    return None


def _noop_promote(_workspace, _models):
    return None


def _noop_verify_reusable(_run, _evidence):
    return None


def _noop_fault(_point, _detail=None):
    return None


def _verify_reusable_recorder(run, evidence):
    """Revalidate persisted MLflow output before skipping expensive work."""
    from qlib.workflow import R
    from quantpits.training.records import build_model_record_entry

    declared = evidence.to_result().entry
    recorder = R.get_recorder(
        recorder_id=declared.recorder_id,
        experiment_name=declared.experiment_name,
    )
    observed = build_model_record_entry(
        key=declared.key,
        operation=declared.operation,
        experiment_name=declared.experiment_name,
        recorder=recorder,
        requested_anchor=declared.requested_anchor,
        dataset_test_end=declared.dataset_test_end,
        fit_start=declared.fit_start,
        fit_end=declared.fit_end,
        source_recorder_id=declared.source_recorder_id,
        source_experiment_name=declared.source_experiment_name,
        source_operation=declared.source_operation,
        workspace_root=run.prepared.ctx.root,
        config_fingerprint=declared.config_fingerprint,
    )
    fields = (
        "key", "operation", "status", "recorder_id", "experiment_name",
        "experiment_id", "artifact_path", "requested_anchor", "prediction_start",
        "prediction_end", "prediction_rows", "dataset_test_end", "source_recorder_id",
        "source_experiment_name", "source_operation", "config_fingerprint",
    )
    if any(getattr(observed, name) != getattr(declared, name) for name in fields):
        raise TrainingStateConflictError("reusable target recorder differs from durable evidence")


@dataclass(frozen=True)
class TrainingExecutionHooks:
    activate_workspace: Callable[[str], None]
    init_qlib: Callable[[], None]
    calculate_dates: Callable[[], Mapping[str, Any]]
    prepare_cache: Callable[[ResolvedTrainingRun], Any] = _noop_cache
    run_static_target: Callable[[TrainingTargetRequest], TrainingTargetResult] = run_static_target
    run_cpcv_target: Callable[[TrainingTargetRequest], TrainingTargetResult] = run_cpcv_target
    promote_static: Callable[[str, Tuple[str, ...]], None] = _noop_promote
    verify_reusable_target: Callable[[ResolvedTrainingRun, TrainingTargetEvidence], None] = _noop_verify_reusable
    fault_hook: Callable[[str, Optional[str]], None] = _noop_fault
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
        verify_reusable_target=_verify_reusable_recorder,
    )


def _relative(prepared: PreparedTrainingRun, path: Path) -> str:
    return path.resolve().relative_to(prepared.ctx.root.resolve()).as_posix()


def _public_outcomes(
    results, published_keys=frozenset(), already_published=frozenset(), durable_outcomes=None,
):
    values = []
    for item in results:
        durable = dict((durable_outcomes or {}).get(item.key, {}))
        value = {"key": item.key, "operation": item.operation, "outcome": item.outcome}
        if item.entry is not None:
            value.update({
                "recorder_id": item.entry.recorder_id,
                "experiment_name": item.entry.experiment_name,
                "published": bool(durable.get("published")) or item.key in published_keys or item.key in already_published,
                "published_this_attempt": bool(durable.get("published_this_attempt")) or item.key in published_keys,
                "already_published": bool(durable.get("already_published")) or item.key in already_published,
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
        publication_status=None, postprocess_status=None, attempt_id=None,
        receipt_fingerprint=None, closure_steps=None,
        committed_outputs=None, logical_started_at=None, logical_finished_at=None,
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
            attempt_id=attempt_id,
            receipt_fingerprint=receipt_fingerprint,
            committed_outputs=(
                tuple(receipt.committed_outputs) if receipt else tuple(committed_outputs or ())
            ),
            closure_steps=dict(closure_steps or {}),
            logical_started_at=logical_started_at,
            logical_finished_at=logical_finished_at,
        )

    @staticmethod
    def _validate_resume(run, state):
        expected = (run.prepared.options.family, run.prepared.options.action, run.resume_fingerprint, run.anchor_date, tuple(item.key for item in run.targets))
        actual = (state.family, state.action, state.resume_fingerprint, state.anchor_date, state.target_keys)
        if actual != expected:
            raise TrainingStateConflictError("resume state differs from resolved execution")

    @staticmethod
    def _receipt_fingerprint(receipt):
        return sha256_bytes((json.dumps(
            receipt.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n").encode("utf-8"))

    @staticmethod
    def _logical_timing(prepared, prior, attempt_started):
        if prior is None:
            return attempt_started, None
        if prior.logical_started_at:
            return prior.logical_started_at, prior.logical_finished_at
        candidate = (
            prepared.ctx.path(prior.manifest_path)
            if prior.manifest_path
            else manifest_path(prepared.ctx, prepared.plan.command, prepared.plan.run_id)
        )
        if candidate.is_file():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise TrainingStateConflictError("manifest_timing_evidence_invalid") from exc
            if (
                value.get("run_id") != prepared.plan.run_id
                or value.get("command") != prepared.plan.command
                or not value.get("started_at")
            ):
                raise TrainingStateConflictError("manifest_timing_identity_mismatch")
            return value["started_at"], value.get("finished_at")
        if prior.phase in ("prepared", "executing") and not prior.outcomes:
            return attempt_started, None
        raise TrainingStateConflictError(
            "logical_run_timing_missing; clear or explicitly migrate this V3 state"
        )

    def _run_targets(
        self, run, state_repo, prior, state_baseline, attempt_id, logical_started_at,
    ):
        if prior is None and run.prepared.options.resume:
            raise TrainingStateConflictError("resume requested but no compatible training state exists")
        outcome_map = {key: dict(value) for key, value in (prior.outcomes.items() if prior else ())}
        for value in outcome_map.values():
            if value.get("published") is True and prior is not None:
                value["published_this_attempt"] = False
                value["already_published"] = True
        if prior is None:
            prepared_state = self._state_for(
                run, outcome_map, phase="prepared", attempt_id=attempt_id,
                logical_started_at=logical_started_at,
            )
            state_baseline = state_repo.save(prepared_state, expected=state_baseline)
            state_baseline = state_repo.save(replace(prepared_state, phase="executing"), expected=state_baseline)
        elif prior.phase not in ("executing", "targets_complete"):
            state_baseline = state_repo.save(replace(
                prior, phase="executing", publication_transaction_id=None,
                publication_status=None, postprocess_status=None,
                manifest_path=None, manifest_fingerprint=None,
                aggregate_error_code=None, receipt_fingerprint=None,
                committed_outputs=(), closure_steps={}, logical_finished_at=None,
                logical_started_at=logical_started_at,
            ), expected=state_baseline)

        evidence_repo = TrainingTargetEvidenceRepository(run.prepared.ctx, run.prepared.plan.run_id)
        reusable = {}
        for target in run.targets:
            saved = outcome_map.get(target.key, {})
            evidence_path = evidence_repo.path_for(target.key)
            orphan_evidence = evidence_path.is_file() and not saved.get("target_evidence_fingerprint")
            if (saved.get("outcome") == "success" and not saved.get("published")) or orphan_evidence:
                relative = saved.get("target_evidence_path") or evidence_repo.relative_path(target.key)
                fingerprint = saved.get("target_evidence_fingerprint")
                if not relative or (not fingerprint and not orphan_evidence):
                    if prior is not None and prior.phase == "targets_complete":
                        raise TrainingEvidenceConflictError("completed target has no durable evidence")
                    continue
                if relative != evidence_repo.relative_path(target.key):
                    raise TrainingEvidenceConflictError("target evidence path differs from target identity")
                evidence = evidence_repo.load(target.key, expected_fingerprint=fingerprint)
                expected = (
                    run.prepared.plan.run_id, target.key, target.operation, run.anchor_date,
                    run.output_experiment_name, run.prepared.plan_fingerprint,
                    run.resume_fingerprint, run.execution_fingerprint,
                )
                actual = (
                    evidence.run_id, evidence.target_key, evidence.operation, evidence.anchor_date,
                    evidence.output_experiment_name, evidence.plan_fingerprint,
                    evidence.resume_fingerprint, evidence.execution_fingerprint,
                )
                if actual != expected:
                    raise TrainingEvidenceConflictError("training target evidence differs from resolved run")
                if saved.get("attempt_id") and evidence.attempt_id != saved.get("attempt_id"):
                    raise TrainingEvidenceConflictError("training target evidence attempt differs from state")
                try:
                    self.hooks.verify_reusable_target(run, evidence)
                except TrainingStateConflictError:
                    raise
                except Exception as exc:
                    raise TrainingEvidenceConflictError(
                        "reusable target recorder failed integrity verification"
                    ) from exc
                reusable[target.key] = evidence.to_result()
                if orphan_evidence:
                    _raw, evidence_baseline = read_with_baseline(
                        evidence_path, display_path=evidence_repo.relative_path(target.key)
                    )
                    outcome_map[target.key] = {
                        "outcome": "success", "operation": evidence.operation,
                        "recorder_id": evidence.entry.get("recorder_id"),
                        "error_code": None, "published": False,
                        "history_event": dict(evidence.history_payload) if evidence.history_payload else None,
                        "target_evidence_path": evidence_repo.relative_path(target.key),
                        "target_evidence_fingerprint": evidence_baseline.fingerprint,
                        "attempt_id": evidence.attempt_id,
                    }

        decision = classify_training_recovery(TrainingRecoveryObservation(
            phase=prior.phase if prior is not None else None,
            target_keys=tuple(item.key for item in run.targets),
            reusable_target_keys=tuple(reusable),
            failed_target_keys=tuple(
                key for key, value in outcome_map.items() if value.get("outcome") == "failed"
            ),
        ))
        if decision.action == "fail_closed":
            raise TrainingStateConflictError(decision.reason_code)
        if decision.action not in ("new_run", "run_targets", "prepare_publication"):
            raise TrainingStateConflictError("unexpected target recovery decision: %s" % decision.reason_code)
        runnable_keys = frozenset(decision.runnable_target_keys)
        runnable = tuple(target for target in run.targets if target.key in runnable_keys)
        if prior is not None and prior.phase == "targets_complete" and runnable:
            failed_state = replace(prior, phase="failed", aggregate_error_code="retry_failed_targets")
            state_baseline = state_repo.save(failed_state, expected=state_baseline)
            executing_state = replace(
                failed_state, phase="executing", attempt_id=attempt_id,
                aggregate_error_code=None,
            )
            state_baseline = state_repo.save(executing_state, expected=state_baseline)
        cache = self.hooks.prepare_cache(run) if runnable else None
        current = TrainingRecordRepository.for_workspace(run.prepared.ctx).load().entry_map
        results = []
        for target in run.targets:
            saved = outcome_map.get(target.key, {})
            if target.key in reusable:
                results.append(reusable[target.key])
                continue
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
            if result.outcome == "success":
                source_identity = None if source is None else {
                    "recorder_id": source.recorder_id,
                    "experiment_name": source.experiment_name,
                    "operation": source.operation,
                }
                evidence = TrainingTargetEvidence(
                    run_id=run.prepared.plan.run_id, attempt_id=attempt_id,
                    target_key=result.key, operation=result.operation, outcome=result.outcome,
                    entry=result.entry.to_dict(), performance=result.performance,
                    history_payload=result.history_event, source_identity=source_identity,
                    anchor_date=run.anchor_date,
                    output_experiment_name=run.output_experiment_name,
                    plan_fingerprint=run.prepared.plan_fingerprint,
                    resume_fingerprint=run.resume_fingerprint,
                    execution_fingerprint=run.execution_fingerprint,
                    observed_at=self.hooks.clock().replace(microsecond=0).isoformat(),
                )
                evidence_path, evidence_fp = evidence_repo.write(evidence)
                self.hooks.fault_hook("after_target_evidence", target.key)
                outcome.update({
                    "target_evidence_path": evidence_path,
                    "target_evidence_fingerprint": evidence_fp,
                    "attempt_id": attempt_id,
                })
            outcome_map[target.key] = outcome
            state_baseline = state_repo.save(
                self._state_for(
                    run, outcome_map, phase="executing", attempt_id=attempt_id,
                    logical_started_at=logical_started_at,
                ), expected=state_baseline
            )
            self.hooks.fault_hook("after_target_state", target.key)
        if tuple(item.key for item in results) != tuple(item.key for item in run.targets):
            raise TrainingRunnerContractError("target result cardinality differs from resolved run")
        state_baseline = state_repo.save(
            self._state_for(
                run, outcome_map, phase="targets_complete", attempt_id=attempt_id,
                logical_started_at=logical_started_at,
            ), expected=state_baseline
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
        evidence_repository = TrainingTargetEvidenceRepository(
            run.prepared.ctx, run.prepared.plan.run_id
        )
        results = []
        published = frozenset(receipt.published_keys)
        for target in run.targets:
            saved = dict(prior.outcomes.get(target.key, {}))
            if target.key in published:
                path = saved.get("target_evidence_path")
                fingerprint = saved.get("target_evidence_fingerprint")
                if path != evidence_repository.relative_path(target.key) or not fingerprint:
                    raise TrainingStateConflictError(
                        "published target lacks durable result evidence"
                    )
                results.append(
                    evidence_repository.load(
                        target.key, expected_fingerprint=fingerprint
                    ).to_result()
                )
            elif saved.get("published") is True:
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

    def _verify_receipt_target_evidence(self, run, prior, receipt):
        repository = TrainingTargetEvidenceRepository(
            run.prepared.ctx, run.prepared.plan.run_id
        )
        targets = {item.key: item for item in run.targets}
        for key in receipt.published_keys:
            saved = dict(prior.outcomes.get(key, {}))
            path = saved.get("target_evidence_path")
            fingerprint = saved.get("target_evidence_fingerprint")
            if path != repository.relative_path(key) or not fingerprint:
                    raise TrainingEvidenceConflictError(
                        "published target lacks durable evidence for recorder verification"
                )
            evidence = repository.load(key, expected_fingerprint=fingerprint)
            target = targets.get(key)
            if target is None or (
                evidence.operation != target.operation
                or evidence.anchor_date != run.anchor_date
                or evidence.output_experiment_name != run.output_experiment_name
                or evidence.plan_fingerprint != run.prepared.plan_fingerprint
                or evidence.resume_fingerprint != run.resume_fingerprint
                or evidence.execution_fingerprint != run.execution_fingerprint
            ):
                raise TrainingEvidenceConflictError(
                    "published target evidence differs from resolved run"
                )
            try:
                self.hooks.verify_reusable_target(run, evidence)
            except TrainingStateConflictError:
                raise
            except Exception as exc:
                raise TrainingEvidenceConflictError(
                    "published target recorder failed integrity verification"
                ) from exc

    @staticmethod
    def _verify_state_receipt(state, receipt):
        receipt_fingerprint = TrainingExecutionService._receipt_fingerprint(receipt)
        if state.publication_transaction_id and (
            state.publication_transaction_id != receipt.transaction_id
        ):
            raise TrainingStateConflictError("state_receipt_transaction_mismatch")
        if state.receipt_fingerprint and state.receipt_fingerprint != receipt_fingerprint:
            raise TrainingStateConflictError("state_receipt_fingerprint_mismatch")
        if state.committed_outputs and tuple(state.committed_outputs) != tuple(receipt.committed_outputs):
            raise TrainingStateConflictError("state_receipt_ledger_mismatch")

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
    def _append_history(run, results, receipt, outcomes):
        repository = TrainingHistoryRepository.for_run(run)
        warnings = []
        wrote = False
        by_key = {item.key: item for item in results}
        for key in receipt.published_keys:
            item = by_key.get(key)
            if item is None or item.history_event is None:
                continue
            try:
                entry = item.entry
                durable = dict(outcomes.get(key, {}))
                event = {
                    "schema_version": 2,
                    "event_id": sha256_bytes((
                        "%s|%s|%s|%s" % (
                            TrainingExecutionService._receipt_fingerprint(receipt),
                            item.key, entry.recorder_id if entry else "", item.operation,
                        )
                    ).encode("utf-8")),
                    "run_id": run.prepared.plan.run_id,
                    "attempt_id": durable.get("attempt_id"),
                    "transaction_id": receipt.transaction_id,
                    "receipt_fingerprint": TrainingExecutionService._receipt_fingerprint(receipt),
                    "target_key": item.key,
                    "model_name": item.key.rsplit("@", 1)[0],
                    "family": run.prepared.options.family,
                    "operation": item.operation,
                    "outcome": item.outcome,
                    "recorder_id": entry.recorder_id if entry else None,
                    "source_recorder_id": durable.get("source_recorder_id"),
                    "source_experiment_name": durable.get("source_experiment_name"),
                    "source_operation": durable.get("source_operation"),
                    "anchor_date": run.anchor_date,
                    "plan_fingerprint": run.prepared.plan_fingerprint,
                    "execution_fingerprint": run.execution_fingerprint,
                    "published_this_attempt": bool(durable.get("published_this_attempt")),
                    "already_published": bool(durable.get("already_published")),
                    "metrics": dict(item.history_event),
                }
                wrote = repository.append(event) or wrote
            except Exception as exc:
                warnings.append("training_history_append_failed:%s" % type(exc).__name__)
        return tuple(warnings), wrote

    def _close_receipt(self, run, results, receipt, outcomes):
        """Run and durably checkpoint each idempotent post-publication step."""
        repository = TrainingClosureRepository(
            run.prepared.ctx, run.prepared.plan.run_id, receipt.transaction_id
        )
        existing, _baseline = repository.load()
        receipt_fingerprint = self._receipt_fingerprint(receipt)
        if existing is not None and (
            existing.run_id != run.prepared.plan.run_id
            or existing.transaction_id != receipt.transaction_id
            or existing.receipt_fingerprint != receipt_fingerprint
        ):
            raise TrainingStateConflictError("training closure belongs to another publication")
        steps = dict(existing.steps if existing is not None else {})
        warnings = list(existing.warnings if existing is not None else ())
        step_errors = dict(existing.step_errors if existing is not None else {})
        wrote_history = False

        def persist():
            repository.save(TrainingClosureState(
                run.prepared.plan.run_id, receipt.transaction_id,
                receipt_fingerprint, dict(steps), tuple(warnings),
                tuple(receipt.committed_outputs),
                existing.manifest_fingerprint if existing is not None else None,
                existing.operator_log_fingerprint if existing is not None else None,
                dict(step_errors),
            ))

        if steps.get("receipt_verified") != "completed":
            steps["receipt_verified"] = "completed"
            persist()
        if steps.get("history_appended") != "completed":
            history_warnings, wrote_history = self._append_history(
                run, results, receipt, outcomes
            )
            warnings.extend(item for item in history_warnings if item not in warnings)
            steps["history_appended"] = "warning" if history_warnings else "completed"
            if history_warnings:
                step_errors["history_appended"] = history_warnings[0]
            else:
                step_errors.pop("history_appended", None)
            persist()
            self.hooks.fault_hook("after_history_closure", None)
        if steps.get("promotion_applied") != "completed":
            promotion_warnings = self._promote(run, receipt)
            warnings.extend(item for item in promotion_warnings if item not in warnings)
            steps["promotion_applied"] = "warning" if promotion_warnings else "completed"
            if promotion_warnings:
                step_errors["promotion_applied"] = promotion_warnings[0]
            else:
                step_errors.pop("promotion_applied", None)
            persist()
            self.hooks.fault_hook("after_promotion_closure", None)
        return tuple(warnings), wrote_history, steps

    @staticmethod
    def _history_output(run):
        filename = "prediction_history.jsonl" if run.prepared.options.action == "predict_only" else "training_history.jsonl"
        return OutputRef("data/%s" % filename, kind="state", overwrite=True)

    @staticmethod
    def _save_closure(run, receipt, steps, warnings=()):
        repository = TrainingClosureRepository(
            run.prepared.ctx, run.prepared.plan.run_id, receipt.transaction_id
        )
        existing, _baseline = repository.load()
        repository.save(TrainingClosureState(
            run.prepared.plan.run_id, receipt.transaction_id,
            TrainingExecutionService._receipt_fingerprint(receipt), dict(steps), tuple(warnings),
            tuple(receipt.committed_outputs),
            existing.manifest_fingerprint if existing is not None else None,
            existing.operator_log_fingerprint if existing is not None else None,
            dict(existing.step_errors if existing is not None else {}),
        ))
        return OutputRef(
            "data/training_runs/%s/closure-%s.json" % (
                run.prepared.plan.run_id, receipt.transaction_id,
            ),
            kind="state", overwrite=True,
        )

    @staticmethod
    def _mark_manifest_closed(run, receipt, manifest_fingerprint, warnings=()):
        repository = TrainingClosureRepository(
            run.prepared.ctx, run.prepared.plan.run_id, receipt.transaction_id
        )
        existing, _baseline = repository.load()
        receipt_fingerprint = TrainingExecutionService._receipt_fingerprint(receipt)
        if existing is not None and (
            existing.run_id != run.prepared.plan.run_id
            or existing.transaction_id != receipt.transaction_id
            or existing.receipt_fingerprint != receipt_fingerprint
        ):
            raise TrainingStateConflictError("training closure belongs to another publication")
        steps = dict(existing.steps if existing is not None else {})
        steps.setdefault("receipt_verified", "completed")
        steps["manifest_verified"] = "completed"
        repository.save(TrainingClosureState(
            run.prepared.plan.run_id, receipt.transaction_id,
            receipt_fingerprint, steps,
            tuple(existing.warnings if existing is not None else warnings),
            tuple(receipt.committed_outputs), manifest_fingerprint,
            existing.operator_log_fingerprint if existing is not None else None,
            dict(existing.step_errors if existing is not None else {}),
        ))
        return steps

    @staticmethod
    def _mark_publication_bound(run, receipt, warnings=()):
        repository = TrainingClosureRepository(
            run.prepared.ctx, run.prepared.plan.run_id, receipt.transaction_id
        )
        existing, _baseline = repository.load()
        receipt_fingerprint = TrainingExecutionService._receipt_fingerprint(receipt)
        if existing is None:
            existing = TrainingClosureState(
                run.prepared.plan.run_id, receipt.transaction_id, receipt_fingerprint,
                {"receipt_verified": "completed"}, tuple(warnings),
                tuple(receipt.committed_outputs),
            )
        steps = dict(existing.steps)
        steps["receipt_verified"] = "completed"
        steps["state_publication_bound"] = "completed"
        repository.save(TrainingClosureState(
            existing.run_id, existing.transaction_id, existing.receipt_fingerprint,
            steps, existing.warnings, tuple(receipt.committed_outputs),
            existing.manifest_fingerprint, existing.operator_log_fingerprint,
            existing.step_errors,
        ))
        return steps

    @staticmethod
    def _mark_terminal_closed(run, receipt, warnings=()):
        repository = TrainingClosureRepository(
            run.prepared.ctx, run.prepared.plan.run_id, receipt.transaction_id
        )
        existing, _baseline = repository.load()
        if existing is None:
            raise TrainingStateConflictError("training closure is missing after terminal state save")
        steps = dict(existing.steps)
        steps["terminal_state_saved"] = "completed"
        repository.save(TrainingClosureState(
            existing.run_id, existing.transaction_id, existing.receipt_fingerprint,
            steps, tuple(existing.warnings or warnings), existing.committed_outputs,
            existing.manifest_fingerprint, existing.operator_log_fingerprint,
            existing.step_errors,
        ))
        return steps

    @staticmethod
    def _mark_closure_step(run, receipt, step, warnings=(), evidence_fingerprint=None):
        repository = TrainingClosureRepository(
            run.prepared.ctx, run.prepared.plan.run_id, receipt.transaction_id
        )
        existing, _baseline = repository.load()
        if existing is None:
            raise TrainingStateConflictError("training closure is missing")
        steps = dict(existing.steps)
        steps[step] = "completed"
        operator_fingerprint = existing.operator_log_fingerprint
        if step == "operator_log_linked":
            if not evidence_fingerprint:
                raise TrainingStateConflictError("operator-log closure lacks durable evidence")
            operator_fingerprint = evidence_fingerprint
        repository.save(TrainingClosureState(
            existing.run_id, existing.transaction_id, existing.receipt_fingerprint,
            steps, tuple(existing.warnings or warnings), existing.committed_outputs,
            existing.manifest_fingerprint, operator_fingerprint, existing.step_errors,
        ))
        return steps

    @staticmethod
    def _closure_complete(steps, *, no_manifest, include_terminal=True):
        return closure_complete(
            steps, no_manifest=no_manifest, include_terminal=include_terminal
        )

    def execute(self, prepared: PreparedTrainingRun) -> TrainingRunSummary:
        attempt_started = self.hooks.clock().replace(microsecond=0).isoformat()
        logical_started = attempt_started
        logical_finished = None
        manifest_rel = None
        manifest_fp = None
        run = None
        results = ()
        published_keys = frozenset()
        already_published = frozenset()
        receipt = None
        warnings = ()
        committed = []
        manifest_closure_steps = {}
        active_closure_steps = {}
        attempt_id = "attempt-%s" % uuid.uuid4().hex[:12]
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
                    if prepared.resume_state is not None:
                        if (
                            prior is None
                            or prior.run_id != prepared.resume_state.run_id
                            or state_baseline.fingerprint != prepared.resume_state.state_fingerprint
                        ):
                            raise TrainingStateConflictError("resume state changed after command planning")
                    if prior is not None and not prepared.options.resume:
                        raise TrainingStateConflictError("existing training state requires --resume or explicit clear")
                    logical_started, logical_finished = self._logical_timing(
                        prepared, prior, attempt_started
                    )
                    if prior is not None and not prior.logical_started_at:
                        prior = replace(
                            prior, logical_started_at=logical_started,
                            logical_finished_at=logical_finished,
                        )
                        state_baseline = state_repo.save(prior, expected=state_baseline)
                    if prior is not None:
                        recoverable_publication = (
                            prior.publication_transaction_id
                            and prior.phase in (
                                "publication_prepared", "publication_committed", "closing", "failed", "completed"
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
                        fault_hook=self.hooks.fault_hook,
                    ) if prior is not None and prepared.options.resume else None
                    publication_observation = (
                        coordinator.inspect_recovery() if coordinator is not None else None
                    )
                    publication_decision = None
                    if publication_observation is not None:
                        publication_decision = classify_training_recovery(
                            TrainingRecoveryObservation(
                                phase=prior.phase,
                                target_keys=prior.target_keys,
                                reusable_target_keys=tuple(
                                    key for key, value in prior.outcomes.items()
                                    if value.get("outcome") in ("success", "preserved")
                                ),
                                failed_target_keys=tuple(
                                    key for key, value in prior.outcomes.items()
                                    if value.get("outcome") == "failed"
                                ),
                                intent_present=publication_observation["intent_present"],
                                receipt_present=publication_observation["receipt_present"],
                                member_states=publication_observation["member_states"],
                                publication_unknown=(
                                    "unknown" in publication_observation["member_states"]
                                ),
                                closure_pending=tuple(
                                    key for key, value in prior.closure_steps.items()
                                    if value != "completed"
                                ),
                                transaction_id=publication_observation["transaction_id"],
                            )
                        )
                        if publication_decision.action == "fail_closed":
                            raise TrainingStateConflictError(publication_decision.reason_code)
                    recovered = (
                        coordinator.recover()
                        if publication_decision is not None
                        and publication_decision.action in ("recover_publication", "close_postprocess")
                        else None
                    )
                    if recovered is not None:
                        if prior.phase in ("targets_complete", "failed"):
                            prior = replace(
                                prior, phase="publication_prepared",
                                publication_transaction_id=recovered.transaction_id,
                                publication_status="recovering",
                            )
                            state_baseline = state_repo.save(prior, expected=state_baseline)
                        if prior.phase == "publication_prepared":
                            prior = replace(
                                prior, phase="publication_committed",
                                publication_transaction_id=recovered.transaction_id,
                                publication_status=recovered.status,
                                receipt_fingerprint=self._receipt_fingerprint(recovered),
                                committed_outputs=tuple(recovered.committed_outputs),
                                recovery_backfill_required=False,
                            )
                            state_baseline = state_repo.save(prior, expected=state_baseline)
                        self._verify_state_receipt(prior, recovered)
                        bound_steps = self._mark_publication_bound(run, recovered, warnings)
                        recovered_manifest_path = prior.manifest_path
                        recovered_manifest_fp = prior.manifest_fingerprint
                        if bound_steps.get("manifest_verified") == "completed":
                            closure_state, _closure_baseline = TrainingClosureRepository(
                                run.prepared.ctx, run.prepared.plan.run_id,
                                recovered.transaction_id,
                            ).load()
                            candidate = manifest_path(
                                run.prepared.ctx, prepared.plan.command,
                                prepared.plan.run_id,
                            )
                            if (
                                closure_state is None
                                or not closure_state.manifest_fingerprint
                                or not candidate.is_file()
                                or fingerprint_file(candidate)
                                != closure_state.manifest_fingerprint
                            ):
                                raise TrainingStateConflictError(
                                    "manifest_closure_evidence_mismatch"
                                )
                            recovered_manifest_path = _relative(prepared, candidate)
                            recovered_manifest_fp = closure_state.manifest_fingerprint
                        if (
                            prior.recovery_backfill_required
                            or prior.closure_steps != bound_steps
                            or prior.manifest_path != recovered_manifest_path
                            or prior.manifest_fingerprint != recovered_manifest_fp
                        ):
                            prior = replace(
                                prior, closure_steps=bound_steps,
                                manifest_path=recovered_manifest_path,
                                manifest_fingerprint=recovered_manifest_fp,
                                publication_transaction_id=recovered.transaction_id,
                                publication_status=recovered.status,
                                receipt_fingerprint=self._receipt_fingerprint(recovered),
                                committed_outputs=tuple(recovered.committed_outputs),
                                recovery_backfill_required=False,
                            )
                            state_baseline = state_repo.save(prior, expected=state_baseline)
                        outcome_map = {key: dict(value) for key, value in prior.outcomes.items()}
                        for key in recovered.published_keys:
                            outcome_map[key]["published"] = True
                            if not (
                                outcome_map[key].get("published_this_attempt")
                                or outcome_map[key].get("already_published")
                            ):
                                outcome_map[key].update({
                                    "published_this_attempt": False,
                                    "already_published": True,
                                })
                        already_published = frozenset(
                            key for key, value in outcome_map.items()
                            if value.get("already_published") is True
                        )
                        has_failed_targets = any(
                            value.get("outcome") == "failed" for value in outcome_map.values()
                        )
                        self._verify_receipt_target_evidence(run, prior, recovered)
                        recovered_results = self._results_from_receipt(run, prior, recovered)
                        # Receipt identity, not the previously observed phase,
                        # authorizes the same retryable closure path.
                        history_warnings, history_written, closure_steps = self._close_receipt(
                            run, recovered_results, recovered, outcome_map
                        )
                        active_closure_steps = dict(closure_steps)
                        warnings += history_warnings
                        if history_written:
                            committed.append(self._history_output(run))
                        committed.append(self._save_closure(run, recovered, closure_steps, warnings))
                        prior = replace(
                            prior, phase="completed" if prior.phase == "completed" else "closing",
                            outcomes=outcome_map,
                            postprocess_status="completed_with_warnings" if warnings else "completed",
                            closure_steps=closure_steps,
                        )
                        state_baseline = state_repo.save(prior, expected=state_baseline)
                        if has_failed_targets:
                            if prior.phase != "failed":
                                prior = replace(
                                    prior, phase="failed",
                                    aggregate_error_code="retry_failed_targets",
                                )
                                state_baseline = state_repo.save(prior, expected=state_baseline)
                            results, outcome_map, state_baseline = self._run_targets(
                                run, state_repo, prior, state_baseline, attempt_id,
                                logical_started,
                            )
                            coordinator = TrainingPublicationCoordinator.fresh_for_run(
                                run, clock=self.hooks.clock, fault_hook=self.hooks.fault_hook
                            )
                        else:
                            receipt = recovered
                            recovered_prior = prior
                            results = recovered_results
                            committed.extend(self._output_refs(receipt))
                            oplog.transaction_id = receipt.transaction_id
                    else:
                        results, outcome_map, state_baseline = self._run_targets(
                            run, state_repo, prior, state_baseline, attempt_id,
                            logical_started,
                        )
                    coordinator = coordinator or TrainingPublicationCoordinator.fresh_for_run(
                        run, clock=self.hooks.clock, fault_hook=self.hooks.fault_hook
                    )
                    intent = None if receipt is not None else coordinator.prepare(results)
                    if intent is not None:
                        oplog.transaction_id = intent.transaction_id
                        prepared_state = self._state_for(
                            run, outcome_map, phase="publication_prepared",
                            transaction_id=intent.transaction_id,
                            publication_status="prepared",
                            attempt_id=attempt_id,
                            logical_started_at=logical_started,
                        )
                        state_baseline = state_repo.save(prepared_state, expected=state_baseline)
                        receipt = coordinator.commit(intent)
                        coordinator.verify_receipt(intent, receipt)
                        published_keys = frozenset(receipt.published_keys)
                        for key in published_keys:
                            outcome_map[key].update({
                                "published": True,
                                "published_this_attempt": True,
                                "already_published": False,
                            })
                        state_baseline = state_repo.save(
                            self._state_for(
                                run, outcome_map, phase="publication_committed", receipt=receipt,
                                attempt_id=attempt_id,
                                receipt_fingerprint=self._receipt_fingerprint(receipt),
                                logical_started_at=logical_started,
                            ),
                            expected=state_baseline,
                        )
                        bound_steps = self._mark_publication_bound(run, receipt, warnings)
                        state_baseline = state_repo.save(
                            self._state_for(
                                run, outcome_map, phase="publication_committed", receipt=receipt,
                                attempt_id=attempt_id,
                                receipt_fingerprint=self._receipt_fingerprint(receipt),
                                closure_steps=bound_steps,
                                logical_started_at=logical_started,
                            ),
                            expected=state_baseline,
                        )
                        committed.extend(self._output_refs(receipt))
                        oplog.transaction_id = receipt.transaction_id
                        closure_warnings, history_written, closure_steps = self._close_receipt(
                            run, results, receipt, outcome_map
                        )
                        active_closure_steps = dict(closure_steps)
                        warnings += closure_warnings
                        if history_written:
                            committed.append(self._history_output(run))
                        committed.append(self._save_closure(run, receipt, closure_steps, warnings))
                        state_baseline = state_repo.save(
                            self._state_for(
                                run, outcome_map, phase="closing", receipt=receipt,
                                postprocess_status="completed_with_warnings" if warnings else "completed",
                                attempt_id=attempt_id,
                                receipt_fingerprint=self._receipt_fingerprint(receipt),
                                closure_steps=closure_steps,
                                logical_started_at=logical_started,
                            ),
                            expected=state_baseline,
                        )
                    failed = tuple(item for item in results if item.outcome == "failed")
                    public = _public_outcomes(
                        results, published_keys, already_published, outcome_map
                    )
                    status = "failed" if failed else "success"
                    finished = self.hooks.clock().replace(microsecond=0).isoformat()
                    if logical_finished is None and not failed:
                        logical_finished = finished
                        timing_state, observed_timing = state_repo.load_locked()
                        if (
                            timing_state is None
                            or observed_timing.fingerprint != state_baseline.fingerprint
                        ):
                            raise TrainingStateConflictError(
                                "training state changed before logical finish binding"
                            )
                        state_baseline = state_repo.save(
                            replace(timing_state, logical_finished_at=logical_finished),
                            expected=state_baseline,
                        )
                    if receipt is not None:
                        committed = list(self._output_refs(receipt))
                        if (
                            active_closure_steps.get("history_appended") == "completed"
                            and any(item.history_event is not None for item in results)
                        ):
                            committed.append(self._history_output(run))
                        committed.append(OutputRef(
                            "data/training_runs/%s/closure-%s.json" % (
                                run.prepared.plan.run_id, receipt.transaction_id,
                            ),
                            kind="state", overwrite=True,
                        ))
                    result = CommandResult(
                        plan=prepared.plan, status=status, started_at=logical_started,
                        finished_at=logical_finished or finished,
                        outputs=tuple(committed), records=self._records(run, public, receipt), warnings=warnings,
                        error={"type": "TrainingExecutionError", "message": "training execution failed"} if failed else None,
                    )
                    premanifest_ready = receipt is None or all(
                        active_closure_steps.get(name) == "completed"
                        for name in (
                            "receipt_verified", "state_publication_bound",
                            "history_appended", "promotion_applied",
                        )
                    )
                    if premanifest_ready and not prepared.options.no_manifest:
                        try:
                            path, manifest_fp = write_or_adopt_run_manifest(
                                prepared.ctx, manifest_from_result(result),
                                allow_failed_supersession=True,
                                allow_verified_success_adoption=True,
                                expected_receipt_ledger=(
                                    receipt.committed_outputs if receipt is not None else None
                                ),
                            )
                        except RunManifestConflictError as exc:
                            raise TrainingManifestConflictError("manifest_conflict") from exc
                        manifest_rel = _relative(prepared, path)
                        oplog.set_run_manifest(
                            run_id=prepared.plan.run_id, manifest_path=manifest_rel
                        )
                        if receipt is not None:
                            manifest_closure_steps = self._mark_manifest_closed(
                                run, receipt, manifest_fp, warnings
                            )
                        self.hooks.fault_hook("after_manifest_closure", manifest_rel)
                    elif prepared.options.no_manifest:
                        oplog.set_run_manifest(
                            run_id=prepared.plan.run_id, manifest_path=None
                        )
                    operator_ready = receipt is not None and premanifest_ready and (
                        prepared.options.no_manifest
                        or manifest_closure_steps.get("manifest_verified") == "completed"
                    )
                    if operator_ready:
                        oplog.set_plan_fingerprint(run.prepared.plan_fingerprint)
                        oplog.set_result({
                            "status": status, "target_count": len(results),
                            "execution_fingerprint": run.execution_fingerprint,
                            "publication_applied": True,
                            "warnings": list(warnings),
                        })
                        try:
                            operator_log_fingerprint = oplog.commit()
                        except Exception as exc:
                            raise TrainingOperatorLogError(
                                "operator_log_write_failed"
                            ) from exc
                        self.hooks.fault_hook(
                            "after_operator_log_append", operator_log_fingerprint
                        )
                        manifest_closure_steps = self._mark_closure_step(
                            run, receipt, "operator_log_linked", warnings,
                            evidence_fingerprint=operator_log_fingerprint,
                        )
                    closure_steps_for_final = dict(
                        manifest_closure_steps
                        or active_closure_steps
                        or (recovered_prior.closure_steps if recovered_prior is not None else {})
                        or (prior.closure_steps if prior is not None else {})
                    )
                    closure_ready = receipt is None or self._closure_complete(
                        closure_steps_for_final, no_manifest=prepared.options.no_manifest,
                        include_terminal=False,
                    )
                    final_phase = "failed" if failed else ("completed" if closure_ready else "closing")
                    if recovered_prior is not None:
                        final_state = replace(
                            recovered_prior, phase=final_phase,
                            outcomes=outcome_map,
                            publication_transaction_id=receipt.transaction_id,
                            publication_status=receipt.status,
                            postprocess_status="completed_with_warnings" if not closure_ready else "completed",
                            manifest_path=manifest_rel, manifest_fingerprint=manifest_fp,
                            aggregate_error_code="training_execution_error" if failed else None,
                            closure_steps=closure_steps_for_final,
                            logical_started_at=logical_started,
                            logical_finished_at=logical_finished,
                        )
                    else:
                        final_state = self._state_for(
                            run, outcome_map, phase=final_phase, receipt=receipt,
                            error_code="training_execution_error" if failed else None,
                            manifest_path=manifest_rel, manifest_fingerprint=manifest_fp,
                            postprocess_status=(
                                "completed_with_warnings" if warnings else "completed"
                            ) if receipt is not None else None,
                            attempt_id=attempt_id,
                            receipt_fingerprint=self._receipt_fingerprint(receipt) if receipt else None,
                            closure_steps=closure_steps_for_final if receipt else {},
                            logical_started_at=logical_started,
                            logical_finished_at=logical_finished,
                        )
                    state_baseline = state_repo.save(final_state, expected=state_baseline)
                    self.hooks.fault_hook("after_terminal_state", final_phase)
                    if receipt is not None and final_phase in ("failed", "completed"):
                        terminal_steps = self._mark_terminal_closed(run, receipt, warnings)
                        final_state = replace(final_state, closure_steps=terminal_steps)
                        state_baseline = state_repo.save(final_state, expected=state_baseline)
                        closure_ready = self._closure_complete(
                            terminal_steps, no_manifest=prepared.options.no_manifest
                        )
                    if not failed and final_phase == "completed" and closure_ready:
                        state_repo.clear(expected=state_baseline, require_terminal=True)
                        try:
                            self._mark_closure_step(run, receipt, "state_cleared", warnings)
                        except Exception:
                            # Absence of the terminal state is authoritative;
                            # the sidecar marker is diagnostic only.
                            pass
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
                    public = _public_outcomes(
                        results, published_keys, already_published, outcome_map
                    )
                    # Preserve the last recoverable lifecycle boundary.  A
                    # failure to save this diagnostic state must not hide the
                    # original exception or overwrite a concurrent owner.
                    if run is not None and state_baseline is not None:
                        try:
                            current_state, observed_state = state_repo.load_locked()
                            if (
                                current_state is not None
                                and observed_state.fingerprint == state_baseline.fingerprint
                                and current_state.phase not in ("failed", "completed")
                            ):
                                failure_phase = (
                                    "closing"
                                    if receipt is not None and current_state.phase == "closing"
                                    else "failed"
                                )
                                state_baseline = state_repo.save(replace(
                                    current_state, phase=failure_phase,
                                    aggregate_error_code=getattr(exc, "code", "training_execution_failed"),
                                    attempt_id=attempt_id,
                                ), expected=state_baseline)
                        except Exception:
                            pass
                    if not prepared.options.no_manifest and manifest_rel is None:
                        failure_records = self._records(run, public, receipt)
                        failure_records.setdefault(
                            "plan_fingerprint", prepared.plan_fingerprint
                        )
                        result = CommandResult(
                            plan=prepared.plan, status="failed", started_at=logical_started,
                            finished_at=logical_finished or finished,
                            outputs=tuple(committed), records=failure_records, warnings=warnings,
                            error={"type": type(exc).__name__, "message": "training execution failed"},
                        )
                        try:
                            path, _failure_manifest_fp = write_or_adopt_run_manifest(
                                prepared.ctx, manifest_from_result(result),
                                allow_failed_supersession=True,
                            )
                        except RunManifestConflictError:
                            path = manifest_path(
                                prepared.ctx, prepared.plan.command,
                                prepared.plan.run_id,
                            )
                            if not path.is_file():
                                raise
                        manifest_rel = _relative(prepared, path)
                        oplog.set_run_manifest(
                            run_id=prepared.plan.run_id, manifest_path=manifest_rel
                        )
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
            outcomes=_public_outcomes(
                results, published_keys, already_published, outcome_map
            ), manifest_path=manifest_rel,
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
