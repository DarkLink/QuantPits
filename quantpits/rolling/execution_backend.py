"""Exact-unit kernel and explicit recorder/evidence backend boundary."""

from __future__ import annotations

import json
import hashlib
import os
import stat
from pathlib import Path

from quantpits.rolling.errors import (
    RollingExecutionContractError,
    RollingExecutionPreflightError,
)
from quantpits.rolling.evidence import (
    RollingEvidenceSetInspection,
    RollingUnitEvidenceRequest,
)
from quantpits.rolling.execution import (
    RollingExecutionBatchResult,
    RollingExecutionScope,
    RollingExecutionUnitResult,
    RollingUnitRunnerObservation,
    _BATCH_TOKEN,
    _RESULT_TOKEN,
    preflight_rolling_execution,
)
from quantpits.rolling.repository import RollingStateRepository
from quantpits.rolling.state import RollingStateUnitClaim, RollingStateV2Snapshot
from quantpits.rolling.identity import workspace_fingerprint
from quantpits.rolling.windows import observe_rolling_business_sessions


def _attempt(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise RollingExecutionContractError("attempt_id must be a non-empty trimmed string")
    return value


def _extensions(scope):
    return json.dumps({
        "execution_protocol_version": scope.execution_protocol_version,
        "scope_fingerprint": scope.scope_fingerprint,
    }, sort_keys=True, separators=(",", ":"))


def _snapshot(scope, phase, attempt_id, units):
    return RollingStateV2Snapshot(
        workspace_fingerprint=scope.run_identity.workspace_fingerprint,
        run_id=scope.run_identity.fingerprint,
        family=scope.run_identity.family,
        action=scope.run_identity.action,
        plan_fingerprint=scope.run_identity.plan_fingerprint,
        execution_fingerprint=scope.run_identity.fingerprint,
        config_fingerprint=scope.run_identity.config_fingerprint,
        anchor_date=scope.run_identity.anchor_date,
        target_keys=scope.run_identity.target_keys,
        window_keys=scope.run_identity.window_keys,
        attempt_id=attempt_id,
        phase=phase,
        units=tuple(units),
        _extensions_json=_extensions(scope),
    )


def _claim(unit, status, record_id=None, evidence_id=None, extensions=None):
    return RollingStateUnitClaim(
        unit.unit_key[0], unit.unit_key[1], status,
        record_id=record_id, evidence_id=evidence_id,
        _extensions_json=(
            json.dumps(extensions, sort_keys=True, separators=(",", ":"))
            if extensions is not None else None
        ),
    )


def _source_extensions(request, attempt_id, extra=None):
    values = {"attempt_id": attempt_id}
    if request is not None:
        values.update({
            "source_manifest_fingerprint": request.source_manifest_fingerprint,
            "experiment_name": request.experiment_name,
            "experiment_id": request.experiment_id,
            "recorder_id": request.recorder_id,
            "source_operation": request.source_operation,
            "artifacts": [item.to_fingerprint_dict() for item in request.artifacts],
        })
    values.update(extra or {})
    return values


def _batch(scope, results):
    results = tuple(results)
    status = (
        "interrupted" if any(item.status == "interrupted" for item in results)
        else "failed" if any(item.status == "failed" for item in results)
        else "blocked" if not results or any(item.status == "blocked" for item in results)
        else "success"
    )
    return RollingExecutionBatchResult(
        scope.requested_unit_keys, results, status,
        "rolling_execution_batch_%s" % status,
        _authority=_BATCH_TOKEN,
    )


def _blocked(scope, reason_code, attempt_id=None):
    return _batch(scope, tuple(
        RollingExecutionUnitResult(
            unit.unit_key, unit.position, "blocked", False,
            attempt_id=attempt_id, reason_code=reason_code,
        ) for unit in scope.units
    ))


def _invalidate_terminal_batch(scope, batch, reason_code):
    """Fail closed when a post-unit guard cannot establish batch durability."""

    invalidated = []
    for item in batch.unit_results:
        if item.status == "executed_success":
            status, did_execute = "failed", True
        elif item.status == "reused_success":
            status, did_execute = "blocked", False
        else:
            status, did_execute = item.status, item.did_execute
        invalidated.append(RollingExecutionUnitResult(
            item.unit_key, item.position, status, did_execute,
            attempt_id=item.attempt_id, reason_code=reason_code,
        ))
    return _batch(scope, tuple(invalidated))


class RollingExecutionKernel:
    """Sole owner of Phase 34 terminal unit and batch outcomes."""

    def __init__(self, repository, backend, runner):
        if not isinstance(repository, RollingStateRepository):
            raise RollingExecutionContractError("kernel requires RollingStateRepository")
        for owner_name, owner in (("backend", backend), ("runner", runner)):
            context = getattr(owner, "context", repository.context)
            if (
                context.root != repository.context.root
                or context.data_dir != repository.context.data_dir
            ):
                raise RollingExecutionContractError(
                    "%s workspace context is foreign" % owner_name
                )
        self.repository = repository
        self.backend = backend
        self.runner = runner

    def _commit(self, state, baseline, evidence=None):
        receipt = (
            self.repository.commit(state, baseline)
            if evidence is None
            else self.repository.commit_evidence_authorized(state, baseline, evidence)
        )
        if receipt.status not in ("committed", "unchanged"):
            raise RollingExecutionPreflightError("state CAS did not commit: %s" % receipt.status)
        return receipt.cas_baseline

    def _lease(self):
        from quantpits.training.lease import TrainingExecutionLease

        root = self.repository.context.root.resolve(strict=True)
        data = self.repository.context.data_dir.resolve(strict=True)
        try:
            data.relative_to(root)
        except ValueError:
            raise RollingExecutionPreflightError("training lease parent escapes workspace")
        locks = self.repository.context.data_dir / "locks"
        if locks.is_symlink():
            raise RollingExecutionPreflightError("training lease parent is a symlink")
        if locks.exists():
            try:
                locks.resolve(strict=True).relative_to(root)
            except (OSError, ValueError):
                raise RollingExecutionPreflightError("training lease parent escapes workspace")
        return TrainingExecutionLease.for_workspace(self.repository.context)

    def _verify_lease(self, lease, expected=None):
        root = self.repository.context.root.resolve(strict=True)
        data = self.repository.context.data_dir.resolve(strict=True)
        parent = lease.path.parent
        try:
            root_node = os.stat(str(root), follow_symlinks=False)
            data_node = os.stat(str(data), follow_symlinks=False)
            parent_node = os.lstat(str(parent))
            lock_node = os.lstat(str(lease.path))
            data.relative_to(root)
            parent.resolve(strict=True).relative_to(root)
            opened = os.fstat(lease._handle.fileno())
        except (AttributeError, OSError, ValueError) as exc:
            raise RollingExecutionPreflightError("training lease identity is unavailable") from exc
        if (
            stat.S_ISLNK(parent_node.st_mode)
            or not stat.S_ISDIR(parent_node.st_mode)
            or stat.S_ISLNK(lock_node.st_mode)
            or not stat.S_ISREG(lock_node.st_mode)
            or (lock_node.st_dev, lock_node.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise RollingExecutionPreflightError("training lease identity changed")
        observed = (
            (root_node.st_dev, root_node.st_ino),
            (data_node.st_dev, data_node.st_ino),
            (parent_node.st_dev, parent_node.st_ino),
            (lock_node.st_dev, lock_node.st_ino),
            (opened.st_dev, opened.st_ino),
        )
        if expected is not None and observed != expected:
            raise RollingExecutionPreflightError("training lease ancestry changed")
        return observed

    def _run_with_lease(self, scope, attempt_id, operation):
        lease = None
        try:
            lease = self._lease()
            lease.acquire(
                run_id="phase34-%s-%s" % (
                    scope.run_identity.fingerprint[:16], attempt_id,
                ),
            )
            lease_identity = self._verify_lease(lease)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            if lease is not None:
                lease.release()
            raise
        except Exception:
            if lease is not None:
                lease.release()
            return _blocked(scope, "rolling_execution_lease_blocked", attempt_id)
        try:
            result = operation(scope, attempt_id)
            try:
                self._verify_lease(lease, lease_identity)
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                return _invalidate_terminal_batch(
                    scope, result, "rolling_execution_lease_postcondition_uncertain",
                )
            return result
        finally:
            lease.release()

    @staticmethod
    def _validate_scope(scope):
        if not isinstance(scope, RollingExecutionScope):
            raise RollingExecutionContractError("kernel requires RollingExecutionScope")

    def _validate_tracking_identity(self, scope):
        try:
            observed = self.backend.tracking_identity()
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            raise RollingExecutionPreflightError(
                "tracking backend identity is unavailable"
            ) from exc
        expected_workspace = workspace_fingerprint(self.repository.context.root)
        expected_backend = getattr(self.backend, "backend_fingerprint", None)
        if (
            not isinstance(observed, dict)
            or observed.get("workspace_fingerprint") != expected_workspace
            or observed.get("workspace_fingerprint")
            != scope.run_identity.workspace_fingerprint
            or observed.get("backend_fingerprint") != expected_backend
            or observed.get("present") is not True
            or observed.get("contained") is not True
            or observed.get("foreign") is not False
        ):
            raise RollingExecutionPreflightError(
                "tracking backend is absent, foreign, or not physically contained"
            )

    def _validate_scope_sources(self, scope):
        root = Path(self.repository.context.root).resolve(strict=True)
        for target in scope.targets:
            path = (root / target.workflow_relative_path).absolute()
            try:
                physical = path.resolve(strict=True)
                physical.relative_to(root)
                data = physical.read_bytes()
            except (OSError, ValueError) as exc:
                raise RollingExecutionPreflightError(
                    "workflow source is missing or physically foreign"
                ) from exc
            if hashlib.sha256(data).hexdigest() != target.workflow_fingerprint:
                raise RollingExecutionPreflightError("workflow source fingerprint changed")
        provider = getattr(self.backend, "calendar_sessions", None)
        if not callable(provider):
            raise RollingExecutionPreflightError("business calendar provider is unavailable")
        observed = observe_rolling_business_sessions(
            tuple(item.window for item in scope.windows), provider,
        )
        if tuple(item.expected_sessions for item in observed) != tuple(
            item.expected_sessions for item in scope.windows
        ):
            raise RollingExecutionPreflightError("business session inventory changed")

    @staticmethod
    def _evidence_for_requests(backend, scope, requests):
        if not isinstance(requests, tuple):
            raise RollingExecutionContractError("evidence requests must be an ordered tuple")
        evidence = backend.inspect(scope, requests)
        if not isinstance(evidence, RollingEvidenceSetInspection):
            raise RollingExecutionContractError("backend returned foreign evidence")
        if evidence.requested_unit_keys != tuple(item.unit_key for item in requests):
            raise RollingExecutionContractError("evidence changed requested identity/order")
        return evidence

    @staticmethod
    def _success_evidence(backend, scope, requests_by_key, claims):
        keys = tuple(
            (claim.target_key, claim.window_key)
            for claim in claims if claim.status == "success"
        )
        requests = tuple(requests_by_key[key] for key in keys)
        if not requests:
            return None
        evidence = RollingExecutionKernel._evidence_for_requests(backend, scope, requests)
        if evidence.status != "all_valid":
            raise RollingExecutionPreflightError("successful state members lost exact evidence")
        return evidence

    def execute(self, scope, attempt_id):
        self._validate_scope(scope)
        attempt_id = _attempt(attempt_id)
        preflight = preflight_rolling_execution(scope)
        if not preflight.preflight_allowed:
            return _blocked(scope, "rolling_execution_capability_blocked")
        try:
            self._validate_tracking_identity(scope)
            self._validate_scope_sources(scope)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(scope, "rolling_execution_tracking_backend_blocked")
        return self._run_with_lease(scope, attempt_id, self._execute_locked)

    def _execute_locked(self, scope, attempt_id):
        try:
            self._validate_tracking_identity(scope)
            self._validate_scope_sources(scope)
            view = self.repository.inspect_readonly()
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(
                scope, "rolling_execution_state_observation_blocked", attempt_id,
            )
        if view.inspection.classification != "missing":
            return _blocked(
                scope, "rolling_execution_fresh_state_not_missing", attempt_id,
            )
        try:
            baseline = self._commit(_snapshot(scope, "prepared", None, ()), view.baseline)
            claims = [_claim(unit, "pending") for unit in scope.units]
            baseline = self._commit(_snapshot(scope, "executing", attempt_id, claims), baseline)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(
                scope, "rolling_execution_initial_state_blocked", attempt_id,
            )
        results = []
        requests_by_key = {}
        state_uncertain = False
        for unit in scope.units:
            if state_uncertain:
                results.append(RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "blocked", False,
                    attempt_id=attempt_id,
                    reason_code="rolling_execution_state_uncertain",
                ))
                continue
            claims[unit.position] = _claim(unit, "running")
            try:
                evidence = self._success_evidence(self.backend, scope, requests_by_key, claims)
                baseline = self._commit(
                    _snapshot(scope, "executing", attempt_id, claims), baseline, evidence,
                )
            except Exception:
                state_uncertain = True
                results.append(RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "failed", False,
                    attempt_id=attempt_id,
                    reason_code="rolling_execution_running_state_failed",
                ))
                continue
            did_execute = False
            request = None
            try:
                self._validate_tracking_identity(scope)
                self._validate_scope_sources(scope)
                did_execute = True
                observation = self.runner.execute(scope, unit, attempt_id)
                if not isinstance(observation, RollingUnitRunnerObservation):
                    raise RollingExecutionContractError("runner returned a foreign observation")
                if observation.unit_key != unit.unit_key or observation.attempt_id != attempt_id:
                    raise RollingExecutionContractError("runner observation identity mismatch")
                if observation.candidate_status != "candidate_success":
                    raise RollingExecutionPreflightError(observation.failure_code)
                request = self.backend.commit_execution_manifest(scope, unit, observation)
                if not isinstance(request, RollingUnitEvidenceRequest) or request.unit_key != unit.unit_key:
                    raise RollingExecutionContractError("backend returned a foreign evidence request")
                requests_by_key[unit.unit_key] = request
                evidence = self._evidence_for_requests(self.backend, scope, (request,))
                result = evidence.unit_results[0]
                if result.classification != "valid":
                    raise RollingExecutionPreflightError("runner success lacks exact valid evidence")
                claims[unit.position] = _claim(
                    unit, "success", observation.recorder_id, result.evidence_fingerprint,
                    _source_extensions(request, attempt_id),
                )
                all_success_evidence = self._success_evidence(
                    self.backend, scope, requests_by_key, claims,
                )
                baseline = self._commit(
                    _snapshot(scope, "executing", attempt_id, claims), baseline,
                    all_success_evidence,
                )
                results.append(RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "executed_success", True,
                    attempt_id, observation.recorder_id, result.evidence_fingerprint,
                    "rolling_execution_evidence_committed",
                    _authority=_RESULT_TOKEN,
                ))
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                claims[unit.position] = _claim(
                    unit, "failed", extensions=_source_extensions(
                        request, attempt_id, {"interrupted_attempt_id": attempt_id},
                    ),
                )
                try:
                    evidence = self._success_evidence(self.backend, scope, requests_by_key, claims)
                    self._commit(_snapshot(scope, "failed", attempt_id, claims), baseline, evidence)
                except Exception:
                    pass
                raise
            except Exception as exc:
                failed_request = requests_by_key.pop(unit.unit_key, None) or request
                claims[unit.position] = _claim(
                    unit, "failed", extensions=_source_extensions(
                        failed_request, attempt_id,
                        {"failure_code": exc.__class__.__name__},
                    ),
                )
                try:
                    evidence = self._success_evidence(self.backend, scope, requests_by_key, claims)
                    baseline = self._commit(
                        _snapshot(scope, "executing", attempt_id, claims), baseline, evidence,
                    )
                except Exception:
                    state_uncertain = True
                results.append(RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "failed", did_execute,
                    attempt_id=attempt_id,
                    reason_code="rolling_execution_unit_failed",
                ))
        if results and all(item.status == "executed_success" for item in results):
            try:
                evidence = self._success_evidence(
                    self.backend, scope, requests_by_key, claims,
                )
                self._commit(
                    _snapshot(scope, "units_complete", attempt_id, claims), baseline,
                    evidence,
                )
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                return _invalidate_terminal_batch(
                    scope, _batch(scope, results),
                    "rolling_execution_units_complete_uncertain",
                )
        elif not state_uncertain:
            try:
                evidence = self._success_evidence(self.backend, scope, requests_by_key, claims)
                self._commit(_snapshot(scope, "failed", attempt_id, claims), baseline, evidence)
            except Exception:
                pass
        return _batch(scope, results)

    def resume(self, scope, attempt_id):
        self._validate_scope(scope)
        attempt_id = _attempt(attempt_id)
        preflight = preflight_rolling_execution(scope)
        if not preflight.preflight_allowed:
            return _blocked(scope, "rolling_execution_capability_blocked")
        try:
            self._validate_tracking_identity(scope)
            self._validate_scope_sources(scope)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(scope, "rolling_execution_tracking_backend_blocked")
        return self._run_with_lease(scope, attempt_id, self._resume_locked)

    def _resume_locked(self, scope, attempt_id):
        try:
            self._validate_tracking_identity(scope)
            self._validate_scope_sources(scope)
            view = self.repository.inspect_readonly()
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(
                scope, "rolling_execution_state_observation_blocked", attempt_id,
            )
        if view.inspection.classification != "valid_versioned":
            return _blocked(
                scope, "rolling_execution_resume_state_invalid", attempt_id,
            )
        state = view.inspection.snapshot
        if (
            state.workspace_fingerprint != scope.run_identity.workspace_fingerprint
            or state.run_id != scope.run_identity.fingerprint
            or state.execution_fingerprint != scope.run_identity.fingerprint
            or state.target_keys != scope.run_identity.target_keys
            or state.window_keys != scope.run_identity.window_keys
            or state.extensions != json.loads(_extensions(scope))
        ):
            return _blocked(
                scope, "rolling_execution_resume_scope_mismatch", attempt_id,
            )
        if state.phase not in (
            "prepared", "executing", "failed", "units_complete",
        ):
            return _blocked(
                scope, "rolling_execution_resume_phase_blocked", attempt_id,
            )
        if state.phase == "prepared":
            claims = [_claim(unit, "pending") for unit in scope.units]
            try:
                baseline = self._commit(
                    _snapshot(scope, "failed", None, claims), view.baseline,
                )
                reconciliation_attempt = "prepared-reconciliation-%s" % (
                    scope.run_identity.fingerprint[:16],
                )
                if reconciliation_attempt == attempt_id:
                    reconciliation_attempt += "-old"
                running = [_claim(unit, "running") for unit in scope.units]
                self._commit(
                    _snapshot(
                        scope, "executing", reconciliation_attempt, running,
                    ),
                    baseline,
                )
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                return _blocked(
                    scope, "rolling_execution_prepared_reconciliation_blocked",
                    attempt_id,
                )
            return self._resume_locked(scope, attempt_id)
        if (
            tuple(
                (item.target_key, item.window_key) for item in state.units
            ) != scope.requested_unit_keys
            or any(
                item.status not in ("pending", "running", "success", "failed")
                for item in state.units
            )
        ):
            return _blocked(
                scope, "rolling_execution_resume_unit_scope_blocked",
                state.attempt_id,
            )
        try:
            requests = self.backend.requests_for_state(scope, state)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(
                scope, "rolling_execution_source_observation_blocked",
                state.attempt_id,
            )
        if (
            not isinstance(requests, tuple)
            or len(requests) != len(scope.units)
            or any(not isinstance(item, RollingUnitEvidenceRequest) for item in requests)
            or tuple(item.unit_key for item in requests) != scope.requested_unit_keys
        ):
            return _blocked(
                scope, "rolling_execution_resume_request_identity_blocked",
                state.attempt_id,
            )
        try:
            evidence = self._evidence_for_requests(self.backend, scope, requests)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(
                scope, "rolling_execution_evidence_observation_blocked",
                state.attempt_id,
            )
        results = [None] * len(scope.units)
        retry_units = []
        requests_by_key = {item.unit_key: item for item in requests}
        claims = list(state.units)
        reconciled_success = False
        for unit, claim, observed in zip(scope.units, state.units, evidence.unit_results):
            if claim.status == "success":
                if observed.classification == "valid" and observed.evidence_fingerprint == claim.evidence_id:
                    results[unit.position] = RollingExecutionUnitResult(
                        unit.unit_key, unit.position, "reused_success", False,
                        state.attempt_id, claim.record_id, claim.evidence_id,
                        "rolling_execution_original_source_reused",
                        _authority=_RESULT_TOKEN,
                    )
                else:
                    results[unit.position] = RollingExecutionUnitResult(
                        unit.unit_key, unit.position, "blocked", False,
                        attempt_id=state.attempt_id,
                        reason_code="rolling_execution_committed_evidence_invalid",
                    )
            elif state.phase == "executing" and observed.classification == "valid":
                request = requests[unit.position]
                recorder_id = dict(observed.source_summary)["recorder_id"]
                claims[unit.position] = _claim(
                    unit, "success", recorder_id, observed.evidence_fingerprint,
                    _source_extensions(request, state.attempt_id),
                )
                results[unit.position] = RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "reused_success", False,
                    state.attempt_id, recorder_id, observed.evidence_fingerprint,
                    "rolling_execution_original_attempt_reconciled",
                    _authority=_RESULT_TOKEN,
                )
                reconciled_success = True
            elif observed.classification == "missing":
                retry_units.append(unit)
                requests_by_key.pop(unit.unit_key, None)
            else:
                results[unit.position] = RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "blocked", False,
                    attempt_id=state.attempt_id,
                    reason_code="rolling_execution_nonmissing_invalid_evidence",
                )
        baseline = view.baseline
        if reconciled_success:
            try:
                success_evidence = self._success_evidence(
                    self.backend, scope, requests_by_key, claims,
                )
                baseline = self._commit(
                    _snapshot(scope, "executing", state.attempt_id, claims), baseline,
                    success_evidence,
                )
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                return _blocked(
                    scope, "rolling_execution_reconciliation_state_blocked",
                    state.attempt_id,
                )
        if not retry_units:
            if results and all(item.status == "reused_success" for item in results):
                try:
                    success_evidence = self._success_evidence(
                        self.backend, scope, requests_by_key, claims,
                    )
                    self._commit(
                        _snapshot(scope, "units_complete", state.attempt_id, claims), baseline,
                        success_evidence,
                    )
                except (KeyboardInterrupt, SystemExit, GeneratorExit):
                    raise
                except Exception:
                    return _invalidate_terminal_batch(
                        scope, _batch(scope, results),
                        "rolling_execution_units_complete_uncertain",
                    )
            return _batch(scope, results)
        if any(item is not None and item.status == "blocked" for item in results):
            for unit in retry_units:
                results[unit.position] = RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "blocked", False,
                    attempt_id=state.attempt_id,
                    reason_code="rolling_execution_retry_blocked_by_invalid_evidence",
                )
            return _batch(scope, results)
        try:
            success_evidence = self._success_evidence(
                self.backend, scope, requests_by_key, claims,
            )
            if state.phase == "executing":
                for unit in retry_units:
                    claims[unit.position] = _claim(unit, "failed")
                baseline = self._commit(
                    _snapshot(scope, "failed", state.attempt_id, claims), baseline,
                    success_evidence,
                )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(
                scope, "rolling_execution_retry_precondition_blocked",
                state.attempt_id,
            )
        if state.phase not in ("failed", "executing"):
            raise RollingExecutionPreflightError("state phase cannot start a retry attempt")
        for unit in retry_units:
            claims[unit.position] = _claim(unit, "running")
        try:
            baseline = self._commit(
                _snapshot(scope, "executing", attempt_id, claims), baseline,
                success_evidence,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _blocked(
                scope, "rolling_execution_retry_state_blocked", attempt_id,
        )
        for unit in retry_units:
            request = None
            did_execute = False
            try:
                self._validate_tracking_identity(scope)
                self._validate_scope_sources(scope)
                did_execute = True
                observation = self.runner.execute(scope, unit, attempt_id)
                if not isinstance(observation, RollingUnitRunnerObservation):
                    raise RollingExecutionContractError("runner returned a foreign observation")
                if observation.unit_key != unit.unit_key or observation.attempt_id != attempt_id:
                    raise RollingExecutionContractError("retry observation identity mismatch")
                if observation.candidate_status != "candidate_success":
                    raise RollingExecutionPreflightError(observation.failure_code)
                request = self.backend.commit_execution_manifest(scope, unit, observation)
                one = self._evidence_for_requests(self.backend, scope, (request,))
                observed = one.unit_results[0]
                if observed.classification != "valid":
                    raise RollingExecutionPreflightError("retry evidence is invalid")
                requests_by_key[unit.unit_key] = request
                claims[unit.position] = _claim(
                    unit, "success", observation.recorder_id, observed.evidence_fingerprint,
                    _source_extensions(request, attempt_id),
                )
                success_evidence = self._success_evidence(
                    self.backend, scope, requests_by_key, claims,
                )
                baseline = self._commit(
                    _snapshot(scope, "executing", attempt_id, claims), baseline,
                    success_evidence,
                )
                results[unit.position] = RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "executed_success", True,
                    attempt_id, observation.recorder_id, observed.evidence_fingerprint,
                    "rolling_execution_retry_evidence_committed",
                    _authority=_RESULT_TOKEN,
                )
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                claims[unit.position] = _claim(
                    unit, "failed", extensions=_source_extensions(
                        request, attempt_id, {"interrupted_attempt_id": attempt_id},
                    ),
                )
                try:
                    success_evidence = self._success_evidence(
                        self.backend, scope, requests_by_key, claims,
                    )
                    self._commit(
                        _snapshot(scope, "failed", attempt_id, claims), baseline,
                        success_evidence,
                    )
                except Exception:
                    pass
                raise
            except Exception as exc:
                requests_by_key.pop(unit.unit_key, None)
                claims[unit.position] = _claim(
                    unit, "failed", extensions=_source_extensions(
                        request, attempt_id,
                        {"failure_code": exc.__class__.__name__},
                    ),
                )
                results[unit.position] = RollingExecutionUnitResult(
                    unit.unit_key, unit.position, "failed", did_execute,
                    attempt_id=attempt_id,
                    reason_code="rolling_execution_retry_failed",
                )
        if all(item.status in ("executed_success", "reused_success") for item in results):
            try:
                success_evidence = self._success_evidence(
                    self.backend, scope, requests_by_key, claims,
                )
                self._commit(
                    _snapshot(scope, "units_complete", attempt_id, claims), baseline,
                    success_evidence,
                )
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                return _invalidate_terminal_batch(
                    scope, _batch(scope, results),
                    "rolling_execution_units_complete_uncertain",
                )
        else:
            try:
                success_evidence = self._success_evidence(
                    self.backend, scope, requests_by_key, claims,
                )
                self._commit(
                    _snapshot(scope, "failed", attempt_id, claims), baseline,
                    success_evidence,
                )
            except Exception:
                pass
        return _batch(scope, results)
