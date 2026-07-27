"""Exact, immutable and non-current Rolling aggregate candidate contracts.

The module is import-pure.  It owns no MLflow or Qlib client and grants
positive authority only to objects rebuilt by its inspectors and kernel.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import numbers
import struct
from dataclasses import InitVar, dataclass, field
from typing import Any, Mapping, Optional, Protocol

from quantpits.rolling.errors import RollingAggregateContractError
from quantpits.rolling.evidence import (
    RollingEvidenceSetInspection,
    RollingUnitEvidenceRequest,
    _load_prediction_pickle,
    _rebuild_evidence_set,
)
from quantpits.rolling.execution import RollingExecutionScope
from quantpits.rolling.identity import RollingTargetIdentity
from quantpits.rolling.repository import (
    RollingStateRepository,
    RollingStateRepositoryView,
)
from quantpits.rolling.state import RollingStateV2Snapshot
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


AGGREGATE_PROTOCOL_VERSION = "rolling_aggregate_candidate_v2"
CANDIDATE_EXPERIMENTS = {
    "rolling": "Rolling_Aggregate_Candidates",
    "cpcv_rolling": "CPCV_Rolling_Aggregate_Candidates",
}
SOURCE_SET_STATUSES = ("all_valid", "incomplete", "observation_drifted")
CANDIDATE_CLASSIFICATIONS = (
    "missing", "duplicate", "foreign", "identity_mismatch", "partial",
    "corrupt", "not_comparable", "drifted", "valid",
)
TARGET_STATUSES = (
    "materialized_success", "reused_success", "failed", "blocked",
    "indeterminate",
)
BATCH_STATUSES = ("success", "failed", "blocked", "indeterminate")
_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)
_SOURCE_TOKEN = object()
_SOURCE_SET_TOKEN = object()
_CANDIDATE_TOKEN = object()
_TARGET_TOKEN = object()
_BATCH_TOKEN = object()
_DIGEST_CHARS = frozenset("0123456789abcdef")


def _contract(message: str) -> None:
    raise RollingAggregateContractError(message)


def _text(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str) or not value or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or value.startswith(("/", "\\"))
        or "://" in value
    ):
        _contract("%s must be a public non-empty trimmed identifier" % field_name)
    return value


def _digest(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in _DIGEST_CHARS for char in value)
    ):
        _contract("%s must be a lowercase SHA-256" % field_name)
    return value


def _strict_tuple(value: Any, field_name: str) -> tuple:
    if not isinstance(value, tuple):
        _contract("%s must be an ordered tuple" % field_name)
    return value


def _state_snapshot(view: RollingStateRepositoryView) -> RollingStateV2Snapshot:
    if not isinstance(view, RollingStateRepositoryView):
        _contract("state_repository_view must be canonical")
    inspection = view.inspection
    snapshot = inspection.snapshot
    if (
        inspection.classification != "valid_versioned"
        or not isinstance(snapshot, RollingStateV2Snapshot)
        or snapshot.phase != "units_complete"
        or not view.baseline.existed
        or view.baseline.fingerprint != inspection.fingerprint
    ):
        _contract("aggregate scope requires exact units_complete State V2")
    return snapshot


@dataclass(frozen=True)
class RollingAggregateScope:
    execution_scope: RollingExecutionScope
    state_repository_view: RollingStateRepositoryView
    aggregate_attempt_id: str
    target_keys: tuple
    window_keys: tuple
    requested_unit_keys: tuple
    state_fingerprint: str
    state_baseline_fingerprint: str
    candidate_keys: tuple
    protocol_version: str = AGGREGATE_PROTOCOL_VERSION
    scope_fingerprint: str = field(init=False)
    _builder_authority: bool = field(init=False, repr=False, compare=False)
    _authority: InitVar[Any] = None

    def __post_init__(self, _authority: Any) -> None:
        if _authority is not _SOURCE_TOKEN:
            _contract("aggregate scopes are canonical-builder-owned")
        if not isinstance(self.execution_scope, RollingExecutionScope):
            _contract("execution_scope must be canonical")
        _state_snapshot(self.state_repository_view)
        _text(self.aggregate_attempt_id, "aggregate_attempt_id")
        targets = _strict_tuple(self.target_keys, "target_keys")
        windows = _strict_tuple(self.window_keys, "window_keys")
        units = _strict_tuple(self.requested_unit_keys, "requested_unit_keys")
        candidates = _strict_tuple(self.candidate_keys, "candidate_keys")
        if not targets or len(targets) != len(set(targets)):
            _contract("aggregate target identity must be non-empty and unique")
        if not windows or len(windows) != len(set(windows)):
            _contract("aggregate window identity must be non-empty and unique")
        expected = tuple((target, window) for target in targets for window in windows)
        if units != expected or units != self.execution_scope.requested_unit_keys:
            _contract("aggregate requested identity/order/cardinality changed")
        if len(candidates) != len(targets) or len(candidates) != len(set(candidates)):
            _contract("candidate keys do not preserve target cardinality")
        _digest(self.state_fingerprint, "state_fingerprint")
        _digest(self.state_baseline_fingerprint, "state_baseline_fingerprint")
        if self.protocol_version != AGGREGATE_PROTOCOL_VERSION:
            _contract("aggregate protocol version is unsupported")
        payload = self.to_fingerprint_dict()
        object.__setattr__(self, "scope_fingerprint", fingerprint_value(payload))
        expected_candidates = tuple(fingerprint_value({
            "protocol_version": self.protocol_version,
            "scope_fingerprint": self.scope_fingerprint,
            "target_key": target,
            "member_keys": [
                list((target, window)) for window in self.window_keys
            ],
        }) for target in self.target_keys)
        if self.candidate_keys != expected_candidates:
            _contract("candidate keys do not bind the exact aggregate scope")
        object.__setattr__(self, "_builder_authority", True)

    @property
    def family(self) -> str:
        return self.execution_scope.run_identity.family

    def to_fingerprint_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "execution_scope_fingerprint": self.execution_scope.scope_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "state_baseline_fingerprint": self.state_baseline_fingerprint,
            "aggregate_attempt_id": self.aggregate_attempt_id,
            "target_keys": list(self.target_keys),
            "window_keys": list(self.window_keys),
            "requested_unit_keys": [list(item) for item in self.requested_unit_keys],
        }

    def to_public_dict(self) -> dict:
        payload = self.to_fingerprint_dict()
        payload["scope_fingerprint"] = self.scope_fingerprint
        payload["candidate_keys"] = list(self.candidate_keys)
        payload["authority"] = "audit_only"
        return payload


def build_rolling_aggregate_scope(
    execution_scope: RollingExecutionScope,
    state_repository_view: RollingStateRepositoryView,
    aggregate_attempt_id: str,
) -> RollingAggregateScope:
    snapshot = _state_snapshot(state_repository_view)
    if not isinstance(execution_scope, RollingExecutionScope):
        _contract("execution_scope must be canonical")
    expected_units = execution_scope.requested_unit_keys
    state_units = tuple((item.target_key, item.window_key) for item in snapshot.units)
    if (
        snapshot.workspace_fingerprint
        != execution_scope.run_identity.workspace_fingerprint
        or snapshot.family != execution_scope.run_identity.family
        or snapshot.action != execution_scope.run_identity.action
        or snapshot.plan_fingerprint != execution_scope.run_identity.plan_fingerprint
        or snapshot.execution_fingerprint != execution_scope.run_identity.fingerprint
        or snapshot.config_fingerprint != execution_scope.run_identity.config_fingerprint
        or snapshot.anchor_date != execution_scope.run_identity.anchor_date
        or snapshot.target_keys != execution_scope.run_identity.target_keys
        or snapshot.window_keys != execution_scope.run_identity.window_keys
        or state_units != expected_units
        or any(item.status != "success" for item in snapshot.units)
    ):
        _contract("State V2 and exact execution scope do not match")
    target_keys = tuple(item.target_key for item in execution_scope.targets)
    window_keys = tuple(item.window_key for item in execution_scope.windows)
    base = {
        "protocol_version": AGGREGATE_PROTOCOL_VERSION,
        "execution_scope_fingerprint": execution_scope.scope_fingerprint,
        "state_fingerprint": state_repository_view.inspection.fingerprint,
        "state_baseline_fingerprint": state_repository_view.baseline.fingerprint,
        "aggregate_attempt_id": _text(aggregate_attempt_id, "aggregate_attempt_id"),
        "target_keys": list(target_keys),
        "window_keys": list(window_keys),
        "requested_unit_keys": [list(item) for item in expected_units],
    }
    scope_fingerprint = fingerprint_value(base)
    candidate_keys = tuple(fingerprint_value({
        "protocol_version": AGGREGATE_PROTOCOL_VERSION,
        "scope_fingerprint": scope_fingerprint,
        "target_key": target,
        "member_keys": [list((target, window)) for window in window_keys],
    }) for target in target_keys)
    return RollingAggregateScope(
        execution_scope, state_repository_view, aggregate_attempt_id,
        target_keys, window_keys, expected_units,
        state_repository_view.inspection.fingerprint,
        state_repository_view.baseline.fingerprint,
        candidate_keys, _authority=_SOURCE_TOKEN,
    )


@dataclass(frozen=True)
class RollingAggregateSourceUnit:
    unit_key: tuple
    request_fingerprint: str
    evidence_fingerprint: str
    recorder_id: str
    sessions: tuple
    canonical_rows: tuple
    canonical_values: tuple
    content_fingerprint: str
    _authority: InitVar[Any] = None
    _inspector_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: Any) -> None:
        if _authority is not _SOURCE_TOKEN:
            _contract("valid aggregate sources are inspector-owned")
        if not isinstance(self.unit_key, tuple) or len(self.unit_key) != 2:
            _contract("source unit key must be a target/window pair")
        _digest(self.request_fingerprint, "request_fingerprint")
        _digest(self.evidence_fingerprint, "evidence_fingerprint")
        _text(self.recorder_id, "recorder_id")
        sessions = _strict_tuple(self.sessions, "sessions")
        rows = _strict_tuple(self.canonical_rows, "canonical_rows")
        values = _strict_tuple(self.canonical_values, "canonical_values")
        if not sessions or sessions != tuple(sorted(set(sessions))):
            _contract("source sessions are not canonical")
        if not rows or len(rows) != len(values) or rows != tuple(sorted(set(rows))):
            _contract("source rows are not canonical and unique")
        if tuple(sorted(set(row[0] for row in rows))) != sessions:
            _contract("source rows and sessions disagree")
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            _contract("source values must be finite binary64")
        _digest(self.content_fingerprint, "content_fingerprint")
        if self.content_fingerprint != _content_fingerprint(rows, values):
            _contract("source content fingerprint is inconsistent")
        object.__setattr__(self, "_inspector_authority", True)

    @property
    def capabilities(self) -> tuple:
        return ("render", "aggregate_source")

    def to_public_dict(self) -> dict:
        return {
            "unit_key": list(self.unit_key),
            "request_fingerprint": self.request_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "recorder_id": self.recorder_id,
            "sessions": list(self.sessions),
            "row_count": len(self.canonical_rows),
            "content_fingerprint": self.content_fingerprint,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class RollingAggregateSourceSetInspection:
    requested_unit_keys: tuple
    unit_results: tuple
    status: str
    reason_code: str
    evidence_set_fingerprint: str
    _authority: InitVar[Any] = None
    _inspector_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: Any) -> None:
        if _authority is not _SOURCE_SET_TOKEN:
            _contract("aggregate source sets are inspector-owned")
        requested = _strict_tuple(self.requested_unit_keys, "requested_unit_keys")
        results = _strict_tuple(self.unit_results, "unit_results")
        if any(not isinstance(item, RollingAggregateSourceUnit) for item in results):
            _contract("aggregate source set contains a foreign member")
        if tuple(item.unit_key for item in results) != requested:
            _contract("aggregate source set changed requested identity")
        expected = "all_valid" if results and len(results) == len(requested) else "incomplete"
        if self.status != expected or self.reason_code != "rolling_aggregate_source_set_%s" % expected:
            _contract("aggregate source status disagrees with members")
        _digest(self.evidence_set_fingerprint, "evidence_set_fingerprint")
        object.__setattr__(self, "_inspector_authority", True)

    @property
    def fingerprint(self) -> str:
        return fingerprint_value(self.to_public_dict())

    def to_public_dict(self) -> dict:
        return {
            "requested_unit_keys": [list(item) for item in self.requested_unit_keys],
            "unit_results": [item.to_public_dict() for item in self.unit_results],
            "status": self.status,
            "reason_code": self.reason_code,
            "evidence_set_fingerprint": self.evidence_set_fingerprint,
        }


def _canonical_frame(data: bytes, expected_sessions: tuple) -> tuple:
    try:
        import pandas as pd
        payload = _load_prediction_pickle(data)
    except _CONTROL:
        raise
    except Exception as exc:
        _contract("prediction safe decode failed: %s" % exc.__class__.__name__)
    if isinstance(payload, pd.Series):
        payload = payload.to_frame(name="score")
    if not isinstance(payload, pd.DataFrame) or len(payload.columns) != 1 or payload.empty:
        _contract("prediction must be a non-empty one-column frame")
    index = payload.index
    if not isinstance(index, pd.MultiIndex):
        _contract("prediction index must be a MultiIndex")
    names = tuple("" if item is None else str(item).lower() for item in index.names)
    date_positions = [i for i, name in enumerate(names) if name in ("datetime", "date", "session")]
    instrument_positions = [i for i, name in enumerate(names) if name in ("instrument", "symbol", "code")]
    if len(date_positions) != 1 or len(instrument_positions) != 1:
        _contract("prediction index levels are not canonical")
    dates = pd.to_datetime(index.get_level_values(date_positions[0]), errors="raise")
    if getattr(dates, "tz", None) is not None or any(item != item.normalize() for item in dates):
        _contract("prediction sessions must be timezone-naive midnight dates")
    instruments = tuple(str(item) for item in index.get_level_values(instrument_positions[0]))
    if any(not item or item != item.strip() for item in instruments):
        _contract("prediction instruments are invalid")
    scores = payload.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(scores.dtype) or pd.api.types.is_bool_dtype(scores.dtype):
        _contract("prediction score must be numeric and bool-free")
    rows = tuple(zip((item.date().isoformat() for item in dates), instruments))
    if rows != tuple(sorted(set(rows))):
        _contract("prediction rows must be sorted and unique")
    values = []
    for raw in scores:
        if isinstance(raw, (bool,)) or (
            isinstance(raw, numbers.Integral) and abs(int(raw)) > 2 ** 53
        ):
            _contract("prediction score conversion would hide information")
        value = float(raw)
        if not math.isfinite(value):
            _contract("prediction scores must be finite")
        values.append(0.0 if value == 0.0 else value)
    sessions = tuple(sorted(set(item[0] for item in rows)))
    if sessions != expected_sessions:
        _contract("prediction sessions do not exactly match the requested window")
    return rows, tuple(values)


def _content_fingerprint(rows: tuple, values: tuple) -> str:
    digest = hashlib.sha256()
    for row, value in zip(rows, values):
        digest.update(row[0].encode("utf-8") + b"\0")
        digest.update(row[1].encode("utf-8") + b"\0")
        digest.update(struct.pack(">d", 0.0 if value == 0.0 else value))
    return digest.hexdigest()


def _index_fingerprint(rows: tuple) -> str:
    digest = hashlib.sha256()
    for session, instrument in rows:
        digest.update(session.encode("utf-8") + b"\0")
        digest.update(instrument.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _value_fingerprint(values: tuple) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack(">d", 0.0 if value == 0.0 else value))
    return digest.hexdigest()


CANDIDATE_MANIFEST_CORE_FIELDS = frozenset({
    "schema_version",
    "protocol",
    "scope_fingerprint",
    "aggregate_attempt_id",
    "target_key",
    "candidate_key",
    "member_unit_keys",
    "source_set_fingerprint",
    "source_request_fingerprints",
    "source_evidence_fingerprints",
    "source_recorder_ids",
    "source_sessions",
    "source_row_counts",
    "source_content_fingerprints",
    "expected_sessions",
    "row_count",
    "candidate_index_fingerprint",
    "candidate_value_fingerprint",
    "content_fingerprint",
    "checked_predicates",
})


def _candidate_manifest_contract_fingerprint(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping):
        _contract("candidate manifest contract must be a mapping")
    if frozenset(manifest) != CANDIDATE_MANIFEST_CORE_FIELDS:
        _contract("candidate manifest core fields are not exact")
    return fingerprint_value(dict(manifest))


class RollingAggregateSourceBackend(Protocol):
    def inspect(self, scope: RollingExecutionScope, requests: tuple) -> RollingEvidenceSetInspection: ...
    def prediction_bytes(self, request: RollingUnitEvidenceRequest) -> bytes: ...


class RollingAggregateCandidateBackend(Protocol):
    def inventory(self, aggregate_scope: RollingAggregateScope) -> Mapping[str, Any]: ...
    def inspect_candidate(self, aggregate_scope: RollingAggregateScope, target_key: str, candidate_key: str, expected_manifest_contract_fingerprint: str) -> Mapping[str, Any]: ...
    def create_candidate(self, aggregate_scope: RollingAggregateScope, target_key: str, candidate_key: str, prediction_bytes: bytes, manifest: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def protected_snapshot(self, aggregate_scope: RollingAggregateScope) -> str: ...
    def backend_identity(self, aggregate_scope: RollingAggregateScope) -> str: ...


def _validate_source_state_join(aggregate_scope, requests, evidence):
    snapshot = aggregate_scope.state_repository_view.inspection.snapshot
    for unit, claim, request, observed in zip(
        aggregate_scope.execution_scope.units, snapshot.units,
        requests, evidence.unit_results,
    ):
        extensions = claim.extensions
        expected_artifacts = [
            item.to_fingerprint_dict() for item in request.artifacts
        ]
        if (
            claim.status != "success"
            or claim.record_id != request.recorder_id
            or claim.evidence_id != observed.evidence_fingerprint
            or not isinstance(extensions, dict)
            or extensions.get("attempt_id")
            != request.run_identity.attempt_id
            or extensions.get("source_manifest_fingerprint")
            != request.source_manifest_fingerprint
            or extensions.get("source_protocol") != request.source_protocol
            or extensions.get("source_publication_key")
            != request.source_publication_key
            or extensions.get("experiment_name") != request.experiment_name
            or extensions.get("experiment_id") != request.experiment_id
            or extensions.get("recorder_id") != request.recorder_id
            or extensions.get("source_operation") != request.source_operation
            or extensions.get("artifacts") != expected_artifacts
            or request.unit_key != unit.unit_key
            or request.run_identity.workspace_fingerprint
            != aggregate_scope.execution_scope.run_identity.workspace_fingerprint
            or request.run_identity.family
            != aggregate_scope.execution_scope.run_identity.family
            or request.run_identity.action
            != aggregate_scope.execution_scope.run_identity.action
            or request.run_identity.plan_fingerprint
            != aggregate_scope.execution_scope.run_identity.plan_fingerprint
            or request.run_identity.config_fingerprint
            != aggregate_scope.execution_scope.run_identity.config_fingerprint
            or request.run_identity.target_keys
            != aggregate_scope.execution_scope.run_identity.target_keys
            or request.run_identity.window_keys
            != aggregate_scope.execution_scope.run_identity.window_keys
            or request.run_identity.runtime_params_fingerprint
            != aggregate_scope.execution_scope.run_identity.runtime_params_fingerprint
        ):
            _contract("source request/evidence does not exactly join State")


def inspect_rolling_aggregate_sources(
    context: WorkspaceContext,
    aggregate_scope: RollingAggregateScope,
    source_requests: tuple,
    source_backend: RollingAggregateSourceBackend,
) -> RollingAggregateSourceSetInspection:
    if not isinstance(context, WorkspaceContext) or context != aggregate_scope.execution_scope.prepared.ctx:
        _contract("aggregate source context is foreign")
    requests = _strict_tuple(source_requests, "source_requests")
    if (
        any(not isinstance(item, RollingUnitEvidenceRequest) for item in requests)
        or tuple(item.unit_key for item in requests) != aggregate_scope.requested_unit_keys
    ):
        _contract("source requests changed requested identity/order/cardinality")
    evidence = _rebuild_evidence_set(
        source_backend.inspect(aggregate_scope.execution_scope, requests)
    )
    if (
        evidence.status != "all_valid"
        or evidence.requested_unit_keys != aggregate_scope.requested_unit_keys
    ):
        _contract("source evidence is incomplete or drifted")
    _validate_source_state_join(aggregate_scope, requests, evidence)
    results = []
    physical_ids = set()
    prior_sessions = {}
    for request, observed in zip(requests, evidence.unit_results):
        if (
            observed.classification != "valid"
            or observed.request_fingerprint != request.source_manifest_fingerprint
            or observed.source_protocol != "execution_bound_v1"
            or not observed._inspector_provenance
        ):
            _contract("source evidence lacks inspector authority")
        recorder_id = dict(observed.source_summary)["recorder_id"]
        if recorder_id in physical_ids:
            _contract("one physical recorder cannot represent two requested units")
        physical_ids.add(recorder_id)
        raw = source_backend.prediction_bytes(request)
        if not isinstance(raw, bytes):
            _contract("source backend returned non-byte prediction data")
        rows, values = _canonical_frame(raw, request.expected_prediction_sessions)
        sessions = tuple(sorted(set(row[0] for row in rows)))
        target_sessions = prior_sessions.setdefault(request.unit_key[0], set())
        if target_sessions.intersection(sessions):
            _contract("requested windows overlap")
        if target_sessions and min(sessions) <= max(target_sessions):
            _contract("requested window sessions are out of order")
        target_sessions.update(sessions)
        results.append(RollingAggregateSourceUnit(
            request.unit_key, request.source_manifest_fingerprint,
            observed.evidence_fingerprint, recorder_id, sessions, rows, values,
            _content_fingerprint(rows, values), _authority=_SOURCE_TOKEN,
        ))
    return RollingAggregateSourceSetInspection(
        aggregate_scope.requested_unit_keys, tuple(results), "all_valid",
        "rolling_aggregate_source_set_all_valid", evidence.fingerprint,
        _authority=_SOURCE_SET_TOKEN,
    )


@dataclass(frozen=True)
class RollingAggregateCandidateInspection:
    target_key: str
    candidate_key: str
    classification: str
    recorder_id: Optional[str] = None
    manifest_fingerprint: Optional[str] = None
    content_fingerprint: Optional[str] = None
    row_count: Optional[int] = None
    _authority: InitVar[Any] = None
    _inspector_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: Any) -> None:
        RollingTargetIdentity.parse(self.target_key)
        _digest(self.candidate_key, "candidate_key")
        if self.classification not in CANDIDATE_CLASSIFICATIONS:
            _contract("unknown candidate classification")
        valid = self.classification == "valid"
        if valid and _authority is not _CANDIDATE_TOKEN:
            _contract("valid candidate inspections are inspector-owned")
        facts = (self.recorder_id, self.manifest_fingerprint, self.content_fingerprint, self.row_count)
        if valid:
            if any(item is None for item in facts) or type(self.row_count) is not int or self.row_count <= 0:
                _contract("valid candidate lacks exact observed facts")
            _text(self.recorder_id, "recorder_id")
            _digest(self.manifest_fingerprint, "manifest_fingerprint")
            _digest(self.content_fingerprint, "content_fingerprint")
        elif any(item is not None for item in facts):
            _contract("invalid candidate cannot grant candidate facts")
        object.__setattr__(self, "_inspector_authority", valid)

    @property
    def capabilities(self) -> tuple:
        return ("render", "audit", "candidate_reference") if self.classification == "valid" else ("render",)

    def to_public_dict(self) -> dict:
        return {
            "target_key": self.target_key,
            "candidate_key": self.candidate_key,
            "classification": self.classification,
            "reason_code": "rolling_aggregate_candidate_%s" % self.classification,
            "recorder_id": self.recorder_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "row_count": self.row_count,
            "capabilities": list(self.capabilities),
        }


def _candidate_from_observation(scope, target_key, candidate_key, observation, expected):
    if not isinstance(observation, Mapping):
        _contract("candidate observation must be a mapping")
    classification = observation.get("classification")
    if classification != "valid":
        return RollingAggregateCandidateInspection(
            target_key, candidate_key,
            classification if classification in CANDIDATE_CLASSIFICATIONS else "not_comparable",
        )
    if (
        observation.get("candidate_key") != candidate_key
        or observation.get("target_key") != target_key
        or observation.get("scope_fingerprint") != scope.scope_fingerprint
        or observation.get("aggregate_attempt_id") != scope.aggregate_attempt_id
        or observation.get("content_fingerprint") != expected["content_fingerprint"]
        or observation.get("row_count") != expected["row_count"]
        or observation.get("manifest_contract_fingerprint")
        != expected["manifest_contract_fingerprint"]
    ):
        return RollingAggregateCandidateInspection(target_key, candidate_key, "identity_mismatch")
    return RollingAggregateCandidateInspection(
        target_key, candidate_key, "valid", observation.get("recorder_id"),
        observation.get("manifest_fingerprint"),
        observation.get("content_fingerprint"), observation.get("row_count"),
        _authority=_CANDIDATE_TOKEN,
    )


@dataclass(frozen=True)
class RollingAggregateTargetResult:
    target_key: str
    requested_unit_keys: tuple
    status: str
    did_write: Optional[bool]
    candidate: Optional[RollingAggregateCandidateInspection]
    reason_code: str
    _authority: InitVar[Any] = None
    _kernel_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: Any) -> None:
        RollingTargetIdentity.parse(self.target_key)
        units = _strict_tuple(self.requested_unit_keys, "requested_unit_keys")
        if not units or any(item[0] != self.target_key for item in units):
            _contract("target result unit identity is invalid")
        if self.status not in TARGET_STATUSES or self.reason_code != "rolling_aggregate_target_%s" % self.status:
            _contract("target status and reason disagree")
        if self.did_write not in (True, False, None):
            _contract("did_write must be a strict tri-state")
        success = self.status in ("materialized_success", "reused_success")
        if success:
            if _authority is not _TARGET_TOKEN or not isinstance(self.candidate, RollingAggregateCandidateInspection):
                _contract("successful target results are kernel-owned")
            if self.candidate.classification != "valid" or self.candidate.target_key != self.target_key:
                _contract("successful target candidate is invalid")
        elif self.candidate is not None:
            _contract("non-success target cannot grant candidate capability")
        if self.status == "materialized_success" and self.did_write is not True:
            _contract("materialized_success requires did_write=true")
        if self.status == "reused_success" and self.did_write is not False:
            _contract("reused_success requires did_write=false")
        if self.status == "indeterminate" and self.did_write is not None:
            _contract("indeterminate requires did_write=null")
        if self.status == "blocked" and self.did_write is not False:
            _contract("blocked result must be known no-write")
        if self.status == "failed" and self.did_write not in (True, False):
            _contract("failed result must have an observed write classification")
        object.__setattr__(self, "_kernel_authority", success and _authority is _TARGET_TOKEN)

    @property
    def capabilities(self) -> tuple:
        return self.candidate.capabilities if self.candidate is not None else ("render",)

    def to_public_dict(self) -> dict:
        return {
            "target_key": self.target_key,
            "requested_unit_keys": [list(item) for item in self.requested_unit_keys],
            "status": self.status,
            "reason_code": self.reason_code,
            "did_write": self.did_write,
            "candidate": self.candidate.to_public_dict() if self.candidate else None,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class RollingAggregateBatchResult:
    requested_target_keys: tuple
    target_results: tuple
    status: str
    reason_code: str
    inventory_counts: tuple = ()
    _authority: InitVar[Any] = None
    _kernel_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: Any) -> None:
        targets = _strict_tuple(self.requested_target_keys, "requested_target_keys")
        results = _strict_tuple(self.target_results, "target_results")
        if not targets or any(not isinstance(item, RollingAggregateTargetResult) for item in results):
            _contract("batch members are invalid")
        if tuple(item.target_key for item in results) != targets:
            _contract("batch changed requested target identity/order/cardinality")
        expected = (
            "indeterminate" if any(item.status == "indeterminate" for item in results)
            else "failed" if any(item.status == "failed" for item in results)
            else "blocked" if any(item.status == "blocked" for item in results)
            else "success"
        )
        if self.status != expected or self.reason_code != "rolling_aggregate_batch_%s" % expected:
            _contract("batch status/reason disagrees with terminal members")
        if expected == "success" and _authority is not _BATCH_TOKEN:
            _contract("successful aggregate batches are kernel-owned")
        counts = _strict_tuple(self.inventory_counts, "inventory_counts")
        if counts:
            if (
                _authority is not _BATCH_TOKEN
                or len(counts) != 4
                or any(type(item) is not int or item < 0 for item in counts)
                or counts[0] != counts[1] + counts[2] + counts[3]
            ):
                _contract("candidate inventory counts lack inspector authority")
        elif expected == "success":
            _contract("successful batch lacks raw inventory partition")
        object.__setattr__(self, "_kernel_authority", expected == "success" and _authority is _BATCH_TOKEN)

    @property
    def capabilities(self) -> tuple:
        return ("render", "audit", "publication_input") if self.status == "success" else ("render", "audit")

    @property
    def fingerprint(self) -> str:
        return fingerprint_value(self.to_public_dict())

    def to_public_dict(self) -> dict:
        payload = {
            "requested_target_keys": list(self.requested_target_keys),
            "target_results": [item.to_public_dict() for item in self.target_results],
            "status": self.status,
            "reason_code": self.reason_code,
            "n_requested": len(self.target_results),
            "n_materialized": sum(item.status == "materialized_success" for item in self.target_results),
            "n_reused": sum(item.status == "reused_success" for item in self.target_results),
            "n_failed": sum(item.status == "failed" for item in self.target_results),
            "n_blocked": sum(item.status == "blocked" for item in self.target_results),
            "n_indeterminate": sum(item.status == "indeterminate" for item in self.target_results),
            "capabilities": list(self.capabilities),
        }
        payload["candidate_inventory"] = (
            {
                "raw_inventory_count": self.inventory_counts[0],
                "n_requested_owned": self.inventory_counts[1],
                "n_orphan_owned": self.inventory_counts[2],
                "n_unassigned": self.inventory_counts[3],
            }
            if self.inventory_counts else None
        )
        return payload


def _prediction_bytes(rows, values):
    import pandas as pd
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(session), instrument) for session, instrument in rows],
        names=("datetime", "instrument"),
    )
    frame = pd.DataFrame({"score": list(values)}, index=index, dtype="float64")
    output = io.BytesIO()
    frame.to_pickle(output)
    return output.getvalue()


class RollingAggregateCandidateKernel:
    def __init__(self, repository, source_backend, candidate_backend):
        if not isinstance(repository, RollingStateRepository):
            _contract("kernel repository must be RollingStateRepository")
        self.repository = repository
        self.source_backend = source_backend
        self.candidate_backend = candidate_backend

    def materialize(self, aggregate_scope):
        if not isinstance(aggregate_scope, RollingAggregateScope):
            _contract("materialize requires RollingAggregateScope")
        current_view = self.repository.inspect_readonly()
        if current_view != aggregate_scope.state_repository_view:
            return self._blocked_batch(aggregate_scope)
        try:
            protected_before = self.candidate_backend.protected_snapshot(
                aggregate_scope,
            )
            backend_before = self.candidate_backend.backend_identity(
                aggregate_scope,
            )
            _digest(protected_before, "protected_snapshot")
            _digest(backend_before, "candidate_backend_identity")
        except _CONTROL:
            raise
        except Exception:
            return self._blocked_batch(aggregate_scope)
        try:
            requests = self.source_backend.requests_for_state(
                aggregate_scope.execution_scope,
                current_view.inspection.snapshot,
            )
            sources = inspect_rolling_aggregate_sources(
                self.repository.context, aggregate_scope, requests,
                self.source_backend,
            )
        except _CONTROL:
            raise
        except Exception:
            return self._blocked_batch(aggregate_scope)
        results = []
        by_target = {
            target: tuple(item for item in sources.unit_results if item.unit_key[0] == target)
            for target in aggregate_scope.target_keys
        }
        try:
            inventory = self.candidate_backend.inventory(aggregate_scope)
        except _CONTROL:
            raise
        except Exception:
            return self._blocked_batch(aggregate_scope)
        if not self._valid_inventory(inventory):
            return self._blocked_batch(aggregate_scope)
        stop_new_writes = False
        for position, target in enumerate(aggregate_scope.target_keys):
            units = by_target[target]
            unit_keys = tuple(item.unit_key for item in units)
            candidate_key = aggregate_scope.candidate_keys[position]
            rows = tuple(row for item in units for row in item.canonical_rows)
            values = tuple(value for item in units for value in item.canonical_values)
            if rows != tuple(sorted(set(rows))):
                results.append(self._result(target, unit_keys, "blocked", False))
                continue
            expected_sessions = tuple(
                session for item in units for session in item.sessions
            )
            manifest = {
                "schema_version": 2,
                "protocol": AGGREGATE_PROTOCOL_VERSION,
                "scope_fingerprint": aggregate_scope.scope_fingerprint,
                "aggregate_attempt_id": aggregate_scope.aggregate_attempt_id,
                "target_key": target,
                "candidate_key": candidate_key,
                "member_unit_keys": [list(item.unit_key) for item in units],
                "source_set_fingerprint": sources.fingerprint,
                "source_request_fingerprints": [
                    item.request_fingerprint for item in units
                ],
                "source_evidence_fingerprints": [
                    item.evidence_fingerprint for item in units
                ],
                "source_recorder_ids": [item.recorder_id for item in units],
                "source_sessions": [
                    list(item.sessions) for item in units
                ],
                "source_row_counts": [
                    len(item.canonical_rows) for item in units
                ],
                "source_content_fingerprints": [
                    item.content_fingerprint for item in units
                ],
                "expected_sessions": list(expected_sessions),
                "row_count": len(rows),
                "candidate_index_fingerprint": _index_fingerprint(rows),
                "candidate_value_fingerprint": _value_fingerprint(values),
                "content_fingerprint": _content_fingerprint(rows, values),
                "checked_predicates": [
                    "source_identity_order_cardinality",
                    "source_state_join",
                    "source_terminal_evidence",
                    "source_session_exactness",
                    "source_non_overlap",
                    "candidate_index_exactness",
                    "candidate_value_exactness",
                    "candidate_content_exactness",
                ],
            }
            expected = {
                "content_fingerprint": manifest["content_fingerprint"],
                "row_count": len(rows),
                "manifest_contract_fingerprint":
                    _candidate_manifest_contract_fingerprint(manifest),
            }
            before_target_inventory = inventory
            if stop_new_writes:
                results.append(self._result(target, unit_keys, "blocked", False))
                continue
            try:
                existing = self.candidate_backend.inspect_candidate(
                    aggregate_scope, target, candidate_key,
                    expected["manifest_contract_fingerprint"],
                )
                inspected = _candidate_from_observation(
                    aggregate_scope, target, candidate_key, existing, expected,
                )
                if inspected.classification == "valid":
                    if self._postconditions_stable(
                        aggregate_scope, current_view, sources,
                        protected_before, backend_before,
                    ):
                        results.append(self._result(
                            target, unit_keys, "reused_success", False,
                            inspected,
                        ))
                    else:
                        results.append(self._result(
                            target, unit_keys, "indeterminate", None,
                        ))
                        stop_new_writes = True
                    continue
                if inspected.classification != "missing":
                    results.append(self._result(target, unit_keys, "blocked", False))
                    continue
                before_target_inventory = self.candidate_backend.inventory(
                    aggregate_scope,
                )
                if not self._valid_inventory(before_target_inventory):
                    results.append(self._result(target, unit_keys, "blocked", False))
                    stop_new_writes = True
                    continue
                observation = self.candidate_backend.create_candidate(
                    aggregate_scope, target, candidate_key,
                    _prediction_bytes(rows, values), manifest,
                )
                inspected = _candidate_from_observation(
                    aggregate_scope, target, candidate_key, observation, expected,
                )
                if inspected.classification != "valid":
                    results.append(self._result(target, unit_keys, "indeterminate", None))
                    stop_new_writes = True
                    continue
                if not self._postconditions_stable(
                    aggregate_scope, current_view, sources,
                    protected_before, backend_before,
                ):
                    results.append(self._result(target, unit_keys, "indeterminate", None))
                    stop_new_writes = True
                    continue
                results.append(self._result(
                    target, unit_keys, "materialized_success", True, inspected,
                ))
            except _CONTROL:
                raise
            except Exception:
                try:
                    after_error = self.candidate_backend.inventory(
                        aggregate_scope,
                    )
                    if not self._valid_inventory(after_error):
                        raise ValueError("inventory unavailable")
                    delta = (
                        after_error["raw_count"]
                        - before_target_inventory["raw_count"]
                    )
                    if delta not in (0, 1):
                        raise ValueError("inventory delta is not comparable")
                    experiment_created = (
                        not before_target_inventory["experiment_present"]
                        and after_error["experiment_present"]
                    )
                    results.append(self._result(
                        target, unit_keys, "failed",
                        bool(delta) or experiment_created,
                    ))
                except _CONTROL:
                    raise
                except Exception:
                    results.append(self._result(
                        target, unit_keys, "indeterminate", None,
                    ))
                    stop_new_writes = True
        return self._batch(aggregate_scope, tuple(results), observe_inventory=True)

    def _postconditions_stable(
        self, aggregate_scope, current_view, sources,
        protected_before, backend_before,
    ):
        try:
            requests_after = self.source_backend.requests_for_state(
                aggregate_scope.execution_scope,
                current_view.inspection.snapshot,
            )
            sources_after = inspect_rolling_aggregate_sources(
                self.repository.context, aggregate_scope, requests_after,
                self.source_backend,
            )
            return (
                sources_after.fingerprint == sources.fingerprint
                and self.repository.inspect_readonly() == current_view
                and self.candidate_backend.protected_snapshot(
                    aggregate_scope,
                ) == protected_before
                and self.candidate_backend.backend_identity(
                    aggregate_scope,
                ) == backend_before
            )
        except _CONTROL:
            raise
        except Exception:
            return False

    @staticmethod
    def _valid_inventory(inventory):
        return (
            isinstance(inventory, Mapping)
            and type(inventory.get("raw_count")) is int
            and inventory["raw_count"] >= 0
            and isinstance(inventory.get("candidates"), tuple)
            and inventory["raw_count"] == len(inventory["candidates"])
            and isinstance(inventory.get("fingerprint"), str)
            and len(inventory["fingerprint"]) == 64
            and type(inventory.get("experiment_present")) is bool
        )

    @staticmethod
    def _result(target, units, status, did_write, candidate=None):
        return RollingAggregateTargetResult(
            target, units, status, did_write, candidate,
            "rolling_aggregate_target_%s" % status,
            _authority=_TARGET_TOKEN if status.endswith("_success") else None,
        )

    def _blocked_batch(self, scope):
        return self._batch(scope, tuple(
            self._result(
                target,
                tuple((target, window) for window in scope.window_keys),
                "blocked", False,
            ) for target in scope.target_keys
        ), observe_inventory=False)

    def _inventory_counts(self, scope):
        inventory = self.candidate_backend.inventory(scope)
        if not self._valid_inventory(inventory):
            _contract("terminal candidate inventory is not comparable")
        requested = set(scope.candidate_keys)
        requested_owned = orphan_owned = unassigned = 0
        for item in inventory["candidates"]:
            if not isinstance(item, Mapping):
                unassigned += 1
                continue
            candidate_key = item.get("candidate_key")
            if candidate_key in requested:
                requested_owned += 1
                continue
            try:
                _digest(candidate_key, "inventory candidate_key")
                RollingTargetIdentity.parse(item.get("target_key"))
                _text(item.get("scope_fingerprint"), "inventory scope")
                _text(
                    item.get("aggregate_attempt_id"),
                    "inventory aggregate attempt",
                )
                orphan_owned += 1
            except RollingAggregateContractError:
                unassigned += 1
        return (
            inventory["raw_count"], requested_owned,
            orphan_owned, unassigned,
        )

    def _batch(self, scope, results, observe_inventory):
        status = (
            "indeterminate" if any(item.status == "indeterminate" for item in results)
            else "failed" if any(item.status == "failed" for item in results)
            else "blocked" if any(item.status == "blocked" for item in results)
            else "success"
        )
        counts = ()
        if observe_inventory:
            try:
                counts = self._inventory_counts(scope)
            except _CONTROL:
                raise
            except Exception:
                if status == "success":
                    results = tuple(self._result(
                        item.target_key, item.requested_unit_keys,
                        "indeterminate", None,
                    ) for item in results)
                    status = "indeterminate"
        return RollingAggregateBatchResult(
            scope.target_keys, results, status,
            "rolling_aggregate_batch_%s" % status,
            counts,
            _authority=_BATCH_TOKEN if counts or status == "success" else None,
        )


def materialize_rolling_aggregate_candidates(
    aggregate_scope: RollingAggregateScope,
    repository: RollingStateRepository,
    source_backend: RollingAggregateSourceBackend,
    candidate_backend: RollingAggregateCandidateBackend,
) -> RollingAggregateBatchResult:
    return RollingAggregateCandidateKernel(
        repository, source_backend, candidate_backend,
    ).materialize(aggregate_scope)


def rolling_aggregate_result_json(result: RollingAggregateBatchResult) -> str:
    if not isinstance(result, RollingAggregateBatchResult):
        _contract("JSON rendering requires a typed aggregate batch")
    payload = result.to_public_dict()
    payload["fingerprint"] = result.fingerprint
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def render_rolling_aggregate_result(result: RollingAggregateBatchResult) -> str:
    if not isinstance(result, RollingAggregateBatchResult):
        _contract("human rendering requires a typed aggregate batch")
    return (
        "Rolling aggregate: status=%s requested=%d materialized=%d reused=%d "
        "failed=%d blocked=%d indeterminate=%d fingerprint=%s"
        % (
            result.status, len(result.target_results),
            sum(item.status == "materialized_success" for item in result.target_results),
            sum(item.status == "reused_success" for item in result.target_results),
            sum(item.status == "failed" for item in result.target_results),
            sum(item.status == "blocked" for item in result.target_results),
            sum(item.status == "indeterminate" for item in result.target_results),
            result.fingerprint,
        )
    )
