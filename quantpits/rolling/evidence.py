"""Immutable, read-only evidence inspection for Rolling execution units.

This module deliberately owns no backend client and imports neither MLflow nor
Qlib.  A caller supplies a small metadata-only backend; artifact bytes are read
and checked here so metadata cannot assert containment, integrity, or coverage.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import pickle
import stat
from dataclasses import InitVar, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlparse

from quantpits.rolling.errors import RollingEvidenceContractError, RollingIdentityError
from quantpits.rolling.identity import (
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingWindowIdentity,
    normalize_iso_date,
    parse_rolling_window_key,
    workspace_fingerprint,
)
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


SOURCE_PROTOCOLS = ("execution_bound_v1", "legacy_unverified")
ARTIFACT_ROLES = ("prediction", "supporting")
CLASSIFICATIONS = (
    "missing", "duplicate", "foreign", "identity_mismatch", "partial",
    "corrupt", "coverage_short", "not_comparable", "legacy_unverified",
    "drifted", "valid",
)
SET_STATUSES = ("all_valid", "incomplete", "none_valid", "observation_drifted")
CLASSIFICATION_REASONS = {
    "missing": "rolling_evidence_missing",
    "duplicate": "rolling_evidence_duplicate",
    "foreign": "rolling_evidence_foreign",
    "identity_mismatch": "rolling_evidence_identity_mismatch",
    "partial": "rolling_evidence_partial",
    "corrupt": "rolling_evidence_corrupt",
    "coverage_short": "rolling_evidence_coverage_short",
    "not_comparable": "rolling_evidence_not_comparable",
    "legacy_unverified": "rolling_evidence_legacy_unverified",
    "drifted": "rolling_evidence_drifted",
    "valid": "rolling_evidence_valid",
}
_CLASSIFICATION_PRIORITY = {
    name: index for index, name in enumerate((
        "drifted", "duplicate", "missing", "foreign", "legacy_unverified",
        "identity_mismatch", "partial", "corrupt", "not_comparable",
        "coverage_short", "valid",
    ))
}
SET_REASONS = {
    "all_valid": "rolling_evidence_set_all_valid",
    "incomplete": "rolling_evidence_set_incomplete",
    "none_valid": "rolling_evidence_set_none_valid",
    "observation_drifted": "rolling_evidence_set_drifted",
}
_BOUND_ACTIONS = {
    "cold_start", "merge", "resume", "daily", "retrain_models", "retrain_last",
}
_DIGEST_CHARS = frozenset("0123456789abcdef")
_INSPECTION_TOKEN = object()
_VALID_CHECKS = frozenset({
    "candidate_cardinality", "source_identity", "artifact_root_containment",
    "artifact_node_kind", "artifact_byte_fingerprint", "artifact_public_path_recheck",
    "prediction_schema", "prediction_index_unique", "prediction_scores_finite",
    "prediction_session_coverage",
})
_VALID_SUMMARY_KEYS = frozenset({
    "backend_fingerprint", "experiment_name", "experiment_id", "recorder_id",
    "target_key", "window_key", "source_protocol", "source_publication_key",
    "source_operation", "source_manifest_fingerprint",
})
_SAFE_PREDICTION_PICKLE_GLOBALS = frozenset({
    ("builtins", "slice"),
    ("copyreg", "_reconstructor"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas._libs.tslibs.nattype", "__nat_unpickle"),
    ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
    ("pandas.core.arrays.datetimes", "DatetimeArray"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.datetimes", "DatetimeIndex"),
    ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
    ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas.core.series", "Series"),
})


def _contract(message: str) -> None:
    raise RollingEvidenceContractError(message)


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        _contract("%s must be a non-empty trimmed string" % field)
    return value


def _public_identifier(value: Any, field: str) -> str:
    value = _identifier(value, field)
    lowered = value.lower()
    windows_absolute = len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in ("/", "\\")
    if (
        value.startswith(("/", "\\"))
        or lowered.startswith("file:")
        or "://" in value
        or windows_absolute
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        _contract("%s is not a public canonical identifier" % field)
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _DIGEST_CHARS for char in value):
        _contract("%s must be a lowercase SHA-256 digest" % field)
    return value


def _strict_size(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        _contract("%s must be a non-negative integer" % field)
    return value


def _artifact_key(value: Any) -> str:
    value = _identifier(value, "logical_key")
    if "\\" in value or value.startswith("/"):
        _contract("artifact logical_key must be a relative POSIX path")
    path = PurePosixPath(value)
    if value in (".", "..") or any(part in ("", ".", "..") for part in path.parts):
        _contract("artifact logical_key is not canonical")
    if path.as_posix() != value:
        _contract("artifact logical_key is not canonical")
    return value


def _tuple(value: Any, field: str) -> tuple:
    if not isinstance(value, tuple):
        _contract("%s must be an ordered tuple" % field)
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _contract("%s must be a mapping" % field)
    return value


def _public_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _public_summary(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_public_summary(item) for item in value]
    return value


def _prefer(current: str, candidate: str) -> str:
    return candidate if _CLASSIFICATION_PRIORITY[candidate] < _CLASSIFICATION_PRIORITY[current] else current


def _validate_source_summary(summary: tuple, field: str = "source_summary") -> tuple:
    summary = _tuple(summary, field)
    if any(not isinstance(item, tuple) or len(item) != 2 for item in summary):
        _contract("%s must contain key/value pairs" % field)
    allowed = {
        "backend_fingerprint", "experiment_name", "experiment_id", "recorder_id",
        "target_key", "window_key", "source_protocol", "source_publication_key",
        "source_operation", "source_manifest_fingerprint",
    }
    if len(summary) != len({item[0] for item in summary}):
        _contract("%s keys must be unique" % field)
    for key, value in summary:
        if key not in allowed or (value is not None and not isinstance(value, str)):
            _contract("%s contains a non-public fact" % field)
        if isinstance(value, str):
            _public_identifier(value, "%s value" % field)
    return summary


def _unit_evidence_fingerprint(
    request_fingerprint: str,
    summary: tuple,
    observations: tuple,
    coverage: "RollingPredictionCoverage",
    checked: tuple,
) -> str:
    return fingerprint_value({
        "request_fingerprint": request_fingerprint,
        "source_summary": _public_summary(summary),
        "artifacts": [item.to_public_dict() for item in observations],
        "coverage": coverage.to_public_dict(),
        "checked": list(checked),
        "classification": "valid",
    })


@dataclass(frozen=True)
class RollingArtifactExpectation:
    logical_key: str
    role: str
    size_bytes: int
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_key", _artifact_key(self.logical_key))
        if self.role not in ARTIFACT_ROLES:
            _contract("artifact role must be prediction or supporting")
        _strict_size(self.size_bytes, "size_bytes")
        _digest(self.fingerprint, "fingerprint")

    def to_fingerprint_dict(self) -> dict[str, Any]:
        return {
            "logical_key": self.logical_key,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class RollingUnitEvidenceRequest:
    run_identity: RollingRunIdentity
    target_key: str
    window_identity: RollingWindowIdentity
    source_protocol: str
    source_publication_key: str
    source_operation: str
    experiment_name: str
    experiment_id: str
    recorder_id: str
    artifacts: tuple
    expected_prediction_sessions: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.run_identity, RollingRunIdentity):
            _contract("run_identity must be RollingRunIdentity")
        if not isinstance(self.window_identity, RollingWindowIdentity):
            _contract("window_identity must be RollingWindowIdentity")
        target = RollingTargetIdentity.parse(self.target_key)
        if target.family != self.run_identity.family:
            _contract("target family does not match run family")
        if self.target_key not in self.run_identity.target_keys:
            _contract("requested target is outside run identity")
        window = RollingWindowIdentity(**{
            "family": self.window_identity.family,
            "train_start": self.window_identity.train_start,
            "train_end": self.window_identity.train_end,
            "test_start": self.window_identity.test_start,
            "test_end": self.window_identity.test_end,
            "effective_config_fingerprint": self.window_identity.effective_config_fingerprint,
            "valid_start": self.window_identity.valid_start,
            "valid_end": self.window_identity.valid_end,
            "folds": self.window_identity.folds,
        })
        if window.family != self.run_identity.family or window.window_key not in self.run_identity.window_keys:
            _contract("requested window is outside run identity")
        if self.source_protocol not in SOURCE_PROTOCOLS:
            _contract("unsupported Rolling evidence source protocol")
        try:
            source_publication = RollingTargetIdentity.parse(self.source_publication_key)
        except RollingIdentityError as exc:
            _contract("source_publication_key must be a canonical Rolling target: %s" % exc)
        if source_publication.family != self.run_identity.family:
            _contract("source publication family does not match run family")
        if self.source_operation not in _BOUND_ACTIONS:
            _contract("source_operation is not a Rolling unit operation")
        for name in ("experiment_name", "experiment_id", "recorder_id"):
            _public_identifier(getattr(self, name), name)
        artifacts = _tuple(self.artifacts, "artifacts")
        rebuilt = tuple(
            RollingArtifactExpectation(
                item.logical_key, item.role, item.size_bytes, item.fingerprint,
            ) if isinstance(item, RollingArtifactExpectation) else _invalid_artifact_member()
            for item in artifacts
        )
        keys = tuple(item.logical_key for item in rebuilt)
        if not rebuilt or len(keys) != len(set(keys)):
            _contract("artifact expectations must be non-empty and unique")
        if sum(item.role == "prediction" for item in rebuilt) != 1:
            _contract("request must contain exactly one prediction artifact")
        sessions = _tuple(self.expected_prediction_sessions, "expected_prediction_sessions")
        normalized = tuple(normalize_iso_date(item, "expected_prediction_session") for item in sessions)
        if not normalized or normalized != tuple(sorted(set(normalized))):
            _contract("expected prediction sessions must be non-empty, unique, and increasing")
        if normalized[0] != window.test_start or normalized[-1] != window.test_end:
            _contract("expected prediction sessions must match window test boundaries")
        if self.source_protocol == "execution_bound_v1":
            if self.run_identity.action not in _BOUND_ACTIONS:
                _contract("run action cannot claim bound per-window evidence")
            _public_identifier(self.run_identity.attempt_id, "attempt_id")
        object.__setattr__(self, "window_identity", window)
        object.__setattr__(self, "artifacts", rebuilt)
        object.__setattr__(self, "expected_prediction_sessions", normalized)

    @property
    def window_key(self) -> str:
        return self.window_identity.window_key

    @property
    def unit_key(self) -> tuple[str, str]:
        return (self.target_key, self.window_key)

    @property
    def source_manifest_fingerprint(self) -> str:
        return fingerprint_value(self.to_fingerprint_dict())

    def to_fingerprint_dict(self) -> dict[str, Any]:
        return {
            "run": self.run_identity.to_public_dict(),
            "target_key": self.target_key,
            "window": self.window_identity.to_fingerprint_dict(),
            "source_protocol": self.source_protocol,
            "source_publication_key": self.source_publication_key,
            "source_operation": self.source_operation,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "recorder_id": self.recorder_id,
            "artifacts": [item.to_fingerprint_dict() for item in self.artifacts],
            "expected_prediction_sessions": list(self.expected_prediction_sessions),
        }


def _invalid_artifact_member() -> RollingArtifactExpectation:
    _contract("artifacts must contain RollingArtifactExpectation members")


@dataclass(frozen=True)
class RollingArtifactObservation:
    logical_key: str
    role: str
    status: str
    size_bytes: int | None = None
    fingerprint: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        _artifact_key(self.logical_key)
        if self.role not in ARTIFACT_ROLES:
            _contract("artifact observation role is invalid")
        if self.status not in ("valid", "missing", "foreign", "corrupt", "drifted", "not_comparable"):
            _contract("artifact observation status is invalid")
        if self.size_bytes is not None:
            _strict_size(self.size_bytes, "observed size_bytes")
        if self.fingerprint is not None:
            _digest(self.fingerprint, "observed fingerprint")
        if self.status == "valid" and (self.size_bytes is None or self.fingerprint is None):
            _contract("valid artifact observation requires size and fingerprint")
        if self.status == "valid" and self.detail is not None:
            _contract("valid artifact observation cannot contain a failure detail")
        if self.status != "valid" and self.detail is None:
            _contract("blocked artifact observation requires a stable detail")
        if self.status in ("missing", "foreign", "not_comparable") and (
            self.size_bytes is not None or self.fingerprint is not None
        ):
            _contract("unobserved artifact cannot carry byte facts")
        if self.detail is not None:
            _public_identifier(self.detail, "artifact detail")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            key: value for key, value in {
                "logical_key": self.logical_key, "role": self.role,
                "status": self.status, "size_bytes": self.size_bytes,
                "fingerprint": self.fingerprint, "detail": self.detail,
            }.items() if value is not None
        }


@dataclass(frozen=True)
class RollingPredictionCoverage:
    expected_sessions: tuple
    observed_sessions: tuple
    index_fingerprint: str
    row_count: int
    score_column: str

    def __post_init__(self) -> None:
        expected = tuple(normalize_iso_date(item, "expected_session") for item in _tuple(self.expected_sessions, "expected_sessions"))
        observed = tuple(normalize_iso_date(item, "observed_session") for item in _tuple(self.observed_sessions, "observed_sessions"))
        if not expected or expected != tuple(sorted(set(expected))):
            _contract("coverage expected sessions are not canonical")
        if not observed or observed != tuple(sorted(set(observed))):
            _contract("coverage observed sessions are not canonical")
        _digest(self.index_fingerprint, "index_fingerprint")
        _strict_size(self.row_count, "row_count")
        if self.row_count == 0:
            _contract("prediction coverage cannot be empty")
        _public_identifier(self.score_column, "score_column")
        object.__setattr__(self, "expected_sessions", expected)
        object.__setattr__(self, "observed_sessions", observed)

    @property
    def exact(self) -> bool:
        return self.expected_sessions == self.observed_sessions

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "expected_sessions": list(self.expected_sessions),
            "observed_sessions": list(self.observed_sessions),
            "index_fingerprint": self.index_fingerprint,
            "row_count": self.row_count,
            "score_column": self.score_column,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class RollingUnitEvidenceInspection:
    unit_key: tuple
    classification: str
    reason_code: str
    source_protocol: str
    request_fingerprint: str
    checked: tuple = ()
    source_summary: tuple = ()
    artifact_observations: tuple = ()
    prediction_coverage: RollingPredictionCoverage | None = None
    evidence_fingerprint: str | None = None
    warnings: tuple = ()
    blockers: tuple = ()
    candidate_count: int = 0
    _inspection_token: InitVar[Any] = None
    _inspector_provenance: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _inspection_token: Any) -> None:
        if not isinstance(self.unit_key, tuple) or len(self.unit_key) != 2:
            _contract("unit_key must be a target/window tuple")
        RollingTargetIdentity.parse(self.unit_key[0])
        parse_rolling_window_key(self.unit_key[1])
        if self.classification not in CLASSIFICATIONS:
            _contract("unknown evidence classification")
        if self.reason_code != CLASSIFICATION_REASONS[self.classification]:
            _contract("classification and reason_code disagree")
        if self.source_protocol not in SOURCE_PROTOCOLS:
            _contract("inspection source protocol is invalid")
        _digest(self.request_fingerprint, "request_fingerprint")
        _strict_size(self.candidate_count, "candidate_count")
        checked = _tuple(self.checked, "checked")
        if len(checked) != len(set(checked)) or any(not isinstance(item, str) or not item for item in checked):
            _contract("checked predicates must be unique non-empty strings")
        summary = _validate_source_summary(self.source_summary)
        observations = _tuple(self.artifact_observations, "artifact_observations")
        rebuilt = tuple(
            RollingArtifactObservation(
                item.logical_key, item.role, item.status, item.size_bytes,
                item.fingerprint, item.detail,
            ) if isinstance(item, RollingArtifactObservation) else _invalid_observation_member()
            for item in observations
        )
        if self.prediction_coverage is not None and not isinstance(self.prediction_coverage, RollingPredictionCoverage):
            _contract("prediction_coverage must be typed")
        logical_keys = tuple(item.logical_key for item in rebuilt)
        if len(logical_keys) != len(set(logical_keys)):
            _contract("artifact observations must have unique logical keys")
        if self.evidence_fingerprint is not None:
            _digest(self.evidence_fingerprint, "evidence_fingerprint")
        for field, values in (("warnings", self.warnings), ("blockers", self.blockers)):
            values = _tuple(values, field)
            for value in values:
                _public_identifier(value, field)
        if self.classification == "valid":
            if _inspection_token is not _INSPECTION_TOKEN:
                _contract("valid evidence can only be created by the immutable inspector")
            if self.candidate_count != 1:
                _contract("valid evidence requires exactly one observed candidate")
            if self.source_protocol != "execution_bound_v1" or self.evidence_fingerprint is None:
                _contract("valid evidence requires bound source and fingerprint")
            if self.prediction_coverage is None or not self.prediction_coverage.exact:
                _contract("valid evidence requires exact prediction coverage")
            if not rebuilt or any(item.status != "valid" for item in rebuilt):
                _contract("valid evidence requires all artifacts valid")
            if len(logical_keys) != len(set(logical_keys)) or sum(item.role == "prediction" for item in rebuilt) != 1:
                _contract("valid evidence requires unique artifacts and one prediction")
            summary_dict = dict(summary)
            if set(summary_dict) != _VALID_SUMMARY_KEYS:
                _contract("valid evidence requires a complete source summary")
            if summary_dict["source_manifest_fingerprint"] != self.request_fingerprint:
                _contract("valid source summary does not match its request")
            if (
                summary_dict["target_key"] != self.unit_key[0]
                or summary_dict["window_key"] != self.unit_key[1]
                or summary_dict["source_protocol"] != self.source_protocol
            ):
                _contract("valid source summary does not match its unit")
            _digest(summary_dict["backend_fingerprint"], "source backend fingerprint")
            for key in (
                "experiment_name", "experiment_id", "recorder_id",
                "source_publication_key", "source_operation",
            ):
                _public_identifier(summary_dict[key], "source summary %s" % key)
            if not _VALID_CHECKS.issubset(checked):
                _contract("valid evidence is missing executed predicates")
            expected_fingerprint = _unit_evidence_fingerprint(
                self.request_fingerprint, summary, rebuilt,
                self.prediction_coverage, checked,
            )
            if self.evidence_fingerprint != expected_fingerprint:
                _contract("valid evidence fingerprint does not match observed facts")
            if self.blockers:
                _contract("valid evidence cannot contain blockers")
            if self.warnings:
                _contract("valid evidence cannot contain warnings")
        else:
            if self.evidence_fingerprint is not None:
                _contract("blocked evidence cannot carry a valid evidence fingerprint")
            if not self.blockers:
                _contract("blocked evidence requires a stable blocker")
            if self.classification == "partial" and not any(item.status == "missing" for item in rebuilt):
                _contract("partial evidence requires a missing artifact observation")
            if self.classification == "coverage_short" and (
                self.prediction_coverage is None or self.prediction_coverage.exact
            ):
                _contract("coverage_short requires comparable incomplete coverage")
        object.__setattr__(self, "artifact_observations", rebuilt)
        object.__setattr__(self, "_inspector_provenance", _inspection_token is _INSPECTION_TOKEN)

    @property
    def capabilities(self) -> tuple[str, ...]:
        return ("render", "immutable_summary", "reuse_proposal", "recovery_proposal") if self.classification == "valid" else ("render",)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "unit_key": list(self.unit_key),
            "classification": self.classification,
            "reason_code": self.reason_code,
            "source_protocol": self.source_protocol,
            "request_fingerprint": self.request_fingerprint,
            "candidate_count": self.candidate_count,
            "checked": list(self.checked),
            "source_summary": dict(self.source_summary),
            "artifact_observations": [item.to_public_dict() for item in self.artifact_observations],
            "prediction_coverage": self.prediction_coverage.to_public_dict() if self.prediction_coverage else None,
            "evidence_fingerprint": self.evidence_fingerprint,
            "capabilities": list(self.capabilities),
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
        }


def _invalid_observation_member() -> RollingArtifactObservation:
    _contract("artifact observations must contain typed members")


@dataclass(frozen=True)
class RollingOrphanObservation:
    unit_key: tuple
    reason_code: str = "rolling_evidence_orphan"
    source_summary: tuple = ()
    candidate_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.unit_key, tuple) or len(self.unit_key) != 2:
            _contract("orphan unit_key must be a pair")
        RollingTargetIdentity.parse(self.unit_key[0])
        parse_rolling_window_key(self.unit_key[1])
        if self.reason_code != "rolling_evidence_orphan":
            _contract("orphan reason_code is invalid")
        if _strict_size(self.candidate_count, "orphan candidate_count") == 0:
            _contract("orphan candidate_count must be positive")
        _validate_source_summary(self.source_summary, "orphan source_summary")

    @property
    def capabilities(self) -> tuple[str, ...]:
        return ("render", "audit")


@dataclass(frozen=True)
class RollingEvidenceSetInspection:
    requested_unit_keys: tuple
    unit_results: tuple
    orphan_observations: tuple
    inventory_before_fingerprint: str
    inventory_after_fingerprint: str
    status: str
    reason_code: str

    def __post_init__(self) -> None:
        requested = _validate_requested_keys(self.requested_unit_keys)
        results = _tuple(self.unit_results, "unit_results")
        rebuilt = tuple(_rebuild_result(item) for item in results)
        if tuple(item.unit_key for item in rebuilt) != requested:
            _contract("evidence results must exactly preserve requested identity and order")
        orphans = _tuple(self.orphan_observations, "orphan_observations")
        rebuilt_orphans = tuple(_rebuild_orphan(item) for item in orphans)
        orphan_keys = tuple(item.unit_key for item in rebuilt_orphans)
        if len(orphan_keys) != len(set(orphan_keys)) or set(orphan_keys).intersection(requested):
            _contract("orphan observations must be unique and outside requested scope")
        _digest(self.inventory_before_fingerprint, "inventory_before_fingerprint")
        _digest(self.inventory_after_fingerprint, "inventory_after_fingerprint")
        expected_status = _set_status(rebuilt, self.inventory_before_fingerprint, self.inventory_after_fingerprint)
        if self.status != expected_status or self.reason_code != SET_REASONS[expected_status]:
            _contract("evidence set status/reason does not match its members")
        object.__setattr__(self, "requested_unit_keys", requested)
        object.__setattr__(self, "unit_results", rebuilt)
        object.__setattr__(self, "orphan_observations", rebuilt_orphans)

    @property
    def n_requested(self) -> int:
        return len(self.unit_results)

    @property
    def n_valid(self) -> int:
        return sum(item.classification == "valid" for item in self.unit_results)

    @property
    def n_blocked(self) -> int:
        return self.n_requested - self.n_valid

    @property
    def n_missing(self) -> int:
        return sum(item.classification == "missing" for item in self.unit_results)

    @property
    def n_orphan(self) -> int:
        return len(self.orphan_observations)

    @property
    def n_candidates(self) -> int:
        return sum(item.candidate_count for item in self.unit_results) + sum(
            item.candidate_count for item in self.orphan_observations
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "requested_unit_keys": [list(item) for item in self.requested_unit_keys],
            "unit_results": [item.to_public_dict() for item in self.unit_results],
            "orphan_unit_keys": [list(item.unit_key) for item in self.orphan_observations],
            "inventory_before_fingerprint": self.inventory_before_fingerprint,
            "inventory_after_fingerprint": self.inventory_after_fingerprint,
            "status": self.status,
            "reason_code": self.reason_code,
            "n_requested": self.n_requested,
            "n_valid": self.n_valid,
            "n_blocked": self.n_blocked,
            "n_missing": self.n_missing,
            "n_orphan": self.n_orphan,
            "n_candidates": self.n_candidates,
            "fingerprint": self.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_value({
            "requested_unit_keys": [list(item) for item in self.requested_unit_keys],
            "unit_evidence": [item.evidence_fingerprint for item in self.unit_results],
            "unit_request_fingerprints": [item.request_fingerprint for item in self.unit_results],
            "unit_classifications": [item.classification for item in self.unit_results],
            "unit_candidate_counts": [item.candidate_count for item in self.unit_results],
            "orphan_unit_keys": [list(item.unit_key) for item in self.orphan_observations],
            "orphan_candidate_counts": [item.candidate_count for item in self.orphan_observations],
            "inventory_before": self.inventory_before_fingerprint,
            "inventory_after": self.inventory_after_fingerprint,
            "status": self.status,
        })


def _validate_requested_keys(value: Any) -> tuple:
    keys = _tuple(value, "requested_unit_keys")
    if not keys:
        _contract("requested evidence scope cannot be empty")
    for item in keys:
        if not isinstance(item, tuple) or len(item) != 2:
            _contract("requested unit keys must be pairs")
        RollingTargetIdentity.parse(item[0])
        family, _, _, _ = parse_rolling_window_key(item[1])
        if RollingTargetIdentity.parse(item[0]).family != family:
            _contract("requested target/window families disagree")
    if len(keys) != len(set(keys)):
        _contract("requested evidence scope contains duplicate units")
    return keys


def _rebuild_result(item: Any) -> RollingUnitEvidenceInspection:
    if not isinstance(item, RollingUnitEvidenceInspection):
        _contract("unit_results must contain typed inspections")
    if item.classification == "valid" and not item._inspector_provenance:
        _contract("valid evidence lacks immutable inspector provenance")
    return RollingUnitEvidenceInspection(
        item.unit_key, item.classification, item.reason_code, item.source_protocol,
        item.request_fingerprint,
        item.checked, item.source_summary, item.artifact_observations,
        item.prediction_coverage, item.evidence_fingerprint, item.warnings,
        item.blockers, item.candidate_count,
        _inspection_token=_INSPECTION_TOKEN if item._inspector_provenance else None,
    )


def _rebuild_orphan(item: Any) -> RollingOrphanObservation:
    if not isinstance(item, RollingOrphanObservation):
        _contract("orphan_observations must contain typed members")
    return RollingOrphanObservation(
        item.unit_key, item.reason_code, item.source_summary, item.candidate_count,
    )


def _set_status(results: tuple, before: str, after: str) -> str:
    if before != after:
        return "observation_drifted"
    if any(item.classification == "drifted" for item in results):
        return "incomplete"
    valid = sum(item.classification == "valid" for item in results)
    if valid == len(results):
        return "all_valid"
    return "none_valid" if valid == 0 else "incomplete"


class RollingEvidenceBackend(Protocol):
    """Metadata-only backend.  Implementations must not initialize or mutate."""

    def tracking_identity(self) -> Mapping[str, Any]: ...
    def inventory(self, requests: tuple[RollingUnitEvidenceRequest, ...]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _FileSnapshot:
    data: bytes
    size_bytes: int
    fingerprint: str
    device: int
    inode: int


class _RestrictedPredictionUnpickler(pickle.Unpickler):
    """Load the narrow pandas/numpy object graph used by prediction artifacts."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _SAFE_PREDICTION_PICKLE_GLOBALS:
            raise pickle.UnpicklingError("prediction pickle references a forbidden global")
        return super().find_class(module, name)


def _load_prediction_pickle(data: bytes) -> Any:
    return _RestrictedPredictionUnpickler(io.BytesIO(data)).load()


def _root_identity(root: Path) -> tuple[int, int] | None:
    try:
        node = os.lstat(root)
    except OSError:
        return None
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        return None
    return (node.st_dev, node.st_ino)


def _inventory_snapshot(backend: RollingEvidenceBackend, requests: tuple) -> tuple[str, tuple]:
    raw = _mapping(backend.inventory(requests), "backend inventory")
    token = _digest(raw.get("fingerprint"), "inventory fingerprint")
    candidates = raw.get("candidates", ())
    if not isinstance(candidates, (tuple, list)):
        _contract("inventory candidates must be ordered")
    return token, tuple(dict(_mapping(item, "inventory candidate")) for item in candidates)


def _candidate_inventory_fingerprint(candidates: tuple) -> str:
    keys = (
        "workspace_fingerprint", "backend_fingerprint", "experiment_name", "experiment_id",
        "recorder_id", "run_fingerprint", "attempt_id", "plan_fingerprint",
        "config_fingerprint", "target_key", "window_key", "source_protocol",
        "source_publication_key", "source_operation", "source_manifest_fingerprint",
        "artifact_root_uri",
    )
    def safe_value(value):
        if value is None or type(value) in (bool, int, float, str):
            return value
        return {"unsupported_candidate_value": True}
    return fingerprint_value([
        {key: safe_value(candidate.get(key)) for key in keys}
        for candidate in candidates
    ])


def _candidate_unit(candidate: Mapping[str, Any]) -> tuple[str, str] | None:
    target = candidate.get("target_key")
    window = candidate.get("window_key")
    if not isinstance(target, str) or not isinstance(window, str):
        return None
    return (target, window)


def _candidate_summary(candidate: Mapping[str, Any]) -> tuple:
    allowed = (
        "backend_fingerprint", "experiment_name", "experiment_id", "recorder_id",
        "target_key", "window_key", "source_protocol", "source_publication_key",
        "source_operation", "source_manifest_fingerprint",
    )
    summary = []
    for key in allowed:
        value = candidate.get(key)
        if isinstance(value, str):
            try:
                value = _public_identifier(value, "candidate summary")
            except RollingEvidenceContractError:
                value = None
        elif value is not None:
            value = None
        summary.append((key, value))
    return tuple(summary)


def _candidate_relevant(candidate: Mapping[str, Any], request: RollingUnitEvidenceRequest) -> bool:
    return (
        _candidate_unit(candidate) == request.unit_key
        or candidate.get("recorder_id") == request.recorder_id
        or candidate.get("source_manifest_fingerprint") == request.source_manifest_fingerprint
    )


def _identity_mismatch(candidate: Mapping[str, Any], request: RollingUnitEvidenceRequest, backend_identity: Mapping[str, Any]) -> bool:
    expected = {
        "workspace_fingerprint": request.run_identity.workspace_fingerprint,
        "backend_fingerprint": backend_identity.get("backend_fingerprint"),
        "experiment_name": request.experiment_name,
        "experiment_id": request.experiment_id,
        "recorder_id": request.recorder_id,
        "run_fingerprint": request.run_identity.fingerprint,
        "attempt_id": request.run_identity.attempt_id,
        "plan_fingerprint": request.run_identity.plan_fingerprint,
        "config_fingerprint": request.run_identity.config_fingerprint,
        "target_key": request.target_key,
        "window_key": request.window_key,
        "source_protocol": request.source_protocol,
        "source_publication_key": request.source_publication_key,
        "source_operation": request.source_operation,
        "source_manifest_fingerprint": request.source_manifest_fingerprint,
    }
    return any(candidate.get(key) != value for key, value in expected.items())


def _artifact_root(candidate: Mapping[str, Any], root: Path) -> tuple[Path | None, str | None]:
    uri = candidate.get("artifact_root_uri")
    if not isinstance(uri, str) or not uri:
        return None, "not_comparable"
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file") or (parsed.scheme == "file" and parsed.netloc not in ("", "localhost")):
        return None, "not_comparable"
    path = Path(unquote(parsed.path if parsed.scheme else uri))
    if not path.is_absolute():
        return None, "not_comparable"
    try:
        relative = path.relative_to(root)
        if any(part in ("", ".", "..") for part in relative.parts):
            return None, "foreign"
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return None, "foreign"
    return path, None


def _secure_read(
    root: Path,
    artifact_root: Path,
    logical_key: str,
) -> tuple[_FileSnapshot | None, str, str | None, tuple[str, ...]]:
    candidate = artifact_root.joinpath(*PurePosixPath(logical_key).parts)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None, "foreign", "artifact_outside_workspace", ()
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        return None, "foreign", "artifact_outside_workspace", ()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    opened_fds = []
    checked = []
    def done(snapshot, status, detail, predicates):
        for opened_fd in reversed(opened_fds):
            try:
                os.close(opened_fd)
            except OSError:
                pass
        opened_fds.clear()
        return snapshot, status, detail, predicates
    try:
        root_stat = os.lstat(root)
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            return done(None, "foreign", "workspace_root_noncanonical", ())
        root_fd = os.open(root, directory_flags)
        opened_fds.append(root_fd)
        opened_root = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or (opened_root.st_dev, opened_root.st_ino) != (root_stat.st_dev, root_stat.st_ino)
        ):
            return done(None, "drifted", "workspace_root_drift", ())
        parent_fd = root_fd
        for part in relative.parts[:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            except OSError:
                public_part = root.joinpath(*relative.parts[:len(opened_fds)])
                try:
                    part_node = os.lstat(public_part)
                    if stat.S_ISLNK(part_node.st_mode):
                        try:
                            public_part.resolve().relative_to(root)
                        except (OSError, ValueError):
                            return done(None, "foreign", "artifact_symlink_escape", tuple(checked))
                        return done(None, "corrupt", "artifact_symlink_component", tuple(checked))
                except OSError:
                    pass
                raise
            opened_fds.append(next_fd)
            parent_fd = next_fd
            if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
                return done(None, "corrupt", "artifact_ancestor_not_directory", tuple(checked))
        try:
            fd = os.open(relative.parts[-1], file_flags, dir_fd=parent_fd)
        except OSError:
            try:
                node = os.stat(relative.parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                checked.append("artifact_node_kind")
                if stat.S_ISLNK(node.st_mode):
                    return done(None, "corrupt", "artifact_symlink_component", tuple(checked))
                if not stat.S_ISREG(node.st_mode):
                    return done(None, "corrupt", "artifact_node_not_regular", tuple(checked))
            except FileNotFoundError:
                return done(None, "missing", "artifact_missing", tuple(checked))
            raise
        opened_fds.append(fd)
        opened = os.fstat(fd)
        checked.append("artifact_node_kind")
        if not stat.S_ISREG(opened.st_mode):
            return done(None, "corrupt", "artifact_node_not_regular", tuple(checked))
    except FileNotFoundError:
        return done(None, "missing", "artifact_missing", tuple(checked))
    except OSError:
        return done(None, "corrupt", "artifact_path_unreadable", tuple(checked))
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        completed = os.fstat(fd)
        checked.append("artifact_byte_fingerprint")
    except OSError:
        return done(None, "corrupt", "artifact_read_failed", tuple(checked))
    finally:
        for opened_fd in reversed(opened_fds):
            try:
                os.close(opened_fd)
            except OSError:
                pass
        opened_fds.clear()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened, field) != getattr(completed, field) for field in stable_fields):
        return None, "drifted", "artifact_bytes_drift", tuple(checked)
    data = b"".join(chunks)
    return (
        _FileSnapshot(data, len(data), hashlib.sha256(data).hexdigest(), opened.st_dev, opened.st_ino),
        "valid", None, tuple(checked),
    )


def _decode_prediction(
    data: bytes,
    expected: tuple,
) -> tuple[RollingPredictionCoverage | None, str | None, str | None, tuple[str, ...]]:
    try:
        import pandas as pd

        payload = _load_prediction_pickle(data)
    except Exception:
        return None, "corrupt", "prediction_decode_failed", ()
    checked = ["prediction_schema"]
    if isinstance(payload, pd.Series):
        column = str(payload.name if payload.name is not None else "score")
        scores = payload
    elif isinstance(payload, pd.DataFrame) and len(payload.columns) == 1:
        column = str(payload.columns[0])
        scores = payload.iloc[:, 0]
    else:
        return None, "not_comparable", "prediction_schema_invalid", tuple(checked)
    index = payload.index
    if len(payload) == 0 or not isinstance(index, pd.MultiIndex):
        return None, "not_comparable", "prediction_schema_invalid", tuple(checked)
    names = tuple(str(item).lower() if item is not None else "" for item in index.names)
    datetime_positions = [i for i, name in enumerate(names) if name in ("datetime", "date", "session")]
    instrument_positions = [i for i, name in enumerate(names) if name in ("instrument", "symbol", "code")]
    if len(datetime_positions) != 1 or len(instrument_positions) != 1:
        return None, "not_comparable", "prediction_schema_invalid", tuple(checked)
    dt_pos, instrument_pos = datetime_positions[0], instrument_positions[0]
    try:
        raw_datetimes = index.get_level_values(dt_pos)
        if any(isinstance(item, (bool, int, float)) for item in raw_datetimes):
            return None, "corrupt", "prediction_schema_invalid", tuple(checked)
        datetimes = pd.to_datetime(raw_datetimes, errors="raise")
        if getattr(datetimes, "tz", None) is not None:
            return None, "not_comparable", "prediction_schema_invalid", tuple(checked)
        if any(item != item.normalize() for item in datetimes):
            return None, "corrupt", "prediction_schema_invalid", tuple(checked)
        sessions = tuple(item.date().isoformat() for item in datetimes)
        instruments = tuple(str(item) for item in index.get_level_values(instrument_pos))
        if not pd.api.types.is_numeric_dtype(scores.dtype) or pd.api.types.is_bool_dtype(scores.dtype):
            return None, "corrupt", "prediction_schema_invalid", tuple(checked)
        numeric = pd.to_numeric(scores, errors="raise")
    except Exception:
        return None, "corrupt", "prediction_schema_invalid", tuple(checked)
    canonical_index = tuple(zip(sessions, instruments))
    if any(not instrument or instrument != instrument.strip() for instrument in instruments):
        return None, "corrupt", "prediction_schema_invalid", tuple(checked)
    checked.append("prediction_index_unique")
    if len(canonical_index) != len(set(canonical_index)):
        return None, "corrupt", "prediction_index_duplicate", tuple(checked)
    if canonical_index != tuple(sorted(canonical_index)):
        return None, "corrupt", "prediction_index_noncanonical", tuple(checked)
    checked.append("prediction_scores_finite")
    try:
        if any(not math.isfinite(float(value)) for value in numeric):
            return None, "corrupt", "prediction_score_non_finite", tuple(checked)
    except (TypeError, ValueError):
        return None, "corrupt", "prediction_schema_invalid", tuple(checked)
    observed = tuple(sorted(set(sessions)))
    try:
        coverage = RollingPredictionCoverage(
            expected, observed, fingerprint_value([list(item) for item in canonical_index]),
            len(canonical_index), column,
        )
    except RollingEvidenceContractError:
        return None, "not_comparable", "prediction_schema_invalid", tuple(checked)
    checked.append("prediction_session_coverage")
    if observed == expected:
        return coverage, None, None, tuple(checked)
    expected_set, observed_set = set(expected), set(observed)
    if not observed_set.issubset(expected_set):
        return coverage, "identity_mismatch", "prediction_out_of_window", tuple(checked)
    if observed and observed[-1] != expected[-1]:
        detail = "prediction_tail_missing"
    elif observed and observed[0] != expected[0]:
        detail = "prediction_head_missing"
    elif observed_set != expected_set:
        detail = "prediction_internal_gap"
    else:
        detail = "prediction_session_subset"
    return coverage, "coverage_short", detail, tuple(checked)


def _unit_result(
    request: RollingUnitEvidenceRequest,
    classification: str,
    *,
    checked: Sequence[str] = (),
    summary: tuple = (),
    observations: tuple = (),
    coverage: RollingPredictionCoverage | None = None,
    detail: str | None = None,
    candidate_count: int = 0,
) -> RollingUnitEvidenceInspection:
    evidence_fingerprint = None
    blockers = ()
    warnings = ()
    if classification == "valid":
        evidence_fingerprint = _unit_evidence_fingerprint(
            request.source_manifest_fingerprint, summary, observations,
            coverage, tuple(checked),
        )
    else:
        blockers = (detail or CLASSIFICATION_REASONS[classification],)
        if classification == "legacy_unverified":
            warnings = ("source manifest is not execution-bound",)
    return RollingUnitEvidenceInspection(
        request.unit_key, classification, CLASSIFICATION_REASONS[classification],
        request.source_protocol, request.source_manifest_fingerprint,
        tuple(checked), summary, observations, coverage,
        evidence_fingerprint, warnings, blockers, candidate_count,
        _inspection_token=_INSPECTION_TOKEN,
    )


def _inspect_candidate(
    root: Path,
    request: RollingUnitEvidenceRequest,
    candidate: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    initial_checked: Sequence[str],
) -> RollingUnitEvidenceInspection:
    summary = _candidate_summary(candidate)
    checked = list(initial_checked)
    if request.source_protocol == "legacy_unverified":
        return _unit_result(
            request, "legacy_unverified", checked=checked, summary=summary,
            candidate_count=1,
        )
    checked.append("source_identity")
    if _identity_mismatch(candidate, request, backend_identity):
        return _unit_result(
            request, "identity_mismatch", checked=checked, summary=summary,
            candidate_count=1,
        )
    artifact_root, issue = _artifact_root(candidate, root)
    if issue:
        return _unit_result(
            request, issue, checked=checked, summary=summary, candidate_count=1,
        )
    checked.append("artifact_root_containment")
    observations = []
    physical_snapshots = {}
    coverage = None
    classification = "valid"
    detail = None
    for expectation in request.artifacts:
        snapshot, status, node_detail, read_checked = _secure_read(
            root, artifact_root, expectation.logical_key,
        )
        checked.extend(read_checked)
        if status != "valid":
            artifact_status = "missing" if status == "missing" else status
            observations.append(RollingArtifactObservation(
                expectation.logical_key, expectation.role, artifact_status, detail=node_detail,
            ))
            candidate_classification = "partial" if status == "missing" else status
            preferred = _prefer(classification, candidate_classification)
            if preferred != classification:
                classification, detail = preferred, node_detail
            continue
        artifact_status = "valid"
        physical_snapshots[expectation.logical_key] = snapshot
        artifact_detail = None
        if snapshot.size_bytes != expectation.size_bytes:
            artifact_status, artifact_detail = "corrupt", "artifact_size_mismatch"
        elif snapshot.fingerprint != expectation.fingerprint:
            artifact_status, artifact_detail = "corrupt", "artifact_digest_mismatch"
        observations.append(RollingArtifactObservation(
            expectation.logical_key, expectation.role, artifact_status,
            snapshot.size_bytes, snapshot.fingerprint, artifact_detail,
        ))
        if artifact_status != "valid":
            preferred = _prefer(classification, "corrupt")
            if preferred != classification:
                classification, detail = preferred, artifact_detail
        if expectation.role == "prediction" and artifact_status == "valid":
            coverage, coverage_class, coverage_detail, prediction_checked = _decode_prediction(
                snapshot.data, request.expected_prediction_sessions,
            )
            checked.extend(prediction_checked)
            if coverage_class:
                preferred = _prefer(classification, coverage_class)
                if preferred != classification:
                    classification, detail = preferred, coverage_detail
    # Re-establish every canonical public artifact name and exact byte fact after
    # content predicates.  An open descriptor alone is not namespace authority.
    for expectation, observation in zip(request.artifacts, observations):
        if observation.status != "valid":
            continue
        rechecked, status, recheck_detail, _ = _secure_read(
            root, artifact_root, expectation.logical_key,
        )
        checked.append("artifact_public_path_recheck")
        if (
            status != "valid" or rechecked.size_bytes != observation.size_bytes
            or rechecked.fingerprint != observation.fingerprint
            or (rechecked.device, rechecked.inode) != (
                physical_snapshots[expectation.logical_key].device,
                physical_snapshots[expectation.logical_key].inode,
            )
        ):
            classification, detail = "drifted", recheck_detail or "artifact_bytes_drift"
    return _unit_result(
        request, classification, checked=tuple(dict.fromkeys(checked)), summary=summary,
        observations=tuple(observations), coverage=coverage, detail=detail,
        candidate_count=1,
    )


def _validate_requests(context: WorkspaceContext, requests: Any) -> tuple:
    if not isinstance(context, WorkspaceContext):
        _contract("context must be an explicit WorkspaceContext")
    requests = _tuple(requests, "requests")
    if not requests:
        _contract("requests cannot be empty")
    rebuilt = []
    for item in requests:
        if not isinstance(item, RollingUnitEvidenceRequest):
            _contract("requests must contain RollingUnitEvidenceRequest members")
        rebuilt.append(RollingUnitEvidenceRequest(
            item.run_identity, item.target_key, item.window_identity, item.source_protocol,
            item.source_publication_key, item.source_operation, item.experiment_name,
            item.experiment_id, item.recorder_id, item.artifacts,
            item.expected_prediction_sessions,
        ))
    rebuilt = tuple(rebuilt)
    keys = tuple(item.unit_key for item in rebuilt)
    if len(keys) != len(set(keys)):
        _contract("requests contain duplicate units")
    run_fingerprints = {item.run_identity.fingerprint for item in rebuilt}
    if len(run_fingerprints) != 1:
        _contract("requests must belong to one Rolling run identity")
    root = Path(context.root)
    if root != root.resolve() or context.data_dir != root / "data":
        _contract("workspace context is not canonical")
    if workspace_fingerprint(root) != rebuilt[0].run_identity.workspace_fingerprint:
        _contract("workspace context does not match run identity")
    return rebuilt


def inspect_rolling_evidence(context, requests, backend):
    """Return one :class:`RollingEvidenceSetInspection` without mutation."""

    requests = _validate_requests(context, requests)
    root = Path(context.root)
    root_before = _root_identity(root)
    backend_identity_observed = False
    try:
        backend_identity = dict(_mapping(backend.tracking_identity(), "tracking identity"))
        backend_identity_observed = True
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        backend_identity = {}
    expected_workspace = requests[0].run_identity.workspace_fingerprint
    backend_fp = backend_identity.get("backend_fingerprint")
    workspace_fp = backend_identity.get("workspace_fingerprint")
    identity_checked = backend_identity_observed and (
        backend_identity.get("present") is False
        or (
            isinstance(workspace_fp, str)
            and len(workspace_fp) == 64
            and not any(char not in _DIGEST_CHARS for char in workspace_fp)
            and isinstance(backend_fp, str)
            and len(backend_fp) == 64
            and not any(char not in _DIGEST_CHARS for char in backend_fp)
            and type(backend_identity.get("present")) is bool
            and type(backend_identity.get("contained")) is bool
        )
    )
    identity_comparable = (
        identity_checked
        and workspace_fp == expected_workspace
        and backend_identity.get("present") is True
        and backend_identity.get("contained") is True
        and not backend_identity.get("foreign", False)
        and root_before is not None
    )
    inventory_before_observed = False
    try:
        before, candidates = _inventory_snapshot(backend, requests)
        inventory_before_observed = True
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        before, candidates = fingerprint_value({"inventory": "unavailable"}), ()
        identity_comparable = False

    results = []
    for request in requests:
        matches = [item for item in candidates if _candidate_relevant(item, request)]
        checked = []
        if identity_checked:
            checked.append("tracking_identity")
        if inventory_before_observed:
            checked.append("candidate_cardinality")
        if not identity_comparable:
            is_foreign = (
                backend_identity.get("foreign", False)
                or backend_identity.get("contained") is False
                or (
                    backend_identity.get("workspace_fingerprint") is not None
                    and backend_identity.get("workspace_fingerprint") != expected_workspace
                )
            )
            classification = (
                "foreign" if is_foreign
                else "missing" if backend_identity.get("present") is False
                else "not_comparable"
            )
            results.append(_unit_result(
                request, classification, checked=checked,
                candidate_count=len(matches) if inventory_before_observed else 0,
            ))
            continue
        if not matches:
            results.append(_unit_result(request, "missing", checked=checked))
        elif len(matches) > 1:
            results.append(_unit_result(
                request, "duplicate", checked=checked, candidate_count=len(matches),
            ))
        else:
            try:
                result = _inspect_candidate(
                    root, request, matches[0], backend_identity, checked,
                )
            except (RollingEvidenceContractError, RollingIdentityError, OSError, TypeError, ValueError):
                result = _unit_result(
                    request, "not_comparable", checked=checked,
                    detail="candidate_metadata_invalid", candidate_count=1,
                )
            results.append(result)

    requested_keys = tuple(item.unit_key for item in requests)
    orphans = []
    orphan_candidates = {}
    for candidate in candidates:
        unit = _candidate_unit(candidate)
        if unit is None or unit in requested_keys:
            continue
        orphan_candidates.setdefault(unit, []).append(candidate)
    for unit, grouped in orphan_candidates.items():
        try:
            orphan = RollingOrphanObservation(
                unit, source_summary=_candidate_summary(grouped[0]),
                candidate_count=len(grouped),
            )
        except (RollingEvidenceContractError, RollingIdentityError):
            continue
        orphans.append(orphan)
    inventory_after_observed = False
    try:
        after, after_candidates = _inventory_snapshot(backend, requests)
        inventory_after_observed = True
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        after, after_candidates = fingerprint_value({"inventory": "unavailable"}), ()
    after_identity_observed = False
    try:
        after_identity = dict(_mapping(backend.tracking_identity(), "tracking identity after"))
        after_identity_observed = True
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        after_identity = {}
    after_backend_fp = after_identity.get("backend_fingerprint")
    after_workspace_fp = after_identity.get("workspace_fingerprint")
    after_identity_checked = after_identity_observed and (
        after_identity.get("present") is False
        or (
            isinstance(after_workspace_fp, str)
            and len(after_workspace_fp) == 64
            and not any(char not in _DIGEST_CHARS for char in after_workspace_fp)
            and isinstance(after_backend_fp, str)
            and len(after_backend_fp) == 64
            and not any(char not in _DIGEST_CHARS for char in after_backend_fp)
            and type(after_identity.get("present")) is bool
            and type(after_identity.get("contained")) is bool
        )
    )
    root_after = _root_identity(root)
    inventory_changed = (
        inventory_before_observed and inventory_after_observed
        and (
            before != after
            or _candidate_inventory_fingerprint(candidates) != _candidate_inventory_fingerprint(after_candidates)
        )
    )
    availability_changed = (
        inventory_before_observed != inventory_after_observed
        or backend_identity_observed != after_identity_observed
    )
    identity_changed = (
        backend_identity_observed and after_identity_observed
        and backend_identity != after_identity
    )
    root_changed = root_before != root_after
    if inventory_changed or availability_changed or identity_changed or root_changed:
        after = fingerprint_value({
            "token": after,
            "candidate_fingerprint": _candidate_inventory_fingerprint(after_candidates),
        })
        drift_checked = []
        if inventory_before_observed and inventory_after_observed:
            drift_checked.append("inventory_before_after")
        if identity_checked and after_identity_checked:
            drift_checked.append("tracking_identity")
        if root_before is not None and root_after is not None:
            drift_checked.append("workspace_root_recheck")
        drift_detail = "inventory_changed" if inventory_changed else "observation_unavailable" if availability_changed else "backend_identity_changed" if identity_changed else "workspace_root_changed"
        results = [
            _unit_result(
                request, "drifted",
                checked=tuple(dict.fromkeys(result.checked + tuple(drift_checked))),
                detail=drift_detail,
                candidate_count=result.candidate_count,
            )
            for request, result in zip(requests, results)
        ]
    status = _set_status(tuple(results), before, after)
    return RollingEvidenceSetInspection(
        requested_keys, tuple(results), tuple(orphans), before, after,
        status, SET_REASONS[status],
    )
