"""Strict, immutable contracts for model capability inspection.

The public types in this module distinguish a valid representation from an
observed fact.  In particular, only :mod:`quantpits.model_capabilities.inspector`
can create checked predicate facts, authoritative terminal rows, or aggregate
counts.  Deserialised values are audit snapshots and never regain preflight
authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


WRAPPER_KINDS = ("custom", "loss_history", "external_passthrough")
DATASET_PROTOCOLS = (
    "point_in_time", "time_series", "memory_time_series", "daily_market_label", "multi_label",
)
ACTIONS = ("train", "incremental", "predict_only", "resume")
EXECUTION_FAMILIES = ("static", "cpcv", "rolling", "cpcv_rolling")
PROCESSOR_PROFILES = (
    "standard_infer_no_label_drop", "sequence_infer_no_label_drop",
    "memory_infer_tail_preserved", "daily_market_label_infer_tail_preserved",
    "multi_label_cardinality_preserved", "safe_inference",
)
ARTIFACT_PROTOCOLS = ("qlib_recorder_model_v1", "artifact_v1")
DEPENDENCY_PROFILES = ("python_qlib_torch", "python_qlib", "python_example", "python_gpu")
TERMINAL_STATUSES = (
    "supported_verified", "unsupported", "conditional", "coverage_unsafe",
    "not_comparable", "invalid_declaration", "probe_failed",
)
AGGREGATE_STATUSES = (
    "all_required_supported", "partially_supported", "none_supported", "inventory_invalid",
)
PREDICATE_OUTCOMES = ("passed", "failed", "not_checked")
OBSERVATION_KINDS = (
    "actual_class_static_observation", "actual_wrapper_generated_protocol_probe",
    "harness_self_test_only", "audit_replay",
)

_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_AUTHORITY = object()


class ModelCapabilityContractError(ValueError):
    """Raised when a capability representation violates the public contract."""


def _contract(message: str) -> None:
    raise ModelCapabilityContractError(message)


def _public_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _contract("%s must be a non-empty trimmed string" % field_name)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _contract("%s contains a control character" % field_name)
    lowered = value.lower()
    windows_path = len(value) > 2 and value[0].isalpha() and value[1] == ":" and value[2] in ("/", "\\")
    if "/" in value or "\\" in value or windows_path or "://" in value or lowered.startswith("file:"):
        _contract("%s must not contain an absolute path or URI" % field_name)
    return value


def _module(value: Any, field_name: str) -> str:
    value = _public_text(value, field_name)
    if not _MODULE_RE.fullmatch(value):
        _contract("%s must be a canonical dotted module name" % field_name)
    return value


def _class_name(value: Any, field_name: str) -> str:
    value = _public_text(value, field_name)
    if not _CLASS_RE.fullmatch(value):
        _contract("%s must be a canonical class name" % field_name)
    return value


def _enum(value: Any, values: Tuple[str, ...], field_name: str) -> str:
    if not isinstance(value, str) or value not in values:
        _contract("%s has an unsupported representation" % field_name)
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        _contract("%s must be a boolean" % field_name)
    return value


def _ordered_predicates(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, tuple):
        _contract("required_predicates must be an ordered tuple")
    if not value or len(value) != len(set(value)):
        _contract("required_predicates must be non-empty and unique")
    for item in value:
        if not isinstance(item, str) or not _PREDICATE_RE.fullmatch(item):
            _contract("required_predicates contains an invalid predicate name")
    return value


def _json_fact(value: Any, field_name: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        if isinstance(value, str):
            return _public_text(value, field_name)
        return value
    if isinstance(value, tuple):
        return tuple(_json_fact(item, field_name) for item in value)
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            key = _public_text(key, "%s key" % field_name)
            converted[key] = _json_fact(item, field_name)
        return converted
    _contract("%s must be a JSON-safe public fact" % field_name)


def _stable_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RawModelCapabilityDeclaration:
    model_module: str
    model_class: str
    wrapper_kind: str
    dataset_module: str
    dataset_class: str
    dataset_protocol: str
    action: str
    execution_family: str
    processor_profile: str
    artifact_protocol: str
    dependency_profile: str
    required_predicates: Tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_module", _module(self.model_module, "model_module"))
        object.__setattr__(self, "model_class", _class_name(self.model_class, "model_class"))
        object.__setattr__(self, "wrapper_kind", _enum(self.wrapper_kind, WRAPPER_KINDS, "wrapper_kind"))
        object.__setattr__(self, "dataset_module", _module(self.dataset_module, "dataset_module"))
        object.__setattr__(self, "dataset_class", _class_name(self.dataset_class, "dataset_class"))
        object.__setattr__(self, "dataset_protocol", _enum(self.dataset_protocol, DATASET_PROTOCOLS, "dataset_protocol"))
        object.__setattr__(self, "action", _enum(self.action, ACTIONS, "action"))
        object.__setattr__(self, "execution_family", _enum(self.execution_family, EXECUTION_FAMILIES, "execution_family"))
        object.__setattr__(self, "processor_profile", _enum(self.processor_profile, PROCESSOR_PROFILES, "processor_profile"))
        object.__setattr__(self, "artifact_protocol", _enum(self.artifact_protocol, ARTIFACT_PROTOCOLS, "artifact_protocol"))
        object.__setattr__(self, "dependency_profile", _enum(self.dependency_profile, DEPENDENCY_PROFILES, "dependency_profile"))
        object.__setattr__(self, "required_predicates", _ordered_predicates(self.required_predicates))
        object.__setattr__(self, "required", _strict_bool(self.required, "required"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RawModelCapabilityDeclaration":
        if not isinstance(value, Mapping):
            _contract("raw declaration must be a mapping")
        expected = {
            "model_module", "model_class", "wrapper_kind", "dataset_module", "dataset_class",
            "dataset_protocol", "action", "execution_family", "processor_profile",
            "artifact_protocol", "dependency_profile", "required_predicates", "required",
        }
        if set(value) != expected:
            _contract("raw declaration fields must match the strict schema")
        kwargs = dict(value)
        predicates = kwargs["required_predicates"]
        if not isinstance(predicates, (list, tuple)):
            _contract("required_predicates must be an ordered sequence")
        kwargs["required_predicates"] = tuple(predicates)
        return cls(**kwargs)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "model_module": self.model_module,
            "model_class": self.model_class,
            "wrapper_kind": self.wrapper_kind,
            "dataset_module": self.dataset_module,
            "dataset_class": self.dataset_class,
            "dataset_protocol": self.dataset_protocol,
            "action": self.action,
            "execution_family": self.execution_family,
            "processor_profile": self.processor_profile,
            "artifact_protocol": self.artifact_protocol,
            "dependency_profile": self.dependency_profile,
            "required_predicates": list(self.required_predicates),
            "required": self.required,
        }


@dataclass(frozen=True)
class ModelCapabilityIdentity:
    model_module: str
    model_class: str
    wrapper_kind: str
    dataset_module: str
    dataset_class: str
    dataset_protocol: str
    action: str
    execution_family: str
    processor_profile: str
    artifact_protocol: str
    dependency_profile: str

    def __post_init__(self) -> None:
        rebuilt = RawModelCapabilityDeclaration(
            self.model_module, self.model_class, self.wrapper_kind,
            self.dataset_module, self.dataset_class, self.dataset_protocol,
            self.action, self.execution_family, self.processor_profile,
            self.artifact_protocol, self.dependency_profile, ("identity_canonical",), True,
        )
        for field_name in (
            "model_module", "model_class", "wrapper_kind", "dataset_module", "dataset_class",
            "dataset_protocol", "action", "execution_family", "processor_profile",
            "artifact_protocol", "dependency_profile",
        ):
            object.__setattr__(self, field_name, getattr(rebuilt, field_name))

    @classmethod
    def from_declaration(cls, declaration: RawModelCapabilityDeclaration) -> "ModelCapabilityIdentity":
        if not isinstance(declaration, RawModelCapabilityDeclaration):
            _contract("identity requires a canonical raw declaration")
        return cls(**{
            key: declaration.to_public_dict()[key]
            for key in (
                "model_module", "model_class", "wrapper_kind", "dataset_module", "dataset_class",
                "dataset_protocol", "action", "execution_family", "processor_profile",
                "artifact_protocol", "dependency_profile",
            )
        })

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint(self.to_public_dict())

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "model_module": self.model_module, "model_class": self.model_class,
            "wrapper_kind": self.wrapper_kind, "dataset_module": self.dataset_module,
            "dataset_class": self.dataset_class, "dataset_protocol": self.dataset_protocol,
            "action": self.action, "execution_family": self.execution_family,
            "processor_profile": self.processor_profile, "artifact_protocol": self.artifact_protocol,
            "dependency_profile": self.dependency_profile,
        }


@dataclass(frozen=True)
class PredicateFact:
    name: str
    outcome: str
    observation_kind: str
    expected: Any
    observed: Any
    reason: str
    _authority: InitVar[object] = None
    _observed_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: object) -> None:
        if _authority is not _AUTHORITY:
            _contract("predicate facts are inspector-owned")
        if not isinstance(self.name, str) or not _PREDICATE_RE.fullmatch(self.name):
            _contract("predicate fact name is invalid")
        object.__setattr__(self, "outcome", _enum(self.outcome, PREDICATE_OUTCOMES, "predicate outcome"))
        object.__setattr__(self, "observation_kind", _enum(self.observation_kind, OBSERVATION_KINDS, "observation_kind"))
        object.__setattr__(self, "expected", _json_fact(self.expected, "expected"))
        object.__setattr__(self, "observed", _json_fact(self.observed, "observed"))
        object.__setattr__(self, "reason", _public_text(self.reason, "reason"))
        if self.outcome in ("passed", "failed") and (self.expected is None or self.observed is None):
            _contract("checked predicate facts require comparable expected and observed values")
        object.__setattr__(self, "_observed_authority", True)

    @property
    def checked(self) -> bool:
        return self._observed_authority and self.outcome in ("passed", "failed")

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "outcome": self.outcome, "checked": self.checked,
            "observation_kind": self.observation_kind, "expected": self.expected,
            "observed": self.observed, "reason": self.reason,
        }


@dataclass(frozen=True)
class CapabilityReplay:
    identity: ModelCapabilityIdentity
    claimed_status: str
    claimed_predicates: Tuple[Mapping[str, Any], ...]

    @property
    def preflight_allowed(self) -> bool:
        return False

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_public_dict(), "claimed_status": self.claimed_status,
            "claimed_predicates": [dict(item) for item in self.claimed_predicates],
            "authority": "audit_replay", "preflight_allowed": False,
        }


@dataclass(frozen=True)
class ModelCapabilityResult:
    identity: ModelCapabilityIdentity
    raw_position: int
    raw_fingerprint: str
    required: bool
    required_predicates: Tuple[str, ...]
    predicates: Tuple[PredicateFact, ...]
    status: str
    reason: str
    did_probe: bool
    _authority: InitVar[object] = None
    _inspector_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: object) -> None:
        if _authority is not _AUTHORITY:
            _contract("terminal capability rows are inspector-owned")
        if not isinstance(self.identity, ModelCapabilityIdentity):
            _contract("result identity must be canonical")
        if type(self.raw_position) is not int or self.raw_position < 0:
            _contract("result raw_position must be a non-negative integer")
        if not isinstance(self.raw_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", self.raw_fingerprint):
            _contract("result raw_fingerprint must be a lowercase SHA-256")
        object.__setattr__(self, "required", _strict_bool(self.required, "required"))
        object.__setattr__(self, "required_predicates", _ordered_predicates(self.required_predicates))
        if not isinstance(self.predicates, tuple) or any(not isinstance(item, PredicateFact) for item in self.predicates):
            _contract("predicates must be an ordered tuple of inspector facts")
        names = tuple(item.name for item in self.predicates)
        if len(names) != len(set(names)):
            _contract("predicate facts must be unique by name")
        object.__setattr__(self, "status", _enum(self.status, TERMINAL_STATUSES, "status"))
        object.__setattr__(self, "reason", _public_text(self.reason, "reason"))
        object.__setattr__(self, "did_probe", _strict_bool(self.did_probe, "did_probe"))
        facts = {item.name: item for item in self.predicates}
        all_required_passed = all(
            name in facts and facts[name].checked and facts[name].outcome == "passed"
            for name in self.required_predicates
        )
        positive_authority = (
            "protocol_adapter" in facts
            and facts["protocol_adapter"].outcome == "passed"
            and facts["protocol_adapter"].observation_kind == "actual_wrapper_generated_protocol_probe"
            and "capability_identity_match" in facts
            and facts["capability_identity_match"].outcome == "passed"
            and facts["capability_identity_match"].observation_kind == "actual_wrapper_generated_protocol_probe"
        )
        if self.status == "supported_verified" and (
            not self.did_probe or not all_required_passed or not positive_authority
        ):
            _contract("supported_verified requires exact inspector-controlled protocol authority")
        if self.status in ("unsupported", "conditional", "coverage_unsafe", "not_comparable", "probe_failed") and not self.did_probe:
            _contract("observed terminal classifications require did_probe")
        if self.status == "invalid_declaration" and self.did_probe:
            _contract("invalid_declaration cannot claim a protocol probe")
        if self.status == "conditional" and not any(item.name == "dependency_available" and item.outcome == "failed" for item in self.predicates):
            _contract("conditional requires a failed dependency predicate")
        if self.status == "coverage_unsafe" and not any(
            (item.name.startswith("prediction_") or item.name == "processor_tail_safe")
            and item.outcome == "failed"
            for item in self.predicates
        ):
            _contract("coverage_unsafe requires a failed processor or prediction predicate")
        object.__setattr__(self, "_inspector_authority", True)

    @property
    def preflight_allowed(self) -> bool:
        return self._inspector_authority and self.status == "supported_verified"

    @property
    def checked_predicates(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.predicates if item.checked)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_public_dict(), "identity_fingerprint": self.identity.fingerprint,
            "raw_position": self.raw_position, "raw_fingerprint": self.raw_fingerprint,
            "required": self.required, "required_predicates": list(self.required_predicates),
            "predicates": [item.to_public_dict() for item in self.predicates],
            "status": self.status, "reason": self.reason, "did_probe": self.did_probe,
            "preflight_allowed": self.preflight_allowed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityReplay:
        if not isinstance(value, Mapping) or not isinstance(value.get("identity"), Mapping):
            _contract("result replay must contain an identity mapping")
        identity = ModelCapabilityIdentity(**dict(value["identity"]))
        claimed_status = _enum(value.get("status"), TERMINAL_STATUSES, "claimed status")
        predicates = value.get("predicates", ())
        if not isinstance(predicates, (list, tuple)) or any(not isinstance(item, Mapping) for item in predicates):
            _contract("replayed predicates must be mappings")
        return CapabilityReplay(identity, claimed_status, tuple(dict(item) for item in predicates))


@dataclass(frozen=True)
class UnassignedDeclaration:
    position: int
    raw_fingerprint: str
    reason: str
    required: bool

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            _contract("unassigned position must be a non-negative integer")
        if not isinstance(self.raw_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", self.raw_fingerprint):
            _contract("unassigned raw_fingerprint must be a lowercase SHA-256")
        object.__setattr__(self, "reason", _public_text(self.reason, "unassigned reason"))
        object.__setattr__(self, "required", _strict_bool(self.required, "unassigned required"))

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position, "raw_fingerprint": self.raw_fingerprint,
            "reason": self.reason, "required": self.required,
        }


@dataclass(frozen=True)
class CapabilityMatrixReplay:
    claimed_status: str
    snapshot: Mapping[str, Any]

    @property
    def preflight_allowed(self) -> bool:
        return False


@dataclass(frozen=True)
class ModelCapabilityMatrix:
    raw_fingerprints: Tuple[str, ...]
    results: Tuple[ModelCapabilityResult, ...]
    unassigned: Tuple[UnassignedDeclaration, ...]
    _authority: InitVar[object] = None
    _inspector_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority: object) -> None:
        if _authority is not _AUTHORITY:
            _contract("capability aggregate is inspector-owned")
        if not isinstance(self.raw_fingerprints, tuple) or any(
            not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
            for item in self.raw_fingerprints
        ):
            _contract("raw_fingerprints must be an ordered tuple of SHA-256 values")
        if not isinstance(self.results, tuple) or any(not isinstance(item, ModelCapabilityResult) for item in self.results):
            _contract("results must contain canonical terminal rows")
        if not isinstance(self.unassigned, tuple) or any(not isinstance(item, UnassignedDeclaration) for item in self.unassigned):
            _contract("unassigned must contain canonical remainder observations")
        if len(self.raw_fingerprints) != len(self.results) + len(self.unassigned):
            _contract("raw inventory must partition into results and unassigned declarations")
        positions = tuple(item.position for item in self.unassigned)
        if len(positions) != len(set(positions)) or any(position >= len(self.raw_fingerprints) for position in positions):
            _contract("unassigned declaration positions must be unique and in inventory")
        result_positions = tuple(item.raw_position for item in self.results)
        if result_positions != tuple(sorted(result_positions)):
            _contract("terminal results must preserve raw declaration order")
        assigned_positions = result_positions + positions
        if len(assigned_positions) != len(set(assigned_positions)) or set(assigned_positions) != set(range(len(self.raw_fingerprints))):
            _contract("every raw declaration must be assigned exactly once")
        for result in self.results:
            if self.raw_fingerprints[result.raw_position] != result.raw_fingerprint:
                _contract("result raw fingerprint does not match authoritative inventory")
        for remainder in self.unassigned:
            if self.raw_fingerprints[remainder.position] != remainder.raw_fingerprint:
                _contract("unassigned raw fingerprint does not match authoritative inventory")
        object.__setattr__(self, "_inspector_authority", True)

    @property
    def n_declarations(self) -> int:
        return len(self.raw_fingerprints)

    @property
    def n_unassigned_declarations(self) -> int:
        return len(self.unassigned)

    @property
    def n_attempted(self) -> int:
        return sum(item.did_probe for item in self.results)

    @property
    def n_supported(self) -> int:
        return sum(item.status == "supported_verified" for item in self.results)

    @property
    def n_conditional(self) -> int:
        return sum(item.status == "conditional" for item in self.results)

    @property
    def n_blocked(self) -> int:
        return len(self.results) - self.n_supported

    @property
    def classification_counts(self) -> Dict[str, int]:
        return {
            status: sum(item.status == status for item in self.results)
            for status in TERMINAL_STATUSES
        }

    @property
    def n_predicates_checked(self) -> int:
        return sum(len(item.checked_predicates) for item in self.results)

    @property
    def status(self) -> str:
        if any(item.required for item in self.unassigned) or any(item.status == "invalid_declaration" for item in self.results):
            return "inventory_invalid"
        required = tuple(item for item in self.results if item.required)
        supported = sum(item.status == "supported_verified" for item in required)
        if required and supported == len(required):
            return "all_required_supported"
        if supported:
            return "partially_supported"
        return "none_supported"

    @property
    def preflight_allowed(self) -> bool:
        return self._inspector_authority and self.status == "all_required_supported"

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint({
            "raw_fingerprints": list(self.raw_fingerprints),
            "results": [item.to_public_dict() for item in self.results],
            "unassigned": [item.to_public_dict() for item in self.unassigned],
        })

    def query(self, identity: ModelCapabilityIdentity) -> ModelCapabilityResult:
        if not isinstance(identity, ModelCapabilityIdentity):
            _contract("query requires a canonical capability identity")
        matches = tuple(item for item in self.results if item.identity == identity)
        if len(matches) != 1:
            _contract("query identity is unknown or conflicted")
        return matches[0]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "fingerprint": self.fingerprint,
            "n_declarations": self.n_declarations,
            "n_unassigned_declarations": self.n_unassigned_declarations,
            "n_results": len(self.results), "n_attempted": self.n_attempted,
            "n_supported": self.n_supported, "n_conditional": self.n_conditional,
            "n_blocked": self.n_blocked, "classification_counts": self.classification_counts,
            "n_predicates_checked": self.n_predicates_checked,
            "preflight_allowed": self.preflight_allowed,
            "results": [item.to_public_dict() for item in self.results],
            "unassigned": [item.to_public_dict() for item in self.unassigned],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityMatrixReplay:
        if not isinstance(value, Mapping):
            _contract("matrix replay must be a mapping")
        claimed = _enum(value.get("status"), AGGREGATE_STATUSES, "claimed aggregate status")
        return CapabilityMatrixReplay(claimed, dict(value))


def _raw_fingerprint(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=lambda item: type(item).__name__)
    except Exception:
        payload = type(value).__name__
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _predicate_fact(
    name: str,
    outcome: str,
    observation_kind: str,
    expected: Any,
    observed: Any,
    reason: str,
) -> PredicateFact:
    return PredicateFact(name, outcome, observation_kind, expected, observed, reason, _authority=_AUTHORITY)


def _terminal_result(
    declaration: RawModelCapabilityDeclaration,
    predicates: Sequence[PredicateFact],
    status: str,
    reason: str,
    did_probe: bool,
    raw_position: int,
    raw_fingerprint: str,
) -> ModelCapabilityResult:
    return ModelCapabilityResult(
        ModelCapabilityIdentity.from_declaration(declaration), raw_position, raw_fingerprint, declaration.required,
        declaration.required_predicates, tuple(predicates), status, reason, did_probe,
        _authority=_AUTHORITY,
    )


def _capability_matrix(
    raw_fingerprints: Sequence[str],
    results: Sequence[ModelCapabilityResult],
    unassigned: Sequence[UnassignedDeclaration],
) -> ModelCapabilityMatrix:
    return ModelCapabilityMatrix(tuple(raw_fingerprints), tuple(results), tuple(unassigned), _authority=_AUTHORITY)
