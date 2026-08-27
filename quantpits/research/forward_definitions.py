"""Pure typed compiler for one Champion--Challenger shadow definition set."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


DEFINITION_CLAIM = "TYPED_OWNER_DECLARATION_V1"
PROTOCOL_KIND = "SHADOW_FORWARD_PROTOCOL_V1"
EXECUTION_KIND = "SHADOW_FORWARD_EXECUTION_ASSUMPTION_V1"
STRATEGY_KIND = "SHADOW_FORWARD_STRATEGY_V1"
RESEARCH_EPOCH_ID = "PROSPECTIVE_SHADOW_V1"

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECOND_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_AUTHORITY = object()
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)

_FALSE_CLAIMS = (
    "external_references_verified", "epoch_started", "prospective_claim",
    "promotion_capability",
)
_DIGEST_FIELDS = ("algorithm", "domain", "value", "size_bytes")
_SOURCE_FIELDS = ("position", "source_id", "model_artifact_digest")

_PROTOCOL_FIELDS = (
    "schema_version", "definition_kind", "definition_claim", "protocol_id",
    "definition_set_id", "research_epoch_id", "created_at", "effective_cycle",
    "champion_strategy_id", "challenger_strategy_id", "execution_assumption_id",
    "selection_decision", "intent_definition", "bootstrap_rule",
    "champion_ranking_rule", "challenger_ranking_rule", "settlement_rule",
    "promotion_rule", "external_references_verified", "epoch_started",
    "prospective_claim", "promotion_capability",
)
_EXECUTION_EXTRA_FIELDS = (
    "definition_kind", "definition_claim", "created_at", "effective_cycle",
    "external_references_verified", "epoch_started", "prospective_claim",
    "promotion_capability",
)
_STRATEGY_FIELDS = (
    "schema_version", "definition_kind", "definition_claim", "strategy_id", "role",
    "parent_strategy_id", "mutation_level", "created_at", "effective_cycle",
    "data_cutoff", "evidence_cycle_id", "evidence_seal_digest",
    "intent_definition_id", "selection_decision_id", "fusion_definition",
    "ranking_rule", "source_members", "hypothesis", "external_references_verified",
    "epoch_started", "prospective_claim", "promotion_capability",
)
_TOP_FIELDS = (
    "definition_set_id", "protocol", "execution_assumption", "champion_strategy",
    "challenger_strategy",
)


class ForwardDefinitionContractError(ValueError):
    """A raw declaration cannot receive typed compiler authority."""


def _exact_dict(value: Any, fields: Tuple[str, ...], name: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ForwardDefinitionContractError("%s fields are not exact" % name)
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForwardDefinitionContractError("definition is not canonical JSON") from exc


def _definition_id(value: Any, name: str) -> str:
    if type(value) is not str or value in (".", "..") or _ID_RE.fullmatch(value) is None:
        raise ForwardDefinitionContractError("%s is not a canonical definition ID" % name)
    return value


def _strict_text(value: Any, name: str, maximum: int) -> str:
    if (
        type(value) is not str or not value or value != value.strip()
        or len(value) > maximum or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ForwardDefinitionContractError("%s must be strict non-empty text" % name)
    return value


def _utc_second(value: Any, name: str) -> str:
    if type(value) is not str or _UTC_SECOND_RE.fullmatch(value) is None:
        raise ForwardDefinitionContractError("%s must be canonical UTC second precision" % name)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ForwardDefinitionContractError("%s is not a real timestamp" % name) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ForwardDefinitionContractError("%s is not canonical" % name)
    return value


def _calendar_date(value: Any, name: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise ForwardDefinitionContractError("%s must be a canonical date" % name)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ForwardDefinitionContractError("%s is not a real date" % name) from exc
    if parsed.isoformat() != value:
        raise ForwardDefinitionContractError("%s is not canonical" % name)
    return value


def _false_claims(raw: Mapping[str, Any], name: str) -> None:
    for field in _FALSE_CLAIMS:
        if type(raw[field]) is not bool or raw[field] is not False:
            raise ForwardDefinitionContractError("%s.%s must be false" % (name, field))


def _fixed(raw: Mapping[str, Any], field: str, expected: Any, name: str) -> None:
    if type(raw[field]) is not type(expected) or raw[field] != expected:
        raise ForwardDefinitionContractError("%s.%s must be %r" % (name, field, expected))


@dataclass(frozen=True)
class DigestReference:
    algorithm: str
    domain: str
    value: str
    size_bytes: int

    def __post_init__(self) -> None:
        if type(self.algorithm) is not str or self.algorithm != "sha256":
            raise ForwardDefinitionContractError("digest algorithm must be sha256")
        if type(self.domain) is not str or self.domain not in (
            "raw_bytes", "canonical_json", "semantic_config",
        ):
            raise ForwardDefinitionContractError("digest domain is invalid")
        if type(self.value) is not str or _HEX_RE.fullmatch(self.value) is None:
            raise ForwardDefinitionContractError("digest value is invalid")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ForwardDefinitionContractError("digest size_bytes is invalid")

    @classmethod
    def from_mapping(cls, value: Any, name: str, domain: Optional[str] = None) -> "DigestReference":
        raw = _exact_dict(value, _DIGEST_FIELDS, name)
        if raw["algorithm"] != "sha256" or type(raw["algorithm"]) is not str:
            raise ForwardDefinitionContractError("%s.algorithm must be sha256" % name)
        if type(raw["domain"]) is not str or raw["domain"] not in (
            "raw_bytes", "canonical_json", "semantic_config",
        ):
            raise ForwardDefinitionContractError("%s.domain is invalid" % name)
        if domain is not None and raw["domain"] != domain:
            raise ForwardDefinitionContractError("%s.domain must be %s" % (name, domain))
        if type(raw["value"]) is not str or _HEX_RE.fullmatch(raw["value"]) is None:
            raise ForwardDefinitionContractError("%s.value is invalid" % name)
        if type(raw["size_bytes"]) is not int or raw["size_bytes"] < 0:
            raise ForwardDefinitionContractError("%s.size_bytes is invalid" % name)
        return cls(raw["algorithm"], raw["domain"], raw["value"], raw["size_bytes"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm, "domain": self.domain, "value": self.value,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class SourceMember:
    position: int
    source_id: str
    model_artifact_digest: DigestReference

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ForwardDefinitionContractError("source position is invalid")
        _strict_text(self.source_id, "source_id", 256)
        if type(self.model_artifact_digest) is not DigestReference:
            raise ForwardDefinitionContractError("source artifact digest is foreign")
        if self.model_artifact_digest.domain != "raw_bytes":
            raise ForwardDefinitionContractError("source artifact digest must use raw_bytes")

    @classmethod
    def from_mapping(cls, value: Any, name: str) -> "SourceMember":
        raw = _exact_dict(value, _SOURCE_FIELDS, name)
        if type(raw["position"]) is not int or raw["position"] < 0:
            raise ForwardDefinitionContractError("%s.position is invalid" % name)
        return cls(
            raw["position"], _strict_text(raw["source_id"], name + ".source_id", 256),
            DigestReference.from_mapping(
                raw["model_artifact_digest"], name + ".model_artifact_digest", "raw_bytes",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position, "source_id": self.source_id,
            "model_artifact_digest": self.model_artifact_digest.to_dict(),
        }


class _OwnedLeaf:
    _payload: Mapping[str, Any]
    _authority: object

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _AUTHORITY or args or set(kwargs) != {"payload"}:
            raise ForwardDefinitionContractError("typed definitions are compiler-owned")
        object.__setattr__(self, "_payload", MappingProxyType(copy.deepcopy(dict(kwargs["payload"]))))
        object.__setattr__(self, "_authority", _AUTHORITY)

    def to_dict(self) -> Dict[str, Any]:
        if getattr(self, "_authority", None) is not _AUTHORITY:
            raise ForwardDefinitionContractError("typed definition authority is absent")
        return copy.deepcopy(dict(self._payload))


class ShadowForwardProtocol(_OwnedLeaf):
    pass


class ShadowExecutionAssumption(_OwnedLeaf):
    pass


class ShadowStrategyDefinition(_OwnedLeaf):
    pass


def _decision(raw: Any, created_at: str) -> Tuple[Any, Dict[str, Any]]:
    if type(raw) is not dict:
        raise ForwardDefinitionContractError("selection_decision must be an exact object")
    try:
        from quantpits.evidence.contracts import DecisionEvent
        decision = DecisionEvent.from_mapping(raw)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ForwardDefinitionContractError("selection_decision is invalid") from exc
    payload = decision.to_dict()
    try:
        decision_time = datetime.fromisoformat(decision.decision_time.replace("Z", "+00:00"))
        created_time = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if decision_time.astimezone(timezone.utc) > created_time:
            raise ForwardDefinitionContractError("selection decision is later than created_at")
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionContractError:
        raise
    except Exception as exc:
        raise ForwardDefinitionContractError("selection decision time is not comparable") from exc
    return decision, payload


def _intent(raw: Any) -> Tuple[Any, Dict[str, Any]]:
    if type(raw) is not dict:
        raise ForwardDefinitionContractError("intent_definition must be an exact object")
    try:
        from quantpits.research.intents import CurrentRuleIntentDefinition
        value = CurrentRuleIntentDefinition.from_dict(raw)
        payload = value._payload()
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise ForwardDefinitionContractError("intent_definition is invalid") from exc
    return value, payload


def _execution(raw: Any) -> Tuple[Any, Dict[str, Any], str, str]:
    try:
        from quantpits.research.accounting import ExecutionAssumption
        fields = tuple(ExecutionAssumption._FIELDS) + _EXECUTION_EXTRA_FIELDS
        value_raw = _exact_dict(raw, fields, "execution_assumption")
        b0_raw = {field: value_raw[field] for field in ExecutionAssumption._FIELDS}
        value = ExecutionAssumption.from_dict(b0_raw)
        payload = value._payload()
    except _PROCESS_CONTROL:
        raise
    except ForwardDefinitionContractError:
        raise
    except Exception as exc:
        raise ForwardDefinitionContractError("execution_assumption is invalid") from exc
    _fixed(value_raw, "definition_kind", EXECUTION_KIND, "execution_assumption")
    _fixed(value_raw, "definition_claim", DEFINITION_CLAIM, "execution_assumption")
    created_at = _utc_second(value_raw["created_at"], "execution_assumption.created_at")
    effective_cycle = _calendar_date(
        value_raw["effective_cycle"], "execution_assumption.effective_cycle",
    )
    _false_claims(value_raw, "execution_assumption")
    payload.update({field: value_raw[field] for field in _EXECUTION_EXTRA_FIELDS})
    return value, payload, created_at, effective_cycle


def _strategy(raw: Any, expected_role: str) -> Tuple[ShadowStrategyDefinition, Dict[str, Any], Tuple[SourceMember, ...]]:
    name = expected_role.lower() + "_strategy"
    value = _exact_dict(raw, _STRATEGY_FIELDS, name)
    _fixed(value, "schema_version", 1, name)
    _fixed(value, "definition_kind", STRATEGY_KIND, name)
    _fixed(value, "definition_claim", DEFINITION_CLAIM, name)
    _fixed(value, "role", expected_role, name)
    _false_claims(value, name)
    _definition_id(value["strategy_id"], name + ".strategy_id")
    _utc_second(value["created_at"], name + ".created_at")
    _calendar_date(value["effective_cycle"], name + ".effective_cycle")
    _calendar_date(value["data_cutoff"], name + ".data_cutoff")
    _calendar_date(value["evidence_cycle_id"], name + ".evidence_cycle_id")
    _definition_id(value["intent_definition_id"], name + ".intent_definition_id")
    seal = DigestReference.from_mapping(
        value["evidence_seal_digest"], name + ".evidence_seal_digest", "raw_bytes",
    )
    _fixed(
        value, "fusion_definition",
        "PER_MODEL_CROSS_SECTIONAL_PERCENTILE_RANK_EQUAL_MEAN_V1", name,
    )
    if type(value["source_members"]) is not list:
        raise ForwardDefinitionContractError("%s.source_members must be a list" % name)
    expected_count = 4 if expected_role == "CHAMPION" else 3
    if len(value["source_members"]) != expected_count:
        raise ForwardDefinitionContractError("%s source count is not exact" % name)
    members = tuple(
        SourceMember.from_mapping(row, "%s.source_members[%d]" % (name, index))
        for index, row in enumerate(value["source_members"])
    )
    if tuple(member.position for member in members) != tuple(range(expected_count)):
        raise ForwardDefinitionContractError("%s source positions are not exact" % name)
    if len({member.source_id for member in members}) != expected_count:
        raise ForwardDefinitionContractError("%s source IDs are not unique" % name)
    digest_keys = tuple(
        (member.model_artifact_digest.algorithm, member.model_artifact_digest.domain,
         member.model_artifact_digest.value, member.model_artifact_digest.size_bytes)
        for member in members
    )
    if len(set(digest_keys)) != expected_count:
        raise ForwardDefinitionContractError("%s artifact digests are not unique" % name)
    if expected_role == "CHAMPION":
        _fixed(value, "parent_strategy_id", None, name)
        _fixed(value, "mutation_level", "CHAMPION_BASELINE", name)
        _fixed(value, "ranking_rule", "SEALED_PRODUCTION_FULL_RANKING_V1", name)
        _fixed(value, "selection_decision_id", None, name)
        _fixed(value, "hypothesis", None, name)
    else:
        _definition_id(value["parent_strategy_id"], name + ".parent_strategy_id")
        _fixed(value, "mutation_level", "L2_ENSEMBLE_COMPOSITION", name)
        _fixed(value, "ranking_rule", "FROZEN_MEMBER_EQUAL_FUSION_V1", name)
        _strict_text(value["selection_decision_id"], name + ".selection_decision_id", 128)
        _strict_text(value["hypothesis"], name + ".hypothesis", 2000)
    payload = dict(value)
    payload["evidence_seal_digest"] = seal.to_dict()
    payload["source_members"] = [member.to_dict() for member in members]
    return ShadowStrategyDefinition(_authority=_AUTHORITY, payload=payload), payload, members


class CompiledShadowForwardDefinitions:
    """Compiler-owned aggregate that revalidates before yielding storage bytes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _AUTHORITY or args:
            raise ForwardDefinitionContractError("compiled definitions are compiler-owned")
        expected = {
            "definition_set_id", "protocol", "execution_assumption", "champion",
            "challenger", "canonical_members", "request_digest",
        }
        if set(kwargs) != expected:
            raise ForwardDefinitionContractError("compiled definition fields are not exact")
        for name, value in kwargs.items():
            stored_name = "_" + name if name in ("canonical_members", "request_digest") else name
            object.__setattr__(self, stored_name, value)
        object.__setattr__(self, "_authority", _AUTHORITY)

    def _assert_authority(self) -> None:
        if getattr(self, "_authority", None) is not _AUTHORITY:
            raise ForwardDefinitionContractError("compiled definition authority is absent")

    @property
    def compiled_capability(self) -> bool:
        self._assert_authority()
        compile_shadow_forward_definitions(self._current_raw())
        return True

    @property
    def canonical_members(self) -> Tuple[Tuple[str, bytes], ...]:
        fresh = compile_shadow_forward_definitions(self._current_raw())
        return fresh._canonical_members

    @property
    def request_digest(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.to_store_request().request_digest))

    @property
    def external_references_verified(self) -> bool:
        return False

    @property
    def epoch_started(self) -> bool:
        return False

    @property
    def prospective_claim(self) -> bool:
        return False

    @property
    def promotion_capability(self) -> bool:
        return False

    def _current_raw(self) -> Dict[str, Any]:
        self._assert_authority()
        expected = (
            (self.protocol, ShadowForwardProtocol),
            (self.execution_assumption, ShadowExecutionAssumption),
            (self.champion, ShadowStrategyDefinition),
            (self.challenger, ShadowStrategyDefinition),
        )
        if any(type(value) is not leaf_type for value, leaf_type in expected):
            raise ForwardDefinitionContractError("compiled aggregate contains a foreign leaf")
        return {
            "definition_set_id": self.definition_set_id,
            "protocol": self.protocol.to_dict(),
            "execution_assumption": self.execution_assumption.to_dict(),
            "champion_strategy": self.champion.to_dict(),
            "challenger_strategy": self.challenger.to_dict(),
        }

    def to_store_request(self) -> Any:
        fresh = compile_shadow_forward_definitions(self._current_raw())
        return fresh._fresh_store_request()

    def _fresh_store_request(self) -> Any:
        from quantpits.research.definition_store import DefinitionBundleMember, DefinitionBundleRequest
        members = tuple(DefinitionBundleMember(path, data) for path, data in self._canonical_members)
        return DefinitionBundleRequest(self.definition_set_id, members)

    def to_summary_dict(self) -> Dict[str, Any]:
        request = self.to_store_request()
        protocol = self.protocol.to_dict()
        champion = self.champion.to_dict()
        challenger = self.challenger.to_dict()
        return {
            "schema_version": 1,
            "definition_claim": DEFINITION_CLAIM,
            "definition_set_id": self.definition_set_id,
            "protocol_id": protocol["protocol_id"],
            "execution_assumption_id": protocol["execution_assumption_id"],
            "champion_strategy_id": champion["strategy_id"],
            "challenger_strategy_id": challenger["strategy_id"],
            "evidence_cycle_id": champion["evidence_cycle_id"],
            "evidence_seal_digest": dict(champion["evidence_seal_digest"]),
            "request_digest": dict(request.request_digest),
            "compiled_capability": True,
            "external_references_verified": False,
            "epoch_started": False,
            "prospective_claim": False,
            "promotion_capability": False,
        }


def compile_shadow_forward_definitions(raw: Any) -> CompiledShadowForwardDefinitions:
    """Rebuild and compile one exact, internally self-consistent definition set."""
    top = _exact_dict(raw, _TOP_FIELDS, "definition set")
    definition_set_id = _definition_id(top["definition_set_id"], "definition_set_id")

    execution_value, execution_payload, execution_created, execution_cycle = _execution(
        top["execution_assumption"],
    )
    champion, champion_payload, champion_members = _strategy(top["champion_strategy"], "CHAMPION")
    challenger, challenger_payload, challenger_members = _strategy(
        top["challenger_strategy"], "CHALLENGER",
    )

    protocol_raw = _exact_dict(top["protocol"], _PROTOCOL_FIELDS, "protocol")
    _fixed(protocol_raw, "schema_version", 1, "protocol")
    _fixed(protocol_raw, "definition_kind", PROTOCOL_KIND, "protocol")
    _fixed(protocol_raw, "definition_claim", DEFINITION_CLAIM, "protocol")
    _fixed(protocol_raw, "research_epoch_id", RESEARCH_EPOCH_ID, "protocol")
    _fixed(protocol_raw, "bootstrap_rule", "MATCHED_CHAMPION_SNAPSHOT_ONCE_V1", "protocol")
    _fixed(protocol_raw, "champion_ranking_rule", "SEALED_PRODUCTION_FULL_RANKING_V1", "protocol")
    _fixed(protocol_raw, "challenger_ranking_rule", "FROZEN_MEMBER_EQUAL_FUSION_V1", "protocol")
    _fixed(protocol_raw, "settlement_rule", "NEXT_OPEN_SELL_BEFORE_BUY_V1", "protocol")
    _fixed(protocol_raw, "promotion_rule", "EXPLICIT_OWNER_DECISION_ONLY_V1", "protocol")
    _false_claims(protocol_raw, "protocol")
    protocol_id = _definition_id(protocol_raw["protocol_id"], "protocol.protocol_id")
    protocol_set_id = _definition_id(
        protocol_raw["definition_set_id"], "protocol.definition_set_id",
    )
    protocol_created = _utc_second(protocol_raw["created_at"], "protocol.created_at")
    protocol_cycle = _calendar_date(protocol_raw["effective_cycle"], "protocol.effective_cycle")
    for field in ("champion_strategy_id", "challenger_strategy_id", "execution_assumption_id"):
        _definition_id(protocol_raw[field], "protocol." + field)
    decision, decision_payload = _decision(protocol_raw["selection_decision"], protocol_created)
    intent, intent_payload = _intent(protocol_raw["intent_definition"])

    semantic_ids = (
        definition_set_id, protocol_id, execution_value.assumption_id, intent.definition_id,
        champion_payload["strategy_id"], challenger_payload["strategy_id"],
    )
    _definition_id(execution_value.assumption_id, "execution_assumption.assumption_id")
    _definition_id(intent.definition_id, "protocol.intent_definition.definition_id")
    if len(set(semantic_ids)) != len(semantic_ids):
        raise ForwardDefinitionContractError("semantic definition IDs must be mutually distinct")
    if protocol_set_id != definition_set_id:
        raise ForwardDefinitionContractError("protocol definition_set_id does not match")
    links = (
        (protocol_raw["champion_strategy_id"], champion_payload["strategy_id"]),
        (protocol_raw["challenger_strategy_id"], challenger_payload["strategy_id"]),
        (protocol_raw["execution_assumption_id"], execution_value.assumption_id),
    )
    if any(left != right for left, right in links):
        raise ForwardDefinitionContractError("protocol definition links do not match")
    if decision.decision != "APPROVE" or decision.target != challenger_payload["strategy_id"]:
        raise ForwardDefinitionContractError("selection decision does not approve the Challenger")
    if decision.decision_id != challenger_payload["selection_decision_id"]:
        raise ForwardDefinitionContractError("selection decision ID does not match")
    if decision.evidence_cycle_id != challenger_payload["evidence_cycle_id"]:
        raise ForwardDefinitionContractError("selection decision evidence cycle does not match")
    if execution_value.lot_size != intent.production_buy_lot_size:
        raise ForwardDefinitionContractError("execution lot size does not match intent definition")
    if champion_payload["parent_strategy_id"] is not None:
        raise ForwardDefinitionContractError("Champion parent must be null")
    if challenger_payload["parent_strategy_id"] != champion_payload["strategy_id"]:
        raise ForwardDefinitionContractError("Challenger parent does not match Champion")

    common = (protocol_created, execution_created, champion_payload["created_at"], challenger_payload["created_at"])
    if len(set(common)) != 1:
        raise ForwardDefinitionContractError("created_at does not match across definitions")
    cycles = (protocol_cycle, execution_cycle, champion_payload["effective_cycle"], challenger_payload["effective_cycle"])
    if len(set(cycles)) != 1:
        raise ForwardDefinitionContractError("effective_cycle does not match across definitions")
    strategy_common = (
        "data_cutoff", "evidence_cycle_id", "evidence_seal_digest", "intent_definition_id",
        "fusion_definition",
    )
    if any(champion_payload[field] != challenger_payload[field] for field in strategy_common):
        raise ForwardDefinitionContractError("Champion and Challenger evidence semantics do not match")
    if champion_payload["intent_definition_id"] != intent.definition_id:
        raise ForwardDefinitionContractError("strategy intent definition does not match")
    if not (
        champion_payload["data_cutoff"] <= champion_payload["evidence_cycle_id"]
        < champion_payload["effective_cycle"]
    ):
        raise ForwardDefinitionContractError("cutoff/evidence/effective cycle order is invalid")

    champion_keys = tuple(
        (member.source_id, member.model_artifact_digest.to_dict()) for member in champion_members
    )
    challenger_keys = tuple(
        (member.source_id, member.model_artifact_digest.to_dict()) for member in challenger_members
    )
    matched = tuple(row for row in champion_keys if row in challenger_keys)
    if matched != challenger_keys or len(challenger_keys) != len(champion_keys) - 1:
        raise ForwardDefinitionContractError("Challenger is not an exact order-preserving leave-one-out")

    protocol_payload = dict(protocol_raw)
    protocol_payload["selection_decision"] = decision_payload
    protocol_payload["intent_definition"] = intent_payload
    protocol = ShadowForwardProtocol(_authority=_AUTHORITY, payload=protocol_payload)
    execution = ShadowExecutionAssumption(_authority=_AUTHORITY, payload=execution_payload)
    payloads = (
        ("protocol.json", protocol_payload),
        ("execution_assumption.json", execution_payload),
        ("champion_strategy.json", champion_payload),
        ("challenger_strategy.json", challenger_payload),
    )
    canonical_members = tuple((path, _canonical_json(payload)) for path, payload in payloads)

    from quantpits.research.definition_store import DefinitionBundleMember, DefinitionBundleRequest
    request = DefinitionBundleRequest(
        definition_set_id,
        tuple(DefinitionBundleMember(path, data) for path, data in canonical_members),
    )
    return CompiledShadowForwardDefinitions(
        _authority=_AUTHORITY,
        definition_set_id=definition_set_id,
        protocol=protocol,
        execution_assumption=execution,
        champion=champion,
        challenger=challenger,
        canonical_members=canonical_members,
        request_digest=MappingProxyType(dict(request.request_digest)),
    )


__all__ = [
    "CompiledShadowForwardDefinitions", "DigestReference", "ForwardDefinitionContractError",
    "ShadowExecutionAssumption", "ShadowForwardProtocol", "ShadowStrategyDefinition",
    "SourceMember", "compile_shadow_forward_definitions",
]
