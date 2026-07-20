"""Pure recovery proposals derived from immutable Rolling evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantpits.rolling.errors import RollingEvidenceContractError
from quantpits.rolling.evidence import (
    RollingEvidenceSetInspection,
    RollingUnitEvidenceRequest,
    _rebuild_evidence_set,
    _rebuild_result,
    _validate_requested_keys,
)
from quantpits.rolling.identity import RollingTargetIdentity, parse_rolling_window_key


RECOVERY_REASONS = {
    "all_reusable": "rolling_recovery_all_reusable",
    "incomplete": "rolling_recovery_incomplete",
    "no_reusable_evidence": "rolling_recovery_no_reusable_evidence",
    "blocked": "rolling_recovery_blocked",
}


def _contract(message: str) -> None:
    raise RollingEvidenceContractError(message)


@dataclass(frozen=True)
class RollingRecoveryProposal:
    requested_unit_keys: tuple
    reusable_unit_keys: tuple
    unresolved_unit_keys: tuple
    orphan_unit_keys: tuple
    status: str
    reason_code: str
    evidence_set_fingerprint: str

    def __post_init__(self) -> None:
        requested = _validate_requested_keys(self.requested_unit_keys)
        for field in ("reusable_unit_keys", "unresolved_unit_keys", "orphan_unit_keys"):
            if not isinstance(getattr(self, field), tuple):
                _contract("%s must be an ordered tuple" % field)
        reusable = self.reusable_unit_keys
        unresolved = self.unresolved_unit_keys
        orphans = self.orphan_unit_keys
        if reusable != tuple(item for item in requested if item in set(reusable)):
            _contract("reusable units must preserve requested order")
        if unresolved != tuple(item for item in requested if item in set(unresolved)):
            _contract("unresolved units must preserve requested order")
        if len(reusable) != len(set(reusable)) or len(unresolved) != len(set(unresolved)):
            _contract("recovery proposal partitions contain duplicates")
        if set(reusable).intersection(unresolved) or set(reusable).union(unresolved) != set(requested):
            _contract("recovery proposal must exactly partition requested units")
        if len(orphans) != len(set(orphans)) or set(orphans).intersection(requested):
            _contract("recovery orphan units must be unique and unrequested")
        for item in orphans:
            if not isinstance(item, tuple) or len(item) != 2:
                _contract("recovery orphan unit must be a target/window pair")
            target = RollingTargetIdentity.parse(item[0])
            family, _, _, _ = parse_rolling_window_key(item[1])
            if target.family != family:
                _contract("recovery orphan target/window families disagree")
        if self.status not in RECOVERY_REASONS or self.reason_code != RECOVERY_REASONS[self.status]:
            _contract("recovery status and reason_code disagree")
        expected_status = (
            "all_reusable" if len(reusable) == len(requested)
            else "no_reusable_evidence" if not reusable
            else "incomplete"
        )
        if self.status != "blocked" and self.status != expected_status:
            _contract("recovery status disagrees with its partition")
        if self.status == "blocked" and reusable:
            _contract("blocked recovery cannot expose reusable units")
        if (
            not isinstance(self.evidence_set_fingerprint, str)
            or len(self.evidence_set_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.evidence_set_fingerprint)
        ):
            _contract("evidence_set_fingerprint must be SHA-256")

    @property
    def recovery_complete(self) -> bool:
        return self.status == "all_reusable"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return ("render", "proposal_only")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "requested_unit_keys": [list(item) for item in self.requested_unit_keys],
            "reusable_unit_keys": [list(item) for item in self.reusable_unit_keys],
            "unresolved_unit_keys": [list(item) for item in self.unresolved_unit_keys],
            "orphan_unit_keys": [list(item) for item in self.orphan_unit_keys],
            "status": self.status,
            "reason_code": self.reason_code,
            "evidence_set_fingerprint": self.evidence_set_fingerprint,
            "recovery_complete": self.recovery_complete,
            "capabilities": list(self.capabilities),
        }


def classify_rolling_recovery(requests, evidence_set):
    """Return one pure proposal after complete aggregate revalidation."""

    if not isinstance(requests, tuple) or not requests:
        _contract("requests must be a non-empty ordered tuple")
    rebuilt_requests = []
    for item in requests:
        if not isinstance(item, RollingUnitEvidenceRequest):
            _contract("requests must contain RollingUnitEvidenceRequest members")
        rebuilt_requests.append(RollingUnitEvidenceRequest(
            item.run_identity, item.target_key, item.window_identity, item.source_protocol,
            item.source_publication_key, item.source_operation, item.experiment_name,
            item.experiment_id, item.recorder_id, item.artifacts,
            item.expected_prediction_sessions,
        ))
    requested_keys = _validate_requested_keys(tuple(item.unit_key for item in rebuilt_requests))
    if len({item.run_identity.fingerprint for item in rebuilt_requests}) != 1:
        _contract("recovery requests must belong to one Rolling run identity")
    evidence_set = _rebuild_evidence_set(evidence_set)
    rebuilt_results = tuple(_rebuild_result(item) for item in evidence_set.unit_results)
    if evidence_set.requested_unit_keys != requested_keys or tuple(item.unit_key for item in rebuilt_results) != requested_keys:
        _contract("recovery request and evidence identities disagree")
    if any(
        result.request_fingerprint != request.source_manifest_fingerprint
        for request, result in zip(rebuilt_requests, rebuilt_results)
    ):
        _contract("recovery request and evidence fingerprints disagree")
    if evidence_set.status == "observation_drifted" or evidence_set.inventory_before_fingerprint != evidence_set.inventory_after_fingerprint:
        reusable = ()
        unresolved = requested_keys
        status = "blocked"
    else:
        reusable = tuple(item.unit_key for item in rebuilt_results if item.classification == "valid")
        unresolved = tuple(item.unit_key for item in rebuilt_results if item.classification != "valid")
        status = (
            "all_reusable" if len(reusable) == len(requested_keys)
            else "no_reusable_evidence" if not reusable
            else "incomplete"
        )
    return RollingRecoveryProposal(
        requested_keys, reusable, unresolved,
        tuple(item.unit_key for item in evidence_set.orphan_observations),
        status, RECOVERY_REASONS[status], evidence_set.fingerprint,
    )
