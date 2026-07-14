"""Pure recovery classification for the training execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TrainingRecoveryObservation:
    phase: Optional[str]
    target_keys: Tuple[str, ...]
    reusable_target_keys: Tuple[str, ...] = ()
    failed_target_keys: Tuple[str, ...] = ()
    intent_present: bool = False
    receipt_present: bool = False
    publication_unknown: bool = False
    identity_valid: bool = True
    evidence_valid: bool = True
    member_states: Tuple[str, ...] = ()
    closure_pending: Tuple[str, ...] = ()
    transaction_id: Optional[str] = None


@dataclass(frozen=True)
class TrainingRecoveryDecision:
    action: str
    runnable_target_keys: Tuple[str, ...]
    reusable_target_keys: Tuple[str, ...]
    transaction_id: Optional[str]
    pending_closure_steps: Tuple[str, ...]
    reason_code: str


def classify_training_recovery(observation: TrainingRecoveryObservation) -> TrainingRecoveryDecision:
    if not observation.identity_valid:
        return TrainingRecoveryDecision("fail_closed", (), (), observation.transaction_id, (), "publication_identity_mismatch")
    if not observation.evidence_valid:
        return TrainingRecoveryDecision("fail_closed", (), (), observation.transaction_id, (), "target_evidence_mismatch")
    if observation.publication_unknown or "unknown" in observation.member_states:
        return TrainingRecoveryDecision("fail_closed", (), (), observation.transaction_id, (), "publication_output_unknown")
    if observation.receipt_present:
        if observation.member_states and any(value != "postimage" for value in observation.member_states):
            return TrainingRecoveryDecision("fail_closed", (), (), observation.transaction_id, (), "committed_output_changed")
        return TrainingRecoveryDecision(
            "close_postprocess", (), observation.reusable_target_keys,
            observation.transaction_id, observation.closure_pending, "adopt_committed_receipt",
        )
    if observation.intent_present:
        states = set(observation.member_states)
        reason = "recover_known_mixed_publication" if {
            "preimage", "postimage"
        }.issubset(states) else "recover_orphan_intent"
        return TrainingRecoveryDecision(
            "recover_publication", (), observation.reusable_target_keys,
            observation.transaction_id, observation.closure_pending, reason,
        )
    reusable = frozenset(observation.reusable_target_keys)
    runnable = tuple(key for key in observation.target_keys if key not in reusable)
    if observation.phase in ("publication_prepared", "publication_committed", "closing", "completed"):
        return TrainingRecoveryDecision(
            "fail_closed", (), tuple(sorted(reusable)), observation.transaction_id,
            observation.closure_pending, "publication_identity_missing",
        )
    if observation.phase == "targets_complete":
        if runnable:
            if set(runnable).issubset(observation.failed_target_keys):
                return TrainingRecoveryDecision(
                    "run_targets", runnable, tuple(sorted(reusable)), None, (), "retry_failed_targets",
                )
            return TrainingRecoveryDecision("fail_closed", (), tuple(sorted(reusable)), None, (), "target_evidence_missing")
        return TrainingRecoveryDecision("prepare_publication", (), tuple(sorted(reusable)), None, (), "resume_targets_complete")
    if observation.phase in ("prepared", "executing", "failed"):
        return TrainingRecoveryDecision("run_targets", runnable, tuple(sorted(reusable)), None, (), "resume_partial_execution")
    return TrainingRecoveryDecision("new_run", observation.target_keys, (), None, (), "new_run")
