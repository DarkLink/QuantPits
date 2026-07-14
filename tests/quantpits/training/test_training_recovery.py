from quantpits.training.recovery import TrainingRecoveryObservation, classify_training_recovery
import pytest


def test_recovery_classifier_reuses_completed_targets_without_io():
    decision = classify_training_recovery(TrainingRecoveryObservation(
        phase="targets_complete", target_keys=("a@static", "b@static"),
        reusable_target_keys=("a@static", "b@static"),
    ))
    assert decision.action == "prepare_publication"
    assert decision.runnable_target_keys == ()
    assert decision.reason_code == "resume_targets_complete"


def test_recovery_classifier_prefers_receipt_closure():
    decision = classify_training_recovery(TrainingRecoveryObservation(
        phase="targets_complete", target_keys=("a@static",),
        reusable_target_keys=("a@static",), receipt_present=True,
        transaction_id="tx", closure_pending=("manifest_verified",),
    ))
    assert decision.action == "close_postprocess"
    assert decision.pending_closure_steps == ("manifest_verified",)


@pytest.mark.parametrize("observation,action,reason", [
    (TrainingRecoveryObservation(None, ("a",)), "new_run", "new_run"),
    (TrainingRecoveryObservation("prepared", ("a",)), "run_targets", "resume_partial_execution"),
    (TrainingRecoveryObservation("executing", ("a", "b"), ("a",)), "run_targets", "resume_partial_execution"),
    (TrainingRecoveryObservation("targets_complete", ("a",), ("a",)), "prepare_publication", "resume_targets_complete"),
    (TrainingRecoveryObservation("targets_complete", ("a",), (), evidence_valid=False), "fail_closed", "target_evidence_mismatch"),
    (TrainingRecoveryObservation("publication_prepared", ("a",), ("a",)), "fail_closed", "publication_identity_missing"),
    (TrainingRecoveryObservation("targets_complete", ("a",), ("a",), intent_present=True, transaction_id="tx"), "recover_publication", "recover_orphan_intent"),
    (TrainingRecoveryObservation("publication_prepared", ("a",), ("a",), intent_present=True, member_states=("preimage", "postimage"), transaction_id="tx"), "recover_publication", "recover_known_mixed_publication"),
    (TrainingRecoveryObservation("publication_committed", ("a",), ("a",), receipt_present=True, member_states=("postimage",), transaction_id="tx"), "close_postprocess", "adopt_committed_receipt"),
    (TrainingRecoveryObservation("publication_committed", ("a",), ("a",), receipt_present=True, member_states=("preimage",), transaction_id="tx"), "fail_closed", "committed_output_changed"),
    (TrainingRecoveryObservation("failed", ("a",), identity_valid=False), "fail_closed", "publication_identity_mismatch"),
    (TrainingRecoveryObservation("publication_prepared", ("a",), publication_unknown=True), "fail_closed", "publication_output_unknown"),
])
def test_recovery_classifier_decision_matrix(observation, action, reason):
    decision = classify_training_recovery(observation)
    assert decision.action == action
    assert decision.reason_code == reason


def test_targets_complete_retries_only_declared_failed_targets():
    decision = classify_training_recovery(TrainingRecoveryObservation(
        "targets_complete", ("a", "b"), ("a",), failed_target_keys=("b",),
    ))
    assert decision.action == "run_targets"
    assert decision.runnable_target_keys == ("b",)
    assert decision.reason_code == "retry_failed_targets"
