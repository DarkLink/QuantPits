import json
import threading
from dataclasses import replace
from contextlib import contextmanager

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.errors import TrainingPublicationRecoveryError
from quantpits.training.publication import (
    TrainingPublicationCoordinator, TrainingPublicationReceipt,
)
from quantpits.training.records import ModelRecordEntry, ModelRecordOutcome
from quantpits.training.record_repository import TrainingRecordRepository
from quantpits.training.resolved import resolve_training_run
from quantpits.training.runners import TrainingTargetResult
from quantpits.utils.workspace import WorkspaceContext


def _workspace(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "data").mkdir()
    (root / "config/model_registry.yaml").write_text(
        "models:\n  demo:\n    enabled: true\n    yaml_file: demo.yaml\n"
    )
    (root / "config/model_config.json").write_text('{"freq":"week"}')
    (root / "config/demo.yaml").write_text("model: {}\n")
    return root


def _run_and_result(root):
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(family="static", action="full", run_id="publication-run"),
    )
    run = resolve_training_run(prepared, {"anchor_date": "2026-07-10", "test_end_time": "2026-07-10"})
    entry = ModelRecordEntry(
        "demo@static", "demo", "static", "train", "ready", "rid", "Prod_Train_WEEK",
        requested_anchor="2026-07-10", prediction_start="2026-07-10",
        prediction_end="2026-07-10", prediction_rows=1,
    )
    return run, TrainingTargetResult("demo@static", "train", "success", entry, {"IC_Mean": 0.1})


def test_publication_commits_record_last_and_writes_verified_receipt(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(run, clock=lambda: __import__("datetime").datetime(2026, 7, 14))
    intent = coordinator.prepare((result,))
    assert [item.kind for item in sorted(intent.members, key=lambda item: item.commit_order)][-1] == "record"
    receipt = coordinator.commit(intent)
    assert receipt.status == "committed"
    assert json.loads((root / "latest_train_records.json").read_text())["models"] == {"demo@static": "rid"}
    assert coordinator.recover().published_keys == ("demo@static",)


def test_publication_rolls_forward_known_mixed_state_without_target_execution(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(run, clock=lambda: __import__("datetime").datetime(2026, 7, 14))
    intent = coordinator.prepare((result,))
    performance = next(item for item in intent.members if item.kind == "performance")
    target = root / performance.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((coordinator.directory / performance.staged_relative_path).read_bytes())
    observation = coordinator.inspect_recovery()
    assert observation["intent_present"] is True
    assert observation["receipt_present"] is False
    assert set(observation["member_states"]) == {"preimage", "postimage"}
    receipt = coordinator.recover()
    assert receipt.recovery_action == "roll_forward"
    assert (root / "latest_train_records.json").is_file()


def test_publication_recovers_all_preimages_as_orphan_intent(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    intent = coordinator.prepare((result,))

    observation = coordinator.inspect_recovery()
    assert observation["member_states"] == tuple("preimage" for _ in intent.members)
    receipt = coordinator.recover()

    assert receipt.recovery_action == "roll_forward"
    assert all(coordinator._member_state(member) == "postimage" for member in intent.members)


def test_publication_finalizes_all_postimages_without_receipt(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    intent = coordinator.prepare((result,))
    for member in intent.members:
        target = root / member.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((coordinator.directory / member.staged_relative_path).read_bytes())

    observation = coordinator.inspect_recovery()
    assert observation["receipt_present"] is False
    assert observation["member_states"] == tuple("postimage" for _ in intent.members)
    receipt = coordinator.recover()

    assert receipt.recovery_action == "finalize_postimages"


@pytest.mark.parametrize(
    "point,member",
    [
        ("after_member_replace", "output/model_performance_2026-07-10.json"),
        ("after_member_replace", "latest_train_records.json"),
        ("before_receipt_write", None),
    ],
)
def test_publication_recovers_each_current_output_boundary(tmp_path, point, member):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    fired = []

    def fail(actual_point, actual_member=None):
        if not fired and actual_point == point and (member is None or actual_member == member):
            fired.append(True)
            raise RuntimeError("injected publication interruption")

    interrupted = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14), fault_hook=fail,
    )
    intent = interrupted.prepare((result,))
    with pytest.raises(RuntimeError, match="injected"):
        interrupted.commit(intent)
    recovered = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    ).recover()
    assert recovered.status == "committed"
    if point == "before_receipt_write":
        assert recovered.recovery_action == "finalize_postimages"
    assert json.loads((root / "latest_train_records.json").read_text())["models"] == {
        "demo@static": "rid"
    }


@pytest.mark.parametrize("mutation,reason", [
    (lambda receipt: replace(receipt, transaction_id="other"), "receipt_transaction_mismatch"),
    (lambda receipt: replace(receipt, run_id="other"), "receipt_run_mismatch"),
    (lambda receipt: replace(receipt, published_keys=("other@static",)), "receipt_keys_mismatch"),
    (lambda receipt: replace(receipt, committed_outputs=receipt.committed_outputs[:-1]), "receipt_ledger_mismatch"),
    (lambda receipt: replace(receipt, committed_outputs=(
        receipt.committed_outputs[0],
    ) + receipt.committed_outputs), "receipt_member_duplicate"),
    (lambda receipt: replace(receipt, committed_outputs=receipt.committed_outputs + ({
        "path": "extra.json", "kind": "record", "fingerprint": "extra",
    },)), "receipt_ledger_mismatch"),
    (lambda receipt: replace(receipt, committed_outputs=(
        dict(receipt.committed_outputs[0], fingerprint="wrong"),
    ) + receipt.committed_outputs[1:]), "receipt_ledger_mismatch"),
])
def test_receipt_must_prove_exact_intent_bundle(tmp_path, mutation, reason):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    intent = coordinator.prepare((result,))
    receipt = coordinator.commit(intent)
    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.verify_receipt(intent, mutation(receipt))
    assert raised.value.reason_code == reason


@pytest.mark.parametrize("payload,reason", [
    ({"schema_version": 2}, "receipt_schema_invalid"),
    ({
        "schema_version": 1, "transaction_id": "tx", "run_id": "run",
        "status": "prepared", "published_keys": ["demo@static"],
        "committed_outputs": [{"path": "x", "kind": "record", "fingerprint": "fp"}],
    }, "receipt_status_invalid"),
    ({
        "schema_version": 1, "transaction_id": "tx", "run_id": "run",
        "status": "committed", "published_keys": ["demo@static"],
        "committed_outputs": [{"path": "x", "kind": "record"}],
    }, "receipt_member_malformed"),
])
def test_receipt_parser_rejects_malformed_evidence(payload, reason):
    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        TrainingPublicationReceipt.from_dict(payload)
    assert raised.value.reason_code == reason


def test_receipt_recovery_rejects_changed_committed_output(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    coordinator.commit(coordinator.prepare((result,)))
    (root / "latest_train_records.json").write_text("{}\n")
    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.inspect_recovery()
    assert raised.value.reason_code == "receipt_postimage_changed"


def test_receipt_without_intent_fails_closed_with_stable_reason(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    coordinator.commit(coordinator.prepare((result,)))
    coordinator.intent_path.unlink()

    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.inspect_recovery()
    assert raised.value.reason_code == "receipt_without_intent"


def test_malformed_intent_fails_closed_with_stable_reason(tmp_path):
    root = _workspace(tmp_path)
    run, _result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    coordinator.directory.mkdir(parents=True)
    coordinator.intent_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.inspect_recovery()
    assert raised.value.reason_code == "intent_malformed"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda member: member.update(relative_path="../outside.json"), "intent_member_path_invalid"),
        (lambda member: member.update(staged_relative_path="../outside.stage"), "intent_stage_path_invalid"),
        (lambda member: member.update(preimage_relative_path="../outside.preimage"), "intent_preimage_path_invalid"),
        (lambda member: member["preimage"].update(path="other.json"), "intent_preimage_identity_mismatch"),
        (lambda member: member["preimage"].update(existed="yes"), "intent_member_malformed"),
    ],
)
def test_corrupt_intent_members_fail_closed(tmp_path, mutation, reason):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    coordinator.prepare((result,))
    payload = json.loads(coordinator.intent_path.read_text(encoding="utf-8"))
    mutation(payload["members"][0])
    coordinator.intent_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.inspect_recovery()
    assert raised.value.reason_code == reason


def test_duplicate_intent_member_fails_during_read_only_inspection(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    coordinator.prepare((result,))
    payload = json.loads(coordinator.intent_path.read_text(encoding="utf-8"))
    payload["members"].append(dict(payload["members"][0]))
    coordinator.intent_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.inspect_recovery()
    assert raised.value.reason_code == "intent_member_duplicate"


def test_missing_staged_member_fails_closed_with_stable_reason(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    intent = coordinator.prepare((result,))
    (coordinator.directory / intent.members[0].staged_relative_path).unlink()

    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.recover()
    assert raised.value.reason_code == "intent_staged_payload_missing"


def test_corrupt_staged_member_fails_closed_with_stable_reason(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    intent = coordinator.prepare((result,))
    (coordinator.directory / intent.members[0].staged_relative_path).write_bytes(
        b"corrupt-staged-output\n"
    )

    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.recover()
    assert raised.value.reason_code == "intent_staged_payload_corrupt"


def test_unknown_current_member_fails_closed_with_stable_reason(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    )
    intent = coordinator.prepare((result,))
    target = root / intent.members[0].relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"unknown-current-output\n")

    with pytest.raises(TrainingPublicationRecoveryError) as raised:
        coordinator.recover()
    assert raised.value.reason_code == "publication_output_unknown"


def test_publication_holds_canonical_record_lock_across_member_replacement(
    tmp_path, monkeypatch,
):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    active = []

    @contextmanager
    def tracked_lock(_repository):
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    observed = []

    def fault(point, member=None):
        if point == "after_member_replace":
            observed.append((member, bool(active)))

    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14),
        fault_hook=fault,
    )
    intent = coordinator.prepare((result,))
    monkeypatch.setattr(TrainingRecordRepository, "lock", tracked_lock)
    coordinator.commit(intent)
    assert observed
    assert all(held for _member, held in observed)


def test_competing_repository_writer_waits_and_preserves_publication(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    started = threading.Event()
    finished = threading.Event()
    errors = []
    writer = []

    extra = ModelRecordEntry(
        "other@static", "other", "static", "train", "ready", "rid-other",
        "Prod_Train_WEEK", requested_anchor="2026-07-10",
        prediction_start="2026-07-10", prediction_end="2026-07-10",
        prediction_rows=1,
    )

    def repository_write():
        started.set()
        try:
            TrainingRecordRepository.for_workspace(
                WorkspaceContext.from_root(root)
            ).merge((ModelRecordOutcome(
                key="other@static", requested_operation="train",
                outcome="success", entry=extra,
            ),))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            finished.set()

    def interleave(point, _member=None):
        if point == "after_member_replace" and not writer:
            thread = threading.Thread(target=repository_write)
            writer.append(thread)
            thread.start()
            assert started.wait(1)
            assert not finished.wait(0.05)

    coordinator = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14),
        fault_hook=interleave,
    )
    coordinator.commit(coordinator.prepare((result,)))
    assert finished.wait(2)
    writer[0].join(timeout=1)

    assert errors == []
    assert set(json.loads((root / "latest_train_records.json").read_text())["models"]) == {
        "demo@static", "other@static",
    }
