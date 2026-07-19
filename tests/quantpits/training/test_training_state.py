import pytest

from quantpits.training.errors import TrainingStateConflictError
from quantpits.training.state import TrainingRunState, TrainingStateRepository


def state(phase="prepared", run_id="run-1", **kwargs):
    values = dict(
        run_id=run_id, family="static", action="incremental",
        plan_fingerprint="plan", execution_fingerprint="execution",
        resume_fingerprint="resume", anchor_date="2026-07-10",
        target_keys=("demo@static",), outcomes={}, phase=phase,
    )
    values.update(kwargs)
    return TrainingRunState(**values)


def test_state_repository_uses_cas_and_validates_transitions(tmp_path):
    repo = TrainingStateRepository(tmp_path / "run_state.json")
    current, absent = repo.inspect_readonly()
    assert current is None
    prepared = repo.save(state(), expected=absent)
    executing = repo.save(state("executing"), expected=prepared)
    with pytest.raises(TrainingStateConflictError, match="changed"):
        repo.save(state("targets_complete"), expected=prepared)
    with pytest.raises(TrainingStateConflictError, match="transition"):
        repo.save(state("completed"), expected=executing)


def test_state_rejects_other_run_and_published_without_recorder(tmp_path):
    repo = TrainingStateRepository(tmp_path / "run_state.json")
    baseline = repo.save(state())
    with pytest.raises(TrainingStateConflictError, match="another run"):
        repo.save(state("executing", run_id="run-2"), expected=baseline)
    with pytest.raises(TrainingStateConflictError, match="recorder evidence"):
        state("executing", outcomes={"demo@static": {"published": True}})


def test_readonly_state_inspect_creates_nothing(tmp_path):
    repo = TrainingStateRepository(tmp_path / "missing" / "run_state.json")
    assert repo.inspect_readonly()[0] is None
    assert not (tmp_path / "missing").exists()


def test_explicit_clear_accepts_legacy_state_without_parsing_it(tmp_path):
    path = tmp_path / "run_state.json"
    path.write_text('{"mode":"cpcv_incremental","completed":["demo"]}')

    TrainingStateRepository(path).clear()

    assert not path.exists()


def test_terminal_only_clear_rejects_legacy_state_without_deleting_it(tmp_path):
    path = tmp_path / "run_state.json"
    original = b'{"mode":"cpcv_incremental","completed":["demo"]}'
    path.write_bytes(original)

    with pytest.raises(TrainingStateConflictError, match="legacy.*display-only"):
        TrainingStateRepository(path).clear(require_terminal=True)

    assert path.read_bytes() == original


def test_state_v3_round_trips_recovery_evidence_fields():
    value = state(
        "closing", attempt_id="attempt-1", receipt_fingerprint="receipt",
        committed_outputs=({
            "path": "latest_train_records.json", "kind": "record", "fingerprint": "post",
        },),
        closure_steps={
            "receipt_verified": "completed", "state_publication_bound": "completed",
            "history_appended": "completed",
        },
        publication_transaction_id="tx", publication_status="committed",
        logical_started_at="2026-07-14T00:00:00",
    )
    assert TrainingRunState.from_dict(value.to_dict()) == value


def test_state_requires_consistent_publication_provenance():
    with pytest.raises(TrainingStateConflictError, match="newly and previously"):
        state("executing", outcomes={"demo@static": {
            "published": True, "recorder_id": "rid",
            "published_this_attempt": True, "already_published": True,
        }})
    with pytest.raises(TrainingStateConflictError, match="requires published"):
        state("executing", outcomes={"demo@static": {
            "published": False, "recorder_id": "rid",
            "published_this_attempt": True,
        }})


def test_state_rejects_contradictory_publication_and_closure_identity():
    ledger = ({
        "path": "latest_train_records.json", "kind": "record", "fingerprint": "post",
    },)
    with pytest.raises(TrainingStateConflictError, match="receipt evidence"):
        state(
            "closing", publication_transaction_id="tx",
            publication_status="committed",
        )
    with pytest.raises(TrainingStateConflictError, match="publication binding"):
        state(
            "closing", publication_transaction_id="tx", publication_status="committed",
            receipt_fingerprint="receipt", committed_outputs=ledger,
            closure_steps={"receipt_verified": "completed"},
        )
    with pytest.raises(TrainingStateConflictError, match="manifest"):
        state(
            "closing", publication_transaction_id="tx", publication_status="committed",
            receipt_fingerprint="receipt", committed_outputs=ledger,
            closure_steps={
                "receipt_verified": "completed", "state_publication_bound": "completed",
                "manifest_verified": "completed",
            },
        )


def test_completed_closure_step_cannot_regress(tmp_path):
    repo = TrainingStateRepository(tmp_path / "run_state.json")
    timing = {"logical_started_at": "2026-07-14T00:00:00"}
    baseline = repo.save(state("prepared", **timing))
    baseline = repo.save(state("executing", **timing), expected=baseline)
    baseline = repo.save(state("targets_complete", **timing), expected=baseline)
    baseline = repo.save(state(
        "publication_prepared", publication_transaction_id="tx",
        publication_status="prepared", **timing,
    ), expected=baseline)
    ledger = ({
        "path": "latest_train_records.json", "kind": "record", "fingerprint": "post",
    },)
    identity = dict(
        publication_transaction_id="tx", publication_status="committed",
        receipt_fingerprint="receipt", committed_outputs=ledger, **timing,
    )
    baseline = repo.save(state(
        "publication_committed", closure_steps={
            "receipt_verified": "completed", "state_publication_bound": "completed",
            "history_appended": "completed",
        }, **identity,
    ), expected=baseline)
    with pytest.raises(TrainingStateConflictError, match="regressed"):
        repo.save(state(
            "publication_committed", closure_steps={
                "receipt_verified": "completed", "state_publication_bound": "completed",
            }, **identity,
        ), expected=baseline)
