"""Contracts for the canonical locked Rolling State V2 repository."""

import dataclasses
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from quantpits.rolling.errors import RollingIdentityError, RollingStatePathError
from quantpits.rolling.identity import workspace_fingerprint
from quantpits.rolling.repository import (
    RollingStateBaseline,
    RollingStateMutationReceipt,
    RollingStateRepository,
)
from quantpits.rolling.state import (
    RollingStateUnitClaim,
    RollingStateV2Snapshot,
    serialize_rolling_state_v2,
)
from quantpits.utils.workspace import WorkspaceContext


WINDOW_A = "rolling:2024-01-01:2024-03-31:abcdef123456"
WINDOW_B = "rolling:2024-04-01:2024-06-30:111111111111"


def _context(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    return WorkspaceContext.from_root(root)


def _state(context, **changes):
    values = {
        "workspace_fingerprint": workspace_fingerprint(context.root),
        "run_id": "rolling-demo-run",
        "family": "rolling",
        "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
        "anchor_date": "2024-03-31",
        "target_keys": ("demo@rolling",),
        "window_keys": (WINDOW_A, WINDOW_B),
        "attempt_id": None,
        "phase": "prepared",
        "units": (),
    }
    values.update(changes)
    return RollingStateV2Snapshot(**values)


def _commit_prepared(repository, context):
    expected = repository.inspect_readonly().baseline
    receipt = repository.commit(_state(context), expected)
    assert receipt.status == "committed"
    return receipt


def _unit_extensions(values):
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _failure_claim(attempt_id, prior=(), failure_code="RuntimeError", source=None):
    extensions = {"attempt_id": attempt_id, "failure_code": failure_code}
    extensions.update(source or {})
    if prior:
        extensions["prior_attempts"] = list(prior)
    return RollingStateUnitClaim(
        "demo@rolling", WINDOW_A, "failed",
        _extensions_json=_unit_extensions(extensions),
    )


def _retry_claim(attempt_id, prior):
    return RollingStateUnitClaim(
        "demo@rolling", WINDOW_A, "running",
        _extensions_json=_unit_extensions({
            "attempt_id": attempt_id,
            "prior_attempts": list(prior),
        }),
    )


def _competing_commit(root, expected, proposed, start, output):
    context = WorkspaceContext.from_root(root)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    start.wait()
    output.put(repository.commit(proposed, expected).status)


def _hold_lock(path, ready, release):
    with Path(path).open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_repository_factory_uses_canonical_family_path_and_sibling_lock(tmp_path):
    context = _context(tmp_path)
    rolling = RollingStateRepository.for_workspace(context, "rolling")
    cpcv = RollingStateRepository.for_workspace(context, "cpcv_rolling")
    assert rolling.state_path == context.data_dir / "rolling_state.json"
    assert rolling.lock_path == context.data_dir / "rolling_state.json.lock"
    assert cpcv.state_path == context.data_dir / "rolling_state_cpcv.json"
    assert cpcv.lock_path == context.data_dir / "rolling_state_cpcv.json.lock"
    with pytest.raises(RollingStatePathError):
        RollingStateRepository.for_workspace(context, "slide")


def test_repository_rejects_noncanonical_context_and_paths_are_readonly(tmp_path):
    context = _context(tmp_path)
    foreign_data = context.root / "nested" / "data"
    malformed = dataclasses.replace(context, data_dir=foreign_data)
    with pytest.raises(RollingStatePathError, match="not canonical"):
        RollingStateRepository.for_workspace(malformed, "rolling")

    repository = RollingStateRepository.for_workspace(context, "rolling")
    for field, value in (
            ("context", malformed),
            ("family", "cpcv_rolling"),
            ("state_name", "foreign.json"),
            ("lock_name", "foreign.lock"),
            ("relative_path", "data/foreign.json"),
            ("state_path", context.root / "foreign.json"),
            ("lock_path", context.root / "foreign.lock")):
        with pytest.raises(AttributeError):
            setattr(repository, field, value)
    assert repository.state_path == context.root / "data" / "rolling_state.json"
    assert repository.lock_path == context.root / "data" / "rolling_state.json.lock"


def test_readonly_view_is_one_coherent_snapshot_and_creates_nothing(tmp_path):
    root = tmp_path / "Demo_Workspace"
    context = WorkspaceContext.from_root(root)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    before = tuple(root.parent.iterdir())
    view = repository.inspect_readonly()
    assert view.inspection.classification == "missing"
    assert view.baseline == RollingStateBaseline(
        "data/rolling_state.json", "missing", False,
    )
    assert tuple(root.parent.iterdir()) == before
    assert not root.exists()

    (root / "data").mkdir(parents=True)
    payload = serialize_rolling_state_v2(_state(context))
    repository.state_path.write_bytes(payload)
    view = repository.inspect_readonly()
    assert view.baseline.fingerprint == view.inspection.fingerprint
    assert view.baseline.size_bytes == len(payload)
    assert tuple(item.name for item in context.data_dir.iterdir()) == (
        "rolling_state.json",
    )


def test_repository_rejects_direct_parent_and_lock_symlink_or_special_node(
        tmp_path, monkeypatch):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    outside = tmp_path / "outside.json"
    outside.write_text("private sentinel", encoding="utf-8")
    repository.state_path.symlink_to(outside)
    with pytest.raises(RollingStatePathError):
        repository.inspect_readonly()
    assert outside.read_text(encoding="utf-8") == "private sentinel"

    repository.state_path.unlink()
    repository.state_path.mkdir()
    with pytest.raises(RollingStatePathError):
        repository.inspect_readonly()
    repository.state_path.rmdir()

    repository.lock_path.symlink_to(outside)
    expected = repository.inspect_readonly().baseline
    assert repository.commit(_state(context), expected).status == "invalid_source"
    assert outside.read_text(encoding="utf-8") == "private sentinel"

    repository.lock_path.unlink()
    repository.lock_path.mkdir()
    assert repository.commit(_state(context), expected).status == "invalid_source"
    repository.lock_path.rmdir()

    monkeypatch.setattr(
        "quantpits.rolling.repository.secrets.token_hex", lambda _size: "fixed",
    )
    temporary = context.data_dir / ".rolling_state.json.fixed.tmp"
    temporary.symlink_to(outside)
    assert repository.commit(_state(context), expected).status == "write_failed_before_replace"
    assert outside.read_text(encoding="utf-8") == "private sentinel"
    temporary.unlink()
    repository.lock_path.unlink()

    context.data_dir.rmdir()
    outside_dir = tmp_path / "outside-data"
    outside_dir.mkdir()
    context.data_dir.symlink_to(outside_dir, target_is_directory=True)
    assert repository.commit(_state(context), expected).status == "invalid_source"
    assert tuple(outside_dir.iterdir()) == ()


@pytest.mark.parametrize("change", [
    {"existed": 1},
    {"size_bytes": True},
    {"size_bytes": -1},
    {"size_bytes": 1.0},
    {"fingerprint": "A" * 64},
    {"fingerprint": "a" * 63},
    {"path_kind": "missing"},
])
def test_baseline_rejects_contradictory_and_coercible_representations(change):
    values = {
        "relative_path": "data/rolling_state.json",
        "path_kind": "file",
        "existed": True,
        "size_bytes": 1,
        "fingerprint": "a" * 64,
    }
    values.update(change)
    with pytest.raises(RollingIdentityError):
        RollingStateBaseline(**values)
    with pytest.raises(RollingIdentityError):
        RollingStateBaseline(
            "data/rolling_state.json", "missing", False, None, "a" * 64,
        )


def test_receipt_rejects_status_write_and_postimage_contradictions():
    missing = RollingStateBaseline(
        "data/rolling_state.json", "missing", False,
    )
    existing = RollingStateBaseline(
        "data/rolling_state.json", "file", True, 1, "a" * 64,
    )
    with pytest.raises(RollingIdentityError):
        RollingStateMutationReceipt(
            "create", "committed", "rolling_state_committed", False,
            "data/rolling_state.json", missing, existing,
        )
    with pytest.raises(RollingIdentityError):
        RollingStateMutationReceipt(
            "transition", "conflict", "rolling_state_committed", False,
            "data/rolling_state.json", existing, None,
        )
    with pytest.raises(RollingIdentityError):
        RollingStateMutationReceipt(
            "delete", "deleted", "rolling_state_deleted", True,
            "data/rolling_state.json", existing, existing,
        )
    with pytest.raises(RollingIdentityError):
        RollingStateMutationReceipt(
            "transition", "conflict", "rolling_state_cas_conflict", 0,
            "data/rolling_state.json", existing, None,
        )
    with pytest.raises(RollingIdentityError):
        RollingStateMutationReceipt(
            "transition", "unknown", "rolling_state_cas_conflict", False,
            "data/rolling_state.json", existing, None,
        )


def test_receipt_rejects_operation_phase_and_cas_baseline_contradictions():
    missing = RollingStateBaseline(
        "data/rolling_state.json", "missing", False,
    )
    before = RollingStateBaseline(
        "data/rolling_state.json", "file", True, 1, "a" * 64,
    )
    after = RollingStateBaseline(
        "data/rolling_state.json", "file", True, 1, "b" * 64,
    )
    contradictions = (
        ("delete", "committed", "rolling_state_committed", True,
         before, after, "failed", "executing"),
        ("transition", "unchanged", "rolling_state_unchanged", False,
         before, after, "executing", "executing"),
        ("transition", "unchanged", "rolling_state_unchanged", False,
         before, before, "prepared", "executing"),
        ("create", "committed", "rolling_state_committed", True,
         missing, after, None, "executing"),
        ("transition", "committed", "rolling_state_committed", True,
         before, after, "failed", "failed"),
        ("delete", "deleted", "rolling_state_deleted", True,
         before, missing, "prepared", None),
        ("delete", "missing_noop", "rolling_state_missing", False,
         missing, missing, "failed", None),
        ("create", "durability_uncertain",
         "rolling_state_durability_uncertain", True,
         before, after, "executing", "prepared"),
        ("transition", "durability_uncertain",
         "rolling_state_durability_uncertain", True,
         before, after, "failed", "failed"),
        ("delete", "durability_uncertain",
         "rolling_state_durability_uncertain", True,
         before, missing, "executing", None),
    )
    for facts in contradictions:
        with pytest.raises(RollingIdentityError):
            RollingStateMutationReceipt(
                *facts[:4], "data/rolling_state.json", *facts[4:],
            )


def test_receipt_enforces_baseline_phase_and_interrupted_fact_consistency():
    missing = RollingStateBaseline(
        "data/rolling_state.json", "missing", False,
    )
    before = RollingStateBaseline(
        "data/rolling_state.json", "file", True, 1, "a" * 64,
    )
    after = RollingStateBaseline(
        "data/rolling_state.json", "file", True, 1, "b" * 64,
    )
    contradictions = (
        ("transition", "conflict", "rolling_state_cas_conflict", False,
         missing, None, "completed", None),
        ("transition", "durability_uncertain",
         "rolling_state_durability_uncertain", True,
         before, None, "executing", "failed"),
        ("transition", "interrupted", "rolling_state_interrupted", False,
         before, after, "executing", "failed"),
        ("create", "interrupted", "rolling_state_interrupted", True,
         before, after, "prepared", "prepared"),
        ("transition", "interrupted", "rolling_state_interrupted", True,
         missing, after, None, "prepared"),
        ("transition", "interrupted", "rolling_state_interrupted", True,
         before, None, None, None),
        ("delete", "interrupted", "rolling_state_interrupted", True,
         before, missing, "executing", None),
    )
    for facts in contradictions:
        with pytest.raises(RollingIdentityError):
            RollingStateMutationReceipt(
                *facts[:4], "data/rolling_state.json", *facts[4:],
            )
    coherent = (
        ("create", False, None, None, None, None),
        ("create", True, missing, after, None, "prepared"),
        ("transition", False, before, None, "executing", None),
        ("transition", True, before, after, "executing", "failed"),
        ("delete", True, before, missing, "failed", None),
    )
    for facts in coherent:
        (
            operation, did_write, before_fact, after_fact,
            before_phase, after_phase,
        ) = facts
        receipt = RollingStateMutationReceipt(
            operation, "interrupted", "rolling_state_interrupted", did_write,
            "data/rolling_state.json", before_fact, after_fact,
            before_phase, after_phase,
        )
        assert receipt.status == "interrupted"


def test_missing_create_and_pre_evidence_phase_transitions_are_monotonic(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    prepared_receipt = _commit_prepared(repository, context)
    prepared = repository.inspect_readonly()
    assert prepared_receipt.after_baseline == prepared.baseline

    executing_state = _state(
        context, phase="executing", attempt_id="attempt-1",
        units=(RollingStateUnitClaim("demo@rolling", WINDOW_A, "running"),),
    )
    executing = repository.commit(executing_state, prepared.baseline)
    assert executing.status == "committed"
    failure = {"attempt_id": "attempt-1", "failure_code": "RuntimeError"}
    failed_state = dataclasses.replace(
        executing_state, phase="failed",
        units=(_failure_claim("attempt-1"),),
    )
    failed = repository.commit(failed_state, executing.after_baseline)
    assert failed.status == "committed"
    retry_state = dataclasses.replace(
        failed_state, phase="executing", attempt_id="attempt-2",
        units=(_retry_claim("attempt-2", (failure,)),),
    )
    retry = repository.commit(retry_state, failed.after_baseline)
    assert retry.status == "committed"
    assert retry.before_phase == "failed"
    assert retry.after_phase == "executing"


def test_retry_attempt_audit_is_canonical_append_only_and_fails_closed(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    _commit_prepared(repository, context)
    baseline = repository.inspect_readonly().baseline

    running0 = _state(
        context, phase="executing", attempt_id="attempt-0",
        units=(RollingStateUnitClaim("demo@rolling", WINDOW_A, "running"),),
    )
    baseline = repository.commit(running0, baseline).after_baseline
    source0 = {
        "source_manifest_fingerprint": "a" * 64,
        "source_protocol": "execution_bound_v1",
        "source_publication_key": "demo@rolling",
        "experiment_name": "exact-unit-experiment",
        "experiment_id": "experiment-0",
        "recorder_id": "recorder-0",
        "source_operation": "merge",
        "artifacts": [],
    }
    audit0 = dict(
        source0, attempt_id="attempt-0", failure_code="RuntimeError",
    )
    failed0 = dataclasses.replace(
        running0, phase="failed",
        units=(_failure_claim("attempt-0", source=source0),),
    )
    baseline = repository.commit(failed0, baseline).after_baseline
    running1 = dataclasses.replace(
        failed0, phase="executing", attempt_id="attempt-1",
        units=(_retry_claim("attempt-1", (audit0,)),),
    )
    baseline = repository.commit(running1, baseline).after_baseline
    audit1 = {"attempt_id": "attempt-1", "failure_code": "OSError"}
    failed1 = dataclasses.replace(
        running1, phase="failed",
        units=(_failure_claim("attempt-1", (audit0,), "OSError"),),
    )
    baseline = repository.commit(failed1, baseline).after_baseline

    exact = (audit0, audit1)
    canonical = dataclasses.replace(
        failed1, phase="executing", attempt_id="attempt-2",
        units=(_retry_claim("attempt-2", exact),),
    )
    for forged_prior in (
        (audit1,),
        (dict(audit0, failure_code="ValueError"), audit1),
        (dict(audit0, recorder_id="forged-recorder"), audit1),
        (audit1, audit0),
        (audit0, audit1, audit1),
    ):
        forged = dataclasses.replace(
            canonical, units=(_retry_claim("attempt-2", forged_prior),),
        )
        receipt = repository.commit(forged, baseline)
        assert receipt.status == "invalid_transition"
        assert repository.inspect_readonly().baseline == baseline

    for malformed_prior in (None, False, {}, "attempt-0"):
        malformed = RollingStateUnitClaim(
            "demo@rolling", WINDOW_A, "running",
            _extensions_json=_unit_extensions({
                "attempt_id": "attempt-2",
                "prior_attempts": malformed_prior,
            }),
        )
        forged = dataclasses.replace(canonical, units=(malformed,))
        assert repository.commit(forged, baseline).status == "invalid_transition"
        assert repository.inspect_readonly().baseline == baseline

    assert repository.commit(canonical, baseline).status == "committed"


def test_transition_rejects_identity_attempt_phase_and_unit_regression(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    _commit_prepared(repository, context)
    prepared = repository.inspect_readonly()
    executing = _state(context, phase="executing", attempt_id="attempt-1")
    committed = repository.commit(executing, prepared.baseline)

    for field, value in (
            ("run_id", "other-run"),
            ("config_fingerprint", "d" * 64),
            ("target_keys", ("other@rolling",)),
            ("attempt_id", "attempt-2")):
        candidate = dataclasses.replace(executing, **{field: value})
        receipt = repository.commit(candidate, committed.after_baseline)
        assert receipt.status == "invalid_transition"
        assert repository.inspect_readonly().baseline == committed.after_baseline

    running = RollingStateUnitClaim("demo@rolling", WINDOW_A, "running")
    progress = dataclasses.replace(executing, units=(running,))
    progress_receipt = repository.commit(progress, committed.after_baseline)
    regressed = dataclasses.replace(
        progress,
        units=(RollingStateUnitClaim("demo@rolling", WINDOW_A, "pending"),),
    )
    rejected = repository.commit(regressed, progress_receipt.after_baseline)
    assert rejected.status == "invalid_transition"


def test_create_rejects_foreign_workspace_before_temp_or_replace(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    expected = repository.inspect_readonly().baseline
    proposed = _state(context, workspace_fingerprint="f" * 64)
    receipt = repository.commit(proposed, expected)
    assert receipt.status == "invalid_transition"
    assert receipt.did_write is False
    assert not repository.state_path.exists()
    assert not any(item.name.endswith(".tmp") for item in context.data_dir.iterdir())


def test_completion_claim_is_denied_until_evidence_phase(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    _commit_prepared(repository, context)
    prepared = repository.inspect_readonly()
    completed = _state(context, phase="completed")
    assert repository.commit(completed, prepared.baseline).status == "invalid_transition"
    success = _state(
        context,
        phase="executing",
        attempt_id="attempt-1",
        units=(RollingStateUnitClaim(
            "demo@rolling", WINDOW_A, "success", record_id="recorder-demo",
        ),),
    )
    receipt = repository.commit(success, prepared.baseline)
    assert receipt.status == "invalid_transition"
    assert receipt.cas_baseline is None


def test_stale_expected_baseline_conflicts_without_authoritative_write(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    missing = repository.inspect_readonly().baseline
    _commit_prepared(repository, context)
    before = repository.state_path.read_bytes()
    candidate = _state(context, phase="executing", attempt_id="attempt-1")
    receipt = repository.commit(candidate, missing)
    assert receipt.status == "conflict"
    assert receipt.did_write is False
    assert receipt.cas_baseline is None
    assert repository.state_path.read_bytes() == before
    assert not any(item.name.endswith(".tmp") for item in context.data_dir.iterdir())


def test_two_processes_with_one_baseline_produce_one_commit_and_one_conflict(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    _commit_prepared(repository, context)
    expected = repository.inspect_readonly().baseline
    proposed = _state(context, phase="executing", attempt_id="attempt-1")
    process_context = multiprocessing.get_context("fork")
    start = process_context.Event()
    output = process_context.Queue()
    processes = [
        process_context.Process(
            target=_competing_commit,
            args=(context.root, expected, proposed, start, output),
        ) for _index in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    statuses = sorted(output.get(timeout=5) for _index in processes)
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    assert statuses == ["committed", "conflict"]
    assert repository.state_path.read_bytes() == serialize_rolling_state_v2(proposed)


def test_retry_requires_fresh_baseline_and_revalidates_transition(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    _commit_prepared(repository, context)
    prepared = repository.inspect_readonly().baseline
    executing = _state(context, phase="executing", attempt_id="attempt-1")
    first = repository.commit(executing, prepared)
    assert repository.commit(executing, prepared).status == "conflict"
    assert repository.commit(executing, first.after_baseline).status == "unchanged"


def test_nonblocking_lock_contention_returns_stable_no_write_result(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    expected = repository.inspect_readonly().baseline
    process_context = multiprocessing.get_context("fork")
    ready = process_context.Event()
    release = process_context.Event()
    holder = process_context.Process(
        target=_hold_lock, args=(repository.lock_path, ready, release),
    )
    holder.start()
    assert ready.wait(timeout=5)
    receipt = repository.commit(_state(context), expected, blocking=False)
    release.set()
    holder.join(timeout=5)
    assert holder.exitcode == 0
    assert receipt.status == "lock_unavailable"
    assert not repository.state_path.exists()


def test_persistent_unheld_lock_file_is_not_treated_as_contention(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    repository.lock_path.touch()
    expected = repository.inspect_readonly().baseline
    assert repository.commit(_state(context), expected, blocking=False).status == "committed"


def test_missing_fcntl_never_falls_back_to_unlocked_mutation(tmp_path, monkeypatch):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    expected = repository.inspect_readonly().baseline
    monkeypatch.setattr("quantpits.rolling.repository.fcntl", None)
    receipt = repository.commit(_state(context), expected)
    assert receipt.status == "lock_unavailable"
    assert repository.inspect_readonly().inspection.classification == "missing"


def test_compare_and_delete_requires_exact_failed_v2_baseline(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    _commit_prepared(repository, context)
    prepared = repository.inspect_readonly().baseline
    assert repository.delete(prepared).status == "invalid_source"
    with pytest.raises(TypeError):
        repository.delete(prepared, require_failed=False)
    failed = _state(context, phase="failed")
    failed_receipt = repository.commit(failed, prepared)
    stale = prepared
    assert repository.delete(stale).status == "conflict"
    deleted = repository.delete(failed_receipt.after_baseline)
    assert deleted.status == "deleted"
    assert deleted.after_baseline.existed is False
    assert repository.delete(deleted.after_baseline).status == "missing_noop"
    assert not any(item.name.startswith("history") for item in context.data_dir.iterdir())


def test_receipt_fingerprints_are_recomputed_from_actual_bytes(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    receipt = _commit_prepared(repository, context)
    actual = repository.state_path.read_bytes()
    assert receipt.after_baseline.fingerprint == hashlib.sha256(actual).hexdigest()
    assert receipt.after_baseline.size_bytes == len(actual)


def test_noncommitted_receipts_cannot_grant_postimage_or_execution_capability(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    missing = repository.inspect_readonly().baseline
    _commit_prepared(repository, context)
    conflict = repository.commit(_state(context), missing)
    assert conflict.status == "conflict"
    assert conflict.cas_baseline is None
    assert not hasattr(conflict, "execute")
    assert not hasattr(conflict, "publish")


def test_repository_rejects_direct_commit_over_legacy_source_even_with_valid_proposal(tmp_path):
    context = _context(tmp_path)
    repository = RollingStateRepository.for_workspace(context, "rolling")
    repository.state_path.write_text(
        '{"completed_windows": {}, "training_method": "slide"}',
        encoding="utf-8",
    )
    baseline = repository.inspect_readonly().baseline
    before = repository.state_path.read_bytes()
    receipt = repository.commit(_state(context), baseline)
    assert receipt.status == "invalid_source"
    assert repository.state_path.read_bytes() == before


def test_evidence_authorized_success_and_units_complete_reject_impossible_combinations(tmp_path):
    from quantpits.rolling import RollingExecutionKernel
    from tests.quantpits.rolling.execution_support import (
        FakeExecutionBackend,
        FakeRunner,
        linear_capability_result,
        make_scope,
    )

    context = _context(tmp_path)
    scope = make_scope(context, linear_capability_result())
    repository = RollingStateRepository.for_workspace(context, "rolling")
    backend = FakeExecutionBackend(context)
    result = RollingExecutionKernel(repository, backend, FakeRunner(context)).execute(
        scope, "attempt-1",
    )
    assert result.status == "success"
    view = repository.inspect_readonly()
    assert view.inspection.snapshot.phase == "units_complete"
    assert repository.commit(view.inspection.snapshot, view.baseline).status == "invalid_transition"

    evidence = backend.inspect(
        scope, backend.requests_for_state(scope, view.inspection.snapshot),
    )
    unit = dataclasses.replace(
        view.inspection.snapshot.units[0], evidence_id="f" * 64,
    )
    forged = dataclasses.replace(view.inspection.snapshot, units=(unit,))
    receipt = repository.commit_evidence_authorized(
        forged, view.baseline, evidence,
    )
    assert receipt.status == "invalid_transition"
    assert repository.inspect_readonly().baseline == view.baseline

    original = view.inspection.snapshot.units[0]
    original_extensions = original.extensions
    forged_extensions = []
    missing_attempt = dict(original_extensions)
    missing_attempt.pop("attempt_id")
    forged_extensions.append(missing_attempt)
    for field, value in (
        ("attempt_id", "forged-attempt"),
        ("experiment_id", "forged-experiment"),
        ("source_manifest_fingerprint", "f" * 64),
        ("source_operation", "daily"),
        ("artifacts", []),
    ):
        changed = dict(original_extensions)
        changed[field] = value
        forged_extensions.append(changed)
    for extensions in forged_extensions:
        unit = dataclasses.replace(
            original,
            _extensions_json=json.dumps(
                extensions, sort_keys=True, separators=(",", ":"),
            ),
        )
        proposed = dataclasses.replace(
            view.inspection.snapshot, units=(unit,),
        )
        denied = repository.commit_evidence_authorized(
            proposed, view.baseline, evidence,
        )
        assert denied.status == "invalid_transition"
        assert repository.inspect_readonly().baseline == view.baseline
