"""Fault-timeline contracts for Rolling State V2 atomic persistence."""

import dataclasses
import fcntl
from pathlib import Path

import pytest

from quantpits.rolling.identity import workspace_fingerprint
from quantpits.rolling.repository import RollingStateRepository
from quantpits.rolling.state import RollingStateV2Snapshot, serialize_rolling_state_v2
from quantpits.utils.workspace import WorkspaceContext


WINDOW = "rolling:2024-01-01:2024-03-31:abcdef123456"
BEFORE_REPLACE_POINTS = (
    "before_lock",
    "after_lock_before_source_read",
    "after_source_read_before_temp_create",
    "after_temp_create",
    "during_temp_write",
    "after_temp_write_before_fsync",
    "after_temp_fsync",
    "after_cas_recheck_before_replace",
)
AFTER_REPLACE_POINTS = (
    "after_replace_before_directory_fsync",
    "after_directory_fsync_before_reread",
)


def _context(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    return WorkspaceContext.from_root(root)


def _state(context, **changes):
    values = {
        "workspace_fingerprint": workspace_fingerprint(context.root),
        "run_id": "rolling-fault-run",
        "family": "rolling",
        "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
        "anchor_date": "2024-03-31",
        "target_keys": ("demo@rolling",),
        "window_keys": (WINDOW,),
        "attempt_id": None,
        "phase": "prepared",
        "units": (),
    }
    values.update(changes)
    return RollingStateV2Snapshot(**values)


def _prepared(context):
    repository = RollingStateRepository.for_workspace(context, "rolling")
    receipt = repository.commit(
        _state(context), repository.inspect_readonly().baseline,
    )
    assert receipt.status == "committed"
    return repository, receipt


def _assert_lock_reacquirable(path):
    with Path(path).open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_fault_hook_named_points_fire_exactly_once(tmp_path):
    context = _context(tmp_path)
    repository, prepared = _prepared(context)
    points = []
    faulted = RollingStateRepository.for_workspace(
        context, "rolling", fault_hook=points.append,
    )
    proposed = _state(context, phase="executing", attempt_id="attempt-1")
    receipt = faulted.commit(proposed, prepared.after_baseline)
    assert receipt.status == "committed"
    assert points == [
        "before_lock",
        "after_lock_before_source_read",
        "after_source_read_before_temp_create",
        "after_temp_create",
        "during_temp_write",
        "after_temp_write_before_fsync",
        "after_temp_fsync",
        "after_cas_recheck_before_replace",
        "after_replace_before_directory_fsync",
        "after_directory_fsync_before_reread",
    ]
    assert len(points) == len(set(points))


def test_fault_injector_deliberate_violation_is_observable(tmp_path):
    context = _context(tmp_path)
    repository, prepared = _prepared(context)

    def fail(point):
        if point == "after_source_read_before_temp_create":
            raise AssertionError("deliberate fault-probe sentinel")

    faulted = RollingStateRepository.for_workspace(
        context, "rolling", fault_hook=fail,
    )
    proposed = _state(context, phase="executing", attempt_id="attempt-1")
    with pytest.raises(AssertionError, match="fault-probe sentinel"):
        faulted.commit(proposed, prepared.after_baseline)
    assert repository.state_path.read_bytes() == serialize_rolling_state_v2(
        _state(context),
    )


@pytest.mark.parametrize("fault_point", BEFORE_REPLACE_POINTS)
def test_before_replace_faults_preserve_exact_preimage_and_clean_temp(
        tmp_path, fault_point):
    context = _context(tmp_path)
    repository, prepared = _prepared(context)
    preimage = repository.state_path.read_bytes()
    fired = []

    def fail(point):
        if point == fault_point:
            fired.append(point)
            raise OSError("injected before-replace fault")

    faulted = RollingStateRepository.for_workspace(
        context, "rolling", fault_hook=fail,
    )
    proposed = _state(context, phase="executing", attempt_id="attempt-1")
    receipt = faulted.commit(proposed, prepared.after_baseline)
    assert fired == [fault_point]
    assert receipt.status == "write_failed_before_replace"
    assert receipt.did_write is False
    assert repository.state_path.read_bytes() == preimage
    assert not any(item.name.endswith(".tmp") for item in context.data_dir.iterdir())
    _assert_lock_reacquirable(repository.lock_path)


def test_pre_replace_drift_is_conflict_not_overwrite(tmp_path):
    context = _context(tmp_path)
    repository, prepared = _prepared(context)
    foreign_complete = serialize_rolling_state_v2(
        _state(context, phase="failed"),
    )

    def drift(point):
        if point == "after_temp_fsync":
            repository.state_path.write_bytes(foreign_complete)

    faulted = RollingStateRepository.for_workspace(
        context, "rolling", fault_hook=drift,
    )
    proposed = _state(context, phase="executing", attempt_id="attempt-1")
    receipt = faulted.commit(proposed, prepared.after_baseline)
    assert receipt.status == "conflict"
    assert receipt.did_write is False
    assert repository.state_path.read_bytes() == foreign_complete
    assert not any(item.name.endswith(".tmp") for item in context.data_dir.iterdir())


@pytest.mark.parametrize("fault_point", AFTER_REPLACE_POINTS)
def test_after_replace_faults_never_report_committed_without_durability_and_reread(
        tmp_path, fault_point):
    context = _context(tmp_path)
    repository, prepared = _prepared(context)
    proposed = _state(context, phase="executing", attempt_id="attempt-1")
    postimage = serialize_rolling_state_v2(proposed)

    def fail(point):
        if point == fault_point:
            raise OSError("injected after-replace fault")

    faulted = RollingStateRepository.for_workspace(
        context, "rolling", fault_hook=fail,
    )
    receipt = faulted.commit(proposed, prepared.after_baseline)
    assert receipt.status == "durability_uncertain"
    assert receipt.status != "committed"
    assert receipt.did_write is True
    assert repository.state_path.read_bytes() == postimage
    assert receipt.cas_baseline is None
    _assert_lock_reacquirable(repository.lock_path)


@pytest.mark.parametrize("fault_point", [
    "after_lock_before_source_read",
    "after_temp_create",
    "after_temp_fsync",
    "after_replace_before_directory_fsync",
])
def test_interrupt_releases_lock_and_never_exposes_partial_bytes(
        tmp_path, fault_point):
    context = _context(tmp_path)
    repository, prepared = _prepared(context)
    preimage = repository.state_path.read_bytes()
    proposed = _state(context, phase="executing", attempt_id="attempt-1")
    postimage = serialize_rolling_state_v2(proposed)

    def interrupt(point):
        if point == fault_point:
            raise KeyboardInterrupt()

    faulted = RollingStateRepository.for_workspace(
        context, "rolling", fault_hook=interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        faulted.commit(proposed, prepared.after_baseline)
    assert repository.state_path.read_bytes() in (preimage, postimage)
    assert repository.state_path.read_bytes() not in (b"",)
    assert not any(item.name.endswith(".tmp") for item in context.data_dir.iterdir())
    _assert_lock_reacquirable(repository.lock_path)


@pytest.mark.parametrize("fault_point", [
    "after_lock_before_source_read",
    "after_cas_recheck_before_replace",
    "after_replace_before_directory_fsync",
])
def test_delete_faults_preserve_complete_or_missing_authoritative_state(
        tmp_path, fault_point):
    context = _context(tmp_path)
    repository, prepared = _prepared(context)
    failed = _state(context, phase="failed")
    failed_receipt = repository.commit(failed, prepared.after_baseline)
    preimage = repository.state_path.read_bytes()

    def fail(point):
        if point == fault_point:
            raise OSError("injected delete fault")

    faulted = RollingStateRepository.for_workspace(
        context, "rolling", fault_hook=fail,
    )
    receipt = faulted.delete(failed_receipt.after_baseline)
    if fault_point == "after_replace_before_directory_fsync":
        assert receipt.status == "durability_uncertain"
        assert receipt.did_write is True
        assert not repository.state_path.exists()
    else:
        assert receipt.status == "write_failed_before_replace"
        assert receipt.did_write is False
        assert repository.state_path.read_bytes() == preimage
    _assert_lock_reacquirable(repository.lock_path)
