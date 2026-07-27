from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from quantpits.tools.verify_rolling_aggregate_candidate import (
    AggregateGateError,
    CLEANUP_AUTHORIZATION,
    EXECUTE_AUTHORIZATION,
    frozen_scenario,
    cleanup_gate_workspace,
    execute_gate,
    _assert_workspace_write_allowlist,
    _assert_gate_budgets,
    _assert_snapshot_unchanged,
    preflight_evidence,
    scenario_from_mapping,
    validate_binding,
)


def _roots(tmp_path):
    disposable = tmp_path / "Demo_Workspace"
    protected = tmp_path / "Protected_Demo_Workspace"
    disposable.mkdir()
    protected.mkdir()
    return disposable, protected


def test_verify_rolling_aggregate_gate_core_negative_matrix(
    tmp_path, monkeypatch,
):
    scenario = frozen_scenario()
    disposable, protected = _roots(tmp_path)
    commit = "1" * 40
    tree = "2" * 40

    def isolated_git(command, **_kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return commit + "\n"
        if command == ["git", "rev-parse", "HEAD^{tree}"]:
            return tree + "\n"
        if command == [
            "git", "status", "--porcelain", "--untracked-files=no",
        ]:
            return ""
        raise AssertionError("unexpected git command: %r" % (command,))

    monkeypatch.setattr(subprocess, "check_output", isolated_git)
    binding = validate_binding(
        scenario, disposable, protected, commit, tree,
    )
    evidence = preflight_evidence(binding)
    assert evidence["status"] == "preflight_passed"
    assert evidence["budgets"]["training_calls"] == 0
    for change in (
        {"family": "cpcv_rolling"},
        {"target_count": 0},
        {"target_count": 2},
        {"window_count": 1},
        {"window_count": 3},
        {"source_unit_count": 1},
        {"training": True},
        {"expected_new_recorders": 2},
        {"cleanup_default": "delete"},
    ):
        with pytest.raises(AggregateGateError):
            replace(scenario, **change)
    with pytest.raises(AggregateGateError):
        validate_binding(
            scenario, disposable, protected, commit, tree,
            execute=True, authorization="wrong",
        )
    authorized = validate_binding(
        scenario, disposable, protected, commit, tree,
        execute=True, authorization=EXECUTE_AUTHORIZATION,
    )
    assert authorized["execute"] is True
    protected_link = tmp_path / "Protected_Workspace_Link"
    protected_link.symlink_to(protected, target_is_directory=True)
    linked = validate_binding(
        scenario, disposable, protected_link, commit, tree,
    )
    assert linked["protected_workspace"] == protected.resolve()
    disposable_link = tmp_path / "Disposable_Workspace_Link"
    disposable_link.symlink_to(disposable, target_is_directory=True)
    with pytest.raises(AggregateGateError):
        validate_binding(
            scenario, disposable_link, protected, commit, tree,
        )
    from types import SimpleNamespace
    import quantpits.tools.verify_rolling_aggregate_candidate as gate_module
    monkeypatch.setattr(
        gate_module.shutil, "disk_usage",
        lambda _path: SimpleNamespace(free=1),
    )
    with pytest.raises(AggregateGateError):
        validate_binding(
            scenario, disposable, protected, commit, tree,
        )


def test_gate_scenario_rejects_unknown_and_non_strict_fields():
    payload = frozen_scenario().to_public_dict()
    payload["target_count"] = True
    with pytest.raises(AggregateGateError):
        scenario_from_mapping(payload)


def test_gate_write_observer_and_cleanup_fail_closed(tmp_path):
    disposable, protected = _roots(tmp_path)
    before = ()
    candidate_after = (
        ("data/rolling_aggregate_candidates_rolling", "directory", None, None),
        (
            "data/rolling_aggregate_candidates_rolling/run/artifacts/pred.pkl",
            "file", 4, __import__("hashlib").sha256(b"pred").hexdigest(),
        ),
    )
    assert _assert_workspace_write_allowlist(
        before, candidate_after,
    ) == (2, 4)
    declared_lock = (
        ("data", "directory", None, None),
        ("data/locks", "directory", None, None),
        (
            "data/locks/rolling_aggregate_candidate.lock",
            "file", 0, __import__("hashlib").sha256(b"").hexdigest(),
        ),
    )
    assert _assert_workspace_write_allowlist(
        before, declared_lock,
    ) == (3, 0)
    unexpected = disposable / "latest_train_records.json"
    unexpected.write_text("{}", encoding="utf-8")
    after = ((
        "latest_train_records.json", "file", 2,
        __import__("hashlib").sha256(b"{}").hexdigest(),
    ),)
    with pytest.raises(AggregateGateError):
        _assert_workspace_write_allowlist(before, after)
    with pytest.raises(AggregateGateError):
        _assert_gate_budgets(301, 0)
    with pytest.raises(AggregateGateError):
        _assert_gate_budgets(0, 513 * 1024 ** 2)
    with pytest.raises(AggregateGateError):
        _assert_snapshot_unchanged(
            (("protected",),), (("drifted",),),
            "protected workspace",
        )
    scenario = frozen_scenario()
    marker = disposable / "data" / "aggregate_gate_scenario.json"
    marker.parent.mkdir()
    marker.write_text(__import__("json").dumps({
        "protocol": scenario.protocol,
        "scenario_fingerprint": scenario.fingerprint,
    }, sort_keys=True), encoding="utf-8")
    with pytest.raises(AggregateGateError):
        cleanup_gate_workspace(
            disposable, protected, scenario.fingerprint, "wrong",
        )
    outcome = cleanup_gate_workspace(
        disposable, protected, scenario.fingerprint,
        CLEANUP_AUTHORIZATION,
    )
    assert outcome["status"] == "cleanup_completed"
    assert not disposable.exists()
    payload = frozen_scenario().to_public_dict()
    payload["unknown"] = "forged"
    with pytest.raises(AggregateGateError):
        scenario_from_mapping(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("new_candidate_recorders", 1),
        ("runner_calls", 1),
        ("training_calls", 1),
        ("write_bytes", 1),
        ("changed_path_count", 1),
        ("candidate_fingerprint", "f" * 64),
        ("candidate_row_count", 3),
    ],
)
def test_gate_rejects_every_nonzero_or_drifted_second_process_fact(
    tmp_path, monkeypatch, field, value,
):
    import json
    from types import SimpleNamespace
    import quantpits.tools.verify_rolling_aggregate_candidate as gate_module

    primary = {
        "status": "materialized_success",
        "candidate_fingerprint": "a" * 64,
        "candidate_row_count": 4,
        "new_candidate_recorders": 1,
        "training_calls": 0,
        "runner_calls": 2,
        "write_bytes": 4,
        "changed_path_count": 2,
    }
    reuse = {
        "status": "reused_success",
        "candidate_fingerprint": "a" * 64,
        "candidate_row_count": 4,
        "new_candidate_recorders": 0,
        "training_calls": 0,
        "runner_calls": 0,
        "write_bytes": 0,
        "changed_path_count": 0,
    }
    reuse[field] = value
    monkeypatch.setattr(
        gate_module, "_run_real_gate",
        lambda _binding, reuse_only=False: primary,
    )
    monkeypatch.setattr(
        gate_module.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "status": "gate_passed", "result": reuse,
            }),
        ),
    )
    workspace, protected = _roots(tmp_path)
    binding = {
        "workspace": workspace,
        "protected_workspace": protected,
        "scenario_fingerprint": frozen_scenario().fingerprint,
        "commit": "1" * 40,
        "tree": "2" * 40,
    }
    with pytest.raises(AggregateGateError):
        execute_gate(binding)


def test_cleanup_failure_cannot_mutate_preserved_gate_outcome(
    tmp_path,
):
    workspace, protected = _roots(tmp_path)
    outcome = {
        "status": "gate_passed",
        "reason_code": "rolling_aggregate_gate_passed",
        "cleanup": "preserved",
    }
    frozen = dict(outcome)
    with pytest.raises(AggregateGateError):
        cleanup_gate_workspace(
            workspace, protected, "0" * 64,
            CLEANUP_AUTHORIZATION,
        )
    assert outcome == frozen
