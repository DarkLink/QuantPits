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
    _assert_workspace_write_allowlist,
    _assert_gate_budgets,
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
