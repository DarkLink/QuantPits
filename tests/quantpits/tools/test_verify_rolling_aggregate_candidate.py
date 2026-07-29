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
    _GateActivityObserver,
    _WorkspaceMutationObserver,
    _assert_workspace_write_allowlist,
    _assert_gate_budgets,
    _assert_snapshot_unchanged,
    _parser,
    preflight_evidence,
    scenario_from_mapping,
    snapshot_tree,
    validate_binding,
)


def _roots(tmp_path):
    disposable = tmp_path / "Demo_Workspace"
    protected = tmp_path / "Protected_Demo_Workspace"
    disposable.mkdir()
    protected.mkdir()
    return disposable, protected


def _reuse_envelope(binding, result):
    return {
        "status": "gate_passed",
        "reason_code": "rolling_aggregate_gate_reuse_passed",
        "protocol": frozen_scenario().protocol,
        "scenario_fingerprint": binding["scenario_fingerprint"],
        "commit": binding["commit"],
        "tree": binding["tree"],
        "result": result,
    }


def _primary_envelope(binding, result):
    envelope = _reuse_envelope(binding, result)
    envelope["reason_code"] = "rolling_aggregate_gate_primary_passed"
    return envelope


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
        if command == [
            "git", "ls-files", "--others", "--exclude-standard",
            "--", "quantpits",
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
    second_protected = tmp_path / "Second_Protected_Demo_Workspace"
    second_protected.mkdir()
    multiple = validate_binding(
        scenario, disposable, (protected, second_protected),
        commit, tree,
    )
    assert multiple["protected_workspaces"] == (
        protected.resolve(), second_protected.resolve(),
    )
    (second_protected / "forbidden-link").symlink_to(protected)
    linked_evidence = preflight_evidence(multiple)
    assert linked_evidence["status"] == "preflight_passed"
    with pytest.raises(AggregateGateError, match="symlink"):
        snapshot_tree(second_protected)
    linked_snapshot = snapshot_tree(
        second_protected, allow_symlinks=True,
    )
    assert linked_snapshot[0][0:2] == ("forbidden-link", "symlink")
    for change in (
        {"family": "cpcv_rolling"},
        {"target_count": 0},
        {"target_count": 2},
        {"window_count": 0},
        {"window_count": 1},
        {"window_count": 3},
        {"source_unit_count": 0},
        {"source_unit_count": 1},
        {"source_unit_count": 3},
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
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda command, **kwargs: (
            "quantpits/foreign.py\n"
            if command == [
                "git", "ls-files", "--others", "--exclude-standard",
                "--", "quantpits",
            ]
            else isolated_git(command, **kwargs)
        ),
    )
    with pytest.raises(AggregateGateError):
        validate_binding(
            scenario, disposable, protected, commit, tree,
            execute=True, authorization=EXECUTE_AUTHORIZATION,
        )
    monkeypatch.setattr(subprocess, "check_output", isolated_git)
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
    protected_inside = disposable / "protected"
    protected_inside.mkdir()
    with pytest.raises(AggregateGateError, match="overlaps"):
        validate_binding(
            scenario, disposable, protected_inside, commit, tree,
        )
    disposable_inside = protected / "disposable"
    disposable_inside.mkdir()
    with pytest.raises(AggregateGateError, match="overlaps"):
        validate_binding(
            scenario, disposable_inside, protected, commit, tree,
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


def test_gate_deliberate_contract_negatives_use_real_inspectors(
    tmp_path,
):
    import io
    import pandas as pd

    from quantpits.rolling import (
        RollingAggregateContractError,
        inspect_rolling_aggregate_sources,
        materialize_rolling_aggregate_candidates,
    )
    from quantpits.utils.workspace import fingerprint_value
    from tests.quantpits.rolling.aggregate_support import (
        FakeCandidateBackend,
        aggregate_case,
    )

    def case(name):
        return aggregate_case(tmp_path / name)

    context, _scope, _repository, source, aggregate = case("identity")
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    for changed in (
        tuple(reversed(requests)),
        (requests[0], requests[0]),
    ):
        with pytest.raises(RollingAggregateContractError):
            inspect_rolling_aggregate_sources(
                context, aggregate, changed, source,
            )
    _foreign_context, _foreign_scope, _foreign_repository, foreign_source, foreign_aggregate = (
        case("foreign-window")
    )
    foreign_requests = foreign_source.requests_for_state(
        foreign_aggregate.execution_scope,
        foreign_aggregate.state_repository_view.inspection.snapshot,
    )
    foreign = inspect_rolling_aggregate_sources(
        context, aggregate,
        (foreign_requests[0],) + requests[1:],
        source,
    )
    assert foreign.status == "observation_drifted"
    assert "aggregate_source" not in foreign.unit_results[0].capabilities

    context, _scope, _repository, source, aggregate = case("partial-source")
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    source_root = Path(
        source.candidates[requests[0].unit_key]["artifact_root_uri"][7:]
    )
    (source_root / "pred.pkl").unlink()
    partial = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert partial.status == "incomplete"

    context, _scope, _repository, source, aggregate = case("duplicate-source")
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    original_inventory = source.inventory

    def duplicate_inventory(observed_requests):
        inventory = original_inventory(observed_requests)
        rows = inventory["candidates"] + (inventory["candidates"][0],)
        return {
            "fingerprint": fingerprint_value(rows),
            "candidates": rows,
        }

    source.inventory = duplicate_inventory
    duplicate = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert duplicate.status == "incomplete"

    for name, mutation in (
        ("overlap-source", "overlap"),
        ("nonfinite-source", "nonfinite"),
    ):
        context, _scope, _repository, source, aggregate = case(name)
        requests = source.requests_for_state(
            aggregate.execution_scope,
            aggregate.state_repository_view.inspection.snapshot,
        )
        selected = requests[1] if mutation == "overlap" else requests[0]
        frame = pd.read_pickle(io.BytesIO(source.prediction_bytes(selected)))
        if mutation == "overlap":
            first = pd.read_pickle(
                io.BytesIO(source.prediction_bytes(requests[0]))
            )
            frame.index = pd.MultiIndex.from_tuples(
                [first.index[0], frame.index[1]],
                names=frame.index.names,
            )
        else:
            frame.iloc[0, 0] = float("nan")
        output = io.BytesIO()
        frame.to_pickle(output)
        original_bytes = source.prediction_bytes
        source.prediction_bytes = lambda request, selected=selected, data=output.getvalue(), original=original_bytes: (
            data if request.unit_key == selected.unit_key
            else original(request)
        )
        observed = inspect_rolling_aggregate_sources(
            context, aggregate, requests, source,
        )
        assert observed.status == "incomplete"
        selected_position = requests.index(selected)
        assert "aggregate_source" not in (
            observed.unit_results[selected_position].capabilities
        )

    context, _scope, repository, source, aggregate = case("stale-state")
    backend = FakeCandidateBackend(context)
    repository.state_path.write_text("{}", encoding="utf-8")
    stale = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert stale.status == "blocked"
    assert "publication_input" not in stale.capabilities

    context, _scope, repository, source, aggregate = case("partial-candidate")
    backend = FakeCandidateBackend(context)
    backend.candidates[aggregate.candidate_keys[0]] = {
        "classification": "partial",
    }
    partial_candidate = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert partial_candidate.status == "blocked"
    assert "publication_input" not in partial_candidate.capabilities

    context, _scope, repository, source, aggregate = case("duplicate-candidate")
    backend = FakeCandidateBackend(context)

    def inject_duplicate(_scope, callback, create_if_missing=False):
        candidate = backend.candidates[aggregate.candidate_keys[0]]
        backend.candidates["duplicate"] = dict(candidate)
        return callback()

    backend.with_candidate_lock = inject_duplicate
    terminal_duplicate = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert terminal_duplicate.status == "indeterminate"
    assert "publication_input" not in terminal_duplicate.capabilities

    context, _scope, repository, source, aggregate = case("backend-source")
    foreign_context, *_unused = case("backend-foreign")
    backend = FakeCandidateBackend(foreign_context)
    drifted_backend = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert drifted_backend.status == "blocked"
    assert backend.calls == []
    assert "publication_input" not in drifted_backend.capabilities

    context, _scope, _repository, source, aggregate = case(
        "source-namespace-drift",
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    selected = requests[0]
    source_root = Path(
        source.candidates[selected.unit_key]["artifact_root_uri"][7:]
    )
    displaced_source = source_root.with_name(
        source_root.name + "-displaced",
    )
    original_prediction = source.prediction_bytes

    def bytes_then_replace(request):
        data = original_prediction(request)
        if request.unit_key == selected.unit_key and source_root.exists():
            source_root.rename(displaced_source)
            __import__("shutil").copytree(displaced_source, source_root)
        return data

    source.prediction_bytes = bytes_then_replace
    drifted_source = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert drifted_source.status == "observation_drifted"
    assert "aggregate_source" not in (
        drifted_source.unit_results[0].capabilities
    )

    context, _scope, repository, source, aggregate = case(
        "candidate-namespace-drift",
    )
    backend = FakeCandidateBackend(context)
    original_create = backend.create_candidate

    def create_then_drift(*args, **kwargs):
        observation = original_create(*args, **kwargs)
        backend.candidates[
            aggregate.candidate_keys[0]
        ]["namespace_fingerprint"] = "f" * 64
        return observation

    backend.create_candidate = create_then_drift
    drifted_candidate = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert drifted_candidate.status == "indeterminate"
    assert drifted_candidate.target_results[0].did_write is True
    assert "publication_input" not in drifted_candidate.capabilities


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
            "data/rolling_aggregate_candidates_rolling/"
            + "a" * 32 + "/artifacts/pred.pkl",
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
    lifecycle_root = tmp_path / "lifecycle"
    lifecycle_root.mkdir()
    (lifecycle_root / "output").mkdir()
    original_mkdir = __import__("os").mkdir
    original_rename = __import__("os").rename
    original_replace = __import__("os").replace
    pathlib_accessor = getattr(lifecycle_root, "_accessor", None)
    original_pathlib_mkdir = (
        pathlib_accessor.mkdir if pathlib_accessor is not None else None
    )
    original_pathlib_rename = (
        pathlib_accessor.rename if pathlib_accessor is not None else None
    )
    original_pathlib_replace = (
        pathlib_accessor.replace if pathlib_accessor is not None else None
    )
    observer = _WorkspaceMutationObserver(lifecycle_root).start()
    transient = lifecycle_root / "output" / "write-then-delete.bin"
    transient.write_bytes(b"not durable")
    transient.unlink()
    lifecycle_paths = observer.stop()
    assert __import__("os").mkdir is original_mkdir
    assert __import__("os").rename is original_rename
    assert __import__("os").replace is original_replace
    if pathlib_accessor is not None:
        assert pathlib_accessor.mkdir is original_pathlib_mkdir
        assert pathlib_accessor.rename is original_pathlib_rename
        assert pathlib_accessor.replace is original_pathlib_replace
    assert "output/write-then-delete.bin" in lifecycle_paths
    with pytest.raises(AggregateGateError):
        _assert_workspace_write_allowlist(
            (("output", "directory", None, None),),
            (("output", "directory", None, None),),
            lifecycle_paths,
            4,
        )
    staging_paths = (
        "data/.quantpits-aggregate-abc_123",
        "data/.quantpits-aggregate-abc_123/pred.pkl",
        "data/.quantpits-aggregate-abc_123/aggregate_manifest.json",
    )
    assert _assert_workspace_write_allowlist(
        (), (), staging_paths, 8,
    ) == (3, 8)
    with pytest.raises(AggregateGateError):
        _assert_workspace_write_allowlist(
            (), (),
            staging_paths + (
                "data/.quantpits-aggregate-abc_123/foreign.bin",
            ),
            8,
        )
    nested_root = tmp_path / "nested-lifecycle"
    nested_root.mkdir()
    nested_observer = _WorkspaceMutationObserver(nested_root).start()
    for position in range(100):
        created = (
            nested_root / "mlruns" / ("experiment-%03d" % position)
            / "recorder"
        )
        created.mkdir(parents=True)
        nested_transient = created / "write-then-delete.bin"
        nested_transient.write_bytes(b"not durable")
        nested_transient.unlink()
    nested_paths = nested_observer.stop()
    for position in range(100):
        prefix = "mlruns/experiment-%03d/recorder" % position
        assert prefix in nested_paths
        assert prefix + "/write-then-delete.bin" in nested_paths
    rename_root = tmp_path / "rename-lifecycle"
    rename_root.mkdir()
    prepared = tmp_path / "prepared-directory"
    prepared.mkdir()
    (prepared / "write-then-delete.bin").write_bytes(b"not durable")
    rename_observer = _WorkspaceMutationObserver(rename_root).start()
    moved = rename_root / "moved"
    prepared.rename(moved)
    (moved / "write-then-delete.bin").unlink()
    moved.rmdir()
    rename_paths = rename_observer.stop()
    assert "moved" in rename_paths
    assert "moved/write-then-delete.bin" in rename_paths
    for forbidden in (
        "mlruns/arbitrary/forbidden.bin",
        "mlruns/123/" + "f" * 32 + "/tags/mlflow.user",
        "qlib_data/arbitrary/forbidden.bin",
        "data/rolling_aggregate_candidates_rolling/arbitrary/forbidden.bin",
        "data/rolling_aggregate_candidates_rolling/"
        + "f" * 32 + "/artifacts/pred.pkl",
        "data/aggregate_gate_runtime/arbitrary/forbidden.bin",
    ):
        with pytest.raises(AggregateGateError):
            _assert_workspace_write_allowlist(
                (), (), (forbidden,), 1,
            )
    excluded_root = tmp_path / "excluded-lifecycle"
    excluded = excluded_root / "protected-subtree"
    excluded.mkdir(parents=True)
    excluded_observer = _WorkspaceMutationObserver(
        excluded_root, ("protected-subtree",),
    ).start()
    (excluded / "must-not-be-inspected").write_bytes(b"private")
    assert excluded_observer.stop() == ()
    with pytest.raises(AggregateGateError):
        _assert_gate_budgets(301, 0)
    with pytest.raises(AggregateGateError):
        _assert_gate_budgets(float("nan"), 0)
    with pytest.raises(AggregateGateError):
        _assert_gate_budgets(1, True)
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


def test_gate_accepts_repeated_protected_workspace_bindings():
    args = _parser().parse_args([
        "--workspace", "/tmp/disposable-demo",
        "--protected-workspace", "/tmp/protected-production",
        "--protected-workspace", "/tmp/protected-experiment",
        "--commit", "a" * 40,
        "--tree", "b" * 40,
    ])
    assert args.protected_workspace == [
        "/tmp/protected-production",
        "/tmp/protected-experiment",
    ]


def test_snapshot_rejects_hardlinked_file_identity(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"must-not-change")
    __import__("os").link(sentinel, root / "alias.bin")

    with pytest.raises(AggregateGateError, match="canonical regular"):
        snapshot_tree(root)

    assert sentinel.read_bytes() == b"must-not-change"


def test_cleanup_rejects_ancestor_of_protected_workspace_without_deleting(
    tmp_path,
):
    broad = tmp_path / "broad"
    protected = broad / "protected"
    marker = broad / "data" / "aggregate_gate_scenario.json"
    protected.mkdir(parents=True)
    marker.parent.mkdir()
    marker.write_text(__import__("json").dumps({
        "protocol": frozen_scenario().protocol,
        "scenario_fingerprint": frozen_scenario().fingerprint,
    }, sort_keys=True), encoding="utf-8")

    with pytest.raises(AggregateGateError, match="protected or broad"):
        cleanup_gate_workspace(
            broad, protected, frozen_scenario().fingerprint,
            CLEANUP_AUTHORIZATION,
        )

    assert broad.is_dir()
    assert protected.is_dir()
    assert marker.is_file()


def test_cleanup_rejects_aliased_tree_node_without_following_it(
    tmp_path,
):
    disposable, protected = _roots(tmp_path)
    marker = disposable / "data" / "aggregate_gate_scenario.json"
    marker.parent.mkdir()
    marker.write_text(__import__("json").dumps({
        "protocol": frozen_scenario().protocol,
        "scenario_fingerprint": frozen_scenario().fingerprint,
    }, sort_keys=True), encoding="utf-8")
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"must-not-change")
    __import__("os").link(
        sentinel, disposable / "data" / "aliased.bin",
    )

    with pytest.raises(AggregateGateError, match="aliased or special"):
        cleanup_gate_workspace(
            disposable, protected, frozen_scenario().fingerprint,
            CLEANUP_AUTHORIZATION,
        )

    assert sentinel.read_bytes() == b"must-not-change"
    assert marker.exists()
    assert disposable.exists()


def test_cleanup_rejects_marker_public_name_replacement(
    tmp_path, monkeypatch,
):
    disposable, protected = _roots(tmp_path)
    marker = disposable / "data" / "aggregate_gate_scenario.json"
    marker.parent.mkdir()
    payload = __import__("json").dumps({
        "protocol": frozen_scenario().protocol,
        "scenario_fingerprint": frozen_scenario().fingerprint,
    }, sort_keys=True).encode("utf-8")
    marker.write_bytes(payload)
    displaced = disposable / "data" / "displaced-marker.json"
    import quantpits.tools.verify_rolling_aggregate_candidate as gate_module
    original_read = gate_module.os.read
    replaced = [False]

    def replace_after_first_read(descriptor, size):
        chunk = original_read(descriptor, size)
        if chunk and not replaced[0]:
            replaced[0] = True
            marker.rename(displaced)
            marker.write_bytes(payload)
        return chunk

    monkeypatch.setattr(gate_module.os, "read", replace_after_first_read)
    with pytest.raises(AggregateGateError, match="drifted"):
        cleanup_gate_workspace(
            disposable, protected, frozen_scenario().fingerprint,
            CLEANUP_AUTHORIZATION,
        )

    assert marker.read_bytes() == payload
    assert displaced.read_bytes() == payload
    assert disposable.exists()


def test_gate_activity_observer_counts_and_denies_forbidden_actions():
    network = _GateActivityObserver()
    with network:
        network.observe_runner()
        with pytest.raises(AggregateGateError):
            __import__("socket").socket.connect(
                None, ("127.0.0.1", 1),
            )
        with pytest.raises(AggregateGateError):
            __import__("socket").socket.sendto(
                None, b"x", ("127.0.0.1", 1),
            )
    training = _GateActivityObserver()
    with training:
        training_namespace = {"__name__": "qlib.contrib.model.injected"}
        exec("def fit(): return None", training_namespace)
        with pytest.raises(AggregateGateError):
            training_namespace["fit"]()
    gpu = _GateActivityObserver()
    with gpu:
        gpu_namespace = {"__name__": "cupy.cuda.injected"}
        with pytest.raises(AggregateGateError):
            exec("def probe(): return None", gpu_namespace)
    assert network.runner_calls == 1
    assert network.network_calls == 2
    assert training.training_calls == 1
    assert gpu.gpu_calls == 1


def test_gate_enforces_one_total_wall_clock_budget(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import quantpits.tools.verify_rolling_aggregate_candidate as gate_module

    primary = {
        "status": "materialized_success",
        "candidate_fingerprint": "a" * 64,
        "candidate_row_count": 4,
        "new_candidate_recorders": 1,
        "training_calls": 0,
        "gpu_calls": 0,
        "network_calls": 0,
        "runner_calls": 2,
        "protected_unchanged": True,
        "repository_unchanged": True,
        "elapsed_seconds": 50.0,
        "workspace_bytes": 100,
        "write_bytes": 4,
        "changed_path_count": 2,
    }
    reuse = {
        "status": "reused_success",
        "candidate_fingerprint": "a" * 64,
        "candidate_row_count": 4,
        "new_candidate_recorders": 0,
        "training_calls": 0,
        "gpu_calls": 0,
        "network_calls": 0,
        "runner_calls": 0,
        "protected_unchanged": True,
        "repository_unchanged": True,
        "elapsed_seconds": 20.0,
        "workspace_bytes": 100,
        "write_bytes": 0,
        "changed_path_count": 0,
    }
    clock = iter((0.0, 100.0, 299.0))
    observed = {}
    monkeypatch.setattr(gate_module.time, "monotonic", lambda: next(clock))

    workspace, protected = _roots(tmp_path)
    binding = {
        "workspace": workspace,
        "protected_workspace": protected,
        "scenario_fingerprint": frozen_scenario().fingerprint,
        "commit": "1" * 40,
        "tree": "2" * 40,
    }

    def child(*_args, **kwargs):
        command = _args[0]
        if command[-1] == "--internal-primary":
            observed["primary_timeout"] = kwargs["timeout"]
            envelope = _primary_envelope(binding, primary)
        else:
            observed["reuse_timeout"] = kwargs["timeout"]
            envelope = _reuse_envelope(binding, reuse)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(envelope),
        )

    monkeypatch.setattr(gate_module.subprocess, "run", child)
    result = execute_gate(binding)
    assert observed["primary_timeout"] == 300
    assert observed["reuse_timeout"] == 200.0
    assert result["total_elapsed_seconds"] == 299.0

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(_args[0], 300)

    monkeypatch.setattr(gate_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(gate_module.subprocess, "run", timeout)
    with pytest.raises(
        AggregateGateError, match="primary process exceeded",
    ):
        execute_gate(binding)


@pytest.mark.parametrize(
    "field,value",
    [
        ("new_candidate_recorders", 1),
        ("runner_calls", 1),
        ("training_calls", 1),
        ("gpu_calls", 1),
        ("network_calls", 1),
        ("write_bytes", 1),
        ("changed_path_count", 1),
        ("candidate_fingerprint", "f" * 64),
        ("candidate_row_count", 3),
        ("elapsed_seconds", float("nan")),
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
        "gpu_calls": 0,
        "network_calls": 0,
        "runner_calls": 2,
        "protected_unchanged": True,
        "repository_unchanged": True,
        "elapsed_seconds": 50.0,
        "workspace_bytes": 100,
        "write_bytes": 4,
        "changed_path_count": 2,
    }
    reuse = {
        "status": "reused_success",
        "candidate_fingerprint": "a" * 64,
        "candidate_row_count": 4,
        "new_candidate_recorders": 0,
        "training_calls": 0,
        "gpu_calls": 0,
        "network_calls": 0,
        "runner_calls": 0,
        "protected_unchanged": True,
        "repository_unchanged": True,
        "elapsed_seconds": 20.0,
        "workspace_bytes": 100,
        "write_bytes": 0,
        "changed_path_count": 0,
    }
    reuse[field] = value
    workspace, protected = _roots(tmp_path)
    binding = {
        "workspace": workspace,
        "protected_workspace": protected,
        "scenario_fingerprint": frozen_scenario().fingerprint,
        "commit": "1" * 40,
        "tree": "2" * 40,
    }
    def child(command, **_kwargs):
        envelope = (
            _primary_envelope(binding, primary)
            if command[-1] == "--internal-primary"
            else _reuse_envelope(binding, reuse)
        )
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(envelope),
        )

    monkeypatch.setattr(gate_module.subprocess, "run", child)
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
