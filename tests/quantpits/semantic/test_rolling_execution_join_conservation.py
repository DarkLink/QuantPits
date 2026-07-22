from quantpits.rolling import (
    RollingExecutionKernel,
    RollingStateRepository,
    build_rolling_execution_scope,
    map_workflow_capability,
    preflight_rolling_execution,
)
from quantpits.utils.workspace import WorkspaceContext

from tests.quantpits.rolling.execution_support import (
    FakeExecutionBackend,
    FakeRunner,
    linear_capability_matrix,
    linear_capability_result,
    make_scope,
)


def _context(tmp_path):
    root = (tmp_path / "workspace").resolve()
    for name in ("config", "data", "mlruns", "output"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return WorkspaceContext.from_root(root)


def test_zero_one_many_units_preserve_identity_order_and_counts(tmp_path):
    context = _context(tmp_path)
    one = make_scope(context, linear_capability_result(), 1, 1)
    many = make_scope(context, linear_capability_result(), 2, 2)
    zero = build_rolling_execution_scope(
        one.prepared, one.resolved, (), one.targets, (),
    )
    assert zero.requested_unit_keys == ()
    assert preflight_rolling_execution(zero).preflight_allowed is False
    assert one.requested_unit_keys == tuple(item.unit_key for item in one.units)
    assert many.requested_unit_keys == tuple(item.unit_key for item in many.units)
    assert many.requested_unit_keys == tuple(
        (target.target_key, window.window_key)
        for target in many.targets for window in many.windows
    )
    assert len(one.units) == 1
    assert len(many.units) == 4


def test_missing_middle_capability_blocks_without_dropping_later_units(tmp_path):
    context = _context(tmp_path)
    available = make_scope(context, linear_capability_result(), 3, 1)
    middle = map_workflow_capability(
        context, available.targets[1].target_key,
        available.targets[1].workflow_relative_path,
        linear_capability_matrix(), action="predict_only",
    )
    targets = (available.targets[0], middle, available.targets[2])
    scope = build_rolling_execution_scope(
        available.prepared, available.resolved,
        tuple(item.window_key for item in available.windows),
        targets, available.windows,
    )
    preflight = preflight_rolling_execution(scope)
    assert preflight.requested_unit_keys == scope.requested_unit_keys
    assert tuple(item[0] for item in preflight.decisions) == scope.requested_unit_keys
    assert tuple(item[1] for item in preflight.decisions) == (True, False, True)
    assert preflight.preflight_allowed is False
    runner = FakeRunner(context)
    batch = RollingExecutionKernel(
        RollingStateRepository.for_workspace(context, "rolling"),
        FakeExecutionBackend(context), runner,
    ).execute(scope, "attempt-1")
    assert batch.requested_unit_keys == scope.requested_unit_keys
    assert tuple(item.status for item in batch.unit_results) == (
        "blocked", "blocked", "blocked",
    )
    assert batch.n_requested == 3
    assert batch.n_runner_calls == 0
    assert runner.calls == []
    resumed = RollingExecutionKernel(
        RollingStateRepository.for_workspace(context, "rolling"),
        FakeExecutionBackend(context), runner,
    ).resume(scope, "attempt-2")
    assert resumed.requested_unit_keys == scope.requested_unit_keys
    assert tuple(item.status for item in resumed.unit_results) == (
        "blocked", "blocked", "blocked",
    )
    assert runner.calls == []
