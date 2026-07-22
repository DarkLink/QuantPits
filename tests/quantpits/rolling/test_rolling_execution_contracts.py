import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantpits.rolling import (
    RollingExecutionBatchResult,
    RollingExecutionContractError,
    RollingExecutionKernel,
    RollingExecutionScope,
    RollingExecutionUnitResult,
    RollingStateRepository,
    RollingRunIdentity,
    RollingWindowDescriptor,
    RollingWindowExecutionDescriptor,
    RollingWindowIdentity,
    build_rolling_execution_scope,
    map_workflow_capability,
    observe_rolling_business_sessions,
)
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value

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


def test_workflow_mapper_uses_module_class_dataset_action_and_family(tmp_path):
    context = _context(tmp_path)
    workflow = context.config_dir / "linear.yaml"
    workflow.write_text(
        "name: misleading-tag\n"
        "task:\n"
        "  model: {class: LinearModel, module_path: qlib.contrib.model.linear}\n"
        "  dataset: {class: DatasetH, module_path: qlib.data.dataset}\n",
        encoding="utf-8",
    )
    descriptor = map_workflow_capability(
        context, "operator-name-does-not-map@rolling", "config/linear.yaml",
        linear_capability_matrix(),
    )
    assert descriptor.capability_identity.model_module == "qlib.contrib.model.linear"
    assert descriptor.capability_identity.model_class == "LinearModel"
    assert descriptor.capability_identity.dataset_class == "DatasetH"
    assert descriptor.capability_identity.action == "train"
    assert descriptor.capability_identity.execution_family == "rolling"

    workflow.write_text(workflow.read_text().replace("DatasetH", "UnknownDataset"), encoding="utf-8")
    with pytest.raises(RollingExecutionContractError):
        map_workflow_capability(
            context, "operator-name-does-not-map@rolling", "config/linear.yaml",
            linear_capability_matrix(),
        )


def test_business_sessions_are_ordered_exact_and_bound_to_execution_fingerprint(tmp_path):
    context = _context(tmp_path)
    scope = make_scope(context, linear_capability_result())
    sessions_fingerprint = fingerprint_value([
        {"window_key": item.window_key, "expected_sessions": list(item.expected_sessions)}
        for item in scope.windows
    ])
    assert scope.runtime_binding.sessions_fingerprint == sessions_fingerprint
    assert scope.run_identity.runtime_params_fingerprint == scope.runtime_binding.fingerprint
    raw = scope.windows[0].window
    for observed in (
        (),
        ("2026-01-05", "2026-01-05"),
        ("2026-01-06", "2026-01-05"),
        ("2026-01-05",),
    ):
        with pytest.raises(Exception):
            observe_rolling_business_sessions((raw,), lambda _start, _end, values=observed: values)


def test_scope_rebuild_revalidates_members_fingerprints_and_workspace(tmp_path):
    context = _context(tmp_path)
    scope = make_scope(context, linear_capability_result())
    with pytest.raises(RollingExecutionContractError):
        dataclasses.replace(scope)
    with pytest.raises(RollingExecutionContractError):
        RollingExecutionScope(
            scope.run_identity, scope.runtime_binding, scope.targets, scope.windows,
            scope.units, scope.prepared, scope.resolved,
        )
    with pytest.raises(Exception):
        dataclasses.replace(scope.windows[0])
    with pytest.raises(Exception):
        RollingWindowExecutionDescriptor(
            scope.windows[0].window, scope.windows[0].expected_sessions,
        )
    workflow = context.root / scope.targets[0].workflow_relative_path
    workflow.write_text(workflow.read_text() + "changed: true\n", encoding="utf-8")
    runner = FakeRunner(context)
    blocked = RollingExecutionKernel(
        RollingStateRepository.for_workspace(context, "rolling"),
        FakeExecutionBackend(context), runner,
    ).execute(scope, "attempt-1")
    assert blocked.status == "blocked"
    assert runner.calls == []
    with pytest.raises(RollingExecutionContractError):
        map_workflow_capability(
            context, scope.targets[0].target_key,
            "../outside.yaml", linear_capability_matrix(),
        )


def test_scope_builder_requires_prepared_resolved_and_ordered_window_subset(tmp_path):
    context = _context(tmp_path)
    full = make_scope(context, linear_capability_result(), n_targets=2, n_windows=3)
    selected_raw = (full.resolved.windows[0], full.resolved.windows[2])
    selected = observe_rolling_business_sessions(
        selected_raw, lambda start, end: (start, end),
    )
    keys = tuple(item.window_key for item in selected)
    subset = build_rolling_execution_scope(
        full.prepared, full.resolved, keys, full.targets, selected,
    )
    assert subset.run_identity.target_keys == tuple(
        item.target_key for item in full.prepared.targets
    )
    assert subset.run_identity.window_keys == keys
    for invalid in (
        (keys[0], keys[0]),
        tuple(reversed(keys)),
        ("rolling:2026-01-01:2026-01-02:abcdef123456",),
    ):
        with pytest.raises(RollingExecutionContractError):
            build_rolling_execution_scope(
                full.prepared, full.resolved, invalid, full.targets, selected,
            )
    with pytest.raises(RollingExecutionContractError):
        build_rolling_execution_scope(
            full.prepared, full.resolved, keys,
            full.targets[1:] + full.targets[:1], selected,
        )


def test_kernel_rejects_runner_runtime_or_provider_backend_drift(tmp_path):
    context = _context(tmp_path)
    scope = make_scope(context, linear_capability_result())
    repository = RollingStateRepository.for_workspace(context, "rolling")
    backend = FakeExecutionBackend(context)
    runner = FakeRunner(context, runtime_params={"market": "other", "benchmark": "other"})
    result = RollingExecutionKernel(repository, backend, runner).execute(
        scope, "attempt-1",
    )
    assert result.status == "blocked"
    assert runner.calls == []

    foreign_context = WorkspaceContext.from_root(
        context.root, mlflow_uri="sqlite:///%s" % (context.root / "other.db"),
        qlib_data_dir=context.qlib_data_dir, qlib_region=context.qlib_region,
    )
    with pytest.raises(RollingExecutionContractError):
        RollingExecutionKernel(
            repository, FakeExecutionBackend(foreign_context), FakeRunner(context),
        )


def test_execution_contract_import_is_workspace_backend_and_optional_dependency_independent(tmp_path):
    script = (
        "import builtins, os\n"
        "original=builtins.__import__\n"
        "def guarded(name,*args,**kwargs):\n"
        "  if name == 'qlib' or name.startswith('qlib.') or name == 'mlflow' or name.startswith('mlflow.'):\n"
        "    raise AssertionError(name)\n"
        "  return original(name,*args,**kwargs)\n"
        "builtins.__import__=guarded\n"
        "before=os.getcwd()\n"
        "import quantpits.rolling.execution\n"
        "assert os.getcwd() == before\n"
    )
    env = dict(os.environ)
    env.pop("QLIB_WORKSPACE_DIR", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path,
        env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_unit_batch_and_state_impossible_cross_field_combinations_are_rejected():
    key = ("linear@rolling", "rolling:2026-01-05:2026-01-06:abcdef123456")
    with pytest.raises(RollingExecutionContractError):
        RollingExecutionUnitResult(
            key, 0, "executed_success", False,
            "attempt", "record", "a" * 64, "reason",
        )
    with pytest.raises(RollingExecutionContractError):
        RollingExecutionUnitResult(
            key, 0, "reused_success", False,
            "attempt", "record", "a" * 64, "reason",
        )
    failed = RollingExecutionUnitResult(key, 0, "failed", True, reason_code="failed")
    with pytest.raises(RollingExecutionContractError):
        RollingExecutionBatchResult((key,), (failed,), "success", "rolling_execution_batch_success")


def test_phase34_modules_parse_as_python38():
    root = Path(__file__).resolve().parents[3]
    modules = (
        "quantpits/rolling/execution.py",
        "quantpits/rolling/execution_backend.py",
        "quantpits/rolling/unit_runner.py",
        "quantpits/rolling/mlflow_execution_backend.py",
    )
    for relative in modules:
        ast.parse((root / relative).read_text(encoding="utf-8"), feature_version=(3, 8))


def test_phase34_docs_public_signatures_and_capabilities_match():
    root = Path(__file__).resolve().parents[3]
    for relative in (
        "docs/30_ROLLING_TRAINING_GUIDE.md", "docs/en/30_ROLLING_TRAINING_GUIDE.md",
        "docs/33_OPERATIONS.md", "docs/en/33_OPERATIONS.md",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "build_rolling_execution_scope" in text
        assert "resume_rolling_units" in text
        assert "units_complete" in text
        assert "current record" in text.lower() or "current-record" in text.lower()
