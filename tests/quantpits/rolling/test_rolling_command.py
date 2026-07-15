"""Contracts for the Phase 28B Rolling prepared command plan."""

import json
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from quantpits.rolling.command import (
    RollingRunOptions,
    inspect_legacy_state,
    options_from_namespace,
    prepare_rolling_run,
    prepared_plan_json,
    resolve_workspace_context,
)
from quantpits.rolling.errors import (
    RollingActionConflictError,
    RollingResumeStateMissingError,
    RollingStateCorruptError,
    RollingStateUnsupportedError,
    RollingTargetUnknownError,
    RollingWorkflowOutsideWorkspaceError,
)
from quantpits.utils.workspace import WorkspaceContext


def _workspace(tmp_path):
    root = tmp_path / "Demo_Workspace"
    config = root / "config"
    data = root / "data"
    config.mkdir(parents=True)
    data.mkdir()
    (root / "output").mkdir()
    (config / "rolling_config.yaml").write_text(yaml.safe_dump({
        "rolling_start": "2020-01-01",
        "train_years": 3,
        "valid_years": 1,
        "test_step": "3M",
        "training_method": "slide",
    }), encoding="utf-8")
    (config / "model_registry.yaml").write_text(yaml.safe_dump({
        "models": {
            "second": {
                "algorithm": "linear", "dataset": "Alpha158",
                "enabled": True, "yaml_file": "config/second.yaml",
            },
            "first": {
                "algorithm": "linear", "dataset": "Alpha158",
                "enabled": True, "yaml_file": "config/first.yaml",
            },
        }
    }, sort_keys=False), encoding="utf-8")
    (config / "first.yaml").write_text("model: first\n", encoding="utf-8")
    (config / "second.yaml").write_text("model: second\n", encoding="utf-8")
    (config / "model_config.json").write_text("{}\n", encoding="utf-8")
    (config / "strategy_config.yaml").write_text("strategy: TopkDropout\n", encoding="utf-8")
    (config / "prod_config.json").write_text("{}\n", encoding="utf-8")
    return root, WorkspaceContext.from_root(root)


def _namespace(**updates):
    values = {
        "cold_start": False, "merge": False, "retrain_models": None,
        "retrain_last": False, "predict_only": False, "resume": False,
        "backtest_only": False, "clear_state": False, "show_state": False,
        "models": None, "algorithm": None, "dataset": None, "market": None,
        "tag": None, "all_enabled": False, "skip": None,
        "training_method": None, "no_pretrain": False, "cache_size": None,
        "allow_stale_predict": False, "backtest": False, "show_folds": False,
        "explain_plan": False, "dry_run": False, "json_plan": False,
        "run_id": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_action_conflict_is_stable():
    with pytest.raises(RollingActionConflictError) as caught:
        options_from_namespace(_namespace(cold_start=True, resume=True))
    assert caught.value.code == "rolling_action_conflict"
    assert "--cold-start" in str(caught.value)
    assert "--resume" in str(caught.value)


def test_workspace_resolution_prefers_explicit_root(tmp_path, monkeypatch):
    explicit = tmp_path / "Demo_Workspace"
    fallback = tmp_path / "fallback_workspace"
    explicit.mkdir()
    fallback.mkdir()
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(fallback))
    assert resolve_workspace_context().root == fallback.resolve()
    assert resolve_workspace_context(str(explicit)).root == explicit.resolve()


def test_prepared_plan_freezes_registry_order_and_workflows(tmp_path):
    _root, ctx = _workspace(tmp_path)
    prepared = prepare_rolling_run(
        ctx, RollingRunOptions(action="cold_start", models=("first", "second")),
        ("--cold-start", "--models", "first,second"),
    )
    assert [target.model_name for target in prepared.targets] == ["second", "first"]
    assert prepared.plan.metadata["target_keys"] == [
        "second@rolling", "first@rolling",
    ]
    assert all(target.workflow_fingerprint for target in prepared.targets)
    assert prepared.plan.metadata["window_resolution"] == "deferred_to_runtime"
    assert prepared.plan.metadata["zero_write_plan_route"] is True


def test_skip_preserves_registry_order_for_remaining_targets(tmp_path):
    _root, ctx = _workspace(tmp_path)
    prepared = prepare_rolling_run(
        ctx, RollingRunOptions(
            action="cold_start", all_enabled=True, skip=("second",),
        ),
    )
    assert [target.model_name for target in prepared.targets] == ["first"]
    assert prepared.targets[0].selected_by == "all_enabled"


def test_human_and_json_share_plan_and_fingerprint_is_stable(tmp_path):
    _root, ctx = _workspace(tmp_path)
    options = RollingRunOptions(action="cold_start", all_enabled=True)
    first = prepare_rolling_run(
        ctx, options, ("--cold-start", "--all-enabled", "--dry-run"),
    )
    second = prepare_rolling_run(
        ctx, options, ("--cold-start", "--all-enabled", "--json-plan"),
    )
    payload = prepared_plan_json(first)
    assert payload["plan"] == first.plan.to_public_dict()
    assert payload["plan_fingerprint"] == first.plan_fingerprint
    assert first.plan_fingerprint == second.plan_fingerprint


def test_training_method_override_is_in_memory_only(tmp_path):
    root, ctx = _workspace(tmp_path)
    path = root / "config" / "rolling_config.yaml"
    before = path.read_bytes()
    prepared = prepare_rolling_run(
        ctx, RollingRunOptions(
            action="cold_start", all_enabled=True, training_method="cpcv",
        ),
    )
    assert prepared.effective_config["training_method"] == "cpcv"
    assert prepared.plan.metadata["cpcv_fold_resolution"] == "deferred_to_qlib_calendar"
    assert path.read_bytes() == before


def test_unknown_exact_model_fails(tmp_path):
    _root, ctx = _workspace(tmp_path)
    with pytest.raises(RollingTargetUnknownError):
        prepare_rolling_run(
            ctx, RollingRunOptions(action="cold_start", models=("unknown",)),
        )


def test_workflow_must_stay_inside_workspace(tmp_path):
    root, ctx = _workspace(tmp_path)
    outside = tmp_path / "private.yaml"
    outside.write_text("model: private\n", encoding="utf-8")
    registry = yaml.safe_load((root / "config" / "model_registry.yaml").read_text())
    registry["models"]["second"]["yaml_file"] = str(outside)
    (root / "config" / "model_registry.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8",
    )
    with pytest.raises(RollingWorkflowOutsideWorkspaceError):
        prepare_rolling_run(
            ctx, RollingRunOptions(action="cold_start", models=("second",)),
        )


def test_strict_state_classification_and_resume_contract(tmp_path):
    root, ctx = _workspace(tmp_path)
    state_path = root / "data" / "rolling_state.json"
    assert inspect_legacy_state(state_path, root).status == "missing"
    with pytest.raises(RollingResumeStateMissingError):
        prepare_rolling_run(ctx, RollingRunOptions(action="resume", all_enabled=True))

    state_path.write_text("{broken", encoding="utf-8")
    assert inspect_legacy_state(state_path, root).status == "corrupt"
    with pytest.raises(RollingStateCorruptError):
        prepare_rolling_run(ctx, RollingRunOptions(action="merge", all_enabled=True))
    shown = prepare_rolling_run(ctx, RollingRunOptions(action="show_state"))
    assert shown.state.status == "corrupt"

    state_path.write_text("[]", encoding="utf-8")
    assert inspect_legacy_state(state_path, root).status == "unsupported"
    with pytest.raises(RollingStateUnsupportedError):
        prepare_rolling_run(ctx, RollingRunOptions(action="merge", all_enabled=True))
    shown = prepare_rolling_run(ctx, RollingRunOptions(action="show_state"))
    assert shown.state.status == "unsupported"

    state_path.write_text(json.dumps({
        "anchor_date": "2026-07-10", "training_method": "slide",
        "completed_windows": {"0": {"first": "recorder-1"}},
    }), encoding="utf-8")
    resumed = prepare_rolling_run(
        ctx, RollingRunOptions(action="resume", all_enabled=True),
    )
    assert resumed.state.status == "valid_legacy"
    assert resumed.state.completed_units == 1
    assert resumed.anchor_policy.resolution == "legacy_state_hint"


def test_phase28_core_uses_python38_compatible_syntax():
    package = Path(__file__).resolve().parents[3] / "quantpits" / "rolling"
    script = package.parent / "scripts" / "rolling_train.py"
    for path in sorted(package.glob("*.py")) + [script]:
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=8,
        )
