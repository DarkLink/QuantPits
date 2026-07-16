"""Contracts for the Phase 28B Rolling prepared command plan."""

import json
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
    RollingStatePreconditionError,
    RollingStateCorruptError,
    RollingStateRejectedError,
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


def test_prepared_targets_use_canonical_identity_and_preserve_registry_order(tmp_path):
    _root, ctx = _workspace(tmp_path)
    prepared = prepare_rolling_run(
        ctx, RollingRunOptions(action="cold_start", models=("first", "second")),
        ("--cold-start", "--models", "first,second"),
    )
    assert [target.model_name for target in prepared.targets] == ["second", "first"]
    assert prepared.plan.metadata["target_keys"] == [
        "second@rolling", "first@rolling",
    ]
    assert [target.identity.target_key for target in prepared.targets] == [
        "second@rolling", "first@rolling",
    ]
    assert all(target.workflow_fingerprint for target in prepared.targets)
    assert prepared.plan.metadata["window_resolution"] == "deferred_to_runtime"
    assert prepared.plan.metadata["zero_write_plan_route"] is True
    assert "data/history/train_records_<timestamp>.json" in {
        output.path for output in prepared.plan.outputs
    }


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
    with_run_id = prepare_rolling_run(
        ctx,
        RollingRunOptions(action="cold_start", all_enabled=True, run_id="attempt-label"),
        ("--cold-start", "--all-enabled", "--run-id", "attempt-label"),
    )
    payload = prepared_plan_json(first)
    assert payload["plan"] == first.plan.to_public_dict()
    assert payload["plan_fingerprint"] == first.plan_fingerprint
    assert first.plan_fingerprint == second.plan_fingerprint
    assert first.plan_fingerprint == with_run_id.plan_fingerprint
    assert len(first.plan_fingerprint) == 64


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


@pytest.mark.parametrize("action", ["daily", "predict_only", "retrain_last"])
def test_state_dependent_actions_fail_before_backend_when_state_is_missing(
    tmp_path, action,
):
    _root, ctx = _workspace(tmp_path)
    with pytest.raises(RollingStatePreconditionError) as caught:
        prepare_rolling_run(
            ctx, RollingRunOptions(action=action, all_enabled=True),
        )
    assert caught.value.code == "rolling_state_precondition_failed"


def test_phase30_core_uses_python38_compatible_syntax():
    package = Path(__file__).resolve().parents[3] / "quantpits" / "rolling"
    script = package.parent / "scripts" / "rolling_train.py"
    for path in sorted(package.glob("*.py")) + [script]:
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=8,
        )


@pytest.mark.parametrize("case,classification,expected_code", [
    ("zero_byte", "corrupt", "rolling_state_corrupt"),
    ("empty_mapping", "unsupported_schema", "rolling_state_schema_unsupported"),
    ("ambiguous", "ambiguous", "rolling_state_ambiguous"),
    ("foreign", "foreign", "rolling_state_foreign"),
    ("identity_mismatch", "identity_mismatch", "rolling_state_identity_mismatch"),
    ("unverified_completion", "unverified_completion", "rolling_state_completion_unverified"),
])
def test_invalid_state_classes_fail_before_safeguard_lease_env_or_backend(
    tmp_path, case, classification, expected_code, capsys,
):
    from quantpits.scripts import rolling_train
    from quantpits.rolling.identity import workspace_fingerprint

    root, ctx = _workspace(tmp_path)
    state_path = root / "data" / "rolling_state.json"
    versioned = {
        "schema_version": 2,
        "workspace_fingerprint": workspace_fingerprint(root),
        "run_id": "demo-run", "family": "rolling", "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": "d" * 64,
        "anchor_date": "2026-07-10",
        "target_keys": ["first@rolling"],
        "window_keys": ["rolling:2026-04-01:2026-06-30:abcdef123456"],
        "attempt_id": None, "phase": "executing", "units": [],
    }
    if case == "zero_byte":
        state_path.write_bytes(b"")
    elif case == "empty_mapping":
        state_path.write_text("{}", encoding="utf-8")
    elif case == "ambiguous":
        state_path.write_text('{"completed_windows":{"01":{}}}', encoding="utf-8")
    elif case == "foreign":
        state_path.write_text(json.dumps({
            "training_method": "cpcv", "completed_windows": {},
        }), encoding="utf-8")
    else:
        if case == "unverified_completion":
            from quantpits.utils.workspace import fingerprint_value
            versioned["config_fingerprint"] = fingerprint_value({
                "rolling_start": "2020-01-01", "train_years": 3,
                "valid_years": 1, "test_step": "3M",
                "test_step_months": 3, "training_method": "slide",
            })
            versioned["phase"] = "completed"
        state_path.write_text(json.dumps(versioned), encoding="utf-8")
    with mock.patch.object(rolling_train, "_safeguard_explicit_workspace") as safeguard, \
         mock.patch.object(rolling_train, "_activate_legacy_workspace") as activate, \
         mock.patch(
             "quantpits.training.lease.TrainingExecutionLease.for_workspace",
         ) as lease_factory:
        exit_code = rolling_train.main([
            "--workspace", str(root), "--merge", "--all-enabled",
        ])
    assert exit_code == 2
    safeguard.assert_not_called()
    lease_factory.assert_not_called()
    activate.assert_not_called()
    # Re-read through the same Prepared expectation used by the public CLI.
    # A bare classifier call intentionally cannot infer config mismatch.
    shown = prepare_rolling_run(ctx, RollingRunOptions(action="show_state"))
    assert shown.state.classification == classification
    assert shown.state.reason_code == expected_code
    stderr = capsys.readouterr().err
    assert expected_code in stderr


@pytest.mark.parametrize("case,classification", [
    ("missing", "missing"),
    ("zero_byte", "corrupt"),
    ("empty_mapping", "unsupported_schema"),
    ("ambiguous", "ambiguous"),
    ("valid_legacy", "valid_legacy"),
    ("foreign", "foreign"),
    ("valid_versioned", "valid_versioned"),
    ("identity_mismatch", "identity_mismatch"),
    ("unverified_completion", "unverified_completion"),
])
def test_show_state_renders_every_classification_without_mutation(
    tmp_path, case, classification, capsys,
):
    from quantpits.scripts.rolling_train import _render_strict_state
    from quantpits.rolling.identity import workspace_fingerprint
    from quantpits.utils.workspace import fingerprint_value

    root, ctx = _workspace(tmp_path)
    state_path = root / "data" / "rolling_state.json"
    effective_fp = fingerprint_value({
        "rolling_start": "2020-01-01", "train_years": 3,
        "valid_years": 1, "test_step": "3M", "test_step_months": 3,
        "training_method": "slide",
    })
    legacy = {
        "anchor_date": "2026-07-10", "training_method": "slide",
        "completed_windows": {}, "total_windows": 0,
    }
    versioned = {
        "schema_version": 2,
        "workspace_fingerprint": workspace_fingerprint(root),
        "run_id": "demo-run", "family": "rolling", "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": effective_fp,
        "anchor_date": "2026-07-10",
        "target_keys": ["first@rolling"],
        "window_keys": ["rolling:2026-04-01:2026-06-30:abcdef123456"],
        "attempt_id": None, "phase": "executing", "units": [],
    }
    if case == "zero_byte":
        state_path.write_bytes(b"")
    elif case == "empty_mapping":
        state_path.write_text("{}", encoding="utf-8")
    elif case == "ambiguous":
        state_path.write_text('{"completed_windows":{"01":{}}}', encoding="utf-8")
    elif case == "valid_legacy":
        state_path.write_text(json.dumps(legacy), encoding="utf-8")
    elif case == "foreign":
        state_path.write_text(
            json.dumps(dict(legacy, training_method="cpcv")), encoding="utf-8",
        )
    elif case in ("valid_versioned", "identity_mismatch", "unverified_completion"):
        if case == "identity_mismatch":
            versioned["config_fingerprint"] = "d" * 64
        elif case == "unverified_completion":
            versioned["phase"] = "completed"
        state_path.write_text(json.dumps(versioned), encoding="utf-8")
    before = state_path.read_bytes() if state_path.exists() else None
    prepared = prepare_rolling_run(ctx, RollingRunOptions(action="show_state"))
    assert prepared.state.classification == classification
    public_inspection = prepared.plan.metadata["state_inspection"]
    assert public_inspection == prepared.state.to_public_dict()
    assert prepared_plan_json(prepared)["plan"]["metadata"]["state_inspection"] == public_inspection
    assert public_inspection["classification"] == classification
    assert public_inspection["reason_code"] == prepared.state.reason_code
    assert public_inspection["fingerprint"] == prepared.state.fingerprint
    assert _render_strict_state(prepared) == 0
    rendered = capsys.readouterr().out
    assert "Classification: %s" % classification in rendered
    assert "Reason: %s" % prepared.state.reason_code in rendered
    assert (state_path.read_bytes() if state_path.exists() else None) == before


@pytest.mark.parametrize("action", [
    "cold_start", "merge", "resume", "daily", "retrain_models",
    "retrain_last", "predict_only", "backtest_only", "clear_state",
])
def test_valid_legacy_action_compatibility_matrix(tmp_path, action):
    root, ctx = _workspace(tmp_path)
    (root / "data" / "rolling_state.json").write_text(json.dumps({
        "anchor_date": "2026-07-10",
        "training_method": "slide",
        "completed_windows": {"0": {"first": "recorder-1"}},
        "current_window_idx": 0,
        "current_model": "first",
        "total_windows": 1,
    }), encoding="utf-8")
    prepared = prepare_rolling_run(
        ctx,
        RollingRunOptions(
            action=action,
            all_enabled=action not in ("clear_state", "show_state"),
        ),
    )
    assert prepared.state.classification == "valid_legacy"
    assert prepared.options.action == action


def test_valid_v2_is_display_only_until_repository_integration(tmp_path):
    from quantpits.rolling.identity import workspace_fingerprint
    from quantpits.rolling.errors import RollingStateRejectedError
    from quantpits.utils.workspace import fingerprint_value

    root, ctx = _workspace(tmp_path)
    payload = {
        "schema_version": 2,
        "workspace_fingerprint": workspace_fingerprint(root),
        "run_id": "demo-run",
        "family": "rolling",
        "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": fingerprint_value({
            "rolling_start": "2020-01-01",
            "train_years": 3,
            "valid_years": 1,
            "test_step": "3M",
            "test_step_months": 3,
            "training_method": "slide",
        }),
        "anchor_date": "2026-07-10",
        "target_keys": ["first@rolling"],
        "window_keys": ["rolling:2026-04-01:2026-06-30:abcdef123456"],
        "attempt_id": None,
        "phase": "executing",
        "units": [],
    }
    (root / "data" / "rolling_state.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    shown = prepare_rolling_run(ctx, RollingRunOptions(action="show_state"))
    assert shown.state.classification == "valid_versioned"
    assert shown.state.warnings == (
        "V2 CAS repository is available; legacy execution integration remains blocked",
    )
    with pytest.raises(RollingStateRejectedError) as caught:
        prepare_rolling_run(
            ctx, RollingRunOptions(action="cold_start", all_enabled=True),
        )
    assert caught.value.code == "rolling_state_versioned_read_only"
