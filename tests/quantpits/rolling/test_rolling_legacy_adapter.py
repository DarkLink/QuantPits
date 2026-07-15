"""Contracts for the Phase 28C authoritative legacy adapter."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from quantpits.rolling.errors import RollingExecutionError, RollingInputChangedError
from quantpits.rolling.legacy import (
    LegacyRollingExecutionAdapter,
    recheck_prepared_inputs,
)
from quantpits.runtime.command import InputRef
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file


def test_input_baseline_recheck_detects_creation_and_content_drift(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    existing = tmp_path / "config.yaml"
    existing.write_text("value: one\n", encoding="utf-8")
    prepared = SimpleNamespace(
        ctx=ctx,
        options=SimpleNamespace(action="daily"),
        plan=SimpleNamespace(inputs=(
            InputRef("config.yaml", fingerprint=fingerprint_file(existing)),
            InputRef("optional.json", required=False),
        )),
    )
    recheck_prepared_inputs(prepared)
    existing.write_text("value: two\n", encoding="utf-8")
    with pytest.raises(RollingInputChangedError):
        recheck_prepared_inputs(prepared)
    existing.write_text("value: one\n", encoding="utf-8")
    (tmp_path / "optional.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RollingInputChangedError):
        recheck_prepared_inputs(prepared)


def test_adapter_passes_exact_prepared_targets_and_resolved_windows(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    (tmp_path / "data").mkdir()
    facade = SimpleNamespace(run_daily=mock.Mock())
    target = SimpleNamespace(
        model_name="demo",
        target_key="demo@rolling",
        workflow_path="config/demo.yaml",
        legacy_info={"yaml_file": "ignored.yaml", "enabled": True},
    )
    prepared = SimpleNamespace(
        ctx=ctx,
        state=SimpleNamespace(path="data/rolling_state.json"),
        targets=(target,),
        options=SimpleNamespace(action="daily"),
        effective_config={"training_method": "slide"},
        plan=SimpleNamespace(metadata={"family": "rolling"}),
        plan_fingerprint="prepared-fingerprint",
        cli_args=("--models", "demo"),
    )
    resolved = SimpleNamespace(
        prepared=prepared,
        params={"anchor_date": "2026-07-15", "freq": "week"},
        execution_fingerprint="execution-fingerprint",
    )
    args = SimpleNamespace()
    with mock.patch("quantpits.utils.operator_log.OperatorLog") as log_cls:
        log_cls.return_value.__enter__.return_value = log_cls.return_value
        outcome = LegacyRollingExecutionAdapter(facade).execute(
            args, prepared, resolved, "rolling-test",
        )
    facade.run_daily.assert_called_once()
    call = facade.run_daily.call_args
    assert list(call.args[1]) == ["demo"]
    assert call.args[1]["demo"]["yaml_file"] == str(
        Path(tmp_path, "config", "demo.yaml")
    )
    assert call.kwargs["resolved"] is resolved
    assert outcome.execution_fingerprint == "execution-fingerprint"


def test_retrain_edit_occurs_inside_adapter_before_frozen_scope_execution(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    (tmp_path / "data").mkdir()
    events = []
    facade = SimpleNamespace(
        run_cold_start=mock.Mock(side_effect=lambda *_args, **_kwargs: events.append("run")),
    )
    target = SimpleNamespace(
        model_name="demo", target_key="demo@rolling",
        workflow_path="config/demo.yaml",
        legacy_info={"yaml_file": "ignored.yaml", "enabled": True},
    )
    prepared = SimpleNamespace(
        ctx=ctx,
        state=SimpleNamespace(path="data/rolling_state.json"),
        targets=(target,),
        options=SimpleNamespace(action="retrain_models"),
        effective_config={"training_method": "slide"},
        plan=SimpleNamespace(metadata={"family": "rolling"}),
        plan_fingerprint="prepared-fingerprint",
        cli_args=("--retrain-models", "demo"),
    )
    resolved = SimpleNamespace(
        prepared=prepared,
        params={"anchor_date": "2026-07-15", "freq": "week"},
        execution_fingerprint="execution-fingerprint",
    )
    adapter = LegacyRollingExecutionAdapter(facade)
    adapter._edit_retrain_models = mock.Mock(
        side_effect=lambda *_args: events.append("edit"),
    )
    args = SimpleNamespace(merge=False, resume=True)
    with mock.patch("quantpits.utils.operator_log.OperatorLog") as log_cls, \
         mock.patch("quantpits.scripts.rolling.state.RollingState"), \
         mock.patch(
             "quantpits.scripts.deep_analysis.promote_config.update_promote_status",
         ):
        log_cls.return_value.__enter__.return_value = log_cls.return_value
        adapter.execute(args, prepared, resolved, "rolling-test")
    assert events == ["edit", "run"]
    effective_args = facade.run_cold_start.call_args.args[0]
    assert effective_args.merge is True
    assert effective_args.resume is False
    assert args.merge is False
    assert args.resume is True


def test_clear_missing_state_is_truthfully_skipped(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    (tmp_path / "data").mkdir()
    prepared = SimpleNamespace(
        ctx=ctx,
        state=SimpleNamespace(path="data/rolling_state.json", status="missing"),
        targets=(),
        options=SimpleNamespace(action="clear_state"),
        effective_config={"training_method": "slide"},
        plan=SimpleNamespace(metadata={"family": "rolling"}),
        plan_fingerprint="prepared-fingerprint",
        cli_args=("--clear-state",),
    )
    with mock.patch("quantpits.utils.operator_log.OperatorLog") as log_cls, \
         mock.patch("quantpits.scripts.rolling.state.RollingState") as state_cls:
        log_cls.return_value.__enter__.return_value = log_cls.return_value
        outcome = LegacyRollingExecutionAdapter(SimpleNamespace()).execute(
            SimpleNamespace(), prepared, None, "rolling-test",
        )
    assert outcome.status == "skipped"
    assert outcome.reason_code == "rolling_action_skipped"
    assert outcome.did_execute is False
    state_cls.assert_not_called()
    result = log_cls.return_value.set_result.call_args.args[0]
    assert result["status"] == "skipped"
    assert result["did_execute"] is False


def test_adapter_failure_is_logged_and_raised_with_stable_code(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    (tmp_path / "data").mkdir()
    target = SimpleNamespace(
        model_name="demo", target_key="demo@rolling",
        workflow_path="config/demo.yaml", legacy_info={},
    )
    prepared = SimpleNamespace(
        ctx=ctx,
        state=SimpleNamespace(
            path="data/rolling_state.json", status="valid_legacy",
        ),
        targets=(target,), options=SimpleNamespace(action="daily"),
        effective_config={"training_method": "slide"},
        plan=SimpleNamespace(metadata={"family": "rolling"}),
        plan_fingerprint="prepared-fingerprint", cli_args=(),
    )
    resolved = SimpleNamespace(
        prepared=prepared, params={}, execution_fingerprint="execution-fingerprint",
    )
    facade = SimpleNamespace(run_daily=mock.Mock(side_effect=RuntimeError("crash")))
    with mock.patch("quantpits.utils.operator_log.OperatorLog") as log_cls:
        log_cls.return_value.__enter__.return_value = log_cls.return_value
        with pytest.raises(RollingExecutionError) as caught:
            LegacyRollingExecutionAdapter(facade).execute(
                SimpleNamespace(), prepared, resolved, "rolling-test",
            )
    assert caught.value.code == "rolling_execution_failed"
    result = log_cls.return_value.set_result.call_args.args[0]
    assert result["status"] == "failed"
    assert result["reason_code"] == "rolling_execution_failed"


def test_adapter_propagates_explicit_facade_skip_to_log_and_outcome(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    (tmp_path / "data").mkdir()
    target = SimpleNamespace(
        model_name="demo", target_key="demo@rolling",
        workflow_path="config/demo.yaml", legacy_info={},
    )
    prepared = SimpleNamespace(
        ctx=ctx,
        state=SimpleNamespace(
            path="data/rolling_state.json", status="valid_legacy",
        ),
        targets=(target,), options=SimpleNamespace(action="predict_only"),
        effective_config={"training_method": "slide"},
        plan=SimpleNamespace(metadata={"family": "rolling"}),
        plan_fingerprint="prepared-fingerprint", cli_args=(),
    )
    resolved = SimpleNamespace(
        prepared=prepared, params={}, execution_fingerprint="execution-fingerprint",
    )
    facade = SimpleNamespace(run_predict_only=mock.Mock(return_value={
        "status": "skipped", "reason_code": "rolling_action_skipped",
        "message": "predict-only generated no predictions", "did_execute": True,
    }))
    with mock.patch("quantpits.utils.operator_log.OperatorLog") as log_cls:
        log_cls.return_value.__enter__.return_value = log_cls.return_value
        outcome = LegacyRollingExecutionAdapter(facade).execute(
            SimpleNamespace(), prepared, resolved, "rolling-test",
        )
    assert outcome.status == "skipped"
    assert outcome.did_execute is True
    result = log_cls.return_value.set_result.call_args.args[0]
    assert result["status"] == outcome.status
    assert result["reason_code"] == outcome.reason_code
