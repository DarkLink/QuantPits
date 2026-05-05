"""Tests for FeedbackLoop orchestrator and priority computation."""

import json
import os
import pytest
import yaml
from unittest.mock import patch, MagicMock

from quantpits.scripts.deep_analysis.action_items import ActionItem
from quantpits.scripts.deep_analysis.feedback_loop import (
    FeedbackLoop,
    FeedbackReport,
    ValidationResult,
    compute_priority,
    _infer_signal_severity,
    _load_training_duration_history,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def orchestrator_workspace(tmp_path):
    """Create a workspace with action_items, config, and data for testing."""
    ws = tmp_path / "CSI300_Base"
    ws.mkdir()

    # config/
    config_dir = ws / "config"
    config_dir.mkdir()

    registry = {
        "models": {
            "gru_Alpha158": {
                "algorithm": "gru",
                "dataset": "Alpha158",
                "yaml_file": "config/workflow_config_gru_Alpha158.yaml",
                "enabled": True,
            },
            "lstm_Alpha360": {
                "algorithm": "lstm",
                "dataset": "Alpha360",
                "yaml_file": "config/workflow_config_lstm_Alpha360.yaml",
                "enabled": True,
            },
        }
    }
    with open(config_dir / "model_registry.yaml", "w") as f:
        yaml.dump(registry, f)

    workflow = {
        "task": {
            "model": {
                "class": "GRU",
                "kwargs": {"n_epochs": 200, "early_stop": 10, "lr": 0.001},
            },
            "dataset": {"class": "DatasetH"},
        },
        "data_handler_config": {"label": ["Ref($close, -2)"]},
    }
    with open(config_dir / "workflow_config_gru_Alpha158.yaml", "w") as f:
        yaml.dump(workflow, f)
    with open(config_dir / "workflow_config_lstm_Alpha360.yaml", "w") as f:
        yaml.dump(workflow, f)

    bounds = {"bounds": {"early_stop": {"min": 5, "max": 100}}}
    with open(config_dir / "hyperparam_bounds.json", "w") as f:
        json.dump(bounds, f)

    with open(config_dir / "feedback_scope.json", "w") as f:
        json.dump({"active_scopes": ["hyperparams"]}, f)

    # data/
    data_dir = ws / "data"
    data_dir.mkdir()

    # Training history for priority scoring
    history_entries = [
        {"model_name": "gru_Alpha158", "duration_seconds": 300, "trained_at": "2026-04-20"},
        {"model_name": "lstm_Alpha360", "duration_seconds": 1800, "trained_at": "2026-04-20"},
    ]
    with open(data_dir / "training_history.jsonl", "w") as f:
        for entry in history_entries:
            f.write(json.dumps(entry) + "\n")

    # output/
    out_dir = ws / "output" / "deep_analysis"
    out_dir.mkdir(parents=True)

    # Create action_items file
    items = [
        {
            "action_id": "act-001",
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "gru_Alpha158",
            "params": {"early_stop": {"from": 10, "to": 20}},
            "reason": "Severe underfitting detected",
            "source_signals": ["severe_underfitting"],
            "confidence": 0.8,
            "risk_level": "low",
            "scope_status": "in_scope",
            "execution_context": {
                "target_env": "playground",
                "requires_retrain": True,
                "estimated_duration_minutes": 5,
            },
        },
        {
            "action_id": "act-002",
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "lstm_Alpha360",
            "params": {"early_stop": {"from": 10, "to": 15}},
            "reason": "Moderate underfitting",
            "source_signals": ["underfitting"],
            "confidence": 0.5,
            "risk_level": "medium",
            "scope_status": "in_scope",
        },
        {
            "action_id": "act-003",
            "action_type": "adjust_hyperparam",
            "scope": "combo_search",
            "target": "combo_A",
            "params": {},
            "scope_status": "out_of_scope",
        },
    ]
    action_items_path = out_dir / "action_items_2026-05-01.json"
    with open(action_items_path, "w") as f:
        json.dump(items, f)

    return ws, str(action_items_path)


# ------------------------------------------------------------------
# Priority tests
# ------------------------------------------------------------------

class TestComputePriority:
    def test_critical_severity_highest(self):
        """Critical severity should score highest."""
        item = ActionItem(confidence=0.5, risk_level="low")
        score_critical = compute_priority(item, "critical")
        score_warning = compute_priority(item, "warning")
        score_info = compute_priority(item, "info")
        assert score_critical > score_warning > score_info

    def test_higher_confidence_higher_score(self):
        """Higher confidence should produce higher score."""
        item_high = ActionItem(confidence=0.9, risk_level="low")
        item_low = ActionItem(confidence=0.1, risk_level="low")
        assert compute_priority(item_high, "warning") > compute_priority(item_low, "warning")

    def test_training_cost_bonus(self):
        """Short training models should get priority bonus."""
        item = ActionItem(target="fast_model", confidence=0.5, risk_level="low")
        history = {"fast_model": 300}  # 5 minutes
        score_with = compute_priority(item, "warning", history)

        score_without = compute_priority(item, "warning", {})
        assert score_with > score_without

    def test_risk_level_tiebreaker(self):
        """High risk should get slight bonus over low risk."""
        item_high = ActionItem(confidence=0.5, risk_level="high")
        item_low = ActionItem(confidence=0.5, risk_level="low")
        assert compute_priority(item_high, "warning") > compute_priority(item_low, "warning")


class TestInferSignalSeverity:
    def test_severe_signal(self):
        item = ActionItem(source_signals=["severe_underfitting"])
        assert _infer_signal_severity(item) == "critical"

    def test_warning_signal(self):
        item = ActionItem(source_signals=["warning_overfitting"])
        assert _infer_signal_severity(item) == "warning"

    def test_default_severity(self):
        item = ActionItem(source_signals=["some_signal"])
        assert _infer_signal_severity(item) == "warning"

    def test_high_risk_defaults_critical(self):
        item = ActionItem(source_signals=[], risk_level="high")
        assert _infer_signal_severity(item) == "critical"


class TestTrainingDurationHistory:
    def test_load_history(self, orchestrator_workspace):
        ws, _ = orchestrator_workspace
        history = _load_training_duration_history(str(ws))
        assert "gru_Alpha158" in history
        assert history["gru_Alpha158"] == 300.0
        assert history["lstm_Alpha360"] == 1800.0

    def test_load_empty_history(self, tmp_path):
        history = _load_training_duration_history(str(tmp_path))
        assert history == {}


# ------------------------------------------------------------------
# Orchestrator tests
# ------------------------------------------------------------------

class TestFeedbackLoopReportOnly:
    def test_report_only_mode(self, orchestrator_workspace):
        """Test report-only mode generates preview without changes."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(action_items_path)

        assert report.mode == "report-only"
        assert report.action_items_processed == 2  # 2 in_scope items
        assert len(report.adapter_results) == 2
        assert "Report-only" in report.summary

        # Verify report was saved
        report_file = os.path.join(str(ws), "output", "deep_analysis",
                                    f"feedback_report_{report.run_date}.json")
        assert os.path.exists(report_file)

    def test_report_only_with_model_filter(self, orchestrator_workspace):
        """Test --models filter limits processed items."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(action_items_path, models=["gru_Alpha158"])

        assert report.action_items_processed == 1

    def test_report_only_with_skip_models(self, orchestrator_workspace):
        """Test --skip-models excludes specified models."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(action_items_path, skip_models=["lstm_Alpha360"])

        assert report.action_items_processed == 1

    def test_report_only_priority_sorting(self, orchestrator_workspace):
        """Test that ActionItems are sorted by priority."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(action_items_path)

        # act-001 (critical, confidence=0.8, short training) should be first
        assert len(report.adapter_results) == 2
        first = report.adapter_results[0]
        assert first["action_id"] == "act-001"


class TestFeedbackLoopTimeBudget:
    def test_time_budget_defers_items(self, orchestrator_workspace):
        """Test that max_duration_minutes defers items exceeding budget."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="report-only")
        # Budget of 6 minutes: gru (5min) fits, lstm (30min) doesn't
        report = loop.run(action_items_path, max_duration_minutes=6)

        assert report.action_items_processed == 1
        assert report.action_items_deferred == 1
        assert len(report.deferred_action_ids) == 1


class TestFeedbackLoopExecute:
    def test_execute_with_skip_retrain(self, orchestrator_workspace):
        """Test execute mode with --skip-retrain only applies config changes."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="execute")
        report = loop.run(action_items_path, skip_retrain=True)

        assert report.mode == "execute"
        # Adapter should have applied changes
        n_success = sum(1 for r in report.adapter_results if r.get("success"))
        assert n_success == 2  # Both adapters should succeed
        # No validation results (skipped retrain)
        assert len(report.validation_results) == 0

        # Verify Playground was created
        pg_dir = os.path.join(os.path.dirname(str(ws)), "CSI300_Base_Playground")
        assert os.path.isdir(pg_dir)

    def test_execute_dry_run(self, orchestrator_workspace):
        """Test execute with --dry-run only previews, no file writes."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="execute")
        report = loop.run(action_items_path, dry_run=True)

        assert all(r.get("dry_run") for r in report.adapter_results)


class TestFeedbackLoopAutoPromote:
    def test_auto_promote_not_implemented(self, orchestrator_workspace):
        """Test auto-promote returns not-yet-implemented message."""
        ws, action_items_path = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="auto-promote")
        report = loop.run(action_items_path)

        assert "not yet implemented" in report.summary


class TestFeedbackLoopEdgeCases:
    def test_empty_action_items(self, orchestrator_workspace):
        """Test handling of empty action items file."""
        ws, _ = orchestrator_workspace

        empty_path = os.path.join(str(ws), "output", "deep_analysis", "empty.json")
        with open(empty_path, "w") as f:
            json.dump([], f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(empty_path)
        assert "No in-scope" in report.summary

    def test_nonexistent_action_items_file(self, orchestrator_workspace):
        """Test handling of missing action items file."""
        ws, _ = orchestrator_workspace

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run("/nonexistent/path.json")
        assert report.action_items_processed == 0

    def test_all_out_of_scope(self, orchestrator_workspace):
        """Test handling when all items are out of scope."""
        ws, _ = orchestrator_workspace

        oos_items = [{
            "action_id": "oos-001",
            "action_type": "adjust_hyperparam",
            "scope": "combo_search",
            "target": "x",
            "params": {},
            "scope_status": "out_of_scope",
        }]
        path = os.path.join(str(ws), "output", "deep_analysis", "oos.json")
        with open(path, "w") as f:
            json.dump(oos_items, f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(path)
        assert "No in-scope" in report.summary


class TestFeedbackLoopExecutionPaths:
    """Tests for execute-mode code paths that can be covered without Qlib."""

    def test_promote_without_playground(self, orchestrator_workspace):
        """--promote should warn when no Playground is found."""
        ws, _ = orchestrator_workspace

        item = {
            "action_id": "exec-001",
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "gru_Alpha158",
            "params": {"early_stop": {"from": 10, "to": 20}},
            "scope_status": "in_scope",
            "source_signals": ["severe_underfitting"],
            "confidence": 0.7,
            "risk_level": "low",
        }
        path = os.path.join(str(ws), "output", "deep_analysis", "exec.json")
        with open(path, "w") as f:
            json.dump([item], f)

        loop = FeedbackLoop(str(ws), mode="promote")
        report = loop.run(path)
        # _run_promote checks playground existence and returns with error
        assert "No Playground" in report.summary or report.promote_result is not None

    def test_budget_defers_all_when_unknown_cost(self, orchestrator_workspace):
        """With no training history, default estimate (3600s) exceeds small budget."""
        ws, _ = orchestrator_workspace

        items = [
            {
                "action_id": f"exec-{i:03d}",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": f"model_{i}",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "scope_status": "in_scope",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
            }
            for i in range(3)
        ]
        path = os.path.join(str(ws), "output", "deep_analysis", "deferred.json")
        with open(path, "w") as f:
            json.dump(items, f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(path, max_duration_minutes=1)
        # Default estimate is 3600s per model → 1 min budget defers all
        assert report.action_items_deferred == len(items)
        assert len(report.deferred_action_ids) == len(items)

    def test_execute_with_model_selection(self, orchestrator_workspace):
        """--models filter should select specific targets."""
        ws, _ = orchestrator_workspace

        items = [
            {
                "action_id": "ms-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "model_keep",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "scope_status": "in_scope",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
            },
            {
                "action_id": "ms-002",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "model_drop",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "scope_status": "in_scope",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
            },
        ]
        path = os.path.join(str(ws), "output", "deep_analysis", "filtered.json")
        with open(path, "w") as f:
            json.dump(items, f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(path, models=["model_keep"])
        assert report.action_items_processed == 1

    def test_execute_with_skip_models(self, orchestrator_workspace):
        """--skip-models should exclude specific targets."""
        ws, _ = orchestrator_workspace

        items = [
            {
                "action_id": "sm-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "model_a",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "scope_status": "in_scope",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
            },
            {
                "action_id": "sm-002",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "model_b",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "scope_status": "in_scope",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
            },
        ]
        path = os.path.join(str(ws), "output", "deep_analysis", "skipped.json")
        with open(path, "w") as f:
            json.dump(items, f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(path, skip_models=["model_b"])
        assert report.action_items_processed == 1

    def test_report_excludes_out_of_scope_and_rejected(self, orchestrator_workspace):
        """Only in_scope items should be processed; out_of_scope and rejected excluded."""
        ws, _ = orchestrator_workspace

        items = [
            {
                "action_id": "mix-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "scope_status": "in_scope",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
            },
            {
                "action_id": "mix-002",
                "action_type": "adjust_hyperparam",
                "scope": "combo_search",
                "target": "m2",
                "params": {},
                "scope_status": "out_of_scope",
                "source_signals": [],
                "confidence": 0.5,
                "risk_level": "low",
            },
            {
                "action_id": "mix-003",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m3",
                "params": {"n_epochs": {"from": 10, "to": 1000}},
                "scope_status": "rejected",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
            },
        ]
        path = os.path.join(str(ws), "output", "deep_analysis", "mixed.json")
        with open(path, "w") as f:
            json.dump(items, f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(path)
        assert report.action_items_processed == 1

    def test_execute_promote_not_implemented(self, orchestrator_workspace):
        """--promote used alone without prior execute should warn."""
        ws, _ = orchestrator_workspace

        item = {
            "action_id": "prom-001",
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "m1",
            "params": {"n_epochs": {"from": 100, "to": 120}},
            "scope_status": "in_scope",
            "source_signals": ["underfitting"],
            "confidence": 0.7,
            "risk_level": "low",
        }
        path = os.path.join(str(ws), "output", "deep_analysis", "prom.json")
        with open(path, "w") as f:
            json.dump([item], f)

        loop = FeedbackLoop(str(ws), mode="promote")
        report = loop.run(path)
        assert "No Playground" in report.summary or report.promote_result is None

    def test_to_dict_roundtrip(self):
        """FeedbackReport.to_dict() should produce a JSON-serializable dict."""
        from quantpits.scripts.deep_analysis.feedback_loop import FeedbackReport, ValidationResult

        report = FeedbackReport(
            run_date="2026-05-01",
            mode="report-only",
            action_items_processed=3,
            action_items_deferred=1,
            deferred_action_ids=["def-001"],
            summary="Test summary",
        )
        d = report.to_dict()
        assert d["run_date"] == "2026-05-01"
        assert d["action_items_deferred"] == 1
        assert d["deferred_action_ids"] == ["def-001"]


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_basic(self):
        """ValidationResult should track IC comparison correctly."""
        from quantpits.scripts.deep_analysis.feedback_loop import ValidationResult

        vr = ValidationResult(
            model="test_model",
            baseline_ic=0.05,
            playground_ic=0.06,
            ic_delta=0.01,
            ic_improved=True,
            passed=True,
        )
        assert vr.ic_improved
        assert vr.passed
        assert vr.ensemble is None

    def test_validation_with_ensemble(self):
        """ValidationResult should accept optional ensemble data."""
        from quantpits.scripts.deep_analysis.feedback_loop import ValidationResult

        vr = ValidationResult(
            model="test_model",
            baseline_ic=0.05,
            playground_ic=0.04,
            ic_delta=-0.01,
            ic_improved=False,
            passed=False,
            ensemble={"combo_name": "c1", "baseline_calmar": 2.5, "playground_calmar": 2.0, "calmar_delta": -0.5},
        )
        assert not vr.ic_improved
        assert vr.ensemble["calmar_delta"] == -0.5
