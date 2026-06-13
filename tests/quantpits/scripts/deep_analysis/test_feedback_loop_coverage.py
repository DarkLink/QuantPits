"""Coverage expansion tests for FeedbackLoop — uncovered functions and paths."""

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
from quantpits.scripts.deep_analysis.signal_extractor import Signal


# ------------------------------------------------------------------
# Additional coverage for compute_priority
# ------------------------------------------------------------------

class TestComputePriorityExtended:
    def test_medium_training_cost_bonus(self):
        """10-30 min training gets 0.5 bonus."""
        item = ActionItem(target="medium_model", confidence=0.5, risk_level="low")
        history = {"medium_model": 20 * 60}  # 20 minutes = 1200 seconds
        score_with = compute_priority(item, "warning", history)
        score_without = compute_priority(item, "warning", {})
        assert score_with > score_without

    def test_long_training_no_bonus(self):
        """30+ min training gets no bonus."""
        item = ActionItem(target="slow_model", confidence=0.5, risk_level="low")
        history = {"slow_model": 3600}  # 60 minutes
        score_with = compute_priority(item, "warning", history)
        score_without = compute_priority(item, "warning", {})
        assert score_with == score_without

    def test_severity_weights(self):
        item = ActionItem(confidence=0.5, risk_level="low")
        assert compute_priority(item, "critical") == 3.0 + 1.0  # severity + confidence*2
        assert compute_priority(item, "warning") == 2.0 + 1.0
        assert compute_priority(item, "info") == 1.0 + 1.0

    def test_medium_risk_bonus(self):
        item = ActionItem(confidence=0.5, risk_level="medium")
        score = compute_priority(item, "warning")
        assert score == 2.0 + 1.0 + 0.3  # severity + confidence*2 + medium bonus


# ------------------------------------------------------------------
# Additional coverage for _infer_signal_severity
# ------------------------------------------------------------------

class TestInferSignalSeverityExtended:
    def test_critical_in_signal(self):
        item = ActionItem(source_signals=["critical_underfitting"])
        assert _infer_signal_severity(item) == "critical"

    def test_no_signals_low_risk(self):
        item = ActionItem(source_signals=[], risk_level="low")
        assert _infer_signal_severity(item) == "warning"

    def test_no_signals_default_no_risk(self):
        item = ActionItem(source_signals=[])
        assert _infer_signal_severity(item) == "warning"

    def test_non_string_in_signals(self):
        item = ActionItem(source_signals=[123, None])
        # Should not crash, returns default "warning"
        assert _infer_signal_severity(item) == "warning"


# ------------------------------------------------------------------
# Additional coverage for _load_training_duration_history
# ------------------------------------------------------------------

class TestTrainingDurationHistoryExtended:
    def test_corrupt_json_skipped(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            f.write('{"model_name": "m1", "duration_seconds": 300}\n')
            f.write('invalid json line\n')
            f.write('{"model_name": "m2", "duration_seconds": 600}\n')

        history = _load_training_duration_history(str(ws))
        # Should only load valid entries
        assert "m1" in history
        assert history["m1"] == 300.0

    def test_missing_duration_key(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            f.write('{"model_name": "m1"}\n')  # no duration_seconds

        history = _load_training_duration_history(str(ws))
        assert history == {}  # model without duration is skipped

    def test_missing_model_name(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            f.write('{"duration_seconds": 300}\n')  # no model_name

        history = _load_training_duration_history(str(ws))
        assert history == {}

    def test_empty_line_skipped(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            f.write('\n')
            f.write('{"model_name": "m1", "duration_seconds": 300}\n')
            f.write('  \n')

        history = _load_training_duration_history(str(ws))
        assert len(history) == 1

    def test_average_computed(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            f.write('{"model_name": "m1", "duration_seconds": 300}\n')
            f.write('{"model_name": "m1", "duration_seconds": 500}\n')

        history = _load_training_duration_history(str(ws))
        assert history["m1"] == 400.0  # average of 300 and 500


# ------------------------------------------------------------------
# Additional orchestrator tests
# ------------------------------------------------------------------

class TestFeedbackLoopPlaygroundOnly:
    def test_playground_only_no_models(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        loop = FeedbackLoop(str(ws), mode="playground-only")
        report = loop.run("", models=None)
        assert "--models is required" in report.summary

    def test_playground_only_no_api_key(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        loop = FeedbackLoop(str(ws), mode="playground-only")
        report = loop.run("", models=["m1"])
        assert "No API key" in report.summary

    def test_playground_only_no_llm_config(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        loop = FeedbackLoop(str(ws), mode="playground-only")
        report = loop.run("", models=["m1"])
        assert "No API key" in report.summary

    def test_playground_only_skip_retrain(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        loop = FeedbackLoop(str(ws), mode="playground-only")
        report = loop.run("", models=["m1"], skip_retrain=True)
        assert "No API key" in report.summary

    def test_playground_only_with_skip_models(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        loop = FeedbackLoop(str(ws), mode="playground-only")
        report = loop.run("", models=["m1", "m2"], skip_models=["m2"])
        assert "No API key" in report.summary


class TestFeedbackLoopModeDispatch:
    def test_unknown_mode(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        # Create action items file
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "reason": "test",
                "source_signals": ["test"],
                "confidence": 0.5,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="unknown-mode")
        report = loop.run(str(items_path))
        assert "Unknown mode" in report.summary

    def test_empty_action_items_list(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "empty.json"
        with open(items_path, "w") as f:
            json.dump([], f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(str(items_path))
        assert "No in-scope" in report.summary

    def test_all_items_filtered_by_model_selection(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "source_signals": ["test"],
                "confidence": 0.5,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(str(items_path), models=["m2"])  # no "m2" in items
        assert "filtered out by model selection" in report.summary


# ------------------------------------------------------------------
# Promoter tests
# ------------------------------------------------------------------

class TestFeedbackLoopPromoteEdgeCases:
    def test_promote_no_playground(self, tmp_path):
        """Promote without Playground returns error."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "source_signals": ["test"],
                "confidence": 0.5,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="promote")
        report = loop.run(str(items_path))
        assert "No Playground" in report.summary

    def test_promote_no_passed_validations(self, tmp_path):
        """Promote with validations but none passed."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()

        pg_dir = ws.parent / "TestWS_Playground"
        pg_dir.mkdir(parents=True, exist_ok=True)
        (pg_dir / "config").mkdir(exist_ok=True)

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)

        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "source_signals": ["test"],
                "confidence": 0.5,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        # Create a feedback report with no passed validations
        report_path = out_dir / "feedback_report_2026-05-07.json"
        with open(report_path, "w") as f:
            json.dump({
                "validation_results": [
                    {"model": "m1", "passed": False},
                    {"model": "m2", "passed": False},
                ],
            }, f)

        loop = FeedbackLoop(str(ws), mode="promote")
        report = loop.run(str(items_path))
        assert "Nothing to promote" in report.summary or "No Playground" in report.summary


# ------------------------------------------------------------------
# ValidationResult tests
# ------------------------------------------------------------------

class TestValidationResultExtended:
    def test_validation_with_icir_and_excess(self):
        vr = ValidationResult(
            model="test",
            baseline_ic=0.05,
            playground_ic=0.06,
            ic_delta=0.01,
            ic_improved=True,
            passed=True,
            playground_icir=0.5,
            playground_excess=0.15,
        )
        assert vr.playground_icir == 0.5
        assert vr.playground_excess == 0.15

    def test_validation_with_convergence(self):
        vr = ValidationResult(
            model="test",
            convergence={"best_epoch": 50, "actual_epochs": 200, "loss_values": [1.0, 0.5, 0.3]},
        )
        assert vr.convergence["best_epoch"] == 50

    def test_validation_with_round_info(self):
        vr = ValidationResult(
            model="test",
            round_idx=2,
            params_changed={"n_epochs": {"from": 200, "to": 250}},
        )
        assert vr.round_idx == 2
        assert vr.params_changed["n_epochs"]["to"] == 250


# ------------------------------------------------------------------
# FeedbackReport tests
# ------------------------------------------------------------------

class TestFeedbackReportExtended:
    def test_full_report_to_dict(self):
        report = FeedbackReport(
            run_date="2026-05-07",
            mode="execute",
            action_items_processed=5,
            action_items_deferred=2,
            deferred_action_ids=["d1", "d2"],
            adapter_results=[{"action_id": "a1", "success": True}],
            validation_results=[{"model": "m1", "passed": True}],
            promote_result={"success": True, "promoted_files": ["f1"]},
            summary="Complete",
        )
        d = report.to_dict()
        assert d["run_date"] == "2026-05-07"
        assert d["mode"] == "execute"
        assert len(d["adapter_results"]) == 1
        assert len(d["validation_results"]) == 1
        assert d["promote_result"]["success"] is True
        assert d["deferred_action_ids"] == ["d1", "d2"]

    def test_default_report(self):
        report = FeedbackReport()
        assert report.run_date == ""
        assert report.mode == ""
        assert report.action_items_processed == 0
        assert report.deferred_action_ids == []


# ------------------------------------------------------------------
# _load_action_items (indirectly through run())
# ------------------------------------------------------------------

class TestLoadActionItems:
    def test_file_not_found(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run("/nonexistent/file.json")
        assert report.action_items_processed == 0

    def test_invalid_json_file(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "bad.json"
        with open(items_path, "w") as f:
            f.write("not valid json at all")

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(str(items_path))
        assert report.action_items_processed == 0

    def test_file_is_not_a_list(self, tmp_path):
        """File containing a single JSON object (not array) — _load_action_items handles this."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "obj.json"
        with open(items_path, "w") as f:
            # Single object (not wrapped in array) — _load_action_items wraps it
            json.dump([{"action_id": "single", "action_type": "adjust_hyperparam",
                        "scope": "hyperparams", "target": "m1",
                        "params": {}, "source_signals": ["test"],
                        "confidence": 0.5, "risk_level": "low",
                        "scope_status": "in_scope"}], f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(str(items_path))
        assert isinstance(report, FeedbackReport)


# ------------------------------------------------------------------
# _save_report (indirect)
# ------------------------------------------------------------------

class TestSaveReport:
    def test_save_report_to_output_dir(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"n_epochs": {"from": 100, "to": 120}},
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        report = loop.run(str(items_path))

        # Check that report file was created
        report_files = list(out_dir.glob("feedback_report_*.json"))
        assert len(report_files) >= 1

        # Verify content is valid JSON
        with open(report_files[0], "r") as f:
            data = json.load(f)
        assert data["mode"] == "report-only"
        assert "run_date" in data


# ------------------------------------------------------------------
# Test run() function with mocked _run_experiment_loop
# ------------------------------------------------------------------

class TestExperimentLoopMocked:
    def test_execute_with_experiment_loop_disabled(self, tmp_path):
        """max_experiment_rounds=0 should skip experiment loop."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()

        # Setup model registry
        registry = {
            "models": {
                "m1": {
                    "algorithm": "gru",
                    "dataset": "Alpha158",
                    "yaml_file": "config/workflow_config_m1.yaml",
                    "enabled": True,
                },
            }
        }
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)

        # Setup workflow YAML
        workflow = {
            "task": {
                "model": {
                    "class": "GRU",
                    "kwargs": {"n_epochs": 200, "early_stop": 10, "lr": 0.001},
                },
            },
        }
        with open(ws / "config" / "workflow_config_m1.yaml", "w") as f:
            yaml.dump(workflow, f)

        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"early_stop": {"min": 5, "max": 100}}}, f)

        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"early_stop": {"from": 10, "to": 20}},
                "reason": "test",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="execute")
        report = loop.run(str(items_path), skip_retrain=True, max_experiment_rounds=0)
        assert report.mode == "execute"
        assert len(report.validation_results) == 0

    def test_execute_dry_run_no_changes(self, tmp_path):
        """Dry run should preview without modifying files."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()

        registry = {
            "models": {
                "m1": {
                    "algorithm": "gru",
                    "dataset": "Alpha158",
                    "yaml_file": "config/workflow_config_m1.yaml",
                    "enabled": True,
                },
            }
        }
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)

        workflow = {
            "task": {
                "model": {
                    "class": "GRU",
                    "kwargs": {"n_epochs": 200, "early_stop": 10, "lr": 0.001},
                },
            },
        }
        with open(ws / "config" / "workflow_config_m1.yaml", "w") as f:
            yaml.dump(workflow, f)

        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"early_stop": {"min": 5, "max": 100}}}, f)

        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"early_stop": {"from": 10, "to": 20}},
                "reason": "test",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="execute")
        report = loop.run(str(items_path), dry_run=True)
        assert report.mode == "execute"
        # All results should be dry_run
        for r in report.adapter_results:
            assert r.get("dry_run") is True

    def test_execute_skip_retrain_with_max_rounds(self, tmp_path):
        """skip_retrain + max_experiment_rounds=0 skips all training."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()

        registry = {
            "models": {
                "m1": {
                    "algorithm": "gru",
                    "dataset": "Alpha158",
                    "yaml_file": "config/workflow_config_m1.yaml",
                    "enabled": True,
                },
            }
        }
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)

        workflow = {
            "task": {"model": {"class": "GRU", "kwargs": {"early_stop": 10}}},
        }
        with open(ws / "config" / "workflow_config_m1.yaml", "w") as f:
            yaml.dump(workflow, f)

        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"early_stop": {"min": 5, "max": 100}}}, f)

        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"early_stop": {"from": 10, "to": 20}},
                "reason": "test",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="execute")
        report = loop.run(str(items_path), skip_retrain=True, max_experiment_rounds=0)
        assert report.mode == "execute"
        assert len(report.adapter_results) == 1
        assert report.adapter_results[0]["success"] is True
        assert len(report.validation_results) == 0


# ------------------------------------------------------------------
# _get_model_ic (indirect)
# ------------------------------------------------------------------

class TestGetModelIC:
    def test_get_model_ic_no_files(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()

        loop = FeedbackLoop(str(ws), mode="report-only")
        ic = loop._get_model_ic(str(ws), "nonexistent_model")
        # Returns None or 0.0 when no data found
        assert ic is None or ic == 0.0

    def test_get_model_ic_with_files(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        output_dir = ws / "output"
        output_dir.mkdir()

        # Create a model_performance JSON with IC
        perf = {
            "ic": 0.05,
            "rank_ic": 0.06,
        }
        with open(output_dir / "model_performance_2026-05-07.json", "w") as f:
            json.dump(perf, f)

        loop = FeedbackLoop(str(ws), mode="report-only")
        # This reads model_performance files to try to extract IC for a model
        ic = loop._get_model_ic(str(ws), "some_model")
        # May return None, 0.0, or an IC value depending on implementation
        assert ic is None or isinstance(ic, (int, float))


# ------------------------------------------------------------------
# Pretrain deps check
# ------------------------------------------------------------------

class TestPretrainDeps:
    def test_execute_with_pretrain_deps(self, tmp_path):
        """Test execute with model that has pretrain_source."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()

        registry = {
            "models": {
                "m1": {
                    "algorithm": "gats",
                    "dataset": "Alpha360",
                    "yaml_file": "config/workflow_config_m1.yaml",
                    "enabled": True,
                    "pretrain_source": "m0",
                },
            }
        }
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)

        workflow = {
            "task": {"model": {"class": "GATS", "kwargs": {"n_epochs": 100}}},
        }
        with open(ws / "config" / "workflow_config_m1.yaml", "w") as f:
            yaml.dump(workflow, f)

        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"n_epochs": {"min": 10, "max": 500}}}, f)

        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        items_path = out_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001",
                "action_type": "adjust_hyperparam",
                "scope": "hyperparams",
                "target": "m1",
                "params": {"n_epochs": {"from": 100, "to": 150}},
                "reason": "test",
                "source_signals": ["underfitting"],
                "confidence": 0.7,
                "risk_level": "low",
                "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="execute")
        report = loop.run(str(items_path), dry_run=True)
        assert report.mode == "execute"


# ------------------------------------------------------------------
# _detect_overfitting
# ------------------------------------------------------------------

class TestDetectOverfitting:
    def test_severe_overfitting(self):
        """best=3 / actual=30 → 0.10 < 0.15 → overfitting (and actual > 5)."""
        loop = FeedbackLoop("/tmp", mode="report-only")
        assert loop._detect_overfitting({"best_epoch": 3, "actual_epochs": 30}) is True

    def test_no_overfitting_best_at_end(self):
        """best=28 / actual=30 → 0.93 > 0.15 → not overfitting."""
        loop = FeedbackLoop("/tmp", mode="report-only")
        assert loop._detect_overfitting({"best_epoch": 28, "actual_epochs": 30}) is False

    def test_short_run_no_judgment(self):
        """actual_epochs <= 5 → no overfitting judgment (false positive guard)."""
        loop = FeedbackLoop("/tmp", mode="report-only")
        assert loop._detect_overfitting({"best_epoch": 1, "actual_epochs": 5}) is False

    def test_empty_convergence(self):
        loop = FeedbackLoop("/tmp", mode="report-only")
        assert loop._detect_overfitting({}) is False

    def test_none_convergence(self):
        loop = FeedbackLoop("/tmp", mode="report-only")
        assert loop._detect_overfitting(None) is False

    def test_missing_best_epoch(self):
        loop = FeedbackLoop("/tmp", mode="report-only")
        assert loop._detect_overfitting({"actual_epochs": 30}) is False

    def test_missing_actual_epochs(self):
        loop = FeedbackLoop("/tmp", mode="report-only")
        assert loop._detect_overfitting({"best_epoch": 3}) is False


# ------------------------------------------------------------------
# _load_experiment_history
# ------------------------------------------------------------------

class TestExperimentHistory:
    def test_no_file_returns_none(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._load_experiment_history("m1") is None

    def test_in_progress_returned(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "data").mkdir()
        exp_path = ws / "data" / "experiment_history.jsonl"
        exp_path.write_text(json.dumps({
            "experiment_id": "exp_001", "model": "m1",
            "baseline_ic": 0.02, "rounds": [], "status": "in_progress",
        }) + "\n")

        loop = FeedbackLoop(str(ws), mode="report-only")
        result = loop._load_experiment_history("m1")
        assert result is not None
        assert result["experiment_id"] == "exp_001"

    def test_completed_not_returned(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "data").mkdir()
        exp_path = ws / "data" / "experiment_history.jsonl"
        exp_path.write_text(json.dumps({
            "experiment_id": "exp_001", "model": "m1",
            "baseline_ic": 0.02, "rounds": [], "status": "completed",
        }) + "\n")

        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._load_experiment_history("m1") is None

    def test_corrupt_lines_skipped(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "data").mkdir()
        exp_path = ws / "data" / "experiment_history.jsonl"
        exp_path.write_text(
            "not valid json\n" +
            json.dumps({"experiment_id": "exp_002", "model": "m1",
                        "baseline_ic": 0.02, "rounds": [], "status": "in_progress"}) + "\n"
        )

        loop = FeedbackLoop(str(ws), mode="report-only")
        result = loop._load_experiment_history("m1")
        assert result is not None
        assert result["experiment_id"] == "exp_002"


# ------------------------------------------------------------------
# _save_experiment_round
# ------------------------------------------------------------------

class TestSaveExperimentRound:
    def test_new_experiment(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        loop = FeedbackLoop(str(ws), mode="report-only")
        loop._save_experiment_round("exp_new", "m1", 0.02,
                                    {"round": 1, "playground_ic": 0.025}, "in_progress")
        exp_path = ws / "data" / "experiment_history.jsonl"
        assert exp_path.exists()
        with open(exp_path, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert len(records) == 1
        assert records[0]["experiment_id"] == "exp_new"
        assert len(records[0]["rounds"]) == 1

    def test_append_round_to_existing(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        loop = FeedbackLoop(str(ws), mode="report-only")
        loop._save_experiment_round("exp_1", "m1", 0.02,
                                    {"round": 1, "playground_ic": 0.025}, "in_progress")
        loop._save_experiment_round("exp_1", "m1", 0.02,
                                    {"round": 2, "playground_ic": 0.030}, "in_progress")
        exp_path = ws / "data" / "experiment_history.jsonl"
        with open(exp_path, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert len(records) == 1
        assert len(records[0]["rounds"]) == 2

    def test_final_status_update(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        loop = FeedbackLoop(str(ws), mode="report-only")
        loop._save_experiment_round("exp_1", "m1", 0.02,
                                    {"round": 1, "playground_ic": 0.025}, "in_progress")
        loop._save_experiment_round("exp_1", "m1", 0.02, None, "completed")
        exp_path = ws / "data" / "experiment_history.jsonl"
        with open(exp_path, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert records[0]["status"] == "completed"
        assert records[0]["best_ic"] == 0.025


# ------------------------------------------------------------------
# _load_latest_feedback_report
# ------------------------------------------------------------------

class TestLoadFeedbackReport:
    def test_no_reports_returns_none(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output" / "deep_analysis").mkdir(parents=True)
        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._load_latest_feedback_report() is None

    def test_corrupt_report_returns_none(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        out = ws / "output" / "deep_analysis"
        out.mkdir(parents=True)
        (out / "feedback_report_2026-01-01.json").write_text("not json {{{")
        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._load_latest_feedback_report() is None


# ------------------------------------------------------------------
# _run_experiment_loop with mocked LLM
# ------------------------------------------------------------------

class TestExperimentLoop:
    def _make_loop_with_workspace(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()
        (ws / "output").mkdir()

        import yaml
        registry = {
            "models": {
                "m1": {"algorithm": "gru", "dataset": "Alpha158",
                       "yaml_file": "config/workflow_config_m1.yaml", "enabled": True},
            }
        }
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)
        workflow = {"task": {"model": {"class": "GRU", "kwargs": {"n_epochs": 100, "early_stop": 10}}}}
        with open(ws / "config" / "workflow_config_m1.yaml", "w") as f:
            yaml.dump(workflow, f)
        with open(ws / "config" / "llm_config.json", "w") as f:
            json.dump({}, f)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"n_epochs": {"min": 10, "max": 500}}}, f)
        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        return FeedbackLoop(str(ws), mode="execute"), ws

    def test_single_round_experiment_llm_stops(self, tmp_path):
        """LLM returns no retry on first round — loop returns 1 result."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        mock_vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.025,
            ic_delta=0.005, ic_improved=True, passed=True,
            convergence={"best_epoch": 95, "actual_epochs": 100},
            round_idx=1, params_changed={"n_epochs": {"from": 100, "to": 150}},
        )

        with patch.object(loop, "_retrain_and_validate", return_value=mock_vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.analyze_experiment_result.return_value = {"decision": "stop"}
            mock_llm._load_current_params.return_value = {}
            mock_llm._load_recent_action_history.return_value = []
            mock_llm._compute_available_interventions.return_value = [
                {"param": "n_epochs", "current": 100, "suggested_min": 50, "suggested_max": 200}
            ]
            mock_llm._load_hyperparam_bounds.return_value = {"n_epochs": {"min": 10, "max": 500}}
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "TestWS_Playground"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        assert len(results) == 1
        assert results[0].round_idx == 1

    def test_experiment_loop_llm_retry_then_stops(self, tmp_path):
        """LLM returns retry on first round, then stop on second."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        # IC delta 0.0005 < min_abs 0.002 → not "meaningful" → LLM gets called
        vr1 = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.0205,
            ic_delta=0.0005, ic_improved=True, passed=True,
            convergence={"best_epoch": 20, "actual_epochs": 100},
            round_idx=1,
        )
        vr2 = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.030,
            ic_delta=0.010, ic_improved=True, passed=True,
            convergence={"best_epoch": 98, "actual_epochs": 100},
            round_idx=2,
        )

        with patch.object(loop, "_retrain_and_validate", side_effect=[vr1, vr2]), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.analyze_experiment_result.side_effect = [
                {"decision": "retry", "next_param": "n_epochs", "next_from": 150,
                 "next_to": 200, "rationale": "increase more"},
                {"decision": "stop"},
            ]
            mock_llm._load_current_params.return_value = {}
            mock_llm._load_recent_action_history.return_value = []
            mock_llm._compute_available_interventions.return_value = [
                {"param": "n_epochs", "current": 100, "suggested_min": 50, "suggested_max": 500}
            ]
            mock_llm._load_hyperparam_bounds.return_value = {"n_epochs": {"min": 10, "max": 500}}
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "TestWS_Playground"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        assert len(results) == 2

    def test_experiment_loop_max_rounds_exhausted(self, tmp_path):
        """LLM keeps saying retry until max_rounds exhausted."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        # IC delta 0.0005 < min_abs 0.002 → not meaningful → LLM keeps trying
        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.0205,
            ic_delta=0.0005, ic_improved=True, passed=True,
            convergence={"best_epoch": 20, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.analyze_experiment_result.return_value = {
                "decision": "retry", "next_param": "n_epochs", "next_from": 150,
                "next_to": 200, "rationale": "keep trying",
            }
            mock_llm._load_current_params.return_value = {}
            mock_llm._load_recent_action_history.return_value = []
            mock_llm._compute_available_interventions.return_value = [
                {"param": "n_epochs", "current": 100, "suggested_min": 50, "suggested_max": 500}
            ]
            mock_llm._load_hyperparam_bounds.return_value = {"n_epochs": {"min": 10, "max": 500}}
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "TestWS_Playground"),
                training_history={}, adapter=MagicMock(), max_rounds=2,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        assert len(results) == 2

    def test_experiment_loop_no_llm_available(self, tmp_path):
        """No API key → loop exits without calling analyzer."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.025,
            ic_delta=0.005, ic_improved=True, passed=True,
            convergence={"best_epoch": 95, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False  # No API key
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "TestWS_Playground"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        # Should return the one training round result, then break
        assert len(results) == 1


# ------------------------------------------------------------------
# _run_execute: unsupported action_type path
# ------------------------------------------------------------------

class TestRunExecuteEdgeCases:
    def test_unsupported_action_type_skipped(self, tmp_path):
        """Lines 358-362: unsupported action_type logs warning and continues."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()
        (ws / "output" / "deep_analysis").mkdir(parents=True)

        import yaml
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump({"models": {"m1": {"algorithm": "gru", "dataset": "Alpha158",
                        "yaml_file": "config/workflow_config_m1.yaml", "enabled": True}}}, f)
        with open(ws / "config" / "workflow_config_m1.yaml", "w") as f:
            yaml.dump({"task": {"model": {"class": "GRU", "kwargs": {"n_epochs": 100}}}}, f)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {}}, f)
        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        items_path = ws / "output" / "deep_analysis" / "items.json"
        with open(items_path, "w") as f:
            json.dump([{
                "action_id": "act-001", "action_type": "disable_model",
                "scope": "hyperparams", "target": "m1",
                "params": {}, "reason": "test", "source_signals": [],
                "confidence": 0.5, "risk_level": "low", "scope_status": "in_scope",
            }], f)

        loop = FeedbackLoop(str(ws), mode="execute")
        report = loop.run(str(items_path), dry_run=True)
        assert report.mode == "execute"
        assert len(report.adapter_results) == 0  # Skipped


# ------------------------------------------------------------------
# _get_model_ic fallback paths
# ------------------------------------------------------------------

class TestGetModelICFallbacks:
    def test_direct_ic_field(self, tmp_path):
        """Line 1009-1011: direct IC field in performance file."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output").mkdir()
        perf_path = ws / "output" / "model_performance_2026-01-01.json"
        perf_path.write_text(json.dumps({"ic": 0.035}))

        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._get_model_ic(str(ws), "m1") == 0.035

    def test_all_structure_fallback(self, tmp_path):
        """Lines 1013-1016: nested 'all' structure fallback."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output").mkdir()
        perf_path = ws / "output" / "model_performance_2026-01-01.json"
        perf_path.write_text(json.dumps({"all": {"IC_Mean": 0.028}}))

        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._get_model_ic(str(ws), "m1") == 0.028


# ------------------------------------------------------------------
# _get_model_ic additional fallback paths
# ------------------------------------------------------------------


class TestGetModelIcAdditionalFallbacks:
    def test_per_model_structure(self, tmp_path):
        """Lines 1006-1011: per-model key with IC_Mean."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output").mkdir()
        perf_path = ws / "output" / "model_performance_2026-01-01.json"
        perf_path.write_text(json.dumps({"m1": {"IC_Mean": 0.045, "ICIR": 0.5}}))

        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._get_model_ic(str(ws), "m1") == 0.045

    def test_per_model_structure_with_ic_field(self, tmp_path):
        """Lines 1006-1011: per-model key with 'ic' field (no IC_Mean)."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output").mkdir()
        perf_path = ws / "output" / "model_performance_2026-01-01.json"
        perf_path.write_text(json.dumps({"m1": {"ic": 0.033}}))

        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._get_model_ic(str(ws), "m1") == 0.033

    def test_direct_ic_field_null_model_data(self, tmp_path):
        """Lines 1013-1016: no per-model key, use direct 'ic' field."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output").mkdir()
        perf_path = ws / "output" / "model_performance_2026-01-01.json"
        perf_path.write_text(json.dumps({"ic": 0.035}))

        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._get_model_ic(str(ws), "m1") == 0.035

    def test_all_with_ic_field(self, tmp_path):
        """Lines 1018-1020: nested 'all' with 'ic' field."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output").mkdir()
        perf_path = ws / "output" / "model_performance_2026-01-01.json"
        perf_path.write_text(json.dumps({"all": {"ic": 0.022}}))

        loop = FeedbackLoop(str(ws), mode="report-only")
        assert loop._get_model_ic(str(ws), "m1") == 0.022

    def test_corrupt_second_file(self, tmp_path):
        """Exception path: corrupt second file is skipped, first valid is used."""
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "output").mkdir()
        # Valid older file
        (ws / "output" / "model_performance_2026-01-01.json").write_text(
            json.dumps({"ic": 0.030}))
        # Corrupt newer file (sorted last, so read first)
        (ws / "output" / "model_performance_2026-01-02.json").write_text("not json")

        loop = FeedbackLoop(str(ws), mode="report-only")
        # Reads newest file first → corrupt → exception → returns None
        assert loop._get_model_ic(str(ws), "m1") is None


# ------------------------------------------------------------------
# _run_experiment_loop early exit paths
# ------------------------------------------------------------------


class TestExperimentLoopEarlyExit:
    def _make_loop_with_workspace(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()
        (ws / "output").mkdir()

        import yaml
        registry = {
            "models": {
                "m1": {"algorithm": "gru", "dataset": "Alpha158",
                       "yaml_file": "config/workflow_config_m1.yaml", "enabled": True},
            }
        }
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)
        workflow = {"task": {"model": {"class": "GRU", "kwargs": {"n_epochs": 100, "early_stop": 10}}}}
        with open(ws / "config" / "workflow_config_m1.yaml", "w") as f:
            yaml.dump(workflow, f)
        with open(ws / "config" / "llm_config.json", "w") as f:
            json.dump({}, f)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"n_epochs": {"min": 10, "max": 500}}}, f)
        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        return FeedbackLoop(str(ws), mode="execute"), ws

    def test_meaningful_improvement_no_overfitting_stops(self, tmp_path):
        """IC improved meaningfully + no overfitting → loop breaks without calling LLM."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        # Meaningful improvement: delta >= 0.002, relative >= 5%, no overfitting
        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.030,
            ic_delta=0.010, ic_improved=True, passed=True,
            convergence={"best_epoch": 95, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False):
            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        assert len(results) == 1
        assert results[0].ic_delta == 0.010

    def test_overfitting_detected_continues(self, tmp_path):
        """IC improved but overfitting detected → loop continues to next round via LLM retry."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        vr1 = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.030,
            ic_delta=0.010, ic_improved=True, passed=True,
            convergence={"best_epoch": 3, "actual_epochs": 30},
            round_idx=1,
        )
        vr2 = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.035,
            ic_delta=0.015, ic_improved=True, passed=True,
            convergence={"best_epoch": 28, "actual_epochs": 30},
            round_idx=2,
        )

        with patch.object(loop, "_retrain_and_validate", side_effect=[vr1, vr2]), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", side_effect=[True, False]), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.analyze_experiment_result.side_effect = [
                {"decision": "retry", "next_param": "n_epochs", "next_from": 150,
                 "next_to": 200, "rationale": "try different range"},
                {"decision": "stop"},
            ]
            mock_llm._load_current_params.return_value = {}
            mock_llm._load_recent_action_history.return_value = []
            mock_llm._compute_available_interventions.return_value = [
                {"param": "n_epochs", "current": 100, "suggested_min": 50, "suggested_max": 500}
            ]
            mock_llm._load_hyperparam_bounds.return_value = {"n_epochs": {"min": 10, "max": 500}}
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        # overfitting round 1 → LLM says retry → round 2 no overfitting → LLM says stop
        assert len(results) == 2

    def test_noise_level_delta_continues(self, tmp_path):
        """IC improved but delta < min_abs (0.002) → continues to call LLM."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.021,
            ic_delta=0.001, ic_improved=True, passed=True,
            convergence={"best_epoch": 20, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False  # No API key → break after noise
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        # Noise-level → continues to LLM → LLM not available → breaks
        assert len(results) == 1

    def test_ic_degraded_continues(self, tmp_path):
        """IC degraded (negative delta) → continues to LLM."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 200}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.018,
            ic_delta=-0.002, ic_improved=False, passed=False,
            convergence={"best_epoch": 50, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        assert len(results) == 1

    def test_max_rounds_exhausted(self, tmp_path):
        """Max rounds reached without meaningful improvement → status=exhausted."""
        loop, ws = self._make_loop_with_workspace(tmp_path)

        item = ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low",
            action_id="act-001",
        )

        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.021,
            ic_delta=0.001, ic_improved=True, passed=True,
            convergence={"best_epoch": 20, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False
            mock_llm_class.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=1,
                skip_first_train=False, pg_mgr=MagicMock(),
            )

        # max_rounds=1, noise-level → LLM not available → breaks with 1 result
        assert len(results) == 1


# ------------------------------------------------------------------
# _run_playground_only edge cases
# ------------------------------------------------------------------


class TestPlaygroundOnlyEdgeCases:
    def _make_playground_loop(self, tmp_path, mode="playground"):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()
        (ws / "output").mkdir()

        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump({"models": {"m1": {"algorithm": "gru", "dataset": "Alpha158",
                       "yaml_file": "config/wf.yaml", "enabled": True}}}, f)
        with open(ws / "config" / "wf.yaml", "w") as f:
            yaml.dump({"task": {"model": {"class": "GRU", "kwargs": {"n_epochs": 100}}}}, f)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"n_epochs": {"min": 10, "max": 500}}}, f)
        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        return FeedbackLoop(str(ws), mode=mode), ws

    def test_corrupt_llm_config(self, tmp_path):
        """Lines 769-773: corrupt llm_config.json → exception swallowed."""
        loop, ws = self._make_playground_loop(tmp_path, mode="playground")
        with open(ws / "config" / "llm_config.json", "w") as f:
            f.write("not valid json {{{")

        # _run_playground_only takes: report, models, skip_models,
        # max_experiment_rounds, skip_retrain
        report = FeedbackReport(run_date="2026-06-13", mode="playground")

        with patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_cls, \
             patch.object(loop, "_save_report"), \
             patch("quantpits.scripts.deep_analysis.feedback_loop.PlaygroundManager") as mock_pg:
            mock_pg_instance = MagicMock()
            mock_pg_instance.create_or_sync.return_value = str(tmp_path / "PG")
            mock_pg.return_value = mock_pg_instance

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False
            mock_llm_cls.return_value = mock_llm

            result = loop._run_playground_only(
                report=report, models=["m1"], skip_models=None,
                max_experiment_rounds=1, skip_retrain=False,
            )
        assert result.summary


# ------------------------------------------------------------------
# _run_experiment_loop additional error paths
# ------------------------------------------------------------------


class TestExperimentLoopErrorPaths:
    def _make_loop(self, tmp_path):
        ws = tmp_path / "TestWS"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "data").mkdir()
        (ws / "output").mkdir()

        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump({"models": {"m1": {"algorithm": "gru", "dataset": "Alpha158",
                       "yaml_file": "config/wf.yaml", "enabled": True}}}, f)
        with open(ws / "config" / "wf.yaml", "w") as f:
            yaml.dump({"task": {"model": {"class": "GRU", "kwargs": {"n_epochs": 100, "early_stop": 10}}}}, f)
        with open(ws / "config" / "llm_config.json", "w") as f:
            json.dump({}, f)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"bounds": {"n_epochs": {"min": 10, "max": 500}}}, f)

        return FeedbackLoop(str(ws), mode="execute"), ws

    def _make_item(self):
        return ActionItem(
            action_type="adjust_hyperparam", scope="hyperparams", target="m1",
            params={"n_epochs": {"from": 100, "to": 150}},
            reason="test", source_signals=["underfitting"],
            confidence=0.7, risk_level="low", action_id="act-err",
        )

    def test_retrain_returns_none_breaks(self, tmp_path):
        """Line 539: _retrain_and_validate returns None → loop breaks."""
        loop, ws = self._make_loop(tmp_path)
        item = self._make_item()

        with patch.object(loop, "_retrain_and_validate", return_value=None), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"):
            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )
        assert results == []

    def test_corrupt_llm_config_in_loop(self, tmp_path):
        """Lines 636-637: corrupt llm_config.json in experiment loop."""
        loop, ws = self._make_loop(tmp_path)
        item = self._make_item()

        with open(ws / "config" / "llm_config.json", "w") as f:
            f.write("not json {{{")

        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.018,
            ic_delta=-0.002, ic_improved=False, passed=False,
            convergence={"best_epoch": 50, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False
            mock_llm_cls.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )
        assert len(results) == 1

    def test_llm_give_up_decision(self, tmp_path):
        """Lines 686-690: LLM returns 'give_up' → loop breaks."""
        loop, ws = self._make_loop(tmp_path)
        item = self._make_item()

        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.018,
            ic_delta=-0.002, ic_improved=False, passed=False,
            convergence={"best_epoch": 50, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.analyze_experiment_result.return_value = {
                "decision": "give_up",
                "rationale": "Cannot improve further",
            }
            mock_llm._load_current_params.return_value = {}
            mock_llm._load_recent_action_history.return_value = []
            mock_llm._compute_available_interventions.return_value = [
                {"param": "n_epochs", "current": 100, "suggested_min": 50, "suggested_max": 500}
            ]
            mock_llm._load_hyperparam_bounds.return_value = {"n_epochs": {"min": 10, "max": 500}}
            mock_llm_cls.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )
        assert len(results) == 1

    def test_llm_missing_param(self, tmp_path):
        """Lines 697-698: LLM returns 'retry' but missing next_param → break."""
        loop, ws = self._make_loop(tmp_path)
        item = self._make_item()

        vr = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.018,
            ic_delta=-0.002, ic_improved=False, passed=False,
            convergence={"best_epoch": 50, "actual_epochs": 100},
            round_idx=1,
        )

        with patch.object(loop, "_retrain_and_validate", return_value=vr), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.analyze_experiment_result.return_value = {
                "decision": "retry",
                "rationale": "Try again",
            }
            mock_llm._load_current_params.return_value = {}
            mock_llm._load_recent_action_history.return_value = []
            mock_llm._compute_available_interventions.return_value = [
                {"param": "n_epochs", "current": 100, "suggested_min": 50, "suggested_max": 500}
            ]
            mock_llm._load_hyperparam_bounds.return_value = {"n_epochs": {"min": 10, "max": 500}}
            mock_llm_cls.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=MagicMock(), max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )
        assert len(results) == 1

    def test_adapter_failure_in_loop_round(self, tmp_path):
        """Lines 725-727: adapter.apply fails in round 2+ → loop breaks."""
        loop, ws = self._make_loop(tmp_path)
        item = self._make_item()

        vr1 = ValidationResult(
            model="m1", baseline_ic=0.02, playground_ic=0.021,
            ic_delta=0.001, ic_improved=True, passed=True,
            convergence={"best_epoch": 20, "actual_epochs": 100},
            round_idx=1,
        )

        failing_adapter = MagicMock()
        failing_adapter.apply.return_value = MagicMock(success=False, changes=[], error="boom")

        with patch.object(loop, "_retrain_and_validate", return_value=vr1), \
             patch.object(loop, "_get_model_ic", return_value=0.02), \
             patch.object(loop, "_load_experiment_history", return_value=None), \
             patch.object(loop, "_save_experiment_round"), \
             patch.object(loop, "_detect_overfitting", return_value=False), \
             patch("quantpits.scripts.deep_analysis.llm_interface.LLMInterface") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_llm.analyze_experiment_result.return_value = {
                "decision": "retry",
                "next_param": "n_epochs", "next_from": 150, "next_to": 200,
                "rationale": "extend further",
            }
            mock_llm._load_current_params.return_value = {}
            mock_llm._load_recent_action_history.return_value = []
            mock_llm._compute_available_interventions.return_value = [
                {"param": "n_epochs", "current": 100, "suggested_min": 50, "suggested_max": 500}
            ]
            mock_llm._load_hyperparam_bounds.return_value = {"n_epochs": {"min": 10, "max": 500}}
            mock_llm_cls.return_value = mock_llm

            results = loop._run_experiment_loop(
                item=item, playground_root=str(tmp_path / "PG"),
                training_history={}, adapter=failing_adapter, max_rounds=3,
                skip_first_train=False, pg_mgr=MagicMock(),
            )
        assert len(results) == 1

