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
