"""Tests for Feedback Evaluator (HistoryReader + FeedbackEvaluator)."""

import json
import pytest
from datetime import datetime, timedelta
from quantpits.scripts.deep_analysis.feedback_evaluator import (
    HistoryReader, FeedbackEvaluator, FeedbackSnapshot, run_feedback_loop,
)


def _date(offset_days=0):
    return (datetime.now() + timedelta(days=offset_days)).strftime("%Y-%m-%d")


# ------------------------------------------------------------------
# HistoryReader
# ------------------------------------------------------------------

class TestHistoryReader:
    def test_read_no_previous_analysis(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        snapshot = reader.read(current_date=_date(0))
        assert snapshot.period_start == "unknown"
        assert snapshot.operator_actions == []

    def test_find_last_analysis_date(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text("[]")
        (out_dir / "action_items_2026-05-05.json").write_text("[]")
        (out_dir / "action_items_2026-05-10.json").write_text("[]")

        reader = HistoryReader(workspace_root=str(ws))
        prev = reader._find_last_analysis_date("2026-05-10")
        assert prev == "2026-05-05"

    def test_find_last_analysis_date_none_before(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-10.json").write_text("[]")

        reader = HistoryReader(workspace_root=str(ws))
        prev = reader._find_last_analysis_date("2026-05-10")
        assert prev is None

    def test_read_operator_actions(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        data_dir = ws / "data"
        data_dir.mkdir()
        log_path = data_dir / "operator_log.jsonl"
        entries = [
            {"timestamp_start": "2026-05-03T10:00:00", "script": "static_train",
             "args": ["--all-enabled"], "source": "human", "duration_seconds": 100,
             "log_id": "log1"},
            {"timestamp_start": "2026-05-04T11:00:00", "script": "ensemble_fusion",
             "args": [], "source": "human", "duration_seconds": 50, "log_id": "log2"},
            {"timestamp_start": "2026-05-01T09:00:00", "script": "static_train",
             "args": [], "source": "scheduled", "duration_seconds": 200, "log_id": "log3"},
        ]
        with open(log_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        reader = HistoryReader(workspace_root=str(ws))
        actions = reader._read_operator_actions("2026-05-01", "2026-05-10")
        # Only human actions in range
        assert len(actions) == 2
        scripts = {a["script"] for a in actions}
        assert scripts == {"static_train", "ensemble_fusion"}

    def test_detect_model_list_changes(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output"
        out_dir.mkdir(parents=True)

        prev = {"model_a": {"IC_Mean": 0.03}, "model_b": {"IC_Mean": 0.02}}
        curr = {"model_a": {"IC_Mean": 0.04}, "model_c": {"IC_Mean": 0.01}}
        with open(out_dir / "model_performance_2026-04-30.json", "w") as f:
            json.dump(prev, f)
        with open(out_dir / "model_performance_2026-05-10.json", "w") as f:
            json.dump(curr, f)

        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_model_list_changes("2026-04-30", "2026-05-10")
        assert changes["added"] == ["model_c"]
        assert changes["removed"] == ["model_b"]

    def test_compute_performance_deltas(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output"
        out_dir.mkdir(parents=True)

        prev = {"model_a": {"IC_Mean": 0.03, "ICIR": 0.20}}
        curr = {"model_a": {"IC_Mean": 0.04, "ICIR": 0.25}}
        with open(out_dir / "model_performance_2026-04-30.json", "w") as f:
            json.dump(prev, f)
        with open(out_dir / "model_performance_2026-05-10.json", "w") as f:
            json.dump(curr, f)

        reader = HistoryReader(workspace_root=str(ws))
        deltas = reader._compute_performance_deltas("2026-04-30", "2026-05-10")
        assert "model_a" in deltas
        assert deltas["model_a"]["ic_delta"] == 0.01
        assert deltas["model_a"]["icir_delta"] == 0.05

    def test_detect_hyperparam_changes(self, tmp_path):
        ws = tmp_path / "ws"
        data_dir = ws / "data"
        data_dir.mkdir(parents=True)

        records = [
            {"model_name": "model_a", "trained_at": "2026-04-28T10:00:00",
             "n_epochs": 200, "lr": 0.001, "early_stop": 10},
            {"model_name": "model_a", "trained_at": "2026-05-08T10:00:00",
             "n_epochs": 200, "lr": 0.0005, "early_stop": 20},
        ]
        with open(data_dir / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_hyperparam_changes("2026-04-30", "2026-05-10")
        assert len(changes) >= 1
        lr_changes = [c for c in changes if c["param"] == "lr"]
        assert len(lr_changes) == 1
        assert lr_changes[0]["from"] == 0.001
        assert lr_changes[0]["to"] == 0.0005

    def test_read_full_snapshot(self, tmp_path):
        ws = tmp_path / "ws"
        for d in ["output/deep_analysis", "data"]:
            (ws / d).mkdir(parents=True)

        # Previous action items
        (ws / "output" / "deep_analysis" / "action_items_2026-05-01.json").write_text("[]")
        # Operator log
        log_path = ws / "data" / "operator_log.jsonl"
        log_path.write_text(json.dumps({
            "timestamp_start": "2026-05-03T10:00:00", "script": "static_train",
            "args": ["--all-enabled"], "source": "human", "log_id": "x",
        }) + "\n")

        reader = HistoryReader(workspace_root=str(ws))
        snapshot = reader.read(current_date="2026-05-10")
        assert snapshot.period_start == "2026-05-01"
        assert snapshot.period_end == "2026-05-10"
        assert len(snapshot.operator_actions) == 1


# ------------------------------------------------------------------
# FeedbackEvaluator
# ------------------------------------------------------------------

class TestFeedbackEvaluator:
    def test_no_previous_analysis(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(period_start="unknown")
        result = evaluator.evaluate(snapshot, current_date=_date(0))
        assert result["evaluated"] is False
        assert "No previous analysis" in result["reason"]

    def test_evaluates_items(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)

        prev_items = [
            {"action_id": "id1", "action_type": "adjust_hyperparam",
             "target": "model_a", "params": {"lr": {"from": 0.001, "to": 0.0005}},
             "reason": "Test reason"},
        ]
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps(prev_items))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01",
            period_end="2026-05-10",
            operator_actions=[{
                "date": "2026-05-03", "script": "static_train",
                "args": ["--models", "model_a"], "log_id": "x",
                "duration_s": 100,
            }],
            performance_deltas={
                "model_a": {"ic_delta": 0.005, "icir_delta": 0.03,
                            "prev_ic": 0.030, "curr_ic": 0.035,
                            "prev_icir": 0.20, "curr_icir": 0.23},
            },
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["evaluated"] is True
        assert len(result["per_item_evaluations"]) == 1
        ev = result["per_item_evaluations"][0]
        assert ev["quality"] == "correct_effective"  # executed + IC improved

    def test_correct_ignored(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps([
            {"action_id": "id1", "action_type": "adjust_hyperparam",
             "target": "model_a", "params": {"lr": {"to": 0.0005}}, "reason": "x"},
        ]))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01", period_end="2026-05-10",
            operator_actions=[],  # not executed
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["per_item_evaluations"][0]["quality"] == "correct_ignored"

    def test_incorrect_param_adjustment(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps([
            {"action_id": "id1", "action_type": "adjust_hyperparam",
             "target": "model_a", "params": {"lr": {"from": 0.001, "to": 0.0005}},
             "reason": "x"},
        ]))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01", period_end="2026-05-10",
            operator_actions=[{
                "date": "2026-05-03", "script": "static_train",
                "args": ["--models", "model_a"], "log_id": "x", "duration_s": 100,
            }],
            performance_deltas={
                "model_a": {"ic_delta": -0.01, "icir_delta": -0.05,
                            "prev_ic": 0.030, "curr_ic": 0.020,
                            "prev_icir": 0.20, "curr_icir": 0.15},
            },
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["per_item_evaluations"][0]["quality"] == "incorrect"

    def test_enable_model(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps([
            {"action_id": "id1", "action_type": "enable_model",
             "target": "model_b", "params": {}, "reason": "x"},
        ]))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01", period_end="2026-05-10",
            operator_actions=[{
                "date": "2026-05-03", "script": "static_train",
                "args": ["--models", "model_b"], "log_id": "x", "duration_s": 100,
            }],
            model_list_changes={"added": ["model_b"], "removed": []},
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["per_item_evaluations"][0]["quality"] == "correct_effective"

    def test_quality_summary(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps([
            {"action_id": "id1", "action_type": "adjust_hyperparam",
             "target": "a", "params": {"lr": {"to": 0.0005}}, "reason": "x"},
            {"action_id": "id2", "action_type": "adjust_hyperparam",
             "target": "b", "params": {"dropout": {"to": 0.3}}, "reason": "x"},
        ]))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01", period_end="2026-05-10",
            operator_actions=[],  # neither executed
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        qs = result["quality_summary"]
        assert qs["total"] == 2
        assert qs["correct_ignored"] == 2
        assert qs["accuracy"] == 1.0

    def test_self_corrections_disable_pattern(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps([
            {"action_id": "id1", "action_type": "disable_model",
             "target": "model_z", "params": {}, "reason": "x"},
        ]))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01", period_end="2026-05-10",
            operator_actions=[{
                "date": "2026-05-03", "script": "static_train",
                "args": ["--disable", "model_z"], "log_id": "x", "duration_s": 100,
            }],
            performance_deltas={
                "model_z": {"ic_delta": -0.02, "icir_delta": None,
                            "prev_ic": 0.01, "curr_ic": -0.01,
                            "prev_icir": 0.05, "curr_icir": -0.05},
            },
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        eval_q = result["per_item_evaluations"][0]["quality"]
        # Disable was executed AND IC got worse → disable was correct
        assert eval_q == "correct_effective"

    def test_run_feedback_loop_no_data(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        result = run_feedback_loop(str(ws), current_date="2026-05-10")
        assert result["evaluated"] is False


# ------------------------------------------------------------------
# Helper edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_classify_enable_not_executed(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps([
            {"action_id": "id1", "action_type": "enable_model",
             "target": "model_b", "params": {}, "reason": "x"},
        ]))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01", period_end="2026-05-10",
            operator_actions=[],
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["per_item_evaluations"][0]["quality"] == "correct_ignored"

    def test_params_changed_detection(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps([
            {"action_id": "id1", "action_type": "adjust_hyperparam",
             "target": "model_a", "params": {"lr": {"from": 0.001, "to": 0.0005}},
             "reason": "x"},
        ]))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(
            has_meaningful_change=True,
            period_start="2026-05-01", period_end="2026-05-10",
            operator_actions=[],
            hyperparam_changes=[{
                "model": "model_a", "param": "lr",
                "from": 0.001, "to": 0.0005,
            }],
            performance_deltas={
                "model_a": {"ic_delta": 0.0, "icir_delta": 0.0,
                            "prev_ic": 0.030, "curr_ic": 0.030,
                            "prev_icir": 0.20, "curr_icir": 0.20},
            },
        )
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        # params changed but no IC delta → pending_verification
        assert result["per_item_evaluations"][0]["quality"] == "pending_verification"

    def test_corrupt_action_items_file(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text("not valid json")

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(period_start="2026-05-01", period_end="2026-05-10")
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["evaluated"] is False

    def test_snapshot_to_dict(self):
        snap = FeedbackSnapshot(
            period_start="2026-05-01",
            period_end="2026-05-10",
            operator_actions=[{"script": "test"}],
            model_list_changes={"added": ["m1"]},
        )
        d = snap.to_dict()
        assert d["period_start"] == "2026-05-01"
        assert d["operator_actions"] == [{"script": "test"}]


# ------------------------------------------------------------------
# HistoryReader edge cases
# ------------------------------------------------------------------

class TestHistoryReaderEdgeCases:
    def test_find_closest_snapshot_no_files(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        result = reader._find_closest_snapshot("2026-05-10", "output/nonexistent_*.json")
        assert result is None

    def test_find_closest_snapshot_no_date_in_name(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output"
        out_dir.mkdir(parents=True)
        (out_dir / "model_performance_nodate.json").write_text("{}")

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._find_closest_snapshot("2026-05-10",
                                                "output/model_performance_*.json")
        assert result is None

    def test_read_operator_actions_no_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        actions = reader._read_operator_actions("2026-05-01", "2026-05-10")
        assert actions == []

    def test_read_operator_actions_corrupt(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        (ws / "data" / "operator_log.jsonl").write_text(
            '{"timestamp_start": "2026-05-03T10:00:00", "script": "x", "source": "human"}\n'
            'corrupt line\n'
        )
        reader = HistoryReader(workspace_root=str(ws))
        actions = reader._read_operator_actions("2026-05-01", "2026-05-10")
        assert len(actions) == 1  # valid line counted

    def test_detect_model_list_changes_no_files(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_model_list_changes("2026-04-30", "2026-05-10")
        assert changes == {}

    def test_detect_model_list_changes_corrupt(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output"
        out_dir.mkdir(parents=True)
        (out_dir / "model_performance_2026-04-30.json").write_text("not json")
        (out_dir / "model_performance_2026-05-10.json").write_text('{"m1": {"IC_Mean": 0.03}}')

        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_model_list_changes("2026-04-30", "2026-05-10")
        assert changes == {}

    def test_compute_performance_deltas_no_files(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        deltas = reader._compute_performance_deltas("2026-04-30", "2026-05-10")
        assert deltas == {}

    def test_detect_hyperparam_changes_no_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_hyperparam_changes("2026-04-30", "2026-05-10")
        assert changes == []

    def test_detect_hyperparam_changes_corrupt(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        (ws / "data" / "training_history.jsonl").write_text("not json\n")

        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_hyperparam_changes("2026-04-30", "2026-05-10")
        assert changes == []

    def test_compute_performance_deltas_skips_non_dict(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output"
        out_dir.mkdir(parents=True)

        prev = {"m1": "not_a_dict"}
        curr = {"m1": {"IC_Mean": 0.04, "ICIR": 0.25}}
        with open(out_dir / "model_performance_2026-04-30.json", "w") as f:
            json.dump(prev, f)
        with open(out_dir / "model_performance_2026-05-10.json", "w") as f:
            json.dump(curr, f)

        reader = HistoryReader(workspace_root=str(ws))
        deltas = reader._compute_performance_deltas("2026-04-30", "2026-05-10")
        assert deltas == {}

    def test_detect_combo_changes(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output"
        out_dir.mkdir(parents=True)
        (out_dir / "combo_comparison_2026-04-30.csv").write_text("col1,col2\n1,2")
        (out_dir / "combo_comparison_2026-05-10.csv").write_text("col1,col2\n3,4")

        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_combo_changes("2026-04-30", "2026-05-10")
        assert len(changes) == 1
        assert changes[0]["type"] == "combo_comparison_updated"

    def test_detect_combo_changes_no_files(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        changes = reader._detect_combo_changes("2026-04-30", "2026-05-10")
        assert changes == []


# ------------------------------------------------------------------
# FeedbackEvaluator edge cases
# ------------------------------------------------------------------

class TestFeedbackEvaluatorEdgeCases:
    def test_evaluate_with_no_action_items_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(period_start="2026-05-01", period_end="2026-05-10")
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["evaluated"] is False

    def test_evaluate_empty_action_items_list(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text("[]")

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(period_start="2026-05-01", period_end="2026-05-10")
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["evaluated"] is False

    def test_evaluate_with_action_items_not_list(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        (out_dir / "action_items_2026-05-01.json").write_text('{"key": "not a list"}')

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(period_start="2026-05-01", period_end="2026-05-10")
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        assert result["evaluated"] is False

    def test_self_corrections_batch_param(self, tmp_path):
        ws = tmp_path / "ws"
        out_dir = ws / "output" / "deep_analysis"
        out_dir.mkdir(parents=True)
        # 3 models all with same early_stop adjustment
        items = [
            {"action_id": f"id{i}", "action_type": "adjust_hyperparam",
             "target": f"model_{i}", "params": {"early_stop": {"from": 10, "to": 20}},
             "reason": "x"}
            for i in range(3)
        ]
        (out_dir / "action_items_2026-05-01.json").write_text(json.dumps(items))

        evaluator = FeedbackEvaluator(workspace_root=str(ws))
        snapshot = FeedbackSnapshot(period_start="2026-05-01", period_end="2026-05-10")
        result = evaluator.evaluate(snapshot, current_date="2026-05-10")
        corrections = result.get("self_corrections", [])
        # Should have a batch-param rule
        batch_rules = [c for c in corrections if "Batch" in c.get("pattern", "")]
        assert len(batch_rules) >= 1

    def test_quality_summary_empty(self):
        evaluator = FeedbackEvaluator(workspace_root="/tmp")
        summary = evaluator._summarize([])
        assert summary == {"total": 0}

    def test_was_executed_checks_args(self):
        from quantpits.scripts.deep_analysis.feedback_evaluator import FeedbackEvaluator as FE

        action = [{"date": "2026-05-03", "script": "static_train",
                    "args": ["--models", "model_a"], "log_id": "x", "duration_s": 100}]
        item = {"target": "model_a", "action_type": "adjust_hyperparam"}
        assert FE._was_executed(item, FeedbackSnapshot(operator_actions=action)) is True

        # Different model in args
        action2 = [{"date": "2026-05-03", "script": "static_train",
                     "args": ["--models", "model_b"], "log_id": "x", "duration_s": 100}]
        assert FE._was_executed(item, FeedbackSnapshot(operator_actions=action2)) is False


# ------------------------------------------------------------------
# _check_data_advanced fallback + _latest_anchor
# ------------------------------------------------------------------

class TestCheckDataAdvanced:
    """Cover the fallback path of _check_data_advanced (without latest_data_date)."""

    def test_fallback_no_perf_files(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced("2026-05-01", "2026-05-10")
        assert result is False

    def test_fallback_same_file_returns_false(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        perf = {"model_a": {"convergence": {"anchor_date": "2026-05-05"}}}
        with open(ws / "output" / "model_performance_2026-05-10.json", "w") as f:
            json.dump(perf, f)

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced("2026-05-01", "2026-05-10")
        # Both snapshots resolve to the same file → same file → False
        assert result is False

    def test_fallback_prev_anchor_newer_than_curr(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        with open(ws / "output" / "model_performance_2026-05-01.json", "w") as f:
            json.dump({"model_a": {"convergence": {"anchor_date": "2026-05-10"}}}, f)
        with open(ws / "output" / "model_performance_2026-05-10.json", "w") as f:
            json.dump({"model_a": {"convergence": {"anchor_date": "2026-05-05"}}}, f)

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced("2026-05-01", "2026-05-10")
        assert result is False

    def test_fallback_curr_anchor_newer_returns_true(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        with open(ws / "output" / "model_performance_2026-05-01.json", "w") as f:
            json.dump({"model_a": {"convergence": {"anchor_date": "2026-05-01"}}}, f)
        with open(ws / "output" / "model_performance_2026-05-10.json", "w") as f:
            json.dump({"model_a": {"convergence": {"anchor_date": "2026-05-10"}}}, f)

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced("2026-05-01", "2026-05-10")
        assert result is True

    def test_fallback_corrupted_json_returns_false(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        with open(ws / "output" / "model_performance_2026-05-01.json", "w") as f:
            json.dump({"model_a": {"convergence": {"anchor_date": "2026-05-01"}}}, f)
        with open(ws / "output" / "model_performance_2026-05-10.json", "w") as f:
            f.write("not json")

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced("2026-05-01", "2026-05-10")
        assert result is False

    def test_fallback_missing_anchor_returns_false(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        with open(ws / "output" / "model_performance_2026-05-01.json", "w") as f:
            json.dump({"model_a": {"no_convergence": {}}}, f)
        with open(ws / "output" / "model_performance_2026-05-10.json", "w") as f:
            json.dump({"model_a": {"convergence": {}}}, f)

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced("2026-05-01", "2026-05-10")
        assert result is False

    def test_primary_path_with_latest_data_date_newer(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        with open(ws / "output" / "model_performance_2026-05-01.json", "w") as f:
            json.dump({"model_a": {"convergence": {"anchor_date": "2026-05-01"}}}, f)

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced(
            "2026-05-01", "2026-05-10", latest_data_date="2026-05-15",
        )
        assert result is True

    def test_primary_path_with_latest_data_date_not_newer(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        with open(ws / "output" / "model_performance_2026-05-01.json", "w") as f:
            json.dump({"model_a": {"convergence": {"anchor_date": "2026-05-20"}}}, f)

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced(
            "2026-05-01", "2026-05-10", latest_data_date="2026-05-15",
        )
        assert result is False

    def test_primary_path_no_perf_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced(
            "2026-05-01", "2026-05-10", latest_data_date="2026-05-15",
        )
        assert result is False

    def test_primary_path_corrupted_perf_file(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        with open(ws / "output" / "model_performance_2026-05-01.json", "w") as f:
            f.write("garbage")

        reader = HistoryReader(workspace_root=str(ws))
        result = reader._check_data_advanced(
            "2026-05-01", "2026-05-10", latest_data_date="2026-05-15",
        )
        assert result is False


class TestLatestAnchor:
    def test_empty_dict(self):
        assert HistoryReader._latest_anchor({}) is None

    def test_no_convergence(self):
        data = {"model_a": {"ic": 0.05}}
        assert HistoryReader._latest_anchor(data) is None

    def test_empty_anchor_in_convergence(self):
        data = {"model_a": {"convergence": {"anchor_date": ""}}}
        assert HistoryReader._latest_anchor(data) is None

    def test_single_model(self):
        data = {"model_a": {"convergence": {"anchor_date": "2026-05-10"}}}
        assert HistoryReader._latest_anchor(data) == "2026-05-10"

    def test_multiple_models_returns_latest(self):
        data = {
            "model_a": {"convergence": {"anchor_date": "2026-05-01"}},
            "model_b": {"convergence": {"anchor_date": "2026-05-10"}},
            "model_c": {"convergence": {"anchor_date": "2026-05-05"}},
        }
        assert HistoryReader._latest_anchor(data) == "2026-05-10"

    def test_skips_non_dict_values(self):
        data = {
            "model_a": "not a dict",
            "model_b": {"convergence": {"anchor_date": "2026-05-10"}},
        }
        assert HistoryReader._latest_anchor(data) == "2026-05-10"

    def test_model_without_convergence_skipped(self):
        data = {
            "model_a": {"no_convergence": {}},
            "model_b": {"convergence": {"anchor_date": "2026-05-10"}},
        }
        assert HistoryReader._latest_anchor(data) == "2026-05-10"
