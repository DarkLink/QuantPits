"""Tests for Action Aggregator."""

import json
import pytest
from unittest.mock import patch
from quantpits.scripts.deep_analysis.action_aggregator import ActionAggregator


@pytest.fixture
def aggregator(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config").mkdir()
    # No feedback_scope.json → empty scopes
    return ActionAggregator(workspace_root=str(ws))


@pytest.fixture
def aggregator_with_scopes(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config").mkdir()
    (ws / "config" / "feedback_scope.json").write_text(json.dumps({
        "active_scopes": ["hyperparams", "model_selection"]
    }))
    return ActionAggregator(workspace_root=str(ws))


class TestAggregate:
    def test_empty_inputs(self, aggregator):
        result = aggregator.aggregate({}, {}, None)
        assert result["raw_count"] == 0
        assert result["deduped_count"] == 0
        assert result["conflicts"] == []

    def test_collects_per_model_items(self, aggregator):
        model_diags = {
            "model_a": {
                "diagnosis": "needs_tuning",
                "action_items": [{
                    "action_type": "adjust_hyperparam",
                    "scope": "hyperparams",
                    "target": "model_a",
                    "params": {"lr": {"from": 0.001, "to": 0.0005}},
                    "confidence": 0.7,
                }],
            }
        }
        result = aggregator.aggregate(model_diags, {}, None)
        assert result["raw_count"] == 1
        assert result["deduped_count"] == 1
        item = result["deduped_items"][0]
        assert item["action_type"] == "adjust_hyperparam"
        assert item["_source"] == "Per-Model(model_a)"
        assert item["_source_diagnosis"] == "needs_tuning"

    def test_collects_per_combo_items(self, aggregator):
        combo_diags = {
            "combo_x": {
                "diagnosis": "degrading",
                "action_items": [{
                    "action_type": "adjust_weights",
                    "scope": "combo_search",
                    "target": "combo_x",
                    "params": {},
                    "confidence": 0.8,
                }],
            }
        }
        result = aggregator.aggregate({}, combo_diags, None)
        assert result["raw_count"] == 1
        assert result["deduped_count"] == 1
        item = result["deduped_items"][0]
        assert item["_source"] == "Per-Combo(combo_x)"

    def test_collects_exec_risk_items(self, aggregator):
        exec_output = {
            "action_items": [{
                "action_type": "adjust_hyperparam",
                "scope": "strategy_params",
                "target": "execution",
                "params": {},
                "confidence": 0.5,
            }],
        }
        result = aggregator.aggregate({}, {}, exec_output)
        assert result["raw_count"] == 1
        assert result["deduped_count"] == 1
        assert result["deduped_items"][0]["_source"] == "Execution/Risk"

    def test_deduplicate_same_target_and_params(self, aggregator):
        model_diags = {
            "model_a": {
                "diagnosis": "needs_tuning",
                "action_items": [{
                    "action_type": "adjust_hyperparam",
                    "scope": "hyperparams",
                    "target": "model_a",
                    "params": {"lr": {"from": 0.001, "to": 0.0005}},
                    "confidence": 0.7,
                }],
            },
            "model_b": {  # duplicate from a different source for same target
                "diagnosis": "needs_tuning",
                "action_items": [{
                    "action_type": "adjust_hyperparam",
                    "scope": "hyperparams",
                    "target": "model_a",
                    "params": {"lr": {"from": 0.001, "to": 0.0005}},
                    "confidence": 0.9,
                }],
            },
        }
        result = aggregator.aggregate(model_diags, {}, None)
        assert result["raw_count"] == 2
        assert result["deduped_count"] == 1
        # Higher confidence should win
        assert result["deduped_items"][0]["confidence"] == 0.9
        assert "_dedup_note" in result["deduped_items"][0]

    def test_no_dedup_different_params(self, aggregator):
        model_diags = {
            "model_a": {
                "diagnosis": "needs_tuning",
                "action_items": [
                    {"action_type": "adjust_hyperparam", "scope": "hyperparams",
                     "target": "model_a", "params": {"lr": {"to": 0.0005}}, "confidence": 0.7},
                    {"action_type": "adjust_hyperparam", "scope": "hyperparams",
                     "target": "model_a", "params": {"dropout": {"to": 0.3}}, "confidence": 0.8},
                ],
            }
        }
        result = aggregator.aggregate(model_diags, {}, None)
        assert result["deduped_count"] == 2

    def test_scope_filtering(self, aggregator_with_scopes):
        model_diags = {
            "model_a": {
                "diagnosis": "needs_tuning",
                "action_items": [
                    {"action_type": "adjust_hyperparam", "scope": "hyperparams",
                     "target": "model_a", "params": {}, "confidence": 0.7},
                    {"action_type": "disable_model", "scope": "model_selection",
                     "target": "model_a", "params": {}, "confidence": 0.6},
                    {"action_type": "trigger_search", "scope": "combo_search",
                     "target": "model_a", "params": {}, "confidence": 0.5},
                ],
            }
        }
        result = aggregator_with_scopes.aggregate(model_diags, {}, None)
        assert result["in_scope_count"] == 2  # hyperparams + model_selection
        assert result["out_of_scope_count"] == 1  # combo_search


class TestConflicts:
    def test_disable_vs_keep_conflict(self, aggregator):
        model_diags = {
            "model_x": {
                "diagnosis": "should_disable",
                "action_items": [{
                    "action_type": "disable_model",
                    "scope": "model_selection",
                    "target": "model_x",
                    "params": {},
                    "confidence": 0.8,
                }],
            }
        }
        combo_diags = {
            "combo_y": {
                "diagnosis": "degrading",
                "member_assessments": {
                    "model_x": {
                        "per_model_diagnosis": "should_disable",
                        "loo_delta": 0.04,
                        "role": "diversifier",
                        "keep": True,
                        "reason": "Positive LOO delta",
                    }
                },
                "action_items": [],
            }
        }
        result = aggregator.aggregate(model_diags, combo_diags, None)
        conflicts = result["conflicts"]
        assert len(conflicts) >= 1
        conflict = [c for c in conflicts if c["type"] == "disable_vs_keep"][0]
        assert conflict["target"] == "model_x"
        assert conflict["combo"] == "combo_y"

    def test_no_conflict_when_both_agree_disable(self, aggregator):
        model_diags = {
            "model_x": {
                "diagnosis": "should_disable",
                "action_items": [],
            }
        }
        combo_diags = {
            "combo_y": {
                "diagnosis": "needs_member_change",
                "member_assessments": {
                    "model_x": {
                        "keep": False,
                        "role": "harmful",
                        "loo_delta": -0.02,
                    }
                },
                "action_items": [],
            }
        }
        result = aggregator.aggregate(model_diags, combo_diags, None)
        # Both agree on disable → no conflict
        assert len(result["conflicts"]) == 0

    def test_contradictory_param_conflict(self, aggregator):
        model_diags = {
            "model_a": {
                "diagnosis": "needs_tuning",
                "action_items": [{
                    "action_type": "adjust_hyperparam",
                    "scope": "hyperparams",
                    "target": "model_a",
                    "params": {"lr": {"from": 0.001, "to": 0.0005}},
                    "confidence": 0.7,
                }],
            },
            "model_b": {
                "diagnosis": "needs_tuning",
                "action_items": [{
                    "action_type": "adjust_hyperparam",
                    "scope": "hyperparams",
                    "target": "model_a",
                    "params": {"lr": {"from": 0.001, "to": 0.0015}},
                    "confidence": 0.6,
                }],
            },
        }
        result = aggregator.aggregate(model_diags, {}, None)
        # Both items target model_a + lr → deduped to one. Check dedup note exists.
        assert result["deduped_count"] == 1
        assert "_dedup_note" in result["deduped_items"][0]
        assert "2 sources" in result["deduped_items"][0]["_dedup_note"]

    def test_missing_target_uses_model_name(self, aggregator):
        model_diags = {
            "model_z": {
                "diagnosis": "needs_tuning",
                "action_items": [{
                    "action_type": "adjust_hyperparam",
                    "scope": "hyperparams",
                    "params": {"lr": {"to": 0.0005}},
                    "confidence": 0.7,
                }],
            }
        }
        result = aggregator.aggregate(model_diags, {}, None)
        assert result["deduped_items"][0]["target"] == "model_z"


class TestLoadActiveScopes:
    def test_no_config_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        agg = ActionAggregator(workspace_root=str(ws))
        assert agg._active_scopes == []

    def test_with_config(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "feedback_scope.json").write_text(json.dumps({
            "active_scopes": ["hyperparams"]
        }))
        agg = ActionAggregator(workspace_root=str(ws))
        assert agg._active_scopes == ["hyperparams"]

    def test_invalid_json(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "feedback_scope.json").write_text("not json")
        agg = ActionAggregator(workspace_root=str(ws))
        assert agg._active_scopes == []
