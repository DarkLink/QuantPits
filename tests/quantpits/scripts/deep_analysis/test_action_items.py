"""Tests for ActionItem data model and Validator (Phase 3)."""

import json
import os
import pytest

from quantpits.scripts.deep_analysis.action_items import (
    ActionItem,
    ActionItemValidator,
    persist_action_items,
)


# ------------------------------------------------------------------
# ActionItem dataclass
# ------------------------------------------------------------------

class TestActionItem:
    def test_auto_uuid(self):
        item = ActionItem(action_type="adjust_hyperparam", target="gru_Alpha158")
        assert item.action_id  # non-empty
        assert len(item.action_id) == 36  # UUID format

    def test_from_dict(self):
        d = {
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "gru_Alpha158",
            "params": {"n_epochs": {"from": 100, "to": 150}},
            "reason": "underfitting",
            "source_signals": ["underfitting"],
            "confidence": 0.8,
            "risk_level": "low",
            "unknown_field": "ignored",
        }
        item = ActionItem.from_dict(d)
        assert item.action_type == "adjust_hyperparam"
        assert item.target == "gru_Alpha158"
        assert item.confidence == 0.8
        assert item.action_id  # auto-generated

    def test_to_dict_roundtrip(self):
        item = ActionItem(
            action_type="disable_model",
            scope="model_selection",
            target="gats_Alpha158",
            reason="negative contribution",
        )
        d = item.to_dict()
        assert d["action_type"] == "disable_model"
        assert d["scope"] == "model_selection"
        assert "action_id" in d

    def test_execution_context_default(self):
        item = ActionItem()
        assert item.execution_context["target_env"] == "playground"
        assert item.execution_context["requires_retrain"] is True

    def test_from_dict_preserves_action_id(self):
        d = {"action_id": "fixed-id-123", "action_type": "test"}
        item = ActionItem.from_dict(d)
        assert item.action_id == "fixed-id-123"


# ------------------------------------------------------------------
# ActionItemValidator
# ------------------------------------------------------------------

class TestActionItemValidator:
    @pytest.fixture
    def validator_workspace(self, tmp_path):
        """Create a workspace with feedback_scope and hyperparam_bounds."""
        ws = tmp_path / "ws"
        ws.mkdir()
        config_dir = ws / "config"
        config_dir.mkdir()

        # feedback_scope.json
        with open(config_dir / "feedback_scope.json", "w") as f:
            json.dump({
                "active_scopes": ["hyperparams"],
                "available_scopes": {
                    "hyperparams": {"enabled": True},
                    "model_selection": {"enabled": False},
                },
            }, f)

        # hyperparam_bounds.json
        with open(config_dir / "hyperparam_bounds.json", "w") as f:
            json.dump({
                "bounds": {
                    "n_epochs": {"min": 10, "max": 500, "max_change_pct": 50},
                    "lr": {"min": 1e-5, "max": 1e-2, "max_change_pct": 100},
                    "dropout": {"min": 0.0, "max": 0.8, "max_change_pct": None},
                }
            }, f)

        return str(ws)

    def _make_validator(self, ws):
        return ActionItemValidator(
            feedback_scope_path=os.path.join(ws, "config", "feedback_scope.json"),
            hyperparam_bounds_path=os.path.join(ws, "config", "hyperparam_bounds.json"),
            workspace_root=ws,
        )

    def test_in_scope_valid_change(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="adjust_hyperparam",
            scope="hyperparams",
            target="gru_Alpha158",
            params={"n_epochs": {"from": 100, "to": 140}},
        )
        result = v.validate([item])
        assert result[0].scope_status == "in_scope"
        assert result[0].validated_at  # non-empty

    def test_out_of_scope(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="disable_model",
            scope="model_selection",
            target="gats_Alpha158",
        )
        result = v.validate([item])
        assert result[0].scope_status == "out_of_scope"
        assert "model_selection" in result[0].rejected_reason

    def test_rejected_below_min(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="adjust_hyperparam",
            scope="hyperparams",
            params={"n_epochs": {"from": 100, "to": 5}},
        )
        result = v.validate([item])
        assert result[0].scope_status == "rejected"
        assert "below minimum" in result[0].rejected_reason

    def test_rejected_above_max(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="adjust_hyperparam",
            scope="hyperparams",
            params={"n_epochs": {"from": 100, "to": 600}},
        )
        result = v.validate([item])
        assert result[0].scope_status == "rejected"
        assert "above maximum" in result[0].rejected_reason

    def test_rejected_change_pct_exceeded(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="adjust_hyperparam",
            scope="hyperparams",
            params={"n_epochs": {"from": 100, "to": 200}},  # 100% change, limit is 50%
        )
        result = v.validate([item])
        assert result[0].scope_status == "rejected"
        assert "max_change_pct" in result[0].rejected_reason

    def test_null_max_change_pct_allows_any_change(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="adjust_hyperparam",
            scope="hyperparams",
            params={"dropout": {"from": 0.1, "to": 0.7}},  # 600% change, but null limit
        )
        result = v.validate([item])
        assert result[0].scope_status == "in_scope"

    def test_unknown_param_passes_with_warning(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="adjust_hyperparam",
            scope="hyperparams",
            params={"unknown_param": {"from": 1, "to": 2}},
        )
        result = v.validate([item])
        assert result[0].scope_status == "in_scope"

    def test_non_adjust_action_in_scope(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        item = ActionItem(
            action_type="retrain",
            scope="hyperparams",
            target="gru_Alpha158",
        )
        result = v.validate([item])
        assert result[0].scope_status == "in_scope"

    def test_missing_config_files(self, tmp_path):
        ws = str(tmp_path / "empty_ws")
        os.makedirs(ws, exist_ok=True)
        v = ActionItemValidator(
            feedback_scope_path=os.path.join(ws, "config", "feedback_scope.json"),
            hyperparam_bounds_path=os.path.join(ws, "config", "hyperparam_bounds.json"),
            workspace_root=ws,
        )
        item = ActionItem(scope="hyperparams")
        result = v.validate([item])
        # No active scopes → out_of_scope
        assert result[0].scope_status == "out_of_scope"

    def test_multiple_items_mixed(self, validator_workspace):
        v = self._make_validator(validator_workspace)
        items = [
            ActionItem(action_type="adjust_hyperparam", scope="hyperparams",
                       params={"n_epochs": {"from": 100, "to": 130}}),
            ActionItem(action_type="disable_model", scope="model_selection"),
            ActionItem(action_type="adjust_hyperparam", scope="hyperparams",
                       params={"n_epochs": {"from": 100, "to": 5}}),
        ]
        results = v.validate(items)
        assert results[0].scope_status == "in_scope"
        assert results[1].scope_status == "out_of_scope"
        assert results[2].scope_status == "rejected"


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

class TestPersistence:
    def test_persist_action_items(self, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(os.path.join(ws, "data"), exist_ok=True)
        os.makedirs(os.path.join(ws, "output", "deep_analysis"), exist_ok=True)

        items = [
            ActionItem(
                action_type="adjust_hyperparam",
                scope="hyperparams",
                target="gru_Alpha158",
                params={"n_epochs": {"from": 100, "to": 150}},
                scope_status="in_scope",
            ),
        ]

        path = persist_action_items(items, ws, run_date="2026-04-30")

        # Snapshot file
        assert os.path.exists(path)
        with open(path, "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["target"] == "gru_Alpha158"

        # History file
        history_path = os.path.join(ws, "data", "action_item_history.jsonl")
        assert os.path.exists(history_path)
        with open(history_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["_run_date"] == "2026-04-30"

    def test_persist_creates_dirs(self, tmp_path):
        ws = str(tmp_path / "new_ws")
        items = [ActionItem(action_type="test")]
        path = persist_action_items(items, ws, run_date="2026-01-01")
        assert os.path.exists(path)
