"""
Tests for ModelSelectionAdapter — enable/disable models in model_registry.yaml.

Covers: apply(), preview(), _check_loo_delta(), _check_combo_membership(), _backup().
"""

import json
import os

import pytest
from ruamel.yaml import YAML

from quantpits.scripts.deep_analysis.action_items import ActionItem
from quantpits.scripts.deep_analysis.adapters.model_selection_adapter import (
    ModelSelectionAdapter,
)

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(tmp_path, models=None):
    """Write a minimal model_registry.yaml and return its path."""
    if models is None:
        models = {
            "m1": {"algorithm": "gru", "dataset": "Alpha158", "enabled": True, "yaml_file": "gru.yaml"},
            "m2": {"algorithm": "mlp", "dataset": "Alpha158", "enabled": True, "yaml_file": "mlp.yaml"},
            "m3": {"algorithm": "lgb", "dataset": "Alpha360", "enabled": False, "yaml_file": "lgb.yaml"},
        }
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "model_registry.yaml"
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump({"models": models}, f)
    return str(path)


def _make_action(action_type, target, params=None):
    return ActionItem(action_type=action_type, target=target, params=params or {})


# ---------------------------------------------------------------------------
# _check_loo_delta
# ---------------------------------------------------------------------------

class TestCheckLooDelta:
    def test_positive_delta_blocks(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", {"loo_delta": 0.05})
        err = adapter._check_loo_delta(item)
        assert err is not None
        assert "positive" in err
        assert "diversifier" in err

    def test_zero_delta_passes(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", {"loo_delta": 0.0})
        assert adapter._check_loo_delta(item) is None

    def test_negative_delta_passes(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", {"loo_delta": -0.03})
        assert adapter._check_loo_delta(item) is None

    def test_no_params_defaults_to_zero(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", None)
        assert adapter._check_loo_delta(item) is None

    def test_none_delta_passes(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", {"loo_delta": None})
        assert adapter._check_loo_delta(item) is None


# ---------------------------------------------------------------------------
# _check_combo_membership
# ---------------------------------------------------------------------------

class TestCheckComboMembership:
    def test_no_ensemble_file(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        assert adapter._check_combo_membership("m1") == []

    def test_invalid_json_returns_empty(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "ensemble_config.json").write_text("NOT JSON")
        adapter = ModelSelectionAdapter(str(tmp_path))
        assert adapter._check_combo_membership("m1") == []

    def test_model_not_in_any_combo(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_a": {"models": ["m2@static", "m3@static"], "default": False},
            }
        }))
        adapter = ModelSelectionAdapter(str(tmp_path))
        assert adapter._check_combo_membership("m1") == []

    def test_model_in_combo(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_a": {"models": ["m1@static", "m2@static"], "default": False},
            }
        }))
        adapter = ModelSelectionAdapter(str(tmp_path))
        warnings = adapter._check_combo_membership("m1")
        assert len(warnings) == 1
        assert "combo_a" in warnings[0]
        assert "replace_member" in warnings[0]

    def test_model_in_default_combo(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "default_combo": {"models": ["m1@static"], "default": True},
            }
        }))
        adapter = ModelSelectionAdapter(str(tmp_path))
        warnings = adapter._check_combo_membership("m1")
        assert len(warnings) == 1
        assert "(default)" in warnings[0]

    def test_model_without_at_suffix_in_combo(self, tmp_path):
        """Member listed without @suffix — still matches."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_a": {"models": ["m1"], "default": False},
            }
        }))
        adapter = ModelSelectionAdapter(str(tmp_path))
        warnings = adapter._check_combo_membership("m1")
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# _backup
# ---------------------------------------------------------------------------

class TestBackup:
    def test_creates_backup(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        adapter._backup()

        backup_dir = tmp_path / "config" / "_backup"
        assert backup_dir.exists()
        backups = list(backup_dir.iterdir())
        assert len(backups) == 1
        assert "model_registry.yaml." in backups[0].name


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

class TestApply:
    def test_missing_registry_file(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1")
        result = adapter.apply(item)
        assert result.success is False
        assert "not found" in result.error

    def test_model_not_in_registry(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "nonexistent")
        result = adapter.apply(item)
        assert result.success is False
        assert "not found" in result.error

    def test_noop_already_enabled(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("enable_model", "m1")  # m1 already enabled
        result = adapter.apply(item)
        assert result.success is True
        assert "already enabled" in result.error

    def test_noop_already_disabled(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m3")  # m3 already disabled
        result = adapter.apply(item)
        assert result.success is True
        assert "already disabled" in result.error

    def test_disable_blocked_by_loo(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", {"loo_delta": 0.08})
        result = adapter.apply(item)
        assert result.success is False
        assert "LOO delta" in result.error
        assert "positive" in result.error

    def test_successful_enable(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("enable_model", "m3")  # m3 is disabled
        result = adapter.apply(item)
        assert result.success is True
        assert len(result.modified_files) == 1
        assert result.changes[0]["param"] == "enabled"
        assert result.changes[0]["old"] is False
        assert result.changes[0]["new"] is True

        # Verify the file was actually written
        with open(adapter._registry_path, "r") as f:
            doc = _yaml.load(f)
        assert doc["models"]["m3"]["enabled"] is True

    def test_successful_disable(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m2", {"loo_delta": -0.01})
        result = adapter.apply(item)
        assert result.success is True
        assert result.changes[0]["old"] is True
        assert result.changes[0]["new"] is False

        with open(adapter._registry_path, "r") as f:
            doc = _yaml.load(f)
        assert doc["models"]["m2"]["enabled"] is False

    def test_disable_with_combo_warning(self, tmp_path):
        _make_registry(tmp_path)
        config_dir = tmp_path / "config"
        (config_dir / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_a": {"models": ["m1@static"], "default": True},
            }
        }))
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", {"loo_delta": -0.01})
        result = adapter.apply(item)
        assert result.success is True
        assert "combo_a" in result.error  # warnings stored in error field
        assert "default" in result.error

    def test_exception_during_apply(self, tmp_path):
        """Trigger the broad except block by making the registry unreadable after initial load."""
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        # Make the file a directory so the write fails
        os.remove(adapter._registry_path)
        os.makedirs(adapter._registry_path)  # now a directory, write will fail
        item = _make_action("disable_model", "m1", {"loo_delta": -0.01})
        result = adapter.apply(item)
        assert result.success is False
        # Error from the broad except block
        assert result.error


# ---------------------------------------------------------------------------
# preview()
# ---------------------------------------------------------------------------

class TestPreview:
    def test_missing_registry_file(self, tmp_path):
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1")
        result = adapter.preview(item)
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]
        assert result["planned_change"] is None

    def test_invalid_yaml(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "model_registry.yaml").write_text(": bad yaml : :")
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1")
        result = adapter.preview(item)
        assert len(result["errors"]) == 1
        assert "parse" in result["errors"][0].lower()

    def test_model_not_in_registry(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "ghost")
        result = adapter.preview(item)
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]

    def test_noop_already_in_state(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("enable_model", "m1")  # already enabled
        result = adapter.preview(item)
        assert len(result["errors"]) == 1
        assert "already enabled" in result["errors"][0]

    def test_successful_preview_enable(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("enable_model", "m3")
        result = adapter.preview(item)
        assert result["planned_change"] is not None
        assert result["planned_change"]["param"] == "enabled"
        assert result["planned_change"]["current"] is False
        assert result["planned_change"]["new"] is True
        assert result["errors"] == []

    def test_preview_disable_with_loo_block(self, tmp_path):
        _make_registry(tmp_path)
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m1", {"loo_delta": 0.05})
        result = adapter.preview(item)
        assert len(result["errors"]) == 1
        assert "LOO delta" in result["errors"][0]

    def test_preview_disable_with_combo_warnings(self, tmp_path):
        _make_registry(tmp_path)
        config_dir = tmp_path / "config"
        (config_dir / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_x": {"models": ["m2@static"], "default": False},
            }
        }))
        adapter = ModelSelectionAdapter(str(tmp_path))
        item = _make_action("disable_model", "m2", {"loo_delta": 0.0})
        result = adapter.preview(item)
        assert len(result["warnings"]) == 1
        assert "combo_x" in result["warnings"][0]
        # Errors should be empty because LOO passes
        assert result["errors"] == []
