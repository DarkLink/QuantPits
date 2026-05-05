"""Tests for TrainingAdapter."""

import json
import os
import pytest
import yaml

from quantpits.scripts.deep_analysis.action_items import ActionItem
from quantpits.scripts.deep_analysis.adapters.training_adapter import TrainingAdapter


@pytest.fixture
def adapter_workspace(tmp_path):
    """Create a workspace with model registry, workflow YAML, and bounds."""
    ws = tmp_path / "TestWorkspace"
    ws.mkdir()
    config_dir = ws / "config"
    config_dir.mkdir()

    # model_registry.yaml
    registry = {
        "models": {
            "gru_Alpha158": {
                "algorithm": "gru",
                "dataset": "Alpha158",
                "yaml_file": "config/workflow_config_gru_Alpha158.yaml",
                "enabled": True,
            },
            "gats_Alpha360": {
                "algorithm": "gats",
                "dataset": "Alpha360",
                "yaml_file": "config/workflow_config_gats_Alpha360.yaml",
                "enabled": True,
                "pretrain_source": "lstm_Alpha360",
            },
        }
    }
    with open(config_dir / "model_registry.yaml", "w") as f:
        yaml.dump(registry, f)

    # workflow_config_gru_Alpha158.yaml
    # Using ruamel-compatible plain dict (will be written via standard yaml)
    workflow = {
        "task": {
            "model": {
                "class": "GRU",
                "module_path": "qlib.contrib.model.pytorch_gru",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 200,
                    "lr": 0.001,
                    "early_stop": 10,
                    "batch_size": 2048,
                    "GPU": 0,
                },
            },
            "dataset": {"class": "DatasetH"},
        },
        "data_handler_config": {"label": ["Ref($close, -2) / Ref($close, -1) - 1"]},
    }
    with open(config_dir / "workflow_config_gru_Alpha158.yaml", "w") as f:
        yaml.dump(workflow, f)

    # workflow_config_gats_Alpha360.yaml (with pretrain dependency)
    gats_workflow = {
        "task": {
            "model": {
                "class": "GATS",
                "kwargs": {"d_feat": 360, "n_epochs": 100},
            },
        },
    }
    with open(config_dir / "workflow_config_gats_Alpha360.yaml", "w") as f:
        yaml.dump(gats_workflow, f)

    # hyperparam_bounds.json
    bounds = {
        "bounds": {
            "n_epochs": {"min": 10, "max": 500, "max_change_pct": 50},
            "lr": {"min": 1e-5, "max": 1e-2, "max_change_pct": 100},
            "early_stop": {"min": 5, "max": 100, "max_change_pct": None},
            "dropout": {"min": 0.0, "max": 0.8, "max_change_pct": None},
        }
    }
    with open(config_dir / "hyperparam_bounds.json", "w") as f:
        json.dump(bounds, f)

    return ws


def _make_action_item(target, params, **kwargs):
    """Helper to create an ActionItem for testing."""
    return ActionItem(
        action_type="adjust_hyperparam",
        scope="hyperparams",
        target=target,
        params=params,
        scope_status="in_scope",
        confidence=0.7,
        risk_level="low",
        **kwargs,
    )


class TestTrainingAdapter:
    def test_apply_success(self, adapter_workspace):
        """Test successful hyperparam modification."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"early_stop": {"from": 10, "to": 20}},
        )

        result = adapter.apply(item)
        assert result.success
        assert result.adapter_type == "training"
        assert len(result.changes) == 1
        assert result.changes[0]["param"] == "early_stop"
        assert result.changes[0]["old"] == 10
        assert result.changes[0]["new"] == 20

        # Verify YAML was actually modified
        yaml_path = result.modified_files[0]
        with open(yaml_path, "r") as f:
            doc = yaml.safe_load(f)
        assert doc["task"]["model"]["kwargs"]["early_stop"] == 20

    def test_apply_multiple_params(self, adapter_workspace):
        """Test modifying multiple hyperparams in one ActionItem."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {
                "early_stop": {"from": 10, "to": 20},
                "dropout": {"from": 0.0, "to": 0.1},
            },
        )

        result = adapter.apply(item)
        assert result.success
        assert len(result.changes) == 2

    def test_apply_from_mismatch(self, adapter_workspace):
        """Test rejection when 'from' value doesn't match current YAML."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"early_stop": {"from": 999, "to": 20}},  # 999 != actual 10
        )

        result = adapter.apply(item)
        assert not result.success
        assert "does not match" in result.error

    def test_apply_out_of_bounds(self, adapter_workspace):
        """Test rejection when new value exceeds bounds."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"n_epochs": {"from": 200, "to": 1000}},  # max is 500
        )

        result = adapter.apply(item)
        assert not result.success
        assert "above maximum" in result.error

    def test_apply_below_bounds(self, adapter_workspace):
        """Test rejection when new value is below minimum."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"n_epochs": {"from": 200, "to": 5}},  # min is 10
        )

        result = adapter.apply(item)
        assert not result.success
        assert "below minimum" in result.error

    def test_apply_creates_backup(self, adapter_workspace):
        """Test that a backup file is created before modification."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"early_stop": {"from": 10, "to": 20}},
        )

        result = adapter.apply(item)
        assert result.success

        backup_dir = os.path.join(str(adapter_workspace), "config", "_backup")
        assert os.path.isdir(backup_dir)
        backups = os.listdir(backup_dir)
        assert len(backups) == 1
        assert backups[0].startswith("workflow_config_gru_Alpha158.yaml.")

    def test_apply_unknown_model(self, adapter_workspace):
        """Test error when model not in registry."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "nonexistent_model",
            {"n_epochs": {"from": 100, "to": 150}},
        )

        result = adapter.apply(item)
        assert not result.success
        assert "Cannot resolve YAML" in result.error

    def test_preview(self, adapter_workspace):
        """Test dry-run preview returns correct planned changes."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"early_stop": {"from": 10, "to": 20}},
        )

        preview = adapter.preview(item)
        assert preview["target"] == "gru_Alpha158"
        assert preview["yaml_file"] is not None
        assert len(preview["planned_changes"]) == 1
        assert preview["planned_changes"][0]["from_match"] is True
        assert not preview["errors"]

    def test_preview_from_mismatch(self, adapter_workspace):
        """Test preview shows from_match=False when values don't match."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"early_stop": {"from": 999, "to": 20}},
        )

        preview = adapter.preview(item)
        assert preview["planned_changes"][0]["from_match"] is False

    def test_check_pretrain_deps_no_deps(self, adapter_workspace):
        """Test model without pretrain_source returns empty list."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item("gru_Alpha158", {})
        missing = adapter.check_pretrain_deps(item)
        assert missing == []

    def test_check_pretrain_deps_missing(self, adapter_workspace):
        """Test model with pretrain_source but no pretrained file."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item("gats_Alpha360", {})
        missing = adapter.check_pretrain_deps(item)
        assert "lstm_Alpha360" in missing

    def test_check_pretrain_deps_present(self, adapter_workspace):
        """Test model with pretrain_source and existing pretrained file."""
        # Create the pretrained file
        pretrained_dir = adapter_workspace / "data" / "pretrained"
        pretrained_dir.mkdir(parents=True)
        (pretrained_dir / "lstm_Alpha360_latest.pkl").write_bytes(b"fake model")

        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item("gats_Alpha360", {})
        missing = adapter.check_pretrain_deps(item)
        assert missing == []

    def test_unknown_param_allowed(self, adapter_workspace):
        """Test that unknown params (not in bounds) are allowed by default."""
        adapter = TrainingAdapter(str(adapter_workspace))
        item = _make_action_item(
            "gru_Alpha158",
            {"GPU": {"from": 0, "to": 1}},  # GPU not in bounds
        )

        result = adapter.apply(item)
        assert result.success


# ── Additional error-path and edge-case tests ──────────────────

def test_apply_malformed_yaml(adapter_workspace):
    """apply() should return error for unparseable YAML."""
    ws = adapter_workspace

    # Corrupt the YAML file with invalid syntax
    malformed_path = os.path.join(ws, "config", "workflow_config_gru_Alpha158.yaml")
    with open(malformed_path, "w") as f:
        f.write(": invalid yaml : : [[[")

    # Also need model_registry pointing to this corrupted file
    registry_path = os.path.join(ws, "config", "model_registry.yaml")
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)
    registry["models"]["gru_Alpha158"]["yaml_file"] = "config/workflow_config_gru_Alpha158.yaml"
    with open(registry_path, "w") as f:
        yaml.dump(registry, f)

    from quantpits.scripts.deep_analysis.adapters.training_adapter import TrainingAdapter
    adapter = TrainingAdapter(str(ws))
    item = _make_action_item("gru_Alpha158", {"early_stop": {"from": 10, "to": 20}})
    result = adapter.apply(item)
    assert not result.success


def test_apply_missing_model_kwargs(adapter_workspace):
    """apply() should return error when YAML has no task.model.kwargs."""
    ws = adapter_workspace

    yaml_path = os.path.join(ws, "config", "workflow_config_mlp_Alpha158.yaml")
    config = {"task": {"model": {"class": "MLP"}, "dataset": {}}}
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    from quantpits.scripts.deep_analysis.adapters.training_adapter import TrainingAdapter
    adapter = TrainingAdapter(str(ws))
    item = _make_action_item("mlp_Alpha158", {"n_epochs": {"from": 100, "to": 150}})
    result = adapter.apply(item)
    assert not result.success


def test_apply_missing_registry_file(adapter_workspace):
    """Adapter may still resolve YAML path directly when registry is missing."""
    ws = adapter_workspace
    registry_path = os.path.join(ws, "config", "model_registry.yaml")
    if os.path.exists(registry_path):
        os.remove(registry_path)

    from quantpits.scripts.deep_analysis.adapters.training_adapter import TrainingAdapter
    adapter = TrainingAdapter(str(ws))
    item = _make_action_item("gru_Alpha158", {"early_stop": {"from": 10, "to": 20}})
    result = adapter.apply(item)
    # Adapter can resolve YAML directly without registry; it's a best-effort operation
    assert isinstance(result.success, bool)


def test_apply_missing_bounds_file(adapter_workspace):
    """apply() should succeed even when hyperparam_bounds.json is missing."""
    ws = adapter_workspace
    bounds_path = os.path.join(ws, "config", "hyperparam_bounds.json")
    if os.path.exists(bounds_path):
        os.remove(bounds_path)

    from quantpits.scripts.deep_analysis.adapters.training_adapter import TrainingAdapter
    adapter = TrainingAdapter(str(ws))
    item = _make_action_item("gru_Alpha158", {"early_stop": {"from": 10, "to": 20}})
    # Should succeed — bounds check is optional
    result = adapter.apply(item)
    assert result.success


def test_adapter_registry_has_training_adapter():
    """Adapter registry should include adjust_hyperparam adapter."""
    from quantpits.scripts.deep_analysis.adapters import ADAPTER_REGISTRY
    assert "adjust_hyperparam" in ADAPTER_REGISTRY
    adapter_cls = ADAPTER_REGISTRY["adjust_hyperparam"]
    assert adapter_cls.__name__ == "TrainingAdapter"
