"""
Tests for DataSplitAdapter — modifies model_config.json for
``adjust_training_window`` ActionItems.

Covers: apply(), preview(), _load_bounds(), _check_bounds(),
_values_match(), _backup(), and adapter registration.
"""

import json
import os

import pytest

from quantpits.scripts.deep_analysis.action_items import ActionItem
from quantpits.scripts.deep_analysis.adapters import ADAPTER_REGISTRY
from quantpits.scripts.deep_analysis.adapters.data_split_adapter import (
    DataSplitAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_config(config_dir, data=None):
    """Write a minimal model_config.json."""
    if data is None:
        data = {
            "market": "csi300",
            "train_set_windows": 8,
            "valid_set_window": 2,
            "test_set_window": 3,
            "data_slice_mode": "slide",
            "freq": "week",
        }
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "model_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(path)


def _write_bounds(config_dir, bounds=None):
    """Write training_window_bounds.json."""
    if bounds is None:
        bounds = {
            "bounds": {
                "train_set_windows": {"min": 2, "max": 20},
                "valid_set_window": {"min": 1, "max": 6},
                "test_set_window": {"min": 1, "max": 8},
                "data_slice_mode": {"allowed_values": ["slide", "fixed"]},
            }
        }
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "training_window_bounds.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bounds, f)
    return str(path)


@pytest.fixture
def adapter_workspace(tmp_path):
    """Create a workspace with config/model_config.json and bounds."""
    config_dir = tmp_path / "config"
    _write_config(config_dir)
    _write_bounds(config_dir)
    return str(tmp_path)


def _make_item(params, action_type="adjust_training_window"):
    return ActionItem(
        action_type=action_type,
        scope="hyperparams",
        target="global",
        params=params,
    )


# ---------------------------------------------------------------------------
# apply() — success paths
# ---------------------------------------------------------------------------

class TestApplySuccess:
    def test_single_change(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 12}})
        result = adapter.apply(item)
        assert result.success is True
        assert len(result.changes) == 1
        assert result.changes[0]["param"] == "train_set_windows"
        assert result.changes[0]["old"] == 8
        assert result.changes[0]["new"] == 12
        assert result.modified_files[0].endswith("model_config.json")
        # Verify on-disk change
        with open(result.modified_files[0], "r") as f:
            config = json.load(f)
        assert config["train_set_windows"] == 12

    def test_multiple_changes(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({
            "train_set_windows": {"from": 8, "to": 10},
            "valid_set_window": {"from": 2, "to": 3},
        })
        result = adapter.apply(item)
        assert result.success is True
        assert len(result.changes) == 2
        with open(result.modified_files[0], "r") as f:
            config = json.load(f)
        assert config["train_set_windows"] == 10
        assert config["valid_set_window"] == 3

    def test_backup_created(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 12}})
        result = adapter.apply(item)
        assert result.success is True
        backup_dir = os.path.join(adapter_workspace, "config", "_backup")
        assert os.path.isdir(backup_dir)
        backups = os.listdir(backup_dir)
        assert len(backups) == 1
        assert "model_config.json" in backups[0]

    def test_atomic_write_no_tmp_leftover(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 14}})
        adapter.apply(item)
        tmp_path = os.path.join(adapter_workspace, "config", "model_config.json.tmp")
        assert not os.path.exists(tmp_path)

    def test_from_none_passes(self, adapter_workspace):
        """When from is None, the from-match check is skipped."""
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"valid_set_window": {"to": 4}})
        # No "from" key → expected_from is None → values_match not called
        result = adapter.apply(item)
        assert result.success is True
        with open(result.modified_files[0], "r") as f:
            config = json.load(f)
        assert config["valid_set_window"] == 4


# ---------------------------------------------------------------------------
# apply() — rejection paths
# ---------------------------------------------------------------------------

class TestApplyRejection:
    def test_from_mismatch(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 10, "to": 12}})
        result = adapter.apply(item)
        assert result.success is False
        assert "does not match" in result.error

    def test_bounds_min_violation(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 1}})
        result = adapter.apply(item)
        assert result.success is False
        assert "below minimum" in result.error

    def test_bounds_max_violation(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 25}})
        result = adapter.apply(item)
        assert result.success is False
        assert "above maximum" in result.error

    def test_allowed_values_violation(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"data_slice_mode": {"from": "slide", "to": "rotate"}})
        result = adapter.apply(item)
        assert result.success is False
        assert "not in allowed" in result.error

    def test_disallowed_field(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"market": {"from": "csi300", "to": "csi500"}})
        result = adapter.apply(item)
        assert result.success is False
        assert "not an allowed" in result.error

    def test_missing_config_file(self, tmp_path):
        ws = str(tmp_path)
        adapter = DataSplitAdapter(ws)
        item = _make_item({"train_set_windows": {"from": 8, "to": 12}})
        result = adapter.apply(item)
        assert result.success is False
        assert "not found" in result.error

    def test_empty_changes(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({})
        result = adapter.apply(item)
        assert result.success is False
        assert "No valid parameter changes" in result.error

    def test_malformed_json(self, adapter_workspace):
        config_path = os.path.join(adapter_workspace, "config", "model_config.json")
        with open(config_path, "w") as f:
            f.write("not valid json {{{")
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 12}})
        result = adapter.apply(item)
        assert result.success is False

    def test_non_dict_change_spec_skipped_then_empty(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": 10})
        result = adapter.apply(item)
        assert result.success is False
        assert "No valid parameter changes" in result.error


# ---------------------------------------------------------------------------
# preview()
# ---------------------------------------------------------------------------

class TestPreview:
    def test_preview_success(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 12}})
        result = adapter.preview(item)
        assert result["config_file"].endswith("model_config.json")
        assert len(result["planned_changes"]) == 1
        entry = result["planned_changes"][0]
        assert entry["param"] == "train_set_windows"
        assert entry["current"] == 8
        assert entry["to"] == 12
        assert entry["from_match"] is True
        assert "bounds_error" not in entry

    def test_preview_from_mismatch(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 10, "to": 12}})
        result = adapter.preview(item)
        assert result["planned_changes"][0]["from_match"] is False

    def test_preview_missing_config(self, tmp_path):
        adapter = DataSplitAdapter(str(tmp_path))
        item = _make_item({"train_set_windows": {"from": 8, "to": 12}})
        result = adapter.preview(item)
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]

    def test_preview_malformed_json(self, adapter_workspace):
        config_path = os.path.join(adapter_workspace, "config", "model_config.json")
        with open(config_path, "w") as f:
            f.write("not json {{{")
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 12}})
        result = adapter.preview(item)
        assert len(result["errors"]) == 1
        assert "Failed to parse" in result["errors"][0]

    def test_preview_disallowed_field(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"market": {"from": "csi300", "to": "csi500"}})
        result = adapter.preview(item)
        assert result["planned_changes"][0]["disallowed_field"] is True

    def test_preview_bounds_error(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"train_set_windows": {"from": 8, "to": 30}})
        result = adapter.preview(item)
        assert "bounds_error" in result["planned_changes"][0]
        assert "above maximum" in result["planned_changes"][0]["bounds_error"]

    def test_preview_from_none(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        item = _make_item({"valid_set_window": {"to": 4}})
        result = adapter.preview(item)
        assert result["planned_changes"][0]["from_match"] is True


# ---------------------------------------------------------------------------
# _values_match()
# ---------------------------------------------------------------------------

class TestValuesMatch:
    def test_int_match(self):
        assert DataSplitAdapter._values_match(8, 8) is True

    def test_int_mismatch(self):
        assert DataSplitAdapter._values_match(8, 9) is False

    def test_float_match_tolerant(self):
        assert DataSplitAdapter._values_match(8.0, 8) is True
        assert DataSplitAdapter._values_match(8, 8.0) is True
        assert DataSplitAdapter._values_match(8.0000000001, 8) is True

    def test_float_mismatch(self):
        assert DataSplitAdapter._values_match(8.1, 8) is False

    def test_string_match(self):
        assert DataSplitAdapter._values_match("slide", "slide") is True

    def test_string_mismatch(self):
        assert DataSplitAdapter._values_match("slide", "fixed") is False

    def test_both_none(self):
        assert DataSplitAdapter._values_match(None, None) is True

    def test_one_none(self):
        assert DataSplitAdapter._values_match(None, 8) is False
        assert DataSplitAdapter._values_match(8, None) is False


# ---------------------------------------------------------------------------
# _load_bounds()
# ---------------------------------------------------------------------------

class TestLoadBounds:
    def test_loads_and_caches(self, adapter_workspace):
        adapter = DataSplitAdapter(adapter_workspace)
        b1 = adapter._load_bounds()
        assert "train_set_windows" in b1
        assert b1["train_set_windows"]["min"] == 2
        # Second call returns cached
        b2 = adapter._load_bounds()
        assert b1 is b2

    def test_missing_file_returns_empty(self, tmp_path):
        adapter = DataSplitAdapter(str(tmp_path))
        assert adapter._load_bounds() == {}

    def test_corrupt_json_returns_empty(self, adapter_workspace):
        bounds_path = os.path.join(adapter_workspace, "config", "training_window_bounds.json")
        with open(bounds_path, "w") as f:
            f.write("not json {{{")
        adapter = DataSplitAdapter(adapter_workspace)
        assert adapter._load_bounds() == {}

    def test_no_bounds_key_returns_empty(self, adapter_workspace):
        bounds_path = os.path.join(adapter_workspace, "config", "training_window_bounds.json")
        with open(bounds_path, "w") as f:
            json.dump({"other": "data"}, f)
        adapter = DataSplitAdapter(adapter_workspace)
        assert adapter._load_bounds() == {}


# ---------------------------------------------------------------------------
# _check_bounds()
# ---------------------------------------------------------------------------

class TestCheckBounds:
    @pytest.fixture
    def adapter(self, adapter_workspace):
        return DataSplitAdapter(adapter_workspace)

    def test_unknown_param_returns_none(self, adapter):
        assert adapter._check_bounds("unknown_param", 5) is None

    def test_within_bounds(self, adapter):
        assert adapter._check_bounds("train_set_windows", 10) is None

    def test_below_min(self, adapter):
        err = adapter._check_bounds("train_set_windows", 1)
        assert err is not None
        assert "below minimum" in err

    def test_above_max(self, adapter):
        err = adapter._check_bounds("train_set_windows", 30)
        assert err is not None
        assert "above maximum" in err

    def test_allowed_values_pass(self, adapter):
        assert adapter._check_bounds("data_slice_mode", "slide") is None

    def test_allowed_values_fail(self, adapter):
        err = adapter._check_bounds("data_slice_mode", "rotate")
        assert err is not None
        assert "not in allowed" in err

    def test_none_value_skips_numeric_check(self, adapter):
        assert adapter._check_bounds("train_set_windows", None) is None


# ---------------------------------------------------------------------------
# Adapter registration
# ---------------------------------------------------------------------------

def test_adapter_registered():
    assert "adjust_training_window" in ADAPTER_REGISTRY
    # The registered class may be wrapped but should be DataSplitAdapter
    reg = ADAPTER_REGISTRY["adjust_training_window"]
    assert reg is DataSplitAdapter or reg.__name__ == "DataSplitAdapter"
