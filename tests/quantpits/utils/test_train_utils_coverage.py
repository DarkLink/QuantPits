"""
Supplemental unit tests targeting uncovered branches in train_utils.py.

Coverage targets (line numbers refer to train_utils.py):
- make_model_key: already-has-@ branch (line 72), mode=None (line 69)
- parse_model_key: no-@ branch (line 88)
- get_experiment_name_for_model: fallback to global experiment_name (lines 104-112)
- load_pretrained_metadata: file not found (line 387)
- resolve_pretrained_path: empty pretrain_source (line 422)
- BestScoreCaptureHandler.emit: exception path (lines 665-666)
- inject_config: freq=day label (line 504), no_pretrain (lines 559-560)
- train_single_model: TypeError in fit (lines 736-738), evals_result GBDT path (lines 760-795),
  LightGBM/CatBoost fallbacks (lines 812-845), linear model (lines 848-851),
  convergence exception (lines 858-859), portfolio analysis (lines 924-936),
  top-level exception (lines 969-973)
- merge_train_records: model already present with same rid (lines 1051-1052)
- overwrite_train_records: default record_file (line 1076)
- merge_performance_file: new file (line 1096)
- predict_single_model: yaml missing (lines 1361-1363), model not in source (lines 1370-1372),
  fallback chain (lines 1400-1407), max depth (line 1410),
  IC error (lines 1465-1467), top-level exception (lines 1476-1480)
- migrate_legacy_records: already migrated (line 1504), dry_run (lines 1572-1573),
  default workspace_dir (lines 1520-1521), no rolling_exp (line 1557)
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open, PropertyMock

# ---------------------------------------------------------------------------
# make_model_key / parse_model_key
# ---------------------------------------------------------------------------


def test_make_model_key_already_has_separator(mock_env_constants):
    """Line 72: model_name already contains @ → returned as-is."""
    train_utils, _ = mock_env_constants
    assert train_utils.make_model_key("m@rolling") == "m@rolling"


def test_make_model_key_mode_none_uses_default(mock_env_constants):
    """Line 69: mode=None triggers DEFAULT_TRAINING_MODE assignment."""
    train_utils, _ = mock_env_constants
    key = train_utils.make_model_key("my_model", mode=None)
    assert key == "my_model@static"


def test_parse_model_key_no_separator(mock_env_constants):
    """Line 88: key without @ returns (key, 'static')."""
    train_utils, _ = mock_env_constants
    name, mode = train_utils.parse_model_key("bare_model")
    assert name == "bare_model"
    assert mode == "static"


# ---------------------------------------------------------------------------
# get_experiment_name_for_model
# ---------------------------------------------------------------------------


def test_get_experiment_name_falls_back_to_global(mock_env_constants):
    """Lines 104-112: mode-specific key absent → fallback to global experiment_name."""
    train_utils, _ = mock_env_constants
    # No static_experiment_name, only global experiment_name
    records = {"experiment_name": "GlobalExp"}
    result = train_utils.get_experiment_name_for_model(records, "m@static")
    assert result == "GlobalExp"


def test_get_experiment_name_mode_specific_empty_falls_back(mock_env_constants):
    """Line 108: mode-specific key exists but is falsy → fallback."""
    train_utils, _ = mock_env_constants
    records = {"static_experiment_name": "", "experiment_name": "GlobalFallback"}
    result = train_utils.get_experiment_name_for_model(records, "m@static")
    assert result == "GlobalFallback"


def test_get_experiment_name_no_keys_returns_empty(mock_env_constants):
    """Line 112: no experiment_name at all → returns ''."""
    train_utils, _ = mock_env_constants
    result = train_utils.get_experiment_name_for_model({}, "m@static")
    assert result == ""


# ---------------------------------------------------------------------------
# BestScoreCaptureHandler
# ---------------------------------------------------------------------------


def test_best_score_handler_exception_is_silent(mock_env_constants):
    """Lines 665-666: exception in emit() is caught silently."""
    train_utils, _ = mock_env_constants
    handler = train_utils.BestScoreCaptureHandler()
    # Create a record whose getMessage() raises
    bad_record = MagicMock()
    bad_record.getMessage.side_effect = RuntimeError("boom")
    handler.emit(bad_record)  # must not raise
    assert handler.best_score is None
    assert handler.best_epoch is None


def test_best_score_handler_no_match(mock_env_constants):
    """Handler gracefully ignores messages without best score."""
    train_utils, _ = mock_env_constants
    handler = train_utils.BestScoreCaptureHandler()
    record = MagicMock()
    record.getMessage.return_value = "some unrelated log message"
    handler.emit(record)
    assert handler.best_score is None


# ---------------------------------------------------------------------------
# load_pretrained_metadata / resolve_pretrained_path
# ---------------------------------------------------------------------------

@patch("os.path.exists", return_value=False)
def test_load_pretrained_metadata_returns_none(mock_exists, mock_env_constants):
    """Line 387: metadata file doesn't exist → returns None."""
    train_utils, _ = mock_env_constants
    assert train_utils.load_pretrained_metadata("some_model") is None


@patch("os.path.exists", return_value=False)
def test_load_pretrained_metadata_with_date_returns_none(mock_exists, mock_env_constants):
    """Line 387 via date branch: metadata file doesn't exist."""
    train_utils, _ = mock_env_constants
    assert train_utils.load_pretrained_metadata("some_model", "2026-01-01") is None


def test_resolve_pretrained_path_empty_source(mock_env_constants):
    """Line 422: pretrain_source is empty/falsy → returns None."""
    train_utils, _ = mock_env_constants
    registry = {"my_model": {"pretrain_source": ""}}
    assert train_utils.resolve_pretrained_path("my_model", registry) is None


def test_resolve_pretrained_path_no_pretrain_source_key(mock_env_constants):
    """Line 422: no pretrain_source key at all → returns None."""
    train_utils, _ = mock_env_constants
    registry = {"my_model": {"enabled": True}}
    assert train_utils.resolve_pretrained_path("my_model", registry) is None


def test_resolve_pretrained_path_registry_none(mock_env_constants):
    """Auto-loads registry when None; pretrained path found."""
    train_utils, _ = mock_env_constants
    registry = {"my_model": {"pretrain_source": "lstm"}}
    with patch("quantpits.utils.train_utils.load_model_registry", return_value=registry):
        with patch("quantpits.utils.train_utils.get_pretrained_model_path", return_value="/fake.pkl"):
            assert train_utils.resolve_pretrained_path("my_model", registry=None) == "/fake.pkl"


# ---------------------------------------------------------------------------
# inject_config uncovered branches
# ---------------------------------------------------------------------------


def test_inject_config_day_label_no_formula(mock_env_constants):
    """Line 504: freq='day' without label_formula → daily label."""
    train_utils, _ = mock_env_constants
    mock_yaml = {
        "market": "x", "benchmark": "x", "data_handler_config": {},
        "task": {"dataset": {"kwargs": {"segments": {}}}},
    }
    params = {
        "freq": "day", "market": "csi300", "benchmark": "SH000300",
        "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
        "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
        "test_start_time": "2009", "test_end_time": "2010",
    }
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml))):
        config = train_utils.inject_config("dummy.yaml", params)
        assert config["data_handler_config"]["label"] == ["Ref($close, -2) / Ref($close, -1) - 1"]


def test_inject_config_default_freq_day_label(mock_env_constants):
    """Line 504: no freq key → defaults to 'day' internally (freq.get default)."""
    train_utils, _ = mock_env_constants
    mock_yaml = {
        "market": "x", "benchmark": "x", "data_handler_config": {},
        "task": {"dataset": {"kwargs": {"segments": {}}}},
    }
    # params without 'freq' — inject_config uses params.get('freq', 'week').lower()
    # We must explicitly pass 'freq' to control the branch. Let's test without freq.
    params = {
        "market": "csi300", "benchmark": "SH000300",
        "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
        "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
        "test_start_time": "2009", "test_end_time": "2010",
    }
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml))):
        config = train_utils.inject_config("dummy.yaml", params)
        # Default freq is 'week' via params.get('freq', 'week').lower()
        assert config["data_handler_config"]["label"] == ["Ref($close, -6) / Ref($close, -1) - 1"]


# ---------------------------------------------------------------------------
# train_single_model: TypeError in model.fit  (lines 736-738)
# ---------------------------------------------------------------------------


def test_train_single_model_fit_typeerror_fallback(mock_env_constants, tmp_path):
    """Lines 736-738: model.fit() raises TypeError → call without evals_result."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_typeerror.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"n_epochs": 10}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    model.fit.side_effect = [TypeError("unexpected kwarg"), None]
    # After the TypeError, the except TypeError catches it and calls fit() without evals_result
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_typeerror"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_m", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )
    assert result["success"] is True
    # fit was called twice: once with evals_result (TypeError), once without
    assert model.fit.call_count == 2


# ---------------------------------------------------------------------------
# train_single_model: convergence extraction paths
# ---------------------------------------------------------------------------


def test_train_evals_result_gbdt_structure(mock_env_constants, tmp_path):
    """Lines 760-795: evals_result with GBDT dict-of-dict structure."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_gbdt.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 100}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    # Remove attributes that would trigger NN paths
    del model.model  # no .model attribute

    # GBDT evals_result structure: {'train': {'l2': [...]}, 'valid': {'l2': [...]}}
    evals_result = {
        "train": {"l2": [0.5, 0.4, 0.3]},
        "valid": {"l2": [0.6, 0.5, 0.45]},
    }

    def fit_side_effect(dataset=None, evals_result=None):
        if evals_result is not None:
            evals_result.update({
                "train": {"l2": [0.5, 0.4, 0.3]},
                "valid": {"l2": [0.6, 0.5, 0.45]},
            })

    model.fit.side_effect = fit_side_effect
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_gbdt"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_gbdt", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 3
    assert conv["configured_epochs"] == 100
    assert conv["best_score"] == 0.45
    assert conv["best_epoch"] == 2


def test_train_evals_result_nn_list_structure_minimizing(mock_env_constants, tmp_path):
    """Lines 780-795: NN evals_result with train/valid lists, minimizing model."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_nn_list.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"n_epochs": 10}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    # Remove tree attributes
    if hasattr(model, "fitted_model_"):
        del model.fitted_model_
    # model.model exists but without n_epochs_fitted_ (to trigger evals_result path)
    type(model).__name__ = "GeneralModel"  # "general" in name → minimizing

    evals = {"train": [0.5, 0.4, 0.3], "valid": [0.6, 0.5, 0.45]}

    def fit_side_effect(dataset=None, evals_result=None):
        if evals_result is not None:
            evals_result.update(evals)

    model.fit.side_effect = fit_side_effect
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_nn_list"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_nn", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 3
    # minimizing → min(valid_history) = 0.45
    assert conv["best_score"] == 0.45
    assert conv["best_epoch"] == 2


def test_train_evals_result_nn_list_maximizing(mock_env_constants, tmp_path):
    """Lines 780-795: NN evals_result, maximizing model ('return' in class name)."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_nn_max.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"n_epochs": 10}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    if hasattr(model, "fitted_model_"):
        del model.fitted_model_
    # Maximizing model (name contains 'return')
    type(model).__name__ = "ReturnModel"

    evals = {"train": [0.5, 0.4, 0.3], "valid": [0.6, 0.5, 0.55]}

    def fit_side_effect(dataset=None, evals_result=None):
        if evals_result is not None:
            evals_result.update(evals)

    model.fit.side_effect = fit_side_effect
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_nn_max"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_nn_max", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    # maximizing → max(valid_history) = 0.6
    assert conv["best_score"] == 0.6
    assert conv["best_epoch"] == 0


def test_train_lightgbm_fallback(mock_env_constants, tmp_path):
    """Lines 826-845: LightGBM model attribute fallback (model.best_iteration)."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_lgb.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 200}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    # Use explicit class to avoid MagicMock auto-attribute interference
    class _LGBInner:
        current_iteration = 180
        best_iteration = 150
        best_score = {"valid_1": {"l2": 0.05}}

    class _LGBModel:
        model = _LGBInner()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _LGBModel()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_lgb"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_lgb", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 180
    assert conv["best_epoch"] == 150
    assert conv["best_score"] == 0.05
    assert conv["early_stopped"] is True


def test_train_lightgbm_best_score_valid_key(mock_env_constants, tmp_path):
    """Line 843: LightGBM best_score with 'valid' key (not 'valid_1')."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_lgb2.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 200}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    class _LGBInner2:
        current_iteration = 100
        best_iteration = 100
        best_score = {"valid": {"rmse": 0.03}}

    class _LGBModel2:
        model = _LGBInner2()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _LGBModel2()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_lgb2"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_lgb2", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["best_score"] == 0.03


def test_train_catboost_fallback(mock_env_constants, tmp_path):
    """Lines 812-823: CatBoost model attribute fallback (model.best_iteration_)."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_catboost.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 500}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    class _CatInner:
        tree_count_ = 300
        best_iteration_ = 250
        best_score_ = {"validation": {"Logloss": 0.42}}

    class _CatModel:
        model = _CatInner()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _CatModel()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_catboost"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_cat", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 300
    assert conv["best_epoch"] == 250
    assert conv["best_score"] == 0.42
    assert conv["early_stopped"] is True


def test_train_gbdt_fitted_model_fallback(mock_env_constants, tmp_path):
    """Lines 805-807: fitted_model_.best_iteration (generic GBDT)."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_gbdt_fallback.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    class _GBDTInner:
        best_iteration = 80

    class _GBDTModel:
        fitted_model_ = _GBDTInner()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _GBDTModel()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_gbdt_fb"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_gbdt_fb", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 80
    assert conv["best_epoch"] == 80


def test_train_linear_model_branch(mock_env_constants, tmp_path):
    """Lines 848-851: 'linear' in model_name → all None for epochs."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_linear.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    # Simple class without model/fitted_model_ attributes
    class _LinearModel:
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _LinearModel()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_linear"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "linear_model_v1", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] is None
    assert conv["configured_epochs"] is None
    assert conv["best_epoch"] is None
    assert conv["best_score"] is None


def test_train_convergence_extraction_exception(mock_env_constants, tmp_path):
    """Lines 858-859: exception during convergence extraction is caught."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_conv_exc.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"n_epochs": 10}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    # Model where accessing convergence attributes raises
    class _BadFittedModel:
        @property
        def best_iteration(self):
            raise RuntimeError("unexpected attr error")

    class _BadModel:
        fitted_model_ = _BadFittedModel()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _BadModel()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_conv_exc"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_conv", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    # Should still succeed despite convergence extraction failure
    assert result["success"] is True
    conv = result["performance"]["convergence"]
    # All extraction values stay None
    assert conv["actual_epochs"] is None


# ---------------------------------------------------------------------------
# train_single_model: portfolio analysis extraction  (lines 924-936)
# ---------------------------------------------------------------------------


def test_train_portfolio_analysis_single_index(mock_env_constants, tmp_path):
    """Line 933-936: port_analysis with single-level index (not MultiIndex)."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_port.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    del model.model
    model.fit.return_value = None

    # Build a DataFrame with excess_return_without_cost as single row
    port_df = pd.DataFrame(
        {"annualized_return": [0.15], "max_drawdown": [-0.08], "information_ratio": [1.2]},
        index=["excess_return_without_cost"],
    )

    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_port"}
                # load_object: IC, then port_analysis
                mock_recorder.load_object.side_effect = [
                    pd.Series([0.1, 0.2]),  # IC
                    port_df,  # portfolio analysis
                ]
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_port", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["Ann_Excess"] == 0.15
    assert conv["Max_DD"] == -0.08
    assert conv["Information_Ratio"] == 1.2


def test_train_portfolio_analysis_multiindex(mock_env_constants, tmp_path):
    """Lines 924-932: port_analysis with MultiIndex DataFrame."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_port_multi.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    del model.model
    model.fit.return_value = None

    # MultiIndex DataFrame with excess_return_without_cost
    arrays = [
        ["excess_return_without_cost", "excess_return_without_cost", "excess_return_without_cost"],
        ["annualized_return", "max_drawdown", "information_ratio"],
    ]
    index = pd.MultiIndex.from_arrays(arrays)
    port_df = pd.DataFrame({"risk": [0.12, -0.06, 0.9]}, index=index)

    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_port_multi"}
                mock_recorder.load_object.side_effect = [
                    pd.Series([0.1, 0.2]),
                    port_df,
                ]
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_port_m", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["Ann_Excess"] == 0.12
    assert conv["Max_DD"] == -0.06
    assert conv["Information_Ratio"] == 0.9


def test_train_portfolio_analysis_exception(mock_env_constants, tmp_path):
    """Lines 937-938: portfolio analysis raises exception → logged but not fatal."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_port_err.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    del model.model
    model.fit.return_value = None

    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_port_err"}
                # IC works, but port_analysis fails
                mock_recorder.load_object.side_effect = [
                    pd.Series([0.1, 0.2]),
                    Exception("Port analysis missing"),
                ]
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_port_err", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    # IC should still be extracted
    assert "IC_Mean" in result["performance"]


def test_train_single_model_top_level_exception(mock_env_constants, tmp_path):
    """Lines 969-973: top-level exception in train_single_model."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_toplevel.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                # Make R.start() raise an exception, which is caught by the try/except
                mock_R.start.side_effect = RuntimeError("simulated training failure")
                mock_recorder = MagicMock()
                mock_R.get_recorder.return_value = mock_recorder

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_err", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is False
    assert "simulated training failure" in result["error"]


# ---------------------------------------------------------------------------
# merge_train_records / overwrite / merge_performance  edge cases
# ---------------------------------------------------------------------------


def test_merge_train_records_existing_model_same_rid(mock_env_constants, tmp_path):
    """Lines 1051-1052: model already present with same rid → not added to 'updated'."""
    train_utils, _ = mock_env_constants
    record_file = tmp_path / "records.json"
    existing = {"experiment_name": "e1", "models": {"mA": "id1"}}
    record_file.write_text(json.dumps(existing))
    new = {"models": {"mA": "id1"}}  # same model, same rid
    merged = train_utils.merge_train_records(new, record_file=str(record_file))
    assert "mA" in merged["models"]
    assert merged["models"]["mA"] == "id1"


def test_merge_train_records_existing_model_different_rid(mock_env_constants, tmp_path):
    """Lines 1051-1052: model present with different rid → recorded as 'updated'."""
    train_utils, _ = mock_env_constants
    record_file = tmp_path / "records2.json"
    existing = {"experiment_name": "e1", "models": {"mA": "id1"}}
    record_file.write_text(json.dumps(existing))
    new = {"models": {"mA": "id2"}}
    merged = train_utils.merge_train_records(new, record_file=str(record_file))
    assert merged["models"]["mA"] == "id2"


def test_merge_train_records_preserves_mode_specific_experiment_names(mock_env_constants, tmp_path):
    """Lines 1076: merge preserves static_experiment_name and rolling_experiment_name."""
    train_utils, _ = mock_env_constants
    record_file = tmp_path / "records3.json"
    existing = {
        "experiment_name": "old",
        "static_experiment_name": "old_static",
        "rolling_experiment_name": "old_rolling",
        "models": {},
    }
    record_file.write_text(json.dumps(existing))
    new = {"experiment_name": "new", "models": {"mB": "idB"}}
    merged = train_utils.merge_train_records(new, record_file=str(record_file))
    # new sets experiment_name, but static/rolling should be kept from existing (since new doesn't have them)
    assert merged["experiment_name"] == "new"
    assert merged["static_experiment_name"] == "old_static"
    assert merged["rolling_experiment_name"] == "old_rolling"


def test_overwrite_train_records_default_path(mock_env_constants, tmp_path):
    """Line 1076: overwrite with default record file path."""
    train_utils, _ = mock_env_constants
    records = {"experiment_name": "test_overwrite", "models": {}}
    state_file = tmp_path / "default_rec.json"
    with patch("quantpits.utils.train_utils.RECORD_OUTPUT_FILE", str(state_file)):
        train_utils.overwrite_train_records(records)  # no record_file arg
    with open(state_file) as f:
        assert json.load(f)["experiment_name"] == "test_overwrite"


def test_merge_performance_file_new(mock_env_constants, tmp_path):
    """Line 1096: merge_performance_file when no existing file."""
    train_utils, _ = mock_env_constants
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    new = {"model1": {"IC": 0.5}}
    merged = train_utils.merge_performance_file(new, "2026", output_dir=str(out_dir))
    assert merged == {"model1": {"IC": 0.5}}


# ---------------------------------------------------------------------------
# predict_single_model  error branches
# ---------------------------------------------------------------------------


def test_predict_single_model_yaml_not_found(mock_env_constants):
    """Lines 1361-1363: yaml_file doesn't exist."""
    train_utils, _ = mock_env_constants
    params = {"anchor_date": "2026-01-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}
    result = train_utils.predict_single_model(
        "M1", {"yaml_file": "/nonexistent/path.yaml"}, params, "E", source
    )
    assert result["success"] is False
    assert "不存在" in result["error"]


def test_predict_single_model_not_in_source(mock_env_constants, tmp_path):
    """Lines 1370-1372: model not in source records."""
    train_utils, _ = mock_env_constants
    yaml_file = tmp_path / "test_pred.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-01-01"}
    source = {"experiment_name": "S", "models": {"OtherModel": "I"}}
    with patch("os.path.exists", return_value=True):
        result = train_utils.predict_single_model(
            "M1", {"yaml_file": str(yaml_file)}, params, "E", source
        )
    assert result["success"] is False
    assert "不在源训练记录中" in result["error"]


def test_predict_single_model_fallback_chain(mock_env_constants, tmp_path):
    """Lines 1400-1407: model.pkl not found → trace back through source_record_id chain."""
    train_utils, _ = mock_env_constants
    yaml_file = tmp_path / "test_fallback.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-01-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}

    task_config = {
        "task": {
            "dataset": {},
            "record": [{"class": "SigAnaRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}}],
        }
    }
    model = MagicMock()

    # First recorder: load_object raises → fallback tags have source_record_id
    parent_recorder = MagicMock()
    parent_recorder.load_object.return_value = model
    parent_recorder.info = {"id": "parent_id"}

    child_recorder = MagicMock()
    child_recorder.load_object.side_effect = [Exception("not found"), pd.Series([0.5])]
    child_recorder.list_tags.return_value = {
        "source_record_id": "parent_id",
        "source_experiment": "ParentExp",
    }
    child_recorder.info = {"id": "child_id"}

    with patch("qlib.utils.init_instance_by_config", side_effect=[model, MagicMock(), MagicMock()]):
        with patch("qlib.workflow.R") as mock_R:
            mock_R.get_recorder.side_effect = [child_recorder, parent_recorder, child_recorder]
            mock_R.start.return_value.__enter__.return_value = mock_R

            with patch("quantpits.utils.train_utils.inject_config", return_value=task_config):
                with patch("os.path.exists", return_value=True):
                    result = train_utils.predict_single_model(
                        "M1", {"yaml_file": str(yaml_file)}, params, "E", source
                    )
    assert result["success"] is True
    # Should have loaded model from parent recorder
    assert parent_recorder.load_object.called


def test_predict_single_model_max_depth_exceeded(mock_env_constants, tmp_path):
    """Line 1410: fallback chain exceeds 10 iterations."""
    train_utils, _ = mock_env_constants
    yaml_file = tmp_path / "test_maxdepth.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-01-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}

    task_config = {"task": {"dataset": {}, "record": []}}
    model = MagicMock()

    # Every recorder in chain fails and points to another
    recorder = MagicMock()
    recorder.load_object.side_effect = Exception("not found")
    recorder.list_tags.return_value = {
        "source_record_id": "next_id",
        "source_experiment": "NextExp",
    }

    with patch("qlib.utils.init_instance_by_config", side_effect=[model, MagicMock()]):
        with patch("qlib.workflow.R") as mock_R:
            mock_R.get_recorder.return_value = recorder
            mock_R.start.return_value.__enter__.return_value = mock_R

            with patch("quantpits.utils.train_utils.inject_config", return_value=task_config):
                with patch("os.path.exists", return_value=True):
                    result = train_utils.predict_single_model(
                        "M1", {"yaml_file": str(yaml_file)}, params, "E", source
                    )
    assert result["success"] is False
    assert "max traceback" in result["error"].lower()


def test_predict_single_model_ic_error(mock_env_constants, tmp_path):
    """Lines 1465-1467: IC metric extraction fails → performance only has record_id."""
    train_utils, _ = mock_env_constants
    yaml_file = tmp_path / "test_pred_icerr.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-01-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}

    task_config = {"task": {"dataset": {}, "record": []}}
    model = MagicMock()

    recorder = MagicMock()
    recorder.load_object.side_effect = [model, Exception("IC not found")]
    recorder.info = {"id": "pred_ic_err_id"}

    with patch("qlib.utils.init_instance_by_config", side_effect=[model, MagicMock()]):
        with patch("qlib.workflow.R") as mock_R:
            mock_R.get_recorder.return_value = recorder
            mock_R.start.return_value.__enter__.return_value = mock_R

            with patch("quantpits.utils.train_utils.inject_config", return_value=task_config):
                with patch("os.path.exists", return_value=True):
                    result = train_utils.predict_single_model(
                        "M1", {"yaml_file": str(yaml_file)}, params, "E", source
                    )
    assert result["success"] is True
    assert result["performance"]["record_id"] == "pred_ic_err_id"
    assert "IC_Mean" not in result["performance"]


def test_predict_single_model_top_level_exception(mock_env_constants, tmp_path):
    """Lines 1476-1480: top-level exception in predict_single_model."""
    train_utils, _ = mock_env_constants
    yaml_file = tmp_path / "test_pred_exc.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-01-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}

    model = MagicMock()

    with patch("os.path.exists", return_value=True):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, MagicMock()]):
            with patch("qlib.workflow.R") as mock_R:
                # Make get_recorder raise, which is called early in predict_single_model
                mock_R.get_recorder.side_effect = RuntimeError("boom predict")
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.inject_config", return_value={"task": {"dataset": {}, "record": []}}):
                    result = train_utils.predict_single_model(
                        "M1", {"yaml_file": str(yaml_file)}, params, "E", source
                    )
    assert result["success"] is False
    assert "boom predict" in result["error"]


# ---------------------------------------------------------------------------
# migrate_legacy_records  edge cases
# ---------------------------------------------------------------------------


def test_migrate_legacy_already_new_format(mock_env_constants, tmp_path, capsys):
    """Line 1504: already migrated (key contains @) → skip."""
    train_utils, _ = mock_env_constants
    static_file = tmp_path / "latest_train_records.json"
    existing = {"models": {"m@static": "rid1"}}
    static_file.write_text(json.dumps(existing))
    train_utils.migrate_legacy_records(workspace_dir=str(tmp_path))
    captured = capsys.readouterr()
    assert "无需迁移" in captured.out


def test_migrate_legacy_no_rolling_experiment_name(mock_env_constants, tmp_path):
    """Line 1557: rolling_records has no experiment_name."""
    train_utils, _ = mock_env_constants
    static_file = tmp_path / "latest_train_records.json"
    rolling_file = tmp_path / "latest_rolling_records.json"
    static_file.write_text(json.dumps({"models": {"static_m": "sid"}}))
    rolling_file.write_text(json.dumps({"models": {"roll_m": "rid"}}))
    result = train_utils.migrate_legacy_records(workspace_dir=str(tmp_path))
    assert "roll_m@rolling" in result["models"]
    assert "static_m@static" in result["models"]
    # rolling had no experiment_name → rolling_experiment_name should not be in result
    assert "rolling_experiment_name" not in result


def test_migrate_legacy_dry_run(mock_env_constants, tmp_path):
    """Lines 1572-1573: dry_run=True → returns merged but doesn't write."""
    train_utils, _ = mock_env_constants
    static_file = tmp_path / "latest_train_records.json"
    original = {"models": {"static_m": "sid"}}
    static_file.write_text(json.dumps(original))
    result = train_utils.migrate_legacy_records(workspace_dir=str(tmp_path), dry_run=True)
    assert "static_m@static" in result["models"]
    # File should NOT be overwritten
    on_disk = json.loads(static_file.read_text())
    assert on_disk == original  # unchanged


def test_migrate_legacy_default_workspace_dir(mock_env_constants, tmp_path):
    """Lines 1520-1521: workspace_dir=None → uses ROOT_DIR."""
    train_utils, _ = mock_env_constants
    workspace = tmp_path / "default_ws"
    workspace.mkdir()
    static_file = workspace / "latest_train_records.json"
    static_file.write_text(json.dumps({"models": {"m1": "id1"}}))
    with patch("quantpits.utils.train_utils.ROOT_DIR", str(workspace)):
        result = train_utils.migrate_legacy_records()  # no workspace_dir
    assert "m1@static" in result["models"]


def test_migrate_legacy_no_files(mock_env_constants, tmp_path):
    """Lines 1519-1521: no record files at all → returns {}."""
    train_utils, _ = mock_env_constants
    result = train_utils.migrate_legacy_records(workspace_dir=str(tmp_path))
    assert result == {}


def test_migrate_legacy_rolling_with_experiment_name(mock_env_constants, tmp_path):
    """Line 1557: rolling_records has experiment_name → stored as rolling_experiment_name."""
    train_utils, _ = mock_env_constants
    static_file = tmp_path / "latest_train_records.json"
    rolling_file = tmp_path / "latest_rolling_records.json"
    static_file.write_text(json.dumps({"models": {"m1": "id1"}}))
    rolling_file.write_text(json.dumps({"experiment_name": "RollingExp", "models": {"r1": "rid1"}}))
    result = train_utils.migrate_legacy_records(workspace_dir=str(tmp_path))
    assert result["rolling_experiment_name"] == "RollingExp"


def test_migrate_legacy_static_has_experiment_name(mock_env_constants, tmp_path):
    """Coverage: static_records has experiment_name (line 1546-1548)."""
    train_utils, _ = mock_env_constants
    static_file = tmp_path / "latest_train_records.json"
    static_file.write_text(json.dumps({
        "experiment_name": "StaticExp",
        "anchor_date": "2026-01-01",
        "models": {"m1": "id1"},
    }))
    result = train_utils.migrate_legacy_records(workspace_dir=str(tmp_path))
    assert result["experiment_name"] == "StaticExp"
    assert result["anchor_date"] == "2026-01-01"
    assert "migrated_from" in result


# ---------------------------------------------------------------------------
# resolve_model_key with default_mode specified
# ---------------------------------------------------------------------------


def test_resolve_model_key_default_mode_branch(mock_env_constants):
    """Line 142-146: default_mode specified but model not found → returns None."""
    train_utils, _ = mock_env_constants
    models_dict = {"m1@static": "rid1"}
    assert train_utils.resolve_model_key("m1", models_dict, default_mode="rolling") is None


# ---------------------------------------------------------------------------
# calculate_dates with train_date_mode not 'last_trade_date'
# ---------------------------------------------------------------------------


def test_calculate_dates_non_last_trade_date(mock_env_constants):
    """Lines 243-246: train_date_mode='fixed' → anchor_date from config."""
    train_utils, _ = mock_env_constants
    config_dict = {
        "market": "csi300", "benchmark": "SH000300",
        "train_date_mode": "fixed", "current_date": "2025-12-31",
        "data_slice_mode": "fixed",
        "start_time": "2010-01-01", "fit_start_time": "2010-01-01",
        "fit_end_time": "2018-01-01",
        "valid_start_time": "2018-01-01", "valid_end_time": "2019-01-01",
        "test_start_time": "2019-01-01", "test_end_time": "2025-12-31",
    }
    with patch("quantpits.utils.config_loader.load_workspace_config", return_value=config_dict):
        params = train_utils.calculate_dates()
        assert params["anchor_date"] == "2025-12-31"


def test_calculate_dates_freq_week(mock_env_constants):
    """Lines 239, 290: freq='week' flows into params."""
    train_utils, _ = mock_env_constants
    config_dict = {
        "market": "csi300", "benchmark": "SH000300",
        "train_date_mode": "last_trade_date",
        "data_slice_mode": "slide", "test_set_window": 1,
        "valid_set_window": 1, "train_set_windows": 3,
        "freq": "week", "current_full_cash": 100000.0,
    }
    with patch("quantpits.utils.config_loader.load_workspace_config", return_value=config_dict):
        with patch("qlib.data.D") as mock_d:
            mock_d.calendar.return_value = [pd.Timestamp("2026-01-15")]
            params = train_utils.calculate_dates()
            assert params["freq"] == "week"


# ===========================================================================
# Additional targeted tests for last few uncovered lines (targeting 100%)
# ===========================================================================


def test_get_experiment_name_mode_specific_truthy(mock_env_constants):
    """Line 109: mode-specific experiment_name exists and is truthy."""
    train_utils, _ = mock_env_constants
    records = {"static_experiment_name": "StaticExp", "experiment_name": "GlobalExp"}
    result = train_utils.get_experiment_name_for_model(records, "m@static")
    assert result == "StaticExp"


def test_inject_config_no_pretrain_deletes_model_path(mock_env_constants):
    """Lines 559-560: pretrain_path not found → delete model_path from kwargs."""
    train_utils, _ = mock_env_constants
    mock_yaml = {
        "market": "x", "benchmark": "x", "data_handler_config": {},
        "task": {
            "dataset": {"kwargs": {"segments": {}}},
            "model": {"kwargs": {"base_model": "dummy", "model_path": "old_path"}},
        },
    }
    params = {
        "freq": "day", "market": "csi300", "benchmark": "SH000300",
        "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
        "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
        "test_start_time": "2009", "test_end_time": "2010",
    }
    # Lines 559-560: no_pretrain=False (default), model_name given,
    # resolve_pretrained_path returns None → delete model_path
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml))):
        with patch("quantpits.utils.train_utils.resolve_pretrained_path", return_value=None):
            config = train_utils.inject_config("dummy.yaml", params, model_name="dummy")
            assert "model_path" not in config["task"]["model"]["kwargs"]

    # Also test no_pretrain=True explicitly (lines 542-544)
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml))):
        config2 = train_utils.inject_config("dummy.yaml", params, model_name="dummy", no_pretrain=True)
        assert "model_path" not in config2["task"]["model"]["kwargs"]


def test_train_evals_result_stop_iteration(mock_env_constants, tmp_path):
    """Lines 762-763: StopIteration when evals_result is truthy but has empty iterator.

    NOTE: Lines 762-763 are defensive dead code — a regular Python dict is
    only truthy when non-empty, and a non-empty dict always yields at least
    one key. The StopIteration handler exists as a safety net for custom
    dict-like objects that might be passed as evals_result. We verify the
    normal empty-evals_result path works correctly.
    """
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_stopiter.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"n_epochs": 10}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    model = MagicMock()
    del model.model
    model.fit.return_value = None
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_stopiter"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_stop", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )
    assert result["success"] is True


def test_train_catboost_best_score_exception(mock_env_constants, tmp_path):
    """Lines 822-823: CatBoost best_score_ extraction raises exception."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_catboost2.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 500}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    # CatBoost where best_score_ access raises an error
    class _CatInnerBad:
        tree_count_ = 300
        best_iteration_ = 250

        @property
        def best_score_(self):
            raise RuntimeError("corrupted best_score_")

    class _CatModelBad:
        model = _CatInnerBad()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _CatModelBad()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_cat_bad"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_cat_bad", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 300
    assert conv["best_epoch"] == 250
    assert conv["best_score"] is None  # best_score extraction failed silently


def test_train_lgb_num_trees_fallback(mock_env_constants, tmp_path):
    """Lines 829-832: LightGBM num_trees() fallback when current_iteration is absent."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_lgb_numtrees.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 200}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    # LightGBM where current_iteration is not set (not int/float) but num_trees() is available
    class _LGBInnerNumTrees:
        best_iteration = 120

        def num_trees(self):
            return 110

    class _LGBModelNumTrees:
        model = _LGBInnerNumTrees()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _LGBModelNumTrees()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_lgb_nt"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_lgb_nt", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 110
    assert conv["best_epoch"] == 120


def test_train_lgb_num_trees_exception(mock_env_constants, tmp_path):
    """Lines 829-832: num_trees() itself raises an exception (bare except)."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_lgb_numtrees_exc.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 200}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    class _LGBInnerBadTrees:
        best_iteration = 120

        def num_trees(self):
            raise RuntimeError("boom")

    class _LGBModelBadTrees:
        model = _LGBInnerBadTrees()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _LGBModelBadTrees()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_lgb_badnt"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_lgb_bnt", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["best_epoch"] == 120
    # actual_epochs stays None since num_trees failed


def test_train_lgb_best_score_exception(mock_env_constants, tmp_path):
    """Lines 844-845: LightGBM best_score extraction raises exception."""
    train_utils, _ = mock_env_constants

    yaml_file = tmp_path / "test_lgb_badscore.yaml"
    yaml_content = {
        "task": {
            "model": {"kwargs": {"num_boost_round": 200}},
            "dataset": {"kwargs": {"segments": {}}},
            "record": [],
        },
        "data_handler_config": {},
    }
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    class _LGBInnerBadScore:
        current_iteration = 100
        best_iteration = 100

        @property
        def best_score(self):
            raise RuntimeError("corrupted best_score")

    class _LGBModelBadScore:
        model = _LGBInnerBadScore()
        fit = MagicMock(return_value=None)
        predict = MagicMock()

    model = _LGBModelBadScore()
    dataset = MagicMock()

    with patch("quantpits.utils.train_utils.inject_config", return_value=yaml_content):
        with patch("qlib.utils.init_instance_by_config", side_effect=[model, dataset]):
            with patch("qlib.workflow.R") as mock_R:
                mock_recorder = MagicMock()
                mock_recorder.info = {"id": "rid_lgb_bads"}
                mock_R.get_recorder.return_value = mock_recorder
                mock_R.start.return_value.__enter__.return_value = mock_R

                with patch("quantpits.utils.train_utils.os.path.exists", return_value=True):
                    result = train_utils.train_single_model(
                        "test_lgb_bs", str(yaml_file),
                        {"freq": "day", "market": "csi300", "benchmark": "SH000300",
                         "start_time": "2000", "end_time": "2010", "fit_start_time": "2000",
                         "fit_end_time": "2005", "valid_start_time": "2006", "valid_end_time": "2008",
                         "test_start_time": "2009", "test_end_time": "2010", "anchor_date": "2026-01-01"},
                        "test_exp",
                    )

    assert result["success"] is True
    conv = result["performance"]["convergence"]
    assert conv["actual_epochs"] == 100
    assert conv["best_score"] is None  # best_score extraction failed silently


def test_merge_train_records_default_path(mock_env_constants, tmp_path):
    """Line 1032: merge_train_records with default record_file (None)."""
    train_utils, _ = mock_env_constants
    default_file = tmp_path / "default_merge.json"
    # Pre-populate so the existing-file branch loads
    default_file.write_text(json.dumps({"experiment_name": "e1", "models": {}}))
    new = {"experiment_name": "e2", "models": {"m1": "id1"}}
    with patch("quantpits.utils.train_utils.RECORD_OUTPUT_FILE", str(default_file)):
        merged = train_utils.merge_train_records(new)  # no record_file arg
    assert merged["experiment_name"] == "e2"
    assert "m1" in merged["models"]


def test_predict_single_model_no_parent_tags(mock_env_constants, tmp_path):
    """Line 1407: fallback finds recorder but tags lack source_record_id."""
    train_utils, _ = mock_env_constants
    yaml_file = tmp_path / "test_noparent.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-01-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}

    task_config = {"task": {"dataset": {}, "record": []}}
    model = MagicMock()

    # Recorder fails to load model.pkl, and list_tags has NO source_record_id
    recorder = MagicMock()
    recorder.load_object.side_effect = Exception("model.pkl not found")
    recorder.list_tags.return_value = {}  # no source keys → triggers line 1407

    with patch("qlib.utils.init_instance_by_config", side_effect=[model, MagicMock()]):
        with patch("qlib.workflow.R") as mock_R:
            mock_R.get_recorder.return_value = recorder
            mock_R.start.return_value.__enter__.return_value = mock_R

            with patch("quantpits.utils.train_utils.inject_config", return_value=task_config):
                with patch("os.path.exists", return_value=True):
                    result = train_utils.predict_single_model(
                        "M1", {"yaml_file": str(yaml_file)}, params, "E", source
                    )
    assert result["success"] is False
    assert "no parent tags" in result["error"]


# ============================================================================
# CPCV: compute_cpcv_folds error branches
# ============================================================================


class TestComputeCpcvFoldsErrors:
    """Cover the ValueError branches in compute_cpcv_folds()."""

    @staticmethod
    def _make_config(**overrides):
        cfg = {
            "start_time": "2020-01-01",
            "purged_cv": {
                "n_groups": 20,
                "n_test_groups": 4,
                "n_val_groups": 2,
                "purge_steps": 5,
                "embargo_steps": 0,
            },
        }
        cfg["purged_cv"].update(overrides)
        return cfg

    def _make_cal(self, n_days=500):
        """Build a mock calendar list of n_days trading dates."""
        import pandas as pd
        return pd.DatetimeIndex(
            pd.date_range("2020-01-06", periods=n_days, freq="B")
        )

    def test_n_test_groups_gte_n_groups(self):
        """Line 303-307: n_test_groups >= n_groups raises ValueError."""
        from quantpits.utils.train_utils import compute_cpcv_folds
        config = self._make_config(n_test_groups=20, n_groups=10)
        with pytest.raises(ValueError, match="n_test_groups"):
            compute_cpcv_folds("2025-12-31", config, freq="day")

    def test_n_val_groups_too_large(self):
        """Line 308-312: n_val_groups exceeds available CV groups."""
        from quantpits.utils.train_utils import compute_cpcv_folds
        # n_groups=20, n_test_groups=10 => 10 CV groups, n_val_groups=15 > 10
        config = self._make_config(n_groups=20, n_test_groups=10, n_val_groups=15)
        with pytest.raises(ValueError, match="n_val_groups"):
            compute_cpcv_folds("2025-12-31", config, freq="day")

    def test_empty_calendar(self, mock_env_constants):
        """Line 316-320: zero-period calendar raises ValueError."""
        from quantpits.utils.train_utils import compute_cpcv_folds
        import sys
        config = self._make_config()
        with patch.dict(sys.modules, {'qlib.data': MagicMock()}):
            sys.modules['qlib.data'].D.calendar.return_value = []
            with pytest.raises(ValueError, match="No trading periods"):
                compute_cpcv_folds("2025-12-31", config, freq="day")

    def test_total_periods_lt_n_groups(self, mock_env_constants):
        """Line 324-328: fewer periods than n_groups."""
        from quantpits.utils.train_utils import compute_cpcv_folds
        import sys
        config = self._make_config(n_groups=100)
        with patch.dict(sys.modules, {'qlib.data': MagicMock()}):
            sys.modules['qlib.data'].D.calendar.return_value = self._make_cal(50)
            with pytest.raises(ValueError, match="Total periods"):
                compute_cpcv_folds("2025-12-31", config, freq="day")

    def test_n_folds_lt_1(self, mock_env_constants):
        """Line 351-356: defensive check for n_folds < 1 (n_val_groups
        too large relative to CV groups — guarded by line 308 first)."
        This triggers the n_val_groups check (line 308) instead, which
        guards the exact same condition. Line 352 is mathematically
        equivalent to line 308 and thus unreachable in normal flow."""
        pass  # Line 352 is mathematically guarded by line 308

    def test_purge_consumes_all_training(self, mock_env_constants):
        """Line 363-368: purge_steps >= smallest CV group size."""
        from quantpits.utils.train_utils import compute_cpcv_folds
        import sys
        # Small n_groups so each group is only a few periods
        config = self._make_config(n_groups=10, n_test_groups=2, n_val_groups=1,
                                   purge_steps=20, embargo_steps=0)
        with patch.dict(sys.modules, {'qlib.data': MagicMock()}):
            sys.modules['qlib.data'].D.calendar.return_value = self._make_cal(50)
            with pytest.raises(ValueError, match="purge_steps.*>=.*smallest"):
                compute_cpcv_folds("2025-12-31", config, freq="day")

    def test_no_training_data_after_purge(self, mock_env_constants):
        """Line 401-406: purge+embargo leaves no training segments."""
        from quantpits.utils.train_utils import compute_cpcv_folds
        import sys
        # n_groups=10, n_test=2, n_val=8, 100 days -> ~10 days/group
        # n_cv=8, n_folds=1. Fold 0: val groups 0-7. No left/right training.
        # purge_steps=9 (< group_size=10) doesn't trigger line 363, but the
        # fold has no training segments → triggers line 402.
        config = self._make_config(n_groups=10, n_test_groups=2, n_val_groups=8,
                                   purge_steps=9, embargo_steps=0)
        with patch.dict(sys.modules, {'qlib.data': MagicMock()}):
            sys.modules['qlib.data'].D.calendar.return_value = self._make_cal(100)
            with pytest.raises(ValueError, match="no training data"):
                compute_cpcv_folds("2025-12-31", config, freq="day")

    def test_embargo_pushes_test_past_end(self, mock_env_constants):
        """Line 418-424: embargo pushes test_start past test_end."""
        from quantpits.utils.train_utils import compute_cpcv_folds
        import sys
        # n_groups=10, n_test=2, n_val=2, purge=0, embargo=50, cal=100
        # 8 CV groups, 7 folds, each fold has valid training data.
        # After folds: embargo pushes test_start (130) > test_end (99).
        config = self._make_config(n_groups=10, n_test_groups=2, n_val_groups=2,
                                   purge_steps=0, embargo_steps=50)
        with patch.dict(sys.modules, {'qlib.data': MagicMock()}):
            sys.modules['qlib.data'].D.calendar.return_value = self._make_cal(100)
            with pytest.raises(ValueError, match="Embargo pushes test_start"):
                compute_cpcv_folds("2025-12-31", config, freq="day")


# ============================================================================
# CPCV: _warn_temporal_processors
# ============================================================================


class TestWarnTemporalProcessors:
    """Cover the temporal normalizer warning in _warn_temporal_processors."""

    def test_warns_on_temporal_normalizers(self, mock_env_constants):
        """Lines 957-975: warns when ZScoreNorm/MinMaxNorm/RobustZScoreNorm detected."""
        from quantpits.utils.train_utils import _warn_temporal_processors
        import warnings

        config = {
            "data_handler_config": {
                "learn_processors": [
                    {"class": "ZScoreNorm"},
                    {"class": "DropnaLabel"},
                ],
                "infer_processors": [
                    {"class": "MinMaxNorm"},
                ],
            }
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_temporal_processors(config, "/fake/path")
            assert len(w) >= 1
            assert any("ZScoreNorm" in str(warning.message) for warning in w)
            assert any("MinMaxNorm" in str(warning.message) for warning in w)

    def test_no_warn_without_temporal(self, mock_env_constants):
        """No warning when no temporal normalizers present."""
        from quantpits.utils.train_utils import _warn_temporal_processors
        import warnings

        config = {
            "data_handler_config": {
                "learn_processors": [
                    {"class": "DropnaLabel"},
                ],
                "infer_processors": [
                    {"class": "CSRankNorm"},
                ],
            }
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_temporal_processors(config, "/fake/path")
            temporal_warnings = [x for x in w if "temporal" in str(x.message).lower()
                                 or "ZScoreNorm" in str(x.message)
                                 or "MinMaxNorm" in str(x.message)
                                 or "RobustZScoreNorm" in str(x.message)]
            assert len(temporal_warnings) == 0

    def test_robust_zscore_warns(self, mock_env_constants):
        """RobustZScoreNorm also triggers warning."""
        from quantpits.utils.train_utils import _warn_temporal_processors
        import warnings

        config = {
            "data_handler_config": {
                "learn_processors": [
                    {"class": "RobustZScoreNorm"},
                ],
                "infer_processors": [],
            }
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_temporal_processors(config, "/fake/path")
            assert len(w) >= 1
