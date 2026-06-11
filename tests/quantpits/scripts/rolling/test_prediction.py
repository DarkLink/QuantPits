"""
Tests for rolling/prediction.py — filter helpers and record saving.
No Qlib infrastructure needed.
"""

import os
import tempfile
from unittest import mock

import pandas as pd
import pytest

from quantpits.scripts.rolling.prediction import (
    _filter_pred_to_test_segment,
    save_rolling_records,
    _repair_truncated_prediction,
    predict_with_latest_model,
)


# ===========================================================================
# _filter_pred_to_test_segment
# ===========================================================================

class TestFilterPredToTestSegment:
    @staticmethod
    def _make_pred(dates, instruments=("A", "B")):
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(dates), instruments],
            names=["datetime", "instrument"],
        )
        return pd.DataFrame({"score": range(len(idx))}, index=idx)

    def test_all_dates_inside_window_kept(self):
        pred = self._make_pred(["2020-01-02", "2020-01-03", "2020-01-04"])
        window = {"test_start": "2020-01-01", "test_end": "2020-01-05"}
        result = _filter_pred_to_test_segment(pred, window)
        assert len(result) == len(pred)

    def test_boundary_dates_included(self):
        pred = self._make_pred(["2020-01-01", "2020-01-02", "2020-01-05"])
        window = {"test_start": "2020-01-01", "test_end": "2020-01-05"}
        result = _filter_pred_to_test_segment(pred, window)
        assert len(result) == len(pred)

    def test_dates_before_window_excluded(self):
        pred = self._make_pred(["2019-12-30", "2020-01-02"])
        window = {"test_start": "2020-01-01", "test_end": "2020-01-05"}
        result = _filter_pred_to_test_segment(pred, window)
        assert len(result) == 2  # only the 2 rows on 2020-01-02
        dates = result.index.get_level_values("datetime")
        assert (dates >= pd.Timestamp("2020-01-01")).all()

    def test_dates_after_window_excluded(self):
        pred = self._make_pred(["2020-01-04", "2020-01-10"])
        window = {"test_start": "2020-01-01", "test_end": "2020-01-05"}
        result = _filter_pred_to_test_segment(pred, window)
        assert len(result) == 2  # only 2 rows on 2020-01-04

    def test_empty_when_no_dates_in_range(self):
        pred = self._make_pred(["2019-12-30", "2019-12-31"])
        window = {"test_start": "2020-01-01", "test_end": "2020-01-05"}
        result = _filter_pred_to_test_segment(pred, window)
        assert len(result) == 0


# ===========================================================================
# save_rolling_records
# ===========================================================================

class TestSaveRollingRecords:
    @pytest.fixture(autouse=True)
    def _setup_env(self, monkeypatch, tmp_path):
        """Ensure QLIB_WORKSPACE_DIR is set so train_utils can be imported."""
        monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(tmp_path))

    def test_formats_model_at_rolling_keys(self):
        combined_records = {"lstm_model": "rid-1", "gru_model": "rid-2"}
        with mock.patch(
            "quantpits.utils.train_utils.merge_train_records"
        ) as mock_merge:
            save_rolling_records(combined_records, "exp_name", "2025-01-01")
            mock_merge.assert_called_once()
            records = mock_merge.call_args[0][0]
            assert "lstm_model@rolling" in records["models"]
            assert "gru_model@rolling" in records["models"]
            assert records["models"]["lstm_model@rolling"] == "rid-1"
            assert records["models"]["gru_model@rolling"] == "rid-2"

    def test_includes_experiment_name_and_anchor_date(self):
        with mock.patch(
            "quantpits.utils.train_utils.merge_train_records"
        ) as mock_merge:
            save_rolling_records({}, "my_exp", "2025-06-01")
            records = mock_merge.call_args[0][0]
            assert records["experiment_name"] == "my_exp"
            assert records["rolling_experiment_name"] == "my_exp"
            assert records["anchor_date"] == "2025-06-01"
            assert "timestamp" in records
            assert records["models"] == {}


# ===========================================================================
# _repair_truncated_prediction and predict_with_latest_model
# ===========================================================================

class TestRepairAndPredict:
    def test_repair_truncated_prediction(self):
        from unittest.mock import MagicMock, patch
        
        # Setup inputs
        model_name = "lstm"
        model_info = {"yaml_file": "fake.yaml"}
        comp = {"window_idx": 0, "record_id": "rec-1"}
        window_map = {
            0: {
                "train_start": "2020-01-01",
                "train_end": "2020-01-10",
                "valid_start": "2020-01-11",
                "valid_end": "2020-01-15",
                "test_start": "2020-01-16",
                "test_end": "2020-01-20"
            }
        }
        rolling_exp_name = "rolling_exp"
        params_base = {"market": "all"}
        
        # 1. Test when window is missing in window_map
        pred, rep = _repair_truncated_prediction(model_name, model_info, comp, {}, rolling_exp_name, params_base)
        assert pred is None
        assert rep is False
        
        # 2. Test when prediction is not truncated (pred_max >= test_end)
        idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-20")], ["A"]], names=["datetime", "instrument"])
        mock_pred = pd.DataFrame({"score": [1.0]}, index=idx)
        
        mock_rec = MagicMock()
        mock_rec.load_object.side_effect = lambda x: mock_pred if x == "pred.pkl" else MagicMock()
        
        with patch("qlib.workflow.R.get_recorder", return_value=mock_rec):
            pred, rep = _repair_truncated_prediction(model_name, model_info, comp, window_map, rolling_exp_name, params_base)
            assert rep is False
            
        # 3. Test when prediction is truncated and repaired
        idx_trunc = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-17")], ["A"]], names=["datetime", "instrument"])
        mock_pred_trunc = pd.DataFrame({"score": [1.0]}, index=idx_trunc)
        
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([1.5], index=idx)
        
        def mock_load(x):
            if x == "pred.pkl":
                return mock_pred_trunc
            elif x == "model.pkl":
                return mock_model
            raise ValueError()
            
        mock_rec.load_object.side_effect = mock_load
        
        with patch("qlib.workflow.R.get_recorder", return_value=mock_rec), \
             patch("quantpits.utils.train_utils.inject_config", return_value={"task": {"dataset": {}}}), \
             patch("qlib.utils.init_instance_by_config", return_value=MagicMock()):
            pred, rep = _repair_truncated_prediction(model_name, model_info, comp, window_map, rolling_exp_name, params_base)
            assert rep is True
            assert isinstance(pred, pd.DataFrame)

    def test_predict_with_latest_model(self):
        from unittest.mock import MagicMock, patch
        
        mock_state = MagicMock()
        mock_state.get_completed_record_ids.return_value = [
            {"window_idx": 0, "record_id": "rec-1"}
        ]
        
        # Test no completions
        mock_state_empty = MagicMock()
        mock_state_empty.get_completed_record_ids.return_value = []
        res = predict_with_latest_model("lstm", {}, mock_state_empty, "exp", {}, "2020-01-01", [])
        assert res is None
        
        # Test window not found
        windows = [{"window_idx": 1, "test_end": "2020-01-20"}]
        res = predict_with_latest_model("lstm", {}, mock_state, "exp", {}, "2020-01-01", windows)
        assert res is None
        
        # Test successful prediction
        windows = [
            {
                "window_idx": 0,
                "train_start": "2020-01-01",
                "train_end": "2020-01-10",
                "valid_start": "2020-01-11",
                "valid_end": "2020-01-15",
                "test_start": "2020-01-16",
                "test_end": "2020-01-20"
            }
        ]
        
        mock_rec = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([1.0], name="score")
        mock_rec.load_object.return_value = mock_model
        
        with patch("qlib.workflow.R.get_recorder", return_value=mock_rec), \
             patch("quantpits.utils.train_utils.inject_config", return_value={"task": {"dataset": {}}}), \
             patch("qlib.utils.init_instance_by_config", return_value=MagicMock()):
            pred = predict_with_latest_model("lstm", {"yaml_file": "fake.yaml"}, mock_state, "exp", {}, "2020-01-01", windows)
            assert isinstance(pred, pd.DataFrame)
            assert "score" in pred.columns

