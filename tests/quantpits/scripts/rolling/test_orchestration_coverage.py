"""
Coverage tests for rolling/orchestration.py (86% → target 95%+).

Coverage targets (line numbers refer to orchestration.py):
  - run_model_windows: already completed, training failure, KeyboardInterrupt,
    dry_run, cache_size pass-through, zero windows
  - concatenate_rolling_predictions: missing recorder, missing pred,
    Series/DataFrame normalization
  - save_rolling_records: key formatting, experiment_name inclusion, mode suffixes
"""

import sys
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# =========================================================================
# run_model_windows
# =========================================================================

class TestRunModelWindows:
    """Tests for run_model_windows()."""

    def _make_window(self, widx=0):
        return {
            "window_idx": widx,
            "train_start": f"2020-{1+widx*3:02d}-01",
            "train_end": f"2022-{12-widx*3:02d}-31",
            "valid_start": f"2023-{1+widx*3:02d}-01",
            "valid_end": f"2023-{12-widx*3:02d}-31",
            "test_start": f"2024-{1+widx*3:02d}-01",
            "test_end": f"2024-{3+widx*3:02d}-31",
        }

    def test_dry_run_skips_training(self):
        from quantpits.scripts.rolling.orchestration import run_model_windows

        windows = [self._make_window(0), self._make_window(1)]
        state = MagicMock()
        state.is_window_model_done.return_value = False

        train_fn = MagicMock()
        result = run_model_windows(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            windows=windows,
            state=state,
            params_base={"freq": "week"},
            experiment_name="test_exp",
            qlib_config=None,
            train_fn=train_fn,
            dry_run=True,
        )

        assert result == 0  # No windows trained
        train_fn.assert_not_called()

    def test_already_completed_skipped(self):
        from quantpits.scripts.rolling.orchestration import run_model_windows

        windows = [self._make_window(0)]
        state = MagicMock()
        state.is_window_model_done.return_value = True  # already done

        train_fn = MagicMock()
        result = run_model_windows(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            windows=windows,
            state=state,
            params_base={"freq": "week"},
            experiment_name="test_exp",
            qlib_config=None,
            train_fn=train_fn,
        )

        assert result == 0
        train_fn.assert_not_called()

    def test_failed_training(self):
        from quantpits.scripts.rolling.orchestration import run_model_windows

        windows = [self._make_window(0)]
        state = MagicMock()
        state.is_window_model_done.return_value = False

        train_fn = MagicMock(return_value={
            "success": False,
            "error": "Training failed",
        })

        result = run_model_windows(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            windows=windows,
            state=state,
            params_base={"freq": "week"},
            experiment_name="test_exp",
            qlib_config=None,
            train_fn=train_fn,
        )

        assert result == 0  # No successes
        state.mark_window_model_done.assert_not_called()

    def test_successful_training(self):
        from quantpits.scripts.rolling.orchestration import run_model_windows

        windows = [self._make_window(0)]
        state = MagicMock()
        state.is_window_model_done.return_value = False

        train_fn = MagicMock(return_value={
            "success": True,
            "record_id": "rid_001",
            "performance": {"IC_Mean": 0.05, "ICIR": 0.5},
        })

        result = run_model_windows(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            windows=windows,
            state=state,
            params_base={"freq": "week"},
            experiment_name="test_exp",
            qlib_config=None,
            train_fn=train_fn,
        )

        assert result == 1
        state.mark_window_model_done.assert_called_once_with(0, "test_model", "rid_001")

    def test_successful_with_none_performance(self):
        from quantpits.scripts.rolling.orchestration import run_model_windows

        windows = [self._make_window(0)]
        state = MagicMock()
        state.is_window_model_done.return_value = False

        train_fn = MagicMock(return_value={
            "success": True,
            "record_id": "rid_001",
            "performance": {},  # Empty dict, no IC info
        })

        result = run_model_windows(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            windows=windows,
            state=state,
            params_base={"freq": "week"},
            experiment_name="test_exp",
            qlib_config=None,
            train_fn=train_fn,
        )

        assert result == 1

    def test_zero_windows(self):
        from quantpits.scripts.rolling.orchestration import run_model_windows

        state = MagicMock()
        train_fn = MagicMock()

        result = run_model_windows(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            windows=[],  # empty
            state=state,
            params_base={"freq": "week"},
            experiment_name="test_exp",
            qlib_config=None,
            train_fn=train_fn,
        )

        assert result == 0
        train_fn.assert_not_called()

    def test_cache_size_passed_to_train_fn(self):
        from quantpits.scripts.rolling.orchestration import run_model_windows

        windows = [self._make_window(0)]
        state = MagicMock()
        state.is_window_model_done.return_value = False

        train_fn = MagicMock(return_value={
            "success": True, "record_id": "rid_001",
            "performance": {"IC_Mean": 0.05},
        })

        run_model_windows(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            windows=windows,
            state=state,
            params_base={"freq": "week"},
            experiment_name="test_exp",
            qlib_config=None,
            train_fn=train_fn,
            cache_size=1024,
        )

        # Verify cache_size was passed
        call_kwargs = train_fn.call_args[1]
        assert call_kwargs.get("cache_size") == 1024


# =========================================================================
# concatenate_rolling_predictions
# =========================================================================

class TestConcatenateRollingPredictions:
    """Tests for concatenate_rolling_predictions()."""

    def _make_state(self, completions=None):
        state = MagicMock()
        if completions is None:
            completions = [{"window_idx": 0, "record_id": "rid_000"}]
        state.get_completed_record_ids.return_value = completions
        return state

    def _make_window_map(self):
        return {
            0: {
                "window_idx": 0,
                "train_start": "2020-01-01", "train_end": "2022-12-31",
                "test_start": "2023-01-01", "test_end": "2023-03-30",
            },
        }

    def _make_pred(self, n=10, start="2023-01-01"):
        dates = pd.date_range(start, periods=n, freq="W")
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        return pd.DataFrame({"score": range(n)}, index=idx)

    def test_no_completions(self):
        from quantpits.scripts.rolling.orchestration import concatenate_rolling_predictions

        state = MagicMock()
        state.get_completed_record_ids.return_value = []

        with patch("qlib.workflow.R"):
            result = concatenate_rolling_predictions(
                state=state,
                model_names=["test_model"],
                rolling_exp_name="test_exp",
                combined_exp_name="combined_exp",
                anchor_date="2023-03-30",
                windows=[self._make_window_map()[0]],
            )

        assert result == {}

    def test_successful_concatenation(self, mock_qlib_R):
        from quantpits.scripts.rolling.orchestration import concatenate_rolling_predictions

        pred = self._make_pred()
        mock_qlib_R["recorder"].load_object.return_value = pred

        state = self._make_state()
        windows = [self._make_window_map()[0]]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = concatenate_rolling_predictions(
                state=state,
                model_names=["test_model"],
                rolling_exp_name="test_exp",
                combined_exp_name="combined_exp",
                anchor_date="2023-03-30",
                windows=windows,
            )

        assert "test_model" in result

    def test_missing_recorder(self, mock_qlib_R):
        from quantpits.scripts.rolling.orchestration import concatenate_rolling_predictions

        mock_qlib_R["R"].get_recorder.side_effect = Exception("Recorder not found")

        state = self._make_state()
        windows = [self._make_window_map()[0]]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = concatenate_rolling_predictions(
                state=state,
                model_names=["test_model"],
                rolling_exp_name="test_exp",
                combined_exp_name="combined_exp",
                anchor_date="2023-03-30",
                windows=windows,
            )

        # Should not crash, just no valid data
        assert result == {}

    def test_series_prediction_normalization(self, mock_qlib_R):
        """Series prediction is converted to DataFrame."""
        from quantpits.scripts.rolling.orchestration import concatenate_rolling_predictions

        dates = pd.date_range("2023-01-01", periods=10, freq="W")
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        pred_series = pd.Series(range(10), index=idx, name="score")

        mock_qlib_R["recorder"].load_object.return_value = pred_series

        state = self._make_state()
        windows = [self._make_window_map()[0]]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = concatenate_rolling_predictions(
                state=state,
                model_names=["test_model"],
                rolling_exp_name="test_exp",
                combined_exp_name="combined_exp",
                anchor_date="2023-03-30",
                windows=windows,
            )

        assert "test_model" in result

    def test_dataframe_without_score_column(self, mock_qlib_R):
        """DataFrame without 'score' column gets columns renamed."""
        from quantpits.scripts.rolling.orchestration import concatenate_rolling_predictions

        dates = pd.date_range("2023-01-01", periods=10, freq="W")
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        pred_df = pd.DataFrame({"pred": range(10)}, index=idx)

        mock_qlib_R["recorder"].load_object.return_value = pred_df

        state = self._make_state()
        windows = [self._make_window_map()[0]]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = concatenate_rolling_predictions(
                state=state,
                model_names=["test_model"],
                rolling_exp_name="test_exp",
                combined_exp_name="combined_exp",
                anchor_date="2023-03-30",
                windows=windows,
            )

        assert "test_model" in result

    def test_extra_predictions_series(self, mock_qlib_R):
        """Extra predictions as Series."""
        from quantpits.scripts.rolling.orchestration import concatenate_rolling_predictions

        pred = self._make_pred()
        mock_qlib_R["recorder"].load_object.return_value = pred

        # Extra preds as Series
        dates = pd.date_range("2023-04-01", periods=5, freq="W")
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        extra_pred = pd.Series(range(5), index=idx, name="extra")

        state = self._make_state()
        windows = [self._make_window_map()[0]]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = concatenate_rolling_predictions(
                state=state,
                model_names=["test_model"],
                rolling_exp_name="test_exp",
                combined_exp_name="combined_exp",
                anchor_date="2023-03-30",
                windows=windows,
                extra_preds={"test_model": extra_pred},
            )

        assert "test_model" in result

    def test_repair_fn_called_for_non_last_windows(self, mock_qlib_R):
        """Repair function is called for non-last completed windows."""
        from quantpits.scripts.rolling.orchestration import concatenate_rolling_predictions

        pred = self._make_pred()
        mock_qlib_R["recorder"].load_object.return_value = pred

        # Two completions: window 0 and window 1
        state = self._make_state([
            {"window_idx": 0, "record_id": "rid_000"},
            {"window_idx": 1, "record_id": "rid_001"},
        ])
        windows = [
            self._make_window_map()[0],
            {
                "window_idx": 1,
                "train_start": "2020-04-01", "train_end": "2023-03-31",
                "test_start": "2023-04-01", "test_end": "2023-06-30",
            },
        ]

        repair_fn = MagicMock(return_value=(None, False))  # No repair needed

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = concatenate_rolling_predictions(
                state=state,
                model_names=["test_model"],
                rolling_exp_name="test_exp",
                combined_exp_name="combined_exp",
                anchor_date="2023-06-30",
                windows=windows,
                targets={"test_model": {"yaml_file": "model.yaml"}},
                params_base={"freq": "week"},
                repair_fn=repair_fn,
            )

        assert "test_model" in result
        # repair_fn should be called for window 0 (not the last: window 1)
        repair_fn.assert_called()


# =========================================================================
# save_rolling_records
# =========================================================================

class TestSaveRollingRecords:
    """Tests for save_rolling_records()."""

    def test_slide_mode_key_format(self, monkeypatch, tmp_path):
        from quantpits.scripts.rolling.orchestration import save_rolling_records
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        monkeypatch.setattr(train_utils, "RECORD_OUTPUT_FILE", str(records_file))

        save_rolling_records(
            combined_records={"model_a": "rid_001"},
            combined_exp_name="test_exp",
            anchor_date="2023-03-30",
            mode="rolling",
        )

        assert records_file.exists()
        data = json.loads(records_file.read_text())
        assert "models" in data
        assert "model_a@rolling" in data["models"]
        assert data["model_records"]["model_a@rolling"]["operation"] == "legacy_import"
        assert data["model_records"]["model_a@rolling"]["status"] == "legacy_unverified"

    def test_cpcv_mode_key_format(self, monkeypatch, tmp_path):
        from quantpits.scripts.rolling.orchestration import save_rolling_records
        from quantpits.utils import train_utils

        records_file = tmp_path / "records_cpcv.json"
        monkeypatch.setattr(train_utils, "RECORD_OUTPUT_FILE", str(records_file))

        save_rolling_records(
            combined_records={"model_a": "rid_001"},
            combined_exp_name="cpcv_exp",
            anchor_date="2023-03-30",
            mode="cpcv_rolling",
        )

        assert records_file.exists()
        data = json.loads(records_file.read_text())
        assert "model_a@cpcv_rolling" in data["models"]
        # Merge may preserve existing fields; experiment_name should be set
        assert data.get("experiment_name") == "cpcv_exp"

    def test_multiple_models(self, monkeypatch, tmp_path):
        from quantpits.scripts.rolling.orchestration import save_rolling_records
        from quantpits.utils import train_utils

        records_file = tmp_path / "records_multi.json"
        monkeypatch.setattr(train_utils, "RECORD_OUTPUT_FILE", str(records_file))

        save_rolling_records(
            combined_records={"m1": "rid_001", "m2": "rid_002"},
            combined_exp_name="test_exp",
            anchor_date="2023-03-30",
            mode="rolling",
        )

        data = json.loads(records_file.read_text())
        assert len(data["models"]) == 2
        assert "m1@rolling" in data["models"]
        assert "m2@rolling" in data["models"]

    def test_timestamp_present(self, monkeypatch, tmp_path):
        from quantpits.scripts.rolling.orchestration import save_rolling_records
        from quantpits.utils import train_utils

        records_file = tmp_path / "records_ts.json"
        monkeypatch.setattr(train_utils, "RECORD_OUTPUT_FILE", str(records_file))

        save_rolling_records(
            combined_records={"m1": "rid_001"},
            combined_exp_name="test_exp",
            anchor_date="2023-03-30",
            mode="rolling",
        )

        data = json.loads(records_file.read_text())
        assert "timestamp" in data
        assert "experiment_name" in data
        assert data["rolling_experiment_name"] == "test_exp"
