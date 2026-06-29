"""
Coverage tests for rolling/strategy_slide.py (93% → target 98%+).

Coverage targets (line numbers refer to strategy_slide.py):
  - train_window: mock detection branch in train_window_isolated
  - train_window_isolated: subprocess + mock dual paths
  - predict_latest: exception paths
  - repair_truncated: boundary conditions
"""

import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock


# =========================================================================
# parse_step_to_relativedelta
# =========================================================================

class TestParseStepToRelativedelta:
    """Tests for parse_step_to_relativedelta — already well-covered, edge cases only."""

    def test_uppercase_input(self):
        from quantpits.scripts.rolling.strategy_slide import parse_step_to_relativedelta
        rd = parse_step_to_relativedelta("6M")
        assert rd.months == 6

    def test_lowercase_input(self):
        from quantpits.scripts.rolling.strategy_slide import parse_step_to_relativedelta
        rd = parse_step_to_relativedelta("1y")
        assert rd.years == 1

    def test_with_spaces(self):
        from quantpits.scripts.rolling.strategy_slide import parse_step_to_relativedelta
        rd = parse_step_to_relativedelta("  3m  ")
        assert rd.months == 3

    def test_invalid_character(self):
        from quantpits.scripts.rolling.strategy_slide import parse_step_to_relativedelta
        with pytest.raises(ValueError, match="Invalid step format"):
            parse_step_to_relativedelta("3D")


# =========================================================================
# train_window_isolated — mock detection + subprocess
# =========================================================================

class TestTrainWindowIsolatedSlide:
    """Tests for train_window_isolated() in slide strategy."""

    def test_mock_detection_branch(self):
        """When rolling_train.train_window_model is a Mock, use it directly."""
        from quantpits.scripts.rolling.strategy_slide import train_window_isolated

        mock_fn = Mock(return_value={
            "success": True, "record_id": "mock_rid",
            "performance": {"IC_Mean": 0.05},
        })

        fake_rt = MagicMock()
        fake_rt.train_window_model = mock_fn

        with patch.dict(sys.modules, {"rolling_train": fake_rt}):
            result = train_window_isolated(
                qlib_config=None,
                model_name="test_model",
                yaml_file="model.yaml",
                window={
                    "window_idx": 0,
                    "train_start": "2020-01-01", "train_end": "2022-12-31",
                    "valid_start": "2023-01-01", "valid_end": "2023-12-31",
                    "test_start": "2024-01-01", "test_end": "2024-03-31",
                },
                params_base={"freq": "week"},
                experiment_name="test_exp",
            )

        assert result["success"] is True
        mock_fn.assert_called_once()

    def test_subprocess_path(self):
        """Normal path spawns a subprocess."""
        from quantpits.scripts.rolling.strategy_slide import train_window_isolated

        expected = {"success": True, "record_id": "sp_rid",
                    "performance": {"IC_Mean": 0.05}}
        future = MagicMock()
        future.result.return_value = expected
        executor = MagicMock()
        executor.__enter__.return_value.submit.return_value = future
        executor.__exit__.return_value = None

        sys.modules.pop("rolling_train", None)
        with patch("concurrent.futures.ProcessPoolExecutor", return_value=executor):
            with patch("multiprocessing.get_context", return_value="spawn"):
                result = train_window_isolated(
                    qlib_config={"provider_uri": "test"},
                    model_name="test_model",
                    yaml_file="model.yaml",
                    window={
                        "window_idx": 0,
                        "train_start": "2020-01-01", "train_end": "2022-12-31",
                        "valid_start": "2023-01-01", "valid_end": "2023-12-31",
                        "test_start": "2024-01-01", "test_end": "2024-03-31",
                    },
                    params_base={"freq": "week"},
                    experiment_name="test_exp",
                )

        assert result["success"] is True


# =========================================================================
# predict_latest — edge cases
# =========================================================================

class TestPredictLatestSlide:
    """Tests for predict_latest() in slide strategy."""

    def _make_state(self, record_ids):
        state = MagicMock()
        state.get_completed_record_ids.return_value = record_ids
        return state

    def _make_windows(self):
        return [
            {
                "window_idx": 0,
                "train_start": "2020-01-01", "train_end": "2022-12-31",
                "valid_start": "2023-01-01", "valid_end": "2023-12-31",
                "test_start": "2024-01-01", "test_end": "2024-03-31",
            },
            {
                "window_idx": 1,
                "train_start": "2020-04-01", "train_end": "2023-03-31",
                "valid_start": "2023-04-01", "valid_end": "2024-03-31",
                "test_start": "2024-04-01", "test_end": "2024-06-30",
            },
        ]

    def test_mock_detection_branch(self):
        from quantpits.scripts.rolling.strategy_slide import predict_latest

        mock_fn = Mock(return_value=pd.DataFrame({"score": [0.1]}))
        fake_rt = MagicMock()
        fake_rt.predict_with_latest_model = mock_fn

        with patch.dict(sys.modules, {"rolling_train": fake_rt}):
            result = predict_latest(
                model_name="test_model",
                model_info={"yaml_file": "model.yaml"},
                state=self._make_state([{"window_idx": 0, "record_id": "rid_000"}]),
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
                anchor_date="2025-01-01",
                windows=self._make_windows(),
                allow_stale_predict=True,
            )

        mock_fn.assert_called_once()
        assert result is not None

    def test_no_completions(self):
        from quantpits.scripts.rolling.strategy_slide import predict_latest

        state = MagicMock()
        state.get_completed_record_ids.return_value = []

        sys.modules.pop("rolling_train", None)
        result = predict_latest(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            state=state,
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
            anchor_date="2025-01-01",
            windows=self._make_windows(),
        )

        assert result is None

    def test_window_not_found(self):
        from quantpits.scripts.rolling.strategy_slide import predict_latest

        sys.modules.pop("rolling_train", None)
        result = predict_latest(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            state=self._make_state([{"window_idx": 99, "record_id": "rid_099"}]),
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
            anchor_date="2025-01-01",
            windows=self._make_windows(),
        )

        assert result is None

    def test_gap_without_allow_stale(self):
        from quantpits.scripts.rolling.strategy_slide import predict_latest

        # Only window 0 completed, anchor far ahead, gap exists
        sys.modules.pop("rolling_train", None)
        result = predict_latest(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            state=self._make_state([{"window_idx": 0, "record_id": "rid_000"}]),
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
            anchor_date="2025-01-01",
            windows=self._make_windows(),
            allow_stale_predict=False,
        )

        assert result is None

    def test_all_up_to_date(self):
        from quantpits.scripts.rolling.strategy_slide import predict_latest

        windows = self._make_windows()
        sys.modules.pop("rolling_train", None)
        result = predict_latest(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            state=self._make_state([{"window_idx": 1, "record_id": "rid_001"}]),
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
            anchor_date=windows[-1]["test_end"],
            windows=windows,
        )

        assert result is None


# =========================================================================
# repair_truncated — edge cases
# =========================================================================

class TestRepairTruncatedSlide:
    """Tests for repair_truncated() in slide strategy."""

    def _make_window_map(self):
        return {
            0: {
                "window_idx": 0,
                "train_start": "2020-01-01", "train_end": "2022-12-31",
                "valid_start": "2023-01-01", "valid_end": "2023-12-31",
                "test_start": "2024-01-01", "test_end": "2024-03-31",
            },
        }

    def _make_pred(self, start="2024-01-01", n=13, freq="W"):
        dates = pd.date_range(start, periods=n, freq=freq)
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        return pd.DataFrame({"score": range(n)}, index=idx)

    def test_window_not_in_map(self):
        from quantpits.scripts.rolling.strategy_slide import repair_truncated
        result = repair_truncated(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            comp={"window_idx": 99, "record_id": "rid_099"},
            window_map=self._make_window_map(),
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
        )
        pred, repaired = result
        assert pred is None
        assert repaired is False

    def test_prediction_not_truncated(self, mock_qlib_R):
        from quantpits.scripts.rolling.strategy_slide import repair_truncated

        # pred covers the full test range including test_end (2024-03-31)
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="W")
        # Ensure last date is within 1 day of test_end
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        full_pred = pd.DataFrame({"score": range(len(dates))}, index=idx)

        mock_qlib_R["recorder"].load_object.return_value = full_pred

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = repair_truncated(
                model_name="test_model",
                model_info={"yaml_file": "model.yaml"},
                comp={"window_idx": 0, "record_id": "rid_000"},
                window_map=self._make_window_map(),
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
            )

        pred, repaired = result
        assert repaired is False
        assert pred is not None

    def test_cannot_load_model(self, mock_qlib_R):
        from quantpits.scripts.rolling.strategy_slide import repair_truncated

        # Prediction is truncated (early dates, ends before test_end)
        pred_truncated = self._make_pred(start="2024-01-01", n=4)
        # load_object: first = pred.pkl, second = model.pkl (fails)
        mock_qlib_R["recorder"].load_object.side_effect = [
            pred_truncated,
            Exception("Model file corrupt"),
        ]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = repair_truncated(
                model_name="test_model",
                model_info={"yaml_file": "model.yaml"},
                comp={"window_idx": 0, "record_id": "rid_000"},
                window_map=self._make_window_map(),
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
            )

        pred, repaired = result
        assert repaired is False  # repair failed, original returned
