"""
Tests for rolling/backtest.py — backtest execution, portfolio metrics printing,
and indicator printing.

Coverage targets:
  - _print_portfolio_metrics: empty/non-dict input, missing 'return' column,
    short series, annualization math, all sections
  - _print_indicators: None input, empty DataFrame, dict input, missing columns
  - run_combined_backtest: empty predictions, missing model, successful flow,
    MLflow log failure
  - run_backtest_only: no records file, empty records, no matching models,
    mode filtering, mock detection branch
"""

import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock


# =========================================================================
# _print_portfolio_metrics
# =========================================================================

class TestPrintPortfolioMetrics:
    """Tests for _print_portfolio_metrics()."""

    def _make_ts_df(self, n=52, seed=42):
        """Create a realistic backtest time-series DataFrame with bench/return/cost."""
        rng = np.random.RandomState(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            "bench": rng.normal(0.001, 0.02, n),
            "return": rng.normal(0.002, 0.025, n),
            "cost": rng.uniform(0.0001, 0.001, n),
        }, index=dates)
        return df

    def test_non_dict_input_returns_none(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        result = _print_portfolio_metrics(None, "test_model", "week")
        assert result is None

    def test_empty_dict_returns_none_tuple(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        result = _print_portfolio_metrics({}, "test_model", "week")
        # Empty dict → no iterations → returns (None, None, None, None)
        assert result == (None, None, None, None)

    def test_dict_with_non_tuple_value(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        result = _print_portfolio_metrics({"1week": "not_a_tuple"}, "test_model", "week")
        # Not a tuple → continue → returns (None, None, None, None)
        assert result == (None, None, None, None)

    def test_dict_with_empty_tuple(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        result = _print_portfolio_metrics({"1week": ()}, "test_model", "week")
        # Empty tuple → len < 1 → continue → returns (None, None, None, None)
        assert result == (None, None, None, None)

    def test_ts_df_none(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        result = _print_portfolio_metrics({"1week": (None,)}, "test_model", "week")
        # ts_df is None → continue → returns (None, None, None, None)
        assert result == (None, None, None, None)

    def test_ts_df_empty(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        result = _print_portfolio_metrics(
            {"1week": (pd.DataFrame(),)}, "test_model", "week")
        # Empty DataFrame → continue → returns (None, None, None, None)
        assert result == (None, None, None, None)

    def test_ts_df_missing_return_column(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        df = pd.DataFrame({"bench": [0.01, -0.01]})
        result = _print_portfolio_metrics({"1week": (df,)}, "test_model", "week")
        # Missing 'return' column → continue → returns (None, None, None, None)
        assert result == (None, None, None, None)

    def test_short_series_less_than_2(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        # Need at least 2 rows + cost column or else excess-with-cost section fails
        df = pd.DataFrame({
            "bench": [0.01, 0.02],
            "return": [0.02, 0.01],
            "cost": [0.001, 0.001],
        })
        result = _print_portfolio_metrics({"1week": (df,)}, "test_model", "week")
        assert result is not None

    def test_successful_weekly_freq(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        df = self._make_ts_df(52)
        result = _print_portfolio_metrics({"1week": (df,)}, "test_model", "week")
        assert result is not None
        ann_ret, max_dd, excess, ir = result
        assert isinstance(ann_ret, float)
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # max drawdown is negative or zero

    def test_successful_daily_freq(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        rng = np.random.RandomState(42)
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        df = pd.DataFrame({
            "bench": rng.normal(0.0005, 0.01, 252),
            "return": rng.normal(0.001, 0.015, 252),
            "cost": rng.uniform(0.0001, 0.0005, 252),
        }, index=dates)
        result = _print_portfolio_metrics({"1day": (df,)}, "test_model", "day")
        assert result is not None
        ann_ret, max_dd, excess, ir = result
        assert isinstance(ann_ret, float)

    def test_multiple_freq_keys(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        df = self._make_ts_df(52)
        # Only the last key's return value is returned
        result = _print_portfolio_metrics(
            {"1week": (df,), "1month": (df,)}, "test_model", "week")
        assert result is not None

    def test_zero_std_dev(self):
        """When std is zero, IR is zero (no crash)."""
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        dates = pd.date_range("2024-01-01", periods=10, freq="W")
        constant = 0.01
        df = pd.DataFrame({
            "bench": [constant] * 10,
            "return": [constant] * 10,
            "cost": [0.0] * 10,
        }, index=dates)
        # All std ≈ 0 → IR = 0, max_dd = 0
        result = _print_portfolio_metrics({"1week": (df,)}, "test_model", "week")
        # With constant data, std=0, IR should be 0.0
        ann_ret, max_dd, excess, ir = result
        assert ir == 0.0

    def test_with_cost_column_present(self):
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        df = self._make_ts_df(52)
        # cost column is already in _make_ts_df
        result = _print_portfolio_metrics({"1week": (df,)}, "test_model", "week")
        assert result is not None

    def test_tuple_with_two_elements(self):
        """Tuple format: (ts_df, positions_df)."""
        from quantpits.scripts.rolling.backtest import _print_portfolio_metrics
        ts_df = self._make_ts_df(52)
        pos_df = pd.DataFrame({"stock1": [100] * 52})
        result = _print_portfolio_metrics(
            {"1week": (ts_df, pos_df)}, "test_model", "week")
        assert result is not None


# =========================================================================
# _print_indicators
# =========================================================================

class TestPrintIndicators:
    """Tests for _print_indicators()."""

    def test_none_input_returns_none(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        result = _print_indicators(None, "test_model", "week")
        assert result is None

    def test_empty_dataframe(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        result = _print_indicators(pd.DataFrame(), "test_model", "week")
        assert result is None

    def test_dataframe_with_ffr_pa_pos(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        dates = pd.date_range("2024-01-01", periods=10, freq="W")
        df = pd.DataFrame({
            "ffr": [0.5] * 10,
            "pa": [0.3] * 10,
            "pos": [100] * 10,
            "deal_amount": [1e6] * 10,
            "value": [1e7] * 10,
            "count": [50] * 10,
        }, index=dates)
        # Should not raise
        _print_indicators(df, "test_model", "week")

    def test_dict_input_with_tuple(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        dates = pd.date_range("2024-01-01", periods=10, freq="W")
        df = pd.DataFrame({"ffr": [0.5] * 10, "pa": [0.3] * 10, "pos": [100] * 10})
        result = _print_indicators({"1week": (df,)}, "test_model", "week")
        # No return value, just prints
        assert result is None

    def test_dict_input_with_none_df(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        result = _print_indicators({"1week": (None,)}, "test_model", "week")
        assert result is None

    def test_dict_input_empty_df(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        result = _print_indicators({"1week": (pd.DataFrame(),)}, "test_model", "week")
        assert result is None

    def test_dataframe_missing_columns(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _print_indicators(df, "test_model", "week")
        assert result is None

    def test_daily_freq(self):
        from quantpits.scripts.rolling.backtest import _print_indicators
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        df = pd.DataFrame({
            "ffr": np.random.normal(0.5, 0.1, 252),
            "pa": np.random.normal(0.3, 0.1, 252),
            "pos": [100] * 252,
        }, index=dates)
        _print_indicators(df, "test_model", "day")


# =========================================================================
# run_combined_backtest
# =========================================================================

class TestRunCombinedBacktest:
    """Tests for run_combined_backtest()."""

    @staticmethod
    def _make_ts_df(n=52, seed=42):
        """Create a realistic backtest time-series DataFrame with bench/return/cost."""
        rng = np.random.RandomState(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            "bench": rng.normal(0.001, 0.02, n),
            "return": rng.normal(0.002, 0.025, n),
            "cost": rng.uniform(0.0001, 0.001, n),
        }, index=dates)
        return df

    def _make_pred_df(self):
        """Create a realistic combined prediction DataFrame."""
        dates = pd.date_range("2024-01-01", periods=20, freq="W")
        instruments = ["SH600000", "SH600001", "SH600002"]
        idx = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"])
        df = pd.DataFrame({"score": np.random.RandomState(42).normal(0, 1, len(idx))},
                          index=idx)
        return df

    def test_empty_combined_records(self):
        from quantpits.scripts.rolling.backtest import run_combined_backtest
        params_base = {"freq": "week", "benchmark": "SH000300"}
        # Should not raise — just nothing to do
        run_combined_backtest(
            model_names=[],
            combined_records={},
            combined_exp_name="test_exp",
            params_base=params_base,
        )

    def test_model_not_in_combined_records(self):
        from quantpits.scripts.rolling.backtest import run_combined_backtest
        params_base = {"freq": "week", "benchmark": "SH000300"}
        run_combined_backtest(
            model_names=["missing_model"],
            combined_records={},
            combined_exp_name="test_exp",
            params_base=params_base,
        )

    def test_empty_predictions(self):
        """pred.pkl is None or empty → skipped."""
        from quantpits.scripts.rolling.backtest import run_combined_backtest

        params_base = {"freq": "week", "benchmark": "SH000300"}

        recorder = MagicMock()
        recorder.load_object.return_value = None  # empty pred
        recorder.id = "rid_001"

        R_mock = MagicMock()
        R_mock.get_recorder.return_value = recorder

        with patch("quantpits.utils.strategy.load_strategy_config", return_value={}):
            with patch("quantpits.utils.strategy.get_backtest_config",
                      return_value={"account": 1e8, "exchange_kwargs": {}}):
                with patch("qlib.workflow.R", R_mock):
                    run_combined_backtest(
                        model_names=["test_model"],
                        combined_records={"test_model": "rid_001"},
                        combined_exp_name="test_exp",
                        params_base=params_base,
                    )

    def test_successful_backtest_flow(self):
        """Full successful backtest with all mocks."""
        from quantpits.scripts.rolling.backtest import run_combined_backtest

        params_base = {"freq": "week", "benchmark": "SH000300"}

        pred_df = self._make_pred_df()

        recorder = MagicMock()
        recorder.load_object.return_value = pred_df
        recorder.id = "rid_001"
        recorder.log_metrics = MagicMock()
        recorder.save_objects = MagicMock()

        R_mock = MagicMock()
        R_mock.get_recorder.return_value = recorder

        # Mock backtest return: (portfolio_metrics, indicators)
        mock_bt_return = (
            {"1week": (self._make_ts_df(n=20), pd.DataFrame())},
            pd.DataFrame({"ffr": [0.5], "pa": [0.3], "pos": [100]}),
        )

        with patch("quantpits.utils.strategy.load_strategy_config", return_value={}):
            with patch("quantpits.utils.strategy.get_backtest_config",
                      return_value={"account": 1e8, "exchange_kwargs": {}}):
                with patch("quantpits.utils.strategy.create_backtest_strategy",
                          return_value=MagicMock()):
                    with patch("qlib.workflow.R", R_mock):
                        with patch("qlib.backtest.backtest", return_value=mock_bt_return):
                            with patch("qlib.backtest.executor.SimulatorExecutor"):
                                run_combined_backtest(
                                    model_names=["test_model"],
                                    combined_records={"test_model": "rid_001"},
                                    combined_exp_name="test_exp",
                                    params_base=params_base,
                                )

    def test_backtest_exception_caught(self):
        """Exception inside backtest loop is caught and doesn't crash."""
        from quantpits.scripts.rolling.backtest import run_combined_backtest

        params_base = {"freq": "week", "benchmark": "SH000300"}

        R_mock = MagicMock()
        R_mock.get_recorder.side_effect = RuntimeError("MLflow error")

        with patch("quantpits.utils.strategy.load_strategy_config", return_value={}):
            with patch("quantpits.utils.strategy.get_backtest_config",
                      return_value={"account": 1e8, "exchange_kwargs": {}}):
                with patch("qlib.workflow.R", R_mock):
                    # Should not raise
                    run_combined_backtest(
                        model_names=["test_model"],
                        combined_records={"test_model": "rid_001"},
                        combined_exp_name="test_exp",
                        params_base=params_base,
                    )

    def test_mlflow_log_exception_handled(self):
        """MLflow log_metrics/save_objects failure is caught."""
        from quantpits.scripts.rolling.backtest import run_combined_backtest

        params_base = {"freq": "week", "benchmark": "SH000300"}

        pred_df = self._make_pred_df()
        recorder = MagicMock()
        recorder.load_object.return_value = pred_df
        recorder.id = "rid_001"
        recorder.log_metrics.side_effect = RuntimeError("log failed")
        recorder.save_objects.side_effect = RuntimeError("save failed")

        R_mock = MagicMock()
        R_mock.get_recorder.return_value = recorder

        mock_bt_return = (
            {"1week": (self._make_ts_df(n=20), pd.DataFrame())},
            pd.DataFrame({"ffr": [0.5], "pa": [0.3], "pos": [100]}),
        )

        with patch("quantpits.utils.strategy.load_strategy_config", return_value={}):
            with patch("quantpits.utils.strategy.get_backtest_config",
                      return_value={"account": 1e8, "exchange_kwargs": {}}):
                with patch("quantpits.utils.strategy.create_backtest_strategy",
                          return_value=MagicMock()):
                    with patch("qlib.workflow.R", R_mock):
                        with patch("qlib.backtest.backtest", return_value=mock_bt_return):
                            with patch("qlib.backtest.executor.SimulatorExecutor"):
                                run_combined_backtest(
                                    model_names=["test_model"],
                                    combined_records={"test_model": "rid_001"},
                                    combined_exp_name="test_exp",
                                    params_base=params_base,
                                )

    def test_portfolio_metrics_single_element_tuple(self):
        """Portfolio metrics tuple with only 1 element (no positions)."""
        from quantpits.scripts.rolling.backtest import run_combined_backtest

        params_base = {"freq": "week", "benchmark": "SH000300"}
        pred_df = self._make_pred_df()

        recorder = MagicMock()
        recorder.load_object.return_value = pred_df
        recorder.id = "rid_001"
        recorder.log_metrics = MagicMock()
        recorder.save_objects = MagicMock()

        R_mock = MagicMock()
        R_mock.get_recorder.return_value = recorder

        ts_df = self._make_ts_df(20)
        mock_bt_return = (
            {"1week": (ts_df,)},  # single-element tuple
            pd.DataFrame({"ffr": [0.5], "pa": [0.3], "pos": [100]}),
        )

        with patch("quantpits.utils.strategy.load_strategy_config", return_value={}):
            with patch("quantpits.utils.strategy.get_backtest_config",
                      return_value={"account": 1e8, "exchange_kwargs": {}}):
                with patch("quantpits.utils.strategy.create_backtest_strategy",
                          return_value=MagicMock()):
                    with patch("qlib.workflow.R", R_mock):
                        with patch("qlib.backtest.backtest", return_value=mock_bt_return):
                            with patch("qlib.backtest.executor.SimulatorExecutor"):
                                run_combined_backtest(
                                    model_names=["test_model"],
                                    combined_records={"test_model": "rid_001"},
                                    combined_exp_name="test_exp",
                                    params_base=params_base,
                                )


# =========================================================================
# run_backtest_only
# =========================================================================

class TestRunBacktestOnly:
    """Tests for run_backtest_only()."""

    def _make_args(self, **kwargs):
        """Create a minimal CLI args namespace."""
        defaults = {
            "models": None,
            "all_enabled": False,
        }
        defaults.update(kwargs)
        return type("Args", (), defaults)()

    def _make_targets(self):
        return {"model_a": {"algorithm": "gru"}, "model_b": {"algorithm": "mlp"}}

    def test_no_records_file(self, tmp_path):
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils
        import quantpits.scripts.rolling.backtest as bt

        args = self._make_args()
        targets = self._make_targets()

        with patch.object(train_utils, "RECORD_OUTPUT_FILE",
                          str(tmp_path / "nonexistent.json")):
            result = run_backtest_only(args, targets)
        assert result == {
            "status": "failed",
            "reason_code": "rolling_backtest_precondition_failed",
            "message": "找不到有效的 latest_train_records.json 或内容为空。",
            "did_execute": False,
        }

    def test_empty_records(self, tmp_path):
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils
        import quantpits.scripts.rolling.backtest as bt

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({"models": {}}))

        args = self._make_args()
        targets = self._make_targets()
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            result = run_backtest_only(args, targets, params_base)
        assert result["status"] == "failed"
        assert result["reason_code"] == "rolling_backtest_precondition_failed"
        assert result["did_execute"] is False

    def test_records_without_models_key(self, tmp_path):
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({"experiment_name": "test"}))

        args = self._make_args()
        targets = self._make_targets()
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            result = run_backtest_only(args, targets, params_base)
        assert result["status"] == "failed"
        assert result["reason_code"] == "rolling_backtest_precondition_failed"

    def test_no_requested_rolling_family(self, tmp_path):
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({
            "models": {"model_a@cpcv_rolling": "rid_001"},
        }))
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            result = run_backtest_only(
                self._make_args(), self._make_targets(), params_base,
                mode="rolling",
            )
        assert result["status"] == "failed"
        assert result["reason_code"] == "rolling_backtest_precondition_failed"
        assert result["did_execute"] is False

    def test_no_matching_models(self, tmp_path):
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({
            "experiment_name": "test_exp",
            "models": {"other_model@rolling": "rid_001"},
        }))

        args = self._make_args()
        targets = {"model_a": {"algorithm": "gru"}}  # not in records
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            result = run_backtest_only(args, targets, params_base)
        assert result["status"] == "failed"
        assert result["reason_code"] == "rolling_backtest_precondition_failed"
        assert result["did_execute"] is False

    def test_slide_mode(self, tmp_path):
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({
            "experiment_name": "test_exp",
            "rolling_experiment_name": "roll_exp",
            "models": {"model_a@rolling": "rid_001", "model_b@rolling": "rid_002"},
        }))

        args = self._make_args()
        targets = self._make_targets()
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            with patch("quantpits.scripts.rolling.backtest.run_combined_backtest") as mock_rcb:
                result = run_backtest_only(
                    args, targets, params_base, mode="rolling",
                )
                # Should call run_combined_backtest
                mock_rcb.assert_called_once()
        assert result["status"] == "success"
        assert result["reason_code"] == "legacy_partial_visibility"
        assert result["did_execute"] is True

    def test_cpcv_mode(self, tmp_path):
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({
            "experiment_name": "test_exp",
            "cpcv_rolling_experiment_name": "cpcv_exp",
            "models": {"model_a@cpcv_rolling": "rid_001"},
        }))

        args = self._make_args()
        targets = self._make_targets()
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            with patch("quantpits.scripts.rolling.backtest.run_combined_backtest") as mock_rcb:
                result = run_backtest_only(
                    args, targets, params_base, mode="cpcv_rolling",
                )
                mock_rcb.assert_called_once()
        assert result["status"] == "success"

    def test_params_base_none(self, tmp_path):
        """When params_base is None, it's auto-detected."""
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({
            "experiment_name": "test_exp",
            "models": {"model_a@rolling": "rid_001"},
        }))

        args = self._make_args()
        targets = self._make_targets()

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            with patch("quantpits.scripts.rolling.backtest.run_combined_backtest") as mock_rcb:
                # Inject a mock rolling_train with get_base_params into sys.modules
                fake_rt = MagicMock()
                fake_rt.get_base_params.return_value = {"freq": "week", "benchmark": "SH000300"}
                fake_rt.run_combined_backtest = None  # not a Mock, so no mock detection

                with patch.dict(sys.modules, {"rolling_train": fake_rt}):
                    run_backtest_only(args, targets, params_base=None)

    def test_mock_detection_branch(self, tmp_path):
        """When rolling_train has a Mock run_combined_backtest, it's used directly."""
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({
            "experiment_name": "test_exp",
            "models": {"model_a@rolling": "rid_001"},
        }))

        args = self._make_args()
        targets = self._make_targets()
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            fake_rt = MagicMock()
            fake_rt.get_base_params.return_value = params_base
            fake_rt.run_combined_backtest = Mock()  # triggers mock detection

            with patch.dict(sys.modules, {"rolling_train": fake_rt}):
                run_backtest_only(args, targets, params_base)

    def test_targets_by_bare_name(self, tmp_path):
        """Targets contain bare names, records contain model@mode keys."""
        from quantpits.scripts.rolling.backtest import run_backtest_only
        from quantpits.utils import train_utils

        records_file = tmp_path / "records.json"
        records_file.write_text(json.dumps({
            "experiment_name": "test_exp",
            "models": {"model_a@rolling": "rid_001"},
        }))

        args = self._make_args()
        targets = {"model_a": {"algorithm": "gru"}}  # bare name matches model_a@rolling
        params_base = {"freq": "week", "benchmark": "SH000300"}

        with patch.object(train_utils, "RECORD_OUTPUT_FILE", str(records_file)):
            with patch("quantpits.scripts.rolling.backtest.run_combined_backtest") as mock_rcb:
                run_backtest_only(args, targets, params_base)
                mock_rcb.assert_called_once()
