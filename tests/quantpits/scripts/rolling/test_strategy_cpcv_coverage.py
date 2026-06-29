"""
Coverage tests for rolling/strategy_cpcv.py (0% → target 50%+).

Coverage targets (line numbers refer to strategy_cpcv.py):
  - _make_fold_config: n_test_groups=0, start_time override, all keys (lines 37-55)
  - generate_windows: window gen, test_end capped, gap, ValueError on empty cfg (lines 62-136)
  - train_window: YAML not found, no folds, K-fold loop, IC computation, ensemble avg,
    record generation, GPU cleanup (lines 143-458)
  - _train_in_subprocess: delegates to train_window (lines 465-488)
  - train_window_isolated: mock detection, subprocess path (lines 491-528)
  - predict_latest: no completions, latest not found, gap detection, stale skip,
    fold model loading, ensemble prediction (lines 535-673)
  - repair_truncated: not found, not truncated, repair, no fold info, load failure (lines 680-779)
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock, PropertyMock


# =========================================================================
# _make_fold_config
# =========================================================================

class TestMakeFoldConfig:
    """Tests for _make_fold_config() — pure function, no mocking needed."""

    def test_n_test_groups_forced_to_zero(self):
        from quantpits.scripts.rolling.strategy_cpcv import _make_fold_config
        cfg = _make_fold_config("2020-01-01", {
            "n_groups": 10,
            "n_test_groups": 5,  # should be overridden
            "n_val_groups": 2,
            "purge_steps": 5,
            "embargo_steps": 2,
        })
        assert cfg["purged_cv"]["n_test_groups"] == 0
        assert cfg["purged_cv"]["n_groups"] == 10
        assert cfg["purged_cv"]["n_val_groups"] == 2
        assert cfg["start_time"] == "2020-01-01"

    def test_start_time_overridden(self):
        from quantpits.scripts.rolling.strategy_cpcv import _make_fold_config
        cfg = _make_fold_config("2019-06-15", {
            "n_groups": 5, "n_test_groups": 2,
            "n_val_groups": 1, "purge_steps": 3, "embargo_steps": 1,
        })
        assert cfg["start_time"] == "2019-06-15"

    def test_all_keys_present(self):
        from quantpits.scripts.rolling.strategy_cpcv import _make_fold_config
        cfg = _make_fold_config("2020-01-01", {
            "n_groups": 8, "n_test_groups": 3,
            "n_val_groups": 1, "purge_steps": 4, "embargo_steps": 1,
        })
        assert "start_time" in cfg
        assert "purged_cv" in cfg
        for key in ["n_groups", "n_test_groups", "n_val_groups", "purge_steps", "embargo_steps"]:
            assert key in cfg["purged_cv"]


# =========================================================================
# generate_windows
# =========================================================================

class TestGenerateWindows:
    """Tests for generate_windows()."""

    def _mock_folds(self):
        return {
            "folds": [
                {
                    "train_segments": [("2020-01-01", "2022-03-31"), ("2022-05-01", "2022-12-31")],
                    "valid_start_time": "2022-04-01",
                    "valid_end_time": "2022-04-30",
                },
                {
                    "train_segments": [("2020-01-01", "2022-12-01")],
                    "valid_start_time": "2022-12-02",
                    "valid_end_time": "2022-12-31",
                },
            ]
        }

    def test_empty_cpcv_cfg_raises(self):
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        with pytest.raises(ValueError, match="CPCV strategy requires cpcv_cfg"):
            generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="3M",
                anchor_date="2024-12-31",
                cpcv_cfg=None,
            )

    def test_basic_3m_step_windows(self):
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        cpcv_cfg = {
            "n_groups": 10, "n_test_groups": 0,
            "n_val_groups": 2, "purge_steps": 5, "embargo_steps": 2,
        }
        with patch("quantpits.utils.train_utils.compute_cpcv_folds",
                   return_value=self._mock_folds()):
            windows = generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="3M",
                anchor_date="2024-12-31",
                cpcv_cfg=cpcv_cfg,
                freq="week",
            )

        assert len(windows) >= 4

        # Window 0: train domain = 3 years before test_start
        w0 = windows[0]
        assert w0["window_idx"] == 0
        assert w0["train_start"] == "2020-01-01"
        assert w0["test_start"] == "2023-01-01"
        # 3 months from test_start minus 1 day: 2023-01-01 + 3M - 1d
        assert w0["test_end"] in ("2023-03-30", "2023-03-31")  # varies by month boundaries
        assert "cpcv_folds" in w0
        assert len(w0["cpcv_folds"]) == 2

    def test_last_window_capped_by_anchor(self):
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        cpcv_cfg = {
            "n_groups": 5, "n_test_groups": 0,
            "n_val_groups": 1, "purge_steps": 3, "embargo_steps": 1,
        }
        with patch("quantpits.utils.train_utils.compute_cpcv_folds",
                   return_value=self._mock_folds()):
            windows = generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="3M",
                anchor_date="2024-02-15",  # mid-quarter — last window capped
                cpcv_cfg=cpcv_cfg,
            )

        assert len(windows) > 0
        last = windows[-1]
        assert pd.Timestamp(last["test_end"]) <= pd.Timestamp("2024-02-15")

    def test_single_window_when_anchor_equals_test_start(self):
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        cpcv_cfg = {
            "n_groups": 5, "n_test_groups": 0,
            "n_val_groups": 1, "purge_steps": 3, "embargo_steps": 1,
        }
        with patch("quantpits.utils.train_utils.compute_cpcv_folds",
                   return_value=self._mock_folds()):
            windows = generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="3M",
                anchor_date="2023-01-01",  # exactly at first test_start
                cpcv_cfg=cpcv_cfg,
            )

        assert len(windows) == 1
        w0 = windows[0]
        assert w0["test_start"] == "2023-01-01"
        assert w0["test_end"] == "2023-01-01"

    def test_no_windows_when_anchor_too_early(self):
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        cpcv_cfg = {
            "n_groups": 5, "n_test_groups": 0,
            "n_val_groups": 1, "purge_steps": 3, "embargo_steps": 1,
        }
        with patch("quantpits.utils.train_utils.compute_cpcv_folds",
                   return_value=self._mock_folds()):
            windows = generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="3M",
                anchor_date="2022-12-31",  # before first test_start
                cpcv_cfg=cpcv_cfg,
            )

        assert len(windows) == 0

    def test_6m_step(self):
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        cpcv_cfg = {
            "n_groups": 6, "n_test_groups": 0,
            "n_val_groups": 2, "purge_steps": 4, "embargo_steps": 1,
        }
        with patch("quantpits.utils.train_utils.compute_cpcv_folds",
                   return_value=self._mock_folds()):
            windows = generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="6M",
                anchor_date="2024-12-31",
                cpcv_cfg=cpcv_cfg,
            )

        assert len(windows) >= 2
        # First test_end should be ~6 months after test_start
        w0 = windows[0]
        assert w0["test_start"] == "2023-01-01"
        # 6 months minus 1 day — varies by month boundaries
        assert "2023-06-29" <= w0["test_end"] <= "2023-06-30"

    def test_train_domain_always_fixed_length(self):
        """Train domain is always exactly train_years long."""
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        cpcv_cfg = {
            "n_groups": 5, "n_test_groups": 0,
            "n_val_groups": 1, "purge_steps": 3, "embargo_steps": 1,
        }
        with patch("quantpits.utils.train_utils.compute_cpcv_folds",
                   return_value=self._mock_folds()):
            windows = generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="1Y",
                anchor_date="2026-12-31",
                cpcv_cfg=cpcv_cfg,
            )

        for w in windows:
            train_start = pd.Timestamp(w["train_start"])
            train_end = pd.Timestamp(w["train_end"])
            # 3 years minus 1 day
            expected_end = train_start + pd.DateOffset(years=3) - pd.DateOffset(days=1)
            assert train_end == expected_end, \
                f"Window {w['window_idx']}: train=[{w['train_start']}, {w['train_end']}], expected_end={expected_end.date()}"

    def test_windows_non_overlapping_test(self):
        """Test segments must be non-overlapping."""
        from quantpits.scripts.rolling.strategy_cpcv import generate_windows
        cpcv_cfg = {
            "n_groups": 5, "n_test_groups": 0,
            "n_val_groups": 1, "purge_steps": 3, "embargo_steps": 1,
        }
        with patch("quantpits.utils.train_utils.compute_cpcv_folds",
                   return_value=self._mock_folds()):
            windows = generate_windows(
                rolling_start="2020-01-01",
                train_years=3,
                test_step="3M",
                anchor_date="2025-12-31",
                cpcv_cfg=cpcv_cfg,
            )

        assert len(windows) >= 2
        for i in range(len(windows) - 1):
            curr_end = pd.Timestamp(windows[i]["test_end"])
            next_start = pd.Timestamp(windows[i + 1]["test_start"])
            assert curr_end < next_start, \
                f"Window {i} test_end={curr_end.date()} >= Window {i+1} test_start={next_start.date()}"


# =========================================================================
# train_window
# =========================================================================

class TestTrainWindow:
    """Tests for train_window() — the core CPCV training function."""

    def _make_window(self, widx=0):
        return {
            "window_idx": widx,
            "train_start": "2020-01-01",
            "train_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-03-30",
            "cpcv_folds": [
                {
                    "train_segments": [("2020-01-01", "2022-04-30")],
                    "valid_start_time": "2022-05-01",
                    "valid_end_time": "2022-12-31",
                },
                {
                    "train_segments": [("2020-05-01", "2022-12-31")],
                    "valid_start_time": "2020-01-01",
                    "valid_end_time": "2020-04-30",
                },
            ],
        }

    def _make_params_base(self):
        return {
            "freq": "week",
            "benchmark": "SH000300",
            "topk": 20,
            "n_drop": 3,
            "anchor_date": "2023-03-30",
        }

    def test_yaml_not_found(self, tmp_path):
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        result = train_window(
            model_name="test_model",
            yaml_file=str(tmp_path / "nonexistent.yaml"),
            window=self._make_window(),
            params_base=self._make_params_base(),
            experiment_name="test_exp",
        )
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_no_folds_in_window(self):
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        window = self._make_window()
        window["cpcv_folds"] = []

        # Need a real yaml file to pass the existence check
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            result = train_window(
                model_name="test_model",
                yaml_file=yaml_path,
                window=window,
                params_base=self._make_params_base(),
                experiment_name="test_exp",
            )
            assert result["success"] is False
            assert "No CPCV folds" in result["error"]
        finally:
            os.unlink(yaml_path)

    def test_successful_k_fold_training(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """Test the full K-fold training loop with all qlib calls mocked."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        # Setup qlib model mock to return predictions
        pred = pd.DataFrame(
            {"score": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2023-01-01", periods=3, freq="W"),
                 ["SH600000"]],
                names=["datetime", "instrument"],
            ),
        )
        mock_qlib_init["model"].predict.return_value = pred

        # Setup dataset segments for IC computation
        mock_qlib_init["dataset"].segments = {
            "test": None,
            "valid": pd.DatetimeIndex(pd.date_range("2022-05-01", periods=3, freq="W")),
        }
        mock_qlib_init["dataset"].prepare.return_value = pd.Series([0.01, 0.02, 0.03])

        # Setup mock recorder
        recorder = mock_qlib_R["recorder"]
        # IC loading: return a simple series
        ic_series = pd.Series([0.04, 0.05, 0.06])
        recorder.load_object.return_value = ic_series

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            with patch("qlib.workflow.R", mock_qlib_R["R"]):
                with patch("qlib.utils.init_instance_by_config",
                          mock_qlib_init["init_instance"]):
                    with patch("quantpits.utils.train_utils.inject_config_for_fold",
                              mock_inject_config):
                        with patch.dict("sys.modules", {"torch": MagicMock()}):
                            # Also mock torch.cuda
                            import torch as torch_mock
                            torch_mock.cuda.is_available.return_value = False

                            result = train_window(
                                model_name="test_model",
                                yaml_file=yaml_path,
                                window=self._make_window(),
                                params_base=self._make_params_base(),
                                experiment_name="test_exp",
                            )
        finally:
            os.unlink(yaml_path)

        assert result["success"] is True
        assert result["n_folds"] == 2
        assert len(result["fold_scores"]) == 2
        assert result["record_id"] is not None
        assert "IC_Mean" in result["performance"]

    def test_train_window_with_cache(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """Train with HandlerCacheManager enabled (cache_size != 0)."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        pred = pd.DataFrame(
            {"score": [0.1]},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2023-01-01", periods=1, freq="W"), ["SH600000"]],
                names=["datetime", "instrument"],
            ),
        )
        mock_qlib_init["model"].predict.return_value = pred
        mock_qlib_init["dataset"].segments = {"test": None, "valid": None}
        mock_qlib_init["dataset"].prepare.return_value = pd.Series([0.01])

        ic_series = pd.Series([0.04])
        mock_qlib_R["recorder"].load_object.return_value = ic_series

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            # Mock HandlerCacheManager
            with patch("quantpits.utils.handler_cache.HandlerCacheManager") as mock_cache_cls:
                mock_cache_mgr = MagicMock()
                mock_cache_mgr.create_dataset.return_value = mock_qlib_init["dataset"]
                mock_cache_cls.return_value = mock_cache_mgr

                with patch("qlib.workflow.R", mock_qlib_R["R"]):
                    with patch("qlib.utils.init_instance_by_config",
                              mock_qlib_init["init_instance"]):
                        with patch("quantpits.utils.train_utils.inject_config_for_fold",
                                  mock_inject_config):
                            with patch.dict("sys.modules", {"torch": MagicMock()}):
                                result = train_window(
                                    model_name="test_model",
                                    yaml_file=yaml_path,
                                    window=self._make_window(),
                                    params_base=self._make_params_base(),
                                    experiment_name="test_exp",
                                    cache_size=1024,
                                )
        finally:
            os.unlink(yaml_path)

        assert result["success"] is True

    def test_train_window_exception_handling(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """Exception during training is caught and returned in result dict."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        # Make init_instance_by_config raise an exception inside the fold loop
        mock_qlib_init["init_instance"].side_effect = RuntimeError("Training failed")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            with patch("qlib.workflow.R", mock_qlib_R["R"]):
                with patch("qlib.utils.init_instance_by_config",
                          mock_qlib_init["init_instance"]):
                    with patch("quantpits.utils.train_utils.inject_config_for_fold",
                              mock_inject_config):
                        result = train_window(
                            model_name="test_model",
                            yaml_file=yaml_path,
                            window=self._make_window(),
                            params_base=self._make_params_base(),
                            experiment_name="test_exp",
                        )
        finally:
            os.unlink(yaml_path)

        assert result["success"] is False
        assert result["error"] is not None

    def test_ic_computation_via_label_dataframe(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """IC computation when prepare() returns a DataFrame with 'label' column."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        # Model returns a DataFrame with 'score' and 'label' columns
        pred = pd.DataFrame(
            {"score": [0.1, 0.2], "label": [0.01, 0.02]},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2023-01-01", periods=2, freq="W"), ["SH600000"]],
                names=["datetime", "instrument"],
            ),
        )
        mock_qlib_init["model"].predict.side_effect = [
            pred,  # first fold prediction
            # Second call: fold validation prediction (returns DataFrame w/label)
            pred,
            # Third call: second fold prediction
            pred,
            # Fourth call: second fold validation
            pred,
        ]

        mock_qlib_init["dataset"].segments = {"test": None, "valid": None}
        mock_qlib_init["dataset"].prepare.return_value = pd.Series([0.01, 0.02])

        ic_series = pd.Series([0.04])
        mock_qlib_R["recorder"].load_object.return_value = ic_series

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            with patch("qlib.workflow.R", mock_qlib_R["R"]):
                with patch("qlib.utils.init_instance_by_config",
                          mock_qlib_init["init_instance"]):
                    with patch("quantpits.utils.train_utils.inject_config_for_fold",
                              mock_inject_config):
                        with patch.dict("sys.modules", {"torch": MagicMock()}):
                            result = train_window(
                                model_name="test_model",
                                yaml_file=yaml_path,
                                window=self._make_window(),
                                params_base=self._make_params_base(),
                                experiment_name="test_exp",
                            )
        finally:
            os.unlink(yaml_path)

        assert result["success"] is True

    def test_performance_extraction_series_port_analysis(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """Performance extraction when port_analysis is a Series (not DataFrame)."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        pred = pd.DataFrame(
            {"score": [0.1]},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2023-01-01", periods=1, freq="W"), ["SH600000"]],
                names=["datetime", "instrument"],
            ),
        )
        mock_qlib_init["model"].predict.return_value = pred

        # Return a Series for port_analysis (second load_object call)
        port_series = pd.Series({
            "annualized_return": 0.15,
            "max_drawdown": -0.05,
            "information_ratio": 1.2,
        }, name="excess_return_without_cost")
        ic_series = pd.Series([0.04, 0.05])

        # First call = IC, second = port_analysis
        mock_qlib_R["recorder"].load_object.side_effect = [ic_series, port_series]

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            with patch("qlib.workflow.R", mock_qlib_R["R"]):
                with patch("qlib.utils.init_instance_by_config",
                          mock_qlib_init["init_instance"]):
                    with patch("quantpits.utils.train_utils.inject_config_for_fold",
                              mock_inject_config):
                        with patch.dict("sys.modules", {"torch": MagicMock()}):
                            result = train_window(
                                model_name="test_model",
                                yaml_file=yaml_path,
                                window=self._make_window(),
                                params_base=self._make_params_base(),
                                experiment_name="test_exp",
                            )
        finally:
            os.unlink(yaml_path)

        assert result["success"] is True
        assert "Ann_Excess" in result["performance"] or result["performance"].get("record_id") is not None

    def test_nan_predictions_warning(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """When fold predictions contain NaN, a warning fraction is printed."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        # All-NaN predictions trigger the NaN warning
        pred = pd.DataFrame(
            {"score": [np.nan, np.nan]},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2023-01-01", periods=2, freq="W"), ["SH600000"]],
                names=["datetime", "instrument"],
            ),
        )
        mock_qlib_init["model"].predict.return_value = pred

        ic_series = pd.Series([0.04])
        mock_qlib_R["recorder"].load_object.return_value = ic_series

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            with patch("qlib.workflow.R", mock_qlib_R["R"]):
                with patch("qlib.utils.init_instance_by_config",
                          mock_qlib_init["init_instance"]):
                    with patch("quantpits.utils.train_utils.inject_config_for_fold",
                              mock_inject_config):
                        with patch.dict("sys.modules", {"torch": MagicMock()}):
                            result = train_window(
                                model_name="test_model",
                                yaml_file=yaml_path,
                                window=self._make_window(),
                                params_base=self._make_params_base(),
                                experiment_name="test_exp",
                            )
        finally:
            os.unlink(yaml_path)

        # Training should complete, but predictions are all NaN
        assert result["success"] is True

    def test_cache_logging(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """When cache_mgr is present, __str__ is logged after training."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window

        pred = pd.DataFrame(
            {"score": [0.1]},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2023-01-01", periods=1, freq="W"), ["SH600000"]],
                names=["datetime", "instrument"],
            ),
        )
        mock_qlib_init["model"].predict.return_value = pred

        ic_series = pd.Series([0.04])
        mock_qlib_R["recorder"].load_object.return_value = ic_series

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("model: test\n")
            yaml_path = f.name

        try:
            mock_cache_mgr = MagicMock()
            mock_cache_mgr.__str__.return_value = "HandlerCache: 2 entries, 128MB"
            mock_cache_mgr.create_dataset.return_value = mock_qlib_init["dataset"]

            with patch("quantpits.utils.handler_cache.HandlerCacheManager",
                       return_value=mock_cache_mgr):
                with patch("qlib.workflow.R", mock_qlib_R["R"]):
                    with patch("qlib.utils.init_instance_by_config",
                              mock_qlib_init["init_instance"]):
                        with patch("quantpits.utils.train_utils.inject_config_for_fold",
                                  mock_inject_config):
                            with patch.dict("sys.modules", {"torch": MagicMock()}):
                                result = train_window(
                                    model_name="test_model",
                                    yaml_file=yaml_path,
                                    window=self._make_window(),
                                    params_base=self._make_params_base(),
                                    experiment_name="test_exp",
                                    cache_size=512,
                                )
        finally:
            os.unlink(yaml_path)

        assert result["success"] is True


# =========================================================================
# _train_in_subprocess
# =========================================================================

class TestTrainInSubprocess:
    """Tests for _train_in_subprocess()."""

    def test_delegates_to_train_window(self):
        from quantpits.scripts.rolling.strategy_cpcv import _train_in_subprocess

        qlib_config = {"provider_uri": "~/.qlib/qlib_data/cn_data"}
        window = {
            "window_idx": 0,
            "train_start": "2020-01-01", "train_end": "2022-12-31",
            "test_start": "2023-01-01", "test_end": "2023-03-30",
            "cpcv_folds": [],
        }
        params_base = {"freq": "week"}

        mock_C = MagicMock()
        mock_qlib_config = MagicMock()
        mock_qlib_config.C = mock_C

        with patch.dict("sys.modules", {"qlib": MagicMock(), "qlib.config": mock_qlib_config}):
            with patch("quantpits.scripts.rolling.strategy_cpcv.train_window") as mock_train:
                mock_train.return_value = {"success": True, "record_id": "rid_001"}
                result = _train_in_subprocess(
                    qlib_config, "test_model", "model.yaml",
                    window, params_base, "test_exp", False, None,
                )

        mock_train.assert_called_once()
        mock_C.register_from_C.assert_called_once_with(qlib_config)
        assert result["success"] is True


# =========================================================================
# train_window_isolated
# =========================================================================

class TestTrainWindowIsolated:
    """Tests for train_window_isolated()."""

    def test_mock_detection_branch(self):
        """When rolling_train has a Mock train_window_model, use it directly."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window_isolated

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
                window={"window_idx": 0, "cpcv_folds": []},
                params_base={"freq": "week"},
                experiment_name="test_exp",
            )

        assert result["success"] is True
        mock_fn.assert_called_once()

    def test_subprocess_path(self, mock_process_pool):
        """Normal (non-test) path spawns a subprocess."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window_isolated

        expected = {
            "success": True, "record_id": "sp_rid",
            "n_folds": 3, "fold_scores": [0.04, 0.05, 0.06],
            "performance": {"IC_Mean": 0.05, "ICIR": 0.5},
            "error": None,
        }
        executor, future = mock_process_pool(expected)

        # Remove rolling_train to avoid mock detection
        sys.modules.pop("rolling_train", None)
        with patch("concurrent.futures.ProcessPoolExecutor", return_value=executor):
            with patch("multiprocessing.get_context", return_value="spawn"):
                result = train_window_isolated(
                    qlib_config={"provider_uri": "test"},
                    model_name="test_model",
                    yaml_file="model.yaml",
                    window={"window_idx": 0, "cpcv_folds": []},
                    params_base={"freq": "week"},
                    experiment_name="test_exp",
                )

        assert result["success"] is True
        assert result["n_folds"] == 3

    def test_passes_cache_size_through(self, mock_process_pool):
        """cache_size is passed through to subprocess."""
        from quantpits.scripts.rolling.strategy_cpcv import train_window_isolated

        expected = {"success": True, "record_id": "sp_rid"}
        executor, future = mock_process_pool(expected)

        sys.modules.pop("rolling_train", None)
        with patch("concurrent.futures.ProcessPoolExecutor", return_value=executor):
            with patch("multiprocessing.get_context", return_value="spawn"):
                result = train_window_isolated(
                    qlib_config={"provider_uri": "test"},
                    model_name="test_model",
                    yaml_file="model.yaml",
                    window={"window_idx": 0, "cpcv_folds": []},
                    params_base={"freq": "week"},
                    experiment_name="test_exp",
                    cache_size=2048,
                )

        assert result["success"] is True


# =========================================================================
# predict_latest
# =========================================================================

class TestPredictLatest:
    """Tests for predict_latest()."""

    def _make_state(self, record_ids):
        """Create a mock RollingState with given completions."""
        state = MagicMock(name="RollingState")
        state.get_completed_record_ids.return_value = record_ids
        return state

    def _make_model_info(self):
        return {"yaml_file": "model.yaml", "algorithm": "gru"}

    def _make_windows(self):
        return [
            {
                "window_idx": 0,
                "train_start": "2020-01-01", "train_end": "2022-12-31",
                "test_start": "2023-01-01", "test_end": "2023-03-30",
                "cpcv_folds": [{"train_segments": [("2020-01-01", "2022-04-30")]}],
            },
            {
                "window_idx": 1,
                "train_start": "2020-04-01", "train_end": "2023-03-31",
                "test_start": "2023-04-01", "test_end": "2023-06-30",
                "cpcv_folds": [{"train_segments": [("2020-04-01", "2023-03-01")]}],
            },
        ]

    def test_no_completions(self):
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        state = MagicMock(name="RollingState")
        state.get_completed_record_ids.return_value = []

        result = predict_latest(
            model_name="test_model",
            model_info=self._make_model_info(),
            state=state,
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
            anchor_date="2023-07-15",
            windows=self._make_windows(),
        )

        assert result is None

    def test_trained_window_not_in_windows(self):
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        state = self._make_state([{"window_idx": 99, "record_id": "rid_099"}])

        result = predict_latest(
            model_name="test_model",
            model_info=self._make_model_info(),
            state=state,
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
            anchor_date="2023-07-15",
            windows=self._make_windows(),
        )

        assert result is None

    def test_gap_without_allow_stale(self):
        """Gap exists but allow_stale_predict=False → return None."""
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        # Only window 0 completed, but anchor is far ahead
        state = self._make_state([{"window_idx": 0, "record_id": "rid_000"}])

        # Remove rolling_train from sys.modules to avoid mock detection
        with patch.dict(sys.modules, {"rolling_train": MagicMock()}):
            sys.modules.pop("rolling_train", None)
            result = predict_latest(
                model_name="test_model",
                model_info=self._make_model_info(),
                state=state,
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
                anchor_date="2025-01-01",
                windows=self._make_windows(),
            )

        assert result is None

    def test_all_up_to_date(self):
        """No gap → already covered, returns None."""
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        # Latest window completed, no index or date gap
        state = self._make_state([{"window_idx": 1, "record_id": "rid_001"}])
        windows = self._make_windows()

        # anchor = test_end of last window
        sys.modules.pop("rolling_train", None)
        result = predict_latest(
            model_name="test_model",
            model_info=self._make_model_info(),
            state=state,
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
            anchor_date=windows[-1]["test_end"],
            windows=windows,
        )

        assert result is None

    def test_mock_detection_branch(self):
        """When rolling_train.predict_with_latest_model is a Mock, use it."""
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        mock_fn = Mock(return_value=pd.DataFrame({"score": [0.1]}))
        fake_rt = MagicMock()
        fake_rt.predict_with_latest_model = mock_fn

        with patch.dict(sys.modules, {"rolling_train": fake_rt}):
            result = predict_latest(
                model_name="test_model",
                model_info=self._make_model_info(),
                state=self._make_state([{"window_idx": 0, "record_id": "rid_000"}]),
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
                anchor_date="2025-01-01",
                windows=self._make_windows(),
                allow_stale_predict=True,
            )

        mock_fn.assert_called_once()
        assert result is not None

    def test_successful_gap_prediction(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """Gap prediction with fold model loading."""
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        state = self._make_state([{"window_idx": 0, "record_id": "rid_000"}])
        windows = self._make_windows()

        # Mock fold model loading
        fold_model = mock_qlib_init["model"]
        pred = pd.DataFrame(
            {"score": [0.1, 0.2]},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2023-04-01", periods=2, freq="W"), ["SH600000"]],
                names=["datetime", "instrument"],
            ),
        )
        fold_model.predict.return_value = pred

        # setup recorder to return fold models
        mock_qlib_R["recorder"].load_object.side_effect = [
            fold_model,  # model_fold_0.pkl
            Exception("No more folds"),  # model_fold_1.pkl
        ]

        sys.modules.pop("rolling_train", None)
        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            with patch("qlib.utils.init_instance_by_config",
                      mock_qlib_init["init_instance"]):
                with patch("quantpits.utils.train_utils.inject_config_for_fold",
                          mock_inject_config):
                    result = predict_latest(
                        model_name="test_model",
                        model_info=self._make_model_info(),
                        state=state,
                        rolling_exp_name="test_exp",
                        params_base={"freq": "week"},
                        anchor_date="2025-01-01",
                        windows=windows,
                        allow_stale_predict=True,
                    )

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_no_fold_models_found(self, mock_qlib_R):
        """All fold model loads fail → returns None."""
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        state = self._make_state([{"window_idx": 0, "record_id": "rid_000"}])
        windows = self._make_windows()

        # First load fails immediately
        mock_qlib_R["recorder"].load_object.side_effect = Exception("No fold models")

        sys.modules.pop("rolling_train", None)
        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            with patch("qlib.utils.init_instance_by_config"):
                with patch("quantpits.utils.train_utils.inject_config_for_fold"):
                    result = predict_latest(
                        model_name="test_model",
                        model_info=self._make_model_info(),
                        state=state,
                        rolling_exp_name="test_exp",
                        params_base={"freq": "week"},
                        anchor_date="2025-01-01",
                        windows=windows,
                        allow_stale_predict=True,
                    )

        assert result is None

    def test_prediction_series_input(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """Series prediction is converted to DataFrame."""
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        state = self._make_state([{"window_idx": 0, "record_id": "rid_000"}])
        windows = self._make_windows()

        fold_model = mock_qlib_init["model"]
        # Return a Series instead of DataFrame
        dates = pd.date_range("2023-04-01", periods=2, freq="W")
        fold_model.predict.return_value = pd.Series([0.1, 0.2], name="score",
                                                     index=dates)

        mock_qlib_R["recorder"].load_object.side_effect = [
            fold_model,
            Exception("No more folds"),
        ]

        sys.modules.pop("rolling_train", None)
        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            with patch("qlib.utils.init_instance_by_config",
                      mock_qlib_init["init_instance"]):
                with patch("quantpits.utils.train_utils.inject_config_for_fold",
                          mock_inject_config):
                    result = predict_latest(
                        model_name="test_model",
                        model_info=self._make_model_info(),
                        state=state,
                        rolling_exp_name="test_exp",
                        params_base={"freq": "week"},
                        anchor_date="2025-01-01",
                        windows=windows,
                        allow_stale_predict=True,
                    )

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_prediction_exception_caught(self, mock_qlib_R):
        """Exception during prediction is caught and returns None."""
        from quantpits.scripts.rolling.strategy_cpcv import predict_latest

        state = self._make_state([{"window_idx": 0, "record_id": "rid_000"}])
        windows = self._make_windows()

        # Recorder raises exception
        mock_qlib_R["R"].get_recorder.side_effect = RuntimeError("MLflow error")

        sys.modules.pop("rolling_train", None)
        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = predict_latest(
                model_name="test_model",
                model_info=self._make_model_info(),
                state=state,
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
                anchor_date="2025-01-01",
                windows=windows,
                allow_stale_predict=True,
            )

        assert result is None


# =========================================================================
# repair_truncated
# =========================================================================

class TestRepairTruncated:
    """Tests for repair_truncated()."""

    def _make_window_map(self):
        return {
            0: {
                "window_idx": 0,
                "train_start": "2020-01-01", "train_end": "2022-12-31",
                "test_start": "2023-01-01", "test_end": "2023-03-30",
                "cpcv_folds": [
                    {"train_segments": [("2020-01-01", "2022-04-30")]},
                ],
            },
        }

    def _make_comp(self, widx=0):
        return {"window_idx": widx, "record_id": "rid_000"}

    def _make_pred(self, n=10, start="2023-01-01", freq="W"):
        dates = pd.date_range(start, periods=n, freq=freq)
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        return pd.DataFrame({"score": range(n)}, index=idx)

    def test_window_not_in_map(self):
        from quantpits.scripts.rolling.strategy_cpcv import repair_truncated
        result = repair_truncated(
            model_name="test_model",
            model_info={"yaml_file": "model.yaml"},
            comp=self._make_comp(99),  # not in map
            window_map=self._make_window_map(),
            rolling_exp_name="test_exp",
            params_base={"freq": "week"},
        )
        pred, repaired = result
        assert pred is None
        assert repaired is False

    def test_prediction_not_truncated(self, mock_qlib_R):
        """pred_max >= test_end - 1 day → no repair needed."""
        from quantpits.scripts.rolling.strategy_cpcv import repair_truncated

        # Prediction covers up to test_end exactly
        pred = self._make_pred(n=13, start="2023-01-01")  # ends at 2023-03-26
        # test_end is 2023-03-30, but if pred ends at 2023-03-30 → no truncation

        # Create pred that covers the full period including test_end
        dates = pd.date_range("2023-01-01", "2023-03-30", freq="W")
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        pred_full = pd.DataFrame({"score": range(len(dates))}, index=idx)

        mock_qlib_R["recorder"].load_object.return_value = pred_full

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = repair_truncated(
                model_name="test_model",
                model_info={"yaml_file": "model.yaml"},
                comp=self._make_comp(0),
                window_map=self._make_window_map(),
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
            )

        pred, repaired = result
        assert repaired is False
        assert pred is not None

    def test_repair_with_truncated_prediction(self, mock_qlib_R, mock_qlib_init, mock_inject_config):
        """Truncated prediction is auto-repaired."""
        from quantpits.scripts.rolling.strategy_cpcv import repair_truncated

        # Prediction only covers first month, test_end is months later
        pred_truncated = self._make_pred(n=4, start="2023-01-01")  # only first month
        dates = pd.date_range("2023-01-01", "2023-03-30", freq="W")
        idx = pd.MultiIndex.from_product(
            [dates, ["SH600000"]], names=["datetime", "instrument"])
        new_pred = pd.DataFrame({"score": np.random.randn(len(dates))}, index=idx)

        fold_model = mock_qlib_init["model"]
        fold_model.predict.return_value = new_pred

        # load_object: first = pred.pkl (truncated), second = model_fold_0.pkl
        mock_qlib_R["recorder"].load_object.side_effect = [pred_truncated, fold_model]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            with patch("qlib.utils.init_instance_by_config",
                      mock_qlib_init["init_instance"]):
                with patch("quantpits.utils.train_utils.inject_config_for_fold",
                          mock_inject_config):
                    result = repair_truncated(
                        model_name="test_model",
                        model_info={"yaml_file": "model.yaml"},
                        comp=self._make_comp(0),
                        window_map=self._make_window_map(),
                        rolling_exp_name="test_exp",
                        params_base={"freq": "week"},
                    )

        pred, repaired = result
        assert repaired is True
        assert pred is not None

    def test_no_folds_for_repair(self, mock_qlib_R):
        """Window has no cpcv_folds → can't repair."""
        from quantpits.scripts.rolling.strategy_cpcv import repair_truncated

        window_map = {
            0: {
                "window_idx": 0,
                "train_start": "2020-01-01", "train_end": "2022-12-31",
                "test_start": "2023-01-01", "test_end": "2023-03-30",
                "cpcv_folds": [],  # empty!
            },
        }

        pred_truncated = self._make_pred(n=4, start="2023-01-01")
        mock_qlib_R["recorder"].load_object.return_value = pred_truncated

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = repair_truncated(
                model_name="test_model",
                model_info={"yaml_file": "model.yaml"},
                comp=self._make_comp(0),
                window_map=window_map,
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
            )

        pred, repaired = result
        assert repaired is False
        # Original pred returned even though repair failed

    def test_fold_model_load_failure(self, mock_qlib_R):
        """Cannot load fold model → repair fails gracefully."""
        from quantpits.scripts.rolling.strategy_cpcv import repair_truncated

        pred_truncated = self._make_pred(n=4, start="2023-01-01")
        # First call = pred.pkl, second = model_fold_0.pkl (fails)
        mock_qlib_R["recorder"].load_object.side_effect = [
            pred_truncated,
            Exception("Model not found"),
        ]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            result = repair_truncated(
                model_name="test_model",
                model_info={"yaml_file": "model.yaml"},
                comp=self._make_comp(0),
                window_map=self._make_window_map(),
                rolling_exp_name="test_exp",
                params_base={"freq": "week"},
            )

        pred, repaired = result
        assert repaired is False  # repair failed, original returned

    def test_repair_exception_caught(self, mock_qlib_R):
        """Exception during repair is caught, original pred returned."""
        from quantpits.scripts.rolling.strategy_cpcv import repair_truncated

        pred_truncated = self._make_pred(n=4, start="2023-01-01")

        # Second load succeeds but inject raises
        fold_model = MagicMock()
        mock_qlib_R["recorder"].load_object.side_effect = [pred_truncated, fold_model]

        with patch("qlib.workflow.R", mock_qlib_R["R"]):
            with patch("quantpits.utils.train_utils.inject_config_for_fold",
                      side_effect=RuntimeError("Config error")):
                result = repair_truncated(
                    model_name="test_model",
                    model_info={"yaml_file": "model.yaml"},
                    comp=self._make_comp(0),
                    window_map=self._make_window_map(),
                    rolling_exp_name="test_exp",
                    params_base={"freq": "week"},
                )

        pred, repaired = result
        assert repaired is False
