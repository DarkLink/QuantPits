"""Tests for search_utils — coverage gaps in worker_init, load_oos_config, save_run_metadata."""

import json
import os
from unittest.mock import MagicMock, patch

# Must set QLIB_WORKSPACE_DIR before importing quantpits
os.environ.setdefault("QLIB_WORKSPACE_DIR", "/tmp")


class TestLoadOosConfig:
    def test_workspace_root_added_to_search_dirs(self, tmp_path):
        """Line 344: workspace_root/config appended to search dirs."""
        from quantpits.utils import search_utils
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "oos_config.json").write_text('{"key": "from_workspace"}')

        result = search_utils.load_oos_config(workspace_root=str(ws))
        assert result["key"] == "from_workspace"

    def test_exception_swallowed_on_invalid_json(self, tmp_path):
        """Lines 350-354: invalid JSON → pass → return {}."""
        from quantpits.utils import search_utils
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "oos_config.json").write_text("NOT JSON {{{")

        result = search_utils.load_oos_config(workspace_root=str(ws))
        assert result == {}

    def test_no_workspace_root_still_searches_root_dir(self, tmp_path):
        """Line 345: env.ROOT_DIR/config always searched."""
        from quantpits.utils import search_utils
        from quantpits.utils import env
        with patch.object(env, 'ROOT_DIR', str(tmp_path)):
            (tmp_path / "config").mkdir(exist_ok=True)
            (tmp_path / "config" / "oos_config.json").write_text('{"fallback": true}')
            result = search_utils.load_oos_config(workspace_root=None)
            assert result.get("fallback") is True


class TestSaveRunMetadata:
    def test_oos_params_injected(self, tmp_path):
        """Line 369: oos_params added to metadata when oos_config is present."""
        from quantpits.utils import search_utils
        ctx = MagicMock()
        ctx.run_path.return_value = str(tmp_path / "run_metadata.json")

        mock_oos = {"limit_threshold": 0.05}
        with patch.object(search_utils, 'load_oos_config', return_value=mock_oos):
            result = search_utils.save_run_metadata(ctx, {"key": "val"})
            assert result is not None

    def test_no_oos_params_when_empty(self, tmp_path):
        """No oos_params when config is empty."""
        from quantpits.utils import search_utils
        ctx = MagicMock()
        ctx.run_path.return_value = str(tmp_path / "rm.json")

        with patch.object(search_utils, 'load_oos_config', return_value={}):
            search_utils.save_run_metadata(ctx, {"key": "val"})


class TestWorkerInit:
    def test_worker_init_sets_env_vars_and_globals(self):
        """Lines 288-313: worker_init sets env vars, creates Exchange, stores globals."""
        from quantpits.utils import search_utils
        import quantpits.utils.search_utils as su

        mock_exchange = MagicMock()

        with patch.object(su, '_worker_norm_df', None, create=True):
            with patch('quantpits.utils.env.init_qlib'):
                with patch('qlib.backtest.exchange.Exchange', return_value=mock_exchange):
                    su.worker_init(
                        norm_df="fake_df",
                        all_codes=["SH600000"],
                        exchange_freq="day",
                        bt_start="09:30:00",
                        bt_end="15:00:00",
                        exchange_kwargs={"limit_threshold": 0.05},
                        st_config={"strategy": "topk"},
                        bt_config={"benchmark": "SH000300"},
                    )

        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert su._worker_trade_exchange is not None


class TestRunBacktestInWorker:
    def test_delegates_to_run_single_backtest(self):
        """Line 322: run_backtest_in_worker delegates to run_single_backtest."""
        from quantpits.utils import search_utils
        import quantpits.utils.search_utils as su

        mock_result = {"sharpe": 2.0, "cagr": 0.15}
        with patch('quantpits.utils.search_utils.run_single_backtest', return_value=mock_result):
            with patch.object(su, '_worker_norm_df', "fake_df"):
                with patch.object(su, '_worker_trade_exchange', MagicMock()):
                    with patch.object(su, '_worker_bt_start', "09:30"):
                        with patch.object(su, '_worker_bt_end', "15:00"):
                            with patch.object(su, '_worker_st_config', {}):
                                with patch.object(su, '_worker_bt_config', {}):
                                    result = su.run_backtest_in_worker(
                                        combo_models=["m1"], top_k=30, drop_n=5,
                                        benchmark="SH000300", freq="day",
                                    )
        assert result["sharpe"] == 2.0


class TestComputeComboScore:
    def test_overlapping_weight_df(self):
        import pandas as pd
        from quantpits.utils.search_utils import _compute_combo_score
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=2), ["A", "B"]],
            names=["datetime", "instrument"]
        )
        norm_df = pd.DataFrame(
            {"m1": [1.0, 2.0, 3.0, 4.0], "m2": [2.0, 3.0, 4.0, 5.0]},
            index=idx
        )
        weight_df = pd.DataFrame(
            {"m1": [0.6, 0.4], "m2": [0.4, 0.6]},
            index=pd.date_range("2020-01-01", periods=2)
        )
        res = _compute_combo_score(norm_df, ["m1", "m2"], weight_df)
        assert isinstance(res, pd.Series)
        assert len(res) == 4

    def test_non_overlapping_weight_df(self):
        import pandas as pd
        from quantpits.utils.search_utils import _compute_combo_score
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=2), ["A", "B"]],
            names=["datetime", "instrument"]
        )
        norm_df = pd.DataFrame(
            {"m1": [1.0, 2.0, 3.0, 4.0], "m2": [2.0, 3.0, 4.0, 5.0]},
            index=idx
        )
        weight_df = pd.DataFrame(
            {"m1": [0.6, 0.4], "m2": [0.4, 0.6]},
            index=pd.date_range("2020-02-01", periods=2)
        )
        res = _compute_combo_score(norm_df, ["m1", "m2"], weight_df)
        assert isinstance(res, pd.Series)
        assert len(res) == 4


class TestComputeRollingSharpeWeights:
    def test_compute_rolling_sharpe_weights_successful(self):
        import pandas as pd
        import numpy as np
        from quantpits.utils import search_utils
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=5), ["A", "B"]],
            names=["datetime", "instrument"]
        )
        norm_df = pd.DataFrame(
            {"m1": np.random.randn(10), "m2": np.random.randn(10)},
            index=idx
        )
        mock_label_df = pd.DataFrame(
            {"label_col": np.random.randn(10)},
            index=idx
        )
        with patch("qlib.data.D.features", return_value=mock_label_df):
            res = search_utils.compute_rolling_sharpe_weights(
                norm_df, top_k=1, window=3, min_periods=1, label_field=["label_col"]
            )
            assert isinstance(res, pd.DataFrame)
            assert list(res.columns) == ["m1", "m2"]


class TestSplitIsOosByArgs:
    def test_split_by_args_various_cutoffs(self):
        import pandas as pd
        import numpy as np
        from quantpits.utils import search_utils
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=24), ["A"]],
            names=["datetime", "instrument"]
        )
        norm_df = pd.DataFrame({"m1": np.arange(24)}, index=idx)
        
        class Args:
            start_date = "2020-01-02"
            end_date = "2020-01-20"
            exclude_last_years = 1
            exclude_last_months = 2
            
        is_df, oos_df = search_utils.split_is_oos_by_args(norm_df, Args())
        assert isinstance(is_df, pd.DataFrame)
        assert isinstance(oos_df, pd.DataFrame)


class TestExtractGroupModelNames:
    def test_extract_group_model_names_successful(self, tmp_path):
        import pytest
        from quantpits.utils import search_utils
        yaml_content = """
        groups:
          group_a:
            - model_1
            - model_2
          group_b:
            - model_3
        """
        yaml_file = tmp_path / "combo_groups.yaml"
        yaml_file.write_text(yaml_content)
        
        res = search_utils.extract_group_model_names(str(yaml_file))
        assert res == {"model_1", "model_2", "model_3"}
        
    def test_extract_group_model_names_empty_raises(self, tmp_path):
        import pytest
        from quantpits.utils import search_utils
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("groups: {}")
        with pytest.raises(ValueError, match="分组配置为空"):
            search_utils.extract_group_model_names(str(empty_yaml))

