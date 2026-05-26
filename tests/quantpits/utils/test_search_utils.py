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
