import pytest
import os
import sys
import json
import yaml
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "data").mkdir()
    (workspace / "output").mkdir()

    # Create dummy config files
    (workspace / "config" / "model_config.json").write_text(json.dumps({
        "market": "csi300",
        "benchmark": "SH000300",
        "freq": "week"
    }))
    (workspace / "config" / "model_registry.yaml").write_text(yaml.dump({
        "models": {
            "m1": {"algorithm": "gru", "dataset": "Alpha158", "enabled": True, "yaml_file": "gru.yaml"},
            "m2": {"algorithm": "mlp", "dataset": "Alpha158", "enabled": True, "yaml_file": "mlp.yaml"},
            "m3": {"algorithm": "lgb", "dataset": "Alpha360", "enabled": False, "yaml_file": "lgb.yaml", "tags": ["tree"]},
        }
    }))

    import importlib
    script_dir = os.path.join(os.getcwd(), "quantpits/scripts")
    if script_dir not in sys.path:
        sys.path.append(script_dir)

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    for mod_name in ['env', 'quantpits.utils.env', 'train_utils',
                     'quantpits.utils.train_utils',
                     'static_train', 'quantpits.scripts.static_train']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from quantpits.scripts import static_train as st
    yield st, workspace


# ===================================================================
# CLI parse_args tests
# ===================================================================

class TestParseArgs:
    def test_full_mode(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--full']):
            args = st.parse_args()
        assert args.full is True
        assert args.predict_only is False

    def test_predict_only_mode(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--predict-only', '--all-enabled']):
            args = st.parse_args()
        assert args.predict_only is True
        assert args.all_enabled is True

    def test_incremental_models(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--models', 'gru,mlp', '--dry-run']):
            args = st.parse_args()
        assert args.models == 'gru,mlp'
        assert args.dry_run is True
        assert args.full is False
        assert args.predict_only is False

    def test_resume_flag(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--models', 'm1', '--resume']):
            args = st.parse_args()
        assert args.resume is True

    def test_list_flag(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--list']):
            args = st.parse_args()
        assert args.list is True

    def test_filter_args(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--algorithm', 'gru', '--skip', 'm2']):
            args = st.parse_args()
        assert args.algorithm == 'gru'
        assert args.skip == 'm2'


# ===================================================================
# resolve_target_models tests (now in train_utils)
# ===================================================================

class TestResolveTargetModels:
    def test_resolve_by_names(self, mock_env):
        from quantpits.utils.train_utils import resolve_target_models
        args = MagicMock()
        args.models = "m1,m2"
        args.skip = None
        args.all_enabled = False
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None

        targets = resolve_target_models(args)
        assert "m1" in targets
        assert "m2" in targets

    def test_resolve_all_enabled(self, mock_env):
        from quantpits.utils.train_utils import resolve_target_models
        args = MagicMock()
        args.models = None
        args.all_enabled = True
        args.skip = "m2"
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None

        targets = resolve_target_models(args)
        assert "m1" in targets
        assert "m2" not in targets

    def test_resolve_by_algorithm(self, mock_env):
        from quantpits.utils.train_utils import resolve_target_models
        args = MagicMock()
        args.models = None
        args.all_enabled = False
        args.skip = None
        args.algorithm = "gru"
        args.dataset = None
        args.market = None
        args.tag = None

        targets = resolve_target_models(args)
        assert "m1" in targets
        assert len(targets) == 1

    def test_resolve_none(self, mock_env):
        from quantpits.utils.train_utils import resolve_target_models
        args = MagicMock()
        args.models = None
        args.all_enabled = False
        args.skip = None
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None

        result = resolve_target_models(args)
        assert result is None


# ===================================================================
# Full train tests
# ===================================================================

class TestRunFullTrain:
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_single_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('json.dump')
    def test_full_train_smoke(self, mock_json, mock_backup, mock_overwrite,
                               mock_train, mock_dates, mock_init, mock_env):
        st, _ = mock_env
        mock_dates.return_value = {"freq": "week", "anchor_date": "2026-03-01"}
        mock_train.return_value = {
            "success": True,
            "record_id": "rid1",
            "performance": {"IC_Mean": 0.1, "ICIR": 0.5}
        }

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        st.run_full_train(args)

        mock_init.assert_called_once()
        # Should train enabled models (m1 and m2 are enabled)
        assert mock_train.call_count == 2
        mock_overwrite.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_single_model')
    def test_full_train_dry_run(self, mock_train, mock_dates, mock_init, mock_env):
        st, _ = mock_env

        args = MagicMock()
        args.dry_run = True
        args.experiment_name = None
        args.no_pretrain = False

        st.run_full_train(args)

        # dry-run should NOT call init_qlib or train_single_model
        # (init_qlib is called before dry-run check, but train is not)
        mock_train.assert_not_called()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_single_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('json.dump')
    def test_full_train_with_failure(self, mock_json, mock_backup, mock_overwrite,
                                     mock_train, mock_dates, mock_init, mock_env):
        st, _ = mock_env
        mock_dates.return_value = {"freq": "week", "anchor_date": "2026-03-01"}
        # First model succeeds, second fails
        mock_train.side_effect = [
            {"success": True, "record_id": "rid1", "performance": {"IC_Mean": 0.1}},
            {"success": False, "error": "OOM"}
        ]

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        st.run_full_train(args)

        assert mock_train.call_count == 2
        mock_overwrite.assert_called_once()


# ===================================================================
# Incremental train tests
# ===================================================================

class TestRunIncrementalTrain:
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_single_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.merge_performance_file')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.clear_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_success(self, mock_table, mock_clear, mock_save,
                                  mock_perf, mock_merge, mock_train,
                                  mock_dates, mock_init, mock_env):
        st, _ = mock_env
        mock_dates.return_value = {"anchor_date": "2026-01-01", "freq": "week"}
        mock_train.return_value = {"success": True, "record_id": "rid1", "performance": {"ICIR": 0.1}}

        args = MagicMock()
        args.models = "m1"
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"m1": {"yaml_file": "gru.yaml"}}
        st.run_incremental_train(args, targets)

        mock_train.assert_called_once()
        mock_merge.assert_called_once()
        mock_clear.assert_called_once()

    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_dry_run(self, mock_table, mock_env):
        st, _ = mock_env
        args = MagicMock()
        args.resume = False
        args.dry_run = True

        targets = {"m1": {"yaml_file": "gru.yaml"}}
        st.run_incremental_train(args, targets)
        # Should not attempt training

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_single_model')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_failure(self, mock_table, mock_save, mock_train,
                                  mock_dates, mock_init, mock_env):
        st, _ = mock_env
        mock_dates.return_value = {"anchor_date": "2026-01-01", "freq": "week"}
        mock_train.return_value = {"success": False, "error": "Disk full"}

        args = MagicMock()
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"m1": {"yaml_file": "gru.yaml"}}
        st.run_incremental_train(args, targets)

        mock_train.assert_called_once()
        # State should NOT be cleared on failure

    @patch('quantpits.utils.train_utils.load_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_resume(self, mock_table, mock_load, mock_env):
        st, _ = mock_env
        mock_load.return_value = {"completed": ["m1"]}

        args = MagicMock()
        args.resume = True
        args.dry_run = True

        targets = {"m1": {"yaml_file": "gru.yaml"}, "m2": {"yaml_file": "mlp.yaml"}}
        st.run_incremental_train(args, targets)
        # m1 should be skipped, m2 should remain


# ===================================================================
# Predict-only tests
# ===================================================================

class TestRunPredictOnly:
    @patch('quantpits.utils.train_utils.predict_single_model')
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.merge_performance_file')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_predict_only_success(self, mock_table, mock_perf, mock_merge,
                                   mock_dates, mock_init, mock_predict, mock_env, tmp_path):
        st, _ = mock_env
        source_file = tmp_path / "records.json"
        with open(source_file, 'w') as f:
            json.dump({"models": {"m1": "rid1"}, "experiment_name": "train"}, f)

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_predict.return_value = {"success": True, "record_id": "new_rid", "performance": {"IC_Mean": 0.05}}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"m1": {"yaml_file": "gru.yaml"}}
        st.run_predict_only(args, targets)

        mock_predict.assert_called_once()
        mock_merge.assert_called_once()

    @patch('quantpits.utils.train_utils.print_model_table')
    def test_predict_only_dry_run(self, mock_table, mock_env, tmp_path):
        st, _ = mock_env
        source_file = tmp_path / "records.json"
        with open(source_file, 'w') as f:
            json.dump({"models": {"m1": "rid1"}, "experiment_name": "train"}, f)

        args = MagicMock()
        args.dry_run = True
        args.source_records = str(source_file)
        args.no_pretrain = False

        targets = {"m1": {"yaml_file": "gru.yaml"}}
        st.run_predict_only(args, targets)

    def test_predict_only_no_source_file(self, mock_env):
        st, _ = mock_env
        args = MagicMock()
        args.source_records = "/nonexistent/records.json"

        targets = {"m1": {"yaml_file": "gru.yaml"}}
        st.run_predict_only(args, targets)
        # Should print error and return

    @patch('quantpits.utils.train_utils.predict_single_model')
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_predict_only_model_not_in_source(self, mock_table, mock_dates, mock_init,
                                               mock_predict, mock_env, tmp_path):
        st, _ = mock_env
        source_file = tmp_path / "records.json"
        with open(source_file, 'w') as f:
            json.dump({"models": {"m1": "rid1"}, "experiment_name": "train"}, f)

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.experiment_name = None
        args.no_pretrain = False

        # m2 is not in source, m1 is
        targets = {"m1": {"yaml_file": "gru.yaml"}, "m2": {"yaml_file": "mlp.yaml"}}

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_predict.return_value = {"success": True, "record_id": "rid", "performance": {}}

        st.run_predict_only(args, targets)
        # Only m1 should be predicted
        mock_predict.assert_called_once()


# ===================================================================
# Main routing tests
# ===================================================================

class TestMain:
    def test_main_no_selection(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py']):
            st.main()  # Should print error and return

    def test_main_list(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--list']):
            with patch('quantpits.utils.train_utils.show_model_list') as mock_show:
                st.main()
                mock_show.assert_called_once()

    def test_main_show_state(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--show-state']):
            with patch('quantpits.scripts.static_train.show_state') as mock_show:
                st.main()
                mock_show.assert_called_once()

    def test_main_clear_state(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--clear-state']):
            with patch('quantpits.utils.train_utils.clear_run_state') as mock_clear:
                st.main()
                mock_clear.assert_called_once()

    def test_main_full(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--full']):
            with patch('quantpits.scripts.static_train.run_full_train') as mock_run:
                st.main()
                mock_run.assert_called_once()

    def test_main_predict_only(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--predict-only', '--all-enabled']):
            with patch('quantpits.utils.train_utils.resolve_target_models', return_value={"m1": {}}):
                with patch('quantpits.scripts.static_train.run_predict_only') as mock_run:
                    st.main()
                    mock_run.assert_called_once()

    def test_main_incremental(self, mock_env):
        st, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--models', 'm1']):
            with patch('quantpits.utils.train_utils.resolve_target_models', return_value={"m1": {}}):
                with patch('quantpits.scripts.static_train.run_incremental_train') as mock_run:
                    st.main()
                    mock_run.assert_called_once()


# ===================================================================
# show_model_list tests (now in train_utils)
# ===================================================================

class TestShowModelList:
    def test_show_list_no_filter(self, mock_env):
        from quantpits.utils.train_utils import show_model_list
        args = MagicMock()
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None

        show_model_list(args)  # Should not raise

    def test_show_list_with_filter(self, mock_env):
        from quantpits.utils.train_utils import show_model_list
        args = MagicMock()
        args.algorithm = "gru"
        args.dataset = None
        args.market = None
        args.tag = None

        show_model_list(args)

    def test_show_list_with_source(self, mock_env, tmp_path):
        from quantpits.utils.train_utils import show_model_list
        source_file = tmp_path / "records.json"
        with open(source_file, 'w') as f:
            json.dump({"models": {"m1": "rid1"}}, f)

        args = MagicMock()
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None

        show_model_list(args, source_records_file=str(source_file))


# ===================================================================
# predict_single_model tests (now in train_utils)
# ===================================================================

class TestPredictSingleModel:
    def test_yaml_missing(self, mock_env, tmp_path):
        from quantpits.utils.train_utils import predict_single_model
        model_info = {'yaml_file': str(tmp_path / 'nonexistent.yaml')}
        result = predict_single_model(
            'test_model', model_info, {}, 'exp', {'models': {'test_model': 'rid1'}}
        )
        assert result['success'] is False
        assert 'YAML' in result['error']

    def test_model_not_in_source(self, mock_env, tmp_path):
        from quantpits.utils.train_utils import predict_single_model
        yaml_file = tmp_path / 'model.yaml'
        yaml_file.write_text('test: true')
        model_info = {'yaml_file': str(yaml_file)}
        result = predict_single_model(
            'test_model', model_info, {}, 'exp', {'models': {}}
        )
        assert result['success'] is False
        assert '不在源训练记录中' in result['error']

    @patch('quantpits.utils.train_utils.inject_config')
    @patch('qlib.utils.init_instance_by_config', create=True)
    @patch('qlib.workflow.R', create=True)
    def test_predict_success(self, mock_R, mock_init, mock_inject, mock_env, tmp_path):
        import pandas as pd
        from quantpits.utils.train_utils import predict_single_model

        yaml_file = tmp_path / 'model.yaml'
        yaml_file.write_text('test: true')
        model_info = {'yaml_file': str(yaml_file)}

        mock_inject.return_value = {
            'task': {
                'dataset': {'class': 'DatasetH'},
                'record': []
            }
        }

        mock_model = MagicMock()
        mock_pred = pd.Series([1.0, 2.0])
        mock_model.predict.return_value = mock_pred

        mock_source_recorder = MagicMock()
        mock_source_recorder.load_object.return_value = mock_model

        mock_recorder = MagicMock()
        mock_recorder.info = {'id': 'new_rid'}
        mock_recorder.load_object.side_effect = Exception("no ic")

        mock_R.get_recorder.side_effect = [mock_source_recorder, mock_recorder]
        mock_R.start.return_value.__enter__ = MagicMock()
        mock_R.start.return_value.__exit__ = MagicMock(return_value=False)

        source_records = {'models': {'test_model': 'rid1'}, 'experiment_name': 'train'}
        params = {'anchor_date': '2020-01-01'}

        result = predict_single_model(
            'test_model', model_info, params, 'exp', source_records
        )

        assert result['success'] is True
        assert result['record_id'] == 'new_rid'


# ===================================================================
# show_state tests
# ===================================================================

class TestShowState:
    def test_show_state_empty(self, mock_env):
        st, _ = mock_env
        with patch('quantpits.utils.train_utils.load_run_state', return_value=None):
            st.show_state()  # Should not raise

    def test_show_state_with_data(self, mock_env):
        st, _ = mock_env
        state = {
            "started_at": "2026-01-01 10:00:00",
            "mode": "incremental",
            "experiment_name": "test",
            "anchor_date": "2026-01-01",
            "completed": ["m1"],
            "target_models": ["m1", "m2"],
            "failed": {"m3": "error msg"}
        }
        with patch('quantpits.utils.train_utils.load_run_state', return_value=state):
            st.show_state()  # Should not raise
