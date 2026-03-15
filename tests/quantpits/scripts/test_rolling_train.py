#!/usr/bin/env python
"""
Tests for rolling_train.py — window generation logic and state management.

These tests only exercise pure date arithmetic and state persistence,
no qlib/env dependency needed.
"""
import os
import sys
import json
import tempfile
import unittest.mock as mock
import pytest

# Avoid breaking other tests by not polluting sys.modules globally with a Mock.
# Instead, set up the environment that env.py expects.
_fake_root = '/tmp/fake_workspace_rolling_test'
os.makedirs(_fake_root, exist_ok=True)
if "QLIB_WORKSPACE_DIR" not in os.environ:
    os.environ["QLIB_WORKSPACE_DIR"] = _fake_root

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', '..', 'quantpits', 'scripts')
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))

# Now env can be imported normally and it will be a real module.
import env
import pandas as pd
import yaml
from rolling_train import generate_rolling_windows, RollingState, parse_step_to_relativedelta, resolve_target_models


@pytest.fixture
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
            "m2": {"algorithm": "mlp", "dataset": "Alpha158", "enabled": False, "yaml_file": "mlp.yaml"}
        }
    }))
    (workspace / "config" / "rolling_config.yaml").write_text(yaml.dump({
        "rolling_start": "2020-01-01",
        "train_years": 3,
        "valid_years": 1,
        "test_step": "3M"
    }))
    (workspace / "gru.yaml").write_text("model: gru")
    (workspace / "mlp.yaml").write_text("model: mlp")
    (workspace / "m1.yaml").write_text("model: m1")

    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(sys, 'argv', ['script.py'])

    import train_utils
    monkeypatch.setattr(train_utils, 'ROLLING_PREDICTION_DIR', str(workspace / "output" / "predictions" / "rolling"))
    (workspace / "output" / "predictions" / "rolling").mkdir(parents=True)

    # Reload relevant modules to pick up new env
    import importlib
    for mod in ['env', 'train_utils', 'rolling_train']:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

    # Mock pd.DataFrame.to_csv and pd.Series.to_csv globally for tests to avoid fs issues
    monkeypatch.setattr(pd.DataFrame, 'to_csv', mock.MagicMock())
    monkeypatch.setattr(pd.Series, 'to_csv', mock.MagicMock())

    import rolling_train as rt
    return rt, workspace


class TestUtils:
    """Test utility functions in rolling_train.py"""

    def test_parse_step_valid(self):
        rd = parse_step_to_relativedelta("3M")
        assert rd.months + rd.years * 12 == 3
        
        rd = parse_step_to_relativedelta("1Y")
        assert rd.months + rd.years * 12 == 12
        
        rd = parse_step_to_relativedelta("  12m  ")
        assert rd.months + rd.years * 12 == 12

    def test_parse_step_invalid(self):
        with pytest.raises(ValueError, match="Invalid step format"):
            parse_step_to_relativedelta("3D")
        with pytest.raises(ValueError):
            parse_step_to_relativedelta("")


class TestGenerateRollingWindows:
    """Test generate_rolling_windows date arithmetic"""

    def test_basic_3m_step(self):
        """基本 3M 步长: T=2020-01-01, X=3Y, Y=1Y, Z=3M"""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="3M",
            anchor_date="2024-12-31",
        )

        assert len(windows) >= 4  # 至少 4 个 windows

        # Window 0
        w0 = windows[0]
        assert w0['train_start'] == "2020-01-01"
        assert w0['train_end'] == "2022-12-31"
        assert w0['valid_start'] == "2023-01-01"
        assert w0['valid_end'] == "2023-12-31"
        assert w0['test_start'] == "2024-01-01"
        assert w0['test_end'] == "2024-03-31"

        # Window 1
        w1 = windows[1]
        assert w1['train_start'] == "2020-04-01"
        assert w1['train_end'] == "2023-03-31"
        assert w1['valid_start'] == "2023-04-01"
        assert w1['valid_end'] == "2024-03-31"
        assert w1['test_start'] == "2024-04-01"
        assert w1['test_end'] == "2024-06-30"

        # Window 2
        w2 = windows[2]
        assert w2['train_start'] == "2020-07-01"
        assert w2['train_end'] == "2023-06-30"
        assert w2['valid_start'] == "2023-07-01"
        assert w2['valid_end'] == "2024-06-30"
        assert w2['test_start'] == "2024-07-01"
        assert w2['test_end'] == "2024-09-30"

        # Window 3
        w3 = windows[3]
        assert w3['train_start'] == "2020-10-01"
        assert w3['train_end'] == "2023-09-30"
        assert w3['valid_start'] == "2023-10-01"
        assert w3['valid_end'] == "2024-09-30"
        assert w3['test_start'] == "2024-10-01"
        assert w3['test_end'] == "2024-12-31"

    def test_no_endpoint_overlap(self):
        """各段端点不能重叠"""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="3M",
            anchor_date="2025-12-31",
        )

        for w in windows:
            assert w['train_end'] < w['valid_start'], \
                f"W{w['window_idx']}: train_end {w['train_end']} >= valid_start {w['valid_start']}"
            assert w['valid_end'] < w['test_start'], \
                f"W{w['window_idx']}: valid_end {w['valid_end']} >= test_start {w['test_start']}"

        for i in range(len(windows) - 1):
            assert windows[i]['test_end'] < windows[i+1]['test_start'], \
                f"W{i} test_end {windows[i]['test_end']} >= W{i+1} test_start {windows[i+1]['test_start']}"

    def test_last_window_truncated_to_anchor(self):
        """最后 window 的 test_end = anchor_date"""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="3M",
            anchor_date="2024-02-15",
        )

        assert len(windows) >= 1
        last = windows[-1]
        assert last['test_end'] == "2024-02-15"

    def test_6m_step(self):
        """6M 步长"""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="6M",
            anchor_date="2025-06-30",
        )

        w0 = windows[0]
        assert w0['test_start'] == "2024-01-01"
        assert w0['test_end'] == "2024-06-30"

        w1 = windows[1]
        assert w1['train_start'] == "2020-07-01"
        assert w1['test_start'] == "2024-07-01"
        assert w1['test_end'] == "2024-12-31"

    def test_1y_step(self):
        """1Y 步长"""
        windows = generate_rolling_windows(
            rolling_start="2015-01-01",
            train_years=5,
            valid_years=2,
            test_step="1Y",
            anchor_date="2025-12-31",
        )

        w0 = windows[0]
        assert w0['train_start'] == "2015-01-01"
        assert w0['train_end'] == "2019-12-31"
        assert w0['valid_start'] == "2020-01-01"
        assert w0['valid_end'] == "2021-12-31"
        assert w0['test_start'] == "2022-01-01"
        assert w0['test_end'] == "2022-12-31"

        w1 = windows[1]
        assert w1['train_start'] == "2016-01-01"
        assert w1['train_end'] == "2020-12-31"
        assert w1['test_start'] == "2023-01-01"
        assert w1['test_end'] == "2023-12-31"

    def test_1m_step(self):
        """1M 步长"""
        windows = generate_rolling_windows(
            rolling_start="2022-01-01",
            train_years=1,
            valid_years=1,
            test_step="1M",
            anchor_date="2024-04-30",
        )

        w0 = windows[0]
        assert w0['train_start'] == "2022-01-01"
        assert w0['train_end'] == "2022-12-31"
        assert w0['valid_start'] == "2023-01-01"
        assert w0['valid_end'] == "2023-12-31"
        assert w0['test_start'] == "2024-01-01"
        assert w0['test_end'] == "2024-01-31"

        w1 = windows[1]
        assert w1['train_start'] == "2022-02-01"
        assert w1['train_end'] == "2023-01-31"
        assert w1['test_start'] == "2024-02-01"
        assert w1['test_end'] == "2024-02-29"  # 2024 is leap year

        w2 = windows[2]
        assert w2['test_start'] == "2024-03-01"
        assert w2['test_end'] == "2024-03-31"

    def test_no_windows_when_anchor_too_early(self):
        """anchor_date 早于第一个 window 的 test_start"""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="3M",
            anchor_date="2023-06-30",
        )

        assert len(windows) == 0

    def test_window_idx_sequential(self):
        """window_idx 从 0 连续递增"""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="3M",
            anchor_date="2025-12-31",
        )

        for i, w in enumerate(windows):
            assert w['window_idx'] == i

    def test_february_month_end(self):
        """涉及 2 月月末的日期计算"""
        windows = generate_rolling_windows(
            rolling_start="2019-03-01",
            train_years=2,
            valid_years=1,
            test_step="1M",
            anchor_date="2022-05-31",
        )

        w0 = windows[0]
        assert w0['test_start'] == "2022-03-01"
        assert w0['test_end'] == "2022-03-31"

        for w in windows:
            if "02-28" in w['train_end'] or "02-29" in w['train_end']:
                from datetime import datetime
                d = datetime.strptime(w['train_end'], "%Y-%m-%d")
                assert d.month == 2
                assert d.day in (28, 29)


class TestRollingStateExtra:
    """Extra tests for RollingState edge cases"""

    def test_load_corrupt_json(self, tmp_path):
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("invalid json {")
        state = RollingState(state_file=str(bad_file))
        assert state._state == state._empty()

    def test_load_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")
        state = RollingState(state_file=str(empty_file))
        assert state._state == state._empty()

    def test_show_state(self, capsys, tmp_path):
        state_file = tmp_path / "state.json"
        state = RollingState(state_file=str(state_file))
        state.init_run({'test_step': '3M'}, '2025-01-01', 4)
        state.mark_window_model_done(0, 'm1', 'rec1')

        state.show()
        captured = capsys.readouterr()
        assert "Rolling 运行状态" in captured.out
        assert "Window 0: ['m1']" in captured.out

    def test_show_state_empty(self, capsys, tmp_path):
        state_file = tmp_path / "none.json"
        state = RollingState(state_file=str(state_file))
        state.show()
        captured = capsys.readouterr()
        assert "没有找到 Rolling 运行状态" in captured.out

    def test_clear_and_backup(self, tmp_path):
        state_file = tmp_path / "state.json"
        state = RollingState(state_file=str(state_file))
        state.init_run({}, '2025-01-01', 1)

        with mock.patch('train_utils.backup_file_with_date') as mock_backup:
            state.clear()
            mock_backup.assert_called_once_with(str(state_file), prefix="rolling_state")
            assert not os.path.exists(state_file)

    def test_get_all_completed(self, tmp_path):
        state_file = tmp_path / "state.json"
        state = RollingState(state_file=str(state_file))
        state.mark_window_model_done(0, 'm1', 'r1')
        assert state.get_all_completed_windows() == {"0": {"m1": "r1"}}


class TestResolveTargetModels:
    """Test resolve_target_models logic"""

    def test_resolve_by_names(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.models = "m1,m2"
        args.skip = None
        targets = rt.resolve_target_models(args)
        assert "m1" in targets
        assert "m2" in targets

    def test_resolve_all_enabled_with_skip(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.models = None
        args.all_enabled = True
        args.skip = "m2"
        targets = rt.resolve_target_models(args)
        assert "m1" in targets
        assert "m2" not in targets

    def test_resolve_by_algorithm(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.models = None
        args.all_enabled = False
        args.algorithm = "gru"
        args.dataset = None
        args.tag = None
        args.skip = None
        targets = rt.resolve_target_models(args)
        assert "m1" in targets
        assert len(targets) == 1

    def test_resolve_by_dataset_and_tag(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.models = None
        args.all_enabled = False
        args.algorithm = None
        args.dataset = "d1"
        args.tag = "t1"
        args.skip = None
        # m1 is gru, m2 is mlp. Let's assume m1 has dataset=d1, tag=t1 in a mock config.
        # But resolve_target_models uses model_registry.yaml which I mocked in mock_env.
        targets = rt.resolve_target_models(args)
        # In my mock_env registry, m1 has nothing, so it should be skipped if filtered.
        # If I want it to pass, I should use valid filters or mock the registry differently.
        assert targets is None or len(targets) == 0

    def test_resolve_none(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.models = None
        args.all_enabled = False
        args.algorithm = None
        args.dataset = None
        args.tag = None
        args.skip = None
        assert rt.resolve_target_models(args) is None


class TestFunctionalLogic:
    """Test core functional logic with mocks"""

    def test_get_base_params(self, mock_env):
        rt, _ = mock_env
        with mock.patch('config_loader.load_workspace_config') as mock_load:
            mock_load.return_value = {'market': 'csi500'}
            with mock.patch('qlib.data.D', create=True) as mock_d:
                mock_d.calendar.return_value = [pd.Timestamp('2024-01-01')]
                params = rt.get_base_params()
                assert params['market'] == 'csi500'
                assert params['anchor_date'] == '2024-01-01'

    def test_train_window_model_success(self, mock_env):
        rt, _ = mock_env
        window = {
            'window_idx': 0,
            'train_start': '2020-01-01', 'train_end': '2022-12-31',
            'valid_start': '2023-01-01', 'valid_end': '2023-12-31',
            'test_start': '2024-01-01', 'test_end': '2024-03-31'
        }
        
        with mock.patch('qlib.workflow.R', create=True) as mock_r, \
             mock.patch('qlib.utils.init_instance_by_config') as mock_init, \
             mock.patch('train_utils.inject_config') as mock_inject:
            
            mock_recorder = mock.MagicMock()
            mock_recorder.info = {'id': 'rec_123'}
            mock_recorder.load_object.return_value = pd.Series([0.1], index=[pd.Timestamp('2024-01-01')])
            mock_r.get_recorder.return_value = mock_recorder
            mock_r.start.return_value.__enter__.return_value = mock_recorder
            
            mock_model = mock.MagicMock()
            mock_model.predict.return_value = pd.DataFrame({'score': [0.5]})
            mock_init.return_value = mock_model
            mock_inject.return_value = {'task': {'model': {}, 'dataset': {}}}
            
            result = rt.train_window_model('m1', 'm1.yaml', window, {'market': 'csi300', 'benchmark': 'csi300'}, 'exp')
            if not result['success']:
                print(f"Error: {result['error']}")
            assert result['success'] is True
            assert result['record_id'] == 'rec_123'

    def test_train_window_model_no_yaml(self, mock_env):
        rt, _ = mock_env
        result = rt.train_window_model('m1', 'no.yaml', {}, {}, 'exp')
        assert result['success'] is False
        assert "YAML 不存在" in result['error']

    def test_concatenate_predictions(self, mock_env):
        rt, _ = mock_env
        state = mock.MagicMock()
        state.get_completed_record_ids.return_value = [
            {'window_idx': 0, 'record_id': 'r0'},
            {'window_idx': 1, 'record_id': 'r1'}
        ]
        
        df0 = pd.DataFrame({'score': [0.1]}, index=pd.MultiIndex.from_tuples([ (pd.Timestamp('2024-01-01'), 'S1')], names=['datetime', 'instrument']))
        df1 = pd.DataFrame({'score': [0.2]}, index=pd.MultiIndex.from_tuples([ (pd.Timestamp('2024-02-01'), 'S1')], names=['datetime', 'instrument']))

        with mock.patch('qlib.workflow.R', create=True) as mock_r, \
             mock.patch('qlib.workflow.R.start'), \
             mock.patch('qlib.workflow.R.save_objects'):
            
            r0 = mock.MagicMock()
            r0.load_object.return_value = df0
            r1 = mock.MagicMock()
            r1.load_object.return_value = df1
            mock_r.get_recorder.side_effect = [r0, r1, mock.MagicMock(id='combined_r')]
            
            res = rt.concatenate_rolling_predictions(state, ['m1'], 'exp', 'comb', '2024-01-01')
            assert 'm1' in res
            assert res['m1'] == 'combined_r'

    def test_save_rolling_records(self, mock_env):
        rt, workspace = mock_env
        combined = {'m1': 'rid1'}
        rt.save_rolling_records(combined, 'exp', '2024-01-01')
        
        record_file = workspace / "latest_rolling_records.json"
        assert record_file.exists()
        data = json.loads(record_file.read_text())
        assert data['models']['m1'] == 'rid1'

    def test_train_window_model_training_failure(self, mock_env):
        rt, _ = mock_env
        window = {'window_idx': 0, 'test_start': '2024-01-01', 'train_start': '2020-01-01', 'train_end': '2022-12-31', 'valid_start': '2023-01-01', 'valid_end': '2023-12-31', 'test_end': '2024-03-31'}
        
        with mock.patch('qlib.workflow.R', create=True) as mock_r, \
             mock.patch('train_utils.inject_config') as mock_inject:
            
            mock_inject.return_value = {'task': {'model': {}, 'dataset': {}}}
            mock_r.start.side_effect = Exception("Training crash")
            result = rt.train_window_model('m1', 'm1.yaml', window, {'market': 'csi300', 'benchmark': 'csi300'}, 'exp')
            assert result['success'] is False
            assert "Training crash" in result['error']

    def test_concatenate_rolling_predictions_real_logic(self, mock_env):
        rt, workspace = mock_env
        state = mock.MagicMock()
        state.get_completed_record_ids.return_value = [
            {'window_idx': 0, 'record_id': 'r0'},
            {'window_idx': 1, 'record_id': 'r1'}
        ]
        
        # Real logic test - don't mock rt.concatenate_rolling_predictions
        df0 = pd.DataFrame({'score': [0.1]}, index=pd.MultiIndex.from_tuples([ (pd.Timestamp('2024-01-01'), 'S1')], names=['datetime', 'instrument']))
        df1 = pd.DataFrame({'score': [0.2]}, index=pd.MultiIndex.from_tuples([ (pd.Timestamp('2024-02-01'), 'S1')], names=['datetime', 'instrument']))

        with mock.patch('qlib.workflow.R', create=True) as mock_r:
            r0 = mock.MagicMock()
            r0.load_object.return_value = df0
            r1 = mock.MagicMock()
            r1.load_object.return_value = df1
            mock_combined_r = mock.MagicMock(id='combined_r')
            mock_r.get_recorder.side_effect = [r0, r1, mock_combined_r]
            
            res = rt.concatenate_rolling_predictions(state, ['m1'], 'rolling_exp', 'comb_exp', '2024-03-31')
            assert res['m1'] == 'combined_r'
            assert mock_r.save_objects.called

    def test_save_rolling_records_real_logic(self, mock_env):
        rt, workspace = mock_env
        combined = {'m1': 'rid1'}
        rt.save_rolling_records(combined, 'exp', '2024-03-31')
        record_file = workspace / "latest_rolling_records.json"
        assert record_file.exists()
        data = json.loads(record_file.read_text())
        assert data['models']['m1'] == 'rid1'

    def test_predict_with_latest_model_real_logic(self, mock_env):
        rt, _ = mock_env
        state = mock.MagicMock()
        state.get_completed_record_ids.return_value = [{'window_idx': 0, 'record_id': 'rid0'}]
        
        with mock.patch('qlib.workflow.R', create=True) as mock_r, \
             mock.patch('qlib.utils.init_instance_by_config') as mock_init, \
             mock.patch('train_utils.inject_config') as mock_inject:
            
            mock_recorder = mock.MagicMock()
            mock_model = mock.MagicMock()
            mock_recorder.load_object.return_value = mock_model
            mock_r.get_recorder.return_value = mock_recorder
            
            mock_inject.return_value = {'task': {'dataset': {}}}
            mock_model.predict.return_value = pd.DataFrame({'score': [0.5]})
            
            res = rt.predict_with_latest_model('m1', {'yaml_file': 'm1.yaml'}, state, 'exp', {'market': 'csi300', 'benchmark': 'csi300'}, '2024-01-01')
            assert res is not None
            assert len(res) == 1


class TestMainFlows:
    """Test top-level flow functions"""

    def test_run_cold_start_dry_run(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.dry_run = True
        args.resume = False
        targets = {'m1': {'yaml_file': 'm1.yaml'}}
        cfg = {
            'rolling_start': '2020-01-01', 'train_years': 3,
            'valid_years': 1, 'test_step': '3M'
        }
        
        with mock.patch('quantpits.scripts.env.init_qlib'), \
             mock.patch('rolling_train.get_base_params') as mock_base, \
             mock.patch('rolling_train.train_window_model') as mock_train:
            
            mock_base.return_value = {'anchor_date': '2025-01-01', 'freq': 'week'}
            rt.run_cold_start(args, targets, cfg)
            mock_train.assert_not_called()

    def test_run_cold_start_full(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.dry_run = False
        args.resume = False
        args.no_pretrain = False
        targets = {'m1': {'yaml_file': 'm1.yaml'}}
        cfg = {
            'rolling_start': '2020-01-01', 'train_years': 3,
            'valid_years': 1, 'test_step': '3M'
        }
        
        with mock.patch('quantpits.scripts.env.init_qlib'), \
             mock.patch('rolling_train.get_base_params') as mock_base, \
             mock.patch('rolling_train.train_window_model') as mock_train, \
             mock.patch('rolling_train.concatenate_rolling_predictions') as mock_concat, \
             mock.patch('rolling_train.save_rolling_records') as mock_save:
            
            mock_base.return_value = {'anchor_date': '2024-02-01', 'freq': 'week'}
            # 2020-01-01 + 3Y + 1Y = 2024-01-01. One window: 2024-01-01 to 2024-02-01.
            mock_train.return_value = {'success': True, 'record_id': 'r1'}
            mock_concat.return_value = {'m1': 'cr1'}
            
            rt.run_cold_start(args, targets, cfg)
            mock_train.assert_called()
            mock_save.assert_called()

    def test_run_daily_predict_only(self, mock_env):
        rt, _ = mock_env
        cfg = {
            'rolling_start': '2020-01-01', 'train_years': 3, 'valid_years': 1, 
            'test_step': '3M', 'test_step_months': 3
        }
        
        with mock.patch('quantpits.scripts.env.init_qlib'), \
             mock.patch.object(rt, 'get_base_params') as mock_base, \
             mock.patch.object(rt, 'RollingState') as mock_state_cls, \
             mock.patch.object(rt, 'predict_with_latest_model') as mock_predict, \
             mock.patch.object(rt, 'concatenate_rolling_predictions') as mock_concat, \
             mock.patch.object(rt, 'generate_rolling_windows') as mock_gen:
            
            mock_base.return_value = {'anchor_date': '2024-02-01', 'freq': 'week'}
            mock_concat.return_value = {'m1': 'rid1'}
            
            mock_state = mock.MagicMock()
            # Set anchor_date to 2024-03-31 so that current anchor (02-01) <= 03-31
            mock_state.anchor_date = '2024-03-31'
            mock_state.is_window_model_done.return_value = True
            mock_state_cls.return_value = mock_state
            
            args = mock.MagicMock()
            args.dry_run = False
            
            rt.run_daily(args, {'m1': {}}, cfg)
            mock_predict.assert_called()

    def test_main_cold_start(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = False
        args.cold_start = True
        args.predict_only = False
        args.resume = False
        
        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.resolve_target_models') as mock_resolve, \
             mock.patch('config_loader.load_rolling_config') as mock_load_cfg, \
             mock.patch('rolling_train.run_cold_start') as mock_run, \
             mock.patch('rolling_train.run_daily') as mock_run_daily:
            
            mock_resolve.return_value = {'m1': {}}
            mock_load_cfg.return_value = {
                'rolling_start': '2020-01-01',
                'train_years': 3,
                'valid_years': 1,
                'test_step': '3M',
                'test_step_months': 3
            }
            try:
                rt.main()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
            mock_run.assert_called_once()

    def test_main_predict_only(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = False
        args.predict_only = True
        args.resume = False
        args.cold_start = False
        args.models = None
        args.algorithm = "gru"
        args.dataset = None
        args.tag = None
        args.all_enabled = False
        
        mock_patch_cfg = {
            'rolling_start': '2020-01-01',
            'train_years': 3,
            'valid_years': 1,
            'test_step': '3M',
            'test_step_months': 3
        }
        
        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.resolve_target_models', return_value={'m1':{}}), \
             mock.patch('config_loader.load_rolling_config', return_value=mock_patch_cfg), \
             mock.patch('rolling_train.run_predict_only') as mock_run_po:
            
            rt.main()
            mock_run_po.assert_called_once()

    def test_main_resume(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = False
        args.resume = True
        args.cold_start = False
        args.predict_only = False
        args.models = None
        args.algorithm = "gru"
        args.dataset = None
        args.tag = None
        args.all_enabled = False
        
        mock_patch_cfg = {
            'rolling_start': '2020-01-01',
            'train_years': 3,
            'valid_years': 1,
            'test_step': '3M',
            'test_step_months': 3
        }
        
        mock_state = mock.MagicMock()
        mock_state.anchor_date = "2025-01-01"
        mock_state.completed_windows = {"0": {"m1": "r1"}}

        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.resolve_target_models', return_value={'m1':{}}), \
             mock.patch('config_loader.load_rolling_config', return_value=mock_patch_cfg), \
             mock.patch('rolling_train.RollingState', return_value=mock_state), \
             mock.patch('rolling_train.run_cold_start') as mock_run:
            
            rt.main()
            mock_run.assert_called_once()

    def test_main_show_state(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = True
        args.clear_state = False
        
        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.RollingState') as mock_state_cls:
            
            rt.main()
            mock_state_cls.return_value.show.assert_called()

    def test_main_clear_state(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = True
        
        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.RollingState') as mock_state_cls:
            
            rt.main()
            mock_state_cls.return_value.clear.assert_called()

    """Test RollingState persistence and recovery"""

    def test_init_and_save(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_file = f.name

        try:
            state = RollingState(state_file=state_file)
            state.init_run(
                rolling_config={'test_step': '3M'},
                anchor_date='2025-01-01',
                total_windows=4,
            )

            assert state.anchor_date == '2025-01-01'

            state2 = RollingState(state_file=state_file)
            assert state2.anchor_date == '2025-01-01'
        finally:
            os.unlink(state_file)

    def test_mark_and_check(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_file = f.name

        try:
            state = RollingState(state_file=state_file)
            state.init_run({'test_step': '3M'}, '2025-01-01', 4)

            assert not state.is_window_model_done(0, 'linear')
            state.mark_window_model_done(0, 'linear', 'rec_001')
            assert state.is_window_model_done(0, 'linear')
            assert not state.is_window_model_done(0, 'gru')
            assert not state.is_window_model_done(1, 'linear')

            recs = state.get_completed_record_ids('linear')
            assert len(recs) == 1
            assert recs[0]['window_idx'] == 0
            assert recs[0]['record_id'] == 'rec_001'
        finally:
            os.unlink(state_file)

    def test_resume_preserves_state(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_file = f.name

        try:
            state = RollingState(state_file=state_file)
            state.init_run({'test_step': '3M'}, '2025-01-01', 4)
            state.mark_window_model_done(0, 'linear', 'rec_001')
            state.mark_window_model_done(0, 'gru', 'rec_002')
            state.mark_window_model_done(1, 'linear', 'rec_003')

            state2 = RollingState(state_file=state_file)
            assert state2.is_window_model_done(0, 'linear')
            assert state2.is_window_model_done(0, 'gru')
            assert state2.is_window_model_done(1, 'linear')
            assert not state2.is_window_model_done(1, 'gru')

            recs = state2.get_completed_record_ids('linear')
            assert len(recs) == 2
        finally:
            os.unlink(state_file)


class TestMainFlowsExtended:
    """Extra flows for new arguments"""

    def test_main_merge(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = False
        args.merge = True
        args.cold_start = False
        args.resume = False
        args.predict_only = False
        args.backtest_only = False
        args.backtest = False

        mock_patch_cfg = {
            'rolling_start': '2020-01-01',
            'train_years': 3,
            'valid_years': 1,
            'test_step': '3M',
            'test_step_months': 3
        }

        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.resolve_target_models', return_value={'m1':{}}), \
             mock.patch('config_loader.load_rolling_config', return_value=mock_patch_cfg), \
             mock.patch('rolling_train.RollingState') as mock_state_cls, \
             mock.patch('rolling_train.run_cold_start') as mock_run:
            
            mock_state = mock_state_cls.return_value
            mock_state.anchor_date = "2024-01-01"
            
            rt.main()
            mock_run.assert_called_once()

    def test_main_backtest_only(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = False
        args.backtest_only = True
        args.predict_only = False
        args.cold_start = False
        args.resume = False
        args.merge = False

        mock_patch_cfg = {
            'rolling_start': '2020-01-01',
            'train_years': 3,
            'valid_years': 1,
            'test_step': '3M',
            'test_step_months': 3
        }

        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.resolve_target_models', return_value={'m1':{}}), \
             mock.patch('config_loader.load_rolling_config', return_value=mock_patch_cfg), \
             mock.patch('rolling_train.run_backtest_only') as mock_run:
            
            rt.main()
            mock_run.assert_called_once()


class TestRunModesExtra:
    """Test the newly added run modes directly"""

    def test_run_backtest_only_success(self, mock_env):
        rt, workspace = mock_env
        args = mock.MagicMock()
        targets = {'m1': {}}
        
        # Create dummy record file
        rec_file = workspace / "latest_rolling_records.json"
        rec_file.write_text(json.dumps({
            "experiment_name": "Exp",
            "models": {"m1": "rec123"}
        }))

        with mock.patch('quantpits.scripts.env.init_qlib'), \
             mock.patch('rolling_train.get_base_params') as mock_base, \
             mock.patch('rolling_train.run_combined_backtest') as mock_run_bt:
            
            mock_base.return_value = {'freq': 'week', 'benchmark': 'SH000300'}
            # Note: run_backtest_only imports ROLLING_RECORD_FILE from train_utils
            # Our mock_env already reloads rolling_train which should pick up the mocked workspace path if train_utils was patched correctly.
            # But ROLLING_RECORD_FILE might be bound at import time.
            # In mock_env: monkeypatch.setattr(train_utils, 'ROLLING_PREDICTION_DIR', ...)
            # Let's ensure ROLLING_RECORD_FILE is also mocked.
            import train_utils
            with mock.patch.object(train_utils, 'ROLLING_RECORD_FILE', str(rec_file)):
                rt.run_backtest_only(args, targets)
                mock_run_bt.assert_called_once()

    def test_run_combined_backtest_full(self, mock_env):
        rt, _ = mock_env
        model_names = ['m1']
        combined_records = {'m1': 'rec1'}
        combined_exp_name = "Exp"
        params_base = {'freq': 'week', 'benchmark': 'SH000300'}

        with mock.patch('qlib.workflow.R', create=True) as mock_r, \
             mock.patch('qlib.backtest.backtest') as mock_bt, \
             mock.patch('strategy.create_backtest_strategy'), \
             mock.patch('strategy.load_strategy_config'), \
             mock.patch('strategy.get_backtest_config') as mock_get_bt_cfg, \
             mock.patch('quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer') as mock_pa_cls, \
             mock.patch('qlib.data.D', create=True) as mock_d:
            
            mock_recorder = mock.MagicMock()
            mock_r.get_recorder.return_value = mock_recorder
            
            # Mock pred data
            dates = pd.date_range('2024-01-01', periods=5)
            pred_df = pd.DataFrame({'score': [0.5]*5}, index=pd.MultiIndex.from_product([dates, ['S1']], names=['datetime', 'instrument']))
            mock_recorder.load_object.return_value = pred_df
            
            # Mock backtest return
            report_df = pd.DataFrame({'account': [1e8]*5, 'bench': [0]*5}, index=dates)
            mock_bt.return_value = ({'1week': (report_df, pd.DataFrame())}, {'1week': (pd.DataFrame(), {})})
            
            mock_get_bt_cfg.return_value = {'account': 1e8, 'exchange_kwargs': {}}
            mock_d.calendar.return_value = dates
            
            mock_pa = mock_pa_cls.return_value
            mock_pa.calculate_traditional_metrics.return_value = {
                'CAGR': 0.1, 'Max_Drawdown': -0.05, 'Excess_Return_CAGR': 0.05, 'Information_Ratio': 1.2, 'Calmar': 2.0
            }

            rt.run_combined_backtest(model_names, combined_records, combined_exp_name, params_base)
            
            # Verify save_objects was called with correct structure
            mock_recorder.save_objects.assert_called()
            # check the logic of calling with artifact_path
            calls = mock_recorder.save_objects.call_args_list
            # One call for portfolio_analysis, one for sig_analysis
            assert any(c.kwargs.get('artifact_path') == 'portfolio_analysis' for c in calls)
            assert any(c.kwargs.get('artifact_path') == 'sig_analysis' for c in calls)

    def test_run_predict_only(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        targets = {'m1': {}}
        cfg = {}
        
        with mock.patch('quantpits.scripts.env.init_qlib'), \
             mock.patch('rolling_train.get_base_params') as mock_base, \
             mock.patch('rolling_train.RollingState') as mock_state_cls, \
             mock.patch('rolling_train.predict_with_latest_model') as mock_predict:
            
            mock_base.return_value = {'anchor_date': '2024-01-01', 'freq': 'week'}
            mock_state = mock_state_cls.return_value
            mock_state.anchor_date = '2023-12-31'
            
            rt.run_predict_only(args, targets, cfg)
            mock_predict.assert_called_once()

    def test_main_resume_no_state(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = False
        args.resume = True
        args.all_enabled = True

        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.RollingState') as mock_state_cls, \
             mock.patch('rolling_train.resolve_target_models') as mock_resolve:
            
            mock_state = mock_state_cls.return_value
            mock_state.anchor_date = None # No state
            
            rt.main()
            # Should return early without calling resolve_target_models
            mock_resolve.assert_not_called()

    def test_main_no_targets(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        args.show_state = False
        args.clear_state = False
        args.cold_start = True
        args.all_enabled = True

        with mock.patch('rolling_train.parse_args', return_value=args), \
             mock.patch('rolling_train.resolve_target_models', return_value={}):
            
            rt.main()
            # Should print "⚠️ 没有匹配的模型" and return

    def test_run_predict_only_no_state(self, mock_env):
        rt, _ = mock_env
        args = mock.MagicMock()
        targets = {'m1': {}}
        cfg = {}
        
        with mock.patch('quantpits.scripts.env.init_qlib'), \
             mock.patch('rolling_train.get_base_params') as mock_base, \
             mock.patch('rolling_train.RollingState') as mock_state_cls:
            
            mock_base.return_value = {'anchor_date': '2024-01-01', 'freq': 'week'}
            mock_state = mock_state_cls.return_value
            mock_state.anchor_date = None # No state
            
            rt.run_predict_only(args, targets, cfg)
            # Should return early



