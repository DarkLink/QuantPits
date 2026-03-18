import os
import json
import yaml
import pytest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
import qlib.workflow

# Mock the constants in train_utils before importing
@pytest.fixture(autouse=True)
def mock_env_constants(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    config_dir = workspace / "config"
    config_dir.mkdir()
    data_dir = workspace / "data"
    data_dir.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    import importlib
    importlib.reload(env)

    # Now we can import train_utils
    from quantpits.utils import train_utils
    importlib.reload(train_utils)
    
    yield train_utils, workspace

def test_run_state(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    state_file = tmp_path / "run_state.json"
    
    # Load empty
    assert train_utils.load_run_state(str(state_file)) is None
    
    # Save 
    state = {
        "mode": "incremental",
        "completed": ["model1"]
    }
    train_utils.save_run_state(state, str(state_file))
    
    # Load
    loaded = train_utils.load_run_state(str(state_file))
    assert loaded["mode"] == "incremental"
    assert "model1" in loaded["completed"]
    
    # Clear
    with patch('quantpits.utils.train_utils.HISTORY_DIR', str(tmp_path / "history")):
        train_utils.clear_run_state(str(state_file))
        assert not os.path.exists(state_file)
        
    train_utils.clear_run_state(str(state_file))

def test_load_model_registry_default(mock_env_constants):
    train_utils, workspace = mock_env_constants
    with patch('quantpits.utils.train_utils.REGISTRY_FILE', "dummy.yaml"):
        with patch('builtins.open', mock_open(read_data=yaml.dump({'models': {}}))):
            registry = train_utils.load_model_registry()
            assert isinstance(registry, dict)

def test_run_state_default_file(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    state_file = tmp_path / "run_state.json"
    with patch('quantpits.utils.train_utils.RUN_STATE_FILE', str(state_file)):
        train_utils.save_run_state({"mode": "test"})
        assert train_utils.load_run_state()["mode"] == "test"
        with patch('quantpits.utils.train_utils.HISTORY_DIR', str(tmp_path / "history")):
            train_utils.clear_run_state()
        assert not os.path.exists(state_file)
        # Clear again
        train_utils.clear_run_state()

def test_load_model_registry(mock_env_constants):
    train_utils, workspace = mock_env_constants
    
    mock_registry = {
        'models': {
            'model1': {'enabled': True, 'algorithm': 'lstm'},
            'model2': {'enabled': False, 'algorithm': 'lgb'}
        }
    }
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_registry))):
        registry = train_utils.load_model_registry("dummy.yaml")
        assert 'model1' in registry
        assert 'model2' in registry
        assert registry['model1']['algorithm'] == 'lstm'

def test_get_enabled_models(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_registry = {
        'model1': {'enabled': True, 'algorithm': 'lstm'},
        'model2': {'enabled': False, 'algorithm': 'lgb'},
        'model3': {'enabled': True, 'algorithm': 'xgb'}
    }
    
    enabled = train_utils.get_enabled_models(mock_registry)
    assert len(enabled) == 2
    assert 'model1' in enabled
    assert 'model3' in enabled
    assert 'model2' not in enabled
    
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        enabled_default = train_utils.get_enabled_models()
        assert len(enabled_default) == 2

def test_backup_file_with_date(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello")
    
    history_dir = tmp_path / "history"
    
    backup_path = train_utils.backup_file_with_date(
        str(source_file), 
        history_dir=str(history_dir), 
        prefix="backup_prefix"
    )
    
    assert os.path.exists(backup_path)
    assert "backup_prefix" in backup_path
    with open(backup_path, 'r') as f:
        assert f.read() == "hello"

    assert train_utils.backup_file_with_date("nonexistent_file.txt") is None

def test_calculate_dates_slide(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    config_dict = {
        "market": "csi300",
        "benchmark": "SH000300",
        "train_date_mode": "last_trade_date",
        "data_slice_mode": "slide",
        "test_set_window": 1,
        "valid_set_window": 1,
        "train_set_windows": 3,
        "freq": "day",
        "current_full_cash": 200000.0
    }
        
    with patch('quantpits.utils.config_loader.load_workspace_config') as mock_load:
        mock_load.return_value = config_dict
        with patch('qlib.data.D') as mock_d:
            # Mock calendar to anchor on 2026-03-01
            mock_d.calendar.return_value = [pd.Timestamp("2026-03-01")]
            
            params = train_utils.calculate_dates()
            
            assert params["market"] == "csi300"
            assert params["account"] == 200000.0
            assert params["anchor_date"] == "2026-03-01"
            assert params["test_end_time"] == "2026-03-01"
            assert params["freq"] == "day"

def test_calculate_dates_fixed(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    config_dict = {
        "market": "csi300",
        "benchmark": "SH000300",
        "train_date_mode": "fixed",
        "current_date": "2026-01-01",
        "data_slice_mode": "fixed",
        "start_time": "2010-01-01",
        "fit_start_time": "2010-01-01",
        "fit_end_time": "2015-01-01",
        "valid_start_time": "2015-01-01",
        "valid_end_time": "2016-01-01",
        "test_start_time": "2016-01-01",
        "test_end_time": "2026-01-01"
    }
        
    with patch('quantpits.utils.config_loader.load_workspace_config') as mock_load:
        mock_load.return_value = config_dict
        params = train_utils.calculate_dates()
        assert params["anchor_date"] == "2026-01-01"
        assert params["start_time"] == "2010-01-01"
        assert params["fit_end_time"] == "2015-01-01"
def test_merge_train_records(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    record_file = tmp_path / "records.json"
    
    # Initial records
    initial = {
        "experiment_name": "exp1",
        "models": {
            "modelA": "id_A1",
            "modelB": "id_B1"
        }
    }
    record_file.write_text(json.dumps(initial))
    
    # New records to merge
    new_records = {
        "experiment_name": "exp2", # Should not override existing if we only merge models maybe? Wait, merge behavior replaces some metadata.
        "models": {
            "modelB": "id_B2", # update
            "modelC": "id_C1"  # new
        }
    }
    
    merged = train_utils.merge_train_records(new_records, record_file=str(record_file))
    
    assert merged["experiment_name"] == "exp2"
    assert merged["models"]["modelA"] == "id_A1" # Preserved
    assert merged["models"]["modelB"] == "id_B2" # Updated
    assert merged["models"]["modelC"] == "id_C1" # Added

    # default file
    with patch('quantpits.utils.train_utils.RECORD_OUTPUT_FILE', str(record_file)):
        with patch('quantpits.utils.train_utils.backup_file_with_date'):
            merged2 = train_utils.merge_train_records({"models": {"modelD": "id_D1"}})
            assert "modelD" in merged2["models"]

def test_merge_performance_file(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    perf1 = {"modelA": {"IC_Mean": 0.1}}
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    perf_file = out_dir / "model_performance_2026-01-01.json"
    perf_file.write_text(json.dumps(perf1))
    
    new_perf = {"modelB": {"IC_Mean": 0.2}, "modelA": {"IC_Mean": 0.15}}
    
    with patch('quantpits.utils.train_utils.HISTORY_DIR', str(tmp_path / "history")):
        merged = train_utils.merge_performance_file(new_perf, "2026-01-01", output_dir=str(out_dir))
    
    assert "modelB" in merged
    assert merged["modelB"]["IC_Mean"] == 0.2
    assert merged["modelA"]["IC_Mean"] == 0.15 # updated

    # Test default output_dir
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)):
        with patch('quantpits.utils.train_utils.HISTORY_DIR', str(tmp_path / "history")):
            # Note: since perf1 was not written to ROOT_DIR/output, it will be treated as new
            (tmp_path / "output").mkdir(parents=True, exist_ok=True)
            merged2 = train_utils.merge_performance_file(new_perf, "2026-01-01")
            assert merged2["modelB"]["IC_Mean"] == 0.2

def test_overwrite_train_records(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    records = {"experiment_name": "test_overwrite"}
    
    with patch('quantpits.utils.train_utils.backup_file_with_date'):
        train_utils.overwrite_train_records(records, record_file=str(tmp_path / 'rec.json'))
        
    with open(tmp_path / 'rec.json', 'r') as f:
        assert json.load(f)["experiment_name"] == "test_overwrite"
        
    # default record file
    with patch('quantpits.utils.train_utils.RECORD_OUTPUT_FILE', str(tmp_path / 'rec_default.json')):
        with patch('quantpits.utils.train_utils.backup_file_with_date'):
            train_utils.overwrite_train_records(records)
            assert os.path.exists(tmp_path / 'rec_default.json')

def test_get_models_by_names(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_registry = {
        'model1': {'enabled': True, 'algorithm': 'lstm'},
        'model2': {'enabled': False, 'algorithm': 'lgb'},
    }
    
    named = train_utils.get_models_by_names(["model1", "MISSING"], mock_registry)
    assert len(named) == 1
    assert "model1" in named

    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        named2 = train_utils.get_models_by_names(["model1"])
        assert "model1" in named2

def test_get_models_by_filter(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_registry = {
        'model1': {'algorithm': 'lstm', 'dataset': 'Alpha158', 'market': 'csi300', 'tags': ['ts', 'tree']},
        'model2': {'algorithm': 'gru', 'dataset': 'Alpha360', 'market': 'csi500', 'tags': ['nn']},
    }
    
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        # registry is None, matches model1
        res = train_utils.get_models_by_filter(algorithm='lstm')
        assert 'model1' in res and 'model2' not in res
        
        res = train_utils.get_models_by_filter(registry=mock_registry, dataset='Alpha360')
        assert 'model2' in res and 'model1' not in res
        
        res = train_utils.get_models_by_filter(registry=mock_registry, market='csi500')
        assert 'model2' in res
        
        res = train_utils.get_models_by_filter(registry=mock_registry, tag='tree')
        assert 'model1' in res
        # Unmatched tag should skip
        res = train_utils.get_models_by_filter(registry=mock_registry, tag='missing')
        assert len(res) == 0

def test_print_model_table(mock_env_constants, capsys):
    train_utils, _ = mock_env_constants
    
    mock_registry = {
        'model1': {'enabled': True, 'algorithm': 'lstm', 'dataset': 'd1', 'market': 'csi300', 'tags': ['t1', 't2']}
    }
    
    train_utils.print_model_table(mock_registry, "Test Title")
    captured = capsys.readouterr()
    assert "Test Title" in captured.out
    assert "model1" in captured.out
    assert "lstm" in captured.out
    assert "t1, t2" in captured.out

def test_inject_config(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_yaml = {
        'market': 'old_market',
        'benchmark': 'old_benchmark',
        'data_handler_config': {},
        'task': {
            'dataset': {'kwargs': {'segments': {}}}
        }
    }
    
    params = {
        'freq': 'week',
        'market': 'new_market',
        'benchmark': 'new_benchmark',
        'start_time': '2000',
        'end_time': '2010',
        'fit_start_time': '2000',
        'fit_end_time': '2005',
        'valid_start_time': '2006',
        'valid_end_time': '2008',
        'test_start_time': '2009',
        'test_end_time': '2010'
    }
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_yaml))):
        config = train_utils.inject_config("dummy.yaml", params)
        
        assert config['market'] == 'new_market'
        assert config['benchmark'] == 'new_benchmark'
        assert config['data_handler_config']['label'] == ["Ref($close, -6) / Ref($close, -1) - 1"]
        assert config['task']['dataset']['kwargs']['segments']['train'] == ['2000', '2005']

def test_inject_config_extra_branches(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_yaml = {
        'market': 'old',
        'benchmark': 'old',
        'strategy': {'params': {}},
        'data_handler_config': {},
        'task': {
            'dataset': {'kwargs': {'segments': {}}},
            'record': [{'class': 'PortAnaRecord', 'kwargs': {'config': {}}}, {'class': 'SigAnaRecord'}],
            'model': {'kwargs': {'base_model': 'dummy', 'model_path': 'old_path', 'd_feat': 20}}
        },
        'port_analysis_config': {}
    }
    
    params = {
        'freq': 'day',
        'market': 'new_market',
        'benchmark': 'new_benchmark',
        'start_time': '2000', 'end_time': '2010', 'fit_start_time': '2000', 'fit_end_time': '2005',
        'valid_start_time': '2006', 'valid_end_time': '2008', 'test_start_time': '2009', 'test_end_time': '2010',
        # Used for line 325 (label_formula in params)
        'label_formula': 'MyLabel',
        # Used for line 310-312 (strategy params)
        'topk': 50, 'n_drop': 5, 'buy_suggestion_factor': 4,
        'account': 100000.0
    }
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_yaml))):
        with patch('quantpits.utils.strategy.generate_port_analysis_config') as mock_gen_pa:
            mock_gen_pa.return_value = {'backtest': {'account': 10}}
            with patch('quantpits.utils.train_utils.resolve_pretrained_path') as mock_resolve:
                # Test no_pretrain = True
                cfg1 = train_utils.inject_config("dummy.yaml", params, model_name="dummy_model", no_pretrain=True)
                assert 'model_path' not in cfg1['task']['model']['kwargs']
                assert cfg1['strategy']['params']['topk'] == 50
                assert cfg1['data_handler_config']['label'] == ['MyLabel']
                assert cfg1['task']['record'][1]['kwargs']['ann_scaler'] == 252
                
                # Test no_pretrain = False, resolve returns path
                mock_resolve.return_value = '/path/to/pretrain.pkl'
                with patch('quantpits.utils.train_utils.validate_pretrain_compatibility') as mock_val:
                    cfg2 = train_utils.inject_config("dummy.yaml", params, model_name="dummy_model", no_pretrain=False)
                    assert cfg2['task']['model']['kwargs']['model_path'] == '/path/to/pretrain.pkl'
                    mock_val.assert_called_once()
                
                # Test no_pretrain = False, resolve returns None
                mock_resolve.return_value = None
                cfg3 = train_utils.inject_config("dummy.yaml", params, model_name="dummy_model", no_pretrain=False)
                assert 'model_path' not in cfg3['task']['model']['kwargs']


# ── train_single_model ───────────────────────────────────────────────────

def test_train_single_model(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    yaml_file = tmp_path / "test_model.yaml"
    base_config = {
        "data_handler_config": {},
        "task": {
            "model": {"class": "DummyModel"},
            "dataset": {"class": "DummyDataset", "kwargs": {"segments": {}}},
            "record": [
                {"class": "SigAnaRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}}
            ]
        }
    }
    yaml_file.write_text(yaml.dump(base_config))
    
    params = {
        "freq": "day",
        "market": "csi300",
        "benchmark": "SH000300",
        "start_time": "2010",
        "end_time": "2020",
        "fit_start_time": "2010",
        "fit_end_time": "2015",
        "valid_start_time": "2015",
        "valid_end_time": "2016",
        "test_start_time": "2016",
        "test_end_time": "2020",
        "anchor_date": "2026-03-01",
        "account": 100000.0
    }
    
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    with patch('quantpits.utils.train_utils.PREDICTION_OUTPUT_DIR', str(pred_dir)):
        with patch('qlib.utils.init_instance_by_config') as mock_init_instance:
            with patch('qlib.workflow.R') as mock_R:
                # Setup mocks for models and recorder
                mock_model = MagicMock()
                mock_dataset = MagicMock()
                mock_pred = pd.DataFrame({"score": [1, 2]})
                mock_model.predict.return_value = mock_pred
                
                mock_record_obj = MagicMock()
                
                # Map side effects for init_instance based on config
                def side_effect_init(cfg, recorder=None):
                    cls = cfg.get("class")
                    if cls == "DummyModel": return mock_model
                    if cls == "DummyDataset": return mock_dataset
                    return mock_record_obj
                    
                mock_init_instance.side_effect = side_effect_init
                
                mock_recorder = MagicMock()
                mock_recorder.info = {'id': 'test_rid_123'}
                mock_ic_series = pd.Series([0.1, 0.2, 0.3])
                mock_recorder.load_object.return_value = mock_ic_series
                
                mock_R.get_recorder.return_value = mock_recorder
                
                # Run
                result = train_utils.train_single_model("DummyLGBM", str(yaml_file), params, "TestExp")
                
                assert result['success'] is True
                assert result['record_id'] == 'test_rid_123'
                
                # Verify performance extraction
                assert 'IC_Mean' in result['performance']
                assert np.isclose(result['performance']['IC_Mean'], 0.2)
                
                # Verify R interactions
                mock_R.start.assert_called_once()
                mock_R.set_tags.assert_called_with(model="DummyLGBM", anchor_date="2026-03-01")

                # Test failure in load_object
                mock_recorder.load_object.side_effect = Exception("IC Error")
                result_ic_error = train_utils.train_single_model("DummyLGBM", str(yaml_file), params, "TestExp")
                assert result_ic_error['success'] is True
                assert 'IC_Mean' not in result_ic_error['performance']

    # Test file doesn't exist
    res_no_file = train_utils.train_single_model("DummyLGBM", "nonexistent.yaml", params, "TestExp")
    assert not res_no_file['success']
    assert "不存在" in res_no_file['error']

    # Test generic exception inside try block
    with patch('os.path.exists', return_value=True):
        with patch('quantpits.utils.train_utils.inject_config', return_value=base_config):
            with patch('qlib.workflow.R'):
                with patch('qlib.utils.init_instance_by_config', side_effect=Exception("Inner Error")):
                    res_err = train_utils.train_single_model("Dummy", "any.yaml", params, "Exp")
                    assert not res_err['success']
                    assert "Inner Error" in res_err['error']


class TestPretrainManagement:
    @patch('quantpits.utils.train_utils.PRETRAINED_DIR', '/tmp/mock_pretrain_dir')
    @patch('os.path.exists')
    def test_get_pretrained_model_path(self, mock_exists):
        from quantpits.utils.train_utils import get_pretrained_model_path
        
        # Test with anchor_date
        mock_exists.return_value = True
        path = get_pretrained_model_path("lstm_Alpha158", "2026-03-01")
        assert path == "/tmp/mock_pretrain_dir/lstm_Alpha158_2026-03-01.pkl"
        
        # Test latest
        path = get_pretrained_model_path("lstm_Alpha158")
        assert path == "/tmp/mock_pretrain_dir/lstm_Alpha158_latest.pkl"
        
        # Test not exists
        mock_exists.return_value = False
        assert get_pretrained_model_path("lstm_Alpha158") is None

    @patch('quantpits.utils.train_utils._get_inner_model')
    @patch('quantpits.utils.train_utils.PRETRAINED_DIR', '/tmp/mock_pretrain_dir')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_pretrained_model(self, mock_file, mock_copy, mock_makedirs, mock_get_inner):
        from quantpits.utils.train_utils import save_pretrained_model
        
        # Mock model
        mock_inner = MagicMock()
        mock_inner.state_dict.return_value = {"weight": [1, 2, 3]}
        mock_get_inner.return_value = mock_inner
        
        # Use patch.dict to mock torch in sys.modules so 'import torch' and 'torch.save' work
        mock_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': mock_torch}):
            saved_path = save_pretrained_model(
                MagicMock(), "lstm_Alpha158", "2026-03-01", 
                d_feat=20, hidden_size=64, num_layers=2
            )
        
        assert saved_path == "/tmp/mock_pretrain_dir/lstm_Alpha158_2026-03-01.pkl"
        mock_torch.save.assert_called_once_with({"weight": [1, 2, 3]}, saved_path)
        
        # Verify JSON writing
        mock_file.assert_any_call("/tmp/mock_pretrain_dir/lstm_Alpha158_2026-03-01.json", 'w')
        
        # Verify copying to latest
        assert mock_copy.call_count == 2 # pkl and json

    @patch('quantpits.utils.train_utils.PRETRAINED_DIR', '/tmp/mock_pretrain_dir')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"d_feat": 20}')
    def test_load_pretrained_metadata(self, mock_file, mock_exists):
        from quantpits.utils.train_utils import load_pretrained_metadata
        mock_exists.return_value = True
        
        meta = load_pretrained_metadata("lstm_Alpha158")
        assert meta["d_feat"] == 20
        mock_exists.assert_called_with("/tmp/mock_pretrain_dir/lstm_Alpha158_latest.json")

        # Test with anchor_date and existing file
        meta_date = load_pretrained_metadata("lstm_Alpha158", "2026-03-01")
        assert meta_date["d_feat"] == 20
        mock_exists.assert_called_with("/tmp/mock_pretrain_dir/lstm_Alpha158_2026-03-01.json")

        # Test non-existing
        mock_exists.return_value = False
        assert load_pretrained_metadata("lstm_Alpha158") is None

    def test_get_inner_model(self):
        from quantpits.utils.train_utils import _get_inner_model
        class MockInner: pass
        class MockModel:
            def __init__(self):
                self.LSTM_model = MockInner()
        assert isinstance(_get_inner_model(MockModel()), MockInner)
        
        class MockModel2:
            def __init__(self):
                self.gru_model = MockInner()
        assert isinstance(_get_inner_model(MockModel2()), MockInner)
        
        class BadModel: pass
        with pytest.raises(ValueError):
            _get_inner_model(BadModel())

    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_pretrained_model_path')
    def test_resolve_pretrained_path(self, mock_get_path, mock_load_registry):
        from quantpits.utils.train_utils import resolve_pretrained_path
        
        mock_load_registry.return_value = {
            "gats": {"pretrain_source": "lstm"},
            "lgbm": {}
        }
        
        mock_get_path.return_value = "/tmp/lstm_latest.pkl"
        
        # Has pretrain_source
        path = resolve_pretrained_path("gats")
        assert path == "/tmp/lstm_latest.pkl"
        mock_get_path.assert_called_with("lstm")
        
        # No pretrain_source
        assert resolve_pretrained_path("lgbm") is None
        
        # pretrain_source but file missing
        mock_get_path.return_value = None
        assert resolve_pretrained_path("gats") is None

    @patch('quantpits.utils.train_utils.load_pretrained_metadata')
    def test_validate_pretrain_compatibility(self, mock_load_meta):
        from quantpits.utils.train_utils import validate_pretrain_compatibility
        
        # Match
        mock_load_meta.return_value = {"d_feat": 20}
        validate_pretrain_compatibility("gats", "/path/to/lstm_latest.pkl", 20)
        
        # Mismatch
        with pytest.raises(ValueError, match="Feature 不匹配"):
            validate_pretrain_compatibility("gats", "/path/to/lstm_latest.pkl", 6)
            
        # No metadata
        mock_load_meta.return_value = None
        validate_pretrain_compatibility("gats", "/path/to/lstm_latest.pkl", 20) # should not raise

def test_predict_single_model(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    yaml_file = tmp_path / "test_model.yaml"
    base_config = {
        "task": {
            "model": {"class": "DummyModel"},
            "dataset": {"class": "DummyDataset"},
            "record": [
                {"class": "SigAnaRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}}
            ]
        }
    }
    yaml_file.write_text(yaml.dump(base_config))
    params = {"anchor_date": "2026-03-01"}
    model_info = {"yaml_file": str(yaml_file)}
    source_records = {"experiment_name": "SrcExp", "models": {"Model1": "SrcId"}}
    
    with patch('qlib.utils.init_instance_by_config') as mock_init_instance:
        with patch('qlib.workflow.R') as mock_R:
            mock_model = MagicMock()
            mock_dataset = MagicMock()
            mock_model.predict.return_value = pd.DataFrame()
            
            def side_effect_init(cfg, recorder=None):
                cls = cfg.get("class")
                if cls == "DummyModel": return mock_model
                if cls == "DummyDataset": return mock_dataset
                return MagicMock() # record obj
                
            mock_init_instance.side_effect = side_effect_init
            
            mock_recorder = MagicMock()
            mock_recorder.info = {'id': 'pred_rid'}
            def load_obj_side_effect(*args, **kwargs):
                if args and "model.pkl" in args[0]:
                    return mock_model
                if kwargs and kwargs.get('name', '') == "model.pkl":
                    return mock_model
                
                # Mock IC series
                mock_ic = MagicMock()
                mock_ic.mean.return_value = 0.5
                mock_ic.std.return_value = 0.1
                return mock_ic
            mock_recorder.load_object.side_effect = load_obj_side_effect
            
            mock_R.get_recorder.return_value = mock_recorder
            
            # 1. Test success branch
            with patch('quantpits.utils.train_utils.inject_config', return_value=base_config):
                with patch('os.path.exists', return_value=True):
                    res = train_utils.predict_single_model("Model1", model_info, params, "Exp", source_records)
                    assert res['success']
                    assert 'IC_Mean' in res['performance']
                    assert np.isclose(res['performance']['IC_Mean'], 0.5)

            # 1b. Test IC error branch
            with patch('quantpits.utils.train_utils.inject_config', return_value=base_config):
                with patch('os.path.exists', return_value=True):
                    def error_ic_side_effect(*args, **kwargs):
                        if args and "model.pkl" in args[0]:
                            return mock_model
                        raise Exception("IC Error")
                    mock_recorder.load_object.side_effect = error_ic_side_effect
                    res_ic_err = train_utils.predict_single_model("Model1", model_info, params, "Exp", source_records)
                    assert res_ic_err['success']
                    assert 'IC_Mean' not in res_ic_err['performance']
            
            # 2. Test main exception inside try block
            with patch('quantpits.utils.train_utils.inject_config', return_value=base_config):
                with patch('os.path.exists', return_value=True):
                    with patch('qlib.utils.init_instance_by_config', side_effect=ValueError("Init Err")):
                        res2 = train_utils.predict_single_model("Model1", model_info, params, "Exp", source_records)
                        assert not res2['success']
                        assert "Init Err" in res2['error']
            
            # 3. Test missing yaml
            bad_info = {"yaml_file": "not_exist.yaml"}
            res3 = train_utils.predict_single_model("Model1", bad_info, params, "Exp", source_records)
            assert not res3['success']
            
            # 4. Test model missing in source records
            res4 = train_utils.predict_single_model("MissingModel", model_info, params, "Exp", source_records)
            assert not res4['success']
            assert "不在源训练记录中" in res4['error']

def test_resolve_target_models(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    class Args:
        pass
        
    mock_registry = {
        'model1': {'enabled': True, 'algorithm': 'lstm'},
        'model2': {'enabled': False, 'algorithm': 'lgb'}
    }
    
    # 1. args.models
    args1 = Args()
    args1.models = "model1, model2"
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res1 = train_utils.resolve_target_models(args1)
        assert 'model1' in res1 and 'model2' in res1

    # 2. args.all_enabled
    args2 = Args()
    args2.all_enabled = True
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res2 = train_utils.resolve_target_models(args2)
        assert 'model1' in res2 and 'model2' not in res2

    # 3. filters
    args3 = Args()
    args3.algorithm = "lgb"
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res3 = train_utils.resolve_target_models(args3)
        assert 'model1' not in res3 and 'model2' in res3

    # 4. skip
    args4 = Args()
    args4.models = "model1, model2"
    args4.skip = "model2"
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res4 = train_utils.resolve_target_models(args4)
        assert 'model1' in res4 and 'model2' not in res4

    # 5. no targets
    args5 = Args()
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        assert train_utils.resolve_target_models(args5) is None

def test_show_model_list(mock_env_constants, tmp_path, capsys):
    train_utils, _ = mock_env_constants
    
    class Args:
        pass
        
    mock_registry = {
        'model1': {'enabled': True, 'dataset': 'Alpha158'},
        'model2': {'enabled': False, 'dataset': 'Alpha360'}
    }
    
    args1 = Args()
    args1.algorithm = "lstm" # to trigger filters
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        with patch('quantpits.utils.train_utils.get_models_by_filter', return_value={'model1': mock_registry['model1']}):
            train_utils.show_model_list(args1)
            
    args2 = Args() # all models
    source_records_file = tmp_path / "records.json"
    source_records_file.write_text(json.dumps({"models": {"model1": "id1"}}))
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        train_utils.show_model_list(args2, source_records_file=str(source_records_file))
