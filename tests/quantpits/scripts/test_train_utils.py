import os
import json
import yaml
import pytest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np

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
    
    from quantpits.scripts import env
    import importlib
    importlib.reload(env)

    # Now we can import train_utils
    from quantpits.scripts import train_utils
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
    with patch('quantpits.scripts.train_utils.HISTORY_DIR', str(tmp_path / "history")):
        train_utils.clear_run_state(str(state_file))
        assert not os.path.exists(state_file)
        
    train_utils.clear_run_state(str(state_file)) # safe to call when not exists

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
        
    with patch('config_loader.load_workspace_config') as mock_load:
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
        
    with patch('config_loader.load_workspace_config') as mock_load:
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

def test_merge_performance_file(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    perf1 = {"modelA": {"IC_Mean": 0.1}}
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    perf_file = out_dir / "model_performance_2026-01-01.json"
    perf_file.write_text(json.dumps(perf1))
    
    new_perf = {"modelB": {"IC_Mean": 0.2}, "modelA": {"IC_Mean": 0.15}}
    
    with patch('quantpits.scripts.train_utils.HISTORY_DIR', str(tmp_path / "history")):
        merged = train_utils.merge_performance_file(new_perf, "2026-01-01", output_dir=str(out_dir))
    
    assert "modelB" in merged
    assert merged["modelB"]["IC_Mean"] == 0.2
    assert merged["modelA"]["IC_Mean"] == 0.15 # updated

def test_get_models_by_names(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_registry = {
        'model1': {'enabled': True, 'algorithm': 'lstm'},
        'model2': {'enabled': False, 'algorithm': 'lgb'},
    }
    
    named = train_utils.get_models_by_names(["model1", "MISSING"], mock_registry)
    assert len(named) == 1
    assert "model1" in named

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
    
    with patch('quantpits.scripts.train_utils.PREDICTION_OUTPUT_DIR', str(pred_dir)):
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
                
                # Verify prediction saved
                pred_path = os.path.join(str(pred_dir), "DummyLGBM_2026-03-01.csv")
                assert os.path.exists(pred_path)
                
                # Verify R interactions
                mock_R.start.assert_called_once()
                mock_R.set_tags.assert_called_with(model="DummyLGBM", anchor_date="2026-03-01")


class TestPretrainManagement:
    @patch('quantpits.scripts.train_utils.PRETRAINED_DIR', '/tmp/mock_pretrain_dir')
    @patch('os.path.exists')
    def test_get_pretrained_model_path(self, mock_exists):
        from quantpits.scripts.train_utils import get_pretrained_model_path
        
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

    @patch('quantpits.scripts.train_utils._get_inner_model')
    @patch('quantpits.scripts.train_utils.PRETRAINED_DIR', '/tmp/mock_pretrain_dir')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_pretrained_model(self, mock_file, mock_copy, mock_makedirs, mock_get_inner):
        from quantpits.scripts.train_utils import save_pretrained_model
        
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

    @patch('quantpits.scripts.train_utils.PRETRAINED_DIR', '/tmp/mock_pretrain_dir')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"d_feat": 20}')
    def test_load_pretrained_metadata(self, mock_file, mock_exists):
        from quantpits.scripts.train_utils import load_pretrained_metadata
        mock_exists.return_value = True
        
        meta = load_pretrained_metadata("lstm_Alpha158")
        assert meta["d_feat"] == 20
        mock_exists.assert_called_with("/tmp/mock_pretrain_dir/lstm_Alpha158_latest.json")

    @patch('quantpits.scripts.train_utils.load_model_registry')
    @patch('quantpits.scripts.train_utils.get_pretrained_model_path')
    def test_resolve_pretrained_path(self, mock_get_path, mock_load_registry):
        from quantpits.scripts.train_utils import resolve_pretrained_path
        
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

    @patch('quantpits.scripts.train_utils.load_pretrained_metadata')
    def test_validate_pretrain_compatibility(self, mock_load_meta):
        from quantpits.scripts.train_utils import validate_pretrain_compatibility
        
        # Match
        mock_load_meta.return_value = {"d_feat": 20}
        validate_pretrain_compatibility("gats", "/path/to/lstm_latest.pkl", 20)
        
        # Mismatch
        with pytest.raises(ValueError, match="Feature 不匹配"):
            validate_pretrain_compatibility("gats", "/path/to/lstm_latest.pkl", 6)
            
        # No metadata
        mock_load_meta.return_value = None
        validate_pretrain_compatibility("gats", "/path/to/lstm_latest.pkl", 20) # should not raise
