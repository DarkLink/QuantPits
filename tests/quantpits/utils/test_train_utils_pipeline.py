import os
import yaml
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

@pytest.fixture
def mock_qlib_objects():
    mock_model = MagicMock()
    mock_dataset = MagicMock()
    mock_recorder = MagicMock()
    mock_recorder.info = {'id': 'test_rid_123'}
    return mock_model, mock_dataset, mock_recorder

def test_calculate_dates_slide(mock_env_constants):
    train_utils, _ = mock_env_constants
    config_dict = {
        "market": "csi300", "benchmark": "SH000300", "train_date_mode": "last_trade_date",
        "data_slice_mode": "slide", "test_set_window": 1, "valid_set_window": 1,
        "train_set_windows": 3, "freq": "day", "current_full_cash": 200000.0
    }
    with patch('quantpits.utils.config_loader.load_workspace_config', return_value=config_dict):
        with patch('qlib.data.D') as mock_d:
            mock_d.calendar.return_value = [pd.Timestamp("2026-03-01")]
            params = train_utils.calculate_dates()
            assert params["anchor_date"] == "2026-03-01"
            assert params["test_end_time"] == "2026-03-01"

def test_calculate_dates_fixed(mock_env_constants):
    train_utils, _ = mock_env_constants
    config_dict = {
        "market": "csi300", "benchmark": "SH000300", "train_date_mode": "fixed",
        "current_date": "2026-01-01", "data_slice_mode": "fixed",
        "start_time": "2010-01-01", "fit_start_time": "2010-01-01", "fit_end_time": "2015-01-01",
        "valid_start_time": "2015-01-01", "valid_end_time": "2016-01-01",
        "test_start_time": "2016-01-01", "test_end_time": "2026-01-01"
    }
    with patch('quantpits.utils.config_loader.load_workspace_config', return_value=config_dict):
        params = train_utils.calculate_dates()
        assert params["anchor_date"] == "2026-01-01"
        assert params["fit_end_time"] == "2015-01-01"

def test_inject_config(mock_env_constants):
    train_utils, _ = mock_env_constants
    mock_yaml = {
        'market': 'old_market', 'benchmark': 'old_benchmark', 'data_handler_config': {},
        'task': {'dataset': {'kwargs': {'segments': {}}}}
    }
    params = {
        'freq': 'week', 'market': 'new_market', 'benchmark': 'new_benchmark',
        'start_time': '2000', 'end_time': '2010', 'fit_start_time': '2000', 'fit_end_time': '2005',
        'valid_start_time': '2006', 'valid_end_time': '2008', 'test_start_time': '2009', 'test_end_time': '2010'
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
        'strategy': {'params': {}}, 'data_handler_config': {},
        'task': {
            'dataset': {'kwargs': {'segments': {}}},
            'record': [{'class': 'PortAnaRecord', 'kwargs': {'config': {}}}, {'class': 'SigAnaRecord'}],
            'model': {'kwargs': {'base_model': 'dummy', 'model_path': 'old_path', 'd_feat': 20}}
        },
        'port_analysis_config': {}
    }
    params = {
        'freq': 'day', 'market': 'new', 'benchmark': 'new', 
        'start_time': '2000', 'end_time': '2010', 'fit_start_time': '2000', 'fit_end_time': '2005',
        'valid_start_time': '2006', 'valid_end_time': '2008', 'test_start_time': '2009', 'test_end_time': '2010',
        'label_formula': 'MyLabel', 'topk': 50, 'n_drop': 5, 'buy_suggestion_factor': 4, 'account': 100000.0
    }
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_yaml))):
        # Ensure backtest has account key
        with patch('quantpits.utils.strategy.generate_port_analysis_config', return_value={'backtest': {'account': 0}}):
            with patch('quantpits.utils.train_utils.resolve_pretrained_path') as mock_resolve:
                cfg1 = train_utils.inject_config("dummy.yaml", params, model_name="dummy", no_pretrain=True)
                assert 'model_path' not in cfg1['task']['model']['kwargs']
                assert cfg1['strategy']['params']['topk'] == 50
                
                mock_resolve.return_value = '/path.pkl'
                with patch('quantpits.utils.train_utils.validate_pretrain_compatibility'):
                    cfg2 = train_utils.inject_config("dummy.yaml", params, model_name="dummy", no_pretrain=False)
                    assert cfg2['task']['model']['kwargs']['model_path'] == '/path.pkl'

def test_train_single_model(mock_env_constants, mock_qlib_objects, tmp_path):
    train_utils, workspace = mock_env_constants
    mock_model, mock_dataset, mock_recorder = mock_qlib_objects
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml.dump({"task": {"model": {}, "dataset": {}, "record": []}}))
    params = {"freq": "day", "market": "c3", "anchor_date": "2026-03-01"}
    
    # Use placeholders to trigger lines 692-695
    task_config = {
        "task": {
            "model": {}, "dataset": {}, 
            "record": [{"class": "SigAnaRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}}]
        }
    }
    
    with patch('quantpits.utils.train_utils.inject_config', return_value=task_config):
        with patch('qlib.utils.init_instance_by_config', side_effect=[mock_model, mock_dataset, MagicMock(), mock_model, mock_dataset, MagicMock()]):
            with patch('qlib.workflow.R') as mock_R:
                mock_R.get_recorder.return_value = mock_recorder
                mock_model.predict.return_value = pd.DataFrame({"score": [1]})
                mock_recorder.load_object.side_effect = [pd.Series([0.1, 0.2])]
                
                res = train_utils.train_single_model("M1", str(yaml_file), params, "E")
                assert res['success']
                
                # Test Failure Case: IC Error
                mock_recorder.load_object.side_effect = Exception("IC Error")
                res_ic_err = train_utils.train_single_model("M1", str(yaml_file), params, "E")
                assert res_ic_err['success']
                assert 'IC_Mean' not in res_ic_err['performance']
                
    # Test file doesn't exist (line 649-651)
    res_no_file = train_utils.train_single_model("M1", "not_exist.yaml", params, "E")
    assert not res_no_file['success']

def test_predict_single_model(mock_env_constants, mock_qlib_objects, tmp_path):
    train_utils, _ = mock_env_constants
    mock_model, mock_dataset, mock_recorder = mock_qlib_objects
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-03-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}
    
    task_config = {
        "task": {
            "dataset": {}, 
            "record": [{"class": "SigAnaRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}}]
        }
    }
    
    with patch('qlib.utils.init_instance_by_config', side_effect=[mock_model, mock_dataset, MagicMock()]):
        with patch('qlib.workflow.R') as mock_R:
            mock_R.get_recorder.return_value = mock_recorder
            mock_recorder.load_object.side_effect = [mock_model, pd.Series([0.5])]
            with patch('quantpits.utils.train_utils.inject_config', return_value=task_config):
                with patch('os.path.exists', return_value=True):
                    res = train_utils.predict_single_model("M1", {"yaml_file": str(yaml_file)}, params, "E", source)
                    assert res['success']

def test_show_model_list(mock_env_constants, capsys):
    train_utils, _ = mock_env_constants
    class Args:
        algorithm = None
        dataset = None
        market = None
        tag = None
    args = Args()
    with patch('quantpits.utils.train_utils.load_model_registry', return_value={"m1": {"enabled": True}}):
        train_utils.show_model_list(args)
    captured = capsys.readouterr()
    assert "m1" in captured.out

class TestPretrainManagement:
    def test_get_pretrained_model_path(self, mock_env_constants):
        train_utils, _ = mock_env_constants
        with patch('os.path.exists', return_value=True):
            assert "lstm_latest.pkl" in train_utils.get_pretrained_model_path("lstm")
            assert "lstm_2026.pkl" in train_utils.get_pretrained_model_path("lstm", "2026")

    @patch('quantpits.utils.train_utils._get_inner_model')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_pretrained_model(self, mock_file, mock_copy, mock_makedirs, mock_get_inner, mock_env_constants):
        train_utils, _ = mock_env_constants
        mock_inner = MagicMock(); mock_get_inner.return_value = mock_inner
        mock_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': mock_torch}):
            path = train_utils.save_pretrained_model(MagicMock(), "lstm", "2026", d_feat=2, hidden_size=6, num_layers=2)
            assert "lstm_2026.pkl" in path
            assert mock_torch.save.called
            assert mock_copy.call_count == 2

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"d_feat": 20}')
    def test_load_pretrained_metadata(self, mock_file, mock_exists, mock_env_constants):
        train_utils, _ = mock_env_constants
        meta = train_utils.load_pretrained_metadata("lstm")
        assert meta["d_feat"] == 20
        assert train_utils.load_pretrained_metadata("lstm", "2026")["d_feat"] == 20

    @patch('quantpits.utils.train_utils.load_model_registry', return_value={"gats": {"pretrain_source": "lstm"}})
    @patch('quantpits.utils.train_utils.get_pretrained_model_path', return_value="/p.pkl")
    def test_resolve_pretrained_path(self, mock_get, mock_load, mock_env_constants, capsys):
        train_utils, _ = mock_env_constants
        assert train_utils.resolve_pretrained_path("gats") == "/p.pkl"
        
        # Branch: not found
        mock_get.return_value = None
        assert train_utils.resolve_pretrained_path("gats") is None
        captured = capsys.readouterr()
        assert "未找到" in captured.out

    def test_validate_pretrain_compatibility(self, mock_env_constants):
        train_utils, _ = mock_env_constants
        with patch('quantpits.utils.train_utils.load_pretrained_metadata', return_value={"d_feat": 20}):
            # Match
            train_utils.validate_pretrain_compatibility("m", "/lstm_latest.pkl", 20)
            # Mismatch
            with pytest.raises(ValueError, match="Feature 不匹配"):
                train_utils.validate_pretrain_compatibility("m", "/lstm_latest.pkl", 10)
        # No metadata
        with patch('quantpits.utils.train_utils.load_pretrained_metadata', return_value=None):
            train_utils.validate_pretrain_compatibility("m", "/lstm_latest.pkl", 20)

    def test_get_inner_model(self, mock_env_constants):
        train_utils, _ = mock_env_constants
        class M: pass
        obj = M(); obj.LSTM_model = "inner"
        assert train_utils._get_inner_model(obj) == "inner"
        with pytest.raises(ValueError): train_utils._get_inner_model(M())
