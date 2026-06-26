import os
import yaml
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open

# ---------------------------------------------------------------------------
# Test inject_config_for_fold
# ---------------------------------------------------------------------------

def test_inject_config_for_fold(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    dummy_yaml = """
market: csi300
benchmark: SH000300
strategy:
  class: TopkDropoutStrategy
  params:
    topk: 50
    n_drop: 5
data_handler_config:
  class: Alpha158
  module_path: qlib.contrib.data.handler
  kwargs:
    instruments: csi300
    start_time: '2015-01-01'
    end_time: '2020-01-01'
task:
  dataset:
    class: TSDatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: Alpha158
      segments:
        train: ['2015-01-01', '2019-01-01']
        valid: ['2019-01-02', '2020-01-01']
  model:
    class: LGBModel
    kwargs:
      base_model: lgb
      d_feat: 158
      model_path: /old/path
  record:
    - class: PortAnaRecord
      kwargs:
        config: {}
        model: <MODEL>
        dataset: <DATASET>
    - class: SigAnaRecord
      kwargs: {}
port_analysis_config:
  backtest:
    start_time: '2020-01-02'
    end_time: '2020-06-01'
    account: 10000000
    benchmark: SH000300
"""
    params = {
        'market': 'csi300_test',
        'benchmark': 'SH000300_test',
        'test_start_time': '2020-01-02',
        'test_end_time': '2020-06-01',
        'anchor_date': '2020-06-01',
        'label_formula': 'Ref($close, -2) / Ref($close, -1) - 1',
    }
    
    fold = {
        'fold_idx': 0,
        'train_segments': [['2015-01-01', '2017-12-31'], ['2018-06-01', '2018-12-31']],
        'valid_start_time': '2018-01-01',
        'valid_end_time': '2018-05-31',
    }
    
    with patch("builtins.open", mock_open(read_data=dummy_yaml)):
        config = train_utils.inject_config_for_fold(
            "dummy.yaml", params, fold, model_name="lgb", no_pretrain=False
        )
        
    assert config['market'] == 'csi300_test'
    assert config['benchmark'] == 'SH000300_test'
    
    dh = config['data_handler_config']
    # fold_train_start = min('2015-01-01', '2018-06-01') = '2015-01-01'
    assert dh['fit_start_time'] == '2015-01-01'
    # fold_train_end = max('2017-12-31', '2018-12-31') = '2018-12-31'
    assert dh['fit_end_time'] == '2018-12-31'
    assert dh['label'] == ['Ref($close, -2) / Ref($close, -1) - 1']
    
    ds_cfg = config['task']['dataset']
    # Class swapped to PurgedTSDatasetH
    assert ds_cfg['class'] == 'PurgedTSDatasetH'
    assert ds_cfg['module_path'] == 'quantpits.data.cpcv_dataset'
    
    segs = ds_cfg['kwargs']['segments']
    assert segs['train'] == fold['train_segments']
    assert segs['valid'] == ['2018-01-01', '2018-05-31']
    assert segs['test'] == ['2020-01-02', '2020-06-01']


# ---------------------------------------------------------------------------
# Test train_cpcv_model & predict_cpcv_model
# ---------------------------------------------------------------------------

def test_train_cpcv_model_and_predict(mock_env_constants, monkeypatch):
    train_utils, workspace = mock_env_constants
    
    # 1. Prepare dummy yaml file
    yaml_file = os.path.join(workspace, "cpcv_task.yaml")
    dummy_yaml = {
        'market': 'csi300',
        'benchmark': 'SH000300',
        'data_handler_config': {
            'class': 'Alpha158',
            'kwargs': {'start_time': '2015-01-01', 'end_time': '2020-01-01'}
        },
        'task': {
            'dataset': {
                'class': 'DatasetH',
                'kwargs': {
                    'handler': {'class': 'Alpha158'},
                    'segments': {'train': [], 'valid': [], 'test': []}
                }
            },
            'model': {
                'class': 'LGBModel',
                'kwargs': {'d_feat': 158}
            },
            'record': [
                {'class': 'SigAnaRecord', 'kwargs': {'model': '<MODEL>', 'dataset': '<DATASET>'}},
                {'class': 'PortAnaRecord', 'kwargs': {'model': '<MODEL>', 'dataset': '<DATASET>', 'config': {}}}
            ]
        },
        'port_analysis_config': {
            'backtest': {
                'start_time': '2020-01-01',
                'end_time': '2020-06-01',
                'account': 10000000,
                'benchmark': 'SH000300'
            }
        }
    }
    with open(yaml_file, 'w') as f:
        yaml.dump(dummy_yaml, f)
        
    params = {
        'market': 'csi300',
        'benchmark': 'SH000300',
        'test_start_time': '2020-01-01',
        'test_end_time': '2020-06-01',
        'anchor_date': '2020-06-01',
        'cpcv_folds': [
            {
                'fold_idx': 0,
                'train_segments': [['2015-01-01', '2018-12-31']],
                'valid_start_time': '2019-01-01',
                'valid_end_time': '2019-12-31',
            }
        ]
    }
    
    # Mock Qlib dynamic instantiation
    mock_model = MagicMock()
    # Mock predict output to be pandas Series or DataFrame
    mock_pred = pd.Series([0.1, 0.2, 0.3], index=pd.MultiIndex.from_tuples(
        [('2020-01-02', '600000.SH'), ('2020-01-02', '600001.SH'), ('2020-01-02', '600002.SH')],
        names=['datetime', 'instrument']
    ))
    mock_model.predict.return_value = mock_pred
    
    mock_dataset = MagicMock()
    mock_dataset.segments = {
        'train': [['2015-01-01', '2018-12-31']],
        'valid': ['2019-01-01', '2019-12-31'],
        'test': ['2020-01-01', '2020-06-01'],
    }
    
    # Mock dataset.prepare() to return a series for validation IC calculation
    mock_val_label = pd.Series([0.05, 0.15, -0.02], index=mock_pred.index)
    mock_dataset.prepare.return_value = mock_val_label

    def mock_init_instance(config, recorder=None):
        if 'LGBModel' in str(config):
            return mock_model
        elif 'Record' in str(config.get('class', '')):
            # Mock the record object
            rec_obj = MagicMock()
            return rec_obj
        else:
            return mock_dataset
            
    monkeypatch.setattr('qlib.utils.init_instance_by_config', mock_init_instance)
    
    # Mock MLflow R
    mock_recorder = MagicMock()
    mock_recorder.id = "mock_cpcv_run_123"
    
    # Mock loader logic
    stored_objects = {}
    def mock_save_objects(**kwargs):
        stored_objects.update(kwargs)
    mock_recorder.save_objects.side_effect = mock_save_objects
    mock_recorder.list_objects.return_value = ["model_fold_0.pkl"]
    mock_recorder.load_object.side_effect = lambda key: stored_objects[key]
    
    mock_exp = MagicMock()
    mock_exp.get_recorder.return_value = mock_recorder
    
    class MockR:
        def start(self, experiment_name=None):
            class MockContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return MockContext()
        def set_tags(self, **kwargs):
            pass
        def log_params(self, **kwargs):
            pass
        def get_recorder(self):
            return mock_recorder
        def get_exp(self, experiment_name=None):
            return mock_exp
            
    monkeypatch.setattr('qlib.workflow.R', MockR())
    
    # 2. Run train_cpcv_model
    res_train = train_utils.train_cpcv_model(
        model_name="lgb",
        params=params,
        yaml_file=yaml_file,
        experiment_name="test_cpcv",
        no_pretrain=True,
    )
    
    assert res_train['success'] is True
    assert res_train['record_id'] == "mock_cpcv_run_123"
    assert res_train['n_folds'] == 1
    assert "model_fold_0.pkl" in stored_objects
    assert "pred.pkl" in stored_objects
    
    # 3. Run predict_cpcv_model
    model_info = {
        'record_id': 'mock_cpcv_run_123',
        'yaml_file': yaml_file,
    }
    res_pred = train_utils.predict_cpcv_model(
        model_name="lgb",
        model_info=model_info,
        params=params,
        experiment_name="test_cpcv",
        no_pretrain=True,
    )
    
    assert res_pred['success'] is True
    assert res_pred['record_id'] == "mock_cpcv_run_123"


def test_train_cpcv_model_errors(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    # 1. YAML file not found
    res = train_utils.train_cpcv_model(
        model_name="lgb",
        params={},
        yaml_file="nonexistent.yaml",
        experiment_name="test_cpcv",
    )
    assert res['success'] is False
    assert "YAML config not found" in res['error']
    
    # 2. Empty cpcv_folds
    yaml_file = os.path.join(train_utils.ROOT_DIR, "empty.yaml")
    with open(yaml_file, 'w') as f:
        yaml.dump({}, f)
        
    res = train_utils.train_cpcv_model(
        model_name="lgb",
        params={'cpcv_folds': []},
        yaml_file=yaml_file,
        experiment_name="test_cpcv",
    )
    assert res['success'] is False
    assert "No CPCV folds" in res['error']


def test_predict_cpcv_model_errors(mock_env_constants, monkeypatch):
    train_utils, workspace = mock_env_constants
    
    # 1. Missing record_id
    res = train_utils.predict_cpcv_model(
        model_name="lgb",
        model_info={},
        params={},
        experiment_name="test_cpcv",
    )
    assert res['success'] is False
    assert "No record_id" in res['error']
    
    # Mock R for successful experiment manager/recorder fetch, but missing YAML or other errors
    mock_recorder = MagicMock()
    mock_recorder.list_objects.return_value = ["model_fold_0.pkl"]
    
    mock_exp = MagicMock()
    mock_exp.get_recorder.return_value = mock_recorder
    class MockR:
        def get_exp(self, experiment_name=None):
            return mock_exp
    monkeypatch.setattr('qlib.workflow.R', MockR())
    
    # 2. Missing yaml_file
    res = train_utils.predict_cpcv_model(
        model_name="lgb",
        model_info={'record_id': 'rid_123'},
        params={},
        experiment_name="test_cpcv",
    )
    assert res['success'] is False
    assert "YAML file not found" in res['error']
    
    # 3. Source recorder loading fails
    class MockRFail:
        def get_exp(self, experiment_name=None):
            raise Exception("MLflow load failed")
    monkeypatch.setattr('qlib.workflow.R', MockRFail())
    
    res = train_utils.predict_cpcv_model(
        model_name="lgb",
        model_info={'record_id': 'rid_123', 'yaml_file': 'dummy.yaml'},
        params={},
        experiment_name="test_cpcv",
    )
    assert res['success'] is False
    assert "Failed to load source recorder" in res['error']


def test_train_cpcv_model_labels_float_indices(mock_env_constants, monkeypatch):
    train_utils, workspace = mock_env_constants
    
    yaml_file = os.path.join(workspace, "cpcv_task.yaml")
    dummy_yaml = {
        'market': 'csi300',
        'benchmark': 'SH000300',
        'data_handler_config': {
            'class': 'Alpha158',
            'kwargs': {'start_time': '2015-01-01', 'end_time': '2020-01-01'}
        },
        'task': {
            'dataset': {
                'class': 'DatasetH',
                'kwargs': {
                    'handler': {'class': 'Alpha158'},
                    'segments': {'train': [], 'valid': [], 'test': []}
                }
            },
            'model': {
                'class': 'LGBModel',
                'kwargs': {'d_feat': 158}
            },
            'record': []
        }
    }
    with open(yaml_file, 'w') as f:
        yaml.dump(dummy_yaml, f)
        
    params = {
        'market': 'csi300',
        'benchmark': 'SH000300',
        'test_start_time': '2020-01-01',
        'test_end_time': '2020-06-01',
        'anchor_date': '2020-06-01',
        'cpcv_folds': [
            {
                'fold_idx': 0,
                'train_segments': [['2015-01-01', '2018-12-31']],
                'valid_start_time': '2019-01-01',
                'valid_end_time': '2019-12-31',
            }
        ]
    }
    
    mock_model = MagicMock()
    mock_pred = pd.Series([0.1, 0.2], index=pd.MultiIndex.from_tuples(
        [('2020-01-02', '600000.SH'), ('2020-01-02', '600001.SH')],
        names=['datetime', 'instrument']
    ))
    mock_model.predict.return_value = mock_pred
    
    class MockTSDataSampler:
        def __init__(self):
            self.idx_map = np.array([[0, 0], [1, 0]], dtype=float)
            self.idx_arr = np.array([[1.0], [0.0]], dtype=float) # float values
            self.data_arr = np.array([[100.0], [200.0], [300.0]], dtype=float)
            self.nan_idx = 2
            
        def get_index(self):
            return mock_pred.index
            
        def __len__(self):
            return len(self.idx_map)
            
    mock_dataset = MagicMock()
    mock_dataset.segments = {
        'train': [['2015-01-01', '2018-12-31']],
        'valid': ['2019-01-01', '2019-12-31'],
        'test': ['2020-01-01', '2020-06-01'],
    }
    mock_dataset.prepare.return_value = MockTSDataSampler()

    def mock_init_instance(config, recorder=None):
        if 'LGBModel' in str(config):
            return mock_model
        return mock_dataset
            
    monkeypatch.setattr('qlib.utils.init_instance_by_config', mock_init_instance)
    
    # Mock MLflow R
    mock_recorder = MagicMock()
    mock_recorder.id = "mock_cpcv_run_123"
    mock_recorder.list_objects.return_value = []
    
    class MockR:
        def start(self, experiment_name=None):
            class MockContext:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return MockContext()
        def set_tags(self, **kwargs): pass
        def log_params(self, **kwargs): pass
        def get_recorder(self): return mock_recorder
        
    monkeypatch.setattr('qlib.workflow.R', MockR())
    
    res = train_utils.train_cpcv_model(
        model_name="lgb",
        params=params,
        yaml_file=yaml_file,
        experiment_name="test_cpcv_float_indexing",
    )
    assert res['success'] is True
