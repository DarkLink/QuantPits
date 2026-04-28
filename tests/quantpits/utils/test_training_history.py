import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

# We will import train_single_model inside the test functions or use a fixture

@pytest.fixture
def mock_qlib():
    with patch('qlib.utils.init_instance_by_config') as mock_init, \
         patch('qlib.workflow.R') as mock_r:
        
        # Mock Recorder
        mock_recorder = MagicMock()
        mock_recorder.info = {'id': 'test_record_123'}
        mock_r.get_recorder.return_value = mock_recorder
        mock_r.start.return_value.__enter__.return_value = mock_r
        
        yield mock_init, mock_r, mock_recorder

@pytest.fixture
def mock_params():
    return {
        'anchor_date': '2026-04-25',
        'market': 'csi300',
        'benchmark': 'SH000300',
        'start_time': '2020-01-01',
        'end_time': '2026-04-25',
        'fit_start_time': '2020-01-01',
        'fit_end_time': '2025-12-31',
        'valid_start_time': '2026-01-01',
        'valid_end_time': '2026-03-31',
        'test_start_time': '2026-04-01',
        'test_end_time': '2026-04-25'
    }

def test_convergence_log_nn_early_stop(mock_env_constants, mock_qlib, mock_params, tmp_path):
    train_utils, _ = mock_env_constants
    from quantpits.utils.train_utils import train_single_model
    mock_init, mock_r, mock_recorder = mock_qlib
    
    # Mock Model with NN attributes
    mock_model = MagicMock()
    # Explicitly delete attributes that might be detected as Tree models
    if hasattr(mock_model, 'fitted_model_'):
        del mock_model.fitted_model_
    
    mock_model.model.n_epochs_fitted_ = 45
    mock_init.side_effect = [mock_model, MagicMock()] # model, dataset
    
    yaml_content = {
        'task': {
            'model': {
                'kwargs': {'n_epochs': 200}
            },
            'dataset': {
                'kwargs': {'segments': {}}
            }
        },
        'data_handler_config': {}
    }
    
    yaml_file = tmp_path / "test.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
    
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True):
        
        result = train_single_model("test_model", str(yaml_file), mock_params, "test_exp")
        
        if not result['success']:
            print(f"ERROR: {result['error']}")
        assert result['success'] is True
        conv = result['performance']['convergence']
        assert conv['actual_epochs'] == 45
        assert conv['configured_epochs'] == 200
        assert conv['early_stopped'] is True
        assert conv['converged'] is False

def test_convergence_log_nn_full(mock_env_constants, mock_qlib, mock_params, tmp_path):
    train_utils, _ = mock_env_constants
    from quantpits.utils.train_utils import train_single_model
    mock_init, mock_r, mock_recorder = mock_qlib
    
    # Mock Model with NN attributes
    mock_model = MagicMock()
    if hasattr(mock_model, 'fitted_model_'):
        del mock_model.fitted_model_
    
    mock_model.model.n_epochs_fitted_ = 200
    mock_init.side_effect = [mock_model, MagicMock()]
    
    yaml_content = {
        'task': {
            'model': {
                'kwargs': {'n_epochs': 200}
            },
            'dataset': {
                'kwargs': {'segments': {}}
            }
        },
        'data_handler_config': {}
    }
    
    yaml_file = tmp_path / "test.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
    
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True):
        
        result = train_single_model("test_model", str(yaml_file), mock_params, "test_exp")
        
        conv = result['performance']['convergence']
        assert conv['actual_epochs'] == 200
        assert conv['early_stopped'] is False
        assert conv['converged'] is True

def test_convergence_log_tree_model(mock_env_constants, mock_qlib, mock_params, tmp_path):
    train_utils, _ = mock_env_constants
    from quantpits.utils.train_utils import train_single_model
    mock_init, mock_r, mock_recorder = mock_qlib
    
    # Mock Model with Tree attributes
    mock_model = MagicMock()
    del mock_model.model # No .model attribute like NN
    mock_model.fitted_model_.best_iteration = 150
    mock_init.side_effect = [mock_model, MagicMock()]
    
    yaml_content = {
        'task': {
            'model': {
                'kwargs': {} # No n_epochs
            },
            'dataset': {
                'kwargs': {'segments': {}}
            }
        },
        'data_handler_config': {}
    }
    
    yaml_file = tmp_path / "test.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
    
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True):
        
        result = train_single_model("test_model", str(yaml_file), mock_params, "test_exp")
        
        conv = result['performance']['convergence']
        assert conv['actual_epochs'] == 150
        assert conv['configured_epochs'] is None
        assert conv['early_stopped'] is False
        assert conv['converged'] is None

def test_convergence_log_appended_to_jsonl(mock_env_constants, mock_qlib, mock_params, tmp_path):
    train_utils, _ = mock_env_constants
    from quantpits.utils.train_utils import train_single_model
    mock_init, mock_r, mock_recorder = mock_qlib
    
    mock_model = MagicMock()
    if hasattr(mock_model, 'fitted_model_'):
        del mock_model.fitted_model_
    if hasattr(mock_model, 'model'):
        del mock_model.model
        
    mock_init.side_effect = [mock_model, MagicMock()] * 2
    
    yaml_content = {
        'task': {
            'model': {'kwargs': {}}, 
            'dataset': {'kwargs': {'segments': {}}}
        }, 
        'data_handler_config': {}
    }
    yaml_file = tmp_path / "test.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
    
    history_file = tmp_path / "data" / "training_history.jsonl"
    
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True):
        
        train_single_model("model1", str(yaml_file), mock_params, "test_exp")
        train_single_model("model2", str(yaml_file), mock_params, "test_exp")
        
        assert history_file.exists()
        lines = history_file.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])['model_name'] == "model1"
        assert json.loads(lines[1])['model_name'] == "model2"

def test_convergence_log_failure_does_not_affect_training(mock_env_constants, mock_qlib, mock_params, tmp_path):
    train_utils, _ = mock_env_constants
    from quantpits.utils.train_utils import train_single_model
    mock_init, mock_r, mock_recorder = mock_qlib
    
    mock_model = MagicMock()
    if hasattr(mock_model, 'fitted_model_'):
        del mock_model.fitted_model_
    if hasattr(mock_model, 'model'):
        del mock_model.model

    mock_init.side_effect = [mock_model, MagicMock()]
    
    yaml_content = {
        'task': {
            'model': {'kwargs': {}}, 
            'dataset': {'kwargs': {'segments': {}}}
        }, 
        'data_handler_config': {}
    }
    yaml_file = tmp_path / "test.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
        
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=IOError("Disk Full")) as mock_file:
        
        # Note: we need to be careful with builtins.open mock because it might break yaml.dump or other things
        # But here train_single_model calls open() for the jsonl file.
        # Actually, let's mock os.makedirs to fail or something more specific.
        pass

    # Alternative way to test failure silence:
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True), \
         patch('quantpits.utils.train_utils.os.makedirs', side_effect=Exception("Perm Error")):
        
        result = train_single_model("test_model", str(yaml_file), mock_params, "test_exp")
        assert result['success'] is True
        assert 'convergence' in result['performance']


def test_convergence_log_log_capture(mock_env_constants, mock_qlib, mock_params, tmp_path):
    train_utils, _ = mock_env_constants
    from quantpits.utils.train_utils import train_single_model
    mock_init, mock_r, mock_recorder = mock_qlib
    
    import logging
    
    # Mock Model that logs best score during fit
    class LoggingModel:
        def fit(self, **kwargs):
            logger = logging.getLogger("qlib.ADD")
            logger.info("bootstrap_fit best score: 0.063768 @ 7")
        def predict(self, **kwargs):
            return MagicMock()
            
    mock_model = LoggingModel()
    mock_init.side_effect = [mock_model, MagicMock()] # model, dataset
    
    yaml_content = {
        'task': {
            'model': {'kwargs': {'n_epochs': 200}},
            'dataset': {'kwargs': {'segments': {}}}
        },
        'data_handler_config': {}
    }
    yaml_file = tmp_path / "test.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
        
    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True):
        
        result = train_single_model("test_model", str(yaml_file), mock_params, "test_exp")
        
        assert result['success'] is True
        conv = result['performance']['convergence']
        assert conv['best_score'] == 0.063768
        assert conv['best_epoch'] == 7


def test_convergence_log_add_model_capture(mock_env_constants, mock_qlib, mock_params, tmp_path):
    """ADD model with custom training loop: logs epochs/loss/early-stop to logger, not evals_result."""
    train_utils, _ = mock_env_constants
    from quantpits.utils.train_utils import train_single_model
    mock_init, mock_r, mock_recorder = mock_qlib

    import logging

    class ADDModel:
        def fit(self, **kwargs):
            logger = logging.getLogger("qlib.ADD")
            for step in range(17):
                logger.info("Epoch%d:", step)
                logger.info("training...")
                logger.info("evaluating...")
                logger.info(
                    "pre_excess_loss/train: 0.97, loss/train: -0.97, "
                    "mse/train: -0.97, ic/train: 0.18"
                )
            logger.info("Epoch17:")
            logger.info("training...")
            logger.info("evaluating...")
            logger.info(
                "pre_excess_loss/train: 0.97, loss/train: -0.970555, "
                "mse/train: -0.970555, ic/train: 0.18"
            )
            logger.info("early stop")
            logger.info("bootstrap_fit best score: 0.063768 @ 7")

        def predict(self, **kwargs):
            return MagicMock()

    mock_model = ADDModel()
    mock_init.side_effect = [mock_model, MagicMock()]

    yaml_content = {
        'task': {
            'model': {'kwargs': {'n_epochs': 200}},
            'dataset': {'kwargs': {'segments': {}}}
        },
        'data_handler_config': {}
    }
    yaml_file = tmp_path / "test.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)

    with patch('quantpits.utils.train_utils.ROOT_DIR', str(tmp_path)), \
         patch('quantpits.utils.train_utils.os.path.exists', return_value=True):

        result = train_single_model("add_Alpha360", str(yaml_file), mock_params, "test_exp")

        assert result['success'] is True
        conv = result['performance']['convergence']
        assert conv['actual_epochs'] == 18, f"Expected 18 epochs, got {conv['actual_epochs']}"
        assert conv['configured_epochs'] == 200
        assert conv['early_stopped'] is True
        assert conv['converged'] is False
        assert conv['best_score'] == 0.063768
        assert conv['best_epoch'] == 7
        assert conv['final_train_loss'] == -0.970555
