import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open

# Setup path for imports
@pytest.fixture(scope="session", autouse=True)
def setup_path():
    SCRIPT_DIR = os.path.join(os.getcwd(), 'quantpits/scripts')
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    yield

@pytest.fixture
def mock_pretrain(monkeypatch):
    # Mock heavy modules
    mock_qlib = MagicMock()
    mock_qlib_workflow = MagicMock()
    mock_qlib_utils = MagicMock()
    mock_qlib_constant = MagicMock()
    mock_mlflow = MagicMock()
    
    # Ensure qlib is a package and has submodules
    monkeypatch.setitem(sys.modules, 'qlib', mock_qlib)
    monkeypatch.setitem(sys.modules, 'qlib.workflow', mock_qlib_workflow)
    monkeypatch.setitem(sys.modules, 'qlib.utils', mock_qlib_utils)
    monkeypatch.setitem(sys.modules, 'qlib.constant', mock_qlib_constant)
    monkeypatch.setitem(sys.modules, 'mlflow', mock_mlflow)
    
    from quantpits.utils import env
    from quantpits.scripts import pretrain
    import importlib
    importlib.reload(env)
    importlib.reload(pretrain)
    
    return pretrain, mock_qlib_utils, mock_qlib_workflow

def test_parse_args(mock_pretrain):
    pretrain, _, _ = mock_pretrain
    with patch('sys.argv', ['pretrain.py', '--models', 'lstm_Alpha158']):
        args = pretrain.parse_args()
        assert args.models == "lstm_Alpha158"

def test_show_list(mock_pretrain):
    pretrain, _, _ = mock_pretrain
    with patch('quantpits.utils.train_utils.load_model_registry') as mock_load, \
         patch('quantpits.utils.train_utils.get_models_by_filter') as mock_filter, \
         patch('quantpits.utils.train_utils.print_model_table') as mock_print:
        
        # Case 1: models exist
        mock_load.return_value = {"m1": {"pretrain_source": "s1"}}
        mock_filter.return_value = {"s1": {}}
        pretrain.show_list()
        mock_print.assert_called_once()
        
        # Case 2: no base models
        mock_filter.return_value = {}
        pretrain.show_list()
        assert mock_print.call_count == 1 # Still 1 from previous call

@patch('quantpits.utils.train_utils.PRETRAINED_DIR', '/tmp/mock_pretrain')
def test_show_pretrained(mock_pretrain):
    pretrain, _, _ = mock_pretrain
    
    # Case 1: Directory doesn't exist
    with patch('os.path.exists', return_value=False):
        pretrain.show_pretrained()
        
    # Case 2: Directory empty
    with patch('os.path.exists', return_value=True), \
         patch('os.listdir', return_value=[]):
        pretrain.show_pretrained()
        
    # Case 3: Files exist (with and without metadata)
    files = ["m1.pkl", "m1.json", "m2.pkl"]
    with patch('os.path.exists') as mock_exists, \
         patch('os.listdir', return_value=files), \
         patch('os.path.getsize', return_value=1024), \
         patch('builtins.open', mock_open(read_data='{"d_feat": 20}')):
        
        def exists_side_effect(path):
            if path == '/tmp/mock_pretrain': return True
            if path.endswith('.json'): return 'm1.json' in path
            return False
            
        mock_exists.side_effect = exists_side_effect
        pretrain.show_pretrained()

def test_pretrain_for_upper_model(mock_pretrain):
    pretrain, mock_utils, mock_workflow = mock_pretrain
    params = {'anchor_date': '2026-03-13'}
    
    # Mock R.start as a context manager
    mock_workflow.R.start.return_value.__enter__.return_value = MagicMock()

    with patch('quantpits.utils.train_utils.load_model_registry') as mock_registry, \
         patch('quantpits.utils.train_utils.inject_config') as mock_inject, \
         patch('quantpits.utils.train_utils.save_pretrained_model'), \
         patch('os.path.exists', return_value=True):
        
        # Success case
        mock_registry.return_value = {
            "upper": {"pretrain_source": "base", "yaml_file": "u.yaml"},
            "base": {"yaml_file": "b.yaml"}
        }
        mock_inject.side_effect = [{'task': {'dataset': {}}}, {'task': {'model': {}}}]
        assert pretrain.pretrain_for_upper_model("upper", params, "Exp") is True
        
        # Failure: upper model not in registry
        mock_inject.side_effect = None # Reset side_effect
        assert pretrain.pretrain_for_upper_model("none", params, "Exp") is False
        
        # Failure: no pretrain_source
        mock_registry.return_value["upper"] = {"yaml_file": "u.yaml"}
        assert pretrain.pretrain_for_upper_model("upper", params, "Exp") is False
        
        # Failure: base model not in registry
        mock_registry.return_value["upper"] = {"pretrain_source": "none", "yaml_file": "u.yaml"}
        assert pretrain.pretrain_for_upper_model("upper", params, "Exp") is False
        
        # Failure: YAML doesn't exist (upper)
        mock_registry.return_value["upper"] = {"pretrain_source": "base", "yaml_file": "u.yaml"}
        with patch('os.path.exists', side_effect=[False, True]):
            assert pretrain.pretrain_for_upper_model("upper", params, "Exp") is False
            
        # Failure: YAML doesn't exist (base)
        with patch('os.path.exists', side_effect=[True, False]):
            assert pretrain.pretrain_for_upper_model("upper", params, "Exp") is False
            
    # Exception case
    # Here we trigger the exception INSIDE the with R.start block in pretrain.py
    # Lines 220-227 are the block. fit() is line 233.
    with patch('quantpits.utils.train_utils.load_model_registry') as mock_registry, \
         patch('quantpits.utils.train_utils.inject_config') as mock_inject, \
         patch('qlib.utils.init_instance_by_config', side_effect=Exception("Crash")):
        
        mock_registry.return_value = {
            "upper": {"pretrain_source": "base", "yaml_file": "u.yaml"},
            "base": {"yaml_file": "b.yaml"}
        }
        mock_inject.side_effect = [{'task': {'dataset': {}}}, {'task': {'model': {}}}]
        with patch('os.path.exists', return_value=True):
            assert pretrain.pretrain_for_upper_model("upper", params, "Exp") is False

def test_pretrain_base_model(mock_pretrain):
    pretrain, _, mock_workflow = mock_pretrain
    params = {'anchor_date': '2026-03-13'}
    info = {'yaml_file': 'b.yaml'}
    
    # Mock R.start as a context manager
    mock_workflow.R.start.return_value.__enter__.return_value = MagicMock()

    with patch('quantpits.utils.train_utils.inject_config') as mock_inject, \
         patch('quantpits.utils.train_utils.save_pretrained_model'), \
         patch('os.path.exists', return_value=True):
        
        mock_inject.return_value = {'task': {'model': {}, 'dataset': {}}}
        assert pretrain.pretrain_base_model("base", info, params, "Exp") is True
        
        # Failure: YAML not found
        with patch('os.path.exists', return_value=False):
            assert pretrain.pretrain_base_model("base", info, params, "Exp") is False
            
    # Exception case triggering inside the block
    with patch('quantpits.utils.train_utils.inject_config') as mock_inject, \
         patch('qlib.utils.init_instance_by_config', side_effect=Exception("Crash")):
        mock_inject.return_value = {'task': {'model': {}, 'dataset': {}}}
        with patch('os.path.exists', return_value=True):
            assert pretrain.pretrain_base_model("base", info, params, "Exp") is False

def test_run_pretrain(mock_pretrain):
    pretrain, _, _ = mock_pretrain
    args = MagicMock(for_model=None, models=None, tag=None, dry_run=False, experiment_name=None)
    
    with patch('quantpits.utils.train_utils.calculate_dates', return_value={'freq': 'week', 'anchor_date': '2026-03-13'}), \
         patch('quantpits.utils.train_utils.load_model_registry', return_value={}), \
         patch('quantpits.utils.env.init_qlib'):
        
        # --for model dry run
        args.for_model = "upper"
        args.dry_run = True
        pretrain.run_pretrain(args)
        
        # --for model run
        args.dry_run = False
        with patch.object(pretrain, 'pretrain_for_upper_model', return_value=True):
            pretrain.run_pretrain(args)
            
        # --for model run failure
        with patch.object(pretrain, 'pretrain_for_upper_model', return_value=False):
            pretrain.run_pretrain(args)
            
        # --models mode
        args.for_model = None
        args.models = "m1,m2"
        with patch('quantpits.utils.train_utils.get_models_by_names', return_value={"m1": {}}), \
             patch('quantpits.utils.train_utils.print_model_table'), \
             patch.object(pretrain, 'pretrain_base_model', side_effect=[True, False]):
            pretrain.run_pretrain(args)
            
        # --tag mode
        args.models = None
        args.tag = "base"
        with patch('quantpits.utils.train_utils.get_models_by_filter', return_value={"m1": {}}), \
             patch('quantpits.utils.train_utils.print_model_table'), \
             patch.object(pretrain, 'pretrain_base_model', return_value=True):
            pretrain.run_pretrain(args)
            
        # no models/tags
        args.tag = None
        pretrain.run_pretrain(args)
        
        # match no models
        args.tag = "none"
        with patch('quantpits.utils.train_utils.get_models_by_filter', return_value={}):
            pretrain.run_pretrain(args)
            
        # dry run tag
        args.tag = "base"
        args.dry_run = True
        with patch('quantpits.utils.train_utils.get_models_by_filter', return_value={"m1": {}}), \
             patch('quantpits.utils.train_utils.print_model_table'):
            pretrain.run_pretrain(args)

def test_main(mock_pretrain):
    pretrain, _, _ = mock_pretrain
    
    with patch.object(pretrain, 'parse_args') as mock_parse, \
         patch.object(pretrain, 'show_list') as mock_list, \
         patch.object(pretrain, 'show_pretrained') as mock_show, \
         patch.object(pretrain, 'run_pretrain') as mock_run:
        
        # --list
        mock_parse.return_value = MagicMock(list=True, show_pretrained=False)
        pretrain.main()
        mock_list.assert_called_once()
        
        # --show-pretrained
        mock_parse.return_value = MagicMock(list=False, show_pretrained=True)
        pretrain.main()
        mock_show.assert_called_once()
        
        # No selection
        mock_parse.return_value = MagicMock(list=False, show_pretrained=False, models=None, tag=None, for_model=None)
        pretrain.main()
        
        # With selection calls run_pretrain
        mock_parse.return_value = MagicMock(list=False, show_pretrained=False, models="m1", tag=None, for_model=None)
        pretrain.main()
        mock_run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
