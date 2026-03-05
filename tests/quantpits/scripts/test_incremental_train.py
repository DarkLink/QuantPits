import pytest
import os
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
            "m2": {"algorithm": "mlp", "dataset": "Alpha158", "enabled": False, "yaml_file": "mlp.yaml"}
        }
    }))
    
    import sys
    import importlib
    script_dir = os.path.join(os.getcwd(), "quantpits/scripts")
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    # Reload all possible names to ensure they pick up the new QLIB_WORKSPACE_DIR
    for mod_name in ['env', 'quantpits.scripts.env', 'train_utils', 'quantpits.scripts.train_utils', 'incremental_train', 'quantpits.scripts.incremental_train']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
            
    from quantpits.scripts import incremental_train as it
    yield it, workspace

def test_resolve_target_models_names(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.models = "m1,m2"
    args.skip = None
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    
    targets = it.resolve_target_models(args)
    assert "m1" in targets
    assert "m2" in targets

def test_resolve_target_models_all_enabled(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.models = None
    args.all_enabled = True
    args.skip = "m2"
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    
    targets = it.resolve_target_models(args)
    assert "m1" in targets
    assert "m2" not in targets

def test_resolve_target_models_filter(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.models = None
    args.all_enabled = False
    args.skip = None
    args.algorithm = "gru"
    args.dataset = None
    args.market = None
    args.tag = None
    
    targets = it.resolve_target_models(args)
    assert "m1" in targets
    assert len(targets) == 1

@patch('train_utils.train_single_model')
@patch('train_utils.save_run_state')
@patch('train_utils.print_model_table')
@patch('quantpits.scripts.incremental_train.resolve_target_models')
@patch('quantpits.scripts.train_utils.calculate_dates')
def test_run_incremental_train_failed(mock_dates, mock_resolve, mock_print_tbl, mock_save, mock_train, mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.models = "m1"
    args.resume = False
    args.dry_run = False
    args.experiment_name = None
    
    mock_resolve.return_value = {"m1": {"yaml_file": "gru.yaml"}}
    mock_dates.return_value = {"anchor_date": "2020-01-01", "freq": "week", "market": "csi300", "benchmark": "SH000300"}
    mock_train.return_value = {"success": False, "error": "Disk full"}
    
    it.run_incremental_train(args)
    mock_train.assert_called_once()
    # Should not call clear_run_state

def test_show_list(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    
    with patch('train_utils.print_model_table'):
        it.show_list(args)

def test_show_state(mock_env):
    it, workspace = mock_env
    with patch('train_utils.load_run_state') as mock_load:
        mock_load.return_value = {"completed": ["m1"], "target_models": ["m1", "m2"], "failed": {"m3": "error"}}
        it.show_state()

def test_main_clear_state(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.list = False
    args.show_state = False
    args.clear_state = True
    
    with patch('quantpits.scripts.incremental_train.parse_args', return_value=args):
        with patch('train_utils.clear_run_state') as mock_clear:
            it.main()
            mock_clear.assert_called_once()

def test_main_show_state(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.list = False
    args.show_state = True
    args.clear_state = False
    
    with patch('quantpits.scripts.incremental_train.parse_args', return_value=args):
        with patch('quantpits.scripts.incremental_train.show_state') as mock_show:
            it.main()
            mock_show.assert_called_once()

def test_show_list_filter(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.algorithm = "gru"
    args.dataset = None
    args.market = None
    args.tag = None
    
    with patch('train_utils.print_model_table'):
        it.show_list(args)

def test_run_incremental_train_no_targets(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.models = "nonexistent"
    args.resume = False
    
    with patch('quantpits.scripts.incremental_train.resolve_target_models', return_value={}):
        it.run_incremental_train(args) # Should print warning and return

def test_main_no_selection(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.list = False
    args.show_state = False
    args.clear_state = False
    args.models = None
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.all_enabled = False
    
    with patch('quantpits.scripts.incremental_train.parse_args', return_value=args):
        it.main() # Should print error and return

@patch('quantpits.scripts.incremental_train.init_qlib', create=True)
@patch('train_utils.calculate_dates')
@patch('train_utils.train_single_model')
@patch('train_utils.merge_train_records')
@patch('train_utils.save_run_state')
@patch('train_utils.clear_run_state')
@patch('train_utils.print_model_table')
@patch('quantpits.scripts.incremental_train.resolve_target_models')
@patch('quantpits.scripts.env.safeguard')
def test_run_incremental_train_success(mock_safeguard, mock_resolve, mock_print_tbl, mock_clear, mock_save, mock_merge, mock_train, mock_dates, mock_init, mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.models = "m1"
    args.resume = False
    args.dry_run = False
    args.experiment_name = None
    
    mock_resolve.return_value = {"m1": {"yaml_file": "gru.yaml"}}
    mock_dates.return_value = {"anchor_date": "2020-01-01", "freq": "week", "market": "csi300", "benchmark": "SH000300"}
    mock_train.return_value = {"success": True, "record_id": "rid1", "performance": {"ICIR": 0.1}}
    
    it.run_incremental_train(args)
    
    mock_train.assert_called_once()
    mock_merge.assert_called_once()
    mock_clear.assert_called_once()

@patch('quantpits.scripts.train_utils.load_run_state')
def test_run_incremental_train_resume(mock_load_state, mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.models = "m1,m2"
    args.resume = True
    args.dry_run = True # skip actual train
    
    mock_load_state.return_value = {"completed": ["m1"]}
    
    with patch('quantpits.scripts.train_utils.print_model_table'):
        it.run_incremental_train(args)
        # Should filter out m1

def test_main_list(mock_env):
    it, workspace = mock_env
    args = MagicMock()
    args.list = True
    args.show_state = False
    args.clear_state = False
    
    with patch('quantpits.scripts.incremental_train.parse_args', return_value=args):
        with patch('quantpits.scripts.incremental_train.show_list') as mock_list:
            it.main()
            mock_list.assert_called_once()
