import pytest
import os
import json
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "output").mkdir()
    
    import sys
    import importlib
    script_dir = os.path.join(os.getcwd(), "quantpits/scripts")
    if script_dir not in sys.path:
        sys.path.append(script_dir)
        
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    # Reload all possible names to ensure they pick up the new QLIB_WORKSPACE_DIR
    for mod_name in ['env', 'quantpits.utils.env', 'prod_predict_only', 'quantpits.scripts.prod_predict_only']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
            
    from quantpits.scripts import prod_predict_only as ppo
    yield ppo

@patch('quantpits.utils.train_utils.get_models_by_names')
@patch('quantpits.utils.train_utils.get_enabled_models')
@patch('quantpits.utils.train_utils.get_models_by_filter')
@patch('quantpits.utils.train_utils.load_model_registry')
def test_resolve_target_models_by_names(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    ppo = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    mock_names.return_value = {"m1": {}}
    
    args = MagicMock()
    args.models = "m1"
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = ppo.resolve_target_models(args)
    assert targets == {"m1": {}}

@patch('quantpits.utils.train_utils.get_models_by_names')
@patch('quantpits.utils.train_utils.get_enabled_models')
@patch('quantpits.utils.train_utils.get_models_by_filter')
@patch('quantpits.utils.train_utils.load_model_registry')
def test_resolve_target_models_by_filter(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    ppo = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    mock_filter.return_value = {"m1": {}}
    
    args = MagicMock()
    args.models = None
    args.all_enabled = False
    args.algorithm = "lstm"
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = ppo.resolve_target_models(args)
    assert targets == {"m1": {}}

@patch('quantpits.utils.train_utils.load_model_registry')
def test_resolve_target_models_none(mock_load, mock_env):
    ppo = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    args = MagicMock()
    args.models = None
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = ppo.resolve_target_models(args)
    assert targets is None

def test_parse_args(mock_env):
    ppo = mock_env
    import sys
    with patch.object(sys, 'argv', ['script.py', '--models', 'gru,mlp', '--dry-run']):
        args = ppo.parse_args()
    assert args.models == 'gru,mlp'
    assert args.dry_run is True

    with patch.object(sys, 'argv', ['script.py', '--all-enabled', '--skip', 'catboost']):
        args = ppo.parse_args()
    assert args.all_enabled is True
    assert args.skip == 'catboost'

    with patch.object(sys, 'argv', ['script.py', '--list']):
        args = ppo.parse_args()
    assert args.list is True

@patch('quantpits.utils.train_utils.load_model_registry')
@patch('quantpits.utils.train_utils.get_enabled_models')
@patch('quantpits.utils.train_utils.get_models_by_names')
@patch('quantpits.utils.train_utils.get_models_by_filter')
def test_resolve_with_skip(mock_filter, mock_names, mock_enabled, mock_load, mock_env):
    ppo = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}, "m3": {}}
    mock_enabled.return_value = {"m1": {}, "m2": {}, "m3": {}}

    args = MagicMock()
    args.models = None
    args.all_enabled = True
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = "m2,m3"

    targets = ppo.resolve_target_models(args)
    assert "m1" in targets
    assert "m2" not in targets
    assert "m3" not in targets

@patch('quantpits.utils.train_utils.inject_config')
@patch('qlib.utils.init_instance_by_config', create=True)
@patch('qlib.workflow.R', create=True)
def test_predict_single_model_yaml_missing(mock_R, mock_init, mock_inject, mock_env, tmp_path):
    ppo = mock_env
    model_info = {'yaml_file': str(tmp_path / 'nonexistent.yaml')}
    result = ppo.predict_single_model(
        'test_model', model_info, {}, 'exp', {'models': {'test_model': 'rid1'}}
    )
    assert result['success'] is False
    assert 'YAML' in result['error']

@patch('quantpits.utils.train_utils.inject_config')
@patch('qlib.utils.init_instance_by_config', create=True)
@patch('qlib.workflow.R', create=True)
def test_predict_single_model_not_in_source(mock_R, mock_init, mock_inject, mock_env, tmp_path):
    ppo = mock_env
    yaml_file = tmp_path / 'model.yaml'
    yaml_file.write_text('test: true')
    model_info = {'yaml_file': str(yaml_file)}
    result = ppo.predict_single_model(
        'test_model', model_info, {}, 'exp', {'models': {}}
    )
    assert result['success'] is False
    assert '不在源训练记录中' in result['error']

@patch('quantpits.utils.train_utils.inject_config')
@patch('qlib.utils.init_instance_by_config', create=True)
@patch('qlib.workflow.R', create=True)
def test_predict_single_model_success(mock_R, mock_init, mock_inject, mock_env, tmp_path):
    import pandas as pd
    ppo = mock_env
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

    with patch('quantpits.utils.train_utils.PREDICTION_OUTPUT_DIR', str(tmp_path)):
        result = ppo.predict_single_model(
            'test_model', model_info, params, 'exp', source_records
        )

    assert result['success'] is True
    assert result['record_id'] == 'new_rid'

@patch('quantpits.utils.train_utils.load_model_registry')
@patch('quantpits.utils.train_utils.get_models_by_filter')
@patch('quantpits.utils.train_utils.print_model_table')
def test_show_list(mock_print_table, mock_filter, mock_load, mock_env, tmp_path):
    ppo = mock_env
    mock_load.return_value = {
        "m1": {"enabled": True}, "m2": {"enabled": False}
    }

    args = MagicMock()
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.source_records = str(tmp_path / 'nonexistent.json')

    ppo.show_list(args)
    mock_print_table.assert_called_once()

@patch('quantpits.utils.train_utils.load_model_registry')
@patch('quantpits.utils.train_utils.get_models_by_filter')
@patch('quantpits.utils.train_utils.print_model_table')
def test_show_list_with_source(mock_print_table, mock_filter, mock_load, mock_env, tmp_path):
    import json
    ppo = mock_env
    mock_load.return_value = {
        "m1": {"enabled": True}, "m2": {"enabled": False}
    }

    source_file = tmp_path / 'records.json'
    with open(source_file, 'w') as f:
        json.dump({"models": {"m1": "rid1"}}, f)

    args = MagicMock()
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.source_records = str(source_file)

    ppo.show_list(args)
    mock_print_table.assert_called_once()

@patch('quantpits.utils.train_utils.load_model_registry')
@patch('quantpits.utils.train_utils.get_enabled_models')
@patch('quantpits.utils.train_utils.get_models_by_names')
@patch('quantpits.utils.train_utils.get_models_by_filter')
@patch('quantpits.utils.train_utils.calculate_dates')
@patch('quantpits.utils.train_utils.merge_train_records')
@patch('quantpits.utils.train_utils.merge_performance_file')
@patch('quantpits.utils.train_utils.print_model_table')
def test_run_predict_only_dry_run(mock_table, mock_merge_perf, mock_merge_rec,
                                  mock_dates, mock_filter, mock_names,
                                  mock_enabled, mock_load, mock_env, tmp_path):
    import json
    ppo = mock_env

    source_file = tmp_path / 'records.json'
    with open(source_file, 'w') as f:
        json.dump({"models": {"m1": "rid1"}, "experiment_name": "train"}, f)

    mock_load.return_value = {"m1": {}}
    mock_names.return_value = {"m1": {}}

    args = MagicMock()
    args.models = "m1"
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    args.source_records = str(source_file)
    args.dry_run = True

    ppo.run_predict_only(args)
    mock_dates.assert_not_called()

@patch('quantpits.scripts.prod_predict_only.init_qlib', create=True)
@patch('quantpits.utils.train_utils.calculate_dates')
@patch('quantpits.utils.train_utils.merge_train_records')
@patch('quantpits.utils.train_utils.merge_performance_file')
@patch('quantpits.scripts.prod_predict_only.predict_single_model')
@patch('quantpits.scripts.prod_predict_only.resolve_target_models')
@patch('quantpits.utils.train_utils.print_model_table')
def test_run_predict_only_success(mock_table, mock_resolve, mock_predict, 
                                  mock_merge_perf, mock_merge_rec,
                                  mock_dates, mock_init, mock_env, tmp_path):
    import json
    ppo = mock_env
    source_file = tmp_path / 'records.json'
    with open(source_file, 'w') as f:
        json.dump({"models": {"m1": "rid1"}, "experiment_name": "train", "anchor_date": "2020-01-01"}, f)

    mock_resolve.return_value = {"m1": {}}
    mock_dates.return_value = {"anchor_date": "2020-01-08", "freq": "week"}
    mock_predict.return_value = {"success": True, "record_id": "new_rid", "performance": {"IC_Mean": 0.05}}

    args = MagicMock()
    args.models = "m1"
    args.dry_run = False
    args.source_records = str(source_file)
    args.experiment_name = "Prod_Predict"

    ppo.run_predict_only(args)
    mock_predict.assert_called_once()
    mock_merge_rec.assert_called_once()
    mock_merge_perf.assert_called_once()

@patch('quantpits.scripts.prod_predict_only.init_qlib', create=True)
@patch('quantpits.utils.train_utils.calculate_dates')
@patch('quantpits.scripts.prod_predict_only.predict_single_model')
@patch('quantpits.scripts.prod_predict_only.resolve_target_models')
def test_run_predict_only_failed_model(mock_resolve, mock_predict, 
                                       mock_dates, mock_init, mock_env, tmp_path):
    import json
    ppo = mock_env
    source_file = tmp_path / 'records.json'
    with open(source_file, 'w') as f:
        json.dump({"models": {"m1": "rid1"}}, f)

    mock_resolve.return_value = {"m1": {}}
    mock_dates.return_value = {"anchor_date": "2020-01-08"}
    mock_predict.return_value = {"success": False, "error": "Prediction timeout"}

    args = MagicMock()
    args.models = "m1"
    args.dry_run = False
    args.source_records = str(source_file)
    args.experiment_name = "Prod_Predict"

    ppo.run_predict_only(args)
    mock_predict.assert_called_once()

def test_main_no_selection(mock_env):
    ppo = mock_env
    import sys
    with patch.object(sys, 'argv', ['script.py']):
        ppo.main()

def test_main_list(mock_env):
    ppo = mock_env
    import sys
    with patch.object(sys, 'argv', ['script.py', '--list']):
        with patch.object(ppo, 'show_list') as mock_show:
            ppo.main()
            mock_show.assert_called_once()
