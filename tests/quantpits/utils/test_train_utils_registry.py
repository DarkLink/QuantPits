import os
import yaml
import json
from unittest.mock import patch, mock_open

def test_load_model_registry_default(mock_env_constants):
    train_utils, _ = mock_env_constants
    with patch('quantpits.utils.train_utils.REGISTRY_FILE', "dummy.yaml"):
        with patch('builtins.open', mock_open(read_data=yaml.dump({'models': {}}))):
            registry = train_utils.load_model_registry()
            assert isinstance(registry, dict)

def test_load_model_registry(mock_env_constants):
    train_utils, _ = mock_env_constants
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

def test_get_models_by_names(mock_env_constants, capsys):
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
        # registry is None
        res = train_utils.get_models_by_filter(algorithm='lstm')
        assert 'model1' in res and 'model2' not in res
        
        # specific registry
        res = train_utils.get_models_by_filter(registry=mock_registry, dataset='Alpha360')
        assert 'model2' in res and 'model1' not in res
        
        # market filter
        res = train_utils.get_models_by_filter(registry=mock_registry, market='csi500')
        assert 'model2' in res
        
        # tag filter
        res = train_utils.get_models_by_filter(registry=mock_registry, tag='tree')
        assert 'model1' in res
        # Unmatched tag
        res = train_utils.get_models_by_filter(registry=mock_registry, tag='missing')
        assert len(res) == 0

def test_resolve_model_key_exact_match(mock_env_constants):
    train_utils, _ = mock_env_constants
    models_dict = {
        "m1@static": "rid1",
        "m2@rolling": "rid2",
        "bare": "rid3"
    }
    assert train_utils.resolve_model_key("m1@static", models_dict) == "m1@static"
    assert train_utils.resolve_model_key("bare", models_dict) == "bare"
    assert train_utils.resolve_model_key("missing", models_dict) is None
    # Branch: already has @ but not found
    assert train_utils.resolve_model_key("m1@missing", models_dict) is None

def test_resolve_model_keys(mock_env_constants):
    train_utils, _ = mock_env_constants
    models_dict = {"m1@static": "rid1", "m2@static": "rid2"}
    res = train_utils.resolve_model_keys(["m1", "m2", "m3"], models_dict)
    assert res == [("m1", "m1@static"), ("m2", "m2@static"), ("m3", None)]

def test_filter_models_by_mode(mock_env_constants):
    train_utils, _ = mock_env_constants
    models_dict = {"m1@static": "r1", "m2@rolling": "r2"}
    assert train_utils.filter_models_by_mode(models_dict, None) == models_dict
    assert train_utils.filter_models_by_mode(models_dict, "static") == {"m1@static": "r1"}

def test_strip_mode_from_keys(mock_env_constants, capsys):
    train_utils, _ = mock_env_constants
    models_dict = {"m1@static": "r1", "m2@rolling": "r2", "m1@rolling": "r3"}
    stripped = train_utils.strip_mode_from_keys(models_dict)
    assert "m1" in stripped and "m2" in stripped
    assert stripped["m1"] == "r3" # Last one wins
    captured = capsys.readouterr()
    assert "去模式后 key 冲突" in captured.out

def test_resolve_model_key_with_default_mode(mock_env_constants):
    train_utils, _ = mock_env_constants
    models_dict = {"m1@static": "rid1", "m1@rolling": "rid2"}
    assert train_utils.resolve_model_key("m1", models_dict, default_mode="static") == "m1@static"
    assert train_utils.resolve_model_key("m1", models_dict, default_mode="rolling") == "m1@rolling"
    # Branch: default mode not found
    assert train_utils.resolve_model_key("m1", {}, default_mode="static") is None

def test_resolve_model_key_conflict(mock_env_constants, capsys):
    train_utils, _ = mock_env_constants
    models_dict = {"m@static": "rid1", "m@rolling": "rid2"}
    res = train_utils.resolve_model_key("m", models_dict)
    assert res == "m@static"
    captured = capsys.readouterr()
    assert "多个训练模式中存在" in captured.out

def test_resolve_target_models(mock_env_constants):
    train_utils, _ = mock_env_constants
    class Args: pass
    mock_registry = {'model1': {'enabled': True, 'algorithm': 'lstm'}, 'model2': {'enabled': False, 'algorithm': 'lgb'}}
    
    # args.models
    args1 = Args(); args1.models = "model1, model2"
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res1 = train_utils.resolve_target_models(args1)
        assert 'model1' in res1 and 'model2' in res1

    # args.all_enabled
    args2 = Args(); args2.all_enabled = True
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res2 = train_utils.resolve_target_models(args2)
        assert 'model1' in res2 and 'model2' not in res2

    # filters
    args3 = Args(); args3.algorithm = "lgb"
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res3 = train_utils.resolve_target_models(args3)
        assert 'model2' in res3

    # skip
    args4 = Args(); args4.models = "model1, model2"; args4.skip = "model2"
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        res4 = train_utils.resolve_target_models(args4)
        assert 'model1' in res4 and 'model2' not in res4

    # no targets
    args5 = Args()
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        assert train_utils.resolve_target_models(args5) is None

def test_print_model_table(mock_env_constants, capsys):
    train_utils, _ = mock_env_constants
    mock_registry = {'m1': {'enabled': True, 'algorithm': 'lstm', 'dataset': 'd1', 'market': 'c300', 'tags': ['t1']}}
    train_utils.print_model_table(mock_registry, "Test Title")
    captured = capsys.readouterr()
    assert "Test Title" in captured.out
    assert "m1" in captured.out

def test_show_model_list(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    class Args: pass
    mock_registry = {'model1': {'enabled': True, 'dataset': 'Alpha158'}, 'model2': {'enabled': False, 'dataset': 'Alpha360'}}
    args1 = Args(); args1.algorithm = "lstm"
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        with patch('quantpits.utils.train_utils.get_models_by_filter', return_value={'model1': mock_registry['model1']}):
            train_utils.show_model_list(args1)
    
    args2 = Args()
    source_records_file = tmp_path / "records.json"
    source_records_file.write_text(json.dumps({"models": {"model1": "id1"}}))
    with patch('quantpits.utils.train_utils.load_model_registry', return_value=mock_registry):
        train_utils.show_model_list(args2, source_records_file=str(source_records_file))
