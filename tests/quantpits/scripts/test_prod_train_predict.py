import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "output").mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.chdir(workspace)
    
    from quantpits.utils import env
    from quantpits.scripts import prod_train_predict
    import importlib
    importlib.reload(env)
    importlib.reload(prod_train_predict)
    
    yield prod_train_predict

@patch('quantpits.scripts.prod_train_predict.calculate_dates')
@patch('quantpits.scripts.prod_train_predict.load_model_registry')
@patch('quantpits.scripts.prod_train_predict.get_enabled_models')
@patch('quantpits.scripts.prod_train_predict.print_model_table')
@patch('quantpits.scripts.prod_train_predict.train_single_model')
@patch('quantpits.scripts.prod_train_predict.overwrite_train_records')
@patch('quantpits.scripts.prod_train_predict.backup_file_with_date')
@patch('json.dump')
def test_run_train_predict_smoke(mock_json_dump, mock_backup, mock_overwrite, mock_train, mock_print, mock_enabled, mock_load, mock_dates, mock_env):
    ptp = mock_env
    
    # Setup mocks
    mock_dates.return_value = {"freq": "week", "anchor_date": "2026-03-01"}
    mock_load.return_value = {}
    mock_enabled.return_value = {"m1": {"yaml_file": "m1.yaml"}}
    
    # Mock train_single_model to return a success dictionary
    mock_train.return_value = {
        "success": True,
        "record_id": "rid1",
        "performance": {"IC_Mean": 0.1, "ICIR": 0.5}
    }
    
    # Run
    ptp.run_train_predict()
    
    # Assertions
    mock_enabled.assert_called_once()
    mock_train.assert_called_once_with("m1", "m1.yaml", mock_dates.return_value, "Prod_Train_WEEK")
    mock_overwrite.assert_called_once()
    mock_backup.assert_called_once()
    mock_json_dump.assert_called_once()
