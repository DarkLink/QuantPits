import pytest
import os
import json
import pandas as pd
from unittest.mock import MagicMock, patch
from quantpits.scripts import migrate_records

@pytest.fixture
def mock_record_files(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    
    # Files defined in quantpits.utils.train_utils
    static_file = root / "latest_train_records.json"
    rolling_file = root / "latest_rolling_records.json"
    
    return root, static_file, rolling_file

def test_main_no_files(mock_record_files, monkeypatch, capsys):
    root, static_file, rolling_file = mock_record_files
    
    # Ensure files don't exist
    assert not static_file.exists()
    assert not rolling_file.exists()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py', '--workspace', str(root)])
    
    migrate_records.main()
    captured = capsys.readouterr()
    # Should print warning about no files found
    assert "没有找到任何旧格式的训练记录文件" in captured.out

def test_main_already_migrated(mock_record_files, monkeypatch, capsys):
    root, static_file, rolling_file = mock_record_files
    
    # Create migrated static file
    static_data = {
        "models": {
            "model1@static": {"experiment": "exp1"}
        }
    }
    with open(static_file, 'w') as f:
        json.dump(static_data, f)
        
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py', '--workspace', str(root)])
    
    migrate_records.main()
    captured = capsys.readouterr()
    # Should print that it is already in model@mode format
    assert "静态记录已经是 model@mode 格式" in captured.out

def test_main_migration_success(mock_record_files, monkeypatch, capsys):
    root, static_file, rolling_file = mock_record_files
    
    # Create legacy static file
    static_data = {"models": {"model1": {"exp": "s1"}}}
    with open(static_file, 'w') as f:
        json.dump(static_data, f)
        
    # Create legacy rolling file
    rolling_data = {"models": {"model2": {"exp": "r1"}}}
    with open(rolling_file, 'w') as f:
        json.dump(rolling_data, f)
        
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py', '--workspace', str(root)])
    
    # Mock the actual migration utility to avoid side effects and complex logic
    migrated_result = {
        "models": {
            "model1@static": {"exp": "s1"},
            "model2@rolling": {"exp": "r1"}
        }
    }
    
    # Mock the actual migration utility at its source
    with patch('quantpits.utils.train_utils.migrate_legacy_records', return_value=migrated_result):
        migrate_records.main()
        captured = capsys.readouterr()
        assert "迁移完成！" in captured.out

def test_main_dry_run(mock_record_files, monkeypatch, capsys):
    root, static_file, rolling_file = mock_record_files
    
    static_data = {"models": {"model1": {"exp": "s1"}}}
    with open(static_file, 'w') as f:
        json.dump(static_data, f)
        
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py', '--workspace', str(root), '--dry-run'])
    
    migrated_result = {"models": {"model1@static": {"exp": "s1"}}}
    
    with patch('quantpits.utils.train_utils.migrate_legacy_records', return_value=migrated_result) as mock_migrate:
        migrate_records.main()
        captured = capsys.readouterr()
        assert "干运行模式" in captured.out
        assert mock_migrate.called
