import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# 模拟 classify_history 依赖的 env
import sys
import os

def test_classify_history_dry_run(monkeypatch, tmp_path, capsys):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    # Mock sys.argv to simulate running the script
    monkeypatch.setattr(sys, 'argv', ['classify_history.py', '--dry-run'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    # Path setup is done locally inside the test but we need to patch the script's behavior
    mock_df = pd.DataFrame({
        "trade_date": ["2026-03-01"],
        "instrument": ["000001"],
        "trade_type": ["BUY"],
        "trade_class": ["S"],
        "suggestion_date": ["2026-02-28"],
        "suggestion_rank": [1]
    })
    
    with patch("quantpits.scripts.analysis.trade_classifier.classify_trades", return_value=mock_df) as mock_classify:
        with patch("quantpits.scripts.analysis.trade_classifier._print_summary") as mock_summary:
            from quantpits.tools import classify_history
            import importlib
            importlib.reload(classify_history)
            
            classify_history.main()
            
            mock_classify.assert_called_once_with(verbose=True)
            mock_summary.assert_called_once_with(mock_df)
            
            captured = capsys.readouterr()
            assert "[DRY-RUN] Skipping file write" in captured.out
            assert "Historical Trade Classification" in captured.out

def test_classify_history_verbose(monkeypatch, tmp_path, capsys):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    monkeypatch.setattr(sys, 'argv', ['classify_history.py', '--verbose'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    mock_df = pd.DataFrame({
        "trade_date": ["2026-03-01"],
        "instrument": ["000001"],
        "trade_type": ["BUY"],
        "trade_class": ["S"],
        "suggestion_date": ["2026-02-28"],
        "suggestion_rank": [1]
    })
    
    with patch("quantpits.scripts.analysis.trade_classifier.classify_trades", return_value=mock_df):
        with patch("quantpits.scripts.analysis.trade_classifier.save_classification") as mock_save:
            from quantpits.tools import classify_history
            import importlib
            importlib.reload(classify_history)
            
            classify_history.main()
            
            mock_save.assert_called_once_with(mock_df)
            
            captured = capsys.readouterr()
            assert "Detailed Classification" in captured.out
            assert "SIGNAL" in captured.out
            assert "Classification complete" in captured.out
