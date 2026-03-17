import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    out_dir = workspace / "output"
    out_dir.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    # We must patch matplotlib backend before importing plot_model_opinions so it doesn't try to open windows
    import matplotlib
    matplotlib.use('Agg')
    
    from quantpits.utils import env
    from quantpits.tools import plot_model_opinions
    import importlib
    importlib.reload(env)
    importlib.reload(plot_model_opinions)
    
    os.chdir(str(workspace))
    yield plot_model_opinions, out_dir

def test_main_no_csv(mock_env, capsys, monkeypatch):
    pmo, out_dir = mock_env
    # Empty output dir
    
    monkeypatch.setattr('sys.argv', ['script.py'])
    pmo.main()
    
    captured = capsys.readouterr()
    assert "未找到" in captured.out

@patch('matplotlib.pyplot.savefig')
def test_rank_extraction_regex(mock_savefig, mock_env, capsys, monkeypatch):
    pmo, out_dir = mock_env
    
    # Create fake CSV
    csv_path = out_dir / "model_opinions_2020-01-01.csv"
    df = pd.DataFrame({
        "m1": ["BUY (1)", "-- (10)"],
        "m2": ["SELL (300)", "BUY (5)"]
    }, index=["A", "B"])
    df.to_csv(csv_path)
    
    monkeypatch.setattr('sys.argv', ['script.py', '--input', str(csv_path)])
    pmo.main()
    
    captured = capsys.readouterr()
    assert "已保存至" in captured.out
    mock_savefig.assert_called_once()
    saved_path = mock_savefig.call_args[0][0]
    assert "model_opinions_2020-01-01_linechart.png" in saved_path

@patch('matplotlib.pyplot.savefig')
def test_rank_extraction_no_rank(mock_savefig, mock_env, capsys, monkeypatch):
    pmo, out_dir = mock_env
    
    # Create fake CSV with NO parenthesis
    csv_path = out_dir / "model_opinions_no_rank.csv"
    df = pd.DataFrame({
        "m1": ["BUY", "--"],
        "m2": ["SELL", "BUY"]
    }, index=["A", "B"])
    df.to_csv(csv_path)
    
    monkeypatch.setattr('sys.argv', ['script.py', '--input', str(csv_path)])
    pmo.main()
    
    captured = capsys.readouterr()
    assert "已保存至" in captured.out
    mock_savefig.assert_called_once()

def test_main_wrong_input(mock_env, capsys, monkeypatch):
    pmo, out_dir = mock_env
    monkeypatch.setattr('sys.argv', ['script.py', '--input', 'non_existent.csv'])
    pmo.main()
    captured = capsys.readouterr()
    assert "找不到文件" in captured.out

@patch('matplotlib.pyplot.savefig')
def test_main_auto_find(mock_savefig, mock_env, capsys, monkeypatch):
    pmo, out_dir = mock_env
    # Create two files, auto-find should pick the latest (alphabetically)
    (out_dir / "model_opinions_2020-01-01.csv").write_text("index,m1\nA,BUY (1)")
    (out_dir / "model_opinions_2020-01-02.csv").write_text("index,m1\nA,BUY (1)")
    
    monkeypatch.setattr('sys.argv', ['script.py'])
    pmo.main()
    
    captured = capsys.readouterr()
    assert "正在读取数据: output/model_opinions_2020-01-02.csv" in captured.out
    assert "已保存至" in captured.out
