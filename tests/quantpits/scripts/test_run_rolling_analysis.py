import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "output").mkdir()
    
    import sys
    script_dir = os.path.join(os.getcwd(), "quantpits/scripts")
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    
    from quantpits.scripts import env, run_rolling_analysis
    import importlib
    importlib.reload(env)
    importlib.reload(run_rolling_analysis)
    
    # Mock os.chdir to avoid changing working directory during tests
    monkeypatch.setattr(os, 'chdir', lambda x: None)
    
    yield run_rolling_analysis, workspace

@patch('quantpits.scripts.run_rolling_analysis.init_qlib')
@patch('quantpits.scripts.run_rolling_analysis.PortfolioAnalyzer')
@patch('quantpits.scripts.run_rolling_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_rolling_analysis.get_daily_features')
def test_compute_rolling_metrics_basic(mock_features, mock_exec, mock_port, mock_init, mock_env):
    rra, workspace = mock_env
    
    # Mock Portfolio returns
    mock_port_instance = mock_port.return_value
    dates = pd.to_datetime([f"2020-01-{i:02d}" for i in range(1, 16)])
    mock_port_instance.calculate_daily_returns.return_value = pd.Series(np.random.rand(15) * 0.02, index=dates)
    mock_port_instance.daily_amount = pd.DataFrame({"CSI300": np.random.rand(15) * 100 + 100}, index=dates)
    
    # Mock Features - Multiple instruments to hit factor logic
    instruments = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    mock_features.return_value = pd.DataFrame({
        "close": np.random.rand(len(idx)) + 10,
        "volume": np.random.rand(len(idx)) * 1000 + 500
    }, index=idx)
    
    # Mock Execution
    mock_exec_instance = mock_exec.return_value
    mock_exec_instance.calculate_slippage_and_delay.return_value = pd.DataFrame({
        "成交日期": dates,
        "Exec_Slippage": [0.001]*15,
        "Delay_Cost": [0.001]*15,
        "Total_Friction": [0.002]*15
    })
    mock_exec_instance.trade_log = pd.DataFrame({
        "证券代码": ["A", "A"],
        "成交日期": pd.to_datetime(["2020-01-01", "2020-01-15"]),
        "交易类别": ["买入", "卖出"],
        "成交价格": [10.0, 11.0],
        "成交数量": [100, 100]
    })
    
    rra.compute_rolling_metrics(windows=[6], sub_window=5)
    
    # Check if output file exists
    out_file = workspace / "output" / "rolling_metrics_6.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert "Portfolio_Return" in df.columns

def test_compute_rolling_metrics_empty(mock_env):
    rra, workspace = mock_env
    with patch('quantpits.scripts.run_rolling_analysis.PortfolioAnalyzer') as mock_port:
        mock_port.return_value.calculate_daily_returns.return_value = pd.Series(dtype=float)
        rra.compute_rolling_metrics() # Should return early

def test_main(mock_env):
    rra, workspace = mock_env
    with patch('quantpits.scripts.run_rolling_analysis.compute_rolling_metrics') as mock_compute:
        with patch('quantpits.scripts.run_rolling_analysis.load_market_config', return_value=('csi300', {})):
            rra.main()
            mock_compute.assert_called_once()
