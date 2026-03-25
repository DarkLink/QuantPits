import pytest
import os
import pandas as pd
import numpy as np
import importlib
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
    
    # Reload all possible names to ensure they pick up the new QLIB_WORKSPACE_DIR
    for mod_name in ['env', 'quantpits.utils.env', 'run_rolling_health_report', 'quantpits.scripts.run_rolling_health_report']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
            
    from quantpits.scripts import run_rolling_health_report as rrhr
    from quantpits.utils import env
    
    # Mock os.chdir to avoid changing working directory
    monkeypatch.setattr(os, 'chdir', lambda x: None)
    
    yield rrhr, workspace

def create_mock_csvs(workspace, n=70):
    dates = pd.date_range("2020-01-01", periods=n)
    df = pd.DataFrame({
        "Date": dates,
        "Exec_Slippage_Mean": np.random.rand(n) * 0.001,
        "Delay_Cost_Mean": np.random.rand(n) * 0.001,
        "Idiosyncratic_Alpha": np.random.rand(n) * 0.01 + 0.01,
        "Exposure_Liquidity": np.random.rand(n),
        "Exposure_Momentum": np.random.rand(n),
        "Exposure_Volatility": np.random.rand(n),
        "Win_Rate": [0.55] * n,
        "Payoff_Ratio": [1.5] * n
    })
    df.to_csv(workspace / "output" / "rolling_metrics_20.csv", index=False)
    df.to_csv(workspace / "output" / "rolling_metrics_60.csv", index=False)
    return df

def test_evaluate_health_missing_files(mock_env):
    rrhr, workspace = mock_env
    # files don't exist
    rrhr.evaluate_health()

def test_evaluate_health_not_enough_data(mock_env):
    rrhr, workspace = mock_env
    create_mock_csvs(workspace, n=10)
    rrhr.evaluate_health()

def test_evaluate_health_normal(mock_env):
    rrhr, workspace = mock_env
    create_mock_csvs(workspace, n=70)
    rrhr.evaluate_health()
    assert (workspace / "output" / "rolling_health_report.md").exists()

def test_evaluate_health_alerts(mock_env):
    rrhr, workspace = mock_env
    df = create_mock_csvs(workspace, n=70)
    
    # Trigger Slippage Alert (Z-Score < -2)
    # Mean ~0.0005, Std ~0.0003. Setting current to -1.0 should trigger it.
    df.loc[69, "Exec_Slippage_Mean"] = -1.0
    
    # Trigger Size Alert (Percentile < 0.05)
    df["Exposure_Liquidity"] = 0.5
    df.loc[69, "Exposure_Liquidity"] = -10.0
    
    # Trigger Alpha Decay (20d < 60d and < 0)
    df["Idiosyncratic_Alpha"] = 0.01
    df.loc[65:69, "Idiosyncratic_Alpha"] = -0.01
    
    # Reuse the 60 metrics for 20 for simplicity
    df.to_csv(workspace / "output" / "rolling_metrics_60.csv", index=False)
    df.to_csv(workspace / "output" / "rolling_metrics_20.csv", index=False)
    
    rrhr.evaluate_health()
    report = (workspace / "output" / "rolling_health_report.md").read_text()
    assert "🔴" in report
    assert "执行摩擦崩盘" in report
