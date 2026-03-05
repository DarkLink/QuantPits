import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    (workspace / "output").mkdir()
    (workspace / "data").mkdir()
    (workspace / "data" / "order_history").mkdir()
    
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, run_analysis
    import importlib
    importlib.reload(env)
    
    # Reload run_analysis to pick up reloaded env
    importlib.reload(run_analysis)
    
    # Patch ROOT_DIR in run_analysis
    monkeypatch.setattr(run_analysis, 'ROOT_DIR', str(workspace))
    
    yield run_analysis, workspace

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.SingleModelAnalyzer')
@patch('quantpits.scripts.run_analysis.EnsembleAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
def test_main_full(mock_port, mock_exec, mock_ens, mock_single, 
                   mock_fwd, mock_load_pred, mock_market, mock_init, 
                   mock_env):
    ra, workspace = mock_env
    
    # Mock return values for utils
    mock_market.return_value = ("csi300", "SH000300")
    
    # Mock data for load_model_predictions
    idx = pd.MultiIndex.from_tuples([("A", pd.to_datetime("2020-01-01"))], names=["instrument", "datetime"])
    mock_df = pd.DataFrame({"score": [0.5]}, index=idx)
    mock_load_pred.return_value = mock_df
    
    mock_fwd.return_value = pd.DataFrame({"score": [0.01]}, index=idx)
    
    # Mock analyzers
    mock_sma = MagicMock()
    mock_single.return_value = mock_sma
    mock_sma.calculate_rank_ic.return_value = (pd.Series([0.05]), 0.6, 1.5)
    mock_sma.calculate_ic_decay.return_value = {"T+1": 0.05}
    mock_sma.calculate_quantile_spread.return_value = pd.DataFrame({"Spread": [0.001]})
    mock_sma.calculate_long_only_ic.return_value = (pd.Series([0.06]), 0.06)
    
    mock_ea = MagicMock()
    mock_ens.return_value = mock_ea
    mock_ea.calculate_signal_correlation.return_value = pd.DataFrame([[1.0]], index=["m1"], columns=["m1"])
    mock_ea.calculate_marginal_contribution.return_value = {
        "Full_Ensemble_Sharpe": 2.0,
        "Marginal_Contributions": {"m1": 0.1}
    }
    mock_ea.calculate_ensemble_ic_metrics.return_value = {"Rank_IC_Mean": 0.07}
    
    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
        "Delay_Cost": [0.001], 
        "Exec_Slippage": [0.0005], 
        "Total_Friction": [0.0015], 
        "成交金额": [100000],
        "交易类别": ["买入"]
    })
    mock_ex.calculate_path_dependency.return_value = pd.DataFrame({"MFE": [0.01], "MAE": [-0.01], "成交金额": [100000]})
    mock_ex.analyze_explicit_costs.return_value = {"fee_ratio": 0.0005}
    mock_ex.analyze_order_discrepancies.return_value = {"total_missed_count": 0}
    mock_ex.trade_log = pd.DataFrame({"trade_class": ["S"], "成交金额": [100000]})
    
    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_traditional_metrics.return_value = {"CAGR": 0.15, "Volatility": 0.20}
    mock_pa.calculate_factor_exposure.return_value = {"Beta_Market": 1.0}
    mock_pa.calculate_style_exposures.return_value = {}
    mock_pa.calculate_holding_metrics.return_value = {"Turnover": 0.1}
    mock_pa.calculate_classified_returns.return_value = {
        "class_df": pd.DataFrame(),
        "manual_buys": pd.DataFrame(),
        "manual_sells": pd.DataFrame()
    }
    
    # Run main
    import sys
    report_path = str(workspace / "output" / "report.md")
    with patch.object(sys, 'argv', ['script.py', '--models', 'm1', 'm2', '--output', 'output/report.md']):
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    assert "# Comprehensive Analysis Report" in content
    assert "Model: m1" in content

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
def test_main_no_models(mock_port, mock_exec, mock_market, mock_init, mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame()
    mock_ex.calculate_path_dependency.return_value = pd.DataFrame()
    mock_ex.analyze_explicit_costs.return_value = {}
    mock_ex.analyze_order_discrepancies.return_value = {}
    mock_ex.trade_log = pd.DataFrame()
    
    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_traditional_metrics.return_value = {}
    mock_pa.calculate_factor_exposure.return_value = {}
    mock_pa.calculate_style_exposures.return_value = {}
    mock_pa.calculate_holding_metrics.return_value = {}
    mock_pa.calculate_classified_returns.return_value = {}
    
    import sys
    report_path = str(workspace / "output" / "report_no_models.md")
    with patch.object(sys, 'argv', ['script.py', '--output', 'output/report_no_models.md']):
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    assert "None (Portfolio/Execution Only)" in content

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.SingleModelAnalyzer')
@patch('quantpits.scripts.run_analysis.EnsembleAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
def test_main_shareable(mock_port, mock_exec, mock_ens, mock_single, 
                        mock_fwd, mock_load_pred, mock_market, mock_init, 
                        mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    # Mock data for load_model_predictions
    idx = pd.MultiIndex.from_tuples([("A", pd.to_datetime("2020-01-01"))], names=["instrument", "datetime"])
    mock_df = pd.DataFrame({"score": [0.5]}, index=idx)
    mock_load_pred.return_value = mock_df
    mock_fwd.return_value = pd.DataFrame({"score": [0.01]}, index=idx)
    
    # Mock analyzers
    mock_sma = MagicMock()
    mock_single.return_value = mock_sma
    mock_sma.calculate_rank_ic.return_value = (pd.Series([0.05]), 0.6, 1.5)
    mock_sma.calculate_ic_decay.return_value = {"T+1": 0.05}
    mock_sma.calculate_quantile_spread.return_value = pd.DataFrame({"Spread": [0.001]})
    mock_sma.calculate_long_only_ic.return_value = (pd.Series([0.06]), 0.06)
    
    mock_ea = MagicMock()
    mock_ens.return_value = mock_ea
    mock_ea.calculate_signal_correlation.return_value = pd.DataFrame([[1.0]], index=["m1"], columns=["m1"])
    mock_ea.calculate_marginal_contribution.return_value = {
        "Full_Ensemble_Sharpe": 2.0,
        "Marginal_Contributions": {"m1": 0.1}
    }
    mock_ea.calculate_ensemble_ic_metrics.return_value = {"Rank_IC_Mean": 0.07}
    
    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
        "Delay_Cost": [0.001], 
        "Exec_Slippage": [0.0005], 
        "Total_Friction": [0.0015], 
        "成交金额": [100000],
        "交易类别": ["买入"],
        "Absolute_Slippage_Amount": [50.0]  # Add this for dividend calc
    })
    mock_ex.calculate_path_dependency.return_value = pd.DataFrame({"MFE": [0.01], "MAE": [-0.01], "成交金额": [100000]})
    mock_ex.analyze_explicit_costs.return_value = {"fee_ratio": 0.0005, "total_fees": 10.0, "total_dividend": 25.0} # 25/50 = 50%
    mock_ex.analyze_order_discrepancies.return_value = {"total_missed_count": 0}
    # Important: slippage and classification use different parts of mock_ex
    mock_ex.trade_log = pd.DataFrame({"trade_class": ["S"], "成交金额": [100000]})
    
    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_traditional_metrics.return_value = {"CAGR": 0.15, "Volatility": 0.20}
    mock_pa.calculate_factor_exposure.return_value = {"Beta_Market": 1.0}
    mock_pa.calculate_style_exposures.return_value = {}
    mock_pa.calculate_holding_metrics.return_value = {"Turnover": 0.1}
    mock_pa.calculate_classified_returns.return_value = {
        "class_df": pd.DataFrame(),
        "manual_buys": pd.DataFrame({"成交金额": [100], "证券代码": ["S1"]}),
        "manual_sells": pd.DataFrame()
    }
    
    import sys
    report_path = str(workspace / "output" / "shareable.md")
    with patch.object(sys, 'argv', ['script.py', '--models', 'm1', '--output', 'output/shareable.md', '--shareable']):
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    
    assert "Comprehensive Analysis Report (Shareable)" in content
    assert "Absolute Slippage Amount" not in content
    assert "Total explicit fees amount" not in content
    assert "Dividend Offset as % of Total Slippage" in content
    assert "50.00%" in content # 25 / 50 
    assert "Amount Pct" in content
    assert "Manual trade details redacted for privacy" in content
    assert "¥" not in content
    assert "S1" not in content # Redacted stock code
