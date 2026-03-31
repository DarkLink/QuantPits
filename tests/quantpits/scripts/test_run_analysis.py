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
    
    from quantpits.utils import env
    from quantpits.scripts import run_analysis
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
    mock_ex.trade_log = pd.DataFrame({"trade_class": ["S"], "成交金额": [100000], "交易类别": ["买入"]})
    
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
    mock_pa.calculate_classified_returns.return_value = {
        "class_df": pd.DataFrame(),
        "manual_buys": pd.DataFrame(),
        "manual_sells": pd.DataFrame()
    }
    
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
    mock_ex.trade_log = pd.DataFrame({"trade_class": ["S"], "成交金额": [100000], "交易类别": ["买入"]})
    
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


@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.SingleModelAnalyzer')
@patch('quantpits.scripts.run_analysis.EnsembleAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
def test_main_coverage_edges(mock_port, mock_exec, mock_ens, mock_single, 
                             mock_fwd, mock_load_pred, mock_market, mock_init, 
                             mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    idx = pd.MultiIndex.from_tuples([("A", pd.to_datetime("2020-01-01"))], names=["instrument", "datetime"])
    mock_fwd.return_value = pd.DataFrame({"score": [0.01]}, index=idx)
    
    # Empty preds for m1, normal preds for m2
    def load_pred_func(m, *args):
        if m == 'm1': return pd.DataFrame()
        return pd.DataFrame({"score": [0.5]}, index=idx)
    mock_load_pred.side_effect = load_pred_func
    
    mock_sma = MagicMock()
    mock_single.return_value = mock_sma
    mock_sma.calculate_rank_ic.return_value = (pd.Series([0.05]), 0.6, 1.5)
    mock_sma.calculate_ic_decay.return_value = {}  # Cover empty ic_decay
    mock_sma.calculate_quantile_spread.return_value = pd.DataFrame({"Spread": [0.001]})
    mock_sma.calculate_long_only_ic.return_value = (pd.Series([0.06]), 0.06)

    mock_ea = MagicMock()
    mock_ens.return_value = mock_ea
    mock_ea.calculate_signal_correlation.return_value = pd.DataFrame([[1.0]], index=["m2"], columns=["m2"])
    mock_ea.calculate_marginal_contribution.return_value = {
        "Full_Ensemble_Sharpe": 2.0,
        "Marginal_Contributions": {"m2": 0.1}
    }
    mock_ea.calculate_ensemble_ic_metrics.return_value = {"Rank_IC_Mean": 0.07, "ICIR": 0.08, "IC_Win_Rate": 0.55, "Spread_Mean": 0.01, "Long_Only_IC_Mean": 0.09}

    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
        "Delay_Cost": [0.001] * 20, 
        "Exec_Slippage": [0.0005] * 20, 
        "Total_Friction": [0.0015] * 20, 
        "成交金额": [100000] * 20,
        "交易类别": ["买入"] * 10 + ["卖出"] * 10,
        "ADV_Participation_Rate": [0.06] * 20,
        "Absolute_Slippage_Amount": [-100.0] * 20 
    })
    mock_ex.calculate_path_dependency.return_value = pd.DataFrame({"MFE": [0.01], "MAE": [-0.01], "成交金额": [100000]})
    mock_ex.analyze_explicit_costs.return_value = {"fee_ratio": 0.0005, "total_fees": 10.0, "total_dividend": -25.0} 
    mock_ex.analyze_order_discrepancies.return_value = {
        "total_missed_count": 120,
        "total_days_with_misses": 5,
        "avg_missed_buys_return": 0.05,
        "theoretical_avg_substitute_return": 0.01,
        "realized_avg_substitute_return": 0.01,
        "theoretical_substitute_bias_impact": -0.04,
        "realized_substitute_bias_impact": -0.04
    }
    
    mock_ex.trade_log = pd.DataFrame({
        "trade_class": ["U", "M"], 
        "成交金额": [100000, 200000], 
        "交易类别": ["买入", "买入"],
        "成交日期": [pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01")],
        "证券代码": ["S1", "S2"]
    })
    
    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_traditional_metrics.return_value = {
        "CAGR": 0.15, "Volatility": 0.20, 
        "Turnover_Rate_Annual": 1500.2,
        "Max_Time_Under_Water_Days": 1
    }
    mock_pa.calculate_factor_exposure.return_value = {
        "Beta_Market": 1.0, "Barra_Liquidity_Exp": 0.1, "Barra_Momentum_Exp": 0.1, "Barra_Volatility_Exp": 0.1,
        "Factor_Annualized": {"size": 0.1, "momentum": 0.1, "volatility": 0.1},
        "Multi_Factor_Intercept": 0.05
    }
    mock_pa.calculate_style_exposures.return_value = {}
    mock_pa.calculate_holding_metrics.return_value = {"Avg_Daily_Holdings_Count": 15.3}
    mock_pa.calculate_classified_returns.return_value = {
        "class_df": pd.DataFrame({"S": [0.1]}),
        "manual_buys": pd.DataFrame({"成交日期": [pd.to_datetime("2020-01-01")], "成交金额": [100], "证券代码": ["S1"], "交易类别": ["买入"]}),
        "manual_sells": pd.DataFrame({"成交日期": [pd.to_datetime("2020-01-01")], "成交金额": [200], "证券代码": ["S2"], "交易类别": ["卖出"]})
    }
    
    import sys
    report_path = str(workspace / "output" / "edges.md")
    
    with patch.object(sys, 'argv', ['script.py', '--models', 'm1', 'm2', '--start-date', 'INVALID_DATE', '--output', 'output/edges.md', '--shareable']):
        ra.main()
        
    assert os.path.exists(report_path)

def test_format_functions_logic(mock_env):
    ra, _ = mock_env
    
    # Use ra.main context or direct access if they were helper functions in main
    # Actually they are local functions nested in main. 
    # To test them without modifying source, I need to trigger them through main() with specific arguments and mocks.
    # However, testing shareable=True already covers some.
    pass

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.SingleModelAnalyzer')
@patch('quantpits.scripts.run_analysis.EnsembleAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
def test_main_friction_and_discrepancy_comprehensive(mock_port, mock_exec, mock_ens, mock_single, 
                                                    mock_fwd, mock_load_pred, mock_market, mock_init, 
                                                    mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    # 1. Mock execution details to cover lines 210-211, 213-215, 226-227, 229-231
    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
        "Delay_Cost": [0.001, 0.002, 0.003, 0.004], 
        "Exec_Slippage": [0.0005, 0.0006, 0.0007, 0.0008], 
        "Total_Friction": [0.0015, 0.0026, 0.0037, 0.0048], 
        "成交金额": [100000, 200000, 300000, 400000],
        "交易类别": ["买入", "卖出", "买入", "卖出"],
        "Absolute_Slippage_Amount": [50.0, 60.0, 70.0, 80.0],
        "ADV_Participation_Rate": [0.0005, 0.005, 0.03, 0.06], # triggers < 0.1%, < 1.0%, < 5.0%, > 5.0%
        "trade_class": ["S", "S", "A", "A"] # All quant
    })
    mock_ex.calculate_path_dependency.return_value = pd.DataFrame({"MFE": [0.01], "MAE": [-0.01], "成交金额": [100000]})
    mock_ex.analyze_explicit_costs.return_value = {"fee_ratio": 0.0005, "total_fees": 10.0, "total_dividend": 5.0}
    
    # 2. Mock discrepancy to cover 267-270
    mock_ex.analyze_order_discrepancies.return_value = {
        "total_missed_count": 10,
        "theoretical_substitute_bias_impact": -0.01,
        "realized_substitute_bias_impact": -0.02,
        "total_days_with_misses": 2,
        "avg_missed_buys_return": 0.02,
        "theoretical_avg_substitute_return": 0.01,
        "realized_avg_substitute_return": 0.00
    }
    mock_ex.trade_log = pd.DataFrame({"trade_class": ["S"], "成交金额": [100000], "交易类别": ["买入"]})
    
    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_traditional_metrics.return_value = {}
    mock_pa.calculate_factor_exposure.return_value = {}
    mock_pa.calculate_style_exposures.return_value = {}
    mock_pa.calculate_holding_metrics.return_value = {}
    mock_pa.calculate_classified_returns.return_value = {"class_df": pd.DataFrame(), "manual_buys": pd.DataFrame(), "manual_sells": pd.DataFrame()}
    
    report_path = str(workspace / "output" / "friction_details.md")
    with patch.object(sys, 'argv', ['script.py', '--output', 'output/friction_details.md']):
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    assert "Absolute Slippage Amount: 120.00" in content # 50 + 70
    assert "Absolute Slippage Amount: 140.00" in content # 60 + 80
    # format_adv in non-shareable is f"{val:.4%}"
    # Buy ADVMean: (0.0005+0.03)/2 = 0.01525 -> 1.5250%
    # Buy ADVMax: 0.03 -> 3.0000%
    assert "ADV Participation Rate (Mean / Max): 1.5250% / 3.0000%" in content
    # Sell ADVMean: (0.005+0.06)/2 = 0.0325 -> 3.2500%
    # Sell ADVMax: 0.06 -> 6.0000%
    assert "ADV Participation Rate (Mean / Max): 3.2500% / 6.0000%" in content
    assert "**Substitution Bias (Unlucky/Loss) (Theoretical)**: -1.0000%" in content
    assert "**Substitution Bias (Unlucky/Loss) (Realized with Cost)**: -2.0000%" in content

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
def test_main_performance_attribution(mock_exec, mock_port, mock_fwd, mock_load_pred, mock_market, mock_init, mock_env):
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
    # Mock data to trigger lines 374-405
    # CAGR must be not None and not NaN
    mock_pa.calculate_traditional_metrics.return_value = {
        "CAGR": 0.25, 
        "Benchmark_CAGR": 0.1,
        "Volatility": 0.15,
        "Turnover_Rate_Annual": 2.5,
        "Max_Time_Under_Water_Days": 10
    }
    mock_pa.calculate_factor_exposure.return_value = {
        "Beta_Market": 1.2,
        "Barra_Liquidity_Exp": 0.5,
        "Barra_Momentum_Exp": -0.2,
        "Barra_Volatility_Exp": 0.1,
        "Factor_Annualized": {
            "size": 0.05,
            "momentum": 0.1,
            "volatility": 0.02
        },
        "Multi_Factor_Intercept": 0.08,
        "Multi_Factor_R_Squared": 0.85,
        "Alpha": 0.05
    }
    mock_pa.calculate_style_exposures.return_value = {"Style_Extra": 0.1} # covers line 295
    mock_pa.calculate_holding_metrics.return_value = {"Avg_Daily_Holdings_Count": 42.4} # covers line 307
    mock_pa.calculate_classified_returns.return_value = {"class_df": pd.DataFrame(), "manual_buys": pd.DataFrame(), "manual_sells": pd.DataFrame()}

    report_path = str(workspace / "output" / "attribution.md")
    with patch.object(sys, 'argv', ['script.py', '--output', 'output/attribution.md']):
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    assert "Performance Attribution" in content
    assert "Beta Return (Exposure to Market): 12.00%" in content
    assert "Idiosyncratic Alpha (Stock Selection / Timing): 8.00%" in content
    assert "Avg_Daily_Holdings_Count**: 42.4" in content # format 1f (line 307)
    assert "Multi_Factor_R_Squared**: 0.8500" in content # format 4f (line 361)
    assert "Alpha**: 5.0000%" in content # format 4% (line 366)

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
def test_main_manual_trades_details(mock_exec, mock_port, mock_market, mock_init, mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame()
    mock_ex.calculate_path_dependency.return_value = pd.DataFrame()
    mock_ex.analyze_explicit_costs.return_value = {}
    mock_ex.analyze_order_discrepancies.return_value = {}
    # Classification distribution needs trade_log (line 425)
    mock_ex.trade_log = pd.DataFrame({
        "交易类别": ["买入", "卖出", "买入", "卖出"],
        "trade_class": ["S", "A", "M", "U"],
        "成交金额": [1000, 2000, 3000, 4000]
    })

    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_traditional_metrics.return_value = {}
    mock_pa.calculate_factor_exposure.return_value = {}
    mock_pa.calculate_style_exposures.return_value = {}
    mock_pa.calculate_holding_metrics.return_value = {}
    
    manual_buys = pd.DataFrame({
        "成交日期": [pd.to_datetime("2024-01-01")],
        "证券代码": ["SH600000"],
        "交易类别": ["买入"],
        "成交金额": [3000]
    })
    manual_sells = pd.DataFrame({
        "成交日期": [pd.to_datetime("2024-01-02")],
        "证券代码": ["SH600001"],
        "交易类别": ["卖出"],
        "成交金额": [2000]
    })
    
    mock_pa.calculate_classified_returns.return_value = {
        "class_df": pd.DataFrame({"class": ["S", "A", "M", "U"]}), # covers line 410
        "manual_buys": manual_buys,
        "manual_sells": manual_sells
    }

    report_path = str(workspace / "output" / "manual_trades.md")
    with patch.object(sys, 'argv', ['script.py', '--output', 'output/manual_trades.md']):
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    assert "Manual Trade Details" in content
    assert "| 2024-01-01 | SH600000 | BUY | ¥3,000 |" in content
    assert "| 2024-01-02 | SH600001 | SELL | ¥2,000 |" in content
    assert "| SIGNAL | 1" in content # Distribution table

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.SingleModelAnalyzer')
@patch('quantpits.scripts.run_analysis.EnsembleAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
def test_main_empty_ic_decay_and_corr(mock_port, mock_exec, mock_ens, mock_single, 
                                      mock_fwd, mock_load_pred, mock_market, mock_init, 
                                      mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    idx = pd.MultiIndex.from_tuples([("A", pd.to_datetime("2020-01-01"))], names=["instrument", "datetime"])
    mock_load_pred.return_value = pd.DataFrame({"score": [0.5]}, index=idx)
    mock_fwd.return_value = pd.DataFrame({"label": [0.01]}, index=idx)
    
    # Mock SingleModelAnalyzer
    mock_sma = MagicMock()
    mock_single.return_value = mock_sma
    mock_sma.calculate_rank_ic.return_value = (pd.Series([0.05]), 0.6, 1.5)
    mock_sma.calculate_ic_decay.return_value = {}
    mock_sma.calculate_quantile_spread.return_value = pd.DataFrame()
    mock_sma.calculate_long_only_ic.return_value = (pd.Series([0.06]), 0.06)

    mock_ea = MagicMock()
    mock_ens.return_value = mock_ea
    mock_ea.calculate_signal_correlation.return_value = pd.DataFrame() # covers line 146
    mock_ea.calculate_marginal_contribution.return_value = {} # covers line 149
    mock_ea.calculate_ensemble_ic_metrics.return_value = {} # covers line 159
    
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
    mock_pa.calculate_classified_returns.return_value = {"class_df": pd.DataFrame(), "manual_buys": pd.DataFrame(), "manual_sells": pd.DataFrame()}

    report_path = str(workspace / "output" / "empty_metrics.md")
    # Multiple models to trigger ensemble check
    # Fix potential ValueError in ra.main if models_preds has duplicate model names
    with patch.object(sys, 'argv', ['script.py', '--models', 'm1', 'm2', '--output', 'output/empty_metrics.md']):
        # We need to make sure load_model_predictions returns unique dfs if called twice
        # Current load_pred_func in test_main_coverage_edges might be reused but here we use default mock
        mock_load_pred.side_effect = [pd.DataFrame({"score": [0.5]}, index=idx), pd.DataFrame({"score": [0.5]}, index=idx)]
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    assert "*N/A*" in content # corr_matrix empty (line 146)

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.SingleModelAnalyzer')
@patch('quantpits.scripts.run_analysis.EnsembleAnalyzer')
@patch('quantpits.scripts.run_analysis.ExecutionAnalyzer')
@patch('quantpits.scripts.run_analysis.PortfolioAnalyzer')
def test_main_shareable_comprehensive(mock_port, mock_exec, mock_ens, mock_single, 
                                      mock_fwd, mock_load_pred, mock_market, mock_init, 
                                      mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    idx = pd.MultiIndex.from_tuples([("A", pd.to_datetime("2020-01-01"))], names=["instrument", "datetime"])
    mock_load_pred.return_value = pd.DataFrame({"score": [0.5]}, index=idx)
    mock_fwd.return_value = pd.DataFrame({"score": [0.01]}, index=idx)
    
    mock_sma = MagicMock()
    mock_single.return_value = mock_sma
    mock_sma.calculate_rank_ic.return_value = (pd.Series([0.05]), 0.6, 1.5)
    mock_sma.calculate_ic_decay.return_value = {"T+1": 0.05}
    mock_sma.calculate_quantile_spread.return_value = pd.DataFrame({"Spread": [0.001]})
    mock_sma.calculate_long_only_ic.return_value = (pd.Series([0.06]), 0.06)
    
    mock_ea = MagicMock()
    mock_ens.return_value = mock_ea
    mock_ea.calculate_signal_correlation.return_value = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["m1", "m2"], columns=["m1", "m2"])
    mock_ea.calculate_marginal_contribution.return_value = {
        "Full_Ensemble_Sharpe": 2.0,
        "Marginal_Contributions": {"m1": 0.1, "m2": -0.05}
    }
    mock_ea.calculate_ensemble_ic_metrics.return_value = {
        "Rank_IC_Mean": 0.07,
        "ICIR": 1.2,
        "IC_Win_Rate": 0.55,
        "Spread_Mean": 0.01,
        "Long_Only_IC_Mean": 0.08
    }
    
    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
        "Delay_Cost": [0.001, 0.002], 
        "Exec_Slippage": [0.0005, 0.0006], 
        "Total_Friction": [0.0015, 0.0026], 
        "成交金额": [100000, 200000],
        "交易类别": ["买入", "卖出"],
        "Absolute_Slippage_Amount": [50.0, 60.0]
    })
    mock_ex.calculate_path_dependency.return_value = pd.DataFrame({"MFE": [0.01], "MAE": [-0.01], "成交金额": [100000]})
    mock_ex.analyze_explicit_costs.return_value = {"fee_ratio": 0.0005, "total_fees": 10.0, "total_dividend": 5.0}
    mock_ex.analyze_order_discrepancies.return_value = {
        "total_missed_count": 10,
        "theoretical_substitute_bias_impact": 0.01,
        "realized_substitute_bias_impact": 0.01,
        "total_days_with_misses": 1,
        "avg_missed_buys_return": 0.02,
        "theoretical_avg_substitute_return": 0.03,
        "realized_avg_substitute_return": 0.03
    }
    mock_ex.trade_log = pd.DataFrame({"trade_class": ["S"], "成交金额": [100000], "交易类别": ["买入"]})
    
    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_traditional_metrics.return_value = {
        "CAGR": 0.15, 
        "Benchmark_CAGR": 0.05,
        "Volatility": 0.1,
        "Turnover_Rate_Annual": 1.5,
        "Max_Time_Under_Water_Days": 42
    }
    mock_pa.calculate_factor_exposure.return_value = {"Beta_Market": 1.0}
    mock_pa.calculate_style_exposures.return_value = {}
    mock_pa.calculate_holding_metrics.return_value = {"Avg_Daily_Holdings_Count": 10.5}
    mock_pa.calculate_classified_returns.return_value = {"class_df": pd.DataFrame({"S": [0.1]}), "manual_buys": pd.DataFrame(), "manual_sells": pd.DataFrame()}

    report_path = str(workspace / "output" / "shareable_full.md")
    with patch.object(sys, 'argv', ['script.py', '--models', 'm1', 'm2', '--start-date', '2024-07-01', '--end-date', '2024-08-31', '--output', 'output/shareable_full.md', '--shareable']):
        ra.main()
    
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
    
    assert "Comprehensive Analysis Report (Shareable)" in content
    assert "~1 Months (Summer 2024 - Summer 2024)" in content # (2024-2024)*12 + 8-7=1
    assert "Rank IC Mean**: 0.1" in content # ensemble_metrics shareable (line 162)
    assert "Drop `m1` -> impact on Sharpe: +0.1" in content # marginal shareable (line 154)
    assert "**Substitution Bias (Lucky/Gain) (Theoretical)**: 1.0%" in content # discrepancy shareable (line 262)
    assert "**Avg_Daily_Holdings_Count**: ~10" in content # holding shareable (line 305)
    assert "**CAGR (252-day basis)**: 15.0%" in content # CAGR shareable (line 325)
    assert "**Max_Time_Under_Water_Days**: 35-45 days" in content # fuzzy days shareable (line 320, 340)
    assert "Math/Compounding Gap (Residual): 10.0%" in content # Attribution shareable (line 399)

@patch('quantpits.scripts.run_analysis.init_qlib')
@patch('quantpits.scripts.run_analysis.load_market_config')
@patch('quantpits.scripts.run_analysis.load_model_predictions')
@patch('quantpits.scripts.run_analysis.get_forward_returns')
@patch('quantpits.scripts.run_analysis.SingleModelAnalyzer')
def test_main_helper_coverage(mock_single, mock_fwd, mock_load_pred, mock_market, mock_init, mock_env):
    ra, workspace = mock_env
    mock_market.return_value = ("csi300", "SH000300")
    
    # Coverage for format_date_range different seasons and exceptions (lines 51-66)
    # Coverage for format_adv different ranges (lines 79-83)
    
    mock_sma = MagicMock()
    mock_single.return_value = mock_sma
    mock_sma.calculate_rank_ic.return_value = (pd.Series([0.05]), 0.6, 1.5)
    mock_sma.calculate_ic_decay.return_value = {}
    
    report_path = str(workspace / "output" / "helpers.md")
    
    # 1. Test different seasons for format_date_range
    # Jan: Winter, Apr: Spring, Jul: Summer, Oct: Fall
    with patch.object(sys, 'argv', ['script.py', '--start-date', '2024-10-01', '--end-date', '2024-12-31', '--output', 'output/helpers.md', '--shareable']):
        ra.main()
    with open(report_path, "r") as f:
        content = f.read()
    assert "Fall 2024" in content
    assert "Winter 2024" in content
    
    # 2. Test format_adv ranges and format_count cases (lines 73-75)
    # Need ExecutionAnalyzer to return specific values
    with patch('quantpits.scripts.run_analysis.ExecutionAnalyzer') as mock_exec:
        mock_ex = MagicMock()
        mock_exec.return_value = mock_ex
        # Test each case of format_adv: < 0.1%, < 1.0%, < 5.0%, > 5.0%
        mock_ex.calculate_path_dependency.return_value = pd.DataFrame()
        mock_ex.analyze_explicit_costs.return_value = {}
        mock_ex.analyze_order_discrepancies.return_value = {}
        mock_ex.trade_log = pd.DataFrame()

        # Case 1: < 0.1%
        mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
            "Delay_Cost": [0], "Exec_Slippage": [0], "Total_Friction": [0], "成交金额": [1000], "交易类别": ["买入"],
            "ADV_Participation_Rate": [0.0005], "trade_class": ["S"]
        })
        with patch.object(sys, 'argv', ['script.py', '--output', 'output/adv1.md', '--shareable']):
            ra.main()
        with open(workspace / "output" / "adv1.md", "r") as f:
            content_adv = f.read()
        assert "< 0.1%" in content_adv

        # Case 2: < 1.0%
        mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
            "Delay_Cost": [0], "Exec_Slippage": [0], "Total_Friction": [0], "成交金额": [1000], "交易类别": ["买入"],
            "ADV_Participation_Rate": [0.005], "trade_class": ["S"]
        })
        with patch.object(sys, 'argv', ['script.py', '--output', 'output/adv2.md', '--shareable']):
            ra.main()
        with open(workspace / "output" / "adv2.md", "r") as f:
            content_adv = f.read()
        assert "< 1.0%" in content_adv

        # Case 3: < 5.0%
        mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
            "Delay_Cost": [0], "Exec_Slippage": [0], "Total_Friction": [0], "成交金额": [1000], "交易类别": ["买入"],
            "ADV_Participation_Rate": [0.03], "trade_class": ["S"]
        })
        with patch.object(sys, 'argv', ['script.py', '--output', 'output/adv3.md', '--shareable']):
            ra.main()
        with open(workspace / "output" / "adv3.md", "r") as f:
            content_adv = f.read()
        assert "< 5.0%" in content_adv

        # Case 4: > 5.0%
        mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
            "Delay_Cost": [0], "Exec_Slippage": [0], "Total_Friction": [0], "成交金额": [1000], "交易类别": ["买入"],
            "ADV_Participation_Rate": [0.08], "trade_class": ["S"]
        })
        with patch.object(sys, 'argv', ['script.py', '--output', 'output/adv4.md', '--shareable']):
            ra.main()
        with open(workspace / "output" / "adv4.md", "r") as f:
            content_adv = f.read()
        assert "> 5.0%" in content_adv

    # 3. Test Spring and Summer seasons (lines 65-66)
    with patch.object(sys, 'argv', ['script.py', '--start-date', '2024-04-01', '--end-date', '2024-07-01', '--output', 'output/seasons.md', '--shareable']):
        ra.main()
    with open(workspace / "output" / "seasons.md", "r") as f:
        content_seasons = f.read()
    assert "Spring 2024" in content_seasons
    assert "Summer 2024" in content_seasons

    # 4. Coverage for else blocks in metrics (lines 345-348) and R_Squared shareable (line 360)
    with patch('quantpits.scripts.run_analysis.PortfolioAnalyzer') as mock_port:
        mock_pa = MagicMock()
        mock_port.return_value = mock_pa
        # Need a metric that goes to 'else' block (line 344)
        mock_pa.calculate_traditional_metrics.return_value = {"Sharpe": 2.1}
        mock_pa.calculate_factor_exposure.return_value = {"Multi_Factor_R_Squared": 0.85}
        mock_pa.calculate_style_exposures.return_value = {}
        mock_pa.calculate_holding_metrics.return_value = {}
        mock_pa.calculate_classified_returns.return_value = {"class_df": pd.DataFrame(), "manual_buys": pd.DataFrame(), "manual_sells": pd.DataFrame()}

        # 4a. Shareable mode
        with patch.object(sys, 'argv', ['script.py', '--output', 'output/extra.md', '--shareable']):
            ra.main()
        with open(workspace / "output" / "extra.md", "r") as f:
            content_extra = f.read()
        assert "Sharpe**: 2.1" in content_extra
        assert "Multi_Factor_R_Squared**: 0.85" in content_extra

        # 4b. Non-shareable mode (line 348)
        with patch.object(sys, 'argv', ['script.py', '--output', 'output/extra2.md']):
            ra.main()
        with open(workspace / "output" / "extra2.md", "r") as f:
            content_extra2 = f.read()
        assert "Sharpe**: 2.1000" in content_extra2

    # 5. Coverage for exception block in format_date_range (line 66)
    with patch.object(sys, 'argv', ['script.py', '--start-date', 'INVALID', '--end-date', 'INVALID', '--output', 'output/invalid_date.md', '--shareable']):
        ra.main()
    with open(workspace / "output" / "invalid_date.md", "r") as f:
        content_invalid = f.read()
    assert "Redacted Range" in content_invalid


