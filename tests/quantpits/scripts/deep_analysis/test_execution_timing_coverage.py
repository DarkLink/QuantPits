import pytest
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from quantpits.scripts.deep_analysis.agents.execution_quality import ExecutionQualityAgent

@pytest.fixture
def mock_execution_context(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = workspace / "data"
    data_dir.mkdir()
    
    ctx = MagicMock()
    ctx.workspace_root = str(workspace)
    ctx.start_date = "2026-04-20"
    ctx.end_date = "2026-04-20"
    ctx.window_label = "test_window"
    
    # ExecutionQualityAgent checks trade_log_df at the beginning
    ctx.trade_log_df = pd.DataFrame({
        '成交日期': [pd.Timestamp("2026-04-20"), pd.Timestamp("2026-04-20")],
        '交易类别': ['证券买入', '证券卖出'],
        '证券代码': ['000001', '000002'],
        '成交价格': [10.0, 20.0],
        '成交数量': [1000, 400],
        '成交金额': [10000.0, 8000.0]
    })
    
    return ctx

def test_execution_timing_analysis_success(mock_execution_context):
    workspace = mock_execution_context.workspace_root
    data_dir = os.path.join(workspace, "data")
    
    # Create mock order log
    order_data = {
        '委托日期': [20260420, 20260420],
        '委托时间': ['09:30:00.00', '09:35:00.00'],
        '交易类别': ['证券买入', '证券卖出'],
        '证券代码': ['000001', '000002'],
        '委托数量': [1000.0, 500.0],
        '成交数量': [1000.0, 400.0],
        '撤单数量': [0.0, 100.0],
        '委托状态': ['已成', '部成已撤']
    }
    pd.DataFrame(order_data).to_csv(os.path.join(data_dir, "raw_order_log_full.csv"), index=False)
    
    # Create mock trade log
    trade_data = {
        '日期': [20260420, 20260420],
        '成交时间': ['09:30:05.00', '09:35:10.00'],
        '交易类别': ['证券买入', '证券卖出'],
        '证券代码': ['000001', '000002'],
        '成交数量': [1000.0, 400.0]
    }
    pd.DataFrame(trade_data).to_csv(os.path.join(data_dir, "raw_trade_log_full.csv"), index=False)
    
    agent = ExecutionQualityAgent()
    # Mock other analyzer calls to focus on timing
    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.calculate_slippage_and_delay", return_value=pd.DataFrame()), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_explicit_costs", return_value={}), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_order_discrepancies", return_value={}):
        findings = agent.analyze(mock_execution_context)
    
    # Verify metrics
    assert 'fill_rate' in findings.raw_metrics
    assert 'cancel_rate' in findings.raw_metrics
    assert 'latency_mean_sec' in findings.raw_metrics
    
    # 1000+400 / 1000+500 = 1400/1500 = 0.9333
    assert abs(findings.raw_metrics['fill_rate'] - 0.9333) < 0.001
    # 100 / 1500 = 0.0666
    assert abs(findings.raw_metrics['cancel_rate'] - 0.0667) < 0.001
    
    # Latencies: 5s and 10s -> mean 7.5s
    assert findings.raw_metrics['latency_mean_sec'] == 7.5
    assert findings.raw_metrics['latency_median_sec'] == 7.5

def test_execution_timing_high_cancel_rate(mock_execution_context):
    workspace = mock_execution_context.workspace_root
    data_dir = os.path.join(workspace, "data")
    
    # Create mock order log with high cancel rate
    order_data = {
        '委托日期': [20260420],
        '委托时间': ['09:30:00.00'],
        '交易类别': ['证券买入'],
        '证券代码': ['000001'],
        '委托数量': [1000.0],
        '成交数量': [200.0],
        '撤单数量': [800.0],
        '委托状态': ['部成已撤']
    }
    pd.DataFrame(order_data).to_csv(os.path.join(data_dir, "raw_order_log_full.csv"), index=False)
    
    # Create mock trade log
    trade_data = {
        '日期': [20260420],
        '成交时间': ['09:30:05.00'],
        '交易类别': ['证券买入'],
        '证券代码': ['000001'],
        '成交数量': [200.0]
    }
    pd.DataFrame(trade_data).to_csv(os.path.join(data_dir, "raw_trade_log_full.csv"), index=False)
    
    agent = ExecutionQualityAgent()
    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.calculate_slippage_and_delay", return_value=pd.DataFrame()), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_explicit_costs", return_value={}), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_order_discrepancies", return_value={}):
        findings = agent.analyze(mock_execution_context)
    
    assert findings.raw_metrics['cancel_rate'] == 0.8
    # Should trigger warning finding
    warning_titles = [f.title for f in findings.findings if f.severity == 'warning']
    assert 'Elevated order cancel rate' in warning_titles

def test_execution_timing_missing_files(mock_execution_context):
    # No files created in data_dir
    agent = ExecutionQualityAgent()
    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.calculate_slippage_and_delay", return_value=pd.DataFrame()), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_explicit_costs", return_value={}), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_order_discrepancies", return_value={}):
        findings = agent.analyze(mock_execution_context)
    
    # Should fallback to deferred status or info finding
    timing_findings = [f for f in findings.findings if 'Execution timing analysis' in f.title]
    assert len(timing_findings) > 0
    assert 'pending granular intraday timestamp data' in timing_findings[0].detail

def test_execution_timing_empty_order_log(mock_execution_context):
    workspace = mock_execution_context.workspace_root
    data_dir = os.path.join(workspace, "data")
    
    # Create empty logs
    pd.DataFrame(columns=['委托日期', '委托时间', '交易类别', '证券代码', '委托数量', '成交数量', '撤单数量', '委托状态']).to_csv(os.path.join(data_dir, "raw_order_log_full.csv"), index=False)
    pd.DataFrame(columns=['日期', '成交时间', '交易类别', '证券代码', '成交数量']).to_csv(os.path.join(data_dir, "raw_trade_log_full.csv"), index=False)
    
    agent = ExecutionQualityAgent()
    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.calculate_slippage_and_delay", return_value=pd.DataFrame()), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_explicit_costs", return_value={}), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_order_discrepancies", return_value={}):
        findings = agent.analyze(mock_execution_context)
    
    assert any('No order data available' in f.detail for f in findings.findings)

def test_execution_timing_error_handling(mock_execution_context):
    workspace = mock_execution_context.workspace_root
    data_dir = os.path.join(workspace, "data")
    
    # Create corrupt files
    with open(os.path.join(data_dir, "raw_order_log_full.csv"), "w") as f:
        f.write("corrupt,data\n1,2,3")
    with open(os.path.join(data_dir, "raw_trade_log_full.csv"), "w") as f:
        f.write("corrupt,data\n1,2,3")
        
    agent = ExecutionQualityAgent()
    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.calculate_slippage_and_delay", return_value=pd.DataFrame()), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_explicit_costs", return_value={}), \
         patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_order_discrepancies", return_value={}):
        findings = agent.analyze(mock_execution_context)
        
    assert 'execution_timing_error' in findings.raw_metrics
    assert any('Execution timing analysis error' in f.title for f in findings.findings)
