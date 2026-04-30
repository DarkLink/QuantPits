"""
Supplemental tests targeting uncovered branches in execution_quality.py.

Coverage targets:
- Lines 41-42: ExecutionAnalyzer init failure
- Lines 103: Elevated sell-side friction alert
- Lines 123-125: friction computation exception
- Lines 151-152: explicit costs exception
- Lines 161-190: substitution bias analysis pipeline (the big gap)
- Lines 210, 214: alternative trade_df column names
- Lines 301-307: no-valid-latency fallback
"""

import os
import sys
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

# Must be set before module import to satisfy env.py
os.environ.setdefault("QLIB_WORKSPACE_DIR", os.getcwd())

from quantpits.scripts.deep_analysis.agents.execution_quality import ExecutionQualityAgent


@pytest.fixture
def exec_ctx(tmp_path):
    """Create a minimal context with trade log data."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "data").mkdir()
    (workspace / "data" / "order_history").mkdir()

    ctx = MagicMock()
    ctx.workspace_root = str(workspace)
    ctx.start_date = "2026-04-20"
    ctx.end_date = "2026-04-20"
    ctx.window_label = "test_window"
    ctx.trade_log_df = pd.DataFrame({
        "成交日期": [pd.Timestamp("2026-04-20"), pd.Timestamp("2026-04-20")],
        "交易类别": ["证券买入", "证券卖出"],
        "证券代码": ["000001", "000002"],
        "成交价格": [10.0, 20.0],
        "成交数量": [1000, 400],
        "成交金额": [10000.0, 8000.0],
    })
    return ctx


def test_execution_analyzer_init_failure(exec_ctx):
    """Lines 41-42: ExecutionAnalyzer init raises → return error findings."""
    agent = ExecutionQualityAgent()
    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer",
               side_effect=RuntimeError("init failed")):
        findings = agent.analyze(exec_ctx)

    assert any("ExecutionAnalyzer init failed" in f.title for f in findings.findings)
    assert findings.raw_metrics.get("error") == "init failed"


def test_friction_computation_exception(exec_ctx):
    """Lines 123-125: friction computation raises → error logged."""
    agent = ExecutionQualityAgent()

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.side_effect = RuntimeError("friction fail")
        ea_inst.analyze_explicit_costs.return_value = {}
        ea_inst.analyze_order_discrepancies.return_value = {}
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    assert "friction_error" in findings.raw_metrics


def test_explicit_costs_exception(exec_ctx):
    """Lines 151-152: explicit costs analysis raises → error logged."""
    agent = ExecutionQualityAgent()

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.return_value = pd.DataFrame()
        ea_inst.analyze_explicit_costs.side_effect = RuntimeError("costs fail")
        ea_inst.analyze_order_discrepancies.return_value = {}
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    assert "cost_error" in findings.raw_metrics


def test_substitution_bias_pipeline(exec_ctx):
    """Lines 161-190: full substitution bias analysis pipeline.

    Creates order_history directory, mock analyze_order_discrepancies
    returning meaningful data, verify findings and recommendations.
    """
    agent = ExecutionQualityAgent()

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.return_value = pd.DataFrame()
        ea_inst.analyze_explicit_costs.return_value = {}
        ea_inst.analyze_order_discrepancies.return_value = {
            "theoretical_substitute_bias_impact": 0.03,
            "realized_substitute_bias_impact": 0.04,  # > 0.02 triggers warning
            "total_missed_count": 5,
            "total_substitute_count": 5,
        }
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    # Check substitution bias in raw_metrics
    sub = findings.raw_metrics.get("substitution_bias", {})
    assert sub.get("theoretical_substitute_bias_impact") == 0.03
    assert sub.get("realized_substitute_bias_impact") == 0.04

    # Should have info finding
    info_findings = [f for f in findings.findings if "Substitution bias" in f.title]
    assert len(info_findings) == 1

    # Should have warning finding (realized > 0.02)
    warn_findings = [f for f in findings.findings if "Significant substitution bias" in f.title]
    assert len(warn_findings) == 1

    # Should have recommendation
    assert any("substitution bias" in r.lower() for r in findings.recommendations)


def test_substitution_bias_below_threshold(exec_ctx):
    """Line 179: realized_bias <= 0.02 → no warning generated."""
    agent = ExecutionQualityAgent()

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.return_value = pd.DataFrame()
        ea_inst.analyze_explicit_costs.return_value = {}
        ea_inst.analyze_order_discrepancies.return_value = {
            "theoretical_substitute_bias_impact": 0.01,
            "realized_substitute_bias_impact": 0.01,
            "total_missed_count": 1,
            "total_substitute_count": 1,
        }
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    warn_findings = [f for f in findings.findings if "Significant substitution bias" in f.title]
    assert len(warn_findings) == 0


def test_substitution_bias_no_order_history_dir(exec_ctx):
    """Lines 162: order_history dirs don't exist → substitution bias skipped."""
    # Remove the order_history dir
    import shutil
    oh_dir = os.path.join(exec_ctx.workspace_root, "data", "order_history")
    shutil.rmtree(oh_dir)

    agent = ExecutionQualityAgent()

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.return_value = pd.DataFrame()
        ea_inst.analyze_explicit_costs.return_value = {}
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    assert "substitution_bias" not in findings.raw_metrics


def test_substitution_bias_exception(exec_ctx):
    """Line 191-192: substitution bias pipeline raises → error captured."""
    agent = ExecutionQualityAgent()

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.return_value = pd.DataFrame()
        ea_inst.analyze_explicit_costs.return_value = {}
        ea_inst.analyze_order_discrepancies.side_effect = RuntimeError("bias fail")
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    assert "substitution_bias_error" in findings.raw_metrics


def test_execution_timing_missing_order_log_fallback(exec_ctx):
    """Lines 301-307: no valid latency data → fallback message."""
    agent = ExecutionQualityAgent()

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.return_value = pd.DataFrame()
        ea_inst.analyze_explicit_costs.return_value = {}
        ea_inst.analyze_order_discrepancies.return_value = {}
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    # Should have timing analysis finding
    timing = [f for f in findings.findings if "Execution timing analysis" in f.title]
    assert len(timing) > 0
    assert "pending granular intraday timestamp data" in timing[0].detail


def test_elevated_sell_friction_alert(exec_ctx):
    """Line 103: sell_total > 0.003 → warning generated."""
    agent = ExecutionQualityAgent()

    friction_df = pd.DataFrame({
        "交易类别": ["证券买入", "证券卖出"],
        "成交金额": [10000.0, 8000.0],
        "Delay_Cost": [0.001, 0.005],
        "Exec_Slippage": [0.001, 0.010],
        "ADV_Participation_Rate": [0.002, 0.001],
    })

    with patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer") as mock_ea:
        ea_inst = MagicMock()
        ea_inst.calculate_slippage_and_delay.return_value = friction_df
        ea_inst.analyze_explicit_costs.return_value = {}
        ea_inst.analyze_order_discrepancies.return_value = {}
        mock_ea.return_value = ea_inst

        with patch("quantpits.scripts.analysis.utils.init_qlib"):
            findings = agent.analyze(exec_ctx)

    warn = [f for f in findings.findings if "sell-side friction" in f.title.lower()]
    assert len(warn) == 1
