import os
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock
from quantpits.scripts.deep_analysis.base_agent import AnalysisContext
from quantpits.scripts.deep_analysis.agents.training_health import TrainingHealthAgent
from quantpits.scripts.deep_analysis.training_context import TrainingModeContext

def test_training_health_agent(tmp_path):
    # Setup mock workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "output").mkdir()

    # Create mock metrics CSVs
    dates = pd.date_range("2026-05-01", periods=100, freq='D')
    
    # 20-day metrics
    df_20 = pd.DataFrame({
        "Date": dates,
        "Idiosyncratic_Alpha": [-0.01 for _ in range(100)], # alpha decay negative values
        "Win_Rate": [0.45 for _ in range(100)],
        "Payoff_Ratio": [1.1 for _ in range(100)]
    })
    df_20.to_csv(workspace / "output" / "rolling_metrics_20.csv", index=False)

    # 60-day metrics (simulate Exec_Slippage_Mean dropping/z-score failure and low size exposure)
    df_60 = pd.DataFrame({
        "Date": dates,
        "Exec_Slippage_Mean": [0.01 for _ in range(99)] + [-0.05], # huge negative drop today
        "Delay_Cost_Mean": [0.01 for _ in range(100)],
        "Idiosyncratic_Alpha": [0.005 for _ in range(100)],
        "Exposure_Liquidity": [0.05 for _ in range(99)] + [-0.10], # extremely low micro-cap exposure
    })
    df_60.to_csv(workspace / "output" / "rolling_metrics_60.csv", index=False)

    # Setup mock TrainingModeContext
    tc = TrainingModeContext(
        workspace_root=str(workspace),
        anchor_date="2026-07-03",
        all_model_keys={"lstm_Alpha158@static": "r1"},
        models_by_name={"lstm_Alpha158": {"static": "r1"}},
        available_modes=["static"],
        rolling_config={"test_step": "3M"},
        rolling_states={
            "rolling": {
                "anchor_date": "2026-03-01", # >90 days ago
                "total_windows": 10,
                "current_window_idx": 9
            }
        },
        cpcv_config={"n_groups": 10}
    )

    # Create AnalysisContext
    ctx = AnalysisContext(
        start_date="2026-05-01",
        end_date="2026-07-03",
        window_label="1y",
        workspace_root=str(workspace),
        training_context=tc
    )

    agent = TrainingHealthAgent()
    results = agent.analyze(ctx)

    # Verify findings
    assert results.agent_name == "Training Health"
    
    finding_types = [f.title for f in results.findings]
    print("DEBUG findings:", finding_types)

    # Check for Mode Coverage finding
    assert any("Incomplete training modes" in t for t in finding_types)

    # Check for Rolling Staleness warning
    assert any("Rolling staleness" in t for t in finding_types)

    # Check for Execution Friction critical warning
    assert any("执行摩擦崩盘" in t for t in finding_types)

    # Check for Alpha Decay warning
    assert any("Alpha Decay" in t for t in finding_types)

    # Check for Barra size exposure drift
    assert any("极端微盘漂移" in t for t in finding_types)
