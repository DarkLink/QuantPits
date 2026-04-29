import pytest
import pandas as pd
import os
import shutil
import json
import yaml
from datetime import datetime, timedelta

@pytest.fixture
def mock_workspace(tmp_path):
    """Create a mock workspace structure with minimal data."""
    workspace = tmp_path / "mock_workspace"
    workspace.mkdir()
    
    # Create subdirs
    (workspace / "data").mkdir()
    (workspace / "data" / "order_history").mkdir()
    (workspace / "output").mkdir()
    (workspace / "archive").mkdir()
    (workspace / "config").mkdir()
    
    # Create sample dataframes
    dates = pd.date_range(start="2026-01-01", periods=100, freq='D')
    
    daily_amount = pd.DataFrame({
        '成交日期': dates,
        '当日持仓市值': [1000000 + i*1000 for i in range(100)],
        '当日可用资金': [50000 for _ in range(100)],
        'CSI300': [3000 + i*5 for i in range(100)]  # Add CSI300 data
    })
    daily_amount.to_csv(workspace / "data" / "daily_amount_log_full.csv", index=False)
    
    trade_log = pd.DataFrame({
        '成交日期': dates[::5],
        '证券代码': ['600000.SH' for _ in range(20)],
        '买卖标志': ['买入' if i % 2 == 0 else '卖出' for i in range(20)],
        '交易类别': ['上海A股普通股票竞价买入' if i % 2 == 0 else '上海A股普通股票竞价卖出' for i in range(20)],
        '成交价格': [10.0 for _ in range(20)],
        '成交数量': [1000 for _ in range(20)],
        '成交金额': [10000.0 for _ in range(20)],
        '手续费': [5.0 for _ in range(20)],
        '发生金额': [-10005.0 if i % 2 == 0 else 9995.0 for i in range(20)]
    })
    trade_log.to_csv(workspace / "data" / "trade_log_full.csv", index=False)
    
    holding_log = pd.DataFrame({
        '成交日期': dates[::2],
        '证券代码': ['600000.SH' for _ in range(50)],
        '收盘价值': [10000.0 for _ in range(50)],
        '浮盈收益率': [0.05 for _ in range(50)]
    })
    holding_log.to_csv(workspace / "data" / "holding_log_full.csv", index=False)
    
    trade_class = pd.DataFrame({
        'trade_date': dates[::5].strftime('%Y-%m-%d'),
        'instrument': ['600000.SH' for _ in range(20)],
        'trade_class': ['S' if i % 3 == 0 else ('A' if i % 3 == 1 else 'M') for i in range(20)]
    })
    trade_class.to_csv(workspace / "data" / "trade_classification.csv", index=False)
    
    # Create sample config
    workflow_cfg = {
        'task': {
            'model': {
                'class': 'ALSTMModel',
                'kwargs': {'d_model': 64, 'lr': 0.001}
            }
        },
        'data_handler_config': {'label': ['Ref(RET, -1)']}
    }
    with open(workspace / "config" / "workflow_config_alstm_Alpha158.yaml", 'w') as f:
        yaml.dump(workflow_cfg, f)
        
    # --- Create more mock files for agent coverage ---
    
    # 1. Model Performance JSONs
    perf_data = {
        'ic': 0.05, 'rank_ic': 0.06, 
        'all': {'return': 0.1, 'IC_Mean': 0.05, 'ICIR': 0.5, 'record_id': 'abc12345'},
        'convergence': {
            'early_stopped': True,
            'epochs_done': 20,
            'configured_epochs': 200,
            'duration_s': 150
        }
    }
    for d in ["2026-01-15", "2026-02-15", "2026-03-15"]:
        with open(workspace / "output" / f"model_performance_{d}.json", 'w') as f:
            json.dump(perf_data, f)

    # 2. Ensemble Fusion Configs
    ensemble_dir = workspace / "output" / "ensemble"
    ensemble_dir.mkdir(exist_ok=True)
    fusion_cfg = {'selected_models': ['model_a', 'model_b']}
    for d in ["2026-01-20", "2026-02-20"]:
        with open(ensemble_dir / f"ensemble_fusion_config_combo1_{d}.json", 'w') as f:
            json.dump(fusion_cfg, f)

    # 3. Combo Comparison CSVs
    combo_comp = pd.DataFrame({
        'combo': ['combo1', 'combo1'],
        'total_return': [0.05, 0.12],
        'calmar_ratio': [2.5, 2.8]
    })
    combo_comp.to_csv(ensemble_dir / "combo_comparison_2026-03-20.csv", index=False)

    # 4. Leaderboard CSVs
    leaderboard = pd.DataFrame({
        'name': ['model_a', 'model_b', 'Ensemble'],
        'annualized_excess': [0.15, 0.10, 0.18]
    })
    leaderboard.to_csv(ensemble_dir / "leaderboard_combo1_2026-03-20.csv", index=False)

    # 5. Correlation Matrix
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=['model_a', 'model_b'], index=['model_a', 'model_b'])
    corr.to_csv(ensemble_dir / "correlation_matrix_2026-03-20.csv")

    # 6. Suggestions (for PredictionAuditAgent)
    suggestions = pd.DataFrame({
        'instrument': ['600000.SH', '000001.SZ'],
        'score': [0.9, 0.8]
    })
    suggestions.to_csv(workspace / "output" / "buy_suggestion_2026-03-20.csv", index=False)
    suggestions.to_csv(workspace / "output" / "sell_suggestion_2026-03-20.csv", index=False)
    
    # 7. Model Opinions
    opinions = {'opinions': [{'code': '600000.SH', 'prediction': 0.05, 'rank': 1}]}
    opinions_csv = pd.DataFrame({
        'instrument': ['600000.SH', '000001.SZ'],
        'model_a': ['BUY', 'SELL'],
        'model_b': ['BUY', 'BUY']
    })
    for d in ["2026-01-20", "2026-02-20", "2026-03-20"]:
        with open(workspace / "output" / f"model_opinions_{d}.json", 'w') as f:
            json.dump(opinions, f)
        opinions_csv.to_csv(workspace / "output" / f"model_opinions_{d}.csv", index=False)

    # 8. Training History JSONL
    history_line = {
        "model_name": "model_a", "record_id": "abc12345", 
        "trained_at": "2026-03-10", "duration_seconds": 150,
        "early_stopped": True, "actual_epochs": 20, "configured_epochs": 200
    }
    with open(workspace / "data" / "training_history.jsonl", 'w') as f:
        f.write(json.dumps(history_line) + "\n")

    # 9. Model Contribution JSONs
    contrib_data = {
        "combo": "combo1", "anchor_date": "2026-03-20",
        "contributions": {
            "model_a": {"loo_ic": 0.8, "full_ic": 1.0, "delta": 0.2},
            "model_b": {"loo_ic": 1.1, "full_ic": 1.0, "delta": -0.1}
        }
    }
    with open(ensemble_dir / "model_contribution_combo1_2026-03-20.json", 'w') as f:
        json.dump(contrib_data, f)

    # 10. Run Metadata (for OOS history)
    metadata = {
        "combo_name": "combo1", "run_date": "2026-03-20",
        "oos_metrics": {"oos_calmar": 2.5, "oos_excess_return": 0.1}
    }
    (ensemble_dir / "run1").mkdir()
    with open(ensemble_dir / "run1" / "run_metadata.json", 'w') as f:
        json.dump(metadata, f)
    metadata["run_date"] = "2026-03-25"
    metadata["oos_metrics"]["oos_calmar"] = 2.2
    (ensemble_dir / "run2").mkdir()
    with open(ensemble_dir / "run2" / "run_metadata.json", 'w') as f:
        json.dump(metadata, f)

    return str(workspace)

@pytest.fixture
def mock_analysis_context(mock_workspace):
    from quantpits.scripts.deep_analysis.base_agent import AnalysisContext
    import glob
    
    workspace_root = mock_workspace
    
    def _get_files(pattern):
        return sorted(glob.glob(os.path.join(workspace_root, pattern)))

    return AnalysisContext(
        start_date="2026-01-01",
        end_date="2026-03-31",
        window_label="test_window",
        workspace_root=workspace_root,
        daily_amount_df=pd.read_csv(os.path.join(workspace_root, "data/daily_amount_log_full.csv"), parse_dates=['成交日期']),
        trade_log_df=pd.read_csv(os.path.join(workspace_root, "data/trade_log_full.csv"), parse_dates=['成交日期']),
        holding_log_df=pd.read_csv(os.path.join(workspace_root, "data/holding_log_full.csv"), parse_dates=['成交日期']),
        trade_classification_df=pd.read_csv(os.path.join(workspace_root, "data/trade_classification.csv")),
        model_performance_files=_get_files("output/model_performance_*.json"),
        ensemble_fusion_config_files=_get_files("output/ensemble/ensemble_fusion_config_*.json"),
        combo_comparison_files=_get_files("output/ensemble/combo_comparison_*.csv"),
        leaderboard_files=_get_files("output/ensemble/leaderboard_*.csv"),
        correlation_matrix_files=_get_files("output/ensemble/correlation_matrix_*.csv"),
        buy_suggestion_files=_get_files("output/buy_suggestion_*.csv"),
        sell_suggestion_files=_get_files("output/sell_suggestion_*.csv"),
        model_opinions_files=_get_files("output/model_opinions_*.json"),
        model_contribution_files=_get_files("output/ensemble/model_contribution_*.json")
    )
