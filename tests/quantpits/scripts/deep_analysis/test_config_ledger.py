import pytest
import os
import json
import yaml
from quantpits.scripts.deep_analysis.config_ledger import (
    _parse_workflow_config, snapshot_configs, diff_snapshots
)

def test_parse_workflow_config(tmp_path):
    cfg_path = tmp_path / "workflow_config_test_Alpha158.yaml"
    cfg = {
        'task': {
            'model': {
                'class': 'TestModel',
                'kwargs': {'lr': 0.01, 'n_estimators': 100}
            }
        },
        'data_handler_config': {'label': ['TestLabel']}
    }
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f)
        
    parsed = _parse_workflow_config(str(cfg_path))
    assert parsed['model_class'] == 'TestModel'
    assert parsed['lr'] == 0.01
    assert parsed['n_estimators'] == 100
    assert parsed['model_key'] == 'test_Alpha158'
    assert parsed['feature_set'] == 'Alpha158'

def test_diff_snapshots():
    old = {
        'hyperparams': {
            'm1': {'lr': 0.01, 'depth': 6}
        },
        'ensemble_config': {'default_combo': 'c1'}
    }
    new = {
        'hyperparams': {
            'm1': {'lr': 0.02, 'depth': 7},
            'm2': {'lr': 0.1}
        },
        'ensemble_config': {'default_combo': 'c2'}
    }
    
    changes = diff_snapshots(old, new)
    
    # lr changed (LearningDynamics)
    hp_change = next(c for c in changes if c['key'] == 'm1.lr')
    assert hp_change['old'] == 0.01
    assert hp_change['new'] == 0.02
    assert hp_change['impact_domain'] == 'Hyperparameter'
    assert hp_change['semantic_label'] == 'LearningDynamics'
    
    # depth changed (CapacityAdjustment)
    depth_change = next(c for c in changes if c['key'] == 'm1.depth')
    assert depth_change['impact_domain'] == 'Architecture'
    assert depth_change['semantic_label'] == 'CapacityAdjustment'
    
    # m2 added (NewModel)
    m2_added = next(c for c in changes if c['key'] == 'm2' and c['change'] == 'added')
    assert m2_added['new']['lr'] == 0.1
    assert m2_added['impact_domain'] == 'Architecture'
    assert m2_added['semantic_label'] == 'NewModel'
    
    # ensemble switch (DefaultComboSwitch)
    ens_switch = next(c for c in changes if c['type'] == 'ensemble_switch')
    assert ens_switch['old'] == 'c1'
    assert ens_switch['new'] == 'c2'
    assert ens_switch['impact_domain'] == 'Ensemble'
    assert ens_switch['semantic_label'] == 'DefaultComboSwitch'

def test_annotate_with_llm_context():
    from quantpits.scripts.deep_analysis.config_ledger import annotate_with_llm_context
    changes = [{'type': 'hyperparam', 'key': 'm1.lr', 'old': 0.01, 'new': 0.02}]
    reason = "Improving convergence speed"
    annotated = annotate_with_llm_context(changes, reason, action_item_id="ai-001", critic_score=0.95)
    
    assert annotated[0]['change_reason'] == reason
    assert annotated[0]['action_item_id'] == "ai-001"
    assert annotated[0]['critic_score'] == 0.95
    assert 'annotated_at' in annotated[0]

def test_snapshot_configs(mock_workspace):
    snapshot = snapshot_configs(mock_workspace, snapshot_date="2026-04-20")
    assert snapshot['snapshot_date'] == "2026-04-20"
    assert 'alstm_Alpha158' in snapshot['hyperparams']
    assert snapshot['hyperparams']['alstm_Alpha158']['model_class'] == 'ALSTMModel'


# -------------------------------------------------------------------
# Coverage gap tests
# -------------------------------------------------------------------

def test_snapshot_configs_invalid_json_ensemble(tmp_path):
    """Lines 102-103: invalid ensemble_config.json → except: pass (no crash)."""
    from quantpits.scripts.deep_analysis.config_ledger import snapshot_configs
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True)
    (ws / "config" / "model_registry.yaml").write_text("models: {}\n")
    (ws / "config" / "ensemble_config.json").write_text("NOT JSON {{{")
    snapshot = snapshot_configs(str(ws))
    assert snapshot['snapshot_date'] is not None

def test_snapshot_configs_invalid_yaml_strategy(tmp_path):
    """Lines 108-112: invalid strategy_config.yaml → except: pass (no crash)."""
    from quantpits.scripts.deep_analysis.config_ledger import snapshot_configs
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True)
    (ws / "config" / "model_registry.yaml").write_text("models: {}\n")
    (ws / "config" / "strategy_config.yaml").write_text(": bad yaml: :")
    snapshot = snapshot_configs(str(ws))
    assert snapshot['snapshot_date'] is not None

def test_load_previous_snapshot_no_dir(tmp_path):
    """Line 135: no config_history dir → None."""
    from quantpits.scripts.deep_analysis.config_ledger import load_previous_snapshot
    result = load_previous_snapshot(str(tmp_path))
    assert result is None

def test_load_previous_snapshot_no_files(tmp_path):
    """Line 139: empty history dir → None."""
    from quantpits.scripts.deep_analysis.config_ledger import load_previous_snapshot
    (tmp_path / "data" / "config_history").mkdir(parents=True)
    result = load_previous_snapshot(str(tmp_path))
    assert result is None

def test_load_previous_snapshot_invalid_json(tmp_path):
    """Lines 150-151: invalid snapshot JSON → except: pass → None."""
    from quantpits.scripts.deep_analysis.config_ledger import load_previous_snapshot
    hist_dir = tmp_path / "data" / "config_history"
    hist_dir.mkdir(parents=True)
    (hist_dir / "config_snapshot_2026-01-01.json").write_text("BAD JSON {{{")
    result = load_previous_snapshot(str(tmp_path))
    assert result is None

def test_diff_snapshots_semantic_label(tmp_path):
    """Lines 204-206: d_feat/label_formula → FeatureExpansion/FeatureSelection."""
    from quantpits.scripts.deep_analysis.config_ledger import diff_snapshots
    prev = {'hyperparams': {'m1': {'lr': 0.01, 'd_feat': 10}}}
    curr = {'hyperparams': {'m1': {'lr': 0.01, 'd_feat': 20}}}
    result = diff_snapshots(prev, curr)
    assert len(result) >= 1

def test_generate_changelog(tmp_path):
    from quantpits.scripts.deep_analysis.config_ledger import (
        generate_changelog, save_snapshot, snapshot_configs
    )
    
    ws = tmp_path / "ws"
    config_dir = ws / "config"
    config_dir.mkdir(parents=True)
    data_dir = ws / "data"
    data_dir.mkdir(parents=True)
    
    # Create simple config snapshots
    (config_dir / "workflow_config_m1_Alpha158.yaml").write_text("task:\n  model:\n    class: ALSTMModel\n    kwargs:\n      lr: 0.01")
    
    # Save a first snapshot
    snap1 = snapshot_configs(str(ws), snapshot_date="2026-01-01")
    save_snapshot(str(ws), snap1)
    
    # Modify config for a second snapshot
    (config_dir / "workflow_config_m1_Alpha158.yaml").write_text("task:\n  model:\n    class: ALSTMModel\n    kwargs:\n      lr: 0.02")
    
    # Add training history
    train_hist = data_dir / "training_history.jsonl"
    train_entry = {
        "model_name": "m1_Alpha158",
        "trained_at": "2026-01-02 12:00:00",
        "actual_epochs": 10,
        "IC_Mean": 0.05,
        "ICIR": 1.2
    }
    train_hist.write_text(json.dumps(train_entry) + "\n")
    
    # Add action item history
    action_hist = data_dir / "action_item_history.jsonl"
    action_entry = {
        "action_type": "Tuning",
        "target": "m1_Alpha158",
        "reason": "Optimize learning rate for ALSTMModel",
        "confidence": 0.85,
        "risk_level": "low"
    }
    action_hist.write_text(json.dumps(action_entry) + "\n")
    
    out_md = ws / "CHANGELOG.md"
    
    text = generate_changelog(str(ws), output_path=str(out_md), title="My Custom Changelog")
    
    assert "My Custom Changelog" in text
    assert "Configuration Changes" in text
    assert "m1_Alpha158" in text
    assert "Recent Training Results" in text
    assert "Recent Action Items" in text
    assert out_md.exists()

def test_config_ledger_cli(tmp_path):
    import runpy
    import sys
    from unittest.mock import patch
    
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True)
    
    test_argv = [
        "config_ledger.py",
        "--snapshot",
        "--changelog",
        "--workspace", str(ws)
    ]
    
    with patch.object(sys, 'argv', test_argv):
        runpy.run_module("quantpits.scripts.deep_analysis.config_ledger", run_name="__main__")


