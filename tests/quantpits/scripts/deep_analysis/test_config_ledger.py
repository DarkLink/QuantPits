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
