"""
Configuration History Ledger for the MAS Deep Analysis System.

Snapshots workflow configs, ensemble configs, and strategy configs on each run.
Provides diff detection against previous snapshots to track hyperparameter and
ensemble composition changes over time.
"""

import os
import json
import glob
import yaml
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _parse_workflow_config(yaml_path: str) -> dict:
    """
    Parse a workflow_config_*.yaml and extract key hyperparameters.
    
    Returns a dict with model class, feature set, and kwargs.
    """
    try:
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return {'_error': str(e), '_path': yaml_path}

    result = {'_path': yaml_path}

    # Extract model info from task.model
    task = cfg.get('task', {})
    model_cfg = task.get('model', {})
    if model_cfg:
        result['model_class'] = model_cfg.get('class', 'Unknown')
        result['model_module'] = model_cfg.get('module_path', '')
        kwargs = model_cfg.get('kwargs', {})
        # Extract commonly-tuned hyperparameters
        for key in ['d_model', 'd_feat', 'hidden_size', 'num_layers', 'dropout',
                     'n_epochs', 'lr', 'batch_size', 'loss', 'optimizer',
                     'loss_function', 'iterations', 'learning_rate',
                     'depth', 'l2_leaf_reg', 'num_leaves', 'max_depth',
                     'n_estimators', 'early_stop', 'seed', 'GPU']:
            if key in kwargs:
                result[key] = kwargs[key]

    # Extract data handler info
    dh = cfg.get('data_handler_config', {})
    label_list = dh.get('label', [])
    if label_list:
        result['label_formula'] = label_list[0] if isinstance(label_list, list) else str(label_list)

    # Extract feature set name from filename
    basename = os.path.basename(yaml_path)
    # workflow_config_alstm_Alpha158.yaml -> alstm_Alpha158
    m = re.match(r'workflow_config_(.+)\.yaml', basename)
    if m:
        model_key = m.group(1)
        result['model_key'] = model_key
        # Try to extract feature set
        parts = model_key.split('_')
        for p in parts:
            if p.startswith('Alpha'):
                result['feature_set'] = p
                break

    return result


def snapshot_configs(workspace_root: str, snapshot_date: Optional[str] = None) -> dict:
    """
    Create a snapshot of all current configurations.
    
    Returns:
        dict with keys: snapshot_date, hyperparams, ensemble_config, strategy_config
    """
    if snapshot_date is None:
        snapshot_date = datetime.now().strftime('%Y-%m-%d')

    config_dir = os.path.join(workspace_root, 'config')
    snapshot = {
        'snapshot_date': snapshot_date,
        'hyperparams': {},
        'ensemble_config': {},
        'strategy_config': {},
    }

    # 1. Parse all workflow_config_*.yaml files
    yaml_pattern = os.path.join(config_dir, 'workflow_config_*.yaml')
    for yaml_path in sorted(glob.glob(yaml_pattern)):
        parsed = _parse_workflow_config(yaml_path)
        model_key = parsed.get('model_key', os.path.basename(yaml_path))
        snapshot['hyperparams'][model_key] = parsed

    # 2. Snapshot ensemble_config.json
    ensemble_path = os.path.join(config_dir, 'ensemble_config.json')
    if os.path.exists(ensemble_path):
        try:
            with open(ensemble_path, 'r') as f:
                snapshot['ensemble_config'] = json.load(f)
        except Exception:
            pass

    # 3. Snapshot strategy_config.yaml
    strategy_path = os.path.join(config_dir, 'strategy_config.yaml')
    if os.path.exists(strategy_path):
        try:
            with open(strategy_path, 'r') as f:
                snapshot['strategy_config'] = yaml.safe_load(f)
        except Exception:
            pass

    return snapshot


def save_snapshot(workspace_root: str, snapshot: dict) -> str:
    """Save a config snapshot to the history directory. Returns the saved path."""
    history_dir = os.path.join(workspace_root, 'data', 'config_history')
    os.makedirs(history_dir, exist_ok=True)

    date_str = snapshot.get('snapshot_date', datetime.now().strftime('%Y-%m-%d'))
    out_path = os.path.join(history_dir, f'config_snapshot_{date_str}.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

    return out_path


def load_previous_snapshot(workspace_root: str, before_date: Optional[str] = None) -> Optional[dict]:
    """Load the most recent snapshot before a given date."""
    history_dir = os.path.join(workspace_root, 'data', 'config_history')
    if not os.path.isdir(history_dir):
        return None

    files = sorted(glob.glob(os.path.join(history_dir, 'config_snapshot_*.json')))
    if not files:
        return None

    if before_date:
        files = [f for f in files if os.path.basename(f) < f'config_snapshot_{before_date}.json']

    if not files:
        return None

    try:
        with open(files[-1], 'r') as f:
            return json.load(f)
    except Exception:
        return None


def diff_snapshots(old: dict, new: dict) -> List[dict]:
    """
    Compare two config snapshots and return a list of change records.
    
    Each record: {type: 'hyperparam'|'ensemble'|'strategy', key: ..., old: ..., new: ...}
    """
    changes = []

    # 1. Hyperparameter changes
    old_hp = old.get('hyperparams', {})
    new_hp = new.get('hyperparams', {})
    all_models = set(list(old_hp.keys()) + list(new_hp.keys()))

    for model in sorted(all_models):
        old_m = old_hp.get(model, {})
        new_m = new_hp.get(model, {})
        if model not in old_hp:
            changes.append({'type': 'hyperparam', 'key': model, 'change': 'added', 'new': new_m})
        elif model not in new_hp:
            changes.append({'type': 'hyperparam', 'key': model, 'change': 'removed', 'old': old_m})
        else:
            # Compare individual params (skip _path)
            skip = {'_path', '_error'}
            for k in set(list(old_m.keys()) + list(new_m.keys())) - skip:
                ov = old_m.get(k)
                nv = new_m.get(k)
                if str(ov) != str(nv):
                    changes.append({
                        'type': 'hyperparam', 'key': f'{model}.{k}',
                        'old': ov, 'new': nv
                    })

    # 2. Ensemble config changes
    old_ec = old.get('ensemble_config', {})
    new_ec = new.get('ensemble_config', {})
    if json.dumps(old_ec, sort_keys=True) != json.dumps(new_ec, sort_keys=True):
        # Detect specific combo changes
        old_combos = old_ec.get('combo_groups', old_ec)
        new_combos = new_ec.get('combo_groups', new_ec)
        if isinstance(old_combos, dict) and isinstance(new_combos, dict):
            for combo_name in set(list(old_combos.keys()) + list(new_combos.keys())):
                oc = old_combos.get(combo_name)
                nc = new_combos.get(combo_name)
                if oc != nc:
                    changes.append({
                        'type': 'ensemble', 'key': combo_name,
                        'old': oc, 'new': nc
                    })
        # Detect default combo switch
        old_default = old_ec.get('default_combo')
        new_default = new_ec.get('default_combo')
        if old_default != new_default and (old_default or new_default):
            changes.append({
                'type': 'ensemble_switch', 'key': 'default_combo',
                'old': old_default, 'new': new_default
            })

    # 3. Strategy config changes
    old_sc = json.dumps(old.get('strategy_config', {}), sort_keys=True)
    new_sc = json.dumps(new.get('strategy_config', {}), sort_keys=True)
    if old_sc != new_sc:
        changes.append({
            'type': 'strategy',
            'old': old.get('strategy_config'),
            'new': new.get('strategy_config')
        })

    return changes
