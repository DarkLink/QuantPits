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
    
    Each record: {
        type: 'hyperparam'|'ensemble'|'strategy'|'ensemble_switch', 
        key: ..., 
        old: ..., 
        new: ...,
        impact_domain: ...,
        semantic_label: ...
    }
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
            changes.append({
                'type': 'hyperparam', 'key': model, 'change': 'added', 
                'new': new_m, 'impact_domain': 'Architecture', 'semantic_label': 'NewModel'
            })
        elif model not in new_hp:
            changes.append({
                'type': 'hyperparam', 'key': model, 'change': 'removed', 
                'old': old_m, 'impact_domain': 'Architecture', 'semantic_label': 'ModelDecommission'
            })
        else:
            # Compare individual params (skip _path)
            skip = {'_path', '_error'}
            for k in set(list(old_m.keys()) + list(new_m.keys())) - skip:
                ov = old_m.get(k)
                nv = new_m.get(k)
                if str(ov) != str(nv):
                    # Heuristics for semantic labels
                    domain = 'Hyperparameter'
                    label = 'Tuning'
                    if k in ['d_model', 'hidden_size', 'num_layers', 'depth', 'max_depth']:
                        domain = 'Architecture'
                        label = 'CapacityAdjustment'
                    elif k in ['lr', 'learning_rate', 'optimizer', 'n_epochs', 'iterations', 'early_stop']:
                        label = 'LearningDynamics'
                    elif k in ['dropout', 'l2_leaf_reg', 'num_leaves']:
                        label = 'Regularization'
                    elif k in ['d_feat', 'feature_set', 'label_formula']:
                        domain = 'Dataset'
                        label = 'FeatureExpansion' if nv and (not ov or len(str(nv)) > len(str(ov))) else 'FeatureSelection'

                    changes.append({
                        'type': 'hyperparam', 'key': f'{model}.{k}',
                        'old': ov, 'new': nv,
                        'impact_domain': domain,
                        'semantic_label': label
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
                    label = 'EnsembleSubstitution'
                    if not oc: label = 'NewEnsemble'
                    elif not nc: label = 'EnsembleDecommission'
                    
                    changes.append({
                        'type': 'ensemble', 'key': combo_name,
                        'old': oc, 'new': nc,
                        'impact_domain': 'Ensemble',
                        'semantic_label': label
                    })
        # Detect default combo switch
        old_default = old_ec.get('default_combo')
        new_default = new_ec.get('default_combo')
        if old_default != new_default and (old_default or new_default):
            changes.append({
                'type': 'ensemble_switch', 'key': 'default_combo',
                'old': old_default, 'new': new_default,
                'impact_domain': 'Ensemble',
                'semantic_label': 'DefaultComboSwitch'
            })

    # 3. Strategy config changes
    old_sc_json = json.dumps(old.get('strategy_config', {}), sort_keys=True)
    new_sc_json = json.dumps(new.get('strategy_config', {}), sort_keys=True)
    if old_sc_json != new_sc_json:
        changes.append({
            'type': 'strategy',
            'old': old.get('strategy_config'),
            'new': new.get('strategy_config'),
            'impact_domain': 'Strategy',
            'semantic_label': 'RiskLimitAdjustment' # Default label
        })

    return changes


def annotate_with_llm_context(
    change_records: List[dict],
    reason: str,
    action_item_id: str = None,
    critic_score: float = 0.0
) -> List[dict]:
    """
    为一批 change records 附加 LLM 操作来源信息。
    在 LLM Critic 执行 promote 时调用。

    Args:
        change_records:  diff_snapshots() 返回的变更列表
        reason:          本次变更的文字理由
        action_item_id:  触发此次变更的 ActionItem ID（LLM 操作时填写）
        critic_score:    Critic Agent 的置信度/质量评分
    """
    for record in change_records:
        record['change_reason'] = reason
        record['action_item_id'] = action_item_id
        record['critic_score'] = critic_score
        record['annotated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return change_records


# ------------------------------------------------------------------
# Changelog generator
# ------------------------------------------------------------------

def generate_changelog(
    workspace_root: str,
    output_path: str = "",
    title: str = "",
) -> str:
    """Generate a human-readable Markdown changelog from config diffs and training data.

    Reads the two most recent config snapshots, diffs them, enriches with
    training history and action item data, and writes a formatted changelog.

    Args:
        workspace_root: Path to the production workspace.
        output_path: If non-empty, write the changelog to this file.
        title: Optional title override (default: "Config Changelog — {date}").

    Returns:
        The changelog text as a string.
    """
    import json as _json

    today = datetime.now().strftime('%Y-%m-%d')
    if not title:
        title = f"Config Changelog — {today}"

    lines = [f"# {title}", ""]

    # 1. Config diff
    current = snapshot_configs(workspace_root)
    previous = load_previous_snapshot(workspace_root)
    changes = diff_snapshots(previous, current) if previous else []

    if changes:
        lines.append("## Configuration Changes")
        lines.append("")

        # Group by model
        by_model: dict = {}
        for c in changes:
            model = c.get('key', 'unknown')
            by_model.setdefault(model, []).append(c)

        for model, model_changes in sorted(by_model.items()):
            lines.append(f"### {model}")
            lines.append("")
            lines.append("| Param | Old | New | Domain |")
            lines.append("|-------|-----|-----|--------|")
            for c in model_changes:
                # key format: "model.param" or just "model" (for added/removed)
                key = c.get('key', '?')
                param = key.split('.', 1)[1] if '.' in key else c.get('change', '?')
                old = c.get('old', '-')
                new = c.get('new', '-')
                domain = c.get('impact_domain', c.get('semantic_label', '-'))
                lines.append(f"| {param} | {old} | {new} | {domain} |")
            lines.append("")

    # 2. Training results (last 10 entries, or matching changed models)
    history_path = os.path.join(workspace_root, 'data', 'training_history.jsonl')
    recent_train = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                all_entries = [_json.loads(line) for line in f if line.strip()]
            # Last 10, newest first
            recent_train = all_entries[-10:][::-1]
        except Exception:
            pass

    if recent_train:
        lines.append("## Recent Training Results")
        lines.append("")
        lines.append("| Model | Date | Epochs | IC | ICIR | Status |")
        lines.append("|-------|------|--------|-----|------|--------|")
        for t in recent_train:
            name = t.get('model_name', '?')
            date = t.get('trained_at', '?')[:10]
            epochs = t.get('actual_epochs', '?')
            ic = t.get('IC_Mean', '?')
            icir = t.get('ICIR', '?')
            status = "✅" if (isinstance(ic, (int, float)) and ic > 0.03) else "⚠️"
            if isinstance(ic, float):
                ic = f"{ic:.4f}"
            if isinstance(icir, float):
                icir = f"{icir:.4f}"
            lines.append(f"| {name} | {date} | {epochs} | {ic} | {icir} | {status} |")
        lines.append("")

    # 3. Action items (last 5)
    actions_path = os.path.join(workspace_root, 'data', 'action_item_history.jsonl')
    recent_actions = []
    if os.path.exists(actions_path):
        try:
            with open(actions_path, 'r') as f:
                all_actions = [_json.loads(line) for line in f if line.strip()]
            for a in all_actions[-5:][::-1]:
                recent_actions.append(
                    f"- [{a.get('action_type', '?')}] **{a.get('target', '?')}**: "
                    f"{a.get('reason', '')[:120]} "
                    f"(conf={a.get('confidence', '?')}, risk={a.get('risk_level', '?')})"
                )
        except Exception:
            pass

    if recent_actions:
        lines.append("## Recent Action Items")
        lines.append("")
        lines.extend(recent_actions)
        lines.append("")

    text = "\n".join(lines) + "\n"

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

    return text


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from quantpits.utils import env

    parser = argparse.ArgumentParser(
        description="Config ledger: snapshot, diff, and changelog generation."
    )
    parser.add_argument('--snapshot', action='store_true',
                        help='Create a config snapshot now.')
    parser.add_argument('--changelog', action='store_true',
                        help='Generate a Markdown changelog from recent diffs.')
    parser.add_argument('--output', type=str, default='',
                        help='Changelog output path (default: stdout only).')
    parser.add_argument('--workspace', type=str, default='',
                        help='Workspace path (default: QLIB_WORKSPACE_DIR).')

    args = parser.parse_args()
    root = args.workspace or env.ROOT_DIR

    if args.snapshot:
        snap = snapshot_configs(root)
        path = save_snapshot(root, snap)
        print(f"Snapshot saved to {path}")

    if args.changelog:
        out = args.output or os.path.join(root, f"CHANGELOG_{datetime.now().strftime('%Y-%m-%d')}.md")
        text = generate_changelog(root, output_path=out)
        print(text)
        if args.output:
            print(f"Changelog written to {out}")
