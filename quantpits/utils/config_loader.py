import os
import json
import yaml
from pathlib import Path
from quantpits.utils.constants import MONTHS_PER_YEAR
from quantpits.utils.workspace import WorkspaceContext

def load_workspace_config(workspace_path):
    """
    Unified configuration loader for QuantPits workspaces.
    Merges model_config.json, prod_config.json, and strategy_config.yaml.
    
    Returns a unified dict with all necessary parameters.
    """
    workspace_path = Path(workspace_path)
    config_dir = workspace_path / "config"
    
    # Files
    model_cfg_path = config_dir / "model_config.json"
    prod_cfg_path = config_dir / "prod_config.json"
    strat_cfg_path = config_dir / "strategy_config.yaml"
    
    config = {}
    
    # 1. Load Model Config (Base environment properties)
    if model_cfg_path.exists():
        with open(model_cfg_path, 'r') as f:
            config.update(json.load(f))
            
    # 2. Load Strategy Config (Single Source of Truth for strategy params)
    if strat_cfg_path.exists():
        with open(strat_cfg_path, 'r') as f:
            strat_data = yaml.safe_load(f)
            if strat_data:
                config['strategy'] = strat_data.get('strategy', {})
                config['backtest'] = strat_data.get('backtest', {})
                
                # Promote core strategy params to top-level for convenience/compatibility
                strat_params = config['strategy'].get('params', {})
                config['topk'] = strat_params.get('topk', config.get('TopK'))
                config['n_drop'] = strat_params.get('n_drop', config.get('DropN'))
                config['buy_suggestion_factor'] = strat_params.get('buy_suggestion_factor', config.get('buy_suggestion_factor'))
                
                # Compatibility mapping (Upper case versions)
                config['TopK'] = config['topk']
                config['DropN'] = config['n_drop']
                
    # 3. Load Prod Config (Current state - handles cash/holding)
    if prod_cfg_path.exists():
        with open(prod_cfg_path, 'r') as f:
            prod_data = json.load(f)
            # We only want State fields from prod_config, others should come from model/strategy
            state_fields = [
                'current_date', 'last_processed_date', 'initial_cash', 
                'current_full_cash', 'initial_holding', 'current_cash', 
                'current_holding', 'model', 'experiment_name',
                'current_train_record_id', 'current_pred_record_id'
            ]
            for field in state_fields:
                if field in prod_data:
                    config[field] = prod_data[field]
                    
    # Sanity checks / Cross-file consistency (Optional but recommended)
    # If market/benchmark exist in both, we prefer model_config but can log warnings if they mismatch
    
    return config

def load_workspace_config_artifacts(workspace_path):
    """Validate workspace configs and return structured validation artifacts.

    This helper is read-only and does not change the legacy
    ``load_workspace_config`` behavior.
    """
    from quantpits.config_contracts.workspace import validate_workspace

    ctx = WorkspaceContext.from_root(workspace_path)
    return validate_workspace(ctx)

def load_workspace_config_with_metadata(workspace_path):
    """Load the legacy merged config plus validation/fingerprint metadata."""
    config = load_workspace_config(workspace_path)
    result = load_workspace_config_artifacts(workspace_path)
    metadata = {
        "workspace": result.workspace.as_posix(),
        "ok": result.ok,
        "artifacts": [
            artifact.to_public_dict(workspace=result.workspace)
            for artifact in result.artifacts
        ],
        "fingerprints": {
            artifact.name: artifact.fingerprint
            for artifact in result.artifacts
            if artifact.fingerprint
        },
        "warnings": [message.to_dict() for message in result.warnings],
        "errors": [message.to_dict() for message in result.errors],
    }
    return config, metadata

def _validate_cpcv_params(train_years, n_groups, purge_steps, embargo_steps, freq):
    """Fail-Fast: reject catastrophically bad CPCV params at config load time.

    Even though the train domain is fixed (train_years), bad purge/embargo
    combinations can still destroy right training segments. E.g.:
      train_years=5, n_groups=20 → group≈13w, purge=10+embargo=10 → gap=20w > group.

    If purge+embargo >= 80% of estimated group size, training segments will be
    destroyed in most folds — reject immediately.
    """
    periods_per_year = 52 if freq == 'week' else 252
    estimated_group = (train_years * periods_per_year) / n_groups
    max_gap = purge_steps + embargo_steps

    if max_gap >= estimated_group * 0.8:
        raise ValueError(
            f"CPCV rolling: purge+embargo ({max_gap}) >= 80% of "
            f"estimated group size ({estimated_group:.0f} {freq}s). "
            f"Training segments will be destroyed in most folds. "
            f"Either:\n"
            f"  - Increase train_years (currently {train_years})\n"
            f"  - Reduce n_groups (currently {n_groups})\n"
            f"  - Reduce purge_steps ({purge_steps}) + embargo_steps ({embargo_steps})"
        )

    if max_gap >= estimated_group * 0.5:
        import warnings
        warnings.warn(
            f"CPCV rolling: purge+embargo ({max_gap}) is "
            f"{max_gap / estimated_group:.0%} of estimated group size "
            f"({estimated_group:.0f} {freq}s). "
            f"Late-fold right training segments may be thin. "
            f"Consider increasing train_years or reducing purge/embargo."
        )


def load_rolling_config(workspace_path):
    """
    Load rolling training configuration from config/rolling_config.yaml.

    Returns:
        dict with keys:
            rolling_start (str), train_years (int), valid_years (int),
            test_step (str), test_step_months (int),
            training_method (str: 'slide' or 'cpcv'),
            and CPCV params (when training_method='cpcv').
        None if file doesn't exist.
    """
    from dateutil.relativedelta import relativedelta

    workspace_path = Path(workspace_path)
    rolling_cfg_path = workspace_path / "config" / "rolling_config.yaml"

    if not rolling_cfg_path.exists():
        return None

    with open(rolling_cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        return None

    # Parse test_step: "3M" -> 3 months, "1Y" -> {MONTHS_PER_YEAR} months
    step_str = str(cfg.get('test_step', '3M')).strip().upper()
    if step_str.endswith('M'):
        step_months = int(step_str[:-1])
    elif step_str.endswith('Y'):
        step_months = int(step_str[:-1]) * MONTHS_PER_YEAR
    else:
        raise ValueError(f"Invalid test_step format: '{cfg['test_step']}'. "
                         f"Expected integer months (e.g. 3M) or years (e.g. 1Y)")

    result = {
        'rolling_start': str(cfg['rolling_start']),
        'train_years': int(cfg['train_years']),
        'valid_years': int(cfg['valid_years']),
        'test_step': step_str,
        'test_step_months': step_months,
        'training_method': str(cfg.get('training_method', 'slide')).lower(),
    }

    # Parse CPCV params when training_method is 'cpcv'
    # n_test_groups is forced to 0 by the strategy (Walk-Forward CPCV);
    # the rolling loop defines test boundaries, CPCV only does CV on train domain.
    if result['training_method'] == 'cpcv':
        cpcv = {
            'n_groups': int(cfg.get('cpcv_n_groups', 10)),
            'n_val_groups': int(cfg.get('cpcv_n_val_groups', 1)),
            'purge_steps': int(cfg.get('cpcv_purge_steps', 3)),
            'embargo_steps': int(cfg.get('cpcv_embargo_steps', 5)),
        }
        # Tier 1 Fail-Fast: reject catastrophically bad params immediately
        _validate_cpcv_params(
            result['train_years'],
            cpcv['n_groups'],
            cpcv['purge_steps'],
            cpcv['embargo_steps'],
            freq='week',
        )
        result['cpcv'] = cpcv

    return result


if __name__ == "__main__":
    # Test loading
    import sys
    if len(sys.argv) > 1:
        c = load_workspace_config(sys.argv[1])
        print(json.dumps(c, indent=2))
