#!/usr/bin/env python
"""
CPCV Training Script (Purged Cross-Validation)

Trains K models per model_name using PurgedKFold cross-validation
following Marcos Lopez de Prado's Advances in Financial Machine Learning.
Each fold trains independently with purged, potentially discontiguous
training segments. Final predictions are averaged across folds.

Usage:
  python quantpits/scripts/cv_train.py --all-enabled
  python quantpits/scripts/cv_train.py --models lightgbm_Alpha158,gru_Alpha158
  python quantpits/scripts/cv_train.py --dry-run --all-enabled
  python quantpits/scripts/cv_train.py --predict-only --all-enabled

Modes:
  --full:           Train all enabled models with CPCV, overwrite records
  (default):        Incremental CPCV training on selected models, merge records
  --predict-only:   Use existing CPCV models to predict on new data
  --dry-run:        Preview fold/window plan without training
  --resume:         Resume from last interruption

Configuration:
  Set "data_slice_mode": "purged_cv" in config/model_config.json and add
  the "purged_cv" config block. See docs/01_TRAINING_GUIDE.md for details.
"""

import os
import sys
import json
import argparse
from datetime import datetime

from quantpits.utils import env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(SCRIPT_DIR)

DEFAULT_PREDICT_EXPERIMENT = "Prod_Predict_CPCV"


# ================= CLI =================
def parse_args():
    parser = argparse.ArgumentParser(
        description='CPCV Training: Purged Cross-Validation with K-fold ensemble',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all-enabled                    # CPCV train all enabled models
  %(prog)s --models lightgbm_Alpha158       # CPCV train a single model
  %(prog)s --tag tree                       # CPCV train all tree models
  %(prog)s --dry-run --all-enabled          # Preview folds without training
  %(prog)s --predict-only --all-enabled     # Predict only with existing models
  %(prog)s --models gru --resume            # Resume interrupted CPCV training
  %(prog)s --list                           # List registered models
        """
    )

    mode = parser.add_argument_group('Run Mode')
    mode.add_argument('--full', action='store_true',
                      help='Full CPCV training: train all enabled models, '
                           'overwrite latest_train_records.json')
    mode.add_argument('--predict-only', action='store_true',
                      help='Predict only: use existing CPCV models on new data')

    select = parser.add_argument_group('Model Selection')
    select.add_argument('--models', type=str,
                        help='Model names, comma-separated (e.g., gru,lightgbm_Alpha158)')
    select.add_argument('--algorithm', type=str,
                        help='Filter by algorithm (e.g., lstm, gru, lightgbm)')
    select.add_argument('--dataset', type=str,
                        help='Filter by dataset (e.g., Alpha158, Alpha360)')
    select.add_argument('--market', type=str,
                        help='Filter by market (e.g., csi300)')
    select.add_argument('--tag', type=str,
                        help='Filter by tag (e.g., ts, tree, attention)')
    select.add_argument('--all-enabled', action='store_true',
                        help='All models with enabled=true')

    skip_group = parser.add_argument_group('Exclude & Skip')
    skip_group.add_argument('--skip', type=str,
                            help='Skip models, comma-separated')
    skip_group.add_argument('--resume', action='store_true',
                            help='Resume from last interruption (skip completed models)')

    ctrl = parser.add_argument_group('Run Control')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='Preview plan only, do not train')
    ctrl.add_argument('--experiment-name', type=str, default=None,
                      help='MLflow experiment name '
                           '(default: Prod_Train_CPCV_{FREQ})')
    ctrl.add_argument('--no-pretrain', action='store_true',
                      help='Force random-weight init for basemodels')
    ctrl.add_argument('--source-records', type=str,
                      default='latest_train_records.json',
                      help='Source records file for predict-only')
    ctrl.add_argument('--cache-size', type=int, default=None, metavar='MB',
                      help='Handler cache max memory (MB). Default: auto-detect '
                           '(50%% free RAM). Set 0 to disable.')

    info = parser.add_argument_group('Information')
    info.add_argument('--list', action='store_true',
                      help='List model registry (supports filters)')
    info.add_argument('--show-state', action='store_true',
                      help='Show last run state')
    info.add_argument('--clear-state', action='store_true',
                      help='Clear run state file')

    return parser.parse_args()


# ================= Helpers =================
def _resolve_targets(args, registry):
    """Determine which models to train based on CLI args.

    Delegates to the shared resolve_target_models() in train_utils,
    but additionally validates that data_slice_mode is 'purged_cv'.
    """
    from quantpits.utils.train_utils import resolve_target_models
    from quantpits.utils.config_loader import load_workspace_config

    config = load_workspace_config(ROOT_DIR)
    data_slice_mode = config.get('data_slice_mode', 'slide')

    targets = resolve_target_models(args, registry)
    if targets is None or not targets:
        print("Error: specify --models, --all-enabled, --full, or a filter.")
        sys.exit(1)

    # Apply skip
    if args.skip:
        skip_names = set(s.strip() for s in args.skip.split(',') if s.strip())
        targets = {k: v for k, v in targets.items() if k not in skip_names}

    return targets


# ================= Full CPCV Training =================
def run_full_train_cpcv(args):
    """Full CPCV training: train all enabled models, overwrite records."""
    from quantpits.utils.train_utils import (
        calculate_dates,
        load_model_registry,
        get_enabled_models,
        train_cpcv_model,
        overwrite_train_records,
        backup_file_with_date,
        print_model_table,
        make_model_key,
        PREDICTION_OUTPUT_DIR,
        RECORD_OUTPUT_FILE,
    )

    env.init_qlib()
    params = calculate_dates()

    if params.get('data_slice_mode') != 'purged_cv':
        print("Error: data_slice_mode must be 'purged_cv' in model_config.json")
        sys.exit(1)

    folds = params.get('cpcv_folds', [])
    if not folds:
        print("Error: No CPCV folds generated. Check purged_cv config.")
        sys.exit(1)

    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    registry = load_model_registry()
    enabled_models = get_enabled_models(registry)

    if not enabled_models:
        print("Warning: No enabled=true models found in model_registry.yaml")
        return

    print_model_table(enabled_models, title="CPCV Full Training Models")
    print(f"  CPCV config: {len(folds)} folds per model")
    print(f"  Test set: [{params['test_start_time']}, {params['test_end_time']}]")
    for fi, fold in enumerate(folds):
        n_seg = len(fold['train_segments'])
        seg_desc = " + ".join(f"[{s[0]}, {s[1]}]" for s in fold['train_segments'])
        print(f"  Fold {fi}: valid=[{fold['valid_start_time']}, "
              f"{fold['valid_end_time']}], train=({n_seg} seg) {seg_desc}")

    if args.dry_run:
        print("\n  Dry-run mode: above models would be trained with CPCV")
        return

    # Initialize handler cache (unless --cache-size 0)
    cache_mgr = None
    if args.cache_size != 0:
        from quantpits.utils.handler_cache import (
            HandlerCacheManager, enumerate_tasks_cpcv, pre_analyze,
        )
        cache_mgr = HandlerCacheManager(
            max_size_mb=args.cache_size if args.cache_size else None)
        yaml_paths = {m: info['yaml_file'] for m, info in enabled_models.items()}
        tasks = enumerate_tasks_cpcv(
            list(enabled_models.keys()), yaml_paths, params)
        pre_analyze(tasks, cache_mgr)

    experiment_name = args.experiment_name or f"Prod_Train_CPCV_{params.get('freq', 'week').upper()}"

    current_records = {
        "experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }
    model_performances = {}

    total = len(enabled_models)
    for idx, (model_name, model_info) in enumerate(enabled_models.items(), 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}] CPCV: {model_name}")
        print(f"{'─'*60}")

        yaml_file = model_info['yaml_file']
        result = train_cpcv_model(
            model_name, yaml_file, params, experiment_name,
            no_pretrain=args.no_pretrain,
            cache_mgr=cache_mgr,
        )

        if result['success']:
            model_key = make_model_key(model_name, 'cpcv')
            current_records['models'][model_key] = result['record_id']
            if result['performance']:
                model_performances[model_name] = result['performance']
        else:
            print(f"  FAILED: {model_name} — {result.get('error', 'Unknown')}")

    # Overwrite records (auto-backup)
    overwrite_train_records(current_records)

    # Save performance
    perf_file = os.path.join(ROOT_DIR, "output",
                             f"model_performance_cpcv_{params['anchor_date']}.json")
    os.makedirs(os.path.dirname(perf_file), exist_ok=True)
    backup_file_with_date(perf_file)
    with open(perf_file, 'w') as f:
        json.dump(model_performances, f, indent=4)

    if cache_mgr is not None:
        print(f"\n  Handler Cache: {cache_mgr}")

    print(f"\n{'='*50}")
    print(f"CPCV Full training complete. Experiment: {experiment_name}")
    print(f"Records saved to {RECORD_OUTPUT_FILE}")
    print(f"{'='*50}\n")


# ================= Incremental CPCV Training =================
def run_incremental_train_cpcv(args, targets):
    """Incremental CPCV training on selected models, merge into records."""
    from quantpits.utils.train_utils import (
        calculate_dates,
        train_cpcv_model,
        merge_train_records,
        merge_performance_file,
        save_run_state,
        load_run_state,
        clear_run_state,
        print_model_table,
        make_model_key,
        RECORD_OUTPUT_FILE,
    )

    print_model_table(targets, title="CPCV Training Targets")

    # Resume handling
    completed_models = set()
    if args.resume:
        state = load_run_state()
        if state and state.get('completed'):
            completed_models = set(state['completed'])
            remaining = {k: v for k, v in targets.items()
                         if k not in completed_models}
            if completed_models:
                skipped = [m for m in targets if m in completed_models]
                print(f"  Resume: skip {len(skipped)} completed: "
                      f"{', '.join(skipped)}")
            targets = remaining
            if not targets:
                print("  All target models already completed.")
                return
            print_model_table(targets, title="Remaining CPCV Targets")
        else:
            print("  No prior run state found; starting from scratch.")

    if args.dry_run:
        # Show fold plan before exiting
        env.init_qlib()
        params = calculate_dates()
        folds = params.get('cpcv_folds', [])
        print(f"\n  CPCV Fold Plan ({len(folds)} folds):")
        print(f"  Test set: [{params['test_start_time']}, {params['test_end_time']}]")
        for fi, fold in enumerate(folds):
            n_seg = len(fold['train_segments'])
            seg_desc = " + ".join(f"[{s[0]}, {s[1]}]" for s in fold['train_segments'])
            print(f"  Fold {fi}: valid=[{fold['valid_start_time']}, "
                  f"{fold['valid_end_time']}]")
            print(f"          train=({n_seg} seg) {seg_desc}")
        print(f"\n  Dry-run mode: above {len(targets)} model(s) would be CPCV trained")
        return

    print("\n" + "=" * 60)
    print("CPCV Incremental Training")
    print("=" * 60)

    env.init_qlib()
    params = calculate_dates()

    if params.get('data_slice_mode') != 'purged_cv':
        print("Error: data_slice_mode must be 'purged_cv' in model_config.json")
        sys.exit(1)

    # Initialize handler cache (unless --cache-size 0)
    cache_mgr = None
    if args.cache_size != 0:
        from quantpits.utils.handler_cache import (
            HandlerCacheManager, enumerate_tasks_cpcv, pre_analyze,
        )
        cache_mgr = HandlerCacheManager(
            max_size_mb=args.cache_size if args.cache_size else None)
        yaml_paths = {m: info['yaml_file'] for m, info in targets.items()}
        tasks = enumerate_tasks_cpcv(
            list(targets.keys()), yaml_paths, params)
        pre_analyze(tasks, cache_mgr)

    freq = params.get('freq', 'week').upper()
    experiment_name = args.experiment_name or f"Prod_Train_CPCV_{freq}"

    # Run state
    all_target_names = list(completed_models | set(targets.keys()))
    run_state = {
        'started_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'mode': 'cpcv_incremental',
        'experiment_name': experiment_name,
        'anchor_date': params['anchor_date'],
        'target_models': all_target_names,
        'completed': list(completed_models),
        'failed': {},
        'skipped': [],
    }
    save_run_state(run_state)

    new_records = {
        "experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }
    new_performances = {}

    total = len(targets)
    for idx, (model_name, model_info) in enumerate(targets.items(), 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}] CPCV: {model_name}")
        print(f"{'─'*60}")

        yaml_file = model_info['yaml_file']
        result = train_cpcv_model(
            model_name, yaml_file, params, experiment_name,
            no_pretrain=args.no_pretrain,
            cache_mgr=cache_mgr,
        )

        if result['success']:
            model_key = make_model_key(model_name, 'cpcv')
            new_records['models'][model_key] = result['record_id']
            if result['performance']:
                new_performances[model_name] = result['performance']
            run_state['completed'].append(model_name)
        else:
            run_state['failed'][model_name] = result.get('error', 'Unknown')
            print(f"  FAILED: {model_name} — {result.get('error', 'Unknown')}")

        save_run_state(run_state)

    # Merge records
    if new_records['models']:
        print("\n" + "=" * 60)
        print("Merging CPCV training records")
        print("=" * 60)
        merge_train_records(new_records)
        if new_performances:
            merge_performance_file(new_performances, params['anchor_date'])

    # Summary
    print(f"\n{'='*60}")
    print("CPCV Incremental Training Complete")
    print("=" * 60)
    succeeded = [m for m in run_state['completed'] if m in targets]
    failed = run_state['failed']
    print(f"  OK: {len(succeeded)} models")
    for name in succeeded:
        print(f"    {name}")
    if failed:
        print(f"  FAILED: {len(failed)} models")
        for name, err in failed.items():
            print(f"    {name}: {err[:80]}")
    if cache_mgr is not None:
        print(f"\n  Handler Cache: {cache_mgr}")
    print(f"{'='*60}\n")


# ================= CPCV Predict-Only =================
def run_predict_only_cpcv(args):
    """Predict on new data using existing CPCV-trained models."""
    from quantpits.utils.train_utils import (
        calculate_dates,
        load_model_registry,
        filter_models_by_mode,
        predict_cpcv_model,
        merge_train_records,
        print_model_table,
        make_model_key,
        parse_model_key,
        RECORD_OUTPUT_FILE,
    )

    env.init_qlib()
    params = calculate_dates()

    # Load source records
    source_file = os.path.join(ROOT_DIR, args.source_records)
    if not os.path.exists(source_file):
        print(f"Error: source records file not found: {source_file}")
        sys.exit(1)

    with open(source_file, 'r') as f:
        source_records = json.load(f)

    source_models = source_records.get('models', {})
    cpcv_models = filter_models_by_mode(source_models, 'cpcv')

    if not cpcv_models:
        print("No @cpcv models found in source records.")
        return

    # Apply model selection filters
    registry = load_model_registry()
    if args.models:
        keep = set(m.strip() for m in args.models.split(',') if m.strip())
        cpcv_models = {k: v for k, v in cpcv_models.items()
                       if parse_model_key(k)[0] in keep}
    elif args.tag:
        from quantpits.utils.train_utils import get_models_by_filter
        tagged = get_models_by_filter(registry, tag=args.tag)
        cpcv_models = {k: v for k, v in cpcv_models.items()
                       if parse_model_key(k)[0] in tagged}

    if not cpcv_models:
        print("No @cpcv models match selection.")
        return

    print(f"\nCPCV Predict-Only: {len(cpcv_models)} models")
    for key, rid in cpcv_models.items():
        print(f"  {key} -> {rid}")

    if args.dry_run:
        print("\n  Dry-run: above models would be predicted")
        return

    freq = params.get('freq', 'week').upper()
    experiment_name = args.experiment_name or f"Prod_Predict_CPCV_{freq}"

    new_records = {
        "experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }

    total = len(cpcv_models)
    for idx, (model_key, record_id) in enumerate(cpcv_models.items(), 1):
        model_name = parse_model_key(model_key)[0]
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}] CPCV Predict: {model_name}")
        print(f"{'─'*60}")

        # Get model info from registry
        model_info = registry.get(model_name, {})
        model_info['record_id'] = record_id
        model_info['yaml_file'] = model_info.get('yaml_file')

        result = predict_cpcv_model(
            model_name, model_info, params, experiment_name,
            no_pretrain=args.no_pretrain,
        )

        if result['success']:
            new_records['models'][model_key] = result['record_id']
        else:
            print(f"  FAILED: {model_name} — {result.get('error', 'Unknown')}")

    if new_records['models']:
        merge_train_records(new_records)

    print(f"\nCPCV Predict-Only complete. Experiment: {experiment_name}\n")


# ================= Main =================
def main():
    from quantpits.utils.train_utils import (
        load_model_registry,
        print_model_table,
        load_run_state,
        clear_run_state,
        save_run_state,
        parse_model_key,
    )

    args = parse_args()
    os.chdir(ROOT_DIR)

    # --list
    if args.list:
        registry = load_model_registry()
        targets = _resolve_targets(args, registry) if (
            args.models or args.algorithm or args.dataset or
            args.market or args.tag
        ) else registry
        print_model_table(targets, title="Model Registry")
        return

    # --show-state
    if args.show_state:
        state = load_run_state()
        if state:
            print(json.dumps(state, indent=2, default=str))
        else:
            print("No run state file found.")
        return

    # --clear-state
    if args.clear_state:
        clear_run_state()
        print("Run state cleared.")
        return

    # Determine mode
    if args.predict_only:
        run_predict_only_cpcv(args)
    elif args.full:
        run_full_train_cpcv(args)
    else:
        # Incremental CPCV training (default)
        registry = load_model_registry()
        targets = _resolve_targets(args, registry)
        if not targets:
            print("No models selected. Use --models, --all-enabled, or filters.")
            sys.exit(1)
        run_incremental_train_cpcv(args, targets)


if __name__ == '__main__':
    main()
