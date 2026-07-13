#!/usr/bin/env python
"""
Rolling Training Script (滚动训练)

Independent rolling training logic supporting multiple per-window training
strategies: slide (classic train/valid/test) and CPCV (purged K-fold).

Usage:
  cd QuantPits && python quantpits/scripts/rolling_train.py [options]

Modes:
  --cold-start:     Full rebuild — clear all state, train all windows.
  --merge:          Detect & train only missing windows (daily maintenance).
  --retrain-models: Clear specified models' records, then rebuild.
  --retrain-last:   Clear & retrain only the last window.
  --predict-only:   Predict using latest window's model, no training.
  --resume:         Resume from interrupted run.
  --backtest-only:  Run backtest on existing rolling predictions.

Examples:
  # Slide-mode cold start (default)
  python quantpits/scripts/rolling_train.py --cold-start --all-enabled

  # CPCV-mode cold start (set training_method: cpcv in rolling_config.yaml)
  python quantpits/scripts/rolling_train.py --cold-start --all-enabled

  # Daily update
  python quantpits/scripts/rolling_train.py --merge --all-enabled

  # Predict only
  python quantpits/scripts/rolling_train.py --predict-only --all-enabled

  # Show state
  python quantpits/scripts/rolling_train.py --show-state

  # Resume
  python quantpits/scripts/rolling_train.py --resume
"""

import os
import sys
import argparse

from quantpits.utils import env
from quantpits.utils.constants import MONTHS_PER_YEAR
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(SCRIPT_DIR)


# ==========================================================================
# Strategy dispatch
# ==========================================================================

def _get_strategy(training_method):
    """Resolve strategy module, key suffix, and state file path.

    Args:
        training_method: 'slide' or 'cpcv'.

    Returns:
        (strategy_module, mode_suffix, state_file_path)
    """
    from quantpits.utils.train_utils import ROLLING_STATE_FILE, ROLLING_STATE_FILE_CPCV

    if training_method == 'cpcv':
        from quantpits.scripts.rolling import strategy_cpcv as st
        return st, 'cpcv_rolling', ROLLING_STATE_FILE_CPCV
    else:
        from quantpits.scripts.rolling import strategy_slide as st
        return st, 'rolling', ROLLING_STATE_FILE


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Rolling Training: sliding time-window training + prediction stitching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time: full cold start
  %(prog)s --cold-start --all-enabled

  # After qlib data update: train only new windows
  %(prog)s --merge --all-enabled

  # Rebuild a model after hyperparameter change
  %(prog)s --retrain-models alstm_Alpha158

  # Retrain last window only
  %(prog)s --retrain-last --all-enabled

  # Predict only, no training
  %(prog)s --predict-only --all-enabled

  # View state / resume
  %(prog)s --show-state
  %(prog)s --resume
        """
    )

    mode = parser.add_argument_group('Run Mode')
    mode.add_argument('--cold-start', action='store_true',
                      help='Full cold start: clear ALL training records, '
                           'retrain all windows from scratch.')
    mode.add_argument('--merge', action='store_true',
                      help='Merge mode: detect & train only missing windows. '
                           'Use for: (1) new windows after qlib data update '
                           '(2) appending new models to existing state.')
    mode.add_argument('--retrain-models', type=str,
                      help='Rebuild specified models: clear their records '
                           'then retrain all windows. Comma-separated.')
    mode.add_argument('--retrain-last', action='store_true',
                      help='Retrain last window only. '
                           'Use --models to limit scope.')
    mode.add_argument('--predict-only', action='store_true',
                      help='Predict only: use latest window weights for '
                           'current data. No training.')
    mode.add_argument('--resume', action='store_true',
                      help='Resume from interrupted run.')
    mode.add_argument('--backtest', action='store_true',
                      help='Run full backtest on stitched predictions '
                           'after training completes.')
    mode.add_argument('--backtest-only', action='store_true',
                      help='Backtest only: skip training & prediction.')
    mode.add_argument('--allow-stale-predict', action='store_true',
                      help='Allow predict-only to use old weights on new '
                           'data when untrained windows exist.')

    select = parser.add_argument_group('Model Selection')
    select.add_argument('--models', type=str,
                        help='Model names, comma-separated')
    select.add_argument('--algorithm', type=str,
                        help='Filter by algorithm')
    select.add_argument('--dataset', type=str,
                        help='Filter by dataset')
    select.add_argument('--tag', type=str,
                        help='Filter by tag')
    select.add_argument('--all-enabled', action='store_true',
                        help='All enabled models')
    select.add_argument('--skip', type=str,
                        help='Skip models, comma-separated')

    ctrl = parser.add_argument_group('Run Control')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='Preview plan only, do not train')
    ctrl.add_argument('--show-folds', action='store_true',
                      help='Show CPCV fold details (train segments, validation) '
                           'during dry-run. Ignored for slide mode.')
    ctrl.add_argument('--no-pretrain', action='store_true',
                      help='Skip pretrained model loading')
    ctrl.add_argument('--cache-size', type=int, default=None, metavar='MB',
                      help='Handler cache max memory (MB). Default: auto-detect '
                           '(50%% free RAM). Set 0 to disable. '
                           'CPCV mode: K folds within each window share the cache.')
    ctrl.add_argument('--training-method', type=str, default=None,
                      choices=['slide', 'cpcv'],
                      help='Override training_method from rolling_config.yaml. '
                           'Use to switch modes without editing config.')

    info = parser.add_argument_group('Information')
    info.add_argument('--show-state', action='store_true',
                      help='Show rolling state')
    info.add_argument('--clear-state', action='store_true',
                      help='Clear rolling state')

    return parser.parse_args()


def resolve_target_models(args):
    """Resolve target models (delegates to shared train_utils implementation)."""
    from quantpits.utils.train_utils import resolve_target_models as _resolve
    return _resolve(args)


def get_base_params():
    """Get workspace base params (market, benchmark, etc.)."""
    from quantpits.utils.config_loader import load_workspace_config
    config = load_workspace_config(ROOT_DIR)

    from qlib.data import D
    try:
        last_trade_date = D.calendar(future=False)[-1:][0]
        anchor_date = last_trade_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"⚠️  Warning: cannot get trading calendar from qlib "
              f"(env may be empty or uninitialized), using default '2024-12-31'. "
              f"Error: {e}")
        anchor_date = '2024-12-31'

    return {
        'market': config.get('market', 'csi300'),
        'benchmark': config.get('benchmark', 'SH000300'),
        'topk': config.get('topk', 20),
        'n_drop': config.get('n_drop', 3),
        'buy_suggestion_factor': config.get('buy_suggestion_factor', 2),
        'account': config.get('current_full_cash', 100000.0),
        'freq': config.get('freq', 'week').lower(),
        'anchor_date': anchor_date,
    }


def _print_fold_details(folds, indent=6):
    """Print CPCV fold train/valid segments for inspection."""
    prefix = " " * indent
    for fi, fold in enumerate(folds):
        n_seg = len(fold['train_segments'])
        seg_desc = " | ".join(f"[{s[0]}, {s[1]}]" for s in fold['train_segments'])
        print(f"{prefix}Fold {fi}: valid=[{fold['valid_start_time']}, "
              f"{fold['valid_end_time']}] train=({n_seg}) {seg_desc}")


# ==========================================================================
# Flow functions
# ==========================================================================

def run_cold_start(args, targets, rolling_cfg):
    """Cold start / merge / resume: Model-First loop with subprocess isolation.

    Supports both slide and CPCV strategies via training_method config.
    """
    from quantpits.utils.train_utils import print_model_table
    # NOTE: orchestration/backtest/state functions are referenced via
    # module-level names (re-exported at bottom of file) so that
    # mock.patch('rolling_train.<func>') can intercept calls.
    from quantpits.scripts.rolling.memory import deep_cleanup_after_model
    from qlib.config import C

    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    qlib_config = C

    training_method = rolling_cfg.get('training_method', 'slide')
    st, mode_suffix, state_file = _get_strategy(training_method)

    # Generate windows: module-level name for slide (allows mock.patch),
    # strategy function for CPCV.
    window_kwargs = dict(
        rolling_start=rolling_cfg['rolling_start'],
        train_years=rolling_cfg['train_years'],
        test_step=rolling_cfg['test_step'],
        anchor_date=anchor_date,
    )
    if training_method == 'cpcv':
        window_kwargs['cpcv_cfg'] = rolling_cfg.get('cpcv', {})
        window_kwargs['freq'] = params_base['freq']
        windows = st.generate_windows(**window_kwargs)
    else:
        window_kwargs['valid_years'] = rolling_cfg.get('valid_years', 1)
        windows = generate_rolling_windows(**window_kwargs)

    if not windows:
        print("❌ No rolling windows could be generated — "
              "check rolling_config.yaml")
        print(f"   rolling_start + train_years may exceed anchor_date "
              f"({anchor_date})?")
        return

    # Print windows
    show_folds = getattr(args, 'show_folds', False)
    print(f"\n{'='*70}")
    print(f"📅 Rolling Windows ({len(windows)} total, method={training_method})")
    print(f"{'='*70}")
    for w in windows:
        if training_method == 'cpcv':
            n_folds = len(w.get('cpcv_folds', []))
            print(f"  Window {w['window_idx']:2d}: "
                  f"Train[{w.get('train_start', '?')}, {w.get('train_end', '?')}] "
                  f"Test[{w['test_start']}, {w['test_end']}] "
                  f"({n_folds} folds)")
            if show_folds:
                _print_fold_details(w.get('cpcv_folds', []), indent=6)
        else:
            print(f"  Window {w['window_idx']:2d}: "
                  f"Train[{w['train_start']}, {w['train_end']}] "
                  f"Valid[{w['valid_start']}, {w['valid_end']}] "
                  f"Test[{w['test_start']}, {w['test_end']}]")
    print(f"{'='*70}")

    print_model_table(targets, title=f"Rolling Training Models ({training_method})")

    if args.dry_run:
        print("🔍 Dry-run mode: windows shown above, no training")
        return

    # Initialize state
    state = RollingState(state_file=state_file)
    if not args.resume and not args.merge:
        state.init_run(rolling_cfg, anchor_date, len(windows),
                       training_method=training_method)
    else:
        if not state.anchor_date:
            print("❌ No rolling state — creating new state")
            state.init_run(rolling_cfg, anchor_date, len(windows),
                           training_method=training_method)
        else:
            print(f"⏩ {'Merge' if args.merge else 'Resume'} mode: "
                  f"skipping completed windows")

    rolling_exp_name = f"Rolling_Windows_{freq}"
    combined_exp_name = f"Rolling_Combined_{freq}"

    # ===== Model-First loop =====
    combined_records = {}
    total_trained = 0

    for model_name, model_info in targets.items():
        n_trained = run_model_windows(
            model_name=model_name,
            model_info=model_info,
            windows=windows,
            state=state,
            params_base=params_base,
            experiment_name=rolling_exp_name,
            qlib_config=qlib_config,
            train_fn=st.train_window_isolated,
            no_pretrain=args.no_pretrain,
            dry_run=args.dry_run,
            cache_size=args.cache_size,
        )
        total_trained += n_trained

        # Stitch immediately after all windows for this model are done
        model_combined = concatenate_rolling_predictions(
            state=state,
            model_names=[model_name],
            rolling_exp_name=rolling_exp_name,
            combined_exp_name=combined_exp_name,
            anchor_date=anchor_date,
            windows=windows,
            targets=targets,
            params_base=params_base,
            repair_fn=st.repair_truncated,
        )
        combined_records.update(model_combined)

        deep_cleanup_after_model(model_name)

    # Merge mode: also stitch models not in this run's targets
    if args.merge:
        completed = state.get_all_completed_windows()
        extra_models = set()
        for win, models in completed.items():
            extra_models.update(models.keys())
        extra_models -= set(targets.keys())
        if extra_models:
            extra_combined = concatenate_rolling_predictions(
                state=state,
                model_names=list(extra_models),
                rolling_exp_name=rolling_exp_name,
                combined_exp_name=combined_exp_name,
                anchor_date=anchor_date,
                windows=windows,
                targets=targets,
                params_base=params_base,
                repair_fn=st.repair_truncated,
            )
            combined_records.update(extra_combined)

    # Save records
    if combined_records:
        save_rolling_records(combined_records, combined_exp_name, anchor_date,
                             mode=mode_suffix, verify_recorders=True, config_payload=params_base)
        if args.backtest:
            run_combined_backtest(
                list(combined_records.keys()), combined_records,
                combined_exp_name, params_base,
            )

    # Done
    print(f"\n{'='*60}")
    print(f"✅ Rolling training complete ({training_method})")
    print(f"{'='*60}")
    print(f"  Windows: {len(windows)}")
    print(f"  Models: {len(targets)}")
    print(f"  Trained: {total_trained} tasks")
    print(f"  Combined records: {len(combined_records)}")
    print(f"  Key suffix: @{mode_suffix}")
    print(f"\n  💡 Next steps:")
    print(f"     Brute force: python quantpits/scripts/brute_force_ensemble.py "
          f"--use-groups --training-mode {mode_suffix}")
    print(f"     Fusion: python quantpits/scripts/ensemble_fusion.py "
          f"--from-config --training-mode {mode_suffix}")
    print(f"{'='*60}")


def run_daily(args, targets, rolling_cfg):
    """Daily mode: detect & train new windows. Model-First + subprocess.

    Falls back to predict-only when all windows are up to date.
    """
    # NOTE: orchestration/backtest/state functions are referenced via
    # module-level names (re-exported at bottom of file) so that
    # mock.patch('rolling_train.<func>') can intercept calls.
    from quantpits.scripts.rolling.memory import deep_cleanup_after_model
    from qlib.config import C

    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    qlib_config = C

    training_method = rolling_cfg.get('training_method', 'slide')
    st, mode_suffix, state_file = _get_strategy(training_method)

    state = RollingState(state_file=state_file)

    if not state.anchor_date:
        print("❌ No rolling state — run --cold-start first")
        return

    # Generate windows: use module-level name for slide (allows mock.patch
    # interception in tests), strategy function for CPCV.
    window_kwargs = dict(
        rolling_start=rolling_cfg['rolling_start'],
        train_years=rolling_cfg['train_years'],
        test_step=rolling_cfg['test_step'],
        anchor_date=anchor_date,
    )
    if training_method == 'cpcv':
        window_kwargs['cpcv_cfg'] = rolling_cfg.get('cpcv', {})
        window_kwargs['freq'] = params_base['freq']
        windows = st.generate_windows(**window_kwargs)
    else:
        window_kwargs['valid_years'] = rolling_cfg.get('valid_years', 1)
        windows = generate_rolling_windows(**window_kwargs)

    # Detect new windows: only count a window as "completed" when ALL
    # target models have finished it.  This prevents a new model (with no
    # records) from being silently skipped because other models are done.
    completed = state.get_all_completed_windows()
    target_names = set(targets.keys())
    completed_indices = {
        int(widx_str)
        for widx_str, models in completed.items()
        if target_names.issubset(set(models.keys()))
    }
    new_windows = [w for w in windows if w['window_idx'] not in completed_indices]

    rolling_exp_name = f"Rolling_Windows_{freq}"
    combined_exp_name = f"Rolling_Combined_{freq}"

    if new_windows:
        show_folds = getattr(args, 'show_folds', False)
        print(f"\n🔄 {len(new_windows)} new window(s) need training")
        for w in new_windows:
            suffix = ""
            if training_method == 'cpcv':
                n_folds = len(w.get('cpcv_folds', []))
                suffix = f" Train[{w.get('train_start','?')}, {w.get('train_end','?')}] ({n_folds} folds)"
            print(f"  Window {w['window_idx']}: Test[{w['test_start']}, {w['test_end']}]{suffix}")
            if show_folds and training_method == 'cpcv':
                _print_fold_details(w.get('cpcv_folds', []), indent=6)

        if args.dry_run:
            print("🔍 Dry-run mode: new windows shown above")
            return

        combined_records = {}
        for model_name, model_info in targets.items():
            run_model_windows(
                model_name=model_name,
                model_info=model_info,
                windows=new_windows,
                state=state,
                params_base=params_base,
                experiment_name=rolling_exp_name,
                qlib_config=qlib_config,
                train_fn=st.train_window_isolated,
                no_pretrain=args.no_pretrain,
            )

            model_combined = concatenate_rolling_predictions(
                state=state,
                model_names=[model_name],
                rolling_exp_name=rolling_exp_name,
                combined_exp_name=combined_exp_name,
                anchor_date=anchor_date,
                windows=windows,
                targets=targets,
                params_base=params_base,
                repair_fn=st.repair_truncated,
            )
            combined_records.update(model_combined)

            deep_cleanup_after_model(model_name)

        if combined_records:
            save_rolling_records(combined_records, combined_exp_name, anchor_date,
                                 mode=mode_suffix, verify_recorders=True, config_payload=params_base)
            if args.backtest:
                run_combined_backtest(
                    list(combined_records.keys()), combined_records,
                    combined_exp_name, params_base,
                )

        print(f"\n✅ Rolling update complete ({len(new_windows)} new windows)")

    else:
        # All windows trained — predict-only with latest model
        print(f"\n📊 All windows up to date — predict-only mode...")

        extra_preds = {}
        for model_name, model_info in targets.items():
            pred = st.predict_latest(
                model_name, model_info, state,
                rolling_exp_name, params_base, anchor_date,
                windows=windows,
            )
            if pred is not None and not pred.empty:
                extra_preds[model_name] = pred

        if extra_preds:
            model_names = list(targets.keys())
            combined_records = concatenate_rolling_predictions(
                state, model_names, rolling_exp_name, combined_exp_name,
                anchor_date, windows=windows, extra_preds=extra_preds,
                targets=targets, params_base=params_base,
                repair_fn=st.repair_truncated,
            )
            if combined_records:
                save_rolling_records(combined_records, combined_exp_name,
                                     anchor_date, mode=mode_suffix, verify_recorders=True, config_payload=params_base)
                if args.backtest:
                    run_combined_backtest(model_names, combined_records,
                                          combined_exp_name, params_base)


def run_predict_only(args, targets, rolling_cfg):
    """Predict-only mode: use latest window models for current data."""
    # NOTE: orchestration/backtest/state functions are referenced via
    # module-level names (re-exported at bottom of file) so that
    # mock.patch('rolling_train.<func>') can intercept calls.

    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    allow_stale = getattr(args, 'allow_stale_predict', False)

    training_method = rolling_cfg.get('training_method', 'slide')
    st, mode_suffix, state_file = _get_strategy(training_method)

    state = RollingState(state_file=state_file)
    if not state.anchor_date:
        print("❌ No rolling state — run --cold-start first")
        return

    # Generate windows: module-level for slide (mock.patch compat), strategy for CPCV
    window_kwargs = dict(
        rolling_start=rolling_cfg['rolling_start'],
        train_years=rolling_cfg['train_years'],
        test_step=rolling_cfg['test_step'],
        anchor_date=anchor_date,
    )
    if training_method == 'cpcv':
        window_kwargs['cpcv_cfg'] = rolling_cfg.get('cpcv', {})
        window_kwargs['freq'] = params_base['freq']
        windows = st.generate_windows(**window_kwargs)
    else:
        window_kwargs['valid_years'] = rolling_cfg.get('valid_years', 1)
        windows = generate_rolling_windows(**window_kwargs)

    rolling_exp_name = f"Rolling_Windows_{freq}"
    extra_preds = {}
    gap_models = []
    gap_skipped = []

    for model_name, model_info in targets.items():
        # Detect gap
        completions = state.get_completed_record_ids(model_name) if state.anchor_date else []
        if completions and windows:
            try:
                last_completed = int(completions[-1]['window_idx'])
                last_available = int(windows[-1]['window_idx'])
                if last_completed < last_available:
                    gap_models.append((model_name, last_completed, last_available))
            except (TypeError, ValueError, KeyError):
                pass

        pred = st.predict_latest(
            model_name, model_info, state,
            rolling_exp_name, params_base, anchor_date,
            windows=windows, allow_stale_predict=allow_stale,
        )
        if pred is not None and not pred.empty:
            extra_preds[model_name] = pred
        elif completions and not allow_stale:
            for name, comp, avail in gap_models:
                if name == model_name:
                    gap_skipped.append((name, comp, avail))
                    break

    if gap_skipped:
        print(f"\n{'='*60}")
        print(f"⛔ Models with untrained windows skipped "
              f"(--allow-stale-predict not set):")
        for name, comp, avail in gap_skipped:
            print(f"   {name}: trained up to W{comp}, data up to W{avail} "
                  f"({avail - comp} window(s) gap)")
        print(f"   Option 1: --retrain-models <model>  (recommended)")
        print(f"   Option 2: --allow-stale-predict       (use old weights)")
        print(f"{'='*60}")
    elif gap_models and allow_stale:
        print(f"\n{'='*60}")
        print(f"⚠️  Models with untrained windows — stale predict active:")
        for name, comp, avail in gap_models:
            print(f"   {name}: trained up to W{comp}, data up to W{avail}")
        print(f"   Recommend: --retrain-models <model> soon")
        print(f"{'='*60}")

    if extra_preds:
        combined_exp_name = f"Rolling_Combined_{freq}"
        model_names = list(targets.keys())
        combined_records = concatenate_rolling_predictions(
            state, model_names, rolling_exp_name, combined_exp_name,
            anchor_date, windows=windows, extra_preds=extra_preds,
            targets=targets, params_base=params_base,
            repair_fn=st.repair_truncated,
        )
        if combined_records:
            save_rolling_records(combined_records, combined_exp_name,
                                 anchor_date, mode=mode_suffix, verify_recorders=True, config_payload=params_base)
            if args.backtest:
                run_combined_backtest(model_names, combined_records,
                                      combined_exp_name, params_base)


# ==========================================================================
# Main
# ==========================================================================

def main():
    from quantpits.utils import env as _env
    _env.safeguard("Rolling Train")
    args = parse_args()

    # Load rolling config first (needed for training_method-aware state commands)
    from quantpits.utils.config_loader import load_rolling_config
    rolling_cfg = load_rolling_config(ROOT_DIR)
    if rolling_cfg is None and not (args.show_state or args.clear_state):
        print("❌ config/rolling_config.yaml not found")
        print("   Create a rolling config file first")
        return

    # Determine effective training_method (config + CLI override)
    training_method = (rolling_cfg or {}).get('training_method', 'slide')
    if args.training_method is not None:
        training_method = args.training_method
        if rolling_cfg:
            if args.training_method != rolling_cfg.get('training_method', 'slide'):
                print(f"  ℹ️  --training-method={args.training_method} overrides config")
            rolling_cfg['training_method'] = training_method
    _, mode_suffix, state_file = _get_strategy(training_method)

    # Info commands (now training-method-aware)
    if args.show_state:
        RollingState(state_file=state_file).show()
        return

    if args.clear_state:
        RollingState(state_file=state_file).clear()
        return

    if rolling_cfg is None:
        print("❌ config/rolling_config.yaml not found")
        return

    print(f"\n📋 Rolling Config ({training_method}):")
    print(f"   Start(T): {rolling_cfg['rolling_start']}")
    print(f"   Train(X): {rolling_cfg['train_years']} years")
    if training_method != 'cpcv':
        print(f"   Valid(Y): {rolling_cfg['valid_years']} years")
    print(f"   Step(Z):  {rolling_cfg['test_step']} "
          f"({rolling_cfg['test_step_months']} months)")
    if training_method == 'cpcv':
        cpcv = rolling_cfg.get('cpcv', {})
        print(f"   CPCV:     {cpcv.get('n_groups')} groups, "
              f"{cpcv.get('n_val_groups')} val, "
              f"purge={cpcv.get('purge_steps')}, "
              f"embargo={cpcv.get('embargo_steps')}")

    # --retrain-last
    if args.retrain_last:
        state = RollingState(state_file=state_file)
        if not state.anchor_date:
            print("❌ No rolling state — run --cold-start first")
            return
        last_idx = state.get_last_completed_window_idx()
        if last_idx is None:
            print("ℹ️  No completed windows to retrain")
            return
        if isinstance(args.models, str) and args.models.strip():
            target_models = [m.strip() for m in args.models.split(',') if m.strip()]
            wkey = str(last_idx)
            cw = state.get_all_completed_windows()
            removed = 0
            for model_name in target_models:
                if wkey in cw and model_name in cw[wkey]:
                    del cw[wkey][model_name]
                    removed += 1
            if removed > 0:
                state.save()
                print(f"🔄 Window {last_idx}: cleared {removed} model(s)")
            else:
                print(f"ℹ️  Window {last_idx}: no matching records, skipping")
        else:
            state.remove_window(last_idx)
            print(f"🔄 Window {last_idx}: cleared all records, will retrain")

    # --retrain-models
    if isinstance(args.retrain_models, str) and args.retrain_models.strip():
        state = RollingState(state_file=state_file)
        if not state.anchor_date:
            print("❌ No rolling state — run --cold-start first")
            return
        model_list = [m.strip() for m in args.retrain_models.split(',') if m.strip()]
        if not model_list:
            print("❌ --retrain-models requires at least one model name")
            return
        removed_total = 0
        for model_name in model_list:
            removed = state.remove_model(model_name)
            if removed > 0:
                print(f"🔄 {model_name}: cleared {removed} window record(s)")
                removed_total += removed
            else:
                print(f"ℹ️  {model_name}: no records in state, will train from scratch")
        if removed_total > 0:
            print(f"✅ {removed_total} record(s) cleared, other models unaffected")
        args.models = args.retrain_models
        args.merge = True

    # Resolve target models
    has_selection = any([
        args.models, args.algorithm, args.dataset,
        args.tag, args.all_enabled,
    ])

    if args.resume or args.merge or args.backtest_only or args.retrain_last:
        if not has_selection:
            args.all_enabled = True
            has_selection = True

    if not has_selection:
        print("❌ Specify models to train")
        print("   Use --models, --algorithm, --dataset, --tag, or --all-enabled")
        return

    if args.resume or args.merge:
        state = RollingState(state_file=state_file)
        if not state.anchor_date and args.resume:
            print("❌ No rolling state to resume")
            return

    targets = resolve_target_models(args)
    if targets is None or not targets:
        print("⚠️  No matching models")
        return

    from quantpits.utils.operator_log import OperatorLog
    with OperatorLog("rolling_train", args=sys.argv[1:]) as oplog:
        if args.backtest_only:
            env.init_qlib()
            params_base = get_base_params()
            # Use module-level re-export so mock.patch can intercept
            run_backtest_only(args, targets, params_base, mode=mode_suffix)
        elif args.predict_only:
            run_predict_only(args, targets, rolling_cfg)
        elif args.cold_start or args.resume or args.merge:
            run_cold_start(args, targets, rolling_cfg)
        elif args.retrain_last:
            run_daily(args, targets, rolling_cfg)
        else:
            run_daily(args, targets, rolling_cfg)

        oplog.set_result({
            "n_targets": len(targets),
            "cold_start": args.cold_start,
            "resume": args.resume,
            "predict_only": args.predict_only,
            "training_method": training_method,
        })

        # Update promote status
        if not args.predict_only and not args.backtest_only:
            try:
                from quantpits.scripts.deep_analysis.promote_config import update_promote_status
                update_promote_status(ROOT_DIR, model_names=list(targets.keys()))
            except Exception:
                pass


# ==========================================================================
# Backward-compatible re-exports (for tests and legacy imports)
# Must be before __main__ guard so flow functions can resolve bare names.
# ==========================================================================
from quantpits.scripts.rolling.strategy_slide import (
    generate_rolling_windows,
    parse_step_to_relativedelta,
    train_window_model,
    train_window_model_isolated,
    predict_with_latest_model,
    _repair_truncated_prediction,
)
from quantpits.scripts.rolling.orchestration import (
    run_model_windows,
    concatenate_rolling_predictions,
    save_rolling_records,
    _filter_pred_to_test_segment,
)
from quantpits.scripts.rolling.backtest import (
    run_combined_backtest,
    run_backtest_only,
)
from quantpits.scripts.rolling.state import RollingState
from quantpits.utils.train_utils import resolve_target_models


if __name__ == "__main__":
    main()
