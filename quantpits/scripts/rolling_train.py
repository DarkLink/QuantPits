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

import sys
import argparse

from quantpits.utils.constants import MONTHS_PER_YEAR


# ==========================================================================
# Strategy dispatch
# ==========================================================================

def _get_strategy(training_method, ctx=None):
    """Resolve strategy module, key suffix, and state file path.

    Args:
        training_method: 'slide' or 'cpcv'.

    Returns:
        (strategy_module, mode_suffix, state_file_path)
    """
    if training_method == 'cpcv':
        from quantpits.scripts.rolling import strategy_cpcv as st
        state_file = (str(ctx.data_path('rolling_state_cpcv.json'))
                      if ctx is not None else _legacy_rolling_paths()[1])
        return st, 'cpcv_rolling', state_file
    else:
        from quantpits.scripts.rolling import strategy_slide as st
        state_file = (str(ctx.data_path('rolling_state.json'))
                      if ctx is not None else _legacy_rolling_paths()[0])
        return st, 'rolling', state_file


def _legacy_rolling_paths():
    """Resolve legacy path constants only after runtime activation."""
    from quantpits.utils.train_utils import ROLLING_STATE_FILE, ROLLING_STATE_FILE_CPCV
    return ROLLING_STATE_FILE, ROLLING_STATE_FILE_CPCV


# ==========================================================================
# CLI
# ==========================================================================

def build_parser():
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
    ctrl.add_argument('--workspace', default=None,
                      help='Explicit workspace root (otherwise QLIB_WORKSPACE_DIR).')
    ctrl.add_argument('--explain-plan', action='store_true',
                      help='Explain the read-only Rolling execution plan.')
    ctrl.add_argument('--json-plan', action='store_true',
                      help='Render the read-only Rolling plan as one JSON document.')
    ctrl.add_argument('--run-id', default=None,
                      help='Explicit operation identity for real execution.')

    info = parser.add_argument_group('Information')
    info.add_argument('--show-state', action='store_true',
                      help='Show rolling state')
    info.add_argument('--clear-state', action='store_true',
                      help='Clear rolling state')

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def resolve_target_models(args, registry=None):
    """Resolve target models (delegates to shared train_utils implementation)."""
    from quantpits.utils.train_utils import resolve_target_models as _resolve
    return _resolve(args, registry=registry)


def get_base_params(workspace_root=None, strict_calendar=False):
    """Get workspace base params (market, benchmark, etc.)."""
    if workspace_root is None:
        from quantpits.utils import env as runtime_env
        workspace_root = runtime_env.ROOT_DIR
    from quantpits.utils.config_loader import load_workspace_config
    config = load_workspace_config(workspace_root)

    from qlib.data import D
    try:
        last_trade_date = D.calendar(future=False)[-1:][0]
        anchor_date = last_trade_date.strftime('%Y-%m-%d')
    except Exception as e:
        if strict_calendar:
            from quantpits.rolling.errors import RollingWindowResolutionError
            raise RollingWindowResolutionError(
                "cannot resolve authoritative Qlib trading anchor: %s" % e
            )
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


def _backtest_not_run_result(model_keys, message):
    """Return a truthful precondition failure when --backtest had no records."""
    model_results = [
        {
            "model_key": model_key,
            "recorder_id": None,
            "status": "failed",
            "stage": "recorder_lookup",
            "reason_code": "rolling_backtest_recorder_unavailable",
            "message": message,
            "did_execute": False,
            "backtest_start": None,
            "backtest_end": None,
        }
        for model_key in model_keys
    ]
    return {
        "status": "failed",
        "reason_code": "rolling_backtest_precondition_failed",
        "message": message,
        "did_execute": False,
        "n_requested": len(model_results),
        "n_attempted": 0,
        "n_succeeded": 0,
        "n_failed": len(model_results),
        "model_results": model_results,
    }


def _consume_backtest_result(result, primary_action, requested_models=()):
    """Bind an embedded --backtest outcome to its already-run primary action."""
    if not _is_authoritative_batch_result(result, requested_models):
        message = "backtest returned no authoritative batch result"
        model_results = [
            {
                "model_key": model_key,
                "recorder_id": None,
                "status": "failed",
                "stage": "backtest",
                "reason_code": "rolling_backtest_execution_failed",
                "message": message,
                "did_execute": True,
                "backtest_start": None,
                "backtest_end": None,
            }
            for model_key in requested_models
        ]
        result = {
            "status": "failed",
            "reason_code": "rolling_backtest_execution_failed",
            "message": message,
            "did_execute": True,
            "n_requested": len(model_results),
            "n_attempted": len(model_results),
            "n_succeeded": 0,
            "n_failed": len(model_results),
            "model_results": model_results,
        }
    outcome = dict(result)
    if outcome["status"] == "failed":
        outcome["message"] = (
            "%s may already have persisted state or records; backtest failed: %s"
            % (primary_action, result.get("message") or "unknown failure")
        )
        # The command did execute even when the backtest itself failed before
        # reaching Qlib, because the primary action has already run.
        outcome["did_execute"] = True
    else:
        outcome["message"] = (
            "%s completed; %s"
            % (primary_action, result.get("message") or "backtest completed")
        )
    return outcome


def _requested_backtest_models(targets, combined_records=()):
    """Keep every selected target, then explicit direct-legacy additions."""
    requested = list(targets)
    requested.extend(
        model_key for model_key in combined_records if model_key not in targets
    )
    return requested


def run_cold_start(args, targets, rolling_cfg, resolved=None):
    """Cold start / merge / resume: Model-First loop with subprocess isolation.

    Supports both slide and CPCV strategies via training_method config.
    """
    from quantpits.utils.train_utils import print_model_table
    # NOTE: orchestration/backtest/state functions are referenced via
    # module-level names (re-exported at bottom of file) so that
    # mock.patch('rolling_train.<func>') can intercept calls.
    from quantpits.scripts.rolling.memory import deep_cleanup_after_model
    from qlib.config import C

    if resolved is None:
        from quantpits.utils import env as runtime_env
        runtime_env.init_qlib()
        params_base = get_base_params()
        windows = None
        state_file_override = None
    else:
        params_base = dict(resolved.params)
        windows = list(resolved.legacy_windows)
        state_file_override = str(resolved.prepared.ctx.path(resolved.prepared.state.path))
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    qlib_config = C

    training_method = rolling_cfg.get('training_method', 'slide')
    st, mode_suffix, state_file = _get_strategy(training_method)
    if state_file_override is not None:
        state_file = state_file_override

    # Generate windows: module-level name for slide (allows mock.patch),
    # strategy function for CPCV.
    if windows is None:
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

    if getattr(args, "dry_run", False) is True:
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

    # Direct legacy calls retain their historical expansion behavior.  The
    # authoritative CLI adapter must stay within the Prepared target tuple.
    if args.merge and resolved is None:
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
    backtest_result = None
    if combined_records:
        save_rolling_records(combined_records, combined_exp_name, anchor_date,
                             mode=mode_suffix, verify_recorders=True, config_payload=params_base)
        if args.backtest:
            requested_models = _requested_backtest_models(
                targets, combined_records,
            )
            backtest_result = run_combined_backtest(
                requested_models, combined_records,
                combined_exp_name, params_base,
            )
    elif args.backtest:
        requested_models = list(targets)
        backtest_result = _backtest_not_run_result(
            requested_models, "training produced no combined record to backtest",
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
    if args.backtest:
        return _consume_backtest_result(
            backtest_result, "rolling training", requested_models,
        )


def run_daily(args, targets, rolling_cfg, resolved=None):
    """Daily mode: detect & train new windows. Model-First + subprocess.

    Falls back to predict-only when all windows are up to date.
    """
    # NOTE: orchestration/backtest/state functions are referenced via
    # module-level names (re-exported at bottom of file) so that
    # mock.patch('rolling_train.<func>') can intercept calls.
    from quantpits.scripts.rolling.memory import deep_cleanup_after_model
    from qlib.config import C

    if resolved is None:
        from quantpits.utils import env as runtime_env
        runtime_env.init_qlib()
        params_base = get_base_params()
        windows = None
        state_file_override = None
    else:
        params_base = dict(resolved.params)
        windows = list(resolved.legacy_windows)
        state_file_override = str(resolved.prepared.ctx.path(resolved.prepared.state.path))
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    qlib_config = C

    training_method = rolling_cfg.get('training_method', 'slide')
    st, mode_suffix, state_file = _get_strategy(training_method)
    if state_file_override is not None:
        state_file = state_file_override

    state = RollingState(state_file=state_file)

    if not state.anchor_date:
        print("❌ No rolling state — run --cold-start first")
        return

    # Generate windows: use module-level name for slide (allows mock.patch
    # interception in tests), strategy function for CPCV.
    if windows is None:
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

        backtest_result = None
        if combined_records:
            save_rolling_records(combined_records, combined_exp_name, anchor_date,
                                 mode=mode_suffix, verify_recorders=True, config_payload=params_base)
            if args.backtest:
                requested_models = _requested_backtest_models(
                    targets, combined_records,
                )
                backtest_result = run_combined_backtest(
                    requested_models, combined_records,
                    combined_exp_name, params_base,
                )
        elif args.backtest:
            requested_models = list(targets)
            backtest_result = _backtest_not_run_result(
                requested_models,
                "daily update produced no combined record to backtest",
            )

        print(f"\n✅ Rolling update complete ({len(new_windows)} new windows)")
        if args.backtest:
            return _consume_backtest_result(
                backtest_result, "daily update", requested_models,
            )

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

        backtest_result = None
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
                    backtest_result = run_combined_backtest(
                        model_names, combined_records,
                        combined_exp_name, params_base,
                    )
            elif args.backtest:
                backtest_result = _backtest_not_run_result(
                    model_names,
                    "daily prediction produced no combined record to backtest",
                )
        elif args.backtest:
            model_names = list(targets)
            backtest_result = _backtest_not_run_result(
                model_names, "daily prediction produced no prediction to backtest",
            )
        if args.backtest:
            return _consume_backtest_result(
                backtest_result, "daily prediction", model_names,
            )


def run_predict_only(args, targets, rolling_cfg, resolved=None):
    """Predict-only mode: use latest window models for current data."""
    # NOTE: orchestration/backtest/state functions are referenced via
    # module-level names (re-exported at bottom of file) so that
    # mock.patch('rolling_train.<func>') can intercept calls.

    if resolved is None:
        from quantpits.utils import env as runtime_env
        runtime_env.init_qlib()
        params_base = get_base_params()
        windows = None
        state_file_override = None
    else:
        params_base = dict(resolved.params)
        windows = list(resolved.legacy_windows)
        state_file_override = str(resolved.prepared.ctx.path(resolved.prepared.state.path))
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    allow_stale = getattr(args, 'allow_stale_predict', False)

    training_method = rolling_cfg.get('training_method', 'slide')
    st, mode_suffix, state_file = _get_strategy(training_method)
    if state_file_override is not None:
        state_file = state_file_override

    state = RollingState(state_file=state_file)
    if not state.anchor_date:
        print("❌ No rolling state — run --cold-start first")
        return {
            "status": "failed",
            "reason_code": "rolling_state_precondition_failed",
            "message": "predict-only requires Rolling state with anchor",
            "did_execute": False,
        }

    # Generate windows: module-level for slide (mock.patch compat), strategy for CPCV
    if windows is None:
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
        backtest_result = None
        if combined_records:
            save_rolling_records(combined_records, combined_exp_name,
                                 anchor_date, mode=mode_suffix, verify_recorders=True, config_payload=params_base)
            if args.backtest:
                backtest_result = run_combined_backtest(
                    model_names, combined_records,
                    combined_exp_name, params_base,
                )
        elif args.backtest:
            backtest_result = _backtest_not_run_result(
                model_names, "prediction produced no combined record to backtest",
            )
        if args.backtest:
            return _consume_backtest_result(
                backtest_result, "prediction", model_names,
            )
        return {
            "status": "success",
            "reason_code": "legacy_partial_visibility",
            "message": "generated predictions for %s model(s)" % len(extra_preds),
            "did_execute": True,
        }
    if args.backtest:
        model_names = list(targets)
        return _consume_backtest_result(_backtest_not_run_result(
            model_names, "predict-only generated no prediction to backtest",
        ), "prediction", model_names)
    return {
        "status": "skipped",
        "reason_code": "rolling_action_skipped",
        "message": "predict-only generated no predictions",
        "did_execute": True,
    }


# ==========================================================================
# Main
# ==========================================================================

def _render_information_error(args, code, message):
    """Keep JSON-plan stdout machine-readable during the 28A migration."""
    if getattr(args, "json_plan", False):
        import json
        print(json.dumps({
            "error": {"code": code, "message": message},
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print("❌ [%s] %s" % (code, message), file=sys.stderr)


def _render_prepared_rolling_plan(args, prepared):
    """Render the one authoritative public payload for every plan flag."""
    from quantpits.rolling.command import prepared_plan_json, render_prepared_plan
    if getattr(args, "json_plan", False):
        import json
        print(json.dumps(prepared_plan_json(prepared), ensure_ascii=False,
                         indent=2, sort_keys=True))
        return 0
    print(render_prepared_plan(prepared))
    return 0


def _render_strict_state(prepared):
    inspection = prepared.state
    print("\n📋 Rolling state inspection:")
    print("  Status: %s" % inspection.status)
    print("  Path: %s" % inspection.path)
    if inspection.fingerprint:
        print("  Fingerprint: %s" % inspection.fingerprint)
    if inspection.anchor:
        print("  Anchor: %s" % inspection.anchor)
    if inspection.training_method:
        print("  Training method: %s" % inspection.training_method)
    if inspection.status == "valid_legacy":
        print("  Completed windows: %s" % inspection.completed_windows)
        print("  Completed model×window units: %s" % inspection.completed_units)
        print("  Resume identity: legacy_partial")
    return 0

def _main_impl(args):

    ctx = getattr(args, '_workspace_context', None)
    if ctx is None:
        from quantpits.rolling.command import resolve_workspace_context
        ctx = resolve_workspace_context(getattr(args, 'workspace', None))

    # Load rolling config first (needed for training_method-aware state commands)
    from quantpits.utils.config_loader import load_rolling_config
    rolling_cfg = load_rolling_config(ctx.root)
    if rolling_cfg is None and not (args.show_state or args.clear_state):
        return _render_information_error(
            args, "rolling_config_missing",
            "config/rolling_config.yaml not found",
        )

    # Determine effective training_method (config + CLI override)
    training_method = (rolling_cfg or {}).get('training_method', 'slide')
    if args.training_method is not None:
        training_method = args.training_method
        if rolling_cfg:
            if args.training_method != rolling_cfg.get('training_method', 'slide'):
                stream = sys.stderr if getattr(args, 'json_plan', False) else sys.stdout
                print(f"  ℹ️  --training-method={args.training_method} overrides config",
                      file=stream)
            rolling_cfg['training_method'] = training_method
    if (getattr(args, "dry_run", False) is True or
            getattr(args, "explain_plan", False) is True or
            getattr(args, "json_plan", False) is True):
        prepared = getattr(args, '_prepared_rolling_run', None)
        if prepared is None:
            from quantpits.rolling.command import (
                options_from_namespace, prepare_rolling_run,
            )
            prepared = prepare_rolling_run(
                ctx, options_from_namespace(args), tuple(sys.argv[1:]),
            )
        return _render_prepared_rolling_plan(args, prepared)
    _, mode_suffix, state_file = _get_strategy(training_method, ctx=ctx)

    # Info commands (now training-method-aware)
    if args.show_state:
        RollingState(state_file=state_file, readonly=True).show()
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
        facade_result = None
        if args.backtest_only:
            from quantpits.utils import env as runtime_env
            runtime_env.init_qlib()
            params_base = get_base_params()
            # Use module-level re-export so mock.patch can intercept
            facade_result = run_backtest_only(
                args, targets, params_base, mode=mode_suffix,
            )
        elif args.predict_only:
            facade_result = run_predict_only(args, targets, rolling_cfg)
        elif args.cold_start or args.resume or args.merge:
            facade_result = run_cold_start(args, targets, rolling_cfg)
        elif args.retrain_last:
            facade_result = run_daily(args, targets, rolling_cfg)
        else:
            facade_result = run_daily(args, targets, rolling_cfg)

        result_summary = {
            "n_targets": len(targets),
            "cold_start": args.cold_start,
            "resume": args.resume,
            "predict_only": args.predict_only,
            "training_method": training_method,
        }
        if isinstance(facade_result, dict):
            result_summary.update({
                key: facade_result[key] for key in (
                    "status", "reason_code", "message", "did_execute",
                    "n_requested", "n_attempted", "n_succeeded", "n_failed",
                ) if key in facade_result
            })
        oplog.set_result(result_summary)

        if (isinstance(facade_result, dict) and
                facade_result.get("status") == "failed"):
            _render_information_error(
                args,
                facade_result.get("reason_code") or "rolling_execution_failed",
                facade_result.get("message") or "Rolling action failed",
            )
            return 2

        # Update promote status
        if not args.predict_only and not args.backtest_only:
            try:
                from quantpits.scripts.deep_analysis.promote_config import update_promote_status
                update_promote_status(str(ctx.root), model_names=list(targets.keys()))
            except Exception:
                pass


def _activate_legacy_workspace(ctx):
    """Bind legacy globals and backends to exactly the Prepared context."""
    import os

    os.environ['QLIB_WORKSPACE_DIR'] = str(ctx.root)
    os.environ['QLIB_DATA_DIR'] = str(ctx.qlib_data_dir)
    os.environ['QLIB_REGION'] = str(ctx.qlib_region)
    os.environ['MLFLOW_TRACKING_URI'] = ctx.mlflow_uri
    try:
        from quantpits.utils import env as runtime_env
        runtime_env.set_root_dir(str(ctx.root))
        runtime_env.QLIB_DATA_DIR = str(ctx.qlib_data_dir)
        runtime_env.QLIB_REGION = str(ctx.qlib_region)
        runtime_env.mlflow_backend = ctx.mlflow_uri
        runtime_env.mlruns_dir = (
            str(ctx.mlruns_dir) if ctx.mlflow_uri.startswith('file://') else None
        )
        runtime_env._qlib_initialized = False
        os.environ['MLFLOW_TRACKING_URI'] = ctx.mlflow_uri
        if ctx.mlflow_uri.startswith('file://'):
            os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
        else:
            os.environ.pop('MLFLOW_ALLOW_FILE_STORE', None)
    except Exception as exc:
        from quantpits.rolling.errors import RollingWorkspaceActivationError
        raise RollingWorkspaceActivationError(
            "cannot activate prepared workspace: %s" % exc
        ) from exc
    return runtime_env


def _safeguard_explicit_workspace(ctx, script_name="Rolling Train"):
    """Display the explicit context without importing the legacy environment."""
    import time

    print("=" * 60)
    print("🚦  SAFEGUARD ACTIVATED  🚦")
    print("[%s] is about to run." % script_name)
    print("Active Workspace: \033[1;31;40m%s\033[0m" % ctx.root.name)
    print("Workspace Path  : %s" % ctx.root)
    print("Qlib Data Dir   : %s" % ctx.qlib_data_dir)
    print("Qlib Region     : %s" % ctx.qlib_region)
    print("=" * 60)
    print("Please confirm. (Press Ctrl+C within 3 seconds to abort if this is the wrong workspace!)")
    time.sleep(3)
    print("Executing...")


def main(argv=None):
    """Run the two-tier Prepared/Resolved Rolling command boundary."""
    from quantpits.rolling.command import (
        options_from_namespace, prepare_rolling_run, resolve_workspace_context,
    )
    from quantpits.rolling.errors import RollingCommandError

    args = parse_args(argv)
    cli_args = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    try:
        ctx = resolve_workspace_context(getattr(args, 'workspace', None))
        options = options_from_namespace(args)
        prepared = prepare_rolling_run(ctx, options, cli_args)
    except RollingCommandError as exc:
        return _render_information_error(args, exc.code, str(exc)) or exc.exit_code
    setattr(args, '_workspace_context', ctx)
    setattr(args, '_prepared_rolling_run', prepared)
    information_route = (
        args.show_state or getattr(args, 'dry_run', False) is True or
        getattr(args, 'explain_plan', False) is True or
        getattr(args, 'json_plan', False) is True
    )
    if information_route:
        if args.show_state and not (
                getattr(args, 'dry_run', False) or
                getattr(args, 'explain_plan', False) or
                getattr(args, 'json_plan', False)):
            return _render_strict_state(prepared)
        return _main_impl(args)

    from quantpits.training.lease import TrainingExecutionLease
    from quantpits.training.errors import TrainingCommandError

    _safeguard_explicit_workspace(ctx)
    lease = TrainingExecutionLease.for_workspace(ctx)
    run_id = getattr(args, 'run_id', None)
    if not isinstance(run_id, str) or not run_id.strip():
        import uuid
        run_id = "rolling-%s" % uuid.uuid4().hex[:12]
    try:
        lease.acquire(run_id=run_id)
    except TrainingCommandError as exc:
        _render_information_error(args, exc.code, str(exc))
        return exc.exit_code
    try:
        from quantpits.rolling.legacy import (
            LegacyRollingExecutionAdapter,
            recheck_prepared_inputs,
            validate_prepared_write_paths,
        )
        from quantpits.rolling.windows import resolve_rolling_run

        recheck_prepared_inputs(prepared)
        validate_prepared_write_paths(prepared)
        runtime_env = _activate_legacy_workspace(ctx)
        resolved = None
        if prepared.options.action != "clear_state":
            try:
                runtime_env.init_qlib()
            except Exception as exc:
                from quantpits.rolling.errors import RollingExecutionError
                raise RollingExecutionError(
                    "Qlib initialization failed: %s" % exc
                ) from exc
            params = get_base_params(workspace_root=ctx.root, strict_calendar=True)
            resolved = resolve_rolling_run(prepared, params)
        adapter = LegacyRollingExecutionAdapter(sys.modules[__name__])
        outcome = adapter.execute(args, prepared, resolved, run_id)
        if outcome.status == "failed":
            _render_information_error(
                args, outcome.reason_code or "rolling_execution_failed",
                outcome.message or "Rolling action failed",
            )
            return 2
        if outcome.status == "skipped":
            print("ℹ️  [%s] %s" % (outcome.reason_code, outcome.message))
        return 0
    except RollingCommandError as exc:
        return _render_information_error(args, exc.code, str(exc)) or exc.exit_code
    finally:
        lease.release()


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
    _is_authoritative_batch_result,
    run_combined_backtest,
    run_backtest_only,
)
from quantpits.scripts.rolling.state import RollingState


if __name__ == "__main__":
    sys.exit(main() or 0)
