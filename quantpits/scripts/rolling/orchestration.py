"""
Shared Rolling Orchestration Module

Strategy-agnostic orchestration logic used by rolling_train.py:
  - run_model_windows():       Model-first loop over windows
  - concatenate_rolling_predictions(): Stitch per-window predictions
  - _filter_pred_to_test_segment():   Clip predictions to test segment
  - save_rolling_records():    Write to latest_train_records.json

Strategy-specific functions (window generation, per-window training,
predict_latest, truncation repair) are provided by strategy_*.py modules.
"""

import pandas as pd
from datetime import datetime

from quantpits.scripts.rolling.memory import (
    cleanup_after_window,
    check_memory_pressure,
    deep_cleanup_after_model,
)


# ==========================================================================
# Window-level orchestration (Model-First loop)
# ==========================================================================

def run_model_windows(model_name, model_info, windows, state,
                      params_base, experiment_name, qlib_config,
                      train_fn,
                      no_pretrain=False, dry_run=False, cache_size=None):
    """Train one model across all windows (Model-First inner loop).

    For each window:
      1. Check if already completed (supports --resume)
      2. Call train_fn in subprocess isolation
      3. Mark completion + lightweight cleanup

    Args:
        model_name: Model identifier.
        model_info: Model config dict (must contain 'yaml_file').
        windows: List of window dicts from strategy.generate_windows().
        state: RollingState instance.
        params_base: Base params dict (market, benchmark, etc.).
        experiment_name: MLflow experiment name for per-window runs.
        qlib_config: Qlib config object (C).
        train_fn: Strategy's train_window_isolated function.
        no_pretrain: Skip pretrained model loading.
        dry_run: Print plan only, don't train.

    Returns:
        int: Number of windows actually trained this invocation.
    """
    yaml_file = model_info['yaml_file']
    trained_count = 0
    total_windows = len(windows)

    print(f"\n{'='*60}")
    print(f"🔧 Model: {model_name} ({total_windows} windows)")
    print(f"{'='*60}")

    for window in windows:
        widx = window['window_idx']

        # Check if already completed
        if state.is_window_model_done(widx, model_name):
            print(f"  ✅ Window {widx}: already completed, skipping")
            continue

        if dry_run:
            print(f"  🔍 Window {widx}: "
                  f"Test[{window['test_start']}, {window['test_end']}] (dry-run)")
            continue

        # Memory safety valve
        check_memory_pressure(f"{model_name}|W{widx}")

        # Train in subprocess isolation (strategy-specific)
        result = train_fn(
            qlib_config=qlib_config,
            model_name=model_name,
            yaml_file=yaml_file,
            window=window,
            params_base=params_base,
            experiment_name=experiment_name,
            no_pretrain=no_pretrain,
            cache_size=cache_size,
        )

        if result['success']:
            state.mark_window_model_done(widx, model_name, result['record_id'])
            trained_count += 1

            perf = result.get('performance', {})
            ic = perf.get('IC_Mean')
            icir = perf.get('ICIR')
            perf_str = ""
            if ic is not None:
                perf_str += f" IC={ic:.4f}"
            if icir is not None:
                perf_str += f" ICIR={icir:.4f}"
            print(f"  ✅ Window {widx}: Done{perf_str}")
        else:
            print(f"  ❌ Window {widx}: {result.get('error', 'Unknown error')}")

        # Level 1 cleanup (main-process side; subprocess already exited)
        cleanup_after_window(model_name, widx)

    return trained_count


# ==========================================================================
# Prediction stitching
# ==========================================================================

def _filter_pred_to_test_segment(pred, window):
    """Clip pred to the window's test segment [test_start, test_end].

    Strategy-agnostic: CPCV windows also provide test_start/test_end.
    """
    dates = pred.index.get_level_values('datetime')
    mask = (dates >= pd.Timestamp(window['test_start'])) & \
           (dates <= pd.Timestamp(window['test_end']))
    return pred[mask]


def concatenate_rolling_predictions(state, model_names, rolling_exp_name,
                                    combined_exp_name, anchor_date,
                                    windows, extra_preds=None,
                                    targets=None, params_base=None,
                                    repair_fn=None):
    """Stitch per-window pred.pkl files into a continuous time series.

    For each model:
      1. Detect & repair truncated historical window predictions (strategy-specific)
      2. Load pred.pkl from each window's MLflow recorder
      3. Clip to test segment only (prevents train-set leakage)
      4. pd.concat (dates don't overlap; keep='last' handles any overlap)
      5. Save to Rolling_Combined experiment recorder

    Args:
        state: RollingState instance.
        model_names: List of model names to concatenate.
        rolling_exp_name: Per-window MLflow experiment name.
        combined_exp_name: Combined (stitched) MLflow experiment name.
        anchor_date: Anchor date string.
        windows: Full rolling window list (for test-segment clipping).
        extra_preds: Optional {model_name: DataFrame} from predict-only mode.
        targets: Optional {model_name: model_info} for repair.
        params_base: Optional base params for repair.
        repair_fn: Strategy's repair_truncated function, or None.

    Returns:
        dict: {model_name: combined_record_id}
    """
    from qlib.workflow import R

    print(f"\n{'='*60}")
    print("📦 Stitching Rolling Predictions (test segments only)")
    print(f"{'='*60}")

    # Build window_idx -> window lookup table
    window_map = {w['window_idx']: w for w in windows}

    combined_records = {}
    repaired_total = 0

    for model_name in model_names:
        completions = state.get_completed_record_ids(model_name)
        if not completions:
            print(f"  [{model_name}] No completed windows, skipping")
            continue

        model_info = targets.get(model_name) if targets else None
        last_widx = completions[-1]['window_idx']

        print(f"\n  [{model_name}] Stitching {len(completions)} windows...")

        all_preds = []
        window_ic_values = []   # Collect per-window IC for aggregated ICIR
        window_rank_ic_values = []  # Collect per-window Rank IC
        for comp in completions:
            widx = comp['window_idx']
            try:
                rec = R.get_recorder(
                    recorder_id=comp['record_id'],
                    experiment_name=rolling_exp_name,
                )

                # For non-last windows, check truncation and auto-repair
                # (last window truncation is handled by predict-only, not here)
                pred = None
                if widx != last_widx and model_info and params_base and repair_fn:
                    pred, repaired = repair_fn(
                        model_name, model_info, comp, window_map,
                        rolling_exp_name, params_base,
                    )
                    if repaired:
                        repaired_total += 1

                if pred is None:
                    pred = rec.load_object("pred.pkl")

                # Normalize to single-column score DataFrame
                if isinstance(pred, pd.DataFrame) and 'score' in pred.columns:
                    pred = pred[['score']]
                elif isinstance(pred, pd.Series):
                    pred = pred.to_frame('score')

                # Clip to test segment
                w = window_map.get(widx)
                if w:
                    pred = _filter_pred_to_test_segment(pred, w)

                all_preds.append(pred)
                dates = pred.index.get_level_values('datetime')
                print(f"    Window {widx}: "
                      f"{dates.min().date()} ~ {dates.max().date()}, "
                      f"{len(pred)} rows")

                # Collect per-window IC metrics for aggregated ICIR
                try:
                    raw_metrics = rec.list_metrics()
                    for k, v in raw_metrics.items():
                        if k == 'IC':
                            window_ic_values.append(float(v))
                        elif k == 'Rank IC':
                            window_rank_ic_values.append(float(v))
                except Exception:
                    pass
            except Exception as e:
                print(f"    Window {widx}: FAILED - {e}")

        # Append extra predictions from predict-only mode
        if extra_preds and model_name in extra_preds:
            extra_df = extra_preds[model_name]
            if extra_df is not None and not extra_df.empty:
                if isinstance(extra_df, pd.Series):
                    extra_df = extra_df.to_frame('score')
                elif isinstance(extra_df, pd.DataFrame) and 'score' in extra_df.columns:
                    extra_df = extra_df[['score']]
                elif isinstance(extra_df, pd.DataFrame):
                    extra_df.columns = ['score']

                # Determine filter window for extra_preds
                filter_window = None
                if completions and windows:
                    model_last_widx = completions[-1]['window_idx']
                    global_last_widx = windows[-1]['window_idx']
                    if model_last_widx < global_last_widx:
                        # gap + extra_preds → stale predict mode, extend range
                        filter_window = windows[-1]
                    else:
                        filter_window = window_map.get(model_last_widx)
                elif completions:
                    last_widx_c = completions[-1]['window_idx']
                    filter_window = window_map.get(last_widx_c)
                if filter_window:
                    extra_df = _filter_pred_to_test_segment(extra_df, filter_window)

                all_preds.append(extra_df)
                dts = extra_df.index.get_level_values('datetime')
                print(f"    Extra Pred_Only: {dts.min().date()} ~ {dts.max().date()}, {len(extra_df)} rows")

        if not all_preds:
            print(f"  [{model_name}] No valid prediction data")
            continue

        # Concatenate
        combined_pred = pd.concat(all_preds)
        # Deduplicate: overlapping regions keep latest (extra_preds / later window)
        combined_pred = combined_pred[~combined_pred.index.duplicated(keep='last')]
        combined_pred = combined_pred.sort_index()

        dates = combined_pred.index.get_level_values('datetime')
        print(f"  [{model_name}] Stitched result: "
              f"{dates.min().date()} ~ {dates.max().date()}, "
              f"{len(combined_pred)} rows")

        # Save to Combined experiment
        with R.start(experiment_name=combined_exp_name):
            R.set_tags(
                model=model_name,
                mode='rolling_combined',
                anchor_date=anchor_date,
                n_windows=len(completions),
            )
            R.save_objects(**{"pred.pkl": combined_pred})

            # Save aggregated IC/ICIR metrics from per-window statistics
            ic_metrics = {}
            if window_ic_values:
                ic_arr = [v for v in window_ic_values if v is not None]
                if ic_arr:
                    import numpy as np
                    ic_mean = float(np.mean(ic_arr))
                    ic_std = float(np.std(ic_arr))
                    ic_metrics['IC'] = ic_mean
                    ic_metrics['ICIR'] = ic_mean / ic_std if ic_std > 0 else 0.0
            if window_rank_ic_values:
                ric_arr = [v for v in window_rank_ic_values if v is not None]
                if ric_arr:
                    import numpy as np
                    ric_mean = float(np.mean(ric_arr))
                    ric_std = float(np.std(ric_arr))
                    ic_metrics['Rank IC'] = ric_mean
                    ic_metrics['Rank ICIR'] = ric_mean / ric_std if ric_std > 0 else 0.0
            if ic_metrics:
                R.log_metrics(**ic_metrics)

            combined_rid = R.get_recorder().id

        combined_records[model_name] = combined_rid
        print(f"  [{model_name}] Combined Recorder: {combined_rid}")

    if repaired_total > 0:
        print(f"\n  🔧 Repaired {repaired_total} truncated historical window predictions")

    return combined_records


# ==========================================================================
# Record saving
# ==========================================================================

def save_rolling_records(combined_records, combined_exp_name, anchor_date,
                         mode='rolling', verify_recorders=False, config_payload=None):
    """Save rolling training records to unified latest_train_records.json.

    Uses model@<mode> key format (e.g. model@rolling, model@cpcv_rolling).
    Merges via merge_train_records() to preserve entries from other modes.

    Args:
        combined_records: {model_name: combined_record_id}
        combined_exp_name: MLflow experiment name.
        anchor_date: Anchor date string.
        mode: Key suffix — 'rolling' (slide) or 'cpcv_rolling' (CPCV).
    """
    from quantpits.utils.train_utils import (
        make_model_key, merge_train_records, RECORD_OUTPUT_FILE,
    )

    # Convert to model@mode keys
    rolling_models = {}
    for name, rid in combined_records.items():
        rolling_key = make_model_key(name, mode)
        rolling_models[rolling_key] = rid

    # Set the appropriate experiment_name field
    if mode == 'cpcv_rolling':
        exp_field = 'cpcv_rolling_experiment_name'
    else:
        exp_field = 'rolling_experiment_name'

    records = {
        "experiment_name": combined_exp_name,
        exp_field: combined_exp_name,
        "anchor_date": anchor_date,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": rolling_models,
        "model_records": {},
    }

    # The combined recorder is the authoritative rolling output.  Publish a
    # ready V2 entry only when its persisted prediction can be inspected;
    # legacy/mocked callers fall back to the compatibility adapter in merge.
    if verify_recorders:
        from qlib.workflow import R
        from quantpits.training.records import build_model_record_entry
        from quantpits.utils import env
        from quantpits.utils.workspace import fingerprint_value
        operation = "cpcv_rolling_combine" if mode == "cpcv_rolling" else "rolling_combine"
        for key, recorder_id in rolling_models.items():
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=combined_exp_name)
            records["model_records"][key] = build_model_record_entry(
                key=key, operation=operation, experiment_name=combined_exp_name,
                recorder=recorder, requested_anchor=anchor_date,
                dataset_test_end=anchor_date,
                workspace_root=env.ROOT_DIR,
                config_fingerprint=fingerprint_value(
                    config_payload or {"anchor_date": anchor_date, "mode": mode}
                ),
            ).to_dict()

    merge_train_records(records)

    print(f"\n📋 Rolling records merged to: {RECORD_OUTPUT_FILE}")
    print(f"   Key suffix: @{mode}")
    print(f"   Models: {len(rolling_models)}")
    for key, rid in rolling_models.items():
        print(f"   {key}: {rid}")
