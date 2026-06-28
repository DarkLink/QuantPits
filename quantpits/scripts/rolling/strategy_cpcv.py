"""
CPCV Training Strategy for Rolling Windows (Walk-Forward CPCV)

Rolling loop defines test boundaries directly.
CPCV operates purely on the train domain [train_start, train_end]
with n_test_groups=0 — K-fold cross-validation only, no internal test set.

Exports the standard strategy interface:
  generate_windows, train_window, train_window_isolated,
  predict_latest, repair_truncated

Key mitigations:
  暗雷1: _make_fold_config() overrides start_time per window (train domain).
  暗雷2: Tier-1 Fail-Fast (config load) — no per-window try/catch needed
         since train domain is always exactly train_years long.
  暗雷3: MLflow recorder ID isolation per (model, window).
  暗雷4: Per-fold GPU cleanup inside train_window().
"""

import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from quantpits.scripts.rolling.strategy_slide import parse_step_to_relativedelta
from quantpits.utils.constants import MONTHS_PER_YEAR


# ==========================================================================
# Internal helpers
# ==========================================================================

def _make_fold_config(train_start, cpcv_cfg):
    """暗雷1防护: 构造独立的 config dict，显式覆写 start_time 为 train 域起点。

    compute_cpcv_folds() 内部从 config['start_time'] 读取日历起点。
    如果不覆写，所有窗口都会使用全局 start_time，导致分区混乱。

    n_test_groups 强制为 0：CPCV 只对 train 域做交叉验证，
    test 边界由 Rolling 大循环定义。
    """
    return {
        'start_time': train_start,
        'purged_cv': {
            'n_groups': cpcv_cfg['n_groups'],
            'n_test_groups': 0,  # Walk-Forward CPCV: no internal test set
            'n_val_groups': cpcv_cfg['n_val_groups'],
            'purge_steps': cpcv_cfg['purge_steps'],
            'embargo_steps': cpcv_cfg['embargo_steps'],
        },
    }


# ==========================================================================
# Window generation
# ==========================================================================

def generate_windows(rolling_start, train_years, test_step, anchor_date,
                     cpcv_cfg=None, freq='week', **kwargs):
    """Generate Walk-Forward CPCV rolling windows.

    Rolling loop defines test boundaries directly:
      test_start = T + w*Δ + X years
      test_end   = min(test_start + Δ - 1d, A)

    CPCV operates purely on the train domain [train_start, train_end]
    which is always exactly train_years long. n_test_groups is forced to 0.

    No windows are skipped — the train domain never shrinks.
    Test segments are strictly non-overlapping.

    Args:
        rolling_start: str 'YYYY-MM-DD'.
        train_years: Train domain length (years), always fixed.
        test_step: Rolling step (e.g. '3M', '6M').
        anchor_date: Latest trading date.
        cpcv_cfg: dict with CPCV parameters (n_groups, n_val_groups,
                  purge_steps, embargo_steps). n_test_groups forced to 0.
        freq: 'week' or 'day'.

    Returns:
        list of dict: [{window_idx, train_start, train_end, test_start,
                        test_end, cpcv_folds}]
    """
    from quantpits.utils.train_utils import compute_cpcv_folds

    if cpcv_cfg is None:
        raise ValueError("CPCV strategy requires cpcv_cfg (CPCV parameters)")

    step_delta = parse_step_to_relativedelta(test_step)
    step_months = step_delta.months + step_delta.years * MONTHS_PER_YEAR
    anchor = pd.Timestamp(anchor_date)
    T = pd.Timestamp(rolling_start)

    windows = []
    widx = 0

    while True:
        offset = relativedelta(months=step_months * widx)

        # Rolling loop defines test boundaries
        test_start = T + offset + relativedelta(years=train_years)
        if test_start > anchor:
            break

        test_end = test_start + step_delta - relativedelta(days=1)
        if test_end > anchor:
            test_end = anchor

        # Train domain: exactly train_years before test_start
        train_end = test_start - relativedelta(days=1)
        train_start = train_end - relativedelta(years=train_years) + relativedelta(days=1)

        train_start_str = train_start.strftime('%Y-%m-%d')
        train_end_str = train_end.strftime('%Y-%m-%d')

        # CPCV purely on the train domain (n_test_groups=0 forced)
        tmp_config = _make_fold_config(train_start_str, cpcv_cfg)
        folds_result = compute_cpcv_folds(train_end_str, tmp_config, freq=freq)

        windows.append({
            'window_idx': widx,
            'train_start': train_start_str,
            'train_end': train_end_str,
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'cpcv_folds': folds_result['folds'],
        })

        widx += 1

    return windows


# ==========================================================================
# Single-window K-fold CPCV training
# ==========================================================================

def train_window(model_name, yaml_file, window, params_base,
                 experiment_name, no_pretrain=False, cache_size=None, **kwargs):
    """Train K CPCV folds for a single rolling window.

    For each fold:
      1. inject_config_for_fold() — fold-specific dates + PurgedDataset swap
      2. model.fit(dataset), model.predict(dataset)
      3. Per-fold GPU cleanup (暗雷4)

    After all folds: ensemble-average predictions, save fold models +
    ensemble pred.pkl, run backtest records.

    Args:
        model_name: Model identifier.
        yaml_file: Path to Qlib workflow YAML.
        window: Window dict from generate_windows() (must contain cpcv_folds).
        params_base: Base params dict.
        experiment_name: MLflow experiment name.
        no_pretrain: Skip pretrained model loading.

    Returns:
        dict: {success, record_id, performance, error, n_folds, fold_scores}
    """
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from quantpits.utils.train_utils import inject_config_for_fold

    result = {
        'success': False,
        'record_id': None,
        'performance': None,
        'error': None,
        'n_folds': 0,
        'fold_scores': [],
    }

    if not os.path.exists(yaml_file):
        result['error'] = f"YAML config not found: {yaml_file}"
        print(f"!!! Warning: {yaml_file} not found, skipping...")
        return result

    folds = window.get('cpcv_folds', [])
    if not folds:
        result['error'] = "No CPCV folds in window"
        return result

    widx = window['window_idx']
    print(f"\n>>> CPCV Rolling: {model_name} | Window {widx} "
          f"({len(folds)} folds)")
    print(f"    Train: [{window['train_start']}, {window['train_end']}]")
    print(f"    Test:  [{window['test_start']}, {window['test_end']}]")

    # Build per-fold params: test boundaries from rolling loop
    fold_params = dict(params_base)
    fold_params['test_start_time'] = window['test_start']
    fold_params['test_end_time'] = window['test_end']
    fold_params['anchor_date'] = window['test_end']

    fold_predictions = []
    fold_models = []
    fold_scores = []

    # Per-window HandlerCacheManager: K folds share cached DataHandler instances.
    # The cache lives entirely within this subprocess; OS reclaims on exit.
    cache_mgr = None
    if cache_size and cache_size != 0:
        from quantpits.utils.handler_cache import HandlerCacheManager
        cache_mgr = HandlerCacheManager(max_size_mb=cache_size)

    try:
        with R.start(experiment_name=experiment_name):
            R.set_tags(
                model=model_name,
                window_idx=widx,
                mode='cpcv_rolling_train',
                n_folds=len(folds),
                test_start=window['test_start'],
                test_end=window['test_end'],
            )

            for fold_idx, fold in enumerate(folds):
                print(f"\n  --- Fold {fold_idx + 1}/{len(folds)} ---")
                print(f"  Train segments: {fold['train_segments']}")
                print(f"  Valid: [{fold['valid_start_time']}, "
                      f"{fold['valid_end_time']}]")

                # Inject fold-specific config (switches to PurgedDataset)
                task_config = inject_config_for_fold(
                    yaml_file, fold_params, fold,
                    model_name=model_name, no_pretrain=no_pretrain,
                )

                model_cfg = task_config['task']['model']
                model = init_instance_by_config(model_cfg)
                model.topk = params_base.get('topk', getattr(model, 'topk', 20))
                model.n_drop = params_base.get('n_drop',
                                               getattr(model, 'n_drop', 3))

                # CPCV: disable multiprocessing DataLoader
                if hasattr(model, 'n_jobs'):
                    model.n_jobs = 0

                dataset_cfg = task_config['task']['dataset']

                # Use handler cache when available (CPCV K-fold reuse)
                if cache_mgr is not None:
                    from quantpits.utils.handler_cache import _has_temporal_processors
                    handler_cfg = dataset_cfg['kwargs']['handler']
                    if _has_temporal_processors(handler_cfg):
                        extra_ctx = {'valid': dataset_cfg['kwargs']['segments']['valid']}
                    else:
                        extra_ctx = None
                    dataset = cache_mgr.create_dataset(dataset_cfg, extra_context=extra_ctx)
                else:
                    dataset = init_instance_by_config(dataset_cfg)
                dataset.setup_data()

                # Train
                print(f"  Training fold {fold_idx}...")
                t0 = time.time()
                model.fit(dataset=dataset)
                train_duration = time.time() - t0

                # Predict on test set
                print(f"  Predicting fold {fold_idx}...")
                pred = model.predict(dataset=dataset)
                fold_predictions.append(pred)
                fold_models.append(model)

                # Fold-level validation IC
                try:
                    orig_test_seg = dataset.segments['test']
                    dataset.segments['test'] = dataset.segments['valid']
                    val_pred = model.predict(dataset=dataset)
                    val_label = dataset.prepare("test", col_set="label")
                    dataset.segments['test'] = orig_test_seg

                    if isinstance(val_pred, pd.DataFrame) and 'label' in val_pred.columns:
                        val_label = val_pred['label']
                        val_pred = val_pred['score']
                    elif isinstance(val_label, pd.DataFrame):
                        val_label = val_label.iloc[:, 0]
                    elif hasattr(val_label, 'get_index') and hasattr(val_label, '__len__'):
                        idx = val_label.get_index()
                        if hasattr(val_label, 'samplers'):
                            for s in val_label.samplers:
                                if hasattr(s, 'idx_map') and isinstance(s.idx_map, np.ndarray):
                                    s.idx_map = s.idx_map.astype(int)
                        elif hasattr(val_label, 'idx_map') and isinstance(val_label.idx_map, np.ndarray):
                            val_label.idx_map = val_label.idx_map.astype(int)

                        if hasattr(val_label, 'samplers'):
                            labels_list = []
                            for s in val_label.samplers:
                                if hasattr(s, 'idx_map') and hasattr(s, 'idx_arr') and hasattr(s, 'data_arr'):
                                    rows = s.idx_map[:, 0]
                                    cols = s.idx_map[:, 1]
                                    last_indices = s.idx_arr[rows, cols]
                                    last_indices = np.nan_to_num(last_indices.astype(np.float64), nan=s.nan_idx).astype(int)
                                    if s.data_arr.ndim == 2:
                                        labels_list.append(s.data_arr[last_indices, -1])
                                    else:
                                        labels_list.append(s.data_arr[last_indices])
                            if len(labels_list) == len(val_label.samplers):
                                arr = np.concatenate(labels_list)
                            else:
                                vals = [val_label[i] for i in range(len(val_label))]
                                arr = np.array([float(v.flat[-1]) for v in vals])
                        elif hasattr(val_label, 'idx_map') and hasattr(val_label, 'idx_arr') and hasattr(val_label, 'data_arr'):
                            rows = val_label.idx_map[:, 0]
                            cols = val_label.idx_map[:, 1]
                            last_indices = val_label.idx_arr[rows, cols]
                            last_indices = np.nan_to_num(last_indices.astype(np.float64), nan=val_label.nan_idx).astype(int)
                            if val_label.data_arr.ndim == 2:
                                arr = val_label.data_arr[last_indices, -1]
                            else:
                                arr = val_label.data_arr[last_indices]
                        else:
                            vals = [val_label[i] for i in range(len(val_label))]
                            arr = np.array([float(v.flat[-1]) for v in vals])
                        val_label = pd.Series(arr, index=idx)

                    df = pd.DataFrame({"pred": val_pred, "label": val_label})
                    df = df.dropna()
                    if not df.empty:
                        grp_key = "datetime" if "datetime" in df.index.names else df.index.names[0]
                        ic = df.groupby(level=grp_key).apply(
                            lambda x: x["pred"].corr(x["label"], method="pearson")
                        )
                        mean_ic = float(ic.mean())
                        fold_scores.append(mean_ic)
                        print(f"  Fold {fold_idx} Validation IC: {mean_ic:.4f}")
                    else:
                        fold_scores.append(None)
                except Exception as e:
                    print(f"  ⚠️  Failed to compute validation IC for fold {fold_idx}: {e}")
                    fold_scores.append(None)

                print(f"  Fold {fold_idx} done in {train_duration:.1f}s")

                # ============================================================
                # 暗雷4: Per-fold GPU cleanup (must be inside the fold loop!)
                # Same pattern as train_cpcv_model() in train_utils.py:1682-1692
                # ============================================================
                del model
                del dataset
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass
                gc.collect()

            # ---- Ensemble average across folds ----
            all_preds_df = pd.concat(fold_predictions, axis=1)
            final_pred = all_preds_df.mean(axis=1, skipna=True)

            nan_frac = final_pred.isna().mean()
            if nan_frac > 0:
                print(f"  ⚠️  {nan_frac:.2%} of final predictions are NaN "
                      f"(universal suspensions across all folds)")

            # ---- Save fold models + ensemble pred ----
            recorder = R.get_recorder()
            recorder.save_objects(**{
                f"model_fold_{fi}.pkl": fm for fi, fm in enumerate(fold_models)
            })

            # Build backtest dataset from first fold's config
            backtest_task_config = inject_config_for_fold(
                yaml_file, fold_params, folds[0],
                model_name=model_name, no_pretrain=no_pretrain,
            )
            backtest_dataset_cfg = backtest_task_config['task']['dataset']
            backtest_dataset = init_instance_by_config(backtest_dataset_cfg)
            backtest_dataset.setup_data()

            # Run SigAnaRecord + PortAnaRecord with fold_models[0]
            record_cfgs = backtest_task_config['task'].get('record', [])
            for r_cfg in record_cfgs:
                if r_cfg['kwargs'].get('model') == '<MODEL>':
                    r_cfg['kwargs']['model'] = fold_models[0]
                if r_cfg['kwargs'].get('dataset') == '<DATASET>':
                    r_cfg['kwargs']['dataset'] = backtest_dataset
                r_obj = init_instance_by_config(r_cfg, recorder=recorder)
                r_obj.generate()

            # Overwrite pred.pkl with the K-fold ensemble average
            recorder.save_objects(**{"pred.pkl": final_pred})

            # Extract IC & backtest metrics
            performance = {}
            try:
                ic_series = recorder.load_object("sig_analysis/ic.pkl")
                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std and ic_std != 0 else None
                performance["IC_Mean"] = float(ic_mean) if ic_mean is not None else None
                performance["ICIR"] = float(ic_ir) if ic_ir is not None else None
            except Exception:
                pass

            try:
                port_analysis = recorder.load_object(
                    "portfolio_analysis/port_analysis_1week.pkl"
                )
                if isinstance(port_analysis, pd.DataFrame):
                    if "excess_return_without_cost" in port_analysis.index:
                        metrics_row = port_analysis.loc["excess_return_without_cost"]
                        if isinstance(metrics_row, pd.DataFrame):
                            val_col = ("risk" if "risk" in metrics_row.columns
                                       else metrics_row.columns[0])
                            performance["Ann_Excess"] = float(
                                metrics_row.loc["annualized_return", val_col]
                            )
                            performance["Max_DD"] = float(
                                metrics_row.loc["max_drawdown", val_col]
                            )
                            performance["Information_Ratio"] = float(
                                metrics_row.loc["information_ratio", val_col]
                            )
                        else:
                            performance["Ann_Excess"] = float(
                                metrics_row.get("annualized_return", None) or 0
                            )
                            performance["Max_DD"] = float(
                                metrics_row.get("max_drawdown", None) or 0
                            )
                            performance["Information_Ratio"] = float(
                                metrics_row.get("information_ratio", None) or 0
                            )
            except Exception:
                pass

            performance["record_id"] = recorder.id

            result['success'] = True
            result['record_id'] = recorder.id
            result['n_folds'] = len(folds)
            result['fold_scores'] = fold_scores
            result['performance'] = performance

            if cache_mgr is not None:
                print(f"  Handler Cache: {cache_mgr}")
            print(f"  ✅ CPCV window {widx} complete: {len(folds)} folds, "
                  f"recorder={recorder.id}")

    except Exception as e:
        result['error'] = str(e)
        print(f"!!! Error in CPCV rolling train {model_name} W{widx}: {e}")
        import traceback
        traceback.print_exc()

    return result


# ==========================================================================
# Subprocess isolation
# ==========================================================================

def _train_in_subprocess(qlib_config_dict, model_name, yaml_file, window,
                         params_base, experiment_name, no_pretrain,
                         cache_size):
    """Subprocess entry point: re-initialize qlib then train CPCV window.

    The subprocess runs ALL K folds sequentially, then exits.
    OS reclaims all memory (qlib H cache, PyTorch CUDA context,
    DataHandler DataFrames) on exit.

    暗雷4: Within this subprocess, per-fold GPU cleanup is done
    inside train_window()'s fold loop.
    """
    from qlib.config import C
    C.register_from_C(qlib_config_dict)

    return train_window(
        model_name=model_name,
        yaml_file=yaml_file,
        window=window,
        params_base=params_base,
        experiment_name=experiment_name,
        no_pretrain=no_pretrain,
        cache_size=cache_size,
    )


def train_window_isolated(qlib_config, model_name, yaml_file, window,
                          params_base, experiment_name, no_pretrain=False,
                          cache_size=None, **kwargs):
    """CPCV window training in an independent subprocess.

    Same spawn pattern as slide strategy's train_window_isolated.
    """
    import sys
    import multiprocessing as mp
    import concurrent.futures
    from unittest.mock import Mock

    rt_mod = sys.modules.get('rolling_train')
    if rt_mod and hasattr(rt_mod, 'train_window_model') and isinstance(rt_mod.train_window_model, Mock):
        print("  [Mock Detected] calling mocked training in main process")
        return rt_mod.train_window_model(
            model_name=model_name,
            yaml_file=yaml_file,
            window=window,
            params_base=params_base,
            experiment_name=experiment_name,
            no_pretrain=no_pretrain,
        )

    _mp_ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=_mp_ctx) as executor:
        future = executor.submit(
            _train_in_subprocess,
            qlib_config,
            model_name,
            yaml_file,
            window,
            params_base,
            experiment_name,
            no_pretrain,
            cache_size,
        )
        return future.result()


# ==========================================================================
# Predict with latest model (predict-only / daily mode)
# ==========================================================================

def predict_latest(model_name, model_info, state, rolling_exp_name,
                   params_base, anchor_date, windows,
                   allow_stale_predict=False, **kwargs):
    """Predict current data using the latest CPCV window's K fold models.

    With n_test_groups=0, windows are never skipped — the train domain
    is always exactly train_years long. The latest completed window's
    test_end may be before anchor_date (gap = untrained windows).

    Loads all K model_fold_*.pkl from the latest window's recorder,
    has each predict from the window's test_start to anchor_date,
    then ensemble-averages.
    """
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from quantpits.utils.train_utils import inject_config_for_fold

    # Mock detection: if rolling_train.predict_with_latest_model is patched
    # in tests, return the mock result to avoid real qlib calls.
    import sys as _sys
    _rt = _sys.modules.get('rolling_train')
    if _rt and hasattr(_rt, 'predict_with_latest_model'):
        from unittest.mock import Mock
        if isinstance(_rt.predict_with_latest_model, Mock):
            return _rt.predict_with_latest_model(
                model_name, model_info, state, rolling_exp_name,
                params_base, anchor_date, windows,
                allow_stale_predict=allow_stale_predict, **kwargs)

    completions = state.get_completed_record_ids(model_name)
    if not completions:
        print(f"  [{model_name}] No historical CPCV rolling records.")
        return None

    latest = completions[-1]
    latest_widx = latest['window_idx']

    # Find the trained window
    trained_window = next((w for w in windows if w['window_idx'] == latest_widx), None)
    if not trained_window:
        print(f"  [{model_name}] Cannot find window W{latest_widx} in window list")
        return None

    # Detect gap (untrained windows between latest trained and last available)
    last_available_widx = windows[-1]['window_idx'] if windows else latest_widx
    gap = last_available_widx - latest_widx

    # Determine if there's actually a gap: either by window index OR by date
    has_index_gap = gap > 0
    window_test_end = trained_window.get('test_end')
    has_date_gap = (window_test_end is not None
                    and pd.Timestamp(anchor_date) > pd.Timestamp(window_test_end))

    if has_index_gap or has_date_gap:
        if not allow_stale_predict and has_index_gap:
            print(f"  ⛔ [{model_name}] {gap} untrained windows. "
                  f"Skipping. Use --allow-stale-predict or --retrain-models {model_name}")
            return None
        if has_index_gap:
            print(f"  ⚠️  [{model_name}] {gap} untrained windows "
                  f"(trained up to W{latest_widx}, data up to W{last_available_widx})")
        print(f"     Predicting gap: [{window_test_end}] → {anchor_date}")
        effective_test_start = pd.Timestamp(window_test_end) + pd.Timedelta(days=1)
        effective_test_end = anchor_date
    else:
        print(f"  [{model_name}] All windows up to date (W{latest_widx}), "
              f"stitched prediction already covers {anchor_date}. Nothing to predict.")
        return None

    print(f"  [{model_name}] Loading W{latest_widx} fold models for gap prediction "
          f"[{effective_test_start.date()} ~ {effective_test_end}]...")

    try:
        rec = R.get_recorder(
            recorder_id=latest['record_id'],
            experiment_name=rolling_exp_name,
        )

        yaml_file = model_info['yaml_file']
        folds = trained_window.get('cpcv_folds', [])

        fi = 0
        fold_predictions = []
        while True:
            try:
                fold_model = rec.load_object(f"model_fold_{fi}.pkl")
            except Exception:
                break  # No more fold models

            # Build dataset with test range extended to anchor_date
            fold_params = dict(params_base)
            fold_params['test_start_time'] = effective_test_start.strftime('%Y-%m-%d')
            fold_params['test_end_time'] = effective_test_end
            fold_params['anchor_date'] = effective_test_end

            # Use first fold's structure for dataset config
            ref_fold = folds[0] if folds else {
                'train_segments': [[trained_window['train_start'],
                                    trained_window['train_end']]],
                'valid_start_time': trained_window['train_start'],
                'valid_end_time': trained_window['train_end'],
            }
            task_config = inject_config_for_fold(
                yaml_file, fold_params, ref_fold,
                model_name=model_name,
            )

            dataset_cfg = task_config['task']['dataset']
            dataset = init_instance_by_config(dataset_cfg)
            dataset.setup_data()

            fold_pred = fold_model.predict(dataset=dataset)
            fold_predictions.append(fold_pred)
            fi += 1

        if not fold_predictions:
            print(f"  [{model_name}] No fold models found in recorder")
            return None

        print(f"  [{model_name}] Loaded {len(fold_predictions)} fold models")

        # Ensemble average
        all_preds_df = pd.concat(fold_predictions, axis=1)
        final_pred = all_preds_df.mean(axis=1, skipna=True)

        if isinstance(final_pred, pd.Series):
            final_pred = final_pred.to_frame('score')
        elif isinstance(final_pred, pd.DataFrame) and 'score' not in final_pred.columns:
            final_pred.columns = ['score']

        print(f"  [{model_name}] Ensemble prediction complete: "
              f"{len(final_pred)} rows")
        return final_pred

    except Exception as e:
        print(f"  [{model_name}] CPCV prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================================================
# Truncation repair
# ==========================================================================

def repair_truncated(model_name, model_info, comp, window_map,
                     rolling_exp_name, params_base):
    """CPCV version: detect & repair truncated window predictions.

    Loads all K fold models from the window's recorder, has each
    re-predict on the full test range, ensemble-averages, and
    writes the repaired pred.pkl back.

    Args:
        model_name: Model name.
        model_info: Model config dict.
        comp: {window_idx, record_id} from RollingState.
        window_map: {window_idx: window} lookup.
        rolling_exp_name: Per-window experiment name.
        params_base: Base params dict.

    Returns:
        (pred_df, repaired: bool)
    """
    from qlib.workflow import R
    from qlib.utils import init_instance_by_config
    from quantpits.utils.train_utils import inject_config_for_fold

    widx = comp['window_idx']
    w = window_map.get(widx)
    if not w:
        return None, False

    rec = R.get_recorder(
        recorder_id=comp['record_id'],
        experiment_name=rolling_exp_name,
    )

    pred = rec.load_object("pred.pkl")
    dates = pred.index.get_level_values('datetime')
    pred_max = dates.max()
    test_end = pd.Timestamp(w['test_end'])

    # 1-day tolerance
    if pred_max >= test_end - pd.Timedelta(days=1):
        return pred, False

    print(f"    🔧 Window {widx}: CPCV prediction truncated "
          f"({pred_max.date()} < {test_end.date()}), auto-repairing...")

    folds = w.get('cpcv_folds', [])
    if not folds:
        print(f"    ⚠️  Window {widx}: no fold info, cannot repair")
        return pred, False

    yaml_file = model_info['yaml_file']
    fold_params = dict(params_base)
    fold_params['test_start_time'] = w['test_start']
    fold_params['test_end_time'] = w['test_end']
    fold_params['anchor_date'] = w['test_end']

    try:
        fold_predictions = []
        for fi in range(len(folds)):
            try:
                fold_model = rec.load_object(f"model_fold_{fi}.pkl")
            except Exception:
                print(f"    ⚠️  Window {widx}: cannot load model_fold_{fi}.pkl")
                continue

            task_config = inject_config_for_fold(
                yaml_file, fold_params, folds[fi],
                model_name=model_name,
            )
            dataset_cfg = task_config['task']['dataset']
            dataset = init_instance_by_config(dataset_cfg)
            dataset.setup_data()

            fold_pred = fold_model.predict(dataset=dataset)
            fold_predictions.append(fold_pred)

        if not fold_predictions:
            print(f"    ⚠️  Window {widx}: no fold models loaded, repair failed")
            return pred, False

        all_preds_df = pd.concat(fold_predictions, axis=1)
        new_pred = all_preds_df.mean(axis=1, skipna=True)

        if isinstance(new_pred, pd.Series):
            new_pred = new_pred.to_frame('score')
        elif isinstance(new_pred, pd.DataFrame) and 'score' not in new_pred.columns:
            new_pred.columns = ['score']

        rec.save_objects(**{"pred.pkl": new_pred})

        new_dates = new_pred.index.get_level_values('datetime')
        print(f"    🔧 Window {widx}: CPCV repair complete "
              f"({new_dates.min().date()} ~ {new_dates.max().date()}, "
              f"{len(new_pred)} rows)")

        return new_pred, True

    except Exception as e:
        print(f"    ⚠️  Window {widx}: CPCV repair failed: {e}")
        return pred, False
