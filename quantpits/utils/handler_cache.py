#!/usr/bin/env python
"""
Centralized DataHandler cache for training.

Eliminates redundant disk I/O and processor fitting when multiple models
share the same data handler configuration. Pre-analyzes all training tasks,
builds each unique handler exactly once, and reuses across models/folds/windows.

Usage in training scripts:
    from quantpits.utils.handler_cache import (
        HandlerCacheManager,
        enumerate_tasks_static,
        enumerate_tasks_cpcv,
        enumerate_tasks_rolling,
        pre_analyze,
    )

    cache_mgr = HandlerCacheManager(max_size_mb=args.cache_size)
    tasks = enumerate_tasks_static(selected_models, yaml_paths, params)
    pre_analyze(tasks, cache_mgr)

    for model_name, handler_cfg, dataset_cfg in tasks:
        dataset = cache_mgr.create_dataset(dataset_cfg)
        model.fit(dataset)

    print(cache_mgr.stats)
"""

import copy
import hashlib
import json
import os
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import psutil


# ============================================================================
# Cache key construction
# ============================================================================

def build_cache_key(handler_cfg: dict, extra_context: Optional[dict] = None) -> str:
    """Build a deterministic hash key from a handler config.

    All handler kwargs participate — instruments, label, start_time, end_time,
    fit_start_time, fit_end_time, and all processor configs. extra_context
    includes dataset-level fields that affect correctness (e.g. validation
    boundaries for CPCV fold isolation).

    Uses json.dumps with sort_keys=True for deterministic output. Qlib YAML
    configs produce only lists and dicts (no set objects), so this is safe.
    """
    key_dict = {
        "class": handler_cfg.get("class"),
        "module_path": handler_cfg.get("module_path"),
        "kwargs": _canonicalize(handler_cfg.get("kwargs", {})),
    }
    if extra_context:
        key_dict["_ctx"] = extra_context
    raw = json.dumps(key_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _canonicalize(obj: Any) -> Any:
    """Recursively sort dicts and convert tuples to lists for stable JSON output."""
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


# ============================================================================
# Temporal processor detection (for CPCV cross-fold optimization)
# ============================================================================

CROSSSECTIONAL_PROCESSOR_CLASSES = frozenset({
    'CSZScoreNorm',
    'CSRankNorm',
    'CSZFillna',
    'DropnaLabel',
    'Fillna',
    'FilterCol',
    'DropCol',
})


def _has_temporal_processors(handler_cfg: dict) -> bool:
    """Check if handler config contains any time-dependent normalizer.

    Uses a whitelist strategy: any processor NOT in CROSSSECTIONAL_PROCESSOR_CLASSES
    is assumed to be temporal (time-dependent) to be safe against data leakage.

    Cross-sectional processors (CSZScoreNorm, CSRankNorm, CSZFillna,
    DropnaLabel, Fillna, FilterCol, DropCol) operate per-day or on
    DataFrame structure — immune to temporal leakage.
    """
    kwargs = handler_cfg.get('kwargs', {})
    for proc_list_key in ('infer_processors', 'learn_processors'):
        for proc in kwargs.get(proc_list_key, []):
            # Normalize: qlib allows both "ClassName" (str) and
            # {"class": "ClassName", "kwargs": {...}} (dict) forms.
            cls_name = proc if isinstance(proc, str) else proc.get('class', '')
            if cls_name not in CROSSSECTIONAL_PROCESSOR_CLASSES:
                return True
    return False


# ============================================================================
# Memory estimation
# ============================================================================

def _get_handler_feature_count(handler_cfg: dict) -> int:
    """Estimate the number of features a handler processes.

    Accounts for FilterCol which drops ~20 features from Alpha158.
    """
    cls_name = handler_cfg.get('class', '')
    if cls_name == 'Alpha158':
        kwargs = handler_cfg.get('kwargs', {})
        infer_procs = kwargs.get('infer_processors', [])
        has_filter = any(
            p.get('class') == 'FilterCol' for p in infer_procs
        )
        return 138 if has_filter else 158
    elif cls_name == 'Alpha360':
        return 6
    elif cls_name == 'Alpha360_plus':
        return 10
    elif cls_name == 'Alpha158_plus':
        return 158
    # Conservative default for custom handlers
    return 100


def estimate_handler_memory(handler_cfg: dict) -> int:
    """Estimate peak memory (bytes) for a DataHandler after setup_data().

    Peak is ~2.5× the raw DataFrame size: holds _data, _learn, and _infer
    simultaneously during process_data(). Each cell is float64 (8 bytes),
    with ~20% DataFrame overhead.

    Returns 0 if date ranges cannot be resolved (e.g. before qlib init).
    """
    try:
        from qlib.data import D
    except ImportError:
        return 0

    kwargs = handler_cfg.get('kwargs', {})
    instruments = kwargs.get('instruments', 'csi300')
    start_time = kwargs.get('start_time')
    end_time = kwargs.get('end_time')

    if not start_time or not end_time:
        return 0

    try:
        n_stocks = len(D.instruments(instruments))
        cal = D.calendar(start_time=start_time, end_time=end_time)
        n_days = len(cal) if cal is not None else 0
    except Exception:
        return 0

    if n_stocks == 0 or n_days == 0:
        return 0

    n_features = _get_handler_feature_count(handler_cfg)
    # float64 = 8 bytes, DataFrame overhead ~20%, intermediate copies ~2.5x
    return int(n_days * n_stocks * n_features * 8 * 1.2 * 2.5)


def _auto_detect_memory() -> int:
    """Auto-detect available memory. Returns 50% of free RAM in bytes."""
    mem = psutil.virtual_memory()
    return int(mem.available * 0.5)


# ============================================================================
# HandlerCacheManager
# ============================================================================

class HandlerCacheManager:
    """Caches fitted DataHandler instances, keyed by their full config hash.

    Builds handlers lazily on first access. LRU eviction when memory limit
    is exceeded. Designed for the training loop pattern:
      - register + pre_analyze: discover unique handler configs (no building)
      - training loop: create_dataset -> get_or_build -> model.fit()

    Single-process only: cached handlers hold multi-GB DataFrames internally.
    Do NOT share across processes — Python multiprocessing would pickle-
    serialize the DataFrames.

    Parameters
    ----------
    max_size_mb : int or None
        Maximum handler memory in MB. None = auto-detect (50% of free RAM).
    """

    def __init__(self, max_size_mb: Optional[int] = None):
        self._cache: OrderedDict = OrderedDict()   # key -> DataHandler
        self._configs: dict = {}                    # key -> (handler_cfg, extra_ctx)
        self._memory: dict = {}                     # key -> estimated bytes
        self._total_memory: int = 0
        self._max_memory: int = (
            max_size_mb * 1024 * 1024 if max_size_mb is not None
            else _auto_detect_memory()
        )
        self._hits: int = 0
        self._misses: int = 0
        self._build_times: dict = {}                # key -> seconds

    # -- Registration (pre-analysis phase) --

    def register(
        self, handler_cfg: dict, extra_context: Optional[dict] = None
    ) -> str:
        """Record a handler config needed by upcoming training tasks.

        Does NOT build the handler. Safe to call repeatedly (idempotent).
        Returns the cache key.
        """
        key = build_cache_key(handler_cfg, extra_context)
        if key not in self._configs:
            self._configs[key] = (copy.deepcopy(handler_cfg), extra_context)
            self._memory[key] = estimate_handler_memory(handler_cfg)
        return key

    # -- Handler access (training phase) --

    def get_or_build(
        self, handler_cfg: dict, extra_context: Optional[dict] = None
    ) -> Tuple[Any, bool]:
        """Get cached handler or build + cache it.

        Returns (handler, was_cached). Evicts LRU entries BEFORE building
        to prevent memory spikes (current cache + new handler peak
        simultaneously could trigger OOM kill).
        """
        key = build_cache_key(handler_cfg, extra_context)

        # Cache hit
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key], True

        self._misses += 1

        # Evict BEFORE building — prevent memory spike
        est = estimate_handler_memory(handler_cfg)
        while self._cache and (self._total_memory + est > self._max_memory):
            evict_key, _ = self._cache.popitem(last=False)
            self._total_memory -= self._memory.get(evict_key, 0)

        # Build handler (qlib init must already be done)
        from qlib.utils import init_instance_by_config

        t0 = time.time()
        handler = init_instance_by_config(handler_cfg)
        self._build_times[key] = time.time() - t0

        # Store in cache
        self._cache[key] = handler
        self._total_memory += est
        return handler, False

    # -- Dataset creation --

    def create_dataset(
        self, dataset_cfg: dict, extra_context: Optional[dict] = None
    ) -> Any:
        """Create a DatasetH/TSDatasetH/PurgedDatasetH from config,
        reusing a cached handler when possible.

        Replaces init_instance_by_config(dataset_cfg) in training code.
        Qlib's DatasetH.__init__ accepts DataHandler instances natively
        via accept_types=DataHandler, so we simply swap the handler config
        dict for the cached instance.

        WARNING: Cached handler instances are shared across multiple datasets.
        MUST be used in serial training loop; handler state is shared across datasets.

        Parameters
        ----------
        dataset_cfg : dict
            Dataset config with 'class', 'module_path', 'kwargs' keys.
            kwargs must contain 'handler' (config dict) and 'segments'.
        extra_context : dict or None
            CPCV validation boundaries for cache key isolation.
        """
        from qlib.utils import init_instance_by_config

        handler_cfg = dataset_cfg['kwargs']['handler']
        handler, _ = self.get_or_build(handler_cfg, extra_context)

        # Clone config and inject cached handler instance
        ds_cfg = copy.deepcopy(dataset_cfg)
        ds_cfg['kwargs']['handler'] = handler  # instance, not config dict

        return init_instance_by_config(ds_cfg)

    # -- Statistics --

    @property
    def stats(self) -> dict:
        """Return cache statistics for post-training reporting."""
        return {
            'hits': self._hits,
            'misses': self._misses,
            'handlers_cached': len(self._cache),
            'handlers_registered': len(self._configs),
            'total_memory_mb': round(self._total_memory / (1024 * 1024), 1),
            'max_memory_mb': round(self._max_memory / (1024 * 1024), 1),
            'total_build_time_s': round(sum(self._build_times.values()), 1),
        }

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"HandlerCacheManager(hits={s['hits']}, misses={s['misses']}, "
            f"cached={s['handlers_cached']}, "
            f"mem={s['total_memory_mb']}/{s['max_memory_mb']}MB, "
            f"build_time={s['total_build_time_s']}s)"
        )


# ============================================================================
# Task enumeration functions (one per training mode)
# ============================================================================

def enumerate_tasks_static(
    selected_models: List[str],
    yaml_paths: Dict[str, str],
    params: dict,
) -> List[tuple]:
    """Enumerate static training tasks: 1 task per model.

    All models share the same date params from calculate_dates().
    Models in the same handler group will map to the same cache key.

    Returns:
        List of (model_name, handler_cfg, dataset_cfg) tuples.
    """
    from quantpits.utils.train_utils import inject_config

    tasks = []
    for model_name in selected_models:
        yaml_path = yaml_paths[model_name]
        task_config = inject_config(
            yaml_path, params, model_name=model_name)
        handler_cfg = task_config['task']['dataset']['kwargs']['handler']
        dataset_cfg = task_config['task']['dataset']
        tasks.append((model_name, handler_cfg, dataset_cfg))

    return tasks


def enumerate_tasks_cpcv(
    selected_models: List[str],
    yaml_paths: Dict[str, str],
    params: dict,
) -> List[tuple]:
    """Enumerate CPCV training tasks: K tasks per model (one per fold).

    K depends on purged_cv config (n_groups, n_test_groups, n_val_groups)
    — NOT fixed. Each fold has its own validation period punched out.

    For cross-sectional normalizers (CSZScoreNorm, CSRankNorm):
        Folds with the same fit_start/fit_end share handlers.
        extra_ctx is None — cache key determined by handler config alone.

    For temporal normalizers (ZScoreNorm, RobustZScoreNorm):
        Each fold gets its own handler via validation boundaries in extra_ctx.

    Within a fold, all models in the same handler group share one handler.

    Returns:
        List of (model_name, fold_idx, handler_cfg, dataset_cfg, extra_ctx)
        tuples. extra_ctx is None for cross-sectional, {'valid': [...]} for
        temporal.
    """
    from quantpits.utils.train_utils import inject_config_for_fold

    folds = params.get('cpcv_folds', [])
    if not folds:
        return []

    tasks = []
    for model_name in selected_models:
        yaml_path = yaml_paths[model_name]

        # Detect temporal processors once per model
        # (same pipeline for all folds of this model)
        task_cfg_0 = inject_config_for_fold(
            yaml_path, params, folds[0], model_name=model_name)
        handler_cfg_0 = task_cfg_0['task']['dataset']['kwargs']['handler']
        has_temporal = _has_temporal_processors(handler_cfg_0)

        for fold in folds:
            task_cfg = inject_config_for_fold(
                yaml_path, params, fold, model_name=model_name)
            handler_cfg = task_cfg['task']['dataset']['kwargs']['handler']
            dataset_cfg = task_cfg['task']['dataset']

            if has_temporal:
                valid_seg = dataset_cfg['kwargs']['segments']['valid']
                extra_ctx = {'valid': valid_seg}
            else:
                extra_ctx = None

            tasks.append((
                model_name, fold['fold_idx'], handler_cfg,
                dataset_cfg, extra_ctx,
            ))

    return tasks


def enumerate_tasks_rolling(
    selected_models: List[str],
    yaml_paths: Dict[str, str],
    params: dict,
    rolling_state: Any = None,
) -> List[tuple]:
    """Enumerate rolling training tasks: one per (model, window) pair.

    Skip already-completed pairs when rolling_state is provided (resume).
    Each window has its own handler_cfg (different fit_start/fit_end).
    Within a window, models in the same handler group share one handler.

    Parameters
    ----------
    rolling_state : RollingState or None
        None = cold start (all windows). RollingState instance = resume/merge.

    Returns:
        List of (model_name, window_idx, handler_cfg, dataset_cfg) tuples.
    """
    from quantpits.scripts.rolling.windows import generate_rolling_windows
    from quantpits.utils.train_utils import inject_config

    windows = generate_rolling_windows(
        params.get('rolling_start', '2017-01-01'),
        params.get('train_years', 5),
        params.get('valid_years', 2),
        params.get('test_step', '3M'),
        params.get('anchor_date', ''),
    )

    if not windows:
        return []

    tasks = []
    skipped = 0
    for model_name in selected_models:
        yaml_path = yaml_paths[model_name]
        for win in windows:
            widx = win['window_idx']

            # Resume check: skip completed (model, window) pairs
            if rolling_state is not None:
                try:
                    if rolling_state.is_window_model_done(widx, model_name):
                        skipped += 1
                        continue
                except Exception:
                    pass  # state check failed — proceed with training

            # Build window-specific params
            win_params = _build_rolling_window_params(params, win)
            task_config = inject_config(
                yaml_path, win_params, model_name=model_name)
            handler_cfg = task_config['task']['dataset']['kwargs']['handler']
            dataset_cfg = task_config['task']['dataset']
            tasks.append((model_name, widx, handler_cfg, dataset_cfg))

    if skipped:
        print(f"  Rolling resume: skipped {skipped} already-completed "
              f"(model, window) pairs")

    return tasks


def _build_rolling_window_params(params: dict, window: dict) -> dict:
    """Build date params for a specific rolling window.

    Creates a copy of params with date fields set to this window's boundaries.
    """
    import copy
    win_params = copy.deepcopy(params)
    win_params['start_time'] = window['train_start']
    win_params['end_time'] = window['test_end']
    win_params['fit_start_time'] = window['train_start']
    win_params['fit_end_time'] = window['train_end']
    win_params['valid_start_time'] = window['valid_start']
    win_params['valid_end_time'] = window['valid_end']
    win_params['test_start_time'] = window['test_start']
    win_params['test_end_time'] = window['test_end']
    return win_params


# ============================================================================
# Pre-analysis orchestration
# ============================================================================

def pre_analyze(tasks: List[tuple], cache_mgr: HandlerCacheManager) -> dict:
    """Register all unique handler configs and report grouping.

    Does NOT build handlers — only registers them for lazy construction
    during the training loop.

    Parameters
    ----------
    tasks : list of tuples from enumerate_tasks_*()
        Tuple structure varies by mode:
          - static:  (model_name, handler_cfg, dataset_cfg)
          - cpcv:    (model_name, fold_idx, handler_cfg, dataset_cfg, extra_ctx)
          - rolling: (model_name, window_idx, handler_cfg, dataset_cfg)
    cache_mgr : HandlerCacheManager

    Returns:
        dict mapping cache_key -> list of task indices
    """
    groups = defaultdict(list)

    for i, task in enumerate(tasks):
        # Extract handler_cfg and extra_ctx from the task tuple.
        # Task tuple structure: static/rolling has handler_cfg at index -2,
        # cpcv has handler_cfg at index 2 and extra_ctx at index 4.
        if len(task) == 5:
            # CPCV: (model, fold_idx, handler_cfg, dataset_cfg, extra_ctx)
            handler_cfg = task[2]
            extra_ctx = task[4]
        elif len(task) == 4:
            # Rolling: (model, window_idx, handler_cfg, dataset_cfg)
            handler_cfg = task[2]
            extra_ctx = None
        else:
            # Static: (model, handler_cfg, dataset_cfg)
            handler_cfg = task[1]
            extra_ctx = None

        key = cache_mgr.register(handler_cfg, extra_ctx)
        groups[key].append(i)

    # Report
    print(f"\n{'='*60}")
    print(f"  Handler Cache Pre-Analysis")
    print(f"{'='*60}")
    print(f"  Training tasks:         {len(tasks)}")
    print(f"  Unique handler configs: {len(groups)}")

    total_est = 0
    for key, indices in sorted(groups.items(), key=lambda x: -len(x[1])):
        est_mb = cache_mgr._memory.get(key, 0) / (1024 * 1024)
        total_est += cache_mgr._memory.get(key, 0)
        sample = [tasks[i] for i in indices[:3]]
        names = [t[0] for t in sample]
        suffix = f" (+{len(indices) - 3} more)" if len(indices) > 3 else ""
        n_folds = ""
        if len(sample) > 0 and len(sample[0]) >= 3:
            # Show fold/window info if present
            extra = []
            for t in sample:
                if len(t) >= 3 and isinstance(t[1], int):
                    extra.append(f"fold={t[1]}" if len(t) == 5 else f"win={t[1]}")
            if extra:
                n_folds = f" [{', '.join(extra)}]"
        print(f"  [{key}] {len(indices)} tasks, ~{est_mb:.0f} MB  "
              f"{', '.join(names)}{suffix}{n_folds}")

    print(f"  {'─'*50}")
    print(f"  Total estimated memory: ~{total_est / (1024*1024):.0f} MB "
          f"(limit: {cache_mgr._max_memory / (1024*1024):.0f} MB)")
    if total_est > cache_mgr._max_memory:
        print(f"  ⚠️  Estimated memory exceeds limit! LRU eviction will "
              f"manage pressure. Consider --cache-size to increase limit.")
    print(f"{'='*60}\n")

    return dict(groups)
