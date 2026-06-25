"""Unit tests for quantpits.utils.handler_cache."""

import pytest
from quantpits.utils.handler_cache import (
    HandlerCacheManager,
    build_cache_key,
    _has_temporal_processors,
    _canonicalize,
)


class TestBuildCacheKey:
    def test_identical_configs_same_key(self):
        """Same handler config produces same key regardless of kwarg order."""
        cfg1 = {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': {'a': 1, 'b': 2, 'c': [3, 4]},
        }
        cfg2 = {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': {'c': [3, 4], 'a': 1, 'b': 2},
        }
        assert build_cache_key(cfg1) == build_cache_key(cfg2)

    def test_different_fit_times_different_key(self):
        """Different fit_start_time produces different keys."""
        cfg1 = {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': {'fit_start_time': '2015-01-01', 'fit_end_time': '2020-01-01'},
        }
        cfg2 = {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': {'fit_start_time': '2016-01-01', 'fit_end_time': '2020-01-01'},
        }
        assert build_cache_key(cfg1) != build_cache_key(cfg2)

    def test_different_class_different_key(self):
        """Different handler class produces different keys."""
        cfg1 = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {}}
        cfg2 = {'class': 'Alpha360', 'module_path': 'm', 'kwargs': {}}
        assert build_cache_key(cfg1) != build_cache_key(cfg2)

    def test_extra_context_affects_key(self):
        """Extra context changes the cache key."""
        cfg = {'class': 'X', 'module_path': 'm', 'kwargs': {}}
        k1 = build_cache_key(cfg, extra_context=None)
        k2 = build_cache_key(cfg, extra_context={'valid': ['2020-01-01', '2021-01-01']})
        k3 = build_cache_key(cfg, extra_context={'valid': ['2021-01-01', '2022-01-01']})
        assert k1 != k2
        assert k2 != k3

    def test_same_extra_context_same_key(self):
        """Same extra context produces same key."""
        cfg = {'class': 'X', 'module_path': 'm', 'kwargs': {}}
        ctx = {'valid': ['2020-01-01', '2021-01-01']}
        assert build_cache_key(cfg, ctx) == build_cache_key(cfg, ctx)


class TestTemporalDetection:
    def test_detects_zscore_norm(self):
        cfg = {
            'kwargs': {
                'learn_processors': [{'class': 'ZScoreNorm'}],
            }
        }
        assert _has_temporal_processors(cfg) is True

    def test_detects_robust_zscore_norm(self):
        cfg = {
            'kwargs': {
                'infer_processors': [{'class': 'RobustZScoreNorm'}],
            }
        }
        assert _has_temporal_processors(cfg) is True

    def test_detects_minmax_norm(self):
        cfg = {
            'kwargs': {
                'learn_processors': [{'class': 'MinMaxNorm'}],
            }
        }
        assert _has_temporal_processors(cfg) is True

    def test_cross_sectional_only_returns_false(self):
        cfg = {
            'kwargs': {
                'learn_processors': [
                    {'class': 'DropnaLabel'},
                    {'class': 'CSRankNorm'},
                ],
                'infer_processors': [
                    {'class': 'CSZScoreNorm'},
                    {'class': 'Fillna'},
                    {'class': 'FilterCol'},
                ],
            }
        }
        assert _has_temporal_processors(cfg) is False

    def test_empty_processors_returns_false(self):
        cfg = {'kwargs': {}}
        assert _has_temporal_processors(cfg) is False


class TestCanonicalize:
    def test_sorts_dict_keys(self):
        assert _canonicalize({'b': 2, 'a': 1}) == {'a': 1, 'b': 2}

    def test_recursive_sort(self):
        assert _canonicalize({'z': {'b': 2, 'a': 1}}) == {'z': {'a': 1, 'b': 2}}

    def test_converts_tuples_to_lists(self):
        assert _canonicalize({'x': (1, 2)}) == {'x': [1, 2]}

    def test_preserves_scalars(self):
        assert _canonicalize({'a': 1, 'b': 'hello', 'c': 3.14}) == {'a': 1, 'b': 'hello', 'c': 3.14}


class TestHandlerCacheManager:
    def test_register_returns_key(self):
        mgr = HandlerCacheManager(max_size_mb=1024)
        cfg = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {}}
        key = mgr.register(cfg)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_register_idempotent(self):
        mgr = HandlerCacheManager(max_size_mb=1024)
        cfg = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {}}
        k1 = mgr.register(cfg)
        k2 = mgr.register(cfg)
        assert k1 == k2
        assert mgr.stats['handlers_registered'] == 1

    def test_register_with_extra_context(self):
        mgr = HandlerCacheManager(max_size_mb=1024)
        cfg = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {}}
        k1 = mgr.register(cfg)
        k2 = mgr.register(cfg, extra_context={'valid': ['2020', '2021']})
        assert k1 != k2
        assert mgr.stats['handlers_registered'] == 2

    def test_stats_initial(self):
        mgr = HandlerCacheManager(max_size_mb=1024)
        s = mgr.stats
        assert s['hits'] == 0
        assert s['misses'] == 0
        assert s['handlers_cached'] == 0
        assert s['handlers_registered'] == 0

    def test_repr(self):
        mgr = HandlerCacheManager(max_size_mb=1024)
        r = repr(mgr)
        assert 'HandlerCacheManager' in r
        assert 'hits=0' in r


class TestEnumerateTasksStatic:
    """Test enumerate_tasks_static with mocked inject_config."""

    def test_enumerates_all_models(self, monkeypatch):
        from quantpits.utils.handler_cache import enumerate_tasks_static

        def mock_inject(yaml_path, params, model_name=None):
            return {
                'task': {
                    'dataset': {
                        'class': 'DatasetH',
                        'module_path': 'qlib.data.dataset',
                        'kwargs': {
                            'handler': {
                                'class': 'Alpha158',
                                'module_path': 'qlib.contrib.data.handler',
                                'kwargs': {'fit_start_time': '2015-01-01'},
                            },
                            'segments': {'train': ['2015', '2020']},
                        },
                    },
                },
            }

        monkeypatch.setattr(
            'quantpits.utils.train_utils.inject_config', mock_inject)

        tasks = enumerate_tasks_static(
            ['model_a', 'model_b'],
            {'model_a': 'a.yaml', 'model_b': 'b.yaml'},
            {'freq': 'week'},
        )

        assert len(tasks) == 2
        assert tasks[0][0] == 'model_a'
        assert tasks[1][0] == 'model_b'
        # Both models share same handler config → same cache key
        from quantpits.utils.handler_cache import build_cache_key
        assert build_cache_key(tasks[0][1]) == build_cache_key(tasks[1][1])

    def test_different_handler_classes(self, monkeypatch):
        """Models with different handler classes get different keys."""
        from quantpits.utils.handler_cache import (
            enumerate_tasks_static, build_cache_key)

        call_count = [0]

        def mock_inject(yaml_path, params, model_name=None):
            call_count[0] += 1
            cls = 'Alpha158' if 'a' in (model_name or '') else 'Alpha360'
            return {
                'task': {
                    'dataset': {
                        'class': 'DatasetH',
                        'module_path': 'qlib.data.dataset',
                        'kwargs': {
                            'handler': {
                                'class': cls,
                                'module_path': 'qlib.contrib.data.handler',
                                'kwargs': {},
                            },
                            'segments': {},
                        },
                    },
                },
            }

        monkeypatch.setattr(
            'quantpits.utils.train_utils.inject_config', mock_inject)

        tasks = enumerate_tasks_static(
            ['model_a', 'model_z'],
            {'model_a': 'a.yaml', 'model_z': 'z.yaml'},
            {'freq': 'week'},
        )

        assert len(tasks) == 2
        assert build_cache_key(tasks[0][1]) != build_cache_key(tasks[1][1])


class TestEnumerateTasksCpcv:
    """Test enumerate_tasks_cpcv with mocked inject_config_for_fold."""

    def test_creates_ktasks_per_model(self, monkeypatch):
        from quantpits.utils.handler_cache import enumerate_tasks_cpcv

        folds = [
            {'fold_idx': 0, 'train_segments': [['2016', '2024']],
             'valid_start_time': '2015', 'valid_end_time': '2016'},
            {'fold_idx': 1, 'train_segments': [['2015', '2016'], ['2017', '2024']],
             'valid_start_time': '2016', 'valid_end_time': '2017'},
            {'fold_idx': 2, 'train_segments': [['2015', '2022']],
             'valid_start_time': '2023', 'valid_end_time': '2024'},
        ]

        def mock_inject_fold(yaml_path, params, fold, model_name=None,
                             no_pretrain=False):
            return {
                'task': {
                    'dataset': {
                        'class': 'PurgedDatasetH',
                        'module_path': 'quantpits.data.cpcv_dataset',
                        'kwargs': {
                            'handler': {
                                'class': 'Alpha158',
                                'module_path': 'qlib.contrib.data.handler',
                                'kwargs': {
                                    'fit_start_time': '2015-01-01',
                                    'fit_end_time': '2024-01-01',
                                    'learn_processors': [
                                        {'class': 'CSRankNorm'},
                                    ],
                                },
                            },
                            'segments': {
                                'train': fold['train_segments'],
                                'valid': [fold['valid_start_time'],
                                          fold['valid_end_time']],
                                'test': ['2024', '2026'],
                            },
                        },
                    },
                },
            }

        monkeypatch.setattr(
            'quantpits.utils.train_utils.inject_config_for_fold',
            mock_inject_fold)

        tasks = enumerate_tasks_cpcv(
            ['model_a', 'model_b'],
            {'model_a': 'a.yaml', 'model_b': 'b.yaml'},
            {'cpcv_folds': folds},
        )

        # 2 models × 3 folds = 6 tasks
        assert len(tasks) == 6
        # All tasks should have 5 elements (model, fold_idx, handler_cfg, dataset_cfg, extra_ctx)
        for t in tasks:
            assert len(t) == 5
        # Cross-sectional normalizers → extra_ctx is None
        for t in tasks:
            assert t[4] is None

    def test_temporal_normalizers_get_extra_ctx(self, monkeypatch):
        from quantpits.utils.handler_cache import enumerate_tasks_cpcv

        folds = [
            {'fold_idx': 0, 'train_segments': [['2016', '2024']],
             'valid_start_time': '2015', 'valid_end_time': '2016'},
        ]

        def mock_inject_fold(yaml_path, params, fold, model_name=None,
                             no_pretrain=False):
            return {
                'task': {
                    'dataset': {
                        'class': 'PurgedDatasetH',
                        'module_path': 'quantpits.data.cpcv_dataset',
                        'kwargs': {
                            'handler': {
                                'class': 'Alpha158',
                                'module_path': 'qlib.contrib.data.handler',
                                'kwargs': {
                                    'learn_processors': [
                                        {'class': 'RobustZScoreNorm'},
                                    ],
                                },
                            },
                            'segments': {
                                'train': fold['train_segments'],
                                'valid': [fold['valid_start_time'],
                                          fold['valid_end_time']],
                                'test': ['2024', '2026'],
                            },
                        },
                    },
                },
            }

        monkeypatch.setattr(
            'quantpits.utils.train_utils.inject_config_for_fold',
            mock_inject_fold)

        tasks = enumerate_tasks_cpcv(
            ['model_a'],
            {'model_a': 'a.yaml'},
            {'cpcv_folds': folds},
        )

        assert len(tasks) == 1
        # Temporal normalizer → extra_ctx contains validation boundaries
        assert tasks[0][4] == {'valid': ['2015', '2016']}

    def test_empty_folds_returns_empty(self):
        from quantpits.utils.handler_cache import enumerate_tasks_cpcv
        tasks = enumerate_tasks_cpcv(
            ['model_a'], {'model_a': 'a.yaml'}, {'cpcv_folds': []})
        assert tasks == []


class TestPreAnalyze:
    """Test the pre_analyze orchestration function."""

    def test_registers_all_tasks(self):
        from quantpits.utils.handler_cache import pre_analyze

        mgr = HandlerCacheManager(max_size_mb=1024)
        cfg = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {}}
        ds_cfg = {'class': 'DatasetH', 'module_path': 'm',
                  'kwargs': {'handler': cfg, 'segments': {}}}

        # Static tasks: (model, handler_cfg, dataset_cfg)
        tasks = [
            ('model_a', cfg, ds_cfg),
            ('model_b', cfg, ds_cfg),
            ('model_c', cfg, ds_cfg),
        ]
        groups = pre_analyze(tasks, mgr)

        assert len(groups) == 1  # all 3 share same handler
        assert mgr.stats['handlers_registered'] == 1

    def test_cpcv_task_extraction(self):
        from quantpits.utils.handler_cache import pre_analyze

        mgr = HandlerCacheManager(max_size_mb=1024)
        cfg = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {}}
        ds_cfg = {'class': 'PurgedDatasetH', 'module_path': 'm',
                  'kwargs': {'handler': cfg, 'segments': {}}}

        # CPCV tasks: (model, fold_idx, handler_cfg, dataset_cfg, extra_ctx)
        tasks = [
            ('model_a', 0, cfg, ds_cfg, None),
            ('model_a', 1, cfg, ds_cfg, {'valid': ['2016', '2017']}),
            ('model_b', 0, cfg, ds_cfg, None),
        ]
        groups = pre_analyze(tasks, mgr)

        # 2 unique configs: None extra_ctx and {'valid': ...}
        assert len(groups) == 2
        assert mgr.stats['handlers_registered'] == 2

    def test_zero_tasks(self):
        from quantpits.utils.handler_cache import pre_analyze

        mgr = HandlerCacheManager(max_size_mb=1024)
        groups = pre_analyze([], mgr)
        assert len(groups) == 0
