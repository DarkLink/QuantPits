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

    def test_get_or_build_forces_drop_raw_false(self, monkeypatch):
        mgr = HandlerCacheManager(max_size_mb=1024)
        cfg = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {'drop_raw': True}}

        built_cfg = None

        def mock_init(c, *args, **kwargs):
            nonlocal built_cfg
            built_cfg = c
            return DummyHandler()

        monkeypatch.setattr("qlib.utils.init_instance_by_config", mock_init)

        h, was_cached = mgr.get_or_build(cfg)
        assert was_cached is False
        assert built_cfg is not None
        assert built_cfg['kwargs']['drop_raw'] is False

    def test_get_or_build_parallel(self, monkeypatch):
        import threading
        import time

        mgr = HandlerCacheManager(max_size_mb=1024)
        cfg = {'class': 'Alpha158', 'module_path': 'm', 'kwargs': {}}

        build_count = 0
        build_started = threading.Event()
        build_resume = threading.Event()

        def mock_init(c, *args, **kwargs):
            nonlocal build_count
            build_count += 1
            build_started.set()
            build_resume.wait(timeout=5.0)
            return DummyHandler()

        monkeypatch.setattr("qlib.utils.init_instance_by_config", mock_init)

        results = []
        threads = []

        def worker():
            h, was_cached = mgr.get_or_build(cfg)
            results.append((h, was_cached))

        # Thread 1 starts build
        t1 = threading.Thread(target=worker)
        t1.start()

        # Wait for thread 1 to enter mock_init
        assert build_started.wait(timeout=2.0)
        assert build_count == 1

        # Thread 2 starts build (should block on Event because Thread 1 is building)
        t2 = threading.Thread(target=worker)
        t2.start()

        # Give Thread 2 a moment to block
        time.sleep(0.1)
        assert len(results) == 0  # neither finished yet
        assert build_count == 1   # no second build started

        # Let Thread 1 finish
        build_resume.set()
        t1.join()
        t2.join()

        assert len(results) == 2
        caches = [r[1] for r in results]
        assert False in caches
        assert True in caches
        assert results[0][0] is results[1][0]  # they got the exact same handler instance
        assert build_count == 1  # exactly one build happened!


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


class DummyProcessor:
    def __init__(self, val=0):
        self.val = val
        self._readonly = False
    def fit(self, df):
        self.val = len(df)
    def __call__(self, df):
        return df
    def is_for_infer(self):
        return True
    def readonly(self):
        return self._readonly


class DummyHandler:
    def __init__(self):
        import pandas as pd
        self.data_loader = "dummy_loader"
        self.instruments = "dummy_instruments"
        self.start_time = "2020-01-01"
        self.end_time = "2020-12-31"
        self.fetch_orig = True
        self._data = pd.DataFrame({'col1': [1, 2, 3]})
        self.infer_processors = [DummyProcessor(10)]
        self.learn_processors = [DummyProcessor(20)]
        self.shared_processors = [DummyProcessor(30)]
        self.process_type = "append"
        self._infer = pd.DataFrame({'infer': [10]})
        self._learn = pd.DataFrame({'learn': [20]})


class TestHandlerProxy:
    def test_proxy_initialization_and_isolation(self):
        from quantpits.utils.handler_cache import HandlerProxy
        import pandas as pd

        source = DummyHandler()
        proxy = HandlerProxy(source)

        # Check metadata attributes
        assert proxy.data_loader == "dummy_loader"
        assert proxy.instruments == "dummy_instruments"
        assert proxy.start_time == "2020-01-01"
        assert proxy.end_time == "2020-12-31"
        assert proxy.fetch_orig is True
        assert proxy.process_type == "append"
        assert proxy.drop_raw is False

        # Check raw data sharing
        assert proxy._data is source._data

        # _infer/_learn must be independent (NOT shared references)
        # after process_data() runs during __init__. Since DummyProcessor
        # is a no-op (__call__ returns df as-is), _infer and _learn should
        # be copies derived from _data, not None.
        assert proxy._infer is not source._infer
        assert proxy._learn is not source._learn

        # Check processor deep copy
        assert len(proxy.infer_processors) == 1
        assert proxy.infer_processors[0] is not source.infer_processors[0]
        assert proxy.infer_processors[0].val == source.infer_processors[0].val

        # Verify processor modification isolation
        proxy.infer_processors[0].val = 999
        assert source.infer_processors[0].val == 10

    def test_setup_data_calls_process_data(self, monkeypatch):
        from quantpits.utils.handler_cache import HandlerProxy
        import pandas as pd

        called = []

        class MockedHandlerProxy(HandlerProxy):
            def fit(self):
                called.append("fit")
            def process_data(self, with_fit=False):
                called.append(f"process_data_with_fit_{with_fit}")
            def fit_process_data(self):
                called.append("fit_process_data")

        source = DummyHandler()
        proxy = MockedHandlerProxy(source)

        # __init__ calls process_data() once; clear that initial call
        assert "process_data_with_fit_False" in called
        called.clear()

        proxy.setup_data(init_type="fit_seq")
        assert "fit_process_data" in called

        called.clear()
        proxy.setup_data(init_type="fit_ind")
        assert "fit" in called
        assert "process_data_with_fit_False" in called

        called.clear()
        proxy.setup_data(init_type="load_state")
        assert "process_data_with_fit_False" in called


class TestMemoryEstimationAndRollingTasks:
    def test_get_handler_feature_count(self):
        from quantpits.utils.handler_cache import _get_handler_feature_count
        assert _get_handler_feature_count({'class': 'Alpha158', 'kwargs': {'infer_processors': [{'class': 'FilterCol'}]}}) == 138
        assert _get_handler_feature_count({'class': 'Alpha158', 'kwargs': {'infer_processors': []}}) == 158
        assert _get_handler_feature_count({'class': 'Alpha360'}) == 6
        assert _get_handler_feature_count({'class': 'Alpha360_plus'}) == 10
        assert _get_handler_feature_count({'class': 'Alpha158_plus'}) == 158
        assert _get_handler_feature_count({'class': 'CustomUnknown'}) == 100

    def test_estimate_handler_memory_missing_dates(self):
        from quantpits.utils.handler_cache import estimate_handler_memory
        # missing start/end time
        assert estimate_handler_memory({'kwargs': {}}) == 0

    def test_estimate_handler_memory_import_error(self, monkeypatch):
        from quantpits.utils.handler_cache import estimate_handler_memory
        # simulate qlib import error
        import sys
        real_modules = sys.modules.copy()
        try:
            sys.modules['qlib.data'] = None
            assert estimate_handler_memory({'kwargs': {'start_time': '2020-01-01', 'end_time': '2020-12-31'}}) == 0
        finally:
            sys.modules.update(real_modules)

    def test_estimate_handler_memory_exception_in_qlib(self, monkeypatch):
        import qlib.data
        from unittest.mock import MagicMock
        from quantpits.utils.handler_cache import estimate_handler_memory
        mock_D = MagicMock()
        mock_D.instruments.side_effect = Exception("qlib failed")
        
        monkeypatch.setattr(qlib.data, 'D', mock_D)
        
        cfg = {'class': 'Alpha158', 'kwargs': {'start_time': '2020-01-01', 'end_time': '2020-12-31'}}
        assert estimate_handler_memory(cfg) == 0

    def test_estimate_handler_memory_valid(self, monkeypatch):
        import qlib.data
        from unittest.mock import MagicMock
        from quantpits.utils.handler_cache import estimate_handler_memory
        mock_D = MagicMock()
        mock_D.instruments.return_value = ['stock1', 'stock2']
        mock_D.calendar.return_value = ['day1', 'day2', 'day3']
        
        monkeypatch.setattr(qlib.data, 'D', mock_D)

        cfg = {
            'class': 'Alpha360',
            'kwargs': {
                'start_time': '2020-01-01',
                'end_time': '2020-12-31',
                'instruments': 'csi300'
            }
        }
        # n_days = 3, n_stocks = 2, n_features = 6
        # expected: int(3 * 2 * 6 * 8 * 1.2 * 2.5) = int(863.9999...) = 863
        assert estimate_handler_memory(cfg) in (863, 864)

    def test_auto_detect_memory(self, monkeypatch):
        from unittest.mock import MagicMock
        from quantpits.utils.handler_cache import _auto_detect_memory
        mock_mem = MagicMock(available=1000000)
        monkeypatch.setattr('psutil.virtual_memory', lambda: mock_mem)
        assert _auto_detect_memory() == 500000

    def test_extract_rolling_tasks(self, monkeypatch):
        from unittest.mock import MagicMock
        from quantpits.utils.handler_cache import enumerate_tasks_rolling
        
        # mock windows generator
        mock_windows = [
            {'window_idx': 0, 'train_start': '2015', 'train_end': '2016', 'valid_start': '2016', 'valid_end': '2017', 'test_start': '2017', 'test_end': '2018'}
        ]
        monkeypatch.setattr('quantpits.scripts.rolling.windows.generate_rolling_windows', lambda *a, **k: mock_windows)
        
        # mock inject_config
        mock_task_config = {
            'task': {
                'dataset': {
                    'class': 'DatasetH',
                    'kwargs': {
                        'handler': {'class': 'Alpha158'}
                    }
                }
            }
        }
        monkeypatch.setattr('quantpits.utils.train_utils.inject_config', lambda y, p, model_name: mock_task_config)

        params = {'rolling_start': '2015-01-01'}
        yaml_paths = {'model_a': 'path/to/model_a.yaml'}
        
        # 1. Cold start
        tasks = enumerate_tasks_rolling(['model_a'], yaml_paths, params, rolling_state=None)
        assert len(tasks) == 1
        assert tasks[0][0] == 'model_a'
        assert tasks[0][1] == 0
        assert tasks[0][2] == {'class': 'Alpha158'}

        # 2. Resume (already done)
        mock_state = MagicMock()
        mock_state.is_window_model_done.return_value = True
        tasks = enumerate_tasks_rolling(['model_a'], yaml_paths, params, rolling_state=mock_state)
        assert len(tasks) == 0

        # 3. Resume (state check throws exception -> proceed)
        mock_state.is_window_model_done.side_effect = Exception("failed")
        tasks = enumerate_tasks_rolling(['model_a'], yaml_paths, params, rolling_state=mock_state)
        assert len(tasks) == 1

        # 4. No windows generated
        monkeypatch.setattr('quantpits.scripts.rolling.windows.generate_rolling_windows', lambda *a, **k: [])
        tasks = enumerate_tasks_rolling(['model_a'], yaml_paths, params, rolling_state=None)
        assert tasks == []



