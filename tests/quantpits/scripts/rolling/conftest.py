"""
Shared fixtures for rolling strategy tests.

Provides pre-built mocks for qlib runtime dependencies (R, C, init_instance_by_config,
inject_config, subprocess) so each strategy test file doesn't need to rebuild them.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# =========================================================================
# qlib.workflow.R mock (MLflow recorder)
# =========================================================================

@pytest.fixture
def mock_qlib_R():
    """Mock qlib.workflow.R as a context manager with full recorder chain.

    Returns a dict with:
      - R: the MagicMock for qlib.workflow.R
      - recorder: the MagicMock returned by get_recorder()
      - start: the MagicMock backing R.start(...) context manager
    """
    recorder = MagicMock(name="recorder")
    recorder.id = "mock_recorder_001"
    recorder.info = {"id": "mock_recorder_001"}

    # R.start() is used as a context manager: with R.start(...): ...
    start_cm = MagicMock(name="R.start_cm")
    start_cm.__enter__.return_value = None
    start_cm.__exit__.return_value = None

    R_mock = MagicMock(name="qlib.workflow.R")
    R_mock.start.return_value = start_cm
    R_mock.set_tags = MagicMock()
    R_mock.log_params = MagicMock()
    R_mock.save_objects = MagicMock()
    R_mock.get_recorder.return_value = recorder

    return {"R": R_mock, "recorder": recorder, "start_cm": start_cm}


# =========================================================================
# qlib.utils mock (init_instance_by_config)
# =========================================================================

@pytest.fixture
def mock_qlib_init():
    """Mock qlib.utils.init_instance_by_config to return a MagicMock model/dataset.

    Returns a dict with:
      - init_instance: the MagicMock for init_instance_by_config
      - model: the MagicMock model returned for model configs
      - dataset: the MagicMock dataset returned for dataset configs
    """
    model = MagicMock(name="qlib_model")
    model.topk = 20
    model.n_drop = 3
    model.fit = MagicMock()
    model.predict = MagicMock()

    dataset = MagicMock(name="qlib_dataset")
    dataset.setup_data = MagicMock()
    dataset.prepare = MagicMock()
    dataset.segments = {"test": None, "valid": None}

    def _side_effect(cfg):
        """Dispatch: if the config has 'kwargs' and 'handler', it's a dataset.
        Otherwise treat it as a model."""
        if isinstance(cfg, dict):
            if "kwargs" in cfg and "handler" in cfg.get("kwargs", {}):
                return dataset
            if "class" in cfg and "Dataset" in str(cfg.get("class", "")):
                return dataset
            # Heuristic: dataset configs have 'segments' in kwargs
            if "kwargs" in cfg and "segments" in cfg["kwargs"]:
                return dataset
        return model

    init_instance = MagicMock(name="init_instance_by_config", side_effect=_side_effect)

    return {
        "init_instance": init_instance,
        "model": model,
        "dataset": dataset,
    }


# =========================================================================
# qlib.config.C mock
# =========================================================================

@pytest.fixture
def mock_qlib_C():
    """Mock qlib.config.C.register_from_C."""
    C_mock = MagicMock(name="qlib.config.C")
    C_mock.register_from_C = MagicMock()
    return C_mock


# =========================================================================
# inject_config mock
# =========================================================================

@pytest.fixture
def mock_inject_config():
    """Mock inject_config / inject_config_for_fold to return a valid task_config dict.

    Returns the MagicMock with a .return_value set.
    """
    mock_fn = MagicMock(name="inject_config")
    mock_fn.return_value = {
        "task": {
            "model": {"class": "MockModel", "kwargs": {}},
            "dataset": {
                "class": "MockDataset",
                "kwargs": {
                    "handler": {"class": "MockHandler"},
                    "segments": {
                        "train": ("2020-01-01", "2022-12-31"),
                        "valid": ("2023-01-01", "2023-12-31"),
                        "test": ("2024-01-01", "2024-03-31"),
                    },
                },
            },
            "record": [],
        }
    }
    return mock_fn


# =========================================================================
# subprocess executor mock
# =========================================================================

@pytest.fixture
def mock_process_pool():
    """Mock concurrent.futures.ProcessPoolExecutor for isolated training.

    Returns a factory that creates pre-configured executor/future mocks.
    """
    def _create(result=None):
        if result is None:
            result = {
                "success": True,
                "record_id": "mock_rid_subprocess",
                "performance": {"IC_Mean": 0.05, "ICIR": 0.5},
                "n_folds": 3,
                "fold_scores": [0.04, 0.05, 0.06],
                "error": None,
            }

        future = MagicMock(name="future")
        future.result.return_value = result

        executor = MagicMock(name="ProcessPoolExecutor")
        executor.__enter__.return_value.submit.return_value = future
        executor.__exit__.return_value = None

        return executor, future

    return _create
