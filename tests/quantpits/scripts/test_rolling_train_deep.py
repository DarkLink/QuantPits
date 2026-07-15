"""
Deep mathematical verification of Rolling Training Logic.

This test focuses on independently validating the time slice 
partitioning, prediction truncation, and state management mechanisms 
in rolling_train.py without relying on MLflow or Qlib integrations.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from unittest import mock
from unittest.mock import patch

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', '..', 'quantpits', 'scripts')
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))

from quantpits.utils import env
import yaml

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "data").mkdir()
    (workspace / "output").mkdir()

    # Create dummy config files
    (workspace / "config" / "model_config.json").write_text(json.dumps({
        "market": "csi300",
        "benchmark": "SH000300",
        "freq": "week"
    }))
    (workspace / "config" / "model_registry.yaml").write_text(yaml.dump({
        "models": {
            "m1": {"algorithm": "gru", "dataset": "Alpha158", "enabled": True, "yaml_file": "gru.yaml"},
        }
    }))
    (workspace / "config" / "rolling_config.yaml").write_text(yaml.dump({
        "rolling_start": "2020-01-01",
        "train_years": 3,
        "valid_years": 1,
        "test_step": "3M"
    }))

    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(sys, 'argv', ['script.py'])

    from quantpits.utils import train_utils
    monkeypatch.setattr(train_utils, 'ROLLING_PREDICTION_DIR', str(workspace / "output" / "predictions" / "rolling"))
    (workspace / "output" / "predictions" / "rolling").mkdir(parents=True)

    # Reload relevant modules to pick up new env
    import importlib
    for mod in ['quantpits.utils.env', 'quantpits.utils.train_utils', 'quantpits.scripts.rolling_train']:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

    # Mock pd.DataFrame.to_csv and pd.Series.to_csv globally for tests to avoid fs issues
    monkeypatch.setattr(pd.DataFrame, 'to_csv', mock.MagicMock())
    monkeypatch.setattr(pd.Series, 'to_csv', mock.MagicMock())

    # Mock safeguard to avoid sleep
    monkeypatch.setattr('quantpits.utils.env.safeguard', lambda x, **kwargs: None)

    return workspace

from quantpits.scripts.rolling_train import (
    parse_step_to_relativedelta,
    generate_rolling_windows,
    _filter_pred_to_test_segment,
    RollingState
)

# ============================================================================
# Group 1: Step Parsing Verification
# ============================================================================

class TestStepParsing:
    def test_valid_month(self):
        delta = parse_step_to_relativedelta("3M")
        assert delta.months == 3
        assert delta.years == 0

        delta2 = parse_step_to_relativedelta(" 6M ")
        assert delta2.months == 6
        assert delta2.years == 0

    def test_valid_year(self):
        delta = parse_step_to_relativedelta("1Y")
        assert delta.years == 1
        assert delta.months == 0

    def test_invalid_formats(self):
        with pytest.raises(ValueError, match="Invalid step format"):
            parse_step_to_relativedelta("3D")
        
        with pytest.raises(ValueError):
            parse_step_to_relativedelta("M")
            
        with pytest.raises(ValueError):
            parse_step_to_relativedelta("")

# ============================================================================
# Group 2: Window Generation (Time Slice Partitioning)
# ============================================================================

class TestWindowGeneration:
    def test_standard_full_step_progression(self):
        """Standard rolling window progression with 1-year steps."""
        # Train=3Y, Valid=1Y, TestStep=1Y
        # T="2020-01-01", Anchor="2025-01-01"
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="1Y",
            anchor_date="2025-01-01"
        )
        # Expected:
        # W0: Train[2020-01-01, 2022-12-31], Valid[2023-01-01, 2023-12-31], Test[2024-01-01, 2024-12-31] -> Ends
        # W1: Test starts on 2025-01-01, and Anchor is 2025-01-01. 
        # test_end_full is 2025-12-31, but min(test_end_full, anchor) means test_end = 2025-01-01.
        
        assert len(windows) == 2
        
        w0 = windows[0]
        assert w0['window_idx'] == 0
        assert w0['train_start'] == "2020-01-01"
        assert w0['train_end'] == "2022-12-31"
        assert w0['valid_start'] == "2023-01-01"
        assert w0['valid_end'] == "2023-12-31"
        assert w0['test_start'] == "2024-01-01"
        assert w0['test_end'] == "2024-12-31"
        
        w1 = windows[1]
        assert w1['window_idx'] == 1
        assert w1['train_start'] == "2021-01-01"
        assert w1['train_end'] == "2023-12-31"
        assert w1['valid_start'] == "2024-01-01"
        assert w1['valid_end'] == "2024-12-31"
        assert w1['test_start'] == "2025-01-01"
        assert w1['test_end'] == "2025-01-01"  # Truncated exactly to Anchor

    def test_truncated_final_step(self):
        """Final window is partially through the test step."""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="3M",
            anchor_date="2024-02-15"
        )
        # W0: Test expected to be 2024-01-01 to 2024-03-31, but truncated by anchor.
        assert len(windows) == 1
        w0 = windows[0]
        assert w0['test_start'] == "2024-01-01"
        assert w0['test_end'] == "2024-02-15"

    def test_too_early_anchor(self):
        """Anchor date is before the first test step even begins."""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="1Y",
            anchor_date="2023-12-31"
        )
        # First test_start is 2024-01-01, which is > 2023-12-31.
        assert len(windows) == 0

    def test_exact_boundary_anchor(self):
        """Anchor exactly matches test_start. Generates a 1-day prediction slice."""
        windows = generate_rolling_windows(
            rolling_start="2020-01-01",
            train_years=3,
            valid_years=1,
            test_step="6M",
            anchor_date="2024-01-01"
        )
        assert len(windows) == 1
        w0 = windows[0]
        assert w0['test_start'] == "2024-01-01"
        assert w0['test_end'] == "2024-01-01"

    def test_leap_year_handling(self):
        """Years with leap days handle rolling accurately using relativedelta."""
        windows = generate_rolling_windows(
            rolling_start="2016-02-29",  # Leap day
            train_years=4,
            valid_years=1,
            test_step="1Y",
            anchor_date="2022-03-01"
        )
        # W0 Train: 2016-02-29 + 4Y - 1d = 2020-02-29 (Leap) - 1d = 2020-02-28
        # W0 Valid: 2020-02-29 to 2021-02-28 (1 year from leap day ends on feb 28)
        # W0 Test: 2021-02-28 + 1 year = 2022-02-28
        w0 = windows[0]
        assert w0['train_start'] == "2016-02-29"
        # 2016-02-29 + 4 years = 2020-02-29. Minus 1 day = 2020-02-28.
        assert w0['train_end'] == "2020-02-28" 
        assert w0['valid_start'] == "2020-02-29"
        # 2020-02-29 + 1 year = 2021-02-28. Minus 1 day = 2021-02-27.
        assert w0['valid_end'] == "2021-02-27"
        assert w0['test_start'] == "2021-02-28"
        # Test full end = 2021-02-28 + 1 year = 2022-02-28. Minus 1 day = 2022-02-27.
        assert w0['test_end'] == "2022-02-27"
        
        w1 = windows[1]
        # Train start: 2017-02-28
        assert w1['train_start'] == "2017-02-28"
        # Train end: 2017-02-28 + 4 years = 2021-02-28. Minus 1 day = 2021-02-27.
        assert w1['train_end'] == "2021-02-27"
        assert w1['valid_start'] == "2021-02-28"
        assert w1['test_start'] == "2022-02-28"

# ============================================================================
# Group 3: Prediction Concatenation Filtering
# ============================================================================

class TestPredictionFiltering:
    def test_filtering_within_bounds_single_index(self):
        dates = pd.date_range("2024-01-01", "2024-01-10")
        pred = pd.DataFrame({'score': range(10)})
        # Mock multi-index like Qlib uses typically: datetime, instrument
        arrays = [dates, ['SZ000001']*10]
        index = pd.MultiIndex.from_arrays(arrays, names=('datetime', 'instrument'))
        pred.index = index
        
        window = {
            'test_start': '2024-01-03',
            'test_end': '2024-01-07'
        }
        
        filtered = _filter_pred_to_test_segment(pred, window)
        assert len(filtered) == 5
        filtered_dates = filtered.index.get_level_values('datetime')
        assert filtered_dates.min() == pd.Timestamp('2024-01-03')
        assert filtered_dates.max() == pd.Timestamp('2024-01-07')

    def test_filtering_empty_slice(self):
        dates = pd.date_range("2024-01-01", "2024-01-10")
        pred = pd.DataFrame({'score': range(10)})
        index = pd.MultiIndex.from_arrays([dates, ['SZ000001']*10], names=('datetime', 'instrument'))
        pred.index = index
        
        window = {
            'test_start': '2024-02-01',
            'test_end': '2024-02-28'
        }
        
        filtered = _filter_pred_to_test_segment(pred, window)
        assert len(filtered) == 0

# ============================================================================
# Group 4: Rolling State Machine Verification
# ============================================================================

class TestRollingStateMachine:
    @pytest.fixture
    def state_file(self, tmp_path):
        return str(tmp_path / "test_rolling_state.json")

    def test_init_and_empty(self, state_file):
        state = RollingState(state_file=state_file)
        assert state._state['total_windows'] == 0
        assert not state.anchor_date
        
        state.init_run({'rolling_start': '2020'}, '2024-01-01', 5)
        assert state.anchor_date == '2024-01-01'
        assert state._state['total_windows'] == 5
        
    def test_mark_and_check_done(self, state_file):
        state = RollingState(state_file=state_file)
        state.init_run({}, '2024', 5)
        
        assert not state.is_window_model_done(0, "modelA")
        
        state.mark_window_model_done(0, "modelA", "rid_0_A")
        assert state.is_window_model_done(0, "modelA")
        assert not state.is_window_model_done(0, "modelB")
        assert not state.is_window_model_done(1, "modelA")
        
        state.mark_window_model_done(1, "modelA", "rid_1_A")
        state.mark_window_model_done(0, "modelB", "rid_0_B")
        
        records_a = state.get_completed_record_ids("modelA")
        assert len(records_a) == 2
        assert records_a[0] == {'window_idx': 0, 'record_id': 'rid_0_A'}
        assert records_a[1] == {'window_idx': 1, 'record_id': 'rid_1_A'}
        
        # Test reloading works
        state2 = RollingState(state_file=state_file)
        assert state2.is_window_model_done(1, "modelA")

    def test_remove_window(self, state_file):
        state = RollingState(state_file=state_file)
        state.init_run({}, '2024', 5)
        state.mark_window_model_done(0, "m1", "r0")
        state.mark_window_model_done(1, "m1", "r1")
        
        # Verify removal
        assert state.get_last_completed_window_idx() == 1
        res = state.remove_window(1)
        assert res is True
        assert state.get_last_completed_window_idx() == 0
        assert not state.is_window_model_done(1, "m1")
        
        # Removing non-existent
        assert state.remove_window(99) is False
        
    def test_clear_state(self, state_file):
        state = RollingState(state_file=state_file)
        state.init_run({}, '2024', 5)
        state.mark_window_model_done(0, "m1", "r0")
        
        with patch('quantpits.utils.train_utils.backup_file_with_date'):
            state.clear()
        
        # Re-initialize to verify
        state2 = RollingState(state_file=state_file)
        assert state2._state['total_windows'] == 0
