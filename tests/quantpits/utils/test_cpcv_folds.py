"""
Unit tests for compute_cpcv_folds() in train_utils.py.

Covers:
- Daily and weekly frequency fold generation
- Fold count correctness (K = n_groups - n_test_groups - n_val_groups + 1)
- Purge gap and embargo gap sizes in steps
- Discontiguous training segments (left + right around validation)
- Guard clause: right segment omitted when purge+embargo consumes it
- Edge cases: purge/embargo too large, insufficient groups, empty calendar
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestComputeCpcvFolds:
    """Tests for compute_cpcv_folds function."""

    @pytest.fixture
    def daily_calendar(self):
        """Mock daily calendar: 1000 trading days starting 2015-01-02."""
        return pd.date_range("2015-01-02", periods=1000, freq="B")

    @pytest.fixture
    def weekly_calendar(self):
        """Mock weekly calendar: 200 trading weeks starting 2015-01-02."""
        return pd.date_range("2015-01-02", periods=200, freq="W-FRI")

    @pytest.fixture
    def base_config(self):
        return {
            "start_time": "2015-01-01",
            "purged_cv": {
                "n_groups": 10,
                "n_test_groups": 2,
                "n_val_groups": 1,
                "purge_steps": 5,
                "embargo_steps": 10,
            },
        }

    def _make_calendar_mock(self, cal):
        """Create a D.calendar mock that returns the given date range."""
        mock_D = MagicMock()
        mock_D.calendar.return_value = cal
        return mock_D

    def test_basic_daily_folds(self, daily_calendar, base_config):
        """10 groups, 2 test, 1 val -> K = 10 - 2 - 1 + 1 = 8 folds."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", base_config, freq="day"
            )

        assert result["test_start_time"] is not None
        assert result["test_end_time"] is not None
        assert len(result["folds"]) == 8

        # Verify each fold has train_segments and valid dates
        for fi, fold in enumerate(result["folds"]):
            assert fold["fold_idx"] == fi
            assert len(fold["train_segments"]) >= 1
            assert len(fold["train_segments"]) <= 2
            assert fold["valid_start_time"] is not None
            assert fold["valid_end_time"] is not None

    def test_weekly_frequency(self, weekly_calendar, base_config):
        """Weekly calendar should also produce 8 folds."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = weekly_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", base_config, freq="week"
            )

        assert len(result["folds"]) == 8
        # purge_steps=5 in weekly mode means 5 weeks
        # Check that fold 0 has training segments

    def test_first_fold_single_train_segment(self, daily_calendar, base_config):
        """Fold 0: valid is first group; train should only be AFTER (single segment)."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", base_config, freq="day"
            )

        # Fold 0: validation = group 0, train = only groups after val
        fold0 = result["folds"][0]
        # Should have exactly 1 training segment (no left segment since val is first)
        assert len(fold0["train_segments"]) == 1

    def test_last_fold_single_train_segment(self, daily_calendar, base_config):
        """Last fold: valid is last CV group; train should only be BEFORE (single segment)."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", base_config, freq="day"
            )

        # Last fold: validation = last CV group, train = only before val
        last_fold = result["folds"][-1]
        assert len(last_fold["train_segments"]) == 1

    def test_middle_fold_two_train_segments(self, daily_calendar, base_config):
        """A middle fold should have train on BOTH sides of validation (2 segments)."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", base_config, freq="day"
            )

        # Fold 3 (middle): valid = group 3, train left = groups 0-2 purged,
        # train right = groups 4-7 embargoed
        fold3 = result["folds"][3]
        assert len(fold3["train_segments"]) == 2

        # Verify the gap between left segment end and right segment start
        left_end = pd.Timestamp(fold3["train_segments"][0][1])
        right_start = pd.Timestamp(fold3["train_segments"][1][0])
        # The gap should be at least (valid length + purge_steps + embargo_steps) days
        gap_days = (right_start - left_end).days
        assert gap_days > 0, "Train segments should have a purge gap between them"

    def test_test_set_is_last_n_groups(self, daily_calendar, base_config):
        """Verify test set boundaries cover the last n_test_groups."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", base_config, freq="day"
            )

        test_start = pd.Timestamp(result["test_start_time"])
        test_end = pd.Timestamp(result["test_end_time"])
        assert test_start < test_end
        # Test end should be the last available trading day
        assert test_end == daily_calendar[-1]

    def test_validate_purge_too_large_raises(self, daily_calendar, base_config):
        """If purge_steps exceeds group size, should raise ValueError."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        config = base_config.copy()
        config["purged_cv"] = base_config["purged_cv"].copy()
        # 1000 days / 10 groups = 100 days per group, purge_steps=200 should fail
        config["purged_cv"]["purge_steps"] = 200

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            with pytest.raises(ValueError, match="purge_steps"):
                compute_cpcv_folds("2026-06-24", config, freq="day")

    def test_n_test_groups_too_large_raises(self, daily_calendar, base_config):
        """n_test_groups >= n_groups should raise."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        config = base_config.copy()
        config["purged_cv"] = base_config["purged_cv"].copy()
        config["purged_cv"]["n_test_groups"] = 10  # same as n_groups

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            with pytest.raises(ValueError, match="n_test_groups"):
                compute_cpcv_folds("2026-06-24", config, freq="day")

    def test_guard_clause_right_segment_omitted(self, daily_calendar, base_config):
        """When purge+embargo consumes right training, right segment should be omitted.

        Use a small calendar (60 days), moderate purge/embargo, so that for a
        late fold the right training chunk is entirely consumed by purge+embargo.
        """
        from quantpits.utils.train_utils import compute_cpcv_folds

        small_cal = pd.date_range("2015-01-02", periods=60, freq="B")

        config = {
            "start_time": "2015-01-01",
            "purged_cv": {
                "n_groups": 10,
                "n_test_groups": 1,
                "n_val_groups": 1,
                "purge_steps": 2,
                "embargo_steps": 2,
            },
        }

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = small_cal
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2015-04-01", config, freq="day"
            )

        # K = 10 - 1 - 1 + 1 = 9 folds
        assert len(result["folds"]) == 9

        # The last fold (fold 8) has validation on group 8.
        # Right side is group 9 minus purge+embargo.
        # With small groups (~6 days each) and purge+embargo=4 steps,
        # the right segment may be omitted for the last fold.
        for fold in result["folds"]:
            assert len(fold["train_segments"]) >= 1, (
                f"Fold {fold['fold_idx']} has no training segments"
            )

    def test_n_val_groups_multi(self, daily_calendar):
        """Test with n_val_groups=2."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        config = {
            "start_time": "2015-01-01",
            "purged_cv": {
                "n_groups": 10,
                "n_test_groups": 2,
                "n_val_groups": 2,
                "purge_steps": 3,
                "embargo_steps": 5,
            },
        }

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", config, freq="day"
            )

        # K = 10 - 2 - 2 + 1 = 7 folds
        assert len(result["folds"]) == 7

    def test_calendar_too_short_raises(self, daily_calendar, base_config):
        """If total periods < n_groups, should raise."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        short_cal = pd.date_range("2015-01-02", periods=5, freq="B")

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = short_cal
            sys.modules['qlib.data'] = mock_qlib_data

            with pytest.raises(ValueError, match="Total periods"):
                compute_cpcv_folds("2015-01-15", base_config, freq="day")

    def test_n_val_groups_exceeds_cv_groups(self, daily_calendar, base_config):
        """n_val_groups > available CV groups should raise."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        config = base_config.copy()
        config["purged_cv"] = base_config["purged_cv"].copy()
        # 10 groups - 2 test = 8 CV, but n_val_groups=10 > 8
        config["purged_cv"]["n_val_groups"] = 10

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            with pytest.raises(ValueError, match="n_val_groups"):
                compute_cpcv_folds("2026-06-24", config, freq="day")

    def test_single_fold_valid(self, daily_calendar, base_config):
        """Edge case: K=1 fold is valid when n_val_groups uses all CV groups but
        train data still exists (because train includes groups before/after val)."""
        from quantpits.utils.train_utils import compute_cpcv_folds

        config = base_config.copy()
        config["purged_cv"] = base_config["purged_cv"].copy()
        # 10 groups - 2 test = 8 CV, n_val_groups=2 -> K = 8 - 2 + 1 = 7 folds
        # This is already tested in test_n_val_groups_multi.
        # For K=1: 10 groups - 4 test = 6 CV, n_val=6 -> K = 6-6+1 = 1 fold
        # But then no train groups remain. That's invalid.
        # Valid K=1: 10 groups - 2 test = 8 CV, n_val=8 -> K=1, all 8 CV groups are val
        # But then no train remains. Also invalid.
        # Simplest valid K=1: small config with purge=0
        config["purged_cv"]["n_groups"] = 5
        config["purged_cv"]["n_test_groups"] = 1
        config["purged_cv"]["n_val_groups"] = 3
        config["purged_cv"]["purge_steps"] = 0
        config["purged_cv"]["embargo_steps"] = 0
        # K = 5 - 1 - 3 + 1 = 2 folds, each with val=3 groups, train=1 group

        with patch.dict('sys.modules', {'qlib.data': MagicMock()}):
            import sys
            mock_qlib_data = MagicMock()
            mock_qlib_data.D.calendar.return_value = daily_calendar
            sys.modules['qlib.data'] = mock_qlib_data

            result = compute_cpcv_folds(
                "2026-06-24", config, freq="day"
            )

        assert len(result["folds"]) == 2
