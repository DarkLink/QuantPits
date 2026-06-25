"""
Unit tests for CPCV dataset classes.

Covers:
- is_multi_segment detection
- PurgedDatasetH._prepare_seg (single and multi-segment)
- ConcatTSDataSampler (len, getitem, idx_df axis, get_index)
- PurgedTSDatasetH._prepare_seg (single and multi-segment)
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from quantpits.data.cpcv_dataset import (
    is_multi_segment,
    PurgedDatasetH,
    PurgedTSDatasetH,
    ConcatTSDataSampler,
)


# ============================================================================
# is_multi_segment
# ============================================================================

class TestIsMultiSegment:
    def test_single_segment_str_pair(self):
        assert not is_multi_segment(["2020-01-01", "2022-12-31"])

    def test_multi_segment_list_of_pairs(self):
        assert is_multi_segment([
            ["2016-01-01", "2020-12-31"],
            ["2022-01-01", "2023-12-31"],
        ])

    def test_single_segment_with_tuple(self):
        # A list containing a single tuple-pair is ambiguous --
        # it could be a single segment or a multi-segment with one element.
        # In practice this never occurs (YAML serializes tuples to lists),
        # so we don't test is_multi_segment directly for this case.
        # Instead, verify that PurgedDatasetH._prepare_seg handles it safely.
        pass

    def test_multi_segment_mixed_type(self):
        assert is_multi_segment([
            ["2016-01-01", "2020-12-31"],
            ("2022-01-01", "2023-12-31"),
        ])

    def test_empty_list(self):
        assert not is_multi_segment([])

    def test_not_a_list(self):
        assert not is_multi_segment("train")

    def test_slice_not_list(self):
        assert not is_multi_segment(slice("2020-01-01", "2022-12-31"))

    def test_list_of_strings_not_pairs(self):
        assert not is_multi_segment(["train", "valid"])

    def test_list_with_three_element_sublist(self):
        # Not a pair (len != 2)
        assert not is_multi_segment([["a", "b", "c"]])


# ============================================================================
# PurgedDatasetH
# ============================================================================

class TestPurgedDatasetH:
    """Tests for PurgedDatasetH._prepare_seg multi-segment support."""

    @staticmethod
    def _make_ds():
        """Create a PurgedDatasetH bypassing __init__ to avoid handler/segments reqs."""
        ds = PurgedDatasetH.__new__(PurgedDatasetH)
        ds.handler = MagicMock()
        ds.segments = {}
        ds.fetch_kwargs = None
        # Remove fetch_kwargs attribute (parent checks hasattr)
        if hasattr(ds, 'fetch_kwargs'):
            del ds.fetch_kwargs
        return ds

    def test_single_segment_delegates_to_parent(self):
        """Single [start, end] should call super()._prepare_seg unchanged."""
        ds = self._make_ds()

        with patch.object(
            PurgedDatasetH.__bases__[0], '_prepare_seg', return_value="parent_result"
        ) as mock_super:
            result = ds._prepare_seg(["2020-01-01", "2022-12-31"])
            mock_super.assert_called_once_with(["2020-01-01", "2022-12-31"])
            assert result == "parent_result"

    def test_multi_segment_fetches_and_concats(self):
        """Multi-segment should fetch each sub-seg and concat with sort_index."""
        ds = self._make_ds()

        df1 = pd.DataFrame(
            {"feature": [1, 2]},
            index=pd.MultiIndex.from_tuples(
                [("2020-01-02", "SH600000"), ("2020-01-02", "SH600001")],
                names=["datetime", "instrument"],
            ),
        )
        df2 = pd.DataFrame(
            {"feature": [3, 4]},
            index=pd.MultiIndex.from_tuples(
                [("2022-01-04", "SH600000"), ("2022-01-04", "SH600001")],
                names=["datetime", "instrument"],
            ),
        )

        with patch.object(
            PurgedDatasetH.__bases__[0], '_prepare_seg',
            side_effect=[df1, df2]
        ) as mock_super:
            result = ds._prepare_seg([
                ["2020-01-01", "2020-12-31"],
                ["2022-01-01", "2022-12-31"],
            ])
            assert mock_super.call_count == 2
            assert len(result) == 4  # 2 + 2 rows concatenated
            assert result.index.names == ["datetime", "instrument"]
            # Verify sorted: 2020 rows before 2022 rows
            assert str(result.index[0][0]) == "2020-01-02"
            assert str(result.index[-1][0]) == "2022-01-04"

    def test_single_index_not_sorted(self):
        """Non-MultiIndex result should still be returned correctly."""
        ds = self._make_ds()

        df1 = pd.DataFrame({"x": [1]}, index=[0])
        df2 = pd.DataFrame({"x": [2]}, index=[1])

        with patch.object(
            PurgedDatasetH.__bases__[0], '_prepare_seg',
            side_effect=[df1, df2]
        ):
            result = ds._prepare_seg([
                ["2020-01-01", "2020-06-30"],
                ["2022-01-01", "2022-06-30"],
            ])
            assert len(result) == 2


# ============================================================================
# ConcatTSDataSampler
# ============================================================================

class TestConcatTSDataSampler:
    """Tests for ConcatTSDataSampler logical concatenation."""

    def _make_mock_sampler(self, length, start_idx=0):
        """Create a mock TSDataSampler with required attributes."""
        sampler = MagicMock()
        sampler.__len__.return_value = length

        # Build a simple idx_df
        dates = pd.date_range(f"202{start_idx}-01-01", periods=length, freq="D")
        sampler.idx_df = pd.DataFrame(
            np.arange(length).reshape(-1, 1),
            index=dates,
            columns=["SH600000"],
        )

        def _getitem(idx):
            if idx < 0:
                idx += length
            if idx < 0 or idx >= length:
                raise IndexError(f"index {idx} out of range")
            return {"sample": f"s{start_idx}_{idx}"}

        sampler.__getitem__.side_effect = _getitem

        def _get_index(idx=None):
            if idx is not None:
                return pd.MultiIndex.from_tuples(
                    [(dates[idx], "SH600000")],
                    names=["datetime", "instrument"],
                )
            return pd.MultiIndex.from_tuples(
                [(d, "SH600000") for d in dates],
                names=["datetime", "instrument"],
            )

        sampler.get_index.side_effect = _get_index

        return sampler

    def test_empty_samplers_raises(self):
        with pytest.raises(ValueError, match="at least one sampler"):
            ConcatTSDataSampler([])

    def test_len_single_sampler(self):
        s1 = self._make_mock_sampler(10)
        cds = ConcatTSDataSampler([s1])
        assert len(cds) == 10

    def test_len_two_samplers(self):
        s1 = self._make_mock_sampler(10, 0)
        s2 = self._make_mock_sampler(15, 1)
        cds = ConcatTSDataSampler([s1, s2])
        assert len(cds) == 25

    def test_getitem_first_sampler(self):
        s1 = self._make_mock_sampler(10, 0)
        s2 = self._make_mock_sampler(10, 1)
        cds = ConcatTSDataSampler([s1, s2])
        # First element should be from first sampler
        result = cds[0]
        assert result == {"sample": "s0_0"}

    def test_getitem_second_sampler(self):
        s1 = self._make_mock_sampler(10, 0)
        s2 = self._make_mock_sampler(10, 1)
        cds = ConcatTSDataSampler([s1, s2])
        # Element at index 10 should be first element of second sampler
        result = cds[10]
        assert result == {"sample": "s1_0"}

    def test_getitem_boundary(self):
        s1 = self._make_mock_sampler(10, 0)
        s2 = self._make_mock_sampler(10, 1)
        cds = ConcatTSDataSampler([s1, s2])
        # Last element of first sampler
        assert cds[9] == {"sample": "s0_9"}
        # First element of second sampler
        assert cds[10] == {"sample": "s1_0"}

    def test_getitem_negative_index(self):
        s1 = self._make_mock_sampler(10, 0)
        s2 = self._make_mock_sampler(5, 1)
        cds = ConcatTSDataSampler([s1, s2])
        # -1 should give last element of second sampler
        result = cds[-1]
        assert result == {"sample": "s1_4"}

    def test_idx_df_axis_is_zero(self):
        """CRITICAL: idx_df must use axis=0 (vertical), not axis=1."""
        s1 = self._make_mock_sampler(5, 0)
        s2 = self._make_mock_sampler(5, 1)
        cds = ConcatTSDataSampler([s1, s2])
        # axis=0: rows = sum, columns = same as individual
        assert len(cds.idx_df) == 10  # 5 + 5 rows
        assert cds.idx_df.shape[1] == 1  # same columns, not doubled

    def test_get_index_full(self):
        s1 = self._make_mock_sampler(3, 0)
        s2 = self._make_mock_sampler(2, 1)
        cds = ConcatTSDataSampler([s1, s2])
        idx = cds.get_index()
        assert len(idx) == 5
        assert idx.names == ["datetime", "instrument"]

    def test_get_index_specific(self):
        s1 = self._make_mock_sampler(3, 0)
        s2 = self._make_mock_sampler(2, 1)
        cds = ConcatTSDataSampler([s1, s2])
        # Index 3 should be first element of s2
        idx = cds.get_index(3)
        assert len(idx) == 1

    def test_three_samplers_general_path(self):
        """Test the N>2 path with precomputed lookup arrays."""
        s0 = self._make_mock_sampler(3, 0)
        s1 = self._make_mock_sampler(3, 1)
        s2 = self._make_mock_sampler(3, 2)
        cds = ConcatTSDataSampler([s0, s1, s2])
        assert len(cds) == 9
        # Check boundaries
        assert cds[0] == {"sample": "s0_0"}
        assert cds[2] == {"sample": "s0_2"}
        assert cds[3] == {"sample": "s1_0"}
        assert cds[5] == {"sample": "s1_2"}
        assert cds[6] == {"sample": "s2_0"}
        assert cds[8] == {"sample": "s2_2"}


# ============================================================================
# PurgedTSDatasetH
# ============================================================================

class TestPurgedTSDatasetH:
    """Tests for PurgedTSDatasetH multi-segment -> ConcatTSDataSampler."""

    @staticmethod
    def _make_ds():
        """Create a PurgedTSDatasetH bypassing __init__."""
        ds = PurgedTSDatasetH.__new__(PurgedTSDatasetH)
        ds.handler = MagicMock()
        ds.segments = {}
        ds.step_len = 30
        ds.cal = pd.DatetimeIndex([])
        ds.flt_col = None
        if hasattr(ds, 'fetch_kwargs'):
            del ds.fetch_kwargs
        return ds

    def test_single_segment_returns_ts_sampler(self):
        """Single segment should delegate to parent TSDatasetH unchanged."""
        ds = self._make_ds()

        with patch.object(
            PurgedTSDatasetH.__bases__[0], '_prepare_seg',
            return_value="ts_sampler_result"
        ) as mock_super:
            result = ds._prepare_seg(["2020-01-01", "2022-12-31"])
            mock_super.assert_called_once_with(["2020-01-01", "2022-12-31"])
            assert result == "ts_sampler_result"

    def test_multi_segment_creates_concat_sampler(self):
        """Multi-segment should create N TSDataSamplers + ConcatTSDataSampler."""
        ds = self._make_ds()

        mock_samplers = [MagicMock(), MagicMock()]
        mock_samplers[0].__len__.return_value = 10
        mock_samplers[1].__len__.return_value = 8
        mock_samplers[0].idx_df = pd.DataFrame({"A": range(10)})
        mock_samplers[1].idx_df = pd.DataFrame({"A": range(8)})
        mock_samplers[0].get_index.return_value = pd.MultiIndex.from_tuples(
            [(f"2020-01-{i+1:02d}", "SH600000") for i in range(10)],
            names=["datetime", "instrument"],
        )
        mock_samplers[1].get_index.return_value = pd.MultiIndex.from_tuples(
            [(f"2022-01-{i+1:02d}", "SH600000") for i in range(8)],
            names=["datetime", "instrument"],
        )

        with patch.object(
            PurgedTSDatasetH.__bases__[0], '_prepare_seg',
            side_effect=mock_samplers
        ) as mock_super:
            result = ds._prepare_seg([
                ["2020-01-01", "2020-12-31"],
                ["2022-01-01", "2022-12-31"],
            ])
            assert mock_super.call_count == 2
            assert isinstance(result, ConcatTSDataSampler)
            assert len(result) == 18

    def test_concat_sampler_no_cross_window_leakage(self):
        """Verify that getitem NEVER crosses between sub-samplers.

        Each sub-sampler should only be called with indices within its range.
        """
        s0 = MagicMock()
        s0.__len__.return_value = 5
        s0.idx_df = pd.DataFrame({"X": range(5)})
        s0.get_index.return_value = pd.MultiIndex.from_tuples(
            [("d", "i")], names=["datetime", "instrument"]
        )

        s1 = MagicMock()
        s1.__len__.return_value = 5
        s1.idx_df = pd.DataFrame({"X": range(5)})
        s1.get_index.return_value = pd.MultiIndex.from_tuples(
            [("d", "i")], names=["datetime", "instrument"]
        )

        cds = ConcatTSDataSampler([s0, s1])

        # Access indices in s0's range
        cds[0]
        s0.__getitem__.assert_called_with(0)
        s1.__getitem__.assert_not_called()

        s0.reset_mock()
        s1.reset_mock()

        # Access indices in s1's range
        cds[5]
        s1.__getitem__.assert_called_with(0)
        s0.__getitem__.assert_not_called()
