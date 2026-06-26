"""
CPCV Dataset Classes

Provides DatasetH/TSDatasetH subclasses that support discontiguous
multi-segment training data for Purged Cross-Validation (CPCV).

- PurgedDatasetH: for tree/point-in-time models (LightGBM, XGBoost, Linear, etc.)
  Uses pd.concat to merge discontiguous time chunks -- safe because each row
  is an independent sample.

- PurgedTSDatasetH: for deep learning / time-series models (LSTM, GRU, ALSTM,
  Transformer, TCN, GATs, etc.). Uses ConcatTSDataSampler to logically
  concatenate multiple TSDataSampler instances, ensuring sliding windows
  NEVER cross purge gaps between discontiguous time chunks.

- ConcatTSDataSampler: wraps N TSDataSampler instances with full API surface
  (idx_df, get_index(), __len__, __getitem__) so Qlib models iterate over
  them transparently without bridge-sequence bugs.
"""

from typing import List, Union

import numpy as np
import pandas as pd

from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler


def is_multi_segment(slc) -> bool:
    """Detect if slc is a multi-segment list of [start, end] pairs.

    Single segment:  ["2020-01-01", "2022-12-31"]
    Multi segment:   [["2016-01-01", "2020-12-31"], ["2022-01-01", "2023-12-31"]]
    """
    return (
        isinstance(slc, list)
        and len(slc) > 0
        and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in slc)
    )


# ============================================================================
# PurgedDatasetH — for tree / point-in-time models
# ============================================================================

class PurgedDatasetH(DatasetH):
    """DatasetH that supports discontiguous multi-segment training data.

    When segments['train'] is [[s1,e1],[s2,e2]], fetches each sub-segment
    independently and pd.concat's them. Safe for point-in-time models
    (GBDT, Linear, XGBoost, CatBoost) where each row is an independent
    sample -- no sliding windows can cross sub-segment boundaries.

    Single-segment input delegates to parent DatasetH unchanged.
    """

    def _prepare_seg(self, slc, **kwargs):
        if is_multi_segment(slc):
            frames = []
            for sub_slc in slc:
                frames.append(super()._prepare_seg(sub_slc, **kwargs))
            merged = pd.concat(frames, axis=0)
            # Ensure Qlib's expected MultiIndex ordering (datetime, instrument)
            if isinstance(merged.index, pd.MultiIndex):
                merged = merged.sort_index()
            return merged
        return super()._prepare_seg(slc, **kwargs)


# ============================================================================
# ConcatTSDataSampler — logical concatenation for DL models
# ============================================================================

class ConcatTSDataSampler:
    """Logically concatenates multiple TSDataSampler instances.

    Each sub-sampler independently manages its own calendar, idx_map,
    and sliding windows. Sliding windows NEVER cross between sub-samplers
    because each sub-sampler's _extend_slice only pads within its own
    contiguous time chunk.

    Exposes the full API surface (idx_df, get_index(), __len__, __getitem__)
    that Qlib models and downstream strategies access at runtime.

    Analogous to torch.utils.data.ConcatDataset.
    """

    def __init__(self, samplers: List[TSDataSampler]):
        if not samplers:
            raise ValueError("ConcatTSDataSampler requires at least one sampler")
        self.samplers = samplers
        self._lengths = np.array([len(s) for s in samplers], dtype=np.int64)
        self._cum_lengths = np.cumsum(np.concatenate([[0], self._lengths]))

        # CRITICAL: use axis=0 (vertical concat), NOT axis=1.
        # Sub-samplers have disjoint time ranges (rows) but identical
        # column schema (instruments). axis=1 would outer-join rows
        # and create NaN-filled wide columns, corrupting .flatten() ops.
        self.idx_df = pd.concat([s.idx_df for s in samplers], axis=0)

        # Build flat lookup for O(1) dispatch when N > 2.
        # For the common N <= 2 case, __getitem__ uses inline if-else
        # which is faster than array indexing.
        if len(samplers) > 2:
            self._sampler_map = np.repeat(
                np.arange(len(samplers)), self._lengths
            )
            self._offsets = np.concatenate([[0], np.cumsum(self._lengths)[:-1]])

    # -- Public API (mirrors TSDataSampler) --

    @property
    def empty(self) -> bool:
        return len(self) == 0

    def config(self, **kwargs):
        """Propagate config to all sub-samplers (e.g. fillna_type)."""
        for s in self.samplers:
            s.config(**kwargs)

    def __len__(self) -> int:
        return int(self._cum_lengths[-1])

    def __getitem__(self, idx):
        # Handle batch indexing (list / np.ndarray from PyTorch DataLoader
        # with num_workers > 0). Dispatch each element to the correct
        # sub-sampler, then stack results.
        if isinstance(idx, (list, np.ndarray)):
            results = [self._get_single(int(i)) for i in idx]
            if len(results) == 0:
                return np.array([])
            return np.ascontiguousarray(np.stack(results, axis=0))

        # Handle (datetime, instrument) tuple key (like TSDataSampler)
        if isinstance(idx, tuple) and len(idx) == 2:
            # Delegate to each sub-sampler; the first that has this key wins.
            # This is the same semantics as TSDataSampler.__getitem__.
            for s in self.samplers:
                try:
                    return s[idx]
                except (IndexError, KeyError):
                    continue
            raise KeyError(f"{idx} not found in any sub-sampler")

        return self._get_single(int(idx))

    def _get_single(self, idx: int):
        """Get a single sample by integer index.

        Returns a contiguous numpy array to ensure downstream
        PyTorch tensors have proper memory alignment for CUDA.
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range")
        # Fast path for N <= 2
        if len(self.samplers) == 1:
            result = self.samplers[0][idx]
        elif len(self.samplers) == 2:
            if idx < self._lengths[0]:
                result = self.samplers[0][idx]
            else:
                result = self.samplers[1][idx - self._lengths[0]]
        else:
            s_idx = self._sampler_map[idx]
            local_idx = idx - int(self._offsets[s_idx])
            result = self.samplers[s_idx][local_idx]
        return np.ascontiguousarray(result)

    def get_index(self, idx=None):
        """Return concatenated MultiIndex; models call this for alignment.

        When idx is None, returns the full concatenated index across all
        sub-samplers. When idx is given, delegates to the correct sub-sampler.
        """
        if idx is not None:
            if idx < 0:
                idx += len(self)
            if len(self.samplers) == 1:
                return self.samplers[0].get_index(idx)
            elif len(self.samplers) == 2:
                if idx < self._lengths[0]:
                    return self.samplers[0].get_index(idx)
                else:
                    return self.samplers[1].get_index(idx - self._lengths[0])
            else:
                s_idx = self._sampler_map[idx]
                local_idx = idx - int(self._offsets[s_idx])
                return self.samplers[s_idx].get_index(local_idx)

        # pd.concat accepts Index/MultiIndex objects in Pandas >= 2.0.
        # For older Pandas, fallback to Index.append() to avoid TypeError.
        all_indices = [s.get_index() for s in self.samplers]
        try:
            result = pd.concat(all_indices)
        except TypeError:
            result = all_indices[0]
            for idx in all_indices[1:]:
                result = result.append(idx)
        # .sort_values() works on both Index and MultiIndex
        # (MultiIndex has no .sort_index() method)
        try:
            return result.sort_values()
        except TypeError:
            # Older pandas: sort_values may not exist on plain Index
            return result


# ============================================================================
# PurgedTSDatasetH — for deep learning / time-series models
# ============================================================================

class PurgedTSDatasetH(TSDatasetH):
    """TSDatasetH that supports discontiguous multi-segment training.

    Instead of pd.concat (which would create bridge-sequence bugs --
    sliding windows crossing purge gaps and producing "Frankenstein"
    sequences), creates one TSDataSampler per contiguous sub-segment
    and wraps them in a ConcatTSDataSampler.

    Each sub-TSDataSampler independently manages its own _extend_slice
    and calendar, so sliding windows NEVER cross purge gaps. The
    concatenation is purely logical (index mapping), not physical.

    Single-segment input delegates to parent TSDatasetH unchanged.
    """

    def _prepare_seg(self, slc, **kwargs):
        if is_multi_segment(slc):
            samplers = []
            for sub_slc in slc:
                # Each sub-segment gets its own TSDataSampler with its own
                # calendar-aware _extend_slice -- windows can't cross gaps.
                sampler = super()._prepare_seg(sub_slc, **kwargs)
                samplers.append(sampler)
            return ConcatTSDataSampler(samplers)
        return super()._prepare_seg(slc, **kwargs)


# ============================================================================
# PurgedMTSDatasetH — for TRA models (Memory-Augmented Time Series)
# ============================================================================

# PurgedMTSDatasetH: only defined if qlib.contrib MTSDatasetH is available.
# The TRA model requires MTSDatasetH; it's in qlib.contrib.data.dataset.
try:
    from qlib.contrib.data.dataset import MTSDatasetH, _get_date_parse_fn
    import copy

    class PurgedMTSDatasetH(MTSDatasetH):
        """MTSDatasetH with multi-segment support for CPCV.

        Filters _batch_slices and _daily_slices against the UNION of all
        sub-segment date ranges. The underlying numpy arrays are shared
        (shallow copy), only the slice views change.
        """

        def _prepare_seg(self, slc, **kwargs):
            if is_multi_segment(slc):
                fn = _get_date_parse_fn(self._index[0][1])
                obj = copy.copy(self)  # shallow copy
                obj._data = self._data
                obj._label = self._label
                obj._index = self._index
                obj._memory = self._memory
                obj._zeros = self._zeros

                date_index = self._index.get_level_values(1)
                batch_masks = []
                daily_masks = []
                for sub_slc in slc:
                    start_date = pd.Timestamp(fn(sub_slc[0]))
                    end_date = pd.Timestamp(fn(sub_slc[1]))
                    batch_masks.append(
                        (date_index >= start_date) & (date_index <= end_date)
                    )
                    daily_masks.append(
                        (self._daily_index.values >= start_date)
                        & (self._daily_index.values <= end_date)
                    )

                # Union: keep any sample in ANY sub-segment
                combined_batch = np.any(np.stack(batch_masks, axis=0), axis=0)
                combined_daily = np.any(np.stack(daily_masks, axis=0), axis=0)

                obj._batch_slices = self._batch_slices[combined_batch]
                obj._daily_slices = self._daily_slices[combined_daily]
                obj._daily_index = self._daily_index[combined_daily]
                return obj
            return super()._prepare_seg(slc, **kwargs)

except ImportError:
    PurgedMTSDatasetH = None  # qlib.contrib not available
