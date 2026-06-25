"""
QuantPits Data Layer

Custom dataset classes that extend Qlib's DatasetH/TSDatasetH for
advanced time-series splitting patterns (CPCV, purged cross-validation).
"""

from quantpits.data.cpcv_dataset import (
    PurgedDatasetH,
    PurgedTSDatasetH,
    ConcatTSDataSampler,
    is_multi_segment,
)

try:
    from quantpits.data.cpcv_dataset import PurgedMTSDatasetH
except ImportError:
    PurgedMTSDatasetH = None

__all__ = [
    "PurgedDatasetH",
    "PurgedTSDatasetH",
    "PurgedMTSDatasetH",
    "ConcatTSDataSampler",
    "is_multi_segment",
]
