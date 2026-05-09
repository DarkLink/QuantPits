# GRU with IC-based early stopping.
# Inherits all logic from pytorch_gru.GRU; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_gru_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_gru import GRU as _BaseGRU, GRUModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class GRU(ICMetricMixin, _BaseGRU):
    """GRU with IC/Rank-IC early stopping metric."""
    pass
