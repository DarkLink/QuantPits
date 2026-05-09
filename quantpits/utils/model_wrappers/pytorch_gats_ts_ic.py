# GATs with IC-based early stopping.
# Inherits all logic from pytorch_gats_ts.GATs; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_gats_ts_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_gats_ts import GATs as _BaseGATs, GATModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class GATs(ICMetricMixin, _BaseGATs):
    """GATs with IC/Rank-IC early stopping metric."""
    pass
