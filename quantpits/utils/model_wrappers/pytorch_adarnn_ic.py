# ADARNN with IC-based early stopping.
# Inherits all logic from pytorch_adarnn.ADARNN; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_adarnn_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_adarnn import ADARNN as _BaseADARNN, AdaRNN  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class ADARNN(ICMetricMixin, _BaseADARNN):
    """ADARNN with IC/Rank-IC early stopping metric."""
    pass
