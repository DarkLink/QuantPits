# KRNN with IC-based early stopping.
# Inherits all logic from pytorch_krnn.KRNN; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_krnn_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_krnn import KRNN as _BaseKRNN, KRNNModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class KRNN(ICMetricMixin, _BaseKRNN):
    """KRNN with IC/Rank-IC early stopping metric."""
    pass
