# TCN with IC-based early stopping.
# Inherits all logic from pytorch_tcn.TCN; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_tcn_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_tcn import TCN as _BaseTCN, TCNModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class TCN(ICMetricMixin, _BaseTCN):
    """TCN with IC/Rank-IC early stopping metric."""
    pass
