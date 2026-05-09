# ALSTM with IC-based early stopping (non-timeseries variant).
# Inherits all logic from pytorch_alstm.ALSTM; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_alstm_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_alstm import ALSTM as _BaseALSTM, ALSTMModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class ALSTM(ICMetricMixin, _BaseALSTM):
    """ALSTM with IC/Rank-IC early stopping metric."""
    pass
