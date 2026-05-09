# ALSTM-TS with IC-based early stopping (timeseries variant).
# Inherits all logic from pytorch_alstm_ts.ALSTM; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_alstm_ts_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_alstm_ts import ALSTM as _BaseALSTM, ALSTMModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class ALSTM(ICMetricMixin, _BaseALSTM):
    """ALSTM (TS) with IC/Rank-IC early stopping metric."""
    pass
