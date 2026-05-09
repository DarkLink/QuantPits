# LSTM with IC-based early stopping.
# Inherits all logic from pytorch_lstm.LSTM; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_lstm_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_lstm import LSTM as _BaseLSTM, LSTMModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class LSTM(ICMetricMixin, _BaseLSTM):
    """LSTM with IC/Rank-IC early stopping metric."""
    pass
