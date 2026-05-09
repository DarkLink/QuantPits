# GeneralPTNN with IC-based early stopping.
# Inherits all logic from pytorch_general_nn.GeneralPTNN; only metric_fn is
# overridden via ICMetricMixin, and its output is negated because GeneralPTNN's
# fit() method strictly expects a loss-like metric where LOWER is better
# (val_score < best_score).
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_general_nn_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_general_nn import GeneralPTNN as _BaseGeneralPTNN
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class GeneralPTNN(ICMetricMixin, _BaseGeneralPTNN):
    """GeneralPTNN with IC/Rank-IC early stopping metric."""

    def metric_fn(self, pred, label):
        # ICMetricMixin returns the IC score (higher is better)
        # or the negative loss (higher is better).
        # GeneralPTNN's fit() uses `val_score < best_score` (lower is better).
        # We must return the negative of the metric so that a higher IC
        # becomes a lower negated value.
        # FIX: squeeze pred to avoid [N, 1] * [N] broadcasting to [N, N] in Pearson correlation
        val = super().metric_fn(pred.squeeze(), label.squeeze())
        return -val
