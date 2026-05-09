# TabNet with IC-based early stopping.
# Inherits all logic from pytorch_tabnet.TabnetModel; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_tabnet_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_tabnet import TabnetModel as _BaseTabnetModel, TabNet  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class TabnetModel(ICMetricMixin, _BaseTabnetModel):
    """TabnetModel with IC/Rank-IC early stopping metric."""
    pass
