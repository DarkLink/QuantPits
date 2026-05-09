# SFM with IC-based early stopping.
# Inherits all logic from pytorch_sfm.SFM; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_sfm_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_sfm import SFM as _BaseSFM, SFM_Model  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class SFM(ICMetricMixin, _BaseSFM):
    """SFM with IC/Rank-IC early stopping metric."""
    pass
