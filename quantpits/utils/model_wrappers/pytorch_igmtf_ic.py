# IGMTF with IC-based early stopping.
# Inherits all logic from pytorch_igmtf.IGMTF; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_igmtf_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_igmtf import IGMTF as _BaseIGMTF, IGMTFModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class IGMTF(ICMetricMixin, _BaseIGMTF):
    """IGMTF with IC/Rank-IC early stopping metric."""
    pass
