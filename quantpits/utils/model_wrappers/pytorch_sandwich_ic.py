# Sandwich with IC-based early stopping.
# Inherits all logic from pytorch_sandwich.Sandwich; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_sandwich_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_sandwich import Sandwich as _BaseSandwich, SandwichModel  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class Sandwich(ICMetricMixin, _BaseSandwich):
    """Sandwich with IC/Rank-IC early stopping metric."""
    pass
