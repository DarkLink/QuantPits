# Localformer with IC-based early stopping.
# Inherits all logic from pytorch_localformer.LocalformerModel; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_localformer_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_localformer import LocalformerModel as _BaseLocalformerModel, Transformer  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class LocalformerModel(ICMetricMixin, _BaseLocalformerModel):
    """LocalformerModel with IC/Rank-IC early stopping metric."""
    pass
