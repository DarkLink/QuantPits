# Transformer with IC-based early stopping.
# Inherits all logic from pytorch_transformer.TransformerModel; only metric_fn is
# overridden via ICMetricMixin.
#
# YAML usage:
#   module_path: qlib.contrib.model.pytorch_transformer_ic
#   kwargs:
#     metric: ic   # or rank_ic

from qlib.contrib.model.pytorch_transformer import TransformerModel as _BaseTransformerModel, Transformer  # noqa: F401
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin


class TransformerModel(ICMetricMixin, _BaseTransformerModel):
    """TransformerModel with IC/Rank-IC early stopping metric."""
    pass
