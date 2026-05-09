from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin

class TransformerModelIC(ICMetricMixin, TransformerModel):
    pass
