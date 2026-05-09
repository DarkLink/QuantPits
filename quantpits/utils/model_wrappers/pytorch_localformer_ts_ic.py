from qlib.contrib.model.pytorch_localformer_ts import LocalformerModel
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin

class LocalformerModelIC(ICMetricMixin, LocalformerModel):
    pass
