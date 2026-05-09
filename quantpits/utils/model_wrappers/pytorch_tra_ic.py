from qlib.contrib.model.pytorch_tra import TRAModel
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin

class TRAModelIC(ICMetricMixin, TRAModel):
    def __init__(self, metric="loss", **kwargs):
        self.metric = metric
        super().__init__(**kwargs)
