from qlib.contrib.model.pytorch_tcn_ts import TCN
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin

class TCNIC(ICMetricMixin, TCN):
    pass
