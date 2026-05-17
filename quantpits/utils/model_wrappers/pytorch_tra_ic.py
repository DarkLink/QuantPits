from qlib.contrib.model.pytorch_tra import TRAModel
from quantpits.utils.model_wrappers.ic_metric_mixin import ICMetricMixin

class TRAModelIC(ICMetricMixin, TRAModel):
    def __init__(self, metric="loss", **kwargs):
        self.metric = metric
        super().__init__(**kwargs)

    def __setstate__(self, state):
        # Qlib's Serializable.__getstate__ drops _-prefixed attrs
        # (e.g. _writer) by default. Re-initialize them here so the
        # model survives a pickle roundtrip for --predict-only.
        super().__setstate__(state)
        if not hasattr(self, "_writer"):
            self._writer = None
