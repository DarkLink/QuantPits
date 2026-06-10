from qlib.contrib.model.pytorch_tcn import TCN as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class TCN(StrategyMetricMixin, _Base):
    pass
