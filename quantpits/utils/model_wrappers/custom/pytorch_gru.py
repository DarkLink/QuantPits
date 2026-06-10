from qlib.contrib.model.pytorch_gru import GRU as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class GRU(StrategyMetricMixin, _Base):
    pass
