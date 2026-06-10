from qlib.contrib.model.pytorch_sandwich import Sandwich as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class Sandwich(StrategyMetricMixin, _Base):
    pass
