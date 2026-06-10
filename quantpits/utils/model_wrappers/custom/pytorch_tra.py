from qlib.contrib.model.pytorch_tra import TRAModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class TRAModelIC(StrategyMetricMixin, _Base):
    pass
