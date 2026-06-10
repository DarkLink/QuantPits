from qlib.contrib.model.pytorch_adarnn import ADARNN as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class ADARNN(StrategyMetricMixin, _Base):
    pass
