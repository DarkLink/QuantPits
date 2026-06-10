from qlib.contrib.model.pytorch_krnn import KRNN as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class KRNN(StrategyMetricMixin, _Base):
    pass
