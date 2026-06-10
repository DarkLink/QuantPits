from qlib.contrib.model.pytorch_igmtf import IGMTF as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class IGMTF(StrategyMetricMixin, _Base):
    pass
