from qlib.contrib.model.pytorch_sfm import SFM as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class SFM(StrategyMetricMixin, _Base):
    pass
