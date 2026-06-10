from qlib.contrib.model.pytorch_alstm import ALSTM as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class ALSTM(StrategyMetricMixin, _Base):
    pass
