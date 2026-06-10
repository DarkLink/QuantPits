from qlib.contrib.model.pytorch_lstm import LSTM as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class LSTM(StrategyMetricMixin, _Base):
    pass
