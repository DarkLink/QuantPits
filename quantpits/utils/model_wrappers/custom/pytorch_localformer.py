from qlib.contrib.model.pytorch_localformer import LocalformerModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class LocalformerModel(StrategyMetricMixin, _Base):
    pass
