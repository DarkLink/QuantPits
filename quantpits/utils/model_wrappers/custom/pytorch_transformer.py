from qlib.contrib.model.pytorch_transformer import TransformerModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class TransformerModel(StrategyMetricMixin, _Base):
    pass
