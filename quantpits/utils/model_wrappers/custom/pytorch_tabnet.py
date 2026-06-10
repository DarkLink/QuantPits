from qlib.contrib.model.pytorch_tabnet import TabnetModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin



class TabnetModel(StrategyMetricMixin, _Base):
    pass
