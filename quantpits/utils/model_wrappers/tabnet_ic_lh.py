from quantpits.utils.model_wrappers.pytorch_tabnet_ic import TabnetModel as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class TabnetModel(LossHistoryMixin, _Base):
    pass
