from quantpits.utils.model_wrappers.custom.pytorch_tabnet import TabnetModel as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class TabnetModel(LossHistoryMixin, _Base):
    pass
