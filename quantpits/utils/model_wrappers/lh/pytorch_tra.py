from quantpits.utils.model_wrappers.custom.pytorch_tra import TRAModelIC as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class TRAModelIC(LossHistoryMixin, _Base):
    pass
