from quantpits.utils.model_wrappers.pytorch_tra_ic import TRAModelIC as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class TRAModelIC(LossHistoryMixin, _Base):
    pass
