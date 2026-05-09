from quantpits.utils.model_wrappers.pytorch_gru_ic import GRU as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class GRU(LossHistoryMixin, _Base):
    pass
