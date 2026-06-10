from quantpits.utils.model_wrappers.custom.pytorch_gru import GRU as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class GRU(LossHistoryMixin, _Base):
    pass
