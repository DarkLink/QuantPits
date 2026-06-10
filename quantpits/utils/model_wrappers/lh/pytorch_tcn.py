from quantpits.utils.model_wrappers.custom.pytorch_tcn import TCN as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class TCN(LossHistoryMixin, _Base):
    pass
