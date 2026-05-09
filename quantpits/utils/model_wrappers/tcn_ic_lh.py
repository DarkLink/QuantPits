from quantpits.utils.model_wrappers.pytorch_tcn_ic import TCN as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class TCN(LossHistoryMixin, _Base):
    pass
