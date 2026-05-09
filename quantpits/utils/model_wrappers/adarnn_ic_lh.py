from quantpits.utils.model_wrappers.pytorch_adarnn_ic import ADARNN as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class ADARNN(LossHistoryMixin, _Base):
    pass
