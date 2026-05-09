from quantpits.utils.model_wrappers.pytorch_krnn_ic import KRNN as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class KRNN(LossHistoryMixin, _Base):
    pass
