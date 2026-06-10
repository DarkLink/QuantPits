from quantpits.utils.model_wrappers.custom.pytorch_krnn import KRNN as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class KRNN(LossHistoryMixin, _Base):
    pass
