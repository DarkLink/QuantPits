from quantpits.utils.model_wrappers.custom.pytorch_adarnn import ADARNN as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class ADARNN(LossHistoryMixin, _Base):
    pass
