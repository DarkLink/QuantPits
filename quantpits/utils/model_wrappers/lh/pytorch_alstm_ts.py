from quantpits.utils.model_wrappers.custom.pytorch_alstm_ts import ALSTM as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class ALSTM(LossHistoryMixin, _Base):
    pass
