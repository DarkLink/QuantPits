from quantpits.utils.model_wrappers.pytorch_alstm_ts_ic import ALSTM as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class ALSTM(LossHistoryMixin, _Base):
    pass
