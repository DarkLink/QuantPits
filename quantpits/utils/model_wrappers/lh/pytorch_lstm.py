from quantpits.utils.model_wrappers.custom.pytorch_lstm import LSTM as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class LSTM(LossHistoryMixin, _Base):
    pass
