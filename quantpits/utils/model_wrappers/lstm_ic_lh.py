from quantpits.utils.model_wrappers.pytorch_lstm_ic import LSTM as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class LSTM(LossHistoryMixin, _Base):
    pass
