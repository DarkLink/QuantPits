from quantpits.utils.model_wrappers.pytorch_transformer_ts_ic import TransformerModelIC as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class TransformerModelIC(LossHistoryMixin, _Base):
    pass
