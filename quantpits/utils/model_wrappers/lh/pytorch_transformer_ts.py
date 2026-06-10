from quantpits.utils.model_wrappers.custom.pytorch_transformer_ts import TransformerModelIC as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class TransformerModelIC(LossHistoryMixin, _Base):
    pass
