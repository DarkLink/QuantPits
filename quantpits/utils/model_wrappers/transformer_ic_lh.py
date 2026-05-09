from quantpits.utils.model_wrappers.pytorch_transformer_ic import TransformerModel as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class TransformerModel(LossHistoryMixin, _Base):
    pass
