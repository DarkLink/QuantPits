from quantpits.utils.model_wrappers.custom.pytorch_transformer import TransformerModel as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class TransformerModel(LossHistoryMixin, _Base):
    pass
