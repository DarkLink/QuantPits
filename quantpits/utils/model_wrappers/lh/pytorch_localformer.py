from quantpits.utils.model_wrappers.custom.pytorch_localformer import LocalformerModel as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class LocalformerModel(LossHistoryMixin, _Base):
    pass
