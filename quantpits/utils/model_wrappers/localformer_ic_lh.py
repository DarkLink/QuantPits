from quantpits.utils.model_wrappers.pytorch_localformer_ic import LocalformerModel as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class LocalformerModel(LossHistoryMixin, _Base):
    pass
