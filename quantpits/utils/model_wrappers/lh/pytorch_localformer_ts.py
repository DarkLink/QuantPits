from quantpits.utils.model_wrappers.custom.pytorch_localformer_ts import LocalformerModelIC as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class LocalformerModelIC(LossHistoryMixin, _Base):
    pass
