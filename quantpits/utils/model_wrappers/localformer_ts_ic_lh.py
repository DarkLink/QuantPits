from quantpits.utils.model_wrappers.pytorch_localformer_ts_ic import LocalformerModelIC as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class LocalformerModelIC(LossHistoryMixin, _Base):
    pass
