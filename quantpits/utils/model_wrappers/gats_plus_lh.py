from quantpits.utils.model_wrappers.pytorch_gats_plus import GATsPlus as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class GATsPlus(LossHistoryMixin, _Base):
    pass
