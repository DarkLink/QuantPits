from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATsPlus as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class GATsPlus(LossHistoryMixin, _Base):
    pass
