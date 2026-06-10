from quantpits.utils.model_wrappers.custom.pytorch_igmtf import IGMTF as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class IGMTF(LossHistoryMixin, _Base):
    pass
