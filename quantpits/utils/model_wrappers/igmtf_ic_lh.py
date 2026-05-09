from quantpits.utils.model_wrappers.pytorch_igmtf_ic import IGMTF as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class IGMTF(LossHistoryMixin, _Base):
    pass
