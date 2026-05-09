from quantpits.utils.model_wrappers.pytorch_sfm_ic import SFM as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class SFM(LossHistoryMixin, _Base):
    pass
