from quantpits.utils.model_wrappers.custom.pytorch_sfm import SFM as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class SFM(LossHistoryMixin, _Base):
    pass
