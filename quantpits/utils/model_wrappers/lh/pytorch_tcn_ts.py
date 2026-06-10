from quantpits.utils.model_wrappers.custom.pytorch_tcn_ts import TCNIC as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class TCNIC(LossHistoryMixin, _Base):
    pass
