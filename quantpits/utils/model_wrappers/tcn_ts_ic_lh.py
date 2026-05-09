from quantpits.utils.model_wrappers.pytorch_tcn_ts_ic import TCNIC as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class TCNIC(LossHistoryMixin, _Base):
    pass
