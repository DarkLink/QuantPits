from quantpits.utils.model_wrappers.pytorch_gats_ts_ic import GATs as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class GATs(LossHistoryMixin, _Base):
    pass
