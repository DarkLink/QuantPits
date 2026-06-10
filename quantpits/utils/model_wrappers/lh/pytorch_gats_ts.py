from quantpits.utils.model_wrappers.custom.pytorch_gats_ts import GATs as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class GATs(LossHistoryMixin, _Base):
    pass
