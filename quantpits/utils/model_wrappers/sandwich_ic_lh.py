from quantpits.utils.model_wrappers.pytorch_sandwich_ic import Sandwich as _Base
from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin

class Sandwich(LossHistoryMixin, _Base):
    pass
