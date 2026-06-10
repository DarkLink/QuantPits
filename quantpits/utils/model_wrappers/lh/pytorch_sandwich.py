from quantpits.utils.model_wrappers.custom.pytorch_sandwich import Sandwich as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin

class Sandwich(LossHistoryMixin, _Base):
    pass
