# TRAModel wrapper — compatible with StrategyMetricMixin.
#
# Root cause: TRAModel.__init__() has no **kwargs catchall, so unknown
# parameters (loss, metric, topk, n_drop) cause TypeError.
#
# Fix:
#   1. __init__ absorbs 'loss' before it reaches TRAModel.
#      'metric', 'topk', 'n_drop' are absorbed by StrategyMetricMixin.
#   2. fit() always delegates to TRAModel.fit() regardless of metric setting.
#      TRA's training loop (MTSDatasetH, router/oracle transport, pretrain
#      phase) is too specialised to adapt to StrategyMetricMixin hooks.
#      Native IC-based early stopping applies.

from qlib.contrib.model.pytorch_tra import TRAModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class TRAModelIC(StrategyMetricMixin, _Base):
    """TRAModel with StrategyMetricMixin __init__ compatibility.

    ``metric='ir'`` is silently downgraded to TRA's native IC-based early
    stopping (TRA already uses IC internally).  ``loss`` is accepted and
    stored but has no effect (TRA uses its own transport loss internally).
    """

    def __init__(self, *args, loss="mse", **kwargs):
        # TRAModel.__init__ has no **kwargs; consume 'loss' here so it
        # never reaches TRAModel.  StrategyMetricMixin.__init__ handles
        # 'metric', 'topk', 'n_drop'.
        self.loss = loss
        super().__init__(*args, **kwargs)

    def fit(self, dataset, evals_result=None, save_path=None):
        """Always use TRAModel.fit(); StrategyMetricMixin hooks are not compatible.

        TRA's fit() requires MTSDatasetH and manages its own multi-stage
        training (pretrain → router transport) with IC-based early stopping.
        """
        if evals_result is None:
            evals_result = {}
        # TRAModel.fit() has no save_path param — it saves to self.logdir.
        _Base.fit(self, dataset, evals_result=evals_result)
