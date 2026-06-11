# ADARNN wrapper — adds IR early-stopping.
#
# Root cause: ADARNN has no train_epoch(x, y) method.  Its training uses
# train_AdaRNN(loader_list, epoch, dist, weight) with multi-domain DataLoaders
# and domain-transfer weight matrices.  StrategyMetricMixin's default
# _train_one_epoch hook calls self.train_epoch(x, y) → AttributeError.
#
# Fix:
#   - metric='ir' → custom _fit_ir() that replicates ADARNN's multi-domain
#     loop and adds a mini-backtest IR early-stop after each epoch.
#   - other metrics → delegate to ADARNN.fit() (native ic/mse/icir/etc.).
#
# Note: ADARNN.__init__ already has **_ so 'loss' and extra kwargs are
# silently swallowed; no __init__ override needed.

import copy
import numpy as np
import torch

from qlib.utils import get_or_create_path
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.pytorch_adarnn import ADARNN as _Base, get_stock_loader

from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class ADARNN(StrategyMetricMixin, _Base):
    """ADARNN with IR early-stopping.

    For ``metric='ir'``: runs ADARNN's multi-domain AdaRNN training loop and
    selects the best epoch by portfolio IR (TopK mini-backtest on valid set).

    For other metrics (``ic``, ``mse``, ``icir``, …): delegates to
    ``ADARNN.fit()`` which handles them natively.
    """

    def fit(self, dataset, evals_result=None, save_path=None):
        if getattr(self, "metric", "") == "ir":
            return self._fit_ir(dataset, evals_result, save_path)
        # Native ADARNN.fit() supports ic / icir / ric / ricir / mse / loss.
        return _Base.fit(
            self, dataset,
            evals_result=evals_result if evals_result is not None else {},
            save_path=save_path,
        )

    def _fit_ir(self, dataset, evals_result, save_path):
        """IR-early-stopping fit: replicates ADARNN's multi-domain loop."""
        if evals_result is None:
            evals_result = {}

        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset.")

        # Replicate ADARNN.fit()'s multi-domain DataLoader setup.
        days = df_train.index.get_level_values(level=0).unique()
        train_splits = np.array_split(days, self.n_splits)
        train_splits = [df_train[s[0]: s[-1]] for s in train_splits]
        train_loader_list = [get_stock_loader(df, self.batch_size) for df in train_splits]

        valid_index = df_valid.index
        topk = getattr(self, "topk", 20)
        save_path = get_or_create_path(save_path)

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_ic = 0.0
        weight_mat, dist_mat = None, None
        # Safety: pre-init best_param to current weights.
        best_param = copy.deepcopy(self.model.state_dict())

        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["valid_ic"] = []
        evals_result["valid_rank_ic"] = []

        self.logger.info("training ADARNN (metric=ir, topk=%d)...", topk)
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            weight_mat, dist_mat = self.train_AdaRNN(
                train_loader_list, step, dist_mat, weight_mat
            )
            self.logger.info("evaluating...")

            # Use ADARNN's own infer() for predictions (handles reshape/seq).
            preds_series = self.infer(df_valid["feature"])
            labels = df_valid["label"].squeeze()
            preds = preds_series.values
            labels_arr = labels.values

            val_ic = StrategyMetricMixin._compute_ic(preds, labels_arr, valid_index)
            val_rank_ic = StrategyMetricMixin._compute_rank_ic(preds, labels_arr, valid_index)
            val_ir = StrategyMetricMixin._run_mini_backtest(preds, labels_arr, valid_index, topk)

            # Train metrics via ADARNN's own test_epoch (returns a dict).
            train_metrics = self.test_epoch(df_train)
            train_score = train_metrics.get("ic", 0.0)

            self.logger.info(
                "train_ic %.6f, val_ic %.6f, val_rank_ic %.6f, val_ir %.6f",
                train_score, val_ic, val_rank_ic, val_ir,
            )
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_ir)
            evals_result["valid_ic"].append(val_ic)
            evals_result["valid_rank_ic"].append(val_rank_ic)

            if val_ir > best_score:
                best_score = val_ir
                best_ic = val_ic
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
                self.logger.info("New best IR: %.6f (IC=%.6f)", best_score, best_ic)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best IR: %.6f (IC=%.6f) @ %d", best_score, best_ic, best_epoch)
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()
