# TabNet wrapper — adds IR early-stopping.
#
# Root cause: StrategyMetricMixin.fit() bypasses TabnetModel.fit(), so the
# pretrain phase and FinetuneModel replacement never happen.  When
# StrategyMetricMixin then calls _forward_all (via _collect_predictions), it
# calls inner(x_batch) on the raw TabNet model whose forward() returns a
# (vec, sparse_loss) tuple instead of a tensor → isfinite() TypeError.
#
# Fix (metric='ir'):
#   _fit_ir() mirrors TabnetModel.fit():
#     1. Pretrain phase (if self.pretrain).
#     2. Replace self.tabnet_model with FinetuneModel.
#     3. Reset optimizer to include the new fc layer.
#     4. IR-based training loop using train_epoch / _tabnet_predict.
#
# Fix (other metrics):
#   Delegate to TabnetModel.fit() directly, which handles the pretrain +
#   FinetuneModel replacement correctly.

import copy
import numpy as np
import torch
import torch.optim as optim

from qlib.utils import get_or_create_path
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.pytorch_tabnet import TabnetModel as _Base, FinetuneModel

from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class TabnetModel(StrategyMetricMixin, _Base):
    """TabnetModel with IR early-stopping.

    ``metric='ir'``: ``_fit_ir()`` handles pretrain, FinetuneModel setup, and
    the IR mini-backtest loop.

    Other metrics: ``TabnetModel.fit()`` is used directly (it already manages
    pretrain + FinetuneModel correctly).
    """

    def fit(self, dataset, evals_result=None, save_path=None):
        if getattr(self, "metric", "") == "ir":
            return self._fit_ir(dataset, evals_result, save_path)
        return _Base.fit(
            self, dataset,
            evals_result=evals_result if evals_result is not None else {},
            save_path=save_path,
        )

    def _fit_ir(self, dataset, evals_result, save_path):
        """IR-early-stopping fit: mirrors TabnetModel.fit() + mini-backtest."""
        if evals_result is None:
            evals_result = {}

        # --- Step 1: pretrain (identical to TabnetModel.fit()) ---
        if self.pretrain:
            self.logger.info("Pretrain...")
            self.pretrain_fn(dataset, self.pretrain_file)
            self.logger.info("Load Pretrain model")
            self.tabnet_model.load_state_dict(
                torch.load(self.pretrain_file, map_location=self.device)
            )

        # --- Step 2: FinetuneModel replacement + optimizer reset ---
        self.tabnet_model = FinetuneModel(
            self.out_dim, self.final_out_dim, self.tabnet_model
        ).to(self.device)
        # Reset optimizer so the new fc layer's params are included.
        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.tabnet_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.tabnet_model.parameters(), lr=self.lr)

        # --- Step 3: prepare data ---
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset.")
        df_train.fillna(df_train.mean(), inplace=True)
        df_valid.fillna(df_valid.mean(), inplace=True)

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        valid_index = df_valid.index
        topk = getattr(self, "topk", 20)
        save_path = get_or_create_path(save_path)

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_ic = 0.0
        # Safety: pre-init best_param to current (post-FinetuneModel) weights.
        best_param = copy.deepcopy(self.tabnet_model.state_dict())

        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["valid_ic"] = []
        evals_result["valid_rank_ic"] = []

        self.logger.info("training TabNet (metric=ir, topk=%d)...", topk)
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")

            # Collect validation predictions using the FinetuneModel forward.
            preds = self._tabnet_predict(x_valid)
            labels = np.squeeze(y_valid.values)[:len(preds)]

            val_ic = StrategyMetricMixin._compute_ic(preds, labels, valid_index)
            val_rank_ic = StrategyMetricMixin._compute_rank_ic(preds, labels, valid_index)
            val_ir = StrategyMetricMixin._run_mini_backtest(preds, labels, valid_index, topk)

            train_loss, train_score = self.test_epoch(x_train, y_train)

            self.logger.info(
                "train %.6f, val_ic %.6f, val_rank_ic %.6f, val_ir %.6f",
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
                best_param = copy.deepcopy(self.tabnet_model.state_dict())
                self.logger.info("New best IR: %.6f (IC=%.6f)", best_score, best_ic)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best IR: %.6f (IC=%.6f) @ %d", best_score, best_ic, best_epoch)
        self.tabnet_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def _tabnet_predict(self, x):
        """Batched forward pass over x using FinetuneModel (requires priors).

        Mirrors TabnetModel.predict() but returns a numpy array rather than
        a pd.Series, and handles all samples (no minimum-batch-size skip).
        """
        x_values = torch.from_numpy(x.values)
        x_values[torch.isnan(x_values)] = 0
        self.tabnet_model.eval()
        preds = []
        sample_num = x_values.shape[0]

        for begin in range(0, sample_num, self.batch_size):
            end = min(begin + self.batch_size, sample_num)
            feature = x_values[begin:end].float().to(self.device)
            priors = torch.ones(end - begin, self.d_feat).to(self.device)
            with torch.no_grad():
                pred = self.tabnet_model(feature, priors)
                # FinetuneModel.forward returns fc(vec).squeeze(); guard
                # against 0-dim tensor on last singleton batch.
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                preds.append(pred.detach().cpu().numpy())

        return np.concatenate(preds).ravel() if preds else np.array([])
