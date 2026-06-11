# IGMTF wrapper — adds IC / IR early-stopping with hidden-state pass-through.
#
# Root cause: IGMTF.train_epoch(x, y, train_hidden, train_hidden_day) and
# test_epoch(x, y, train_hidden, train_hidden_day) require two extra positional
# arguments that StrategyMetricMixin's default hooks don't supply.
#
# Fix (metric='ir'):
#   Override three hooks:
#     _train_one_epoch  → compute get_train_hidden(), cache, then train_epoch
#     _eval_one_epoch   → reuse cached hidden for test_epoch
#     _collect_predictions → reuse cached hidden for day-by-day forward pass
#   Also mirror IGMTF.fit()'s pretrained-model loading before the IR loop.
#
# Fix (other metrics):
#   Delegate to IGMTF.fit() directly; it already handles ic / loss natively
#   and will call self.metric_fn / self.loss_fn which resolve through MRO to
#   StrategyMetricMixin (so rank_ic / mse also work).

import numpy as np
import torch

from qlib.contrib.model.pytorch_igmtf import IGMTF as _Base

from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class IGMTF(StrategyMetricMixin, _Base):
    """IGMTF with IC / IR early-stopping.

    ``metric='ir'``: StrategyMetricMixin's IR loop is used with overridden
    hooks that pass the required ``train_hidden`` / ``train_hidden_day``
    arguments.  ``get_train_hidden()`` is called once per epoch in
    ``_train_one_epoch`` and cached for the remaining hooks.

    Other metrics: ``IGMTF.fit()`` is used directly.
    """

    # ------------------------------------------------------------------
    # StrategyMetricMixin hook overrides
    # ------------------------------------------------------------------

    def _train_one_epoch(self, train_data):
        x, y = train_data
        # Recompute hidden at the start of each epoch (IGMTF requires this).
        train_hidden, train_hidden_day = self.get_train_hidden(x)
        # Cache for _eval_one_epoch and _collect_predictions in the same epoch.
        self._epoch_hidden = (train_hidden, train_hidden_day)
        self.train_epoch(x, y, train_hidden, train_hidden_day)

    def _eval_one_epoch(self, data):
        x, y = data
        h, hd = self._epoch_hidden  # always set before _eval_one_epoch is called
        # IGMTF.test_epoch returns (mean_loss, mean_score) — exactly what the mixin expects.
        return self.test_epoch(x, y, h, hd)

    def _collect_predictions(self, valid_data, valid_index):
        x_valid, y_valid = valid_data
        h, hd = self._epoch_hidden

        x_values = x_valid.values
        y_values = np.squeeze(y_valid.values)
        daily_batches = self.get_daily_inter(x_valid, shuffle=False)

        all_preds, all_labels, all_losses = [], [], []
        self.igmtf_model.eval()
        with torch.no_grad():
            for batch in daily_batches:
                feature = torch.from_numpy(x_values[batch]).float().to(self.device)
                label = torch.from_numpy(y_values[batch]).float().to(self.device)
                pred = self.igmtf_model(feature, train_hidden=h, train_hidden_day=hd)
                loss = self.loss_fn(pred, label)
                all_losses.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_labels.append(label.cpu().numpy())

        preds = np.concatenate(all_preds).ravel() if all_preds else np.array([])
        labels = np.concatenate(all_labels).ravel() if all_labels else np.array([])
        mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
        return preds, labels, mean_loss

    # ------------------------------------------------------------------
    # fit() dispatch
    # ------------------------------------------------------------------

    def fit(self, dataset, evals_result=None, save_path=None):
        if getattr(self, "metric", "") == "ir":
            # Load pretrained base model (mirrors IGMTF.fit()'s first step),
            # then hand off to StrategyMetricMixin's IR loop.
            self._load_pretrained()
            return StrategyMetricMixin.fit(
                self, dataset,
                evals_result=evals_result,
                save_path=save_path,
            )
        # For ic / loss / mse / rank_ic: IGMTF.fit() handles them natively.
        # self.metric_fn / self.loss_fn resolve through MRO to StrategyMetricMixin,
        # so rank_ic and ic-loss work even through the base fit() path.
        return _Base.fit(
            self, dataset,
            evals_result=evals_result if evals_result is not None else {},
            save_path=save_path,
        )

    def _load_pretrained(self):
        """Mirror IGMTF.fit()'s pretrained-model loading step."""
        from qlib.contrib.model.pytorch_lstm import LSTMModel
        from qlib.contrib.model.pytorch_gru import GRUModel

        if self.base_model == "LSTM":
            pretrained_model = LSTMModel()
        elif self.base_model == "GRU":
            pretrained_model = GRUModel()
        else:
            return  # unknown base_model — skip pretrain loading

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )

        model_dict = self.igmtf_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items()
            if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.igmtf_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")
