# ADD with IC / IR early-stopping.
# ADD already computes IC/RIC in test_epoch via cal_ic_metrics().
# This wrapper adds IR-based early-stopping (metric='ir').

import numpy as np
import copy
import torch
from qlib.utils import get_or_create_path
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.pytorch_add import ADD as _Base, LSTMModel, GRUModel


class ADD(_Base):
    """ADD with IR early-stopping.  IC/RIC already built-in."""

    def fit(self, dataset, evals_result=None, save_path=None):
        if getattr(self, "metric", "mse") == "ir":
            return self._fit_ir(dataset, evals_result, save_path)
        return super().fit(dataset, evals_result=evals_result,
                          save_path=save_path)

    def _fit_ir(self, dataset, evals_result, save_path):
        from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin as _SM

        if evals_result is None:
            evals_result = {}

        # Replicate ADD's fit() data preparation
        label_train, label_valid = dataset.prepare(
            ["train", "valid"], col_set=["label"], data_key=DataHandlerLP.DK_R,
        )
        self.fit_thresh(label_train)
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        df_train = self.gen_market_label(df_train, label_train)
        df_valid = self.gen_market_label(df_valid, label_valid)

        x_train, y_train, m_train = df_train["feature"], df_train["label"], df_train["market_return"]
        x_valid, y_valid, m_valid = df_valid["feature"], df_valid["label"], df_valid["market_return"]
        valid_index = df_valid.index

        # pretrained base model
        if self.base_model == "LSTM":
            pt = LSTMModel()
        elif self.base_model == "GRU":
            pt = GRUModel()
        else:
            raise ValueError(f"unknown base model `{self.base_model}`")
        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pt.load_state_dict(torch.load(self.model_path, map_location=self.device))
            md = self.ADD_model.enc_excess.state_dict()
            pd_ = {k: v for k, v in pt.rnn.state_dict().items() if k in md}
            md.update(pd_)
            self.ADD_model.enc_excess.load_state_dict(md)
            md = self.ADD_model.enc_market.state_dict()
            pd_ = {k: v for k, v in pt.rnn.state_dict().items() if k in md}
            md.update(pd_)
            self.ADD_model.enc_market.load_state_dict(md)
            self.logger.info("Loading pretrained model Done...")

        # Prepare training data
        y_train_values = np.squeeze(y_train.values)
        m_train_values = np.squeeze(m_train.values.astype(int))

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_ic = 0.0
        topk = getattr(self, 'topk', 20)
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["valid_ic"] = []
        evals_result["valid_rank_ic"] = []

        self.logger.info("training ADD (metric=ir, topk=%d)...", topk)
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train_values, m_train_values)
            self.logger.info("evaluating...")

            # Collect predictions for IR
            x_values = x_valid.values
            y_v = np.squeeze(y_valid.values)
            self.ADD_model.eval()
            all_preds, all_labels = [], []
            daily_batches = self.get_daily_inter(x_valid, shuffle=False)
            with torch.no_grad():
                for batch in daily_batches:
                    x_b = torch.from_numpy(x_values[batch]).float().to(self.device)
                    y_b = torch.from_numpy(y_v[batch]).float().to(self.device)
                    p = self.ADD_model(x_b)["excess"]
                    if p.dim() > 1 and p.shape[-1] == 1:
                        p = p.squeeze(-1)
                    all_preds.append(p.cpu().numpy())
                    all_labels.append(y_b.cpu().numpy())
            preds = np.concatenate(all_preds).ravel() if all_preds else np.array([])
            labels = np.concatenate(all_labels).ravel() if all_labels else np.array([])

            val_ic = _SM._compute_ic(preds, labels, valid_index)
            val_rank_ic = _SM._compute_rank_ic(preds, labels, valid_index)
            val_ir = _SM._run_mini_backtest(preds, labels, valid_index, topk)

            # Also run standard test for logging
            valid_metrics = self.test_epoch(x_valid, y_valid, m_valid)
            self.logger.info("val_ic %.6f, val_rc %.6f, val_ir %.6f, val_mse %.6f",
                            val_ic, val_rank_ic, val_ir, valid_metrics.get("mse", 0))

            evals_result["train"].append(valid_metrics.get("mse", 0))
            evals_result["valid"].append(val_ir)
            evals_result["valid_ic"].append(val_ic)
            evals_result["valid_rank_ic"].append(val_rank_ic)

            if val_ir > best_score:
                best_score = val_ir
                best_ic = val_ic
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.ADD_model.state_dict())
                self.logger.info("New best IR: %.6f (IC=%.6f)", best_score, best_ic)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best IR: %.6f (IC=%.6f) @ %d", best_score, best_ic, best_epoch)
        self.ADD_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()
