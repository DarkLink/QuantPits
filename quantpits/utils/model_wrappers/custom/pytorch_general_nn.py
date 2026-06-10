# GeneralPTNN with IC / IR early-stopping.
#
# GeneralPTNN's fit() uses minimize-based early stopping (val_score < best_score,
# best_score = inf).  IC/IR are higher-is-better, so we negate them.
#
# YAML:
#   metric: ir     → IR-based early stop (negated)
#   metric: ic     → IC-based early stop (negated)
#   metric: loss   → loss-based (no negation needed)

import numpy as np
import copy
import torch
from torch.utils.data import DataLoader

from qlib.utils import get_or_create_path
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.model.utils import ConcatDataset

from qlib.contrib.model.pytorch_general_nn import GeneralPTNN as _Base, TSDatasetH
from quantpits.utils.model_wrappers.mixins.ic import ICMetricMixin


class GeneralPTNN(ICMetricMixin, _Base):
    """GeneralPTNN with IC / Rank-IC / IR early-stopping.

    All metrics are negated to work with GeneralPTNN's minimize-based fit().
    """

    def metric_fn(self, pred, label):
        m = getattr(self, "metric", "mse")
        if m == "ir":
            # ICMetricMixin doesn't handle 'ir' — compute IC directly
            p, l = pred.squeeze(), label.squeeze()
            if p.numel() < 2:
                val = torch.tensor(0.0, device=p.device)
            else:
                p_c, l_c = p - p.mean(), l - l.mean()
                val = torch.sum(p_c * l_c) / (
                    torch.sqrt(torch.sum(p_c ** 2) + 1e-8) *
                    torch.sqrt(torch.sum(l_c ** 2) + 1e-8))
        else:
            val = super().metric_fn(pred.squeeze(), label.squeeze())
        return -val

    def loss_fn(self, pred, label, weight=None):
        if getattr(self, "loss", "mse") == "ic":
            if not hasattr(self, "_ic_loss_module"):
                from quantpits.utils.model_wrappers.mixins.ic import ICLoss
                self._ic_loss_module = ICLoss()
            return self._ic_loss_module(pred, label)
        try:
            return super().loss_fn(pred, label)
        except TypeError:
            return super().loss_fn(pred, label, weight)

    def fit(self, dataset, evals_result=None, save_path=None, reweighter=None):
        if getattr(self, "metric", "mse") == "ir":
            return self._fit_ir(dataset, evals_result, save_path, reweighter)
        return super().fit(dataset, evals_result=evals_result,
                          save_path=save_path, reweighter=reweighter)

    def _fit_ir(self, dataset, evals_result, save_path, reweighter):
        from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin as _SM

        if evals_result is None:
            evals_result = {}
        ists = isinstance(dataset, TSDatasetH)
        dl_train = dataset.prepare("train", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset.")
        if ists:
            dl_train.config(fillna_type="ffill+bfill")
            dl_valid.config(fillna_type="ffill+bfill")
        else:
            dl_train, dl_valid = dl_train.values, dl_valid.values
        valid_index = dl_valid.get_index() if ists else None

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_jobs, drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_jobs, drop_last=False,
        )

        save_path = get_or_create_path(save_path)
        # minimize: best_score starts at inf, negated IR is lower-is-better
        best_score = np.inf
        best_epoch = 0
        best_ic = 0.0
        stop_steps = 0
        topk = getattr(self, 'topk', 20)
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["valid_ic"] = []
        evals_result["valid_rank_ic"] = []

        self.logger.info("training GeneralPTNN (metric=ir, topk=%d)...", topk)
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            # collect predictions
            inner = self._get_inner_module_general()
            inner.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data, _weight in valid_loader:
                    if ists:
                        feature = data[:, :, 0:-1].to(self.device)
                        label = data[:, -1, -1].to(self.device)
                    else:
                        feature = data[:, 0:-1].to(self.device)
                        label = data[:, -1].to(self.device)
                    pred = inner(feature.float())
                    if pred.dim() > 1 and pred.shape[-1] == 1:
                        pred = pred.squeeze(-1)
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(label.cpu().numpy())
            preds = np.concatenate(all_preds).ravel() if all_preds else np.array([])
            labels = np.concatenate(all_labels).ravel() if all_labels else np.array([])

            if valid_index is not None:
                val_ic = _SM._compute_ic(preds, labels, valid_index)
                val_rank_ic = _SM._compute_rank_ic(preds, labels, valid_index)
                val_ir = _SM._run_mini_backtest(preds, labels, valid_index, topk)
            else:
                val_ic = val_rank_ic = 0.0
                val_ir = -np.inf

            train_loss, train_score = self.test_epoch(train_loader)
            neg_ir = -val_ir  # negate for minimization
            self.logger.info("train %.6f, val_ic %.6f, val_rc %.6f, val_ir %.6f (neg %.6f)",
                            train_score, val_ic, val_rank_ic, val_ir, neg_ir)
            evals_result["train"].append(train_score)
            evals_result["valid"].append(neg_ir)
            evals_result["valid_ic"].append(val_ic)
            evals_result["valid_rank_ic"].append(val_rank_ic)

            if neg_ir < best_score:
                best_score = neg_ir
                best_ic = val_ic
                stop_steps = 0
                best_epoch = step
                inner_best = self._get_inner_module_general()
                best_param = copy.deepcopy(inner_best.state_dict())
                self.logger.info("New best IR: %.6f (neg=%.6f, IC=%.6f)",
                                val_ir, best_score, best_ic)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best IR: %.6f (IC=%.6f) @ %d", best_score, best_ic, best_epoch)
        inner_best = self._get_inner_module_general()
        inner_best.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def _get_inner_module_general(self):
        for attr in dir(self):
            if attr.startswith("_"):
                continue
            try:
                obj = getattr(self, attr)
            except Exception:
                continue
            if isinstance(obj, torch.nn.Module) and attr not in ("ic_criterion",):
                return obj
        raise AttributeError("Cannot find inner nn.Module in GeneralPTNN")
