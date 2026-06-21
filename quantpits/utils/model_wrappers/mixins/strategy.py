# StrategyMetricMixin — overrides fit() to support metric='ir' (portfolio IR)
# as an early-stopping criterion, in addition to the standard 'ic', 'rank_ic',
# 'loss' / 'mse'.  Controlled by the existing YAML `metric` parameter — no new
# config key needed.
#
# Usage:
#   YAML:  metric: ir   (or ic, rank_ic, loss, mse)
#   Wrapper:
#     class LSTM(StrategyMetricMixin, _ICWrapper):
#         pass

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy

import torch

from qlib.utils import get_or_create_path
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger


class StrategyMetricMixin:
    """Extends the ``metric`` YAML parameter with ``metric='ir'``.

    YAML ``metric`` values:
    - ``"ir"``       — portfolio IR via lightweight TopK backtest
    - ``"ic"``       — Pearson IC (standard ICMetricMixin behaviour)
    - ``"rank_ic"``  — Spearman Rank IC
    - ``""``, ``"loss"``, ``"mse"`` — negative validation loss

    ``metric='ir'`` is the only value that requires this mixin; the others
    fall through to the parent ``fit()``.

    TopK is injected from the workspace strategy_config.yaml via
    train_single_model() → model.topk = params['topk'].
    """

    # ------------------------------------------------------------------
    def __init__(self, *args, topk=20, n_drop=3, metric="ir", **kwargs):
        self.topk = topk
        self.n_drop = n_drop
        try:
            super().__init__(*args, metric=metric, **kwargs)
        except TypeError:
            super().__init__(*args, **kwargs)
        self.metric = metric  # ensure final value is correct
        if not hasattr(self, 'early_stop'):
            self.early_stop = 20

    # ------------------------------------------------------------------
    # metric_fn — handles ir / ic / rank_ic / loss / mse directly
    # (no dependency on ICMetricMixin).  base qlib model only needs to
    # provide loss_fn, train_epoch, test_epoch.
    # ------------------------------------------------------------------

    def metric_fn(self, pred, label):
        mask = torch.isfinite(pred) & torch.isfinite(label)
        p = pred[mask]
        l = label[mask]
        if self.metric == "ir":
            return self._batch_pearson_ic(p, l)
        elif self.metric == "ic":
            return self._batch_pearson_ic(p, l)
        elif self.metric == "rank_ic":
            return self._batch_rank_ic(p, l)
        elif self.metric in ("", "loss", "mse"):
            mask = torch.isfinite(label)
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError(f"unknown metric `{self.metric}`")

    # ------------------------------------------------------------------
    # loss_fn — support loss='ic'
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # fit() — dispatches on self.metric
    # ------------------------------------------------------------------

    _NEEDS_BACKTEST = {"ir"}          # metrics that need mini-backtest
    _HIGHER_IS_BETTER = {"ir", "ic", "rank_ic"}  # maximisation metrics

    def fit(self, dataset, evals_result=None, save_path=None):
        if self.metric not in self._NEEDS_BACKTEST:
            # 'ic', 'rank_ic', 'loss', 'mse' — delegate to parent fit()
            if evals_result is None:
                evals_result = {}
            return super().fit(dataset, evals_result=evals_result,
                              save_path=save_path)

        # --- metric='ir': custom fit() with mini-backtest ---
        if evals_result is None:
            evals_result = {}

        train_data, valid_data, valid_index = self._prepare_fit_data(dataset)

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_ic = 0.0
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["valid_ic"] = []
        evals_result["valid_rank_ic"] = []

        logger = getattr(self, "logger", get_module_logger("StrategyMetric"))

        logger.info("training with StrategyMetric (metric=%s, topk=%d)...",
                    self.metric, self.topk)
        self.fitted = True
        best_param = copy.deepcopy(self._get_inner_state_dict())

        for step in range(self.n_epochs):
            logger.info("Epoch%d:", step)
            logger.info("training...")
            self._train_one_epoch(train_data)

            logger.info("evaluating...")
            val_preds, val_labels, val_loss = self._collect_predictions(
                valid_data, valid_index,
            )

            val_ic = self._compute_ic(val_preds, val_labels, valid_index)
            val_rank_ic = self._compute_rank_ic(val_preds, val_labels, valid_index)
            val_ir = self._run_mini_backtest(val_preds, val_labels,
                                             valid_index, self.topk)

            train_loss, train_score = self._eval_one_epoch(train_data)

            logger.info(
                "train %.6f, val_loss %.6f, val_ic %.6f, val_rank_ic %.6f, val_ir %.6f",
                train_score, val_loss, val_ic, val_rank_ic, val_ir,
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
                best_param = copy.deepcopy(self._get_inner_state_dict())
                logger.info("New best IR: %.6f (IC=%.6f)", best_score, best_ic)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    logger.info("early stop")
                    break

        logger.info("best IR: %.6f (IC=%.6f) @ %d", best_score, best_ic, best_epoch)
        self._load_inner_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # fit() hooks — override in DataLoader-based wrappers
    # ------------------------------------------------------------------

    def _prepare_fit_data(self, dataset):
        """Default (numpy): return (x_train,y_train), (x_valid,y_valid), valid_index."""
        df_train, df_valid, _ = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset.")
        return (df_train["feature"], df_train["label"]), \
               (df_valid["feature"], df_valid["label"]), \
               df_valid.index

    def _train_one_epoch(self, train_data):
        """Default (numpy): call train_epoch(x, y)."""
        x, y = train_data
        self.train_epoch(x, y)

    def _eval_one_epoch(self, data):
        """Default (numpy): call test_epoch(x, y), return (loss, score)."""
        x, y = data
        return self.test_epoch(x, y)

    def _collect_predictions(self, valid_data, valid_index):
        """Default (numpy): batched forward pass over x_valid, y_valid."""
        x_valid, y_valid = valid_data
        return self._forward_all(x_valid, y_valid.values, valid_index)

    def _forward_all_dataloader(self, data_loader):
        """Collect predictions from a DataLoader (for TS wrappers).

        Handles both weighted (data, weight) and unweighted batch formats.
        Returns (preds_array, labels_array, mean_loss).
        """
        inner = self._get_inner_module()
        inner.eval()
        all_preds, all_labels, all_losses = [], [], []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    data, weight = batch
                else:
                    data = batch
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)
                pred = inner(feature.float())
                if pred.dim() > 1 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                try:
                    loss = self.loss_fn(pred, label)
                except TypeError:
                    loss = self.loss_fn(pred, label, None)
                all_losses.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_labels.append(label.cpu().numpy())

        preds = np.concatenate(all_preds).ravel() if all_preds else np.array([])
        labels = np.concatenate(all_labels).ravel() if all_labels else np.array([])
        mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
        return preds, labels, mean_loss

    # ------------------------------------------------------------------
    # helpers — inner nn.Module discovery via dir() scan
    # ------------------------------------------------------------------

    KNOWN_LOSS_ATTRS = {"ic_criterion", "mse_criterion", "loss_fn", "criterion"}

    @classmethod
    def _find_inner_modules(cls, instance):
        candidates = []
        for name in dir(instance):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(instance, name)
            except Exception:
                continue
            if isinstance(obj, torch.nn.Module) and name not in cls.KNOWN_LOSS_ATTRS:
                n_params = sum(p.numel() for p in obj.parameters())
                candidates.append((n_params, name, obj))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def _get_inner_module(self):
        candidates = self._find_inner_modules(self)
        if not candidates:
            raise AttributeError(
                f"{type(self).__name__}: no torch.nn.Module found. "
                f"Override _get_inner_module()."
            )
        return candidates[0][2]

    def _get_inner_state_dict(self):
        return self._get_inner_module().state_dict()

    def _load_inner_state_dict(self, state_dict):
        self._get_inner_module().load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    def _forward_all(self, x_valid, y_valid, valid_index):
        if hasattr(x_valid, "values"):
            x_values = x_valid.values
        else:
            x_values = np.asarray(x_valid)
        if hasattr(y_valid, "values"):
            y_values = y_valid.values
        else:
            y_values = np.asarray(y_valid)

        n_samples = x_values.shape[0]
        batch_size = getattr(self, "batch_size", 2000)
        device = getattr(self, "device", torch.device("cpu"))
        inner = self._get_inner_module()
        inner.eval()

        all_preds = []
        all_losses = []

        with torch.no_grad():
            for begin in range(0, n_samples, batch_size):
                end = min(begin + batch_size, n_samples)
                if end - begin < 2:
                    break
                x_batch = torch.from_numpy(x_values[begin:end]).float().to(device)
                y_batch = torch.from_numpy(
                    y_values[begin:end].reshape(-1)
                ).float().to(device)

                pred = inner(x_batch)
                if pred.dim() > 1 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)

                try:
                    loss = self.loss_fn(pred, y_batch)
                except TypeError:
                    loss = self.loss_fn(pred, y_batch, None)
                all_losses.append(loss.item())

                all_preds.append(pred.cpu().numpy())

        preds = np.concatenate(all_preds).ravel() if all_preds else np.array([])
        labels = np.asarray(y_values[:len(preds)]).ravel()
        mean_loss = float(np.mean(all_losses)) if all_losses else 0.0

        return preds, labels, mean_loss

    # ------------------------------------------------------------------
    # IC computation (per-day groupby, pandas-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_pearson_ic(pred, label):
        """Pearson IC for a single batch (tensor in, tensor out)."""
        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device)
        p = pred - pred.mean()
        l = label - label.mean()
        cov = torch.sum(p * l)
        p_std = torch.sqrt(torch.sum(p ** 2) + 1e-8)
        l_std = torch.sqrt(torch.sum(l ** 2) + 1e-8)
        return cov / (p_std * l_std + 1e-8)

    @staticmethod
    def _batch_rank_ic(pred, label):
        """Spearman Rank IC for a single batch (via argsort-argsort)."""
        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device)
        def _rank(x):
            order = x.argsort()
            ranks = torch.empty_like(x)
            ranks[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
            return ranks
        return StrategyMetricMixin._batch_pearson_ic(_rank(pred), _rank(label))

    @staticmethod
    def _compute_ic(preds, labels, index):
        if len(preds) < 2:
            return 0.0
        df = pd.DataFrame({"pred": preds, "label": labels}, index=index[:len(preds)])
        grp_key = "datetime" if "datetime" in df.index.names else df.index.names[0]
        ic = df.groupby(level=grp_key).apply(
            lambda x: x["pred"].corr(x["label"], method="pearson")
        )
        return float(ic.mean())

    @staticmethod
    def _compute_rank_ic(preds, labels, index):
        if len(preds) < 2:
            return 0.0
        df = pd.DataFrame({"pred": preds, "label": labels}, index=index[:len(preds)])
        grp_key = "datetime" if "datetime" in df.index.names else df.index.names[0]
        ric = df.groupby(level=grp_key).apply(
            lambda x: x["pred"].corr(x["label"], method="spearman")
        )
        return float(ric.mean())

    # ------------------------------------------------------------------
    # mini-backtest
    # ------------------------------------------------------------------

    @staticmethod
    def _run_mini_backtest(preds, labels, index, topk):
        if len(preds) < 2:
            return -np.inf

        df = pd.DataFrame({"pred": preds, "label": labels}, index=index[:len(preds)])
        grp_key = "datetime" if "datetime" in df.index.names else df.index.names[0]

        daily_returns = []
        for _date, group in df.groupby(level=grp_key):
            if len(group) < topk:
                continue
            top = group.nlargest(topk, "pred")
            ret = top["label"].mean()
            daily_returns.append(ret)

        if len(daily_returns) < 2:
            return -np.inf

        daily_returns = pd.Series(daily_returns)
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()

        if std_ret == 0 or np.isnan(std_ret):
            return -np.inf

        ir = (mean_ret / std_ret) * np.sqrt(52)
        return float(ir)
