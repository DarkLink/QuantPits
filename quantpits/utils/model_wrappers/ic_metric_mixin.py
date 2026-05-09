# IC-based Metric Mixin for Qlib PyTorch Models
#
# Provides IC and Rank-IC as early-stopping metrics without modifying
# the original model implementations.  Training loss remains unchanged
# (typically MSE); only the validation metric used for best-epoch
# selection is overridden.
#
# Usage:
#   class ALSTM(ICMetricMixin, _BaseALSTM):
#       pass  # MRO gives ICMetricMixin.metric_fn priority

import torch
import torch.nn as nn


class ICLoss(nn.Module):
    """1 - Pearson correlation, usable as both a loss and a metric.

    Numerically stabilised with epsilon guards on std computation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        mask = torch.isfinite(label) & torch.isfinite(pred)
        pred = pred[mask]
        label = label[mask]

        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_centered = pred - pred.mean()
        label_centered = label - label.mean()

        covariance = torch.sum(pred_centered * label_centered)
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2) + 1e-8)
        label_std = torch.sqrt(torch.sum(label_centered ** 2) + 1e-8)

        correlation = covariance / (pred_std * label_std + 1e-8)
        return 1.0 - correlation


class ICMetricMixin:
    """Mixin that adds ``metric='ic'`` and ``metric='rank_ic'`` support.

    When mixed into a Qlib pytorch model class, the MRO ensures this
    ``metric_fn`` is called instead of the base class's version.

    Supported metric values:
    - ``""`` or ``"loss"`` : negative training loss (original behaviour)
    - ``"ic"``             : Pearson IC on validation batch
    - ``"rank_ic"``        : Spearman rank IC on validation batch
    """

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        p = pred[mask]
        l = label[mask]

        if self.metric == "ic":
            return self._batch_pearson_ic(p, l)
        elif self.metric == "rank_ic":
            return self._batch_rank_ic(p, l)
        elif self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        elif self.metric == "mse":
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _batch_pearson_ic(pred, label):
        """Pearson correlation as a score (higher is better)."""
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
        """Spearman rank correlation (computed via Pearson on ranks)."""
        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device)

        def _rank(x):
            """Dense ranking via argsort-argsort."""
            order = x.argsort()
            ranks = torch.empty_like(x)
            ranks[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
            return ranks

        return ICMetricMixin._batch_pearson_ic(_rank(pred), _rank(label))
