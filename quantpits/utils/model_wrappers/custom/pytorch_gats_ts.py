import types

import numpy as np
import torch
from torch.utils.data import DataLoader

from qlib.contrib.model.pytorch_gats_ts import GATs as _Base, DailyBatchSampler
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


def _stable_cal_attention(self, x, y):
    """Stable softmax version of GATModel.cal_attention.

    The upstream qlib GATModel applies softmax directly to unbounded
    LeakyReLU output.  When attention scores grow large, exp() overflows
    to inf and softmax produces inf/inf = NaN, which then corrupts all
    parameters via backward().
    """
    x = self.transformation(x)
    y = self.transformation(y)

    sample_num = x.shape[0]
    dim = x.shape[1]
    e_x = x.expand(sample_num, sample_num, dim)
    e_y = torch.transpose(e_x, 0, 1)
    attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
    self.a_t = torch.t(self.a)
    attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
    attention_out = self.leaky_relu(attention_out)
    # subtract max before softmax to prevent exp() overflow
    attention_out = attention_out - attention_out.max(dim=-1, keepdim=True)[0]
    att_weight = self.softmax(attention_out)
    return att_weight


class GATs(StrategyMetricMixin, _Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_cal_attention()
        # Add weight decay to prevent unbounded parameter growth under
        # scale-invariant IC loss, which otherwise allows outputs to
        # drift toward overflow -> inf -> NaN.
        for pg in self.train_optimizer.param_groups:
            pg.setdefault("weight_decay", 1e-5)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._patch_cal_attention()

    def _patch_cal_attention(self):
        self.GAT_model.cal_attention = types.MethodType(
            _stable_cal_attention, self.GAT_model
        )

    def _train_one_epoch(self, train_data):
        self.train_epoch(train_data)
        # Detect NaN in parameters after training epoch and raise early.
        for name, param in self.GAT_model.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"NaN/Inf detected in parameter '{name}' after training epoch"
                )

    def _eval_one_epoch(self, data):
        # Filter NaN scores: a single NaN batch poisons np.mean().
        loss, score = self.test_epoch(data)
        if np.isnan(score):
            score = 0.0
        return loss, score

    def _prepare_fit_data(self, dataset):
        dl_train = dataset.prepare("train", col_set=["feature", "label"])
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"])
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset.")
        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)
        train_loader = DataLoader(
            dl_train, sampler=sampler_train,
            num_workers=self.n_jobs, drop_last=False,
        )
        valid_loader = DataLoader(
            dl_valid, sampler=sampler_valid,
            num_workers=self.n_jobs, drop_last=False,
        )
        return train_loader, valid_loader, dl_valid.get_index()

    def _collect_predictions(self, valid_data, valid_index):
        inner = self._get_inner_module()
        inner.eval()
        all_preds, all_labels, all_losses = [], [], []
        with torch.no_grad():
            for data in valid_data:
                data = data.squeeze()
                feature = data[:, :, 0:-1].to(self.device)  # [batch, seq_len, d_feat]
                label = data[:, -1, -1].to(self.device)     # last timestep, last col
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
