import numpy as np
import torch
from torch.utils.data import DataLoader

from qlib.contrib.model.pytorch_gats_ts import GATs as _Base, DailyBatchSampler
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class GATs(StrategyMetricMixin, _Base):
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

    def _train_one_epoch(self, train_data):
        self.train_epoch(train_data)

    def _eval_one_epoch(self, data):
        return self.test_epoch(data)

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
