import torch
import numpy as np
from torch.utils.data import DataLoader
from qlib.data.dataset.handler import DataHandlerLP

from qlib.contrib.model.pytorch_tcn_ts import TCN as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class TCNIC(StrategyMetricMixin, _Base):
    def _prepare_fit_data(self, dataset):
        dl_train = dataset.prepare("train", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset.")
        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_jobs, drop_last=True,
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False,
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
                data = torch.transpose(data, 1, 2)
                feature = data[:, 0:-1, :].to(self.device)
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
