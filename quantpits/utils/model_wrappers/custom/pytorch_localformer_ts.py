from torch.utils.data import DataLoader
from qlib.data.dataset.handler import DataHandlerLP

from qlib.contrib.model.pytorch_localformer_ts import LocalformerModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class LocalformerModelIC(StrategyMetricMixin, _Base):
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
        return self._forward_all_dataloader(valid_data)
