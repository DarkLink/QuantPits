# TCTS with IC/Rank-IC early stopping.
#
# The base TCTS uses pure MSE for early stopping and has a
# lowest_valid_performance retrain loop that can cause infinite retraining
# when MSE stays above the threshold.  This wrapper replaces MSE-based
# early stopping with IC (Pearson correlation) or Rank-IC, following the
# same pattern used by all other IC wrappers in this directory.
#
# YAML usage:
#   module_path: quantpits.utils.model_wrappers.pytorch_tcts_ic
#   class: TCTS
#   kwargs:
#     metric: ic          # or rank_ic
#     d_feat: 6
#     ...

import copy
import random

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from qlib.contrib.model.pytorch_tcts import TCTS as _BaseTCTS
from qlib.contrib.model.pytorch_tcts import GRUModel, MLPModel


class TCTS(_BaseTCTS):
    """TCTS with IC/Rank-IC early stopping.

    Training loss (weighted MSE) is unchanged; only the validation metric
    used for epoch selection and early stopping is switched to IC.
    """

    def __init__(self, metric="ic", **kwargs):
        super().__init__(**kwargs)
        self.metric = metric

    # ---- metric ----------------------------------------------------------

    def metric_fn(self, pred, label):
        """Compute IC or Rank-IC on numpy arrays.  Higher is better."""
        mask = np.isfinite(label)
        p, l = pred[mask], label[mask]
        if len(p) < 2:
            return 0.0
        if self.metric == "ic":
            return float(np.corrcoef(p, l)[0, 1])
        elif self.metric == "rank_ic":
            from scipy.stats import spearmanr
            return float(spearmanr(p, l)[0])
        return 0.0

    # ---- test_epoch ------------------------------------------------------
    # Same as the base but returns (loss, score) instead of a single float.

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.fore_model.eval()

        preds = []
        labels = []

        indices = np.arange(len(x_values))
        target_col = abs(self.target_label)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(
                x_values[indices[i : i + self.batch_size]]
            ).float().to(self.device)
            label = torch.from_numpy(
                y_values[indices[i : i + self.batch_size]]
            ).float().to(self.device)

            pred = self.fore_model(feature)
            preds.append(pred.cpu().detach().numpy())
            labels.append(label[:, target_col].cpu().numpy())

        pred = np.concatenate(preds)
        label = np.concatenate(labels)

        mse = float(np.mean((pred - label) ** 2))
        score = self.metric_fn(pred, label)
        return mse, score

    # ---- training --------------------------------------------------------
    # Same as the base but maximises score (IC) instead of minimising loss,
    # and prints both metrics per epoch.

    def training(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        verbose=True,
        save_path=None,
    ):
        self.fore_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.weight_model = MLPModel(
            d_feat=self.input_dim + 3 * self.output_dim + 1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.output_dim,
        )
        if self._fore_optimizer.lower() == "adam":
            self.fore_optimizer = optim.Adam(self.fore_model.parameters(), lr=self.fore_lr)
        elif self._fore_optimizer.lower() == "gd":
            self.fore_optimizer = optim.SGD(self.fore_model.parameters(), lr=self.fore_lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(self._fore_optimizer))
        if self._weight_optimizer.lower() == "adam":
            self.weight_optimizer = optim.Adam(self.weight_model.parameters(), lr=self.weight_lr)
        elif self._weight_optimizer.lower() == "gd":
            self.weight_optimizer = optim.SGD(self.weight_model.parameters(), lr=self.weight_lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(self._weight_optimizer))

        self.fitted = False
        self.fore_model.to(self.device)
        self.weight_model.to(self.device)

        best_score = -np.inf
        best_epoch = 0
        stop_round = 0

        for epoch in range(self.n_epochs):
            print("Epoch:", epoch)

            print("training...")
            self.train_epoch(x_train, y_train, x_valid, y_valid)

            print("evaluating...")
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            test_loss, test_score = self.test_epoch(x_test, y_test)

            if verbose:
                print(
                    "valid loss=%.6f score=%.6f, test loss=%.6f score=%.6f"
                    % (val_loss, val_score, test_loss, test_score)
                )

            if val_score > best_score:
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                torch.save(
                    copy.deepcopy(self.fore_model.state_dict()),
                    save_path + "_fore_model.bin",
                )
                torch.save(
                    copy.deepcopy(self.weight_model.state_dict()),
                    save_path + "_weight_model.bin",
                )
            else:
                stop_round += 1
                if stop_round >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6f @ %d" % (best_score, best_epoch))
        best_param = torch.load(save_path + "_fore_model.bin", map_location=self.device)
        self.fore_model.load_state_dict(best_param)
        best_param = torch.load(save_path + "_weight_model.bin", map_location=self.device)
        self.weight_model.load_state_dict(best_param)
        self.fitted = True

        if self.use_gpu:
            torch.cuda.empty_cache()

        return best_score

    # ---- fit -------------------------------------------------------------
    # Single-pass: removes the lowest_valid_performance retrain loop.

    def fit(self, dataset, verbose=True, save_path=None):
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.utils import get_or_create_path

        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        x_test, y_test = df_test["feature"], df_test["label"]

        if save_path is None:
            save_path = get_or_create_path(save_path)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.training(
            x_train, y_train, x_valid, y_valid, x_test, y_test,
            verbose=verbose, save_path=save_path,
        )
