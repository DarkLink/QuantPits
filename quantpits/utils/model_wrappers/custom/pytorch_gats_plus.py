# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from scipy.stats import spearmanr
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_gru import GRUModel


class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        index = self.data_source.get_index()
        dates = index.get_level_values("datetime")
        
        # 建立 日期 -> [所有该日期的行号] 的映射
        # 这样无论物理内存怎么乱，我们都能精准抓取属于同一天的所有行
        self.daily_indices = {}
        for row_idx, date in enumerate(dates):
            if date not in self.daily_indices:
                self.daily_indices[date] = []
            self.daily_indices[date].append(row_idx)
            
        self.sorted_dates = sorted(self.daily_indices.keys())

    def __iter__(self):
        for date in self.sorted_dates:
            yield np.array(self.daily_indices[date])

    def __len__(self):
        return len(self.sorted_dates)


class GATsPlus(Model):
    """GATsPlus Model
    
    Enhanced GATs model supporting IC/Rank IC metrics and Correlation Loss.
    """

    def __init__(
        self,
        d_feat=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="ic",        # 默认改为 ic
        early_stop=20,
        loss="mse",         # 支持 mse 或 corr
        base_model="GRU",
        model_path=None,
        optimizer="adam",
        GPU=0,
        n_jobs=10,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GATsPlus")
        self.logger.info("GATsPlus pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "GATsPlus parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nbase_model : {}"
            "\nmodel_path : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                early_stop,
                optimizer.lower(),
                loss,
                base_model,
                model_path,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GAT_model = GATModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.logger.info("model:\n{:}".format(self.GAT_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GAT_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GAT_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GAT_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GAT_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)
    
    def correlation_loss(self, pred, label):
        """
        Correlation Loss: 1 - Pearson Correlation
        Optimizes IC directly.
        """
        # Ensure sufficient batch size
        if pred.shape[0] < 2:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        vx = pred - torch.mean(pred)
        vy = label - torch.mean(label)
        
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return 1 - cost

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        p = pred[mask]
        l = label[mask]

        if self.loss == "mse":
            return self.mse(p, l)
        if self.loss == "corr":
            return self.correlation_loss(p, l)
        if self.loss == "ic":
            if not hasattr(self, "_ic_loss_module"):
                from quantpits.utils.model_wrappers.mixins.ic import ICLoss
                self._ic_loss_module = ICLoss()
            return self._ic_loss_module(p, l)

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        """
        Calculate metrics per day (batch).
        Note: inputs are torch tensors on device.
        """
        mask = torch.isfinite(label)
        p = pred[mask]
        l = label[mask]

        if self.metric in ("", "loss"):
            # Negative loss so that larger is better (consistent with IC)
            return -self.loss_fn(p, l)

        # Pearson IC
        elif self.metric in ("ic", "ir"):
            # "ir" early-stopping is handled at epoch level via mini-backtest;
            # use Pearson IC here as the per-batch score for train logging.
            if len(p) < 2: return 0.0
            vx = p - torch.mean(p)
            vy = l - torch.mean(l)
            ic = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
            return ic.item()

        # Rank IC (Spearman)
        elif self.metric == "ric":
            if len(p) < 2: return 0.0
            p_np = p.detach().cpu().numpy()
            l_np = l.detach().cpu().numpy()
            return spearmanr(p_np, l_np)[0]

        raise ValueError("unknown metric `%s`" % self.metric)

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, data_loader):
        self.GAT_model.train()

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.GAT_model(feature.float())
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.GAT_model.eval()

        scores = []
        losses = []

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                # Handle cases where correlation might return nan (e.g. constant input)
                if np.isfinite(score):
                    scores.append(score)

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=None,
        save_path=None,
    ):
        if getattr(self, "metric", "") == "ir":
            return self._fit_ir(dataset, evals_result, save_path)
        if evals_result is None:
            evals_result = {}
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)

        train_loader = DataLoader(dl_train, sampler=sampler_train, num_workers=self.n_jobs, drop_last=True)
        valid_loader = DataLoader(dl_valid, sampler=sampler_valid, num_workers=self.n_jobs, drop_last=True)

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            
            self.logger.info("train loss %.6f, valid loss %.6f" % (train_loss, val_loss))
            self.logger.info("train %s %.6f, valid %s %.6f" % (self.metric, train_score, self.metric, val_score))
            
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GAT_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def _fit_ir(self, dataset, evals_result, save_path):
        import pandas as pd
        from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin as _SM

        if evals_result is None:
            evals_result = {}
        dl_train = dataset.prepare("train", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset.")
        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        valid_index = dl_valid.get_index()

        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)
        train_loader = DataLoader(
            dl_train, sampler=sampler_train,
            num_workers=self.n_jobs, drop_last=True,
        )
        valid_loader = DataLoader(
            dl_valid, sampler=sampler_valid,
            num_workers=self.n_jobs, drop_last=True,
        )

        # pretrained base model
        if self.base_model == "LSTM":
            pt = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size,
                          num_layers=self.num_layers)
        elif self.base_model == "GRU":
            pt = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size,
                         num_layers=self.num_layers)
        else:
            raise ValueError(f"unknown base model `{self.base_model}`")
        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pt.load_state_dict(torch.load(self.model_path, map_location=self.device))
        md = self.GAT_model.state_dict()
        md.update({k: v for k, v in pt.state_dict().items() if k in md})
        self.GAT_model.load_state_dict(md)

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

        self.logger.info("training GATsPlus (metric=ir, topk=%d)...", topk)
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            # collect predictions
            self.GAT_model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data in valid_loader:
                    d = data.squeeze()
                    f = d[:, :, 0:-1].to(self.device)
                    l = d[:, -1, -1].to(self.device)  # last timestep, last col (label)
                    p = self.GAT_model(f.float())
                    if p.dim() > 1 and p.shape[-1] == 1:
                        p = p.squeeze(-1)
                    all_preds.append(p.cpu().numpy())
                    all_labels.append(l.cpu().numpy())
            preds = np.concatenate(all_preds).ravel() if all_preds else np.array([])
            labels = np.concatenate(all_labels).ravel() if all_labels else np.array([])

            val_ic = _SM._compute_ic(preds, labels, valid_index)
            val_rank_ic = _SM._compute_rank_ic(preds, labels, valid_index)
            val_ir = _SM._run_mini_backtest(preds, labels, valid_index, topk)

            train_loss, train_score = self.test_epoch(train_loader)
            self.logger.info("train %.6f, val_ic %.6f, val_rank_ic %.6f, val_ir %.6f",
                            train_score, val_ic, val_rank_ic, val_ir)
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_ir)
            evals_result["valid_ic"].append(val_ic)
            evals_result["valid_rank_ic"].append(val_rank_ic)
            if val_ir > best_score:
                best_score = val_ir
                best_ic = val_ic
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
                self.logger.info("New best IR: %.6f (IC=%.6f)", best_score, best_ic)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best IR: %.6f (IC=%.6f) @ %d", best_score, best_ic, best_epoch)
        self.GAT_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        test_loader = DataLoader(dl_test, sampler=sampler_test, num_workers=self.n_jobs)
        self.GAT_model.eval()
        preds = []

        for data in test_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GATModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
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
        attention_out = attention_out - attention_out.max(dim=-1, keepdim=True)[0]
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()