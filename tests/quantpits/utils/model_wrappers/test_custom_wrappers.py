"""
Tests for custom model wrappers (TS wrappers, ADARNN, GeneralPTNN).
These tests verify fit hooks and custom fit loops using mocks to avoid loading qlib.
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Skip all tests if torch is not installed
torch = pytest.importorskip("torch", reason="torch not installed")


# ===========================================================================
# Mocks and Stubs
# ===========================================================================

class MockTSDatasetLoader:
    def __init__(self, length=6):
        self.length = length
        self.empty = False
        
    def config(self, *args, **kwargs):
        pass
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return torch.randn(len(idx), 5, 3)
        # returns sequence shape: [seq_len, feat_dim + 1]
        return torch.randn(5, 3)
        
    def get_index(self):
        return pd.Index(pd.date_range("2020-01-01", periods=self.length), name="datetime")


class MockTSDataset:
    def prepare(self, tag, *args, **kwargs):
        return MockTSDatasetLoader()


class MockDataset:
    def prepare(self, tags, *args, **kwargs):
        idx_train = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=2), ["A", "B", "C"]],
            names=["datetime", "instrument"]
        )
        df_train = pd.DataFrame(
            np.ones((6, 2)),
            index=idx_train,
            columns=pd.MultiIndex.from_tuples([("feature", "f1"), ("feature", "f2")])
        )
        df_train[("label", "l1")] = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]

        idx_valid = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-03", periods=2), ["A", "B", "C"]],
            names=["datetime", "instrument"]
        )
        df_valid = pd.DataFrame(
            np.ones((6, 2)),
            index=idx_valid,
            columns=pd.MultiIndex.from_tuples([("feature", "f1"), ("feature", "f2")])
        )
        df_valid[("label", "l1")] = [1.0, 2.0, 3.0, 4.0, 2.0, 1.0]

        if isinstance(tags, list):
            return df_train, df_valid
        elif tags == "train":
            return df_train
        else:
            return df_valid


# ===========================================================================
# TS Model Wrapper Hook Tests
# ===========================================================================

@pytest.mark.parametrize("module_name, class_name", [
    ("quantpits.utils.model_wrappers.custom.pytorch_alstm_ts", "ALSTM"),
    ("quantpits.utils.model_wrappers.custom.pytorch_localformer_ts", "LocalformerModelIC"),
    ("quantpits.utils.model_wrappers.custom.pytorch_tcn_ts", "TCNIC"),
    ("quantpits.utils.model_wrappers.custom.pytorch_transformer_ts", "TransformerModelIC"),
    ("quantpits.utils.model_wrappers.custom.pytorch_gats_ts", "GATs"),
])
def test_ts_wrapper_hooks(module_name, class_name):
    try:
        mod = __import__(module_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
    except ImportError as exc:
        pytest.skip(f"qlib dependency missing for {module_name}: {exc}")

    # Instantiate via __new__ to avoid calling super().__init__ which requires Qlib
    model = cls.__new__(cls)
    model.batch_size = 2
    model.n_jobs = 1
    model.device = torch.device("cpu")

    # Mock qlib sampler for GATs by patching in the target wrapper module
    class MockSampler:
        def __len__(self):
            return 3
        def __iter__(self):
            return iter([[0, 1], [2, 3], [4, 5]])

    sampler_patch = patch(f"{module_name}.DailyBatchSampler", return_value=MockSampler()) if "pytorch_gats_ts" in module_name else patch("builtins.print")
    with sampler_patch:
        # 1. Test _prepare_fit_data
        dataset = MockTSDataset()
        train_loader, valid_loader, valid_index = model._prepare_fit_data(dataset)
        assert train_loader is not None
        assert valid_loader is not None
        assert len(valid_index) == 6

    # 2. Test _train_one_epoch
    model.train_epoch = MagicMock()
    model._train_one_epoch(train_loader)
    model.train_epoch.assert_called_once_with(train_loader)

    # 3. Test _eval_one_epoch
    model.test_epoch = MagicMock(return_value=(0.1, 0.5))
    loss_val, score_val = model._eval_one_epoch(valid_loader)
    assert loss_val == 0.1
    assert score_val == 0.5
    model.test_epoch.assert_called_once_with(valid_loader)

    # 4. Test _collect_predictions
    mock_inner = MagicMock(spec=torch.nn.Module)
    # Always return [batch, 1] for TS predictions
    mock_inner.return_value = torch.ones(2, 1)
    model._get_inner_module = MagicMock(return_value=mock_inner)
    model.loss_fn = MagicMock(return_value=torch.tensor(0.1))

    preds, labels, mean_loss = model._collect_predictions(valid_loader, valid_index)
    assert len(preds) > 0
    assert len(labels) > 0
    assert isinstance(mean_loss, float)


# ===========================================================================
# ADARNN Wrapper Tests
# ===========================================================================

class TestADARNNWrapper:
    def test_adarnn_fit_ir(self, tmp_path):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_adarnn import ADARNN
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for ADARNN: {exc}")

        model = ADARNN.__new__(ADARNN)
        model.metric = "ir"
        model.n_splits = 2
        model.batch_size = 2
        model.topk = 2
        model.n_epochs = 2
        model.early_stop = 2
        model.device = torch.device("cpu")
        model.logger = MagicMock()

        # Mock model state_dict and train/infer/test methods
        model.model = MagicMock()
        model.model.state_dict.return_value = {"weight": torch.ones(1)}
        model.train_AdaRNN = MagicMock(return_value=(None, None))
        model.infer = MagicMock(return_value=pd.Series([1.0, 2.0, 3.0, 4.0, 2.0, 1.0]))
        model.test_epoch = MagicMock(return_value={"ic": 0.5})

        dataset = MockDataset()
        save_path = tmp_path / "adarnn.pth"

        with patch("quantpits.utils.model_wrappers.custom.pytorch_adarnn.get_stock_loader", return_value=[torch.ones(2, 3, 3)]):
            evals = {}
            model.fit(dataset, evals_result=evals, save_path=save_path)
            assert model.train_AdaRNN.call_count == 2
            assert "train" in evals
            assert len(evals["train"]) == 2

    def test_adarnn_fit_non_ir(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_adarnn import ADARNN
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for ADARNN: {exc}")

        model = ADARNN.__new__(ADARNN)
        model.metric = "ic"

        with patch("quantpits.utils.model_wrappers.custom.pytorch_adarnn._Base.fit", return_value="native_fit") as mock_fit:
            res = model.fit(None)
            assert res == "native_fit"
            mock_fit.assert_called_once()


# ===========================================================================
# GeneralPTNN Wrapper Tests
# ===========================================================================

class TestGeneralPTNNWrapper:
    def test_general_ptnn_fit_ir(self, tmp_path):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_general_nn import GeneralPTNN
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for GeneralPTNN: {exc}")

        model = GeneralPTNN.__new__(GeneralPTNN)
        model.metric = "ir"
        model.n_epochs = 2
        model.batch_size = 2
        model.n_jobs = 1
        model.device = torch.device("cpu")
        model.early_stop = 2
        model.logger = MagicMock()

        # Mock _get_inner_module_general and state_dict
        mock_inner = MagicMock(spec=torch.nn.Module)
        mock_inner.state_dict.return_value = {"param": torch.ones(1)}
        mock_inner.side_effect = lambda x: torch.ones(x.shape[0], 1)
        model._get_inner_module_general = MagicMock(return_value=mock_inner)

        model.train_epoch = MagicMock()
        model.test_epoch = MagicMock(return_value=(0.1, 0.5))

        dataset = MockDataset()
        save_path = tmp_path / "general.pth"

        evals = {}
        model.fit(dataset, evals_result=evals, save_path=save_path)
        assert len(evals["valid"]) == 2
        assert "train" in evals

    def test_general_ptnn_metric_and_loss(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_general_nn import GeneralPTNN
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for GeneralPTNN: {exc}")

        model = GeneralPTNN.__new__(GeneralPTNN)
        model.metric = "ir"
        model.loss = "ic"

        # Test metric_fn for ir
        pred = torch.tensor([1.0, 2.0, 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        score = model.metric_fn(pred, label)
        # Negated Pearson correlation of perfectly correlated vectors is -1.0
        assert abs(score - (-1.0)) < 1e-4

        # Test loss_fn for ic
        loss = model.loss_fn(pred, label)
        assert loss is not None


# ===========================================================================
# Multi-Instrument Mock Dataset for TS models to enable proper daily IC calc
# ===========================================================================

class MockMultiInstrumentTSDatasetLoader:
    def __init__(self, length=6, n_instruments=3):
        self.length = length
        self.n_instruments = n_instruments
        self.empty = False
        
    def config(self, *args, **kwargs):
        pass
        
    def __len__(self):
        return self.length * self.n_instruments
        
    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return torch.randn(len(idx), 5, 3)
        return torch.randn(5, 3)
        
    def get_index(self):
        return pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=self.length), [f"I{i}" for i in range(self.n_instruments)]],
            names=["datetime", "instrument"]
        )


class MockMultiInstrumentTSDataset:
    def prepare(self, tag, *args, **kwargs):
        return MockMultiInstrumentTSDatasetLoader()


# ===========================================================================
# LSTM IC Loss Wrapper Tests
# ===========================================================================

class TestLSTMICModel:
    def test_ic_loss(self):
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss import ICLoss
        loss_fn = ICLoss()
        
        # Less than 2 elements
        pred1 = torch.tensor([1.0], requires_grad=True)
        label1 = torch.tensor([2.0])
        val1 = loss_fn(pred1, label1)
        assert val1.item() == 0.0
        
        # Valid elements
        pred2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        label2 = torch.tensor([1.0, 2.0, 3.0])
        val2 = loss_fn(pred2, label2)
        assert abs(val2.item() - 0.0) < 1e-4
        
        # With nan/inf masks
        pred3 = torch.tensor([1.0, 2.0, float('nan'), 3.0], requires_grad=True)
        label3 = torch.tensor([1.0, 2.0, 3.0, float('inf')])
        val3 = loss_fn(pred3, label3)
        assert abs(val3.item() - 0.0) < 1e-4

    def test_lstmic_model_init_and_properties(self):
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss import LSTMICModel
        
        model = LSTMICModel(d_feat=2, hidden_size=4, num_layers=1, n_epochs=2, batch_size=2, GPU=-1)
        assert model.d_feat == 2
        assert model.hidden_size == 4
        assert not model.fitted
        assert not model.use_gpu
        
        model_sgd = LSTMICModel(d_feat=2, hidden_size=4, num_layers=1, optimizer="gd", seed=42, GPU=-1)
        assert model_sgd.optimizer == "gd"
        
        with pytest.raises(NotImplementedError):
            LSTMICModel(d_feat=2, optimizer="invalid_opt", GPU=-1)

    def test_lstmic_model_loss_fn(self):
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss import LSTMICModel
        model = LSTMICModel(d_feat=2, hidden_size=4, num_layers=1, loss="ic", GPU=-1)
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        
        loss_ic = model.loss_fn(pred, label, None)
        assert abs(loss_ic.item() - 0.0) < 1e-4
        
        model.loss = "mse"
        loss_mse = model.loss_fn(pred, label, None)
        assert abs(loss_mse.item() - 0.0) < 1e-4
        
        weight = torch.tensor([1.0, 2.0, 3.0])
        loss_mse_w = model.loss_fn(pred, label, weight)
        assert abs(loss_mse_w.item() - 0.0) < 1e-4
        
        model.loss = "mix"
        loss_mix = model.loss_fn(pred, label, None)
        assert loss_mix is not None
        
        model.loss = "invalid_loss"
        with pytest.raises(ValueError):
            model.loss_fn(pred, label, None)

    def test_lstmic_model_fit_and_predict(self, tmp_path):
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss import LSTMICModel
        model = LSTMICModel(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            n_epochs=2,
            batch_size=2,
            early_stop=2,
            n_jobs=1,
            GPU=-1,
        )
        
        dataset = MockMultiInstrumentTSDataset()
        save_path = tmp_path / "lstmic.pth"
        
        evals = {}
        model.fit(dataset, evals_result=evals, save_path=save_path)
        assert model.fitted
        assert "train" in evals
        assert "valid" in evals
        assert len(evals["train"]) == 2
        
        preds = model.predict(dataset)
        assert isinstance(preds, pd.Series)
        assert len(preds) == 18
        
        model_ir = LSTMICModel(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            n_epochs=2,
            batch_size=2,
            early_stop=2,
            n_jobs=1,
            GPU=-1,
            metric="ir"
        )
        model_ir.topk = 2
        evals_ir = {}
        model_ir.fit(dataset, evals_result=evals_ir, save_path=save_path)
        assert model_ir.fitted
        assert "train" in evals_ir
        assert "valid" in evals_ir
        assert "valid_ic" in evals_ir
        
        with pytest.raises(ValueError, match="Unsupported reweighter type"):
            model.fit(dataset, reweighter="invalid_reweighter")
            
        from qlib.data.dataset.weight import Reweighter
        mock_reweighter = MagicMock(spec=Reweighter)
        mock_reweighter.reweight.side_effect = lambda x: np.ones(len(x))
        model.fit(dataset, reweighter=mock_reweighter, save_path=save_path)


# ===========================================================================
# LSTM Rank Wrapper Tests
# ===========================================================================

class TestLSTMRankModel:
    def test_lstmrank_model_init_and_properties(self):
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_rank import LSTMRankModel
        
        model = LSTMRankModel(d_feat=2, hidden_size=4, num_layers=1, n_epochs=2, batch_size=2, GPU=-1)
        assert model.d_feat == 2
        assert model.hidden_size == 4
        assert not model.fitted
        assert not model.use_gpu
        
        model_sgd = LSTMRankModel(d_feat=2, hidden_size=4, num_layers=1, optimizer="gd", GPU=-1)
        assert model_sgd.optimizer == "gd"
        
        with pytest.raises(NotImplementedError):
            LSTMRankModel(d_feat=2, optimizer="invalid_opt", GPU=-1)

    def test_lstmrank_model_loss_fn(self):
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_rank import LSTMRankModel
        model = LSTMRankModel(d_feat=2, hidden_size=4, num_layers=1, GPU=-1)
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        
        loss_mse = model.loss_fn(pred, label, None)
        assert abs(loss_mse.item() - 0.0) < 1e-4
        
        model.loss = "invalid_loss"
        with pytest.raises(ValueError):
            model.loss_fn(pred, label, None)

    def test_lstmrank_model_fit_and_predict(self, tmp_path):
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_rank import LSTMRankModel
        model = LSTMRankModel(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            n_epochs=2,
            batch_size=2,
            early_stop=2,
            n_jobs=1,
            GPU=-1,
        )
        
        dataset = MockMultiInstrumentTSDataset()
        save_path = tmp_path / "lstmrank.pth"
        
        evals = {}
        model.fit(dataset, evals_result=evals, save_path=save_path)
        assert model.fitted
        assert "train" in evals
        assert "valid" in evals
        assert len(evals["train"]) == 2
        
        preds = model.predict(dataset)
        assert isinstance(preds, pd.Series)
        assert len(preds) == 18
        
        with pytest.raises(ValueError, match="Unsupported reweighter type"):
            model.fit(dataset, reweighter="invalid_reweighter")
            
        from qlib.data.dataset.weight import Reweighter
        mock_reweighter = MagicMock(spec=Reweighter)
        mock_reweighter.reweight.side_effect = lambda x: np.ones(len(x))
        model.fit(dataset, reweighter=mock_reweighter, save_path=save_path)


# ===========================================================================
# TabNet Wrapper Tests
# ===========================================================================

class TestTabnetModel:
    def test_tabnet_fit_non_ir(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_tabnet import TabnetModel
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for TabNet: {exc}")
            
        model = TabnetModel.__new__(TabnetModel)
        model.metric = "ic"
        
        with patch("quantpits.utils.model_wrappers.custom.pytorch_tabnet._Base.fit", return_value="native_fit") as mock_fit:
            res = model.fit(None)
            assert res == "native_fit"
            mock_fit.assert_called_once()
            
    def test_tabnet_fit_ir(self, tmp_path):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_tabnet import TabnetModel
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for TabNet: {exc}")
            
        model = TabnetModel.__new__(TabnetModel)
        model.metric = "ir"
        model.pretrain = True
        model.pretrain_file = str(tmp_path / "pretrain.pth")
        model.out_dim = 2
        model.final_out_dim = 1
        model.device = torch.device("cpu")
        model.optimizer = "adam"
        model.lr = 0.001
        model.n_epochs = 2
        model.early_stop = 2
        model.batch_size = 2
        model.d_feat = 2
        model.logger = MagicMock()
        
        model.pretrain_fn = MagicMock()
        
        class DummyTabnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(1))
            def forward(self, x, priors):
                return torch.ones(x.shape[0], 2), None
                
        model.tabnet_model = DummyTabnet()
        torch.save(model.tabnet_model.state_dict(), model.pretrain_file)
        
        dataset = MockDataset()
        save_path = tmp_path / "tabnet.pth"
        
        model.train_epoch = MagicMock()
        model.test_epoch = MagicMock(return_value=(0.1, 0.5))
        
        evals = {}
        model.fit(dataset, evals_result=evals, save_path=save_path)
        assert model.fitted
        assert len(evals["train"]) == 2
        
        # Test alternative optimizer SGD and pretrain=False
        model_sgd = TabnetModel.__new__(TabnetModel)
        model_sgd.metric = "ir"
        model_sgd.pretrain = False
        model_sgd.out_dim = 2
        model_sgd.final_out_dim = 1
        model_sgd.device = torch.device("cpu")
        model_sgd.optimizer = "sgd"
        model_sgd.lr = 0.001
        model_sgd.n_epochs = 1
        model_sgd.early_stop = 1
        model_sgd.batch_size = 2
        model_sgd.d_feat = 2
        model_sgd.logger = MagicMock()
        model_sgd.tabnet_model = DummyTabnet()
        model_sgd.train_epoch = MagicMock()
        model_sgd.test_epoch = MagicMock(return_value=(0.1, 0.5))
        
        model_sgd.fit(dataset, evals_result={}, save_path=save_path)
        assert model_sgd.fitted

    def test_tabnet_predict_singleton(self, tmp_path):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_tabnet import TabnetModel
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for TabNet: {exc}")
            
        model = TabnetModel.__new__(TabnetModel)
        model.device = torch.device("cpu")
        model.batch_size = 1
        model.d_feat = 2
        
        class SingletonFinetuneModel(torch.nn.Module):
            def forward(self, x, priors):
                return torch.tensor(1.5)
                
        model.tabnet_model = SingletonFinetuneModel()
        
        df = pd.DataFrame([[1.0, 2.0]], columns=["f1", "f2"])
        preds = model._tabnet_predict(df)
        assert np.array_equal(preds, np.array([1.5]))


# ===========================================================================
# TCTS Wrapper Tests
# ===========================================================================

class MockTCTSDataset(MockDataset):
    def prepare(self, tags, *args, **kwargs):
        if isinstance(tags, list) and len(tags) == 3:
            df_train = super().prepare("train", *args, **kwargs)
            df_valid = super().prepare("valid", *args, **kwargs)
            df_test = super().prepare("test", *args, **kwargs)
            return df_train, df_valid, df_test
        return super().prepare(tags, *args, **kwargs)


class TestTCTSModel:
    def test_tcts_init_and_metric(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_tcts import TCTS
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for TCTS: {exc}")
            
        model = TCTS.__new__(TCTS)
        model.metric = "ic"
        
        assert model.metric_fn(np.array([1.0]), np.array([2.0])) == 0.0
        
        pred = np.array([1.0, 2.0, 3.0])
        label = np.array([1.0, 2.0, 3.0])
        assert abs(model.metric_fn(pred, label) - 1.0) < 1e-4
        
        model.metric = "rank_ic"
        assert abs(model.metric_fn(pred, label) - 1.0) < 1e-4
        
        model.metric = "invalid_metric"
        assert model.metric_fn(pred, label) == 0.0

    def test_tcts_test_epoch(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_tcts import TCTS
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for TCTS: {exc}")
            
        model = TCTS.__new__(TCTS)
        model.batch_size = 2
        model.target_label = 0
        model.device = torch.device("cpu")
        model.metric = "ic"
        
        model.fore_model = MagicMock(spec=torch.nn.Module)
        model.fore_model.eval = MagicMock()
        model.fore_model.side_effect = lambda x: torch.arange(float(x.shape[0]))
        
        data_x = pd.DataFrame(np.arange(8).reshape(4, 2))
        data_y = pd.DataFrame(np.arange(8).reshape(4, 2))
        
        mse, score = model.test_epoch(data_x, data_y)
        assert mse > 0.0
        assert score is not None
        assert not np.isnan(score)
        
    def test_tcts_training_and_fit(self, tmp_path):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_tcts import TCTS
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for TCTS: {exc}")
            
        model = TCTS.__new__(TCTS)
        model.d_feat = 2
        model.hidden_size = 4
        model.num_layers = 1
        model.dropout = 0.0
        model.input_dim = 2
        model.output_dim = 1
        model.device = torch.device("cpu")
        model._fore_optimizer = "adam"
        model._weight_optimizer = "gd"
        model.fore_lr = 0.001
        model.weight_lr = 0.001
        model.n_epochs = 2
        model.early_stop = 2
        model.seed = 42
        model.metric = "ic"
        model.batch_size = 2
        model.use_gpu = False
        
        model.train_epoch = MagicMock()
        model.test_epoch = MagicMock(side_effect=[(0.1, 0.5), (0.2, 0.4), (0.1, 0.5), (0.2, 0.4)])
        
        save_path = str(tmp_path / "tcts_model")
        
        dataset = MockTCTSDataset()
        
        model.fit(dataset, verbose=True, save_path=save_path)
        assert model.fitted
        
        model_gd = TCTS.__new__(TCTS)
        model_gd.d_feat = 2
        model_gd.hidden_size = 4
        model_gd.num_layers = 1
        model_gd.dropout = 0.0
        model_gd.input_dim = 2
        model_gd.output_dim = 1
        model_gd.device = torch.device("cpu")
        model_gd._fore_optimizer = "gd"
        model_gd._weight_optimizer = "adam"
        model_gd.fore_lr = 0.001
        model_gd.weight_lr = 0.001
        model_gd.n_epochs = 1
        model_gd.early_stop = 1
        model_gd.seed = 42
        model_gd.metric = "ic"
        model_gd.batch_size = 2
        model_gd.use_gpu = False
        model_gd.train_epoch = MagicMock()
        model_gd.test_epoch = MagicMock(return_value=(0.1, 0.5))
        
        model_gd.fit(dataset, verbose=False, save_path=save_path)
        
        model_invalid = TCTS.__new__(TCTS)
        model_invalid._fore_optimizer = "invalid"
        model_invalid.d_feat = 2
        model_invalid.hidden_size = 4
        model_invalid.num_layers = 1
        model_invalid.dropout = 0.0
        model_invalid.input_dim = 2
        model_invalid.output_dim = 1
        model_invalid.device = torch.device("cpu")
        with pytest.raises(NotImplementedError):
            model_invalid.training(None, None, None, None, None, None, save_path=save_path)
            
        model_invalid2 = TCTS.__new__(TCTS)
        model_invalid2._fore_optimizer = "adam"
        model_invalid2._weight_optimizer = "invalid"
        model_invalid2.d_feat = 2
        model_invalid2.hidden_size = 4
        model_invalid2.num_layers = 1
        model_invalid2.dropout = 0.0
        model_invalid2.input_dim = 2
        model_invalid2.output_dim = 1
        model_invalid2.device = torch.device("cpu")
        model_invalid2.fore_lr = 0.001
        model_invalid2.weight_lr = 0.001
        with pytest.raises(NotImplementedError):
            model_invalid2.training(None, None, None, None, None, None, save_path=save_path)


# ===========================================================================
# GATsPlus Wrapper Tests
# ===========================================================================

class TestGATsPlusModel:
    def test_daily_batch_sampler(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import DailyBatchSampler
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for DailyBatchSampler: {exc}")
            
        class MockDataSource:
            def get_index(self):
                return pd.MultiIndex.from_product(
                    [pd.date_range("2020-01-01", periods=3), ["A", "B"]],
                    names=["datetime", "instrument"]
                )
        
        sampler = DailyBatchSampler(MockDataSource())
        assert len(sampler) == 3
        indices = list(sampler)
        assert len(indices) == 3
        assert np.array_equal(indices[0], np.array([0, 1]))

    def test_gat_model(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATModel
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for GATModel: {exc}")
            
        # GRU base model
        gat_gru = GATModel(d_feat=2, hidden_size=4, num_layers=1, base_model="GRU")
        assert gat_gru is not None
        
        # LSTM base model
        gat_lstm = GATModel(d_feat=2, hidden_size=4, num_layers=1, base_model="LSTM")
        assert gat_lstm is not None
        
        # Invalid base model
        with pytest.raises(ValueError):
            GATModel(d_feat=2, base_model="invalid")
            
        # Test forward
        x = torch.randn(3, 5, 2)
        pred = gat_gru(x)
        assert pred.shape == (3,)

    def test_gats_plus_init(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATsPlus
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for GATsPlus: {exc}")
            
        model = GATsPlus(d_feat=2, hidden_size=4, num_layers=1, base_model="GRU", optimizer="adam", GPU=-1)
        assert model.d_feat == 2
        assert model.hidden_size == 4
        assert not model.fitted
        
        model_sgd = GATsPlus(d_feat=2, hidden_size=4, num_layers=1, base_model="GRU", optimizer="gd", GPU=-1)
        assert model_sgd.optimizer == "gd"
        
        with pytest.raises(NotImplementedError):
            GATsPlus(d_feat=2, optimizer="invalid", GPU=-1)

        with pytest.raises(ValueError):
            GATsPlus(d_feat=2, base_model="invalid", GPU=-1)

    def test_gats_plus_loss_and_metric(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATsPlus
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for GATsPlus: {exc}")
            
        model = GATsPlus(d_feat=2, hidden_size=4, num_layers=1, base_model="GRU", loss="mse", metric="ic", GPU=-1)
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        
        loss_corr = model.correlation_loss(pred, label)
        assert abs(loss_corr.item() - 0.0) < 1e-4
        
        loss_small = model.correlation_loss(torch.tensor([1.0]), torch.tensor([2.0]))
        assert loss_small.item() == 0.0
        
        loss_mse = model.loss_fn(pred, label)
        assert abs(loss_mse.item() - 0.0) < 1e-4
        
        model.loss = "corr"
        loss_c = model.loss_fn(pred, label)
        assert abs(loss_c.item() - 0.0) < 1e-4
        
        model.loss = "ic"
        loss_ic = model.loss_fn(pred, label)
        assert loss_ic is not None
        
        model.loss = "unknown"
        with pytest.raises(ValueError):
            model.loss_fn(pred, label)
            
        model.metric = "loss"
        model.loss = "mse"
        metric_loss = model.metric_fn(pred, label)
        assert abs(metric_loss.item() - 0.0) < 1e-4
        
        model.metric = "ic"
        metric_ic = model.metric_fn(pred, label)
        assert abs(metric_ic - 1.0) < 1e-4
        
        metric_ic_small = model.metric_fn(torch.tensor([1.0]), torch.tensor([2.0]))
        assert metric_ic_small == 0.0
        
        model.metric = "ric"
        metric_ric = model.metric_fn(pred, label)
        assert abs(metric_ric - 1.0) < 1e-4
        
        metric_ric_small = model.metric_fn(torch.tensor([1.0]), torch.tensor([2.0]))
        assert metric_ric_small == 0.0
        
        model.metric = "unknown"
        with pytest.raises(ValueError):
            model.metric_fn(pred, label)

    def test_gats_plus_daily_inter(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATsPlus
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for GATsPlus: {exc}")
            
        model = GATsPlus(d_feat=2, hidden_size=4, num_layers=1, GPU=-1)
        
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=3), ["A", "B"]],
            names=["datetime", "instrument"]
        )
        df = pd.DataFrame(np.ones((6, 2)), index=idx)
        
        daily_index, daily_count = model.get_daily_inter(df, shuffle=False)
        assert np.array_equal(daily_index, np.array([0, 2, 4]))
        assert np.array_equal(daily_count, np.array([2, 2, 2]))
        
        daily_index_s, daily_count_s = model.get_daily_inter(df, shuffle=True)
        assert len(daily_index_s) == 3
        assert len(daily_count_s) == 3

    def test_gats_plus_fit_and_predict(self, tmp_path):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATsPlus
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing for GATsPlus: {exc}")
            
        model = GATsPlus(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            base_model="GRU",
            metric="ic",
            n_epochs=2,
            early_stop=2,
            n_jobs=1,
            GPU=-1,
            seed=42
        )
        
        dataset = MockMultiInstrumentTSDataset()
        save_path = tmp_path / "gats_plus.pth"
        
        model.fit(dataset, evals_result={}, save_path=save_path)
        assert model.fitted
        
        preds = model.predict(dataset)
        assert len(preds) == 18
        
        model_ir = GATsPlus(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            base_model="GRU",
            metric="ir",
            n_epochs=2,
            early_stop=2,
            n_jobs=1,
            GPU=-1,
            seed=42
        )
        model_ir.topk = 2
        evals_ir = {}
        model_ir.fit(dataset, evals_result=evals_ir, save_path=save_path)
        assert model_ir.fitted
        assert "train" in evals_ir
        assert "valid" in evals_ir
        assert "valid_ic" in evals_ir
        assert "valid_rank_ic" in evals_ir
        
        from qlib.contrib.model.pytorch_gru import GRUModel as QlibGRUModel
        pretrained_gru = QlibGRUModel(d_feat=2, hidden_size=4, num_layers=1)
        model_path_save = tmp_path / "base_model.pth"
        torch.save(pretrained_gru.state_dict(), model_path_save)
        
        model_load = GATsPlus(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            base_model="GRU",
            model_path=str(model_path_save),
            n_epochs=1,
            early_stop=1,
            n_jobs=1,
            GPU=-1,
            seed=42
        )
        model_load.fit(dataset, evals_result={}, save_path=save_path)
        assert model_load.fitted

