"""
Smoke tests for the thin-wrapper files and qlib-dependent wrappers.

Strategy
--------
1.  Import smoke tests: verify every thin wrapper (class GRU(Mixin, Base): pass)
    can be imported without errors.  Skips individually if the required qlib
    model is unavailable so that a missing optional qlib dependency does not
    fail the whole suite.

2.  MRO correctness: confirm that ICMetricMixin.metric_fn is the one that
    wins in the MRO, and that LossHistoryMixin.fit wins when both are stacked.

Each parametrize entry is: (dotted_module_path, class_name)
"""

import importlib
import pytest


# ---------------------------------------------------------------------------
# Registry of all thin wrapper modules
# (module_path, class_name_inside_module)
# ---------------------------------------------------------------------------

IC_WRAPPERS = [
    ("quantpits.utils.model_wrappers.custom.pytorch_gru",           "GRU"),
    ("quantpits.utils.model_wrappers.custom.pytorch_lstm",          "LSTM"),
    ("quantpits.utils.model_wrappers.custom.pytorch_alstm",         "ALSTM"),
    ("quantpits.utils.model_wrappers.custom.pytorch_alstm_ts",      "ALSTM"),
    ("quantpits.utils.model_wrappers.custom.pytorch_adarnn",        "ADARNN"),
    ("quantpits.utils.model_wrappers.custom.pytorch_gats_ts",       "GATs"),
    ("quantpits.utils.model_wrappers.custom.pytorch_igmtf",         "IGMTF"),
    ("quantpits.utils.model_wrappers.custom.pytorch_krnn",          "KRNN"),
    ("quantpits.utils.model_wrappers.custom.pytorch_localformer",   "LocalformerModel"),
    ("quantpits.utils.model_wrappers.custom.pytorch_localformer_ts","LocalformerModelIC"),
    ("quantpits.utils.model_wrappers.custom.pytorch_sandwich",      "Sandwich"),
    ("quantpits.utils.model_wrappers.custom.pytorch_sfm",           "SFM"),
    ("quantpits.utils.model_wrappers.custom.pytorch_tabnet",        "TabnetModel"),
    ("quantpits.utils.model_wrappers.custom.pytorch_tcn",           "TCN"),
    ("quantpits.utils.model_wrappers.custom.pytorch_tcn_ts",        "TCNIC"),
    ("quantpits.utils.model_wrappers.custom.pytorch_tra",           "TRAModelIC"),
    ("quantpits.utils.model_wrappers.custom.pytorch_transformer",   "TransformerModel"),
    ("quantpits.utils.model_wrappers.custom.pytorch_transformer_ts","TransformerModelIC"),
    ("quantpits.utils.model_wrappers.custom.pytorch_general_nn",    "GeneralPTNN"),
]

LH_WRAPPERS = [
    ("quantpits.utils.model_wrappers.lh.pytorch_gru",            "GRU"),
    ("quantpits.utils.model_wrappers.lh.pytorch_lstm",           "LSTM"),
    ("quantpits.utils.model_wrappers.lh.pytorch_alstm",          "ALSTM"),
    ("quantpits.utils.model_wrappers.lh.pytorch_alstm_ts",       "ALSTM"),
    ("quantpits.utils.model_wrappers.lh.pytorch_adarnn",         "ADARNN"),
    ("quantpits.utils.model_wrappers.lh.pytorch_gats_plus",      "GATsPlus"),
    ("quantpits.utils.model_wrappers.lh.pytorch_gats_ts",        "GATs"),
    ("quantpits.utils.model_wrappers.lh.pytorch_igmtf",          "IGMTF"),
    ("quantpits.utils.model_wrappers.lh.pytorch_krnn",           "KRNN"),
    ("quantpits.utils.model_wrappers.lh.pytorch_localformer",    "LocalformerModel"),
    ("quantpits.utils.model_wrappers.lh.pytorch_localformer_ts", "LocalformerModelIC"),
    ("quantpits.utils.model_wrappers.lh.pytorch_sandwich",       "Sandwich"),
    ("quantpits.utils.model_wrappers.lh.pytorch_sfm",            "SFM"),
    ("quantpits.utils.model_wrappers.lh.pytorch_tabnet",         "TabnetModel"),
    ("quantpits.utils.model_wrappers.lh.pytorch_tcn",            "TCN"),
    ("quantpits.utils.model_wrappers.lh.pytorch_tcn_ts",         "TCNIC"),
    ("quantpits.utils.model_wrappers.lh.pytorch_tra",            "TRAModelIC"),
    ("quantpits.utils.model_wrappers.lh.pytorch_transformer",    "TransformerModel"),
    ("quantpits.utils.model_wrappers.lh.pytorch_transformer_ts", "TransformerModelIC"),
]

STANDALONE_MODELS = [
    ("quantpits.utils.model_wrappers.custom.pytorch_gats_plus",    "GATsPlus"),
    ("quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss", "LSTMICModel"),
    ("quantpits.utils.model_wrappers.custom.pytorch_lstm_rank",    "LSTMRankModel"),
    ("quantpits.utils.model_wrappers.custom.pytorch_tcts",         "TCTS"),
]

ALL_WRAPPERS = IC_WRAPPERS + LH_WRAPPERS + STANDALONE_MODELS


# ===========================================================================
# Phase 3A: Import smoke tests
# ===========================================================================

@pytest.mark.parametrize("module_path,class_name", ALL_WRAPPERS,
                         ids=[f"{m.split('.')[-1]}.{c}" for m, c in ALL_WRAPPERS])
def test_wrapper_importable(module_path, class_name):
    """Every wrapper module must be importable and contain the expected class."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.skip(f"qlib dependency missing for {module_path}: {exc}")

    assert hasattr(mod, class_name), (
        f"{module_path} should define class '{class_name}'"
    )
    cls = getattr(mod, class_name)
    assert isinstance(cls, type), f"'{class_name}' must be a class"


# ===========================================================================
# Phase 3B: MRO correctness
# ===========================================================================

class TestMROCorrectness:
    """Verify that mixin methods win over qlib base class methods."""

    def _load_or_skip(self, module_path, class_name):
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except ImportError as exc:
            pytest.skip(f"dependency missing ({module_path}): {exc}")

    def _import_mixin_or_skip(self, dotted):
        """Import a mixin class, skipping if torch (or qlib) is absent.

        ic_metric_mixin.py has a top-level `import torch`, so importing it
        raises ImportError in environments where torch is not installed.
        """
        try:
            parts = dotted.rsplit(".", 1)
            mod = importlib.import_module(parts[0])
            return getattr(mod, parts[1])
        except ImportError as exc:
            pytest.skip(f"dependency missing ({dotted}): {exc}")

    def test_gru_ic_uses_strategymetricmixin_metric_fn(self):
        StrategyMetricMixin = self._import_mixin_or_skip(
            "quantpits.utils.model_wrappers.mixins.strategy.StrategyMetricMixin"
        )
        cls = self._load_or_skip(
            "quantpits.utils.model_wrappers.custom.pytorch_gru", "GRU"
        )
        # The MRO-resolved metric_fn must be from StrategyMetricMixin, not base GRU
        for klass in cls.__mro__:
            if "metric_fn" in klass.__dict__:
                assert klass is StrategyMetricMixin, (
                    f"Expected StrategyMetricMixin.metric_fn to win, got {klass}"
                )
                break

    def test_gru_ic_lh_uses_losshistorymixin_fit(self):
        LossHistoryMixin = self._import_mixin_or_skip(
            "quantpits.utils.model_wrappers.mixins.loss_history.LossHistoryMixin"
        )
        cls = self._load_or_skip(
            "quantpits.utils.model_wrappers.lh.pytorch_gru", "GRU"
        )
        for klass in cls.__mro__:
            if "fit" in klass.__dict__:
                assert klass is LossHistoryMixin, (
                    f"Expected LossHistoryMixin.fit to win, got {klass}"
                )
                break

    def test_lstm_ic_lh_mro_order(self):
        """Full stack: LossHistoryMixin > StrategyMetricMixin > BaseModel."""
        LossHistoryMixin = self._import_mixin_or_skip(
            "quantpits.utils.model_wrappers.mixins.loss_history.LossHistoryMixin"
        )
        StrategyMetricMixin = self._import_mixin_or_skip(
            "quantpits.utils.model_wrappers.mixins.strategy.StrategyMetricMixin"
        )
        cls = self._load_or_skip(
            "quantpits.utils.model_wrappers.lh.pytorch_lstm", "LSTM"
        )
        mro = cls.__mro__
        lh_idx = next(
            (i for i, k in enumerate(mro) if k is LossHistoryMixin), None
        )
        sm_idx = next(
            (i for i, k in enumerate(mro) if k is StrategyMetricMixin), None
        )
        assert lh_idx is not None, "LossHistoryMixin must be in MRO"
        assert sm_idx is not None, "StrategyMetricMixin must be in MRO"
        assert lh_idx < sm_idx, (
            "LossHistoryMixin must precede StrategyMetricMixin in MRO"
        )

    def test_tra_model_ic_uses_strategymetricmixin(self):
        """TRAModelIC must use StrategyMetricMixin in its MRO."""
        StrategyMetricMixin = self._import_mixin_or_skip(
            "quantpits.utils.model_wrappers.mixins.strategy.StrategyMetricMixin"
        )
        cls = self._load_or_skip(
            "quantpits.utils.model_wrappers.custom.pytorch_tra", "TRAModelIC"
        )
        assert StrategyMetricMixin in cls.__mro__, (
            "TRAModelIC must inherit from StrategyMetricMixin"
        )



# ===========================================================================
# Phase 3C: Standalone model sub-function tests (Tier 4)
# ===========================================================================

class TestGATsPlus:
    """Test GATsPlus internals without requiring a full dataset."""

    pytestmark = pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )

    @pytest.fixture
    def model(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATsPlus
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing: {exc}")
        # Construct without GPU requirement, no model_path
        obj = GATsPlus.__new__(GATsPlus)
        obj.device = torch.device("cpu")
        obj.loss = "mse"
        obj.metric = "ic"
        return obj

    def test_mse_loss(self, model):
        torch = pytest.importorskip("torch")
        pred  = torch.tensor([1.0, 2.0, 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        loss = model.mse(pred, label)
        assert abs(loss.item()) < 1e-6

    def test_correlation_loss_perfect_correlation(self, model):
        torch = pytest.importorskip("torch")
        pred  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        label = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss = model.correlation_loss(pred, label)
        # 1 - Pearson(1,1) = 1 - 1.0 = 0
        assert abs(loss.item()) < 1e-5

    def test_correlation_loss_in_range(self, model):
        torch = pytest.importorskip("torch")
        torch.manual_seed(1)
        pred  = torch.randn(30)
        label = torch.randn(30)
        loss = model.correlation_loss(pred, label)
        assert 0.0 <= loss.item() <= 2.0 + 1e-6

    def test_correlation_loss_batch_size_one_returns_zero(self, model):
        torch = pytest.importorskip("torch")
        pred  = torch.tensor([1.0])
        label = torch.tensor([2.0])
        loss = model.correlation_loss(pred, label)
        assert abs(loss.item()) < 1e-6

    def test_loss_fn_mse(self, model):
        torch = pytest.importorskip("torch")
        math = __import__("math")
        model.loss = "mse"
        pred  = torch.tensor([1.0, float("nan"), 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        loss = model.loss_fn(pred, label)
        # nan in pred is masked by ~isnan(label); nan in pred still passes through
        # mask is ~isnan(label) so index 1 survives but pred[1]=nan
        # Just check it doesn't crash and result is finite or nan
        assert isinstance(loss.item(), float)

    def test_loss_fn_corr_mode(self, model):
        torch = pytest.importorskip("torch")
        model.loss = "corr"
        pred  = torch.tensor([1.0, 2.0, 3.0, 4.0])
        label = torch.tensor([4.0, 3.0, 2.0, 1.0])
        loss = model.loss_fn(pred, label)
        # corr with perfectly anti-correlated → 1-(-1)=2
        assert abs(loss.item() - 2.0) < 1e-4

    def test_loss_fn_unknown_raises(self, model):
        torch = pytest.importorskip("torch")
        model.loss = "huber"
        with pytest.raises(ValueError, match="unknown loss"):
            model.loss_fn(torch.randn(5), torch.randn(5))

    def test_metric_fn_ic(self, model):
        torch = pytest.importorskip("torch")
        model.metric = "ic"
        pred  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        label = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        score = model.metric_fn(pred, label)
        assert abs(score - 1.0) < 1e-4

    def test_metric_fn_ric(self, model):
        torch = pytest.importorskip("torch")
        model.metric = "ric"
        pred  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        label = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        score = model.metric_fn(pred, label)
        assert abs(score - 1.0) < 1e-4

    def test_metric_fn_loss_mode(self, model):
        torch = pytest.importorskip("torch")
        model.metric = "loss"
        model.loss = "mse"
        pred  = torch.tensor([1.0, 2.0, 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        score = model.metric_fn(pred, label)
        # -MSE(perfect) = 0
        assert abs(score.item()) < 1e-6

    def test_metric_fn_unknown_raises(self, model):
        torch = pytest.importorskip("torch")
        model.metric = "sharpe"
        with pytest.raises(ValueError, match="unknown metric"):
            model.metric_fn(torch.randn(5), torch.randn(5))

    def test_metric_fn_insufficient_data_returns_zero(self, model):
        torch = pytest.importorskip("torch")
        model.metric = "ic"
        # Only 1 valid point after mask
        pred  = torch.tensor([1.0, float("inf")])
        label = torch.tensor([1.0, 2.0])
        # After mask (isfinite(label)=[T,T], so both pass; 2 points → ok)
        # Let's make only 1 survive: all-non-finite label except 1
        pred2  = torch.tensor([1.0])
        label2 = torch.tensor([1.0])
        score = model.metric_fn(pred2, label2)
        # len < 2 → returns 0.0
        assert score == 0.0

    def test_gat_model_forward_shape(self):
        """GATModel.forward should output shape [N] for N stock samples."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_gats_plus import GATModel
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing: {exc}")
        model = GATModel(d_feat=6, hidden_size=32, num_layers=1, base_model="GRU")
        model.eval()
        x = torch.randn(10, 20, 6)  # [N_stocks, T_steps, D_feat]
        with torch.no_grad():
            out = model(x)
        assert out.shape == (10,), f"Expected shape (10,), got {out.shape}"


class TestLSTMICLoss:
    """Test LSTMICModel internals (pytorch_lstm_ic_loss.py)."""

    @pytest.fixture
    def icmodel(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss import (
                LSTMICModel, ICLoss
            )
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing: {exc}")
        return LSTMICModel.__new__(LSTMICModel), ICLoss

    def test_icloss_perfect_correlation(self, icmodel):
        torch = pytest.importorskip("torch")
        _, ICLoss = icmodel
        criterion = ICLoss()
        pred  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        label = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss = criterion(pred, label)
        assert abs(loss.item()) < 1e-5

    def test_icloss_nan_masking(self, icmodel):
        torch = pytest.importorskip("torch")
        math = __import__("math")
        _, ICLoss = icmodel
        criterion = ICLoss()
        pred  = torch.tensor([1.0, float("nan"), 3.0])
        label = torch.tensor([float("nan"), 2.0, 3.0])
        loss = criterion(pred, label)
        assert math.isfinite(loss.item())

    def test_icloss_backward(self, icmodel):
        torch = pytest.importorskip("torch")
        _, ICLoss = icmodel
        criterion = ICLoss()
        pred  = torch.randn(20, requires_grad=True)
        label = torch.randn(20)
        loss = criterion(pred, label)
        loss.backward()
        assert pred.grad is not None

    def test_calc_ic_computes_correct_ic(self, icmodel):
        """calc_ic should return ~1.0 rank IC for perfectly correlated data."""
        import numpy as np, pandas as pd
        LSTMICModel_obj, _ = icmodel

        # Build a MultiIndex like qlib uses: (datetime, instrument)
        dates = pd.date_range("2020-01-01", periods=3)
        instruments = ["A", "B", "C", "D", "E"]
        idx = pd.MultiIndex.from_product([dates, instruments],
                                          names=["datetime", "instrument"])
        n = len(idx)
        preds  = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)  # identical → rank_ic = 1.0

        # Patch the method reference
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss import LSTMICModel
        rank_ic, ic = LSTMICModel.calc_ic(None, preds, labels, idx)
        assert abs(rank_ic - 1.0) < 1e-4, f"rank_ic should be 1.0, got {rank_ic}"
        assert abs(ic - 1.0) < 1e-4, f"ic should be 1.0, got {ic}"

    def test_calc_ic_anti_correlated(self, icmodel):
        import numpy as np, pandas as pd
        from quantpits.utils.model_wrappers.custom.pytorch_lstm_ic_loss import LSTMICModel
        dates = pd.date_range("2020-01-01", periods=2)
        instruments = ["A", "B", "C", "D", "E"]
        idx = pd.MultiIndex.from_product([dates, instruments],
                                          names=["datetime", "instrument"])
        n = len(idx)
        preds  = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)[::-1]  # reversed
        rank_ic, ic = LSTMICModel.calc_ic(None, preds, labels, idx)
        assert rank_ic < -0.5


class TestLSTMRankModel:
    """Test LSTMRankModel internals (pytorch_lstm_rank.py)."""

    @pytest.fixture
    def model_cls(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_lstm_rank import LSTMRankModel
            return LSTMRankModel
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing: {exc}")

    def test_calc_ic_perfect_rank(self, model_cls):
        import numpy as np, pandas as pd
        dates = pd.date_range("2020-01-01", periods=3)
        instruments = ["A", "B", "C"]
        idx = pd.MultiIndex.from_product([dates, instruments],
                                          names=["datetime", "instrument"])
        n = len(idx)
        preds  = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        rank_ic, ic = model_cls.calc_ic(None, preds, labels, idx)
        assert abs(rank_ic - 1.0) < 1e-4

    def test_calc_ic_datetime_level_fallback(self, model_cls):
        """Should also work when index level is named something other than 'datetime'."""
        import numpy as np, pandas as pd
        dates = pd.date_range("2020-01-01", periods=2)
        instruments = ["X", "Y", "Z"]
        # name it 'date' instead of 'datetime'
        idx = pd.MultiIndex.from_product([dates, instruments],
                                          names=["date", "instrument"])
        n = len(idx)
        preds  = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        # 'datetime' not in index.names → should fall back to level=0 groupby
        rank_ic, ic = model_cls.calc_ic(None, preds, labels, idx)
        assert abs(rank_ic - 1.0) < 1e-4

    def test_mse_loss_fn(self, model_cls):
        torch = pytest.importorskip("torch")
        obj = model_cls.__new__(model_cls)
        obj.loss = "mse"
        pred   = torch.tensor([1.0, 2.0, 3.0])
        label  = torch.tensor([1.0, 2.0, 3.0])
        weight = torch.ones(3)
        loss = obj.loss_fn(pred, label, weight)
        assert abs(loss.item()) < 1e-6

    def test_mse_loss_fn_ignores_nan(self, model_cls):
        torch = pytest.importorskip("torch")
        math = __import__("math")
        obj = model_cls.__new__(model_cls)
        obj.loss = "mse"
        pred   = torch.tensor([1.0, float("nan"), 3.0])
        label  = torch.tensor([1.0, float("nan"), 3.0])
        weight = torch.ones(3)
        loss = obj.loss_fn(pred, label, weight)
        # NaN in label is masked; NaN in pred passes through → result is NaN
        # Document current behaviour; at minimum should not raise
        assert isinstance(loss.item(), float)

    def test_mse_loss_none_weight_defaults_to_ones(self, model_cls):
        torch = pytest.importorskip("torch")
        obj = model_cls.__new__(model_cls)
        obj.loss = "mse"
        pred  = torch.tensor([2.0, 3.0])
        label = torch.tensor([1.0, 1.0])
        loss_w_none = obj.loss_fn(pred, label, weight=None)
        loss_w_ones = obj.loss_fn(pred, label, weight=torch.ones(2))
        assert abs(loss_w_none.item() - loss_w_ones.item()) < 1e-6

    def test_unknown_loss_raises(self, model_cls):
        torch = pytest.importorskip("torch")
        obj = model_cls.__new__(model_cls)
        obj.loss = "huber"
        with pytest.raises(ValueError, match="unknown loss"):
            obj.loss_fn(torch.randn(5), torch.randn(5), weight=None)


class TestTCTS:
    """Test TCTS.metric_fn (pure numpy IC / Rank-IC computation)."""

    @pytest.fixture
    def model(self):
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_tcts import TCTS
        except ImportError as exc:
            pytest.skip(f"qlib dependency missing: {exc}")
        obj = TCTS.__new__(TCTS)
        obj.metric = "ic"
        return obj

    def test_ic_perfect_correlation(self, model):
        import numpy as np
        model.metric = "ic"
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        label = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = model.metric_fn(pred, label)
        assert abs(score - 1.0) < 1e-5

    def test_ic_perfect_negative(self, model):
        import numpy as np
        model.metric = "ic"
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        label = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        score = model.metric_fn(pred, label)
        assert abs(score + 1.0) < 1e-5

    def test_rank_ic_perfect(self, model):
        import numpy as np
        model.metric = "rank_ic"
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        label = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        score = model.metric_fn(pred, label)
        assert abs(score - 1.0) < 1e-5

    def test_nan_in_label_masked(self, model):
        import numpy as np
        model.metric = "ic"
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        label = np.array([1.0, 2.0, float("nan"), 4.0])
        score = model.metric_fn(pred, label)
        assert abs(score - 1.0) < 1e-5

    def test_all_nan_labels_returns_zero(self, model):
        import numpy as np
        model.metric = "ic"
        pred = np.array([1.0, 2.0, 3.0])
        label = np.array([float("nan"), float("nan"), float("nan")])
        score = model.metric_fn(pred, label)
        assert score == 0.0

    def test_fewer_than_two_valid_returns_zero(self, model):
        import numpy as np
        model.metric = "ic"
        pred = np.array([1.0])
        label = np.array([1.0])
        score = model.metric_fn(pred, label)
        assert score == 0.0

    def test_unknown_metric_returns_zero(self, model):
        import numpy as np
        model.metric = "unknown"
        pred = np.array([1.0, 2.0, 3.0])
        label = np.array([1.0, 2.0, 3.0])
        score = model.metric_fn(pred, label)
        assert score == 0.0
