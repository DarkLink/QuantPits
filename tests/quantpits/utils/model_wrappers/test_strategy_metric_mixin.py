"""
Tests for StrategyMetricMixin — static IC/IR helpers and metric_fn dispatch.

Self-contained: numpy, pandas, and torch only.  No qlib init, no dataset,
no GPU required.  Skipped automatically when torch is absent.
"""

import math
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch not installed – skipping StrategyMetricMixin tests")

# Import lazily to skip cleanly when qlib is missing.
try:
    from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin
except ImportError as exc:
    pytest.skip(f"qlib dependency missing: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def t(*values):
    return torch.tensor(values, dtype=torch.float32)


class _StubStrategyModel(StrategyMetricMixin):
    """Minimal stub for testing metric_fn dispatch without a real model."""

    def __init__(self, metric="ic"):
        self.metric = metric

    def loss_fn(self, pred, label):
        return torch.mean((pred - label) ** 2)


# ===========================================================================
# _batch_pearson_ic
# ===========================================================================

class TestBatchPearsonIC:
    def test_perfect_positive(self):
        ic = StrategyMetricMixin._batch_pearson_ic(t(1., 2., 3., 4., 5.), t(1., 2., 3., 4., 5.))
        assert abs(ic.item() - 1.0) < 1e-5

    def test_perfect_negative(self):
        ic = StrategyMetricMixin._batch_pearson_ic(t(1., 2., 3., 4., 5.), t(5., 4., 3., 2., 1.))
        assert abs(ic.item() + 1.0) < 1e-5

    def test_output_bounded(self):
        torch.manual_seed(42)
        p = torch.randn(100)
        l = torch.randn(100)
        ic = StrategyMetricMixin._batch_pearson_ic(p, l)
        assert -1.0 - 1e-6 <= ic.item() <= 1.0 + 1e-6

    def test_constant_pred_does_not_nan(self):
        ic = StrategyMetricMixin._batch_pearson_ic(t(2., 2., 2., 2.), t(1., 2., 3., 4.))
        assert math.isfinite(ic.item())

    def test_constant_label_does_not_nan(self):
        ic = StrategyMetricMixin._batch_pearson_ic(t(1., 2., 3., 4.), t(5., 5., 5., 5.))
        assert math.isfinite(ic.item())

    def test_fewer_than_two_returns_zero_tensor(self):
        ic = StrategyMetricMixin._batch_pearson_ic(t(1.0), t(1.0))
        assert ic.item() == 0.0

    def test_returns_tensor(self):
        ic = StrategyMetricMixin._batch_pearson_ic(t(1., 2., 3.), t(3., 2., 1.))
        assert isinstance(ic, torch.Tensor)


# ===========================================================================
# _batch_rank_ic
# ===========================================================================

class TestBatchRankIC:
    def test_perfect_rank_correlation(self):
        ric = StrategyMetricMixin._batch_rank_ic(t(1., 2., 3., 4.), t(10., 20., 30., 40.))
        assert abs(ric.item() - 1.0) < 1e-5

    def test_perfect_anti_correlation(self):
        ric = StrategyMetricMixin._batch_rank_ic(t(1., 2., 3., 4.), t(40., 30., 20., 10.))
        assert abs(ric.item() + 1.0) < 1e-5

    def test_fewer_than_two_returns_zero(self):
        ric = StrategyMetricMixin._batch_rank_ic(t(5.0), t(5.0))
        assert ric.item() == 0.0

    def test_output_bounded(self):
        torch.manual_seed(99)
        p = torch.randn(200)
        l = torch.randn(200)
        ric = StrategyMetricMixin._batch_rank_ic(p, l)
        assert -1.0 - 1e-6 <= ric.item() <= 1.0 + 1e-6

    def test_monotone_transform_preserves_rank_ic(self):
        p = t(1., 3., 2., 5., 4.)
        l = t(2., 4., 3., 5., 1.)
        ric_orig = StrategyMetricMixin._batch_rank_ic(p, l)
        ric_trans = StrategyMetricMixin._batch_rank_ic(p, l ** 3)
        assert abs(ric_orig.item() - ric_trans.item()) < 1e-4

    def test_returns_tensor(self):
        ric = StrategyMetricMixin._batch_rank_ic(t(1., 2., 3.), t(3., 2., 1.))
        assert isinstance(ric, torch.Tensor)


# ===========================================================================
# _compute_ic  (pandas groupby-based)
# ===========================================================================

class TestComputeIC:
    @staticmethod
    def _make_index(dates, instruments, date_name="datetime"):
        return pd.MultiIndex.from_product(
            [dates, instruments], names=[date_name, "instrument"]
        )

    def test_perfect_correlation(self):
        dates = pd.date_range("2020-01-01", periods=3)
        instruments = ["A", "B", "C", "D", "E"]
        idx = self._make_index(dates, instruments)
        n = len(idx)
        preds = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        ic = StrategyMetricMixin._compute_ic(preds, labels, idx)
        assert abs(ic - 1.0) < 1e-4

    def test_fewer_than_two_points_returns_zero(self):
        preds = np.array([1.0])
        labels = np.array([1.0])
        idx = pd.MultiIndex.from_tuples(
            [("2020-01-01", "A")], names=["datetime", "instrument"]
        )
        ic = StrategyMetricMixin._compute_ic(preds, labels, idx)
        assert ic == 0.0

    def test_fallback_when_no_datetime_level(self):
        dates = pd.date_range("2020-01-01", periods=2)
        instruments = ["X", "Y"]
        idx = self._make_index(dates, instruments, date_name="date")
        n = len(idx)
        preds = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        ic = StrategyMetricMixin._compute_ic(preds, labels, idx)
        assert abs(ic - 1.0) < 1e-4


# ===========================================================================
# _compute_rank_ic  (pandas groupby-based)
# ===========================================================================

class TestComputeRankIC:
    @staticmethod
    def _make_index(dates, instruments, date_name="datetime"):
        return pd.MultiIndex.from_product(
            [dates, instruments], names=[date_name, "instrument"]
        )

    def test_perfect_rank(self):
        dates = pd.date_range("2020-01-01", periods=3)
        instruments = ["A", "B", "C"]
        idx = self._make_index(dates, instruments)
        n = len(idx)
        preds = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        ric = StrategyMetricMixin._compute_rank_ic(preds, labels, idx)
        assert abs(ric - 1.0) < 1e-4

    def test_fewer_than_two_points_returns_zero(self):
        preds = np.array([1.0])
        labels = np.array([1.0])
        idx = pd.MultiIndex.from_tuples(
            [("2020-01-01", "A")], names=["datetime", "instrument"]
        )
        ric = StrategyMetricMixin._compute_rank_ic(preds, labels, idx)
        assert ric == 0.0

    def test_fallback_when_no_datetime_level(self):
        dates = pd.date_range("2020-01-01", periods=2)
        instruments = ["X", "Y", "Z"]
        idx = self._make_index(dates, instruments, date_name="date")
        n = len(idx)
        preds = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        ric = StrategyMetricMixin._compute_rank_ic(preds, labels, idx)
        assert abs(ric - 1.0) < 1e-4


# ===========================================================================
# _run_mini_backtest
# ===========================================================================

class TestRunMiniBacktest:
    @staticmethod
    def _make_index(dates, instruments):
        return pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )

    def test_equal_preds_returns_topk_mean_label(self):
        dates = pd.date_range("2020-01-01", periods=5)
        instruments = ["A", "B", "C", "D", "E"]
        idx = self._make_index(dates, instruments)
        n = len(idx)
        preds = np.ones(n)  # all equal
        labels = np.arange(n, dtype=float)
        ir = StrategyMetricMixin._run_mini_backtest(preds, labels, idx, topk=3)
        assert math.isfinite(ir)

    def test_fewer_than_two_points_returns_neg_inf(self):
        ir = StrategyMetricMixin._run_mini_backtest(
            np.array([1.0]), np.array([1.0]),
            pd.MultiIndex.from_tuples(
                [("2020-01-01", "A")], names=["datetime", "instrument"]
            ), topk=5,
        )
        assert ir == -np.inf

    def test_single_day_returns_neg_inf(self):
        instruments = ["A", "B", "C", "D", "E"]
        idx = self._make_index([pd.Timestamp("2020-01-01")], instruments)
        n = len(idx)
        preds = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        ir = StrategyMetricMixin._run_mini_backtest(preds, labels, idx, topk=3)
        assert ir == -np.inf

    def test_identical_daily_returns_returns_neg_inf(self):
        dates = pd.date_range("2020-01-01", periods=2)
        instruments = ["A", "B", "C", "D", "E"]
        idx = self._make_index(dates, instruments)
        # perfecly identical pred=label across both days → daily returns identical → std=0 → -inf
        preds = np.tile(np.arange(5, dtype=float), 2)
        labels = np.tile(np.arange(5, dtype=float), 2)
        ir = StrategyMetricMixin._run_mini_backtest(preds, labels, idx, topk=3)
        assert ir == -np.inf

    def test_group_smaller_than_topk_skipped(self):
        dates = pd.date_range("2020-01-01", periods=2)
        instruments = ["A", "B"]  # 2 stocks < topk=5
        idx = self._make_index(dates, instruments)
        n = len(idx)
        preds = np.arange(n, dtype=float)
        labels = np.arange(n, dtype=float)
        ir = StrategyMetricMixin._run_mini_backtest(preds, labels, idx, topk=5)
        assert ir == -np.inf


# ===========================================================================
# _find_inner_modules
# ===========================================================================

class TestFindInnerModules:
    def test_finds_nn_module_sorted_by_param_count(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.big = torch.nn.Linear(10, 10)
                self.small = torch.nn.Linear(2, 2)

        instance = FakeModel()
        candidates = StrategyMetricMixin._find_inner_modules(instance)
        assert len(candidates) == 2
        # Sorted by param count descending: big (110) before small (6)
        assert candidates[0][1] == "big"
        assert candidates[1][1] == "small"

    def test_empty_instance_returns_empty_list(self):
        class Empty:
            pass

        candidates = StrategyMetricMixin._find_inner_modules(Empty())
        assert candidates == []

    def test_excludes_known_loss_attrs(self):
        class ModelWithLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.Linear(5, 5)
                self.loss_fn = torch.nn.Linear(3, 3)
                self.criterion = torch.nn.Linear(4, 4)

        instance = ModelWithLoss()
        candidates = StrategyMetricMixin._find_inner_modules(instance)
        names = {name for _, name, _ in candidates}
        assert "loss_fn" not in names
        assert "criterion" not in names
        assert "lstm" in names

    def test_skips_private_attrs(self):
        class ModelWithPrivate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._hidden = torch.nn.Linear(5, 5)
                self.visible = torch.nn.Linear(3, 3)

        instance = ModelWithPrivate()
        candidates = StrategyMetricMixin._find_inner_modules(instance)
        names = {name for _, name, _ in candidates}
        assert "_hidden" not in names
        assert "visible" in names


# ===========================================================================
# metric_fn dispatch
# ===========================================================================

class TestStrategyMetricFn:
    def test_ir_uses_pearson(self):
        model = _StubStrategyModel(metric="ir")
        pred = t(1., 2., 3., 4., 5.)
        label = t(1., 2., 3., 4., 5.)
        score = model.metric_fn(pred, label)
        assert abs(score.item() - 1.0) < 1e-5

    def test_ic_uses_pearson(self):
        model = _StubStrategyModel(metric="ic")
        pred = t(1., 2., 3., 4., 5.)
        label = t(1., 2., 3., 4., 5.)
        score = model.metric_fn(pred, label)
        assert abs(score.item() - 1.0) < 1e-5

    def test_rank_ic_uses_spearman(self):
        model = _StubStrategyModel(metric="rank_ic")
        pred = t(1., 2., 3., 4., 5.)
        label = t(1., 2., 3., 4., 5.)
        score = model.metric_fn(pred, label)
        assert abs(score.item() - 1.0) < 1e-5

    def test_loss_returns_negative_mse(self):
        model = _StubStrategyModel(metric="loss")
        pred = t(1., 2., 3.)
        label = t(1., 2., 3.)
        score = model.metric_fn(pred, label)
        assert abs(score.item()) < 1e-6

    def test_empty_string_same_as_loss(self):
        m_loss = _StubStrategyModel(metric="loss")
        m_empty = _StubStrategyModel(metric="")
        pred = t(1., 2., 3.)
        label = t(4., 5., 6.)
        assert abs(
            m_loss.metric_fn(pred, label).item()
            - m_empty.metric_fn(pred, label).item()
        ) < 1e-6

    def test_mse_same_as_loss(self):
        m_mse = _StubStrategyModel(metric="mse")
        m_loss = _StubStrategyModel(metric="loss")
        pred = t(1., 2., 3.)
        label = t(4., 5., 6.)
        assert abs(
            m_mse.metric_fn(pred, label).item()
            - m_loss.metric_fn(pred, label).item()
        ) < 1e-6

    def test_unknown_metric_raises(self):
        model = _StubStrategyModel(metric="sharpe")
        with pytest.raises(ValueError, match="unknown metric"):
            model.metric_fn(t(1., 2., 3.), t(1., 2., 3.))


# ===========================================================================
# StrategyMetricMixin integration and helper tests
# ===========================================================================

class DummyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Always output [batch, 1]
        return torch.ones(x.shape[0], 1) * self.param


class StubBaseForStrategy:
    def __init__(self):
        self.n_epochs = 2
        self.use_gpu = False
        self.device = torch.device("cpu")
        self.early_stop = 2

    def train_epoch(self, x, y):
        pass

    def test_epoch(self, x, y):
        return 0.1, 0.5

    def loss_fn(self, pred, label, weight=None):
        return torch.mean((pred - label) ** 2)


class DummyStrategyModel(StrategyMetricMixin, StubBaseForStrategy, torch.nn.Module):
    def __init__(self, metric="ir", loss="mse"):
        torch.nn.Module.__init__(self)
        StubBaseForStrategy.__init__(self)
        StrategyMetricMixin.__init__(self, metric=metric)
        self.inner = DummyInner()
        self.loss = loss


class MockDataset:
    def prepare(self, *args, **kwargs):
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

        return df_train, df_valid, None


class ModelWithErrorProperty(StrategyMetricMixin, torch.nn.Module):
    @property
    def bad_attr(self):
        raise ValueError("Simulated error")


class TestStrategyMetricMixinIntegration:
    def test_fit_ir_successful(self, tmp_path):
        import os
        model = DummyStrategyModel(metric="ir")
        dataset = MockDataset()
        save_path = tmp_path / "model.pth"
        evals = {}
        model.fit(dataset, evals_result=evals, save_path=save_path)
        assert "train" in evals
        assert "valid" in evals
        assert len(evals["valid"]) == 2
        assert os.path.exists(save_path)

    def test_fit_delegates_to_parent_for_non_ir(self):
        class StubBaseWithFit(StubBaseForStrategy):
            def fit(self, dataset, evals_result=None, save_path=None):
                return "delegated"

        class DummyStrategyModelWithFit(StrategyMetricMixin, StubBaseWithFit, torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                StubBaseWithFit.__init__(self)
                StrategyMetricMixin.__init__(self, metric="ic")

        model = DummyStrategyModelWithFit()
        res = model.fit(None)
        assert res == "delegated"

    def test_loss_fn_ic(self):
        model = DummyStrategyModel(metric="ic", loss="ic")
        pred = torch.tensor([1.0, 2.0, 3.0])
        label = torch.tensor([1.0, 2.0, 3.0])
        loss = model.loss_fn(pred, label)
        assert loss is not None

    def test_forward_all_dataloader(self):
        model = DummyStrategyModel(metric="ir")
        batch1 = torch.ones(4, 5, 3)
        batch2 = (torch.ones(4, 5, 3), torch.ones(4))
        
        preds, labels, mean_loss = model._forward_all_dataloader([batch1, batch2])
        assert len(preds) == 8
        assert len(labels) == 8
        assert isinstance(mean_loss, float)

    def test_find_inner_modules_exception(self):
        model = ModelWithErrorProperty()
        candidates = StrategyMetricMixin._find_inner_modules(model)
        assert len(candidates) == 0

