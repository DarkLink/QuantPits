"""
Tests for ICMetricMixin and ICLoss.

These tests are deliberately self-contained: they only depend on torch and
the two mixin/loss classes from quantpits.  No qlib init, no dataset, no GPU
required.

Skipped automatically in environments without torch (e.g. the lightweight
Docker CI image that only installs [test] extras).
"""

import math
import pytest
torch = pytest.importorskip("torch", reason="torch not installed – skipping IC mixin tests")

from quantpits.utils.model_wrappers.mixins.ic import ICLoss, ICMetricMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def t(*values):
    """Shorthand: create a float32 CPU tensor from a list of values."""
    return torch.tensor(values, dtype=torch.float32)


class _StubModel(ICMetricMixin):
    """Minimal stub that satisfies ICMetricMixin's dependencies.

    metric and loss_fn are set per-test so we can exercise every branch of
    metric_fn without touching any real model.
    """

    def __init__(self, metric="ic"):
        self.metric = metric

    def loss_fn(self, pred, label):
        """Simple MSE loss for exercising the 'loss' metric path."""
        return torch.mean((pred - label) ** 2)


# ===========================================================================
# ICLoss
# ===========================================================================

class TestICLoss:
    """Tests for ICLoss (1 - Pearson correlation, differentiable)."""

    def setup_method(self):
        self.criterion = ICLoss()

    # --- basic correctness ---------------------------------------------------

    def test_perfect_positive_correlation_returns_zero(self):
        pred  = t(1.0, 2.0, 3.0, 4.0, 5.0)
        label = t(1.0, 2.0, 3.0, 4.0, 5.0)
        loss = self.criterion(pred, label)
        assert abs(loss.item()) < 1e-5, f"expected ≈0, got {loss.item()}"

    def test_perfect_negative_correlation_returns_two(self):
        pred  = t(1.0, 2.0, 3.0, 4.0, 5.0)
        label = t(5.0, 4.0, 3.0, 2.0, 1.0)
        loss = self.criterion(pred, label)
        assert abs(loss.item() - 2.0) < 1e-5, f"expected ≈2, got {loss.item()}"

    def test_output_in_range_zero_to_two(self):
        torch.manual_seed(0)
        pred  = torch.randn(50)
        label = torch.randn(50)
        loss = self.criterion(pred, label)
        assert 0.0 <= loss.item() <= 2.0 + 1e-6

    # --- numerical edge cases ------------------------------------------------

    def test_constant_pred_does_not_nan(self):
        """All-constant pred → std≈0; epsilon guard must prevent NaN."""
        pred  = t(3.0, 3.0, 3.0, 3.0)
        label = t(1.0, 2.0, 3.0, 4.0)
        loss = self.criterion(pred, label)
        assert math.isfinite(loss.item()), "loss must be finite for constant pred"

    def test_constant_label_does_not_nan(self):
        pred  = t(1.0, 2.0, 3.0, 4.0)
        label = t(5.0, 5.0, 5.0, 5.0)
        loss = self.criterion(pred, label)
        assert math.isfinite(loss.item()), "loss must be finite for constant label"

    def test_both_constant_does_not_nan(self):
        pred  = t(1.0, 1.0, 1.0)
        label = t(2.0, 2.0, 2.0)
        loss = self.criterion(pred, label)
        assert math.isfinite(loss.item())

    def test_nan_in_label_is_masked(self):
        pred  = t(1.0, 2.0, float("nan"), 4.0)
        label = t(1.0, 2.0, 3.0, float("nan"))
        # Only indices 0 and 1 survive the finite mask → at least 2 points
        loss = self.criterion(pred, label)
        assert math.isfinite(loss.item())

    def test_inf_in_pred_is_masked(self):
        pred  = t(1.0, float("inf"), 3.0, 4.0)
        label = t(1.0, 2.0, 3.0, 4.0)
        loss = self.criterion(pred, label)
        assert math.isfinite(loss.item())

    def test_fewer_than_two_valid_points_returns_zero(self):
        pred  = t(float("nan"), float("nan"), 1.0)
        label = t(1.0, float("nan"), float("nan"))
        # After masking (finite both), only 0 points survive
        loss = self.criterion(pred, label)
        assert loss.item() == 0.0

    def test_exactly_two_valid_points_finite(self):
        pred  = t(1.0, 2.0, float("nan"))
        label = t(1.0, 2.0, float("nan"))
        loss = self.criterion(pred, label)
        assert math.isfinite(loss.item())

    # --- gradient flow -------------------------------------------------------

    def test_backward_does_not_error(self):
        """ICLoss must be differentiable for training."""
        pred  = torch.randn(20, requires_grad=True)
        label = torch.randn(20)
        loss = self.criterion(pred, label)
        loss.backward()  # must not raise
        assert pred.grad is not None

    def test_constant_pred_backward_does_not_nan(self):
        """Backward through constant pred (edge case for epsilon guard)."""
        pred  = torch.full((5,), 3.0, requires_grad=True)
        label = t(1.0, 2.0, 3.0, 4.0, 5.0)
        loss = self.criterion(pred, label)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any(), "grad must not be NaN"

    def test_batch_size_1_does_not_error(self):
        pred = torch.tensor([[1.0]], requires_grad=True)
        label = torch.tensor([1.0])
        loss = self.criterion(pred, label)
        assert loss.item() == 0.0
        loss.backward()


# ===========================================================================
# ICMetricMixin._batch_pearson_ic
# ===========================================================================

class TestBatchPearsonIC:
    """Tests for the static helper _batch_pearson_ic."""

    def test_perfect_positive(self):
        ic = ICMetricMixin._batch_pearson_ic(t(1., 2., 3.), t(1., 2., 3.))
        assert abs(ic.item() - 1.0) < 1e-5

    def test_perfect_negative(self):
        ic = ICMetricMixin._batch_pearson_ic(t(1., 2., 3.), t(3., 2., 1.))
        assert abs(ic.item() + 1.0) < 1e-5

    def test_output_bounded(self):
        torch.manual_seed(42)
        p = torch.randn(100)
        l = torch.randn(100)
        ic = ICMetricMixin._batch_pearson_ic(p, l)
        assert -1.0 - 1e-6 <= ic.item() <= 1.0 + 1e-6

    def test_constant_pred_does_not_nan(self):
        p = t(2., 2., 2., 2.)
        l = t(1., 2., 3., 4.)
        ic = ICMetricMixin._batch_pearson_ic(p, l)
        assert math.isfinite(ic.item())

    def test_constant_label_does_not_nan(self):
        p = t(1., 2., 3., 4.)
        l = t(5., 5., 5., 5.)
        ic = ICMetricMixin._batch_pearson_ic(p, l)
        assert math.isfinite(ic.item())

    def test_fewer_than_two_returns_zero_tensor(self):
        p = t(1.0)
        l = t(1.0)
        ic = ICMetricMixin._batch_pearson_ic(p, l)
        assert ic.item() == 0.0

    def test_large_batch(self):
        torch.manual_seed(7)
        n = 10000
        p = torch.randn(n)
        l = p * 0.8 + torch.randn(n) * 0.2  # strong positive correlation
        ic = ICMetricMixin._batch_pearson_ic(p, l)
        assert ic.item() > 0.7, "should detect strong positive correlation"

    def test_orthogonal_vectors_near_zero(self):
        # Construct two genuinely uncorrelated vectors
        torch.manual_seed(123)
        n = 1000
        p = torch.randn(n)
        l = torch.randn(n)
        ic = ICMetricMixin._batch_pearson_ic(p, l)
        assert abs(ic.item()) < 0.15, "random vectors should have low IC"

    def test_returns_tensor(self):
        ic = ICMetricMixin._batch_pearson_ic(t(1., 2., 3.), t(3., 2., 1.))
        assert isinstance(ic, torch.Tensor)


# ===========================================================================
# ICMetricMixin._batch_rank_ic
# ===========================================================================

class TestBatchRankIC:
    """Tests for the static helper _batch_rank_ic (Spearman rank IC)."""

    def test_perfect_rank_correlation(self):
        ric = ICMetricMixin._batch_rank_ic(t(1., 2., 3., 4.), t(10., 20., 30., 40.))
        assert abs(ric.item() - 1.0) < 1e-5

    def test_perfect_anti_correlation(self):
        ric = ICMetricMixin._batch_rank_ic(t(1., 2., 3., 4.), t(40., 30., 20., 10.))
        assert abs(ric.item() + 1.0) < 1e-5

    def test_fewer_than_two_returns_zero(self):
        ric = ICMetricMixin._batch_rank_ic(t(5.0), t(5.0))
        assert ric.item() == 0.0

    def test_output_bounded(self):
        torch.manual_seed(99)
        p = torch.randn(200)
        l = torch.randn(200)
        ric = ICMetricMixin._batch_rank_ic(p, l)
        assert -1.0 - 1e-6 <= ric.item() <= 1.0 + 1e-6

    def test_monotone_transform_preserves_rank_ic(self):
        """Applying a monotone transform to label should not change rank IC."""
        p = t(1., 3., 2., 5., 4.)
        l = t(2., 4., 3., 5., 1.)
        ric_original = ICMetricMixin._batch_rank_ic(p, l)
        ric_transformed = ICMetricMixin._batch_rank_ic(p, l ** 3)
        assert abs(ric_original.item() - ric_transformed.item()) < 1e-4

    def test_different_from_pearson_for_nonlinear(self):
        """Rank IC should differ from Pearson IC for non-linear relationships."""
        p = t(1., 2., 3., 4., 5.)
        l = t(1., 4., 9., 16., 25.)  # quadratic: same rank, different values
        pearson = ICMetricMixin._batch_pearson_ic(p, l)
        spearman = ICMetricMixin._batch_rank_ic(p, l)
        # Spearman should be 1.0 (perfect rank), Pearson < 1.0
        assert abs(spearman.item() - 1.0) < 1e-4
        assert pearson.item() < 0.999

    def test_returns_tensor(self):
        ric = ICMetricMixin._batch_rank_ic(t(1., 2., 3.), t(3., 2., 1.))
        assert isinstance(ric, torch.Tensor)


# ===========================================================================
# ICMetricMixin.metric_fn (full dispatch logic)
# ===========================================================================

class TestICMetricMixinMetricFn:
    """Tests for metric_fn — exercises every dispatch branch."""

    # --- ic ------------------------------------------------------------------

    def test_ic_metric_returns_pearson_ic(self):
        model = _StubModel(metric="ic")
        pred  = t(1., 2., 3., 4., 5.)
        label = t(1., 2., 3., 4., 5.)
        score = model.metric_fn(pred, label)
        assert abs(score.item() - 1.0) < 1e-5

    def test_ic_metric_masks_nan(self):
        model = _StubModel(metric="ic")
        # metric_fn masks only isfinite(label); pred values at valid label
        # positions are kept even if they are NaN.
        # Construct a case where all surviving pred values are finite:
        #   pred=[1, 2, nan, 4]  label=[1, nan, 3, 4]
        #   isfinite(label) mask = [T, F, T, T] → pred after mask = [1, nan, 4]
        # Because pred[2]=nan, Pearson IC will return NaN in that case.
        # Test the case where only finite pred values survive:
        pred  = t(1., 2., 3., 4.)
        label = t(1., float("nan"), float("nan"), 4.)
        # mask = [T, F, F, T] → pred=[1,4], label=[1,4] → IC = 1.0
        score = model.metric_fn(pred, label)
        assert math.isfinite(score.item())
        assert abs(score.item() - 1.0) < 1e-5

    # --- rank_ic -------------------------------------------------------------

    def test_rank_ic_metric_returns_spearman(self):
        model = _StubModel(metric="rank_ic")
        pred  = t(1., 2., 3., 4., 5.)
        label = t(1., 2., 3., 4., 5.)
        score = model.metric_fn(pred, label)
        assert abs(score.item() - 1.0) < 1e-5

    def test_rank_ic_masks_nan(self):
        model = _StubModel(metric="rank_ic")
        pred  = t(1., float("nan"), 3., 4.)
        label = t(1., 2., float("nan"), 4.)
        score = model.metric_fn(pred, label)
        assert math.isfinite(score.item())

    # --- loss / "" -----------------------------------------------------------

    def test_loss_metric_returns_negative_loss(self):
        model = _StubModel(metric="loss")
        pred  = t(1., 2., 3.)
        label = t(1., 2., 3.)
        # MSE(pred, label) = 0 → metric_fn should return -0 = 0
        score = model.metric_fn(pred, label)
        assert abs(score.item()) < 1e-6

    def test_empty_string_metric_same_as_loss(self):
        model_loss  = _StubModel(metric="loss")
        model_empty = _StubModel(metric="")
        pred  = t(1., 2., 3.)
        label = t(4., 5., 6.)
        assert abs(
            model_loss.metric_fn(pred, label).item()
            - model_empty.metric_fn(pred, label).item()
        ) < 1e-6

    def test_mse_metric_behaves_like_loss(self):
        model_mse  = _StubModel(metric="mse")
        model_loss = _StubModel(metric="loss")
        pred  = t(1., 2., 3.)
        label = t(4., 5., 6.)
        assert abs(
            model_mse.metric_fn(pred, label).item()
            - model_loss.metric_fn(pred, label).item()
        ) < 1e-6

    def test_loss_metric_with_mse_mismatch(self):
        model = _StubModel(metric="loss")
        pred  = t(0., 0., 0.)
        label = t(1., 1., 1.)
        # MSE = 1.0, so metric = -1.0
        score = model.metric_fn(pred, label)
        assert abs(score.item() + 1.0) < 1e-5

    # --- unknown metric raises -----------------------------------------------

    def test_unknown_metric_raises_value_error(self):
        model = _StubModel(metric="r2")
        with pytest.raises(ValueError, match="unknown metric"):
            model.metric_fn(t(1., 2., 3.), t(1., 2., 3.))

    # --- nan masking in metric_fn applies before dispatch --------------------

    def test_all_nan_label_produces_finite_score(self):
        """If all labels are NaN the mask is empty; should not crash."""
        model = _StubModel(metric="ic")
        pred  = t(1., 2., 3.)
        label = t(float("nan"), float("nan"), float("nan"))
        # After mask: 0 valid points → _batch_pearson_ic returns 0
        score = model.metric_fn(pred, label)
        assert math.isfinite(score.item())


# ===========================================================================
# GeneralPTNN sign inversion and squeeze fix
# ===========================================================================

class TestGeneralPTNNMetricFn:
    """Validate the sign-inversion + squeeze fix in pytorch_general_nn_ic."""

    @pytest.fixture
    def model(self):
        """Import lazily so the test is skipped if qlib is unavailable."""
        try:
            from quantpits.utils.model_wrappers.custom.pytorch_general_nn import GeneralPTNN
        except ImportError as exc:
            pytest.skip(f"qlib not available: {exc}")
        return GeneralPTNN.__new__(GeneralPTNN)

    def test_ic_metric_is_negated(self, model):
        """GeneralPTNN.metric_fn must negate the raw IC (lower = better)."""
        model.metric = "ic"

        def _mse(p, l):
            return torch.mean((p - l) ** 2)

        model.loss_fn = _mse

        pred  = t(1., 2., 3., 4., 5.)
        label = t(1., 2., 3., 4., 5.)
        # Raw IC = 1.0 → after negation = -1.0
        score = model.metric_fn(pred, label)
        assert score.item() < 0, "negated IC should be negative for positive correlation"
        assert abs(score.item() + 1.0) < 1e-4

    def test_column_vector_pred_no_broadcasting(self, model):
        """[N,1] pred must be squeezed before Pearson computation."""
        model.metric = "ic"

        def _mse(p, l):
            return torch.mean((p - l) ** 2)

        model.loss_fn = _mse

        pred_2d = t(1., 2., 3., 4., 5.).unsqueeze(1)  # shape [5, 1]
        label   = t(1., 2., 3., 4., 5.)

        # Without squeeze this would create a [5,5] matrix via broadcasting
        score = model.metric_fn(pred_2d, label)
        assert math.isfinite(score.item()), "squeeze fix must prevent [N,N] broadcast"
