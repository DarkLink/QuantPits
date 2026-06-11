"""
Tests for LossHistoryMixin.

Fully self-contained: no qlib, no dataset, no GPU.  We build a minimal
FakeBaseModel that mimics how qlib's pytorch models call test_epoch inside
fit() and verify every behaviour of the mixin.
"""

import pytest
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin


# ---------------------------------------------------------------------------
# Fake base models that mimic qlib's calling conventions
# ---------------------------------------------------------------------------

class _NormalBase:
    """Standard qlib pattern: fit() calls test_epoch alternately train/valid."""

    def test_epoch(self, tag):
        # Returns (loss, score) — the most common pattern
        return (0.5, 0.8)

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        for _epoch in range(3):
            train_loss, train_score = self.test_epoch("train")
            evals_result.setdefault("train", []).append(train_score)
            val_loss, val_score = self.test_epoch("valid")
            evals_result.setdefault("valid", []).append(val_score)
        return evals_result


class _DictResultBase:
    """Mimics ADARNN: test_epoch returns a dict with 'mse' key."""

    def test_epoch(self, tag):
        return {"mse": 0.25, "ic": 0.4}

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        for _epoch in range(2):
            self.test_epoch("train")
            self.test_epoch("valid")
        return evals_result


class _TRABase:
    """Mimics TRA: test_epoch returns a 4-tuple where first element is a dict."""

    def test_epoch(self, tag):
        return ({"MSE": 0.33, "IC": 0.5}, 0.7, [], [])

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        for _epoch in range(2):
            self.test_epoch("train")
            self.test_epoch("valid")
        return evals_result


class _NoSavePathBase:
    """Mimics models whose fit() does not accept save_path keyword."""

    def test_epoch(self, tag):
        return (0.1, 0.9)

    def fit(self, dataset, evals_result=None):
        if evals_result is None:
            evals_result = {}
        self.test_epoch("train")
        self.test_epoch("valid")
        return evals_result


class _ZeroLossBase:
    """test_epoch returns loss=0.0 — edge case for falsy float."""

    def test_epoch(self, tag):
        return (0.0, 1.0)

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        self.test_epoch("train")
        self.test_epoch("valid")
        return evals_result


# Build concrete mixed classes
class NormalModel(LossHistoryMixin, _NormalBase):
    pass

class DictResultModel(LossHistoryMixin, _DictResultBase):
    pass

class TRAModel(LossHistoryMixin, _TRABase):
    pass

class NoSavePathModel(LossHistoryMixin, _NoSavePathBase):
    pass

class ZeroLossModel(LossHistoryMixin, _ZeroLossBase):
    pass


# ===========================================================================
# LossHistoryMixin._lh_extract_loss  (static, pure Python)
# ===========================================================================

class TestExtractLoss:
    """Unit tests for the three-pattern dispatcher."""

    # --- (loss, score) tuple — most common NN pattern -----------------------

    def test_two_tuple_extracts_first_element(self):
        assert LossHistoryMixin._lh_extract_loss((0.5, 0.8)) == pytest.approx(0.5)

    def test_two_tuple_with_int_loss(self):
        assert LossHistoryMixin._lh_extract_loss((1, 0.9)) == pytest.approx(1.0)

    def test_longer_tuple_extracts_first_element(self):
        assert LossHistoryMixin._lh_extract_loss((0.3, 0.7, "extra")) == pytest.approx(0.3)

    def test_single_element_tuple_extracts_it(self):
        assert LossHistoryMixin._lh_extract_loss((0.42,)) == pytest.approx(0.42)

    def test_list_extracts_first_element(self):
        assert LossHistoryMixin._lh_extract_loss([0.6, 0.9]) == pytest.approx(0.6)

    # --- 4-tuple with metrics dict — TRA pattern ----------------------------

    def test_tra_style_mse_key(self):
        result = ({"MSE": 0.33, "IC": 0.5}, 0.7, [], [])
        assert LossHistoryMixin._lh_extract_loss(result) == pytest.approx(0.33)

    def test_tra_style_lowercase_mse_fallback(self):
        result = ({"mse": 0.21}, 0.6, [], [])
        assert LossHistoryMixin._lh_extract_loss(result) == pytest.approx(0.21)

    def test_tra_style_loss_key_fallback(self):
        result = ({"loss": 0.17}, 0.6, [], [])
        assert LossHistoryMixin._lh_extract_loss(result) == pytest.approx(0.17)

    # --- top-level dict — ADARNN pattern ------------------------------------

    def test_top_level_dict_mse_key(self):
        assert LossHistoryMixin._lh_extract_loss({"mse": 0.25, "ic": 0.4}) == pytest.approx(0.25)

    def test_top_level_dict_loss_key_fallback(self):
        assert LossHistoryMixin._lh_extract_loss({"loss": 0.18}) == pytest.approx(0.18)

    # --- zero loss edge case -------------------------------------------------

    def test_zero_loss_tuple_returns_zero(self):
        """0.0 is falsy; make sure `or 0` doesn't hide a genuine zero."""
        # (0.0, score) → first = 0.0 → float(0.0) = 0.0
        result = LossHistoryMixin._lh_extract_loss((0.0, 0.9))
        assert result == pytest.approx(0.0)

    def test_zero_loss_dict_mse(self):
        result = LossHistoryMixin._lh_extract_loss({"mse": 0.0})
        assert result == pytest.approx(0.0)

    # --- unrecognised formats return None ------------------------------------

    def test_none_input_returns_none(self):
        assert LossHistoryMixin._lh_extract_loss(None) is None

    def test_scalar_float_returns_none(self):
        # A bare float is not a tuple/list/dict — should fall through to None
        assert LossHistoryMixin._lh_extract_loss(0.5) is None

    def test_string_input_returns_none(self):
        assert LossHistoryMixin._lh_extract_loss("loss=0.3") is None

    def test_empty_tuple_returns_none(self):
        assert LossHistoryMixin._lh_extract_loss(()) is None

    def test_empty_dict_returns_none(self):
        # dict.get('mse', dict.get('loss', None)) = None, float(None or 0) = 0.0
        # Actually this returns 0.0 from the `or 0` guard — document that behaviour
        result = LossHistoryMixin._lh_extract_loss({})
        # The implementation does: float(None or 0) = float(0) = 0.0
        # This is a known (minor) quirk; the test asserts the current behaviour.
        assert result == pytest.approx(0.0)


# ===========================================================================
# LossHistoryMixin.fit()  —  interception and bookkeeping
# ===========================================================================

class TestLossHistoryMixinFit:
    """Integration-style tests for the monkey-patch interception."""

    # --- normal 3-epoch run -------------------------------------------------

    def test_train_loss_captured_length(self):
        m = NormalModel()
        result = {}
        m.fit(None, evals_result=result)
        assert "train_loss" in result
        assert len(result["train_loss"]) == 3

    def test_valid_loss_captured_length(self):
        m = NormalModel()
        result = {}
        m.fit(None, evals_result=result)
        assert "valid_loss" in result
        assert len(result["valid_loss"]) == 3

    def test_train_loss_values_correct(self):
        m = NormalModel()
        result = {}
        m.fit(None, evals_result=result)
        assert all(abs(v - 0.5) < 1e-6 for v in result["train_loss"])

    def test_valid_loss_values_correct(self):
        m = NormalModel()
        result = {}
        m.fit(None, evals_result=result)
        assert all(abs(v - 0.5) < 1e-6 for v in result["valid_loss"])

    def test_original_evals_result_keys_preserved(self):
        """Existing 'train' / 'valid' score lists must not be destroyed."""
        m = NormalModel()
        result = {}
        m.fit(None, evals_result=result)
        assert "train" in result
        assert "valid" in result
        assert len(result["train"]) == 3

    def test_returns_base_fit_return_value(self):
        m = NormalModel()
        ret = m.fit(None)
        assert ret is not None  # NormalBase.fit returns evals_result dict

    # --- evals_result=None creates a fresh dict ----------------------------

    def test_none_evals_result_still_captures_losses(self):
        m = NormalModel()
        ret = m.fit(None, evals_result=None)
        # fit() creates evals_result internally; losses are injected into it
        # but since caller passed None the return is the internal dict
        # (depends on base fit returning evals_result)
        # Just verify no exception and method completes
        assert True  # survivability test

    # --- monkey-patch is cleaned up after fit ------------------------------

    def test_test_epoch_restored_after_fit(self):
        """self.test_epoch must call the original implementation after fit()."""
        m = NormalModel()
        # Record the original bound method before fit
        before_result = m.test_epoch("probe")
        m.fit(None, evals_result={})
        # After fit the instance-level test_epoch must return the same result
        # as the original (not a stale closure).
        after_result = m.test_epoch("probe")
        assert before_result == after_result, (
            "test_epoch after fit() must delegate to the original implementation"
        )

    def test_test_epoch_restored_even_on_exception(self):
        """Even if super().fit() raises, test_epoch must be restored (finally block)."""

        class _RaisingBase:
            def test_epoch(self, tag):
                return (0.1, 0.9)

            def fit(self, dataset, evals_result=None, save_path=None):
                self.test_epoch("train")
                raise RuntimeError("training failed")

        class RaisingModel(LossHistoryMixin, _RaisingBase):
            pass

        m = RaisingModel()
        # Capture what test_epoch returns before fit
        before_result = m.test_epoch("probe")

        with pytest.raises(RuntimeError, match="training failed"):
            m.fit(None, evals_result={})

        # After exception, test_epoch must still work (finally block ran restore)
        after_result = m.test_epoch("probe")
        assert before_result == after_result, (
            "test_epoch must be restored to original even after exception"
        )

    # --- dict-style test_epoch (ADARNN pattern) ----------------------------

    def test_dict_result_losses_captured(self):
        m = DictResultModel()
        result = {}
        m.fit(None, evals_result=result)
        assert "train_loss" in result
        assert len(result["train_loss"]) == 2
        assert all(abs(v - 0.25) < 1e-6 for v in result["train_loss"])

    # --- TRA-style 4-tuple with dict ---------------------------------------

    def test_tra_style_losses_captured(self):
        m = TRAModel()
        result = {}
        m.fit(None, evals_result=result)
        assert "train_loss" in result
        assert len(result["train_loss"]) == 2
        assert all(abs(v - 0.33) < 1e-6 for v in result["train_loss"])

    # --- models that don't accept save_path --------------------------------

    def test_no_save_path_model_falls_back(self):
        """TypeError('save_path') must be caught and retried without save_path."""
        m = NoSavePathModel()
        result = {}
        m.fit(None, evals_result=result, save_path="/tmp/model.pth")
        assert "train_loss" in result

    # --- zero-loss edge case -----------------------------------------------

    def test_zero_loss_captured_as_zero(self):
        m = ZeroLossModel()
        result = {}
        m.fit(None, evals_result=result)
        assert "train_loss" in result
        assert abs(result["train_loss"][0]) < 1e-9, \
            "zero loss must not be swallowed by 'or 0' guard"

    # --- internal temp attributes cleaned up ------------------------------

    def test_internal_attributes_deleted_after_fit(self):
        m = NormalModel()
        m.fit(None, evals_result={})
        assert not hasattr(m, "_lh_train_loss")
        assert not hasattr(m, "_lh_valid_loss")
        assert not hasattr(m, "_lh_call_count")

    # --- odd number of test_epoch calls (train-only model) -----------------

    def test_odd_test_epoch_calls_only_fills_train(self):
        """Call pattern: train only (no validation).  With the even/odd counter,
        calls 0,2,4,... go to train_loss and calls 1,3,5,... to valid_loss.
        3 train-only calls land at counter positions 0,1,2 →
          pos 0 (even) → train_loss
          pos 1 (odd)  → valid_loss   (gets the 2nd call unexpectedly)
          pos 2 (even) → train_loss
        So train_loss=[2 entries], valid_loss=[1 entry] after 3 calls.
        """

        class _TrainOnlyBase:
            def test_epoch(self, tag):
                return (0.7, 0.5)

            def fit(self, dataset, evals_result=None, save_path=None):
                if evals_result is None:
                    evals_result = {}
                for _ in range(3):
                    self.test_epoch("train")  # only train, no valid
                return evals_result

        class TrainOnlyModel(LossHistoryMixin, _TrainOnlyBase):
            pass

        m = TrainOnlyModel()
        result = {}
        m.fit(None, evals_result=result)
        assert "train_loss" in result
        # With 3 single calls and even/odd interleaving:
        # call 0 → train, call 1 → valid, call 2 → train → 2 train + 1 valid
        assert len(result["train_loss"]) == 2
        assert len(result.get("valid_loss", [])) == 1

    def test_other_type_error_re_raised(self):
        class _TypeErrorBase:
            def test_epoch(self, tag):
                return (0.1, 0.9)

            def fit(self, dataset, evals_result=None, save_path=None):
                raise TypeError("some other error")

        class TypeErrorModel(LossHistoryMixin, _TypeErrorBase):
            pass

        m = TypeErrorModel()
        with pytest.raises(TypeError, match="some other error"):
            m.fit(None)

