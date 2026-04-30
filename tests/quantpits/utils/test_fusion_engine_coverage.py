"""
Supplemental tests targeting uncovered branches in fusion_engine.py.
Coverage targets: lines 63-65 (date not in eval_df index).
"""

import pandas as pd
import numpy as np
from unittest.mock import patch


def test_calculate_weights_dynamic_path():
    """Lines 58-89: dynamic weight calculation path end-to-end.

    Lines 62-65 (date not in eval_df index) are defensive dead code:
    with a pandas MultiIndex, Timestamp scalars match level-0 lookups,
    so `date not in eval_df.index` is always False when dates come
    from get_level_values. This test exercises the full dynamic path
    including rolling sharpe, thresholding, and normalization.
    """
    from quantpits.utils.fusion_engine import calculate_weights

    dates = pd.date_range("2020-01-01", periods=65, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    np.random.seed(42)
    norm_df = pd.DataFrame({
        "M1": np.random.randn(130),
        "M2": np.random.randn(130),
    }, index=idx)

    label_df = pd.DataFrame({"label": np.random.randn(130)}, index=idx)

    with patch("qlib.data.D") as mock_D:
        mock_D.features.return_value = label_df
        with patch("builtins.print"):
            final_weights, static_weights, is_dynamic = calculate_weights(
                norm_df, {}, "dynamic", {"TopK": 1}, {}
            )

    assert is_dynamic is True
    assert static_weights is None
    assert final_weights.shape == (65, 2)
    # First row should be equal weight (shift(1) → NaN → fillna equal)
    assert abs(final_weights.iloc[0, 0] - 0.5) < 0.01


def test_generate_ensemble_signal_zero_std_detection():
    """Line 171-172: final_score.std() == 0 → warning printed."""
    from quantpits.utils.fusion_engine import generate_ensemble_signal

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2020-01-02"]), ["A"]],
        names=["datetime", "instrument"],
    )
    norm_df = pd.DataFrame({"M1": [1.0, 1.0]}, index=idx)
    static_w = {"M1": 1.0}

    with patch("builtins.print") as mock_print:
        signal = generate_ensemble_signal(norm_df, None, static_w, False)

    assert signal.std() == 0
    assert any("加权可能失败" in str(c) for c in mock_print.call_args_list)


def test_calculate_weights_manual_with_config_fallback():
    """Test manual weight path where manual_weights_str is None → config fallback."""
    from quantpits.utils.fusion_engine import calculate_weights

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"]
    )
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0]}, index=idx)
    ensemble_cfg = {"manual_weights": {"M1": 0.7, "M2": 0.3}}

    with patch("builtins.print"):
        _, static, is_dyn = calculate_weights(
            norm_df, {}, "manual", {}, ensemble_cfg, manual_weights_str=None
        )

    assert not is_dyn
    assert static["M1"] == 0.7
    assert static["M2"] == 0.3


def test_calculate_weights_manual_sum_zero_equal_fallback():
    """Lines 121-123: manual weights sum to 0 → equal fallback."""
    from quantpits.utils.fusion_engine import calculate_weights

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"]
    )
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0]}, index=idx)
    ensemble_cfg = {"manual_weights": {"M1": 0.0, "M2": 0.0}}

    with patch("builtins.print"):
        _, static, is_dyn = calculate_weights(
            norm_df, {}, "manual", {}, ensemble_cfg
        )

    assert not is_dyn
    assert static["M1"] == 0.5
    assert static["M2"] == 0.5


def test_calculate_weights_icir_all_invalid_equal_fallback():
    """Lines 96-98: all ICIR below min_ic → equal weights."""
    from quantpits.utils.fusion_engine import calculate_weights

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), ["A"]], names=["datetime", "instrument"]
    )
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0]}, index=idx)
    model_metrics = {"M1": 0.001, "M2": -0.5}

    with patch("builtins.print"):
        _, static, is_dyn = calculate_weights(
            norm_df, model_metrics, "icir_weighted", {}, {"min_model_ic": 0.01}
        )

    assert not is_dyn
    assert static["M1"] == 0.5
    assert static["M2"] == 0.5
