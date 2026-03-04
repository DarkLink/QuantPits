import pytest
import pandas as pd
import numpy as np

from quantpits.scripts.analysis.ensemble_analyzer import EnsembleAnalyzer


def _make_model_preds(n_models=3, n_dates=10, n_instruments=30, seed=42):
    """Helper: generate synthetic predictions for multiple models."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    instruments = [f"SZ{str(i).zfill(6)}" for i in range(n_instruments)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    preds = {}
    for i in range(n_models):
        scores = rng.standard_normal(len(idx))
        preds[f"model_{i}"] = pd.DataFrame({"score": scores}, index=idx)
    return preds


def _make_returns(n_dates=10, n_instruments=30, seed=99):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    instruments = [f"SZ{str(i).zfill(6)}" for i in range(n_instruments)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    returns = rng.standard_normal(len(idx)) * 0.02
    return pd.DataFrame({"return_1d": returns}, index=idx)


# ── Signal Correlation ───────────────────────────────────────────────────

def test_signal_correlation():
    preds = _make_model_preds(n_models=3)
    ea = EnsembleAnalyzer(preds)
    corr = ea.calculate_signal_correlation()
    assert not corr.empty
    assert corr.shape == (3, 3)
    # Diagonal should be ~1.0
    for model in preds.keys():
        assert abs(corr.loc[model, model] - 1.0) < 0.05


def test_signal_correlation_single_model():
    preds = _make_model_preds(n_models=1)
    ea = EnsembleAnalyzer(preds)
    corr = ea.calculate_signal_correlation()
    assert corr.empty  # Cannot correlate with only 1 model


# ── Marginal Contribution ───────────────────────────────────────────────

def test_marginal_contribution():
    preds = _make_model_preds(n_models=3)
    returns = _make_returns()
    ea = EnsembleAnalyzer(preds)
    result = ea.calculate_marginal_contribution(returns)
    assert "Full_Ensemble_Sharpe" in result
    assert "Marginal_Contributions" in result
    assert len(result["Marginal_Contributions"]) == 3


def test_marginal_contribution_single_model():
    preds = _make_model_preds(n_models=1)
    returns = _make_returns()
    ea = EnsembleAnalyzer(preds)
    result = ea.calculate_marginal_contribution(returns)
    assert result == {}


# ── OOS vs IS Drift ──────────────────────────────────────────────────────

def test_oos_vs_is_drift():
    preds = _make_model_preds(n_models=2)
    ea = EnsembleAnalyzer(preds)

    rng = np.random.default_rng(42)
    is_returns = pd.Series(rng.standard_normal(100) * 0.01 + 0.002)
    oos_returns = pd.Series(rng.standard_normal(50) * 0.01 + 0.001)

    result = ea.track_oos_vs_is_drift(is_returns, oos_returns)
    assert "IS_Sharpe" in result
    assert "OOS_Sharpe" in result
    assert "Decay_Ratio" in result
    assert "Warning" in result


def test_oos_vs_is_drift_zero_is_sharpe():
    preds = _make_model_preds(n_models=2)
    ea = EnsembleAnalyzer(preds)

    is_returns = pd.Series([0.0] * 10)  # Zero std -> 0 sharpe
    oos_returns = pd.Series([0.01] * 10)
    result = ea.track_oos_vs_is_drift(is_returns, oos_returns)
    assert result["IS_Sharpe"] == 0.0
    assert result["Decay_Ratio"] == 1.0  # When IS=0, default ratio should be 1


# ── Ensemble IC Metrics ──────────────────────────────────────────────────

def test_ensemble_ic_metrics():
    preds = _make_model_preds(n_models=3)
    returns = _make_returns()
    ea = EnsembleAnalyzer(preds)
    result = ea.calculate_ensemble_ic_metrics(returns)
    assert "Rank_IC_Mean" in result
    assert "IC_Win_Rate" in result
    assert "ICIR" in result


def test_ensemble_ic_metrics_empty():
    ea = EnsembleAnalyzer({})
    result = ea.calculate_ensemble_ic_metrics(pd.DataFrame())
    assert result == {}
