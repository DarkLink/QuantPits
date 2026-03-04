import pytest
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from quantpits.scripts.analysis.single_model_analyzer import SingleModelAnalyzer


def _make_pred_df(n_dates=5, n_instruments=20, seed=42):
    """Helper: generate a synthetic prediction DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    instruments = [f"SZ{str(i).zfill(6)}" for i in range(n_instruments)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    scores = rng.standard_normal(len(idx))
    return pd.DataFrame({"score": scores}, index=idx)


def _make_returns_df(n_dates=5, n_instruments=20, seed=99):
    """Helper: generate a synthetic forward returns DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    instruments = [f"SZ{str(i).zfill(6)}" for i in range(n_instruments)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    returns = rng.standard_normal(len(idx)) * 0.02
    return pd.DataFrame({"return_1d": returns}, index=idx)


# ── Init ─────────────────────────────────────────────────────────────────

def test_init_no_score_column():
    df = pd.DataFrame({"price": [1.0]}, index=pd.MultiIndex.from_tuples(
        [("2026-01-01", "SZ000001")], names=["datetime", "instrument"]
    ))
    with pytest.raises(ValueError, match="score"):
        SingleModelAnalyzer(df)


def test_init_dropna():
    pred = _make_pred_df(n_dates=2, n_instruments=3)
    pred.iloc[0, 0] = np.nan
    sma = SingleModelAnalyzer(pred)
    assert sma.pred_df["score"].isna().sum() == 0


# ── calculate_rank_ic ────────────────────────────────────────────────────

def test_calculate_rank_ic():
    pred = _make_pred_df()
    returns = _make_returns_df()
    sma = SingleModelAnalyzer(pred)
    daily_ic, ic_win_rate, icir = sma.calculate_rank_ic(returns)
    assert len(daily_ic) > 0
    assert 0 <= ic_win_rate <= 1
    assert isinstance(icir, float)


def test_calculate_rank_ic_empty():
    pred = _make_pred_df()
    # Returns with non-overlapping instruments
    idx = pd.MultiIndex.from_tuples(
        [("2099-01-01", "XX000001")], names=["datetime", "instrument"]
    )
    returns = pd.DataFrame({"return_1d": [0.01]}, index=idx)
    sma = SingleModelAnalyzer(pred)
    result = sma.calculate_rank_ic(returns)
    # Empty overlap returns (Series, 0.0)
    assert len(result) == 2
    daily_ic, ic_win_rate = result
    assert daily_ic.empty or ic_win_rate == 0.0


# ── calculate_cusum ──────────────────────────────────────────────────────

def test_calculate_cusum():
    pred = _make_pred_df()
    sma = SingleModelAnalyzer(pred)
    series = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    result = sma.calculate_cusum(series)
    assert "CUSUM_POS" in result.columns
    assert "CUSUM_NEG" in result.columns
    assert len(result) == len(series)


def test_calculate_cusum_zero_std():
    pred = _make_pred_df()
    sma = SingleModelAnalyzer(pred)
    series = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = sma.calculate_cusum(series)
    assert (result["CUSUM_POS"] == 0).all()
    assert (result["CUSUM_NEG"] == 0).all()


# ── calculate_psi ────────────────────────────────────────────────────────

def test_calculate_psi():
    pred = _make_pred_df(n_dates=10, n_instruments=50)
    sma = SingleModelAnalyzer(pred)
    dates = pred.index.get_level_values("datetime").unique()
    baseline = dates[:5]
    current = dates[5:]
    psi = sma.calculate_psi(baseline, current)
    assert isinstance(psi, float)
    assert psi >= 0  # PSI is always non-negative


def test_calculate_psi_empty():
    pred = _make_pred_df()
    sma = SingleModelAnalyzer(pred)
    psi = sma.calculate_psi([], pred.index.get_level_values("datetime").unique())
    assert np.isnan(psi)


# ── calculate_quantile_spread ────────────────────────────────────────────

def test_calculate_quantile_spread():
    pred = _make_pred_df()
    returns = _make_returns_df()
    sma = SingleModelAnalyzer(pred)
    spread_df = sma.calculate_quantile_spread(returns)
    assert not spread_df.empty
    assert "Spread" in spread_df.columns
    assert "Top_Ret" in spread_df.columns
    assert "Bottom_Ret" in spread_df.columns


# ── calculate_long_only_ic ───────────────────────────────────────────────

def test_calculate_long_only_ic():
    pred = _make_pred_df()
    returns = _make_returns_df()
    sma = SingleModelAnalyzer(pred)
    daily_ic, ic_mean = sma.calculate_long_only_ic(returns, top_k=5)
    assert len(daily_ic) > 0
    assert isinstance(ic_mean, float)


def test_calculate_long_only_ic_empty():
    pred = _make_pred_df()
    idx = pd.MultiIndex.from_tuples(
        [("2099-01-01", "XX000001")], names=["datetime", "instrument"]
    )
    returns = pd.DataFrame({"return_1d": [0.01]}, index=idx)
    sma = SingleModelAnalyzer(pred)
    daily_ic, ic_mean = sma.calculate_long_only_ic(returns)
    assert daily_ic.empty
    assert ic_mean == 0.0
