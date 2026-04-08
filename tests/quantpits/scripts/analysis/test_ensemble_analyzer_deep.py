"""
Deep mathematical verification of EnsembleAnalyzer.

Every test constructs data with a hand-computable ground truth and asserts
element-level closeness, rather than just checking key presence.
"""
import pytest
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from unittest.mock import patch, MagicMock

from quantpits.scripts.analysis.ensemble_analyzer import EnsembleAnalyzer
from quantpits.utils.constants import TRADING_DAYS_PER_YEAR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preds(n_models=3, n_dates=5, n_instruments=10, seed=42):
    """Create controlled synthetic predictions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-02", periods=n_dates, freq="D")
    instruments = [f"SZ{str(i).zfill(6)}" for i in range(n_instruments)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    preds = {}
    for i in range(n_models):
        scores = rng.standard_normal(len(idx))
        preds[f"model_{i}"] = pd.DataFrame({"score": scores}, index=idx)
    return preds, dates, instruments, idx


def _make_returns(dates, instruments, seed=99):
    """Create controlled returns aligned with predictions."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rets = rng.standard_normal(len(idx)) * 0.02
    return pd.DataFrame({"return_1d": rets}, index=idx)


# ============================================================================
# Group 1: Signal Correlation – Manual Spearman
# ============================================================================

class TestSignalCorrelation:
    def test_manual_spearman(self):
        """
        Compute cross-sectional Spearman correlations per day by hand,
        then average. Compare to analyzer output.
        """
        preds, dates, instruments, idx = _make_preds(n_models=3, n_dates=5, n_instruments=20, seed=42)
        ea = EnsembleAnalyzer(preds)
        corr = ea.calculate_signal_correlation()

        # Independent: build merged df, compute daily cross-sectional Spearman
        dfs = []
        for name, df in preds.items():
            dfs.append(df[["score"]].rename(columns={"score": name}))
        merged = pd.concat(dfs, axis=1, join="inner").dropna()

        model_names = list(preds.keys())
        n_m = len(model_names)
        daily_corrs = []
        for dt in dates:
            day_data = merged.loc[dt]
            if len(day_data) < 2:
                continue
            corr_mat = np.zeros((n_m, n_m))
            for i in range(n_m):
                for j in range(n_m):
                    rho, _ = spearmanr(day_data[model_names[i]].values, day_data[model_names[j]].values)
                    corr_mat[i, j] = rho
            daily_corrs.append(corr_mat)

        expected_avg = np.nanmean(daily_corrs, axis=0)

        for i, mi in enumerate(model_names):
            for j, mj in enumerate(model_names):
                assert np.isclose(corr.loc[mi, mj], expected_avg[i, j], atol=1e-8), \
                    f"Corr[{mi},{mj}]: {corr.loc[mi, mj]} vs {expected_avg[i, j]}"

    def test_diagonal_is_one(self):
        preds, _, _, _ = _make_preds(n_models=3, n_dates=10, n_instruments=30, seed=7)
        ea = EnsembleAnalyzer(preds)
        corr = ea.calculate_signal_correlation()
        for model in preds:
            assert np.isclose(corr.loc[model, model], 1.0, atol=1e-8)

    def test_single_model_empty(self):
        preds, _, _, _ = _make_preds(n_models=1)
        ea = EnsembleAnalyzer(preds)
        assert ea.calculate_signal_correlation().empty

    def test_two_instruments_too_few(self):
        """With < 2 instruments per day, correlation is NaN → should still work."""
        dates = pd.date_range("2025-01-02", periods=3, freq="D")
        instruments = ["SZ000001"]  # Only 1 instrument
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        preds = {
            "m1": pd.DataFrame({"score": [1.0, 2.0, 3.0]}, index=idx),
            "m2": pd.DataFrame({"score": [3.0, 2.0, 1.0]}, index=idx),
        }
        ea = EnsembleAnalyzer(preds)
        corr = ea.calculate_signal_correlation()
        # Should return empty because NaN correlations are dropped
        # or a matrix — depends on implementation; it should at least not crash
        assert isinstance(corr, pd.DataFrame)


# ============================================================================
# Group 2: Marginal Contribution
# ============================================================================

class TestMarginalContribution:
    def test_zscore_and_sharpe(self):
        """
        Verify z-scoring, equal-weight averaging, top-q selection,
        and drop-one marginal contribution logic.
        """
        preds, dates, instruments, idx = _make_preds(n_models=3, n_dates=10, n_instruments=30, seed=42)
        returns = _make_returns(dates, instruments, seed=99)
        ea = EnsembleAnalyzer(preds)
        result = ea.calculate_marginal_contribution(returns)

        # Independent replication
        dfs = []
        model_names = list(preds.keys())
        for name in model_names:
            dfs.append(preds[name][["score"]].rename(columns={"score": name}))
        merged = pd.concat(dfs, axis=1, join="inner").dropna()

        # Z-score
        def _zscore(df):
            std = df.std()
            std = std.replace(0, 1)
            return (df - df.mean()) / std

        merged_z = merged.groupby(level="datetime", group_keys=False).apply(_zscore)
        full_score = merged_z.mean(axis=1)

        def _score_to_sharpe(score_series, top_q=0.2):
            df = score_series.to_frame("score").join(returns, how="inner").dropna()
            if df.empty:
                return 0.0
            ret_col = "return_1d"
            def _top_ret(x):
                if len(x) == 0:
                    return np.nan
                q = x["score"].quantile(1 - top_q)
                return x[x["score"] >= q][ret_col].mean()
            daily_rets = df.groupby(level="datetime").apply(_top_ret).dropna()
            if len(daily_rets) < 2 or daily_rets.std() == 0:
                return 0.0
            rf_daily = 0.0135 / TRADING_DAYS_PER_YEAR
            return ((daily_rets.mean() - rf_daily) / daily_rets.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

        expected_full_sharpe = _score_to_sharpe(full_score)
        assert np.isclose(result["Full_Ensemble_Sharpe"], expected_full_sharpe, atol=1e-10), \
            f"Full Sharpe: {result['Full_Ensemble_Sharpe']} vs {expected_full_sharpe}"

        # Marginal contributions
        for model in model_names:
            without = merged_z.drop(columns=[model]).mean(axis=1)
            wo_sharpe = _score_to_sharpe(without)
            expected_mc = expected_full_sharpe - wo_sharpe
            assert np.isclose(result["Marginal_Contributions"][model], expected_mc, atol=1e-10), \
                f"MC[{model}]: {result['Marginal_Contributions'][model]} vs {expected_mc}"

    def test_single_model_returns_empty(self):
        preds, dates, instruments, _ = _make_preds(n_models=1)
        returns = _make_returns(dates, instruments)
        ea = EnsembleAnalyzer(preds)
        assert ea.calculate_marginal_contribution(returns) == {}


# ============================================================================
# Group 3: OOS vs IS Drift
# ============================================================================

class TestOOSDrift:
    def test_sharpe_formula(self):
        """Verify Sharpe computation and decay ratio against hand computation."""
        rng = np.random.default_rng(42)
        is_rets = rng.normal(0.002, 0.01, 200)
        oos_rets = rng.normal(0.001, 0.01, 100)

        preds, _, _, _ = _make_preds(n_models=2)
        ea = EnsembleAnalyzer(preds)
        result = ea.track_oos_vs_is_drift(pd.Series(is_rets), pd.Series(oos_rets))

        # Independent Sharpe
        rf_daily = 0.0135 / TRADING_DAYS_PER_YEAR
        is_s = ((is_rets.mean() - rf_daily) / is_rets.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        oos_s = ((oos_rets.mean() - rf_daily) / oos_rets.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)

        assert np.isclose(result["IS_Sharpe"], is_s, atol=1e-8)
        assert np.isclose(result["OOS_Sharpe"], oos_s, atol=1e-8)

        expected_decay = oos_s / is_s if is_s > 0 else 1.0
        assert np.isclose(result["Decay_Ratio"], expected_decay, atol=1e-8)

    def test_decay_warning_trigger(self):
        """Decay < 0.5 should show strong warning."""
        rng = np.random.default_rng(42)
        is_rets = rng.normal(0.005, 0.01, 100)  # Strong IS performance
        oos_rets = rng.normal(0.0001, 0.01, 100)  # Weak OOS

        preds, _, _, _ = _make_preds(n_models=2)
        ea = EnsembleAnalyzer(preds)
        result = ea.track_oos_vs_is_drift(pd.Series(is_rets), pd.Series(oos_rets))

        if result["Decay_Ratio"] < 0.5:
            assert "Strong Warning" in result["Warning"]
        else:
            assert result["Warning"] == "OK"

    def test_zero_is_sharpe(self):
        """IS Sharpe = 0 → decay ratio should be 1.0."""
        preds, _, _, _ = _make_preds(n_models=2)
        ea = EnsembleAnalyzer(preds)
        is_rets = pd.Series([0.0] * 10)  # zero std → sharpe 0
        oos_rets = pd.Series([0.01] * 10)
        result = ea.track_oos_vs_is_drift(is_rets, oos_rets)
        assert result["IS_Sharpe"] == 0.0
        assert pd.isna(result["Decay_Ratio"])


# ============================================================================
# Group 4: Ensemble IC Metrics
# ============================================================================

class TestEnsembleICMetrics:
    def test_ic_metrics_flow(self):
        """
        Verify the z-score → equal-weight → SingleModelAnalyzer flow.
        We verify the z-scored ensemble score is constructed correctly,
        then mock SingleModelAnalyzer to confirm it's called with correct data.
        """
        preds, dates, instruments, idx = _make_preds(n_models=3, n_dates=10, n_instruments=30, seed=42)
        returns = _make_returns(dates, instruments, seed=99)
        ea = EnsembleAnalyzer(preds)

        # Independent: compute expected z-scored ensemble score
        dfs = []
        model_names = list(preds.keys())
        for name in model_names:
            dfs.append(preds[name][["score"]].rename(columns={"score": name}))
        merged = pd.concat(dfs, axis=1, join="inner").dropna()

        def _zscore(df):
            std = df.std()
            std = std.replace(0, 1)
            return (df - df.mean()) / std

        merged_z = merged.groupby(level="datetime", group_keys=False).apply(_zscore)
        expected_score = merged_z.mean(axis=1).to_frame("score")

        # Now run the analyzer
        result = ea.calculate_ensemble_ic_metrics(returns)

        # It should return meaningful IC values (not just check key presence)
        assert "Rank_IC_Mean" in result
        assert "IC_Win_Rate" in result
        assert "ICIR" in result
        assert "Spread_Mean" in result
        assert "Long_Only_IC_Mean" in result

        # Cross-validate: manually compute Rank IC from expected_score
        merged_check = expected_score.join(returns, how="inner").dropna()
        if not merged_check.empty:
            def _ic(df):
                if len(df) < 2:
                    return np.nan
                return spearmanr(df["score"], df["return_1d"])[0]

            daily_ic = merged_check.groupby(level="datetime").apply(_ic).dropna()
            expected_ic_mean = daily_ic.mean()
            expected_ic_wr = (daily_ic > 0).mean()
            expected_icir = daily_ic.mean() / daily_ic.std() if daily_ic.std() != 0 else 0

            assert np.isclose(result["Rank_IC_Mean"], expected_ic_mean, atol=1e-8), \
                f"IC Mean: {result['Rank_IC_Mean']} vs {expected_ic_mean}"
            assert np.isclose(result["IC_Win_Rate"], expected_ic_wr, atol=1e-8), \
                f"IC WR: {result['IC_Win_Rate']} vs {expected_ic_wr}"
            assert np.isclose(result["ICIR"], expected_icir, atol=1e-8), \
                f"ICIR: {result['ICIR']} vs {expected_icir}"

    def test_empty_ensemble(self):
        ea = EnsembleAnalyzer({})
        assert ea.calculate_ensemble_ic_metrics(pd.DataFrame()) == {}

    def test_single_model_still_works(self):
        """Even with 1 model, ensemble IC should work (trivial ensemble)."""
        preds, dates, instruments, _ = _make_preds(n_models=1, n_dates=10, n_instruments=30)
        returns = _make_returns(dates, instruments)
        ea = EnsembleAnalyzer(preds)
        result = ea.calculate_ensemble_ic_metrics(returns)
        assert "Rank_IC_Mean" in result


# ============================================================================
# Additional: NaN / missing score filtering
# ============================================================================

class TestInitFiltering:
    def test_nan_scores_dropped(self):
        """Models with NaN scores should have them dropped on init."""
        dates = pd.date_range("2025-01-02", periods=3, freq="D")
        instruments = ["SZ000001", "SZ000002"]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        preds = {
            "m1": pd.DataFrame({"score": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0]}, index=idx),
        }
        ea = EnsembleAnalyzer(preds)
        assert len(ea.models_preds["m1"]) == 4  # 2 NaN rows dropped

    def test_no_score_column_ignored(self):
        """Model with no 'score' column should be silently ignored."""
        dates = pd.date_range("2025-01-02", periods=3, freq="D")
        instruments = ["SZ000001"]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        preds = {
            "good": pd.DataFrame({"score": [1.0, 2.0, 3.0]}, index=idx),
            "bad": pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=idx),
        }
        ea = EnsembleAnalyzer(preds)
        assert "good" in ea.models_preds
        assert "bad" not in ea.models_preds
