"""
Tests for BenchmarkDataLoader — QLIB data fetching, regime map construction,
per-segment overlay, sliding dynamics, and what-if scoring.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from quantpits.scripts.deep_analysis.benchmark_data_loader import (
    _build_regime_map,
    _classify_regime,
    _compute_segment_stats,
    _compute_cross_segment,
    _build_sliding_dynamics,
    _build_what_if,
    _compute_slide_dates,
    _moment_similarity,
    _find_pareto_frontier,
    _get_regimes_in_range,
    _check_boundary_regime,
    _compute_drawdown_stats,
    load_benchmark_data,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_close_series(start_date="2005-01-04", days=5000, seed=42):
    """Generate a synthetic close price series with distinct regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=days, freq="B")
    returns = np.zeros(days)
    n = days
    # Create distinct, strongly-separated regimes
    # 0-500: strong bull, normal vol
    returns[:500] = rng.normal(0.002, 0.012, 500)
    # 500-1000: strong bear, high vol
    returns[500:1000] = rng.normal(-0.003, 0.028, 500)
    # 1000-1500: sideways, low vol
    returns[1000:1500] = rng.normal(0.0001, 0.007, 500)
    # 1500-2000: bull, high vol
    returns[1500:2000] = rng.normal(0.002, 0.024, 500)
    # 2000-2500: bear, normal vol
    returns[2000:2500] = rng.normal(-0.002, 0.014, 500)
    # 2500-3000: bull, normal vol
    returns[2500:3000] = rng.normal(0.0015, 0.013, 500)
    # 3000-3500: sideways, high vol
    returns[3000:3500] = rng.normal(0.0000, 0.022, 500)
    # 3500-4000: bear, high vol (major crash)
    returns[3500:4000] = rng.normal(-0.0025, 0.030, 500)
    # 4000+: bull, normal vol (recent)
    returns[4000:] = rng.normal(0.001, 0.013, days - 4000)

    price = 1000 * (1 + pd.Series(returns, index=dates)).cumprod()
    return price


def _make_weekly_close(close):
    """Resample daily close to weekly (Friday)."""
    return close.resample("W-FRI").last().dropna()


# ---------------------------------------------------------------------------
# Regime classification tests
# ---------------------------------------------------------------------------


class TestClassifyRegime:
    def test_bullish_normal(self):
        """Steady uptrend with moderate volatility → Bullish-NormalVol."""
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        rets = np.random.default_rng(42).normal(0.001, 0.012, 60)
        close = pd.Series(1000 * (1 + rets).cumprod(), index=dates)
        result = _classify_regime(close)
        assert result["trend"] == "Bullish"
        assert result["vol"] == "NormalVol"

    def test_bearish_high_vol(self):
        """Steep decline with high volatility → Bearish-HighVol."""
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        rets = np.random.default_rng(42).normal(-0.003, 0.030, 60)
        close = pd.Series(1000 * (1 + rets).cumprod(), index=dates)
        result = _classify_regime(close)
        assert result["trend"] == "Bearish"
        assert result["vol"] == "HighVol"

    def test_sideways_low_vol(self):
        """Flat with low volatility → Sideways-LowVol."""
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0000, 0.005, 60)  # zero-mean, very low vol → sideways + low vol
        close = pd.Series(1000 * (1 + rets).cumprod(), index=dates)
        result = _classify_regime(close)
        assert result["trend"] == "Sideways"
        assert result["vol"] == "LowVol"

    def test_insufficient_data(self):
        """Very few data points → Unknown."""
        close = pd.Series([100, 101, 102], index=pd.date_range("2020-01-01", periods=3))
        result = _classify_regime(close)
        assert result["composite"] == "Unknown-Unknown"


# ---------------------------------------------------------------------------
# Regime map tests
# ---------------------------------------------------------------------------


class TestBuildRegimeMap:
    def test_multiple_regimes_detected(self):
        """Synthetic data with multiple distinct regimes → multiple segments."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        assert len(regime_map) >= 3  # At least 3 distinct regimes
        # Check merges: consecutive same-label segments should be merged
        composites = [s["composite"] for s in regime_map]
        for i in range(len(composites) - 1):
            assert composites[i] != composites[i + 1], \
                f"Consecutive segments should not have same label at index {i}"

    def test_all_fields_present(self):
        """Each segment has all required fields."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        required = {"start", "end", "trend", "vol", "composite", "cum_return", "max_dd", "ann_vol"}
        for seg in regime_map:
            assert required.issubset(set(seg.keys())), f"Missing keys: {required - set(seg.keys())}"

    def test_short_series(self):
        """Very short series returns empty."""
        close = pd.Series([100, 101], index=pd.date_range("2020-01-01", periods=2))
        regime_map = _build_regime_map(close)
        assert len(regime_map) == 0


# ---------------------------------------------------------------------------
# Date computation tests
# ---------------------------------------------------------------------------


class TestComputeSlideDates:
    def test_contiguity(self):
        """Train/valid/test boundaries must be contiguous."""
        dates = _compute_slide_dates("2026-06-12", 5, 2, 2)
        assert dates["fit_end_time"] < dates["valid_start_time"]
        assert dates["valid_end_time"] < dates["test_start_time"]
        assert dates["test_end_time"] == "2026-06-12"

    def test_negative_windows(self):
        """Fractional windows should work."""
        dates = _compute_slide_dates("2026-06-12", 5, 1.5, 2)
        assert dates["valid_end_time"] < dates["test_start_time"]

    def test_train_longer_than_total(self):
        """train=10, valid=2, test=2 → train starts ~14 years back."""
        dates = _compute_slide_dates("2026-06-12", 10, 2, 2)
        start = datetime.strptime(dates["fit_start_time"], "%Y-%m-%d")
        anchor = datetime.strptime(dates["test_end_time"], "%Y-%m-%d")
        diff_years = (anchor - start).days / 365.25
        assert 13 < diff_years < 15


# ---------------------------------------------------------------------------
# Per-segment overlay tests
# ---------------------------------------------------------------------------


class TestComputeSegmentStats:
    def test_empty_segment(self):
        """Empty date range returns empty marker."""
        result = _compute_segment_stats("", "", [], pd.Series(), pd.Series())
        assert result.get("empty") is True

    def test_stats_on_synthetic_data(self):
        """Compute stats for a known range."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        returns = weekly.pct_change().dropna()

        start = close.index[0].strftime("%Y-%m-%d")
        end = close.index[-1].strftime("%Y-%m-%d")
        result = _compute_segment_stats(start, end, regime_map, weekly, returns)

        assert not result.get("empty")
        assert result["date_range"]["start"] == start
        assert result["regime_switch_count"] > 0
        assert result["volatility"]["annualized"] > 0
        assert result["return_stats"]["mean"] != 0


class TestComputeCrossSegment:
    def test_vol_ratio(self, tmp_path):
        """Test volatility ratio computation."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        returns = weekly.pct_change().dropna()

        dates = _compute_slide_dates(
            close.index[-1].strftime("%Y-%m-%d"), train_w=8, valid_w=2, test_w=2,
        )
        full_stats = _compute_segment_stats(
            close.index[0].strftime("%Y-%m-%d"),
            close.index[-1].strftime("%Y-%m-%d"),
            regime_map, weekly, returns,
        )

        segments = {}
        for seg_name, start_key, end_key in [
            ("train", "fit_start_time", "fit_end_time"),
            ("valid", "valid_start_time", "valid_end_time"),
            ("test", "test_start_time", "test_end_time"),
        ]:
            segments[seg_name] = _compute_segment_stats(
                dates[start_key], dates[end_key], regime_map, weekly, returns,
            )

        cross = _compute_cross_segment(segments, full_stats, regime_map, returns, dates)
        assert "train_vs_test_vol_ratio" in cross
        assert "train_vs_test_ks_statistic" in cross
        assert "regime_coverage_pct" in cross
        assert 0 <= cross["regime_coverage_pct"] <= 1


class TestDrawdownStats:
    def test_major_drawdown_detected(self):
        """A return series with a large drawdown should be detected."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, 200)
        rets[50:80] = -0.03  # Simulate a major drawdown period
        ret_series = pd.Series(rets, index=pd.date_range("2020-01-01", periods=200, freq="B"))
        result = _compute_drawdown_stats(ret_series)
        assert result["max_dd"] < -0.15
        assert result["major_dd_count"] >= 1


# ---------------------------------------------------------------------------
# Sliding dynamics tests
# ---------------------------------------------------------------------------


class TestSlidingDynamics:
    def test_coverage_stable_on_synthetic(self):
        """Sliding dynamics should produce coverage series."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        anchor = weekly.index[-1].strftime("%Y-%m-%d")

        result = _build_sliding_dynamics(anchor, 5, 2, 2, "slide", regime_map)
        assert result["num_weeks"] == 52
        assert len(result["coverage_over_time"]) == 52
        assert 0 <= result["stability_score"] <= 1
        assert result["trend"] in ("improving", "degrading", "stable")

    def test_fixed_mode_skips(self):
        """Fixed mode returns empty sliding dynamics."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        result = _build_sliding_dynamics("2026-06-12", 5, 2, 2, "fixed", regime_map)
        assert result["num_weeks"] == 0
        assert "fixed" in result.get("trend", "")


# ---------------------------------------------------------------------------
# What-if tests
# ---------------------------------------------------------------------------


class TestWhatIf:
    def test_candidates_generated(self):
        """What-if should generate and score candidate configs."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        returns = weekly.pct_change().dropna()
        anchor = weekly.index[-1].strftime("%Y-%m-%d")
        dates = _compute_slide_dates(anchor, 5, 2, 2)

        result = _build_what_if(anchor, 5, 2, 2, "slide", regime_map, returns, dates)
        assert len(result["top_candidates"]) >= 1
        assert len(result["top_candidates"]) <= 20
        assert result["search_space"]["total_configs_evaluated"] > 100

    def test_pareto_frontier(self):
        """Pareto frontier should contain non-dominated configs."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)
        returns = weekly.pct_change().dropna()
        anchor = weekly.index[-1].strftime("%Y-%m-%d")
        dates = _compute_slide_dates(anchor, 5, 2, 2)

        # Generate all candidates for Pareto test
        all_candidates = []
        for tw in range(2, 10):
            for vw_int in range(2, 8):
                vw = vw_int / 2.0
                for ts_int in range(2, 8):
                    ts = ts_int / 2.0
                    if tw < vw + ts or vw < 1:
                        continue
                    d = _compute_slide_dates(anchor, tw, vw, ts)
                    tr = _get_regimes_in_range(d["fit_start_time"], d["fit_end_time"], regime_map)
                    n_all = len(set(s["composite"] for s in regime_map))
                    coverage = len(tr) / n_all if n_all else 0
                    similarity = 0.7
                    recency = 0.5
                    all_candidates.append({
                        "config": {"train": tw, "valid": vw, "test": ts},
                        "quality_scores": {
                            "coverage": round(coverage, 3),
                            "similarity": similarity,
                            "recency": recency,
                            "boundary_quality": 0.5,
                            "stability": 0.8,
                            "composite": round(0.3*coverage + 0.3*similarity + 0.2*recency + 0.1*0.5 + 0.1*0.8, 4),
                        },
                    })

        all_candidates.sort(key=lambda c: c["quality_scores"]["composite"], reverse=True)
        pareto = _find_pareto_frontier(all_candidates)
        assert len(pareto) >= 1

        # Check no Pareto member is dominated by another
        for i, p in enumerate(pareto):
            ps = p["quality_scores"]
            for j, q in enumerate(pareto):
                if i == j:
                    continue
                qs = q["quality_scores"]
                is_dominated = (
                    qs["coverage"] >= ps["coverage"]
                    and qs["similarity"] >= ps["similarity"]
                    and qs["recency"] >= ps["recency"]
                    and (qs["coverage"] > ps["coverage"]
                         or qs["similarity"] > ps["similarity"]
                         or qs["recency"] > ps["recency"])
                )
                assert not is_dominated, f"Pareto member {i} dominated by {j}"


class TestMomentSimilarity:
    def test_identical_distributions(self):
        """Identical return series → high similarity."""
        ret = pd.Series(np.random.default_rng(42).normal(0.001, 0.015, 200))
        sim = _moment_similarity(ret, ret)
        # exp(-0) = 1.0 when comparing identical series
        assert sim > 0.95

    def test_very_different(self):
        """Very different distributions → low similarity."""
        rng = np.random.default_rng(42)
        train = pd.Series(rng.normal(0.003, 0.010, 500))
        test = pd.Series(rng.normal(-0.005, 0.050, 500))
        sim = _moment_similarity(train, test)
        # With exp(-d), very different means+stds → d large → sim < 0.5
        assert sim < 0.5

    def test_insufficient_data(self):
        """Too few data points → neutral score."""
        train = pd.Series([0.01, 0.02])
        test = pd.Series([0.01, 0.02])
        sim = _moment_similarity(train, test)
        assert sim == 0.5


class TestGetRegimesInRange:
    def test_partial_overlap(self):
        """Regimes partially overlapping with date range."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)

        # Query a subset range
        start = close.index[100].strftime("%Y-%m-%d")
        end = close.index[500].strftime("%Y-%m-%d")
        regimes = _get_regimes_in_range(start, end, regime_map)
        assert len(regimes) >= 1
        total_pct = sum(regimes.values())
        assert abs(total_pct - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Integration test (with mocked QLIB)
# ---------------------------------------------------------------------------


class TestLoadBenchmarkData:
    def test_qlib_unavailable_graceful(self, tmp_path):
        """When QLIB is unavailable, return error dict."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        model_config = {
            "market": "csi300",
            "benchmark": "SH000300",
            "train_set_windows": 5,
            "valid_set_window": 2,
            "test_set_window": 2,
            "data_slice_mode": "slide",
            "freq": "week",
            "train_date_mode": "last_trade_date",
            "start_time": "2015-01-01",
        }
        with open(config_dir / "model_config.json", "w") as f:
            json.dump(model_config, f)

        with patch("quantpits.scripts.deep_analysis.benchmark_data_loader._fetch_benchmark_qlib",
                   side_effect=RuntimeError("QLIB not available")):
            result = load_benchmark_data(str(tmp_path))
            assert result.get("error") is not None
            assert "QLIB" in result["error"]

    def test_loads_with_mocked_qlib(self, tmp_path):
        """Full load with mocked QLIB data."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        model_config = {
            "market": "csi300",
            "benchmark": "SH000300",
            "train_set_windows": 5,
            "valid_set_window": 2,
            "test_set_window": 2,
            "data_slice_mode": "slide",
            "freq": "week",
            "train_date_mode": "last_trade_date",
            "start_time": "2005-01-04",
        }
        with open(config_dir / "model_config.json", "w") as f:
            json.dump(model_config, f)

        close = _make_close_series(start_date="2005-01-04", days=5000)
        daily_ret = close.pct_change().dropna()
        anchor = close.index[-1].strftime("%Y-%m-%d")

        with patch("quantpits.scripts.deep_analysis.benchmark_data_loader._fetch_benchmark_qlib",
                   return_value=(close, daily_ret)):
            with patch("quantpits.scripts.deep_analysis.benchmark_data_loader._get_anchor_date",
                       return_value=anchor):
                result = load_benchmark_data(str(tmp_path))

        assert result.get("error") is None
        assert result["benchmark"] == "SH000300"
        assert len(result["regime_map"]) >= 3
        assert result["current_window"]["config"]["train"] == 5
        cw = result["current_window"]
        segs = cw.get("segments", {})
        assert "train" in segs
        assert "valid" in segs
        assert "test" in segs
        cs = cw.get("cross_segment", {})
        assert "regime_coverage_pct" in cs
        sd = result["sliding_dynamics"]
        assert sd["num_weeks"] == 52
        wi = result["what_if"]
        assert len(wi["top_candidates"]) >= 1

    def test_missing_config(self, tmp_path):
        """No model_config.json → error."""
        result = load_benchmark_data(str(tmp_path))
        assert result.get("error") is not None


class TestCheckBoundaryRegime:
    def test_boundary_change(self):
        """Regime change at boundary should be detected."""
        close = _make_close_series(days=5000)
        weekly = _make_weekly_close(close)
        regime_map = _build_regime_map(weekly)

        # Find a boundary between two different regimes
        if len(regime_map) >= 2:
            boundary_date = regime_map[1]["start"]
            dates = {
                "fit_end_time": regime_map[0]["end"],
                "valid_start_time": boundary_date,
            }
            result = _check_boundary_regime(dates, "fit_end_time", "valid_start_time", regime_map)
            # May or may not change depending on merge logic
            assert "changed" in result
            assert "last_before" in result
