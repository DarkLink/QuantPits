"""
Benchmark Data Loader — fetches CSI300/SH000300 data from QLIB and
pre-computes market regime analytics for training window analysis.

All computation is deterministic (no LLM). Produces a structured dict
consumed by TrainingWindowAnalyzer and WindowCritic.

Three pillars:
  1. Per-segment overlay: train/valid/test market characteristics
  2. Sliding dynamics: 52-week coverage simulation + cliff edge detection
  3. What-if scoring: enumerate candidate configs → quality scores → Pareto frontier
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QLIB_DATA_START = "2005-01-04"  # Earliest available QLIB data

# Regime classification thresholds (calibrated for Chinese A-share market)
TREND_BULLISH_THRESHOLD = 0.02   # period return > +2%  → Bullish
TREND_BEARISH_THRESHOLD = -0.02  # period return < -2%  → Bearish
VOL_HIGH_THRESHOLD = 0.30        # ann_vol > 30%        → High-Vol (CSI300 baseline ~20-25%)
VOL_LOW_THRESHOLD = 0.15         # ann_vol < 15%        → Low-Vol
REGIME_WINDOW_DAYS = 60          # rolling window for regime detection
REGIME_STEP_DAYS = 20            # step between regime windows

# Recency scoring
RECENCY_MAX_GAP_YEARS = 15.0     # train_end age divisor for recency penalty

# Sliding dynamics
SLIDING_WEEKS = 52

# What-if search bounds
WHATIF_TRAIN_MIN, WHATIF_TRAIN_MAX = 2, 20
WHATIF_VALID_MIN, WHATIF_VALID_MAX = 1, 6
WHATIF_TEST_MIN, WHATIF_TEST_MAX = 1, 8
WHATIF_TOP_N = 20

# Drawdown
MAJOR_DD_THRESHOLD = -0.15

# Quality score weights — similarity and recency dominate because:
# - coverage is non-differentiating when all configs capture all regime types
# - real-world experiments (CSI300_Base CHANGELOG) show older data hurts,
#   not helps — similarity to CURRENT market is what matters
# - boundary and stability are tiebreakers
WEIGHT_COVERAGE = 0.10
WEIGHT_SIMILARITY = 0.45
WEIGHT_RECENCY = 0.25
WEIGHT_BOUNDARY = 0.10
WEIGHT_STABILITY = 0.10

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_benchmark_data(workspace_root: str) -> dict:
    """Load benchmark data and compute all market regime analytics.

    Args:
        workspace_root: Path to the workspace directory.

    Returns:
        A dict with keys: benchmark, fetch_range, freq, error,
        regime_map, current_window, sliding_dynamics, what_if.
        If ``error`` is set, all other analytics fields are empty/None.
    """
    config = _load_model_config(workspace_root)
    if not config:
        return {"error": "model_config.json not found or unreadable", "benchmark": "?"}

    benchmark = config.get("benchmark", "SH000300")
    freq = config.get("freq", "week").lower()

    # --- Step 1: Compute date boundaries ---
    try:
        anchor_date = _get_anchor_date(config)
    except Exception as e:
        return {"error": f"Failed to resolve anchor date: {e}", "benchmark": benchmark}

    mode = config.get("data_slice_mode", "slide")
    current_train = config.get("train_set_windows", 5)
    current_valid = config.get("valid_set_window", 2)
    current_test = config.get("test_set_window", 2)

    if mode == "slide":
        current_dates = _compute_slide_dates(anchor_date, current_train, current_valid, current_test)
    else:
        current_dates = _read_fixed_dates(config)

    # --- Step 2: Fetch benchmark data from QLIB ---
    # Use earliest available QLIB data for richer regime history;
    # model_config start_time may be narrower (e.g. 2015) which limits
    # what-if analysis that needs to see older regimes.
    fetch_start = QLIB_DATA_START
    fetch_end = anchor_date

    try:
        close, daily_returns = _fetch_benchmark_qlib(benchmark, fetch_start, fetch_end)
    except Exception as e:
        logger.warning("Failed to fetch benchmark data from QLIB: %s", e)
        return {"error": f"QLIB data fetch failed: {e}", "benchmark": benchmark,
                "fetch_range": {"start": fetch_start, "end": fetch_end}, "freq": freq}

    if close is None or len(close) < REGIME_WINDOW_DAYS:
        return {"error": f"Insufficient benchmark data ({len(close) if close is not None else 0} obs)",
                "benchmark": benchmark, "fetch_range": {"start": fetch_start, "end": fetch_end}, "freq": freq}

    # Resample to week if needed
    if freq == "week":
        weekly_close = close.resample("W-FRI").last().dropna()
        weekly_returns = weekly_close.pct_change().dropna()
    else:
        weekly_close = close
        weekly_returns = daily_returns

    actual_start = close.index[0].strftime("%Y-%m-%d") if hasattr(close.index[0], 'strftime') else str(close.index[0])[:10]
    actual_end = close.index[-1].strftime("%Y-%m-%d") if hasattr(close.index[-1], 'strftime') else str(close.index[-1])[:10]

    # --- Step 3: Build regime map ---
    regime_map = _build_regime_map(weekly_close)

    # --- Step 4: Per-segment overlay ---
    current_window = _build_current_window(
        current_dates, current_train, current_valid, current_test,
        mode, regime_map, weekly_close, weekly_returns,
    )

    # --- Step 5: Sliding dynamics ---
    sliding_dynamics = _build_sliding_dynamics(
        anchor_date, current_train, current_valid, current_test,
        mode, regime_map,
    )

    # --- Step 6: What-if scoring ---
    what_if = _build_what_if(
        anchor_date, current_train, current_valid, current_test,
        mode, regime_map, weekly_returns, current_dates,
    )

    return {
        "benchmark": benchmark,
        "fetch_range": {"start": actual_start, "end": actual_end},
        "freq": freq,
        "error": None,
        "regime_map": regime_map,
        "regime_summary": _build_regime_summary(regime_map),
        "current_window": current_window,
        "sliding_dynamics": sliding_dynamics,
        "what_if": what_if,
    }


# ---------------------------------------------------------------------------
# Step 1: Config & dates
# ---------------------------------------------------------------------------

def _load_model_config(workspace_root: str) -> dict:
    path = os.path.join(workspace_root, "config", "model_config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load model_config.json: %s", e)
        return {}


def _get_anchor_date(config: dict) -> str:
    """Resolve the anchor date. Prefer QLIB calendar, fall back to config."""
    train_date_mode = config.get("train_date_mode", "last_trade_date")
    if train_date_mode == "last_trade_date":
        try:
            from qlib.data import D
            last_td = D.calendar(future=False)[-1:][0]
            return last_td.strftime("%Y-%m-%d")
        except Exception:
            pass
    return config.get("current_date", datetime.now().strftime("%Y-%m-%d"))


def _compute_slide_dates(anchor_date: str, train_w: float, valid_w: float,
                         test_w: float) -> dict:
    """Replicate train_utils.compute_slide_dates() logic."""
    from quantpits.utils.constants import AVERAGE_CALENDAR_DAYS_PER_YEAR

    def _add_year(input_date: str, n: float) -> Tuple[str, str]:
        dt = datetime.strptime(input_date, "%Y-%m-%d")
        delta = int(n * AVERAGE_CALENDAR_DAYS_PER_YEAR)
        target = dt + timedelta(days=delta)
        next_day = target + timedelta(days=1)
        return target.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d")

    total = train_w + valid_w + test_w
    _, start_time = _add_year(anchor_date, -total)
    fit_end, valid_start = _add_year(anchor_date, -(valid_w + test_w))
    valid_end, test_start = _add_year(anchor_date, -test_w)

    return {
        "start_time": start_time, "fit_start_time": start_time,
        "fit_end_time": fit_end, "valid_start_time": valid_start,
        "valid_end_time": valid_end, "test_start_time": test_start,
        "test_end_time": anchor_date,
    }


def _read_fixed_dates(config: dict) -> dict:
    return {
        "start_time": config.get("start_time", "2008-01-01"),
        "fit_start_time": config.get("fit_start_time", "2008-01-01"),
        "fit_end_time": config.get("fit_end_time", ""),
        "valid_start_time": config.get("valid_start_time", ""),
        "valid_end_time": config.get("valid_end_time", ""),
        "test_start_time": config.get("test_start_time", ""),
        "test_end_time": config.get("test_end_time", ""),
    }


# ---------------------------------------------------------------------------
# Step 2: QLIB data fetch
# ---------------------------------------------------------------------------

def _fetch_benchmark_qlib(benchmark: str, start: str, end: str) -> Tuple[pd.Series, pd.Series]:
    """Fetch benchmark close prices from QLIB. Returns (close, daily_returns)."""
    from qlib.data import D
    from quantpits.utils.env import init_qlib

    init_qlib()

    bench_df = D.features(
        [benchmark],
        ["$close"],
        start_time=start,
        end_time=end,
    )
    raw_close = bench_df["$close"]

    # Handle multi-index: (instrument, datetime) → extract datetime level
    if hasattr(raw_close.index, "get_level_values"):
        raw_close = raw_close.droplevel(0) if raw_close.index.nlevels > 1 else raw_close
    raw_close = raw_close.sort_index()
    raw_close = raw_close.astype(float)

    daily_ret = raw_close.pct_change().dropna()

    return raw_close, daily_ret


# ---------------------------------------------------------------------------
# Step 3: Regime map
# ---------------------------------------------------------------------------

def _classify_regime(close_segment: pd.Series) -> dict:
    """Classify a single segment of close prices into trend + vol labels."""
    if len(close_segment) < 5:
        return {"trend": "Unknown", "vol": "Unknown", "composite": "Unknown-Unknown",
                "cum_return": 0.0, "max_dd": 0.0, "ann_vol": 0.0}

    start_val = close_segment.iloc[0]
    end_val = close_segment.iloc[-1]
    period_return = (end_val / start_val - 1) if start_val != 0 else 0.0

    # Trend
    if period_return > TREND_BULLISH_THRESHOLD:
        trend = "Bullish"
    elif period_return < TREND_BEARISH_THRESHOLD:
        trend = "Bearish"
    else:
        trend = "Sideways"

    # Volatility
    rets = close_segment.pct_change().dropna()
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) >= 5 else 0.0
    if ann_vol > VOL_HIGH_THRESHOLD:
        vol = "HighVol"
    elif ann_vol < VOL_LOW_THRESHOLD:
        vol = "LowVol"
    else:
        vol = "NormalVol"

    # Drawdown
    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    dd = (cum / rolling_max - 1) if not rolling_max.empty else pd.Series([0.0])
    max_dd = float(dd.min()) if not dd.empty else 0.0

    return {
        "trend": trend, "vol": vol,
        "composite": f"{trend}-{vol}",
        "cum_return": float(period_return),
        "max_dd": max_dd,
        "ann_vol": ann_vol,
    }


def _build_regime_map(close: pd.Series) -> List[dict]:
    """Build a regime map by sliding a window over the full price history.

    Returns a list of regime segments, each with start/end dates and labels.
    Consecutive windows with the same composite label are merged.
    """
    segments: List[dict] = []
    if len(close) < REGIME_WINDOW_DAYS:
        return segments

    # Compute regime for each sliding window position
    raw_labels: List[dict] = []
    for i in range(0, len(close) - REGIME_WINDOW_DAYS + 1, REGIME_STEP_DAYS):
        segment = close.iloc[i:i + REGIME_WINDOW_DAYS]
        label = _classify_regime(segment)
        label["start"] = segment.index[0]
        label["end"] = segment.index[-1]
        raw_labels.append(label)

    if not raw_labels:
        return segments

    # Merge consecutive windows with the same composite label
    current = dict(raw_labels[0])
    for next_label in raw_labels[1:]:
        if next_label["composite"] == current["composite"]:
            current["end"] = next_label["end"]
            # Update max_dd to the worse of the two
            current["max_dd"] = min(current["max_dd"], next_label["max_dd"])
            current["cum_return"] = next_label["cum_return"]
            current["ann_vol"] = (current["ann_vol"] + next_label["ann_vol"]) / 2
        else:
            _finalize_regime_segment(current)
            segments.append(current)
            current = dict(next_label)
    _finalize_regime_segment(current)
    segments.append(current)

    return segments


def _finalize_regime_segment(seg: dict) -> None:
    """Convert Timestamp fields to strings for JSON serialization."""
    for key in ("start", "end"):
        if key in seg and hasattr(seg[key], "strftime"):
            seg[key] = seg[key].strftime("%Y-%m-%d")


def _build_regime_summary(regime_map: List[dict]) -> dict:
    """Count unique regimes and their frequencies."""
    counts: Dict[str, int] = {}
    for seg in regime_map:
        c = seg.get("composite", "?")
        counts[c] = counts.get(c, 0) + 1
    return {
        "unique_regimes": sorted(counts.keys()),
        "regime_counts": counts,
        "total_segments": len(regime_map),
    }


# ---------------------------------------------------------------------------
# Step 4: Per-segment overlay
# ---------------------------------------------------------------------------

def _build_current_window(dates: dict, train_w: float, valid_w: float,
                          test_w: float, mode: str, regime_map: List[dict],
                          close: pd.Series, returns: pd.Series) -> dict:
    """Compute per-segment market stats for the current window config."""
    segments = {}
    for seg_name, start_key, end_key in [
        ("train", "fit_start_time", "fit_end_time"),
        ("valid", "valid_start_time", "valid_end_time"),
        ("test", "test_start_time", "test_end_time"),
    ]:
        s, e = dates.get(start_key, ""), dates.get(end_key, "")
        segments[seg_name] = _compute_segment_stats(s, e, regime_map, close, returns)

    # Full-history stats
    full_stats = _compute_segment_stats(
        close.index[0].strftime("%Y-%m-%d") if hasattr(close.index[0], 'strftime') else str(close.index[0])[:10],
        close.index[-1].strftime("%Y-%m-%d") if hasattr(close.index[-1], 'strftime') else str(close.index[-1])[:10],
        regime_map, close, returns,
    )

    # Cross-segment comparisons
    cross = _compute_cross_segment(segments, full_stats, regime_map, returns, dates)

    return {
        "config": {"train": train_w, "valid": valid_w, "test": test_w, "mode": mode,
                   "anchor_date": dates.get("test_end_time", "")},
        "segments": segments,
        "full_history": full_stats,
        "cross_segment": cross,
    }


def _compute_segment_stats(start: str, end: str, regime_map: List[dict],
                           close: pd.Series, returns: pd.Series) -> dict:
    """Compute all market stats for a single date range segment."""
    if not start or not end:
        return {"date_range": {"start": start, "end": end}, "empty": True}

    # Regime coverage
    regimes: Dict[str, float] = {}
    total_days = 0
    seg_start = pd.Timestamp(start)
    seg_end = pd.Timestamp(end)
    for seg in regime_map:
        rs = pd.Timestamp(seg["start"])
        re = pd.Timestamp(seg["end"])
        overlap_start = max(rs, seg_start)
        overlap_end = min(re, seg_end)
        if overlap_start <= overlap_end:
            days = (overlap_end - overlap_start).days + 1
            label = seg.get("composite", "?")
            regimes[label] = regimes.get(label, 0) + days
            total_days += days

    regime_list = []
    if total_days > 0:
        regime_list = [{"label": k, "pct": round(v / total_days, 3)}
                       for k, v in sorted(regimes.items(), key=lambda x: -x[1])]

    # Regime switch count within segment
    switch_count = _count_regime_switches_in_range(start, end, regime_map)

    # Return stats
    ret_slice = pd.Series(dtype=float)
    try:
        mask = (returns.index >= seg_start) & (returns.index <= seg_end)
        ret_slice = returns.loc[mask].dropna()
    except Exception:
        pass

    return_stats = {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0}
    if len(ret_slice) >= 5:
        return_stats = {
            "mean": float(ret_slice.mean()),
            "std": float(ret_slice.std()),
            "skew": float(ret_slice.skew()) if not (np.isnan(ret_slice.skew())) else 0.0,
            "kurt": float(ret_slice.kurtosis()) if not (np.isnan(ret_slice.kurtosis())) else 0.0,
        }

    # Volatility
    ann_vol = float(ret_slice.std() * np.sqrt(252)) if len(ret_slice) >= 5 else 0.0
    vol_label = "HighVol" if ann_vol > VOL_HIGH_THRESHOLD else ("LowVol" if ann_vol < VOL_LOW_THRESHOLD else "NormalVol")

    # Drawdown stats
    dd_stats = _compute_drawdown_stats(ret_slice)

    # Cumulative return
    cum_ret = 0.0
    try:
        seg_close = close[(close.index >= seg_start) & (close.index <= seg_end)]
        if len(seg_close) >= 2:
            cum_ret = float(seg_close.iloc[-1] / seg_close.iloc[0] - 1)
    except Exception:
        pass

    return {
        "date_range": {"start": start, "end": end},
        "empty": len(ret_slice) < 5,
        "num_observations": len(ret_slice),
        "regimes": regime_list,
        "regime_switch_count": switch_count,
        "volatility": {"annualized": round(ann_vol, 4), "label": vol_label},
        "return_stats": return_stats,
        "drawdown_stats": dd_stats,
        "cum_return": round(cum_ret, 4),
    }


def _count_regime_switches_in_range(start: str, end: str, regime_map: List[dict]) -> int:
    """Count how many regime transitions occur within a date range."""
    seg_start = pd.Timestamp(start)
    seg_end = pd.Timestamp(end)
    prev_label = None
    switches = 0
    for seg in regime_map:
        rs = pd.Timestamp(seg["start"])
        if rs > seg_end:
            break
        re = pd.Timestamp(seg["end"])
        if re < seg_start:
            continue
        label = seg.get("composite", "")
        if prev_label is not None and label != prev_label:
            switches += 1
        prev_label = label
    return switches


def _compute_drawdown_stats(returns: pd.Series) -> dict:
    """Compute drawdown statistics for a return series."""
    if len(returns) < 5:
        return {"max_dd": 0.0, "avg_dd": 0.0, "major_dd_count": 0, "major_dds": []}

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd_series = cum / rolling_max - 1

    max_dd = float(dd_series.min())
    avg_dd = float(dd_series.mean())

    # Count major drawdown events (contiguous periods below MAJOR_DD_THRESHOLD)
    major_dds: List[dict] = []
    in_dd = False
    dd_start = None
    dd_peak_depth = 0.0
    for idx, val in dd_series.items():
        if val < MAJOR_DD_THRESHOLD and not in_dd:
            in_dd = True
            dd_start = idx
            dd_peak_depth = val
        elif in_dd:
            if val < dd_peak_depth:
                dd_peak_depth = val
            if val >= -0.02:  # Recovery threshold: DD back within 2%
                major_dds.append({
                    "start": dd_start.strftime("%Y-%m-%d") if hasattr(dd_start, 'strftime') else str(dd_start)[:10],
                    "end": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10],
                    "depth": round(float(dd_peak_depth), 3),
                })
                in_dd = False
    if in_dd and dd_start is not None:
        last_idx = dd_series.index[-1]
        major_dds.append({
            "start": dd_start.strftime("%Y-%m-%d") if hasattr(dd_start, 'strftime') else str(dd_start)[:10],
            "end": last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, 'strftime') else str(last_idx)[:10],
            "depth": round(float(dd_peak_depth), 3),
        })

    return {
        "max_dd": round(max_dd, 4),
        "avg_dd": round(avg_dd, 4),
        "major_dd_count": len(major_dds),
        "major_dds": major_dds,
    }


def _compute_cross_segment(segments: dict, full_stats: dict,
                           regime_map: List[dict], returns: pd.Series,
                           dates: dict) -> dict:
    """Compute cross-segment comparisons."""
    train_seg = segments.get("train", {})
    test_seg = segments.get("test", {})
    valid_seg = segments.get("valid", {})

    # Volatility ratio
    train_vol = train_seg.get("volatility", {}).get("annualized", 0.0)
    test_vol = test_seg.get("volatility", {}).get("annualized", 0.0)
    vol_ratio = (test_vol / train_vol) if train_vol > 0 else 1.0

    # KS statistic — train vs test returns
    train_ret = _slice_returns(returns, dates, "fit_start_time", "fit_end_time")
    test_ret = _slice_returns(returns, dates, "test_start_time", "test_end_time")
    ks_stat = _compute_ks(train_ret, test_ret)

    # Regime coverage
    all_regimes = set()
    for seg in regime_map:
        all_regimes.add(seg.get("composite", ""))
    train_regimes = set()
    for r in train_seg.get("regimes", []):
        train_regimes.add(r.get("label", ""))
    coverage_pct = len(train_regimes) / len(all_regimes) if all_regimes else 0.0
    missing_regimes = sorted(all_regimes - train_regimes)

    # Boundary regime check
    train_to_valid = _check_boundary_regime(dates, "fit_end_time", "valid_start_time", regime_map)
    valid_to_test = _check_boundary_regime(dates, "valid_end_time", "test_start_time", regime_map)

    return {
        "train_vs_test_vol_ratio": round(vol_ratio, 3),
        "train_vs_test_ks_statistic": round(ks_stat, 4),
        "train_to_valid_boundary": train_to_valid,
        "valid_to_test_boundary": valid_to_test,
        "regime_coverage_pct": round(coverage_pct, 3),
        "missing_regimes": missing_regimes,
    }


def _slice_returns(returns: pd.Series, dates: dict,
                   start_key: str, end_key: str) -> pd.Series:
    """Slice return series for a given date range from dates dict."""
    s, e = dates.get(start_key, ""), dates.get(end_key, "")
    if not s or not e:
        return pd.Series(dtype=float)
    try:
        mask = (returns.index >= pd.Timestamp(s)) & (returns.index <= pd.Timestamp(e))
        return returns.loc[mask].dropna()
    except Exception:
        return pd.Series(dtype=float)


def _compute_ks(a: pd.Series, b: pd.Series) -> float:
    """Two-sample KS statistic (0 = identical, 1 = completely different)."""
    from scipy import stats as scipy_stats
    if len(a) < 10 or len(b) < 10:
        return 0.0
    try:
        ks, _ = scipy_stats.ks_2samp(a.values, b.values)
        return float(ks)
    except Exception:
        return 0.0


def _check_boundary_regime(dates: dict, before_key: str, after_key: str,
                           regime_map: List[dict]) -> dict:
    """Check if the regime label changes across a date boundary."""
    before_date = dates.get(before_key, "")
    after_date = dates.get(after_key, "")

    def _find_regime_at(d: str) -> str:
        if not d:
            return ""
        dt = pd.Timestamp(d)
        for seg in regime_map:
            if pd.Timestamp(seg["start"]) <= dt <= pd.Timestamp(seg["end"]):
                return seg.get("composite", "")
        return ""

    last_before = _find_regime_at(before_date)
    first_after = _find_regime_at(after_date)
    return {
        "last_before": last_before, "first_after": first_after,
        "changed": last_before != first_after or (not last_before),
    }


# ---------------------------------------------------------------------------
# Step 5: Sliding dynamics
# ---------------------------------------------------------------------------

def _build_sliding_dynamics(anchor_date: str, train_w: float, valid_w: float,
                            test_w: float, mode: str,
                            regime_map: List[dict]) -> dict:
    """Simulate 52 weeks of sliding windows and track regime coverage."""
    if mode != "slide":
        return {"num_weeks": 0, "coverage_over_time": [], "cliff_edges": [],
                "stability_score": 1.0, "trend": "n/a (fixed mode)"}

    anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d")

    # Get all unique regime labels
    all_regimes = set()
    for seg in regime_map:
        all_regimes.add(seg.get("composite", ""))
    n_all = len(all_regimes)

    coverage_series: List[dict] = []
    prev_regime_count = None
    cliff_edges: List[dict] = []

    for week_offset in range(SLIDING_WEEKS):
        past_anchor = anchor_dt - timedelta(days=7 * week_offset)
        past_anchor_str = past_anchor.strftime("%Y-%m-%d")
        dates = _compute_slide_dates(past_anchor_str, train_w, valid_w, test_w)

        train_regimes = _get_regimes_in_range(
            dates["fit_start_time"], dates["fit_end_time"], regime_map,
        )
        coverage_pct = len(train_regimes) / n_all if n_all > 0 else 0.0

        coverage_series.append({
            "anchor": past_anchor_str,
            "regime_count": len(train_regimes),
            "coverage_pct": round(coverage_pct, 3),
            "dominant_regime": max(train_regimes, key=train_regimes.get) if train_regimes else "?",
        })

        # Detect cliff edge: regime count dropped
        if prev_regime_count is not None and len(train_regimes) < prev_regime_count:
            lost = set()
            # Find what was lost by looking at the previous week's regimes
            prev_dates = _compute_slide_dates(
                (anchor_dt - timedelta(days=7 * (week_offset - 1))).strftime("%Y-%m-%d"),
                train_w, valid_w, test_w,
            )
            prev_regimes = _get_regimes_in_range(
                prev_dates["fit_start_time"], prev_dates["fit_end_time"], regime_map,
            )
            lost = set(prev_regimes.keys()) - set(train_regimes.keys())
            for lost_regime in lost:
                # Check if this is within the "warning window" (4 weeks from now)
                cliff_edges.append({
                    "anchor": past_anchor_str,
                    "lost_regime": lost_regime,
                    "weeks_until": week_offset,  # 0 = now, positive = weeks ago
                })

        prev_regime_count = len(train_regimes)

    # Stability score: 1 - normalized std of coverage
    coverages = [c["coverage_pct"] for c in coverage_series]
    std_cov = float(np.std(coverages)) if len(coverages) > 1 else 0.0
    stability = round(1.0 - min(std_cov / 0.3, 1.0), 3)  # std=0.3 → stability=0

    # Trend: is coverage improving or degrading?
    if len(coverages) >= 8:
        first_half = np.mean(coverages[:len(coverages)//2])
        second_half = np.mean(coverages[len(coverages)//2:])
        diff = second_half - first_half
        if diff > 0.02:
            trend = "improving"
        elif diff < -0.02:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return {
        "num_weeks": SLIDING_WEEKS,
        "coverage_over_time": coverage_series,
        "cliff_edges": cliff_edges[:10],  # keep top 10
        "stability_score": stability,
        "trend": trend,
    }


def _get_regimes_in_range(start: str, end: str, regime_map: List[dict]) -> Dict[str, float]:
    """Get regime labels and their fraction within a date range."""
    regimes: Dict[str, float] = {}
    total = 0
    rs_dt = pd.Timestamp(start) if start else None
    re_dt = pd.Timestamp(end) if end else None
    if rs_dt is None or re_dt is None:
        return regimes
    for seg in regime_map:
        ss = pd.Timestamp(seg["start"])
        se = pd.Timestamp(seg["end"])
        overlap = max(ss, rs_dt)
        overlap_end = min(se, re_dt)
        if overlap <= overlap_end:
            days = (overlap_end - overlap).days + 1
            label = seg.get("composite", "?")
            regimes[label] = regimes.get(label, 0) + days
            total += days
    # Normalize to fractions
    if total > 0:
        regimes = {k: round(v / total, 3) for k, v in regimes.items()}
    return regimes


# ---------------------------------------------------------------------------
# Step 6: What-if scoring
# ---------------------------------------------------------------------------

def _build_what_if(anchor_date: str, current_train: float, current_valid: float,
                   current_test: float, mode: str, regime_map: List[dict],
                   returns: pd.Series, current_dates: dict) -> dict:
    """Enumerate and score candidate window configs."""
    if mode != "slide":
        return {"top_candidates": [], "pareto_frontier": [], "search_space": {}}

    all_regimes = set()
    for seg in regime_map:
        all_regimes.add(seg.get("composite", ""))
    n_all = len(all_regimes)

    anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d")

    candidates: List[dict] = []
    for train_w in range(WHATIF_TRAIN_MIN, WHATIF_TRAIN_MAX + 1):
        for valid_w_int in range(WHATIF_VALID_MIN * 2, WHATIF_VALID_MAX * 2 + 1):
            valid_w = valid_w_int / 2.0  # 0.5 increments: 1.0, 1.5, 2.0, ...
            for test_w_int in range(WHATIF_TEST_MIN * 2, WHATIF_TEST_MAX * 2 + 1):
                test_w = test_w_int / 2.0

                # Constraints
                if train_w < valid_w + test_w:
                    continue
                if valid_w < 1:
                    continue

                dates = _compute_slide_dates(anchor_date, train_w, valid_w, test_w)
                train_regimes = _get_regimes_in_range(
                    dates["fit_start_time"], dates["fit_end_time"], regime_map,
                )

                # Coverage score: % of full-history regime types in training
                coverage = len(train_regimes) / n_all if n_all > 0 else 0.0

                # Similarity: train vs candidate's own test window.
                # Scaled by test_window_adequacy to penalise short tests.
                train_ret = _slice_returns(returns, dates, "fit_start_time", "fit_end_time")
                test_ret = _slice_returns(returns, dates, "test_start_time", "test_end_time")
                raw_similarity = _moment_similarity(train_ret, test_ret)
                test_adequacy = min(test_w / 2.0, 1.0)
                similarity = round(raw_similarity * test_adequacy, 3)

                # Regime relevance: how well does the training regime
                # distribution match the test period distribution?
                # Uses Jensen-Shannon distance — penalises training
                # windows dominated by regimes unlike the current market.
                test_regimes = _get_regimes_in_range(
                    dates["test_start_time"], dates["test_end_time"], regime_map,
                )
                regime_relevance = _regime_distribution_similarity(
                    train_regimes, test_regimes,
                )

                # Recency: penalise both ends of the training window.
                # train_end_age: gap from end of training to now
                # train_start_age: how far back does data go
                # Composite: geometric mean — punishes extremes on either end.
                train_end = datetime.strptime(dates["fit_end_time"], "%Y-%m-%d")
                train_start = datetime.strptime(dates["fit_start_time"], "%Y-%m-%d")
                train_end_age = (anchor_dt - train_end).days / 365.25
                train_start_age = (anchor_dt - train_start).days / 365.25
                end_recency = 1.0 - min(train_end_age / RECENCY_MAX_GAP_YEARS, 1.0)
                start_freshness = 1.0 - min(train_start_age / (RECENCY_MAX_GAP_YEARS * 1.5), 1.0)
                recency = round(np.sqrt(end_recency * start_freshness), 3)

                # Boundary quality: penalty if near a regime transition
                boundary_quality = _compute_boundary_quality(dates, regime_map)

                # Stability: Jaccard similarity of regimes across nearby anchors
                stability = _approximate_stability(
                    anchor_date, train_w, valid_w, test_w, regime_map, n_all,
                )

                composite = round(
                    WEIGHT_COVERAGE * coverage +
                    WEIGHT_SIMILARITY * similarity +
                    WEIGHT_RECENCY * recency +
                    WEIGHT_BOUNDARY * boundary_quality +
                    WEIGHT_STABILITY * stability +
                    0.10 * regime_relevance,  # bonus for regime-aligned training
                    4,
                )

                # Mark if this is the current config
                is_current = (train_w == current_train and valid_w == current_valid
                              and test_w == current_test)

                # New regimes gained vs current
                current_train_regimes = _get_regimes_in_range(
                    current_dates["fit_start_time"], current_dates["fit_end_time"], regime_map,
                )
                new_regimes = sorted(set(train_regimes.keys()) - set(current_train_regimes.keys()))

                candidates.append({
                    "config": {"train": train_w, "valid": valid_w, "test": test_w,
                               "train_date_range": {
                                   "start": dates["fit_start_time"],
                                   "end": dates["fit_end_time"],
                               }},
                    "quality_scores": {
                        "coverage": round(coverage, 3),
                        "similarity": round(similarity, 3),
                        "recency": recency,
                        "boundary_quality": round(boundary_quality, 3),
                        "stability": round(stability, 3),
                        "regime_relevance": round(regime_relevance, 3),
                        "composite": composite,
                    },
                    "is_current": is_current,
                    "new_regimes_vs_current": new_regimes,
                })

    # Sort by composite score
    candidates.sort(key=lambda c: c["quality_scores"]["composite"], reverse=True)

    # Find Pareto frontier (non-dominated in coverage × similarity × recency)
    pareto = _find_pareto_frontier(candidates)

    return {
        "top_candidates": candidates[:WHATIF_TOP_N],
        "pareto_frontier": pareto[:10],
        "search_space": {
            "train_range": [WHATIF_TRAIN_MIN, WHATIF_TRAIN_MAX],
            "valid_range": [WHATIF_VALID_MIN, WHATIF_VALID_MAX],
            "test_range": [WHATIF_TEST_MIN, WHATIF_TEST_MAX],
            "total_configs_evaluated": len(candidates),
        },
    }


def _moment_similarity(train_ret: pd.Series, test_ret: pd.Series) -> float:
    """Compute similarity using exponential decay on moment distance.

    Uses mean, std, skew to compare train vs test return distributions.
    Returns a score in [0, 1] where 1 = identical.
    exp(-d) gives sharper differentiation than 1/(1+d):
      d=0.2 → 0.82, d=0.5 → 0.61, d=1.0 → 0.37, d=2.0 → 0.14
    """
    if len(train_ret) < 10 or len(test_ret) < 10:
        return 0.5

    t_mean, t_std = float(train_ret.mean()), float(train_ret.std())
    e_mean, e_std = float(test_ret.mean()), float(test_ret.std())

    if e_std == 0:
        return 0.5

    # Normalized differences in mean and std
    d_mean = abs(t_mean - e_mean) / e_std
    d_std = abs(t_std - e_std) / e_std

    # Add skewness difference if both have enough data
    d_skew = 0.0
    if len(train_ret) >= 30 and len(test_ret) >= 30:
        t_skew = float(train_ret.skew())
        e_skew = float(test_ret.skew())
        if not (np.isnan(t_skew) or np.isnan(e_skew)):
            d_skew = abs(t_skew - e_skew) / max(abs(e_skew), 0.5)

    d = np.sqrt(d_mean ** 2 + d_std ** 2 + 0.5 * d_skew ** 2)
    return round(float(np.exp(-d)), 3)


def _compute_boundary_quality(dates: dict, regime_map: List[dict]) -> float:
    """Score how well window boundaries align with regime transitions.

    Penalises boundaries that fall near a regime transition (within 60 days).
    A boundary far from any transition = high quality.
    Returns score in [0, 1].
    """
    penalty = 0.0

    for before_key, after_key in [
        ("fit_end_time", "valid_start_time"),
        ("valid_end_time", "test_start_time"),
    ]:
        boundary_str = dates.get(after_key, "")
        if not boundary_str:
            penalty += 0.5
            continue
        try:
            boundary_dt = pd.Timestamp(boundary_str)
        except Exception:
            penalty += 0.5
            continue

        # Find the nearest regime transition to this boundary
        min_gap_days = float("inf")
        for seg in regime_map:
            seg_start = pd.Timestamp(seg["start"])
            seg_end = pd.Timestamp(seg["end"])
            # Check gap to segment start (beginning of a regime = transition in)
            gap = abs((boundary_dt - seg_start).days)
            min_gap_days = min(min_gap_days, gap)
            # Check gap to segment end (end of a regime = transition out)
            gap = abs((boundary_dt - seg_end).days)
            min_gap_days = min(min_gap_days, gap)

        # Within 30 days of a transition → heavy penalty
        # 30-90 days → light penalty
        # >90 days → no penalty
        if min_gap_days < 30:
            penalty += 0.5
        elif min_gap_days < 90:
            penalty += 0.25

    return round(1.0 - min(penalty, 1.0), 3)


def _approximate_stability(anchor_date: str, train_w: float, valid_w: float,
                           test_w: float, regime_map: List[dict],
                           n_all: int, num_samples: int = 12) -> float:
    """Stability via Jaccard similarity of regime sets across nearby anchors.

    Higher = regime composition in training is consistent week-over-week.
    Uses a mix of regime identity (which regimes) and proportion similarity.
    """
    anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d")
    regime_sets: List[Dict[str, float]] = []
    for i in range(num_samples):
        past = anchor_dt - timedelta(days=7 * i * 2)  # every 2 weeks
        dates = _compute_slide_dates(past.strftime("%Y-%m-%d"), train_w, valid_w, test_w)
        regimes = _get_regimes_in_range(
            dates["fit_start_time"], dates["fit_end_time"], regime_map,
        )
        regime_sets.append(regimes)

    if len(regime_sets) < 2:
        return 1.0

    # Compute pairwise Jaccard + proportion similarity, average
    similarities = []
    for i in range(len(regime_sets) - 1):
        a, b = regime_sets[i], regime_sets[i + 1]
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        if not keys_a and not keys_b:
            similarities.append(1.0)
            continue
        if not keys_a or not keys_b:
            similarities.append(0.0)
            continue
        # Jaccard on regime labels
        jaccard = len(keys_a & keys_b) / len(keys_a | keys_b)
        # Proportion similarity on shared labels (cosine-like)
        shared = keys_a & keys_b
        if shared:
            prop_sim = sum(min(a[k], b[k]) for k in shared) / sum(max(a[k], b[k]) for k in shared)
        else:
            prop_sim = 0.0
        similarities.append(0.6 * jaccard + 0.4 * prop_sim)

    avg_sim = float(np.mean(similarities)) if similarities else 1.0
    return round(avg_sim, 3)


def _regime_distribution_similarity(train_regimes: Dict[str, float],
                                     test_regimes: Dict[str, float]) -> float:
    """Compute Jensen-Shannon-like similarity between regime distributions.

    Returns score in [0, 1] where 1 = identical distributions.
    A training window dominated by regimes absent from the test period
    gets a low score — penalising inclusion of irrelevant old data.
    """
    if not train_regimes or not test_regimes:
        return 0.5

    all_labels = sorted(set(list(train_regimes.keys()) + list(test_regimes.keys())))
    if len(all_labels) <= 1:
        return 1.0

    p = np.array([train_regimes.get(k, 0.0) for k in all_labels])
    q = np.array([test_regimes.get(k, 0.0) for k in all_labels])

    # Normalise
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    # Jensen-Shannon distance
    m = (p + q) / 2

    def _kl(a, b):
        # Kullback-Leibler divergence, avoiding log(0)
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    js_div = (_kl(p, m) + _kl(q, m)) / 2

    # Convert to similarity: exp(-js_div) maps [0, inf] → [1, 0]
    # Typical JS values: 0 (identical) to ~0.7 (very different)
    similarity = float(np.exp(-js_div * 3))  # scale factor 3 to spread values
    return round(similarity, 3)


def _find_pareto_frontier(candidates: List[dict]) -> List[dict]:
    """Find Pareto-optimal candidates across (coverage, similarity, recency)."""
    pareto = []
    for c in candidates:
        scores = c["quality_scores"]
        dominated = False
        for other in pareto:
            os = other["quality_scores"]
            if (os["coverage"] >= scores["coverage"]
                    and os["similarity"] >= scores["similarity"]
                    and os["recency"] >= scores["recency"]):
                # At least one strictly better
                if (os["coverage"] > scores["coverage"]
                        or os["similarity"] > scores["similarity"]
                        or os["recency"] > scores["recency"]):
                    dominated = True
                    break
        if not dominated:
            # Remove any existing that this one dominates
            pareto = [p for p in pareto if not (
                scores["coverage"] >= p["quality_scores"]["coverage"]
                and scores["similarity"] >= p["quality_scores"]["similarity"]
                and scores["recency"] >= p["quality_scores"]["recency"]
                and (scores["coverage"] > p["quality_scores"]["coverage"]
                     or scores["similarity"] > p["quality_scores"]["similarity"]
                     or scores["recency"] > p["quality_scores"]["recency"])
            )]
            pareto.append(c)

    # Add a "strength" label for LLM context
    for p in pareto:
        scores = p["quality_scores"]
        strengths = []
        if scores["coverage"] >= 0.7:
            strengths.append("highest coverage")
        if scores["similarity"] >= 0.75:
            strengths.append("best similarity")
        if scores["recency"] >= 0.80:
            strengths.append("most recent data")
        if scores["boundary_quality"] >= 0.75:
            strengths.append("clean boundaries")
        p["strength"] = ", ".join(strengths) if strengths else "balanced"

    return pareto
