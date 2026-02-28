#!/usr/bin/env python3
"""
Trade Classifier Engine — 实盘交易信号归因

将每笔买入/卖出交易分为三类：
  S  SIGNAL      — 标的在 suggestion 文件中，排名 ≤ 实际操作股票数 N
  A  SUBSTITUTE  — 标的在 suggestion 文件中，排名 > N（替代标的）
  M  MANUAL      — 标的完全不在任何 suggestion 文件中

分类结果保存为独立的 trade_classification.csv，不修改 trade_log_full.csv。
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quantpits.scripts.analysis.utils import ROOT_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
ORDER_DIR = os.path.join(DATA_DIR, "order_history")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CLASSIFICATION_FILE = os.path.join(DATA_DIR, "trade_classification.csv")
TRADE_LOG_FILE = os.path.join(DATA_DIR, "trade_log_full.csv")

BUY_TYPES = ["上海A股普通股票竞价买入", "深圳A股普通股票竞价买入"]
SELL_TYPES = ["上海A股普通股票竞价卖出", "深圳A股普通股票竞价卖出"]


# ---------------------------------------------------------------------------
# Suggestion File Loader
# ---------------------------------------------------------------------------

def _scan_suggestion_dates(prefix="buy_suggestion"):
    """Scan order_history/ and output/ to discover all available suggestion file dates."""
    dates = []
    
    # 1. Check order_history
    pattern1 = os.path.join(ORDER_DIR, f"{prefix}_*.csv")
    for f in sorted(glob.glob(pattern1)):
        base = os.path.basename(f)
        date_str = base.replace(f"{prefix}_", "").replace(".csv", "")
        try:
            dates.append(pd.Timestamp(date_str))
        except Exception:
            pass
            
    # 2. Check output directory
    pattern2 = os.path.join(OUTPUT_DIR, f"{prefix}_*.csv")
    for f in sorted(glob.glob(pattern2)):
        base = os.path.basename(f)
        # e.g., buy_suggestion_ensemble_2026-02-24.csv
        date_part = base.replace(".csv", "")[-10:]
        try:
            dates.append(pd.Timestamp(date_part))
        except Exception:
            pass
            
    return sorted(list(set(dates)))


def _find_suggestion_date(trade_date, available_dates, strict_after="2026-02-13"):
    """
    For a given trade_date, find the matching suggestion file date.

    Strategy:
      1. Exact match on trade_date
      2. For history (<= strict_after): fallback by searching backwards up to 7 days.
      3. For future (> strict_after): exact match only. No file = Manual.
    Returns the matched date or None.
    """
    td = pd.Timestamp(trade_date)
    # Exact match
    if td in available_dates:
        return td
        
    # Strict matching for future data
    if strict_after and td > pd.Timestamp(strict_after):
        return None

    # Fallback for historical data: search backwards up to 7 days
    for delta in range(1, 8):
        candidate = td - pd.Timedelta(days=delta)
        if candidate in available_dates:
            return candidate
    return None


def _load_suggestion(prefix, date):
    """Load a buy/sell suggestion CSV for a given date. Returns a DataFrame."""
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    
    # Priority 1: Check output directory (latest generated ones)
    out_pattern1 = os.path.join(OUTPUT_DIR, f"{prefix}_*_{date_str}.csv")
    out_files = sorted(glob.glob(out_pattern1))
    if not out_files:
        out_pattern2 = os.path.join(OUTPUT_DIR, f"{prefix}_{date_str}.csv")
        out_files = sorted(glob.glob(out_pattern2))
        
    if out_files:
        path = out_files[-1]
    else:
        # Priority 2: Check order_history directory
        path = os.path.join(ORDER_DIR, f"{prefix}_{date_str}.csv")

    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Ensure instrument column exists
    if "instrument" not in df.columns:
        return pd.DataFrame()
    return df


def _add_prefix(code):
    """Add SH/SZ prefix to a 6‑digit stock code."""
    if pd.isna(code):
        return code
    code = str(code).strip()
    if code.startswith("6"):
        return "SH" + code
    elif code.startswith("0") or code.startswith("3"):
        return "SZ" + code
    return code


# ---------------------------------------------------------------------------
# Core Classification
# ---------------------------------------------------------------------------

def classify_trades(verbose=False, trade_dates=None):
    """
    Classify all buy/sell trades in trade_log_full.csv.

    If trade_dates is strictly defined, only filters and classifies trades from those days.

    Returns a DataFrame with columns:
        trade_date, instrument, trade_type, trade_class,
        suggestion_date, suggestion_rank
    """
    # 1. Load trade log
    if not os.path.exists(TRADE_LOG_FILE):
        raise FileNotFoundError(f"Trade log not found: {TRADE_LOG_FILE}")

    trade_df = pd.read_csv(TRADE_LOG_FILE)

    # Normalize date column
    if "成交日期" in trade_df.columns:
        trade_df["成交日期"] = pd.to_datetime(trade_df["成交日期"])
    else:
        raise ValueError("Column '成交日期' not found in trade log")

    # Filter to specific trade dates if requested
    if trade_dates:
        target_dates = [pd.Timestamp(d) for d in trade_dates]
        trade_df = trade_df[trade_df["成交日期"].isin(target_dates)]
        
    if trade_df.empty:
        return pd.DataFrame()

    # Filter only buy/sell trades (exclude dividends, interest, etc.)
    all_types = BUY_TYPES + SELL_TYPES
    trade_only = trade_df[trade_df["交易类别"].isin(all_types)].copy()

    # Normalize instrument codes: extract 6-digit code from 证券代码
    # Note: in the trade log, 证券代码 may already have prefix like SH601066
    # or may be raw 6-digit code. We handle both.
    if "证券代码" in trade_only.columns:
        trade_only["instrument"] = trade_only["证券代码"].apply(_normalize_instrument)
    else:
        raise ValueError("Column '证券代码' not found in trade log")

    # 2. Scan available suggestion dates
    buy_dates_set = set(_scan_suggestion_dates("buy_suggestion"))
    sell_dates_set = set(_scan_suggestion_dates("sell_suggestion"))

    # Load config to get buy_suggestion_factor
    config_file = os.path.join(ROOT_DIR, "config", "prod_config.json")
    factor = 3
    if os.path.exists(config_file):
        try:
            import json
            with open(config_file, "r") as f:
                cfg = json.load(f)
            factor = cfg.get("buy_suggestion_factor", 3)
        except Exception:
            pass

    # 3. Group trades by (date, type) and classify
    import math
    results = []

    for trade_date, day_group in trade_only.groupby("成交日期"):
        # Split into buys and sells for this day
        buys = day_group[day_group["交易类别"].isin(BUY_TYPES)]
        sells = day_group[day_group["交易类别"].isin(SELL_TYPES)]

        # Deduplicate: same instrument on same day may appear multiple times
        # (e.g. partial fills). We count unique instruments.
        buy_instruments = buys["instrument"].unique().tolist()
        sell_instruments = sells["instrument"].unique().tolist()

        n_buy = len(buy_instruments)
        n_sell = len(sell_instruments)

        # Find matching suggestion files
        buy_sugg_date = _find_suggestion_date(trade_date, buy_dates_set)
        sell_sugg_date = _find_suggestion_date(trade_date, sell_dates_set)

        # Load suggestion data
        buy_sugg = _load_suggestion("buy_suggestion", buy_sugg_date) if buy_sugg_date else pd.DataFrame()
        sell_sugg = _load_suggestion("sell_suggestion", sell_sugg_date) if sell_sugg_date else pd.DataFrame()

        # Calculate algorithmic signal counts (intended DropN bounds)
        alg_n_buy = math.ceil(len(buy_sugg) / factor) if not buy_sugg.empty else 0
        alg_n_sell = len(sell_sugg) if not sell_sugg.empty else 0

        # Build instrument → rank mapping from suggestion files
        buy_rank = {}
        if not buy_sugg.empty:
            buy_rank = {row["instrument"]: idx for idx, row in buy_sugg.iterrows()}

        sell_rank = {}
        if not sell_sugg.empty:
            sell_rank = {row["instrument"]: idx for idx, row in sell_sugg.iterrows()}

        # Classify each buy
        for inst in buy_instruments:
            if inst in buy_rank:
                rank = buy_rank[inst]
                cls = "S" if rank < alg_n_buy else "A"
            else:
                rank = None
                cls = "M"
            results.append({
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "instrument": inst,
                "trade_type": "BUY",
                "trade_class": cls,
                "suggestion_date": buy_sugg_date.strftime("%Y-%m-%d") if buy_sugg_date else "",
                "suggestion_rank": rank if rank is not None else "",
                "signal_count": alg_n_buy,
            })

        # Classify each sell
        for inst in sell_instruments:
            if inst in sell_rank:
                rank = sell_rank[inst]
                cls = "S" if rank < alg_n_sell else "A"
            else:
                rank = None
                cls = "M"
            results.append({
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "instrument": inst,
                "trade_type": "SELL",
                "trade_class": cls,
                "suggestion_date": sell_sugg_date.strftime("%Y-%m-%d") if sell_sugg_date else "",
                "suggestion_rank": rank if rank is not None else "",
                "signal_count": alg_n_sell,
            })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        print("No trades to classify.")
        return result_df

    # Sort by date
    result_df = result_df.sort_values("trade_date").reset_index(drop=True)

    if verbose:
        _print_summary(result_df)

    return result_df


def _normalize_instrument(code):
    """
    Normalize instrument code to SH/SZ prefix format.
    Handles: 'SH601066', '601066', etc.
    """
    if pd.isna(code):
        return code
    code = str(code).strip()
    # Already has prefix
    if code.startswith("SH") or code.startswith("SZ"):
        return code
    # 6-digit code
    return _add_prefix(code)


def save_classification(df, path=None, append=False, trade_dates=None):
    """Save classification DataFrame to CSV. Supports incremental appends overriding specific dates."""
    if path is None:
        path = CLASSIFICATION_FILE
        
    if append and os.path.exists(path) and trade_dates:
        # Load existing cleanly and wipe out overlapping dates
        existing_df = pd.read_csv(path)
        existing_df = existing_df[~existing_df["trade_date"].isin(trade_dates)]
        final_df = pd.concat([existing_df, df], ignore_index=True)
        final_df = final_df.sort_values("trade_date").reset_index(drop=True)
    else:
        final_df = df
        
    final_df.to_csv(path, index=False)
    print(f"  → Saved classification records to {path}")


def load_classification(path=None):
    """Load existing classification CSV if available."""
    if path is None:
        path = CLASSIFICATION_FILE
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _print_summary(df):
    """Print a summary of the classification results."""
    total = len(df)
    if total == 0:
        print("No trades classified.")
        return

    print("\n" + "=" * 50)
    print("  Trade Classification Summary")
    print("=" * 50)

    for tt in ["BUY", "SELL"]:
        sub = df[df["trade_type"] == tt]
        if sub.empty:
            continue
        n = len(sub)
        signal = len(sub[sub["trade_class"] == "S"])
        subst = len(sub[sub["trade_class"] == "A"])
        manual = len(sub[sub["trade_class"] == "M"])
        print(f"\n  {tt} ({n} trades):")
        print(f"    SIGNAL:     {signal:>4d}  ({signal/n*100:5.1f}%)")
        print(f"    SUBSTITUTE: {subst:>4d}  ({subst/n*100:5.1f}%)")
        print(f"    MANUAL:     {manual:>4d}  ({manual/n*100:5.1f}%)")

    # Overall
    signal = len(df[df["trade_class"] == "S"])
    subst = len(df[df["trade_class"] == "A"])
    manual = len(df[df["trade_class"] == "M"])
    print(f"\n  TOTAL ({total} trades):")
    print(f"    SIGNAL:     {signal:>4d}  ({signal/total*100:5.1f}%)")
    print(f"    SUBSTITUTE: {subst:>4d}  ({subst/total*100:5.1f}%)")
    print(f"    MANUAL:     {manual:>4d}  ({manual/total*100:5.1f}%)")

    # List all MANUAL trades
    manuals = df[df["trade_class"] == "M"]
    if not manuals.empty:
        print(f"\n  Manual Trades ({len(manuals)}):")
        for _, row in manuals.iterrows():
            print(f"    {row['trade_date']}  {row['instrument']:>10s}  {row['trade_type']}")
    print("=" * 50 + "\n")
