#!/usr/bin/env python3
"""
Historical Trade Classification Backfill

一次性脚本：为 CSI300_Base 工作区的全部历史交易记录批量生成分类标签。

Usage:
    cd QuantPits

    # 预览模式（不写入文件）
    python quantpits/scripts/classify_history.py --dry-run --verbose

    # 正式运行
    python quantpits/scripts/classify_history.py --verbose
"""

import os
import sys
import argparse

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import env
os.chdir(env.ROOT_DIR)

# Project root (where quantpits package is) is the parent of SCRIPT_DIR's parent
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(PROJECT_ROOT)

from quantpits.scripts.analysis.trade_classifier import (
    classify_trades,
    save_classification,
    _print_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Classify all historical trades as SIGNAL / SUBSTITUTE / MANUAL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print classification summary without writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-trade classification results",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Historical Trade Classification Backfill")
    print(f"  Workspace: {env.ROOT_DIR}")
    print("=" * 60)

    # Run classification
    result_df = classify_trades(verbose=True)

    if result_df.empty:
        print("No trades found to classify. Exiting.")
        return

    # Always print summary
    if not args.verbose:
        _print_summary(result_df)

    # Verbose: print per-trade details
    if args.verbose:
        print("\n--- Detailed Classification ---")
        for _, row in result_df.iterrows():
            cls_label = {"S": "SIGNAL", "A": "SUBSTITUTE", "M": "MANUAL"}[
                row["trade_class"]
            ]
            sugg_info = ""
            if row["suggestion_date"]:
                sugg_info = f"  (sugg: {row['suggestion_date']}, rank: {row['suggestion_rank']})"
            print(
                f"  {row['trade_date']}  {row['instrument']:>10s}  "
                f"{row['trade_type']:<4s}  → {cls_label:<10s}{sugg_info}"
            )
        print()

    # Save
    if args.dry_run:
        print("[DRY-RUN] Skipping file write.")
    else:
        save_classification(result_df)
        print("✅ Classification complete.")


if __name__ == "__main__":
    main()
