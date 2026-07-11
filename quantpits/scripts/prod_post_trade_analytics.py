#!/usr/bin/env python3
"""
Production Post-Trade Analytics 批量处理脚本

处理实盘每日订单和成交数据，仅供分析使用，追加到历史 CSV 日志中。
本脚本完全独立于交割单的现金/持仓更新。

使用方法:
    cd QuantPits
    source workspaces/Demo_Workspace/run_env.sh
    python quantpits/scripts/prod_post_trade_analytics.py                # 正常运行
    python quantpits/scripts/prod_post_trade_analytics.py --dry-run       # 仅预览，不写文件
    python quantpits/scripts/prod_post_trade_analytics.py --end-date 2026-02-10  # 指定结束日期
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantpits.scripts.brokers import get_adapter
from quantpits.utils import env

DATA_DIR = os.path.join(env.ROOT_DIR, "data")
ORDER_LOG_FILE = os.path.join(DATA_DIR, "raw_order_log_full.csv")
TRADE_LOG_FILE = os.path.join(DATA_DIR, "raw_trade_log_full.csv")
ORDER_TRADE_STATE_FILE = os.path.join(DATA_DIR, ".order_trade_state.json")

def load_prod_config():
    """使用 config_loader 加载统一配置以获取进度日期等信息"""
    from quantpits.utils.config_loader import load_workspace_config
    return load_workspace_config(env.ROOT_DIR)


def _load_command_run_config(ctx, options):
    """Build execution preparation through legacy patchable config helpers."""
    from quantpits.post_trade.command import PostTradeRunConfig

    prod = load_prod_config()
    state_cursor = prod.get("last_processed_date", prod.get("current_date"))
    if not state_cursor:
        raise ValueError("prod_config.json does not define current_date/last_processed_date")
    return PostTradeRunConfig(
        prod_config=prod,
        cashflow_config={},
        broker_name=options.broker or prod.get("broker", "gtja"),
        state_cursor=state_cursor,
        legacy_execution_cursor=load_order_trade_last_date(None),
    )


def load_order_trade_last_date(fallback_date):
    """加载 order/trade 的最后处理日期，独立于 prod_config。

    避免 prod_post_trade.py（交割）更新 config 后导致本脚本跳过日期。
    """
    import json
    if os.path.exists(ORDER_TRADE_STATE_FILE):
        with open(ORDER_TRADE_STATE_FILE, "r") as f:
            state = json.load(f)
        return state.get("last_processed_date", fallback_date)
    return fallback_date


def save_order_trade_last_date(date_str):
    """保存 order/trade 最后处理日期到独立状态文件"""
    import json
    with open(ORDER_TRADE_STATE_FILE, "w") as f:
        json.dump({"last_processed_date": date_str}, f)

def get_trade_dates(start_date, end_date):
    """使用 qlib 日历获取交易日列表"""
    from qlib.data import D
    try:
        trade_dates = D.calendar(start_time=start_date, end_time=end_date)
        return [d.strftime("%Y-%m-%d") for d in trade_dates]
    except Exception:
        return []

def process_analytics_for_day(date_str, adapter, dry_run=False):
    order_file = os.path.join(DATA_DIR, f"{date_str}-order.xlsx")
    trade_file = os.path.join(DATA_DIR, f"{date_str}-trade.xlsx")
    
    print(f"\nProcessing analytics for {date_str}...")
    
    # Process Orders
    if os.path.exists(order_file):
        df_order = adapter.read_orders(order_file)
        if not df_order.empty:
            print(f"  Orders found: {len(df_order)} rows")
            if not dry_run:
                # 追加到 log
                if os.path.exists(ORDER_LOG_FILE):
                    exist_df = pd.read_csv(ORDER_LOG_FILE, dtype={"证券代码": str})
                    full_df = pd.concat([exist_df, df_order], ignore_index=True)
                else:
                    full_df = df_order
                # 去重保存
                full_df.drop_duplicates().to_csv(ORDER_LOG_FILE, index=False)
        else:
            print("  Orders: No valid rows found after filtering.")
    else:
        print(f"  Orders: File not found ({date_str}-order.xlsx)")
        
    # Process Trades
    if os.path.exists(trade_file):
        df_trade = adapter.read_trades(trade_file)
        if not df_trade.empty:
            print(f"  Trades found: {len(df_trade)} rows")
            if not dry_run:
                # 追加到 log
                if os.path.exists(TRADE_LOG_FILE):
                    exist_df = pd.read_csv(TRADE_LOG_FILE, dtype={"证券代码": str})
                    full_df = pd.concat([exist_df, df_trade], ignore_index=True)
                else:
                    full_df = df_trade
                # 去重保存
                full_df.drop_duplicates().to_csv(TRADE_LOG_FILE, index=False)
        else:
            print("  Trades: No valid rows found after filtering.")
    else:
        print(f"  Trades: File not found ({date_str}-trade.xlsx)")

def main():
    parser = argparse.ArgumentParser(description="处理每日订单和成交用于分析")
    from quantpits.post_trade.command import (
        PostTradeCommandDependencies, add_post_trade_arguments, execute_prepared, options_from_namespace,
        prepare_post_trade_run, render_prepared,
    )
    add_post_trade_arguments(parser, default_scope="execution")
    args = parser.parse_args()
    try:
        options = options_from_namespace(args)
        if options.scope != "execution":
            raise ValueError("analytics compatibility command only supports --scope execution")
        prepared = prepare_post_trade_run(
            env.get_workspace_context(), options, cli_args=tuple(sys.argv[1:]),
            dependencies=PostTradeCommandDependencies(load_run_config=_load_command_run_config),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.explain_plan or args.json_plan:
        print(render_prepared(prepared))
        return
    if not args.dry_run:
        env.safeguard("Prod Post Trade Analytics")
    try:
        summary = execute_prepared(prepared, get_adapter(prepared.config.broker_name))
    except Exception as exc:
        from quantpits.post_trade.contracts import PostTradeExecutionError
        if isinstance(exc, (PostTradeExecutionError, ValueError)):
            print("[ERROR] %s" % exc)
            raise SystemExit(1)
        raise
    if args.dry_run:
        print(render_prepared(summary.prepared))
        print("\n[DRY-RUN] Strict execution-evidence validation passed. No files written.")
    else:
        count = len(summary.ingestion.ingested_sources) if summary.ingestion else 0
        print("\nAnalytics ingestion completed: %d source receipts." % count)

if __name__ == "__main__":
    main()
