#!/usr/bin/env python3
"""
Production Post-Trade Analytics 批量处理脚本

处理实盘每日订单和成交数据，仅供分析使用，追加到历史 CSV 日志中。
本脚本完全独立于交割单的现金/持仓更新。

使用方法:
    cd QuantPits
    source workspaces/CSI300_Base/run_env.sh
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

# 需要先进入对应 workspace，这通常由 env.ROOT_DIR 管理
os.chdir(env.ROOT_DIR)

DATA_DIR = "data"
ORDER_LOG_FILE = os.path.join(DATA_DIR, "raw_order_log_full.csv")
TRADE_LOG_FILE = os.path.join(DATA_DIR, "raw_trade_log_full.csv")

def load_prod_config():
    """使用 config_loader 加载统一配置以获取进度日期等信息"""
    from quantpits.utils.config_loader import load_workspace_config
    return load_workspace_config(env.ROOT_DIR)

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
    env.safeguard("Prod Post Trade Analytics")
    env.init_qlib()

    parser = argparse.ArgumentParser(description="处理每日订单和成交用于分析")
    parser.add_argument("--end-date", type=str, default=None, help="结束日期")
    parser.add_argument("--dry-run", action="store_true", help="仅预览")
    parser.add_argument("--broker", type=str, default="gtja", help="券商标识")
    args = parser.parse_args()

    config = load_prod_config()
    last_processed_date = config.get("last_processed_date", config["current_date"])
    
    broker_name = args.broker or config.get("broker", "gtja")
    try:
        adapter = get_adapter(broker_name)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    start_date = (datetime.strptime(last_processed_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    trade_dates = get_trade_dates(start_date, end_date)
    
    print(f"Last processed date (from config): {last_processed_date}")
    print(f"Analytics Date range: {start_date} to {end_date}")
    
    if not trade_dates:
        print("\nNo trade dates to process.")
        return

    for date_str in trade_dates:
        process_analytics_for_day(date_str, adapter, dry_run=args.dry_run)
        
    print(f"\n{'='*50}")
    print("Analytics Batch processing completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
