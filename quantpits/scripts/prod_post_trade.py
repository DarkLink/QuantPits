#!/usr/bin/env python3
"""
Production Post-Trade 批量处理脚本

处理实盘交易数据：读取交易软件导出文件，更新持仓、现金、日志。
本脚本与预测、融合、回测等模块完全解耦。

使用方法:
    cd QuantPits
    python quantpits/scripts/prod_post_trade.py                # 正常运行
    python quantpits/scripts/prod_post_trade.py --dry-run       # 仅预览，不写文件
    python quantpits/scripts/prod_post_trade.py --end-date 2026-02-10  # 指定结束日期
    python quantpits/scripts/prod_post_trade.py --verbose       # 详细输出
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import copy
from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd

from quantpits.scripts.brokers import get_adapter
from quantpits.scripts.brokers.base import SELL_TYPES, BUY_TYPES, INTEREST_TYPES

from quantpits.utils import env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join(env.ROOT_DIR, "config")
DATA_DIR = os.path.join(env.ROOT_DIR, "data")
PROD_CONFIG_FILE = os.path.join(env.ROOT_DIR, "config", "prod_config.json")
CASHFLOW_CONFIG_FILE = os.path.join(CONFIG_DIR, "cashflow.json")
TRADE_LOG_FILE = os.path.join(DATA_DIR, "trade_log_full.csv")
HOLDING_LOG_FILE = os.path.join(DATA_DIR, "holding_log_full.csv")
DAILY_LOG_FILE = os.path.join(DATA_DIR, "daily_amount_log_full.csv")
EMPTY_TRADE_FILE = os.path.join(DATA_DIR, "emp-table.xlsx")
_PREPARED_SETTLEMENTS = {}


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------
def load_prod_config():
    """使用 config_loader 加载统一配置"""
    from quantpits.utils.config_loader import load_workspace_config
    return load_workspace_config(env.ROOT_DIR)


def load_cashflow_config():
    """加载 cashflow.json，支持新旧两种格式。

    新格式 (推荐):
        {"cashflows": {"2026-02-03": 50000, "2026-02-06": -20000}}

    旧格式 (向后兼容):
        {"cash_flow_today": 50000}
        → 将在第一个处理日应用该值
    """
    with open(CASHFLOW_CONFIG_FILE, "r") as f:
        config = json.load(f)
    return config


def _load_command_run_config(ctx, options):
    """Build the prepared snapshot through legacy patchable loaders."""
    from quantpits.post_trade.command import PostTradeRunConfig

    prod = load_prod_config()
    raw_prod = json.loads(ctx.config_path("prod_config.json").read_text(encoding="utf-8"))
    cashflow = load_cashflow_config()
    state_cursor = prod.get("last_processed_date", prod.get("current_date"))
    if not state_cursor:
        raise ValueError("prod_config.json does not define current_date/last_processed_date")
    legacy_path = ctx.data_path(".order_trade_state.json")
    legacy_cursor = None
    if legacy_path.exists():
        legacy_cursor = json.loads(legacy_path.read_text(encoding="utf-8")).get("last_processed_date")
    return PostTradeRunConfig(
        prod_config=prod,
        cashflow_config=cashflow,
        broker_name=options.broker or prod.get("broker", "gtja"),
        state_cursor=state_cursor,
        legacy_execution_cursor=legacy_cursor,
        prod_state_document=raw_prod,
    )


def get_cashflow_for_date(cashflow_config, date_str, is_first_day=False):
    """获取指定日期的 cashflow 金额。

    Args:
        cashflow_config: cashflow 配置字典
        date_str: 日期字符串 YYYY-MM-DD
        is_first_day: 是否为批次中的第一天（旧格式兼容用）

    Returns:
        Decimal: 当日 cashflow（正=入金, 负=出金, 0=无操作）
    """
    # 新格式：按日期索引
    if "cashflows" in cashflow_config:
        cashflows = cashflow_config["cashflows"]
        return Decimal(str(cashflows.get(date_str, 0)))

    # 旧格式兼容：cash_flow_today 只在第一个处理日应用
    if is_first_day:
        return Decimal(str(cashflow_config.get("cash_flow_today", 0)))

    return Decimal("0")


def save_prod_config(config):
    """保存 prod_config.json，仅保留动态状态字段以减少冗余"""
    # 定义状态字段白名单
    state_keys = {
        "current_date", "last_processed_date", 
        "current_cash", "current_holding", 
        "model", "experiment_name", "broker"
    }
    
    # 提取状态数据
    state_config = {k: v for k, v in config.items() if k in state_keys}
    
    with open(PROD_CONFIG_FILE, "w") as f:
        json.dump(state_config, f, indent=4, ensure_ascii=False)


def archive_cashflows(cashflow_config):
    """处理完成后归档 cashflow 配置。

    新格式：将 cashflows 中的条目移到 processed 子键
    旧格式：重置 cash_flow_today 为 0
    """
    if "cashflows" in cashflow_config:
        processed = cashflow_config.get("processed", {})
        for date_str, amount in cashflow_config["cashflows"].items():
            processed[date_str] = amount
        cashflow_config["cashflows"] = {}
        cashflow_config["processed"] = processed
    else:
        cashflow_config["cash_flow_today"] = 0

    with open(CASHFLOW_CONFIG_FILE, "w") as f:
        json.dump(cashflow_config, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Trade Date Helpers
# ---------------------------------------------------------------------------
def init_qlib():
    """初始化 Qlib（委托给 env.init_qlib）"""
    env.init_qlib()


def get_trade_dates(start_date, end_date, *, strict=False):
    """使用 qlib 日历获取交易日列表"""
    from qlib.data import D
    try:
        trade_dates = D.calendar(start_time=start_date, end_time=end_date)
        return [d.strftime("%Y-%m-%d") for d in trade_dates]
    except Exception as exc:
        if strict:
            from quantpits.post_trade.contracts import PostTradeExecutionError
            raise PostTradeExecutionError(
                "Failed to resolve Qlib trading calendar for %s..%s: %s"
                % (start_date, end_date, exc)
            ) from exc
        return []


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def add_prefix(instrument):
    """添加交易所前缀 (6→SH, 0/3→SZ)"""
    if pd.isna(instrument):
        return instrument
    instrument = str(instrument)
    if instrument.startswith("6"):
        return "SH" + instrument
    elif instrument.startswith("0") or instrument.startswith("3"):
        return "SZ" + instrument
    return instrument


def load_trade_file(date_string, model, adapter):
    """加载交易文件，如不存在则使用空模板。使用具体的券商适配器进行读取。"""
    prepared = _PREPARED_SETTLEMENTS.get(date_string)
    if prepared is not None:
        df = prepared.copy()
        if not df.empty:
            df["model"] = model
        return df
    file_path = os.path.join(DATA_DIR, f"{date_string}-table.xlsx")
    if not os.path.exists(file_path):
        file_path = EMPTY_TRADE_FILE

    df = adapter.read_settlement(file_path)
    if not df.empty:
        df["model"] = model
    return df


def to_decimal_columns(df, columns):
    """将指定列转为 Decimal 类型"""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: Decimal(str(x)))


# ---------------------------------------------------------------------------
# Core Processing
# ---------------------------------------------------------------------------
def process_single_day(current_date_string, current_cash, current_holding,
                       model, market, benchmark, cashflow_today,
                       adapter, verbose=False):
    """处理单日交易数据。

    Args:
        current_date_string: 日期字符串
        current_cash: 当前现金 (Decimal)
        current_holding: 当前持仓列表
        model: 模型名称
        market: 市场
        benchmark: 基准指数代码
        cashflow_today: 当日 cashflow 金额 (Decimal)
        adapter: 券商适配器实例
        verbose: 是否详细输出

    Returns:
        (cash_after, updated_holding): 更新后的现金和持仓
    """
    from qlib.data import D
    from qlib.data.ops import Feature

    print(f"\n{'='*50}")
    print(f"Processing date: {current_date_string}")
    if cashflow_today != 0:
        print(f"  *** Cashflow: {cashflow_today:+} ***")
    print(f"{'='*50}")

    # --- 加载交易文件 ---
    trade_detail = load_trade_file(current_date_string, model, adapter)

    if not trade_detail.empty and "证券代码" in trade_detail.columns:
        trade_detail["证券代码"] = trade_detail["证券代码"].apply(add_prefix)
        if "交收日期" in trade_detail.columns:
            trade_detail["成交日期"] = pd.to_datetime(
                trade_detail["交收日期"], format="%Y%m%d"
            ).dt.strftime("%Y-%m-%d")

        # 保存交易详情
        output_file = os.path.join(DATA_DIR, f"trade_detail_{current_date_string}.csv")
        trade_detail.to_csv(output_file, index=False)

        # 更新交易日志
        if os.path.exists(TRADE_LOG_FILE):
            exists_trade_log = pd.read_csv(TRADE_LOG_FILE)
            full_trade_log = pd.concat(
                [exists_trade_log, trade_detail], ignore_index=True
            )
        else:
            full_trade_log = trade_detail
        full_trade_log.drop_duplicates().to_csv(TRADE_LOG_FILE, index=False)

    # --- 转换持仓为 DataFrame ---
    current_holding_df = pd.DataFrame(current_holding)
    if not current_holding_df.empty and "value" in current_holding_df.columns:
        current_holding_df["value"] = current_holding_df["value"].apply(lambda x: Decimal(str(x)))
        current_holding_df["amount"] = current_holding_df["amount"].apply(lambda x: Decimal(str(x)))

    # --- 处理卖出 ---
    total_sell_amount = Decimal("0")
    sell_merged_df = pd.DataFrame()

    if not trade_detail.empty and "资金发生数" in trade_detail.columns:
        current_trade_detail = trade_detail.loc[trade_detail["model"] == model]
        sold_trade_detail = current_trade_detail.loc[
            current_trade_detail["交易类别"].isin(SELL_TYPES)
        ].copy()

        if not sold_trade_detail.empty:
            to_decimal_columns(
                sold_trade_detail, ["成交价格", "成交金额", "资金发生数", "成交数量"]
            )

            holding_for_merge = current_holding_df.copy()
            holding_for_merge.rename(
                columns={
                    "instrument": "证券代码",
                    "value": "持仓数量",
                    "amount": "持仓成本",
                },
                inplace=True,
            )
            holding_for_merge["持仓均价"] = (
                holding_for_merge["持仓成本"] / holding_for_merge["持仓数量"]
            )

            sell_merged_df = pd.merge(
                sold_trade_detail, holding_for_merge, on=["证券代码"], how="inner"
            )
            if not sell_merged_df.empty:
                sell_merged_df["收益额"] = sell_merged_df["资金发生数"] - (
                    sell_merged_df["持仓均价"] * sell_merged_df["成交数量"]
                ).apply(lambda x: round(x, 2))
                sell_merged_df["收益率"] = (
                    sell_merged_df["收益额"] / sell_merged_df["资金发生数"]
                ).apply(lambda x: round(x, 5))
                total_sell_amount = sell_merged_df["资金发生数"].sum()
                print(f"  Sell amount: {total_sell_amount}")

                if verbose:
                    for _, row in sell_merged_df.iterrows():
                        print(
                            f"    {row['证券代码']} qty={row['成交数量']} "
                            f"amt={row['资金发生数']} pnl={row['收益额']}"
                        )

    # --- 处理买入 ---
    total_buy_amount = Decimal("0")
    buy_merged_df = pd.DataFrame()

    if not trade_detail.empty and "交易类别" in trade_detail.columns:
        current_trade_detail = trade_detail.loc[trade_detail["model"] == model]
        bought_trade_detail = current_trade_detail.loc[
            current_trade_detail["交易类别"].isin(BUY_TYPES)
        ].copy()

        if not bought_trade_detail.empty:
            to_decimal_columns(
                bought_trade_detail, ["成交价格", "成交金额", "资金发生数", "成交数量"]
            )
            total_buy_amount = -1 * bought_trade_detail["资金发生数"].sum()
            buy_merged_df = bought_trade_detail
            print(f"  Buy amount: {total_buy_amount}")

            if verbose:
                for _, row in bought_trade_detail.iterrows():
                    print(
                        f"    {row['证券代码']} qty={row['成交数量']} "
                        f"amt={row['资金发生数']}"
                    )

    # --- 处理利息/红利 ---
    total_interest_amount = Decimal("0")
    if not trade_detail.empty and "交易类别" in trade_detail.columns:
        interest_detail = trade_detail.loc[
            trade_detail["交易类别"].isin(INTEREST_TYPES)
        ].copy()
        if not interest_detail.empty:
            interest_detail["资金发生数"] = interest_detail["资金发生数"].apply(
                lambda x: Decimal(str(x))
            )
            total_interest_amount = interest_detail["资金发生数"].sum()
            print(f"  Interest/Dividend: {total_interest_amount}")

    # --- 更新现金 (包含 cashflow) ---
    cash_after = (
        current_cash
        + Decimal(str(total_sell_amount))
        - Decimal(str(total_buy_amount))
        + total_interest_amount
        + cashflow_today
    )
    cash_components = f"{current_cash}"
    if total_sell_amount:
        cash_components += f" +sell({total_sell_amount})"
    if total_buy_amount:
        cash_components += f" -buy({total_buy_amount})"
    if total_interest_amount:
        cash_components += f" +int({total_interest_amount})"
    if cashflow_today:
        cash_components += f" +cf({cashflow_today})"
    print(f"  Cash: {cash_components} = {cash_after}")

    # --- 更新持仓 ---
    new_holding_df = current_holding_df.copy()
    new_holding_df.rename(
        columns={
            "instrument": "证券代码",
            "value": "持仓数量",
            "amount": "持仓成本",
        },
        inplace=True,
    )
    new_holding_df = new_holding_df.set_index("证券代码")

    # 处理卖出更新
    if not sell_merged_df.empty:
        for _, row in sell_merged_df.iterrows():
            code = row["证券代码"]
            if code in new_holding_df.index:
                new_holding_df.loc[code, "持仓数量"] -= row["成交数量"]
                new_holding_df.loc[code, "持仓成本"] -= row["资金发生数"]
                if new_holding_df.loc[code, "持仓数量"] == 0:
                    new_holding_df.drop(index=code, inplace=True)

    # 处理买入更新
    if not buy_merged_df.empty:
        for _, row in buy_merged_df.iterrows():
            code = row["证券代码"]
            bought_qty = row["成交数量"]
            bought_amt = -1 * row["资金发生数"]
            if code in new_holding_df.index:
                new_holding_df.loc[code, "持仓数量"] += bought_qty
                new_holding_df.loc[code, "持仓成本"] += bought_amt
            else:
                new_holding_df.loc[code] = {
                    "持仓数量": bought_qty,
                    "持仓成本": bought_amt,
                }

    new_holding_df.reset_index(inplace=True)

    # --- 获取收盘价 ---
    instruments = D.instruments(market="all")
    current_close = Feature("close") / Feature("factor")
    features_df = D.features(
        start_time=current_date_string,
        instruments=instruments,
        fields=[current_close],
    )
    features_df_reset = features_df.reset_index()
    features_df_reset.rename(
        columns={
            "instrument": "证券代码",
            "datetime": "成交日期",
            "Div($close,$factor)": "收盘价格",
        },
        inplace=True,
    )
    features_df_reset["收盘价格"] = features_df_reset["收盘价格"].apply(
        lambda x: Decimal(str(x))
    )

    # --- 获取基准收盘价 ---
    try:
        features_benchmark = D.features(
            start_time=current_date_string,
            instruments=[benchmark],
            fields=[current_close],
        )
        current_benchmark = features_benchmark.loc[
            (benchmark, current_date_string), "Div($close,$factor)"
        ]
        current_benchmark = round(current_benchmark.item(), 2)
    except Exception:
        current_benchmark = 0

    # --- 合并持仓与收盘价 ---
    new_holding_df["成交日期"] = pd.to_datetime(current_date_string)
    if "持仓均价" not in new_holding_df.columns:
        new_holding_df["持仓均价"] = new_holding_df["持仓成本"].apply(
            lambda x: Decimal(str(x))
        ) / new_holding_df["持仓数量"].apply(lambda x: Decimal(str(x)))

    holding_merged_df = pd.merge(
        new_holding_df, features_df_reset, on=["证券代码", "成交日期"], how="inner"
    )

    if not holding_merged_df.empty:
        holding_merged_df["收盘价格"] = holding_merged_df["收盘价格"].apply(
            lambda x: round(float(x), 2)
        )
        holding_merged_df["收盘价值"] = (
            holding_merged_df["持仓数量"].apply(lambda x: float(x))
            * holding_merged_df["收盘价格"]
        ).apply(lambda x: round(x, 2))
        holding_merged_df["当前浮盈"] = holding_merged_df[
            "收盘价值"
        ] - holding_merged_df["持仓成本"].apply(lambda x: float(x))
        holding_merged_df["浮盈收益率"] = (
            holding_merged_df["当前浮盈"]
            / holding_merged_df["持仓成本"].apply(lambda x: float(x))
        ).apply(lambda x: round(x, 5))
        holding_merged_df["成交日期"] = holding_merged_df["成交日期"].dt.strftime(
            "%Y-%m-%d"
        )

    # 添加现金行
    cash_row = pd.DataFrame(
        [
            {
                "成交日期": current_date_string,
                "证券代码": "CASH",
                "持仓数量": float(cash_after),
                "持仓成本": float(cash_after),
                "持仓均价": 1,
                "收盘价格": 1,
                "收盘价值": float(cash_after),
                "当前浮盈": 0,
                "浮盈收益率": 0,
            }
        ]
    )
    holding_merged_df = pd.concat([holding_merged_df, cash_row], ignore_index=True)

    # --- 更新持仓日志 ---
    if os.path.exists(HOLDING_LOG_FILE):
        exists_log = pd.read_csv(HOLDING_LOG_FILE)
        full_log = pd.concat([exists_log, holding_merged_df], ignore_index=True)
    else:
        full_log = holding_merged_df
    full_log.drop_duplicates().to_csv(HOLDING_LOG_FILE, index=False)

    # --- 更新每日金额汇总 ---
    holding_merged_df["持仓成本"] = holding_merged_df["持仓成本"].apply(
        lambda x: float(x)
    )
    holding_merged_df["收盘价值"] = holding_merged_df["收盘价值"].apply(
        lambda x: float(x)
    )
    daily_sum = (
        holding_merged_df.groupby("成交日期")[["持仓成本", "收盘价值"]]
        .sum(numeric_only=True)
        .reset_index()
    )
    daily_sum["当前浮盈"] = (daily_sum["收盘价值"] - daily_sum["持仓成本"]).round(2)
    daily_sum["当前浮盈率"] = (daily_sum["当前浮盈"] / daily_sum["持仓成本"]).round(5)
    daily_sum["CSI300"] = str(current_benchmark)
    daily_sum["CASHFLOW"] = str(float(cashflow_today))

    if os.path.exists(DAILY_LOG_FILE):
        exists_daily = pd.read_csv(DAILY_LOG_FILE)
        full_daily = pd.concat([exists_daily, daily_sum], ignore_index=True)
    else:
        full_daily = daily_sum
    full_daily.drop_duplicates().to_csv(DAILY_LOG_FILE, index=False)

    # --- 转换回原始格式 ---
    updated_holding = []
    for _, row in new_holding_df.iterrows():
        updated_holding.append(
            {
                "instrument": row["证券代码"],
                "value": str(row["持仓数量"]),
                "amount": str(row["持仓成本"]),
            }
        )

    print(f"  Holdings: {len(updated_holding)} positions")

    return cash_after, updated_holding, float(daily_sum["收盘价值"].iloc[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Production Post-Trade 批量处理 — 处理实盘交易数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quantpits/scripts/prod_post_trade.py                        # 正常运行
  python quantpits/scripts/prod_post_trade.py --dry-run              # 仅预览
  python quantpits/scripts/prod_post_trade.py --end-date 2026-02-10  # 指定结束日期
  python quantpits/scripts/prod_post_trade.py --verbose              # 详细输出

Cashflow 配置 (config/cashflow.json):
  新格式: {"cashflows": {"2026-02-03": 50000, "2026-02-06": -20000}}
  旧格式: {"cash_flow_today": 50000}  (仅在首日应用)
        """,
    )
    from quantpits.post_trade.command import add_post_trade_arguments
    return add_post_trade_arguments(parser, default_scope="all")


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main():
    args = parse_args()
    from quantpits.post_trade.command import (
        PostTradeCommandDependencies, execute_prepared, options_from_namespace,
        prepare_post_trade_run, render_prepared,
    )
    try:
        options = options_from_namespace(args)
        prepared = prepare_post_trade_run(
            env.get_workspace_context(), options, cli_args=tuple(sys.argv[1:]),
            dependencies=PostTradeCommandDependencies(load_run_config=_load_command_run_config),
        )
    except ValueError as exc:
        build_parser().error(str(exc))
    if args.explain_plan or args.json_plan:
        print(render_prepared(prepared))
        return
    if not args.dry_run:
        env.safeguard("Prod Post Trade")

    cwd = os.getcwd()

    # Strict preflight and execution-evidence ingestion happen before legacy state writers.
    try:
        adapter = get_adapter(prepared.config.broker_name)
        from quantpits.post_trade.service import PostTradeService
        from quantpits.post_trade.valuation import QlibValuationProvider
        state_service = PostTradeService(QlibValuationProvider())

        def _run_state(effective_prepared, parsed):
            dates = tuple(
                item.trade_date for item in effective_prepared.catalog.settlement_sources
                if item.status in {"present", "assumed_empty"}
            )
            return state_service.run_state(effective_prepared, parsed, dates)

        summary = execute_prepared(
            prepared, adapter, init_qlib=init_qlib,
            resolve_trade_dates=lambda start, end: get_trade_dates(start, end, strict=True),
            state_callback=_run_state,
        )
    except Exception as exc:
        from quantpits.post_trade.contracts import PostTradeExecutionError
        if isinstance(exc, (PostTradeExecutionError, ValueError)):
            print("[ERROR] %s" % exc)
            raise SystemExit(1)
        raise
    if args.dry_run:
        print(render_prepared(summary.prepared))
        if summary.state_result is not None:
            print("\n[DRY-RUN] Strict broker intake validation passed. State calculation passed: %s" % json.dumps(summary.state_result.public_summary(), sort_keys=True))
        else:
            print("\n[DRY-RUN] Strict broker intake validation passed. No files written.")
        return

    # Execution-only is a strict terminal route: it never initializes or
    # mutates account state.
    if prepared.options.scope == "execution":
        print("Execution evidence ingestion completed.")
        return

    state_result = summary.state_result
    trade_dates = list(state_result.change_set.processed_dates) if state_result else []
    if not trade_dates:
        print("\nNo trade dates to process.")
        return

    # --- 执行交易分类 ---
    try:
        from quantpits.scripts.analysis.trade_classifier import classify_trades, save_classification
        print(f"\n{'='*50}")
        print(f"Running Trade Classification Engine for {len(trade_dates)} dates...")
        classified_df = classify_trades(verbose=False, trade_dates=trade_dates)
        if not classified_df.empty:
            save_classification(classified_df, append=True, trade_dates=trade_dates)
            print("Trade classification updated.")
        else:
            print("No new trades required classification.")
    except Exception as e:
        print(f"\n[WARN] Failed to run trade classification: {e}")
        import traceback
        traceback.print_exc()

    # --- 完成 ---
    print(f"\n{'='*50}")
    print(f"Batch processing completed!")
    print(f"Processed {len(trade_dates)} trade dates")
    print("State invariants: passed")
    print(f"Final holdings: {len(state_result.change_set.final_state.positions)} positions")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
