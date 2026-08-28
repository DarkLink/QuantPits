#!/usr/bin/env python
"""
monkey_benchmark.py — 🐒 随机猴子 Monte Carlo 基准测试

高效批量模拟 N 次随机选股，生成统计分布，与真实模型的 IC/ICIR/收益/Sharpe/回撤
做假设检验。数据集只构建 1 次，随机循环 N 次，跳过 MLflow/State V3 开销。

用法:
    # 跑 1000 个猴子，和所有 enabled 模型比较
    python quantpits/scripts/monkey_benchmark.py --n-trials 1000

    # 指定要对比的真实模型
    python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --models gru,lightgbm_Alpha158

    # 输出直方图
    python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --plot

    # 快速模式（仅 IC/ICIR，跳过回测）
    python quantpits/scripts/monkey_benchmark.py --n-trials 5000 --ic-only
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
from quantpits.utils import env
from quantpits.utils.constants import TRADING_DAYS_PER_YEAR

os.chdir(env.ROOT_DIR)
ROOT_DIR = env.ROOT_DIR


# ============================================================================
# 数据准备（一次性开销）
# ============================================================================

def load_label_and_returns(norm_df_or_dates, instruments, freq="week"):
    """加载前向收益标签和日频收益率。

    Args:
        norm_df_or_dates: 带 datetime 索引的 DataFrame 或 DatetimeIndex
        instruments: 股票列表
        freq: 交易频率 ('week'/'day')

    Returns:
        label_series: pd.Series, index=(datetime, instrument), 前向 N 日收益率
        returns_wide: pd.DataFrame, index=datetime, columns=instrument, 日频收益率
        bench_returns: pd.Series, index=datetime, 基准日频收益率
        common_dates: DatetimeIndex
    """
    from qlib.data import D

    if isinstance(norm_df_or_dates, pd.DatetimeIndex):
        dates = norm_df_or_dates
    else:
        dates = norm_df_or_dates.index.get_level_values("datetime").unique().sort_values()

    start_date = str(dates.min().date())
    end_date = str(dates.max().date())

    # 前向收益标签（与 YAML label 一致：周频用 6 天前瞻）
    ref_days = 6 if freq == "week" else 2
    label_expr = f"Ref($close, -{ref_days}) / Ref($close, -1) - 1"

    print(f"加载标签数据: {label_expr}")
    label_df = D.features(instruments, [label_expr],
                          start_time=start_date, end_time=end_date)
    label_df.columns = ["label"]
    label_series = label_df["label"]

    # 日频收益率
    print("加载日频收益率...")
    ret_df = D.features(instruments, ["Ref($close, -1)/$close - 1"],
                        start_time=start_date, end_time=end_date)
    ret_df.columns = ["return"]
    returns_wide = ret_df["return"].unstack(level="instrument")

    common_dates = dates.intersection(returns_wide.index)
    returns_wide = returns_wide.loc[common_dates]

    # 基准收益
    try:
        bench_df = D.features(["SH000300"], ["$close"],
                              start_time=start_date, end_time=end_date)
        bench_close = bench_df["$close"]
        bench_returns = bench_close.pct_change(1).shift(-1)
        if hasattr(bench_returns.index, "get_level_values"):
            bench_returns.index = bench_returns.index.get_level_values("datetime")
        bench_returns = bench_returns.reindex(common_dates)
    except Exception:
        bench_returns = pd.Series(0.0, index=common_dates)

    return label_series, returns_wide, bench_returns, common_dates


def load_real_model_metrics(train_records, selected_models=None):
    """从已有训练记录加载真实模型的 IC/ICIR 指标。

    Returns:
        dict: model_name -> {"ic": float, "icir": float}
    """
    from qlib.workflow import R
    from quantpits.utils.train_utils import get_experiment_name_for_model

    models = train_records.get("models", {})
    if selected_models is None:
        selected_models = list(models.keys())

    metrics = {}
    for model_name in selected_models:
        if model_name not in models:
            continue
        record_id = models[model_name]
        try:
            exp_name = get_experiment_name_for_model(train_records, model_name)
            recorder = R.get_recorder(
                recorder_id=record_id, experiment_name=exp_name
            )

            # 读取 IC 序列
            ic_series = recorder.load_object("sig_analysis/ic.pkl")
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            icir = ic_mean / ic_std if ic_std > 0 else 0.0

            metrics[model_name] = {"ic": ic_mean, "icir": icir}
        except Exception as e:
            print(f"  [{model_name}] 无法加载指标: {e}")

    return metrics


# ============================================================================
# Monte Carlo 核心
# ============================================================================

def compute_daily_ic(scores, labels, dates_level):
    """按日计算 Pearson IC。

    Args:
        scores: pd.Series, 随机分数 (index=MultiIndex)
        labels: pd.Series, 前向收益标签 (index=MultiIndex)
        dates_level: str, datetime level name

    Returns:
        ic_series: pd.Series, 每日 IC
    """
    df = pd.DataFrame({"score": scores, "label": labels}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    ic = df.groupby(level=dates_level).apply(
        lambda x: x["score"].corr(x["label"]) if len(x) >= 5 else np.nan
    ).dropna()
    return ic


def vectorized_topk_backtest(random_scores_wide, returns_np, top_k,
                              cost_rate, rebalance_freq):
    """向量化 TopK 回测，复用 brute_force_fast 的核心逻辑。

    Args:
        random_scores_wide: (T, N) ndarray, 随机分数矩阵
        returns_np: (T, N) ndarray, 日频收益率矩阵
        top_k: TopK 持仓数
        cost_rate: 单次换手交易费用率
        rebalance_freq: 调仓频率（天数）

    Returns:
        net_returns: (T,) ndarray, 每日净收益率
    """
    T, N = random_scores_wide.shape
    k = min(top_k, N)

    # 每个调仓日的 TopK
    rebalance_indices = np.arange(0, T, rebalance_freq)
    scores_reb = random_scores_wide[rebalance_indices]

    if k < N:
        topk_reb = np.argpartition(-scores_reb, k, axis=1)[:, :k]
    else:
        topk_reb = np.tile(np.arange(N), (len(rebalance_indices), 1))

    # 广播到每天
    day_to_reb = np.arange(T) // rebalance_freq
    day_to_reb = np.clip(day_to_reb, 0, len(rebalance_indices) - 1)
    actual_holdings = topk_reb[day_to_reb]

    # 日收益
    row_idx = np.arange(T)[:, None]
    daily_returns = np.mean(returns_np[row_idx, actual_holdings], axis=1)

    # 换手费用
    turnover_costs = np.zeros(T, dtype=np.float32)
    if cost_rate > 0 and len(rebalance_indices) > 1:
        mask_reb = np.zeros((len(rebalance_indices), N), dtype=bool)
        reb_row_idx = np.arange(len(rebalance_indices))[:, None]
        mask_reb[reb_row_idx, topk_reb] = True
        turnovers = np.sum(mask_reb[1:] & ~mask_reb[:-1], axis=1) / k
        turnover_costs[rebalance_indices[1:]] = turnovers * cost_rate

    return daily_returns - turnover_costs


def compute_backtest_metrics(net_returns, bench_returns_np):
    """从日频净收益率计算组合绩效指标。"""
    days = len(net_returns)
    if days == 0:
        return {}

    nav = np.cumprod(1 + net_returns)
    final_nav = nav[-1]
    total_ret = final_nav - 1.0

    years = days / TRADING_DAYS_PER_YEAR
    ann_ret = (final_nav ** (1 / years)) - 1 if final_nav > 0 else -1.0

    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    max_dd = float(np.min(drawdown))

    if np.std(net_returns) > 0:
        sharpe = float(np.mean(net_returns) / np.std(net_returns)
                       * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0

    # 基准
    bench_nav = np.cumprod(1 + bench_returns_np)
    bench_final = bench_nav[-1] if len(bench_nav) > 0 else 1.0
    bench_cagr = (bench_final ** (1 / years)) - 1 if bench_final > 0 else -1.0

    return {
        "ann_ret": float(ann_ret),
        "total_ret": float(total_ret),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "excess_ret": float(ann_ret - bench_cagr),
    }


def run_monte_carlo(n_trials, label_series, returns_wide, bench_returns,
                    common_dates, top_k, cost_rate, rebalance_freq,
                    ic_only=False):
    """执行 N 次随机猴子模拟。

    Args:
        n_trials: 模拟次数
        label_series: 前向收益标签 (MultiIndex: datetime, instrument)
        returns_wide: 日频收益率宽表 (datetime x instrument)
        bench_returns: 基准日频收益率 (datetime)
        common_dates: 公共交易日
        top_k: TopK 持仓数
        cost_rate: 交易费用率
        rebalance_freq: 调仓频率
        ic_only: 仅计算 IC/ICIR，跳过回测

    Returns:
        pd.DataFrame: 每次模拟的指标
    """
    # 预对齐数据
    label_aligned = label_series.reindex(
        pd.MultiIndex.from_product(
            [common_dates, returns_wide.columns],
            names=["datetime", "instrument"]
        )
    )
    valid_mask = label_aligned.notna()
    label_for_ic = label_aligned[valid_mask]

    # 回测用矩阵
    if not ic_only:
        returns_np = returns_wide.reindex(common_dates).values.astype(np.float32)
        returns_np = np.nan_to_num(returns_np, nan=0.0)
        bench_np = bench_returns.reindex(common_dates).fillna(0).values.astype(np.float32)
        T, N = returns_np.shape

    results = []
    rng = np.random.default_rng()

    for i in tqdm(range(n_trials), desc="🐒 Running monkeys", unit="trial"):
        # 生成随机分数
        random_full = pd.Series(
            rng.uniform(0, 1, size=len(label_aligned)),
            index=label_aligned.index,
        )

        # IC 计算
        ic_series = compute_daily_ic(
            random_full[valid_mask], label_for_ic, "datetime"
        )
        ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
        ic_std = float(ic_series.std()) if len(ic_series) > 0 else 1.0
        icir = ic_mean / ic_std if ic_std > 0 else 0.0

        trial_result = {"ic": ic_mean, "icir": icir}

        # 回测
        if not ic_only:
            scores_wide = random_full.unstack(level="instrument")
            scores_wide = scores_wide.reindex(
                index=common_dates, columns=returns_wide.columns
            )
            scores_np = scores_wide.values.astype(np.float32)
            scores_np = np.nan_to_num(scores_np, nan=-np.inf)

            net_ret = vectorized_topk_backtest(
                scores_np, returns_np, top_k, cost_rate, rebalance_freq
            )
            bt_metrics = compute_backtest_metrics(net_ret, bench_np)
            trial_result.update(bt_metrics)

        results.append(trial_result)

    return pd.DataFrame(results)


# ============================================================================
# 统计分析 & 输出
# ============================================================================

def print_distribution(series, name, pct_format=False):
    """打印分布统计。"""
    fmt = lambda v: f"{v*100:.2f}%" if pct_format else f"{v:.4f}"
    print(f"  Mean:    {fmt(series.mean())} ± {fmt(series.std())}")
    print(f"  Median:  {fmt(series.median())}")
    print(f"  5%ile:   {fmt(series.quantile(0.05))}")
    print(f"  95%ile:  {fmt(series.quantile(0.95))}")
    print(f"  Min:     {fmt(series.min())}")
    print(f"  Max:     {fmt(series.max())}")


def compute_percentile(value, distribution):
    """计算 value 在分布中的百分位排名。"""
    return float(np.mean(distribution <= value) * 100)


def compute_pvalue(value, distribution):
    """计算 value 显著优于分布的 p-value（单侧）。"""
    return float(np.mean(distribution >= value))


def print_results(monkey_df, real_metrics, ic_only=False):
    """打印完整的比较结果。"""
    n = len(monkey_df)
    print(f"\n{'='*70}")
    print(f"🐒 Monkey Benchmark — {n} Random Trials")
    print(f"{'='*70}")

    # 随机分布
    print(f"\n--- Random IC Distribution ---")
    print_distribution(monkey_df["ic"], "IC")

    print(f"\n--- Random ICIR Distribution ---")
    print_distribution(monkey_df["icir"], "ICIR")

    if not ic_only and "ann_ret" in monkey_df.columns:
        print(f"\n--- Random Annualized Return Distribution ---")
        print_distribution(monkey_df["ann_ret"], "Ann_Ret", pct_format=True)

        print(f"\n--- Random Sharpe Distribution ---")
        print_distribution(monkey_df["sharpe"], "Sharpe")

        print(f"\n--- Random Max Drawdown Distribution ---")
        print_distribution(monkey_df["max_dd"], "Max_DD", pct_format=True)

    # 真实模型对比
    if real_metrics:
        print(f"\n{'='*70}")
        print("Your Models vs Monkeys:")
        print(f"{'='*70}")

        header_parts = ["Model", "IC", "ICIR", "IC %ile", "IC p-val"]
        if not ic_only and "ann_ret" in monkey_df.columns:
            header_parts.extend(["Ann_Ret %ile", "Sharpe %ile"])

        # 打印表头
        fmt_header = " | ".join(f"{h:>14s}" for h in header_parts)
        print(f"\n{fmt_header}")
        print("-" * len(fmt_header))

        for model_name, m in sorted(real_metrics.items()):
            ic_val = m.get("ic", 0)
            icir_val = m.get("icir", 0)
            ic_pct = compute_percentile(ic_val, monkey_df["ic"].values)
            ic_pval = compute_pvalue(ic_val, monkey_df["ic"].values)

            parts = [
                f"{model_name:>14s}",
                f"{ic_val:>14.4f}",
                f"{icir_val:>14.4f}",
                f"{ic_pct:>13.1f}%",
                f"{ic_pval:>14.4f}",
            ]

            if not ic_only and "ann_ret" in monkey_df.columns:
                # 需要真实模型的回测指标才能比较；这里仅用 IC 分布
                parts.extend(["           N/A", "           N/A"])

            print(" | ".join(parts))

        # 判定
        print()
        sig_count = 0
        for model_name, m in real_metrics.items():
            pval = compute_pvalue(m.get("ic", 0), monkey_df["ic"].values)
            if pval < 0.05:
                sig_count += 1

        if sig_count == len(real_metrics):
            print(f"Verdict: All {len(real_metrics)} models significantly "
                  f"beat monkeys (p < 0.05) ✅")
        elif sig_count > 0:
            print(f"Verdict: {sig_count}/{len(real_metrics)} models "
                  f"significantly beat monkeys (p < 0.05) ⚠️")
        else:
            print(f"Verdict: No model significantly beats monkeys (p < 0.05) ❌")


def plot_distributions(monkey_df, real_metrics, output_dir, ic_only=False):
    """生成分布直方图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n = len(monkey_df)

    metrics_to_plot = [
        ("ic", "IC", False),
        ("icir", "ICIR", False),
    ]
    if not ic_only and "ann_ret" in monkey_df.columns:
        metrics_to_plot.extend([
            ("ann_ret", "Annualized Return", True),
            ("sharpe", "Sharpe Ratio", False),
            ("max_dd", "Max Drawdown", True),
        ])

    for col, label, is_pct in metrics_to_plot:
        if col not in monkey_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        data = monkey_df[col].dropna()
        if is_pct:
            data = data * 100

        ax.hist(data, bins=50, alpha=0.7, color="#4FC3F7",
                edgecolor="#0288D1", label=f"Random ({n} trials)")

        # 标注真实模型
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(real_metrics), 1)))
        for i, (model_name, m) in enumerate(real_metrics.items()):
            val = m.get("ic", 0) if col in ("ic", "icir") else None
            if col == "icir":
                val = m.get("icir", 0)
            if val is not None:
                plot_val = val * 100 if is_pct else val
                ax.axvline(plot_val, color=colors[i], linestyle="--",
                          linewidth=2, label=f"{model_name} ({plot_val:.4f})")

        unit = "%" if is_pct else ""
        ax.set_xlabel(f"{label} {unit}")
        ax.set_ylabel("Count")
        ax.set_title(f"🐒 Monkey Benchmark: {label} Distribution ({n} trials)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(output_dir, f"monkey_{col}_distribution.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  图表已保存: {fig_path}")


def save_results_csv(monkey_df, output_dir):
    """保存完整模拟结果到 CSV。"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "monkey_trials.csv")
    monkey_df.to_csv(csv_path, index=True)
    print(f"  完整结果已保存: {csv_path}")
    return csv_path


# ============================================================================
# 主流程
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="🐒 Monkey Benchmark — Monte Carlo 随机选股基准测试"
    )
    parser.add_argument("--n-trials", type=int, default=1000,
                        help="随机模拟次数 (default: 1000)")
    parser.add_argument("--models", type=str, default=None,
                        help="要比较的真实模型名，逗号分隔 (default: 所有 enabled)")
    parser.add_argument("--yaml", type=str, default=None,
                        help="用于构建 dataset 的 workflow YAML "
                             "(default: 第一个 enabled 模型的 YAML)")
    parser.add_argument("--ic-only", action="store_true",
                        help="仅计算 IC/ICIR，跳过回测（更快）")
    parser.add_argument("--plot", action="store_true",
                        help="生成分布直方图")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Workspace 路径")
    parser.add_argument("--training-mode", type=str, default="static",
                        help="训练记录模式过滤 (default: static)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Workspace
    if args.workspace:
        env.set_root_dir(args.workspace)

    print(f"\n{'='*70}")
    print(f"🐒 Monkey Benchmark — Monte Carlo 随机选股基准测试")
    print(f"{'='*70}")
    print(f"Workspace: {env.ROOT_DIR}")
    print(f"Trials:    {args.n_trials}")
    print(f"Mode:      {'IC only' if args.ic_only else 'Full (IC + Backtest)'}")
    print()

    # 加载配置
    from quantpits.utils.config_loader import load_workspace_config
    model_config = load_workspace_config(ROOT_DIR)
    freq = model_config.get("freq", "week")
    rebalance_freq = 5 if freq == "week" else 1

    # 加载策略配置
    from quantpits.utils import strategy as st_module
    st_config = st_module.load_strategy_config()
    top_k = st_config["strategy"]["params"]["topk"]
    bt_config = st_module.get_backtest_config(st_config)
    cost_rate = (bt_config["exchange_kwargs"].get("open_cost", 0.0005)
                 + bt_config["exchange_kwargs"].get("close_cost", 0.0015))

    print(f"TopK:      {top_k}")
    print(f"Freq:      {freq} (rebalance every {rebalance_freq} days)")
    print(f"Cost Rate: {cost_rate:.4f}")

    # 初始化 Qlib
    env.init_qlib()

    # 加载训练记录
    record_file = os.path.join(ROOT_DIR, "latest_train_records.json")
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            train_records = json.load(f)
    else:
        train_records = {"models": {}}
        print("⚠️  未找到训练记录，将仅运行随机模拟（无真实模型对比）")

    # 确定要比较的真实模型
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",")]
    else:
        selected_models = list(train_records.get("models", {}).keys())

    # 获取真实模型指标
    real_metrics = {}
    if selected_models and train_records.get("models"):
        print(f"\n加载真实模型指标 ({len(selected_models)} 个)...")
        real_metrics = load_real_model_metrics(train_records, selected_models)
        for name, m in real_metrics.items():
            print(f"  [{name}] IC={m['ic']:.4f}, ICIR={m['icir']:.4f}")

    # 确定 dataset 范围 — 使用训练记录中的日期范围
    # 从模型的 pred.pkl 获取时间覆盖范围
    from qlib.data import D

    market = model_config.get("market", "csi300")
    instruments = D.instruments(market)
    instrument_list = D.list_instruments(instruments, as_list=True)

    # 获取时间范围
    test_start = model_config.get("test_start_time", model_config.get("backtest_start_time"))
    test_end = model_config.get("test_end_time", model_config.get("backtest_end_time"))

    if not test_start or not test_end:
        print("❌ 无法确定测试日期范围，请检查 model_config.json")
        sys.exit(1)

    print(f"\n测试区间: {test_start} ~ {test_end}")
    print(f"市场: {market} ({len(instrument_list)} 只股票)")

    # 构建日期和股票 index
    all_dates = D.calendar(start_time=test_start, end_time=test_end,
                           freq="day")
    all_dates = pd.DatetimeIndex(all_dates)

    # 加载标签和收益率（一次性）
    print("\n构建数据矩阵（一次性开销）...")
    t0 = time.time()
    label_series, returns_wide, bench_returns, common_dates = \
        load_label_and_returns(all_dates, instrument_list, freq=freq)
    print(f"数据准备完成: {time.time() - t0:.1f}s")
    print(f"  交易日: {len(common_dates)}")
    print(f"  股票数: {returns_wide.shape[1]}")
    print(f"  标签数: {label_series.notna().sum()}")

    # Monte Carlo 模拟
    print(f"\n开始 Monte Carlo 模拟 ({args.n_trials} trials)...")
    t0 = time.time()
    monkey_df = run_monte_carlo(
        n_trials=args.n_trials,
        label_series=label_series,
        returns_wide=returns_wide,
        bench_returns=bench_returns,
        common_dates=common_dates,
        top_k=top_k,
        cost_rate=cost_rate,
        rebalance_freq=rebalance_freq,
        ic_only=args.ic_only,
    )
    elapsed = time.time() - t0
    print(f"模拟完成: {elapsed:.1f}s "
          f"({elapsed/args.n_trials*1000:.1f}ms/trial)")

    # 输出结果
    print_results(monkey_df, real_metrics, ic_only=args.ic_only)

    # 保存结果
    output_dir = os.path.join(ROOT_DIR, "output", "monkey_benchmark")
    save_results_csv(monkey_df, output_dir)

    # 直方图
    if args.plot:
        print(f"\n生成分布图...")
        plot_distributions(monkey_df, real_metrics, output_dir,
                           ic_only=args.ic_only)

    print(f"\n{'='*70}")
    print(f"🐒 Done! Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
