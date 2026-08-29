#!/usr/bin/env python
"""
monkey_benchmark.py — 🐒 随机猴子 Monte Carlo 统计基准测试

高效批量模拟 N 次随机选股，生成统计分布，与真实模型的 IC/ICIR/年化收益/Sharpe/最大回撤
做严格假设检验。

关键设计（解决策略回测与动态成分股对齐问题）：
1. 精确对齐 TopkDropoutStrategy (TopK + DropN)：
   真实交易并非每期全量推倒重来，而是每期仅淘汰打分最靠后的 DropN 支股票，保留其余大部分持仓。
   本脚本精确还原生产环境中的 TopkDropout 调仓状态机，真实反映换手摩擦与持仓惯性。
2. 严格动态股票池（Dynamic Universe）：
   每一天仅针对当天属于目标市场指数的有效成分股打随机分与选股。
   出池股票自动触发强制淘汰，绝不引入全历史已退市或未进池标的造成的 0 收益稀释。
3. 收益与基准对齐：
   若 TopK 设为大于当日成分股总数的值，则每期全选当日所有有效成分股（等权持有），
   组合收益将完全等价于成分股等权全市场基准。

用法:
    # 跑 1000 个猴子，默认从 strategy_config.yaml 读取 TopK 与 DropN
    python quantpits/scripts/monkey_benchmark.py --n-trials 1000

    # 自定义 TopK 与 DropN
    python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --topk 20 --n-drop 3

    # 生成分布直方图
    python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --plot

    # 仅计算 IC/ICIR（跳过回测，超快）
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
# 路径与常量设置
# ---------------------------------------------------------------------------
from quantpits.utils import env
from quantpits.utils.constants import TRADING_DAYS_PER_YEAR


# ============================================================================
# Stage 1: 动态股票池与收益率矩阵构建
# ============================================================================

def load_market_data(market, test_start, test_end, freq="week", benchmark="SH000300"):
    """
    基于 Qlib 动态成分股获取特征、标签与日频收益率。
    
    保证：
    1. 在任意日期 t，仅获取当天处于 market（如 csi300）内的有效股票。
    2. 返回宽表（DatetimeIndex x InstrumentIndex），非当日成分股的值为 NaN。
    """
    from qlib.data import D

    print(f"\n--- [Stage 1] 加载动态股票池数据 (Market: {market}, 基准: {benchmark}, 日期: {test_start} ~ {test_end}) ---")
    
    # 获取动态市场对象（Qlib 内部会自动按日解析成员变更）
    instruments = D.instruments(market=market)

    # 1. 标签字段（周频预测未来 6 日收益，日频预测未来 2 日收益）
    ref_days = 6 if freq == "week" else 2
    label_field = f"Ref($close, -{ref_days})/Ref($close, -1) - 1"
    return_field = "Ref($close, -1)/$close - 1"

    print(f"  标签公式: {label_field}")
    print(f"  日频收益: {return_field}")

    # 拉取动态成分股的多列特征
    t0 = time.time()
    feat_df = D.features(
        instruments,
        [label_field, return_field],
        start_time=test_start,
        end_time=test_end,
    )
    print(f"  特征拉取完成: {time.time() - t0:.2f}s, 包含 {len(feat_df)} 条有效 (日期, 股票) 记录")

    # 转换为宽表 (T x N)
    label_wide = feat_df[label_field].unstack(level="instrument")
    returns_wide = feat_df[return_field].unstack(level="instrument")

    # 统一日期索引（仅取共同存在的有效交易日）
    common_dates = returns_wide.index.sort_values()
    label_wide = label_wide.reindex(common_dates)
    returns_wide = returns_wide.reindex(common_dates)

    # 有效成分股布尔掩码 (T x N)：True 代表当日在该市场池内且有行情
    valid_mask = ~returns_wide.isna().values
    
    daily_counts = valid_mask.sum(axis=1)
    print(f"  测试集交易日: {len(common_dates)} 天")
    print(f"  每日有效成分股数: 均值 {daily_counts.mean():.1f} 只 "
          f"(Min: {daily_counts.min()}, Max: {daily_counts.max()})")
    print(f"  全历史涉及标的总数: {returns_wide.shape[1]} 只")

    # 2. 加载基准指数收益率 (例如 SH000300)
    try:
        bench_df = D.features([benchmark], ["$close"], start_time=test_start, end_time=test_end)
        bench_close = bench_df["$close"]
        bench_returns = bench_close.pct_change(1).shift(-1)
        if hasattr(bench_returns.index, "get_level_values"):
            bench_returns.index = bench_returns.index.get_level_values("datetime")
        bench_returns = bench_returns.reindex(common_dates).fillna(0.0)
        print(f"  基准指数 ({benchmark}) 收益加载成功: {len(bench_returns)} 天")
    except Exception as e:
        print(f"  基准收益加载失败 ({e})，使用 0 替代")
        bench_returns = pd.Series(0.0, index=common_dates)

    return label_wide, returns_wide, valid_mask, bench_returns, common_dates


def load_real_model_metrics(train_records, selected_models=None):
    """从已保存的 MLflow / training_records 中读取真实模型的 IC / ICIR 指标。"""
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
            recorder = R.get_recorder(recorder_id=record_id, experiment_name=exp_name)

            ic_mean, icir, rank_ic, rank_icir = None, None, None, None

            # 读取 IC 指标
            try:
                ic_series = recorder.load_object("sig_analysis/ic.pkl")
                ic_m = float(ic_series.mean())
                ic_s = float(ic_series.std())
                ic_mean = ic_m
                icir = ic_m / ic_s if ic_s > 0 else 0.0
            except Exception:
                pass

            # 读取 Rank IC 指标
            try:
                ric_series = recorder.load_object("sig_analysis/ric.pkl")
                ric_m = float(ric_series.mean())
                ric_s = float(ric_series.std())
                rank_ic = ric_m
                rank_icir = ric_m / ric_s if ric_s > 0 else 0.0
            except Exception:
                pass

            if ic_mean is not None:
                metrics[model_name] = {
                    "ic": ic_mean,
                    "icir": icir,
                    "rank_ic": rank_ic if rank_ic is not None else ic_mean,
                    "rank_icir": rank_icir if rank_icir is not None else icir,
                }
        except Exception as e:
            print(f"  [{model_name}] 无法读取历史记录: {e}")

    return metrics


# ============================================================================
# Stage 2: 向量化 TopkDropout Monte Carlo 引擎
# ============================================================================

def compute_daily_ic_fast(scores_mat, label_mat, valid_mask):
    """
    按日计算 Pearson IC 与 Spearman Rank IC。
    
    仅在每日 valid_mask 为 True 且 label 不为 NaN 的有效成分股截面上计算。
    """
    T = scores_mat.shape[0]
    daily_ic = []
    daily_rank_ic = []

    for t in range(T):
        mask_t = valid_mask[t] & ~np.isnan(label_mat[t])
        n_valid = np.sum(mask_t)
        if n_valid < 5:
            continue

        s = scores_mat[t, mask_t]
        y = label_mat[t, mask_t]

        # 1. Pearson IC
        s_diff = s - np.mean(s)
        y_diff = y - np.mean(y)
        s_std = np.sqrt(np.sum(s_diff ** 2))
        y_std = np.sqrt(np.sum(y_diff ** 2))
        if s_std > 1e-8 and y_std > 1e-8:
            ic = np.sum(s_diff * y_diff) / (s_std * y_std)
            daily_ic.append(ic)

        # 2. Spearman Rank IC
        s_rank = s.argsort().argsort()
        y_rank = y.argsort().argsort()
        sr_diff = s_rank - np.mean(s_rank)
        yr_diff = y_rank - np.mean(y_rank)
        sr_std = np.sqrt(np.sum(sr_diff ** 2))
        yr_std = np.sqrt(np.sum(yr_diff ** 2))
        if sr_std > 1e-8 and yr_std > 1e-8:
            ric = np.sum(sr_diff * yr_diff) / (sr_std * yr_std)
            daily_rank_ic.append(ric)

    if len(daily_ic) == 0:
        return 0.0, 0.0, 0.0, 0.0

    ic_arr = np.array(daily_ic)
    ric_arr = np.array(daily_rank_ic)

    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0

    ric_mean = float(np.mean(ric_arr))
    ric_std = float(np.std(ric_arr))
    ricir = ric_mean / ric_std if ric_std > 1e-8 else 0.0

    return ic_mean, icir, ric_mean, ricir


def run_single_monkey_backtest(scores_mat, returns_mat, valid_mask, top_k, n_drop, cost_rate, rebalance_freq):
    """
    对单只猴子的随机排序执行 TopkDropout 回测。
    
    精确对齐 Qlib TopkDropoutStrategy 逻辑：
    - 初始期：买入 TopK 只股票。
    - 后续调仓日：
      1. 找出出池股票（强制淘汰）。
      2. 找出当前持仓中在当前截面打分最靠后的至多 (n_drop - n_forced) 只股票进行卖出。
      3. 从未持有的前排候选股票中买入补齐至 TopK。
      4. 严格按照实际买卖数量计算换手率并扣除摩擦成本。
    - 若 n_drop <= 0 或 n_drop >= top_k，则退化为全量调仓。
    """
    T, N = scores_mat.shape
    rebalance_indices = np.arange(0, T, rebalance_freq)

    reb_holdings_mask = np.zeros((len(rebalance_indices), N), dtype=bool)
    turnover_costs = np.zeros(T, dtype=np.float64)

    curr_held_set = set()

    for i, t in enumerate(rebalance_indices):
        valid_indices = np.where(valid_mask[t])[0]
        n_valid = len(valid_indices)
        if n_valid == 0:
            continue

        k = min(top_k, n_valid)
        scores_valid = scores_mat[t, valid_indices]
        
        # 按分数从高到低排序当前有效池中的标的
        sorted_rel_order = np.argsort(-scores_valid)
        ranked_candidates = valid_indices[sorted_rel_order]

        is_full_rebalance = (n_drop is None) or (n_drop is not None and n_drop < 0) or (n_drop is not None and n_drop >= k)

        if i == 0 or is_full_rebalance:
            # 初始建仓，或全量调仓模式
            new_held_set = set(ranked_candidates[:k])
        else:
            # TopK Dropout 模式 (含 n_drop=0 纯持有模式)
            valid_set = set(valid_indices)
            forced_exit = curr_held_set - valid_set
            eligible_held = curr_held_set & valid_set

            # 对仍有效的持仓按当前分数从低到高（最差在前）排序
            if len(eligible_held) > 0:
                held_arr = np.array(list(eligible_held))
                held_scores = scores_mat[t, held_arr]
                worst_to_best_held = held_arr[np.argsort(held_scores)]
            else:
                worst_to_best_held = np.array([], dtype=int)

            # 正常淘汰名额 = max(0, n_drop - len(forced_exit))
            normal_drop_count = max(0, n_drop - len(forced_exit))
            normal_dropped = set(worst_to_best_held[:normal_drop_count])

            kept_held = eligible_held - normal_dropped

            # 补齐缺口
            needed_buys = max(0, k - len(kept_held))
            new_buys = []
            if needed_buys > 0:
                for cand in ranked_candidates:
                    if cand not in kept_held:
                        new_buys.append(cand)
                        if len(new_buys) == needed_buys:
                            break

            new_held_set = kept_held | set(new_buys)

        # 换手率计算
        if i > 0 and cost_rate > 0:
            buys_count = len(new_held_set - curr_held_set)
            turnover = buys_count / max(k, 1)
            turnover_costs[t] = turnover * cost_rate

        curr_held_set = new_held_set
        reb_holdings_mask[i, list(curr_held_set)] = True

    # 广播到每一天
    day_to_reb_idx = np.clip(np.arange(T) // rebalance_freq, 0, len(rebalance_indices) - 1)
    daily_holdings_mask = reb_holdings_mask[day_to_reb_idx]

    # 计算每日持仓等权收益率
    daily_gross_returns = np.zeros(T, dtype=np.float64)
    for t in range(T):
        held = daily_holdings_mask[t]
        k_held = np.sum(held)
        if k_held > 0:
            ret_vals = returns_mat[t, held]
            valid_ret = ret_vals[~np.isnan(ret_vals)]
            if len(valid_ret) > 0:
                daily_gross_returns[t] = np.mean(valid_ret)

    # 净收益率
    net_returns = daily_gross_returns - turnover_costs
    return net_returns


def compute_performance_metrics(net_returns, bench_returns_arr):
    """计算净值指标：总收益、年化复合收益率 (CAGR)、Sharpe、最大回撤、超额收益等。"""
    days = len(net_returns)
    if days == 0:
        return {}

    nav = np.cumprod(1.0 + net_returns)
    final_nav = float(nav[-1])
    total_ret = final_nav - 1.0

    years = days / TRADING_DAYS_PER_YEAR
    ann_ret = (final_nav ** (1.0 / years)) - 1.0 if final_nav > 0 else -1.0

    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    max_dd = float(np.min(drawdown))

    std_ret = np.std(net_returns)
    sharpe = float(np.mean(net_returns) / std_ret * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_ret > 1e-8 else 0.0

    # 基准
    bench_nav = np.cumprod(1.0 + bench_returns_arr)
    bench_final = float(bench_nav[-1]) if len(bench_nav) > 0 else 1.0
    bench_total = bench_final - 1.0
    bench_cagr = (bench_final ** (1.0 / years)) - 1.0 if bench_final > 0 else -1.0

    bench_running_max = np.maximum.accumulate(bench_nav)
    bench_drawdown = (bench_nav - bench_running_max) / bench_running_max
    bench_max_dd = float(np.min(bench_drawdown))

    std_bench = np.std(bench_returns_arr)
    bench_sharpe = float(np.mean(bench_returns_arr) / std_bench * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_bench > 1e-8 else 0.0

    return {
        "ann_ret": float(ann_ret),
        "total_ret": float(total_ret),
        "max_dd": float(max_dd),
        "sharpe": float(sharpe),
        "excess_ret": float(total_ret - bench_total),
        "ann_excess": float(ann_ret - bench_cagr),
        "bench_cagr": float(bench_cagr),
        "bench_total": float(bench_total),
        "bench_max_dd": float(bench_max_dd),
        "bench_sharpe": float(bench_sharpe),
    }


def run_monte_carlo_trials(
    n_trials, label_mat, returns_mat, valid_mask, bench_returns_arr,
    top_k, n_drop, cost_rate, rebalance_freq, ic_only=False
):
    """批量执行 N 次独立猴子模拟。"""
    T, N = returns_mat.shape
    results = []
    rng = np.random.default_rng()

    for _ in tqdm(range(n_trials), desc="🐒 模拟猴子随机选股", unit="只"):
        # 1. 针对 (T, N) 生成均匀随机数 [0, 1)
        raw_scores = rng.uniform(0.0, 1.0, size=(T, N))

        # 2. 将非当日成分股的分数置为 -inf，彻底与当日有效池隔离
        scores = np.where(valid_mask, raw_scores, -np.inf)

        # 3. 计算 IC 指标
        ic_mean, icir, rank_ic, rank_icir = compute_daily_ic_fast(scores, label_mat, valid_mask)
        trial_dict = {
            "ic": ic_mean,
            "icir": icir,
            "rank_ic": rank_ic,
            "rank_icir": rank_icir,
        }

        # 4. 执行 TopkDropout 回测
        if not ic_only:
            net_returns = run_single_monkey_backtest(
                scores, returns_mat, valid_mask, top_k, n_drop, cost_rate, rebalance_freq
            )
            perf = compute_performance_metrics(net_returns, bench_returns_arr)
            trial_dict.update(perf)

        results.append(trial_dict)

    return pd.DataFrame(results)


# ============================================================================
# Stage 3: 统计报表与图表可视化
# ============================================================================

def format_distribution(series, pct=False):
    """格式化分布指标。"""
    fmt = lambda x: f"{x*100:+.2f}%" if pct else f"{x:+.4f}"
    return {
        "Mean": fmt(series.mean()),
        "Std": fmt(series.std()),
        "Median": fmt(series.median()),
        "5%ile": fmt(series.quantile(0.05)),
        "95%ile": fmt(series.quantile(0.95)),
        "Min": fmt(series.min()),
        "Max": fmt(series.max()),
    }


def print_comparison_report(monkey_df, real_metrics, top_k, n_drop, benchmark_name="SH000300", ic_only=False):
    """打印详细的统计检验结果报告。"""
    n = len(monkey_df)
    if n_drop == 0:
        drop_desc = "DropN=0 (零主动淘汰 / Buy and Hold 纯持有)"
    elif n_drop is not None and 0 < n_drop < top_k:
        drop_desc = f"DropN={n_drop} (每期淘汰打分最差的 {n_drop} 支)"
    else:
        drop_desc = "全量调仓 (Full Rebalance)"
    print(f"\n{'='*78}")
    print(f"🐒 Monkey Benchmark — {n} 次独立随机选股统计分析")
    print(f"{'='*78}")
    print(f"  策略参数: TopK={top_k}, {drop_desc}")
    if not ic_only and "bench_cagr" in monkey_df.columns:
        b_cagr = monkey_df["bench_cagr"].iloc[0]
        b_tot = monkey_df["bench_total"].iloc[0]
        b_shp = monkey_df.get("bench_sharpe", pd.Series([0.0])).iloc[0]
        b_mdd = monkey_df.get("bench_max_dd", pd.Series([0.0])).iloc[0]
        print(f"  同期基准 ({benchmark_name}): 年化收益 (CAGR)={b_cagr*100:+.2f}%, 累计收益={b_tot*100:+.2f}%, "
              f"Sharpe={b_shp:+.4f}, 最大回撤={b_mdd*100:+.2f}%")

    print(f"\n📊 [猴子基准统计分布 (Monte Carlo Distribution)]")
    
    cols_to_show = [
        ("ic", "IC (Pearson)", False),
        ("icir", "ICIR (Pearson)", False),
        ("rank_ic", "Rank IC (Spearman)", False),
        ("rank_icir", "Rank ICIR (Spearman)", False),
    ]
    if not ic_only and "ann_ret" in monkey_df.columns:
        cols_to_show.extend([
            ("ann_ret", "年化收益率 (CAGR)", True),
            ("total_ret", "累计总收益率 (Total Return)", True),
            ("sharpe", "Sharpe 比率", False),
            ("max_dd", "最大回撤 (Max Drawdown)", True),
            ("ann_excess", "年化超额收益 (Ann Excess vs Bench)", True),
        ])

    table_data = []
    for col, name, pct in cols_to_show:
        if col in monkey_df.columns:
            stats = format_distribution(monkey_df[col], pct=pct)
            table_data.append([
                name,
                f"{stats['Mean']} ± {stats['Std']}",
                stats["Median"],
                f"[{stats['5%ile']}, {stats['95%ile']}]",
                f"[{stats['Min']}, {stats['Max']}]",
            ])

    headers = ["指标", "均值 ± 标准差", "中位数", "90% 置信区间 (5%~95%)", "极端范围 (Min~Max)"]
    df_report = pd.DataFrame(table_data, columns=headers)
    print(df_report.to_string(index=False))

    # 与真实模型假设检验对比
    if real_metrics:
        print(f"\n{'='*78}")
        print("🎯 [真实模型 vs 猴子基准检验 (Hypothesis Testing)]")
        print(f"{'='*78}")

        comp_rows = []
        for model_name, m in sorted(real_metrics.items()):
            m_ic = m.get("ic", 0.0)
            m_icir = m.get("icir", 0.0)
            m_ric = m.get("rank_ic", m_ic)

            # 计算在猴子分布中的百分位与单侧 p-value
            ic_pct = (monkey_df["ic"] <= m_ic).mean() * 100.0
            ic_pval = (monkey_df["ic"] >= m_ic).mean()

            ric_pct = (monkey_df["rank_ic"] <= m_ric).mean() * 100.0
            ric_pval = (monkey_df["rank_ic"] >= m_ric).mean()

            # 判定结论
            sig = "✅ 极显著 (p<0.01)" if ic_pval < 0.01 else ("✅ 显著 (p<0.05)" if ic_pval < 0.05 else "❌ 未能打败猴子 (p≥0.05)")

            comp_rows.append([
                model_name,
                f"{m_ic:+.4f}",
                f"{m_icir:+.4f}",
                f"{m_ric:+.4f}",
                f"{ic_pct:5.1f}%",
                f"{ic_pval:.4f}",
                sig,
            ])

        m_headers = ["模型名称", "IC", "ICIR", "Rank IC", "IC 百分位", "p-value", "显著性结论"]
        df_models = pd.DataFrame(comp_rows, columns=m_headers)
        print(df_models.to_string(index=False))


def plot_distributions(monkey_df, real_metrics, output_dir, ic_only=False):
    """绘制分布直方图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n = len(monkey_df)

    metrics_map = [
        ("ic", "IC (Pearson)", False),
        ("icir", "ICIR (Pearson)", False),
        ("rank_ic", "Rank IC (Spearman)", False),
    ]
    if not ic_only and "ann_ret" in monkey_df.columns:
        metrics_map.extend([
            ("ann_ret", "Annualized Return (CAGR)", True),
            ("sharpe", "Sharpe Ratio", False),
            ("max_dd", "Max Drawdown", True),
        ])

    for col, title, is_pct in metrics_map:
        if col not in monkey_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        data = monkey_df[col].dropna()
        if is_pct:
            data = data * 100.0

        ax.hist(
            data, bins=40, alpha=0.75, color="#1976D2", edgecolor="#0D47A1",
            label=f"Monkeys (N={n})"
        )

        # 标注均值线与 95% 置信线
        q05 = data.quantile(0.05)
        q95 = data.quantile(0.95)
        ax.axvline(q05, color="#78909C", linestyle=":", label=f"5%ile ({q05:+.2f})")
        ax.axvline(q95, color="#78909C", linestyle=":", label=f"95%ile ({q95:+.2f})")

        # 标注真实模型
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(real_metrics), 1)))
        for i, (m_name, m_val) in enumerate(real_metrics.items()):
            val = m_val.get(col, m_val.get("ic" if "ic" in col else None))
            if val is not None:
                p_val = val * 100.0 if is_pct else val
                ax.axvline(
                    p_val, color=colors[i], linestyle="--", linewidth=2,
                    label=f"{m_name} ({p_val:+.4f})"
                )

        unit = " (%)" if is_pct else ""
        ax.set_xlabel(f"{title}{unit}")
        ax.set_ylabel("Count (Frequency)")
        ax.set_title(f"🐒 Monkey Benchmark: {title} Distribution (N={n})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, f"monkey_{col}_distribution.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  图表已保存: {save_path}")


# ============================================================================
# 主入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="🐒 Monkey Benchmark — 严格动态成分股与 TopkDropout 随机选股基准测试"
    )
    parser.add_argument("--n-trials", type=int, default=1000,
                        help="随机模拟次数 (default: 1000)")
    parser.add_argument("--models", type=str, default=None,
                        help="要比较的真实模型名，逗号分隔 (default: 训练记录中的所有模型)")
    parser.add_argument("--topk", type=int, default=None,
                        help="TopK 选股数量 (default: 读取 strategy_config.yaml)")
    parser.add_argument("--n-drop", type=int, default=None,
                        help="DropN 淘汰数量 (default: 读取 strategy_config.yaml)")
    parser.add_argument("--ic-only", action="store_true",
                        help="仅计算 IC/ICIR/Rank IC，跳过回测（超快）")
    parser.add_argument("--plot", action="store_true",
                        help="生成分布直方图")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Workspace 路径")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.workspace:
        env.set_root_dir(args.workspace)

    ROOT_DIR = env.ROOT_DIR
    os.chdir(ROOT_DIR)

    print(f"\n{'='*78}")
    print(f"🐒 QuantPits Monkey Benchmark — 随机猴子统计基准分析系统")
    print(f"{'='*78}")
    print(f"Workspace: {ROOT_DIR}")
    print(f"Trials:    {args.n_trials} 次独立随机实验")
    print(f"Mode:      {'IC / Rank IC 快速模式' if args.ic_only else '全指标模式 (IC + TopkDropout 组合回测)'}")

    # 1. 加载配置
    from quantpits.utils.config_loader import load_workspace_config
    model_config = load_workspace_config(ROOT_DIR)
    freq = model_config.get("freq", "week")
    rebalance_freq = 5 if freq == "week" else 1
    market = model_config.get("market", "csi300")
    benchmark = model_config.get("benchmark", "SH000300")

    # 2. 策略与回测参数
    from quantpits.utils import strategy as st_module
    st_config = st_module.load_strategy_config()
    strategy_params = st_config.get("strategy", {}).get("params", {})
    top_k = args.topk or strategy_params.get("topk", 20)
    n_drop = args.n_drop if args.n_drop is not None else strategy_params.get("n_drop", 3)
    
    bt_config = st_module.get_backtest_config(st_config)
    cost_rate = (bt_config["exchange_kwargs"].get("open_cost", 0.0005)
                 + bt_config["exchange_kwargs"].get("close_cost", 0.0015))

    print(f"Market:    {market}")
    print(f"Benchmark: {benchmark}")
    print(f"TopK:      {top_k}")
    print(f"DropN:     {n_drop} (每期最多淘汰最差的 {n_drop} 支股票)")
    print(f"Freq:      {freq} (每 {rebalance_freq} 个交易日调仓一次)")
    print(f"Cost Rate: 单边摩擦合计 {cost_rate*100:.2f}%")

    # 3. 初始化 Qlib
    env.init_qlib()

    # 4. 加载测试区间
    test_start = model_config.get("test_start_time", model_config.get("backtest_start_time"))
    test_end = model_config.get("test_end_time", model_config.get("backtest_end_time"))
    if not test_start or not test_end:
        print("❌ 无法从 model_config.json 确定 test_start_time / test_end_time！")
        sys.exit(1)

    # 5. 加载数据
    label_wide, returns_wide, valid_mask, bench_returns, common_dates = load_market_data(
        market=market, test_start=test_start, test_end=test_end, freq=freq, benchmark=benchmark
    )

    label_mat = label_wide.values.astype(np.float64)
    returns_mat = returns_wide.values.astype(np.float64)
    bench_arr = bench_returns.values.astype(np.float64)

    # 6. 读取已有真实模型指标
    record_file = os.path.join(ROOT_DIR, "latest_train_records.json")
    real_metrics = {}
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            train_records = json.load(f)
        sel_models = [m.strip() for m in args.models.split(",")] if args.models else None
        real_metrics = load_real_model_metrics(train_records, sel_models)
        if real_metrics:
            print(f"\n已成功载入 {len(real_metrics)} 个已训练真实模型指标用于对比:")
            for m_name, m_data in real_metrics.items():
                print(f"  • {m_name:24s}: IC={m_data['ic']:+.4f}, ICIR={m_data['icir']:+.4f}, "
                      f"Rank IC={m_data['rank_ic']:+.4f}")

    # 7. 运行 Monte Carlo 模拟
    print(f"\n--- [Stage 2] 执行 Monte Carlo 模拟 ({args.n_trials} 只随机猴子) ---")
    t0 = time.time()
    monkey_df = run_monte_carlo_trials(
        n_trials=args.n_trials,
        label_mat=label_mat,
        returns_mat=returns_mat,
        valid_mask=valid_mask,
        bench_returns_arr=bench_arr,
        top_k=top_k,
        n_drop=n_drop,
        cost_rate=cost_rate,
        rebalance_freq=rebalance_freq,
        ic_only=args.ic_only,
    )
    elapsed = time.time() - t0
    print(f"模拟完成: 耗时 {elapsed:.2f}s (平均每只猴子 {elapsed/args.n_trials*1000:.2f}ms)")

    # 8. 打印分析报告
    print_comparison_report(
        monkey_df=monkey_df,
        real_metrics=real_metrics,
        top_k=top_k,
        n_drop=n_drop,
        benchmark_name=benchmark,
        ic_only=args.ic_only,
    )

    # 9. 保存结果与图表
    output_dir = os.path.join(ROOT_DIR, "output", "monkey_benchmark")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "monkey_trials_distribution.csv")
    monkey_df.to_csv(csv_path, index=False)
    print(f"\n  完整模拟结果已保存: {csv_path}")

    if args.plot:
        print(f"\n--- [Stage 3] 绘制分布直方图 ---")
        plot_distributions(monkey_df, real_metrics, output_dir, ic_only=args.ic_only)

    print(f"\n{'='*78}")
    print(f"🐒 Benchmark Completed! 报告与图表已保存至: {output_dir}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
