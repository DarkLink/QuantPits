#!/usr/bin/env python
"""
Brute Force Fast - 向量化快速暴力穷举组合回测 + 结果分析

使用 NumPy/CuPy 矩阵运算替代 qlib 官方 backtest，速度提升约 5000 倍。
精度有所降低（忽略涨跌停、简化交易费用），但组合排序高度一致，
适合做粗筛，选出 Top 候选后用原版精确验证。

运行方式：
  cd QuantPits && python quantpits/scripts/brute_force_fast.py

常用命令：
  # 快速测试（最多 3 个模型的组合）
  python quantpits/scripts/brute_force_fast.py --max-combo-size 3

  # 仅分析已有结果（不重新跑回测）
  python quantpits/scripts/brute_force_fast.py --analysis-only

  # 从上次中断处继续
  python quantpits/scripts/brute_force_fast.py --resume

  # 只跑回测、跳过分析
  python quantpits/scripts/brute_force_fast.py --skip-analysis

  # 使用 GPU 加速
  python quantpits/scripts/brute_force_fast.py --use-gpu

  # 自定义交易费用（双边 0.3%）
  python quantpits/scripts/brute_force_fast.py --cost-rate 0.003
"""

import os
import sys
import json
import gc
import itertools
import logging
import argparse
import time
import pickle
from datetime import datetime
from collections import Counter
from itertools import chain
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
from quantpits.utils import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
# os.chdir(ROOT_DIR)  # 已在上方完成

# ---------------------------------------------------------------------------
# GPU 加速：自动检测 CuPy
# ---------------------------------------------------------------------------
_USE_GPU = False
xp = np  # default to numpy


def _init_gpu(force_gpu=False, force_no_gpu=False):
    """初始化 GPU 支持（CuPy）"""
    global _USE_GPU, xp

    if force_no_gpu:
        _USE_GPU = False
        xp = np
        print("GPU: 已禁用 (--no-gpu)")
        return

    try:
        import cupy as cp
        # 测试 GPU 是否可用
        cp.array([1.0])
        _USE_GPU = True
        xp = cp
        device = cp.cuda.Device()
        print(f"GPU: ✅ 已启用 CuPy (Device: {device.id}, "
              f"Memory: {device.mem_info[1] / 1024**3:.1f} GB)")
    except Exception as e:
        if force_gpu:
            print(f"GPU: ❌ CuPy 不可用: {e}")
            print("请安装 CuPy: pip install cupy-cuda12x")
            sys.exit(1)
        _USE_GPU = False
        xp = np
        print(f"GPU: 未启用 (CuPy 不可用，使用 NumPy)")


def _to_numpy(arr):
    """将 CuPy 数组转回 NumPy"""
    if _USE_GPU:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BACKTEST_CONFIG = {
    "account": 100_000_000,
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
}


# ============================================================================
# Stage 0: 初始化 & 配置加载
# ============================================================================

def init_qlib():
    """初始化 Qlib（委托给 env.init_qlib）"""
    env.init_qlib()


def load_config(record_file="latest_train_records.json"):
    """使用 config_loader 加载统一配置"""
    from quantpits.utils.config_loader import load_workspace_config
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            train_records = json.load(f)
    else:
        train_records = {"models": {}, "experiment_name": "unknown"}

    model_config = load_workspace_config(ROOT_DIR)

    return train_records, model_config


# ============================================================================
# Stage 1: 加载预测数据 + 收益率矩阵
# ============================================================================

def zscore_norm(series):
    """按天 Z-Score 归一化 (减均值，除标准差)"""
    def _norm_func(x):
        std = x.std()
        if std == 0:
            return x - x.mean()
        return (x - x.mean()) / std
    return series.groupby(level="datetime", group_keys=False).apply(_norm_func)


def load_predictions(train_records):
    """
    从 Qlib Recorder 加载所有模型的预测值，归一化后返回宽表。

    Returns:
        norm_df: DataFrame, index=(datetime, instrument), columns=model_names
        model_metrics: dict, model_name -> ICIR
    """
    from qlib.workflow import R

    experiment_name = train_records["experiment_name"]
    models = train_records["models"]

    all_preds = []
    model_metrics = {}

    print(f"\n{'='*60}")
    print("Stage 1: 加载模型预测数据")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Models ({len(models)}): {list(models.keys())}")

    for model_name, record_id in models.items():
        try:
            recorder = R.get_recorder(
                recorder_id=record_id, experiment_name=experiment_name
            )

            # 加载预测值
            pred = recorder.load_object("pred.pkl")
            if isinstance(pred, pd.Series):
                pred = pred.to_frame("score")
            pred.columns = [model_name]
            all_preds.append(pred)

            # 读取 ICIR 指标
            raw_metrics = recorder.list_metrics()
            metric_val = 0.0
            for k, v in raw_metrics.items():
                if "ICIR" in k:
                    metric_val = v
                    break
            model_metrics[model_name] = metric_val

            print(f"  [{model_name}] OK. Preds={len(pred)}, ICIR={metric_val:.4f}")
        except Exception as e:
            print(f"  [{model_name}] FAILED: {e}")

    print(f"\n成功加载 {len(all_preds)}/{len(models)} 个模型")

    if not all_preds:
        raise ValueError("未加载到任何预测数据！")

    # 合并 & Z-Score 归一化
    merged_df = pd.concat(all_preds, axis=1).dropna()
    print(f"合并后数据维度: {merged_df.shape}")

    norm_df = pd.DataFrame(index=merged_df.index)
    for col in merged_df.columns:
        norm_df[col] = zscore_norm(merged_df[col])

    return norm_df, model_metrics


def load_returns_matrix(norm_df, freq="week"):
    """
    从 qlib 加载个股**日频**收益率，并对齐到 norm_df 的索引。

    无论 freq 是 'week' 还是 'day'，都加载日频收益率。
    周频调仓逻辑在回测引擎中处理（每 rebalance_freq 天调仓一次），
    而不是在收益率矩阵中用前瞻 5 天收益率来模拟。

    返回:
        returns_wide: DataFrame, index=datetime, columns=instrument (日频收益率)
        benchmark_returns: Series, index=datetime (日频基准收益率)
        dates: DatetimeIndex
        instruments: list
    """
    from qlib.data import D

    print(f"\n--- 加载收益率数据 (交易频率={freq}, 收益率=日频) ---")

    # 获取唯一日期和股票列表
    dates = norm_df.index.get_level_values("datetime").unique().sort_values()
    instruments = norm_df.index.get_level_values("instrument").unique().tolist()

    start_date = str(dates.min().date())
    end_date = str(dates.max().date())

    # 始终加载日频收益率: 今日收盘 → 明日收盘
    # Ref($close, -1) 是明天的收盘价 (qlib 的 Ref 负数=未来)
    ret_df = D.features(
        instruments,
        ["Ref($close, -1)/$close - 1"],
        start_time=start_date,
        end_time=end_date,
    )
    ret_df.columns = ["return"]

    # 转为宽表 (datetime x instrument)
    returns_wide = ret_df["return"].unstack(level="instrument")

    # 只保留 norm_df 中存在的日期
    common_dates = dates.intersection(returns_wide.index)
    returns_wide = returns_wide.loc[common_dates]

    print(f"收益率矩阵维度: {returns_wide.shape} "
          f"(日期={returns_wide.shape[0]}, 股票={returns_wide.shape[1]})")

    # 加载基准日频收益率
    try:
        bench_df = D.features(
            ["SH000300"],
            ["$close"],
            start_time=start_date,
            end_time=end_date,
        )
        bench_close = bench_df["$close"]
        # 日频: 今日收盘 → 明日收盘
        bench_returns = bench_close.pct_change(1).shift(-1)

        # 对齐到交易日期
        if hasattr(bench_returns.index, "get_level_values"):
            bench_returns.index = bench_returns.index.get_level_values("datetime")
        bench_returns = bench_returns.reindex(common_dates)
        print(f"基准收益加载成功: {len(bench_returns.dropna())} 个交易日")
    except Exception as e:
        print(f"基准收益加载失败: {e}，使用 0 替代")
        bench_returns = pd.Series(0.0, index=common_dates)

    return returns_wide, bench_returns, common_dates, instruments


def prepare_matrices(norm_df, returns_wide, common_dates):
    """
    将 DataFrame 转换为对齐的 NumPy 矩阵。

    Returns:
        scores_np: dict, model_name -> (T, N) array
        returns_np: (T, N) array
        model_names: list
        date_index: DatetimeIndex
        instrument_index: Index
    """
    print("\n--- 构建矩阵 ---")

    model_names = list(norm_df.columns)

    # 将 norm_df 转为 per-model 宽表
    scores_np = {}
    for model in model_names:
        model_series = norm_df[model]
        model_wide = model_series.unstack(level="instrument")
        # 对齐到公共日期和股票
        common_instruments = returns_wide.columns.intersection(model_wide.columns)
        model_aligned = model_wide.reindex(
            index=common_dates, columns=common_instruments
        )
        scores_np[model] = model_aligned.values.astype(np.float32)

    # 对齐收益率矩阵
    common_instruments = returns_wide.columns
    for model in model_names:
        model_wide = norm_df[model].unstack(level="instrument")
        common_instruments = common_instruments.intersection(model_wide.columns)

    returns_aligned = returns_wide[common_instruments].reindex(common_dates)
    returns_np = returns_aligned.values.astype(np.float32)

    # 重新对齐 scores
    for model in model_names:
        model_wide = norm_df[model].unstack(level="instrument")
        scores_np[model] = model_wide.reindex(
            index=common_dates, columns=common_instruments
        ).values.astype(np.float32)

    # NaN 处理：将 NaN 收益率设为 0，NaN 分数设为 -inf（不被选中）
    nan_mask = np.isnan(returns_np)
    returns_np = np.nan_to_num(returns_np, nan=0.0)
    for model in model_names:
        scores_np[model] = np.where(
            nan_mask | np.isnan(scores_np[model]),
            -np.inf,
            scores_np[model],
        )

    print(f"最终矩阵维度: T={returns_np.shape[0]} 天, "
          f"N={returns_np.shape[1]} 只股票, "
          f"M={len(model_names)} 个模型")

    return scores_np, returns_np, model_names, common_dates, common_instruments


def split_is_oos_by_args(norm_df, args):
    """根据参数将 norm_df 划分为 IS 和 OOS，委托给 search_utils"""
    from quantpits.utils.search_utils import split_is_oos_by_args as _split
    return _split(norm_df, args)


# ============================================================================
# Stage 2: 相关性分析
# ============================================================================

def correlation_analysis(norm_df, output_dir, anchor_date):
    """计算并保存预测值相关性矩阵"""
    print(f"\n{'='*60}")
    print("Stage 2: 相关性分析")
    print(f"{'='*60}")

    corr_matrix = norm_df.corr()
    print("\n=== 模型预测相关性矩阵 ===")
    print(corr_matrix.round(4))

    # 保存
    corr_path = os.path.join(output_dir, f"correlation_matrix_{anchor_date}.csv")
    corr_matrix.to_csv(corr_path)
    print(f"\n相关性矩阵已保存: {corr_path}")

    return corr_matrix


# ============================================================================
# Stage 2.5: 模型分组 & 组合生成
# ============================================================================

def load_combo_groups(group_config_path, available_models):
    """加载分组配置，委托给 search_utils"""
    from quantpits.utils.search_utils import load_combo_groups as _load
    return _load(group_config_path, available_models)


def generate_grouped_combinations(groups, min_combo_size=1, max_combo_size=0):
    """基于分组生成组合，委托给 search_utils"""
    from quantpits.utils.search_utils import generate_grouped_combinations as _gen
    return _gen(groups, min_combo_size, max_combo_size)


# ============================================================================
# Stage 3: 向量化快速回测 (核心)
# ============================================================================

def _vectorized_topk_backtest_single(
    combo_score_np, returns_np, top_k, cost_rate, rebalance_freq=5,
):
    """
    对单个组合进行向量化回测 — 周频调仓 + 日频净值。

    模拟 qlib 的 TopkDropoutStrategy + SimulatorExecutor 行为：
    - 每 rebalance_freq 个交易日做一次 TopK 选股（调仓）
    - 非调仓日保持上一期的持仓不变
    - 每个交易日按日频收益率更新组合收益
    - 换手费用仅在调仓日扣除

    Args:
        combo_score_np: (T, N) 融合信号矩阵
        returns_np: (T, N) 日频收益率矩阵
        top_k: TopK 持仓数
        cost_rate: 单次换手交易费用率
        rebalance_freq: 调仓频率(交易日数), week=5, day=1

    Returns:
        numpy array: (T,) 每日净收益率序列
    """
    T, N = combo_score_np.shape
    k = min(top_k, N)

    # 预计算每天的 TopK（仅在调仓日计算，节省 argpartition 开销）
    rebalance_indices = xp.arange(0, T, rebalance_freq)
    combo_score_reb = combo_score_np[rebalance_indices]
    
    if k < N:
        topk_reb = xp.argpartition(-combo_score_reb, k, axis=1)[:, :k]
    else:
        topk_reb = xp.tile(xp.arange(N), (len(rebalance_indices), 1))

    # 将调仓日的持仓广播到每一天: day_to_reb_idx 将 [0..T-1] 映射到相应的调仓期索引
    day_to_reb_idx = xp.arange(T) // rebalance_freq
    actual_holdings = topk_reb[day_to_reb_idx]

    # 每天都用当前持仓计算日收益
    row_indices = xp.arange(T)[:, None]
    daily_returns = xp.mean(returns_np[row_indices, actual_holdings], axis=1)

    turnover_costs = xp.zeros(T, dtype=xp.float32)

    # 计算换手费用（仅在有两个及以上调仓期时才可能有换手）
    if cost_rate > 0 and len(rebalance_indices) > 1:
        # 仅针对调仓日构建 boolean mask 来计算集合差集
        mask_reb = xp.zeros((len(rebalance_indices), N), dtype=bool)
        row_indices_reb = xp.arange(len(rebalance_indices))[:, None]
        mask_reb[row_indices_reb, topk_reb] = True
        
        # curr & ~prev 即为新增持仓
        mask_curr = mask_reb[1:]
        mask_prev = mask_reb[:-1]
        
        turnovers = xp.sum(mask_curr & ~mask_prev, axis=1) / k
        # 将换手费用记录在发生调仓的当天 (索引 1 开始的 rebalance_indices)
        turnover_costs[rebalance_indices[1:]] = turnovers * cost_rate

    # 扣除交易费用
    net_returns = daily_returns - turnover_costs

    return _to_numpy(net_returns)


def _vectorized_topk_backtest_batch(
    combo_scores_batch, returns_np, top_k, cost_rate, rebalance_freq=5,
):
    """
    批量向量化回测（多个组合同时处理）。

    Args:
        combo_scores_batch: list of (T, N) arrays - 多个组合的融合信号
        returns_np: (T, N) 日频收益率矩阵
        top_k: TopK
        cost_rate: 交易费用率
        rebalance_freq: 调仓频率(交易日数)

    Returns:
        list of numpy arrays: 每个组合的日频净收益率序列
    """
    results = []
    for combo_score in combo_scores_batch:
        ret = _vectorized_topk_backtest_single(
            combo_score, returns_np, top_k, cost_rate, rebalance_freq,
        )
        results.append(ret)
    return results


def compute_metrics(net_returns, bench_returns_np, freq="day"):
    """
    从日频净收益率序列计算绩效指标。

    注意：net_returns 現在始终是日频的（即使是周频调仓策略），
    因为我们在回测引擎中按日累计净值。periods 固定为 252。

    Args:
        net_returns: (T,) 日频组合净收益率
        bench_returns_np: (T,) 日频基准收益率
        freq: 保留参数，始终使用 252 (日频)

    Returns:
        dict: 指标字典
    """
    # 收益率序列始终是日频，计算年数时始终使用 252
    periods_per_year = 252
    days = len(net_returns)
    if days == 0:
        return {}

    # 净值曲线
    nav = np.cumprod(1 + net_returns)
    final_nav = nav[-1]
    total_ret = final_nav - 1.0

    # 几何年化收益 (CAGR)
    years = days / periods_per_year
    ann_ret = (final_nav ** (1 / years)) - 1 if final_nav > 0 else -1.0
    

    # 最大回撤
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    max_dd = np.min(drawdown)

    # 基准收益
    bench_nav = np.cumprod(1 + bench_returns_np)
    bench_final_nav = bench_nav[-1] if len(bench_nav) > 0 else 1.0
    bench_ret_total = bench_final_nav - 1.0
    bench_cagr = (bench_final_nav ** (1 / years)) - 1 if bench_final_nav > 0 else -1.0

    # 超额 (几何总超额 vs 几何年化超额)
    excess_ret = total_ret - bench_ret_total
    ann_excess = ann_ret - bench_cagr

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Sharpe (简化版，维持日频波动率年化)
    periods = 252 # 波动率计算通常用日频基准
    if np.std(net_returns) > 0:
        sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(periods)
    else:
        sharpe = 0

    return {
        "Ann_Ret": ann_ret,
        "Max_DD": max_dd,
        "Excess_Ret": excess_ret,
        "Ann_Excess": ann_excess,
        "Total_Ret": total_ret,
        "Final_NAV": final_nav * BACKTEST_CONFIG["account"],
        "Calmar": calmar,
        "Sharpe": sharpe,
    }


def brute_force_fast_backtest(
    scores_np, returns_np, model_names, bench_returns_np,
    top_k, freq, cost_rate, batch_size,
    min_combo_size, max_combo_size, output_dir, anchor_date,
    resume=False, rebalance_freq=5, use_groups=False, group_config=None,
):
    """
    向量化快速暴力穷举所有模型组合并回测。

    Returns:
        results_df: DataFrame，所有组合的回测结果
    """
    print(f"\n{'='*60}")
    print("Stage 3: 向量化快速回测")
    print(f"{'='*60}")

    max_size = max_combo_size if max_combo_size > 0 else len(model_names)
    max_size = min(max_size, len(model_names))

    # ── 生成组合 ──
    if use_groups and group_config:
        print(f"\n📦 分组穷举模式 (配置: {group_config})")
        groups = load_combo_groups(group_config, model_names)
        print(f"有效分组 ({len(groups)}个):")
        total_product = 1
        for gname, models in groups.items():
            print(f"  [{gname}] ({len(models)}个): {models}")
            total_product *= len(models)

        all_combinations = generate_grouped_combinations(
            groups, min_combo_size, max_combo_size
        )
        print(f"\n分组笛卡尔积组合数: {len(all_combinations)}")
    else:
        print(f"待穷举模型 ({len(model_names)}个): {model_names}")
        print(f"组合大小范围: {min_combo_size} ~ {max_size}")
        all_combinations = []
        for r in range(min_combo_size, max_size + 1):
            all_combinations.extend(itertools.combinations(model_names, r))

    print(f"调仓频率: 每 {rebalance_freq} 个交易日, TopK={top_k}")
    print(f"交易费用率: {cost_rate:.4f}")
    print(f"批处理大小: {batch_size}")
    print(f"收益率: 日频, 净值: 日频累计")
    print(f"计算后端: {'CuPy (GPU)' if _USE_GPU else 'NumPy (CPU)'}")

    print(f"总组合数: {len(all_combinations)}")

    # Resume: 加载已有结果
    csv_path = os.path.join(output_dir, f"brute_force_fast_results_{anchor_date}.csv")
    done_combos = set()
    existing_results = []

    if resume and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_results = existing_df.to_dict("records")
        done_combos = set(existing_df["models"].tolist())
        print(f"Resume 模式: 已有 {len(done_combos)} 个组合，跳过")

    # 过滤已完成的组合
    pending = [
        c for c in all_combinations if ",".join(c) not in done_combos
    ]
    print(f"待回测组合数: {len(pending)}")

    if not pending:
        print("所有组合已完成！")
        results_df = pd.DataFrame(existing_results)
    else:
        # 将数据移到 GPU（如果启用）
        returns_gpu = xp.asarray(returns_np)
        scores_gpu = {m: xp.asarray(v) for m, v in scores_np.items()}

        results = list(existing_results)
        t0 = time.time()

        # 分批处理
        for batch_start in tqdm(
            range(0, len(pending), batch_size),
            desc="Fast Backtesting",
            total=(len(pending) + batch_size - 1) // batch_size,
        ):
            batch_combos = pending[batch_start : batch_start + batch_size]

            # 为每个组合计算融合信号
            batch_scores = []
            for combo in batch_combos:
                # 等权融合
                combo_score = sum(scores_gpu[m] for m in combo) / len(combo)
                batch_scores.append(combo_score)

            # 批量回测
            batch_returns = _vectorized_topk_backtest_batch(
                batch_scores, returns_gpu, top_k, cost_rate, rebalance_freq,
            )

            # 计算指标
            for combo, net_ret in zip(batch_combos, batch_returns):
                metrics = compute_metrics(net_ret, bench_returns_np, freq)
                metrics["models"] = ",".join(combo)
                metrics["n_models"] = len(combo)
                results.append(metrics)

            # 定期保存 & GC
            if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
                gc.collect()

        elapsed = time.time() - t0
        speed = len(pending) / elapsed if elapsed > 0 else 0
        print(f"\n回测速度: {speed:.1f} 组合/秒 (共 {elapsed:.1f} 秒)")

        # 清理 GPU 内存
        if _USE_GPU:
            del returns_gpu, scores_gpu
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()

        results_df = pd.DataFrame(results)

    # 排序并保存
    if not results_df.empty:
        results_df = results_df.sort_values(
            by="Ann_Excess", ascending=False
        ).reset_index(drop=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\n穷举完成！结果已保存至: {csv_path}")
        print(f"有效组合数: {len(results_df)}")
    else:
        print("警告: 无有效回测结果")

    return results_df


# ============================================================================
# Main
# ============================================================================

def main():
    from quantpits.utils import env
    env.safeguard("Brute Force Fast")
    parser = argparse.ArgumentParser(
        description="⚡ 快速向量化暴力穷举组合回测 + 结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试 (最多 3 个模型)
  python quantpits/scripts/brute_force_fast.py --max-combo-size 3

  # 使用 GPU 加速
  python quantpits/scripts/brute_force_fast.py --use-gpu

  # 从中断处继续
  python quantpits/scripts/brute_force_fast.py --resume
        """,
    )
    parser.add_argument(
        "--record-file", type=str, default="latest_train_records.json",
        help="训练记录文件路径 (默认: latest_train_records.json)",
    )
    parser.add_argument(
        "--training-mode", type=str, default=None,
        choices=["static", "rolling"],
        help="训练模式过滤 (默认 None=全部，static 或 rolling)",
    )
    parser.add_argument(
        "--max-combo-size", type=int, default=0,
        help="最大组合大小 (0=全部, 默认: 0)",
    )
    parser.add_argument(
        "--min-combo-size", type=int, default=1,
        help="最小组合大小 (默认: 1)",
    )
    parser.add_argument(
        "--freq", type=str, default=None, choices=["day", "week"],
        help="回测交易频率 (默认: 从 model_config 读取)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/brute_force_fast",
        help="输出目录 (默认: output/brute_force_fast)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从已有 CSV 继续 (跳过已完成的组合)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="回测开始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="回测结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--exclude-last-years", type=int, default=0,
        help="在 IS 阶段排除最后 N 年的数据（留作 OOS）",
    )
    parser.add_argument(
        "--exclude-last-months", type=int, default=0,
        help="在 IS 阶段排除最后 N 个月的数据（留作 OOS）",
    )
    parser.add_argument(
        "--use-groups", action="store_true",
        help="使用模型分组遍历 (同组只选一个)",
    )
    parser.add_argument(
        "--group-config", type=str, default="config/combo_groups.yaml",
        help="分组配置文件路径 (默认: config/combo_groups.yaml)",
    )
    # 快速模式特有参数
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="批量处理大小 (默认: 512)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="强制使用 GPU (CuPy)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="强制禁用 GPU",
    )
    parser.add_argument(
        "--cost-rate", type=float, default=0.002,
        help="单次换手交易费用率 (默认: 0.002 = 双边 0.2%%)",
    )
    args = parser.parse_args()

    # 初始化
    print("=" * 60)
    print("⚡ Brute Force Fast - 向量化快速暴力穷举回测")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # GPU 初始化
    _init_gpu(force_gpu=args.use_gpu, force_no_gpu=args.no_gpu)

    init_qlib()

    # Stage 0: 初始化 & 配置加载
    train_records, model_config = load_config(args.record_file)
    anchor_date = train_records.get("anchor_date", "unknown")
    
    # 确定频率 (优先级: 命令行参数 > model_config > 默认 week)
    freq = args.freq or model_config.get("freq", "week")
    args.freq = freq
    print(f"当前交易频率: {freq}")
    top_k = model_config.get("TopK", 22)
    drop_n = model_config.get("DropN", 3)
    benchmark = model_config.get("benchmark", "SH000300")

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: 加载预测数据 (应用 training-mode 过滤)
    if getattr(args, 'training_mode', None):
        from quantpits.utils.train_utils import filter_models_by_mode
        filtered = filter_models_by_mode(train_records.get('models', {}), args.training_mode)
        train_records = dict(train_records)
        train_records['models'] = filtered
        print(f"训练模式过滤: {args.training_mode} (剩余 {len(filtered)} 个模型)")
    norm_df, model_metrics = load_predictions(train_records)

    # 划分数据集 (IS / OOS)
    is_norm_df, oos_norm_df = split_is_oos_by_args(norm_df, args)
    if is_norm_df.empty:
        print("错误: IS 期无数据！请检查日期参数。")
        sys.exit(1)

    # Stage 2: 相关性分析
    corr_matrix = correlation_analysis(is_norm_df, args.output_dir, anchor_date)

    # 确定调仓频率 (周频=5天, 日频=1天)
    rebalance_freq = 5 if freq == "week" else 1

    # 加载日频收益率矩阵 (无论 freq 是什么都加载日频，全量加载)
    returns_wide, bench_returns, common_dates, instruments = load_returns_matrix(
        norm_df, freq=freq
    )

    # 构建 IS 对齐矩阵
    is_dates = is_norm_df.index.get_level_values("datetime").unique().sort_values()
    is_common_dates = is_dates.intersection(returns_wide.index)
    
    scores_np, returns_np, model_names, date_index, inst_index = prepare_matrices(
        is_norm_df, returns_wide, is_common_dates
    )

    # 对齐基准收益 (IS)
    bench_returns_np = bench_returns.reindex(date_index).fillna(0).values.astype(np.float32)

    # Stage 3: 快速回测
    results_df = brute_force_fast_backtest(
        scores_np=scores_np,
        returns_np=returns_np,
        model_names=model_names,
        bench_returns_np=bench_returns_np,
        top_k=top_k,
        freq=args.freq,
        cost_rate=args.cost_rate,
        batch_size=args.batch_size,
        min_combo_size=args.min_combo_size,
        max_combo_size=args.max_combo_size,
        output_dir=args.output_dir,
        anchor_date=anchor_date,
        resume=args.resume,
        rebalance_freq=rebalance_freq,
        use_groups=args.use_groups,
        group_config=args.group_config,
    )

    # 导出 Metadata
    metadata_path = os.path.join(args.output_dir, f"run_metadata_{anchor_date}.json")
    is_dates_all = is_norm_df.index.get_level_values("datetime")
    oos_dates_all = oos_norm_df.index.get_level_values("datetime")
    
    import json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "anchor_date": anchor_date,
            "script_used": "brute_force_fast",
            "freq": args.freq,
            "cost_rate": args.cost_rate,
            "record_file": args.record_file,
            "training_mode": getattr(args, "training_mode", None),
            "is_start_date": str(is_dates_all.min().date()) if not is_dates_all.empty else None,
            "is_end_date": str(is_dates_all.max().date()) if not is_dates_all.empty else None,
            "oos_start_date": str(oos_dates_all.min().date()) if not oos_dates_all.empty else None,
            "oos_end_date": str(oos_dates_all.max().date()) if not oos_dates_all.empty else None,
            "exclude_last_years": args.exclude_last_years,
            "exclude_last_months": args.exclude_last_months,
            "rebalance_freq": rebalance_freq,
            "use_groups": args.use_groups,
            "group_config": args.group_config
        }, f, indent=4)
    print(f"\n✅ 元数据已保存: {metadata_path}")
    print("请使用 quantpits/scripts/analyze_ensembles.py 单独进行分析与 OOS 验证。")

    print(f"\n{'='*60}")
    print(f"全部完成！ 耗时结束于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
