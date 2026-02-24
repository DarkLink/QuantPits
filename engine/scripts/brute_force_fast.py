#!/usr/bin/env python
"""
Brute Force Fast - 向量化快速暴力穷举组合回测 + 结果分析

使用 NumPy/CuPy 矩阵运算替代 qlib 官方 backtest，速度提升约 5000 倍。
精度有所降低（忽略涨跌停、简化交易费用），但组合排序高度一致，
适合做粗筛，选出 Top 候选后用原版精确验证。

运行方式：
  cd QuantPits && python engine/scripts/brute_force_fast.py

常用命令：
  # 快速测试（最多 3 个模型的组合）
  python engine/scripts/brute_force_fast.py --max-combo-size 3

  # 仅分析已有结果（不重新跑回测）
  python engine/scripts/brute_force_fast.py --analysis-only

  # 完整穷举 + 分析
  python engine/scripts/brute_force_fast.py

  # 从上次中断处继续
  python engine/scripts/brute_force_fast.py --resume

  # 只跑回测、跳过分析
  python engine/scripts/brute_force_fast.py --skip-analysis

  # 使用 GPU 加速
  python engine/scripts/brute_force_fast.py --use-gpu

  # 自定义交易费用（双边 0.3%）
  python engine/scripts/brute_force_fast.py --cost-rate 0.003
"""

import os
import sys
import json
import gc
import itertools
import logging
import argparse
import time
from datetime import datetime
from collections import Counter
from itertools import chain

import yaml
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
import env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
os.chdir(ROOT_DIR)

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
    """初始化 Qlib"""
    import qlib
    from qlib.constant import REG_CN
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)


def load_config(record_file="latest_train_records.json"):
    """加载训练记录和模型配置"""
    with open(record_file, "r") as f:
        train_records = json.load(f)

    with open("config/model_config.json", "r") as f:
        model_config = json.load(f)

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
    """根据参数将 norm_df 划分为 IS (In-Sample) 和 OOS (Out-of-Sample)"""
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    
    dates = norm_df.index.get_level_values("datetime").unique().sort_values()
    max_date = dates.max()
    
    cutoff_date = max_date
    if args.exclude_last_years > 0:
        cutoff_date = cutoff_date - pd.DateOffset(years=args.exclude_last_years)
    if args.exclude_last_months > 0:
        cutoff_date = cutoff_date - pd.DateOffset(months=args.exclude_last_months)
        
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    if end_date and end_date < cutoff_date:
        cutoff_date = end_date
        
    is_mask = norm_df.index.get_level_values("datetime") <= cutoff_date
    if start_date:
        is_mask &= norm_df.index.get_level_values("datetime") >= start_date
        
    is_norm_df = norm_df[is_mask]
    
    oos_mask = norm_df.index.get_level_values("datetime") > cutoff_date
    oos_norm_df = norm_df[oos_mask]
    
    print(f"\n=== 数据集划分 (In-Sample / Out-Of-Sample) ===")
    if not is_norm_df.empty:
        print(f"IS 期  : {is_norm_df.index.get_level_values('datetime').min().date()} ~ {is_norm_df.index.get_level_values('datetime').max().date()} (共 {len(is_norm_df.index.get_level_values('datetime').unique())} 天)")
    else:
        print("IS 期  : 无数据")
        
    if not oos_norm_df.empty:
        print(f"OOS 期 : {oos_norm_df.index.get_level_values('datetime').min().date()} ~ {oos_norm_df.index.get_level_values('datetime').max().date()} (共 {len(oos_norm_df.index.get_level_values('datetime').unique())} 天)")
    else:
        print("OOS 期 : 无数据")
        
    return is_norm_df, oos_norm_df


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
    """
    加载分组配置，验证模型名，返回有效分组。

    Args:
        group_config_path: combo_groups.yaml 路径
        available_models: 当前加载到的模型列表 (norm_df.columns)

    Returns:
        groups: dict, group_name -> list of valid model names
    """
    with open(group_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_groups = cfg.get("groups", {})
    if not raw_groups:
        raise ValueError(f"分组配置为空: {group_config_path}")

    available_set = set(available_models)
    groups = {}
    skipped_models = []

    for gname, models in raw_groups.items():
        valid = [m for m in models if m in available_set]
        invalid = [m for m in models if m not in available_set]
        if invalid:
            skipped_models.extend(invalid)
            print(f"  ⚠️  组 [{gname}] 中以下模型不存在于预测数据中，已忽略: {invalid}")
        if valid:
            groups[gname] = valid
        else:
            print(f"  ⚠️  组 [{gname}] 无有效模型，已跳过")

    if skipped_models:
        print(f"  共忽略 {len(skipped_models)} 个无效模型")

    # 检查未分组的模型 (仅打印提示，不自动参与)
    grouped_models = set()
    for models in groups.values():
        grouped_models.update(models)
    ungrouped = available_set - grouped_models
    if ungrouped:
        print(f"  ℹ️  以下模型未在任何分组中，将被排除: {sorted(ungrouped)}")

    return groups


def generate_grouped_combinations(groups, min_combo_size=1, max_combo_size=0):
    """
    基于分组生成组合：从所有组的子集中，每组选一个模型，做笛卡尔积。

    为支持 min/max combo size，我们枚举组的子集（选哪些组参与），
    然后对参与的组做 itertools.product。

    Args:
        groups: dict, group_name -> list of models
        min_combo_size: 最小组合大小 (选几个组)
        max_combo_size: 最大组合大小 (0=全部组)

    Returns:
        list of tuples, 每个 tuple 是一个模型组合
    """
    group_names = list(groups.keys())
    n_groups = len(group_names)
    max_size = max_combo_size if max_combo_size > 0 else n_groups
    max_size = min(max_size, n_groups)

    all_combinations = []

    # 枚举选哪些组参与 (选 r 个组的组合)
    for r in range(min_combo_size, max_size + 1):
        for group_subset in itertools.combinations(group_names, r):
            # 对选中的组做笛卡尔积
            model_lists = [groups[g] for g in group_subset]
            for combo in itertools.product(*model_lists):
                all_combinations.append(combo)

    return all_combinations


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
    topk_reb = xp.argpartition(-combo_score_reb, k, axis=1)[:, :k]

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
    # 收益率序列始终是日频，用 252 年化
    periods = 252

    # 净值曲线
    nav = np.cumprod(1 + net_returns)
    final_nav = nav[-1] if len(nav) > 0 else 1.0
    total_ret = final_nav - 1.0

    # 年化收益
    ann_ret = np.mean(net_returns) * periods

    # 最大回撤
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    max_dd = np.min(drawdown)

    # 基准收益
    bench_ret_total = np.prod(1 + bench_returns_np) - 1.0

    # 超额
    excess_ret = total_ret - bench_ret_total
    ann_excess = ann_ret - (np.mean(bench_returns_np) * periods)

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Sharpe (简化版)
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
# Stage 4: 结果分析 (复用原版逻辑)
# ============================================================================

def analyze_results(
    results_df, corr_matrix, norm_df, train_records, output_dir, anchor_date, top_n=50,
):
    """对暴力穷举结果进行全面分析"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n{'='*60}")
    print("Stage 4: 结果分析")
    print(f"{'='*60}")

    if results_df.empty:
        print("无数据可分析！")
        return

    # 预处理
    df = results_df.copy()
    df["model_list"] = df["models"].apply(lambda x: x.split(","))
    df["is_single"] = df["n_models"] == 1

    report_lines = []  # 收集报告文本

    # -----------------------------------------------------------------------
    # 4.1 Top 组合展示
    # -----------------------------------------------------------------------
    display_cols = ["models", "n_models", "Ann_Ret", "Max_DD", "Ann_Excess", "Calmar"]
    available_cols = [c for c in display_cols if c in df.columns]
    fmt = {
        "Ann_Ret": "{:.2%}".format,
        "Max_DD": "{:.2%}".format,
        "Ann_Excess": "{:.2%}".format,
        "Calmar": "{:.2f}".format,
    }
    fmt = {k: v for k, v in fmt.items() if k in df.columns}

    print("\n🏆 === 综合表现最佳 Top 20 (按年化超额) ===")
    top20_str = df[available_cols].head(20).to_string(formatters=fmt)
    print(top20_str)
    report_lines.append("=== Top 20 (按年化超额) ===\n" + top20_str)

    robust_df = df[df["Ann_Ret"] > 0.10].sort_values(by="Max_DD", ascending=False)
    if not robust_df.empty:
        print("\n🛡️ === 最稳健组合 Top 10 (年化>10%, 按回撤排序) ===")
        robust_str = robust_df[available_cols].head(10).to_string(formatters=fmt)
        print(robust_str)
        report_lines.append("\n=== 最稳健组合 Top 10 ===\n" + robust_str)

    # -----------------------------------------------------------------------
    # 4.2 模型归因分析 (Model Attribution)
    # -----------------------------------------------------------------------
    print(f"\n📊 === 模型归因分析 (Top/Bottom {top_n}) ===")

    top_combinations = df.sort_values("Calmar", ascending=False).head(top_n)
    bottom_combinations = df.sort_values("Calmar", ascending=True).head(top_n)

    def get_model_counts(series_of_lists):
        all_models = list(chain.from_iterable(series_of_lists))
        return pd.Series(Counter(all_models)).sort_values(ascending=False)

    top_counts = get_model_counts(top_combinations["model_list"])
    bottom_counts = get_model_counts(bottom_combinations["model_list"])

    attribution = pd.DataFrame(
        {"Top_Count": top_counts, "Bottom_Count": bottom_counts}
    ).fillna(0)
    attribution["Net_Score"] = attribution["Top_Count"] - attribution["Bottom_Count"]
    attribution = attribution.sort_values("Net_Score", ascending=False)

    print(attribution)
    report_lines.append("\n=== 模型归因 (Net Score) ===\n" + attribution.to_string())

    attr_path = os.path.join(output_dir, f"model_attribution_{anchor_date}.csv")
    attribution.to_csv(attr_path)
    print(f"归因表已保存: {attr_path}")

    # 归因条形图
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(attribution))
        width = 0.35
        ax.bar(
            x - width / 2, attribution["Top_Count"], width,
            label=f"In Top {top_n}", color="forestgreen", alpha=0.7,
        )
        ax.bar(
            x + width / 2, attribution["Bottom_Count"], width,
            label=f"In Bottom {top_n}", color="firebrick", alpha=0.7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(attribution.index, rotation=45, ha="right")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Model Importance Analysis (Top/Bottom {top_n} by Calmar) [FAST]"
        )
        ax.legend()
        plt.tight_layout()
        attr_fig_path = os.path.join(
            output_dir, f"model_attribution_{anchor_date}.png"
        )
        plt.savefig(attr_fig_path, dpi=150)
        plt.close()
        print(f"归因图已保存: {attr_fig_path}")
    except Exception as e:
        print(f"归因图绘制失败: {e}")

    # -----------------------------------------------------------------------
    # 4.3 相关性 vs 绩效分析
    # -----------------------------------------------------------------------
    print("\n🔗 === 相关性 vs 绩效分析 ===")

    # 提取单模型表现
    single_model_perf = (
        df[df["n_models"] == 1].set_index("models")["Ann_Excess"].to_dict()
    )

    # 使用基于 unique 组合的预先计算，避免 100万行 的 row-by-row pd.apply
    unique_combos = df["models"].unique()
    combo_to_metrics = {}

    for combo_str in unique_combos:
        models_list = combo_str.split(",")
        n = len(models_list)

        # 组合内部平均相关性
        if n < 2:
            avg_corr = 1.0
        else:
            try:
                sub = corr_matrix.loc[models_list, models_list].values
                # 使用 numpy triu 提取上三角（不含对角线）
                upper_tri = sub[np.triu_indices(n, k=1)]
                avg_corr = np.mean(upper_tri) if len(upper_tri) > 0 else 1.0
            except KeyError:
                avg_corr = np.nan

        # 多样性红利
        avg_individual_ret = np.mean(
            [single_model_perf.get(m, np.nan) for m in models_list]
        )
        combo_to_metrics[combo_str] = (avg_corr, avg_individual_ret)

    # 映射回 DataFrame
    metrics_df = pd.DataFrame.from_dict(
        combo_to_metrics, orient="index", columns=["avg_corr", "avg_ind"]
    )
    
    # 将指标 merge 进原表
    df = df.merge(metrics_df, left_on="models", right_index=True, how="left")
    df["diversity_bonus"] = df["Ann_Excess"] - df["avg_ind"]
    df = df.drop(columns=["avg_ind"])

    # 找黄金组合
    golden = df[
        (df["avg_corr"] < 0.3) & (df["Calmar"] > df["Calmar"].quantile(0.9))
    ]
    if not golden.empty:
        print(
            f"发现 {len(golden)} 个'黄金组合'：内部相关性 < 0.3 且 Calmar 前 10%"
        )
        print(f"典型代表: {golden.iloc[0]['models']}")
        golden_cols = [c for c in available_cols + ["avg_corr"] if c in df.columns]
        report_lines.append(
            f"\n=== 黄金组合 ({len(golden)} 个) ===\n"
            + golden[golden_cols].head(5).to_string(formatters=fmt)
        )
    else:
        print("未发现显著的'低相关性-高收益'组合")
        report_lines.append("\n=== 黄金组合: 未发现 ===")

    # 相关性 vs Calmar 散点图
    try:
        multi_df = df[df["n_models"] > 1].dropna(subset=["avg_corr"])
        if not multi_df.empty:
            # 防内存溢出：如果组合超过 50,000，随机抽样 50,000 进行绘图
            MAX_PLOT_POINTS = 50000
            if len(multi_df) > MAX_PLOT_POINTS:
                print(f"数据量过大 ({len(multi_df)})，随机抽样 {MAX_PLOT_POINTS} 个点进行绘图")
                plot_df = multi_df.sample(n=MAX_PLOT_POINTS, random_state=42)
            else:
                plot_df = multi_df

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # 图1: 风险-收益全景图
            scatter = axes[0].scatter(
                plot_df["Max_DD"].abs(), plot_df["Ann_Excess"],
                c=plot_df["n_models"], cmap="viridis", alpha=0.6,
                s=plot_df["n_models"] * 10 + 20,
            )
            singles = df[df["n_models"] == 1]
            axes[0].scatter(
                singles["Max_DD"].abs(), singles["Ann_Excess"],
                color="red", marker="x", s=100, label="Single Model",
            )
            axes[0].set_xlabel("Max Drawdown (Absolute)")
            axes[0].set_ylabel("Ann Excess Return")
            axes[0].set_title("Risk vs Return [FAST]")
            axes[0].legend()
            plt.colorbar(scatter, ax=axes[0], label="# Models")

            # 图2: 相关性 vs Calmar
            sns.scatterplot(
                x="avg_corr", y="Calmar", hue="n_models",
                palette="viridis", data=plot_df, ax=axes[1], alpha=0.7,
            )
            sns.regplot(
                x="avg_corr", y="Calmar", data=plot_df,
                scatter=False, ax=axes[1], color="red",
                line_kws={"linestyle": "--"},
            )
            axes[1].set_title("Correlation vs Calmar [FAST]")
            axes[1].set_xlabel("Avg Intra-Ensemble Correlation")

            plt.tight_layout()
            scatter_path = os.path.join(
                output_dir, f"risk_return_scatter_{anchor_date}.png"
            )
            plt.savefig(scatter_path, dpi=150)
            plt.close()
            print(f"散点图已保存: {scatter_path}")
    except Exception as e:
        print(f"散点图绘制失败: {e}")

    # 模型数量分组统计
    group_stats = df.groupby("n_models")[["Ann_Excess", "Calmar"]].agg(
        ["median", "mean", "max"]
    )
    print("\n=== 按模型数量分组统计 ===")
    print(group_stats.round(4))
    report_lines.append(
        "\n=== 按模型数量分组统计 ===\n" + group_stats.round(4).to_string()
    )

    # -----------------------------------------------------------------------
    # 4.4 Portfolio Optimization (Top 10 单模型)
    # -----------------------------------------------------------------------
    print("\n📐 === 优化权重分析 ===")
    try:
        from scipy.optimize import minimize

        # 使用 norm_df 的日均值作为简化收益代理
        top_singles = (
            df[df["n_models"] == 1]
            .sort_values("Calmar", ascending=False)
            .head(10)["models"]
            .tolist()
        )

        valid_models = [m for m in top_singles if m in norm_df.columns]

        if len(valid_models) >= 2:
            # 使用预测值排名的 TopK 收益作为近似收益
            print(f"使用 {len(valid_models)} 个 Top 单模型进行优化")

            # 从 norm_df 直接计算相关性和 variance 进行优化
            subset_corr = norm_df[valid_models].corr()
            # 使用单模型回测结果构造收益向量
            mu = pd.Series(
                {m: df[df["models"] == m]["Ann_Excess"].values[0] for m in valid_models}
            )
            # 简化 cov：使用 corr * vol
            individual_vol = pd.Series(
                {m: abs(df[df["models"] == m]["Max_DD"].values[0]) for m in valid_models}
            )
            cov = subset_corr * np.outer(individual_vol.values, individual_vol.values)

            num = len(valid_models)
            init_guess = [1.0 / num] * num
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bounds = tuple((0.0, 1.0) for _ in range(num))

            # Max Sharpe
            def neg_sharpe(w, mu, cov):
                ret = np.dot(w, mu)
                vol = np.sqrt(np.dot(w.T, np.dot(cov.values, w)))
                return -(ret / (vol + 1e-9))

            opt_sharpe = minimize(
                neg_sharpe, init_guess, args=(mu, cov),
                method="SLSQP", bounds=bounds, constraints=constraints,
            )

            # Risk Parity
            def risk_parity_obj(w, cov):
                w = np.array(w)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov.values, w)))
                mrc = np.dot(cov.values, w) / (port_vol + 1e-9)
                rc = w * mrc
                target = port_vol / len(w)
                return np.sum((rc - target) ** 2)

            opt_rp = minimize(
                risk_parity_obj, init_guess, args=(cov,),
                method="SLSQP", bounds=bounds, constraints=constraints,
            )

            weights_df = pd.DataFrame(
                {
                    "Equal_Weight": init_guess,
                    "Max_Sharpe": opt_sharpe.x,
                    "Risk_Parity": opt_rp.x,
                },
                index=valid_models,
            )

            print("\n=== 智能优化权重对比 ===")
            print(weights_df.round(4))
            report_lines.append(
                "\n=== 优化权重对比 ===\n" + weights_df.round(4).to_string()
            )

            opt_path = os.path.join(
                output_dir, f"optimization_weights_{anchor_date}.csv"
            )
            weights_df.to_csv(opt_path)
            print(f"优化权重已保存: {opt_path}")

        else:
            print("有效模型不足 2 个，跳过优化")
    except Exception as e:
        print(f"优化分析失败: {e}")

    # -----------------------------------------------------------------------
    # 4.5 综合报告
    # -----------------------------------------------------------------------
    best_combo = df.iloc[0]
    best_diversity_idx = df["diversity_bonus"].idxmax() if "diversity_bonus" in df.columns else None
    best_diversity = df.loc[best_diversity_idx] if best_diversity_idx is not None and pd.notna(best_diversity_idx) else None

    summary = []
    summary.append("=" * 60)
    summary.append("⚡ 快速模式自动分析报告摘要")
    summary.append("=" * 60)
    summary.append("(注意：快速模式指标为近似值，排序与原版高度一致)")

    summary.append(f"\n1. 最佳组合 (年化超额):")
    summary.append(f"   模型: {best_combo['models']}")
    summary.append(f"   模型数: {best_combo['n_models']}")
    summary.append(f"   年化收益: {best_combo['Ann_Ret']:.2%}")
    summary.append(f"   年化超额: {best_combo['Ann_Excess']:.2%}")
    summary.append(f"   最大回撤: {best_combo['Max_DD']:.2%}")
    summary.append(f"   Calmar: {best_combo['Calmar']:.2f}")

    if "avg_corr" in best_combo.index and pd.notna(best_combo.get("avg_corr")):
        summary.append(f"   内部相关性: {best_combo['avg_corr']:.4f}")

    if best_diversity is not None and pd.notna(best_diversity.get("diversity_bonus")):
        summary.append(f"\n2. 最大多样性红利组合:")
        summary.append(f"   模型: {best_diversity['models']}")
        summary.append(f"   Diversity Bonus: {best_diversity['diversity_bonus']:.4%}")

    summary.append(f"\n3. 建议保留的核心模型 (MVP):")
    summary.append(f"   {attribution.index[:3].tolist()}")
    summary.append("=" * 60)

    summary_text = "\n".join(summary)
    print(f"\n{summary_text}")

    # 写入报告文件
    report_path = os.path.join(output_dir, f"analysis_report_fast_{anchor_date}.txt")
    full_report = summary_text + "\n\n" + "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n完整报告已保存: {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="⚡ 快速向量化暴力穷举组合回测 + 结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试 (最多 3 个模型)
  python engine/scripts/brute_force_fast.py --max-combo-size 3

  # 仅分析已有结果
  python engine/scripts/brute_force_fast.py --analysis-only

  # 完整穷举 + 分析
  python engine/scripts/brute_force_fast.py

  # 使用 GPU 加速
  python engine/scripts/brute_force_fast.py --use-gpu

  # 从中断处继续
  python engine/scripts/brute_force_fast.py --resume
        """,
    )
    parser.add_argument(
        "--record-file", type=str, default="latest_train_records.json",
        help="训练记录文件路径 (默认: latest_train_records.json)",
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
        "--freq", type=str, default="week", choices=["day", "week"],
        help="回测交易频率 (默认: week)",
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="分析时 Top/Bottom N (默认: 50)",
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
        "--auto-test-top", type=int, default=0,
        help="自动在 OOS 数据上测试排名前 N 的组合",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="跳过分析阶段 (仅回测)",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="仅分析已有 CSV 结果 (不跑回测)",
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

    # 加载配置
    train_records, model_config = load_config(args.record_file)
    anchor_date = train_records.get(
        "anchor_date", datetime.now().strftime("%Y-%m-%d")
    )
    top_k = model_config.get("TopK", 22)
    drop_n = model_config.get("DropN", 3)
    benchmark = model_config.get("benchmark", "SH000300")

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: 加载预测数据
    norm_df, model_metrics = load_predictions(train_records)

    # 划分数据集 (IS / OOS)
    is_norm_df, oos_norm_df = split_is_oos_by_args(norm_df, args)
    if is_norm_df.empty:
        print("错误: IS 期无数据！请检查日期参数。")
        sys.exit(1)

    # Stage 2: 相关性分析
    corr_matrix = correlation_analysis(is_norm_df, args.output_dir, anchor_date)

    if not args.analysis_only:
        # 确定调仓频率 (周频=5天, 日频=1天)
        rebalance_freq = 5 if args.freq == "week" else 1

        # 加载日频收益率矩阵 (无论 freq 是什么都加载日频，全量加载)
        returns_wide, bench_returns, common_dates, instruments = load_returns_matrix(
            norm_df, freq=args.freq
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
    else:
        # 直接读取已有结果
        import glob
        csv_path = os.path.join(
            args.output_dir, f"brute_force_fast_results_{anchor_date}.csv"
        )
        if not os.path.exists(csv_path):
            pattern = os.path.join(args.output_dir, "brute_force_fast_results_*.csv")
            files = sorted(glob.glob(pattern))
            if files:
                csv_path = files[-1]
                print(f"使用最新结果文件: {csv_path}")
            else:
                print(f"错误: 未找到结果文件 ({pattern})")
                sys.exit(1)
        results_df = pd.read_csv(csv_path)
        print(f"\n已加载现有结果: {csv_path} ({len(results_df)} 条)")

    # Stage 4: 分析
    if not args.skip_analysis and not results_df.empty:
        analyze_results(
            results_df=results_df,
            corr_matrix=corr_matrix,
            norm_df=is_norm_df,
            train_records=train_records,
            output_dir=args.output_dir,
            anchor_date=anchor_date,
            top_n=args.top_n,
        )

    # Stage 5: OOS 验证
    if getattr(args, "auto_test_top", 0) > 0 and not results_df.empty:
        if oos_norm_df.empty:
            print("\n⚠️ 无法进行 ⚡ 快速 OOS 验证：无 OOS 数据！请配合使用 --exclude-last-years 或类似参数。")
        else:
            print(f"\n{'='*60}")
            print(f"Stage 5: ⚡ 快速 Out-Of-Sample (OOS) 验证 (Top {args.auto_test_top})")
            print(f"{'='*60}")
            
            rebalance_freq = 5 if args.freq == "week" else 1
            if args.analysis_only:
                # 若使用了 analysis_only，需要单独加载一下全量收益矩阵
                returns_wide, bench_returns, common_dates, instruments = load_returns_matrix(
                    norm_df, freq=args.freq
                )
                
            oos_dates = oos_norm_df.index.get_level_values("datetime").unique().sort_values()
            oos_common_dates = oos_dates.intersection(returns_wide.index)
            
            oos_scores_np, oos_returns_np, _, oos_date_index, _ = prepare_matrices(
                oos_norm_df, returns_wide, oos_common_dates
            )
            oos_bench_returns_np = bench_returns.reindex(oos_date_index).fillna(0).values.astype(np.float32)

            top_combos = results_df.head(args.auto_test_top)["models"].tolist()
            
            oos_returns_gpu = xp.asarray(oos_returns_np)
            oos_scores_gpu = {m: xp.asarray(v) for m, v in oos_scores_np.items()}
            
            oos_results = []
            for combo_str in tqdm(top_combos, desc="OOS Testing"):
                combo = combo_str.split(",")
                combo_score = sum(oos_scores_gpu[m] for m in combo) / len(combo)
                
                net_ret = _vectorized_topk_backtest_single(
                    combo_score, oos_returns_gpu, top_k, args.cost_rate, rebalance_freq,
                )
                metrics = compute_metrics(net_ret, oos_bench_returns_np, args.freq)
                metrics["models"] = combo_str
                metrics["n_models"] = len(combo)
                oos_results.append(metrics)
                
            if _USE_GPU:
                del oos_returns_gpu, oos_scores_gpu
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                
            if oos_results:
                oos_df = pd.DataFrame(oos_results)
                oos_path = os.path.join(args.output_dir, f"oos_validation_{anchor_date}.csv")
                oos_df.to_csv(oos_path, index=False)
                print(f"OOS 结果已保存: {oos_path}")
                
                print("\n⚡ 快速 OOS 验证结果:")
                display_cols = ["models", "Ann_Ret", "Max_DD", "Ann_Excess", "Calmar"]
                fmt = {
                    "Ann_Ret": "{:.2%}".format,
                    "Max_DD": "{:.2%}".format,
                    "Ann_Excess": "{:.2%}".format,
                    "Calmar": "{:.2f}".format,
                }
                print(oos_df[display_cols].to_string(formatters=fmt))

    print(f"\n{'='*60}")
    print(f"全部完成！ 耗时结束于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
