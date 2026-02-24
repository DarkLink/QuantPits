#!/usr/bin/env python
"""
Brute Force Ensemble - 暴力穷举组合回测 + 结果分析

将所有模型的预测结果进行暴力组合，对每个组合做等权融合+回测，
最后对结果进行全面分析（模型归因、相关性、聚类、优化权重等）。

运行方式：
  cd QuantPits && python engine/scripts/brute_force_ensemble.py

常用命令：
  # 快速测试（最多3个模型的组合）
  python engine/scripts/brute_force_ensemble.py --max-combo-size 3

  # 仅分析已有结果（不重新跑回测）
  python engine/scripts/brute_force_ensemble.py --analysis-only

  # 完整穷举 + 分析
  python engine/scripts/brute_force_ensemble.py

  # 从上次中断处继续
  python engine/scripts/brute_force_ensemble.py --resume

  # 只跑回测、跳过分析
  python engine/scripts/brute_force_ensemble.py --skip-analysis

  # 使用模型分组穷举 (每组只选一个) — 大幅减少组合数
  python engine/scripts/brute_force_ensemble.py --use-groups

  # 指定自定义分组配置
  python engine/scripts/brute_force_ensemble.py --use-groups --group-config config/my_groups.yaml
"""

import os
import sys
import json
import gc
import signal
import itertools
import logging
import argparse
from datetime import datetime
from collections import Counter
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ---------------------------------------------------------------------------
# 全局中断标志 & 信号处理
# ---------------------------------------------------------------------------
_shutdown = False
_original_sigint = None
_original_sigterm = None


def _signal_handler(signum, frame):
    """收到 SIGINT/SIGTERM 后标记安全中断"""
    global _shutdown
    if _shutdown:
        # 第二次中断 → 强制退出
        print("\n\n⛔ 再次收到中断信号，强制退出...")
        sys.exit(1)
    _shutdown = True
    sig_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n\n⚠️  收到 {sig_name}，将在当前批次完成后安全退出...")
    print("   (再次按 Ctrl+C 强制退出)")


def _install_signal_handlers():
    """安装信号处理器"""
    global _original_sigint, _original_sigterm
    _original_sigint = signal.getsignal(signal.SIGINT)
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _restore_signal_handlers():
    """恢复原始信号处理器"""
    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)
    if _original_sigterm is not None:
        signal.signal(signal.SIGTERM, _original_sigterm)


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
# Stage 1: 加载预测数据
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
    
    # OOS 是截止日之后的数据（如果有指定 start_date/end_date，暂不限制 OOS 的末尾跨度，或只保留剩下的）
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
# Stage 3: 暴力穷举回测
# ============================================================================

def extract_report_df(metrics):
    """从回测结果中提取 report DataFrame"""
    if isinstance(metrics, dict):
        val = list(metrics.values())[0]
        return val[0] if isinstance(val, tuple) else val
    elif isinstance(metrics, tuple):
        first = metrics[0]
        if isinstance(first, pd.DataFrame):
            return first
        elif isinstance(first, tuple) and len(first) >= 1:
            return first[0]
        return metrics
    return metrics


def run_single_backtest(
    combo_models, norm_df, top_k, drop_n, benchmark, freq,
    trade_exchange, bt_start, bt_end
):
    """对指定的模型组合进行回测，返回指标字典或 None"""
    from qlib.backtest import backtest_loop
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.utils import CommonInfrastructure
    from qlib.backtest.account import Account
    from qlib.contrib.strategy import TopkDropoutStrategy

    # 1. 合成信号 (等权均值，归一化后的)
    combo_score = norm_df[list(combo_models)].mean(axis=1)

    # 2. 准备组件
    # 注意: Account 必须每次新建，不能复用 (状态会累积)
    trade_account = Account(init_cash=BACKTEST_CONFIG["account"])
    
    # CommonInfrastructure 组合: 新 Account + 共享 Exchange
    common_infra = CommonInfrastructure(
        trade_account=trade_account,
        trade_exchange=trade_exchange
    )

    strategy = TopkDropoutStrategy(
        signal=combo_score, topk=top_k, n_drop=drop_n, only_tradable=True
    )
    # 策略需要关联 infra
    strategy.reset_common_infra(common_infra)

    executor_obj = SimulatorExecutor(
        time_per_step=freq,
        generate_portfolio_metrics=True,
        verbose=False,
        common_infra=common_infra,
    )

    # 3. 回测
    try:
        # 使用 backtest_loop 直接运行，避免 backtest() 函数内部重建 Exchange
        raw_portfolio_metrics, _ = backtest_loop(
            start_time=bt_start,
            end_time=bt_end,
            trade_strategy=strategy,
            trade_executor=executor_obj,
        )

        # 4. 提取结果
        report = extract_report_df(raw_portfolio_metrics)

        initial_cash = BACKTEST_CONFIG["account"]
        final_nav = report.iloc[-1]["account"]
        ann_ret = report["return"].mean() * 52

        report["nav"] = report["account"]
        report["max_nav"] = report["nav"].cummax()
        report["drawdown"] = (report["nav"] - report["max_nav"]) / report["max_nav"]
        max_dd = report["drawdown"].min()

        total_ret = (final_nav / initial_cash) - 1
        bench_ret = (
            (report.iloc[-1]["bench"] - report.iloc[0]["bench"])
            / report.iloc[0]["bench"]
        )
        excess_ret = total_ret - bench_ret
        ann_excess = ann_ret - (report["bench"].mean() * 52)

        return {
            "models": ",".join(combo_models),
            "n_models": len(combo_models),
            "Ann_Ret": ann_ret,
            "Max_DD": max_dd,
            "Excess_Ret": excess_ret,
            "Ann_Excess": ann_excess,
            "Total_Ret": total_ret,
            "Final_NAV": final_nav,
            "Calmar": ann_ret / abs(max_dd) if max_dd != 0 else 0,
        }
    except Exception:
        # import traceback
        # traceback.print_exc()
        return None


def _append_results_to_csv(csv_path, results, write_header=False):
    """将一批结果追加写入 CSV 文件"""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def brute_force_backtest(
    norm_df, top_k, drop_n, benchmark, freq,
    min_combo_size, max_combo_size, output_dir, anchor_date, resume=False,
    n_jobs=4, use_groups=False, group_config=None,
    batch_size=50,
):
    """
    暴力穷举所有模型组合并回测。

    支持:
    - 分批执行 + 增量保存 (防止崩溃丢失进度)
    - SIGINT/SIGTERM 安全中断
    - 模型分组穷举 (--use-groups)

    Returns:
        results_df: DataFrame，所有组合的回测结果
    """
    global _shutdown
    _shutdown = False

    from qlib.backtest.exchange import Exchange

    print(f"\n{'='*60}")
    print("Stage 3: 暴力穷举回测 (Batched Threading + Checkpoint)")
    print(f"{'='*60}")

    model_candidates = list(norm_df.columns)

    # 准备共享的 Exchange 对象
    print("Initializing Shared Exchange...")
    bt_start = str(norm_df.index.get_level_values(0).min().date())
    bt_end = str(norm_df.index.get_level_values(0).max().date())
    all_codes = sorted(norm_df.index.get_level_values(1).unique().tolist())

    exchange_kwargs = BACKTEST_CONFIG["exchange_kwargs"].copy()
    exchange_freq = exchange_kwargs.pop("freq", "day")

    trade_exchange = Exchange(
        freq=exchange_freq,
        start_time=bt_start,
        end_time=bt_end,
        codes=all_codes,
        **exchange_kwargs
    )
    print(f"Shared Exchange Initialized. Period: {bt_start} ~ {bt_end}, Instruments: {len(all_codes)}")

    # ── 生成组合 ──
    if use_groups and group_config:
        print(f"\n📦 分组穷举模式 (配置: {group_config})")
        groups = load_combo_groups(group_config, model_candidates)
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
        max_size = max_combo_size if max_combo_size > 0 else len(model_candidates)
        max_size = min(max_size, len(model_candidates))
        print(f"待穷举模型 ({len(model_candidates)}个): {model_candidates}")
        print(f"组合大小范围: {min_combo_size} ~ {max_size}")

        all_combinations = []
        for r in range(min_combo_size, max_size + 1):
            all_combinations.extend(itertools.combinations(model_candidates, r))

    print(f"总组合数: {len(all_combinations)}")
    print(f"回测频率: {freq}, TopK={top_k}, DropN={drop_n}")
    print(f"并发线程数: {n_jobs}, 批次大小: {batch_size}")

    # ── Resume: 加载已有结果 ──
    csv_path = os.path.join(output_dir, f"brute_force_results_{anchor_date}.csv")
    done_combos = set()

    if resume and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_combos = set(existing_df["models"].tolist())
        print(f"Resume 模式: 已有 {len(done_combos)} 个组合，跳过")

    # 过滤已完成的组合
    pending = [
        c for c in all_combinations if ",".join(c) not in done_combos
    ]
    print(f"待回测组合数: {len(pending)}")

    if not pending:
        print("所有组合已完成！")
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
        else:
            results_df = pd.DataFrame()
    else:
        # 如果不是 resume，先写一个空的带 header 的 CSV
        need_header = not (resume and os.path.exists(csv_path))
        if need_header:
            # 写 header
            header_cols = [
                "models", "n_models", "Ann_Ret", "Max_DD",
                "Excess_Ret", "Ann_Excess", "Total_Ret", "Final_NAV", "Calmar"
            ]
            pd.DataFrame(columns=header_cols).to_csv(csv_path, index=False)

        # 临时静默 Qlib 日志
        qlib_log = logging.getLogger("qlib")
        original_level = qlib_log.level
        qlib_log.setLevel(logging.WARNING)

        # 安装信号处理器
        _install_signal_handlers()

        completed_count = len(done_combos)
        total_count = len(all_combinations)
        failed_count = 0

        # ── 分批处理 ──
        pbar = tqdm(
            total=len(pending),
            desc=f"Brute Force (Threads={n_jobs})",
            unit="combo",
        )

        try:
            for batch_start in range(0, len(pending), batch_size):
                if _shutdown:
                    break

                batch = pending[batch_start : batch_start + batch_size]
                batch_results = []

                # 使用 ThreadPoolExecutor 处理当前批次
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_combo = {
                        executor.submit(
                            run_single_backtest,
                            combo, norm_df, top_k, drop_n, benchmark, freq,
                            trade_exchange, bt_start, bt_end
                        ): combo
                        for combo in batch
                    }

                    for future in as_completed(future_to_combo):
                        if _shutdown:
                            # 收到中断，不再等待其他 future
                            # 但已提交的会继续完成 (ThreadPoolExecutor 的行为)
                            pass
                        try:
                            res = future.result()
                            if res:
                                batch_results.append(res)
                                completed_count += 1
                            else:
                                failed_count += 1
                        except Exception:
                            failed_count += 1
                        pbar.update(1)

                # 批次完成 → 增量写入 CSV
                if batch_results:
                    _append_results_to_csv(csv_path, batch_results, write_header=False)

                # 释放内存
                del batch_results
                gc.collect()

                if _shutdown:
                    break

        finally:
            pbar.close()
            _restore_signal_handlers()
            qlib_log.setLevel(original_level)

        # 打印完成/中断状态
        if _shutdown:
            print(f"\n⚠️  已安全中断！")
            print(f"   已完成: {completed_count}/{total_count} 组合")
            print(f"   失败: {failed_count} 个")
            print(f"   结果已保存至: {csv_path}")
            print(f"   使用 --resume 继续未完成的组合")
        else:
            print(f"\n✅ 回测全部完成！")
            print(f"   有效: {completed_count - len(done_combos)}, 失败: {failed_count}")

        # 读取完整结果 (包括 resume 的)
        results_df = pd.read_csv(csv_path)

    # 排序并重新保存
    if not results_df.empty:
        results_df = results_df.sort_values(
            by="Ann_Excess", ascending=False
        ).reset_index(drop=True)
        results_df.to_csv(csv_path, index=False)
        print(f"结果已排序保存: {csv_path} ({len(results_df)} 条)")
    else:
        print("警告: 无有效回测结果")

    return results_df


# ============================================================================
# Stage 4: 结果分析
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
    fmt = {
        "Ann_Ret": "{:.2%}".format,
        "Max_DD": "{:.2%}".format,
        "Ann_Excess": "{:.2%}".format,
        "Calmar": "{:.2f}".format,
    }

    print("\n🏆 === 综合表现最佳 Top 20 (按年化超额) ===")
    top20_str = df[display_cols].head(20).to_string(formatters=fmt)
    print(top20_str)
    report_lines.append("=== Top 20 (按年化超额) ===\n" + top20_str)

    robust_df = df[df["Ann_Ret"] > 0.10].sort_values(by="Max_DD", ascending=False)
    if not robust_df.empty:
        print("\n🛡️ === 最稳健组合 Top 10 (年化>10%, 按回撤排序) ===")
        robust_str = robust_df[display_cols].head(10).to_string(formatters=fmt)
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
            f"Model Importance Analysis (Top/Bottom {top_n} by Calmar)"
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

    # 使用基于 unique 组合的预先计算，避免百万行的 row-by-row pd.apply
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
        report_lines.append(
            f"\n=== 黄金组合 ({len(golden)} 个) ===\n"
            + golden[display_cols + ["avg_corr"]].head(5).to_string(formatters=fmt)
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
            axes[0].set_title("Risk vs Return")
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
            axes[1].set_title("Correlation vs Calmar")
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
    # 4.4 Cluster Dendrogram (基于回测报告的超额收益)
    # -----------------------------------------------------------------------
    print("\n🌳 === 层次聚类分析 ===")
    try:
        from qlib.workflow import R
        from scipy.cluster.hierarchy import dendrogram, linkage

        experiment_name = train_records["experiment_name"]
        models = train_records["models"]

        returns_dict = {}
        excess_dict = {}

        for model_name, record_id in models.items():
            try:
                recorder = R.get_recorder(
                    recorder_id=record_id, experiment_name=experiment_name
                )
                report = recorder.load_object(
                    "portfolio_analysis/report_normal_1week.pkl"
                )
                returns_dict[model_name] = report["return"]
                if "bench" in report.columns:
                    excess_dict[model_name] = report["return"] - report["bench"]
            except Exception as e:
                print(f"  [跳过] {model_name}: {e}")

        if excess_dict:
            all_excess = pd.DataFrame(excess_dict).dropna()

            if len(all_excess.columns) >= 2:
                corr_excess = all_excess.corr()
                linked = linkage(corr_excess, "ward")

                fig, ax = plt.subplots(figsize=(12, 7))
                dendrogram(
                    linked, orientation="top", labels=corr_excess.columns.tolist(),
                    distance_sort="descending", show_leaf_counts=True, ax=ax,
                )
                ax.set_title("Model Alpha Cluster Dendrogram")
                ax.set_ylabel("Euclidean Distance (Dissimilarity)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                dendro_path = os.path.join(
                    output_dir, f"cluster_dendrogram_{anchor_date}.png"
                )
                plt.savefig(dendro_path, dpi=150)
                plt.close()
                print(f"聚类图已保存: {dendro_path}")
            else:
                print("模型数量不足，跳过聚类")
        else:
            print("无超额收益数据，跳过聚类")
    except Exception as e:
        print(f"聚类分析失败: {e}")

    # -----------------------------------------------------------------------
    # 4.5 Portfolio Optimization (Top 10 单模型)
    # -----------------------------------------------------------------------
    print("\n📐 === 优化权重分析 ===")
    try:
        from scipy.optimize import minimize

        if "all_excess" not in dir() or all_excess is None or all_excess.empty:
            # 尝试使用 returns_dict
            if returns_dict:
                all_returns_opt = pd.DataFrame(returns_dict).dropna()
            else:
                raise ValueError("无收益率数据")
        else:
            all_returns_opt = pd.DataFrame(returns_dict).dropna()

        # 选取 Top 10 单模型
        top_singles = (
            df[df["n_models"] == 1]
            .sort_values("Calmar", ascending=False)
            .head(10)["models"]
            .tolist()
        )
        valid_models = [m for m in top_singles if m in all_returns_opt.columns]

        if len(valid_models) >= 2:
            subset = all_returns_opt[valid_models]
            mu = subset.mean() * 252
            cov = subset.cov() * 252
            num = len(valid_models)
            init_guess = [1.0 / num] * num
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bounds = tuple((0.0, 1.0) for _ in range(num))

            # Max Sharpe
            def neg_sharpe(w, mu, cov):
                ret = np.dot(w, mu)
                vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                return -(ret / (vol + 1e-9))

            opt_sharpe = minimize(
                neg_sharpe, init_guess, args=(mu, cov),
                method="SLSQP", bounds=bounds, constraints=constraints,
            )

            # Risk Parity
            def risk_parity_obj(w, cov):
                w = np.array(w)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                mrc = np.dot(cov, w) / (port_vol + 1e-9)
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

            # 简单回测对比
            def calc_stats(ret_series, name):
                ann_ret = ret_series.mean() * 252
                ann_vol = ret_series.std() * np.sqrt(252)
                sharpe = ann_ret / (ann_vol + 1e-9)
                return f"{name}: Sharpe={sharpe:.2f}, Ann Ret={ann_ret:.2%}, Vol={ann_vol:.2%}"

            ew_ret = subset.mean(axis=1)
            ms_ret = subset.dot(opt_sharpe.x)
            rp_ret = subset.dot(opt_rp.x)

            print("\n绩效总结:")
            stats_lines = [
                calc_stats(ew_ret, "Equal Weight"),
                calc_stats(ms_ret, "Max Sharpe  "),
                calc_stats(rp_ret, "Risk Parity "),
            ]
            for line in stats_lines:
                print(f"  {line}")
            report_lines.append(
                "\n=== 优化绩效 ===\n" + "\n".join(stats_lines)
            )

            # 净值曲线
            try:
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(
                    (1 + ew_ret).cumprod(), label="Equal Weight",
                    linestyle="--", color="black",
                )
                ax.plot(
                    (1 + ms_ret).cumprod(), label="Max Sharpe", color="red",
                )
                ax.plot(
                    (1 + rp_ret).cumprod(), label="Risk Parity", color="green",
                )
                ax.set_title("Brute Force EW vs Mathematical Optimization")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                opt_fig_path = os.path.join(
                    output_dir, f"optimization_equity_{anchor_date}.png"
                )
                plt.savefig(opt_fig_path, dpi=150)
                plt.close()
                print(f"净值曲线已保存: {opt_fig_path}")
            except Exception as e:
                print(f"净值图绘制失败: {e}")

        else:
            print("有效模型不足 2 个，跳过优化")
    except Exception as e:
        print(f"优化分析失败: {e}")

    # -----------------------------------------------------------------------
    # 4.6 综合报告
    # -----------------------------------------------------------------------
    best_combo = df.iloc[0]
    best_diversity_idx = df["diversity_bonus"].idxmax()
    best_diversity = df.loc[best_diversity_idx] if pd.notna(best_diversity_idx) else None

    summary = []
    summary.append("=" * 60)
    summary.append("自动分析报告摘要")
    summary.append("=" * 60)

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
    report_path = os.path.join(output_dir, f"analysis_report_{anchor_date}.txt")
    full_report = summary_text + "\n\n" + "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n完整报告已保存: {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="暴力穷举模型组合回测 + 结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试 (最多 3 个模型)
  python engine/scripts/brute_force_ensemble.py --max-combo-size 3

  # 仅分析已有结果
  python engine/scripts/brute_force_ensemble.py --analysis-only

  # 完整穷举 + 分析
  python engine/scripts/brute_force_ensemble.py

  # 从中断处继续
  python engine/scripts/brute_force_ensemble.py --resume
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
        "--output-dir", type=str, default="output/brute_force",
        help="输出目录 (默认: output/brute_force)",
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
        "--resume", action="store_true",
        help="从已有 CSV 继续 (跳过已完成的组合)",
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
        "--n-jobs", type=int, default=4,
        help="并发回测线程数 (默认: 4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="每批处理的组合数 (默认: 50，影响 checkpoint 粒度和内存占用)",
    )
    parser.add_argument(
        "--use-groups", action="store_true",
        help="启用分组穷举模式 (每组只选一个模型)",
    )
    parser.add_argument(
        "--group-config", type=str, default="config/combo_groups.yaml",
        help="分组配置文件路径 (默认: config/combo_groups.yaml)",
    )
    args = parser.parse_args()

    # 初始化
    print("=" * 60)
    print("Brute Force Ensemble - 暴力穷举组合回测")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

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

    # Stage 2: 相关性分析 (基于 IS)
    corr_matrix = correlation_analysis(is_norm_df, args.output_dir, anchor_date)

    if not args.analysis_only:
        # Stage 3: 暴力回测 (基于 IS)
        results_df = brute_force_backtest(
            norm_df=is_norm_df,
            top_k=top_k,
            drop_n=drop_n,
            benchmark=benchmark,
            freq=args.freq,
            min_combo_size=args.min_combo_size,
            max_combo_size=args.max_combo_size,
            output_dir=args.output_dir,
            anchor_date=anchor_date,
            resume=args.resume,
            n_jobs=args.n_jobs,
            use_groups=args.use_groups,
            group_config=args.group_config,
            batch_size=args.batch_size,
        )
    else:
        # 直接读取已有结果
        csv_path = os.path.join(
            args.output_dir, f"brute_force_results_{anchor_date}.csv"
        )
        if not os.path.exists(csv_path):
            import glob
            pattern = os.path.join(args.output_dir, "brute_force_results_*.csv")
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
            print("\n⚠️ 无法进行 OOS 验证：无 OOS 数据！请配合使用 --exclude-last-years 或类似参数。")
        else:
            print(f"\n{'='*60}")
            print(f"Stage 5: 自动 Out-Of-Sample (OOS) 验证 (Top {args.auto_test_top})")
            print(f"{'='*60}")
            
            top_combos = results_df.head(args.auto_test_top)["models"].tolist()
            
            # Init OOS Exchange
            from qlib.backtest.exchange import Exchange
            bt_start_oos = str(oos_norm_df.index.get_level_values(0).min().date())
            bt_end_oos = str(oos_norm_df.index.get_level_values(0).max().date())
            all_codes_oos = sorted(oos_norm_df.index.get_level_values(1).unique().tolist())
            
            exchange_kwargs = BACKTEST_CONFIG["exchange_kwargs"].copy()
            exchange_freq = exchange_kwargs.pop("freq", "day")
            
            trade_exchange_oos = Exchange(
                freq=exchange_freq,
                start_time=bt_start_oos,
                end_time=bt_end_oos,
                codes=all_codes_oos,
                **exchange_kwargs
            )
            
            oos_results = []
            for combo_str in tqdm(top_combos, desc="OOS Testing"):
                combo = combo_str.split(",")
                res = run_single_backtest(
                    combo, oos_norm_df, top_k, drop_n, benchmark, args.freq,
                    trade_exchange_oos, bt_start_oos, bt_end_oos
                )
                if res:
                    oos_results.append(res)
                    
            if oos_results:
                oos_df = pd.DataFrame(oos_results)
                oos_path = os.path.join(args.output_dir, f"oos_validation_{anchor_date}.csv")
                oos_df.to_csv(oos_path, index=False)
                print(f"OOS 结果已保存: {oos_path}")
                
                print("\nOOS 验证结果:")
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
