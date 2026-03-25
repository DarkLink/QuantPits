#!/usr/bin/env python
"""
Brute Force Ensemble - 暴力穷举组合回测 + 结果分析

将所有模型的预测结果进行暴力组合，对每个组合做等权融合+回测，
最后对结果进行全面分析（模型归因、相关性、聚类、优化权重等）。

运行方式：
  cd QuantPits && python quantpits/scripts/brute_force_ensemble.py

常用命令：
  # 快速测试（最多3个模型的组合）
  python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

  # 仅分析已有结果（不重新跑回测）
  python quantpits/scripts/brute_force_ensemble.py --analysis-only

  # 从上次中断处继续
  python quantpits/scripts/brute_force_ensemble.py --resume

  # 只跑回测、跳过分析
  python quantpits/scripts/brute_force_ensemble.py --skip-analysis

  # 使用模型分组穷举 (每组只选一个) — 大幅减少组合数
  python quantpits/scripts/brute_force_ensemble.py --use-groups

  # 指定自定义分组配置
  python quantpits/scripts/brute_force_ensemble.py --use-groups --group-config config/my_groups.yaml
"""

import os
import sys
import json
import gc
import signal
import itertools
import logging
import argparse
import yaml
import warnings
from datetime import datetime
from collections import Counter
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed

from quantpits.utils import env
os.chdir(env.ROOT_DIR)

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
# 已在上方导入并切换目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# BACKTEST_CONFIG has been migrated to strategy provider `config/strategy_config.yaml`

# ---------------------------------------------------------------------------
# 全局中断标志 & 信号处理 (委托给 search_utils)
# ---------------------------------------------------------------------------
from quantpits.utils.search_utils import (
    _signal_handler, _install_signal_handlers, _restore_signal_handlers,
    run_single_backtest, _append_results_to_csv,
    split_is_oos_by_args, load_combo_groups, generate_grouped_combinations,
)
import quantpits.utils.search_utils as _su

# 暴露 _shutdown 作为模块属性的别名，保持向后兼容
_shutdown = False  # 运行时实际使用 _su._shutdown


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
# Stage 1: 加载预测数据
# ============================================================================

def load_predictions(train_records):
    """
    从 Qlib Recorder 加载所有模型的预测值，归一化后返回宽表。
    """
    from quantpits.utils.predict_utils import load_predictions_from_recorder
    norm_df, model_metrics, _ = load_predictions_from_recorder(train_records)
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
    """根据参数将 norm_df 划分为 IS (In-Sample) 和 OOS (Out-of-Sample)，委托给 search_utils"""
    from quantpits.utils.search_utils import split_is_oos_by_args as _split
    return _split(norm_df, args)


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
# Stage 3: 暴力穷举回测
# ============================================================================

def run_single_backtest(
    combo_models, norm_df, top_k, drop_n, benchmark, freq,
    trade_exchange, bt_start, bt_end, st_config=None, bt_config=None
):
    """对指定的模型组合进行回测，委托给 search_utils"""
    from quantpits.utils.search_utils import run_single_backtest as _run
    return _run(
        combo_models, norm_df, top_k, drop_n, benchmark, freq,
        trade_exchange, bt_start, bt_end, st_config, bt_config
    )


def _append_results_to_csv(csv_path, results, write_header=False):
    """将一批结果追加写入 CSV 文件，委托给 search_utils"""
    from quantpits.utils.search_utils import _append_results_to_csv as _append
    _append(csv_path, results, write_header)


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
    _su._shutdown = False

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

    from quantpits.utils import strategy
    st_config = strategy.load_strategy_config()
    bt_config = strategy.get_backtest_config(st_config)
    
    exchange_kwargs = bt_config["exchange_kwargs"].copy()
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
                if _su._shutdown:
                    break

                batch = pending[batch_start : batch_start + batch_size]
                batch_results = []

                # 使用 ThreadPoolExecutor 处理当前批次
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_combo = {
                        executor.submit(
                            run_single_backtest,
                            combo,
                            norm_df,
                            top_k,
                            drop_n,
                            benchmark,
                            freq,
                            trade_exchange,
                            bt_start,
                            bt_end,
                            st_config,
                            bt_config
                        ): combo
                        for combo in batch
                    }

                    for future in as_completed(future_to_combo):
                        if _su._shutdown:
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

                if _su._shutdown:
                    break

        finally:
            pbar.close()
            _restore_signal_handlers()
            qlib_log.setLevel(original_level)

        # 打印完成/中断状态
        if _su._shutdown:
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
# Main
# ============================================================================

def main():
    from quantpits.utils import env
    env.safeguard("Brute Force Ensemble")
    parser = argparse.ArgumentParser(
        description="暴力穷举模型组合回测 + 结果分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试 (最多 3 个模型)
  python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

  # 从中断处继续
  python quantpits/scripts/brute_force_ensemble.py --resume
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
        "--resume", action="store_true",
        help="从已有 CSV 继续 (跳过已完成的组合)",
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
    
    train_records, model_config = load_config(args.record_file)
    
    # 确定频率
    freq = args.freq or model_config.get("freq", "week")
    args.freq = freq
    print(f"当前交易频率: {freq}")
    
    anchor_date = train_records.get(
        "anchor_date", datetime.now().strftime("%Y-%m-%d")
    )
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

    # 导出 Metadata
    metadata_path = os.path.join(args.output_dir, f"run_metadata_{anchor_date}.json")
    is_dates_all = is_norm_df.index.get_level_values("datetime")
    oos_dates_all = oos_norm_df.index.get_level_values("datetime")
    import json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "anchor_date": anchor_date,
            "script_used": "brute_force_ensemble",
            "freq": args.freq,
            "record_file": args.record_file,
            "training_mode": getattr(args, "training_mode", None),
            "is_start_date": str(is_dates_all.min().date()) if not is_dates_all.empty else None,
            "is_end_date": str(is_dates_all.max().date()) if not is_dates_all.empty else None,
            "oos_start_date": str(oos_dates_all.min().date()) if not oos_dates_all.empty else None,
            "oos_end_date": str(oos_dates_all.max().date()) if not oos_dates_all.empty else None,
            "exclude_last_years": args.exclude_last_years,
            "exclude_last_months": args.exclude_last_months,
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
