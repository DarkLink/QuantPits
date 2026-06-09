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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    worker_init, run_backtest_in_worker, compute_rolling_sharpe_weights,
    extract_group_model_names,
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

def load_predictions(train_records, norm_method="rank", selected_models=None):
    """
    从 Qlib Recorder 加载所有模型的预测值，归一化后返回宽表。
    """
    from quantpits.utils.predict_utils import load_predictions_from_recorder
    norm_df, model_metrics, _ = load_predictions_from_recorder(
        train_records, selected_models=selected_models, norm_method=norm_method
    )
    return norm_df, model_metrics


# ============================================================================
# Stage 2: 相关性分析
# ============================================================================

def correlation_analysis(norm_df, output_dir, anchor_date=None):
    """计算并保存预测值相关性矩阵

    Args:
        norm_df: 归一化预测数据
        output_dir: 输出目录 (通常是 RunContext.is_dir)
        anchor_date: 可选日期后缀 (新结构下不再需要)
    """
    print(f"\n{'='*60}")
    print("Stage 2: 相关性分析")
    print(f"{'='*60}")

    corr_matrix = norm_df.corr()
    print("\n=== 模型预测相关性矩阵 ===")
    print(corr_matrix.round(4))

    # 保存
    suffix = f"_{anchor_date}" if anchor_date else ""
    corr_path = os.path.join(output_dir, f"correlation_matrix{suffix}.csv")
    corr_matrix.to_csv(corr_path)
    print(f"\n相关性矩阵已保存: {corr_path}")

    return corr_matrix



# ============================================================================
# Stage 2.5: 模型分组 & 组合生成
# ============================================================================
# split_is_oos_by_args, load_combo_groups, generate_grouped_combinations,
# run_single_backtest, _append_results_to_csv are imported from search_utils
# at the top of this file.


# ============================================================================
# Stage 3: 暴力穷举回测
# ============================================================================


def brute_force_backtest(
    norm_df, top_k, drop_n, benchmark, freq,
    min_combo_size, max_combo_size, output_dir, anchor_date=None, resume=False,
    n_jobs=4, use_groups=False, group_config=None,
    batch_size=50, results_filename="results.csv",
    weight_method="equal",
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

    print(f"\n{'='*60}")
    print("Stage 3: 暴力穷举回测 (Batched ProcessPool + Checkpoint)")
    print(f"{'='*60}")

    model_candidates = list(norm_df.columns)

    # 准备 Exchange 构造参数（不在主进程构造 Exchange，由 worker 各自创建）
    bt_start = str(norm_df.index.get_level_values(0).min().date())
    bt_end = str(norm_df.index.get_level_values(0).max().date())
    all_codes = sorted(norm_df.index.get_level_values(1).unique().tolist())

    from quantpits.utils import strategy
    st_config = strategy.load_strategy_config()
    bt_config = strategy.get_backtest_config(st_config)

    exchange_kwargs = bt_config["exchange_kwargs"].copy()
    exchange_freq = exchange_kwargs.pop("freq", "day")

    print(f"Exchange params prepared. Period: {bt_start} ~ {bt_end}, Instruments: {len(all_codes)}")

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
    print(f"并发进程数: {n_jobs}, 批次大小: {batch_size}")

    # ── Resume: 加载已有结果 ──
    csv_path = os.path.join(output_dir, results_filename)
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

        # ── 分批处理 (单一进程池，spawn 避免 fork 锁继承) ──
        pbar = tqdm(
            total=len(pending),
            desc=f"Brute Force (Workers={n_jobs})",
            unit="combo",
        )

        # ── 动态权重预计算 (仅在主进程，每模型全局计算一次) ──
        weight_df = None
        if weight_method == "rolling_sharpe":
            weight_df = compute_rolling_sharpe_weights(norm_df, top_k)

        # spawn: 每个 worker 是全新 Python 进程，不继承主进程的 Qlib 锁状态
        # 同时限制 BLAS 线程数，避免 8 worker × 32 OpenBLAS threads 撑爆 pthread_create
        import multiprocessing as mp
        _mp_ctx = mp.get_context("spawn")

        _saved_env = {}
        for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            _saved_env[_key] = os.environ.get(_key)
            os.environ[_key] = "1"

        try:
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=_mp_ctx,
                initializer=worker_init,
                initargs=(
                    norm_df, exchange_kwargs, all_codes,
                    bt_start, bt_end, exchange_freq,
                    st_config, bt_config, weight_df,
                ),
            ) as executor:
                for batch_start in range(0, len(pending), batch_size):
                    if _su._shutdown:
                        break

                    batch = pending[batch_start : batch_start + batch_size]
                    batch_results = []

                    future_to_combo = {
                        executor.submit(
                            run_backtest_in_worker,
                            combo,
                            top_k,
                            drop_n,
                            benchmark,
                            freq,
                        ): combo
                        for combo in batch
                    }

                    for future in as_completed(future_to_combo):
                        if _su._shutdown:
                            for f in future_to_combo:
                                f.cancel()
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

        finally:
            pbar.close()
            _restore_signal_handlers()
            qlib_log.setLevel(original_level)
            # 恢复 BLAS 线程设置
            for _key, _val in _saved_env.items():
                if _val is not None:
                    os.environ[_key] = _val
                else:
                    os.environ.pop(_key, None)

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
        "--output-dir", type=str, default="output/ensemble_runs",
        help="输出根目录 (默认: output/ensemble_runs)",
    )
    parser.add_argument(
        "--run-label", type=str, default="",
        help="运行标签，注入输出目录名以避免同日期多次运行互相覆盖。"
             "例如 --run-label rolling 会产生 brute_force_2026-06-05_rolling/",
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
        help="并发回测进程数 (默认: 4)",
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
    parser.add_argument(
        "--norm-method", type=str, default="rank", choices=["zscore", "rank"],
        help="截面归一化方法 (默认: zscore)",
    )
    parser.add_argument(
        "--weight-method", type=str, default="equal",
        choices=["equal", "rolling_sharpe"],
        help="融合权重方法 (默认: equal, 可选: rolling_sharpe 动态权重)",
    )
    args = parser.parse_args()

    from quantpits.utils.operator_log import OperatorLog
    with OperatorLog("brute_force_ensemble", args=sys.argv[1:]) as oplog:
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

        # 构建 RunContext
        from quantpits.utils.run_context import RunContext
        ctx = RunContext(
            base_dir=args.output_dir,
            script_name="brute_force",
            anchor_date=anchor_date,
            run_label=args.run_label.strip() if args.run_label else "",
        )
        ctx.ensure_dirs()
        print(f"输出目录: {ctx.run_dir}")

        # Stage 1: 加载预测数据 (应用 training-mode 过滤)
        if getattr(args, 'training_mode', None):
            from quantpits.utils.train_utils import filter_models_by_mode
            filtered = filter_models_by_mode(train_records.get('models', {}), args.training_mode)
            train_records = dict(train_records)
            train_records['models'] = filtered
            print(f"训练模式过滤: {args.training_mode} (剩余 {len(filtered)} 个模型)")

        # Stage 1b: 应用 --use-groups 过滤 (在加载前缩小模型范围，避免全量计算)
        selected_models = None
        if args.use_groups and args.group_config:
            group_models = extract_group_model_names(args.group_config)
            current_models = set(train_records.get('models', {}).keys())
            selected_models = sorted(group_models & current_models)
            missing = group_models - current_models
            if missing:
                print(f"warning: 分组配置中的模型不在训练记录中: {sorted(missing)}，已忽略")
            if not selected_models:
                print("错误: --use-groups 过滤后无有效模型可加载！")
                print(f"  分组配置中的模型: {sorted(group_models)}")
                print(f"  训练记录中的模型: {sorted(current_models)}")
                sys.exit(1)
            print(f"分组过滤: 从 {len(current_models)} 个模型中选择 {len(selected_models)} 个")

        norm_df, model_metrics = load_predictions(
            train_records, norm_method=args.norm_method, selected_models=selected_models
        )
        
        # 划分数据集 (IS / OOS)
        is_norm_df, oos_norm_df = split_is_oos_by_args(norm_df, args)
        if is_norm_df.empty:
            print("错误: IS 期无数据！请检查日期参数。")
            sys.exit(1)

        # Stage 2: 相关性分析
        corr_matrix = correlation_analysis(is_norm_df, ctx.is_dir)

        # Stage 3: 暴力回测 (基于 IS)
        results_df = brute_force_backtest(
            norm_df=is_norm_df,
            top_k=top_k,
            drop_n=drop_n,
            benchmark=benchmark,
            freq=args.freq,
            min_combo_size=args.min_combo_size,
            max_combo_size=args.max_combo_size,
            output_dir=ctx.is_dir,
            resume=args.resume,
            n_jobs=args.n_jobs,
            use_groups=args.use_groups,
            group_config=args.group_config,
            batch_size=args.batch_size,
            weight_method=args.weight_method,
        )

        # 导出 Metadata
        is_dates_all = is_norm_df.index.get_level_values("datetime")
        oos_dates_all = oos_norm_df.index.get_level_values("datetime")
        
        from quantpits.utils.search_utils import save_run_metadata
        run_label = args.run_label.strip() if getattr(args, 'run_label', None) else ""
        metadata_path = save_run_metadata(ctx, {
            "anchor_date": anchor_date,
            "script_used": "brute_force_ensemble",
            "run_label": run_label,
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
            "group_config": args.group_config,
            "norm_method": args.norm_method,
            "weight_method": args.weight_method,
        })
        print(f"请使用以下命令进行分析与 OOS 验证:")
        print(f"  python quantpits/scripts/analyze_ensembles.py --metadata {metadata_path}")

        print(f"\n{'='*60}")
        print(f"全部完成！ 耗时结束于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {ctx.run_dir}")
        print(f"{'='*60}")

        oplog.set_result({
            "n_models": len(norm_df.columns),
            "n_combinations": len(results_df) if not results_df.empty else 0,
            "anchor_date": anchor_date
        })


if __name__ == "__main__":
    main()
