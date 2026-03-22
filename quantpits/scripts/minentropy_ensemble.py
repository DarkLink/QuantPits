#!/usr/bin/env python
"""
MinEntropy (mRMR) Ensemble - 基于最小熵（最小冗余最大相关）的等权组合搜索

此脚本不使用暴力穷举，而是通过评估单模型的预测相关性(Redundancy)和预测能力(Relevance)，
采用贪婪算法(mRMR)高效地搜索出具有高度多样性且表现优异的模型组合。
该方法有效避免了在大量模型中暴力搜索带来的过拟合问题。

运行方式：
  cd QuantPits && python quantpits/scripts/minentropy_ensemble.py
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
from concurrent.futures import ThreadPoolExecutor, as_completed

from qlib.backtest.exchange import Exchange
from quantpits.utils import strategy

from quantpits.utils import env
os.chdir(env.ROOT_DIR)

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from quantpits.scripts.brute_force_ensemble import (
    _install_signal_handlers, _restore_signal_handlers, _signal_handler,
    init_qlib, load_config, load_predictions, split_is_oos_by_args,
    correlation_analysis, run_single_backtest, _append_results_to_csv
)

# 借用 brute_force 中的 _shutdown
import quantpits.scripts.brute_force_ensemble as bfe_module

def minentropy_backtest(
    norm_df, top_k, drop_n, benchmark, freq,
    max_combo_size, output_dir, anchor_date, resume=False,
    n_jobs=4, batch_size=50,
):
    """
    基于 mRMR (Minimum Redundancy Maximum Relevance) 计算候选集，然后回测。
    """
    bfe_module._shutdown = False
    
    # 强制确保索引名称正确，防止某些操作导致名称丢失
    if not norm_df.empty:
        norm_df.index.names = ["datetime", "instrument"]
    
    print(f"\n{'='*60}")
    print("Stage 3: MinEntropy (mRMR) 组合搜索与回测")
    print(f"{'='*60}")

    model_candidates = list(norm_df.columns)
    if not model_candidates:
        print("没有可用模型！")
        return pd.DataFrame()

    print("Initializing Shared Exchange...")
    bt_start = str(norm_df.index.get_level_values(0).min().date())
    bt_end = str(norm_df.index.get_level_values(0).max().date())
    all_codes = sorted(norm_df.index.get_level_values(1).unique().tolist())

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

    # 1. 计算相关性矩阵 (Redundancy basis)
    print("\n[ mRMR Step 1 ] 计算预测值相关性矩阵...")
    corr_matrix = norm_df.corr().fillna(0)

    # 2. 单模型验证 (Relevance basis)
    print(f"\n[ mRMR Step 2 ] 准备单模型基准性能测试用于 Relevance...")
    single_results = {}
    print(f"正在快速评估 {len(model_candidates)} 个单模型...")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        future_to_model = {
            executor.submit(
                run_single_backtest,
                [m], norm_df, top_k, drop_n, benchmark, freq,
                trade_exchange, bt_start, bt_end, st_config, bt_config
            ): m
            for m in model_candidates
        }
        for future in tqdm(as_completed(future_to_model), total=len(model_candidates), desc="Single Models"):
            m = future_to_model[future]
            try:
                res = future.result()
                if res and res.get("Ann_Excess") is not None:
                    single_results[m] = res["Ann_Excess"]
                else:
                    single_results[m] = 0.0
            except Exception as e:
                single_results[m] = 0.0

    # 归一化 Relevance
    if not single_results:
        print("所有单模型评估失败，无法进行 mRMR 搜索！")
        return pd.DataFrame()
        
    s_vals = list(single_results.values())
    min_score, max_score = min(s_vals), max(s_vals)
    range_score = max_score - min_score if (max_score - min_score) > 1e-6 else 1.0
    
    def scale_relevance(raw_score):
        return (raw_score - min_score) / range_score

    # 3. mRMR 贪婪搜索组合
    print(f"\n[ mRMR Step 3 ] 开始 mRMR 组合空间搜索 (max_combo_size={max_combo_size})...")
    # 设置不同的惩罚力度 lambda，0.0 代表纯贪婪收益，越大约倾向于寻找低相关性互补的模型
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    all_combinations_set = set()
    
    for lmbda in lambdas:
        for start_m in model_candidates:
            selected = [start_m]
            # 记录不同长度的过程组合（比如当 max_combo_size=5，记录长度为1,2,3,4,5的所有组合）
            all_combinations_set.add(tuple(sorted(selected)))
            
            for _ in range(max_combo_size - 1):
                best_score = -9999.0
                best_m = None
                
                for m in model_candidates:
                    if m in selected:
                        continue
                    
                    relevance = scale_relevance(single_results[m])
                    # Redundancy: 候选模型与已选模型的相关性均值
                    redundancy = np.mean([corr_matrix.loc[m, y] for y in selected])
                    
                    score = relevance - lmbda * redundancy
                    if score > best_score:
                        best_score = score
                        best_m = m
                        
                if best_m is not None:
                    selected.append(best_m)
                    all_combinations_set.add(tuple(sorted(selected)))

    # 提取有效的独立组合
    all_combinations = [list(c) for c in all_combinations_set if len(c) > 1]
    # (单模型之前测过了，这里仅对 >1 的组合跑详细回测保存，单模型的记录已经在 results 里其实最好也存下来)
    all_combinations = [list(c) for c in all_combinations_set] 
    # 从元组转为列表，保证组合顺序对内一致
    all_combinations_str = [",".join(c) for c in all_combinations]
    
    print(f"mRMR 启发式搜索为您精选出 {len(all_combinations)} 个高质量候选组合！(若穷举可能达千万级)")
    
    # ── Resume: 加载已有结果 ──
    csv_path = os.path.join(output_dir, f"minentropy_results_{anchor_date}.csv")
    done_combos = set()

    if resume and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_combos = set(existing_df["models"].tolist())
        print(f"Resume 模式: 已有 {len(done_combos)} 个组合，跳过")

    pending = [c for c in all_combinations if ",".join(c) not in done_combos]
    print(f"待回测组合数: {len(pending)}")

    if not pending:
        print("所有组合已完成！")
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
        else:
            results_df = pd.DataFrame()
    else:
        need_header = not (resume and os.path.exists(csv_path))
        if need_header:
            header_cols = [
                "models", "n_models", "Ann_Ret", "Max_DD",
                "Excess_Ret", "Ann_Excess", "Total_Ret", "Final_NAV", "Calmar"
            ]
            pd.DataFrame(columns=header_cols).to_csv(csv_path, index=False)

        qlib_log = logging.getLogger("qlib")
        original_level = qlib_log.level
        qlib_log.setLevel(logging.WARNING)

        _install_signal_handlers()

        completed_count = len(done_combos)
        total_count = len(all_combinations)
        failed_count = 0

        pbar = tqdm(
            total=len(pending),
            desc=f"MinEntropy Eval (Threads={n_jobs})",
            unit="combo",
        )

        try:
            for batch_start in range(0, len(pending), batch_size):
                if bfe_module._shutdown:
                    break

                batch = pending[batch_start : batch_start + batch_size]
                batch_results = []

                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_combo = {
                        executor.submit(
                            run_single_backtest,
                            combo, norm_df, top_k, drop_n, benchmark, freq,
                            trade_exchange, bt_start, bt_end, st_config, bt_config
                        ): combo
                        for combo in batch
                    }

                    for future in as_completed(future_to_combo):
                        if bfe_module._shutdown:
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

                if batch_results:
                    _append_results_to_csv(csv_path, batch_results, write_header=False)

                del batch_results
                gc.collect()

                if bfe_module._shutdown:
                    break

        finally:
            pbar.close()
            _restore_signal_handlers()
            qlib_log.setLevel(original_level)

        if bfe_module._shutdown:
            print(f"\n⚠️  已安全中断！")
            print(f"   已完成: {completed_count}/{total_count} 组合")
            print(f"   结果保存至: {csv_path}")
        else:
            print(f"\n✅ 回测全部完成！")
            print(f"   有效: {completed_count - len(done_combos)}, 失败: {failed_count}")

        results_df = pd.read_csv(csv_path)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by="Ann_Excess", ascending=False
        ).reset_index(drop=True)
        results_df.to_csv(csv_path, index=False)
        print(f"结果已排序保存: {csv_path} ({len(results_df)} 条)")
    else:
        print("警告: 无有效回测结果")

    return results_df


def main():
    env.safeguard("MinEntropy Ensemble Search")
    parser = argparse.ArgumentParser(
        description="MinEntropy (mRMR) 组合搜索验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--record-file", type=str, default="latest_train_records.json")
    parser.add_argument("--training-mode", type=str, default=None, choices=["static", "rolling"])
    parser.add_argument("--max-combo-size", type=int, default=5, help="智能搜索的最大组合大小 (默认: 5)")
    parser.add_argument("--freq", type=str, default=None, choices=["day", "week"])
    parser.add_argument("--output-dir", type=str, default="output/brute_force")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--exclude-last-years", type=int, default=0)
    parser.add_argument("--exclude-last-months", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--mrmr-candidate-size", type=int, default=10, help="mRMR 每步保留的候选池大小 (默认: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("MinEntropy Ensemble - 基于信息论的低冗余高收益组合搜寻 (mRMR)")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    init_qlib()
    
    train_records, model_config = load_config(args.record_file)
    freq = args.freq or model_config.get("freq", "week")
    args.freq = freq
    
    anchor_date = train_records.get("anchor_date", datetime.now().strftime("%Y-%m-%d"))
    top_k = model_config.get("TopK", 22)
    drop_n = model_config.get("DropN", 3)
    benchmark = model_config.get("benchmark", "SH000300")

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: 加载预测数据 (应用 training-mode 过滤)
    if getattr(args, 'training_mode', None):
        from quantpits.utils.train_utils import filter_models_by_mode
        filtered = filter_models_by_mode(train_records.get('models', {}), args.training_mode)
        train_records = dict(train_records)
        train_records['models'] = filtered
        print(f"训练模式过滤: {args.training_mode} (剩余 {len(filtered)} 个模型)")

    norm_df, _ = load_predictions(train_records)
    if not norm_df.empty:
        norm_df.index.names = ["datetime", "instrument"]
    is_norm_df, oos_norm_df = split_is_oos_by_args(norm_df, args)
    if is_norm_df.empty:
        print("错误: IS 期无数据！请检查日期参数。")
        sys.exit(1)

    # 1. 相关性分析 (借用 brute_force 的功能落盘)
    correlation_analysis(is_norm_df, args.output_dir, anchor_date)

    # 2. MinEntropy mRMR 回测
    results_df = minentropy_backtest(
        norm_df=is_norm_df,
        top_k=top_k,
        drop_n=drop_n,
        benchmark=benchmark,
        freq=args.freq,
        max_combo_size=args.max_combo_size,
        output_dir=args.output_dir,
        anchor_date=anchor_date,
        resume=args.resume,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
    )

    # 3. 导出 Metadata (标记 script_used: minentropy 以便后续处理)
    metadata_path = os.path.join(args.output_dir, f"minentropy_metadata_{anchor_date}.json")
    is_dates_all = is_norm_df.index.get_level_values("datetime") if not is_norm_df.empty else pd.Series()
    oos_dates_all = oos_norm_df.index.get_level_values("datetime") if not oos_norm_df.empty else pd.Series()
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "anchor_date": anchor_date,
            "script_used": "minentropy",
            "freq": args.freq,
            "record_file": args.record_file,
            "training_mode": getattr(args, "training_mode", None),
            "is_start_date": str(is_dates_all.min().date()) if not is_dates_all.empty else None,
            "is_end_date": str(is_dates_all.max().date()) if not is_dates_all.empty else None,
            "oos_start_date": str(oos_dates_all.min().date()) if not oos_dates_all.empty else None,
            "oos_end_date": str(oos_dates_all.max().date()) if not oos_dates_all.empty else None,
            "exclude_last_years": args.exclude_last_years,
            "exclude_last_months": args.exclude_last_months,
            "mrmr_candidate_size": args.mrmr_candidate_size,
        }, f, indent=4)
    print(f"\n✅ 元数据已保存: {metadata_path}")
    print("请使用 quantpits/scripts/analyze_ensembles.py 单独进行分析与 OOS 验证。")

if __name__ == "__main__":
    main()
