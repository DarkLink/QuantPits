"""
组合搜索共享工具 — 信号处理、回测执行、IS/OOS 切分、分组穷举

从 brute_force_ensemble.py / brute_force_fast.py / minentropy_ensemble.py / analyze_ensembles.py
中抽取的公共逻辑，消除 hacky cross-script import 和代码重复。
"""

import os
import sys
import signal
import itertools
import yaml
import pandas as pd
import numpy as np


# ============================================================================
# 全局中断标志 & 信号处理
# ============================================================================
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
# 单组合回测核心
# ============================================================================

def run_single_backtest(
    combo_models, norm_df, top_k, drop_n, benchmark, freq,
    trade_exchange, bt_start, bt_end, st_config=None, bt_config=None
):
    """对指定的模型组合进行回测，返回指标字典或 None"""
    from quantpits.utils import strategy
    from quantpits.utils.backtest_utils import run_backtest_with_strategy, standard_evaluate_portfolio

    if st_config is None:
        st_config = strategy.load_strategy_config()
    if bt_config is None:
        bt_config = strategy.get_backtest_config(st_config)

    # 1. 合成信号 (等权均值，归一化后的) (仅在当前组合子集上求交集 dropna)
    combo_score = norm_df[list(combo_models)].dropna(how='any').mean(axis=1)

    import copy
    st_config = copy.deepcopy(st_config)
    st_config["strategy"]["params"]["topk"] = top_k
    st_config["strategy"]["params"]["n_drop"] = drop_n

    strategy_inst = strategy.create_backtest_strategy(combo_score, st_config)

    # 2. 回测
    try:
        report, _ = run_backtest_with_strategy(
            strategy_inst=strategy_inst,
            trade_exchange=trade_exchange,
            freq=freq,
            account_cash=bt_config["account"],
            bt_start=bt_start,
            bt_end=bt_end
        )

        # 3. 标准化计算结果
        st_config_inner = strategy.load_strategy_config()
        benchmark_col = st_config_inner.get('benchmark', 'SH000300')
        
        metrics = standard_evaluate_portfolio(report, benchmark_col, freq)

        return {
            "models": ",".join(combo_models),
            "n_models": len(combo_models),
            "Ann_Ret": metrics.get("CAGR_252", 0),
            "Max_DD": metrics.get("Max_Drawdown", 0),
            "Excess_Ret": metrics.get("Absolute_Return", 0) - metrics.get("Benchmark_Absolute_Return", 0),
            "Ann_Excess": metrics.get("Excess_Return_CAGR_252", 0),
            "Total_Ret": metrics.get("Absolute_Return", 0),
            "Final_NAV": report.iloc[-1]["account"],
            "Calmar": metrics.get("Calmar", 0) if pd.notna(metrics.get("Calmar")) else 0,
        }
    except Exception as e:
        print(f"  [ERROR] Combo {combo_models} failed: {e}")
        return None


# ============================================================================
# CSV 增量写入
# ============================================================================

def _append_results_to_csv(csv_path, results, write_header=False):
    """将一批结果追加写入 CSV 文件"""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ============================================================================
# IS/OOS 数据集切分
# ============================================================================

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
    
    # OOS 是截止日之后的数据
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
# 模型分组 & 组合生成
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
