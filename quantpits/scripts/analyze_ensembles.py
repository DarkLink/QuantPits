#!/usr/bin/env python
"""
Analyze Ensembles - 多维度 OOS (Out-Of-Sample) 候选池验证分析

用法:
  python quantpits/scripts/analyze_ensembles.py --metadata output/brute_force/run_metadata_2026-03-21.json

此脚本将读取运行生成的 metadata 与 IS 回测结果，自动恢复数据集划分，
并在 OOS 数据上对多个不同“派系”的优秀候选组合（高收益、高稳健、低相关等）进行准确的回测验证。
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import chain
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from quantpits.utils import env
os.chdir(env.ROOT_DIR)
env.init_qlib()

from quantpits.utils import strategy
from quantpits.utils.backtest_utils import run_backtest_with_strategy, standard_evaluate_portfolio
from qlib.backtest.exchange import Exchange

def run_single_backtest_oos(
    combo_models, norm_df, top_k, drop_n, benchmark, freq,
    trade_exchange, bt_start, bt_end, st_config=None, bt_config=None
):
    """单独运行一次标准回测，用于 OOS 精确验证"""
    if st_config is None:
        st_config = strategy.load_strategy_config()
    if bt_config is None:
        bt_config = strategy.get_backtest_config(st_config)

    # 1. 合成信号
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

        st_config_inner = strategy.load_strategy_config()
        benchmark_col = st_config_inner.get('benchmark', 'SH000300')
        metrics = standard_evaluate_portfolio(report, benchmark_col, freq)

        return {
            "models": ",".join(combo_models),
            "n_models": len(combo_models),
            "Ann_Ret": metrics.get("CAGR", 0),
            "Max_DD": metrics.get("Max_Drawdown", 0),
            "Excess_Ret": metrics.get("Absolute_Return", 0) - metrics.get("Benchmark_Absolute_Return", 0),
            "Ann_Excess": metrics.get("Excess_Return_CAGR", 0),
            "Total_Ret": metrics.get("Absolute_Return", 0),
            "Final_NAV": report.iloc[-1]["account"],
            "Calmar": metrics.get("Calmar", 0) if pd.notna(metrics.get("Calmar")) else 0,
        }
    except Exception as e:
        print(f"  [ERROR] Combo {combo_models} failed: {e}")
        return None




# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def generate_is_visualizations_and_report(df, metadata_dir, anchor_date, prefix="", top_n=50):
    print("\n📊 === 生成 IS 阶段可视化与分析报告 ===")
    
    report_lines = []
    
    # 格式化
    fmt = {
        "Ann_Excess": "{:.2%}".format,
        "Max_DD": "{:.2%}".format,
        "Calmar": "{:.2f}".format,
        "avg_corr": "{:.4f}".format,
    }
    display_cols = ["models", "n_models", "Ann_Excess", "Max_DD", "Calmar"]
    
    # 1. 模型归因分析 (Model Attribution)
    try:
        top_combinations = df.sort_values("Calmar", ascending=False).head(top_n)
        bottom_combinations = df.sort_values("Calmar", ascending=True).head(top_n)

        def get_model_counts(series_of_lists):
            all_models = list(chain.from_iterable(series_of_lists))
            return pd.Series(Counter(all_models)).sort_values(ascending=False)

        top_counts = get_model_counts(top_combinations["models"].apply(lambda x: str(x).split(",")))
        bottom_counts = get_model_counts(bottom_combinations["models"].apply(lambda x: str(x).split(",")))

        attribution = pd.DataFrame(
            {"Top_Count": top_counts, "Bottom_Count": bottom_counts}
        ).fillna(0)
        attribution["Net_Score"] = attribution["Top_Count"] - attribution["Bottom_Count"]
        attribution = attribution.sort_values("Net_Score", ascending=False)

        attr_path = os.path.join(metadata_dir, f"{prefix}model_attribution_{anchor_date}.csv")
        attribution.to_csv(attr_path)

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
        ax.set_title(f"Model Importance Analysis (Top/Bottom {top_n} by Calmar)")
        ax.legend()
        plt.tight_layout()
        attr_fig_path = os.path.join(metadata_dir, f"{prefix}model_attribution_{anchor_date}.png")
        plt.savefig(attr_fig_path, dpi=150)
        plt.close()
        print(f"归因图已保存: {attr_fig_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"模型归因分析失败: {e}")
        attribution = pd.DataFrame()

    # 读取相关性矩阵并计算 avg_corr
    try:
        corr_file = os.path.join(metadata_dir, f"{prefix}correlation_matrix_{anchor_date}.csv")
        # 兼容旧版本没有任何前缀的情况
        if not os.path.exists(corr_file):
            corr_file = os.path.join(metadata_dir, f"correlation_matrix_{anchor_date}.csv")
        if os.path.exists(corr_file):
            corr_matrix = pd.read_csv(corr_file, index_col=0)
            
            single_model_perf = df[df["n_models"] == 1].set_index("models")["Ann_Excess"].to_dict()
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
                avg_individual_ret = np.mean([single_model_perf.get(m, np.nan) for m in models_list])
                combo_to_metrics[combo_str] = (avg_corr, avg_individual_ret)

            metrics_df = pd.DataFrame.from_dict(combo_to_metrics, orient="index", columns=["avg_corr", "avg_ind"])
            df = df.merge(metrics_df, left_on="models", right_index=True, how="left")
            df["diversity_bonus"] = df["Ann_Excess"] - df["avg_ind"]
            df = df.drop(columns=["avg_ind"])
    except Exception as e:
        print(f"相关性计算失败: {e}")

    # 生成文字版 Report
    try:
        df_sorted = df.sort_values("Calmar", ascending=False)
        report_lines.append("=== Top 20 组合 (按 Calmar) ===")
        cols_to_show = display_cols + (["avg_corr"] if "avg_corr" in df.columns else [])
        report_lines.append(df_sorted[cols_to_show].head(20).to_string(formatters=fmt))

        robust = df[df["Ann_Ret"] > 0].sort_values("Calmar", ascending=False)
        report_lines.append("\n=== 绝对稳健组合 (收益>0, 按 Calmar) ===")
        report_lines.append(robust[cols_to_show].head(10).to_string(formatters=fmt))

        if "avg_corr" in df.columns:
            golden = df[(df["avg_corr"] < 0.3) & (df["Calmar"] > df["Calmar"].quantile(0.9))]
            if not golden.empty:
                report_lines.append(f"\n=== 黄金组合 ({len(golden)} 个) ===")
                report_lines.append(golden[cols_to_show].head(10).to_string(formatters=fmt))
            else:
                report_lines.append("\n=== 黄金组合: 未发现 ===")

        group_stats = df.groupby("n_models")[["Ann_Excess", "Calmar"]].agg(["median", "mean", "max"])
        report_lines.append("\n=== 按模型数量分组统计 ===")
        report_lines.append(group_stats.round(4).to_string())

        # 汇总前言
        summary = []
        summary.append("=" * 60)
        summary.append(f"自动分析报告摘要 ({anchor_date})")
        summary.append("=" * 60)

        best_combo = df_sorted.iloc[0] if not df_sorted.empty else None
        if best_combo is not None:
            summary.append(f"\n1. 最佳组合 (按 Calmar):")
            summary.append(f"   模型: {best_combo['models']}")
            summary.append(f"   模型数: {best_combo['n_models']}")
            summary.append(f"   年化超额: {best_combo['Ann_Excess']:.2%}")
            summary.append(f"   最大回撤: {best_combo['Max_DD']:.2%}")
            summary.append(f"   Calmar: {best_combo['Calmar']:.2f}")
            if "avg_corr" in best_combo and pd.notna(best_combo["avg_corr"]):
                summary.append(f"   内部相关性: {best_combo['avg_corr']:.4f}")

        if "diversity_bonus" in df.columns:
            best_diversity_idx = df["diversity_bonus"].idxmax()
            best_diversity = df.loc[best_diversity_idx] if pd.notna(best_diversity_idx) else None
            if best_diversity is not None and pd.notna(best_diversity.get("diversity_bonus")):
                summary.append(f"\n2. 最大多样性红利组合:")
                summary.append(f"   模型: {best_diversity['models']}")
                summary.append(f"   Diversity Bonus: {best_diversity['diversity_bonus']:.4%}")

        if not attribution.empty:
            summary.append(f"\n3. 建议保留的核心模型 (MVP):")
            summary.append(f"   {attribution.index[:3].tolist()}")
        
        summary.append("=" * 60)
        
        report_path = os.path.join(metadata_dir, f"{prefix}analysis_report_{anchor_date}.txt")
        full_report = "\n".join(summary) + "\n\n" + "\n".join(report_lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"IS 综合评估报告已保存: {report_path}")

    except Exception as e:
        print(f"IS 综合评估报告生成失败: {e}")

    # 2. 风险-收益散点图 (Risk Return Scatter) & 相关性 vs 实盘图
    try:
        df = df.copy()
        df["Ann_Excess"] = pd.to_numeric(df["Ann_Excess"], errors="coerce")
        df["Max_DD"] = pd.to_numeric(df["Max_DD"], errors="coerce")
        df["n_models"] = pd.to_numeric(df["n_models"], errors="coerce")
        
        if "n_models" in df.columns:
            multi_df = df[df["n_models"] > 1].copy()
            if not multi_df.empty:
                MAX_PLOT_POINTS = 50000
                if len(multi_df) > MAX_PLOT_POINTS:
                    plot_df = multi_df.sample(n=MAX_PLOT_POINTS, random_state=42)
                else:
                    plot_df = multi_df

                has_corr = "avg_corr" in plot_df.columns and plot_df["avg_corr"].notna().any()
                if has_corr:
                    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                    ax0, ax1 = axes[0], axes[1]
                else:
                    fig, ax0 = plt.subplots(figsize=(10, 8))

                scatter = ax0.scatter(
                    plot_df["Max_DD"].abs(), plot_df["Ann_Excess"],
                    c=plot_df["n_models"], cmap="viridis", alpha=0.6,
                    s=plot_df["n_models"] * 10 + 20,
                )
                singles = df[df["n_models"] == 1]
                if not singles.empty:
                    ax0.scatter(
                        singles["Max_DD"].abs(), singles["Ann_Excess"],
                        color="red", marker="x", s=100, label="Single Model",
                    )
                ax0.set_xlabel("Max Drawdown (Absolute)")
                ax0.set_ylabel("Ann Excess Return")
                ax0.set_title("IS Risk vs Return (All Subsets)")
                ax0.legend()
                plt.colorbar(scatter, ax=ax0, label="# Models")
                
                if has_corr:
                    sns.scatterplot(
                        x="avg_corr", y="Calmar", hue="n_models",
                        palette="viridis", data=plot_df, ax=ax1, alpha=0.7,
                    )
                    # Use regplot
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        sns.regplot(
                            x="avg_corr", y="Calmar", data=plot_df,
                            scatter=False, ax=ax1, color="red",
                            line_kws={"linestyle": "--"},
                        )
                    ax1.set_title("IS Correlation vs Calmar")
                    ax1.set_xlabel("Avg Intra-Ensemble Correlation")

                plt.tight_layout()
                scatter_path = os.path.join(metadata_dir, f"{prefix}risk_return_scatter_{anchor_date}.png")
                plt.savefig(scatter_path, dpi=150)
                plt.close()
                print(f"IS 散点图已保存: {scatter_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"IS 散点图生成跳过/失败: {e}")

    return df  # Return df in case it has updated avg_corr

def generate_dendrogram(metadata_dir, anchor_date, prefix=""):
    print("\n🌳 === 生成聚类树状图 ===")
    try:
        corr_file = os.path.join(metadata_dir, f"{prefix}correlation_matrix_{anchor_date}.csv")
        if not os.path.exists(corr_file):
            corr_file = os.path.join(metadata_dir, f"correlation_matrix_{anchor_date}.csv")
        if os.path.exists(corr_file):
            corr_matrix = pd.read_csv(corr_file, index_col=0)
            # Ward linkage expects distance, but we can feed it corr_matrix if we convert it to distance
            # For correlation matrix, distance = col_num dimension matrix? Actually linkage takes squareform distance or observation vectors.
            # Convert correlation to distance matrix: D = 1 - corr
            distance_matrix = 1 - corr_matrix.fillna(0)
            import scipy.spatial.distance as ssd
            dist_array = ssd.squareform(distance_matrix.clip(0, 2))
            
            linked = linkage(dist_array, "ward")
            fig, ax = plt.subplots(figsize=(12, 7))
            dendrogram(
                linked, orientation="top", labels=corr_matrix.columns.tolist(),
                distance_sort="descending", show_leaf_counts=True, ax=ax,
            )
            ax.set_title("Model Prediction Cluster Dendrogram (Ward)")
            ax.set_ylabel("Ward Distance")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            dendro_path = os.path.join(metadata_dir, f"{prefix}cluster_dendrogram_{anchor_date}.png")
            plt.savefig(dendro_path, dpi=150)
            plt.close()
            print(f"聚类图已保存: {dendro_path}")
        else:
            print("找不到相关性矩阵文件，跳过聚类图。")
    except Exception as e:
        print(f"聚类分析失败: {e}")

def generate_oos_visualizations(oos_df, metadata_dir, anchor_date, prefix=""):
    print("\n📈 === 生成 OOS 阶段可视化分析 ===")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        categories = oos_df["Pool_Sources"].unique()
        colors = sns.color_palette("husl", len(categories))
        
        for i, cat in enumerate(categories):
            cat_df = oos_df[oos_df["Pool_Sources"] == cat]
            ax.scatter(
                cat_df["Max_DD"].abs(), cat_df["Ann_Excess"],
                label=cat, color=colors[i], s=100, alpha=0.8, edgecolors="white"
            )
            
        ax.set_xlabel("OOS Max Drawdown (Absolute)")
        ax.set_ylabel("OOS Ann Excess Return")
        ax.set_title("OOS Validation: Risk vs Return of Selected Candidates")
        ax.legend(title="Candidate Pool Source", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scatter_path = os.path.join(metadata_dir, f"{prefix}oos_risk_return_{anchor_date}.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"OOS 散点图已保存: {scatter_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"OOS 散点图绘制失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="多维度组合分析及 OOS 验证")
    parser.add_argument("--metadata", type=str, required=True, help="穷举生成的 metadata JSON 文件路径")
    parser.add_argument("--top-n", type=int, default=5, help="各个维度默认提取的 Top 数量 (默认: 5)")
    parser.add_argument("--top-n-yield", type=int, help="绝对收益派 Top 数量 (覆盖 --top-n)")
    parser.add_argument("--top-n-robust", type=int, help="绝对稳健派 Top 数量 (覆盖 --top-n)")
    parser.add_argument("--top-n-defensive", type=int, help="极致防守派 Top 数量 (覆盖 --top-n)")
    parser.add_argument("--top-n-mvp", type=int, help="单模型基准 Top 数量 (覆盖 --top-n)")
    parser.add_argument("--top-n-diversity", type=int, help="黄金多样式 Top 数量 (覆盖 --top-n)")
    parser.add_argument("--training-mode", type=str, default="", help="过滤特定训练模式的模型 (如 'static', 'incremental')")
    parser.add_argument("--max-workers", type=int, default=4, help="OOS回测的并发线程数 (默认: 1)")
    args = parser.parse_args()

    # 1. 解析 Metadata
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    anchor_date = meta["anchor_date"]
    script_used = meta["script_used"]
    freq = meta["freq"]
    record_file = meta["record_file"]
    oos_start_date = meta.get("oos_start_date")
    oos_end_date = meta.get("oos_end_date")

    print("=" * 60)
    print("🚀 Analyze Ensembles - IS 多维提取与 OOS 验证")
    print(f"数据源: {script_used} (发布于 {anchor_date})")
    print(f"OOS 验证周期间隔: {oos_start_date} ~ {oos_end_date}")
    print("=" * 60)

    has_oos = bool(oos_start_date and oos_end_date and str(oos_start_date).lower() != "none" and str(oos_end_date).lower() != "none")
    if not has_oos:
        print("⚠️ 警告：元数据中未找到有效的 OOS 数据周期！本次运行仅执行 IS (In-Sample) 全量分析，将跳过 OOS 验证环节。")

    metadata_dir = os.path.dirname(os.path.abspath(args.metadata))
    
    # 2. 读取 IS Results CSV
    if "fast" in script_used:
        results_file = os.path.join(metadata_dir, f"brute_force_fast_results_{anchor_date}.csv")
    elif "minentropy" in script_used:
        results_file = os.path.join(metadata_dir, f"minentropy_results_{anchor_date}.csv")
    else:
        results_file = os.path.join(metadata_dir, f"brute_force_results_{anchor_date}.csv")

    if not os.path.exists(results_file):
        print(f"❌ 找不到 IS 结果文件: {results_file}")
        sys.exit(1)
        
    df = pd.read_csv(results_file)
    print(f"\n✅ 成功加载 IS 结果文件，共 {len(df)} 组策略记录。")

    if args.training_mode:
        def match_mode(models_str):
            models = str(models_str).split(",")
            return all(m.endswith(f"@{args.training_mode}") for m in models)
        
        df = df[df["models"].apply(match_mode)].copy()
        print(f"🎯 应用 --training-mode '{args.training_mode}' 过滤后，剩余 {len(df)} 组策略记录参加评估。")
        if df.empty:
            print("❌ 过滤后无符合条件的记录，分析终止。")
            sys.exit(0)

    prefix = "" if script_used == "brute_force_ensemble" else f"{script_used.replace('_ensemble', '')}_"

    # 2.5 生成 IS 全量结果可视化
    try:
        df = generate_is_visualizations_and_report(df, metadata_dir, anchor_date, prefix=prefix, top_n=args.top_n)
        generate_dendrogram(metadata_dir, anchor_date, prefix=prefix)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"IS 可视化图表生成跳过/失败: {e}")

    if not has_oos:
        print("\n✅ IS 分析已完成。由于未切分 OOS 数据，后续 OOS 验证环节已跳过。")
        return

    # 3. 构建多维 OOS 候选池
    print("\n📦 === 构建多维 OOS 候选池 ===")
    candidates = {}
    
    df["Ann_Excess"] = pd.to_numeric(df["Ann_Excess"], errors="coerce")
    df["Calmar"] = pd.to_numeric(df["Calmar"], errors="coerce")
    df["Max_DD"] = pd.to_numeric(df["Max_DD"], errors="coerce")
    df["n_models"] = pd.to_numeric(df["n_models"], errors="coerce")
    
    n_yield = args.top_n_yield if args.top_n_yield is not None else args.top_n
    n_robust = args.top_n_robust if args.top_n_robust is not None else args.top_n
    n_defensive = args.top_n_defensive if args.top_n_defensive is not None else args.top_n
    n_mvp = args.top_n_mvp if args.top_n_mvp is not None else args.top_n
    n_diversity = args.top_n_diversity if args.top_n_diversity is not None else args.top_n

    # Pool 1: 绝对收益派 (Yield)
    yield_pool = df.sort_values(by="Ann_Excess", ascending=False).head(n_yield)
    for model in yield_pool["models"]:
        candidates.setdefault(model, set()).add("Yield_Top")
        
    # Pool 2: 绝对稳健派 (Robust)
    robust_pool = df[df["Ann_Ret"] > 0].sort_values(by="Calmar", ascending=False).head(n_robust)
    for model in robust_pool["models"]:
        candidates.setdefault(model, set()).add("Robust_Top")

    # Pool 3: 极致防守派 (Defensive)
    defensive_pool = df[df["Ann_Excess"] > 0.05].copy()
    if not defensive_pool.empty:
        # 最大的意味着回撤绝对值最小 (最接近 0)
        defensive_pool = defensive_pool.sort_values(by="Max_DD", ascending=False).head(n_defensive)
        for model in defensive_pool["models"]:
            candidates.setdefault(model, set()).add("Defensive_Top")

    # Pool 4: 单模型基准 (MVP Baseline)
    mvp_pool = df[df["n_models"] == 1].sort_values(by="Ann_Excess", ascending=False).head(n_mvp)
    for model in mvp_pool["models"]:
        candidates.setdefault(model, set()).add("MVP_Base")
        
    # Pool 5: 黄金多样式 (Diversity Bonus) (若相关列存在)
    if "avg_corr" in df.columns:
        df["avg_corr"] = pd.to_numeric(df["avg_corr"], errors="coerce")
        golden_pool = df[df["avg_corr"] < 0.3].sort_values(by="Calmar", ascending=False).head(n_diversity)
        for model in golden_pool["models"]:
            candidates.setdefault(model, set()).add("Golden_Diversity")

    unique_candidates = list(candidates.keys())
    print(f"共提取 {len(unique_candidates)} 个独特的超级组合进行 OOS 测试。\n")

    # 4. 加载原始预测数据供 OOS 回测
    print("⏳ 正在结合 Metadata 重新对齐 OOS 评估矩阵...")
    from quantpits.utils.config_loader import load_workspace_config
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            train_records = json.load(f)
    else:
        print(f"❌ 找不到 train_records: {record_file}")
        sys.exit(1)
        
    model_config = load_workspace_config(env.ROOT_DIR)
    
    from quantpits.utils.predict_utils import load_predictions_from_recorder
    # 获取归一化过后的全局预测得分
    unique_models = set()
    for c in unique_candidates:
        unique_models.update(c.split(","))
    models_to_load_list = list(unique_models)
    norm_df, _, _ = load_predictions_from_recorder(train_records, selected_models=models_to_load_list)
        
    # 根据元数据精确切分 OOS 区间
    start_date = pd.to_datetime(oos_start_date)
    end_date = pd.to_datetime(oos_end_date)
    oos_mask = (norm_df.index.get_level_values("datetime") >= start_date) & (norm_df.index.get_level_values("datetime") <= end_date)
    oos_norm_df = norm_df[oos_mask]
    
    if oos_norm_df.empty:
        print("❌ 错误：OOS 切分区间内无数据！可能未按指定的排除配置提供足够的日期长度。")
        sys.exit(1)
        
    print(f"✅ OOS 数据准备完毕，共 {len(oos_norm_df.index.get_level_values('datetime').unique())} 个交易日。")
    
    # 5. 执行 OOS 回测验证
    st_config = strategy.load_strategy_config()
    bt_config = strategy.get_backtest_config(st_config)
    
    top_k = model_config.get("TopK", 22)
    drop_n = model_config.get("DropN", 3)
    benchmark = model_config.get("benchmark", "SH000300")
    
    exchange_kwargs = bt_config["exchange_kwargs"].copy()
    exchange_freq = exchange_kwargs.pop("freq", "day")
    all_codes_oos = sorted(oos_norm_df.index.get_level_values(1).unique().tolist())
    
    trade_exchange_oos = Exchange(
        freq=exchange_freq,
        start_time=oos_start_date,
        end_time=oos_end_date,
        codes=all_codes_oos,
        **exchange_kwargs
    )
    
    oos_results = []
    print(f"\n⚔️ === 开始多维精准 OOS 回测验证 (Threads={args.max_workers}) ===")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_combo = {}
        for combo_str in unique_candidates:
            combo = combo_str.split(",")
            future = executor.submit(
                run_single_backtest_oos,
                combo, oos_norm_df, top_k, drop_n, benchmark, freq,
                trade_exchange_oos, oos_start_date, oos_end_date,
                st_config, bt_config
            )
            future_to_combo[future] = combo_str
            
        for future in tqdm(as_completed(future_to_combo), total=len(unique_candidates), desc="OOS Evaluation"):
            combo_str = future_to_combo[future]
            try:
                res = future.result()
                if res:
                    res["Pool_Sources"] = " | ".join(candidates[combo_str])
                    is_row = df[df["models"] == combo_str].iloc[0]
                    res["IS_Ann_Excess"] = is_row["Ann_Excess"]
                    res["IS_Calmar"] = is_row["Calmar"]
                    oos_results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"OOS 评估组合 {combo_str} 失败: {e}")
            
    # 6. 生成报告
    if not oos_results:
        print("没有可用的 OOS 结果。")
        return
        
    oos_df = pd.DataFrame(oos_results)
    oos_df = oos_df.sort_values("Ann_Excess", ascending=False)
    
    out_csv = os.path.join(metadata_dir, f"{prefix}oos_multi_analysis_{anchor_date}.csv")
    oos_df.to_csv(out_csv, index=False)
    
    # 7. 生成 OOS 可视化散点图
    generate_oos_visualizations(oos_df, metadata_dir, anchor_date, prefix=prefix)

    print("\n🏆 全维 OOS 验证成绩 (Top 15):")
    disp_cols = ["models", "Pool_Sources", "Ann_Excess", "Max_DD", "Calmar", "IS_Ann_Excess", "IS_Calmar"]
    fmt = {
        "Ann_Excess": "{:.2%}".format,
        "Max_DD": "{:.2%}".format,
        "Calmar": "{:.2f}".format,
        "IS_Ann_Excess": "{:.2%}".format,
        "IS_Calmar": "{:.2f}".format,
    }
    
    report_text = "="*60 + "\\n"
    report_text += f"OOS 综合验证分析报告 ({anchor_date})\\n"
    report_text += "="*60 + "\\n\\n"
    report_text += oos_df[disp_cols].head(15).to_string(formatters=fmt)
    
    print(oos_df[disp_cols].head(15).to_string(formatters=fmt))
    
    report_path = os.path.join(metadata_dir, f"{prefix}oos_report_{anchor_date}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"\\n✅ 完整分析报告已保存: {report_path}")
    print(f"📊 详细聚合明细已保存: {out_csv}")

if __name__ == "__main__":
    main()
