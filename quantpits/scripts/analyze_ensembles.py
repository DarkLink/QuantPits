#!/usr/bin/env python
"""
Analyze Ensembles - 多维度 OOS (Out-Of-Sample) 候选池验证分析

用法:
  python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_2026-04-03/run_metadata.json

此脚本将读取运行生成的 metadata 与 IS 回测结果，自动恢复数据集划分，
并在 OOS 数据上对多个不同"派系"的优秀候选组合（高收益、高稳健、低相关等）进行准确的回测验证。

产出物按 RunContext 分层:
  - IS 可视化/报告 → ctx.is_dir (is/)
  - OOS 回测/报告 → ctx.oos_dir (oos/)
  - 汇总摘要 → ctx.run_dir (summary.md)
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
from quantpits.utils.run_context import RunContext
from qlib.backtest.exchange import Exchange

def run_single_backtest_oos(
    combo_models, norm_df, top_k, drop_n, benchmark, freq,
    trade_exchange, bt_start, bt_end, st_config=None, bt_config=None
):
    """单独运行一次标准回测，用于 OOS 精确验证 (委托给 search_utils)"""
    from quantpits.utils.search_utils import run_single_backtest
    return run_single_backtest(
        combo_models, norm_df, top_k, drop_n, benchmark, freq,
        trade_exchange, bt_start, bt_end, st_config, bt_config
    )


# ==========================================
# IS Results Discovery
# ==========================================

def _find_is_results_csv(ctx, meta):
    """
    在 ctx.is_dir 中查找 IS 回测结果 CSV 文件。

    支持新结构（results.csv）和旧结构（带日期后缀的文件名）。
    """
    is_dir = ctx.is_dir
    anchor_date = meta["anchor_date"]
    script_used = meta.get("script_used", "brute_force_ensemble")

    # 新结构优先: results.csv
    candidates = [
        os.path.join(is_dir, "results.csv"),
    ]

    # 旧结构兼容（LegacyRunContext 时 is_dir == metadata_dir）
    if "fast" in script_used:
        candidates.append(os.path.join(is_dir, f"brute_force_fast_results_{anchor_date}.csv"))
    elif "minentropy" in script_used:
        candidates.append(os.path.join(is_dir, f"minentropy_results_{anchor_date}.csv"))
    else:
        candidates.append(os.path.join(is_dir, f"brute_force_results_{anchor_date}.csv"))

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def _find_correlation_csv(ctx, meta):
    """在 ctx.is_dir 中查找相关性矩阵 CSV。"""
    is_dir = ctx.is_dir
    anchor_date = meta["anchor_date"]

    candidates = [
        os.path.join(is_dir, "correlation_matrix.csv"),
        os.path.join(is_dir, f"correlation_matrix_{anchor_date}.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def generate_is_visualizations_and_report(df, is_dir, anchor_date, top_n=50, corr_file=None):
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
    attribution = pd.DataFrame()
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

        attr_path = os.path.join(is_dir, "model_attribution.csv")
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
        attr_fig_path = os.path.join(is_dir, "model_attribution.png")
        plt.savefig(attr_fig_path, dpi=150)
        plt.close()
        print(f"归因图已保存: {attr_fig_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"模型归因分析失败: {e}")

    # 读取相关性矩阵并计算 avg_corr
    try:
        if corr_file and os.path.exists(corr_file):
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
        
        report_path = os.path.join(is_dir, "analysis_report.txt")
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
                scatter_path = os.path.join(is_dir, "risk_return_scatter.png")
                plt.savefig(scatter_path, dpi=150)
                plt.close()
                print(f"IS 散点图已保存: {scatter_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"IS 散点图生成跳过/失败: {e}")

    return df  # Return df in case it has updated avg_corr

def generate_dendrogram(is_dir, corr_file=None):
    print("\n🌳 === 生成聚类树状图 ===")
    try:
        if corr_file and os.path.exists(corr_file):
            corr_matrix = pd.read_csv(corr_file, index_col=0)
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

            dendro_path = os.path.join(is_dir, "cluster_dendrogram.png")
            plt.savefig(dendro_path, dpi=150)
            plt.close()
            print(f"聚类图已保存: {dendro_path}")
        else:
            print("找不到相关性矩阵文件，跳过聚类图。")
    except Exception as e:
        print(f"聚类分析失败: {e}")

def generate_oos_visualizations(oos_df, oos_dir):
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
        
        scatter_path = os.path.join(oos_dir, "oos_risk_return.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"OOS 散点图已保存: {scatter_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"OOS 散点图绘制失败: {e}")


def generate_summary_md(ctx, meta, df, oos_df=None, top_n=10):
    """生成人类可读的 summary.md 一页纸总结。"""
    lines = []
    anchor_date = meta["anchor_date"]
    script_used = meta.get("script_used", "unknown")

    lines.append(f"# Ensemble Search Report — {anchor_date}\n")

    # Run Info
    lines.append("## Run Info\n")
    lines.append(f"- **Script**: `{script_used}`")
    lines.append(f"- **IS Period**: {meta.get('is_start_date', '?')} ~ {meta.get('is_end_date', '?')}")
    if meta.get("oos_start_date") and str(meta.get("oos_start_date")).lower() != "none":
        lines.append(f"- **OOS Period**: {meta['oos_start_date']} ~ {meta['oos_end_date']}")
    else:
        lines.append("- **OOS Period**: _(not configured)_")
    lines.append(f"- **Anchor Date**: {anchor_date}")
    lines.append(f"- **Total Combinations Evaluated**: {len(df)}")
    lines.append("")

    # IS Highlights
    lines.append("## IS Highlights\n")

    if df.empty or "Calmar" not in df.columns:
        lines.append("_(No IS results available)_\n")
        df_sorted = df
        top_is = df.head(0)
    else:
        df_sorted = df.sort_values("Calmar", ascending=False)
        top_is = df_sorted.head(top_n)

    lines.append("| Rank | Models | Ann_Excess | Max_DD | Calmar |")
    lines.append("|------|--------|-----------|--------|--------|")
    for rank, (_, row) in enumerate(top_is.iterrows(), 1):
        models_short = row["models"]
        if len(models_short) > 60:
            models_short = models_short[:57] + "..."
        ann_excess = f"{row['Ann_Excess']:.2%}" if pd.notna(row.get("Ann_Excess")) else "?"
        max_dd = f"{row['Max_DD']:.2%}" if pd.notna(row.get("Max_DD")) else "?"
        calmar = f"{row['Calmar']:.2f}" if pd.notna(row.get("Calmar")) else "?"
        lines.append(f"| {rank} | {models_short} | {ann_excess} | {max_dd} | {calmar} |")
    lines.append("")

    # OOS Validation
    if oos_df is not None and not oos_df.empty:
        lines.append("## OOS Validation\n")
        oos_sorted = oos_df.sort_values("Ann_Excess", ascending=False).head(top_n)
        lines.append("| Rank | Models | OOS_Excess | OOS_MaxDD | IS→OOS | Pool |")
        lines.append("|------|--------|-----------|----------|--------|------|")
        for rank, (_, row) in enumerate(oos_sorted.iterrows(), 1):
            models_short = row["models"]
            if len(models_short) > 50:
                models_short = models_short[:47] + "..."
            oos_excess = f"{row['Ann_Excess']:.2%}" if pd.notna(row.get("Ann_Excess")) else "?"
            oos_dd = f"{row['Max_DD']:.2%}" if pd.notna(row.get("Max_DD")) else "?"
            is_excess = f"{row.get('IS_Ann_Excess', '?'):.2%}" if pd.notna(row.get("IS_Ann_Excess")) else "?"
            arrow = f"{is_excess} → {oos_excess}"
            pool = str(row.get("Pool_Sources", ""))[:30]
            lines.append(f"| {rank} | {models_short} | {oos_excess} | {oos_dd} | {arrow} | {pool} |")
        lines.append("")

    # Files in This Run
    lines.append("## Files in This Run\n")
    for subdir_name, subdir_path in [("is", ctx.is_dir), ("oos", ctx.oos_dir), ("root", ctx.run_dir)]:
        if os.path.isdir(subdir_path):
            files = sorted(os.listdir(subdir_path))
            files = [f for f in files if os.path.isfile(os.path.join(subdir_path, f))]
            if files:
                for fname in files:
                    if subdir_name == "root":
                        lines.append(f"- `{fname}`")
                    else:
                        lines.append(f"- `{subdir_name}/{fname}`")
    lines.append("")

    summary_path = ctx.run_path("summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📋 Summary 已保存: {summary_path}")

    # Generate HTML summary to avoid truncation
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif; margin: 20px; line-height: 1.6; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f6f8fa; }",
        "h1, h2 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Ensemble Search Report — {anchor_date}</h1>",
        "<h2>Run Info</h2>",
        "<ul>",
        f"<li><b>Script</b>: <code>{script_used}</code></li>",
        f"<li><b>IS Period</b>: {meta.get('is_start_date', '?')} ~ {meta.get('is_end_date', '?')}</li>"
    ]
    if meta.get("oos_start_date") and str(meta.get("oos_start_date")).lower() != "none":
        html_lines.append(f"<li><b>OOS Period</b>: {meta['oos_start_date']} ~ {meta['oos_end_date']}</li>")
    else:
        html_lines.append("<li><b>OOS Period</b>: <i>(not configured)</i></li>")
    html_lines.append(f"<li><b>Anchor Date</b>: {anchor_date}</li>")
    html_lines.append(f"<li><b>Total Combinations Evaluated</b>: {len(df)}</li>")
    html_lines.append("</ul>")

    html_lines.append("<h2>IS Highlights</h2>")
    if df.empty or "Calmar" not in df.columns:
        html_lines.append("<p><i>(No IS results available)</i></p>")
    else:
        html_lines.append("<table>")
        html_lines.append("<tr><th>Rank</th><th>Models</th><th>Ann_Excess</th><th>Max_DD</th><th>Calmar</th></tr>")
        for rank, (_, row) in enumerate(top_is.iterrows(), 1):
            models_full = row['models']
            ann_excess = f"{row['Ann_Excess']:.2%}" if pd.notna(row.get("Ann_Excess")) else "?"
            max_dd = f"{row['Max_DD']:.2%}" if pd.notna(row.get("Max_DD")) else "?"
            calmar = f"{row['Calmar']:.2f}" if pd.notna(row.get("Calmar")) else "?"
            html_lines.append(f"<tr><td>{rank}</td><td>{models_full}</td><td>{ann_excess}</td><td>{max_dd}</td><td>{calmar}</td></tr>")
        html_lines.append("</table>")

    if oos_df is not None and not oos_df.empty:
        html_lines.append("<h2>OOS Validation</h2>")
        html_lines.append("<table>")
        html_lines.append("<tr><th>Rank</th><th>Models</th><th>OOS_Excess</th><th>OOS_MaxDD</th><th>IS&rarr;OOS</th><th>Pool</th></tr>")
        for rank, (_, row) in enumerate(oos_sorted.iterrows(), 1):
            models_full = row['models']
            oos_excess = f"{row['Ann_Excess']:.2%}" if pd.notna(row.get("Ann_Excess")) else "?"
            oos_dd = f"{row['Max_DD']:.2%}" if pd.notna(row.get("Max_DD")) else "?"
            is_excess = f"{row.get('IS_Ann_Excess', '?'):.2%}" if pd.notna(row.get("IS_Ann_Excess")) else "?"
            arrow = f"{is_excess} &rarr; {oos_excess}"
            pool = str(row.get("Pool_Sources", ""))
            html_lines.append(f"<tr><td>{rank}</td><td>{models_full}</td><td>{oos_excess}</td><td>{oos_dd}</td><td>{arrow}</td><td>{pool}</td></tr>")
        html_lines.append("</table>")

    html_lines.append("<h2>Files in This Run</h2><ul>")
    for subdir_name, subdir_path in [("is", ctx.is_dir), ("oos", ctx.oos_dir), ("root", ctx.run_dir)]:
        if os.path.isdir(subdir_path):
            files = sorted(os.listdir(subdir_path))
            files = [f for f in files if os.path.isfile(os.path.join(subdir_path, f))]
            if files:
                for fname in files:
                    if subdir_name == "root":
                        html_lines.append(f"<li><code>{fname}</code></li>")
                    else:
                        html_lines.append(f"<li><code>{subdir_name}/{fname}</code></li>")
    html_lines.append("</ul></body></html>")

    html_path = ctx.run_path("summary.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    print(f"📋 Summary HTML 已保存: {html_path}")


# ==========================================
# OOS HTML REPORT
# ==========================================

POOL_COLORS = {
    "Yield_Top":       ("#f59e0b", "#78350f"),   # amber
    "Robust_Top":      ("#10b981", "#064e3b"),   # emerald
    "Defensive_Top":   ("#3b82f6", "#1e3a5f"),   # blue
    "Golden_Diversity":("#a855f7", "#4c1d95"),   # purple
    "MVP_Base":        ("#6b7280", "#1f2937"),   # gray
}


def _pool_badge_html(pool_name: str) -> str:
    """Return a styled <span> badge for a pool name."""
    fg, bg = POOL_COLORS.get(pool_name, ("#e5e7eb", "#374151"))
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
        f'border-radius:4px;padding:2px 7px;font-size:11px;font-weight:600;'
        f'white-space:nowrap;">{pool_name}</span>'
    )


def _pool_sources_html(pool_sources_str: str) -> str:
    """Convert e.g. 'Yield_Top | Robust_Top' to badge HTML."""
    pools = [p.strip() for p in str(pool_sources_str).split("|") if p.strip()]
    return " ".join(_pool_badge_html(p) for p in pools)


def _is_oos_arrow_html(is_val: float, oos_val: float) -> str:
    """IS → OOS arrow with colour coding."""
    is_str = f"{is_val:.1%}" if pd.notna(is_val) else "?"
    oos_str = f"{oos_val:.1%}" if pd.notna(oos_val) else "?"
    if pd.notna(oos_val) and oos_val >= 0:
        color = "#10b981"
    else:
        color = "#ef4444"
    return f'<span style="color:#9ca3af">{is_str}</span> → <span style="color:{color};font-weight:600">{oos_str}</span>'


def _oos_table_rows(sub_df: pd.DataFrame, show_rank: bool = True) -> str:
    """Generate <tr> rows for OOS result table."""
    rows = []
    for rank, (_, row) in enumerate(sub_df.iterrows(), 1):
        models = row.get("models", "")
        n = int(row.get("n_models", 1))
        ann_excess = row.get("Ann_Excess", float("nan"))
        max_dd = row.get("Max_DD", float("nan"))
        calmar = row.get("Calmar", float("nan"))
        is_ann = row.get("IS_Ann_Excess", float("nan"))
        is_calmar = row.get("IS_Calmar", float("nan"))
        pool_sources = row.get("Pool_Sources", "")
        pool_count = int(row.get("_pool_count", 1))

        calmar_str = f"{calmar:.2f}" if pd.notna(calmar) else "?"
        is_calmar_str = f"{is_calmar:.2f}" if pd.notna(is_calmar) else "?"
        max_dd_str = f"{max_dd:.1%}" if pd.notna(max_dd) else "?"

        oos_color = "#10b981" if pd.notna(ann_excess) and ann_excess >= 0 else "#ef4444"

        rank_td = f'<td style="color:#6b7280;font-size:12px">{rank}</td>' if show_rank else ""

        pool_count_badge = ""
        if pool_count >= 2:
            pc_color = "#f59e0b" if pool_count >= 3 else "#60a5fa"
            pool_count_badge = f' <span style="background:{pc_color};color:#000;border-radius:10px;padding:1px 6px;font-size:10px;font-weight:700">{pool_count}×</span>'

        rows.append(
            f"<tr>"
            f"{rank_td}"
            f'<td style="font-size:11px;max-width:320px;word-break:break-all">'
            f"{models}{pool_count_badge}</td>"
            f'<td style="text-align:center;color:#9ca3af">{n}</td>'
            f'<td style="text-align:right">{_is_oos_arrow_html(is_ann, ann_excess)}</td>'
            f'<td style="text-align:right;color:#9ca3af">{max_dd_str}</td>'
            f'<td style="text-align:right;color:#e5e7eb">{calmar_str}</td>'
            f'<td style="text-align:right;color:#9ca3af;font-size:11px">{is_calmar_str}</td>'
            f'<td style="font-size:11px">{_pool_sources_html(pool_sources)}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


TABLE_HEADER = (
    "<thead><tr>"
    "<th>#</th>"
    "<th>Models</th>"
    "<th style='text-align:center'>N</th>"
    "<th style='text-align:right'>IS→OOS Excess</th>"
    "<th style='text-align:right'>Max DD</th>"
    "<th style='text-align:right'>OOS Calmar</th>"
    "<th style='text-align:right'>IS Calmar</th>"
    "<th>Pools</th>"
    "</tr></thead>"
)


def generate_oos_html_report(oos_df: pd.DataFrame, oos_dir: str, anchor_date: str) -> str:
    """
    生成富交互式 OOS 分析 HTML 报告。

    核心特性:
    - 跨维度共识面板 (Consensus Panel): 同时上榜 2+ 个派系的组合优先展示
    - 各派系独立 Tab: 快速按维度浏览各派系的 Top 候选
    - 彩色 Pool 徽章 + IS→OOS 收益双色箭头

    返回保存的 HTML 文件路径。
    """
    print("\n🌐 === 生成 OOS 富交互 HTML 报告 ===")

    df = oos_df.copy()

    # ---- 预处理 ----
    for col in ["Ann_Excess", "Max_DD", "Calmar", "IS_Ann_Excess", "IS_Calmar", "n_models"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算每个组合出现的派系数量
    df["_pools_list"] = df["Pool_Sources"].apply(
        lambda x: [p.strip() for p in str(x).split("|") if p.strip()]
    )
    df["_pool_count"] = df["_pools_list"].apply(len)

    # 所有出现的派系（按出现频次排序）
    all_pools = sorted(
        set(p for pools in df["_pools_list"] for p in pools),
        key=lambda p: list(POOL_COLORS.keys()).index(p) if p in POOL_COLORS else 99
    )

    total = len(df)
    oos_positive = int((df["Ann_Excess"] > 0).sum())
    multi_dim = int((df["_pool_count"] >= 2).sum())

    # ---- 共识排序: pool_count DESC, Calmar DESC ----
    consensus_df = df.sort_values(["_pool_count", "Calmar"], ascending=[False, False])
    multi_df = consensus_df[consensus_df["_pool_count"] >= 2]

    # ---- 构建各派系 Tab 内容 ----
    tabs_nav = []
    tabs_content = []
    for i, pool in enumerate(all_pools):
        fg, bg = POOL_COLORS.get(pool, ("#e5e7eb", "#374151"))
        pool_subset = df[df["_pools_list"].apply(lambda pl: pool in pl)].sort_values("Calmar", ascending=False)
        tab_id = f"tab_{pool}"
        pane_id = f"pane_{pool}"
        count = len(pool_subset)
        pos_count = int((pool_subset["Ann_Excess"] > 0).sum())

        tabs_nav.append(
            f'<button class="tab-btn" id="{tab_id}" onclick="showTab(\'{pane_id}\')" '
            f'style="border-left:3px solid {fg}">'
            f'<span style="color:{fg};font-weight:700">{pool}</span>'
            f'<span style="color:#6b7280;font-size:11px;margin-left:6px">{count} combos / {pos_count} OOS+</span>'
            f'</button>'
        )

        rows_html = _oos_table_rows(pool_subset.head(30), show_rank=True)
        tabs_content.append(
            f'<div class="tab-pane" id="{pane_id}" style="display:none">'
            f'<h3 style="color:{fg};margin:0 0 12px">{pool} &mdash; Top {min(30, count)} by OOS Calmar</h3>'
            f'<table>{TABLE_HEADER}<tbody>{rows_html}</tbody></table>'
            f'</div>'
        )

    # ---- Pool 覆盖统计 ----
    pool_stats_rows = []
    for pool in all_pools:
        fg, _ = POOL_COLORS.get(pool, ("#e5e7eb", "#374151"))
        sub = df[df["_pools_list"].apply(lambda pl: pool in pl)]
        count = len(sub)
        pos = int((sub["Ann_Excess"] > 0).sum())
        pos_rate = pos / count if count else 0
        median_calmar = sub["Calmar"].median()
        pool_stats_rows.append(
            f"<tr>"
            f'<td>{_pool_badge_html(pool)}</td>'
            f'<td style="text-align:center">{count}</td>'
            f'<td style="text-align:center;color:{"#10b981" if pos_rate >= 0.5 else "#ef4444"}">{pos}/{count} ({pos_rate:.0%})</td>'
            f'<td style="text-align:center">{median_calmar:.2f}</td>'
            f"</tr>"
        )
    pool_stats_html = "\n".join(pool_stats_rows)

    # ---- 构建最终 HTML ----
    consensus_rows_html = _oos_table_rows(multi_df, show_rank=True)
    all_rows_html = _oos_table_rows(consensus_df, show_rank=True)  # full table sorted consensus-first

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OOS Multi-Pool Analysis &mdash; {anchor_date}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 0 0 60px;
}}
.hero {{
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 60%);
    border-bottom: 1px solid #334155;
    padding: 28px 32px 20px;
}}
.hero h1 {{ font-size: 22px; font-weight: 700; color: #f1f5f9; }}
.hero p {{ color: #94a3b8; font-size: 13px; margin-top: 4px; }}
.metrics-bar {{
    display: flex; gap: 24px; margin-top: 16px; flex-wrap: wrap;
}}
.metric-card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 18px;
    min-width: 120px;
}}
.metric-card .val {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
.metric-card .lbl {{ font-size: 11px; color: #64748b; margin-top: 2px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px 24px 0; }}
section {{ margin-bottom: 36px; }}
section > h2 {{
    font-size: 16px; font-weight: 700; color: #cbd5e1;
    border-left: 4px solid #6366f1;
    padding-left: 12px;
    margin-bottom: 14px;
}}
.panel {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    overflow: hidden;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
thead tr {{ background: #0f172a; }}
th {{
    padding: 9px 12px;
    color: #64748b;
    font-weight: 600;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .5px;
    border-bottom: 1px solid #334155;
}}
td {{
    padding: 8px 12px;
    border-bottom: 1px solid #1e293b;
    vertical-align: middle;
}}
tr:hover td {{ background: #1e3a5f22; }}
.pools-sidebar {{
    display: flex;
    gap: 20px;
    align-items: flex-start;
}}
.tab-list {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 200px;
    flex-shrink: 0;
}}
.tab-btn {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 10px 14px;
    cursor: pointer;
    text-align: left;
    color: #e2e8f0;
    font-size: 12px;
    transition: background .15s;
    display: flex;
    flex-direction: column;
    gap: 2px;
}}
.tab-btn:hover, .tab-btn.active {{ background: #334155; }}
.tab-panes {{ flex: 1; min-width: 0; }}
.tab-pane .panel {{ border-radius: 8px; }}
.badge-multi {{
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed, #f59e0b);
    color: #fff;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
    margin-left: 4px;
}}
</style>
</head>
<body>
<div class="hero">
  <h1>🔭 OOS Multi-Pool Analysis &mdash; {anchor_date}</h1>
  <p>Out-of-Sample validation across {len(all_pools)} selection dimensions. Combos sorted by cross-dimensional consensus first.</p>
  <div class="metrics-bar">
    <div class="metric-card"><div class="val">{total}</div><div class="lbl">Total OOS Combos</div></div>
    <div class="metric-card"><div class="val" style="color:#10b981">{oos_positive}</div><div class="lbl">OOS Positive Excess</div></div>
    <div class="metric-card"><div class="val" style="color:#f59e0b">{multi_dim}</div><div class="lbl">Cross-Dim Consensus (2+)</div></div>
    <div class="metric-card"><div class="val">{len(all_pools)}</div><div class="lbl">Selection Pools</div></div>
  </div>
</div>
<div class="container">

<!-- === POOL STATS === -->
<section>
<h2>Pool Coverage Overview</h2>
<div class="panel">
<table>
<thead><tr><th>Pool</th><th style='text-align:center'>Combos</th><th style='text-align:center'>OOS Positive</th><th style='text-align:center'>Median Calmar</th></tr></thead>
<tbody>{pool_stats_html}</tbody>
</table>
</div>
</section>

<!-- === CONSENSUS PANEL === -->
<section>
<h2>🏆 Cross-Dimension Consensus (appearing in 2+ pools)</h2>
<div class="panel">
{'<table>' + TABLE_HEADER + '<tbody>' + consensus_rows_html + '</tbody></table>' if not multi_df.empty else '<p style="padding:20px;color:#64748b">No combos appear in 2+ pools.</p>'}
</div>
</section>

<!-- === PER-POOL TABS === -->
<section>
<h2>Per-Pool Breakdown</h2>
<div class="pools-sidebar">
  <div class="tab-list" id="tabList">
    {''.join(tabs_nav)}
  </div>
  <div class="tab-panes" id="tabPanes">
    {''.join(f'<div class="panel" style="padding:16px">{c}</div>' for c in tabs_content)}
  </div>
</div>
</section>

<!-- === FULL TABLE === -->
<section>
<h2>All OOS Results (Consensus-First Ordering)</h2>
<div class="panel">
<table>{TABLE_HEADER}<tbody>{all_rows_html}</tbody></table>
</div>
</section>

</div>
<script>
function showTab(paneId) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  var pane = document.getElementById(paneId);
  if (pane) pane.style.display = 'block';
  // find matching button
  document.querySelectorAll('.tab-btn').forEach(b => {{
    if (b.getAttribute('onclick') && b.getAttribute('onclick').includes(paneId)) {{
      b.classList.add('active');
    }}
  }});
}}
// Show first tab by default
var firstPane = document.querySelector('.tab-pane');
if (firstPane) firstPane.style.display = 'block';
var firstBtn = document.querySelector('.tab-btn');
if (firstBtn) firstBtn.classList.add('active');
</script>
</body>
</html>
"""

    out_path = os.path.join(oos_dir, "oos_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 OOS HTML 报告已保存: {out_path}")
    return out_path


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
    parser.add_argument("--max-workers", type=int, default=4, help="OOS回测的并发线程数 (默认: 4)")
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

    # 构建 RunContext (自动检测新/旧目录结构)
    ctx = RunContext.from_metadata(args.metadata)
    ctx.ensure_dirs()

    print("=" * 60)
    print("🚀 Analyze Ensembles - IS 多维提取与 OOS 验证")
    print(f"数据源: {script_used} (发布于 {anchor_date})")
    print(f"OOS 验证周期间隔: {oos_start_date} ~ {oos_end_date}")
    print(f"输出目录: {ctx.run_dir}")
    print("=" * 60)

    has_oos = bool(oos_start_date and oos_end_date and str(oos_start_date).lower() != "none" and str(oos_end_date).lower() != "none")
    if not has_oos:
        print("⚠️ 警告：元数据中未找到有效的 OOS 数据周期！本次运行仅执行 IS (In-Sample) 全量分析，将跳过 OOS 验证环节。")

    # 2. 读取 IS Results CSV
    results_file = _find_is_results_csv(ctx, meta)
    if not results_file:
        print(f"❌ 找不到 IS 结果文件，已搜索: {ctx.is_dir}")
        sys.exit(1)
        
    df = pd.read_csv(results_file)
    print(f"\n✅ 成功加载 IS 结果文件: {os.path.basename(results_file)}，共 {len(df)} 组策略记录。")

    if args.training_mode:
        def match_mode(models_str):
            models = str(models_str).split(",")
            return all(m.endswith(f"@{args.training_mode}") for m in models)
        
        df = df[df["models"].apply(match_mode)].copy()
        print(f"🎯 应用 --training-mode '{args.training_mode}' 过滤后，剩余 {len(df)} 组策略记录参加评估。")
        if df.empty:
            print("❌ 过滤后无符合条件的记录，分析终止。")
            sys.exit(0)

    # 找到相关性矩阵 CSV
    corr_file = _find_correlation_csv(ctx, meta)

    # 2.5 生成 IS 全量结果可视化
    try:
        df = generate_is_visualizations_and_report(
            df, ctx.is_dir, anchor_date, top_n=args.top_n, corr_file=corr_file
        )
        generate_dendrogram(ctx.is_dir, corr_file=corr_file)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"IS 可视化图表生成跳过/失败: {e}")

    if not has_oos:
        generate_summary_md(ctx, meta, df, oos_df=None)
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
        generate_summary_md(ctx, meta, df, oos_df=None)
        return
        
    oos_df = pd.DataFrame(oos_results)
    oos_df = oos_df.sort_values("Ann_Excess", ascending=False)
    
    out_csv = os.path.join(ctx.oos_dir, "oos_multi_analysis.csv")
    oos_df.to_csv(out_csv, index=False)
    
    # 7. 生成 OOS 可视化散点图
    generate_oos_visualizations(oos_df, ctx.oos_dir)

    print("\n🏆 全维 OOS 验证成绩 (Top 15):")
    disp_cols = ["models", "Pool_Sources", "Ann_Excess", "Max_DD", "Calmar", "IS_Ann_Excess", "IS_Calmar"]
    fmt = {
        "Ann_Excess": "{:.2%}".format,
        "Max_DD": "{:.2%}".format,
        "Calmar": "{:.2f}".format,
        "IS_Ann_Excess": "{:.2%}".format,
        "IS_Calmar": "{:.2f}".format,
    }
    
    # 使用真正的换行符写入报告
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"OOS 综合验证分析报告 ({anchor_date})")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(oos_df[disp_cols].head(15).to_string(formatters=fmt))
    
    print(oos_df[disp_cols].head(15).to_string(formatters=fmt))
    
    report_path = os.path.join(ctx.oos_dir, "oos_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n✅ 完整分析报告已保存: {report_path}")
    print(f"📊 详细聚合明细已保存: {out_csv}")

    # 7b. 生成 OOS 富交互 HTML 报告
    try:
        generate_oos_html_report(oos_df, ctx.oos_dir, anchor_date)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"OOS HTML 报告生成失败 (非致命): {e}")

    # 8. 生成 summary.md
    generate_summary_md(ctx, meta, df, oos_df=oos_df)

if __name__ == "__main__":
    main()
