#!/usr/bin/env python
"""
迁移旧 output/brute_force/ 和 output/brute_force_fast/ 目录到新的
output/ensemble_runs/ 按运行归档结构。

用法:
  python quantpits/scripts/migrate_ensemble_outputs.py
  python quantpits/scripts/migrate_ensemble_outputs.py --dry-run   # 仅预览
  python quantpits/scripts/migrate_ensemble_outputs.py --source output/brute_force --source output/brute_force_fast
"""

import os
import re
import shutil
import argparse
import json
from collections import defaultdict
from pathlib import Path


# 已知的文件名模式 → (script_name, is/oos/root, new_filename)
# 带日期后缀的文件通过正则匹配
PATTERNS = [
    # IS 阶段产出
    (r"^brute_force_results_(\d{4}-\d{2}-\d{2})\.csv$",
     "brute_force", "is", "results.csv"),
    (r"^brute_force_fast_results_(\d{4}-\d{2}-\d{2})\.csv$",
     "brute_force_fast", "is", "results.csv"),
    (r"^minentropy_results_(\d{4}-\d{2}-\d{2})\.csv$",
     "minentropy", "is", "results.csv"),
    (r"^correlation_matrix_(\d{4}-\d{2}-\d{2})\.csv$",
     None, "is", "correlation_matrix.csv"),  # script_name from metadata/context
    (r"^(minentropy_)?correlation_matrix_(\d{4}-\d{2}-\d{2})\.csv$",
     None, "is", "correlation_matrix.csv"),  # with optional prefix
    (r"^(minentropy_)?analysis_report_(\d{4}-\d{2}-\d{2})\.txt$",
     None, "is", "analysis_report.txt"),
    (r"^analysis_report_fast_(\d{4}-\d{2}-\d{2})\.txt$",
     "brute_force_fast", "is", "analysis_report.txt"),
    (r"^(minentropy_)?model_attribution_(\d{4}-\d{2}-\d{2})\.csv$",
     None, "is", "model_attribution.csv"),
    (r"^(minentropy_)?model_attribution_(\d{4}-\d{2}-\d{2})\.png$",
     None, "is", "model_attribution.png"),
    (r"^(minentropy_)?risk_return_scatter_(\d{4}-\d{2}-\d{2})\.png$",
     None, "is", "risk_return_scatter.png"),
    (r"^(minentropy_)?cluster_dendrogram_(\d{4}-\d{2}-\d{2})\.png$",
     None, "is", "cluster_dendrogram.png"),
    # OOS 阶段产出
    (r"^(minentropy_)?oos_multi_analysis_(\d{4}-\d{2}-\d{2})\.csv$",
     None, "oos", "oos_multi_analysis.csv"),
    (r"^(minentropy_)?oos_report_(\d{4}-\d{2}-\d{2})\.txt$",
     None, "oos", "oos_report.txt"),
    (r"^(minentropy_)?oos_risk_return_(\d{4}-\d{2}-\d{2})\.png$",
     None, "oos", "oos_risk_return.png"),
    (r"^(minentropy_)?oos_validation_(\d{4}-\d{2}-\d{2})\.csv$",
     None, "oos", "oos_validation.csv"),
    # 根目录产出
    (r"^run_metadata_(\d{4}-\d{2}-\d{2})\.json$",
     None, "root", "run_metadata.json"),
    (r"^minentropy_metadata_(\d{4}-\d{2}-\d{2})\.json$",
     "minentropy", "root", "run_metadata.json"),
    # 优化权重等
    (r"^optimization_weights_(\d{4}-\d{2}-\d{2})\.csv$",
     None, "is", "optimization_weights.csv"),
    (r"^optimization_equity_(\d{4}-\d{2}-\d{2})\.png$",
     None, "is", "optimization_equity.png"),
]


def _extract_date(filename):
    """从文件名中提取日期字符串 YYYY-MM-DD"""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else None


def _detect_script_from_dir(source_dir):
    """根据目录名推断 script_name"""
    basename = os.path.basename(source_dir.rstrip("/"))
    if "brute_force_fast" in basename:
        return "brute_force_fast"
    elif "brute_force" in basename:
        return "brute_force"
    return None


def _detect_script_from_metadata(source_dir, date_str):
    """尝试从 metadata JSON 读取 script_used"""
    for pattern in [
        f"run_metadata_{date_str}.json",
        f"minentropy_metadata_{date_str}.json",
    ]:
        path = os.path.join(source_dir, pattern)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    meta = json.load(f)
                raw = meta.get("script_used", "")
                # 标准化
                mapping = {
                    "brute_force_ensemble": "brute_force",
                    "brute_force_fast": "brute_force_fast",
                    "minentropy": "minentropy",
                }
                return mapping.get(raw, raw)
            except Exception:
                pass
    return None


def plan_migration(source_dir, target_base):
    """
    扫描 source_dir 中的文件，规划迁移方案。

    Returns:
        list of (src_path, dst_path, date_str, script_name)
    """
    if not os.path.isdir(source_dir):
        print(f"源目录不存在: {source_dir}")
        return []

    default_script = _detect_script_from_dir(source_dir)
    files = sorted(os.listdir(source_dir))
    plan = []

    # 先按日期分组以确定每个日期的 script_name
    date_to_script = {}
    for filename in files:
        date_str = _extract_date(filename)
        if date_str and date_str not in date_to_script:
            script = _detect_script_from_metadata(source_dir, date_str)
            if script:
                date_to_script[date_str] = script

    for filename in files:
        filepath = os.path.join(source_dir, filename)
        if not os.path.isfile(filepath):
            continue

        date_str = _extract_date(filename)
        if not date_str:
            continue  # 不含日期的文件跳过

        matched = False
        for pattern, pat_script, stage, new_name in PATTERNS:
            m = re.match(pattern, filename)
            if m:
                # 确定 script_name
                script = pat_script or date_to_script.get(date_str) or default_script or "unknown"

                # 处理 minentropy 前缀: 如果文件名有 minentropy_ 前缀，强制 script=minentropy
                if filename.startswith("minentropy_"):
                    script = "minentropy"

                run_dir = os.path.join(target_base, f"{script}_{date_str}")
                if stage == "root":
                    dst = os.path.join(run_dir, new_name)
                elif stage == "is":
                    dst = os.path.join(run_dir, "is", new_name)
                else:
                    dst = os.path.join(run_dir, "oos", new_name)

                plan.append((filepath, dst, date_str, script))
                matched = True
                break

        if not matched:
            # 未匹配的带日期文件，放到对应 run 的 is/ 下保留原名
            script = date_to_script.get(date_str) or default_script or "unknown"
            if filename.startswith("minentropy_"):
                script = "minentropy"
            run_dir = os.path.join(target_base, f"{script}_{date_str}")
            dst = os.path.join(run_dir, "is", filename)
            plan.append((filepath, dst, date_str, script))

    return plan


def execute_migration(plan, dry_run=False):
    """执行迁移计划"""
    if not plan:
        print("无文件需要迁移。")
        return

    # 按 run_dir 分组统计
    runs = defaultdict(list)
    for src, dst, date_str, script in plan:
        run_dir = os.path.dirname(dst)
        if run_dir.endswith("/is") or run_dir.endswith("/oos"):
            run_dir = os.path.dirname(run_dir)
        runs[run_dir].append((src, dst))

    print(f"\n📦 迁移计划: {len(plan)} 个文件 → {len(runs)} 个运行目录\n")

    for run_dir in sorted(runs.keys()):
        items = runs[run_dir]
        print(f"  📁 {os.path.basename(run_dir)}/")
        for src, dst in items:
            rel_dst = os.path.relpath(dst, run_dir)
            print(f"     {'→' if not dry_run else '~'} {rel_dst}  ← {os.path.basename(src)}")

    if dry_run:
        print(f"\n⚠️ DRY RUN 模式 — 以上操作未实际执行。移除 --dry-run 执行迁移。")
        return

    # 实际执行
    print(f"\n🔄 执行迁移...")
    moved = 0
    for src, dst, _, _ in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"  ⚠️ 目标已存在, 跳过: {dst}")
            continue
        shutil.copy2(src, dst)
        moved += 1

    print(f"\n✅ 迁移完成! 共复制 {moved}/{len(plan)} 个文件。")
    print(f"原始目录未删除，确认无误后可手动删除。")


def main():
    parser = argparse.ArgumentParser(
        description="迁移旧版 ensemble 输出到新的 per-run 目录结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", type=str, nargs="+",
        default=["output/brute_force", "output/brute_force_fast"],
        help="源目录列表 (默认: output/brute_force output/brute_force_fast)",
    )
    parser.add_argument(
        "--target", type=str, default="output/ensemble_runs",
        help="目标根目录 (默认: output/ensemble_runs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览，不实际执行",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Ensemble Output Migration Tool")
    print("=" * 60)

    all_plan = []
    for source in args.source:
        if os.path.isdir(source):
            print(f"\n📂 扫描源目录: {source}")
            plan = plan_migration(source, args.target)
            all_plan.extend(plan)
        else:
            print(f"  ⏭ 跳过 (不存在): {source}")

    execute_migration(all_plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
