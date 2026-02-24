#!/usr/bin/env python3
"""
QuantPits 带日期文件归档工具

常态化归档工具，每周运行后清理历史文件：
- 自动扫描 output/ 及子目录中 *_YYYY-MM-DD* 和 YYYY-MM-DD-* 格式的文件
- 按逻辑文件名分组，保留最新 N 个日期版本，旧版移入 archive/
- 订单建议和交易明细归档到 data/order_history/
- 支持 --dry-run 模式预览

用法:
    python engine/scripts/archive_dated_files.py --dry-run          # 预览模式
    python engine/scripts/archive_dated_files.py                     # 实际归档
    python engine/scripts/archive_dated_files.py --keep 2            # 保留最近2个版本
    python engine/scripts/archive_dated_files.py --include-notebooks # 同时归档 legacy notebooks
    python engine/scripts/archive_dated_files.py --cleanup-legacy    # 清理测试/实验遗留文件
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import env

# ── 路径定义 ─────────────────────────────────────────────────────────────
ROOT_DIR = env.ROOT_DIR
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive")
ORDER_HISTORY_DIR = os.path.join(DATA_DIR, "order_history")

# ── 日期模式 ─────────────────────────────────────────────────────────────
# 匹配 _YYYY-MM-DD 或 _YYYY-MM-DD_HHMMSS（文件名中间或末尾）
DATE_PATTERN_SUFFIX = re.compile(r'_(\d{4}-\d{2}-\d{2})(?:_(\d{6}))?')
# 匹配 YYYY-MM-DD- 开头的文件名（如交易软件导出的 xlsx）
DATE_PATTERN_PREFIX = re.compile(r'^(\d{4}-\d{2}-\d{2})-')

# ── 交易数据文件模式（归档到 order_history） ─────────────────────────────
TRADE_DATA_PATTERNS = [
    r'^buy_suggestion_',
    r'^sell_suggestion_',
    r'^model_opinions_',
    r'^trade_detail_',
    r'^\d{4}-\d{2}-\d{2}-table\.xlsx$',
]

# ── Legacy notebooks（已重构到 scripts/）──────────────────────────────────
LEGACY_NOTEBOOKS = [
    "weekly_ensemble_predict.ipynb",
    "weekly_ensemble_predict_v2.ipynb",
    "weekly_ensemble_predict_v3.ipynb",
    "brute_force_data_check.ipynb",
    "weekly_order_gen.ipynb",
    "weekly_order_gen_ensemble.ipynb",
    "weekly_order_gen_v2.ipynb",
    "weekly_post_trade.ipynb",
]

# ── 测试/实验遗留文件 ────────────────────────────────────────────────────
LEGACY_ITEMS = {
    # 目录
    "output/Alpha158_full": "misc/Alpha158_full",
    "output/analysis": "misc/analysis",
    # 单文件
    "data/emp-table.xlsx": "misc/emp-table.xlsx",
    "scripts/compare_results.py": "misc/compare_results.py",
    "scripts/model_compare.py": "misc/model_compare.py",
    "scripts/ensemble_predict.py": "misc/ensemble_predict.py",
    # output/ 根目录的早期测试散落文件（已被子目录的正式输出取代）
    "output/brute_force_results_2026-02-06.csv": "misc/brute_force_results_2026-02-06.csv",
    "output/best_model_2026-02-06.txt": "misc/best_model_2026-02-06.txt",
    "output/ensemble_config_2026-02-06.json": "misc/ensemble_config_2026-02-06.json",
    "output/ensemble_performance_2026-02-06.png": "misc/ensemble_performance_2026-02-06.png",
    "output/leaderboard_2026-02-06.csv": "misc/leaderboard_2026-02-06.csv",
    "output/model_comparison_2026-02-06.json": "misc/model_comparison_2026-02-06.json",
}

# ── 不应被归档的文件（累计日志等）─────────────────────────────────────────
PROTECTED_PATTERNS = [
    r'_full\.csv$',       # trade_log_full.csv, holding_log_full.csv, etc.
    r'^run_state\.json$',
    r'^model_log\.csv$',
]


def extract_date_info(filename):
    """
    从文件名中提取日期和逻辑文件名（去掉日期部分的前缀）

    Returns:
        (logical_name, date_str, sort_key) or None
        - logical_name: 去掉日期后的文件名前缀（用来分组）
        - date_str: YYYY-MM-DD 格式的日期字符串
        - sort_key: 用于排序的字符串（日期+可选时间戳）
    """
    # 尝试前缀模式：YYYY-MM-DD-xxx
    m = DATE_PATTERN_PREFIX.match(filename)
    if m:
        date_str = m.group(1)
        # 逻辑名：日期后面的部分（如 "table.xlsx"）
        rest = filename[len(m.group(0)):]
        name, ext = os.path.splitext(rest)
        logical_name = f"*-{rest}"  # 通配前缀
        return logical_name, date_str, date_str

    # 尝试后缀模式：xxx_YYYY-MM-DD 或 xxx_YYYY-MM-DD_HHMMSS
    m = DATE_PATTERN_SUFFIX.search(filename)
    if m:
        date_str = m.group(1)
        timestamp = m.group(2) or ""
        # 逻辑名：日期之前的部分 + 日期之后的扩展名
        prefix = filename[:m.start()]
        suffix = filename[m.end():]
        logical_name = f"{prefix}*{suffix}"
        sort_key = f"{date_str}_{timestamp}" if timestamp else date_str
        return logical_name, date_str, sort_key

    return None


def is_protected(filename):
    """检查文件是否受保护（不应归档）"""
    for pattern in PROTECTED_PATTERNS:
        if re.search(pattern, filename):
            return True
    return False


def is_trade_data(filename):
    """检查文件是否属于交易数据"""
    for pattern in TRADE_DATA_PATTERNS:
        if re.match(pattern, filename):
            return True
    return False


def scan_dated_files(directory, relative_prefix=""):
    """
    扫描目录中的带日期文件，返回分组信息

    Returns:
        dict: {(relative_dir, logical_name): [(sort_key, filename, full_path), ...]}
    """
    groups = defaultdict(list)

    if not os.path.isdir(directory):
        return groups

    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)

        # 跳过目录（子目录由调用者递归处理）
        if os.path.isdir(full_path):
            continue

        # 跳过受保护的文件
        if is_protected(entry):
            continue

        info = extract_date_info(entry)
        if info:
            logical_name, date_str, sort_key = info
            rel_dir = relative_prefix
            groups[(rel_dir, logical_name)].append((sort_key, entry, full_path))

    return groups


def get_anchor_date(override=None):
    """
    获取锚点日期（最近一个交易日，通常是周五）

    优先级：
    1. CLI 传入的 --anchor-date
    2. latest_train_records.json 中的 anchor_date
    3. 如果都没有则报错

    Returns:
        str: YYYY-MM-DD 格式的锚点日期
    """
    if override:
        return override

    records_file = os.path.join(ROOT_DIR, "latest_train_records.json")
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            records = json.load(f)
        anchor = records.get('anchor_date', '')
        if anchor:
            return anchor

    raise ValueError(
        "无法确定锚点日期。请通过 --anchor-date YYYY-MM-DD 指定，"
        "或确保 latest_train_records.json 中包含 anchor_date 字段。"
    )


def plan_archive(anchor_date):
    """
    计算需要归档的文件：日期 < anchor_date 的文件归档，>= anchor_date 的保留

    Args:
        anchor_date: 锚点日期字符串 YYYY-MM-DD

    Returns:
        list of (source_path, dest_path, category)
    """
    moves = []

    # ── 1. 扫描 output/ 根目录 ──────────────────────────────────────────
    groups = scan_dated_files(OUTPUT_DIR, "output")

    # ── 2. 扫描 output/ 子目录 ──────────────────────────────────────────
    for subdir in ["predictions", "ensemble", "brute_force", "brute_force_fast", "ranking"]:
        sub_path = os.path.join(OUTPUT_DIR, subdir)
        sub_groups = scan_dated_files(sub_path, f"output/{subdir}")
        groups.update(sub_groups)

    # ── 3. 扫描 data/ 目录 ──────────────────────────────────────────────
    data_groups = scan_dated_files(DATA_DIR, "data")
    groups.update(data_groups)

    # ── 4. 处理每个文件：日期 < anchor_date 的归档 ─────────────────────
    for (rel_dir, logical_name), files in sorted(groups.items()):
        for sort_key, filename, full_path in files:
            # sort_key 格式为 YYYY-MM-DD 或 YYYY-MM-DD_HHMMSS
            file_date = sort_key[:10]  # 提取 YYYY-MM-DD 部分
            if file_date < anchor_date:
                # 根据文件类型决定归档目标
                if is_trade_data(filename):
                    dest = os.path.join(ORDER_HISTORY_DIR, filename)
                    category = "trade_data"
                else:
                    dest = os.path.join(ARCHIVE_DIR, rel_dir, filename)
                    category = "output"
                moves.append((full_path, dest, category))

    return moves


def archive_legacy_notebooks(dry_run=False):
    """归档已重构的 legacy notebooks"""
    moves = []
    notebooks_dir = os.path.join(ROOT_DIR, "notebooks")
    archive_nb_dir = os.path.join(ARCHIVE_DIR, "notebooks")

    for nb in LEGACY_NOTEBOOKS:
        src = os.path.join(notebooks_dir, nb)
        if os.path.exists(src):
            dest = os.path.join(archive_nb_dir, nb)
            moves.append((src, dest, "notebook"))

    # notebooks/output/ 目录（空的）
    nb_output = os.path.join(notebooks_dir, "output")
    if os.path.isdir(nb_output) and not os.listdir(nb_output):
        if not dry_run:
            os.rmdir(nb_output)
            print(f"  🗑️  删除空目录: notebooks/output/")
        else:
            print(f"  🗑️  [DRY-RUN] 将删除空目录: notebooks/output/")

    return moves


def archive_legacy_items(dry_run=False):
    """归档测试/实验遗留文件"""
    moves = []

    for src_rel, dest_rel in LEGACY_ITEMS.items():
        src = os.path.join(ROOT_DIR, src_rel)
        dest = os.path.join(ARCHIVE_DIR, dest_rel)
        if os.path.exists(src):
            moves.append((src, dest, "legacy"))

    return moves


def execute_moves(moves, dry_run=False):
    """执行文件移动操作"""
    if not moves:
        print("  ✅ 没有需要归档的文件")
        return 0

    # 按类别统计
    by_category = defaultdict(list)
    for src, dest, cat in moves:
        by_category[cat].append((src, dest))

    total = 0
    category_labels = {
        "output": "📦 输出文件归档",
        "trade_data": "💰 交易数据归档",
        "notebook": "📓 Legacy Notebook 归档",
        "legacy": "🧹 测试/实验遗留文件",
    }

    for cat, cat_moves in by_category.items():
        label = category_labels.get(cat, cat)
        print(f"\n{'='*60}")
        print(f"  {label} ({len(cat_moves)} 个文件)")
        print(f"{'='*60}")

        for src, dest in sorted(cat_moves):
            src_rel = os.path.relpath(src, ROOT_DIR)
            dest_rel = os.path.relpath(dest, ROOT_DIR)

            if dry_run:
                print(f"  📋 {src_rel}")
                print(f"     → {dest_rel}")
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.isdir(src):
                    shutil.move(src, dest)
                else:
                    shutil.move(src, dest)
                print(f"  ✅ {src_rel}")
                print(f"     → {dest_rel}")
            total += 1

    return total


def print_summary(moves, dry_run=False):
    """打印归档汇总"""
    by_category = defaultdict(int)
    for _, _, cat in moves:
        by_category[cat] += 1

    mode = "[DRY-RUN 预览]" if dry_run else "[已完成]"
    print(f"\n{'='*60}")
    print(f"  📊 归档汇总 {mode}")
    print(f"{'='*60}")
    for cat, count in sorted(by_category.items()):
        labels = {
            "output": "输出文件",
            "trade_data": "交易数据",
            "notebook": "Legacy Notebooks",
            "legacy": "测试/实验遗留",
        }
        print(f"  • {labels.get(cat, cat)}: {count} 个")
    print(f"  • 总计: {len(moves)} 个文件")

    if dry_run:
        print(f"\n  💡 使用不带 --dry-run 参数运行以实际执行归档")


def main():
    parser = argparse.ArgumentParser(
        description="QuantPits 带日期文件归档工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python engine/scripts/archive_dated_files.py --dry-run           # 预览模式
  python engine/scripts/archive_dated_files.py                      # 实际归档（自动读取锚点日期）
  python engine/scripts/archive_dated_files.py --anchor-date 2026-02-13  # 指定锚点日期
  python engine/scripts/archive_dated_files.py --include-notebooks  # 同时归档 legacy notebooks
  python engine/scripts/archive_dated_files.py --cleanup-legacy     # 清理测试/实验遗留
  python engine/scripts/archive_dated_files.py --all                # 全部归档（含 notebooks + legacy）
        """
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，只显示将要移动的文件，不实际操作")
    parser.add_argument("--anchor-date", type=str, default=None,
                        help="锚点日期 YYYY-MM-DD（默认从 latest_train_records.json 读取）。"
                             "该日期及之后的文件保留，之前的归档")
    parser.add_argument("--skip-trade-data", action="store_true",
                        help="跳过交易数据归档（buy/sell_suggestion, model_opinions, trade_detail 等）")
    parser.add_argument("--include-notebooks", action="store_true",
                        help="同时归档已重构的 legacy notebooks")
    parser.add_argument("--cleanup-legacy", action="store_true",
                        help="清理测试/实验遗留文件（Alpha158_full, compare_results.py 等）")
    parser.add_argument("--all", action="store_true",
                        help="全部归档（等同于 --include-notebooks --cleanup-legacy）")

    args = parser.parse_args()

    if args.all:
        args.include_notebooks = True
        args.cleanup_legacy = True

    # 获取锚点日期
    anchor_date = get_anchor_date(args.anchor_date)

    print(f"{'='*60}")
    print(f"  QuantPits 文件归档工具")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模式: {'🔍 DRY-RUN 预览' if args.dry_run else '🚀 实际执行'}")
    print(f"  锚点日期: {anchor_date}")
    print(f"  规则: 日期 < {anchor_date} 的文件归档，>= 的保留")
    print(f"{'='*60}")

    all_moves = []

    # ── 带日期文件归档 ──────────────────────────────────────────────────
    print(f"\n📂 扫描带日期文件...")
    dated_moves = plan_archive(anchor_date)

    if args.skip_trade_data:
        dated_moves = [(s, d, c) for s, d, c in dated_moves if c != "trade_data"]

    all_moves.extend(dated_moves)

    # ── Legacy notebooks ───────────────────────────────────────────────
    if args.include_notebooks:
        print(f"\n📓 扫描 legacy notebooks...")
        nb_moves = archive_legacy_notebooks(dry_run=args.dry_run)
        all_moves.extend(nb_moves)

    # ── 测试/实验遗留 ──────────────────────────────────────────────────
    if args.cleanup_legacy:
        print(f"\n🧹 扫描测试/实验遗留文件...")
        legacy_moves = archive_legacy_items(dry_run=args.dry_run)
        all_moves.extend(legacy_moves)

    # ── 执行 ─────────────────────────────────────────────────────────
    if all_moves:
        execute_moves(all_moves, dry_run=args.dry_run)
        print_summary(all_moves, dry_run=args.dry_run)
    else:
        print(f"\n  ✅ 当前没有需要归档的文件，一切整洁！")

    return 0


if __name__ == "__main__":
    sys.exit(main())
