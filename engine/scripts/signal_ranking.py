#!/usr/bin/env python3
"""
Signal Ranking - 生成股票信号评分排名（独立于订单生成）

工作流位置：融合预测 → **信号排名（本脚本）**

用途：将融合预测分数归一化为 -100 ~ +100 的推荐指数，生成 Top N 排名 CSV，
适合分享给他人作为参考。

运行方式：
  cd QuantPits

  # 为 default combo 生成 Top 300 排名
  python engine/scripts/signal_ranking.py

  # 为所有 combo 各生成一份
  python engine/scripts/signal_ranking.py --all-combos

  # 为指定 combo 生成
  python engine/scripts/signal_ranking.py --combo combo_A

  # 自定义 TopN
  python engine/scripts/signal_ranking.py --top-n 500

  # 指定预测文件
  python engine/scripts/signal_ranking.py --prediction-file output/predictions/ensemble_2026-02-13.csv

参数：
  --combo             指定 combo 名称
  --all-combos        为所有 combo 各生成一份
  --prediction-file   直接指定预测文件路径
  --top-n             输出 Top N 个标的 (默认 300)
  --output-dir        输出目录 (默认 output/ranking)
  --dry-run           仅打印，不写入文件
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR

PREDICTION_DIR = os.path.join(ROOT_DIR, "output", "predictions")
ENSEMBLE_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "ensemble_config.json")


# ============================================================================
# 配置解析 (复用 ensemble_fusion.py 的逻辑)
# ============================================================================
def parse_ensemble_config():
    """
    解析 ensemble_config.json，兼容新旧格式。

    Returns:
        combos: dict, combo_name -> {"models": [], "method": str, "default": bool}
    """
    if not os.path.exists(ENSEMBLE_CONFIG_FILE):
        return {}

    with open(ENSEMBLE_CONFIG_FILE, 'r') as f:
        config = json.load(f)

    if 'combos' in config:
        return config['combos']
    elif 'models' in config:
        # 旧格式
        return {
            'legacy': {
                'models': config['models'],
                'method': config.get('ensemble_method', 'equal'),
                'default': True,
            }
        }
    return {}


def get_default_combo(combos):
    """返回 default combo 的 (name, config)"""
    for name, cfg in combos.items():
        if cfg.get('default', False):
            return name, cfg
    if combos:
        first_name = next(iter(combos))
        return first_name, combos[first_name]
    return None, None


# ============================================================================
# 信号评分核心逻辑
# ============================================================================
def generate_signal_scores(pred_df, top_n=300):
    """
    将预测分数归一化为 -100 ~ +100 的推荐指数。

    Args:
        pred_df: DataFrame with 'score' column, index=(instrument, datetime)
        top_n: 输出 Top N 个标的

    Returns:
        result_df: DataFrame with columns ['推荐指数'], index=股票代码, 按推荐指数降序
    """
    # 取最新一天
    if 'datetime' in pred_df.index.names:
        latest_date = pred_df.index.get_level_values('datetime').max()
        if len(pred_df.index.get_level_values('datetime').unique()) > 1:
            daily_df = pred_df.xs(latest_date, level='datetime').copy()
        else:
            daily_df = pred_df.copy()
            if 'datetime' in daily_df.index.names:
                daily_df = daily_df.droplevel('datetime')
    else:
        daily_df = pred_df.copy()
        latest_date = None

    # 归一化到 -100 ~ +100
    raw_min = daily_df['score'].min()
    raw_max = daily_df['score'].max()
    raw_range = raw_max - raw_min

    if raw_range == 0:
        daily_df['signal_score'] = 0.00
    else:
        daily_df['signal_score'] = ((daily_df['score'] - raw_min) / raw_range * 200) - 100

    # 排序并取 Top N
    result_df = daily_df.sort_values(by='signal_score', ascending=False).head(top_n)
    result_df['推荐指数'] = result_df['signal_score'].round(2)

    output_df = result_df[['推荐指数']].copy()
    output_df.index.name = '股票代码'

    return output_df, latest_date


def find_prediction_file(combo_name=None, anchor_date=None):
    """
    查找预测文件。

    Args:
        combo_name: combo 名称，None 表示查找 default ensemble
        anchor_date: 日期限制

    Returns:
        pred_file: 文件路径
    """
    if combo_name:
        pattern = os.path.join(PREDICTION_DIR, f"ensemble_{combo_name}_*.csv")
    else:
        # 查找不带 combo name 的通用 ensemble 文件
        pattern = os.path.join(PREDICTION_DIR, "ensemble_*.csv")

    files = sorted(glob.glob(pattern))

    if combo_name:
        # 精确匹配带 combo_name 的文件
        if not files:
            raise FileNotFoundError(
                f"未找到 combo '{combo_name}' 的预测文件。\n"
                f"搜索路径: {pattern}\n"
                "请先运行 ensemble_fusion.py"
            )
        return files[-1]
    else:
        # 过滤掉带 combo_name 的文件（它们包含两个 _ 分隔的部分在 ensemble_ 之后）
        # ensemble_2026-02-13.csv vs ensemble_combo_A_2026-02-13.csv
        generic_files = []
        for f in files:
            basename = os.path.basename(f)
            # 移除 "ensemble_" 前缀和 ".csv" 后缀，检查剩余部分是否像日期
            rest = basename[len("ensemble_"):-len(".csv")]
            # 如果剩余部分看起来像一个日期 (YYYY-MM-DD)，则是通用文件
            if len(rest) == 10 and rest[4] == '-' and rest[7] == '-':
                generic_files.append(f)

        if not generic_files:
            # 退化：直接使用最新的任何 ensemble 文件
            if files:
                return files[-1]
            raise FileNotFoundError(
                "未找到 ensemble 预测文件。\n"
                f"搜索路径: {pattern}\n"
                "请先运行 ensemble_fusion.py"
            )
        return generic_files[-1]


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Signal Ranking - 生成股票信号评分排名',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 为 default combo 生成 Top 300
  python engine/scripts/signal_ranking.py

  # 为所有 combo 各生成
  python engine/scripts/signal_ranking.py --all-combos

  # 指定 combo
  python engine/scripts/signal_ranking.py --combo combo_A

  # 自定义 Top N
  python engine/scripts/signal_ranking.py --top-n 500
"""
    )
    parser.add_argument('--combo', type=str,
                        help='指定 combo 名称')
    parser.add_argument('--all-combos', action='store_true',
                        help='为所有 combo 各生成一份')
    parser.add_argument('--prediction-file', type=str,
                        help='直接指定预测文件路径')
    parser.add_argument('--top-n', type=int, default=300,
                        help='输出 Top N 个标的 (默认 300)')
    parser.add_argument('--output-dir', type=str, default='output/ranking',
                        help='输出目录 (默认 output/ranking)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印，不写入文件')
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("# Signal Ranking — 信号排名")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    if args.dry_run:
        print("\n⚠️  DRY-RUN 模式: 不会写入任何文件")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 确定要处理的任务列表 ----
    # 每个任务: (label, pred_file)
    tasks = []

    if args.prediction_file:
        if not os.path.exists(args.prediction_file):
            print(f"Error: 指定的预测文件不存在: {args.prediction_file}")
            sys.exit(1)
        tasks.append(('custom', args.prediction_file))

    elif args.all_combos:
        combos = parse_ensemble_config()
        if not combos:
            print("Error: ensemble_config.json 中没有 combos")
            sys.exit(1)
        for name, cfg in combos.items():
            try:
                pred_file = find_prediction_file(combo_name=name)
                tasks.append((name, pred_file))
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        if not tasks:
            print("Error: 没有找到任何 combo 的预测文件")
            sys.exit(1)
        print(f"\n多组合模式: 共 {len(tasks)} 个 combo")

    elif args.combo:
        pred_file = find_prediction_file(combo_name=args.combo)
        tasks.append((args.combo, pred_file))

    else:
        # Default: 使用最新 ensemble 预测
        pred_file = find_prediction_file()
        tasks.append(('default', pred_file))

    # ---- 逐任务处理 ----
    generated_files = []

    for label, pred_file in tasks:
        print(f"\n{'='*60}")
        print(f"处理: {label}")
        print(f"预测文件: {pred_file}")
        print(f"{'='*60}")

        # 加载预测
        pred_df = pd.read_csv(pred_file, index_col=[0, 1], parse_dates=[1])
        print(f"预测数据量: {len(pred_df)} 条")

        # 生成信号评分
        output_df, latest_date = generate_signal_scores(pred_df, top_n=args.top_n)

        date_str = str(latest_date)[:10] if latest_date else 'unknown'
        print(f"最新日期  : {date_str}")
        print(f"Top {args.top_n} 标的: {len(output_df)} 个")

        # 显示 Top 5
        print(f"\nTop 5:")
        print(output_df.head(5))

        # 保存
        output_filename = os.path.join(
            args.output_dir,
            f"Signal_{label}_{date_str}_Top{args.top_n}.csv"
        )

        if args.dry_run:
            print(f"\n[DRY-RUN] 不写入: {output_filename}")
        else:
            output_df.to_csv(output_filename, encoding='utf-8-sig')
            print(f"\n✅ 已保存: {output_filename}")
            generated_files.append(output_filename)

    # ---- 完成 ----
    print(f"\n{'#'*60}")
    print("# ✅ Signal Ranking 完成!")
    print(f"{'#'*60}")
    print(f"处理了 {len(tasks)} 个预测源")
    for label, pred_file in tasks:
        print(f"  {label}: {os.path.basename(pred_file)}")
    if generated_files:
        print(f"\n生成的文件:")
        for f in generated_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
