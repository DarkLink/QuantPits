#!/usr/bin/env python3
"""
Signal Ranking - 生成股票信号评分排名（独立于订单生成）

工作流位置：融合预测 → **信号排名（本脚本）**

用途：将融合预测分数归一化为 -100 ~ +100 的推荐指数，生成 Top N 排名 CSV，
适合分享给他人作为参考。

运行方式：
  cd QuantPits

  # 为 default combo 生成 Top 300 排名
  python quantpits/scripts/signal_ranking.py

  # 为所有 combo 各生成一份
  python quantpits/scripts/signal_ranking.py --all-combos

  # 为指定 combo 生成
  python quantpits/scripts/signal_ranking.py --combo combo_A

  # 自定义 TopN
  python quantpits/scripts/signal_ranking.py --top-n 500

  # 自定义 TopN
  python quantpits/scripts/signal_ranking.py --top-n 500

参数：
  --combo             指定 combo 名称
  --all-combos        为所有 combo 各生成一份
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
from quantpits.utils import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR

PREDICTION_DIR = os.path.join(ROOT_DIR, "output", "predictions")
ENSEMBLE_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "ensemble_config.json")


# ============================================================================
# 配置解析 (委托给 ensemble_utils)
# ============================================================================
def parse_ensemble_config(config_file=None):
    """
    解析 ensemble_config.json，兼容新旧格式。

    Args:
        config_file: 配置文件路径 (默认 ENSEMBLE_CONFIG_FILE)

    Returns:
        combos: dict, combo_name -> {"models": [], "method": str, "default": bool}
    """
    _config_file = config_file or ENSEMBLE_CONFIG_FILE
    if not os.path.exists(_config_file):
        return {}

    with open(_config_file, 'r') as f:
        config = json.load(f)

    from quantpits.utils.ensemble_utils import parse_ensemble_config as _parse
    combos, _ = _parse(config)
    return combos


def get_default_combo(combos):
    """返回 default combo 的 (name, config)"""
    from quantpits.utils.ensemble_utils import get_default_combo as _get
    return _get(combos)


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


def get_prediction_from_recorder(combo_name=None):
    """
    从 Qlib Recorder 获取预测数据。
    """
    from qlib.workflow import R
    from quantpits.utils.ensemble_utils import load_ensemble_records, resolve_combo_record_id

    records = load_ensemble_records(ROOT_DIR)
    combo_name, record_id = resolve_combo_record_id(records, combo_name)

    recorder = R.get_recorder(recorder_id=record_id, experiment_name="Ensemble_Fusion")
    pred_df = recorder.load_object("pred.pkl")
    if isinstance(pred_df, pd.Series):
         pred_df = pred_df.to_frame('score')
    return pred_df, record_id

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
  python quantpits/scripts/signal_ranking.py

  # 为所有 combo 各生成
  python quantpits/scripts/signal_ranking.py --all-combos

  # 指定 combo
  python quantpits/scripts/signal_ranking.py --combo combo_A

  # 自定义 Top N
  python quantpits/scripts/signal_ranking.py --top-n 500
"""
    )
    parser.add_argument('--combo', type=str,
                        help='指定 combo 名称')
    parser.add_argument('--all-combos', action='store_true',
                        help='为所有 combo 各生成一份')
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
    env.init_qlib()

    # ---- 确定要处理的任务列表 ----
    # 每个任务: (label, pred_df, source_desc)
    tasks = []

    if args.all_combos:
        combos = parse_ensemble_config()
        if not combos:
            print("Error: ensemble_config.json 中没有 combos")
            sys.exit(1)
        for name, cfg in combos.items():
            try:
                pred_df, rid = get_prediction_from_recorder(combo_name=name)
                tasks.append((name, pred_df, f"recorder {rid}"))
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        if not tasks:
            print("Error: 没有找到任何 combo 的预测记录")
            sys.exit(1)
        print(f"\n多组合模式: 共 {len(tasks)} 个 combo")

    elif args.combo:
        pred_df, rid = get_prediction_from_recorder(combo_name=args.combo)
        tasks.append((args.combo, pred_df, f"recorder {rid}"))

    else:
        # Default: 使用最新 ensemble 预测
        pred_df, rid = get_prediction_from_recorder()
        tasks.append(('default', pred_df, f"recorder {rid}"))

    # ---- 逐任务处理 ----
    generated_files = []

    for label, pred_df, source_desc in tasks:
        print(f"\n{'='*60}")
        print(f"处理: {label}")
        print(f"来源: {source_desc}")
        print(f"{'='*60}")

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
    for label, _, source_desc in tasks:
        print(f"  {label}: {source_desc}")
    if generated_files:
        print(f"\n生成的文件:")
        for f in generated_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
