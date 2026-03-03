#!/usr/bin/env python3
"""
Order Generation - 基于融合/单模型预测生成买卖订单

工作流位置：训练 → 穷举 → 融合回测 → Post-Trade → **订单生成（本脚本）**

前置条件：
  - 已运行预测/融合脚本，output/predictions/ 中有预测结果
  - 已运行 post-trade 脚本，prod_config.json 中有最新持仓和现金

运行方式：
  # 使用最新融合预测
  python quantpits/scripts/order_gen.py

  # 使用单模型预测（不融合）
  python quantpits/scripts/order_gen.py --model gru

  # 指定预测文件
  python quantpits/scripts/order_gen.py --prediction-file output/predictions/ensemble_2026-02-06.csv

  # 仅预览
  python quantpits/scripts/order_gen.py --dry-run

参数：
  --model            使用单模型预测（从 output/predictions/{model}_{date}.csv 加载）
  --prediction-file  直接指定预测文件路径
  --output-dir       输出目录 (默认 output)
  --dry-run          仅打印订单计划，不写入文件
  --verbose          显示详细的排名和价格信息
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from quantpits.scripts import env
os.chdir(env.ROOT_DIR)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
# 已在上方导入并切换目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR

CONFIG_FILE = os.path.join(ROOT_DIR, "config", "prod_config.json")
CASHFLOW_FILE = os.path.join(ROOT_DIR, "config", "cashflow.json")
PREDICTION_DIR = os.path.join(ROOT_DIR, "output", "predictions")
ENSEMBLE_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "ensemble_config.json")


# ============================================================================
# Stage 0: 初始化 & 配置加载
# ============================================================================
def init_qlib():
    """初始化 Qlib（委托给 env.init_qlib）"""
    env.init_qlib()


def get_anchor_date():
    """获取锚点日期（最近的前一交易日）"""
    from qlib.data import D
    last_trade_date = D.calendar(future=False)[-1:][0]
    return last_trade_date.strftime('%Y-%m-%d')


def load_configs():
    """加载 prod_config.json 和 cashflow.json"""
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    cashflow_config = {}
    if os.path.exists(CASHFLOW_FILE):
        with open(CASHFLOW_FILE, 'r') as f:
            cashflow_config = json.load(f)

    return config, cashflow_config


def get_cashflow_today(cashflow_config, anchor_date):
    """获取当日 cashflow 金额（支持新旧两种格式）"""
    # 新格式: {"cashflows": {"2026-02-03": 50000}}
    cashflows = cashflow_config.get('cashflows', {})
    if cashflows and anchor_date in cashflows:
        return float(cashflows[anchor_date])

    # 旧格式: {"cash_flow_today": 50000}
    return float(cashflow_config.get('cash_flow_today', 0))


# ============================================================================
# Stage 1: 加载预测数据
# ============================================================================
def load_predictions(prediction_file=None, model_name=None, anchor_date=None):
    """
    加载预测数据。

    优先级: prediction_file > model_name > 自动搜索 ensemble CSV

    Args:
        prediction_file: 直接指定的预测文件路径
        model_name: 单模型名称
        anchor_date: 锚点日期

    Returns:
        pred_df: DataFrame with 'score' column, index=(instrument,) or (instrument, datetime)
        source_desc: str, 预测来源描述
    """
    if prediction_file:
        # 直接指定文件
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"指定的预测文件不存在: {prediction_file}")
        pred_df = pd.read_csv(prediction_file, index_col=[0, 1], parse_dates=[1])
        return pred_df, f"指定文件: {prediction_file}"

    if model_name:
        # 按模型名搜索
        pattern = os.path.join(PREDICTION_DIR, f"{model_name}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"未找到模型 {model_name} 的预测文件。\n"
                f"搜索路径: {pattern}\n"
                f"请先运行训练/预测脚本。"
            )
        pred_file = files[-1]
        pred_df = pd.read_csv(pred_file, index_col=[0, 1], parse_dates=[1])
        return pred_df, f"单模型: {model_name} ({os.path.basename(pred_file)})"

    # 自动搜索 ensemble 预测
    # 优先级: ensemble_YYYY-MM-DD.csv (default combo 的向后兼容副本)
    #       > ensemble_default_YYYY-MM-DD.csv (显式 default combo)
    #       > ensemble_*.csv (任意 combo)
    pred_file = None

    # 1) 向后兼容格式: ensemble_YYYY-MM-DD.csv (无 combo 名)
    compat_pattern = os.path.join(PREDICTION_DIR, "ensemble_[0-9]*.csv")
    compat_files = sorted(glob.glob(compat_pattern))
    if compat_files:
        pred_file = compat_files[-1]

    # 2) 若无，尝试 ensemble_default_YYYY-MM-DD.csv
    if not pred_file:
        default_pattern = os.path.join(PREDICTION_DIR, "ensemble_default_*.csv")
        default_files = sorted(glob.glob(default_pattern))
        if default_files:
            pred_file = default_files[-1]

    # 3) 若仍无，回退到任意 ensemble_*.csv（按日期排序）
    if not pred_file:
        pattern = os.path.join(PREDICTION_DIR, "ensemble_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                "未找到 ensemble 预测文件。\n"
                f"搜索路径: {pattern}\n"
                "请先运行 ensemble_fusion.py，或使用 --model 指定单模型。"
            )
        pred_file = files[-1]
    pred_df = pd.read_csv(pred_file, index_col=[0, 1], parse_dates=[1])

    # 尝试加载对应的 ensemble 配置
    config_pattern = os.path.join(ROOT_DIR, "output", "ensemble", "ensemble_fusion_config_*.json")
    config_files = sorted(glob.glob(config_pattern))
    # 也检查旧格式位置
    if not config_files:
        config_pattern = os.path.join(ROOT_DIR, "output", "ensemble_config_*.json")
        config_files = sorted(glob.glob(config_pattern))

    ensemble_info = ""
    if config_files:
        try:
            with open(config_files[-1], 'r') as f:
                ens_cfg = json.load(f)
            method = ens_cfg.get('weight_mode', ens_cfg.get('method', 'unknown'))
            models = ens_cfg.get('models_used', [])
            ensemble_info = f"\n  融合方式: {method}\n  模型组合: {', '.join(models)}"
        except Exception:
            pass

    return pred_df, f"Ensemble 融合 ({os.path.basename(pred_file)}){ensemble_info}"


# ============================================================================
# Stage 2: 价格数据获取
# ============================================================================
def get_price_data(anchor_date, market):
    """
    获取当日复权价格和涨跌停估价。

    Args:
        anchor_date: 锚点日期
        market: 市场名称

    Returns:
        price_df: DataFrame with columns [current_close, possible_max, possible_min]
    """
    from qlib.data import D
    from qlib.data.ops import Feature

    instruments = D.instruments(market=market)
    current_close = Feature("close") / Feature("factor")
    possible_max = Feature("close") / Feature("factor") * 1.1
    possible_min = Feature("close") / Feature("factor") * 0.9

    features_df = D.features(
        instruments=instruments,
        start_time=anchor_date,
        fields=[current_close, possible_max, possible_min]
    )

    features_df.rename(columns={
        'Div($close,$factor)': 'current_close',
        'Mul(Div($close,$factor),1.1)': 'possible_max',
        'Mul(Div($close,$factor),0.9)': 'possible_min'
    }, inplace=True)

    return features_df


# ============================================================================
# Stage 3: 排序与持仓分析
# ============================================================================
# Note: analyze_positions, generate_sell_orders, and generate_buy_orders
# have been extracted to quantpits/scripts/strategy.py to decouple strategy logic.


# ============================================================================
# Stage 3.5: 多模型判断表
# ============================================================================
def _load_pred_latest_day(pred_source, source_type, valid_instruments=None):
    """
    统一加载预测数据并返回最新一天的 DataFrame（按 score 降序，index=instrument）。

    支持:
      - CSV 文件 (score 列 或 列名为 '0')
      - Qlib Recorder pkl DataFrame

    Args:
        pred_source: 文件路径或 DataFrame
        source_type: 'model', 'combo', 'model_pkl'
        valid_instruments: set, 可选, 仅保留这些标的
    """
    if source_type == 'model_pkl':
        df = pred_source.copy()
    else:
        df = pd.read_csv(pred_source)
        # 统一列名: 单模型 CSV 的预测列可能叫 '0'
        if '0' in df.columns and 'score' not in df.columns:
            df = df.rename(columns={'0': 'score'})
        # 设置 multi-index
        if 'instrument' in df.columns and 'datetime' in df.columns:
            df = df.set_index(['instrument', 'datetime'])

    # 确保有 score 列
    if 'score' not in df.columns:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            df = df.rename(columns={num_cols[0]: 'score'})
        else:
            return None

    # 取最新一天
    if 'datetime' in df.index.names:
        latest_date = df.index.get_level_values('datetime').max()
        if len(df.index.get_level_values('datetime').unique()) > 1:
            daily_df = df.xs(latest_date, level='datetime')
        else:
            daily_df = df.droplevel('datetime') if 'datetime' in df.index.names else df
    elif 'datetime' in df.columns:
        latest_date = df['datetime'].max()
        daily_df = df[df['datetime'] == latest_date].set_index('instrument')
    else:
        daily_df = df

    # 确保 index 是 instrument
    if 'instrument' in daily_df.columns:
        daily_df = daily_df.set_index('instrument')

    # 过滤到有效标的（与 analyze_positions 的 price merge 对齐）
    if valid_instruments is not None:
        daily_df = daily_df[daily_df.index.isin(valid_instruments)]

    return daily_df.sort_values('score', ascending=False)


def generate_model_opinions(focus_instruments, current_holding_instruments,
                            top_k, drop_n, buy_suggestion_factor,
                            sorted_df, output_dir, next_trade_date_string,
                            dry_run=False):
    """
    加载所有 combo 和单一模型的预测，对每个标的生成判断。

    判断逻辑（与 analyze_positions 一致）：
      - 持仓在候选池内 → HOLD
      - 持仓池外, 最差 DropN → SELL
      - 持仓池外, 非最差 → HOLD
      - 非持仓, 排名靠前 → BUY / BUY*
      - 非持仓, 排名靠后 → --

    Args:
        focus_instruments: list, 关注标的列表
        current_holding_instruments: list, 当前持仓代码列表
        top_k: TopK 阈值
        drop_n: DropN 阈值
        buy_suggestion_factor: 买入倍数
        sorted_df: DataFrame, analyze_positions 输出的排序数据（已与价格合并）
        output_dir: 输出目录
        next_trade_date_string: 下一交易日
        dry_run: 是否 dry-run 模式

    Returns:
        opinions_df, combo_info
    """
    # 有效标的集合（只考虑有价格数据的）
    valid_instruments = set(sorted_df.index.tolist())

    # 加载 ensemble 配置
    combos = {}
    if os.path.exists(ENSEMBLE_CONFIG_FILE):
        with open(ENSEMBLE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        if 'combos' in config:
            combos = config['combos']
        elif 'models' in config:
            combos = {'legacy': {
                'models': config['models'],
                'default': True,
            }}

    # 收集所有预测源: (label, source, source_type, details)
    sources = []
    combo_info = {}

    # 1) Combo 预测
    for combo_name, cfg in combos.items():
        combo_info[combo_name] = cfg.get('models', [])
        pattern = os.path.join(PREDICTION_DIR, f"ensemble_{combo_name}_*.csv")
        files = sorted(glob.glob(pattern))
        if files:
            sources.append((f"combo_{combo_name}", files[-1], 'combo', combo_name))
            continue
        if cfg.get('default', False):
            pattern2 = os.path.join(PREDICTION_DIR, "ensemble_*.csv")
            generic_files = []
            for f_path in sorted(glob.glob(pattern2)):
                basename = os.path.basename(f_path)
                rest = basename[len("ensemble_"):-len(".csv")]
                if len(rest) == 10 and rest[4] == '-' and rest[7] == '-':
                    generic_files.append(f_path)
            if generic_files:
                sources.append((f"combo_{combo_name}", generic_files[-1], 'combo', combo_name))

    # 2) 单一模型预测 (CSV 优先, Qlib Recorder 后备)
    all_single_models = set()
    for cfg in combos.values():
        all_single_models.update(cfg.get('models', []))
    for model_name in sorted(all_single_models):
        pattern = os.path.join(PREDICTION_DIR, f"{model_name}_*.csv")
        files = sorted(glob.glob(pattern))
        if files:
            sources.append((f"model_{model_name}", files[-1], 'model', model_name))
        else:
            try:
                train_records_file = os.path.join(ROOT_DIR, 'config', 'latest_train_records.json')
                if os.path.exists(train_records_file):
                    with open(train_records_file, 'r') as f:
                        train_records = json.load(f)
                    record_id = train_records.get('models', {}).get(model_name)
                    if record_id:
                        from qlib.workflow import R
                        experiment_name = train_records.get('experiment_name', 'prod_train')
                        recorder = R.get_recorder(recorder_id=record_id,
                                                  experiment_name=experiment_name)
                        pred_pkl = recorder.load_object('pred.pkl')
                        if pred_pkl is not None and len(pred_pkl) > 0:
                            sources.append((f"model_{model_name}", pred_pkl, 'model_pkl', model_name))
            except Exception:
                pass

    if not sources:
        print("  未找到额外预测文件，跳过多模型判断")
        return None, {}

    # 在所有来源前插入 order_basis（使用 sorted_df，与实际订单完全一致）
    sources.insert(0, ("order_basis", None, 'sorted_df', 'order_basis'))

    print(f"  预测源: {len(sources)} 个 "
          f"(order_basis: 1, combo: {sum(1 for s in sources if s[2] == 'combo')}, "
          f"model: {sum(1 for s in sources if s[2] in ('model', 'model_pkl'))})")

    # 对每个预测源，模拟 analyze_positions 的换仓逻辑
    holding_set = set(current_holding_instruments)

    # 预加载所有预测源（过滤到有效标的集合，与 analyze_positions 对齐）
    pred_cache = {}  # label -> sorted DataFrame or None
    for label, pred_source, source_type, detail in sources:
        if source_type == 'sorted_df':
            # 直接使用 analyze_positions 的输出（已排序、已合并价格）
            pred_cache[label] = sorted_df[['score']].sort_values('score', ascending=False)
            continue
        try:
            pred_cache[label] = _load_pred_latest_day(
                pred_source, source_type, valid_instruments=valid_instruments
            )
        except Exception:
            pred_cache[label] = None

    # 对每个预测源，模拟完整的换仓决策（与 analyze_positions 一致）
    source_decisions = {}  # label -> {instrument: action}
    for label, pred_source, source_type, detail in sources:
        sorted_preds = pred_cache.get(label)
        if sorted_preds is None:
            source_decisions[label] = {}
            continue

        decisions = {}
        all_instruments = sorted_preds.index.tolist()  # 已按 score 降序

        # 1) 确定候选池 = top (TopK + DropN * factor)
        pool_size = top_k + drop_n * buy_suggestion_factor
        pool_instruments = set(all_instruments[:pool_size])

        # 2) 持仓中在池内的 → 暂定 HOLD
        held_in_pool = [inst for inst in all_instruments if inst in holding_set and inst in pool_instruments]
        # 持仓中在池外的 → 卖出候选（取最差 DropN）
        held_outside_pool = [inst for inst in all_instruments if inst in holding_set and inst not in pool_instruments]
        sell_set = set(held_outside_pool[-drop_n:]) if held_outside_pool else set()

        # 3) 持仓决策：卖出 or 持有
        held_in_ranking = [inst for inst in all_instruments if inst in holding_set]
        for inst in held_in_ranking:
            decisions[inst] = 'SELL' if inst in sell_set else 'HOLD'

        # 4) 买入决策
        hold_final_set = set(held_in_ranking) - sell_set
        buy_count = max(0, top_k - len(hold_final_set))

        # 非持仓的池内标的，按排名取 buy_count * factor 个
        non_held_in_pool = [inst for inst in all_instruments[:pool_size] if inst not in holding_set]
        buy_primary = set(non_held_in_pool[:buy_count])
        buy_backup = set(non_held_in_pool[buy_count:buy_count * buy_suggestion_factor])

        for inst in non_held_in_pool:
            if inst in buy_primary:
                decisions[inst] = 'BUY'
            elif inst in buy_backup:
                decisions[inst] = 'BUY*'
            else:
                decisions[inst] = '--'

        # 池外的非持仓
        for inst in all_instruments[pool_size:]:
            if inst not in holding_set:
                decisions[inst] = '--'

        # 附加排名信息
        ranks = {inst: idx + 1 for idx, inst in enumerate(all_instruments)}
        for inst in decisions:
            if inst in ranks and decisions[inst] != '-':
                decisions[inst] = f"{decisions[inst]} ({ranks[inst]})"

        source_decisions[label] = decisions

    # 组装 opinions 表
    opinion_rows = []
    for instrument in focus_instruments:
        row = {'instrument': instrument}
        for label, pred_source, source_type, detail in sources:
            decisions = source_decisions.get(label, {})
            row[label] = decisions.get(instrument, '-')
        opinion_rows.append(row)

    opinions_df = pd.DataFrame(opinion_rows)
    if not opinions_df.empty:
        opinions_df = opinions_df.set_index('instrument')

    # 构造 combo 信息
    model_to_combos = {}
    for combo_name, models in combo_info.items():
        for m in models:
            model_to_combos.setdefault(m, []).append(combo_name)

    # 保存
    csv_file = os.path.join(output_dir, f"model_opinions_{next_trade_date_string}.csv")
    json_file = os.path.join(output_dir, f"model_opinions_{next_trade_date_string}.json")

    json_data = {
        'trade_date': next_trade_date_string,
        'combo_composition': combo_info,
        'model_to_combos': model_to_combos,
        'thresholds': {
            'TopK': top_k, 'DropN': drop_n,
            'buy_suggestion_factor': buy_suggestion_factor,
        },
        'legend': {
            'BUY': '非持仓, 排名靠前的买入候选 (数量 = 卖出数)',
            'BUY*': '非持仓, 备选买入 (应对停牌等情况)',
            'HOLD': '持仓, 继续持有',
            'SELL': '持仓, TopK 之外的最差 DropN',
            '--': '非持仓, 不在买入候选范围',
            '-': '无数据',
            '说明': '决策后的括号内数字表示该模型或组合下的预测排名',
        },
        'sources': [(label,
                     'sorted_df (与订单一致)' if f is None
                     else (os.path.basename(f) if isinstance(f, str) else f'pkl:{detail}'),
                     stype) for label, f, stype, detail in sources],
    }

    if dry_run:
        print(f"  [DRY-RUN] 不写入: {csv_file}")
        print(f"  [DRY-RUN] 不写入: {json_file}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        opinions_df.to_csv(csv_file)
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"  多模型判断表: {csv_file}")
        print(f"  模型信息汇总: {json_file}")

    return opinions_df, combo_info


# ============================================================================
# Stage 6: 输出与汇总
# ============================================================================
def save_orders(sell_orders, buy_orders, next_trade_date_string, output_dir,
                source_label, dry_run=False):
    """
    保存订单 CSV 文件和汇总订单 JSON。

    Args:
        sell_orders: 卖出订单列表
        buy_orders: 买入订单列表
        next_trade_date_string: 下一交易日
        output_dir: 输出目录
        source_label: 预测来源标签（用于文件命名）
        dry_run: 是否 dry-run 模式

    Returns:
        sell_file, buy_file: 保存的文件路径（dry-run 时返回目标路径但不实际写入）
    """
    sell_df = pd.DataFrame(sell_orders)
    buy_df = pd.DataFrame(buy_orders)

    sell_file = os.path.join(output_dir, f"sell_suggestion_{source_label}_{next_trade_date_string}.csv")
    buy_file = os.path.join(output_dir, f"buy_suggestion_{source_label}_{next_trade_date_string}.csv")

    if dry_run:
        print(f"\n[DRY-RUN] 以下文件不会被写入:")
        print(f"  卖出订单: {sell_file}")
        print(f"  买入订单: {buy_file}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        if not sell_df.empty:
            sell_df.to_csv(sell_file, index=False)
        if not buy_df.empty:
            buy_df.to_csv(buy_file, index=False)
        print(f"\n📁 订单文件已保存:")
        print(f"  卖出订单: {sell_file}")
        print(f"  买入订单: {buy_file}")

    return sell_file, buy_file


# ============================================================================
# Main
# ============================================================================
def main():
    import env
    env.safeguard("Order Generation")
    parser = argparse.ArgumentParser(
        description='Order Generation - 基于融合/单模型预测生成买卖订单',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用最新融合预测
  python quantpits/scripts/order_gen.py

  # 使用单模型预测（不融合）
  python quantpits/scripts/order_gen.py --model gru

  # 指定预测文件
  python quantpits/scripts/order_gen.py --prediction-file output/predictions/ensemble_2026-02-06.csv

  # 仅预览
  python quantpits/scripts/order_gen.py --dry-run
"""
    )
    parser.add_argument('--model', type=str,
                        help='使用单模型预测（从 output/predictions/{model}_{date}.csv 加载）')
    parser.add_argument('--prediction-file', type=str,
                        help='直接指定预测文件路径')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录 (默认 output)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印订单计划，不写入文件')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细的排名和价格信息')

    args = parser.parse_args()

    # ---- Stage 0: 初始化 ----
    print(f"\n{'#'*60}")
    print("# Order Generation — 订单生成")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    if args.dry_run:
        print("\n⚠️  DRY-RUN 模式: 不会写入任何文件")

    init_qlib()
    anchor_date = get_anchor_date()
    config, cashflow_config = load_configs()

    import strategy
    strategy_config = strategy.load_strategy_config()
    order_gen = strategy.create_order_generator(strategy_config)
    
    # 策略参数（由 Strategy Provider 统一管理）
    st_params = strategy.get_strategy_params(strategy_config)
    top_k = st_params.get('topk', 20)
    drop_n = st_params.get('n_drop', 3)
    buy_suggestion_factor = st_params.get('buy_suggestion_factor', 2)

    # 运行时状态变量（来自 prod_config）
    market = config.get('market', 'csi300')
    current_cash = float(config.get('current_cash', 0))
    current_holding = config.get('current_holding', [])
    cash_flow_today = get_cashflow_today(cashflow_config, anchor_date)

    print(f"\n{'='*60}")
    print("Stage 0: 配置加载")
    print(f"{'='*60}")
    print(f"锚点日期   : {anchor_date}")
    print(f"市场       : {market}")
    print(f"TopK       : {top_k}")
    print(f"DropN      : {drop_n}")
    print(f"买入倍数   : {buy_suggestion_factor}")
    print(f"当前现金   : {current_cash:,.2f}")
    print(f"当前持仓   : {len(current_holding)} 个")
    if cash_flow_today != 0:
        print(f"当日出入金 : {cash_flow_today:+,.2f}")

    # ---- Stage 1: 加载预测 ----
    print(f"\n{'='*60}")
    print("Stage 1: 加载预测数据")
    print(f"{'='*60}")

    pred_df, source_desc = load_predictions(
        prediction_file=args.prediction_file,
        model_name=args.model,
        anchor_date=anchor_date
    )

    print(f"预测来源   : {source_desc}")
    print(f"预测数据量 : {len(pred_df)} 条")

    latest_date = pred_df.index.get_level_values('datetime').max()
    n_instruments = pred_df.xs(latest_date, level='datetime').shape[0] if \
        len(pred_df.index.get_level_values('datetime').unique()) > 1 else len(pred_df)
    print(f"最新日期   : {latest_date}")
    print(f"当日标的数 : {n_instruments}")

    # ---- Stage 2: 价格数据 ----
    print(f"\n{'='*60}")
    print("Stage 2: 获取价格数据")
    print(f"{'='*60}")

    price_df = get_price_data(anchor_date, market)
    print(f"价格数据   : {len(price_df)} 条")

    # ---- 获取下一交易日 ----
    from qlib.data import D
    target_dates = D.calendar(start_time=anchor_date, future=True)[:2]
    if len(target_dates) >= 2:
        next_trade_date = target_dates[1]
    else:
        next_trade_date = target_dates[0]
    next_trade_date_string = next_trade_date.strftime('%Y-%m-%d')
    print(f"下一交易日 : {next_trade_date_string}")

    # ---- Stage 3: 持仓分析 ----
    print(f"\n{'='*60}")
    print("Stage 3: 排序与持仓分析")
    print(f"{'='*60}")

    hold_final, sell_candidates, buy_candidates, sorted_df, buy_count = order_gen.analyze_positions(
        pred_df, price_df, current_holding
    )

    print(f"继续持有   : {len(hold_final)} 个")
    print(f"计划卖出   : {len(sell_candidates)} 个")
    print(f"需要买入   : {buy_count} 个")
    print(f"买入候选   : {len(buy_candidates)} 个")

    if args.verbose:
        print(f"\n--- 继续持有 ---")
        if len(hold_final) > 0:
            print(hold_final[['score', 'current_close']].to_string())

        print(f"\n--- 卖出候选 ---")
        if len(sell_candidates) > 0:
            print(sell_candidates[['score', 'current_close']].to_string())

        print(f"\n--- 买入候选 ---")
        if len(buy_candidates) > 0:
            print(buy_candidates[['score', 'current_close']].to_string())

    # ---- Stage 3.5: 多模型判断表 ----
    print(f"\n{'='*60}")
    print("Stage 3.5: 多模型判断表")
    print(f"{'='*60}")

    # 收集关注标的（持仓 + 买入候选 + 卖出候选）
    focus_instruments = []
    for df in [hold_final, sell_candidates, buy_candidates]:
        if len(df) > 0:
            idx = df.index.get_level_values('instrument') if 'instrument' in df.index.names else df.index
            focus_instruments.extend(idx.tolist())
    focus_instruments = list(dict.fromkeys(focus_instruments))  # 去重保序

    # 当前持仓代码列表
    current_holding_instruments = [h['instrument'] for h in current_holding]

    opinions_df, combo_info = generate_model_opinions(
        focus_instruments, current_holding_instruments,
        top_k, drop_n, buy_suggestion_factor,
        sorted_df, args.output_dir, next_trade_date_string, dry_run=args.dry_run
    )

    if opinions_df is not None and not opinions_df.empty and args.verbose:
        print(f"\n--- 多模型判断（关注标的） ---")
        print(opinions_df.to_string())
        if combo_info:
            print(f"\n--- Combo 组成 ---")
            for name, models in combo_info.items():
                print(f"  {name}: {', '.join(models)}")

    # ---- Stage 4: 卖出订单 ----
    print(f"\n{'='*60}")
    print("Stage 4: 生成卖出订单")
    print(f"{'='*60}")

    sell_orders, sell_amount = order_gen.generate_sell_orders(
        sell_candidates, current_holding, next_trade_date_string
    )

    if sell_orders:
        sell_df = pd.DataFrame(sell_orders)
        print(f"\n卖出订单 ({len(sell_orders)} 笔，预估回收 {sell_amount:,.2f}):")
        print(sell_df.to_string(index=False))
    else:
        print("无卖出订单")

    # ---- Stage 5: 买入订单 ----
    print(f"\n{'='*60}")
    print("Stage 5: 生成买入订单")
    print(f"{'='*60}")

    available_cash = current_cash + sell_amount + cash_flow_today
    print(f"可用现金   : {current_cash:,.2f} (余额)")
    if sell_amount > 0:
        print(f"           + {sell_amount:,.2f} (预估卖出回收)")
    if cash_flow_today != 0:
        print(f"           + {cash_flow_today:+,.2f} (出入金)")
    print(f"           = {available_cash:,.2f} (总可用)")
    if buy_count > 0:
        print(f"每股预算   : {available_cash / buy_count:,.2f}")

    buy_orders = order_gen.generate_buy_orders(
        buy_candidates, buy_count, available_cash, next_trade_date_string
    )

    if buy_orders:
        buy_df = pd.DataFrame(buy_orders)
        amounts = sorted([o['estimated_amount'] for o in buy_orders])
        # 预估区间：取 top buy_count 个最小/最大金额
        min_n = amounts[:buy_count]
        max_n = amounts[-buy_count:] if len(amounts) >= buy_count else amounts
        min_total = sum(min_n)
        max_total = sum(max_n)
        print(f"\n买入备选 ({len(buy_orders)} 笔, 实际买入 {buy_count} 个):")
        print(buy_df.to_string(index=False))
        if min_total == max_total:
            print(f"\n💰 预估支出 : {min_total:,.2f}")
        else:
            print(f"\n💰 预估支出区间 : {min_total:,.2f} ~ {max_total:,.2f}")
    else:
        print("无买入订单")

    # ---- Stage 6: 保存 ----
    print(f"\n{'='*60}")
    print("Stage 6: 保存订单")
    print(f"{'='*60}")

    # 确定文件命名标签
    if args.model:
        source_label = args.model
    elif args.prediction_file:
        source_label = "custom"
    else:
        source_label = "ensemble"

    sell_file, buy_file = save_orders(
        sell_orders, buy_orders, next_trade_date_string,
        args.output_dir, source_label, dry_run=args.dry_run
    )

    # ---- 完成 ----
    print(f"\n{'#'*60}")
    print("# ✅ 订单生成完成!")
    print(f"{'#'*60}")
    print(f"📅 交易日   : {next_trade_date_string}")
    print(f"📊 预测来源 : {source_desc.split(chr(10))[0]}")
    print(f"📌 继续持有 : {len(hold_final)}")
    print(f"📤 卖出     : {len(sell_orders)}")
    print(f"📥 买入     : {len(buy_orders)}")
    if buy_orders:
        total_buy = sum(o['estimated_amount'] for o in buy_orders)
        print(f"💰 预估支出 : {total_buy:,.2f}")
    if sell_orders:
        print(f"💰 预估回收 : {sell_amount:,.2f}")


if __name__ == "__main__":
    main()
