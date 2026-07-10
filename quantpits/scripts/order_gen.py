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

  # 指定融合组合
  python quantpits/scripts/order_gen.py --combo combo_A

  # 轻量查看执行计划（不初始化 Qlib）
  python quantpits/scripts/order_gen.py --explain-plan

  # 仅预览
  python quantpits/scripts/order_gen.py --dry-run

参数：
  --model            使用单模型预测（从 Qlib 记录加载）
  --output-dir       输出目录 (默认 output)
  --dry-run          执行完整订单计算，但不写入文件
  --verbose          显示详细的排名和价格信息
  --explain-plan     仅打印轻量执行计划，不初始化 Qlib
  --json-plan        输出机器可读 JSON 计划
"""

import os
import sys
import json
from pathlib import Path
from quantpits.utils import env
from quantpits.utils.workspace import WorkspaceContext

import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
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


def _compat_workspace_context(ctx=None):
    if ctx is not None:
        return ctx
    return WorkspaceContext.from_root(ROOT_DIR)


def load_configs(*, ctx=None):
    """使用 config_loader 加载统一配置并读取 cashflow.json"""
    from quantpits.utils.config_loader import load_workspace_config
    workspace = _compat_workspace_context(ctx)
    config = load_workspace_config(workspace.root)

    cashflow_config = {}
    cashflow_file = workspace.config_path("cashflow.json")
    if cashflow_file.exists():
        with cashflow_file.open('r', encoding='utf-8') as f:
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
def load_predictions(model_name=None, anchor_date=None, record_file=None, combo_name=None, *, ctx=None, source=None):
    """
    加载预测数据。

    优先级: model_name > 自动搜索 ensemble 记录
    """
    if source is not None:
        from quantpits.order.execution import load_resolved_prediction

        loaded = load_resolved_prediction(source)
        return loaded.data, loaded.description

    from qlib.workflow import R
    workspace = _compat_workspace_context(ctx)
    
    if model_name:
        from quantpits.utils.train_utils import resolve_model_key, get_experiment_name_for_model
        record_file = record_file or workspace.path("latest_train_records.json")
        if not os.path.exists(record_file):
            fallback_record_file = workspace.config_path("latest_train_records.json")
            if fallback_record_file.exists():
                record_file = fallback_record_file
        if not os.path.exists(record_file):
            raise FileNotFoundError(f"无法找到训练记录文件: {record_file}")
        with open(record_file, 'r') as f:
            records = json.load(f)
        models = records.get("models", {})
        # 支持 model@mode 和裸名
        full_key = resolve_model_key(model_name, models)
        if not full_key:
            raise ValueError(f"模型 {model_name} 的训练记录未找到")
        
        record_id = models[full_key]
        experiment_name = get_experiment_name_for_model(records, full_key)
        recorder = R.get_recorder(recorder_id=record_id, experiment_name=experiment_name)
        pred_df = recorder.load_object("pred.pkl")
        if isinstance(pred_df, pd.Series):
            pred_df = pred_df.to_frame('score')
        return pred_df, f"单模型: {full_key} (Record: {record_id})"

    # 自动搜索 ensemble 记录
    records_file = workspace.config_path("ensemble_records.json")
    if not os.path.exists(records_file):
         raise FileNotFoundError("未找到 ensemble_records.json，请先运行 ensemble_fusion.py")
    
    with open(records_file, 'r') as f:
         ensemble_records = json.load(f)
         
    combo_display = combo_name
    if not combo_display:
         combo_display = ensemble_records.get("default_combo")
    
    combos = ensemble_records.get("combos", {})
    if not combo_display or combo_display not in combos:
         # Fallback to the first available combo
         if not combos:
              raise ValueError("ensemble_records.json 中没有有效的融合记录")
         combo_display = list(combos.keys())[-1]
         
    record_id = combos[combo_display]
    recorder = R.get_recorder(recorder_id=record_id, experiment_name="Ensemble_Fusion")
    pred_df = recorder.load_object("pred.pkl")
    if isinstance(pred_df, pd.Series):
        pred_df = pred_df.to_frame('score')
        
    engine_desc = f"Ensemble 融合: {combo_display} (Record: {record_id})"
    return pred_df, engine_desc


# ============================================================================
# Stage 2: 价格数据获取
# ============================================================================
def get_price_data(anchor_date, market, instruments=None):
    """
    获取当日复权价格和涨跌停估价。

    Args:
        anchor_date: 锚点日期
        market: 市场名称
        instruments: 可选, 直接指定标的列表(覆盖 market 自动获取)

    Returns:
        price_df: DataFrame with columns [current_close, possible_max, possible_min]
    """
    from qlib.data import D
    from qlib.data.ops import Feature

    if instruments is None:
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
def _load_pred_latest_day(df, valid_instruments=None):
    """
    统一过滤预测数据并返回最新一天的 DataFrame（按 score 降序，index=instrument）。

    支持:
      - Qlib Recorder pkl DataFrame

    Args:
        df: DataFrame
        valid_instruments: set, 可选, 仅保留这些标的
    """
    df = df.copy()

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
                            dry_run=False, record_file=None, *, ctx=None):
    """Compatibility wrapper around the pure opinion engine."""
    from quantpits.order.command import OrderRunConfig
    from quantpits.order.opinions import ModelOpinionsRequest, build_model_opinions
    from quantpits.order.persistence import (
        OrderPersistenceRequest,
        build_order_artifact_paths,
        persist_order_artifacts,
    )

    workspace = _compat_workspace_context(ctx)

    def read_json(path):
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}

    train_path = Path(record_file) if record_file else workspace.path("latest_train_records.json")
    if not train_path.is_absolute():
        train_path = workspace.path(train_path.as_posix())
    if not train_path.exists():
        train_path = workspace.config_path("latest_train_records.json")

    run_config = OrderRunConfig(
        merged_config={},
        cashflow_config={},
        strategy_config={},
        ensemble_config=read_json(workspace.config_path("ensemble_config.json")),
        ensemble_records=read_json(workspace.config_path("ensemble_records.json")),
        train_records=read_json(train_path),
    )

    def load_prediction(record_id, experiment_name):
        from qlib.workflow import R

        return R.get_recorder(
            recorder_id=record_id,
            experiment_name=experiment_name,
        ).load_object("pred.pkl")

    result = build_model_opinions(
        ModelOpinionsRequest(
            focus_instruments=tuple(focus_instruments),
            current_holding_instruments=tuple(current_holding_instruments),
            top_k=top_k,
            drop_n=drop_n,
            buy_suggestion_factor=buy_suggestion_factor,
            sorted_predictions=sorted_df,
            trade_date=next_trade_date_string,
            run_config=run_config,
            load_prediction=load_prediction,
        )
    )
    if result is None:
        print("  未找到额外预测文件，跳过多模型判断")
        return None, {}

    paths = build_order_artifact_paths(output_dir, next_trade_date_string, "ensemble")
    if dry_run:
        print(f"  [DRY-RUN] 不写入: {paths.opinion_csv}")
        print(f"  [DRY-RUN] 不写入: {paths.opinion_json}")
    else:
        persist_order_artifacts(
            OrderPersistenceRequest(
                ctx=workspace,
                output_dir=Path(output_dir),
                trade_date=next_trade_date_string,
                source_label="ensemble",
                sell_orders=(),
                buy_orders=(),
                opinions=result,
            )
        )
        print(f"  多模型判断表: {paths.opinion_csv}")
        print(f"  模型信息汇总: {paths.opinion_json}")
    return result.dataframe, result.combo_composition


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
    from quantpits.order.persistence import atomic_write_csv, build_order_artifact_paths

    sell_df = pd.DataFrame(sell_orders)
    buy_df = pd.DataFrame(buy_orders)
    paths = build_order_artifact_paths(output_dir, next_trade_date_string, source_label)
    sell_file = paths.sell_csv.as_posix()
    buy_file = paths.buy_csv.as_posix()

    if dry_run:
        print(f"\n[DRY-RUN] 以下文件不会被写入:")
        print(f"  卖出订单: {sell_file}")
        print(f"  买入订单: {buy_file}")
    else:
        if not sell_df.empty:
            atomic_write_csv(sell_df, paths.sell_csv, index=False)
        if not buy_df.empty:
            atomic_write_csv(buy_df, paths.buy_csv, index=False)
        print(f"\n📁 订单文件已保存:")
        print(f"  卖出订单: {sell_file}")
        print(f"  买入订单: {buy_file}")

    return sell_file, buy_file


def get_next_trade_date(anchor_date):
    """Compatibility wrapper for the typed calendar boundary."""
    from quantpits.order.execution import resolve_next_trade_date

    return resolve_next_trade_date(anchor_date)


def _load_opinion_prediction(record_id, experiment_name):
    from qlib.workflow import R

    return R.get_recorder(recorder_id=record_id, experiment_name=experiment_name).load_object("pred.pkl")


def default_order_execution_hooks():
    """Late-bind script compatibility hooks for the reusable order service."""
    from quantpits.order.execution import LoadedOrderPrediction, OrderExecutionHooks
    from quantpits.order.opinions import build_model_opinions
    from quantpits.order.persistence import persist_order_artifacts
    from quantpits.utils import strategy

    def load_source(source):
        data, description = load_predictions(source=source)
        return LoadedOrderPrediction(data=data, source=source, description=description)

    return OrderExecutionHooks(
        init_qlib=init_qlib,
        get_anchor_date=get_anchor_date,
        get_next_trade_date=get_next_trade_date,
        load_predictions=load_source,
        get_price_data=get_price_data,
        create_order_generator=strategy.create_order_generator,
        get_strategy_params=strategy.get_strategy_params,
        build_model_opinions=build_model_opinions,
        persist_artifacts=persist_order_artifacts,
    )


def run_order_generation(prepared):
    """Run order generation through the reusable service boundary."""
    from quantpits.order.service import OrderGenerationService

    return OrderGenerationService(default_order_execution_hooks()).execute(prepared)


def _default_order_run_config(ctx, options):
    from quantpits.order.command import load_order_run_config
    from quantpits.utils import strategy

    merged_config, cashflow_config = load_configs(ctx=ctx)
    strategy_config = strategy.load_strategy_config(ctx.root)
    return load_order_run_config(
        ctx,
        options,
        merged_config=merged_config,
        cashflow_config=cashflow_config,
        strategy_config=strategy_config,
    )


def default_order_command_dependencies():
    from quantpits.order.command import OrderCommandDependencies

    return OrderCommandDependencies(
        get_workspace_context=env.get_workspace_context,
        load_run_config=_default_order_run_config,
        safeguard=env.safeguard,
        execute=run_order_generation,
    )


def main(argv=None):
    from quantpits.order.command import (
        OrderCommandRequest,
        OrderPlanError,
        build_order_arg_parser,
        run_order_command,
    )
    from quantpits.order.execution import OrderExecutionError

    parser = build_order_arg_parser()
    cli_args = tuple(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    try:
        outcome = run_order_command(
            OrderCommandRequest(args=args, cli_args=cli_args),
            default_order_command_dependencies(),
        )
    except OrderPlanError as exc:
        parser.error(str(exc))
        return
    except OrderExecutionError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    if outcome.rendered_output is not None:
        print(outcome.rendered_output)


if __name__ == "__main__":
    main()
