#!/usr/bin/env python
"""
Ensemble Fusion - 对用户选定的模型组合进行融合预测、回测和风险分析

工作流位置：训练 → 暴力穷举 → 手动选组合 → **融合回测（本脚本）** → 订单生成

支持多组合模式：ensemble_config.json 中可定义多个 combo，标记一个 default。

运行方式：
  cd QuantPits

  # 等权融合（最常用）
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158,alstm_Alpha158

  # ICIR 加权
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method icir_weighted

  # 手动权重
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method manual --weights "gru:0.6,linear_Alpha158:0.4"

  # 动态权重（滚动 TopK Sharpe）
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method dynamic

  # 从 ensemble_config.json 读取 default combo
  python quantpits/scripts/ensemble_fusion.py --from-config

  # 运行指定的 combo
  python quantpits/scripts/ensemble_fusion.py --combo combo_A

  # 运行所有 combo 并生成跨组合对比
  python quantpits/scripts/ensemble_fusion.py --from-config-all

  # 跳过回测
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --no-backtest

参数：
  --models            逗号分隔的模型名列表（直接指定，优先级最高）
  --from-config       从 ensemble_config.json 读取 default combo
  --from-config-all   运行 ensemble_config.json 中所有 combo
  --combo             运行指定名称的 combo
  --method            权重模式: equal / icir_weighted / manual / dynamic (默认 equal)
  --weights           手动权重, 如 "gru:0.6,linear_Alpha158:0.4"
  --freq              回测频率: day / week (默认 week)
  --record-file       训练记录文件 (默认 latest_train_records.json)
  --output-dir        输出目录 (默认 output/ensemble)
  --no-backtest       跳过回测
  --no-charts         跳过图表生成
  --detailed-analysis 生成详尽的回测分析报告（类似实盘分析）
  --verbose-backtest  开启 Qlib 回测的详细模式
"""

import sys
from pathlib import Path

from quantpits.utils import env

import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
ROOT_DIR = env.ROOT_DIR  # Backward compatibility only; do not use for writes.

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# BACKTEST_CONFIG has been migrated to strategy provider `config/strategy_config.yaml`


# ============================================================================
# Stage 0: 初始化 & 配置加载
# ============================================================================
def init_qlib():
    """初始化 Qlib（委托给 env.init_qlib）"""
    env.init_qlib()


def load_config(record_file="latest_train_records.json"):
    """使用显式 WorkspaceContext 加载 ensemble 运行配置。"""
    from quantpits.ensemble.config import load_ensemble_run_config

    run_config = load_ensemble_run_config(env.get_workspace_context(), record_file=record_file)
    return run_config.train_records, run_config.model_config, run_config.ensemble_config


def parse_ensemble_config(ensemble_config):
    """解析 ensemble_config.json，兼容新旧格式。(委托给 ensemble_utils)"""
    from quantpits.utils.ensemble_utils import parse_ensemble_config as _parse
    return _parse(ensemble_config)


def get_default_combo(combos):
    """返回 default combo 的 (name, config)。(委托给 ensemble_utils)"""
    from quantpits.utils.ensemble_utils import get_default_combo as _get
    return _get(combos)


# ============================================================================
# Stage 1: 加载预测数据
# ============================================================================
def load_selected_predictions(train_records, selected_models, norm_method="rank"):
    """
    从 Qlib Recorder 加载选定模型的预测值，归一化后返回宽表。
    """
    from quantpits.utils.predict_utils import load_predictions_from_recorder
    return load_predictions_from_recorder(train_records, selected_models, norm_method=norm_method)


def filter_norm_df_by_args(norm_df, args):
    """根据参数截取 norm_df 的时间窗口"""
    dates = norm_df.index.get_level_values("datetime").unique().sort_values()
    max_date = dates.max()
    min_date = dates.min()
    
    start_date = pd.to_datetime(args.start_date) if args.start_date else min_date
    end_date = pd.to_datetime(args.end_date) if args.end_date else max_date
    
    # 如果指定了 --only-last-years / --only-last-months，覆盖 start_date
    # brute force 的 OOS 定义是 strictly > (max_date - offset)，这里保持一致
    if getattr(args, "only_last_years", 0) > 0 or getattr(args, "only_last_months", 0) > 0:
        cutoff_date = max_date
        if args.only_last_years > 0:
            cutoff_date -= pd.DateOffset(years=args.only_last_years)
        if args.only_last_months > 0:
            cutoff_date -= pd.DateOffset(months=args.only_last_months)
        
        mask = (norm_df.index.get_level_values("datetime") > cutoff_date) & \
               (norm_df.index.get_level_values("datetime") <= end_date)
    else:
        mask = (norm_df.index.get_level_values("datetime") >= start_date) & \
               (norm_df.index.get_level_values("datetime") <= end_date)
               
    filtered_df = norm_df[mask]
    
    print(f"\n=== 时间窗口过滤 ===")
    if not filtered_df.empty:
        actual_start = filtered_df.index.get_level_values('datetime').min().date()
        actual_end = filtered_df.index.get_level_values('datetime').max().date()
        days_count = len(filtered_df.index.get_level_values('datetime').unique())
        print(f"数据范围  : {actual_start} ~ {actual_end} (共 {days_count} 天交易日)")
    else:
        print("数据范围  : 无数据")
        
    return filtered_df


# ============================================================================
# Stage 2: 相关性分析
# ============================================================================
def correlation_analysis(norm_df, output_dir, anchor_date, combo_name=None):
    """计算并保存选定模型的预测值相关性矩阵"""
    from quantpits.ensemble.analytics import (
        CorrelationAnalysisRequest,
        run_correlation_analysis,
    )

    result = run_correlation_analysis(
        CorrelationAnalysisRequest(
            norm_df=norm_df,
            output_dir=output_dir,
            anchor_date=anchor_date,
            combo_name=combo_name,
        )
    )
    return result.matrix


# ============================================================================
# Stage 3: 权重计算
# ============================================================================
def calculate_weights(norm_df, model_metrics, method, model_config,
                      ensemble_config, manual_weights_str=None):
    """计算各模型权重 (委托给 fusion_engine)"""
    from quantpits.utils.fusion_engine import calculate_weights as _calc
    return _calc(norm_df, model_metrics, method, model_config,
                 ensemble_config, manual_weights_str)


# ============================================================================
# Stage 4: 信号融合
# ============================================================================
def generate_ensemble_signal(norm_df, final_weights, static_weights, is_dynamic):
    """生成融合信号 (委托给 fusion_engine)"""
    from quantpits.utils.fusion_engine import generate_ensemble_signal as _gen
    return _gen(norm_df, final_weights, static_weights, is_dynamic)


def calculate_loo_contribution(norm_df, final_score):
    """
    计算 Leave-One-Out (LOO) 贡献度。
    
    使用 IC(LOO_score, final_score) 作为代理指标，衡量剔除该模型后，
    融合信号与原始融合信号的相关性变化。相关性下降越多（delta 越大），
    说明该模型对最终信号的贡献越大。
    
    Returns:
        dict: {model_name: {"loo_ic": float, "full_ic": float, "delta": float}}
    """
    from quantpits.ensemble.analytics import calculate_loo_contribution as _calculate

    return _calculate(norm_df, final_score)


def save_model_contribution_snapshot(*, output_dir, anchor_date, combo_name, contributions):
    """保存 LOO 模型贡献度快照。"""
    from quantpits.ensemble.analytics import save_model_contribution_snapshot as _save

    return _save(
        output_dir=output_dir,
        anchor_date=anchor_date,
        combo_name=combo_name,
        contributions=contributions,
    )


# ============================================================================
# Fusion Run Ledger
# ============================================================================
def append_to_fusion_ledger(
    workspace_root: str,
    run_date: str,
    combo_name: str,
    models: list,
    method: str,
    is_default: bool,
    eval_window: dict,
    metrics: dict,
    sub_model_metrics: dict = None,
    loo_contributions: dict = None,
    cli_args: list = None,
):
    """
    将本次 ensemble_fusion.py 的回测结果追加写入 data/fusion_run_ledger.jsonl。

    每条记录代表一次手动/配置驱动的融合回测快照，供 RLFF EnsembleEvolutionAgent
    追踪组合性能趋势和 IS/OOS 衰减，无需依赖 brute_force 搜索的 run_metadata.json。

    Args:
        workspace_root: 工作区根目录
        run_date:       执行日期 (YYYY-MM-DD)
        combo_name:     组合名称（None 时记为 'default'）
        models:         模型名称列表
        method:         权重模式 ('equal'/'icir_weighted'/...)
        is_default:     是否为 default combo
        eval_window:    评估窗口配置 {
                            'start': str, 'end': str,
                            'only_last_years': int, 'only_last_months': int
                        }
        metrics:        Ensemble 绩效指标字典（annualized_return/excess/max_drawdown/calmar/...）
        sub_model_metrics: 子模型绩效 {model_name: {annualized_return, annualized_excess, ...}}
        loo_contributions: LOO 贡献度 {model_name: {delta, loo_ic, full_ic}}
        cli_args:       sys.argv[1:]
    """
    from quantpits.ensemble.ledger import FusionLedgerEntry, append_fusion_ledger

    append_fusion_ledger(
        workspace_root,
        FusionLedgerEntry(
            run_date=run_date,
            combo_name=combo_name,
            models=tuple(models),
            method=method,
            is_default=is_default,
            eval_window=eval_window,
            metrics=metrics,
            sub_model_metrics=sub_model_metrics or {},
            loo_contributions=loo_contributions or {},
            cli_args=tuple(cli_args or ()),
        ),
    )


# ============================================================================
# Stage 5: 保存预测结果
# ============================================================================
def _workspace_root_path(workspace_root=None) -> Path:
    from quantpits.ensemble.persistence import workspace_root_path

    return workspace_root_path(workspace_root)


def _workspace_bound_path(workspace_root: Path, path_value) -> Path:
    from quantpits.ensemble.persistence import workspace_bound_path

    return workspace_bound_path(workspace_root, path_value)


def save_predictions(final_score, anchor_date, experiment_name, method,
                     model_names, model_metrics, static_weights, is_dynamic,
                     output_dir, combo_name=None, is_default=False,
                     prediction_dir=None, save_csv=False, workspace_root=None):
    """
    保存融合预测和配置。
    """
    from quantpits.ensemble.persistence import (
        PredictionSaveRequest,
        save_ensemble_predictions,
    )

    result = save_ensemble_predictions(
        PredictionSaveRequest(
            final_score=final_score,
            anchor_date=anchor_date,
            experiment_name=experiment_name,
            method=method,
            model_names=tuple(model_names),
            model_metrics=model_metrics,
            static_weights=static_weights,
            is_dynamic=is_dynamic,
            output_dir=output_dir,
            combo_name=combo_name,
            is_default=is_default,
            prediction_dir=prediction_dir,
            save_csv=save_csv,
            workspace_root=workspace_root,
        )
    )
    return result.returned_ref


# ============================================================================
# Stage 6: 回测
# ============================================================================
def run_backtest(final_score, top_k, drop_n, benchmark, freq, st_config=None, bt_config=None, verbose=False):
    """运行回测"""
    from quantpits.ensemble.backtest import run_backtest as _run_backtest

    return _run_backtest(
        final_score,
        top_k,
        drop_n,
        benchmark,
        freq,
        st_config=st_config,
        bt_config=bt_config,
        verbose=verbose,
    )

def run_detailed_backtest_analysis(executor_obj, combo_name, anchor_date, output_dir, freq, benchmark='SH000300'):
    """运行详尽的回测分析报告 (委托给 backtest_report)"""
    from quantpits.ensemble.backtest import run_detailed_backtest_analysis as _run

    return _run(executor_obj, combo_name, anchor_date, output_dir, freq, benchmark=benchmark)



# ============================================================================
# Stage 7: 风险分析 & 排行榜
# ============================================================================
def calculate_safe_risk(returns, freq):
    """确保输入为 Series，输出为扁平字典。"""
    from quantpits.ensemble.risk_report import calculate_safe_risk as _calculate_safe_risk

    return _calculate_safe_risk(returns, freq)


def risk_analysis_and_leaderboard(report_df, norm_df, train_records,
                                  loaded_models, freq, output_dir, anchor_date,
                                  combo_name=None):
    """风险分析与排行榜生成"""
    from quantpits.ensemble.risk_report import (
        risk_analysis_and_leaderboard as _risk_analysis_and_leaderboard,
    )

    return _risk_analysis_and_leaderboard(
        report_df,
        norm_df,
        train_records,
        loaded_models,
        freq,
        output_dir,
        anchor_date,
        combo_name=combo_name,
    )


# ============================================================================
# Stage 8: 可视化
# ============================================================================
def generate_charts(all_reports, report_df, final_weights, is_dynamic,
                    freq, anchor_date, output_dir, combo_name=None):
    """生成可视化图表"""
    from quantpits.ensemble.charts import generate_charts as _generate_charts

    return _generate_charts(
        all_reports,
        report_df,
        final_weights,
        is_dynamic,
        freq,
        anchor_date,
        output_dir,
        combo_name=combo_name,
    )


# ============================================================================
# Combo Comparison (multi-combo mode)
# ============================================================================
def compare_combos(combo_results, anchor_date, output_dir, freq):
    """
    生成跨组合对比汇总。

    Args:
        combo_results: list of dict, 每个 combo 的结果
            [{"name": str, "models": list, "method": str, "is_default": bool,
              "pred_file": str, "report_df": DataFrame or None, ...}]
        anchor_date: 锚点日期
        output_dir: 输出目录
        freq: 回测频率 (day/week)
    """
    from quantpits.ensemble.comparison import compare_combos as _compare_combos

    return _compare_combos(combo_results, anchor_date, output_dir, freq)


# ============================================================================
# Single Combo Pipeline
# ============================================================================
def _single_combo_pipeline_hooks():
    from quantpits.ensemble.pipeline import SingleComboPipelineHooks

    return SingleComboPipelineHooks(
        correlation_analysis=correlation_analysis,
        calculate_weights=calculate_weights,
        generate_ensemble_signal=generate_ensemble_signal,
        save_predictions=save_predictions,
        run_backtest=run_backtest,
        run_detailed_backtest_analysis=run_detailed_backtest_analysis,
        risk_analysis_and_leaderboard=risk_analysis_and_leaderboard,
        generate_charts=generate_charts,
        calculate_loo_contribution=calculate_loo_contribution,
        save_model_contribution_snapshot=save_model_contribution_snapshot,
        append_to_fusion_ledger=append_to_fusion_ledger,
        get_workspace_root=lambda: env.get_workspace_context().root.as_posix(),
    )


def run_single_combo(combo_name, selected_models, method, manual_weights_str,
                     norm_df, model_metrics, loaded_models,
                     train_records, model_config, ensemble_config,
                     anchor_date, experiment_name, args,
                     is_default=False):
    """
    对单个 combo 执行完整的 Stage 2-10 流水线。

    Args:
        combo_name: 组合名称（None 表示 --models 直接指定模式）
        selected_models: 该 combo 的模型列表
        method: 权重模式
        manual_weights_str: 手动权重字符串
        norm_df: 全部模型的归一化预测宽表
        model_metrics: 模型 ICIR 指标
        loaded_models: 已加载的模型列表
        is_default: 是否为 default combo

    Returns:
        dict with combo result info
    """
    from quantpits.ensemble.pipeline import (
        SingleComboPipelineRequest,
        run_single_combo_pipeline,
    )

    result = run_single_combo_pipeline(
        SingleComboPipelineRequest(
            combo_name=combo_name,
            selected_models=selected_models,
            method=method,
            manual_weights_str=manual_weights_str,
            norm_df=norm_df,
            model_metrics=model_metrics,
            loaded_models=loaded_models,
            train_records=train_records,
            model_config=model_config,
            ensemble_config=ensemble_config,
            anchor_date=anchor_date,
            experiment_name=experiment_name,
            args=args,
            is_default=is_default,
        ),
        hooks=_single_combo_pipeline_hooks(),
    )
    return None if result is None else result.to_legacy_dict()


def build_arg_parser():
    from quantpits.ensemble.command import build_ensemble_arg_parser

    return build_ensemble_arg_parser()


def default_ensemble_hooks():
    from quantpits.ensemble import EnsembleExecutionHooks

    return EnsembleExecutionHooks(
        init_qlib=init_qlib,
        load_selected_predictions=load_selected_predictions,
        filter_norm_df_by_args=filter_norm_df_by_args,
        run_single_combo=run_single_combo,
        compare_combos=compare_combos,
    )


def default_ensemble_command_dependencies():
    """Build late-bound command dependencies for legacy patch compatibility."""

    from quantpits.ensemble import EnsembleFusionService, EnsembleRunConfig
    from quantpits.ensemble.command import EnsembleCommandDependencies

    def load_run_config(ctx, record_file):
        del ctx  # load_config resolves the active workspace through the compatibility facade.
        train_records, model_config, ensemble_config = load_config(record_file)
        return EnsembleRunConfig(train_records, model_config, ensemble_config)

    return EnsembleCommandDependencies(
        get_workspace_context=env.get_workspace_context,
        load_run_config=load_run_config,
        safeguard=env.safeguard,
        service_factory=lambda: EnsembleFusionService(default_ensemble_hooks()),
    )


# ============================================================================
# Main
# ============================================================================
def main(argv=None):
    parser = build_arg_parser()
    cli_args = tuple(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    from quantpits.ensemble.command import (
        EnsembleCommandRequest,
        EnsembleCommandUsageError,
        run_ensemble_command,
    )
    from quantpits.ensemble.execution import EnsembleExecutionError
    from quantpits.utils.ensemble_plan import EnsemblePlanError

    try:
        outcome = run_ensemble_command(
            EnsembleCommandRequest(args=args, cli_args=cli_args),
            default_ensemble_command_dependencies(),
        )
    except (EnsembleCommandUsageError, EnsemblePlanError) as exc:
        parser.error(str(exc))
    except EnsembleExecutionError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    if outcome.rendered_output is not None:
        print(outcome.rendered_output)


if __name__ == "__main__":
    main()
