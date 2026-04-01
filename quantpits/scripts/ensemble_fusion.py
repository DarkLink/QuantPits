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

import os
import sys
import json
import argparse
from datetime import datetime

from quantpits.utils import env
os.chdir(env.ROOT_DIR)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
# 已在上方导入并切换目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(SCRIPT_DIR)

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
    """使用 config_loader 加载统一配置"""
    from quantpits.utils.config_loader import load_workspace_config
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            train_records = json.load(f)
    else:
        train_records = {"models": {}, "experiment_name": "unknown"}

    # 使用 config_loader 加载工作区配置 (取代直接读取 model_config.json)
    model_config = load_workspace_config(ROOT_DIR)

    ensemble_config = {}
    if os.path.exists("config/ensemble_config.json"):
        with open("config/ensemble_config.json", "r") as f:
            ensemble_config = json.load(f)

    return train_records, model_config, ensemble_config


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
def load_selected_predictions(train_records, selected_models):
    """
    从 Qlib Recorder 加载选定模型的预测值，归一化后返回宽表。
    """
    from quantpits.utils.predict_utils import load_predictions_from_recorder
    return load_predictions_from_recorder(train_records, selected_models)


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
    print(f"\n{'='*60}")
    print("Stage 2: 相关性分析（仅选定模型）")
    print(f"{'='*60}")

    corr_matrix = norm_df.corr()
    print("\n模型预测相关性矩阵:")
    print(corr_matrix.round(4))

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{combo_name}" if combo_name else ""
    corr_file = os.path.join(output_dir, f"correlation_matrix{suffix}_{anchor_date}.csv")
    corr_matrix.to_csv(corr_file)
    print(f"\n相关性矩阵已保存: {corr_file}")

    # 统计
    n = len(corr_matrix)
    if n > 1:
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper.stack().mean()
        max_corr = upper.stack().max()
        min_corr = upper.stack().min()
        print(f"\n相关性统计: 均值={avg_corr:.4f}, 最大={max_corr:.4f}, 最小={min_corr:.4f}")

    return corr_matrix


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


# ============================================================================
# Stage 5: 保存预测结果
# ============================================================================
def save_predictions(final_score, anchor_date, experiment_name, method,
                     model_names, model_metrics, static_weights, is_dynamic,
                     output_dir, combo_name=None, is_default=False,
                     prediction_dir=None, save_csv=False):
    """
    保存融合预测和配置。
    """
    from quantpits.utils.predict_utils import save_predictions_to_recorder
    ensemble_df = final_score.to_frame('score')

    combo_display = combo_name if combo_name else "ensemble_raw"
    record_id = save_predictions_to_recorder(
        pred=ensemble_df,
        experiment_name="Ensemble_Fusion",
        model_name=combo_display,
        tags={
            "anchor_date": anchor_date,
            "weight_mode": method,
            "combo_name": combo_display
        }
    )
    print(f"\nEnsemble 预测已保存至 Recorder: {record_id} (Experiment: Ensemble_Fusion)")

    # 更新 ensemble_records.json
    records_file = os.path.join(ROOT_DIR, "config", "ensemble_records.json")
    os.makedirs(os.path.dirname(records_file), exist_ok=True)
    
    ensemble_records = {}
    if os.path.exists(records_file):
        with open(records_file, "r") as f:
            ensemble_records = json.load(f)
            
    ensemble_records.setdefault("combos", {})
    ensemble_records["combos"][combo_display] = record_id
    if is_default:
        ensemble_records["default_combo"] = combo_display
        ensemble_records["default_record_id"] = record_id
        
    with open(records_file, "w") as f:
        json.dump(ensemble_records, f, indent=4)

    # 可选保存 CSV
    pred_file = None
    if save_csv:
        pred_dir = prediction_dir or os.path.join("output", "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        if combo_name:
            pred_file = os.path.join(pred_dir, f"ensemble_{combo_name}_{anchor_date}.csv")
        else:
            pred_file = os.path.join(pred_dir, f"ensemble_{anchor_date}.csv")

        ensemble_df.to_csv(pred_file)
        print(f"Ensemble CSV 已额外保存: {pred_file}")
        
        if combo_name and is_default:
            compat_file = os.path.join(pred_dir, f"ensemble_{anchor_date}.csv")
            ensemble_df.to_csv(compat_file)

    # 保存配置
    os.makedirs(output_dir, exist_ok=True)
    config_out = {
        'recorder_id': record_id,
        'anchor_date': anchor_date,
        'experiment_name': experiment_name,
        'weight_mode': method,
        'models_used': model_names,
        'model_metrics': {m: round(v, 4) for m, v in model_metrics.items()},
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    if combo_name:
        config_out['combo_name'] = combo_name
        config_out['is_default'] = is_default
    if not is_dynamic and static_weights:
        config_out['static_weights'] = {m: round(w, 4) for m, w in static_weights.items()}

    suffix = f"_{combo_name}" if combo_name else ""
    config_file = os.path.join(output_dir, f"ensemble_fusion_config{suffix}_{anchor_date}.json")
    with open(config_file, 'w') as f:
        json.dump(config_out, f, indent=4, ensure_ascii=False)
    print(f"Config 已保存: {config_file}")

    return pred_file or record_id


# ============================================================================
# Stage 6: 回测
# ============================================================================
def run_backtest(final_score, top_k, drop_n, benchmark, freq, st_config=None, bt_config=None, verbose=False):
    """运行回测"""
    from quantpits.utils import strategy
    from quantpits.utils.backtest_utils import run_backtest_with_strategy, standard_evaluate_portfolio
    from qlib.backtest.exchange import Exchange

    if st_config is None:
        st_config = strategy.load_strategy_config()
    if bt_config is None:
        bt_config = strategy.get_backtest_config(st_config)

    bt_start = str(final_score.index.get_level_values(0).min().date())
    bt_end = str(final_score.index.get_level_values(0).max().date())

    print(f"\n{'='*60}")
    print("Stage 6: 回测")
    print(f"{'='*60}")
    print(f"Backtest Range: {bt_start} ~ {bt_end}")
    print(f"Freq: {freq}")
    print(f"Verbose: {verbose}")

    strategy_inst = strategy.create_backtest_strategy(final_score, st_config)

    # 准备共享 Exchange (ensemble 原本不需要共享，但接口需要传 trade_exchange)
    all_codes = sorted(final_score.index.get_level_values(1).unique().tolist())
    exchange_kwargs = bt_config["exchange_kwargs"].copy()
    exchange_freq = exchange_kwargs.pop("freq", "day")

    trade_exchange = Exchange(
        freq=exchange_freq,
        start_time=bt_start,
        end_time=bt_end,
        codes=all_codes,
        **exchange_kwargs
    )

    print(f"\n开始回测...")
    report_df, executor_obj = run_backtest_with_strategy(
        strategy_inst=strategy_inst,
        trade_exchange=trade_exchange,
        freq=freq,
        account_cash=bt_config['account'],
        bt_start=bt_start,
        bt_end=bt_end
    )

    if report_df is not None:
        metrics = standard_evaluate_portfolio(report_df, benchmark, freq)
        
        annualized_return = metrics.get('CAGR_252', 0)
        max_drawdown = metrics.get('Max_Drawdown', 0)
        bench_cum_ret = metrics.get('Benchmark_Absolute_Return', 0)
        total_return = metrics.get('Absolute_Return', 0)
        calmar = metrics.get('Calmar', 0)

        initial_cash = bt_config['account']
        final_nav = report_df.iloc[-1]['nav']

        print(f'\n{"="*20} 回测绩效报告 {"="*20}')
        print(f'回测区间     : {bt_start} ~ {bt_end}')
        print(f'初始资金     : {initial_cash:,.2f}')
        print(f'最终净值     : {final_nav:,.2f}')
        print(f'策略累计收益 : {total_return*100:.2f}%')
        print(f'基准累计收益 : {bench_cum_ret*100:.2f}% (超额: {(total_return-bench_cum_ret)*100:.2f}%)')
        print(f'年化收益率   : {annualized_return*100:.2f}%')
        print(f'最大回撤     : {max_drawdown*100:.2f}%')
        if not pd.isna(calmar):
            print(f'Calmar Ratio : {calmar:.4f}')

    else:
        print("【错误】未能提取回测数据")

    return report_df, executor_obj

def run_detailed_backtest_analysis(executor_obj, combo_name, anchor_date, output_dir, freq, benchmark='SH000300'):
    """运行详尽的回测分析报告 (委托给 backtest_report)"""
    from quantpits.utils.backtest_report import run_detailed_backtest_analysis as _run
    return _run(executor_obj, combo_name, anchor_date, output_dir, freq, benchmark)



# ============================================================================
# Stage 7: 风险分析 & 排行榜
# ============================================================================
def calculate_safe_risk(returns, freq):
    """确保输入为 Series，输出为扁平字典"""
    from qlib.contrib.evaluate import risk_analysis

    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0] if not returns.empty else pd.Series(dtype=float)

    try:
        risk = risk_analysis(returns, freq=freq)
        if isinstance(risk, pd.DataFrame):
            risk = risk.iloc[:, 0]
        return risk.to_dict()
    except Exception as e:
        print(f"Risk calculation failed: {e}")
        return {}


def risk_analysis_and_leaderboard(report_df, norm_df, train_records,
                                  loaded_models, freq, output_dir, anchor_date,
                                  combo_name=None):
    """风险分析与排行榜生成"""
    from qlib.workflow import R
    from quantpits.utils.train_utils import get_experiment_name_for_model

    experiment_name = train_records.get('experiment_name', '')
    models = train_records.get('models', {})

    print(f"\n{'='*60}")
    print("Stage 7: 风险分析 & 排行榜")
    print(f"{'='*60}")

    # 1. Ensemble 风险指标
    leaderboard_data = []
    all_reports = {}

    if report_df is not None:
        print(">>> Ensemble 模型风险分析:")
        from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
        from quantpits.utils import strategy
        st_config = strategy.load_strategy_config()
        benchmark = st_config.get('benchmark', 'SH000300')
        
        # Explicitly convert returns to NAV before up-sampling
        da_df = pd.DataFrame(index=report_df.index)
        da_df['收盘价值'] = report_df['account']
        da_df[benchmark] = (1 + report_df['bench']).cumprod()
        if not isinstance(da_df.index, pd.DatetimeIndex):
            da_df.index = pd.to_datetime(da_df.index)
        
        # Consistent Up-sampling for correct duration (252-day basis)
        from qlib.data import D
        bt_start_dt = da_df.index.min()
        bt_end_dt = da_df.index.max()
        daily_dates = D.calendar(start_time=bt_start_dt, end_time=bt_end_dt, freq='day')
        da_df = da_df.reindex(daily_dates, method='ffill').dropna(subset=['收盘价值'])
        da_df.index.name = '成交日期'
        da_df = da_df.reset_index()
        
        pa = PortfolioAnalyzer(
            daily_amount_df=da_df, 
            trade_log_df=pd.DataFrame(), 
            holding_log_df=pd.DataFrame(),
            benchmark_col=benchmark, 
            freq=freq
        )
        metrics = pa.calculate_traditional_metrics()
        
        # We still want the detailed breakdown for Ensemble in the log
        # But we'll use our internal logic for consistency
        r_strat = report_df['return']
        r_bench = report_df['bench']
        r_excess = r_strat - r_bench
        
        risk_strat = pd.Series({
            'mean': r_strat.mean(),
            'std': r_strat.std(),
            'annualized_return': metrics.get('CAGR_252', 0),
            'information_ratio': metrics.get('Information_Ratio_(Arithmetic)', 0),
            'max_drawdown': metrics.get('Max_Drawdown', 0)
        })
        risk_bench = pd.Series({
            'mean': r_bench.mean(),
            'std': r_bench.std(),
            'annualized_return': metrics.get('Benchmark_CAGR_252', 0),
            'information_ratio': metrics.get('Benchmark_Sharpe', 0),
            'max_drawdown': metrics.get('Benchmark_Max_Drawdown', 0)
        })
        risk_excess = pd.Series({
            'mean': r_excess.mean(),
            'std': r_excess.std(),
            'annualized_return': metrics.get('Excess_Return_CAGR_252', 0),
            'information_ratio': metrics.get('Information_Ratio_(Arithmetic)', 0),
            'max_drawdown': (report_df['account']/report_df['account'].cummax() - (1+r_bench).cumprod()/((1+r_bench).cumprod().cummax())).min() # rough proxy
        })
        # Note: excess max drawdown is harder, but the core ones are annualized_return and IR

        ensemble_wide_df = pd.concat(
            [risk_strat, risk_bench, risk_excess],
            axis=1, keys=['account', 'bench', 'excess']
        )
        print(ensemble_wide_df)

        # 添加到排行榜
        metrics_for_lb = risk_strat.to_dict()
        metrics_for_lb['annualized_excess'] = metrics.get('Excess_Return_CAGR_252', 0)
        metrics_for_lb['name'] = 'Ensemble'
        leaderboard_data.append(metrics_for_lb)
        all_reports['Ensemble'] = report_df

    # 2. 子模型排行榜
    print('\n>>> 生成子模型对比排行榜...')
    freq_val = 'week' if freq == 'week' else 'day'
    freq_suffix = '1week' if freq_val == 'week' else '1day'
    report_filename = f"portfolio_analysis/report_normal_{freq_suffix}.pkl"

    # 从当前 combo_norm_df 获取评价区间，以便子模型评估对齐
    eval_start = str(norm_df.index.get_level_values('datetime').min().date())
    eval_end = str(norm_df.index.get_level_values('datetime').max().date())

    for model_name in loaded_models:
        record_id = models.get(model_name)
        if not record_id:
            continue
        try:
            model_exp_name = get_experiment_name_for_model(train_records, model_name)
            recorder = R.get_recorder(recorder_id=record_id, experiment_name=model_exp_name)
            hist_report = recorder.load_object(report_filename)
            
            # 裁剪历史报告到当前的评价区间
            hist_report = hist_report[(hist_report.index >= pd.to_datetime(eval_start)) & (hist_report.index <= pd.to_datetime(eval_end))]
            all_reports[model_name] = hist_report

            if 'return' in hist_report.columns and not hist_report.empty:
                # Up-sample sub-model report for consistent metric calculation
                sub_da_df = pd.DataFrame(index=hist_report.index)
                sub_da_df['收盘价值'] = hist_report['account']
                sub_da_df[benchmark] = (1 + hist_report['bench']).cumprod()
                if not isinstance(sub_da_df.index, pd.DatetimeIndex):
                    sub_da_df.index = pd.to_datetime(sub_da_df.index)
                
                # Use the same daily_dates for alignment
                from qlib.data import D
                sub_bt_start = sub_da_df.index.min()
                sub_bt_end = sub_da_df.index.max()
                sub_daily_dates = D.calendar(start_time=sub_bt_start, end_time=sub_bt_end, freq='day')
                
                sub_da_df = sub_da_df.reindex(sub_daily_dates, method='ffill').dropna(subset=['收盘价值'])
                sub_da_df.index.name = '成交日期'
                sub_da_df = sub_da_df.reset_index()
                
                sub_pa = PortfolioAnalyzer(
                    daily_amount_df=sub_da_df, 
                    trade_log_df=pd.DataFrame(), 
                    holding_log_df=pd.DataFrame(),
                    benchmark_col=benchmark, 
                    freq=freq
                )
                sub_metrics_pa = sub_pa.calculate_traditional_metrics()
                
                metrics = {
                    'name': model_name,
                    'annualized_return': sub_metrics_pa.get('CAGR_252', 0),
                    'annualized_excess': sub_metrics_pa.get('Excess_Return_CAGR_252', 0),
                    'information_ratio': sub_metrics_pa.get('Information_Ratio_(Arithmetic)', 0),
                    'max_drawdown': sub_metrics_pa.get('Max_Drawdown', 0)
                }
                leaderboard_data.append(metrics)
        except Exception as e:
            print(f"  [跳过] {model_name}: {e}")

    # 3. 输出排行榜
    leaderboard_df = None
    if leaderboard_data:
        leaderboard_df = pd.DataFrame(leaderboard_data).set_index('name')
        leaderboard_df = leaderboard_df.apply(pd.to_numeric, errors='coerce')

        sort_col = 'annualized_return' if 'annualized_return' in leaderboard_df.columns else leaderboard_df.columns[0]
        leaderboard_df = leaderboard_df.sort_values(sort_col, ascending=False)

        print(f"\n{'='*10} 绩效对比 {'='*10}")
        display_cols = [c for c in ['annualized_return', 'annualized_excess', 'information_ratio', 'max_drawdown'] if c in leaderboard_df.columns]
        if display_cols:
            print(leaderboard_df[display_cols])
        else:
            print(leaderboard_df)

        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{combo_name}" if combo_name else ""
        lb_file = os.path.join(output_dir, f"leaderboard{suffix}_{anchor_date}.csv")
        leaderboard_df.to_csv(lb_file)
        print(f"\n排行榜已保存: {lb_file}")

    return all_reports, leaderboard_df


# ============================================================================
# Stage 8: 可视化
# ============================================================================
def generate_charts(all_reports, report_df, final_weights, is_dynamic,
                    freq, anchor_date, output_dir, combo_name=None):
    """生成可视化图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print("Stage 8: 可视化")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Part 1: 净值曲线
    if all_reports:
        plt.figure(figsize=(12, 6))
        for name, r_df in all_reports.items():
            if 'return' not in r_df.columns:
                continue
            cum_ret = (1 + r_df['return']).cumprod()
            style = {'color': 'red', 'linewidth': 2.5, 'zorder': 10} if name == 'Ensemble' else {'alpha': 0.4, 'linewidth': 1}
            plt.plot(cum_ret.index, cum_ret.values, label=name, **style)

        if report_df is not None and 'bench' in report_df.columns:
            bench_cum = (1 + report_df['bench']).cumprod()
            plt.plot(bench_cum.index, bench_cum.values, label='Benchmark', color='black', linestyle='--', alpha=0.8)

        plt.title(f'Cumulative Return Comparison ({freq})')
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        suffix = f"_{combo_name}" if combo_name else ""
        chart_file = os.path.join(output_dir, f"ensemble_nav{suffix}_{anchor_date}.png")
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"净值曲线已保存: {chart_file}")

    # Part 2: 动态权重分布图
    if is_dynamic and final_weights is not None:
        fig, ax = plt.subplots(figsize=(14, 6))
        final_weights.plot.area(ax=ax, alpha=0.7, linewidth=0.5)
        ax.set_title('Dynamic Weight Distribution (Rolling Sharpe)', fontsize=14)
        ax.set_ylabel('Weight')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        suffix = f"_{combo_name}" if combo_name else ""
        weight_file = os.path.join(output_dir, f"ensemble_weights{suffix}_{anchor_date}.png")
        plt.savefig(weight_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"权重分布图已保存: {weight_file}")


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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n{'#'*60}")
    print("# 跨组合对比")
    print(f"{'#'*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 对比表
    comparison_data = []
    for result in combo_results:
        row = {
            'combo': result['name'],
            'models': ', '.join(result['models']),
            'method': result['method'],
            'is_default': result['is_default'],
        }
        report_df = result.get('report_df')
        if report_df is not None:
            from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
            from quantpits.utils import strategy
            st_config = strategy.load_strategy_config()
            benchmark = st_config.get('benchmark', 'SH000300')
            
            # Prepare daily_amount_df (Standardize and convert returns to NAV before up-sampling)
            da_df = pd.DataFrame(index=report_df.index)
            da_df['收盘价值'] = report_df['account']
            da_df[benchmark] = (1 + report_df['bench']).cumprod()
            if not isinstance(da_df.index, pd.DatetimeIndex):
                da_df.index = pd.to_datetime(da_df.index)
            da_df.index.name = '成交日期'
            
            # Up-sample to daily frequency to ensure PortfolioAnalyzer's day-counting is correct
            from qlib.data import D
            bt_start_dt = da_df.index.min()
            bt_end_dt = da_df.index.max()
            daily_dates = D.calendar(start_time=bt_start_dt, end_time=bt_end_dt, freq='day')
            da_df = da_df.reindex(daily_dates, method='ffill').dropna(subset=['收盘价值'])
            da_df = da_df.reset_index().rename(columns={'index': '成交日期'})
            
            pa = PortfolioAnalyzer(
                daily_amount_df=da_df, 
                trade_log_df=pd.DataFrame(), 
                holding_log_df=pd.DataFrame(),
                benchmark_col=benchmark, 
                freq=freq
            )
            metrics = pa.calculate_traditional_metrics()
            
            row.update({
                'total_return': round(metrics.get('Absolute_Return', 0) * 100, 2),
                'annualized_return': round(metrics.get('CAGR_252', 0) * 100, 2),
                'annualized_excess': round(metrics.get('Excess_Return_CAGR_252', 0) * 100, 2),
                'max_drawdown': round(metrics.get('Max_Drawdown', 0) * 100, 2),
                'calmar_ratio': round(metrics.get('Calmar', 1.0) if not pd.isna(metrics.get('Calmar')) else 0.0, 4),
                'excess_return': round(metrics.get('Absolute_Return', 0) * 100 - metrics.get('Benchmark_Absolute_Return', 0) * 100, 2),
            })
        comparison_data.append(row)

    comp_df = pd.DataFrame(comparison_data)
    comp_file = os.path.join(output_dir, f"combo_comparison_{anchor_date}.csv")
    comp_df.to_csv(comp_file, index=False)
    print(f"\n对比表已保存: {comp_file}")

    # 显示对比
    print(f"\n{'='*20} 组合对比 {'='*20}")
    display_cols = [c for c in ['combo', 'is_default', 'total_return', 'annualized_return', 'annualized_excess',
                                'max_drawdown', 'calmar_ratio', 'excess_return']
                    if c in comp_df.columns]
    if display_cols:
        print(comp_df[display_cols].to_string(index=False))

    # 2. 净值对比图
    has_reports = [r for r in combo_results if r.get('report_df') is not None]
    if has_reports:
        plt.figure(figsize=(14, 7))
        for result in has_reports:
            report_df = result['report_df']
            if 'return' in report_df.columns:
                cum_ret = (1 + report_df['return']).cumprod()
                label = f"{result['name']}{'  ★' if result['is_default'] else ''}"
                lw = 2.5 if result['is_default'] else 1.2
                plt.plot(cum_ret.index, cum_ret.values, label=label, linewidth=lw)

        # Benchmark
        first_report = has_reports[0]['report_df']
        if 'bench' in first_report.columns:
            bench_cum = (1 + first_report['bench']).cumprod()
            plt.plot(bench_cum.index, bench_cum.values, label='Benchmark',
                     color='black', linestyle='--', alpha=0.7)

        plt.title(f'Combo Comparison - Cumulative Returns ({anchor_date})')
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        chart_file = os.path.join(output_dir, f"combo_comparison_{anchor_date}.png")
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比图已保存: {chart_file}")

    return comp_df


# ============================================================================
# Single Combo Pipeline
# ============================================================================
def run_single_combo(combo_name, selected_models, method, manual_weights_str,
                     norm_df, model_metrics, loaded_models,
                     train_records, model_config, ensemble_config,
                     anchor_date, experiment_name, args,
                     is_default=False):
    """
    对单个 combo 执行完整的 Stage 2-8 流水线。

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
    print(f"\n{'@'*60}")
    if combo_name:
        default_tag = " [DEFAULT]" if is_default else ""
        print(f"@ Combo: {combo_name}{default_tag}")
    print(f"@ 模型: {', '.join(selected_models)}")
    print(f"@ 权重: {method}")
    print(f"{'@'*60}")

    top_k = model_config.get('TopK', 22)
    drop_n = model_config.get('DropN', 3)
    benchmark = model_config.get('benchmark', 'SH000300')

    # 取该 combo 涉及的模型子集
    combo_models = [m for m in selected_models if m in norm_df.columns]
    if not combo_models:
        print(f"Warning: combo {combo_name} 没有有效模型，跳过")
        return None

    combo_norm_df = norm_df[combo_models].dropna(how='any')
    combo_metrics = {m: model_metrics.get(m, 0) for m in combo_models}

    # ---- Stage 2: 相关性分析 ----
    combo_output_dir = args.output_dir
    corr_matrix = correlation_analysis(combo_norm_df, combo_output_dir, anchor_date, combo_name=combo_name)

    # ---- Stage 3: 权重计算 ----
    # 为 combo 构造一个 mini ensemble_config
    combo_ensemble_config = dict(ensemble_config)
    final_weights, static_weights, is_dynamic = calculate_weights(
        combo_norm_df, combo_metrics, method,
        model_config, combo_ensemble_config, manual_weights_str
    )

    # ---- Stage 4: 信号融合 ----
    final_score = generate_ensemble_signal(
        combo_norm_df, final_weights, static_weights, is_dynamic
    )

    # ---- Stage 5: 保存预测 ----
    prediction_dir = getattr(args, 'prediction_dir', None)
    pred_file = save_predictions(
        final_score, anchor_date, experiment_name, method,
        combo_models, combo_metrics, static_weights, is_dynamic,
        combo_output_dir, combo_name=combo_name, is_default=is_default,
        prediction_dir=prediction_dir, save_csv=getattr(args, 'save_csv', False)
    )

    # ---- Stage 6: 回测 ----
    report_df = None
    executor_obj = None
    if not args.no_backtest:
        report_df, executor_obj = run_backtest(
            final_score, top_k, drop_n, benchmark, args.freq, 
            verbose=args.verbose_backtest
        )
        
        if args.detailed_analysis and executor_obj is not None:
            run_detailed_backtest_analysis(
                executor_obj, combo_name, anchor_date, combo_output_dir, args.freq,
                benchmark=benchmark
            )

    # ---- Stage 7: 风险分析 ----
    all_reports = {}
    leaderboard_df = None
    if report_df is not None:
        all_reports, leaderboard_df = risk_analysis_and_leaderboard(
            report_df, combo_norm_df, train_records, combo_models,
            args.freq, combo_output_dir, anchor_date, combo_name=combo_name
        )

    # ---- Stage 8: 可视化 ----
    if not args.no_charts and all_reports:
        generate_charts(
            all_reports, report_df, final_weights, is_dynamic,
            args.freq, anchor_date, combo_output_dir, combo_name=combo_name
        )

    return {
        'name': combo_name or 'default',
        'models': combo_models,
        'method': method,
        'is_default': is_default,
        'pred_file': pred_file,
        'report_df': report_df,
        'leaderboard_df': leaderboard_df,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Fusion - 对选定模型组合进行融合预测、回测和风险分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 等权融合
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158,alstm_Alpha158

  # ICIR 加权
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method icir_weighted

  # 从 ensemble_config.json 读取 default combo
  python quantpits/scripts/ensemble_fusion.py --from-config

  # 运行指定 combo
  python quantpits/scripts/ensemble_fusion.py --combo combo_A

  # 运行所有 combo 并生成对比
  python quantpits/scripts/ensemble_fusion.py --from-config-all

  # 手动权重
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method manual --weights "gru:0.6,linear_Alpha158:0.4"
"""
    )
    parser.add_argument('--models', type=str,
                        help='逗号分隔的模型名列表（直接指定）')
    parser.add_argument('--from-config', action='store_true',
                        help='从 ensemble_config.json 读取 default combo')
    parser.add_argument('--from-config-all', action='store_true',
                        help='运行 ensemble_config.json 中所有 combo')
    parser.add_argument('--combo', type=str,
                        help='运行指定名称的 combo')
    parser.add_argument('--method', type=str, default='equal',
                        choices=['equal', 'icir_weighted', 'manual', 'dynamic'],
                        help='权重模式 (默认 equal，--models 模式下使用)')
    parser.add_argument('--weights', type=str,
                        help='手动权重, 如 "gru:0.6,linear_Alpha158:0.4"')
    parser.add_argument('--freq', type=str, default=None,
                        choices=['day', 'week'],
                        help='回测频率 (默认从 model_config 读取)')
    parser.add_argument('--record-file', type=str, default='latest_train_records.json',
                        help='训练记录文件 (默认 latest_train_records.json)')
    parser.add_argument('--training-mode', type=str, default=None,
                        choices=['static', 'rolling'],
                        help='训练模式过滤 (默认 None=自动解析，static 或 rolling)')
    parser.add_argument('--output-dir', type=str, default='output/ensemble',
                        help='输出目录 (默认 output/ensemble)')
    parser.add_argument('--prediction-dir', type=str, default=None,
                        help='预测 CSV 输出目录 (默认 output/predictions)')
    parser.add_argument('--no-backtest', action='store_true',
                        help='跳过回测')
    parser.add_argument('--no-charts', action='store_true',
                        help='跳过图表生成')
    parser.add_argument('--start-date', type=str, default=None,
                        help='预测数据过滤的开始日期 YYYY-MM-DD (包含该日)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='预测数据过滤的结束日期 YYYY-MM-DD (包含该日)')
    parser.add_argument('--only-last-years', type=int, default=0,
                        help='仅使用最后 N 年的预测数据 (作为 OOS 测试集)')
    parser.add_argument('--only-last-months', type=int, default=0,
                        help='仅使用最后 N 个月的预测数据 (作为 OOS 测试集)')
    parser.add_argument('--detailed-analysis', action='store_true', help='生成详尽的回测分析报告（类似实盘分析）')
    parser.add_argument('--verbose-backtest', action='store_true', help='开启 Qlib 回测的详细日志模式')
    parser.add_argument('--save-csv', action='store_true', help='除了保存为 Qlib Recorder 外，同时输出 predictions csv')

    args = parser.parse_args()

    from quantpits.utils import env
    env.safeguard("Ensemble Fusion")

    # ---- 验证参数 ----
    if not args.models and not args.from_config and not args.from_config_all and not args.combo:
        parser.error("必须指定 --models、--from-config、--from-config-all 或 --combo")

    # ---- Stage 0: 初始化 ----
    print(f"\n{'#'*60}")
    print("# Ensemble Fusion")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    init_qlib()
    train_records, model_config, ensemble_config = load_config(args.record_file)

    # 确定频率
    if args.freq:
        args.freq = args.freq
    else:
        args.freq = model_config.get('freq', 'week')
    print(f"当前交易频率: {args.freq}")

    anchor_date = train_records.get('anchor_date', datetime.now().strftime('%Y-%m-%d'))
    experiment_name = train_records['experiment_name']
    all_models_dict = train_records.get('models', {})
    # 获取 --training-mode（CLI 级别默认模式）
    cli_training_mode = getattr(args, 'training_mode', None)

    from quantpits.utils.train_utils import resolve_model_key

    # ---- 确定要运行的 combo 列表 ----
    combos_to_run = []  # list of (name, models, method, manual_weights_str, is_default)

    if args.models:
        # 直接指定模型列表模式 — 解析 model@mode key
        raw_models = [m.strip() for m in args.models.split(',')]
        selected_models = []
        for m in raw_models:
            full_key = resolve_model_key(m, all_models_dict, cli_training_mode)
            if full_key:
                selected_models.append(full_key)
            else:
                print(f"\nWarning: 模型 '{m}' 不在训练记录中，将被跳过")
        if not selected_models:
            print("Error: 没有有效的模型")
            sys.exit(1)
        combos_to_run.append((None, selected_models, args.method, args.weights, True))

    elif args.combo:
        # 运行指定的 combo
        combos, global_config = parse_ensemble_config(ensemble_config)
        if args.combo not in combos:
            print(f"Error: combo '{args.combo}' 不在 ensemble_config.json 中")
            print(f"可用的 combo: {list(combos.keys())}")
            sys.exit(1)
        cfg = combos[args.combo]
        # combo 级 training_mode > CLI 级
        combo_mode = cfg.get('training_mode', cli_training_mode)
        resolved = []
        for m in cfg['models']:
            full_key = resolve_model_key(m, all_models_dict, combo_mode)
            if full_key:
                resolved.append(full_key)
            else:
                print(f"Warning: combo '{args.combo}' 中模型 '{m}' 不在训练记录中")
        is_def = cfg.get('default', False)
        combos_to_run.append((args.combo, resolved, cfg.get('method', 'equal'),
                              None, is_def))

    elif args.from_config_all:
        # 运行所有 combo
        combos, global_config = parse_ensemble_config(ensemble_config)
        if not combos:
            print("Error: ensemble_config.json 中没有 combos")
            sys.exit(1)
        for name, cfg in combos.items():
            combo_mode = cfg.get('training_mode', cli_training_mode)
            resolved = []
            for m in cfg['models']:
                full_key = resolve_model_key(m, all_models_dict, combo_mode)
                if full_key:
                    resolved.append(full_key)
                else:
                    print(f"Warning: combo '{name}' 中模型 '{m}' 不在训练记录中")
            is_def = cfg.get('default', False)
            combos_to_run.append((name, resolved, cfg.get('method', 'equal'),
                                  None, is_def))
        print(f"多组合模式: 共 {len(combos_to_run)} 个 combo")
        for name, models, method, _, is_def in combos_to_run:
            tag = " [DEFAULT]" if is_def else ""
            print(f"  {name}{tag}: {models} ({method})")

    else:  # args.from_config
        # 读取 default combo
        combos, global_config = parse_ensemble_config(ensemble_config)
        default_name, default_cfg = get_default_combo(combos)
        if not default_cfg:
            print("Error: ensemble_config.json 中没有 default combo")
            sys.exit(1)
        # 如果命令行未指定 method，使用 combo 中配置的
        method = default_cfg.get('method', 'equal')
        if args.method != 'equal':  # 用户显式指定了 method
            method = args.method
        combo_mode = default_cfg.get('training_mode', cli_training_mode)
        resolved = []
        for m in default_cfg['models']:
            full_key = resolve_model_key(m, all_models_dict, combo_mode)
            if full_key:
                resolved.append(full_key)
            else:
                print(f"Warning: default combo 中模型 '{m}' 不在训练记录中")
        combos_to_run.append((default_name, resolved, method,
                              args.weights, True))
        print(f"从 ensemble_config.json 加载 default combo: {default_name}")
        print(f"模型: {resolved}")
        print(f"权重: {method}")

    # ---- 验证所有 combo 的模型（已在上面 resolve 时处理）----
    all_needed_models = set()
    for _, models, _, _, _ in combos_to_run:
        all_needed_models.update(models)

    if not all_needed_models:
        print("Error: 没有有效的模型")
        sys.exit(1)

    print(f"\n所有 combo 涉及的模型并集 ({len(all_needed_models)}): {sorted(all_needed_models)}")

    # ---- Stage 1: 一次性加载所有模型预测 ----
    norm_df, model_metrics, loaded_models = load_selected_predictions(
        train_records, sorted(all_needed_models)
    )

    # ---- 时间窗口过滤 ----
    norm_df = filter_norm_df_by_args(norm_df, args)
    if norm_df.empty:
        print("Error: 过滤后没有预测数据，请检查日期参数。")
        sys.exit(1)

    # ---- 逐 combo 运行 Stage 2-8 ----
    combo_results = []
    for combo_name, models, method, manual_w, is_default in combos_to_run:
        # 过滤掉实际未加载成功的模型
        valid_models = [m for m in models if m in loaded_models]
        if not valid_models:
            print(f"\nWarning: combo {combo_name} 没有有效模型，跳过")
            continue

        result = run_single_combo(
            combo_name=combo_name,
            selected_models=valid_models,
            method=method,
            manual_weights_str=manual_w,
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
        )
        if result:
            combo_results.append(result)

    # ---- 多组合对比 ----
    if len(combo_results) > 1:
        compare_combos(combo_results, anchor_date, args.output_dir, args.freq)

    # ---- 完成 ----
    print(f"\n{'#'*60}")
    print("# 完成!")
    print(f"{'#'*60}")
    for result in combo_results:
        default_tag = " [DEFAULT]" if result['is_default'] else ""
        print(f"组合 {result['name']}{default_tag}: {', '.join(result['models'])}")
        print(f"  权重模式 : {result['method']}")
        print(f"  预测文件 : {result['pred_file']}")
        if result.get('report_df') is not None:
            from quantpits.utils import strategy
            bt_config = strategy.get_backtest_config()
            initial_cash = bt_config['account']
            final_nav = result['report_df'].iloc[-1]['account']
            total_return = (final_nav - initial_cash) / initial_cash
            print(f"  策略收益 : {total_return*100:.2f}%")
    print(f"输出目录   : {args.output_dir}")


if __name__ == "__main__":
    main()
