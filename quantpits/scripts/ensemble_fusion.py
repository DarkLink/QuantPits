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
"""

import os
import sys
import json
import argparse
from datetime import datetime

import env
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
BACKTEST_CONFIG = {
    "account": 100_000_000,
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
}


# ============================================================================
# Stage 0: 初始化 & 配置加载
# ============================================================================
def init_qlib():
    """初始化 Qlib"""
    import qlib
    from qlib.constant import REG_CN
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)


def load_config(record_file="latest_train_records.json"):
    """加载训练记录、模型配置和 ensemble 配置"""
    with open(record_file, "r") as f:
        train_records = json.load(f)

    with open("config/model_config.json", "r") as f:
        model_config = json.load(f)

    ensemble_config = {}
    if os.path.exists("config/ensemble_config.json"):
        with open("config/ensemble_config.json", "r") as f:
            ensemble_config = json.load(f)

    return train_records, model_config, ensemble_config


def parse_ensemble_config(ensemble_config):
    """
    解析 ensemble_config.json，兼容新旧格式。

    新格式 (多组合):
        {"combos": {"combo_A": {"models": [...], "method": "equal", "default": true}, ...}}

    旧格式 (单组合):
        {"models": [...], "ensemble_method": "manual", "manual_weights": {...}}

    Returns:
        combos: dict, combo_name -> {"models": [], "method": str, "default": bool, ...}
        global_config: dict, min_model_ic 等全局配置
    """
    if 'combos' in ensemble_config:
        # 新格式
        combos = ensemble_config['combos']
        global_config = {k: v for k, v in ensemble_config.items() if k != 'combos'}
        return combos, global_config
    elif 'models' in ensemble_config:
        # 旧格式 → 自动迁移为 legacy combo
        legacy_combo = {
            'models': ensemble_config['models'],
            'method': ensemble_config.get('ensemble_method', 'equal'),
            'default': True,
            'description': '从旧格式自动迁移',
        }
        if 'manual_weights' in ensemble_config:
            legacy_combo['manual_weights'] = ensemble_config['manual_weights']
        combos = {'legacy': legacy_combo}
        global_config = {k: v for k, v in ensemble_config.items()
                         if k not in ('models', 'ensemble_method', 'manual_weights', 'use_ensemble')}
        return combos, global_config
    else:
        return {}, {}


def get_default_combo(combos):
    """返回 default combo 的 (name, config)，如果没有则返回第一个"""
    for name, cfg in combos.items():
        if cfg.get('default', False):
            return name, cfg
    # 没有标记 default 的，取第一个
    if combos:
        first_name = next(iter(combos))
        return first_name, combos[first_name]
    return None, None


# ============================================================================
# Stage 1: 加载预测数据
# ============================================================================
def zscore_norm(series):
    """按天 Z-Score 归一化 (减均值，除标准差)"""
    def _norm_func(x):
        std = x.std()
        if std == 0:
            return x - x.mean()
        return (x - x.mean()) / std
    return series.groupby(level='datetime', group_keys=False).apply(_norm_func)


def load_selected_predictions(train_records, selected_models):
    """
    从 Qlib Recorder 加载选定模型的预测值，归一化后返回宽表。

    Args:
        train_records: 训练记录字典
        selected_models: 要加载的模型名列表

    Returns:
        norm_df: DataFrame, index=(datetime, instrument), columns=model_names
        model_metrics: dict, model_name -> ICIR
    """
    from qlib.workflow import R

    experiment_name = train_records["experiment_name"]
    models = train_records["models"]

    all_preds = []
    model_metrics = {}
    loaded_models = []

    print(f"\n{'='*60}")
    print("Stage 1: 加载选定模型预测数据")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Selected Models ({len(selected_models)}): {selected_models}")

    for model_name in selected_models:
        if model_name not in models:
            print(f"  [{model_name}] SKIPPED: 不在训练记录中")
            continue

        record_id = models[model_name]
        try:
            recorder = R.get_recorder(
                recorder_id=record_id, experiment_name=experiment_name
            )

            # 加载预测值
            pred = recorder.load_object("pred.pkl")
            if isinstance(pred, pd.Series):
                pred = pred.to_frame("score")
            pred.columns = [model_name]
            all_preds.append(pred)
            loaded_models.append(model_name)

            # 读取 ICIR 指标
            raw_metrics = recorder.list_metrics()
            metric_val = 0.0
            for k, v in raw_metrics.items():
                if "ICIR" in k:
                    metric_val = v
                    break
            model_metrics[model_name] = metric_val

            print(f"  [{model_name}] OK. Preds={len(pred)}, ICIR={metric_val:.4f}")
        except Exception as e:
            print(f"  [{model_name}] FAILED: {e}")

    print(f"\n成功加载 {len(all_preds)}/{len(selected_models)} 个模型")

    if not all_preds:
        raise ValueError("未加载到任何预测数据！")

    # 合并 & Z-Score 归一化
    merged_df = pd.concat(all_preds, axis=1).dropna()
    print(f"合并后数据维度: {merged_df.shape}")

    norm_df = pd.DataFrame(index=merged_df.index)
    for col in merged_df.columns:
        norm_df[col] = zscore_norm(merged_df[col])

    return norm_df, model_metrics, loaded_models


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
    """
    计算各模型权重。

    Args:
        norm_df: 归一化后的预测宽表
        model_metrics: {model_name: ICIR}
        method: 权重模式
        model_config: 模型配置
        ensemble_config: ensemble 配置
        manual_weights_str: 手动权重字符串 "model1:w1,model2:w2"

    Returns:
        final_weights: 动态权重 DataFrame (dynamic 模式) 或 None
        static_weights: 静态权重 dict (非 dynamic 模式) 或 None
        is_dynamic: bool
    """
    model_names = list(norm_df.columns)

    top_k = model_config.get('TopK', 22)
    min_ic = ensemble_config.get('min_model_ic', 0.01)

    print(f"\n{'='*60}")
    print(f"Stage 3: 权重计算 (Mode: {method})")
    print(f"{'='*60}")

    if method == 'dynamic':
        # ---- 滚动 TopK Sharpe 动态权重 ----
        from qlib.data import D

        ROLLING_WINDOW = 60
        MIN_SHARPE_THRESHOLD = 0.0
        EVAL_TOP_K = top_k
        LABEL_FIELD = ['Ref($close, -6)/$close - 1']

        print(f">>> 使用动态权重 (Rolling TopK Sharpe, Window={ROLLING_WINDOW})")

        # 加载真实 Label
        instruments = norm_df.index.get_level_values('instrument').unique().tolist()
        start_date = norm_df.index.get_level_values('datetime').min()
        end_date = norm_df.index.get_level_values('datetime').max()
        label_df = D.features(instruments, LABEL_FIELD, start_time=start_date, end_time=end_date)
        label_df.columns = ['label']
        eval_df = norm_df.join(label_df, how='inner')

        # 计算每个模型每天的 TopK 平均收益
        dates = eval_df.index.get_level_values('datetime').unique().sort_values()
        perf_dict = {m: [] for m in model_names}

        for date in dates:
            if date not in eval_df.index:
                for m in model_names:
                    perf_dict[m].append(0)
                continue
            day_data = eval_df.loc[date]
            for model in model_names:
                top_stocks = day_data.nlargest(EVAL_TOP_K, model)
                perf_dict[model].append(top_stocks['label'].mean())

        daily_topk_ret = pd.DataFrame(perf_dict, index=dates)

        # 滚动 Sharpe
        rolling_mean = daily_topk_ret.rolling(window=ROLLING_WINDOW, min_periods=20).mean()
        rolling_std = daily_topk_ret.rolling(window=ROLLING_WINDOW, min_periods=20).std()
        rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)).fillna(0)

        # 熔断 + 归一化
        raw_weights = rolling_sharpe.copy()
        raw_weights[raw_weights < MIN_SHARPE_THRESHOLD] = 0
        weight_sum = raw_weights.sum(axis=1)
        equal_w = pd.DataFrame(1.0 / len(model_names), index=raw_weights.index, columns=raw_weights.columns)
        final_weights = raw_weights.div(weight_sum, axis=0).fillna(equal_w)
        final_weights = final_weights.shift(1).fillna(1.0 / len(model_names))

        print('\n平均权重分布:')
        for m, w in final_weights.mean().sort_values(ascending=False).items():
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)")

        return final_weights, None, True

    elif method == 'icir_weighted':
        # ---- 静态 ICIR 加权 ----
        print(f">>> 使用 ICIR 加权 (min_ic={min_ic})")
        valid = {m: max(0, v) for m, v in model_metrics.items() if m in model_names and v > min_ic}
        if not valid:
            print("Warning: 无有效 ICIR，使用等权")
            static_weights = {m: 1.0 / len(model_names) for m in model_names}
        else:
            total = sum(valid.values())
            static_weights = {m: valid.get(m, 0) / total if total > 0 else 1.0 / len(valid) for m in model_names}

        for m, w in sorted(static_weights.items(), key=lambda x: -x[1]):
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)  ICIR={model_metrics.get(m, 0):.4f}")

        return None, static_weights, False

    elif method == 'manual':
        # ---- 手动权重 ----
        print(">>> 使用手动权重")
        if manual_weights_str:
            manual_w = {}
            for item in manual_weights_str.split(','):
                parts = item.strip().split(':')
                if len(parts) == 2:
                    manual_w[parts[0].strip()] = float(parts[1].strip())
        else:
            manual_w = ensemble_config.get('manual_weights', {})

        total = sum(manual_w.get(m, 0) for m in model_names)
        if total == 0:
            print("Warning: 手动权重总和为 0，使用等权")
            static_weights = {m: 1.0 / len(model_names) for m in model_names}
        else:
            static_weights = {m: manual_w.get(m, 0) / total for m in model_names}

        for m, w in static_weights.items():
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)")

        return None, static_weights, False

    else:  # equal
        # ---- 等权 ----
        print(f">>> 使用等权 ({len(model_names)} 个模型)")
        static_weights = {m: 1.0 / len(model_names) for m in model_names}

        for m, w in static_weights.items():
            icir = model_metrics.get(m, 0)
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)  ICIR={icir:.4f}")

        return None, static_weights, False


# ============================================================================
# Stage 4: 信号融合
# ============================================================================
def generate_ensemble_signal(norm_df, final_weights, static_weights, is_dynamic):
    """生成融合信号"""
    model_names = list(norm_df.columns)

    print(f"\n{'='*60}")
    print("Stage 4: 生成 Ensemble 融合信号")
    print(f"{'='*60}")

    final_score = pd.Series(0.0, index=norm_df.index, name='score')

    if is_dynamic:
        for model in model_names:
            w = final_weights[model]
            weighted_pred = norm_df[model].mul(w, level='datetime')
            final_score += weighted_pred
    else:
        for model in model_names:
            w = static_weights.get(model, 0)
            if w > 0:
                final_score += norm_df[model] * w

    # 统计检查
    print(f"\n=== Ensemble Signal 统计 ===")
    print(f"数据量: {len(final_score)}")
    print(f"Min: {final_score.min():.4f}, Max: {final_score.max():.4f}, Mean: {final_score.mean():.4f}")
    print(f"Std: {final_score.std():.4f}")

    if final_score.std() == 0:
        print("!!! 警告：最终结果标准差为0，加权可能失败 !!!")
    else:
        print("加权成功。")

    return final_score


# ============================================================================
# Stage 5: 保存预测结果
# ============================================================================
def save_predictions(final_score, anchor_date, experiment_name, method,
                     model_names, model_metrics, static_weights, is_dynamic,
                     output_dir, combo_name=None, is_default=False):
    """
    保存融合预测和配置。

    Args:
        combo_name: 组合名称（多组合模式下使用）
        is_default: 是否为 default combo（额外保存不带 combo_name 的兼容文件）
    """
    # 保存预测
    os.makedirs("output/predictions", exist_ok=True)
    ensemble_df = final_score.to_frame('score')

    # 文件命名：带 combo_name 或不带
    if combo_name:
        pred_file = f"output/predictions/ensemble_{combo_name}_{anchor_date}.csv"
    else:
        pred_file = f"output/predictions/ensemble_{anchor_date}.csv"

    ensemble_df.to_csv(pred_file)
    print(f"\nEnsemble 预测已保存: {pred_file}")
    print(f"Total: {len(ensemble_df)} records")

    # default combo 额外保存一份兼容文件
    if combo_name and is_default:
        compat_file = f"output/predictions/ensemble_{anchor_date}.csv"
        ensemble_df.to_csv(compat_file)
        print(f"Default 兼容文件: {compat_file}")

    # 保存配置
    os.makedirs(output_dir, exist_ok=True)
    config_out = {
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

    return pred_file


# ============================================================================
# Stage 6: 回测
# ============================================================================
def extract_report_df(metrics):
    """从回测结果中提取 report DataFrame"""
    if isinstance(metrics, dict):
        val = list(metrics.values())[0]
        return val[0] if isinstance(val, tuple) else val
    elif isinstance(metrics, tuple):
        first = metrics[0]
        if isinstance(first, pd.DataFrame):
            return first
        elif isinstance(first, tuple) and len(first) >= 1:
            return first[0]
        return metrics
    return metrics


def run_backtest(final_score, top_k, drop_n, benchmark, freq):
    """运行回测"""
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy import TopkDropoutStrategy

    bt_start = str(final_score.index.get_level_values(0).min().date())
    bt_end = str(final_score.index.get_level_values(0).max().date())

    print(f"\n{'='*60}")
    print("Stage 6: 回测")
    print(f"{'='*60}")
    print(f"Backtest Range: {bt_start} ~ {bt_end}")
    print(f"Freq: {freq}, TopK={top_k}, DropN={drop_n}")

    strategy = TopkDropoutStrategy(
        signal=final_score,
        topk=top_k,
        n_drop=drop_n,
        only_tradable=True
    )

    executor_obj = SimulatorExecutor(
        time_per_step=freq,
        generate_portfolio_metrics=True,
        verbose=False
    )

    print(f"\n开始回测...")
    raw_portfolio_metrics, raw_indicators = backtest(
        executor=executor_obj,
        strategy=strategy,
        start_time=bt_start,
        end_time=bt_end,
        account=BACKTEST_CONFIG['account'],
        benchmark=benchmark,
        exchange_kwargs=BACKTEST_CONFIG['exchange_kwargs']
    )

    report_df = extract_report_df(raw_portfolio_metrics)

    if report_df is not None:
        initial_cash = BACKTEST_CONFIG['account']
        final_nav = report_df.iloc[-1]['account']
        ann_scaler = 52 if freq == 'week' else 252
        annualized_return = report_df['return'].mean() * ann_scaler

        report_df['nav'] = report_df['account']
        report_df['max_nav'] = report_df['nav'].cummax()
        report_df['drawdown'] = (report_df['nav'] - report_df['max_nav']) / report_df['max_nav']
        max_drawdown = report_df['drawdown'].min()

        bench_ret = (report_df.iloc[-1]['bench'] - report_df.iloc[0]['bench']) / report_df.iloc[0]['bench']
        total_return = (final_nav / initial_cash) - 1

        print(f'\n{"="*20} 回测绩效报告 {"="*20}')
        print(f'回测区间     : {bt_start} ~ {bt_end}')
        print(f'初始资金     : {initial_cash:,.2f}')
        print(f'最终净值     : {final_nav:,.2f}')
        print(f'策略累计收益 : {total_return*100:.2f}%')
        print(f'基准累计收益 : {bench_ret*100:.2f}% (超额: {(total_return-bench_ret)*100:.2f}%)')
        print(f'年化收益率   : {annualized_return*100:.2f}%')
        print(f'最大回撤     : {max_drawdown*100:.2f}%')
        if max_drawdown != 0:
            print(f'Calmar Ratio : {annualized_return / abs(max_drawdown):.4f}')
    else:
        print("【错误】未能提取回测数据")

    return report_df


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

    experiment_name = train_records['experiment_name']
    models = train_records['models']

    print(f"\n{'='*60}")
    print("Stage 7: 风险分析 & 排行榜")
    print(f"{'='*60}")

    # 1. Ensemble 风险指标
    leaderboard_data = []
    all_reports = {}

    if report_df is not None:
        print(">>> Ensemble 模型风险分析:")
        r_strat = report_df['return']
        r_bench = report_df['bench']
        r_excess = r_strat - r_bench

        risk_strat = pd.Series(calculate_safe_risk(r_strat, freq))
        risk_bench = pd.Series(calculate_safe_risk(r_bench, freq))
        risk_excess = pd.Series(calculate_safe_risk(r_excess, freq))

        ensemble_wide_df = pd.concat(
            [risk_strat, risk_bench, risk_excess],
            axis=1, keys=['account', 'bench', 'excess']
        )
        print(ensemble_wide_df)

        # 添加到排行榜
        metrics = risk_strat.to_dict()
        metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        metrics['name'] = 'Ensemble'
        leaderboard_data.append(metrics)
        all_reports['Ensemble'] = report_df

    # 2. 子模型排行榜
    print('\n>>> 生成子模型对比排行榜...')
    freq_val = 'week' if freq == 'week' else 'day'
    freq_suffix = '1week' if freq_val == 'week' else '1day'
    report_filename = f"portfolio_analysis/report_normal_{freq_suffix}.pkl"

    for model_name in loaded_models:
        record_id = models.get(model_name)
        if not record_id:
            continue
        try:
            recorder = R.get_recorder(recorder_id=record_id, experiment_name=experiment_name)
            hist_report = recorder.load_object(report_filename)
            all_reports[model_name] = hist_report

            if 'return' in hist_report.columns:
                metrics = calculate_safe_risk(hist_report['return'], freq=freq)
                metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
                metrics['name'] = model_name
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
        display_cols = [c for c in ['annualized_return', 'information_ratio', 'max_drawdown'] if c in leaderboard_df.columns]
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
            initial_cash = BACKTEST_CONFIG['account']
            final_nav = report_df.iloc[-1]['account']
            total_return = (final_nav - initial_cash) / initial_cash
            ann_scaler = 52 if freq == 'week' else 252
            ann_return = report_df['return'].mean() * ann_scaler

            report_df = report_df.copy()
            report_df['nav'] = report_df['account']
            report_df['max_nav'] = report_df['nav'].cummax()
            report_df['drawdown'] = (report_df['nav'] - report_df['max_nav']) / report_df['max_nav']
            max_dd = report_df['drawdown'].min()

            bench_ret = (report_df.iloc[-1]['bench'] - report_df.iloc[0]['bench']) / report_df.iloc[0]['bench']

            row.update({
                'total_return': round(total_return * 100, 2),
                'annualized_return': round(ann_return * 100, 2),
                'max_drawdown': round(max_dd * 100, 2),
                'calmar_ratio': round(ann_return / abs(max_dd), 4) if max_dd != 0 else None,
                'excess_return': round((total_return - bench_ret) * 100, 2),
            })
        comparison_data.append(row)

    comp_df = pd.DataFrame(comparison_data)
    comp_file = os.path.join(output_dir, f"combo_comparison_{anchor_date}.csv")
    comp_df.to_csv(comp_file, index=False)
    print(f"\n对比表已保存: {comp_file}")

    # 显示对比
    print(f"\n{'='*20} 组合对比 {'='*20}")
    display_cols = [c for c in ['combo', 'is_default', 'total_return', 'annualized_return',
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

    combo_norm_df = norm_df[combo_models]
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
    pred_file = save_predictions(
        final_score, anchor_date, experiment_name, method,
        combo_models, combo_metrics, static_weights, is_dynamic,
        combo_output_dir, combo_name=combo_name, is_default=is_default
    )

    # ---- Stage 6: 回测 ----
    report_df = None
    if not args.no_backtest:
        report_df = run_backtest(final_score, top_k, drop_n, benchmark, args.freq)

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
    parser.add_argument('--output-dir', type=str, default='output/ensemble',
                        help='输出目录 (默认 output/ensemble)')
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
    args = parser.parse_args()

    import env
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
    available_models = list(train_records['models'].keys())

    # ---- 确定要运行的 combo 列表 ----
    combos_to_run = []  # list of (name, models, method, manual_weights_str, is_default)

    if args.models:
        # 直接指定模型列表模式（原始行为）
        selected_models = [m.strip() for m in args.models.split(',')]
        missing = [m for m in selected_models if m not in available_models]
        if missing:
            print(f"\nWarning: 以下模型不在训练记录中，将被跳过: {missing}")
            selected_models = [m for m in selected_models if m not in missing]
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
        is_def = cfg.get('default', False)
        combos_to_run.append((args.combo, cfg['models'], cfg.get('method', 'equal'),
                              None, is_def))

    elif args.from_config_all:
        # 运行所有 combo
        combos, global_config = parse_ensemble_config(ensemble_config)
        if not combos:
            print("Error: ensemble_config.json 中没有 combos")
            sys.exit(1)
        for name, cfg in combos.items():
            is_def = cfg.get('default', False)
            combos_to_run.append((name, cfg['models'], cfg.get('method', 'equal'),
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
        combos_to_run.append((default_name, default_cfg['models'], method,
                              args.weights, True))
        print(f"从 ensemble_config.json 加载 default combo: {default_name}")
        print(f"模型: {default_cfg['models']}")
        print(f"权重: {method}")

    # ---- 验证所有 combo 的模型 ----
    all_needed_models = set()
    for _, models, _, _, _ in combos_to_run:
        valid_models = [m for m in models if m in available_models]
        missing = [m for m in models if m not in available_models]
        if missing:
            print(f"\nWarning: 以下模型不在训练记录中，将被跳过: {missing}")
        all_needed_models.update(valid_models)

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
            initial_cash = BACKTEST_CONFIG['account']
            final_nav = result['report_df'].iloc[-1]['account']
            total_return = (final_nav - initial_cash) / initial_cash
            print(f"  策略收益 : {total_return*100:.2f}%")
    print(f"输出目录   : {args.output_dir}")


if __name__ == "__main__":
    main()
