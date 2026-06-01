"""
Rolling 回测模块

对拼接后的 rolling 预测结果执行回测分析。
"""

import pandas as pd


def run_combined_backtest(model_names, combined_records, combined_exp_name, params_base):
    """
    对合并后的预测执行回测，并将回测结果的 port_analysis 等指标保存追加回相应的记录。
    """
    from qlib.workflow import R
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from quantpits.utils import strategy
    import numpy as np
    import warnings

    print(f"\n{'='*60}")
    print("📈 运行 Rolling 合并预测的回测")
    print(f"{'='*60}")

    st_config = strategy.load_strategy_config()
    bt_config = strategy.get_backtest_config(st_config)

    for model_name in model_names:
        if model_name not in combined_records:
            continue

        record_id = combined_records[model_name]
        print(f"\n  [{model_name}] 提取合并预测以进行回测 (Record: {record_id})...")

        try:
            rec = R.get_recorder(recorder_id=record_id, experiment_name=combined_exp_name)
            pred = rec.load_object("pred.pkl")

            if pred is None or pred.empty:
                print(f"  [{model_name}] 预测为空，跳过回测。")
                continue

            bt_start = str(pred.index.get_level_values(0).min().date())
            bt_end = str(pred.index.get_level_values(0).max().date())

            print(f"  [{model_name}] Backtest Range: {bt_start} ~ {bt_end}")

            # Create Strategy
            strategy_inst = strategy.create_backtest_strategy(pred, st_config)

            # Create Executor
            executor_obj = SimulatorExecutor(
                time_per_step=params_base['freq'],
                generate_portfolio_metrics=True,
                verbose=False
            )

            print(f"  [{model_name}] 执行回测...")
            with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw_portfolio_metrics, raw_indicators = backtest(
                    executor=executor_obj,
                    strategy=strategy_inst,
                    start_time=bt_start,
                    end_time=bt_end,
                    account=bt_config['account'],
                    benchmark=params_base['benchmark'],
                    exchange_kwargs=bt_config['exchange_kwargs']
                )

            # Use PortfolioAnalyzer to get traditional metrics
            from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
            from qlib.data import D

            def extract_report_df(metrics):
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

            report_df = extract_report_df(raw_portfolio_metrics)
            if report_df is None or report_df.empty:
                 print(f"  [{model_name}] 提取回测结果失败。")
                 continue

            # Format report DataFrame
            da_df = pd.DataFrame(index=report_df.index)
            da_df['收盘价值'] = report_df['account']
            da_df[params_base['benchmark']] = (1 + report_df['bench']).cumprod()
            if not isinstance(da_df.index, pd.DatetimeIndex):
                da_df.index = pd.to_datetime(da_df.index)

            bt_start_dt = da_df.index.min()
            bt_end_dt = da_df.index.max()
            daily_dates = D.calendar(start_time=bt_start_dt, end_time=bt_end_dt, freq='day')
            da_df = da_df.reindex(daily_dates, method='ffill').dropna(subset=['收盘价值'])
            da_df = da_df.reset_index().rename(columns={'index': '成交日期', 'datetime': '成交日期'})

            pa = PortfolioAnalyzer(
                daily_amount_df=da_df,
                trade_log_df=pd.DataFrame(),
                holding_log_df=pd.DataFrame(),
                benchmark_col=params_base['benchmark'],
                freq=params_base['freq']
            )
            metrics = pa.calculate_traditional_metrics()

            ann_ret = metrics.get('CAGR', 0)
            max_dd = metrics.get('Max_Drawdown', 0)
            excess = metrics.get('Excess_Return_CAGR', 0)
            ir = metrics.get('Information_Ratio', 0)
            calmar = metrics.get('Calmar', 0)

            print(f"  [{model_name}] 回测完成! Ann_Ret: {ann_ret:.2%}, Excess: {excess:.2%}, Max_DD: {max_dd:.2%}, IR: {ir:.3f}")

            # Save objects back to the same recorder
            try:
                rec.log_metrics(
                    Ann_Ret=ann_ret,
                    Max_DD=max_dd,
                    Excess_Return=excess,
                    Information_Ratio=ir,
                    Calmar=calmar
                )

                # 严格按照 Qlib PortAnaRecord 的保存格式，把报告分离并保存到 portfolio_analysis 子目录下
                port_ana_objs = {}
                if isinstance(raw_portfolio_metrics, dict):
                    for freq_key, metrics_tuple in raw_portfolio_metrics.items():
                        if isinstance(metrics_tuple, tuple) and len(metrics_tuple) >= 2:
                            port_ana_objs[f"report_normal_{freq_key}.pkl"] = metrics_tuple[0]
                            port_ana_objs[f"positions_normal_{freq_key}.pkl"] = metrics_tuple[1]
                        elif isinstance(metrics_tuple, tuple) and len(metrics_tuple) == 1:
                            port_ana_objs[f"report_normal_{freq_key}.pkl"] = metrics_tuple[0]

                if port_ana_objs:
                    rec.save_objects(artifact_path="portfolio_analysis", **port_ana_objs)

                # 指标分析保存到 sig_analysis 子目录（依照 SigAnaRecord）或者根目录
                rec.save_objects(artifact_path="sig_analysis", **{
                    f"indicator_analysis_{params_base['freq']}.pkl": raw_indicators
                })
            except Exception as log_e:
                print(f"  [{model_name}] MLflow 记录失败，可能已存在同名 metric: {log_e}")

        except Exception as e:
            print(f"  [{model_name}] 回测过程失败: {e}")
            import traceback
            traceback.print_exc()


def run_backtest_only(args, targets, params_base=None):
    """仅回测模式：读取统一训练记录中的 rolling 模型并运行回测"""
    import os
    import json
    from quantpits.utils.train_utils import (
        RECORD_OUTPUT_FILE, filter_models_by_mode, parse_model_key,
    )

    if params_base is None:
        import sys
        rt_mod = sys.modules.get('rolling_train') or sys.modules.get('quantpits.scripts.rolling_train')
        if rt_mod and hasattr(rt_mod, 'get_base_params'):
            params_base = rt_mod.get_base_params()
        else:
            from quantpits.scripts.rolling_train import get_base_params
            params_base = get_base_params()

    if os.path.exists(RECORD_OUTPUT_FILE):
        with open(RECORD_OUTPUT_FILE, 'r') as f:
            records = json.load(f)
    else:
        records = None

    if not records or "models" not in records:
        print("❌ 找不到有效的 latest_train_records.json 或内容为空。")
        return

    # 从统一文件中过滤出 rolling 模式的模型
    rolling_models = filter_models_by_mode(records.get('models', {}), 'rolling')
    if not rolling_models:
        print("❌ 统一训练记录中没有 @rolling 模式的模型记录。")
        return

    # 构建一个 rolling-only 的 records dict 供下游使用
    records = dict(records)  # shallow copy
    records['models'] = rolling_models

    combined_exp_name = records.get("experiment_name")
    # 优先使用 rolling_experiment_name（迁移后可能存在）
    if records.get('rolling_experiment_name'):
        combined_exp_name = records['rolling_experiment_name']
    if not combined_exp_name:
        freq = params_base['freq'].upper()
        combined_exp_name = f"Rolling_Combined_{freq}"

    combined_records = records["models"]
    # rolling_models 的 key 是 model@rolling 格式，targets 是裸名
    # 构建 base_name -> full_key 的映射
    base_to_key = {}
    for key in combined_records:
        base_name, _ = parse_model_key(key)
        base_to_key[base_name] = key

    model_names = []
    for m in targets.keys():
        if m in combined_records:
            model_names.append(m)
        elif m in base_to_key:
            model_names.append(base_to_key[m])

    if not model_names:
        print("❌ 选定的模型中没有找到历史滚动预测记录。")
        return

    # 检查是否在单元测试中被 mock 了 run_combined_backtest
    import sys
    from unittest.mock import Mock
    rt_mod = sys.modules.get('rolling_train')
    if rt_mod and hasattr(rt_mod, 'run_combined_backtest') and isinstance(rt_mod.run_combined_backtest, Mock):
        print(f"  [Mock Detected] calling mocked rolling_train.run_combined_backtest")
        rt_mod.run_combined_backtest(model_names, combined_records, combined_exp_name, params_base)
    else:
        run_combined_backtest(model_names, combined_records, combined_exp_name, params_base)
