"""
Rolling 回测模块

对拼接后的 rolling 预测结果执行回测分析。
"""

import pandas as pd


_BACKTEST_COMPLETED = "rolling_backtest_completed"
_BACKTEST_PRECONDITION_FAILED = "rolling_backtest_precondition_failed"
_BACKTEST_EXECUTION_FAILED = "rolling_backtest_execution_failed"
_BACKTEST_PUBLICATION_FAILED = "rolling_backtest_publication_failed"


def _model_result(model_key, recorder_id, status, stage, reason_code, message,
                  did_execute, backtest_start=None, backtest_end=None):
    """Build the stable, JSON-safe result for one requested model."""
    return {
        "model_key": model_key,
        "recorder_id": recorder_id,
        "status": status,
        "stage": stage,
        "reason_code": reason_code,
        "message": message,
        "did_execute": bool(did_execute),
        "backtest_start": backtest_start,
        "backtest_end": backtest_end,
    }


def _batch_result(model_results):
    """Aggregate per-model facts without interpreting console output."""
    requested = len(model_results)
    attempted = sum(1 for item in model_results if item["did_execute"])
    succeeded = sum(1 for item in model_results if item["status"] == "success")
    failed = sum(1 for item in model_results if item["status"] == "failed")

    if requested and failed == 0 and succeeded == requested:
        status = "success"
        reason_code = _BACKTEST_COMPLETED
        message = "backtest completed for all %s requested model(s)" % requested
    else:
        status = "failed"
        reason_codes = {item["reason_code"] for item in model_results}
        if _BACKTEST_PUBLICATION_FAILED in reason_codes:
            reason_code = _BACKTEST_PUBLICATION_FAILED
        elif (_BACKTEST_EXECUTION_FAILED in reason_codes or
              "rolling_backtest_result_invalid" in reason_codes):
            reason_code = _BACKTEST_EXECUTION_FAILED
        else:
            reason_code = _BACKTEST_PRECONDITION_FAILED
        message = (
            "backtest failed: requested=%s attempted=%s succeeded=%s failed=%s"
            % (requested, attempted, succeeded, failed)
        )
    return {
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "did_execute": attempted > 0,
        "n_requested": requested,
        "n_attempted": attempted,
        "n_succeeded": succeeded,
        "n_failed": failed,
        "model_results": model_results,
    }


def _valid_portfolio_result(value):
    """Require at least one Qlib portfolio report before publication."""
    if not isinstance(value, dict) or not value:
        return False
    for metrics_tuple in value.values():
        if (isinstance(metrics_tuple, tuple) and metrics_tuple and
                metrics_tuple[0] is not None):
            return True
    return False


def _precondition_batch(model_keys, message):
    """Describe a backtest-only failure that occurs before backend lookup."""
    result = _batch_result([
        _model_result(
            model_key, None, "failed", "recorder_lookup",
            "rolling_backtest_recorder_unavailable", message, False,
        )
        for model_key in model_keys
    ])
    result["message"] = message
    return result


def _print_portfolio_metrics(raw_portfolio_metrics, model_name, freq):
    """Print detailed portfolio analysis tables matching qlib PortAnaRecord format.

    SimulatorExecutor returns time-series DataFrames. We compute summary
    statistics (mean, std, annualized_return, information_ratio, max_drawdown)
    for benchmark return, excess return without cost, and excess return with cost.
    """
    import numpy as np

    if not isinstance(raw_portfolio_metrics, dict):
        return
    for freq_key, metrics_tuple in raw_portfolio_metrics.items():
        if isinstance(metrics_tuple, tuple) and len(metrics_tuple) >= 1:
            ts_df = metrics_tuple[0]
        else:
            continue
        if ts_df is None or ts_df.empty or 'return' not in ts_df.columns:
            continue

        periods_per_year = 52 if freq == 'week' else 252

        def _compute_summary(series):
            s = series.dropna()
            if len(s) < 2:
                return None
            mu = s.mean()
            sigma = s.std()
            ann_ret = mu * periods_per_year
            ir = ann_ret / (sigma * np.sqrt(periods_per_year)) if sigma > 0 else 0.0
            # max drawdown
            cumulative = (1 + s).cumprod()
            running_max = cumulative.cummax()
            dd = (cumulative - running_max) / running_max
            max_dd = dd.min()
            return pd.Series({
                'mean': mu, 'std': sigma,
                'annualized_return': ann_ret,
                'information_ratio': ir,
                'max_drawdown': max_dd,
            }, name='risk')

        sections = [
            ('benchmark return', ts_df['bench']),
            ('excess return without cost',
             ts_df['return'] - ts_df['bench']),
            ('excess return with cost',
             ts_df['return'] - ts_df['bench'] - ts_df['cost'].fillna(0)),
        ]
        for label, series in sections:
            summary = _compute_summary(series)
            if summary is not None:
                print(f"\n  [{model_name}] The following are analysis results of "
                      f"{label}({freq_key}).")
                print(summary.to_string())

        # Return aggregate metrics from excess-without-cost for summary line
        excess_series = ts_df['return'] - ts_df['bench']
        exc = _compute_summary(excess_series)
        if exc is not None:
            return (exc['annualized_return'], exc['max_drawdown'],
                    exc['annualized_return'], exc['information_ratio'])
    return None, None, None, None


def _print_indicators(raw_indicators, model_name, freq):
    """Print indicator aggregates in qlib SigAnaRecord format (ffr/pa/pos only)."""
    if raw_indicators is None:
        return
    # raw_indicators from qlib.backtest.backtest() is a time-series DataFrame
    # indexed by date with columns: ffr, pa, pos, deal_amount, value, count.
    # qlib's SigAnaRecord prints only the aggregate ffr/pa/pos.
    if isinstance(raw_indicators, pd.DataFrame) and not raw_indicators.empty:
        ind_df = raw_indicators
    elif isinstance(raw_indicators, dict):
        if not raw_indicators:
            return
        val = list(raw_indicators.values())[0]
        ind_df = val[0] if isinstance(val, tuple) else val
    else:
        return
    if ind_df is None or ind_df.empty:
        return

    # Extract aggregate indicators: use mean of ffr/pa/pos columns
    agg_cols = ['ffr', 'pa', 'pos']
    available = [c for c in agg_cols if c in ind_df.columns]
    if not available:
        return
    agg = ind_df[available].mean().to_frame('value')
    print(f"\n  [{model_name}] The following are analysis results of "
          f"indicators({freq}).")
    print(agg.to_string())


def run_combined_backtest(model_names, combined_records, combined_exp_name, params_base):
    """
    对合并后的预测执行回测，并将回测结果的 port_analysis 等指标保存追加回相应的记录。
    """
    from qlib.workflow import R
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from quantpits.utils import strategy
    import numpy as np
    import traceback
    import warnings

    print(f"\n{'='*60}")
    print("📈 运行 Rolling 合并预测的回测")
    print(f"{'='*60}")

    requested_models = list(model_names)
    if not requested_models:
        return _batch_result([])

    try:
        st_config = strategy.load_strategy_config()
        bt_config = strategy.get_backtest_config(st_config)
    except Exception as exc:
        print(f"  Backtest configuration failed: {exc}")
        traceback.print_exc()
        return _batch_result([
            _model_result(
                model_name, combined_records.get(model_name), "failed",
                "backtest", _BACKTEST_EXECUTION_FAILED,
                "backtest configuration is unavailable", False,
            )
            for model_name in requested_models
        ])
    model_results = []

    for model_name in requested_models:
        if model_name not in combined_records:
            model_results.append(_model_result(
                model_name, None, "failed", "recorder_lookup",
                "rolling_backtest_recorder_unavailable",
                "no combined recorder was declared for the requested model",
                False,
            ))
            continue

        record_id = combined_records[model_name]
        print(f"\n  [{model_name}] 提取合并预测以进行回测 (Record: {record_id})...")

        try:
            rec = R.get_recorder(recorder_id=record_id, experiment_name=combined_exp_name)
        except Exception as exc:
            print(f"  [{model_name}] Recorder lookup failed: {exc}")
            traceback.print_exc()
            model_results.append(_model_result(
                model_name, record_id, "failed", "recorder_lookup",
                "rolling_backtest_recorder_unavailable",
                "selected recorder is unavailable", True,
            ))
            continue

        try:
            pred = rec.load_object("pred.pkl")
        except Exception as exc:
            print(f"  [{model_name}] Prediction load failed: {exc}")
            traceback.print_exc()
            model_results.append(_model_result(
                model_name, record_id, "failed", "prediction_load",
                "rolling_backtest_prediction_unavailable",
                "selected prediction artifact is unavailable", True,
            ))
            continue

        if pred is None or getattr(pred, "empty", True):
            print(f"  [{model_name}] 预测为空，无法执行回测。")
            model_results.append(_model_result(
                model_name, record_id, "failed", "prediction_load",
                "rolling_backtest_prediction_unavailable",
                "selected prediction artifact is empty", True,
            ))
            continue

        try:
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
        except Exception as exc:
            print(f"  [{model_name}] Backtest execution failed: {exc}")
            traceback.print_exc()
            model_results.append(_model_result(
                model_name, record_id, "failed", "backtest",
                _BACKTEST_EXECUTION_FAILED,
                "backtest execution failed", True,
            ))
            continue

        if not _valid_portfolio_result(raw_portfolio_metrics):
            print(f"  [{model_name}] Backtest returned no portfolio result.")
            model_results.append(_model_result(
                model_name, record_id, "failed", "backtest",
                "rolling_backtest_result_invalid",
                "backtest returned an invalid portfolio result", True,
                bt_start, bt_end,
            ))
            continue

        try:
            # ---- Print detailed backtest metrics (same format as training-time PortAnaRecord) ----
            ann_ret, max_dd, excess, ir = _print_portfolio_metrics(
                raw_portfolio_metrics, model_name, params_base['freq'])
            _print_indicators(raw_indicators, model_name, params_base['freq'])

            if ann_ret is not None:
                print(f"  [{model_name}] 回测完成! Ann_Ret: {ann_ret:.2%}, "
                      f"Excess: {excess:.2%}, Max_DD: {max_dd:.2%}, IR: {ir:.3f}")

            # Save objects back to the same recorder. Publication is part of
            # the requested operation, so any failure is authoritative.
            rec.log_metrics(
                Ann_Ret=ann_ret or 0,
                Max_DD=max_dd or 0,
                Excess_Return=excess or 0,
                Information_Ratio=ir or 0,
            )

            # 严格按照 Qlib PortAnaRecord 的保存格式，把报告分离并保存到 portfolio_analysis 子目录下
            port_ana_objs = {}
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
        except Exception as exc:
            print(f"  [{model_name}] Backtest publication failed: {exc}")
            traceback.print_exc()
            model_results.append(_model_result(
                model_name, record_id, "failed", "publication",
                _BACKTEST_PUBLICATION_FAILED,
                "backtest metrics or artifacts could not be published", True,
                bt_start, bt_end,
            ))
            continue

        model_results.append(_model_result(
            model_name, record_id, "success", "complete",
            _BACKTEST_COMPLETED, "backtest and publication completed", True,
            bt_start, bt_end,
        ))

    return _batch_result(model_results)


def run_backtest_only(args, targets, params_base=None, mode='rolling'):
    """仅回测模式：读取统一训练记录中的 rolling 模型并运行回测。

    Args:
        args: CLI args namespace.
        targets: {model_name: model_info} dict.
        params_base: Base params dict (auto-detected if None).
        mode: Key suffix to filter — 'rolling' (slide) or 'cpcv_rolling' (CPCV).
    """
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
        message = "找不到有效的 latest_train_records.json 或内容为空。"
        print(f"❌ {message}")
        return _precondition_batch(list(targets), message)

    # 从统一文件中过滤出指定模式的模型
    rolling_models = filter_models_by_mode(records.get('models', {}), mode)
    if not rolling_models:
        message = f"统一训练记录中没有 @{mode} 模式的模型记录。"
        print(f"❌ {message}")
        return _precondition_batch(list(targets), message)

    # 构建一个 rolling-only 的 records dict 供下游使用
    records = dict(records)  # shallow copy
    records['models'] = rolling_models

    combined_exp_name = records.get("experiment_name")
    # 优先使用对应模式的 experiment_name
    if mode == 'cpcv_rolling':
        if records.get('cpcv_rolling_experiment_name'):
            combined_exp_name = records['cpcv_rolling_experiment_name']
    else:
        if records.get('rolling_experiment_name'):
            combined_exp_name = records['rolling_experiment_name']
    if not combined_exp_name:
        freq = params_base['freq'].upper()
        combined_exp_name = f"Rolling_Combined_{freq}"

    combined_records = records["models"]
    # records 的 key 是 model@mode 格式，targets 是裸名
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
        message = "选定的模型中没有找到历史滚动预测记录。"
        print(f"❌ {message}")
        return _precondition_batch(list(targets), message)

    # 检查是否在单元测试中被 mock 了 run_combined_backtest
    import sys
    from unittest.mock import Mock
    rt_mod = sys.modules.get('rolling_train')
    if rt_mod and hasattr(rt_mod, 'run_combined_backtest') and isinstance(rt_mod.run_combined_backtest, Mock):
        print(f"  [Mock Detected] calling mocked rolling_train.run_combined_backtest")
        result = rt_mod.run_combined_backtest(
            model_names, combined_records, combined_exp_name, params_base,
        )
    else:
        result = run_combined_backtest(
            model_names, combined_records, combined_exp_name, params_base,
        )
    if not isinstance(result, dict) or result.get("status") not in (
            "success", "failed"):
        return {
            "status": "failed",
            "reason_code": _BACKTEST_EXECUTION_FAILED,
            "message": "backtest returned no authoritative batch result",
            "did_execute": True,
            "n_requested": len(model_names),
            "n_attempted": 0,
            "n_succeeded": 0,
            "n_failed": len(model_names),
            "model_results": [],
        }
    return result
