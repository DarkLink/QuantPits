"""
Rolling 预测拼接与保存模块

负责将各 window 的 pred.pkl 拼接成完整时间序列，
以及使用最新模型对当前数据进行预测。
"""

import pandas as pd
from datetime import datetime


def _filter_pred_to_test_segment(pred, window):
    """将 pred 过滤到 window 的 test 段日期范围 [test_start, test_end]"""
    dates = pred.index.get_level_values('datetime')
    mask = (dates >= pd.Timestamp(window['test_start'])) & \
           (dates <= pd.Timestamp(window['test_end']))
    return pred[mask]


def _repair_truncated_prediction(model_name, model_info, comp, window_map,
                                  rolling_exp_name, params_base):
    """
    检测并修复因训练时数据不完整而被截断的 window 预测。

    当一个窗口不再是"最后一个窗口"时，后续数据更新可能已补全其 test 范围。
    此函数检查 pred.pkl 是否覆盖完整的 test 段，若不完整则用原模型权重重新预测。

    Args:
        model_name: 模型名
        model_info: 模型配置 (含 yaml_file)
        comp: dict {window_idx, record_id}，已完成窗口记录
        window_map: dict {window_idx: window}，当前完整窗口列表
        rolling_exp_name: per-window MLflow 实验名
        params_base: 基础参数

    Returns:
        (pred_df, repaired: bool)
    """
    from qlib.workflow import R
    from quantpits.utils.train_utils import inject_config
    from qlib.utils import init_instance_by_config

    widx = comp['window_idx']
    w = window_map.get(widx)
    if not w:
        return None, False

    rec = R.get_recorder(
        recorder_id=comp['record_id'],
        experiment_name=rolling_exp_name,
    )

    pred = rec.load_object("pred.pkl")
    dates = pred.index.get_level_values('datetime')
    pred_max = dates.max()
    test_end = pd.Timestamp(w['test_end'])

    # 允许 1 天的容差（跨周末等情况）
    if pred_max >= test_end - pd.Timedelta(days=1):
        return pred, False

    print(f"    🔧 Window {widx}: 预测被截断 ({pred_max.date()} < {test_end.date()}), "
          f"自动补全...")

    try:
        model = rec.load_object("model.pkl")
    except Exception as e:
        print(f"    ⚠️  Window {widx}: 无法加载模型权重 ({e}), 跳过修复")
        return pred, False

    yaml_file = model_info['yaml_file']
    params = dict(params_base)
    params['start_time'] = w['train_start']
    params['end_time'] = w['test_end']
    params['fit_start_time'] = w['train_start']
    params['fit_end_time'] = w['train_end']
    params['valid_start_time'] = w['valid_start']
    params['valid_end_time'] = w['valid_end']
    params['test_start_time'] = w['test_start']
    params['test_end_time'] = w['test_end']

    task_config = inject_config(yaml_file, params, model_name=model_name)
    dataset_cfg = task_config['task']['dataset']
    dataset = init_instance_by_config(dataset_cfg)

    new_pred = model.predict(dataset=dataset)
    if isinstance(new_pred, pd.Series):
        new_pred = new_pred.to_frame('score')
    elif isinstance(new_pred, pd.DataFrame) and 'score' not in new_pred.columns:
        new_pred.columns = ['score']

    # 写回 MLflow，覆盖旧的截断预测
    rec.save_objects(**{"pred.pkl": new_pred})

    new_dates = new_pred.index.get_level_values('datetime')
    print(f"    🔧 Window {widx}: 补全完成 "
          f"({new_dates.min().date()} ~ {new_dates.max().date()}, "
          f"{len(new_pred)} rows)")

    return new_pred, True


def concatenate_rolling_predictions(state, model_names, rolling_exp_name,
                                    combined_exp_name, anchor_date,
                                    windows, extra_preds=None,
                                    targets=None, params_base=None):
    """
    将各 window 的 pred.pkl 拼接成完整时间序列。

    对每个模型：
    1. 检测并修复因训练时数据不完整而被截断的历史窗口预测
    2. 从各 window recorder 加载 pred.pkl
    3. 截取仅 test 段的预测（避免训练集内预测泄漏到下游回测）
    4. pd.concat (日期不重叠)
    5. 保存到 Rolling_Combined 实验的新 recorder

    Args:
        state: RollingState
        model_names: 要拼接的模型名列表（可为单个模型的列表）
        rolling_exp_name: per-window 实验名
        combined_exp_name: 拼接后实验名
        anchor_date: 锚点日期
        windows: list of dict, 完整的 rolling windows 列表（用于 test 段截取）
        extra_preds: dict {model_name: DataFrame}, 额外的 predict-only 预测
        targets: dict {model_name: model_info}, 提供 yaml_file 用于修复截断窗口
        params_base: dict, 基础参数（提供 market/benchmark 等，用于修复时构建 dataset）

    Returns:
        dict: {model_name: combined_record_id}
    """
    from qlib.workflow import R

    print(f"\n{'='*60}")
    print("📦 拼接 Rolling 预测 (仅 test 段)")
    print(f"{'='*60}")

    # 构建 window_idx -> window 的快速查找表
    window_map = {w['window_idx']: w for w in windows}

    combined_records = {}
    repaired_total = 0

    for model_name in model_names:
        completions = state.get_completed_record_ids(model_name)
        if not completions:
            print(f"  [{model_name}] 无已完成的 window, 跳过")
            continue

        model_info = targets.get(model_name) if targets else None
        last_widx = completions[-1]['window_idx']

        print(f"\n  [{model_name}] 拼接 {len(completions)} 个 windows...")

        all_preds = []
        for comp in completions:
            widx = comp['window_idx']
            try:
                rec = R.get_recorder(
                    recorder_id=comp['record_id'],
                    experiment_name=rolling_exp_name
                )

                # 对非最后窗口检测截断并自动修复
                # （最后窗口的截断由 --predict-only 处理，不在此修复）
                pred = None
                if widx != last_widx and model_info and params_base:
                    pred, repaired = _repair_truncated_prediction(
                        model_name, model_info, comp, window_map,
                        rolling_exp_name, params_base,
                    )
                    if repaired:
                        repaired_total += 1

                if pred is None:
                    pred = rec.load_object("pred.pkl")

                # 统一为单列 score DataFrame，避免下游 columns 不匹配
                if isinstance(pred, pd.DataFrame) and 'score' in pred.columns:
                    pred = pred[['score']]
                elif isinstance(pred, pd.Series):
                    pred = pred.to_frame('score')

                # 截取 test 段：仅保留 [test_start, test_end] 范围内的预测
                w = window_map.get(widx)
                if w:
                    pred = _filter_pred_to_test_segment(pred, w)

                all_preds.append(pred)
                dates = pred.index.get_level_values('datetime')
                print(f"    Window {widx}: "
                      f"{dates.min().date()} ~ {dates.max().date()}, "
                      f"{len(pred)} rows")
            except Exception as e:
                print(f"    Window {widx}: FAILED - {e}")

        if extra_preds and model_name in extra_preds:
            extra_df = extra_preds[model_name]
            if extra_df is not None and not extra_df.empty:
                # 统一为单列 score DataFrame
                if isinstance(extra_df, pd.Series):
                    extra_df = extra_df.to_frame('score')
                elif isinstance(extra_df, pd.DataFrame) and 'score' in extra_df.columns:
                    extra_df = extra_df[['score']]
                elif isinstance(extra_df, pd.DataFrame):
                    extra_df.columns = ['score']

                # extra_preds 来自 predict_with_latest_model，也需要截取 test 段。
                # 如果该模型存在 gap（最新完成 window 落后于可用 window），
                # 且 extra_preds 非空（意味着 --allow-stale-predict 已启用），
                # 则扩展到最后一个可用 window 的 test 范围。
                filter_window = None
                if completions and windows:
                    model_last_widx = completions[-1]['window_idx']
                    global_last_widx = windows[-1]['window_idx']
                    if model_last_widx < global_last_widx:
                        # gap + extra_preds → stale predict 模式，扩展范围
                        filter_window = windows[-1]
                    else:
                        filter_window = window_map.get(model_last_widx)
                elif completions:
                    last_widx = completions[-1]['window_idx']
                    filter_window = window_map.get(last_widx)
                if filter_window:
                    extra_df = _filter_pred_to_test_segment(extra_df, filter_window)

                all_preds.append(extra_df)
                dts = extra_df.index.get_level_values('datetime')
                print(f"    Extra Pred_Only: {dts.min().date()} ~ {dts.max().date()}, {len(extra_df)} rows")

        if not all_preds:
            print(f"  [{model_name}] 无有效预测数据")
            continue

        # 拼接
        combined_pred = pd.concat(all_preds)
        # 去重：重叠区域保留最新 (extra_preds / 后续 window) 的预测
        combined_pred = combined_pred[~combined_pred.index.duplicated(keep='last')]
        combined_pred = combined_pred.sort_index()

        dates = combined_pred.index.get_level_values('datetime')
        print(f"  [{model_name}] 拼接结果: "
              f"{dates.min().date()} ~ {dates.max().date()}, "
              f"{len(combined_pred)} rows")

        # 保存到 Combined 实验
        with R.start(experiment_name=combined_exp_name):
            R.set_tags(
                model=model_name,
                mode='rolling_combined',
                anchor_date=anchor_date,
                n_windows=len(completions),
            )
            R.save_objects(**{"pred.pkl": combined_pred})
            combined_rid = R.get_recorder().id

        combined_records[model_name] = combined_rid

        print(f"  [{model_name}] Combined Recorder: {combined_rid}")

    if repaired_total > 0:
        print(f"\n  🔧 共修复 {repaired_total} 个被截断的历史窗口预测")

    return combined_records


def save_rolling_records(combined_records, combined_exp_name, anchor_date):
    """
    保存 rolling 训练记录到统一的 latest_train_records.json

    使用 model@rolling key 格式写入统一记录文件，通过 merge 方式保留其他模式的记录。
    """
    from quantpits.utils.train_utils import (
        make_model_key, merge_train_records, RECORD_OUTPUT_FILE,
    )

    # 将 rolling 模型的 key 转为 model@rolling 格式
    rolling_models = {}
    for name, rid in combined_records.items():
        rolling_key = make_model_key(name, 'rolling')
        rolling_models[rolling_key] = rid

    records = {
        "experiment_name": combined_exp_name,
        "rolling_experiment_name": combined_exp_name,
        "anchor_date": anchor_date,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": rolling_models,
    }

    merge_train_records(records)

    print(f"\n📋 Rolling 记录已合并到统一文件: {RECORD_OUTPUT_FILE}")
    print(f"   模型数: {len(rolling_models)}")
    for key, rid in rolling_models.items():
        print(f"   {key}: {rid}")


def predict_with_latest_model(model_name, model_info, state,
                              rolling_exp_name, params_base, anchor_date,
                              windows, allow_stale_predict=False):
    """
    使用最近一个 window 训练的模型对最新数据预测。

    用于日常模式中距离上次 rolling 未超过 step 的情况。

    Args:
        allow_stale_predict: 是否允许用旧权重预测新窗口数据。
            默认 False，gap 时只预测已训练窗口范围。
    """
    from quantpits.utils.train_utils import inject_config
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    completions = state.get_completed_record_ids(model_name)
    if not completions:
        print(f"  [{model_name}] 无历史 rolling 训练记录，请先通过以下方式添加模型：")
        print(f"     --merge --models {model_name}       (追加新模型，不影响现有)")
        print(f"     --retrain-models {model_name}        (重建模型，不影响其他)")
        print(f"  ⚠️  请勿使用 --cold-start，会清除所有其他模型的训练记录！")
        return None

    # 取最新 window 的模型
    latest = completions[-1]
    widx = latest['window_idx']

    # 检测是否存在未训练的窗口 (state 中最新 window 落后于当前可用数据)
    last_window = windows[-1] if windows else None
    last_widx = last_window['window_idx'] if last_window else widx
    gap_windows = last_widx - widx

    if gap_windows > 0:
        if allow_stale_predict:
            print(f"  ⚠️  [{model_name}] 有 {gap_windows} 个新窗口未训练 "
                  f"(state 最新 W{widx}, 当前数据到 W{last_widx})")
            print(f"     已开启 --allow-stale-predict，将用 W{widx} 权重尽力预测全部可用数据")
            print(f"     建议尽快 --retrain-models {model_name}")
        else:
            print(f"  ⛔ [{model_name}] 有 {gap_windows} 个新窗口未训练 "
                  f"(state 最新 W{widx}, 当前数据到 W{last_widx})")
            print(f"     未开启 --allow-stale-predict，仅预测已训练窗口范围")
            print(f"     如需覆盖全部数据: --allow-stale-predict 或 --retrain-models {model_name}")
            return None  # 跳过，不生成任何预测

    print(f"  [{model_name}] 加载 Window {widx} 模型进行预测...")

    window = next((w for w in windows if w['window_idx'] == widx), None)
    if not window:
        print(f"  [{model_name}] 无法找到对应的 window 数据划分: {widx}")
        return None

    # 当有未训练窗口且允许 stale predict 时，扩展预测范围到最后一个可用窗口
    # 的 test_end，避免拼接时最新数据缺失。
    effective_test_end = (
        last_window.get('test_end') if (gap_windows > 0 and allow_stale_predict)
        else window.get('test_end', window.get('test_end_time'))
    )

    try:
        rec = R.get_recorder(
            recorder_id=latest['record_id'],
            experiment_name=rolling_exp_name
        )
        model = rec.load_object("model.pkl")

        # 构建最新数据的 dataset
        yaml_file = model_info['yaml_file']
        params = dict(params_base)
        params['anchor_date'] = anchor_date

        # 补齐基于该 window 的日期范围，以满足 inject_config 检查。
        # 当存在未训练窗口时，end_time 扩展到最后一个可用窗口，
        # 确保 dataset 能覆盖到最新数据。
        params['start_time'] = window['train_start']
        params['end_time'] = effective_test_end
        params['fit_start_time'] = window['train_start']
        params['fit_end_time'] = window['train_end']
        params['valid_start_time'] = window['valid_start']
        params['valid_end_time'] = window['valid_end']
        params['test_start_time'] = window['test_start']
        params['test_end_time'] = effective_test_end

        task_config = inject_config(yaml_file, params, model_name=model_name)

        dataset_cfg = task_config['task']['dataset']
        dataset = init_instance_by_config(dataset_cfg)

        pred = model.predict(dataset=dataset)

        # 统一为单列 score DataFrame，与训练时 SigAnaRecord 保存格式对齐
        if isinstance(pred, pd.Series):
            pred = pred.to_frame('score')
        elif isinstance(pred, pd.DataFrame) and 'score' not in pred.columns:
            pred.columns = ['score']

        print(f"  [{model_name}] 预测完成: Recorder={latest['record_id']}")

        return pred

    except Exception as e:
        print(f"  [{model_name}] 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None
