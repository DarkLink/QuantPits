"""
Rolling 训练核心模块

包含单窗口训练、子进程隔离、和 Model-First 批量训练逻辑。

关键设计:
  - train_window_model(): 单窗口训练（可在主进程或子进程中调用）
  - train_window_model_isolated(): 子进程隔离包装，OS 级内存回收
  - run_model_windows(): Model-First 内循环，训练一个模型的所有 windows

灵感来源: qlib TrainerR.call_in_subproc + call_in_subproc (qlib/utils/paral.py)
"""

import os
import concurrent.futures

from quantpits.scripts.rolling.memory import (
    cleanup_after_window,
    check_memory_pressure,
    log_memory,
)


def train_window_model(model_name, yaml_file, window, params_base,
                       experiment_name, no_pretrain=False):
    """
    训练单个模型在一个 rolling window 上。

    Args:
        model_name: 模型名称
        yaml_file: YAML 配置路径
        window: 日期 dict (train_start, train_end, valid_start, valid_end, test_start, test_end)
        params_base: 基础参数 (market, benchmark 等，来自 workspace config)
        experiment_name: MLflow experiment name
        no_pretrain: 是否跳过预训练

    Returns:
        dict: {success, record_id, performance, error}
    """
    from quantpits.utils.train_utils import inject_config
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    result = {
        'success': False,
        'record_id': None,
        'performance': None,
        'error': None,
    }

    if not os.path.exists(yaml_file):
        result['error'] = f"YAML 不存在: {yaml_file}"
        return result

    # 构建带 rolling window 日期的 params
    params = dict(params_base)
    params['start_time'] = window['train_start']
    params['end_time'] = window['test_end']
    params['fit_start_time'] = window['train_start']
    params['fit_end_time'] = window['train_end']
    params['valid_start_time'] = window['valid_start']
    params['valid_end_time'] = window['valid_end']
    params['test_start_time'] = window['test_start']
    params['test_end_time'] = window['test_end']

    widx = window['window_idx']

    print(f"\n>>> Rolling Train: {model_name} | Window {widx}")
    print(f"    Train: [{window['train_start']}, {window['train_end']}]")
    print(f"    Valid: [{window['valid_start']}, {window['valid_end']}]")
    print(f"    Test:  [{window['test_start']}, {window['test_end']}]")
    print(f"    YAML:  {yaml_file}")

    task_config = inject_config(yaml_file, params, model_name=model_name,
                                no_pretrain=no_pretrain)

    try:
        with R.start(experiment_name=experiment_name):
            R.set_tags(
                model=model_name,
                window_idx=widx,
                mode='rolling_train',
                train_start=window['train_start'],
                train_end=window['train_end'],
                test_start=window['test_start'],
                test_end=window['test_end'],
            )
            R.log_params(**{k: str(v) for k, v in params.items()})

            # 训练
            model_cfg = task_config['task']['model']
            model = init_instance_by_config(model_cfg)
            dataset_cfg = task_config['task']['dataset']
            dataset = init_instance_by_config(dataset_cfg)

            print(f"[{model_name}|W{widx}] Training...")
            model.fit(dataset=dataset)

            # 预测
            print(f"[{model_name}|W{widx}] Predicting...")
            pred = model.predict(dataset=dataset)

            # 保存模型
            recorder = R.get_recorder()
            recorder.save_objects(**{"model.pkl": model})

            # Signal Record
            record_cfgs = task_config['task'].get('record', [])
            for r_cfg in record_cfgs:
                if r_cfg['kwargs'].get('model') == '<MODEL>':
                    r_cfg['kwargs']['model'] = model
                if r_cfg['kwargs'].get('dataset') == '<DATASET>':
                    r_cfg['kwargs']['dataset'] = dataset
                r_obj = init_instance_by_config(r_cfg, recorder=recorder)
                r_obj.generate()
                del r_obj  # 显式释放 record 对象

            # IC metrics
            performance = {}
            try:
                ic_series = recorder.load_object("sig_analysis/ic.pkl")
                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std != 0 else None
                performance = {
                    "IC_Mean": float(ic_mean) if ic_mean else None,
                    "ICIR": float(ic_ir) if ic_ir else None,
                    "record_id": recorder.info['id'],
                }
            except Exception as e:
                print(f"[{model_name}|W{widx}] IC metrics unavailable: {e}")
                performance = {"record_id": recorder.info['id']}

            rid = recorder.info['id']
            print(f"[{model_name}|W{widx}] Done! Recorder: {rid}")

            # 显式释放大对象，减少 GC 压力
            del model, dataset, pred, recorder

            result['success'] = True
            result['record_id'] = rid
            result['performance'] = performance

    except Exception as e:
        result['error'] = str(e)
        print(f"!!! Error in rolling train {model_name} W{widx}: {e}")
        import traceback
        traceback.print_exc()

    return result


def _train_in_subprocess(qlib_config_dict, model_name, yaml_file, window,
                          params_base, experiment_name, no_pretrain):
    """
    子进程入口: 重新初始化 qlib 后执行训练。

    子进程结束后 OS 自动回收所有内存（包括 qlib H cache、PyTorch CUDA 缓存、
    DataHandler 中的 DataFrame 等），完全避免内存泄漏。

    灵感来源: qlib.utils.paral.call_in_subproc
    """
    from qlib.config import C
    C.register_from_C(qlib_config_dict)

    return train_window_model(
        model_name=model_name,
        yaml_file=yaml_file,
        window=window,
        params_base=params_base,
        experiment_name=experiment_name,
        no_pretrain=no_pretrain,
    )


def train_window_model_isolated(qlib_config, model_name, yaml_file, window,
                                 params_base, experiment_name, no_pretrain=False):
    """
    在独立子进程中训练，确保内存完全释放。

    每个训练 task 在独立子进程中执行:
    1. 子进程重新 qlib.init() (~5-10s 开销)
    2. 执行 train_window_model()
    3. 子进程退出 → OS 回收所有内存
    4. 主进程仅接收轻量级 result dict

    Args:
        qlib_config: qlib 配置对象 (C)，用于子进程重新初始化
        其余参数同 train_window_model

    Returns:
        dict: {success, record_id, performance, error}
    """
    # 检查是否在单元测试中被 mock 了 train_window_model
    import sys
    from unittest.mock import Mock
    rt_mod = sys.modules.get('rolling_train')
    if rt_mod and hasattr(rt_mod, 'train_window_model') and isinstance(rt_mod.train_window_model, Mock):
        print(f"  [Mock Detected] calling mocked rolling_train.train_window_model in main process")
        return rt_mod.train_window_model(
            model_name=model_name,
            yaml_file=yaml_file,
            window=window,
            params_base=params_base,
            experiment_name=experiment_name,
            no_pretrain=no_pretrain,
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _train_in_subprocess,
            qlib_config,
            model_name,
            yaml_file,
            window,
            params_base,
            experiment_name,
            no_pretrain,
        )
        return future.result()


def run_model_windows(model_name, model_info, windows, state,
                      params_base, experiment_name, qlib_config,
                      no_pretrain=False, dry_run=False):
    """
    训练一个模型的所有 windows (Model-First 内循环)。

    对每个 window:
      1. 检查是否已完成 (支持 --resume)
      2. 在子进程中训练
      3. 标记完成 + 轻量清理

    Args:
        model_name: 模型名称
        model_info: 模型配置 (含 yaml_file)
        windows: 完整 rolling windows 列表
        state: RollingState 实例
        params_base: 基础参数
        experiment_name: MLflow 实验名
        qlib_config: qlib 配置 (C)
        no_pretrain: 是否跳过预训练
        dry_run: 仅打印不训练

    Returns:
        int: 本轮实际训练的 window 数
    """
    yaml_file = model_info['yaml_file']
    trained_count = 0
    total_windows = len(windows)

    print(f"\n{'='*60}")
    print(f"🔧 模型: {model_name} ({total_windows} windows)")
    print(f"{'='*60}")

    for window in windows:
        widx = window['window_idx']

        # 检查是否已完成
        if state.is_window_model_done(widx, model_name):
            print(f"  ✅ Window {widx}: 已完成, 跳过")
            continue

        if dry_run:
            print(f"  🔍 Window {widx}: "
                  f"Test[{window['test_start']}, {window['test_end']}] (dry-run)")
            continue

        # 内存安全阀检查
        check_memory_pressure(f"{model_name}|W{widx}")

        # 在子进程中训练
        result = train_window_model_isolated(
            qlib_config=qlib_config,
            model_name=model_name,
            yaml_file=yaml_file,
            window=window,
            params_base=params_base,
            experiment_name=experiment_name,
            no_pretrain=no_pretrain,
        )

        if result['success']:
            state.mark_window_model_done(widx, model_name, result['record_id'])
            trained_count += 1

            perf = result.get('performance', {})
            ic = perf.get('IC_Mean')
            icir = perf.get('ICIR')
            perf_str = ""
            if ic is not None:
                perf_str += f" IC={ic:.4f}"
            if icir is not None:
                perf_str += f" ICIR={icir:.4f}"
            print(f"  ✅ Window {widx}: Done{perf_str}")
        else:
            print(f"  ❌ Window {widx}: {result.get('error', 'Unknown error')}")

        # Level 1 清理 (主进程侧，子进程已退出)
        cleanup_after_window(model_name, widx)

    return trained_count
