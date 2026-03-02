#!/usr/bin/env python
"""
Predict-Only 脚本 (Production Predict Only)
使用已训练好的模型对最新数据进行预测和回测，不重新训练。

核心语义：
- 从 latest_train_records.json 获取已有模型的 recorder_id
- 加载 model.pkl，用最新数据生成新预测
- 在 Prod_Predict_{Freq} 实验下创建新 Recorder（含 pred.pkl + SignalRecord）
- 以 merge 方式更新 latest_train_records.json，保证下游穷举/融合兼容

运行方式：cd QuantPits && python quantpits/scripts/prod_predict_only.py [options]

示例：
  # 预测所有 enabled 模型
  python quantpits/scripts/prod_predict_only.py --all-enabled

  # 预测指定模型
  python quantpits/scripts/prod_predict_only.py --models gru,mlp

  # 按标签筛选
  python quantpits/scripts/prod_predict_only.py --tag tree

  # Dry-run（仅查看计划）
  python quantpits/scripts/prod_predict_only.py --models gru,mlp --dry-run

  # 查看可用模型
  python quantpits/scripts/prod_predict_only.py --list
"""

import os
import sys
import json
import argparse
from datetime import datetime

import env
os.chdir(env.ROOT_DIR)

DEFAULT_EXPERIMENT_NAME = "Prod_Predict"


# init_qlib 由 env.init_qlib() 集中管理（延迟导入 qlib，在解析参数之后调用）


def parse_args():
    parser = argparse.ArgumentParser(
        description='仅预测：使用已有模型对最新数据进行预测和回测，不重新训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --all-enabled                           # 预测所有 enabled 模型
  %(prog)s --models gru,mlp                        # 预测指定模型
  %(prog)s --algorithm lstm                        # 预测所有 LSTM 系列
  %(prog)s --tag tree                              # 预测所有 tree 标签模型
  %(prog)s --all-enabled --skip catboost_Alpha158  # 跳过指定模型
  %(prog)s --models gru --dry-run                  # 仅打印计划，不预测
  %(prog)s --list                                  # 列出所有注册模型
        """
    )

    # 模型选择
    select = parser.add_argument_group('模型选择')
    select.add_argument('--models', type=str,
                        help='指定模型名，逗号分隔 (如: gru,mlp,alstm_Alpha158)')
    select.add_argument('--algorithm', type=str,
                        help='按算法筛选 (如: lstm, gru, lightgbm)')
    select.add_argument('--dataset', type=str,
                        help='按数据集筛选 (如: Alpha158, Alpha360)')
    select.add_argument('--market', type=str,
                        help='按市场筛选 (如: csi300)')
    select.add_argument('--tag', type=str,
                        help='按标签筛选 (如: ts, tree, attention)')
    select.add_argument('--all-enabled', action='store_true',
                        help='预测所有 enabled=true 的模型')

    # 排除
    skip_group = parser.add_argument_group('排除')
    skip_group.add_argument('--skip', type=str,
                            help='跳过指定模型，逗号分隔')

    # 数据来源
    source = parser.add_argument_group('数据来源')
    source.add_argument('--source-records', type=str,
                        default='latest_train_records.json',
                        help='源训练记录文件，用于获取已有模型 (默认: latest_train_records.json)')

    # 运行控制
    ctrl = parser.add_argument_group('运行控制')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='仅打印待预测模型列表，不实际执行')
    ctrl.add_argument('--experiment-name', type=str,
                      default=DEFAULT_EXPERIMENT_NAME,
                      help=f'MLflow 实验名称 (默认: {DEFAULT_EXPERIMENT_NAME})')

    # 信息查看
    info = parser.add_argument_group('信息查看')
    info.add_argument('--list', action='store_true',
                      help='列出模型注册表（可结合筛选条件）')

    return parser.parse_args()


def resolve_target_models(args):
    """
    根据 CLI 参数解析目标模型列表

    Returns:
        dict: {model_name: model_info} 或 None（未指定选择条件）
    """
    from train_utils import (
        load_model_registry,
        get_enabled_models,
        get_models_by_filter,
        get_models_by_names,
    )

    registry = load_model_registry()

    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
        targets = get_models_by_names(model_names, registry)
    elif args.all_enabled:
        targets = get_enabled_models(registry)
    elif args.algorithm or args.dataset or args.market or args.tag:
        targets = get_models_by_filter(
            registry,
            algorithm=args.algorithm,
            dataset=args.dataset,
            market=args.market,
            tag=args.tag
        )
    else:
        return None

    # 应用 --skip
    if args.skip:
        skip_names = [m.strip() for m in args.skip.split(',')]
        targets = {k: v for k, v in targets.items() if k not in skip_names}
        if skip_names:
            print(f"⏭️  跳过模型: {', '.join(skip_names)}")

    return targets


def predict_single_model(model_name, model_info, params, experiment_name, source_records):
    """
    使用已有模型对新数据进行预测（不训练）

    流程：
    1. 从 source_records 获取原始 recorder_id
    2. 从原 recorder 加载 model.pkl
    3. 用 inject_config() 构建新的 dataset（新的日期范围）
    4. model.predict(dataset)
    5. 在新实验下创建 Recorder，保存 pred.pkl 和 SignalRecord
    6. 计算 IC 等指标

    Args:
        model_name: 模型名称
        model_info: 模型注册表信息（含 yaml_file 等）
        params: 日期参数（来自 calculate_dates）
        experiment_name: 新 MLflow 实验名称
        source_records: 源训练记录 dict

    Returns:
        dict: {
            'success': bool,
            'record_id': str or None,
            'performance': dict or None,
            'error': str or None
        }
    """
    from train_utils import inject_config, PREDICTION_OUTPUT_DIR

    result = {
        'success': False,
        'record_id': None,
        'performance': None,
        'error': None
    }

    yaml_file = model_info['yaml_file']
    if not os.path.exists(yaml_file):
        result['error'] = f"YAML 配置文件不存在: {yaml_file}"
        print(f"!!! Warning: {yaml_file} not found, skipping...")
        return result

    # 检查模型是否存在于源记录中
    source_models = source_records.get('models', {})
    if model_name not in source_models:
        result['error'] = f"模型 '{model_name}' 不在源训练记录中，无法加载已有模型"
        print(f"!!! Error: {result['error']}")
        return result

    source_record_id = source_models[model_name]
    source_experiment = source_records.get('experiment_name', 'Weekly_Production_Train')

    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    print(f"\n>>> Predict-Only: {model_name}")
    print(f"    Source: experiment={source_experiment}, recorder={source_record_id}")
    print(f"    YAML: {yaml_file}")

    try:
        # 1. 从源 recorder 加载模型
        print(f"[{model_name}] Loading model from source recorder...")
        source_recorder = R.get_recorder(
            recorder_id=source_record_id,
            experiment_name=source_experiment
        )
        model = source_recorder.load_object("model.pkl")
        print(f"[{model_name}] Model loaded successfully")

        # 2. 构建新的 dataset（使用新日期范围）
        task_config = inject_config(yaml_file, params)

        dataset_cfg = task_config['task']['dataset']
        dataset = init_instance_by_config(dataset_cfg)

        # 3. 在新实验下创建 Recorder 并预测
        with R.start(experiment_name=experiment_name):
            R.set_tags(
                model=model_name,
                anchor_date=params['anchor_date'],
                mode='predict_only',
                source_experiment=source_experiment,
                source_record_id=source_record_id,
            )
            R.log_params(**params)

            # 预测
            print(f"[{model_name}] Predicting...")
            pred = model.predict(dataset=dataset)

            # 保存预测结果为 CSV（便于人工查看）
            os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
            pred_file = os.path.join(
                PREDICTION_OUTPUT_DIR,
                f"{model_name}_{params['anchor_date']}.csv"
            )
            pred.to_csv(pred_file)
            print(f"[{model_name}] Predictions saved to {pred_file}")

            # 运行 SignalRecord（生成 pred.pkl + sig_analysis/ic.pkl）
            record_cfgs = task_config['task'].get('record', [])
            recorder = R.get_recorder()

            for r_cfg in record_cfgs:
                if r_cfg['kwargs'].get('model') == '<MODEL>':
                    r_cfg['kwargs']['model'] = model
                if r_cfg['kwargs'].get('dataset') == '<DATASET>':
                    r_cfg['kwargs']['dataset'] = dataset

                r_obj = init_instance_by_config(r_cfg, recorder=recorder)
                r_obj.generate()

            # 获取 IC 指标
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
                print(f"[{model_name}] Could not get IC metrics: {e}")
                performance = {"record_id": recorder.info['id']}

            rid = recorder.info['id']
            print(f"[{model_name}] Finished! New Recorder ID: {rid}")

            result['success'] = True
            result['record_id'] = rid
            result['performance'] = performance

    except Exception as e:
        result['error'] = str(e)
        print(f"!!! Error running predict-only for {model_name}: {e}")
        import traceback
        traceback.print_exc()

    return result


def run_predict_only(args):
    """执行 predict-only 流程"""
    from train_utils import (
        calculate_dates,
        merge_train_records,
        merge_performance_file,
        print_model_table,
        PREDICTION_OUTPUT_DIR,
        RECORD_OUTPUT_FILE,
    )

    # 加载源训练记录
    source_file = args.source_records
    if not os.path.exists(source_file):
        print(f"❌ 源训练记录文件不存在: {source_file}")
        print("   请先运行训练脚本生成 latest_train_records.json")
        return

    with open(source_file, 'r') as f:
        source_records = json.load(f)

    print(f"📂 源训练记录: {source_file}")
    print(f"   实验: {source_records.get('experiment_name', 'N/A')}")
    print(f"   锚点日期: {source_records.get('anchor_date', 'N/A')}")
    print(f"   模型数: {len(source_records.get('models', {}))}")

    # 解析目标模型
    targets = resolve_target_models(args)
    if targets is None:
        print("❌ 错误: 请指定至少一种模型选择方式")
        print("   使用 --models, --algorithm, --dataset, --tag, 或 --all-enabled")
        print("   使用 --help 查看完整帮助")
        return

    if not targets:
        print("⚠️  没有匹配的模型")
        return

    # 检查哪些模型在源记录中存在
    source_models = source_records.get('models', {})
    available = {k: v for k, v in targets.items() if k in source_models}
    missing = {k: v for k, v in targets.items() if k not in source_models}

    if missing:
        print(f"\n⚠️  以下模型不在源训练记录中，将跳过:")
        for name in missing:
            print(f"    - {name}")

    if not available:
        print("❌ 没有可预测的模型（所有选定模型都不在源训练记录中）")
        return

    print_model_table(available, title="待预测模型")

    # Dry-run 模式
    if args.dry_run:
        print("🔍 Dry-run 模式: 以上模型将被预测，但本次不会实际执行")
        print("   去掉 --dry-run 参数以实际运行")
        return

    # ===== 开始预测 =====
    print("\n" + "=" * 60)
    print("🚀 开始 Predict-Only")
    print("=" * 60)

    # 初始化 Qlib
    env.init_qlib()

    # 计算日期
    params = calculate_dates()
    freq = params.get('freq', 'week').upper()

    experiment_name = args.experiment_name
    if experiment_name == DEFAULT_EXPERIMENT_NAME:
        experiment_name = f"{DEFAULT_EXPERIMENT_NAME}_{freq}"

    # 收集结果
    new_records = {
        "experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }
    new_performances = {}
    failed_models = {}

    total = len(available)
    for idx, (model_name, model_info) in enumerate(available.items(), 1):
        print(f"\n{'─' * 60}")
        print(f"  [{idx}/{total}] {model_name}")
        print(f"{'─' * 60}")

        result = predict_single_model(
            model_name, model_info, params,
            experiment_name, source_records
        )

        if result['success']:
            new_records['models'][model_name] = result['record_id']
            if result['performance']:
                new_performances[model_name] = result['performance']
        else:
            failed_models[model_name] = result.get('error', 'Unknown error')
            print(f"❌ 模型 {model_name} 预测失败: {result.get('error', 'Unknown')}")

    # ===== 合并记录 =====
    if new_records['models']:
        print("\n" + "=" * 60)
        print("📦 合并预测记录")
        print("=" * 60)

        # 合并 latest_train_records.json（merge 语义）
        merged = merge_train_records(new_records)

        # 合并性能文件
        if new_performances:
            merge_performance_file(new_performances, params['anchor_date'])

    # ===== 预测总结 =====
    print(f"\n{'=' * 60}")
    print("📊 Predict-Only 完成")
    print("=" * 60)

    succeeded = [m for m in new_records['models']]
    print(f"  ✅ 成功: {len(succeeded)} 个模型")
    for name in succeeded:
        perf = new_performances.get(name, {})
        ic = perf.get('IC_Mean', 'N/A')
        icir = perf.get('ICIR', 'N/A')
        ic_str = f"{ic:.4f}" if isinstance(ic, float) else ic
        icir_str = f"{icir:.4f}" if isinstance(icir, float) else icir
        print(f"    {name}: IC={ic_str}, ICIR={icir_str}")

    if failed_models:
        print(f"  ❌ 失败: {len(failed_models)} 个模型")
        for name, err in failed_models.items():
            print(f"    {name}: {err[:80]}")

    if missing:
        print(f"  ⏭️  跳过（不在源记录中）: {len(missing)} 个模型")
        for name in missing:
            print(f"    {name}")

    print(f"\n  📂 实验名: {experiment_name}")
    print(f"  📋 记录已合并到: latest_train_records.json")
    print(f"  📊 预测 CSV 在: output/predictions/")
    print(f"\n  💡 后续步骤:")
    print(f"     穷举: python quantpits/scripts/brute_force_fast.py --max-combo-size 3")
    print(f"     融合: python quantpits/scripts/ensemble_fusion.py --models <模型列表>")
    print(f"{'=' * 60}\n")


def show_list(args):
    """列出模型注册表"""
    from train_utils import (
        load_model_registry,
        get_models_by_filter,
        print_model_table,
    )

    registry = load_model_registry()

    # 应用筛选条件
    if args.algorithm or args.dataset or args.market or args.tag:
        models = get_models_by_filter(
            registry,
            algorithm=args.algorithm,
            dataset=args.dataset,
            market=args.market,
            tag=args.tag
        )
        title = "筛选结果"
    else:
        models = registry
        title = "全部注册模型"

    print_model_table(models, title=title)

    # 打印启用/禁用统计
    enabled_count = sum(1 for m in models.values() if m.get('enabled', False))
    disabled_count = len(models) - enabled_count
    print(f"  启用: {enabled_count}  |  禁用: {disabled_count}")

    # 检查源训练记录中哪些模型可用
    source_file = args.source_records
    if os.path.exists(source_file):
        with open(source_file, 'r') as f:
            source_records = json.load(f)
        source_models = source_records.get('models', {})
        available = [name for name in models if name in source_models]
        print(f"\n  源记录 ({source_file}):")
        print(f"    已训练可预测: {len(available)} / {len(models)}")
        if available:
            print(f"    可用: {', '.join(available)}")
        not_available = [name for name in models if name not in source_models]
        if not_available:
            print(f"    无记录: {', '.join(not_available)}")
    else:
        print(f"\n  ⚠️  源记录文件不存在: {source_file}")


def main():
    import env
    env.safeguard("Prod Predict Only")
    args = parse_args()

    # 信息查看类命令（不需要 Qlib 初始化）
    if args.list:
        show_list(args)
        return

    # 检查是否指定了模型
    has_selection = any([
        args.models, args.algorithm, args.dataset,
        args.market, args.tag, args.all_enabled
    ])

    if not has_selection:
        print("❌ 请指定要预测的模型")
        print("   使用 --help 查看完整帮助")
        print("   使用 --list 查看所有可用模型")
        return

    run_predict_only(args)


if __name__ == "__main__":
    main()
