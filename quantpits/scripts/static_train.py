#!/usr/bin/env python
"""
Static Training Script (静态训练)
统一入口：全量训练、增量训练、仅预测。

运行方式：cd QuantPits && python quantpits/scripts/static_train.py [options]

模式说明：
  全量训练(--full):    训练所有 enabled 模型，全量覆写 latest_train_records.json
  增量训练(默认):       训练指定模型，merge 方式更新记录
  仅预测(--predict-only): 不训练，使用已有模型预测新数据

示例：
  # 全量训练
  python quantpits/scripts/static_train.py --full

  # 增量训练指定模型
  python quantpits/scripts/static_train.py --models gru,mlp

  # 增量训练所有 enabled（merge 模式）
  python quantpits/scripts/static_train.py --all-enabled

  # 仅预测
  python quantpits/scripts/static_train.py --predict-only --all-enabled

  # 按标签训练
  python quantpits/scripts/static_train.py --tag tree

  # Dry-run
  python quantpits/scripts/static_train.py --models gru --dry-run

  # 断点恢复
  python quantpits/scripts/static_train.py --models gru,mlp --resume

  # 查看模型注册表
  python quantpits/scripts/static_train.py --list

  # 查看运行状态
  python quantpits/scripts/static_train.py --show-state
"""

import os
import sys
import json
import argparse
from datetime import datetime

from quantpits.utils import env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(SCRIPT_DIR)
# os.chdir(ROOT_DIR)  # Moved to main() to avoid import-time side effects

DEFAULT_PREDICT_EXPERIMENT = "Prod_Predict"


# ================= CLI =================
def parse_args():
    parser = argparse.ArgumentParser(
        description='静态训练：全量训练、增量训练、仅预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --full                                  # 全量训练所有 enabled 模型
  %(prog)s --models gru,mlp                        # 增量训练指定模型
  %(prog)s --all-enabled                           # 增量训练所有 enabled 模型
  %(prog)s --predict-only --all-enabled            # 仅预测
  %(prog)s --tag tree                              # 按标签训练
  %(prog)s --models gru --dry-run                  # 预览训练计划
  %(prog)s --models gru,mlp --resume               # 断点恢复
  %(prog)s --list                                  # 列出模型注册表
  %(prog)s --show-state                            # 查看运行状态
        """
    )

    mode = parser.add_argument_group('运行模式')
    mode.add_argument('--full', action='store_true',
                      help='全量训练：训练所有 enabled 模型，全量覆写 latest_train_records.json')
    mode.add_argument('--predict-only', action='store_true',
                      help='仅预测：使用已有模型对最新数据预测，不重新训练')

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
                        help='所有 enabled=true 的模型')

    skip_group = parser.add_argument_group('排除与跳过')
    skip_group.add_argument('--skip', type=str,
                            help='跳过指定模型，逗号分隔')
    skip_group.add_argument('--resume', action='store_true',
                            help='从上次中断处继续（跳过已完成的模型）')

    ctrl = parser.add_argument_group('运行控制')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='仅打印待训练/预测模型列表，不实际执行')
    ctrl.add_argument('--experiment-name', type=str, default=None,
                      help='MLflow 实验名称 (默认: Prod_Train_{FREQ} / Prod_Predict_{FREQ})')
    ctrl.add_argument('--no-pretrain', action='store_true',
                      help='忽略 pretrain_source，使用随机权重初始化 basemodel')
    ctrl.add_argument('--source-records', type=str,
                      default='latest_train_records.json',
                      help='predict-only 的源训练记录文件 (默认: latest_train_records.json)')

    info = parser.add_argument_group('信息查看')
    info.add_argument('--list', action='store_true',
                      help='列出模型注册表（可结合筛选条件）')
    info.add_argument('--show-state', action='store_true',
                      help='显示上次运行状态')
    info.add_argument('--clear-state', action='store_true',
                      help='清除运行状态文件')

    return parser.parse_args()


# ================= 全量训练 =================
def run_full_train(args):
    """全量训练所有 enabled 模型，overwrite 记录"""
    from quantpits.utils.train_utils import (
        calculate_dates,
        load_model_registry,
        get_enabled_models,
        train_single_model,
        overwrite_train_records,
        backup_file_with_date,
        print_model_table,
        make_model_key,
        PREDICTION_OUTPUT_DIR,
        RECORD_OUTPUT_FILE,
    )

    env.init_qlib()
    params = calculate_dates()

    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    registry = load_model_registry()
    enabled_models = get_enabled_models(registry)

    if not enabled_models:
        print("⚠️  没有找到 enabled=true 的模型，请检查 config/model_registry.yaml")
        return

    print_model_table(enabled_models, title="全量训练模型列表")

    if args.dry_run:
        print("🔍 Dry-run 模式: 以上模型将被训练，但本次不会实际执行")
        return

    experiment_name = args.experiment_name or f"Prod_Train_{params.get('freq', 'week').upper()}"

    current_records = {
        "experiment_name": experiment_name,
        "static_experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }

    model_performances = {}

    total = len(enabled_models)
    for idx, (model_name, model_info) in enumerate(enabled_models.items(), 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}] {model_name}")
        print(f"{'─'*60}")

        yaml_file = model_info['yaml_file']
        result = train_single_model(model_name, yaml_file, params, experiment_name,
                                    no_pretrain=args.no_pretrain)

        if result['success']:
            model_key = make_model_key(model_name, 'static')
            current_records['models'][model_key] = result['record_id']
            if result['performance']:
                model_performances[model_name] = result['performance']
        else:
            print(f"❌ 模型 {model_name} 训练失败: {result.get('error', 'Unknown')}")

    # 全量覆写记录（自动备份历史）
    overwrite_train_records(current_records)

    # 保存模型成绩对比
    perf_file = f"output/model_performance_{params['anchor_date']}.json"
    backup_file_with_date(perf_file, prefix=f"model_performance_{params['anchor_date']}")
    with open(perf_file, 'w') as f:
        json.dump(model_performances, f, indent=4)

    print(f"\n{'='*50}")
    print(f"All tasks finished. Experiment: {experiment_name}")
    print(f"Records saved to {RECORD_OUTPUT_FILE}")
    print(f"Performance comparison saved to {perf_file}")
    print(f"\nModel Performances:")
    for name, perf in model_performances.items():
        ic = perf.get('IC_Mean', 'N/A')
        icir = perf.get('ICIR', 'N/A')
        ic_str = f"{ic:.4f}" if isinstance(ic, float) else ic
        icir_str = f"{icir:.4f}" if isinstance(icir, float) else icir
        print(f"  {name}: IC={ic_str}, ICIR={icir_str}")
    print(f"{'='*50}\n")


# ================= 增量训练 =================
def run_incremental_train(args, targets):
    """增量训练指定模型，merge 方式更新记录"""
    from quantpits.utils.train_utils import (
        calculate_dates,
        train_single_model,
        merge_train_records,
        merge_performance_file,
        save_run_state,
        load_run_state,
        clear_run_state,
        print_model_table,
        make_model_key,
        RECORD_OUTPUT_FILE,
    )

    # 打印待训练模型
    print_model_table(targets, title="待训练模型")

    # 处理 resume 模式
    completed_models = set()
    if args.resume:
        state = load_run_state()
        if state and state.get('completed'):
            completed_models = set(state['completed'])
            remaining = {k: v for k, v in targets.items() if k not in completed_models}
            if completed_models:
                skipped = [m for m in targets if m in completed_models]
                print(f"⏩ Resume 模式: 跳过已完成的 {len(skipped)} 个模型: {', '.join(skipped)}")
            targets = remaining

            if not targets:
                print("✅ 所有目标模型已在上次运行中完成")
                return

            print_model_table(targets, title="剩余待训练模型")
        else:
            print("ℹ️  没有找到上次运行状态，将从头开始训练")

    # Dry-run 模式
    if args.dry_run:
        print("🔍 Dry-run 模式: 以上模型将被训练，但本次不会实际执行")
        print("   去掉 --dry-run 参数以实际运行训练")
        return

    # ===== 开始训练 =====
    print("\n" + "="*60)
    print("🚀 开始增量训练")
    print("="*60)

    env.init_qlib()
    params = calculate_dates()
    freq = params.get('freq', 'week').upper()

    experiment_name = args.experiment_name or f"Prod_Train_{freq}"

    # 初始化运行状态
    all_target_names = list(completed_models | set(targets.keys()))
    run_state = {
        'started_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'mode': 'incremental',
        'experiment_name': experiment_name,
        'anchor_date': params['anchor_date'],
        'target_models': all_target_names,
        'completed': list(completed_models),
        'failed': {},
        'skipped': []
    }
    save_run_state(run_state)

    # 训练结果收集
    new_records = {
        "experiment_name": experiment_name,
        "static_experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }
    new_performances = {}

    total = len(targets)
    for idx, (model_name, model_info) in enumerate(targets.items(), 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}] {model_name}")
        print(f"{'─'*60}")

        yaml_file = model_info['yaml_file']

        result = train_single_model(
            model_name, yaml_file, params, experiment_name,
            no_pretrain=args.no_pretrain
        )

        if result['success']:
            model_key = make_model_key(model_name, 'static')
            new_records['models'][model_key] = result['record_id']
            if result['performance']:
                new_performances[model_name] = result['performance']

            # 更新运行状态
            run_state['completed'].append(model_name)
        else:
            run_state['failed'][model_name] = result.get('error', 'Unknown error')
            print(f"❌ 模型 {model_name} 训练失败: {result.get('error', 'Unknown')}")

        # 实时保存运行状态（防止中断丢失进度）
        save_run_state(run_state)

    # ===== 合并记录 =====
    if new_records['models']:
        print("\n" + "="*60)
        print("📦 合并训练记录")
        print("="*60)

        merge_train_records(new_records)

        if new_performances:
            merge_performance_file(new_performances, params['anchor_date'])

    # ===== 训练总结 =====
    print(f"\n{'='*60}")
    print("📊 增量训练完成")
    print("="*60)

    succeeded = run_state['completed']
    this_run_completed = [m for m in succeeded if m in targets]
    failed = run_state['failed']

    print(f"  ✅ 成功: {len(this_run_completed)} 个模型")
    for name in this_run_completed:
        perf = new_performances.get(name, {})
        ic = perf.get('IC_Mean', 'N/A')
        icir = perf.get('ICIR', 'N/A')
        ic_str = f"{ic:.4f}" if isinstance(ic, float) else ic
        icir_str = f"{icir:.4f}" if isinstance(icir, float) else icir
        print(f"    {name}: IC={ic_str}, ICIR={icir_str}")

    if failed:
        print(f"  ❌ 失败: {len(failed)} 个模型")
        for name, err in failed.items():
            print(f"    {name}: {err[:80]}")
        print(f"\n  💡 提示: 使用 --resume 参数可跳过已成功的模型，重新训练失败的模型")

    print(f"{'='*60}\n")

    # 训练全部成功时清除运行状态
    if not failed:
        clear_run_state()


# ================= 仅预测 =================
def run_predict_only(args, targets):
    """使用已有模型对最新数据预测，不重训"""
    from quantpits.utils.train_utils import (
        calculate_dates,
        merge_train_records,
        merge_performance_file,
        predict_single_model,
        print_model_table,
        make_model_key,
        resolve_model_key,
        PREDICTION_OUTPUT_DIR,
        RECORD_OUTPUT_FILE,
    )

    # 加载源训练记录
    source_file = args.source_records
    if not os.path.exists(source_file):
        print(f"❌ 源训练记录文件不存在: {source_file}")
        print("   请先运行训练（--full 或 --models）生成 latest_train_records.json")
        return

    with open(source_file, 'r') as f:
        source_records = json.load(f)

    print(f"📂 源训练记录: {source_file}")
    print(f"   实验: {source_records.get('experiment_name', 'N/A')}")
    print(f"   锚点日期: {source_records.get('anchor_date', 'N/A')}")
    print(f"   模型数: {len(source_records.get('models', {}))}")

    # 检查哪些模型在源记录中存在
    source_models = source_records.get('models', {})
    available = {}
    missing = {}
    for k, v in targets.items():
        if resolve_model_key(k, source_models, default_mode='static'):
            available[k] = v
        else:
            missing[k] = v

    if missing:
        print(f"\n⚠️  以下模型不在源训练记录中，将跳过:")
        for name in missing:
            print(f"    - {name}")

    if not available:
        print("❌ 没有可预测的模型（所有选定模型都不在源训练记录中）")
        return

    print_model_table(available, title="待预测模型")

    if args.dry_run:
        print("🔍 Dry-run 模式: 以上模型将被预测，但本次不会实际执行")
        return

    # ===== 开始预测 =====
    print("\n" + "=" * 60)
    print("🚀 开始 Predict-Only")
    print("=" * 60)

    env.init_qlib()
    params = calculate_dates()
    freq = params.get('freq', 'week').upper()

    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = f"{DEFAULT_PREDICT_EXPERIMENT}_{freq}"

    # 收集结果
    new_records = {
        "experiment_name": experiment_name,
        "static_experiment_name": experiment_name,
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
            experiment_name, source_records,
            no_pretrain=args.no_pretrain
        )

        if result['success']:
            model_key = make_model_key(model_name, 'static')
            new_records['models'][model_key] = result['record_id']
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

        merge_train_records(new_records)

        if new_performances:
            merge_performance_file(new_performances, params['anchor_date'])

    # ===== 预测总结 =====
    print(f"\n{'=' * 60}")
    print("📊 Predict-Only 完成")
    print("=" * 60)

    succeeded = [m for m in available if m in new_performances]
    print(f"  ✅ 成功: {len(succeeded)} 个模型")
    for name in succeeded:
        perf = new_performances.get(name, {})
        ic = perf.get('IC_Mean', 'N/A')
        icir = perf.get('ICIR', 'N/A')
        ic_str = f"{ic:.4f}" if isinstance(ic, float) else ic
        icir_str = f"{icir:.4f}" if isinstance(icir, float) else icir
        
        model_key = make_model_key(name, 'static')
        print(f"    {model_key}: IC={ic_str}, ICIR={icir_str}")

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
    print(f"\n  💡 后续步骤:")
    print(f"     穷举: python quantpits/scripts/brute_force_fast.py --max-combo-size 3")
    print(f"     融合: python quantpits/scripts/ensemble_fusion.py --models <模型列表>")
    print(f"{'=' * 60}\n")


# ================= 信息命令 =================
def show_state():
    """显示运行状态"""
    from quantpits.utils.train_utils import load_run_state

    state = load_run_state()
    if state is None:
        print("ℹ️  没有找到运行状态文件")
        return

    print("\n📋 上次运行状态:")
    print(f"  开始时间: {state.get('started_at', 'N/A')}")
    print(f"  运行模式: {state.get('mode', 'N/A')}")
    print(f"  实验名称: {state.get('experiment_name', 'N/A')}")
    print(f"  锚点日期: {state.get('anchor_date', 'N/A')}")

    completed = state.get('completed', [])
    failed = state.get('failed', {})
    targets = state.get('target_models', [])
    remaining = [m for m in targets if m not in completed and m not in failed]

    print(f"\n  目标模型: {len(targets)} 个")
    if completed:
        print(f"  ✅ 已完成: {len(completed)} - {', '.join(completed)}")
    if failed:
        print(f"  ❌ 失败: {len(failed)}")
        for name, err in failed.items():
            print(f"      {name}: {err[:80]}")
    if remaining:
        print(f"  ⏳ 未执行: {len(remaining)} - {', '.join(remaining)}")


# ================= 主入口 =================
def main():
    os.chdir(ROOT_DIR)
    from quantpits.utils import env as _env
    _env.safeguard("Static Train")
    args = parse_args()

    # 信息查看类命令（不需要 Qlib 初始化）
    if args.list:
        from quantpits.utils.train_utils import show_model_list, RECORD_OUTPUT_FILE
        source_file = args.source_records if args.predict_only else None
        show_model_list(args, source_records_file=source_file)
        return

    if args.show_state:
        show_state()
        return

    if args.clear_state:
        from quantpits.utils.train_utils import clear_run_state
        clear_run_state()
        return

    from quantpits.utils.operator_log import OperatorLog
    with OperatorLog("static_train", args=sys.argv[1:]) as oplog:
        # 判断运行模式
        if args.full:
            # 全量模式：忽略模型选择参数
            run_full_train(args)
            return

        # 增量模式和 predict-only 都需要模型选择
        has_selection = any([
            args.models, args.algorithm, args.dataset,
            args.market, args.tag, args.all_enabled
        ])

        if not has_selection:
            print("❌ 请指定要训练/预测的模型")
            print("   使用 --models, --algorithm, --dataset, --tag, 或 --all-enabled")
            print("   使用 --full 全量训练所有 enabled 模型")
            print("   使用 --help 查看完整帮助")
            print("   使用 --list 查看所有可用模型")
            return

        from quantpits.utils.train_utils import resolve_target_models
        targets = resolve_target_models(args)
        if targets is None or not targets:
            print("⚠️  没有匹配的模型")
            return

        if args.predict_only:
            run_predict_only(args, targets)
        else:
            run_incremental_train(args, targets)

        oplog.set_result({
            "n_targets": len(targets),
            "predict_only": args.predict_only
        })


if __name__ == "__main__":
    main()
