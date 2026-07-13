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

DEFAULT_PREDICT_EXPERIMENT = "Prod_Predict"


# ================= CLI =================
def build_parser():
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
    ctrl.add_argument('--cache-size', type=int, default=None, metavar='MB',
                      help='Handler 缓存最大内存 (MB)，默认自动检测 (50%% 空闲 RAM)。'
                           '设为 0 禁用缓存。')
    ctrl.add_argument('--workspace', default=None, help='显式 workspace 根目录')
    ctrl.add_argument('--explain-plan', action='store_true', help='只打印轻量执行计划，不初始化 Qlib 或写文件')
    ctrl.add_argument('--json-plan', action='store_true', help='以单一 JSON 文档输出轻量执行计划')
    ctrl.add_argument('--run-id', default=None, help='显式运行 ID，用于 plan/manifest 对齐')
    ctrl.add_argument('--no-manifest', action='store_true', help='真实执行时不写 RunManifest')

    info = parser.add_argument_group('信息查看')
    info.add_argument('--list', action='store_true',
                      help='列出模型注册表（可结合筛选条件）')
    info.add_argument('--show-state', action='store_true',
                      help='显示上次运行状态')
    info.add_argument('--clear-state', action='store_true',
                      help='清除运行状态文件')

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


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

    # Initialize handler cache (unless --cache-size 0)
    cache_mgr = None
    if args.cache_size != 0:
        from quantpits.utils.handler_cache import (
            HandlerCacheManager, enumerate_tasks_static, pre_analyze,
        )
        cache_mgr = HandlerCacheManager(
            max_size_mb=args.cache_size if args.cache_size else None)
        # Pre-analyze: discover unique handler configs across all models
        yaml_paths = {m: info['yaml_file'] for m, info in enabled_models.items()}
        tasks = enumerate_tasks_static(
            list(enabled_models.keys()), yaml_paths, params)
        pre_analyze(tasks, cache_mgr)

    experiment_name = args.experiment_name or f"Prod_Train_{params.get('freq', 'week').upper()}"

    current_records = {
        "experiment_name": experiment_name,
        "static_experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "model_records": {},
    }

    model_performances = {}
    failed_models = []

    total = len(enabled_models)
    for idx, (model_name, model_info) in enumerate(enabled_models.items(), 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}] {model_name}")
        print(f"{'─'*60}")

        yaml_file = model_info['yaml_file']
        result = train_single_model(model_name, yaml_file, params,
                                    experiment_name,
                                    no_pretrain=args.no_pretrain,
                                    cache_mgr=cache_mgr)

        if result['success']:
            model_key = make_model_key(model_name, 'static')
            current_records['models'][model_key] = result['record_id']
            if result.get('record_entry'):
                current_records['model_records'][model_key] = result['record_entry']
            if result['performance']:
                model_performances[model_name] = result['performance']
        else:
            failed_models.append(model_name)
            print(f"❌ 模型 {model_name} 训练失败: {result.get('error', 'Unknown')}")

    # Full publication is all-or-nothing. Preserve the previous registry if
    # any selected model failed or lacked verified recorder evidence.
    incomplete = failed_models or set(current_records['model_records']) != set(current_records['models'])
    if incomplete:
        print("❌ 全量训练未完整成功，保留现有训练记录，不发布部分结果")
    else:
        overwrite_train_records(current_records)

    # 保存模型成绩对比
    perf_file = os.path.join(env.ROOT_DIR, "output", f"model_performance_{params['anchor_date']}.json")
    os.makedirs(os.path.dirname(perf_file), exist_ok=True)
    backup_file_with_date(perf_file, prefix=f"model_performance_{params['anchor_date']}")
    with open(perf_file, 'w') as f:
        json.dump(model_performances, f, indent=4)

    if cache_mgr is not None:
        print(f"\n  Handler Cache: {cache_mgr}")

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
    return {
        "success": not bool(incomplete), "failed": tuple(failed_models),
        "succeeded": tuple(sorted(current_records["models"])),
        "published": not bool(incomplete), "anchor_date": params["anchor_date"],
        "experiment_name": experiment_name,
    }


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

    # Initialize handler cache (unless --cache-size 0)
    cache_mgr = None
    if args.cache_size != 0:
        from quantpits.utils.handler_cache import (
            HandlerCacheManager, enumerate_tasks_static, pre_analyze,
        )
        cache_mgr = HandlerCacheManager(
            max_size_mb=args.cache_size if args.cache_size else None)
        yaml_paths = {m: info['yaml_file'] for m, info in targets.items()}
        tasks = enumerate_tasks_static(
            list(targets.keys()), yaml_paths, params)
        pre_analyze(tasks, cache_mgr)

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
    run_state['run_id'] = getattr(args, 'run_id', None)
    run_state['plan_fingerprint'] = getattr(args, 'plan_fingerprint', None)
    save_run_state(run_state)

    # 训练结果收集
    new_records = {
        "experiment_name": experiment_name,
        "static_experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "model_records": {},
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
            no_pretrain=args.no_pretrain,
            cache_mgr=cache_mgr,
        )

        if result['success']:
            model_key = make_model_key(model_name, 'static')
            new_records['models'][model_key] = result['record_id']
            if result.get('record_entry'):
                new_records['model_records'][model_key] = result['record_entry']
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

    if cache_mgr is not None:
        print(f"\n  Handler Cache: {cache_mgr}")

    print(f"{'='*60}\n")

    # 训练全部成功时清除运行状态
    if not failed:
        clear_run_state()
    return {
        "success": not bool(failed), "failed": tuple(sorted(failed)),
        "succeeded": tuple(sorted(new_records["models"])),
        "published": bool(new_records['models']), "anchor_date": params["anchor_date"],
        "experiment_name": experiment_name,
    }


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
    if not os.path.isabs(source_file):
        source_file = os.path.join(env.ROOT_DIR, source_file)
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

    # Initialize handler cache (unless --cache-size 0)
    cache_mgr = None
    if args.cache_size != 0:
        from quantpits.utils.handler_cache import (
            HandlerCacheManager, enumerate_tasks_static, pre_analyze,
        )
        cache_mgr = HandlerCacheManager(
            max_size_mb=args.cache_size if args.cache_size else None)
        yaml_paths = {m: info['yaml_file'] for m, info in available.items()}
        tasks = enumerate_tasks_static(
            list(available.keys()), yaml_paths, params)
        pre_analyze(tasks, cache_mgr)

    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = f"{DEFAULT_PREDICT_EXPERIMENT}_{freq}"

    # 收集结果
    new_records = {
        "experiment_name": experiment_name,
        "static_experiment_name": experiment_name,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "model_records": {},
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
            no_pretrain=args.no_pretrain,
            cache_mgr=cache_mgr,
        )

        if result['success']:
            model_key = make_model_key(model_name, 'static')
            new_records['models'][model_key] = result['record_id']
            if result.get('record_entry'):
                new_records['model_records'][model_key] = result['record_entry']
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

    if cache_mgr is not None:
        print(f"\n  Handler Cache: {cache_mgr}")

    print(f"\n  📂 实验名: {experiment_name}")
    print(f"  📋 记录已合并到: latest_train_records.json")
    print(f"\n  💡 后续步骤:")
    print(f"     穷举: python quantpits/scripts/brute_force_fast.py --max-combo-size 3")
    print(f"     融合: python quantpits/scripts/ensemble_fusion.py --models <模型列表>")
    print(f"{'=' * 60}\n")
    return {
        "success": not bool(failed_models), "failed": tuple(sorted(failed_models)),
        "succeeded": tuple(sorted(new_records["models"])),
        "published": bool(new_records['models']), "anchor_date": params["anchor_date"],
        "experiment_name": experiment_name,
    }


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
def _legacy_execute(prepared, args):
    from quantpits.utils.train_utils import load_model_registry
    registry = load_model_registry()
    target_names = {item.model_name for item in prepared.targets}
    targets = {key: value for key, value in registry.items() if key in target_names}
    args.source_records = str((prepared.ctx.root / prepared.options.source_records).resolve())
    args.run_id = prepared.plan.run_id
    args.plan_fingerprint = prepared.plan_fingerprint
    if prepared.options.action == "full":
        return run_full_train(args)
    if prepared.options.action == "predict_only":
        return run_predict_only(args, targets)
    return run_incremental_train(args, targets)


def main(argv=None):
    args = parse_args(argv)
    ctx = env.get_workspace_context(args.workspace)

    # 信息查看类命令（不需要 Qlib 初始化）
    if args.list:
        env.set_root_dir(str(ctx.root))
        from quantpits.utils.train_utils import show_model_list, RECORD_OUTPUT_FILE
        source_file = str((ctx.root / args.source_records).resolve()) if args.predict_only else None
        show_model_list(args, source_records_file=source_file)
        return

    if args.show_state:
        env.set_root_dir(str(ctx.root))
        show_state()
        return

    if args.clear_state:
        env.safeguard("Static Train: clear state", workspace_root=ctx.root)
        env.set_root_dir(str(ctx.root))
        from quantpits.utils.train_utils import clear_run_state
        clear_run_state()
        return

    from quantpits.training.command import (
        options_from_namespace, prepare_training_run, prepared_plan_json, render_prepared_plan,
    )
    from quantpits.training.errors import TrainingCommandError
    from quantpits.training.service import TrainingExecutionHooks, TrainingExecutionService
    try:
        options = options_from_namespace(args, "static")
        cli_args = tuple(argv if argv is not None else sys.argv[1:])
        prepared = prepare_training_run(ctx=ctx, options=options, cli_args=cli_args)
        if options.json_plan:
            print(json.dumps(prepared_plan_json(prepared), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if options.explain_plan:
            print(render_prepared_plan(prepared))
            return 0
        env.safeguard("Static Train", workspace_root=ctx.root)
        service = TrainingExecutionService(TrainingExecutionHooks(
            activate_workspace=env.set_root_dir,
            init_qlib=env.init_qlib,
            execute_legacy=lambda value: _legacy_execute(value, args),
        ))
        service.execute(prepared)
        if not args.predict_only:
            try:
                from quantpits.scripts.deep_analysis.promote_config import update_promote_status
                update_promote_status(str(ctx.root), model_names=[item.model_name for item in prepared.targets])
            except Exception:
                pass
        return 0
    except TrainingCommandError as exc:
        print("❌ %s" % exc, file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
