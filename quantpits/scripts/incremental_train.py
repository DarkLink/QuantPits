#!/usr/bin/env python
"""
增量训练脚本 (Incremental Training)
选择性训练个别模型，以 merge 方式更新训练记录，不影响其他模型。

核心语义：
- 同名模型 → 覆盖 recorder ID 和性能数据
- 新增模型 → 追加到记录
- 未训练模型 → 保留原有记录不变

运行方式：cd QuantPits && python quantpits/scripts/incremental_train.py [options]

示例：
  # 指定模型训练
  python quantpits/scripts/incremental_train.py --models gru,mlp

  # 按算法筛选
  python quantpits/scripts/incremental_train.py --algorithm lstm

  # 按数据集筛选
  python quantpits/scripts/incremental_train.py --dataset Alpha360

  # 按标签筛选
  python quantpits/scripts/incremental_train.py --tag tree

  # 训练所有 enabled 模型（等同全量但以 merge 方式保存）
  python quantpits/scripts/incremental_train.py --all-enabled

  # 从上次中断处继续
  python quantpits/scripts/incremental_train.py --models gru,mlp,alstm_Alpha158 --resume

  # 跳过某些模型
  python quantpits/scripts/incremental_train.py --all-enabled --skip catboost_Alpha158

  # Dry-run（仅打印待训练模型，不实际训练）
  python quantpits/scripts/incremental_train.py --models gru,mlp --dry-run
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 设置工作目录
import env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(SCRIPT_DIR)
os.chdir(ROOT_DIR)

# init_qlib 由 env.init_qlib() 集中管理（延迟导入 qlib，在解析参数之后调用）


def parse_args():
    parser = argparse.ArgumentParser(
        description='增量训练：选择性训练个别模型，以 merge 方式更新记录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --models gru,mlp                    # 训练指定模型
  %(prog)s --algorithm lstm                     # 训练所有 LSTM 系列
  %(prog)s --dataset Alpha360                   # 训练所有 Alpha360 模型
  %(prog)s --tag tree                           # 训练所有 tree 标签模型
  %(prog)s --all-enabled                        # 训练所有 enabled 模型
  %(prog)s --models gru,mlp --resume            # 从上次中断处继续
  %(prog)s --all-enabled --skip catboost_Alpha158  # 跳过指定模型
  %(prog)s --models gru --dry-run               # 仅打印计划，不训练
  %(prog)s --list                               # 列出所有注册模型
  %(prog)s --list --algorithm gru               # 列出所有 GRU 模型
        """
    )
    
    # 模型选择（互斥组：至少指定一种选择方式）
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
                        help='训练所有 enabled=true 的模型')
    
    # 排除 / 跳过
    skip_group = parser.add_argument_group('排除与跳过')
    skip_group.add_argument('--skip', type=str,
                            help='跳过指定模型，逗号分隔')
    skip_group.add_argument('--resume', action='store_true',
                            help='从上次中断处继续（跳过已完成的模型）')
    
    # 运行控制
    ctrl = parser.add_argument_group('运行控制')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='仅打印待训练模型列表，不实际训练')
    ctrl.add_argument('--experiment-name', type=str, default=None,
                      help='MLflow 实验名称 (默认: Prod_Train_{FREQ})')
    
    # 信息查看
    info = parser.add_argument_group('信息查看')
    info.add_argument('--list', action='store_true',
                      help='列出模型注册表（可结合筛选条件）')
    info.add_argument('--show-state', action='store_true',
                      help='显示上次运行状态')
    info.add_argument('--clear-state', action='store_true',
                      help='清除运行状态文件')
    
    return parser.parse_args()


def resolve_target_models(args):
    """
    根据 CLI 参数解析目标模型列表
    
    Returns:
        dict: {model_name: model_info}
    """
    from train_utils import (
        load_model_registry,
        get_enabled_models,
        get_models_by_filter,
        get_models_by_names,
    )
    
    registry = load_model_registry()
    
    if args.models:
        # 按名称指定
        model_names = [m.strip() for m in args.models.split(',')]
        targets = get_models_by_names(model_names, registry)
    elif args.all_enabled:
        # 所有 enabled 模型
        targets = get_enabled_models(registry)
    elif args.algorithm or args.dataset or args.market or args.tag:
        # 按条件筛选
        targets = get_models_by_filter(
            registry,
            algorithm=args.algorithm,
            dataset=args.dataset,
            market=args.market,
            tag=args.tag
        )
    else:
        return None  # 没有指定任何选择条件
    
    # 应用 --skip
    if args.skip:
        skip_names = [m.strip() for m in args.skip.split(',')]
        targets = {k: v for k, v in targets.items() if k not in skip_names}
        if skip_names:
            print(f"⏭️  跳过模型: {', '.join(skip_names)}")
    
    return targets


def run_incremental_train(args):
    """执行增量训练"""
    from train_utils import (
        calculate_dates,
        train_single_model,
        merge_train_records,
        merge_performance_file,
        save_run_state,
        load_run_state,
        clear_run_state,
        print_model_table,
        backup_file_with_date,
        RECORD_OUTPUT_FILE,
    )
    
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
    
    # 打印待训练模型
    print_model_table(targets, title="待训练模型")
    
    # 处理 resume 模式
    completed_models = set()
    if args.resume:
        state = load_run_state()
        if state and state.get('completed'):
            completed_models = set(state['completed'])
            # 过滤已完成的模型
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
    
    # 初始化 Qlib
    env.init_qlib()
    
    # 计算日期
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
        
        result = train_single_model(model_name, yaml_file, params, experiment_name)
        
        if result['success']:
            new_records['models'][model_name] = result['record_id']
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
        
        # 合并 latest_train_records.json
        merged = merge_train_records(new_records)
        
        # 合并性能文件
        if new_performances:
            merge_performance_file(new_performances, params['anchor_date'])
    
    # ===== 训练总结 =====
    print(f"\n{'='*60}")
    print("📊 增量训练完成")
    print("="*60)
    
    succeeded = run_state['completed']
    # 只显示本次训练的（排除 resume 带来的历史完成）
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
    
    # 按数据集分组统计
    datasets = {}
    for name, info in models.items():
        ds = info.get('dataset', 'unknown')
        datasets.setdefault(ds, []).append(name)
    
    print(f"\n  按数据集分布:")
    for ds, names in sorted(datasets.items()):
        print(f"    {ds}: {len(names)} ({', '.join(names)})")


def show_state():
    """显示运行状态"""
    from train_utils import load_run_state
    
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


def main():
    args = parse_args()
    
    # 信息查看类命令（不需要 Qlib 初始化）
    if args.list:
        show_list(args)
        return
    
    if args.show_state:
        show_state()
        return
    
    if args.clear_state:
        from train_utils import clear_run_state
        clear_run_state()
        return
    
    # 检查是否指定了模型
    has_selection = any([
        args.models, args.algorithm, args.dataset,
        args.market, args.tag, args.all_enabled
    ])
    
    if not has_selection:
        print("❌ 请指定要训练的模型")
        print("   使用 --help 查看完整帮助")
        print("   使用 --list 查看所有可用模型")
        return
    
    run_incremental_train(args)


if __name__ == "__main__":
    main()
