#!/usr/bin/env python
"""
预训练脚本 (Pretrain)
训练基础模型（LSTM/GRU）并导出 state_dict，供上层模型（GATs/ADD/IGMTF 等）使用。

核心语义：
- 预训练 ≠ 正式训练：不写入 latest_train_records.json
- 产出保存在 data/pretrained/ 目录下
- 附带 JSON sidecar 元数据（d_feat, hidden_size, num_layers 等），用于兼容性校验

运行方式：cd QuantPits && python quantpits/scripts/pretrain.py [options]

示例：
  # 预训练指定基础模型（用基础模型自己的 YAML）
  python quantpits/scripts/pretrain.py --models lstm_Alpha158

  # 为特定上层模型预训练（用上层模型的 dataset 配置 + 基础模型架构）
  python quantpits/scripts/pretrain.py --for gats_Alpha158_plus

  # 预训练所有 basemodel 标签的模型
  python quantpits/scripts/pretrain.py --tag basemodel

  # 查看已有预训练文件
  python quantpits/scripts/pretrain.py --show-pretrained

  # 列出所有可预训练的模型
  python quantpits/scripts/pretrain.py --list

  # Dry-run
  python quantpits/scripts/pretrain.py --models lstm_Alpha158 --dry-run
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
os.chdir(ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description='预训练：训练基础模型并导出 state_dict 供上层模型使用',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --models lstm_Alpha158               # 预训练指定模型
  %(prog)s --for gats_Alpha158_plus             # 为特定上层模型预训练
  %(prog)s --tag basemodel                      # 预训练所有 basemodel
  %(prog)s --models lstm_Alpha158 --dry-run     # 仅打印计划
  %(prog)s --list                               # 列出可预训练模型
  %(prog)s --show-pretrained                    # 显示已有预训练文件
        """
    )

    # 模型选择
    select = parser.add_argument_group('模型选择')
    select.add_argument('--models', type=str,
                        help='指定基础模型名，逗号分隔 (如: lstm_Alpha158,gru_Alpha360)')
    select.add_argument('--tag', type=str, default=None,
                        help='按标签筛选 (如: basemodel)')
    select.add_argument('--for', dest='for_model', type=str,
                        help='为特定上层模型预训练 (用上层模型的 dataset 配置)')

    # 运行控制
    ctrl = parser.add_argument_group('运行控制')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='仅打印计划，不实际训练')
    ctrl.add_argument('--experiment-name', type=str, default=None,
                      help='MLflow 实验名称 (默认: Pretrain_{FREQ})')

    # 信息查看
    info = parser.add_argument_group('信息查看')
    info.add_argument('--list', action='store_true',
                      help='列出所有可预训练的基础模型')
    info.add_argument('--show-pretrained', action='store_true',
                      help='显示已有预训练文件及元数据')

    return parser.parse_args()


def show_list():
    """列出所有可预训练的基础模型"""
    from quantpits.utils.train_utils import (
        load_model_registry,
        get_models_by_filter,
        print_model_table,
    )

    registry = load_model_registry()

    # 显示所有 basemodel 标签的模型
    base_models = get_models_by_filter(registry, tag='basemodel')
    if base_models:
        print_model_table(base_models, title="可预训练的基础模型 (tag=basemodel)")
    else:
        print("⚠️  没有标记为 basemodel 的模型")

    # 显示哪些上层模型依赖预训练
    print("\n📋 上层模型预训练依赖:")
    for name, info in registry.items():
        pretrain_source = info.get('pretrain_source')
        if pretrain_source:
            print(f"  {name} → {pretrain_source}")


def show_pretrained():
    """显示已有预训练文件"""
    from quantpits.utils.train_utils import PRETRAINED_DIR

    if not os.path.exists(PRETRAINED_DIR):
        print(f"📂 预训练目录不存在: {PRETRAINED_DIR}")
        return

    pkl_files = sorted([f for f in os.listdir(PRETRAINED_DIR) if f.endswith('.pkl')])
    if not pkl_files:
        print(f"📂 预训练目录为空: {PRETRAINED_DIR}")
        return

    print(f"\n📂 预训练目录: {PRETRAINED_DIR}")
    print(f"{'─' * 70}")

    for pkl_file in pkl_files:
        pkl_path = os.path.join(PRETRAINED_DIR, pkl_file)
        size_mb = os.path.getsize(pkl_path) / (1024 * 1024)

        # 尝试加载对应的 JSON 元数据
        json_file = pkl_file.replace('.pkl', '.json')
        json_path = os.path.join(PRETRAINED_DIR, json_file)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
            d_feat = meta.get('d_feat', '?')
            hidden = meta.get('hidden_size', '?')
            layers = meta.get('num_layers', '?')
            saved_at = meta.get('saved_at', '?')
            print(f"  {pkl_file:40s} {size_mb:6.2f} MB  "
                  f"d_feat={d_feat} hidden={hidden} layers={layers}  "
                  f"saved={saved_at}")
        else:
            print(f"  {pkl_file:40s} {size_mb:6.2f} MB  (no metadata)")

    print(f"{'─' * 70}")
    print(f"  共 {len(pkl_files)} 个文件")


def pretrain_for_upper_model(upper_model_name, params, experiment_name):
    """为特定上层模型预训练基础模型
    
    使用上层模型的 dataset 配置 + 基础模型的 model 架构。
    """
    from quantpits.utils.train_utils import (
        load_model_registry,
        inject_config,
        save_pretrained_model,
        PRETRAINED_DIR,
    )
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    registry = load_model_registry()

    # 1. 解析上层模型的 pretrain_source
    upper_info = registry.get(upper_model_name)
    if upper_info is None:
        print(f"❌ 上层模型 '{upper_model_name}' 不在注册表中")
        return False

    pretrain_source = upper_info.get('pretrain_source')
    if not pretrain_source:
        print(f"❌ 模型 '{upper_model_name}' 没有 pretrain_source 字段")
        return False

    base_info = registry.get(pretrain_source)
    if base_info is None:
        print(f"❌ 基础模型 '{pretrain_source}' 不在注册表中")
        return False

    upper_yaml = upper_info['yaml_file']
    base_yaml = base_info['yaml_file']

    if not os.path.exists(upper_yaml):
        print(f"❌ 上层模型 YAML 不存在: {upper_yaml}")
        return False
    if not os.path.exists(base_yaml):
        print(f"❌ 基础模型 YAML 不存在: {base_yaml}")
        return False

    print(f"\n>>> Pretrain --for {upper_model_name}")
    print(f"    上层模型: {upper_model_name} ({upper_yaml})")
    print(f"    基础模型: {pretrain_source} ({base_yaml})")
    print(f"    策略: 上层模型的 dataset + 基础模型的 model 架构")

    # 2. 构建混合配置
    # 使用上层模型的 YAML 获取 dataset（保证 feature 一致）
    upper_config = inject_config(upper_yaml, params, model_name=None, no_pretrain=True)
    # 使用基础模型的 YAML 获取 model 配置
    base_config = inject_config(base_yaml, params, model_name=None, no_pretrain=True)

    # 组合：基础 model + 上层 dataset
    model_cfg = base_config['task']['model']
    dataset_cfg = upper_config['task']['dataset']

    # 提取 d_feat 等参数
    model_kwargs = model_cfg.get('kwargs', {})
    d_feat = model_kwargs.get('d_feat', 6)
    hidden_size = model_kwargs.get('hidden_size', 64)
    num_layers = model_kwargs.get('num_layers', 2)

    print(f"    d_feat={d_feat}, hidden_size={hidden_size}, num_layers={num_layers}")

    try:
        with R.start(experiment_name=experiment_name):
            R.set_tags(
                model=pretrain_source,
                mode='pretrain',
                for_model=upper_model_name,
                anchor_date=params['anchor_date'],
            )

            # 初始化并训练
            model = init_instance_by_config(model_cfg)
            dataset = init_instance_by_config(dataset_cfg)

            print(f"[{pretrain_source}] Training (pretrain for {upper_model_name})...")
            model.fit(dataset=dataset)

            # 保存预训练 state_dict
            save_pretrained_model(
                model, pretrain_source, params['anchor_date'],
                d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers
            )

            print(f"[{pretrain_source}] Pretrain complete!")
            return True

    except Exception as e:
        print(f"!!! Pretrain error for {pretrain_source}: {e}")
        import traceback
        traceback.print_exc()
        return False


def pretrain_base_model(model_name, model_info, params, experiment_name):
    """预训练单个基础模型（使用自己的 YAML 配置）"""
    from quantpits.utils.train_utils import (
        inject_config,
        save_pretrained_model,
    )
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    yaml_file = model_info['yaml_file']
    if not os.path.exists(yaml_file):
        print(f"!!! Warning: {yaml_file} not found, skipping...")
        return False

    print(f"\n>>> Pretrain: {model_name} from {yaml_file}")

    task_config = inject_config(yaml_file, params, model_name=None, no_pretrain=True)

    model_cfg = task_config['task']['model']
    model_kwargs = model_cfg.get('kwargs', {})
    d_feat = model_kwargs.get('d_feat', 6)
    hidden_size = model_kwargs.get('hidden_size', 64)
    num_layers = model_kwargs.get('num_layers', 2)

    print(f"    d_feat={d_feat}, hidden_size={hidden_size}, num_layers={num_layers}")

    try:
        with R.start(experiment_name=experiment_name):
            R.set_tags(
                model=model_name,
                mode='pretrain',
                anchor_date=params['anchor_date'],
            )

            model = init_instance_by_config(model_cfg)
            dataset_cfg = task_config['task']['dataset']
            dataset = init_instance_by_config(dataset_cfg)

            print(f"[{model_name}] Training (pretrain)...")
            model.fit(dataset=dataset)

            save_pretrained_model(
                model, model_name, params['anchor_date'],
                d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers
            )

            print(f"[{model_name}] Pretrain complete!")
            return True

    except Exception as e:
        print(f"!!! Pretrain error for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pretrain(args):
    """执行预训练流程"""
    from quantpits.utils.train_utils import (
        calculate_dates,
        load_model_registry,
        get_models_by_filter,
        get_models_by_names,
        print_model_table,
    )

    # 初始化 Qlib
    env.init_qlib()

    # 计算日期
    params = calculate_dates()
    freq = params.get('freq', 'week').upper()
    experiment_name = args.experiment_name or f"Pretrain_{freq}"

    # --for 模式
    if args.for_model:
        if args.dry_run:
            print(f"🔍 Dry-run: 将为 {args.for_model} 预训练基础模型")
            return

        success = pretrain_for_upper_model(args.for_model, params, experiment_name)

        print(f"\n{'=' * 60}")
        if success:
            print(f"✅ 预训练完成")
        else:
            print(f"❌ 预训练失败")
        print(f"{'=' * 60}\n")
        return

    # --models 或 --tag 模式
    registry = load_model_registry()

    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
        targets = get_models_by_names(model_names, registry)
    elif args.tag:
        targets = get_models_by_filter(registry, tag=args.tag)
    else:
        print("❌ 请指定 --models, --tag, 或 --for")
        return

    if not targets:
        print("⚠️  没有匹配的模型")
        return

    print_model_table(targets, title="待预训练模型")

    if args.dry_run:
        print("🔍 Dry-run: 以上模型将被预训练，但本次不实际执行")
        return

    # 执行预训练
    print(f"\n{'=' * 60}")
    print("🧠 开始预训练")
    print(f"{'=' * 60}")

    succeeded = []
    failed = []
    total = len(targets)

    for idx, (model_name, model_info) in enumerate(targets.items(), 1):
        print(f"\n{'─' * 60}")
        print(f"  [{idx}/{total}] {model_name}")
        print(f"{'─' * 60}")

        success = pretrain_base_model(model_name, model_info, params, experiment_name)
        if success:
            succeeded.append(model_name)
        else:
            failed.append(model_name)

    # 总结
    print(f"\n{'=' * 60}")
    print("📊 预训练完成")
    print(f"{'=' * 60}")

    if succeeded:
        print(f"  ✅ 成功: {len(succeeded)} 个 — {', '.join(succeeded)}")
    if failed:
        print(f"  ❌ 失败: {len(failed)} 个 — {', '.join(failed)}")

    print(f"\n  📂 预训练文件在: data/pretrained/")
    print(f"  💡 查看详情: python quantpits/scripts/pretrain.py --show-pretrained")
    print(f"{'=' * 60}\n")


def main():
    args = parse_args()

    # 信息查看类命令
    if args.list:
        show_list()
        return

    if args.show_pretrained:
        show_pretrained()
        return

    # 检查是否指定了模型
    has_selection = any([args.models, args.tag, args.for_model])

    if not has_selection:
        print("❌ 请指定要预训练的模型")
        print("   使用 --help 查看完整帮助")
        print("   使用 --list 查看可预训练的模型")
        return

    run_pretrain(args)


if __name__ == "__main__":
    main()
