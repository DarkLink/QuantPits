#!/usr/bin/env python3
"""
迁移工具：将旧的训练记录格式转换为统一的 model@mode 格式。

用法:
  # 干运行（仅预览，不写入）
  python quantpits/scripts/migrate_records.py --dry-run

  # 执行迁移
  python quantpits/scripts/migrate_records.py

  # 指定工作区目录
  python quantpits/scripts/migrate_records.py --workspace ./workspaces/CSI300_Base
"""
import argparse
import json
import os
import sys

# 确保 quantpits 可导入
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="迁移旧训练记录到统一 model@mode 格式"
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="工作区根目录 (默认: 当前 env.ROOT_DIR)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览迁移结果，不写入文件"
    )
    args = parser.parse_args()

    from quantpits.utils import env

    if args.workspace:
        root_dir = os.path.abspath(args.workspace)
    else:
        root_dir = env.ROOT_DIR
    print(f"工作区: {root_dir}")

    from quantpits.utils.train_utils import (
        RECORD_OUTPUT_FILE,
        LEGACY_ROLLING_RECORD_FILE,
        migrate_legacy_records,
    )

    # 检查文件状态
    static_file = os.path.join(root_dir, os.path.basename(RECORD_OUTPUT_FILE))
    rolling_file = os.path.join(root_dir, os.path.basename(LEGACY_ROLLING_RECORD_FILE))

    print(f"\n{'='*60}")
    print("文件状态检查")
    print(f"{'='*60}")
    print(f"静态记录: {static_file}")
    print(f"  存在: {os.path.exists(static_file)}")
    print(f"滚动记录: {rolling_file}")
    print(f"  存在: {os.path.exists(rolling_file)}")

    if not os.path.exists(static_file) and not os.path.exists(rolling_file):
        print("\n⚠️  没有找到任何旧格式的训练记录文件，无需迁移。")
        return

    # 预览当前内容
    print(f"\n{'='*60}")
    print("当前记录内容")
    print(f"{'='*60}")

    if os.path.exists(static_file):
        with open(static_file, 'r') as f:
            static_data = json.load(f)
        models = static_data.get('models', {})
        has_mode_keys = any('@' in k for k in models.keys())
        print(f"\n📄 {RECORD_OUTPUT_FILE}:")
        print(f"  实验名: {static_data.get('experiment_name', 'N/A')}")
        print(f"  模型数: {len(models)}")
        print(f"  已有 @mode 格式: {has_mode_keys}")
        for k, v in models.items():
            print(f"    {k}: {v}")

        if has_mode_keys:
            print("\n✅ 静态记录已经是 model@mode 格式，无需迁移。")
            if not os.path.exists(rolling_file):
                print("   滚动记录也不存在，全部完成。")
                return

    if os.path.exists(rolling_file):
        with open(rolling_file, 'r') as f:
            rolling_data = json.load(f)
        models = rolling_data.get('models', {})
        print(f"\n📄 {LEGACY_ROLLING_RECORD_FILE}:")
        print(f"  实验名: {rolling_data.get('experiment_name', 'N/A')}")
        print(f"  模型数: {len(models)}")
        for k, v in models.items():
            print(f"    {k}: {v}")

    # 执行迁移
    print(f"\n{'='*60}")
    if args.dry_run:
        print("🔍 干运行模式 — 以下是迁移预览")
    else:
        print("🚀 执行迁移")
    print(f"{'='*60}")

    result = migrate_legacy_records(root_dir, dry_run=args.dry_run)

    if result:
        print(f"\n迁移后统一文件内容:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        models = result.get('models', {})
        static_count = sum(1 for k in models if k.endswith('@static'))
        rolling_count = sum(1 for k in models if k.endswith('@rolling'))
        print(f"\n📊 统计:")
        print(f"  @static 模型: {static_count}")
        print(f"  @rolling 模型: {rolling_count}")
        print(f"  总计: {len(models)}")

        if args.dry_run:
            print("\n⚠️  这是干运行，未写入任何文件。")
            print("   移除 --dry-run 参数以执行实际迁移。")
        else:
            print(f"\n✅ 迁移完成！")
            print(f"   统一文件: {static_file}")
            if os.path.exists(rolling_file + '.bak'):
                print(f"   旧滚动文件备份: {rolling_file}.bak")
    else:
        print("\n⚠️  无需迁移或迁移跳过。")


if __name__ == "__main__":
    main()
