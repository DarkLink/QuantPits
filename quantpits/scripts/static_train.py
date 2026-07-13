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

import sys
import json
import argparse

from quantpits.utils import env

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


# ================= 信息命令 =================
def show_state(ctx=None):
    """显示运行状态"""
    ctx = ctx or env.get_workspace_context()
    path = ctx.data_path("run_state.json")
    if not path.is_file():
        print("ℹ️  没有找到运行状态文件")
        return
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, default=str))


# ================= 主入口 =================
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
        show_state(ctx)
        return

    if args.clear_state:
        env.safeguard("Static Train: clear state", workspace_root=ctx.root)
        from quantpits.training.state import TrainingStateRepository
        TrainingStateRepository(ctx.data_path("run_state.json")).clear()
        return

    from quantpits.training.command import (
        options_from_namespace, prepare_training_run, prepared_plan_json, render_prepared_plan,
    )
    from quantpits.training.errors import TrainingCommandError
    from quantpits.training.service import TrainingExecutionService, default_execution_hooks
    from quantpits.utils.train_utils import calculate_dates
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
        service = TrainingExecutionService(default_execution_hooks(
            activate_workspace=env.set_root_dir,
            init_qlib=env.init_qlib,
            calculate_dates=calculate_dates,
        ))
        summary = service.execute(prepared)
        if not args.predict_only:
            try:
                from quantpits.scripts.deep_analysis.promote_config import update_promote_status
                published = [
                    item["key"].rsplit("@", 1)[0]
                    for item in summary.outcomes if item.get("published") is True
                ]
                if published:
                    update_promote_status(str(ctx.root), model_names=published)
            except Exception as exc:
                print("Warning: promote status update failed (%s)" % type(exc).__name__, file=sys.stderr)
        return 0
    except TrainingCommandError as exc:
        print("❌ %s" % exc, file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
