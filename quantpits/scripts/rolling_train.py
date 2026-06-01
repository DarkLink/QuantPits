#!/usr/bin/env python
"""
Rolling Training Script (滚动训练)
独立于静态训练的滚动训练逻辑，支持冷启动、日常滚动、断点恢复。

运行方式：cd QuantPits && python quantpits/scripts/rolling_train.py [options]

模式说明：
  冷启动：从 rolling_start(T) 生成所有 windows，全部训练+预测+拼接
  日常模式：检测距上次 rolling 是否超过 step，是则滚动训练新 window，否则仅预测

示例：
  # 冷启动（首次运行必须）
  python quantpits/scripts/rolling_train.py --cold-start --all-enabled

  # 冷启动指定模型
  python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158

  # 日常模式（自动判断训练/预测）
  python quantpits/scripts/rolling_train.py --all-enabled

  # 强制仅预测
  python quantpits/scripts/rolling_train.py --predict-only --all-enabled

  # Dry-run（仅显示 windows 划分）
  python quantpits/scripts/rolling_train.py --cold-start --dry-run --models linear_Alpha158

  # 查看状态
  python quantpits/scripts/rolling_train.py --show-state

  # 断点恢复
  python quantpits/scripts/rolling_train.py --resume

  # 重训最后一个 window
  python quantpits/scripts/rolling_train.py --retrain-last --all-enabled
"""

import os
import sys
import argparse

from quantpits.utils import env
from quantpits.utils.constants import MONTHS_PER_YEAR
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(SCRIPT_DIR)

# Sub-modules
from quantpits.scripts.rolling.windows import (
    generate_rolling_windows,
    parse_step_to_relativedelta,
)
from quantpits.scripts.rolling.state import RollingState
from quantpits.scripts.rolling.training import (
    run_model_windows,
    train_window_model,
    train_window_model_isolated,
)
from quantpits.scripts.rolling.prediction import (
    concatenate_rolling_predictions,
    save_rolling_records,
    predict_with_latest_model,
    _filter_pred_to_test_segment,
)
from quantpits.scripts.rolling.backtest import (
    run_combined_backtest,
    run_backtest_only,
)
from quantpits.scripts.rolling.memory import (
    deep_cleanup_after_model,
    log_memory,
)


# ================= CLI =================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Rolling 训练：滚动时间窗口训练 + 预测拼接',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --cold-start --all-enabled           # 冷启动所有 enabled 模型
  %(prog)s --cold-start --models linear_Alpha158  # 冷启动指定模型
  %(prog)s --all-enabled                         # 日常模式
  %(prog)s --predict-only --all-enabled          # 仅预测
  %(prog)s --cold-start --dry-run --all-enabled  # 查看 windows 划分
  %(prog)s --show-state                          # 查看状态
  %(prog)s --resume                              # 断点恢复
        """
    )

    mode = parser.add_argument_group('运行模式')
    mode.add_argument('--cold-start', action='store_true',
                      help='冷启动：生成所有 windows 并训练')
    mode.add_argument('--predict-only', action='store_true',
                      help='仅使用最新模型预测，不训练')
    mode.add_argument('--resume', action='store_true',
                      help='从断点恢复训练')
    mode.add_argument('--merge', action='store_true',
                      help='合并冷启动：在已有 rolling 状态基础上追加新模型')
    mode.add_argument('--retrain-last', action='store_true',
                      help='重训最后一个 window：清除 state 中最后 window 的记录后走日常流程')
    mode.add_argument('--backtest', action='store_true',
                      help='训练拼接完成后，对产出的合成预测进行全量回测')
    mode.add_argument('--backtest-only', action='store_true',
                      help='仅对 latest_rolling_records.json 中的模型进行回测 (跳过训练预测)')

    select = parser.add_argument_group('模型选择')
    select.add_argument('--models', type=str,
                        help='指定模型名，逗号分隔')
    select.add_argument('--algorithm', type=str,
                        help='按算法筛选')
    select.add_argument('--dataset', type=str,
                        help='按数据集筛选')
    select.add_argument('--tag', type=str,
                        help='按标签筛选')
    select.add_argument('--all-enabled', action='store_true',
                        help='所有 enabled 模型')
    select.add_argument('--skip', type=str,
                        help='跳过指定模型，逗号分隔')

    ctrl = parser.add_argument_group('运行控制')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='仅显示 windows 划分，不训练')
    ctrl.add_argument('--no-pretrain', action='store_true',
                      help='不加载预训练模型')

    info = parser.add_argument_group('信息查看')
    info.add_argument('--show-state', action='store_true',
                      help='显示 rolling 状态')
    info.add_argument('--clear-state', action='store_true',
                      help='清除 rolling 状态')

    return parser.parse_args()


def resolve_target_models(args):
    """解析目标模型列表（委托给 train_utils 共享实现）"""
    from quantpits.utils.train_utils import resolve_target_models as _resolve
    return _resolve(args)


def get_base_params():
    """获取 workspace 基础参数 (market, benchmark 等)"""
    from quantpits.utils.config_loader import load_workspace_config
    config = load_workspace_config(ROOT_DIR)

    from qlib.data import D
    last_trade_date = D.calendar(future=False)[-1:][0]
    anchor_date = last_trade_date.strftime('%Y-%m-%d')

    return {
        'market': config.get('market', 'csi300'),
        'benchmark': config.get('benchmark', 'SH000300'),
        'topk': config.get('topk', 20),
        'n_drop': config.get('n_drop', 3),
        'buy_suggestion_factor': config.get('buy_suggestion_factor', 2),
        'account': config.get('current_full_cash', 100000.0),
        'freq': config.get('freq', 'week').lower(),
        'anchor_date': anchor_date,
    }


# ================= 主流程 =================
def run_cold_start(args, targets, rolling_cfg):
    """
    冷启动：Model-First 循环 + 子进程隔离。

    外循环遍历模型，内循环遍历 windows，每个 task 在独立子进程中执行。
    每个模型完成后立即拼接预测并深度清理内存。
    """
    from quantpits.utils.train_utils import print_model_table
    from qlib.config import C

    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    qlib_config = C  # 捕获当前 qlib 配置，传给子进程

    # 生成 windows
    windows = generate_rolling_windows(
        rolling_start=rolling_cfg['rolling_start'],
        train_years=rolling_cfg['train_years'],
        valid_years=rolling_cfg['valid_years'],
        test_step=rolling_cfg['test_step'],
        anchor_date=anchor_date,
    )

    if not windows:
        print("❌ 无法生成任何 rolling window — 请检查 rolling_config.yaml")
        print(f"   rolling_start + train_years + valid_years 是否晚于 anchor_date ({anchor_date})?")
        return

    # 打印 windows
    print(f"\n{'='*70}")
    print(f"📅 Rolling Windows ({len(windows)} 个)")
    print(f"{'='*70}")
    for w in windows:
        print(f"  Window {w['window_idx']:2d}: "
              f"Train[{w['train_start']}, {w['train_end']}] "
              f"Valid[{w['valid_start']}, {w['valid_end']}] "
              f"Test[{w['test_start']}, {w['test_end']}]")
    print(f"{'='*70}")

    print_model_table(targets, title="Rolling 训练模型")

    if args.dry_run:
        print("🔍 Dry-run 模式：以上为 windows 划分，不实际训练")
        return

    # 初始化状态
    state = RollingState()
    if not args.resume and not args.merge:
        state.init_run(rolling_cfg, anchor_date, len(windows))
    else:
        if not state.anchor_date:
            print("❌ 无 rolling 状态可恢复，将新建状态")
            state.init_run(rolling_cfg, anchor_date, len(windows))
        else:
            print(f"⏩ {'Merge' if args.merge else 'Resume'} 模式：跳过已完成窗格")

    rolling_exp_name = f"Rolling_Windows_{freq}"
    combined_exp_name = f"Rolling_Combined_{freq}"

    # ===== Model-First 循环 =====
    combined_records = {}
    total_trained = 0

    for model_name, model_info in targets.items():
        # 训练该模型的所有 windows (子进程隔离)
        n_trained = run_model_windows(
            model_name=model_name,
            model_info=model_info,
            windows=windows,
            state=state,
            params_base=params_base,
            experiment_name=rolling_exp_name,
            qlib_config=qlib_config,
            no_pretrain=args.no_pretrain,
            dry_run=args.dry_run,
        )
        total_trained += n_trained

        # 该模型所有 windows 完成后，立即拼接
        model_combined = concatenate_rolling_predictions(
            state=state,
            model_names=[model_name],
            rolling_exp_name=rolling_exp_name,
            combined_exp_name=combined_exp_name,
            anchor_date=anchor_date,
            windows=windows,
        )
        combined_records.update(model_combined)

        # 深度清理: qlib MemCache + GPU + GC
        deep_cleanup_after_model(model_name)

    # Merge 模式: 需要拼接之前已训练但不在本次 targets 中的模型
    if args.merge:
        completed = state.get_all_completed_windows()
        extra_models = set()
        for win, models in completed.items():
            extra_models.update(models.keys())
        extra_models -= set(targets.keys())
        if extra_models:
            extra_combined = concatenate_rolling_predictions(
                state=state,
                model_names=list(extra_models),
                rolling_exp_name=rolling_exp_name,
                combined_exp_name=combined_exp_name,
                anchor_date=anchor_date,
                windows=windows,
            )
            combined_records.update(extra_combined)

    # 保存记录
    if combined_records:
        save_rolling_records(combined_records, combined_exp_name, anchor_date)
        if args.backtest:
            run_combined_backtest(
                list(combined_records.keys()), combined_records,
                combined_exp_name, params_base
            )

    # 完成
    print(f"\n{'='*60}")
    print("✅ Rolling 冷启动完成")
    print(f"{'='*60}")
    print(f"  Windows: {len(windows)}")
    print(f"  Models: {len(targets)}")
    print(f"  Trained: {total_trained} tasks")
    print(f"  Combined records: {len(combined_records)}")
    print(f"\n  💡 后续步骤:")
    print(f"     穷举: python quantpits/scripts/brute_force_fast.py "
          f"--record-file latest_rolling_records.json")
    print(f"     融合: python quantpits/scripts/ensemble_fusion.py "
          f"--from-config --record-file latest_rolling_records.json")
    print(f"{'='*60}")


def run_daily(args, targets, rolling_cfg):
    """
    日常模式：检测是否需要新 window 训练。
    同样使用 Model-First + 子进程隔离。
    """
    from qlib.config import C

    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()
    qlib_config = C

    state = RollingState()

    if not state.anchor_date:
        print("❌ 无 rolling 状态，请先运行 --cold-start")
        return

    # 生成到当前 anchor 的 windows
    windows = generate_rolling_windows(
        rolling_start=rolling_cfg['rolling_start'],
        train_years=rolling_cfg['train_years'],
        valid_years=rolling_cfg['valid_years'],
        test_step=rolling_cfg['test_step'],
        anchor_date=anchor_date,
    )

    # 检查有无新 window 需要训练
    completed = state.get_all_completed_windows()
    completed_indices = {int(k) for k in completed.keys()}
    new_windows = [w for w in windows if w['window_idx'] not in completed_indices]

    rolling_exp_name = f"Rolling_Windows_{freq}"
    combined_exp_name = f"Rolling_Combined_{freq}"

    if new_windows:
        print(f"\n🔄 检测到 {len(new_windows)} 个新 window 需要训练")
        for w in new_windows:
            print(f"  Window {w['window_idx']}: Test[{w['test_start']}, {w['test_end']}]")

        if args.dry_run:
            print("🔍 Dry-run 模式：以上为需训练的新 windows")
            return

        # Model-First: 训练新 windows
        combined_records = {}
        for model_name, model_info in targets.items():
            n_trained = run_model_windows(
                model_name=model_name,
                model_info=model_info,
                windows=new_windows,
                state=state,
                params_base=params_base,
                experiment_name=rolling_exp_name,
                qlib_config=qlib_config,
                no_pretrain=args.no_pretrain,
            )

            # 拼接该模型（包含所有已完成的 windows，不仅仅是新的）
            model_combined = concatenate_rolling_predictions(
                state=state,
                model_names=[model_name],
                rolling_exp_name=rolling_exp_name,
                combined_exp_name=combined_exp_name,
                anchor_date=anchor_date,
                windows=windows,
            )
            combined_records.update(model_combined)

            deep_cleanup_after_model(model_name)

        if combined_records:
            save_rolling_records(combined_records, combined_exp_name, anchor_date)
            if args.backtest:
                run_combined_backtest(
                    list(combined_records.keys()), combined_records,
                    combined_exp_name, params_base
                )

        print(f"\n✅ Rolling 滚动更新完成 (新训练 {len(new_windows)} 个 windows)")

    else:
        # 所有 window 已训练，使用最新模型对当前 anchor_date 范围预测
        print(f"\n📊 所有 windows 已训练完毕，执行 predict-only...")

        extra_preds = {}
        for model_name, model_info in targets.items():
            pred = predict_with_latest_model(
                model_name, model_info, state,
                rolling_exp_name, params_base, anchor_date, windows=windows
            )
            if pred is not None and not pred.empty:
                extra_preds[model_name] = pred

        if extra_preds:
            model_names = list(targets.keys())
            combined_records = concatenate_rolling_predictions(
                state, model_names, rolling_exp_name, combined_exp_name, anchor_date,
                windows=windows, extra_preds=extra_preds,
            )
            if combined_records:
                save_rolling_records(combined_records, combined_exp_name, anchor_date)
                if args.backtest:
                    run_combined_backtest(model_names, combined_records, combined_exp_name, params_base)


def run_predict_only(args, targets, rolling_cfg):
    """仅预测模式"""
    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()

    state = RollingState()
    if not state.anchor_date:
        print("❌ 无 rolling 状态，请先运行 --cold-start")
        return

    # 为 predict-only 解析当前日期下的 windows (保证 latest window 的 test_end 达到 anchor_date)
    windows = generate_rolling_windows(
        rolling_start=rolling_cfg['rolling_start'],
        train_years=rolling_cfg['train_years'],
        valid_years=rolling_cfg['valid_years'],
        test_step=rolling_cfg['test_step'],
        anchor_date=anchor_date,
    )

    rolling_exp_name = f"Rolling_Windows_{freq}"
    extra_preds = {}

    for model_name, model_info in targets.items():
        pred = predict_with_latest_model(
            model_name, model_info, state,
            rolling_exp_name, params_base, anchor_date, windows=windows
        )
        if pred is not None and not pred.empty:
            extra_preds[model_name] = pred

    if extra_preds:
        combined_exp_name = f"Rolling_Combined_{freq}"
        model_names = list(targets.keys())
        combined_records = concatenate_rolling_predictions(
            state, model_names, rolling_exp_name, combined_exp_name, anchor_date,
            windows=windows, extra_preds=extra_preds,
        )
        if combined_records:
            save_rolling_records(combined_records, combined_exp_name, anchor_date)
            # 允许在 predict-only 后也触发 backtest
            if args.backtest:
                run_combined_backtest(model_names, combined_records, combined_exp_name, params_base)


def main():
    from quantpits.utils import env as _env
    _env.safeguard("Rolling Train")
    args = parse_args()

    # 信息查看命令
    if args.show_state:
        RollingState().show()
        return

    if args.clear_state:
        RollingState().clear()
        return

    # 加载 rolling 配置
    from quantpits.utils.config_loader import load_rolling_config
    rolling_cfg = load_rolling_config(ROOT_DIR)
    if rolling_cfg is None:
        print("❌ 找不到 config/rolling_config.yaml")
        print("   请先创建 rolling 配置文件")
        return

    print(f"\n📋 Rolling 配置:")
    print(f"   起点(T): {rolling_cfg['rolling_start']}")
    print(f"   训练(X): {rolling_cfg['train_years']} 年")
    print(f"   验证(Y): {rolling_cfg['valid_years']} 年")
    print(f"   步长(Z): {rolling_cfg['test_step']} ({rolling_cfg['test_step_months']} 个月)")

    # --retrain-last: 清除最后一个 window 的训练记录，然后走日常流程
    if args.retrain_last:
        state = RollingState()
        if not state.anchor_date:
            print("❌ 无 rolling 状态，请先 --cold-start")
            return
        last_idx = state.get_last_completed_window_idx()
        if last_idx is not None:
            state.remove_window(last_idx)
            print(f"🔄 已清除 Window {last_idx} 的训练记录，将重新训练")
        else:
            print("ℹ️  没有已完成的 window 可以重训")
            return

    # 解析目标模型
    has_selection = any([
        args.models, args.algorithm, args.dataset,
        args.tag, args.all_enabled
    ])

    if args.resume or args.merge or args.backtest_only or args.retrain_last:
        if not has_selection:
            args.all_enabled = True
            has_selection = True

    if not has_selection:
        print("❌ 请指定要训练的模型")
        print("   使用 --models, --algorithm, --dataset, --tag, 或 --all-enabled")
        return

    if args.resume or args.merge:
        state = RollingState()
        if not state.anchor_date and args.resume:
            print("❌ 无 rolling 状态可恢复")
            return

    targets = resolve_target_models(args)
    if targets is None or not targets:
        print("⚠️  没有匹配的模型")
        return

    from quantpits.utils.operator_log import OperatorLog
    with OperatorLog("rolling_train", args=sys.argv[1:]) as oplog:
        # 选择运行模式
        if args.backtest_only:
            env.init_qlib()
            params_base = get_base_params()
            run_backtest_only(args, targets, params_base)
        elif args.predict_only:
            run_predict_only(args, targets, rolling_cfg)
        elif args.cold_start or args.resume or args.merge:
            run_cold_start(args, targets, rolling_cfg)
        elif args.retrain_last:
            # --retrain-last 走日常流程（会检测到被清除的 window 并重训）
            run_daily(args, targets, rolling_cfg)
        else:
            run_daily(args, targets, rolling_cfg)

        oplog.set_result({
            "n_targets": len(targets),
            "cold_start": args.cold_start,
            "resume": args.resume,
            "predict_only": args.predict_only
        })

        # Update promote status: promoted_pending_retrain → active
        if not args.predict_only and not args.backtest_only:
            try:
                from quantpits.scripts.deep_analysis.promote_config import update_promote_status
                update_promote_status(ROOT_DIR, model_names=list(targets.keys()))
            except Exception:
                pass  # Non-critical — don't block main training flow


if __name__ == "__main__":
    main()
