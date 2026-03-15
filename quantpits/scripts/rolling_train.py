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
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(SCRIPT_DIR)


# ================= 日期工具 =================
def parse_step_to_relativedelta(step_str):
    """将 '3M' / '1Y' 等解析为 relativedelta"""
    step_str = step_str.strip().upper()
    if step_str.endswith('M'):
        return relativedelta(months=int(step_str[:-1]))
    elif step_str.endswith('Y'):
        return relativedelta(years=int(step_str[:-1]))
    else:
        raise ValueError(f"Invalid step format: '{step_str}'. Use e.g. 3M or 1Y")


def generate_rolling_windows(rolling_start, train_years, valid_years,
                             test_step, anchor_date):
    """
    生成所有 rolling windows 的日期范围。

    Window n:
      Train: [T + n*Z,       T + X + n*Z − 1d]
      Valid: [T + X + n*Z,   T + X + Y + n*Z − 1d]
      Test:  [T + X + Y + n*Z, T + X + Y + (n+1)*Z − 1d]

    最后一个 window 的 test_end = anchor_date（不足完整 Z 也做）。

    Args:
        rolling_start: str, "2020-01-01"
        train_years: int, X
        valid_years: int, Y
        test_step: str, Z ("3M", "6M", "1Y")
        anchor_date: str, 最新日期

    Returns:
        list of dict: [{
            'window_idx': int,
            'train_start': str, 'train_end': str,
            'valid_start': str, 'valid_end': str,
            'test_start': str,  'test_end': str,
        }]
    """
    T = datetime.strptime(rolling_start, "%Y-%m-%d")
    X = relativedelta(years=train_years)
    Y = relativedelta(years=valid_years)
    Z = parse_step_to_relativedelta(test_step)
    anchor = datetime.strptime(anchor_date, "%Y-%m-%d")

    # Convert Z to total months for easy multiplication
    z_months = (Z.years or 0) * 12 + (Z.months or 0)

    windows = []
    n = 0

    while True:
        nZ = relativedelta(months=n * z_months)
        n1Z = relativedelta(months=(n + 1) * z_months)

        train_start = T + nZ
        train_end = T + X + nZ - timedelta(days=1)
        valid_start = T + X + nZ
        valid_end = T + X + Y + nZ - timedelta(days=1)
        test_start = T + X + Y + nZ
        test_end_full = T + X + Y + n1Z - timedelta(days=1)

        # 若 test_start 已超过 anchor_date，停止
        if test_start > anchor:
            break

        # 最后一个 window: test_end = min(test_end_full, anchor_date)
        test_end = min(test_end_full, anchor)

        windows.append({
            'window_idx': n,
            'train_start': train_start.strftime("%Y-%m-%d"),
            'train_end': train_end.strftime("%Y-%m-%d"),
            'valid_start': valid_start.strftime("%Y-%m-%d"),
            'valid_end': valid_end.strftime("%Y-%m-%d"),
            'test_start': test_start.strftime("%Y-%m-%d"),
            'test_end': test_end.strftime("%Y-%m-%d"),
        })

        # 若已到 anchor，不再继续
        if test_end >= anchor:
            break

        n += 1

    return windows


# ================= 状态管理 =================
class RollingState:
    """Rolling 训练状态管理，支持断点恢复"""

    def __init__(self, state_file=None):
        from train_utils import ROLLING_STATE_FILE
        self.state_file = state_file or ROLLING_STATE_FILE
        self._state = self._load()

    def _load(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        return self._empty()

    def _empty(self):
        return {
            'started_at': None,
            'rolling_config': {},
            'anchor_date': None,
            'completed_windows': {},  # {window_idx_str: {model_name: record_id}}
            'current_window_idx': None,
            'current_model': None,
            'total_windows': 0,
        }

    def save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self._state, f, indent=4)

    def init_run(self, rolling_config, anchor_date, total_windows):
        self._state = self._empty()
        self._state['started_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._state['rolling_config'] = rolling_config
        self._state['anchor_date'] = anchor_date
        self._state['total_windows'] = total_windows
        self.save()

    def is_window_model_done(self, window_idx, model_name):
        key = str(window_idx)
        return key in self._state['completed_windows'] and \
               model_name in self._state['completed_windows'][key]

    def mark_window_model_done(self, window_idx, model_name, record_id):
        key = str(window_idx)
        if key not in self._state['completed_windows']:
            self._state['completed_windows'][key] = {}
        self._state['completed_windows'][key][model_name] = record_id
        self._state['current_window_idx'] = window_idx
        self._state['current_model'] = model_name
        self.save()

    def get_completed_record_ids(self, model_name):
        """获取某模型在所有已完成 windows 的 record_ids (按 window_idx 排序)"""
        result = []
        for key in sorted(self._state['completed_windows'].keys(), key=int):
            models = self._state['completed_windows'][key]
            if model_name in models:
                result.append({
                    'window_idx': int(key),
                    'record_id': models[model_name],
                })
        return result

    def get_all_completed_windows(self):
        return self._state['completed_windows']

    @property
    def anchor_date(self):
        return self._state.get('anchor_date')

    def clear(self):
        if os.path.exists(self.state_file):
            from train_utils import backup_file_with_date
            backup_file_with_date(self.state_file, prefix="rolling_state")
            os.remove(self.state_file)
            print("🗑️  Rolling 状态已清除")

    def show(self):
        if not self._state.get('started_at'):
            print("ℹ️  没有找到 Rolling 运行状态")
            return

        s = self._state
        print("\n📋 Rolling 运行状态:")
        print(f"  开始时间: {s.get('started_at', 'N/A')}")
        print(f"  锚点日期: {s.get('anchor_date', 'N/A')}")
        print(f"  总 Windows: {s.get('total_windows', 0)}")

        completed = s.get('completed_windows', {})
        n_completed_windows = len(completed)
        total_models = sum(len(m) for m in completed.values())
        print(f"  已完成 Windows: {n_completed_windows}")
        print(f"  已完成 模型×Window: {total_models}")

        if completed:
            for widx in sorted(completed.keys(), key=int):
                models = completed[widx]
                print(f"    Window {widx}: {list(models.keys())}")

        cw = s.get('current_window_idx')
        cm = s.get('current_model')
        if cw is not None:
            print(f"  最后完成: Window {cw}, 模型 {cm}")


# ================= 训练单 Window =================
def train_window_model(model_name, yaml_file, window, params_base,
                       experiment_name, no_pretrain=False):
    """
    训练单个模型在一个 rolling window 上

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
    from train_utils import inject_config, ROLLING_PREDICTION_DIR
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

            # 保存 per-window CSV
            os.makedirs(ROLLING_PREDICTION_DIR, exist_ok=True)
            pred_file = os.path.join(
                ROLLING_PREDICTION_DIR,
                f"{model_name}_w{widx}_{window['test_start']}_{window['test_end']}.csv"
            )
            pred.to_csv(pred_file)
            print(f"[{model_name}|W{widx}] Per-window CSV: {pred_file}")

            # Signal Record
            record_cfgs = task_config['task'].get('record', [])
            for r_cfg in record_cfgs:
                if r_cfg['kwargs'].get('model') == '<MODEL>':
                    r_cfg['kwargs']['model'] = model
                if r_cfg['kwargs'].get('dataset') == '<DATASET>':
                    r_cfg['kwargs']['dataset'] = dataset
                r_obj = init_instance_by_config(r_cfg, recorder=recorder)
                r_obj.generate()

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

            result['success'] = True
            result['record_id'] = rid
            result['performance'] = performance

    except Exception as e:
        result['error'] = str(e)
        print(f"!!! Error in rolling train {model_name} W{widx}: {e}")
        import traceback
        traceback.print_exc()

    return result


# ================= 预测拼接 =================
def concatenate_rolling_predictions(state, model_names, rolling_exp_name,
                                    combined_exp_name, anchor_date):
    """
    将各 window 的 pred.pkl 拼接成完整时间序列。

    对每个模型：
    1. 从各 window recorder 加载 pred.pkl (仅 test 段)
    2. pd.concat (日期不重叠)
    3. 保存到 Rolling_Combined 实验的新 recorder
    4. 保存拼接后的 CSV

    Args:
        state: RollingState
        model_names: 要拼接的模型名列表
        rolling_exp_name: per-window 实验名
        combined_exp_name: 拼接后实验名
        anchor_date: 锚点日期

    Returns:
        dict: {model_name: combined_record_id}
    """
    from qlib.workflow import R
    from train_utils import ROLLING_PREDICTION_DIR
    import pandas as pd

    print(f"\n{'='*60}")
    print("📦 拼接 Rolling 预测")
    print(f"{'='*60}")

    combined_records = {}

    for model_name in model_names:
        completions = state.get_completed_record_ids(model_name)
        if not completions:
            print(f"  [{model_name}] 无已完成的 window, 跳过")
            continue

        print(f"\n  [{model_name}] 拼接 {len(completions)} 个 windows...")

        all_preds = []
        for comp in completions:
            try:
                rec = R.get_recorder(
                    recorder_id=comp['record_id'],
                    experiment_name=rolling_exp_name
                )
                pred = rec.load_object("pred.pkl")
                all_preds.append(pred)
                dates = pred.index.get_level_values('datetime')
                print(f"    Window {comp['window_idx']}: "
                      f"{dates.min().date()} ~ {dates.max().date()}, "
                      f"{len(pred)} rows")
            except Exception as e:
                print(f"    Window {comp['window_idx']}: FAILED - {e}")

        if not all_preds:
            print(f"  [{model_name}] 无有效预测数据")
            continue

        # 拼接
        combined_pred = pd.concat(all_preds)
        # 去重（以防万一）
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

        # 保存拼接后的 CSV
        csv_path = os.path.join(
            ROLLING_PREDICTION_DIR,
            f"{model_name}_{anchor_date}.csv"
        )
        combined_pred.to_csv(csv_path)
        print(f"  [{model_name}] Combined CSV: {csv_path}")
        print(f"  [{model_name}] Combined Recorder: {combined_rid}")

    return combined_records


def save_rolling_records(combined_records, combined_exp_name, anchor_date):
    """
    保存 latest_rolling_records.json

    格式与 latest_train_records.json 一致，以便下游直接 --record-file 读取。
    """
    from train_utils import ROLLING_RECORD_FILE, backup_file_with_date

    records = {
        "experiment_name": combined_exp_name,
        "anchor_date": anchor_date,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "rolling",
        "models": combined_records,
    }

    backup_file_with_date(ROLLING_RECORD_FILE, prefix="rolling_records")

    with open(ROLLING_RECORD_FILE, 'w') as f:
        json.dump(records, f, indent=4)

    print(f"\n📋 Rolling 记录已保存: {ROLLING_RECORD_FILE}")
    print(f"   模型数: {len(combined_records)}")
    for name, rid in combined_records.items():
        print(f"   {name}: {rid}")


# ================= 日常预测 =================
def predict_with_latest_model(model_name, model_info, state,
                              rolling_exp_name, params_base, anchor_date):
    """
    使用最近一个 window 训练的模型对最新数据预测。

    用于日常模式中距离上次 rolling 未超过 step 的情况。
    """
    from train_utils import inject_config, ROLLING_PREDICTION_DIR
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    completions = state.get_completed_record_ids(model_name)
    if not completions:
        print(f"  [{model_name}] 无历史 rolling 模型，需要先 --cold-start")
        return None

    # 取最新 window 的模型
    latest = completions[-1]
    print(f"  [{model_name}] 加载 Window {latest['window_idx']} 模型进行预测...")

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
        task_config = inject_config(yaml_file, params, model_name=model_name)

        dataset_cfg = task_config['task']['dataset']
        dataset = init_instance_by_config(dataset_cfg)

        pred = model.predict(dataset=dataset)

        os.makedirs(ROLLING_PREDICTION_DIR, exist_ok=True)
        pred_file = os.path.join(
            ROLLING_PREDICTION_DIR,
            f"{model_name}_predict_{anchor_date}.csv"
        )
        pred.to_csv(pred_file)
        print(f"  [{model_name}] 预测完成: {pred_file}")

        return pred

    except Exception as e:
        print(f"  [{model_name}] 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    """解析目标模型列表"""
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
    elif args.algorithm or args.dataset or args.tag:
        targets = get_models_by_filter(
            registry,
            algorithm=args.algorithm,
            dataset=args.dataset,
            tag=args.tag,
        )
    else:
        return None

    if args.skip:
        skip_names = [m.strip() for m in args.skip.split(',')]
        targets = {k: v for k, v in targets.items() if k not in skip_names}

    return targets


def get_base_params():
    """获取 workspace 基础参数 (market, benchmark 等)"""
    from config_loader import load_workspace_config
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
    """冷启动：所有 windows 训练 + 拼接"""
    from train_utils import print_model_table

    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()

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
    if getattr(args, 'resume', False) is not True and getattr(args, 'merge', False) is not True:
        state.init_run(rolling_cfg, anchor_date, len(windows))
    else:
        if not state.anchor_date:
            print("❌ 无 rolling 状态可恢复，将新建状态")
            state.init_run(rolling_cfg, anchor_date, len(windows))
        else:
            print(f"⏩ {'Merge' if getattr(args, 'merge', False) is True else 'Resume'} 模式：跳过已完成窗格")

    rolling_exp_name = f"Rolling_Windows_{freq}"
    combined_exp_name = f"Rolling_Combined_{freq}"

    # 训练每个 window × 每个 model
    total_tasks = len(windows) * len(targets)
    done_count = 0

    for window in windows:
        widx = window['window_idx']
        for model_name, model_info in targets.items():
            done_count += 1
            tag = f"[{done_count}/{total_tasks}]"

            if state.is_window_model_done(widx, model_name):
                print(f"\n{tag} ⏩ {model_name} W{widx} 已完成，跳过")
                continue

            print(f"\n{'─'*60}")
            print(f"  {tag} {model_name} | Window {widx}")
            print(f"{'─'*60}")

            result = train_window_model(
                model_name=model_name,
                yaml_file=model_info['yaml_file'],
                window=window,
                params_base=params_base,
                experiment_name=rolling_exp_name,
                no_pretrain=args.no_pretrain,
            )

            if result['success']:
                state.mark_window_model_done(widx, model_name, result['record_id'])
            else:
                print(f"❌ {model_name} W{widx} 训练失败: {result.get('error', 'Unknown')}")

    # 拼接预测
    if getattr(args, 'merge', False) is True:
        completed = state.get_all_completed_windows()
        all_models = set(targets.keys())
        for win, models in completed.items():
            all_models.update(models.keys())
        model_names = list(all_models)
    else:
        model_names = list(targets.keys())

    combined_records = concatenate_rolling_predictions(
        state, model_names, rolling_exp_name, combined_exp_name, anchor_date
    )

    if combined_records:
        save_rolling_records(combined_records, combined_exp_name, anchor_date)
        if getattr(args, 'backtest', False) is True:
            run_combined_backtest(model_names, combined_records, combined_exp_name, params_base)

    # 完成
    print(f"\n{'='*60}")
    print("✅ Rolling 冷启动完成")
    print(f"{'='*60}")
    print(f"  Windows: {len(windows)}")
    print(f"  Models: {len(targets)}")
    print(f"  Combined records: {len(combined_records)}")
    print(f"\n  💡 后续步骤:")
    print(f"     穷举: python quantpits/scripts/brute_force_fast.py "
          f"--record-file latest_rolling_records.json")
    print(f"     融合: python quantpits/scripts/ensemble_fusion.py "
          f"--from-config --record-file latest_rolling_records.json")
    print(f"{'='*60}")


def run_daily(args, targets, rolling_cfg):
    """日常模式：检测是否需要新 window 训练"""
    env.init_qlib()
    params_base = get_base_params()
    anchor_date = params_base['anchor_date']
    freq = params_base['freq'].upper()

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

        # 训练新 windows
        for window in new_windows:
            widx = window['window_idx']
            for model_name, model_info in targets.items():
                if state.is_window_model_done(widx, model_name):
                    continue

                result = train_window_model(
                    model_name=model_name,
                    yaml_file=model_info['yaml_file'],
                    window=window,
                    params_base=params_base,
                    experiment_name=rolling_exp_name,
                    no_pretrain=args.no_pretrain,
                )

                if result['success']:
                    state.mark_window_model_done(widx, model_name, result['record_id'])
                else:
                    print(f"❌ {model_name} W{widx} 失败: {result.get('error')}")

        # 重新拼接
        model_names = list(targets.keys())
        combined_records = concatenate_rolling_predictions(
            state, model_names, rolling_exp_name, combined_exp_name, anchor_date
        )
        if combined_records:
            save_rolling_records(combined_records, combined_exp_name, anchor_date)
            if getattr(args, 'backtest', False) is True:
                run_combined_backtest(model_names, combined_records, combined_exp_name, params_base)

        print(f"\n✅ Rolling 滚动更新完成 (新训练 {len(new_windows)} 个 windows)")

    else:
        print(f"\n📊 所有 windows 已训练完毕，执行 predict-only...")

        for model_name, model_info in targets.items():
            predict_with_latest_model(
                model_name, model_info, state,
                rolling_exp_name, params_base, anchor_date
            )


def run_combined_backtest(model_names, combined_records, combined_exp_name, params_base):
    """
    对合并后的预测执行回测，并将回测结果的 port_analysis 等指标保存追加回相应的记录。
    """
    from qlib.workflow import R
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    import strategy
    import numpy as np
    import warnings
    
    print(f"\n{'='*60}")
    print("📈 运行 Rolling 合并预测的回测")
    print(f"{'='*60}")
    
    st_config = strategy.load_strategy_config()
    bt_config = strategy.get_backtest_config(st_config)
    
    for model_name in model_names:
        if model_name not in combined_records:
            continue
            
        record_id = combined_records[model_name]
        print(f"\n  [{model_name}] 提取合并预测以进行回测 (Record: {record_id})...")
        
        try:
            rec = R.get_recorder(recorder_id=record_id, experiment_name=combined_exp_name)
            pred = rec.load_object("pred.pkl")
            
            if pred is None or pred.empty:
                print(f"  [{model_name}] 预测为空，跳过回测。")
                continue
                
            bt_start = str(pred.index.get_level_values(0).min().date())
            bt_end = str(pred.index.get_level_values(0).max().date())
            
            print(f"  [{model_name}] Backtest Range: {bt_start} ~ {bt_end}")
            
            # Create Strategy
            strategy_inst = strategy.create_backtest_strategy(pred, st_config)
            
            # Create Executor
            executor_obj = SimulatorExecutor(
                time_per_step=params_base['freq'],
                generate_portfolio_metrics=True,
                verbose=False
            )
            
            print(f"  [{model_name}] 执行回测...")
            with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw_portfolio_metrics, raw_indicators = backtest(
                    executor=executor_obj,
                    strategy=strategy_inst,
                    start_time=bt_start,
                    end_time=bt_end,
                    account=bt_config['account'],
                    benchmark=params_base['benchmark'],
                    exchange_kwargs=bt_config['exchange_kwargs']
                )
            
            # Use PortfolioAnalyzer to get traditional metrics
            from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
            import pandas as pd
            from qlib.data import D
            
            def extract_report_df(metrics):
                if isinstance(metrics, dict):
                    val = list(metrics.values())[0]
                    return val[0] if isinstance(val, tuple) else val
                elif isinstance(metrics, tuple):
                    first = metrics[0]
                    if isinstance(first, pd.DataFrame):
                        return first
                    elif isinstance(first, tuple) and len(first) >= 1:
                        return first[0]
                    return metrics
                return metrics
                
            report_df = extract_report_df(raw_portfolio_metrics)
            if report_df is None or report_df.empty:
                 print(f"  [{model_name}] 提取回测结果失败。")
                 continue
                 
            # Format report DataFrame
            da_df = pd.DataFrame(index=report_df.index)
            da_df['收盘价值'] = report_df['account']
            da_df[params_base['benchmark']] = (1 + report_df['bench']).cumprod()
            if not isinstance(da_df.index, pd.DatetimeIndex):
                da_df.index = pd.to_datetime(da_df.index)
            
            bt_start_dt = da_df.index.min()
            bt_end_dt = da_df.index.max()
            daily_dates = D.calendar(start_time=bt_start_dt, end_time=bt_end_dt, freq='day')
            da_df = da_df.reindex(daily_dates, method='ffill').dropna(subset=['收盘价值'])
            da_df = da_df.reset_index().rename(columns={'index': '成交日期', 'datetime': '成交日期'})
            
            pa = PortfolioAnalyzer(
                daily_amount_df=da_df, 
                trade_log_df=pd.DataFrame(), 
                holding_log_df=pd.DataFrame(),
                benchmark_col=params_base['benchmark'], 
                freq=params_base['freq']
            )
            metrics = pa.calculate_traditional_metrics()
            
            ann_ret = metrics.get('CAGR', 0)
            max_dd = metrics.get('Max_Drawdown', 0)
            excess = metrics.get('Excess_Return_CAGR', 0)
            ir = metrics.get('Information_Ratio', 0)
            calmar = metrics.get('Calmar', 0)
            
            print(f"  [{model_name}] 回测完成! Ann_Ret: {ann_ret:.2%}, Excess: {excess:.2%}, Max_DD: {max_dd:.2%}, IR: {ir:.3f}")
            
            # Save objects back to the same recorder
            # By calling methods directly on the existing `rec` object
            try:
                rec.log_metrics(
                    Ann_Ret=ann_ret,
                    Max_DD=max_dd,
                    Excess_Return=excess,
                    Information_Ratio=ir,
                    Calmar=calmar
                )
                
                # 严格按照 Qlib PortAnaRecord 的保存格式，把报告分离并保存到 portfolio_analysis 子目录下
                port_ana_objs = {}
                if isinstance(raw_portfolio_metrics, dict):
                    for freq_key, metrics_tuple in raw_portfolio_metrics.items():
                        if isinstance(metrics_tuple, tuple) and len(metrics_tuple) >= 2:
                            port_ana_objs[f"report_normal_{freq_key}.pkl"] = metrics_tuple[0]
                            port_ana_objs[f"positions_normal_{freq_key}.pkl"] = metrics_tuple[1]
                        elif isinstance(metrics_tuple, tuple) and len(metrics_tuple) == 1:
                            port_ana_objs[f"report_normal_{freq_key}.pkl"] = metrics_tuple[0]

                if port_ana_objs:
                    rec.save_objects(artifact_path="portfolio_analysis", **port_ana_objs)
                
                # 指标分析保存到 sig_analysis 子目录（依照 SigAnaRecord）或者根目录
                rec.save_objects(artifact_path="sig_analysis", **{
                    f"indicator_analysis_{params_base['freq']}.pkl": raw_indicators
                })
            except Exception as log_e:
                print(f"  [{model_name}] MLflow 记录失败，可能已存在同名 metric: {log_e}")
                
        except Exception as e:
            print(f"  [{model_name}] 回测过程失败: {e}")
            import traceback
            traceback.print_exc()


def run_backtest_only(args, targets):
    """仅回测模式：读取 latest_rolling_records.json 直接运行"""
    import os, json
    from train_utils import ROLLING_RECORD_FILE
    env.init_qlib()
    params_base = get_base_params()
    
    if os.path.exists(ROLLING_RECORD_FILE):
        with open(ROLLING_RECORD_FILE, 'r') as f:
            records = json.load(f)
    else:
        records = None
        
    if not records or "models" not in records:
        print("❌ 找不到有效的 latest_rolling_records.json 或内容为空。")
        return
        
    combined_exp_name = records.get("experiment_name")
    if not combined_exp_name:
        freq = params_base['freq'].upper()
        combined_exp_name = f"Rolling_Combined_{freq}"
        
    combined_records = records["models"]
    model_names = []
    
    for m in targets.keys():
        if m in combined_records:
            model_names.append(m)
            
    if not model_names:
        print("❌ 选定的模型中没有找到历史滚动预测记录。")
        return
        
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

    rolling_exp_name = f"Rolling_Windows_{freq}"

    for model_name, model_info in targets.items():
        predict_with_latest_model(
            model_name, model_info, state,
            rolling_exp_name, params_base, anchor_date
        )


def main():
    args = parse_args()

    # 信息查看命令
    if args.show_state:
        RollingState().show()
        return

    if args.clear_state:
        RollingState().clear()
        return

    # 加载 rolling 配置
    from config_loader import load_rolling_config
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

    # 解析目标模型
    has_selection = any([
        args.models, args.algorithm, args.dataset,
        args.tag, args.all_enabled
    ])

    if getattr(args, 'resume', False) is True or getattr(args, 'merge', False) is True or getattr(args, 'backtest_only', False) is True:
        if not has_selection:
            args.all_enabled = True
            has_selection = True

    if not has_selection:
        print("❌ 请指定要训练的模型")
        print("   使用 --models, --algorithm, --dataset, --tag, 或 --all-enabled")
        return

    if getattr(args, 'resume', False) is True or getattr(args, 'merge', False) is True:
        state = RollingState()
        if not state.anchor_date and getattr(args, 'resume', False) is True:
            print("❌ 无 rolling 状态可恢复")
            return

    targets = resolve_target_models(args)
    if targets is None or not targets:
        print("⚠️  没有匹配的模型")
        return

    # 选择运行模式
    if getattr(args, 'backtest_only', False) is True:
        run_backtest_only(args, targets)
    elif args.predict_only:
        run_predict_only(args, targets, rolling_cfg)
    elif args.cold_start or getattr(args, 'resume', False) is True or getattr(args, 'merge', False) is True:
        run_cold_start(args, targets, rolling_cfg)
    else:
        run_daily(args, targets, rolling_cfg)


if __name__ == "__main__":
    main()
