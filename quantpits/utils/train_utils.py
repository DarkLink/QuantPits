#!/usr/bin/env python
"""
训练共享工具模块 (Train Utilities)
从生产训练逻辑中提取的共享工具，供全量训练和增量训练复用。

主要功能：
- 日期计算 (calculate_dates)
- YAML 参数注入 (inject_config)
- 模型注册表管理 (load_model_registry, get_models_by_filter)
- 单模型训练流程 (train_single_model)
- 历史备份 (backup_file_with_date)
- 训练记录合并 (merge_train_records)
- 运行状态管理 (save_run_state, load_run_state)
"""

import os
import json
import yaml
import shutil
from datetime import datetime, timedelta
from quantpits.utils.constants import TRADING_DAYS_PER_YEAR, TRADING_WEEKS_PER_YEAR, AVERAGE_CALENDAR_DAYS_PER_YEAR

# 注意: qlib 相关导入（D, init_instance_by_config, R）在需要的函数内部延迟导入，
# 这样 --list, --show-state 等不需要训练的命令可以在没有 qlib 的环境中运行。


# ================= 路径常量 =================
import gc
import traceback

from quantpits.utils import env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR

REGISTRY_FILE = os.path.join(ROOT_DIR, "config", "model_registry.yaml")
MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "model_config.json")
PROD_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "prod_config.json")
RECORD_OUTPUT_FILE = os.path.join(ROOT_DIR, "latest_train_records.json")
PREDICTION_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "predictions")
ROLLING_PREDICTION_DIR = os.path.join(ROOT_DIR, "output", "predictions", "rolling")
HISTORY_DIR = os.path.join(ROOT_DIR, "data", "history")
RUN_STATE_FILE = os.path.join(ROOT_DIR, "data", "run_state.json")
ROLLING_STATE_FILE = os.path.join(ROOT_DIR, "data", "rolling_state.json")
# Legacy: kept only for migration from old dual-file format
LEGACY_ROLLING_RECORD_FILE = os.path.join(ROOT_DIR, "latest_rolling_records.json")
PRETRAINED_DIR = os.path.join(ROOT_DIR, "data", "pretrained")

# Unified training mode system
KNOWN_TRAINING_MODES = ['static', 'rolling']
DEFAULT_TRAINING_MODE = 'static'
MODE_SEPARATOR = '@'


# ================= Model Key Helpers =================
def make_model_key(model_name, mode=None):
    """构造 model@mode 复合 key

    Args:
        model_name: 模型名称 (如 'gru_Alpha158')
        mode: 训练模式 (如 'static', 'rolling'). None 则使用 DEFAULT_TRAINING_MODE

    Returns:
        str: 复合 key (如 'gru_Alpha158@static')
    """
    if mode is None:
        mode = DEFAULT_TRAINING_MODE
    if MODE_SEPARATOR in model_name:
        # 已经是复合 key，直接返回
        return model_name
    return f"{model_name}{MODE_SEPARATOR}{mode}"


def parse_model_key(key):
    """解析 model@mode 复合 key

    Args:
        key: 复合 key 或裸模型名

    Returns:
        tuple: (model_name, mode). 无 '@' 时 mode 默认 'static'
    """
    if MODE_SEPARATOR in key:
        parts = key.rsplit(MODE_SEPARATOR, 1)
        return parts[0], parts[1]
    return key, DEFAULT_TRAINING_MODE


def get_experiment_name_for_model(records_dict, model_key):
    """
    根据模型 key 的 mode 获取对应的 experiment_name。
    优先读取 JSON 顶层的 `static_experiment_name` / `rolling_experiment_name`，
    为了向后兼容，如果没找到则回退到 `experiment_name`。

    Args:
        records_dict: 包含 experiment_name 等字段的记录字典
        model_key: 模型的复合 key (如 'gru_Alpha158@static')

    Returns:
        str: 该模型对应的 experiment_name
    """
    _, mode = parse_model_key(model_key)
    
    # 优先尝试模式特定的 experiment_name
    mode_exp_key = f"{mode}_experiment_name"
    if mode_exp_key in records_dict and records_dict[mode_exp_key]:
        return records_dict[mode_exp_key]
        
    # 回退到老的全局 experiment_name
    return records_dict.get('experiment_name', '')



def resolve_model_key(name, models_dict, default_mode=None):
    """将裸模型名或完整 key 解析为 models_dict 中实际存在的 key

    解析优先级：
    1. name 本身是完整 key 且存在 → 直接返回
    2. name 是裸名 + default_mode 指定 → 返回 name@default_mode
    3. name 是裸名 + 无 default_mode → 按 KNOWN_TRAINING_MODES 顺序搜索
    4. 找不到 → 返回 None

    Args:
        name: 模型名或 model@mode 完整 key
        models_dict: 训练记录中的 models dict
        default_mode: 默认训练模式，None 则自动搜索

    Returns:
        str or None: 解析后的完整 key，找不到返回 None
    """
    # 1. 直接匹配（完整 key 或旧格式裸名）
    if name in models_dict:
        return name

    # 2. 如果已经包含 @，说明是指定了模式但不存在
    if MODE_SEPARATOR in name:
        return None

    # 3. 指定了 default_mode
    if default_mode:
        full_key = make_model_key(name, default_mode)
        if full_key in models_dict:
            return full_key
        return None

    # 4. 自动搜索所有已知模式
    found = []
    for mode in KNOWN_TRAINING_MODES:
        full_key = make_model_key(name, mode)
        if full_key in models_dict:
            found.append(full_key)

    if len(found) == 1:
        return found[0]
    elif len(found) > 1:
        # 多个模式都存在，优先返回第一个（static），同时 warn
        print(f"⚠️  模型 '{name}' 在多个训练模式中存在: {found}，使用 {found[0]}")
        print(f"   如需指定模式，请使用 '{name}@rolling' 或 --training-mode 参数")
        return found[0]

    return None


def resolve_model_keys(names, models_dict, default_mode=None):
    """批量解析模型名为完整 key

    Args:
        names: 模型名列表
        models_dict: 训练记录中的 models dict
        default_mode: 默认训练模式

    Returns:
        resolved: list of (original_name, full_key) — full_key 为 None 表示未找到
    """
    results = []
    for name in names:
        full_key = resolve_model_key(name, models_dict, default_mode)
        results.append((name, full_key))
    return results


def filter_models_by_mode(models_dict, mode):
    """按训练模式过滤 models dict

    Args:
        models_dict: {model_key: record_id}
        mode: 训练模式 (如 'static', 'rolling'). None 则不过滤

    Returns:
        dict: 过滤后的 {model_key: record_id}
    """
    if mode is None:
        return models_dict
    return {
        k: v for k, v in models_dict.items()
        if parse_model_key(k)[1] == mode
    }


def strip_mode_from_keys(models_dict):
    """从 model@mode key 中去掉 @mode 部分，返回 {bare_name: record_id}

    如果去掉 mode 后有重复的 bare_name，后者覆盖前者并 warn。

    Args:
        models_dict: {model_key: record_id}

    Returns:
        dict: {bare_name: record_id}
    """
    result = {}
    for key, rid in models_dict.items():
        bare_name, _ = parse_model_key(key)
        if bare_name in result:
            print(f"⚠️  去模式后 key 冲突: '{bare_name}' 出现多次，使用最后一个")
        result[bare_name] = rid
    return result


# ================= 日期计算 =================
def calculate_dates():
    """根据统一配置计算训练日期窗口"""
    from qlib.data import D
    from quantpits.utils.config_loader import load_workspace_config

    # 加载统一配置 (取代对 MODEL_CONFIG_FILE 和 PROD_CONFIG_FILE 的直接读取)
    config = load_workspace_config(ROOT_DIR)

    train_date_mode = config.get('train_date_mode', 'last_trade_date')
    data_slice_mode = config.get('data_slice_mode', 'slide')

    test_set_window = config.get("test_set_window", 3)
    valid_set_window = config.get("valid_set_window", 2)
    train_set_windows = config.get("train_set_windows", 8)

    # 频次配置
    freq = config.get("freq", "week").lower()
    
    # 确定锚点日期
    if train_date_mode == 'last_trade_date':
        last_trade_date = D.calendar(future=False)[-1:][0]
        anchor_date = last_trade_date.strftime('%Y-%m-%d')
    else:
        anchor_date = config.get('current_date', datetime.now().strftime('%Y-%m-%d'))

    def add_year_with_nextday(input_date, n):
        input_date_obj = datetime.strptime(input_date, "%Y-%m-%d")
        # 允许 fractional years
        delta_days = int(n * AVERAGE_CALENDAR_DAYS_PER_YEAR)
        target_date = input_date_obj + timedelta(days=delta_days)
        next_day = target_date + timedelta(days=1)
        return target_date.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d")

    if data_slice_mode == 'slide':
        _, start_time = add_year_with_nextday(anchor_date, -1 * (train_set_windows + valid_set_window + test_set_window))
        fit_start_time = start_time
        fit_end_time, valid_start_time = add_year_with_nextday(anchor_date, -1 * (valid_set_window + test_set_window))
        valid_end_time, test_start_time = add_year_with_nextday(anchor_date, -1 * test_set_window)
        test_end_time = anchor_date
    else:
        start_time = config.get("start_time", "2008-01-01")
        fit_start_time = config.get("fit_start_time", "2008-01-01")
        fit_end_time = config.get("fit_end_time", "")
        valid_start_time = config.get("valid_start_time", "")
        valid_end_time = config.get("valid_end_time", "")
        test_start_time = config.get("test_start_time", "")
        test_end_time = config.get("test_end_time", "")

    # 获取当前资金信息 (从统一配置中的 current_full_cash 获取)
    account = config.get("current_full_cash", 100000.0)
    
    date_params = {
        "market": config.get("market", "csi300"),
        "benchmark": config.get("benchmark", "SH000300"),
        "topk": config.get("topk", 20),
        "n_drop": config.get("n_drop", 3),
        "buy_suggestion_factor": config.get("buy_suggestion_factor", 2),
        "account": account,
        "start_time": start_time,
        "end_time": test_end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "valid_start_time": valid_start_time,
        "valid_end_time": valid_end_time,
        "test_start_time": test_start_time,
        "test_end_time": test_end_time,
        "anchor_date": anchor_date,
        "freq": freq
    }

    print("\n=== Config Loaded via config_loader ===")
    for k, v in date_params.items():
        print(f"{k}: {v}")
    print("========================================\n")

    return date_params


# ================= 预训练管理 =================
def get_pretrained_model_path(base_model_name, anchor_date=None):
    """获取预训练模型的路径
    
    Args:
        base_model_name: 基础模型名（如 'lstm_Alpha158'）
        anchor_date: 指定日期版本，None 则用 latest
    
    Returns:
        str: 预训练模型路径，不存在返回 None
    """
    if anchor_date:
        path = os.path.join(PRETRAINED_DIR, f"{base_model_name}_{anchor_date}.pkl")
    else:
        path = os.path.join(PRETRAINED_DIR, f"{base_model_name}_latest.pkl")
    return path if os.path.exists(path) else None


def save_pretrained_model(model, base_model_name, anchor_date,
                          d_feat, hidden_size, num_layers):
    """保存预训练模型的 state_dict 和元数据
    
    Args:
        model: 训练好的 qlib 模型对象
        base_model_name: 基础模型名（如 'lstm_Alpha158'）
        anchor_date: 训练锚点日期
        d_feat: 输入特征维度
        hidden_size: 隐层大小
        num_layers: 层数
    
    Returns:
        str: 保存的 .pkl 文件路径
    """
    import torch
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    
    inner_model = _get_inner_model(model)
    state_dict = inner_model.state_dict()
    
    # 保存带日期的版本
    dated_path = os.path.join(PRETRAINED_DIR, f"{base_model_name}_{anchor_date}.pkl")
    torch.save(state_dict, dated_path)
    
    # 保存 JSON sidecar 元数据
    metadata = {
        "base_model_name": base_model_name,
        "anchor_date": anchor_date,
        "d_feat": d_feat,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = dated_path.replace(".pkl", ".json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 更新 latest（复制而非 symlink，避免跨平台问题）
    latest_pkl = os.path.join(PRETRAINED_DIR, f"{base_model_name}_latest.pkl")
    latest_json = os.path.join(PRETRAINED_DIR, f"{base_model_name}_latest.json")
    shutil.copy2(dated_path, latest_pkl)
    shutil.copy2(meta_path, latest_json)
    
    print(f"🧠 预训练模型已保存: {dated_path}")
    print(f"   元数据: {meta_path}")
    print(f"   Latest: {latest_pkl}")
    return dated_path


def load_pretrained_metadata(base_model_name, anchor_date=None):
    """加载预训练模型的 JSON sidecar 元数据
    
    Args:
        base_model_name: 基础模型名
        anchor_date: 日期版本，None 用 latest
    
    Returns:
        dict or None: 元数据，不存在返回 None
    """
    if anchor_date:
        meta_path = os.path.join(PRETRAINED_DIR, f"{base_model_name}_{anchor_date}.json")
    else:
        meta_path = os.path.join(PRETRAINED_DIR, f"{base_model_name}_latest.json")
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


def _get_inner_model(model):
    """从 qlib 模型包装器中获取内部 pytorch nn.Module
    
    支持的模型属性名: LSTM_model, lstm_model, gru_model, GRU_model
    """
    for attr in ['LSTM_model', 'lstm_model', 'gru_model', 'GRU_model']:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError(f"无法从 {type(model).__name__} 中提取内部模型, "
                     f"已检查属性: LSTM_model, lstm_model, gru_model, GRU_model")


def resolve_pretrained_path(model_name, registry=None):
    """解析模型的预训练依赖路径
    
    根据 model_registry 中的 pretrain_source 字段，
    查找对应的预训练模型文件。
    
    Args:
        model_name: 模型名（如 'gats_Alpha158_plus'）
        registry: 模型注册表，None 则自动加载
    
    Returns:
        str or None: 预训练模型路径，无依赖或未找到返回 None
    """
    if registry is None:
        registry = load_model_registry()
    
    model_info = registry.get(model_name, {})
    pretrain_source = model_info.get('pretrain_source')
    
    if not pretrain_source:
        return None
    
    path = get_pretrained_model_path(pretrain_source)
    if path is None:
        print(f"⚠️  模型 '{model_name}' 需要预训练模型 '{pretrain_source}'，但未找到")
        print(f"   请先运行: python quantpits/scripts/pretrain.py --models {pretrain_source}")
        print(f"   或: python quantpits/scripts/pretrain.py --for {model_name}")
        print(f"   将使用随机权重初始化 basemodel")
    return path


def validate_pretrain_compatibility(model_name, pretrain_path, upper_d_feat):
    """校验预训练模型与上层模型的 d_feat 兼容性
    
    Args:
        model_name: 上层模型名
        pretrain_path: 预训练文件路径
        upper_d_feat: 上层模型的 d_feat
    
    Raises:
        ValueError: d_feat 不匹配时
    """
    # 从 pretrain_path 推导 base_model_name
    basename = os.path.basename(pretrain_path)
    # 格式: {base_model_name}_{date}.pkl 或 {base_model_name}_latest.pkl
    base_name = basename.rsplit('_', 1)[0] if '_' in basename else basename.replace('.pkl', '')
    
    metadata = load_pretrained_metadata(base_name)
    if metadata is None:
        # 无元数据（旧格式预训练文件），跳过校验
        print(f"  ⚠️  预训练模型无元数据，跳过 d_feat 校验")
        return
    
    pretrain_d_feat = metadata.get('d_feat')
    if pretrain_d_feat is not None and pretrain_d_feat != upper_d_feat:
        raise ValueError(
            f"❌ Feature 不匹配: {model_name} (d_feat={upper_d_feat}) "
            f"vs {basename} (d_feat={pretrain_d_feat})\n"
            f"   请重新预训练: python quantpits/scripts/pretrain.py --for {model_name}"
        )


# ================= YAML 注入 =================
def inject_config(yaml_path, params, model_name=None, no_pretrain=False):
    """将参数注入 YAML 配置 (包含频次感知注入 + 预训练 model_path 注入)
    
    Args:
        yaml_path: YAML 配置文件路径
        params: 日期参数 (来自 calculate_dates)
        model_name: 模型名 (用于查找 pretrain_source), None 则跳过预训练注入
        no_pretrain: 强制不加载预训练模型 (使用随机权重)
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    freq = params.get('freq', 'week').lower()

    config['market'] = params['market']
    config['benchmark'] = params['benchmark']

    # 注入策略参数
    if 'strategy' in config and 'params' in config['strategy']:
        config['strategy']['params']['topk'] = params.get('topk', 20)
        config['strategy']['params']['n_drop'] = params.get('n_drop', 3)
        config['strategy']['params']['buy_suggestion_factor'] = params.get('buy_suggestion_factor', 2)

    dh = config['data_handler_config']
    dh['start_time'] = params['start_time']
    dh['end_time'] = params['end_time']
    dh['fit_start_time'] = params['fit_start_time']
    dh['fit_end_time'] = params['fit_end_time']
    dh['instruments'] = params['market']

    # 1. 注入 Label (根据频次)
    # 如果 freq 为 week，标记可能是 Ref(-6) 或 Ref(-2)
    # 策略：如果 model_config 中有 label_formula 则使用，否则根据频次猜测
    if 'label_formula' in params:
        dh['label'] = [params['label_formula']]
    elif freq == 'week':
        # Qlib 默认日频数据下，周收益率通常取 5-6 天
        dh['label'] = ["Ref($close, -6) / Ref($close, -1) - 1"]
    else:
        dh['label'] = ["Ref($close, -2) / Ref($close, -1) - 1"]

    if 'task' in config and 'dataset' in config['task']:
        segs = config['task']['dataset']['kwargs']['segments']
        segs['train'] = [params['fit_start_time'], params['fit_end_time']]
        segs['valid'] = [params['valid_start_time'], params['valid_end_time']]
        segs['test'] = [params['test_start_time'], params['test_end_time']]

    if 'port_analysis_config' in config:
        from quantpits.utils import strategy
        pa = strategy.generate_port_analysis_config(freq=freq)
        pa['backtest']['start_time'] = params['test_start_time']
        pa['backtest']['end_time'] = params['test_end_time']
        pa['backtest']['account'] = params.get('account', pa['backtest']['account'])
        pa['backtest']['benchmark'] = params['benchmark']
        config['port_analysis_config'] = pa

        # YAML anchors (*port_analysis_config) are resolved at parse time,
        # so PortAnaRecord's config kwarg still points to the original empty {}.
        # Patch it directly to use the generated config.
        if 'task' in config and 'record' in config['task']:
            for r_cfg in config['task']['record']:
                if r_cfg.get('class') == 'PortAnaRecord':
                    if 'kwargs' in r_cfg and 'config' in r_cfg['kwargs']:
                        r_cfg['kwargs']['config'] = pa

    # 3. 注入 SigAnaRecord ann_scaler
    if 'task' in config and 'record' in config['task']:
        for r_cfg in config['task']['record']:
            if r_cfg.get('class') == 'SigAnaRecord':
                if 'kwargs' not in r_cfg: r_cfg['kwargs'] = {}
                r_cfg['kwargs']['ann_scaler'] = TRADING_WEEKS_PER_YEAR if freq == 'week' else TRADING_DAYS_PER_YEAR

    # 4. 注入预训练 model_path
    if 'task' in config and 'model' in config['task']:
        model_kwargs = config['task']['model'].get('kwargs', {})
        if 'base_model' in model_kwargs:
            if no_pretrain:
                # --no-pretrain: 强制删除 model_path, 使用随机权重
                if 'model_path' in model_kwargs:
                    del model_kwargs['model_path']
                print(f"  ⏭️  --no-pretrain: 将使用随机权重初始化 basemodel")
            elif model_name:
                pretrain_path = resolve_pretrained_path(model_name)
                if pretrain_path:
                    # 校验 d_feat 兼容性
                    upper_d_feat = model_kwargs.get('d_feat')
                    if upper_d_feat is not None:
                        validate_pretrain_compatibility(
                            model_name, pretrain_path, upper_d_feat
                        )
                    model_kwargs['model_path'] = pretrain_path
                    print(f"  🧠 注入预训练模型: {pretrain_path}")
                else:
                    # 预训练文件不存在, 删除 YAML 中可能存在的 model_path
                    if 'model_path' in model_kwargs:
                        del model_kwargs['model_path']

    return config


# ================= 模型注册表 =================
def load_model_registry(registry_file=None):
    """
    加载模型注册表
    
    Returns:
        dict: {model_name: {algorithm, dataset, market, yaml_file, enabled, tags, notes}}
    """
    if registry_file is None:
        registry_file = REGISTRY_FILE
    
    with open(registry_file, 'r') as f:
        registry = yaml.safe_load(f)
    
    return registry.get('models', {})


def get_enabled_models(registry=None):
    """获取所有 enabled=true 的模型列表"""
    if registry is None:
        registry = load_model_registry()
    
    return {name: info for name, info in registry.items() if info.get('enabled', False)}


def get_models_by_filter(registry=None, algorithm=None, dataset=None, market=None, tag=None):
    """
    按条件筛选模型
    
    Args:
        registry: 模型注册表，None 则自动加载
        algorithm: 按算法筛选 (如 'lstm', 'gru')
        dataset: 按数据集筛选 (如 'Alpha158', 'Alpha360')
        market: 按市场筛选 (如 'csi300')
        tag: 按标签筛选 (如 'ts', 'tree')
    
    Returns:
        dict: 满足条件的模型子集
    """
    if registry is None:
        registry = load_model_registry()
    
    result = {}
    for name, info in registry.items():
        if algorithm and info.get('algorithm', '').lower() != algorithm.lower():
            continue
        if dataset and info.get('dataset', '').lower() != dataset.lower():
            continue
        if market and info.get('market', '').lower() != market.lower():
            continue
        if tag and tag.lower() not in [t.lower() for t in info.get('tags', [])]:
            continue
        result[name] = info
    
    return result


def get_models_by_names(model_names, registry=None):
    """
    按模型名列表获取模型信息
    
    Args:
        model_names: 模型名列表
        registry: 模型注册表，None 则自动加载
    
    Returns:
        dict: 匹配的模型子集
    """
    if registry is None:
        registry = load_model_registry()
    
    result = {}
    for name in model_names:
        name = name.strip()
        if name in registry:
            result[name] = registry[name]
        else:
            print(f"⚠️  警告: 模型 '{name}' 不在注册表中，跳过")
    
    return result


# ================= 单模型训练 =================
def train_single_model(model_name, yaml_file, params, experiment_name, no_pretrain=False):
    """
    训练单个模型的完整流程：训练 → 预测 → Signal Record → IC 计算
    
    需要 qlib 已初始化。
    
    Args:
        model_name: 模型名称
        yaml_file: YAML 配置文件路径
        params: 日期参数（来自 calculate_dates）
        experiment_name: MLflow 实验名称
        no_pretrain: 强制不加载预训练模型
    
    Returns:
        dict: {
            'success': bool,
            'record_id': str or None,
            'performance': dict or None,
            'error': str or None
        }
    """
    result = {
        'success': False,
        'record_id': None,
        'performance': None,
        'error': None
    }
    
    if not os.path.exists(yaml_file):
        result['error'] = f"YAML配置文件不存在: {yaml_file}"
        print(f"!!! Warning: {yaml_file} not found, skipping...")
        return result
    
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    print(f"\n>>> Processing Model: {model_name} from {yaml_file} ...")
    
    task_config = inject_config(yaml_file, params, model_name=model_name, no_pretrain=no_pretrain)
    
    try:
        with R.start(experiment_name=experiment_name):
            R.set_tags(model=model_name, anchor_date=params['anchor_date'])
            R.log_params(**params)
            
            # 初始化模型和数据集
            model_cfg = task_config['task']['model']
            model = init_instance_by_config(model_cfg)
            
            dataset_cfg = task_config['task']['dataset']
            dataset = init_instance_by_config(dataset_cfg)
            
            # 训练
            print(f"[{model_name}] Training...")
            model.fit(dataset=dataset)
            
            # 预测
            print(f"[{model_name}] Predicting...")
            pred = model.predict(dataset=dataset)
            
            # 保存模型对象（供 static_train.py --predict-only 加载）
            print(f"[{model_name}] Saving model to recorder...")
            recorder = R.get_recorder()
            recorder.save_objects(**{"model.pkl": model})
            

            
            # 生成 Signal Record
            record_cfgs = task_config['task'].get('record', [])
            recorder = R.get_recorder()
            
            for r_cfg in record_cfgs:
                if r_cfg['kwargs'].get('model') == '<MODEL>':
                    r_cfg['kwargs']['model'] = model
                if r_cfg['kwargs'].get('dataset') == '<DATASET>':
                    r_cfg['kwargs']['dataset'] = dataset
                
                r_obj = init_instance_by_config(r_cfg, recorder=recorder)
                r_obj.generate()
            
            # 获取模型成绩（IC等）
            performance = {}
            try:
                ic_series = recorder.load_object("sig_analysis/ic.pkl")
                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std != 0 else None
                performance = {
                    "IC_Mean": float(ic_mean) if ic_mean else None,
                    "ICIR": float(ic_ir) if ic_ir else None,
                    "record_id": recorder.info['id']
                }
            except Exception as e:
                print(f"[{model_name}] Could not get IC metrics: {e}")
                performance = {"record_id": recorder.info['id']}
            
            rid = recorder.info['id']
            print(f"[{model_name}] Finished! Recorder ID: {rid}")
            
            result['success'] = True
            result['record_id'] = rid
            result['performance'] = performance
    
    except Exception as e:
        result['error'] = str(e)
        print(f"!!! Error running {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return result


# ================= 历史备份 =================
def backup_file_with_date(file_path, history_dir=None, prefix=None):
    """
    将文件备份到 history 目录，文件名带日期时间戳
    
    Args:
        file_path: 要备份的文件路径
        history_dir: 历史目录，默认 data/history/
        prefix: 备份文件名前缀，默认使用原文件名（不含扩展名）
    
    Returns:
        str: 备份文件路径，如果源文件不存在则返回 None
    """
    if not os.path.exists(file_path):
        return None
    
    if history_dir is None:
        history_dir = HISTORY_DIR
    
    os.makedirs(history_dir, exist_ok=True)
    
    basename = os.path.basename(file_path)
    name, ext = os.path.splitext(basename)
    if prefix:
        name = prefix
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_name = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(history_dir, backup_name)
    
    shutil.copy2(file_path, backup_path)
    print(f"📦 已备份: {file_path} → {backup_path}")
    
    return backup_path


# ================= 训练记录管理 =================
def merge_train_records(new_records, record_file=None):
    """
    增量合并训练记录到 latest_train_records.json
    
    语义：
    - 同名模型 → 覆盖 recorder ID
    - 新模型 → 追加
    - 未出现的模型 → 保留原有记录
    
    Args:
        new_records: 新的训练记录 dict，格式同 latest_train_records.json
        record_file: 记录文件路径
    
    Returns:
        dict: 合并后的完整记录
    """
    if record_file is None:
        record_file = RECORD_OUTPUT_FILE
    
    # 先备份现有文件
    backup_file_with_date(record_file, prefix="train_records")
    
    # 加载现有记录
    existing = {}
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            existing = json.load(f)
    
    # 合并：保留既有模型，覆盖/新增训练过的模型
    merged_models = existing.get('models', {}).copy()
    new_models = new_records.get('models', {})
    
    added = []
    updated = []
    for model_name, rid in new_models.items():
        if model_name in merged_models:
            if merged_models[model_name] != rid:
                updated.append(model_name)
        else:
            added.append(model_name)
        merged_models[model_name] = rid
    
    # 构建合并后的记录
    merged = {
        "experiment_name": new_records.get('experiment_name', existing.get('experiment_name', '')),
        "static_experiment_name": new_records.get('static_experiment_name', existing.get('static_experiment_name', '')),
        "rolling_experiment_name": new_records.get('rolling_experiment_name', existing.get('rolling_experiment_name', '')),
        "anchor_date": new_records.get('anchor_date', existing.get('anchor_date', '')),
        "timestamp": new_records.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "last_incremental_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": merged_models
    }
    
    # 保存
    with open(record_file, 'w') as f:
        json.dump(merged, f, indent=4)
    
    # 打印合并摘要
    preserved = [m for m in merged_models if m not in new_models]
    print(f"\n📋 训练记录合并完成:")
    if updated:
        print(f"  🔄 更新 ({len(updated)}): {', '.join(updated)}")
    if added:
        print(f"  ➕ 新增 ({len(added)}): {', '.join(added)}")
    if preserved:
        print(f"  📌 保留 ({len(preserved)}): {', '.join(preserved)}")
    print(f"  📁 文件: {record_file}")
    
    return merged


def overwrite_train_records(records, record_file=None):
    """
    全量覆写训练记录（用于 static_train.py --full 全量刷新模式）
    覆写前自动备份。
    
    Args:
        records: 完整的训练记录 dict
        record_file: 记录文件路径
    """
    if record_file is None:
        record_file = RECORD_OUTPUT_FILE
    
    # 备份现有文件
    backup_file_with_date(record_file, prefix="train_records")
    
    # 全量覆写
    with open(record_file, 'w') as f:
        json.dump(records, f, indent=4)
    
    print(f"📋 训练记录已全量覆写: {record_file}")


def merge_performance_file(new_performances, anchor_date, output_dir=None):
    """
    合并模型性能文件
    
    Args:
        new_performances: 新的性能数据 dict
        anchor_date: 锚点日期
        output_dir: 输出目录
    
    Returns:
        dict: 合并后的性能数据
    """
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "output")
    
    perf_file = os.path.join(output_dir, f"model_performance_{anchor_date}.json")
    
    # 加载现有性能数据
    existing = {}
    if os.path.exists(perf_file):
        backup_file_with_date(perf_file, prefix=f"model_performance_{anchor_date}")
        with open(perf_file, 'r') as f:
            existing = json.load(f)
    
    # 合并
    merged = existing.copy()
    merged.update(new_performances)
    
    # 保存
    with open(perf_file, 'w') as f:
        json.dump(merged, f, indent=4)
    
    return merged


# ================= 运行状态管理 =================
def save_run_state(state, state_file=None):
    """
    保存运行状态（用于 rerun/resume）
    
    Args:
        state: 运行状态 dict，格式:
            {
                'started_at': str,
                'mode': 'incremental' | 'full',
                'target_models': [str],
                'completed': [str],
                'failed': {model_name: error_msg},
                'skipped': [str]
            }
    """
    if state_file is None:
        state_file = RUN_STATE_FILE
    
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)


def load_run_state(state_file=None):
    """
    加载运行状态
    
    Returns:
        dict or None: 运行状态，不存在则返回 None
    """
    if state_file is None:
        state_file = RUN_STATE_FILE
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return None


def clear_run_state(state_file=None):
    """清除运行状态文件"""
    if state_file is None:
        state_file = RUN_STATE_FILE
    
    if os.path.exists(state_file):
        # 备份到历史
        backup_file_with_date(state_file, prefix="run_state")
        os.remove(state_file)
        print("🗑️  运行状态已清除")


# ================= 工具函数 =================
def print_model_table(models, title="模型列表"):
    """
    以表格形式打印模型列表
    
    Args:
        models: 模型字典 {name: info}
        title: 表格标题
    """
    print(f"\n{'='*70}")
    print(f"  {title} ({len(models)} 个模型)")
    print(f"{'='*70}")
    print(f"  {'模型名':<30} {'算法':<12} {'数据集':<12} {'市场':<8} {'标签'}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8} {'-'*20}")
    
    for name, info in models.items():
        tags_str = ', '.join(info.get('tags', []))
        print(f"  {name:<30} {info.get('algorithm',''):<12} {info.get('dataset',''):<12} {info.get('market',''):<8} {tags_str}")
    
    print(f"{'='*70}\n")


# ================= 共享 CLI 逻辑 =================
def resolve_target_models(args, registry=None):
    """
    根据 CLI 参数解析目标模型列表（共享逻辑）

    支持的 args 属性:
    - models: str, 逗号分隔模型名
    - all_enabled: bool
    - algorithm / dataset / market / tag: str, 筛选条件
    - skip: str, 逗号分隔的排除模型名

    Args:
        args: argparse.Namespace
        registry: 模型注册表 dict, None 则自动加载

    Returns:
        dict: {model_name: model_info} 或 None（未指定选择条件）
    """
    if registry is None:
        registry = load_model_registry()

    if getattr(args, 'models', None):
        model_names = [m.strip() for m in args.models.split(',')]
        targets = get_models_by_names(model_names, registry)
    elif getattr(args, 'all_enabled', False):
        targets = get_enabled_models(registry)
    elif any([getattr(args, 'algorithm', None),
              getattr(args, 'dataset', None),
              getattr(args, 'market', None),
              getattr(args, 'tag', None)]):
        targets = get_models_by_filter(
            registry,
            algorithm=getattr(args, 'algorithm', None),
            dataset=getattr(args, 'dataset', None),
            market=getattr(args, 'market', None),
            tag=getattr(args, 'tag', None),
        )
    else:
        return None  # 没有指定任何选择条件

    # 应用 --skip
    skip_str = getattr(args, 'skip', None)
    if skip_str:
        skip_names = [m.strip() for m in skip_str.split(',')]
        targets = {k: v for k, v in targets.items() if k not in skip_names}
        if skip_names:
            print(f"⏭️  跳过模型: {', '.join(skip_names)}")

    return targets


def show_model_list(args, source_records_file=None):
    """
    列出模型注册表（共享逻辑）

    Args:
        args: 含 algorithm/dataset/market/tag 筛选属性
        source_records_file: 可选，源训练记录文件路径（用于显示可预测模型）
    """
    registry = load_model_registry()

    # 应用筛选条件
    if any([getattr(args, 'algorithm', None),
            getattr(args, 'dataset', None),
            getattr(args, 'market', None),
            getattr(args, 'tag', None)]):
        models = get_models_by_filter(
            registry,
            algorithm=getattr(args, 'algorithm', None),
            dataset=getattr(args, 'dataset', None),
            market=getattr(args, 'market', None),
            tag=getattr(args, 'tag', None),
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

    # 检查源训练记录中哪些模型可用
    if source_records_file and os.path.exists(source_records_file):
        with open(source_records_file, 'r') as f:
            source_records = json.load(f)
        source_models = source_records.get('models', {})
        available = [name for name in models if name in source_models]
        print(f"\n  源记录 ({source_records_file}):")
        print(f"    已训练可预测: {len(available)} / {len(models)}")
        if available:
            print(f"    可用: {', '.join(available)}")
        not_available = [name for name in models if name not in source_models]
        if not_available:
            print(f"    无记录: {', '.join(not_available)}")


def predict_single_model(model_name, model_info, params, experiment_name,
                         source_records, no_pretrain=False):
    """
    使用已有模型对新数据进行预测（不训练）— 共享逻辑

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
        no_pretrain: 是否跳过预训练权重

    Returns:
        dict: {success, record_id, performance, error}
    """
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
    resolved_key = resolve_model_key(model_name, source_models, default_mode='static')
    
    if not resolved_key:
        result['error'] = f"模型 '{model_name}' 不在源训练记录中，无法加载已有模型"
        print(f"!!! Error: {result['error']}")
        return result

    source_record_id = source_models[resolved_key]
    source_experiment = source_records.get('experiment_name', 'Weekly_Production_Train')

    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    print(f"\n>>> Predict-Only: {model_name}")
    print(f"    Source: experiment={source_experiment}, recorder={source_record_id}")
    print(f"    YAML: {yaml_file}")

    try:
        # 1. 从源 recorder 加载模型
        print(f"[{model_name}] Loading model from source recorder...")
        
        # 稳健加载：如果在当前 recorder 里没找到 model.pkl，则根据 source_record_id tag 向上溯源
        current_id = source_record_id
        current_exp = source_experiment
        model = None
        for _ in range(10):
            source_recorder = R.get_recorder(
                recorder_id=current_id,
                experiment_name=current_exp
            )
            try:
                model = source_recorder.load_object("model.pkl")
                break
            except Exception:
                tags = source_recorder.list_tags()
                if 'source_record_id' in tags and 'source_experiment' in tags:
                    print(f"    [Fallback] model.pkl 不在 {current_id} 中，正在向上溯源到 {tags['source_record_id']}...")
                    current_id = tags['source_record_id']
                    current_exp = tags['source_experiment']
                else:
                    raise ValueError(f"model.pkl not found in {current_id} and no parent tags available.")
                    
        if model is None:
            raise ValueError(f"Exceeded max traceback depth of 10 for {model_name}.")
            
        print(f"[{model_name}] Model loaded successfully")

        # 2. 构建新的 dataset（使用新日期范围）
        task_config = inject_config(yaml_file, params, model_name=model_name,
                                    no_pretrain=no_pretrain)

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

            # 重点修复：必须把模型也存入新的 recorder 里面，
            # 否则下一个周期如果继续做仅预测，会因为上个仅预测的记录中没有 model.pkl 而失败
            recorder.save_objects(**{"model.pkl": model})

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


# ================= 迁移工具 =================
def migrate_legacy_records(workspace_dir=None, dry_run=False):
    """将旧格式的双文件记录迁移为统一格式

    旧格式:
    - latest_train_records.json (static 训练)
    - latest_rolling_records.json (rolling 训练)

    新格式:
    - latest_train_records.json (统一文件, key 为 model@mode)

    Args:
        workspace_dir: 工作区根目录, None 则使用 ROOT_DIR
        dry_run: 仅打印计划不实际写入

    Returns:
        dict: 合并后的记录，dry_run=True 时也返回（但不写入）
    """
    if workspace_dir is None:
        workspace_dir = ROOT_DIR

    static_file = os.path.join(workspace_dir, "latest_train_records.json")
    rolling_file = os.path.join(workspace_dir, "latest_rolling_records.json")

    static_records = {}
    rolling_records = {}

    if os.path.exists(static_file):
        with open(static_file, 'r') as f:
            static_records = json.load(f)
    if os.path.exists(rolling_file):
        with open(rolling_file, 'r') as f:
            rolling_records = json.load(f)

    if not static_records and not rolling_records:
        print("⚠️  没有找到任何记录文件，无需迁移")
        return {}

    # 检查是否已经迁移过（key 中包含 @）
    static_models = static_records.get('models', {})
    already_migrated = any(MODE_SEPARATOR in k for k in static_models)
    if already_migrated:
        print("ℹ️  记录文件已是新格式，无需迁移")
        return static_records

    # 构建合并后的 models dict
    merged_models = {}

    # Static models → model@static
    for name, rid in static_models.items():
        new_key = make_model_key(name, 'static')
        merged_models[new_key] = rid

    # Rolling models → model@rolling
    rolling_models = rolling_records.get('models', {})
    for name, rid in rolling_models.items():
        new_key = make_model_key(name, 'rolling')
        merged_models[new_key] = rid

    # 合并元数据（以 static 为主，保留 rolling 的 experiment_name）
    merged = {
        "experiment_name": static_records.get('experiment_name', ''),
        "anchor_date": static_records.get('anchor_date',
                                          rolling_records.get('anchor_date', '')),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "migrated_from": "legacy_dual_file",
        "models": merged_models,
    }

    # 如果有 rolling 的 experiment_name，记录在元数据中
    rolling_exp = rolling_records.get('experiment_name')
    if rolling_exp:
        merged["rolling_experiment_name"] = rolling_exp

    # 打印迁移计划
    print(f"\n{'='*60}")
    print("📦 训练记录迁移计划")
    print(f"{'='*60}")
    print(f"Static 记录 ({len(static_models)} 个模型):")
    for name, rid in static_models.items():
        print(f"  {name} → {make_model_key(name, 'static')}  [{rid[:12]}...]")
    print(f"Rolling 记录 ({len(rolling_models)} 个模型):")
    for name, rid in rolling_models.items():
        print(f"  {name} → {make_model_key(name, 'rolling')}  [{rid[:12]}...]")
    print(f"合并后总计: {len(merged_models)} 个 model@mode 条目")

    if dry_run:
        print("\n🔍 Dry-run 模式: 不实际写入")
        return merged

    # 备份旧文件
    history_dir = os.path.join(workspace_dir, "data", "history")
    if os.path.exists(static_file):
        backup_file_with_date(static_file, history_dir=history_dir,
                              prefix="train_records_pre_migration")
    if os.path.exists(rolling_file):
        backup_file_with_date(rolling_file, history_dir=history_dir,
                              prefix="rolling_records_pre_migration")

    # 写入统一文件
    with open(static_file, 'w') as f:
        json.dump(merged, f, indent=4)
    print(f"\n✅ 统一记录已写入: {static_file}")

    # 删除旧的 rolling 文件
    if os.path.exists(rolling_file):
        os.remove(rolling_file)
        print(f"🗑️  已删除旧文件: {rolling_file}")

    print(f"{'='*60}\n")
    return merged
