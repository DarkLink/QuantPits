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

# 注意: qlib 相关导入（D, init_instance_by_config, R）在需要的函数内部延迟导入，
# 这样 --list, --show-state 等不需要训练的命令可以在没有 qlib 的环境中运行。


# ================= 路径常量 =================
import gc
import traceback

import env

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
ROLLING_RECORD_FILE = os.path.join(ROOT_DIR, "latest_rolling_records.json")
PRETRAINED_DIR = os.path.join(ROOT_DIR, "data", "pretrained")


# ================= 日期计算 =================
def calculate_dates():
    """根据统一配置计算训练日期窗口"""
    from qlib.data import D
    from config_loader import load_workspace_config

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
        delta_days = int(n * 365.25)
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
        import strategy
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
                r_cfg['kwargs']['ann_scaler'] = 52 if freq == 'week' else 252

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
            
            # 保存模型对象（供 prod_predict_only.py 加载）
            print(f"[{model_name}] Saving model to recorder...")
            recorder = R.get_recorder()
            recorder.save_objects(**{"model.pkl": model})
            
            # 保存预测结果
            os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
            pred_file = os.path.join(PREDICTION_OUTPUT_DIR, f"{model_name}_{params['anchor_date']}.csv")
            pred.to_csv(pred_file)
            print(f"[{model_name}] Predictions saved to {pred_file}")
            
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
    全量覆写训练记录（用于 prod_train_predict.py 全量刷新模式）
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
