"""Pure legacy-compatible model key construction and selection."""

KNOWN_TRAINING_MODES = ['static', 'rolling', 'cpcv', 'cpcv_rolling']
DEFAULT_TRAINING_MODE = 'static'
MODE_SEPARATOR = '@'


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


