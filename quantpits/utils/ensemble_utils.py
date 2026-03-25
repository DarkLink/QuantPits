"""
Ensemble 共享工具 — 配置解析、Combo 管理、Records 加载

从 ensemble_fusion.py / signal_ranking.py / order_gen.py 中抽取的公共逻辑，
消除跨脚本的代码重复。
"""

import os
import json
from typing import Dict, Tuple, Optional, Any


# ============================================================================
# Ensemble Config 解析
# ============================================================================

def parse_ensemble_config(ensemble_config: dict) -> Tuple[dict, dict]:
    """
    解析 ensemble_config.json，兼容新旧格式。

    新格式 (多组合):
        {"combos": {"combo_A": {"models": [...], "method": "equal", "default": true}, ...}}

    旧格式 (单组合):
        {"models": [...], "ensemble_method": "manual", "manual_weights": {...}}

    Returns:
        combos: dict, combo_name -> {"models": [], "method": str, "default": bool, ...}
        global_config: dict, min_model_ic 等全局配置
    """
    if 'combos' in ensemble_config:
        # 新格式
        combos = ensemble_config['combos']
        global_config = {k: v for k, v in ensemble_config.items() if k != 'combos'}
        return combos, global_config
    elif 'models' in ensemble_config:
        # 旧格式 → 自动迁移为 legacy combo
        legacy_combo = {
            'models': ensemble_config['models'],
            'method': ensemble_config.get('ensemble_method', 'equal'),
            'default': True,
            'description': '从旧格式自动迁移',
        }
        if 'manual_weights' in ensemble_config:
            legacy_combo['manual_weights'] = ensemble_config['manual_weights']
        combos = {'legacy': legacy_combo}
        global_config = {k: v for k, v in ensemble_config.items()
                         if k not in ('models', 'ensemble_method', 'manual_weights', 'use_ensemble')}
        return combos, global_config
    else:
        return {}, {}


def get_default_combo(combos: dict) -> Tuple[Optional[str], Optional[dict]]:
    """返回 default combo 的 (name, config)，如果没有则返回第一个"""
    for name, cfg in combos.items():
        if cfg.get('default', False):
            return name, cfg
    # 没有标记 default 的，取第一个
    if combos:
        first_name = next(iter(combos))
        return first_name, combos[first_name]
    return None, None


# ============================================================================
# Ensemble Records 加载
# ============================================================================

def load_ensemble_records(root_dir: str) -> dict:
    """
    加载 config/ensemble_records.json。

    Args:
        root_dir: 工作区根目录

    Returns:
        dict: ensemble records 内容，如果文件不存在则返回空 dict

    Raises:
        FileNotFoundError: 当文件不存在时（严格模式可由调用方处理）
    """
    records_file = os.path.join(root_dir, "config", "ensemble_records.json")
    if not os.path.exists(records_file):
        raise FileNotFoundError(
            "未找到 ensemble_records.json，请先运行 ensemble_fusion.py"
        )

    with open(records_file, 'r') as f:
        return json.load(f)


def resolve_combo_record_id(
    ensemble_records: dict,
    combo_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    从 ensemble_records 中解析 combo 的 record_id。

    如果 combo_name 为 None，则使用 default_combo 或最后一个 combo。

    Args:
        ensemble_records: ensemble_records.json 的内容
        combo_name: 指定的 combo 名称

    Returns:
        (combo_name, record_id)

    Raises:
        ValueError: 没有有效的融合记录
        FileNotFoundError: 指定的 combo 不存在
    """
    combos = ensemble_records.get("combos", {})

    if not combo_name:
        combo_name = ensemble_records.get("default_combo")
        if not combo_name:
            if not combos:
                raise ValueError("ensemble_records.json 中没有有效的融合记录")
            combo_name = list(combos.keys())[-1]

    record_id = (
        combos.get(combo_name)
        or combos.get(f"ensemble_{combo_name}")
        or combos.get(combo_name.replace("combo_", ""))
    )
    if not record_id:
        raise FileNotFoundError(f"未找到 combo '{combo_name}' 的记录 ID")

    return combo_name, record_id
