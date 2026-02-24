#!/usr/bin/env python
"""
Weekly Train + Predict Script (全量训练)
周末训练并预测，输出各模型成绩供选择。

运行方式：cd QuantPits && python engine/scripts/weekly_train_predict.py

本脚本为全量训练模式：
- 训练 model_registry.yaml 中所有 enabled=true 的模型
- 完成后 **全量覆写** latest_train_records.json（覆写前自动备份历史）
- 如需增量训练个别模型，请使用 scripts/incremental_train.py
"""

import qlib
import yaml
import json
import pandas as pd
import os
import argparse
import env
from qlib.data import D
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from datetime import datetime, timedelta

# ================= 配置 =================
# 工作目录应该是 QuantPits/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
os.chdir(ROOT_DIR)  # 确保工作目录正确

# 初始化 Qlib
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 导入共享工具
from train_utils import (
    calculate_dates,
    inject_config,
    load_model_registry,
    get_enabled_models,
    train_single_model,
    overwrite_train_records,
    backup_file_with_date,
    print_model_table,
    PREDICTION_OUTPUT_DIR,
    RECORD_OUTPUT_FILE,
)

EXPERIMENT_NAME = "Weekly_Production_Train"


# ================= 主流程 =================
def run_train_predict():
    """训练并预测所有 enabled 模型（全量刷新模式）"""
    params = calculate_dates()

    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    # 从模型注册表加载 enabled 模型
    registry = load_model_registry()
    enabled_models = get_enabled_models(registry)

    if not enabled_models:
        print("⚠️  没有找到 enabled=true 的模型，请检查 config/model_registry.yaml")
        return

    print_model_table(enabled_models, title="全量训练模型列表")

    current_records = {
        "experiment_name": EXPERIMENT_NAME,
        "anchor_date": params['anchor_date'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }

    model_performances = {}

    total = len(enabled_models)
    for idx, (model_name, model_info) in enumerate(enabled_models.items(), 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{total}] {model_name}")
        print(f"{'─'*60}")

        yaml_file = model_info['yaml_file']
        result = train_single_model(model_name, yaml_file, params, EXPERIMENT_NAME)

        if result['success']:
            current_records['models'][model_name] = result['record_id']
            if result['performance']:
                model_performances[model_name] = result['performance']
        else:
            print(f"❌ 模型 {model_name} 训练失败: {result.get('error', 'Unknown')}")

    # 全量覆写记录（自动备份历史）
    overwrite_train_records(current_records)

    # 保存模型成绩对比
    perf_file = f"output/model_performance_{params['anchor_date']}.json"
    backup_file_with_date(perf_file, prefix=f"model_performance_{params['anchor_date']}")
    with open(perf_file, 'w') as f:
        json.dump(model_performances, f, indent=4)

    print(f"\n{'='*50}")
    print(f"All tasks finished. Experiment: {EXPERIMENT_NAME}")
    print(f"Records saved to {RECORD_OUTPUT_FILE}")
    print(f"Performance comparison saved to {perf_file}")
    print(f"\nModel Performances:")
    for name, perf in model_performances.items():
        ic = perf.get('IC_Mean', 'N/A')
        icir = perf.get('ICIR', 'N/A')
        ic_str = f"{ic:.4f}" if isinstance(ic, float) else ic
        icir_str = f"{icir:.4f}" if isinstance(icir, float) else icir
        print(f"  {name}: IC={ic_str}, ICIR={icir_str}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_train_predict()
