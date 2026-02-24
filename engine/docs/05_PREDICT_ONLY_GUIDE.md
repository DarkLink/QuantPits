# Predict-Only 使用指南

## 概述

`scripts/weekly_predict_only.py` 用于**不重新训练**的情况下，使用已有模型对最新数据进行预测和回测。

**使用场景**：数据更新后无需重训，直接用现有模型生成新预测，然后接入穷举/融合流程。

**工作流位置**：~~训练~~ → **仅预测（本步）** → 暴力穷举 → 融合回测 → 订单生成

---

## 快速开始

```bash
cd QuantPits

# 预测所有 enabled 模型
python engine/scripts/weekly_predict_only.py --all-enabled

# 预测指定模型
python engine/scripts/weekly_predict_only.py --models gru,mlp,linear_Alpha158

# 只预测 tree 系列
python engine/scripts/weekly_predict_only.py --tag tree

# 查看会预测哪些模型（不实际执行）
python engine/scripts/weekly_predict_only.py --all-enabled --dry-run
```

---

## 工作原理

```
1. 读取 latest_train_records.json（源记录）
2. 根据选择条件确定目标模型
3. 对每个模型:
   a. 从源 Recorder 加载 model.pkl
   b. 用 model_config.json 计算新的日期窗口
   c. 构建新 dataset、执行 model.predict()
   d. 在 Weekly_Predict_Only 实验下创建新 Recorder
   e. 保存 pred.pkl + 运行 SignalRecord（生成 IC 指标）
   f. 同时输出 CSV 到 output/predictions/
4. Merge 方式更新 latest_train_records.json
```

> [!IMPORTANT]
> 预测结束后，`latest_train_records.json` 中相关模型的 `experiment_name` 和 `record_id` 会更新为新值。下游的穷举/融合脚本直接读取此文件即可，无需任何修改。

---

## 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--models` | 无 | 指定模型名，逗号分隔 |
| `--algorithm` | 无 | 按算法筛选 |
| `--dataset` | 无 | 按数据集筛选 |
| `--market` | 无 | 按市场筛选 |
| `--tag` | 无 | 按标签筛选 |
| `--all-enabled` | - | 预测所有 enabled 模型 |
| `--skip` | 无 | 跳过指定模型，逗号分隔 |
| `--source-records` | `latest_train_records.json` | 源训练记录文件 |
| `--dry-run` | - | 仅打印计划 |
| `--experiment-name` | `Weekly_Predict_Only` | MLflow 实验名称 |
| `--list` | - | 列出模型注册表 |

---

## 模型选择方式

```bash
# 1. 按名称指定
python engine/scripts/weekly_predict_only.py --models gru,mlp

# 2. 按算法筛选
python engine/scripts/weekly_predict_only.py --algorithm lstm

# 3. 按数据集筛选
python engine/scripts/weekly_predict_only.py --dataset Alpha360

# 4. 按标签筛选
python engine/scripts/weekly_predict_only.py --tag tree

# 5. 所有 enabled 模型
python engine/scripts/weekly_predict_only.py --all-enabled

# 6. 排除某些模型
python engine/scripts/weekly_predict_only.py --all-enabled --skip catboost_Alpha158
```

---

## 输出文件

```
output/
├── predictions/
│   ├── gru_2026-02-13.csv              # 预测结果 CSV
│   ├── mlp_2026-02-13.csv
│   └── ...
└── model_performance_2026-02-13.json   # IC/ICIR 指标（合并）

latest_train_records.json               # 更新后的训练记录（含新 record_id）
```

---

## 保存行为 (Merge 语义)

与 `incremental_train.py` 一致：

| 情况 | 行为 |
|------|------|
| 同名模型已存在 | 覆盖其 recorder ID |
| 新增模型 | 追加到记录中 |
| 未预测的模型 | 保留原有记录不变 |

合并前自动备份到 `data/history/`。

---

## 典型工作流

### 场景 1：数据更新后快速预测 + 穷举

```bash
cd QuantPits

# Step 1: 用现有模型预测新数据
python engine/scripts/weekly_predict_only.py --all-enabled

# Step 2: 穷举组合（快速版）
python engine/scripts/brute_force_fast.py --max-combo-size 3

# Step 3: 融合回测
python engine/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# Step 4: 生成订单
# (使用 weekly_order_gen_ensemble.ipynb)
```

### 场景 2：只预测部分模型

```bash
# 只用 tree 系列模型预测
python engine/scripts/weekly_predict_only.py --tag tree

# 然后对这些模型做融合
python engine/scripts/ensemble_fusion.py \
  --models lightgbm_Alpha158,catboost_Alpha158
```

### 场景 3：Dry-run 检查

```bash
# 先看看哪些模型会被预测
python engine/scripts/weekly_predict_only.py --all-enabled --dry-run

# 列出注册表和可用模型
python engine/scripts/weekly_predict_only.py --list
```

---

## 与其他脚本的关系

| 脚本 | 用途 | 是否训练 | 输入 | 输出 |
|------|------|:--------:|------|------|
| `weekly_train_predict.py` | 全量训练+预测 | ✅ | configs | `latest_train_records.json` |
| `incremental_train.py` | 增量训练+预测 | ✅ | configs | `latest_train_records.json` |
| **`weekly_predict_only.py`** | **仅预测** | **❌** | **已有模型** | **`latest_train_records.json`** |
| `brute_force_ensemble.py` | 穷举组合 | - | train records | leaderboard |
| `ensemble_fusion.py` | 融合回测 | - | 选定模型 | 融合预测 + 绩效 |

> 三个输出 `latest_train_records.json` 的脚本使用**相同格式**，下游脚本可以透明切换。

---

## 注意事项

1. **前提条件**：需要先运行过训练脚本，确保 `latest_train_records.json` 存在且包含模型记录
2. **模型不存在**：如果选定模型不在源记录中，会自动跳过并警告
3. **实验名区分**：predict-only 使用 `Weekly_Predict_Only` 实验名，可与训练实验区分
4. **日期窗口**：使用 `model_config.json` 中的日期配置，与训练脚本一致
5. **备份**：每次更新 `latest_train_records.json` 前自动备份到 `data/history/`
