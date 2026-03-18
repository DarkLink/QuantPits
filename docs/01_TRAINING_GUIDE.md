# QuantPits 训练系统使用指南

## 概览

训练系统由三个主脚本组成，共享同一套工具模块和模型注册表：

| 脚本 | 用途 | 训练 | 数据源 | 保存语义 |
|------|------|------|--------|----------|
| `static_train.py --full` | 全量训练 | ✅ | configs | `latest_train_records.json` |
| `static_train.py` | 增量训练 | ✅ | configs | `latest_train_records.json` |
| `static_train.py --predict-only` | 仅预测 | ❌ | 已有模型 | `latest_train_records.json` |
| `pretrain.py` | 基础模型预训练 | ✅ | configs | `data/pretrained/` (state_dict) |

两个脚本都会在修改 `latest_train_records.json` 之前自动备份历史到 `data/history/`。

---

## 文件结构

```text
QuantPits/
├── quantpits/
│   ├── scripts/                      # 系统核心代码
│   │   ├── static_train.py           # 静态训练统一入口（全量/增量/仅预测）
│   │   ├── pretrain.py               # 🧠 基础模型预训练脚本
│   │   ├── check_workflow_yaml.py    # 🔧 YAML配置生产环境参数验证
│   │   └── train_utils.py            # 共享工具模块
│   └── docs/
│       └── 01_TRAINING_GUIDE.md      # 本文档
│
└── workspaces/
    └── <YourWorkspace>/              # 你的隔离工作区
        ├── config/
        │   ├── model_registry.yaml   # 📋 模型注册表（核心配置）
        │   ├── model_config.json     # 日期/市场参数
        │   └── workflow_config_*.yaml# 各模型的 Qlib 工作流配置
        ├── output/
        │   ├── predictions/          # 预测结果 CSV
        │   └── model_performance_*.json # 模型成绩
        ├── data/
        │   ├── history/              # 📦 自动备份的历史文件
        │   ├── pretrained/           # 🧠 预训练基模型 (.pkl + .json)
        │   └── run_state.json        # 增量训练运行状态
        └── latest_train_records.json # 当前训练记录
```

---

## 模型注册表 (`config/model_registry.yaml`)

### 结构

每个模型用三个维度组织：**算法 (algorithm)** + **数据集 (dataset)** + **市场 (market)**

```yaml
models:
  gru:                              # 模型唯一标识名
    algorithm: gru                  # 算法名称
    dataset: Alpha158               # 数据处理器
    market: csi300                  # 目标市场（作为元数据标签用于命令行筛选）
    yaml_file: config/workflow_config_gru.yaml  # Qlib 工作流配置
    enabled: true                   # 是否参与全量训练
    tags: [basemodel, ts]           # 分类标签（用于筛选）
    pretrain_source: lstm_Alpha158  # (可选) 声明依赖的基础模型
    notes: "可选备注"                # 备注信息
```

#### 关键字段说明：
- **`tags: [basemodel]`**: 标记该模型可作为预训练基础模型。
- **`pretrain_source`**: 标记该上层模型依赖哪个基础模型。系统会自动寻找对应的 `_latest.pkl`。

> [!NOTE]
> **关于市场配置的区别**：注册表中的 `market` 字段是**模型元数据标签**，专门用于在执行增量训练或预测时通过 `--market` 参数进行筛选过滤。实际拉取量价数据时，系统依据的是 `model_config.json` 中的全局 `market` 设置。

### 添加新模型

1. 创建 YAML 工作流配置 `config/workflow_config_xxx.yaml`
2. 在 `model_registry.yaml` 添加模型条目
3. 使用 `static_train.py --models xxx` 单独训练验证
4. 确认无误后将 `enabled` 设为 `true`

### 禁用模型

将 `enabled` 设为 `false`，全量训练时会自动跳过。增量训练仍可通过 `--models` 指定运行。

### 可用标签

| 标签 | 含义 | 模型 |
|------|------|------|
| `ts` | 时序模型 | gru, alstm, tcn, sfm, ... |
| `nn` | 神经网络 | mlp, TabNet |
| `tree` | 树模型 | lightgbm, catboost |
| `attention` | 注意力机制 | alstm, transformer, TabNet |
| `baseline` | 基线模型 | linear |
| `graph` | 图模型 | gats |
| `cnn` | 卷积网络 | tcn |
| `basemodel` | 作为其他模型基础 | lstm |

---

## 全量训练 (`static_train.py --full`)

### 使用场景
- 生产环境例行全量训练
- 需要完整刷新所有模型记录的场景

### 运行

```bash
cd QuantPits
python quantpits/scripts/static_train.py --full
```

### 行为
1. 训练 `model_registry.yaml` 中所有 `enabled: true` 的模型
2. 完成后 **全量覆写** `latest_train_records.json`
3. 覆写前自动备份到 `data/history/train_records_YYYY-MM-DD_HHMMSS.json`
4. 性能数据保存到 `output/model_performance_{anchor_date}.json`

---

## 增量训练 (`static_train.py`)

### 使用场景
- 新增了模型，只想训练新模型
- 某个模型调参后需要重新训练
- 原来训练失败的模型需要重跑
- 不想全量重跑浪费时间和资源

### 模型选择方式

```bash
cd QuantPits

# 1. 按名称指定（逗号分隔）
python quantpits/scripts/static_train.py --models gru,mlp

# 2. 按算法筛选
python quantpits/scripts/static_train.py --algorithm lstm

# 3. 按数据集筛选
python quantpits/scripts/static_train.py --dataset Alpha360

# 4. 按标签筛选
python quantpits/scripts/static_train.py --tag tree

# 5. 按市场筛选
python quantpits/scripts/static_train.py --market csi300

# 6. 所有 enabled 模型（merge 模式）
python quantpits/scripts/static_train.py --all-enabled

# 7. 组合使用
python quantpits/scripts/static_train.py --all-enabled --skip catboost_Alpha158
```

### 保存行为 (Merge 语义)

| 情况 | 行为 |
|------|------|
| 同名模型已存在 | 覆盖其 recorder ID 和性能数据 |
| 新增模型 | 追加到记录中 |
| 未训练的模型 | 保留原有记录不变 |

### Dry-run（仅查看计划）

```bash
# 查看将训练哪些模型，不实际执行
python quantpits/scripts/static_train.py --models gru,mlp --dry-run
```

### Rerun / Resume（中断恢复）

如果训练到一半中断（模型死了/手动终止），运行状态会自动保存到 `data/run_state.json`。

```bash
# 查看上次运行状态
python quantpits/scripts/static_train.py --show-state

# 继续上次未完成的训练（跳过已成功的模型）
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158 --resume

# 清除运行状态（重新开始）
python quantpits/scripts/static_train.py --clear-state
```

**注意**：`--resume` 只跳过已完成的模型，**失败的模型会被重新训练**。

### 查看模型注册表

```bash
# 列出所有注册模型
python quantpits/scripts/static_train.py --list

# 按条件筛选查看
python quantpits/scripts/static_train.py --list --algorithm gru
python quantpits/scripts/static_train.py --list --dataset Alpha360
python quantpits/scripts/static_train.py --list --tag tree
```

---

## 日期处理

训练日期和频次由 `config/model_config.json` 控制：

| 参数 | 说明 |
|------|------|
| `train_date_mode` | `last_trade_date`（使用最近交易日）或固定日期 |
| `data_slice_mode` | `slide`（滑动窗口）或 `fixed`（固定日期） |
| `train_set_windows` | 训练集窗口大小（年） |
| `valid_set_window` | 验证集窗口大小（年） |
| `test_set_window` | 测试集窗口大小（年） |
| `freq` | 交易频次 (`week`/`day`) |

### 日期切换注意
- 全量训练和增量训练共享同一个 `model_config.json`
- 如果在增量训练时修改了日期参数，**新训练的模型会使用新日期**，而保留的旧模型仍基于旧日期
- 建议在同一个 anchor_date 窗口内使用增量训练，跨日期时使用全量训练

---

## 历史备份

所有重要文件在修改前会自动备份到 `data/history/`：

```
data/history/
├── train_records_2026-02-11_165306.json      # latest_train_records.json 的历史
├── train_records_2026-02-18_120000.json
├── model_performance_2026-02-06_165306.json   # 性能数据的历史
└── run_state_2026-02-12_150000.json           # 运行状态的历史
```

无需手动操作，系统会自动管理备份。

---

## 典型工作流

### 场景 1：例行例行训练

```bash
cd QuantPits
python quantpits/scripts/static_train.py --full
python quantpits/scripts/ensemble_predict.py --method icir_weighted --backtest
```

### 场景 1b：数据更新后仅预测（不重训）

```bash
cd QuantPits
# 使用已有模型对新数据预测
python quantpits/scripts/static_train.py --predict-only --all-enabled
# 后续穷举/融合流程不变
python quantpits/scripts/brute_force_fast.py --max-combo-size 3
python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158,alstm_Alpha158
```

> 详见 [05_PREDICT_ONLY_GUIDE.md](05_PREDICT_ONLY_GUIDE.md)

### 场景 2：新增一个模型

```bash
# 1. 创建 YAML 配置
# 2. 在 model_registry.yaml 添加条目（先设 enabled: false）
# 3. 单独训练验证
python quantpits/scripts/static_train.py --models new_model_name

# 4. 确认无误后，修改 enabled: true
```

### 场景 3：调参后重跑某个模型

```bash
# 修改 YAML 配置后
python quantpits/scripts/static_train.py --models gru
```

### 场景 4：训练中断恢复

```bash
# 第一次运行（中途中断了）
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360
# ... gru 完成，mlp 失败，后面的还没开始 ...

# 查看状态
python quantpits/scripts/static_train.py --show-state

# 继续运行（跳过已完成的 gru）
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360 --resume
```

### 场景 5：只想跑 tree 系列模型

```bash
python quantpits/scripts/static_train.py --tag tree
# 等价于: --models lightgbm_Alpha158,catboost_Alpha158
```

---

## 配置验证与修复

为确保所有模型的 YAML 文件按预期配置为生产模式（如 `label` 根据频次自动调整，`time_per_step` 匹配频次，`ann_scaler` 匹配频次），提供了自动化验证脚本。**建议在新增或修改 YAML 后运行此检查。**

```bash
# 检查所有的 workflow_config_*.yaml 是否符合生产环境参数要求 (day/week)
python quantpits/tools/check_workflow_yaml.py

# 尝试自动修正所有异常的 YAML 文件（自动将参数转为生产环境要求的格式）
python quantpits/tools/check_workflow_yaml.py --fix
```

---

---

## 基础模型预训练 (`pretrain.py`)

某些复杂模型（如 GATs, ADD, IGMTF）需要一个预训练好的基模型（如 LSTM 或 GRU）作为权重初始化。

### 使用场景
- 需要为上层模型提供初始化权重。
- 修改了 Feature (d_feat)，需要重新训练兼容的基础模型。

### 核心语义
- **预训练不计入训练记录**：不修改 `latest_train_records.json`。
- **元数据校验**：每个预训练文件附带 `.json` 元数据。如果上层模型的 `d_feat` 与预训练文件不符，系统会报错阻断。

### 常用命令

```bash
# 1. 列出可预训练模型及其依赖关系
python quantpits/scripts/pretrain.py --list

# 2. 预训练指定基础模型
python quantpits/scripts/pretrain.py --models lstm_Alpha158

# 3. 为特定上层模型预训练（最推荐：自动对齐 Dataset 配置）
# 即使修改了 Feature，该命令也能确保基础模型与上层模型完全兼容
python quantpits/scripts/pretrain.py --for gats_Alpha158_plus

# 4. 查看已有预训练文件
python quantpits/scripts/pretrain.py --show-pretrained

# 5. 强制使用随机权重（跳过预训练）
# 在全量、增量或仅预测中均可用
python quantpits/scripts/static_train.py --models gats_Alpha158_plus --no-pretrain
```

---

## 关于 LSTM 和 GATs

- `gats_Alpha158_plus` 默认依赖 `lstm_Alpha158`。
- 训练全流程：
  1. 预训练基模型（可选，已有则跳过）：
     `python quantpits/scripts/pretrain.py --for gats_Alpha158_plus`
  2. 训练上层模型：
     `python quantpits/scripts/static_train.py --models gats_Alpha158_plus`

- 如果不想使用预训练模型，只需加上 `--no-pretrain` 标志。


---

## 完整参数一览

```
python quantpits/scripts/static_train.py --help

模式:
  --full                  全量训练：训练所有 enabled 模型，全量覆写 records
  --predict-only          仅预测：使用已有模型对最新数据预测，不重新训练

模型选择:
  --models TEXT           指定模型名，逗号分隔
  --algorithm TEXT        按算法筛选
  --dataset TEXT          按数据集筛选
  --market TEXT           按市场筛选
  --tag TEXT              按标签筛选
  --all-enabled           训练所有 enabled=true 的模型

排除与跳过:
  --skip TEXT             跳过指定模型，逗号分隔
  --resume                从上次中断处继续

运行控制:
  --dry-run               仅打印计划，不训练
  --experiment-name TEXT  MLflow 实验名称

信息查看:
  --list                  列出模型注册表
  --show-state            显示上次运行状态
  --clear-state           清除运行状态文件
```
