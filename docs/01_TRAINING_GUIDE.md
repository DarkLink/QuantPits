# QuantPits 训练系统使用指南

## 概览

训练系统由三个主脚本组成，共享同一套工具模块和模型注册表：

| 脚本 | 用途 | 训练 | 数据源 | 保存语义 |
|------|------|------|--------|----------|
| `static_train.py --full` | 全量训练 | ✅ | configs | `latest_train_records.json` |
| `static_train.py` | 增量训练 | ✅ | configs | `latest_train_records.json` |
| `static_train.py --predict-only` | 仅预测 | ❌ | 已有模型 | `latest_train_records.json` |
| `pretrain.py` | 基础模型预训练 | ✅ | configs | `data/pretrained/` (state_dict) |

static/CPCV 的真实执行通过同一个 typed execution kernel 发布记录。每个 workspace 同时只允许
一个真实训练命令持有 `data/locks/training_execution.lock`；轻量 plan 不创建该文件。当前记录与
本次性能文件通过 `data/training_transactions/<run-id>-<fingerprint>/` 中的 durable intent/receipt
组成可恢复发布单元，`latest_train_records.json` 始终作为最后替换的 current pointer。
本节完整契约当前只适用于 static/CPCV；Rolling/CPCV-rolling 仍使用既有独立流程。其真实变更
命令仅共享 workspace training execution lease，尚未接入 State V3 与 publication coordinator。

### Training Record V2

新写入的训练记录使用 `schema_version: 2`。`model_records` 是权威身份表，每个
`model@mode` 分别记录实际输出 experiment、recorder、operation、预测覆盖日期及可选的
source recorder 谱系；顶层 `models`、`*_experiment_name` 和 `anchor_date` 仅作为旧消费者的
兼容视图。读取 V2 时不得用顶层字段覆盖模型级身份。

旧 V1 文件仍可只读加载，并标记为 `legacy_unverified`。可使用以下命令进行只读审计和
V2 迁移预览；命令不会改写记录或初始化缺失的 MLflow backend：

```bash
python -m quantpits.tools.audit_training_records \
  --workspace workspaces/Demo_Workspace --json
```

默认 JSON 只输出计数与稳定问题码；只有在受控终端显式添加 `--preview` 才会包含内存迁移草案。
可选的 `--verify-mlflow` 会验证 experiment/recorder 与 workspace containment；
`--verify-predictions` 还会读取实际 `pred.pkl` 并核对模型级 prediction end。启用后，审计命令
会将只读 Qlib recorder runtime 显式绑定到 `--workspace`；该运行时初始化不会创建或迁移
缺失的 backend，也不会写入 workspace。全量训练只有在所有目标都成功并生成 verified entry
时才更新 current registry。

生产者只有在输出 recorder 与持久化 `pred.pkl` 验证完成后才应发布 `ready` 条目。失败或
跳过的模型不得替换已有 current pointer。

V2 使用严格版本分派：一旦声明 `schema_version: 2`，一致的 `models` 与 `model_records`
映射就是必需项，不会退回顶层 V1 字段。`ready` 记录必须包含实际预测的 start/end/rows；
predict-only 还必须记录直接 source recorder、experiment 与 operation。融合运行时会再次
核对 recorder 的 experiment/artifact 以及 `pred.pkl` 的覆盖范围，registry 是证据而不是替代验证。

### 轻量执行计划与运行审计

`static_train.py` 与 `cv_train.py` 共用 plan-first command boundary：

```bash
python -m quantpits.scripts.static_train --full --explain-plan
python -m quantpits.scripts.static_train --predict-only --all-enabled --json-plan
python -m quantpits.scripts.cv_train --all-enabled --explain-plan
```

`--explain-plan` / `--json-plan` 只读取 workspace 内的 registry、配置、workflow、当前记录
baseline、可选 resume state 和必要的 source record；不会初始化 Qlib/MLflow、触发 safeguard、
改变 cwd 或写文件。依赖交易日历的
锚点会显示为 `deferred_to_qlib_calendar`，不会伪造日期。旧 `--dry-run` 兼容为相同轻量计划。

真实执行默认写 `output/manifests/{static_train|cv_train}/<run_id>.json` 并关联
`data/operator_log.jsonl`。`--run-id` 可固定身份，`--no-manifest` 只关闭 manifest。
两个命令均支持 `--workspace PATH`。

Safeguard 通过后，service 只初始化一次 Qlib，并把精确日期、顺序固定的 target、source
recorder、输出 experiment 和 current-record baseline 绑定为 `execution_fingerprint`。runner
一次只能处理一个已计划 target，不能重新扫描 registry 或扩大模型集合。Predict-only 在准备
cache 或启动 runner 前还会从 workspace-bound MLflow backend 重新核对每个 source recorder 的
实际 experiment 与 artifact containment；legacy 声明不一致时 fail closed，不能把错误谱系复制
进新的 V2 entry。

真实执行使用带锁、CAS 保护的 Training State V3，阶段依次区分 target execution、publication
prepared/committed、manifest closure 与 terminal status。中断后的同身份 `--resume` 会根据
transaction preimage/postimage fingerprint 确定性 roll-forward 或完成审计，不会仅凭文件存在
推断成功；遇到未知第三版本时 fail closed。

Phase 27 将恢复证据进一步拆清：未指定 `--run-id` 的 `--resume` 会采用持久化 state 中的
logical run ID；每个成功 target 在 publication 前写入不可变、带 SHA-256 的 JSON evidence。
因此，已完成但尚未发布的昂贵模型工作可在验证后复用，而不是重复执行。Receipt 证明 current
outputs 已提交，target evidence 证明模型工作已完成，state 只负责协调恢复。History 事件绑定
receipt/target identity，manifest 使用 write-or-adopt 语义；相同内容可采用，冲突内容 fail closed。
恢复动作由纯 recovery classifier 根据 state、target evidence、intent、receipt 和 current member
状态决定；未知输出或身份不一致不会进入 target/publication 执行。History、promotion、manifest、
OperatorLog 与 terminal-state 是逐步持久化的 closure：warning 步骤保留 state，并在后续同身份
`--resume` 中仅重试未完成步骤；所有 closure 步骤完成后才清理 `run_state.json`。
Receipt 只有在 schema/status、run/transaction、target 顺序、完整 output ledger 及每个 current
postimage fingerprint 都与 intent 精确一致时，才可授权 closure；缺失、重复、额外或已改变的
member 一律 fail closed。OperatorLog 的 manifest-linked JSONL entry 会先 append、flush、fsync，
随后才将 `operator_log_linked` 标为完成；若写入失败，state 保持 `closing`。Manifest 采用精确
canonical bytes，持久化的 logical start/finish timing 在 closure resume 中不会被新 attempt 时间替换。
兼容的旧 State V3 仅在 transaction-bound intent/receipt、完整 target evidence、current postimage 与
可恢复 timing 全部可证明时回填新增字段；否则会提示 clear/显式迁移并 fail closed，不自动迁移。

发布语义：

- full：所有 target 都成功且 recorder evidence 完整时才提交 record/performance bundle；否则保持原字节。
- incremental / predict-only：成功 target 在一个 transaction 中 merge，失败 target 保留旧指针；只要有失败，发布 receipt 仍先持久化，随后命令以 execution failure 结束。
- resume：日期、配置、source 与 target 形成的 resume fingerprint 必须一致，且已发布 recorder 仍是 current pointer；current-record baseline 的预期变化不会伪造冲突。
- manifest：只列 publication receipt 证明的实际 workspace 输出；未发布的 MLflow recorder 仅作为 outcome evidence。
- promotion：static train 的 promotion hook 在 receipt 持久化后执行；失败只形成显式 warning，不回滚已验证的 current pointer。
- history：typed static/CPCV runner 不直接写全局 history 路径；service 只为 receipt 中的新发布项追加带稳定 `event_id` 的 workspace-local 事件，恢复时可幂等重放。

---

## 文件结构

```text
QuantPits/
├── quantpits/
│   ├── scripts/                      # 系统核心脚本
│   │   ├── static_train.py           # 静态训练统一入口（全量/增量/仅预测）
│   │   ├── pretrain.py               # 🧠 基础模型预训练脚本
│   │   └── check_workflow_yaml.py    # 🔧 YAML配置生产环境参数验证
│   ├── utils/                         # 共享工具模块
│   │   ├── train_utils.py            # 日期计算、YAML 注入、模型注册表、记录合并
│   │   ├── predict_utils.py          # 预测数据加载/保存
│   │   ├── config_loader.py          # Workspace 级配置加载
│   │   ├── workspace.py              # 显式 WorkspaceContext 与 fingerprint helper
│   │   ├── strategy.py               # 策略配置/回测策略构建
│   │   └── ...                       # 更多共享模块（详见系统总览）
│   ├── config_contracts/              # Workspace 配置校验、normalize、fingerprint
│   ├── training/                      # plan/runner/lease/State V3/publication journal/service
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
        │   ├── predictions/          # (由 Qlib Recorder 接管，存储于 mlruns)
        │   └── model_performance_*.json # 模型成绩
        ├── data/
        │   ├── history/              # 📦 自动备份的历史文件
        │   ├── pretrained/           # 🧠 预训练基模型 (.pkl + .json)
        │   ├── run_state.json        # 带锁、CAS 保护的 Training State V3
        │   ├── locks/                # 训练 execution/publication advisory locks
        │   ├── training_transactions/# 可恢复 publication intent/receipt
        │   └── training_runs/        # logical run 的 immutable target evidence/closure
        └── latest_train_records.json # 当前训练记录
```

> 测试政策：自动全量测试由项目 owner 手工执行。实现/审阅 agent 默认只做静态检查，不应自动
> 运行 `pytest tests/`、coverage、Docker full-suite 或等价命令；只有无法通过静态审阅判断的具体
> 高风险契约才可运行最小定向测试，并在交接中记录原因与命令。

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
3. record/performance 的精确 preimage 与 postimage 写入 publication transaction 目录
4. 验证性能文件后最后替换 `latest_train_records.json`，再写 committed receipt

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

显式的 `--clear-state` 是 operator 确认过的重新开始边界，因此也可以删除无法恢复的 legacy
schema state，而不会先尝试把它解析为 V3。自动 closure 使用的 terminal-only 清理仍会严格解析
V3 并拒绝 legacy 或非 terminal state。清理生产 state 前应先用 `--show-state` 和文件指纹保存证据。

**注意**：`--resume` 会保留 receipt 已证明且仍为 current pointer 的模型，重新执行失败模型；若
模型工作和 publication 已完成但 manifest/state closure 中断，则只完成审计闭环，不重复训练。
普通 `--resume` 不需要再次传 `--run-id`；如果显式传入，该 ID 必须与 state 完全一致，否则会在
Qlib 初始化前失败。重新选择的有序 target key 也必须与 state 完全一致；缺少、增加、重排或
改选 target 都会在 planning 阶段拒绝，不会初始化 Qlib/MLflow、计算日期、准备 cache 或运行模型。

全量 Python/Docker 测试与真实 Playground static/CPCV predict-only release gate 由 workspace owner
显式执行；实现工具不会自动运行全量套件，也不会自动删除或迁移 legacy state。

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

## CPCV 模式：Purged Cross-Validation

CPCV (Combinatorial Purged Cross-Validation) 基于 Marcos Lopez de Prado《Advances in Financial Machine Learning》，通过将验证期从训练时间线中"挖出"并施加 Purge/Embargo 间隙，让模型同时从验证期之前和之后的数据中学习，从而保持对近期市场模式的适应性。

### 为什么要用 CPCV？

传统的 `slide` 模式（5年训练/2年验证/2年测试）产生的训练数据距今 4-9 年，模型完全接触不到近期市场状态。CPCV 将时间线划分为 N 个等长分组，在每个 Fold 中用 1-2 段非连续的训练数据（分居验证期两侧），使训练数据跨越到验证期之后，大幅缩短"训练→预测"的时间距离。

### 配置

CPCV 通过 `config/model_config.json` 中的 `purged_cv` 配置块独立启用，
**不受 `data_slice_mode` 影响** — `data_slice_mode`（`slide` / `fixed`）仅控制
静态训练的日期窗口划分方式，与 CPCV 正交：

```jsonc
{
    "data_slice_mode": "slide",    // 控制静态训练（slide/fixed），不影响 CPCV
    "purged_cv": {
        "n_groups": 10,            // 将 [start_time, anchor_date] 划分为 N 个等步长分组
        "n_test_groups": 2,        // 末尾的 N 个分组固定为测试集（不参与 CV）
        "n_val_groups": 1,         // 每个 Fold 使用 N 个连续分组作为验证集
        "purge_steps": 5,          // 对称 Purge：验证集两侧各移除 N 个步长（periods）
        "embargo_steps": 10        // 非对称 Embargo：验证集之后额外延迟 N 步才恢复训练
    },
    "start_time": "2015-01-01",   // 总时间范围起点（CPCV 和 slide 模式共用）
    "freq": "week"                 // 步长单位：day 或 week
}
```

> [!NOTE]
> **向后兼容**：原有设置 `data_slice_mode: "purged_cv"` 的配置文件仍然正常工作。
> 如果同时设置了 `data_slice_mode: "slide"` 和 `purged_cv` 块，则 CPCV 和静态训练可共存于同一工作区，
> 分别通过 `cv_train.py` 和 `static_train.py` 独立运行，无需修改配置文件。

**参数语义**：

| 参数 | 含义 | 示例 (freq=week) |
|------|------|-----------------|
| `n_groups` | 总分组数 | 10 组 |
| `n_test_groups` | 末尾 N 组固定为测试集 | 2 组 ≈ 2.3 年 |
| `n_val_groups` | 每 Fold 的验证组数 | 1 组 ≈ 1.15 年 |
| `purge_steps` | 对称移除训练数据（验证两侧各 N 步） | 5 步 = 5 周 |
| `embargo_steps` | 非对称延迟（仅验证之后） | 10 步 = 10 周 |

Fold 数量：**K = n_groups - n_test_groups - n_val_groups + 1**

### 运行

```bash
cd QuantPits

# CPCV 全量训练（所有 enabled 模型）
python quantpits/scripts/cv_train.py --full

# CPCV 增量训练指定模型
python quantpits/scripts/cv_train.py --models lightgbm_Alpha158,gru_Alpha158

# CPCV 按标签训练
python quantpits/scripts/cv_train.py --tag tree

# Dry-run（预览 Fold 划分计划）
python quantpits/scripts/cv_train.py --all-enabled --dry-run

# 仅预测（使用已有 CPCV 模型对新数据预测）
python quantpits/scripts/cv_train.py --predict-only --all-enabled
```

### 下游兼容性

CPCV 训练的模型以 `model_name@cpcv` 键存储在 `latest_train_records.json` 中，与 `@static` 和 `@rolling` 模型共存。下游脚本通过 `--training-mode cpcv` 过滤：

```bash
python quantpits/scripts/ensemble_fusion.py --from-config --training-mode cpcv
```

每个 CPCV 模型在 Recorder 中存储了 K 个 Fold 模型 (`model_fold_0.pkl` … `model_fold_K-1.pkl`) 和一份 K 折平均后的最终预测 (`pred.pkl`)，下游融合流程无需感知 K 折细节。

### 模型类型兼容性

| 模型类型 | 数据集类 | 实现方式 |
|---------|---------|---------|
| 树模型 (LightGBM, XGBoost, CatBoost) | `PurgedDatasetH` | `pd.concat` 非连续时间片段（安全，点预测无序列窗口） |
| 线性模型 (Linear) | `PurgedDatasetH` | 同上 |
| 深度学习 (LSTM, GRU, ALSTM, Transformer, TCN, GATs) | `PurgedTSDatasetH` | `ConcatTSDataSampler` 逻辑级联（滑动窗口不跨越 Purge Gap） |

### 预处理注意事项

- **推荐截面归一化**：`CSZScoreNorm`、`CSRankNorm`（按日/按个股独立计算，天然免疫时序泄露）
- **避免时域归一化**：`ZScoreNorm`、`MinMaxNorm`、`RobustZScoreNorm` 在全局时间范围上 Fit 会导致验证/测试集统计量泄露到训练集。CPCV 训练启动时如果检测到这些算子会发出 `UserWarning`。

---

## 发布历史与恢复证据

typed static/CPCV 不再依赖 `data/history/` 的时间戳备份保证正确性。每次实际发布使用：

```
data/training_transactions/<transaction-id>/
├── intent.json               # preimage/postimage fingerprint 与提交顺序
├── member-*.preimage         # 存在时保存精确旧字节
├── member-*.postimage        # 已 fsync 的目标字节
└── receipt.json              # verified committed outputs
```

同一个 logical run 的 target 与 closure 证据位于：

```text
data/training_runs/<run-id>/
├── targets/<target-key>.json # immutable typed target result + recorder lineage
└── closure-<transaction-id>.json # receipt-linked step status and retry evidence
```

`data/history/` 仍可能包含 legacy/rolling 工具生成的兼容备份，但不是新 static/CPCV 恢复依据。

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
