# M0: 模型 Wrapper 架构总览

## 概述

`quantpits/utils/model_wrappers/` 为 qlib 的 PyTorch 模型提供增强 wrapper，在不修改上游 qlib 代码的前提下扩展：

- **IC / Rank IC / IR / loss 早停** — 统一通过 YAML 的 `metric` 参数控制
- **IC loss**（1 - Pearson 相关系数）— 可作为训练目标替代 MSE
- **Loss 历史记录** — 用于收敛诊断
- **组合 IR 评估** — 在验证集上运行轻量 TopK 回测

核心原则：**一个模型架构一个 wrapper 文件**。引擎代码不硬编码任何策略参数（TopK 从 workspace config 注入）。

---

## 目录结构

```
quantpits/utils/model_wrappers/
├── mixins/                       # 可复用组件（多重继承，按需堆叠）
│   ├── ic.py                     # ICMetricMixin + ICLoss
│   ├── strategy.py               # StrategyMetricMixin（IR回测、IC loss、metric派发）
│   └── loss_history.py           # LossHistoryMixin（捕获每epoch的train/valid loss）
│
├── custom/                       # 生产 wrapper，一个模型一个文件
│   ├── pytorch_lstm.py           # LSTM（numpy范式，metric: ir/ic/rank_ic/loss）
│   ├── pytorch_alstm_ts.py       # ALSTM TS（DataLoader + ConcatDataset + hooks）
│   ├── pytorch_gats_plus.py      # GATsPlus standalone（DailyBatchSampler + 预训练基模型）
│   ├── pytorch_lstm_ic_loss.py   # LSTM IC-loss standalone（自定义test_epoch → preds/labels）
│   ├── pytorch_general_nn.py     # GeneralPTNN（minimize模式，IR取反适配）
│   └── ...（共 24 个 wrapper）
│
├── lh/                           # LossHistory 薄封装，文件名对齐 custom/
│   ├── pytorch_lstm.py           # class LSTM(LossHistoryMixin, _CustomLSTM): pass
│   └── ...（共 19 个 wrapper）
│
└── experiment/                   # 临时实验空间
    └── .gitkeep
```

### 命名约定

| 模式 | 示例 | 含义 |
|------|------|------|
| `pytorch_<架构>.py` | `pytorch_lstm.py` | 标准 wrapper（numpy 或 DataLoader 范式） |
| `pytorch_<架构>_ts.py` | `pytorch_alstm_ts.py` | 时序变体（DataLoader-based fit） |
| `pytorch_<架构>_ic_loss.py` | `pytorch_lstm_ic_loss.py` | Standalone 模型（自定义 fit()） |
| `pytorch_<架构>_rank.py` | `pytorch_lstm_rank.py` | Rank 变体 standalone |

---

## Mixin 层级

每个 wrapper 通过多重继承组合。MRO 按固定层次排列：

```
Wrapper (custom/pytorch_xxx.py)
  └─ StrategyMetricMixin (mixins/strategy.py)
       ├─ metric_fn: 派发 ir → IC, ic → Pearson, rank_ic → Spearman, loss → -MSE
       ├─ loss_fn:   派发 ic → 1-Pearson, mse → 父类
       └─ fit():     派发 ir → IR回测循环, 其他 → 父类
  └─ QlibBase (qlib.contrib.model.pytorch_xxx)
       ├─ train_epoch, test_epoch, predict
       └─ 内部 nn.Module (lstm_model, gru_model, ...)
```

### 各 mixin 职责

| Mixin | 提供 | 使用者 |
|-------|------|--------|
| `StrategyMetricMixin` | `metric_fn`（ir/ic/rank_ic/loss）、`loss_fn`（ic/mse）、`fit()`（IR循环）、mini-backtest、inner-module 发现 | 所有 `custom/` wrapper |
| `ICMetricMixin` | `metric_fn`（ic/rank_ic）、`_batch_pearson_ic`、`_batch_rank_ic` | 仅 `custom/pytorch_general_nn.py`（minimize 取反需要） |
| `LossHistoryMixin` | Monkey-patch `test_epoch` 捕获每 epoch loss | 所有 `lh/` wrapper |

`StrategyMetricMixin` 对大多数模型已完全覆盖 `ICMetricMixin` 的功能。唯一的例外是 `GeneralPTNN`——因为 minimize 取反需要单独保留 `ICMetricMixin`。

---

## `metric` 派发机制

YAML 中唯一的 `metric` 参数同时控制验证分数（早停用）和 batch 级分数（日志用）：

```
YAML metric 值
    │
    ├── "ir"  ──→ StrategyMetricMixin.fit() IR回测循环
    │            metric_fn 退到 Pearson IC（batch日志兼容）
    │
    ├── "ic"  ──→ 父类 fit()（ICMetricMixin 或 qlib 基类）
    │
    ├── "rank_ic" ──→ 父类 fit()
    │
    └── "loss" / "mse" / "" ──→ 父类 fit()，负MSE
```

当 `metric: ir` 时，`fit()` 每 epoch 收集完整验证集预测值，运行轻量 TopK 等权回测（无交易成本），用年化组合 IR 作为早停信号。传统 IC / Rank IC 仍记录到 `evals_result` 用于诊断。

---

## 模型范式

### 1. Numpy-based（多数模型）

标准 qlib fit 模式：`dataset.prepare()` 返回 DataFrame，`train_epoch(x, y)` 收 numpy 数组。`StrategyMetricMixin` 提供默认 hook 实现。

模型：LSTM、GRU、TCN、ALSTM、Transformer、Localformer、SFM、Sandwich、ADARNN、
KRNN、IGMTF、TabNet、TRA（13 个模型，全部 Alpha360）。

### 2. DataLoader-based + hooks（TS 变体）

`train_epoch(data_loader)` 收 DataLoader。不同模型的 DataLoader 构建方式不同（ConcatDataset+权重、简单 DataLoader、DailyBatchSampler）。wrapper 覆盖 `StrategyMetricMixin` 中的四个 hook：

| Hook | 作用 |
|------|------|
| `_prepare_fit_data(dataset)` | 构建 train/valid DataLoader → 返回 (train, valid, index) |
| `_train_one_epoch(train_data)` | 调用 `self.train_epoch(data_loader)` |
| `_eval_one_epoch(data)` | 调用 `self.test_epoch(data_loader)` → (loss, score) |
| `_collect_predictions(valid_data, valid_index)` | 对 valid DataLoader 做 forward → (preds, labels, loss) |

模型：ALSTM_TS、Transformer_TS、Localformer_TS、TCN_TS、GATs_TS（5 个模型）。

### 3. Standalone（自定义 fit）

模型有自己的 `fit()`，无法用标准 hook 模式。IR 支持通过 `_fit_ir()` 方法直接内嵌在 wrapper 文件中，从 `fit()` 派发调用：

```python
def fit(self, dataset, evals_result=None, save_path=None):
    if getattr(self, "metric", "") == "ir":
        return self._fit_ir(dataset, evals_result, save_path)
    return super().fit(dataset, evals_result=evals_result, save_path=save_path)
```

模型：GATsPlus（DailyBatchSampler + 预训练基模型）、LSTMICModel（自定义 test_epoch 返回 preds/labels）、ADD（每日批次 + 市场标签）。

### 4. Minimize-based（GeneralPTNN）

GeneralPTNN 的 `fit()` 使用最小化：`best_score = np.inf`，`val_score < best_score`。
IR 是越高越好，必须在 `_fit_ir()` 中取反后适配。

模型：GRU_Alpha158、MLP_Alpha158。

---

## TopK 注入

引擎代码不硬编码 TopK。流转路径：

```
strategy_config.yaml (topk: 22)
  → load_workspace_config()
    → params['topk']
      → train_single_model() 设置 model.topk = params['topk']
        → StrategyMetricMixin.fit() 使用 self.topk
```

---

## 不适用

树模型 / 线性模型（LightGBM、CatBoost、Linear）使用 sklearn API，不走 PyTorch wrapper 系统。

---

## Wrapper 能力矩阵

`quantpits/model_capabilities/` 提供只读、machine-readable 的 wrapper × dataset × action × execution-family
能力矩阵。public catalog 对 43 个 repository wrapper module（`custom/` 24 个、`lh/` 19 个）以及一个 sanitized
`LinearModel + DatasetH` passthrough declaration 建立 atomic rows；每个模型按 `train`、`incremental`、
`predict_only`、`resume` 与 `static`、`cpcv`、`rolling`、`cpcv_rolling` 展开。catalog 不读取 workspace registry、
workflow、recorder 或 MLflow backend。

terminal status 固定为：

| Status | 含义 | 允许能力 |
|---|---|---|
| `supported_verified` | exact row 的全部 required predicates 都由 inspector 实际检查并通过 | render/query + 后续 execution preflight proposal |
| `unsupported` | exact dataset/action/family 组合明确不兼容 | render only |
| `conditional` | optional dependency 或 device 条件未满足 | render；条件变化后重新 inspect |
| `coverage_unsafe` | prediction tail/gap/unique/finite 或 processor tail predicate 失败 | render only |
| `not_comparable` | 未建立完整 protocol/artifact 可比较事实 | render only |
| `invalid_declaration` | strict raw identity 或 duplicate declaration 非法 | audit only |
| `probe_failed` | isolated row probe 发生普通 operational failure | render only |

import、class resolution、constructor/fit/predict signature 成功都只是 predicate fact，不能单独形成
`supported_verified`。serialized replay、public constructor 和 `dataclasses.replace()` 也不能制造 inspector
provenance、aggregate count 或 preflight capability。矩阵仍未接入 legacy static/CPCV/Rolling CLI；Phase 34
exact-unit kernel 只查询所选 unit 的 exact row。positive row 不是训练、恢复、publication 或模型质量成功证据。
该 kernel 的 execution authority 还要求 Prepared/Resolved scope、完整 runtime/provider/backend identity、runner 前后
唯一新增 recorder provenance、Phase 32 exact recovery proposal 与完整 source-selector State CAS 同时成立。

默认 inspector 已连接 fail-closed protocol adapter registry。唯一 actual positive 是
`qlib.contrib.model.linear.LinearModel + qlib.data.dataset.DatasetH + point_in_time + train + rolling +
qlib_recorder_model_v1 + python_qlib`。它在隔离临时目录内使用 generated fixture，实际完成 fit、model artifact
roundtrip、reload、predict，并检查 processor input/output、artifact source/type 与 tail/gap/unique/finite；相邻
action/family/dataset row 不会自动升级。其他 generated fixture、覆写 dataset behavior、探针自写 artifact sidecar
或从请求 identity 回显 action/profile 都只能作为 `harness_self_test_only`，不能授予 positive authority，未实现
exact adapter 的 row 仍返回 `not_comparable / protocol_adapter_not_available`。
controlled import 还会验证默认 constructor 可绑定；内部 wrappers 检查 `fit(dataset, evals_result)`，上述 external
Linear row 检查其真实 `fit(dataset, reweighter)`/dataset contract，并统一检查 `predict(dataset)`，同时拒绝
workspace/backend activation。zero-write observer 默认覆盖仓库公共边界（包括 `quantpits/`、`docs/`、`tests/` 和
root output/cache），但排除 `.git/`、ignored plan 和私有 `workspaces/`；observer 以文件事件加结构元数据工作，
不读取或渲染 output 业务内容。即使同时检测到写入，子进程或 probe 的
`KeyboardInterrupt`、`SystemExit`、`GeneratorExit` 仍优先透传。

```python
from quantpits.model_capabilities import ModelCapabilityInspector

matrix = ModelCapabilityInspector().inspect_catalog()
payload = matrix.to_public_dict()
```

完整 catalog 仍因其余 required rows 被阻断而 `preflight_allowed=false`；执行方只能 exact-query 所选 unit，不能用
唯一 positive subset 缩小 requested set。
保持 qlib 原始 module_path。
