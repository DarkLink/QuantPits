# OOM-RL 数据基础设施 — Phase 1 & 2

本文档说明 OOM-RL 系统自动收集的结构化数据，以及 Phase 2 对各 Agent 的增强。

---

## 1. OperatorLog — 操作日志

**文件**: `quantpits/utils/operator_log.py`

每次核心脚本运行自动追加一条 JSONL 记录到 `data/operator_log.jsonl`。已集成到 7 个脚本：`static_train.py`, `rolling_train.py`, `ensemble_fusion.py`, `pretrain.py`, `minentropy_ensemble.py`, `brute_force_ensemble.py`, `brute_force_fast.py`。

### 记录格式

```json
{
    "log_id": "20260429_152941_static_train_a1b2",
    "timestamp_start": "2026-04-29T15:29:41.864383",
    "timestamp_end": "2026-04-29T15:29:52.152027",
    "duration_seconds": 10.287,
    "script": "static_train",
    "args": ["--all-enabled"],
    "source": "human",
    "tags": [],
    "notes": "",
    "action_item_id": null,
    "result_summary": {"anchor_date": "2026-04-24", "n_models": 20},
    "exception": null
}
```

### source 字段

| 值 | 含义 |
|----|------|
| `human` | 人工手动运行 |
| `llm_critic` | LLM Critic 触发的操作（Phase 4 Feedback Loop） |
| `scheduled` | 定时任务触发 |

### tags 字段

支持 `["test"]` / `["experiment"]` 标签。带 `test` 标签的记录在后续分析中可被过滤。

### 使用方式

```python
from quantpits.utils.operator_log import OperatorLog

with OperatorLog("static_train", args=sys.argv[1:]) as oplog:
    oplog.set_source("llm_critic")
    oplog.set_action_item_id("55b3a485-dfc0-4e6e-bdb3-e843ab4f5905")
    # ... 脚本主逻辑 ...
    oplog.set_result({"n_models": 20, "anchor_date": "2026-04-24"})
```

---

## 2. Config Ledger — 配置分类账

**文件**: `quantpits/scripts/deep_analysis/config_ledger.py`

每次 `run_deep_analysis.py` 运行时自动对当前配置做快照，保存到 `data/config_history/config_snapshot_{date}.json`。

### 快照内容

- 所有 `workflow_config_*.yaml` 中的超参数（per-model）
- `ensemble_config.json`（活跃组合及其构成）
- `strategy_config.yaml`

### 核心函数

| 函数 | 用途 |
|------|------|
| `snapshot_configs(workspace_root, snapshot_date)` | 对当前配置做完整快照 |
| `save_snapshot(workspace_root, snapshot)` | 持久化到 `data/config_history/` |
| `load_previous_snapshot(workspace_root, before_date)` | 加载最近的先前快照 |
| `diff_snapshots(old, new)` | 对比两个快照，返回字段级变更列表 |
| `annotate_with_llm_context(records, reason, action_item_id, critic_score)` | 为变更记录附加 LLM 操作来源 |

### diff_snapshots 变更记录结构

```python
{
    "type": "hyperparam",          # hyperparam | ensemble | strategy | ensemble_switch
    "key": "gru_Alpha158.n_epochs",
    "old": 200,
    "new": 250,
    "impact_domain": "Hyperparameter",
    "semantic_label": "Tuning"     # Tuning | CapacityAdjustment | Regularization | etc.
}
```

---

## 3. Training History — 训练收敛日志

**文件**: `data/training_history.jsonl`

`train_single_model()` 在每次训练完成后自动追加一条记录。包含训练过程的收敛信息和模型表现指标。

### 记录格式

```json
{
    "model_name": "adarnn_Alpha360",
    "experiment_name": "Prod_Train_WEEK",
    "record_id": "c35fb440b0404678977240f20769e264",
    "anchor_date": "2026-04-24",
    "trained_at": "2026-04-28T16:59:25.228065",
    "duration_seconds": 1538.33,
    "early_stopped": true,
    "actual_epochs": 38,
    "configured_epochs": 200,
    "best_epoch": 17,
    "best_score": 0.05497,
    "converged": false,
    "final_train_loss": 0.03017,
    "IC_Mean": 0.04147,
    "ICIR": 0.34530,
    "Ann_Excess": -0.03828,
    "Max_DD": -0.28192,
    "Information_Ratio": -0.41790
}
```

### 关键字段

| 字段 | 说明 |
|------|------|
| `early_stopped` | 是否触发 early stopping 提前结束训练 |
| `actual_epochs` | 实际训练的 epoch 数 |
| `configured_epochs` | 配置的 epoch 数 |
| `best_epoch` / `best_score` | 最佳 epoch 及对应的验证分数 |
| `converged` | `actual_epochs == configured_epochs` (跑满即为收敛) |
| `IC_Mean` / `ICIR` | 单模型 IC 均值和信息比率 |
| `Ann_Excess` / `Max_DD` | 单模型回测的 excess return 和最大回撤 |
| `duration_seconds` | 训练耗时（用于 Phase 4 优先级调度） |

### 注意事项

- 每次训练追加一条记录（不覆盖），历史训练记录持续积累
- predict-only 周期不产生新记录
- `model_performance_{date}.json` 中的 `convergence` 字段由 `merge_performance_file()` 从旧文件继承，确保 predict-only 覆盖时 convergence 信息不丢失

---

## 4. Fusion Run Ledger — 融合运行分类账

**文件**: `data/fusion_run_ledger.jsonl`

`ensemble_fusion.py` 在每次回测完成后自动追加。由 `EnsembleEvolutionAgent` 消费。

### 关键字段

- `combo_name`: 组合名称
- `run_date`: 运行日期
- `is_oos`: 是否为 OOS (Out-of-Sample) 评估
- `calmar`, `annualized_return`, `max_drawdown`: 回测指标
- `lo_contributions`: Leave-One-Out 模型贡献度 (`{model: {loo_ic, full_ic, delta}}`)

### OOS 评估

OOS 融合通过 `--only-last-years 1` 参数触发：
```bash
python -m quantpits.scripts.ensemble_fusion --from-config-all --only-last-years 1
```

`oos_config.json` 控制默认排除周期：
```json
{
    "exclude_last_years": 1,
    "exclude_last_months": 0
}
```

---

## 5. Phase 2 — Agent 增强

### Model Health Agent

- **训练日期追溯**: 从 `training_history.jsonl` 中提取每模型的最近一次真实训练（非 predict-only）日期
- **收敛状态检测**: 判断模型是否 `underfitting`（early stopped 过早）、`overfitting`（跑满但 IC 低）、正常收敛
- **过时检测**: 标记超过 N 周未训练的模型

### Ensemble Eval Agent

- **OOS 历史对比**: 读取 `fusion_run_ledger.jsonl` 中的 OOS 记录，计算 OOS Calmar 趋势斜率
- **LOO 贡献度**: 消费 `ensemble_fusion.py` 生成的 `model_contribution_{date}.json`
- **组合变更检测**: 三层变化（构成变化 / 活跃组合切换 / 内容变异）

### Market Regime Agent

- 滑窗检测 regime switch 事件及频率
- 输出: 趋势标签（牛市/熊市/震荡）、波动率百分位、回撤深度

### Prediction Audit Agent

- 从 `model_opinions_{date}.json` 拆解每个单模型的独立预测胜率（hit rate）
- 共识 vs 分歧分析
