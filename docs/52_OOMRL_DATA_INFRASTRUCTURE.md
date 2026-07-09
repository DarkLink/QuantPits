# OOM-RL 数据基础设施 — Phase 1, 2 & Runtime Linkage

本文档说明 OOM-RL 系统自动收集的结构化数据、Phase 2 对各 Agent 的增强，以及 runtime plan/manifest 与操作日志的兼容关联字段。

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
    "run_id": null,
    "manifest_path": null,
    "plan_fingerprint": null,
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

### runtime manifest 关联字段

`OperatorLog` 向后兼容地新增了 3 个可选字段，供后续命令接入 `quantpits.runtime.RunManifest` 时使用：

| 字段 | 含义 |
|------|------|
| `run_id` | 本次命令运行的稳定运行 ID |
| `manifest_path` | 对应 `RunManifest` 的 workspace-relative 路径，例如 `output/manifests/{command}/{run_id}.json` |
| `plan_fingerprint` | dry-run plan / public plan dict 的稳定指纹 |

旧脚本无需修改；未设置时字段为 `null`。这些字段只记录标识、路径和指纹，不包含原始配置内容。

### 使用方式

```python
from quantpits.utils.operator_log import OperatorLog

with OperatorLog("static_train", args=sys.argv[1:]) as oplog:
    oplog.set_source("llm_critic")
    oplog.set_action_item_id("55b3a485-dfc0-4e6e-bdb3-e843ab4f5905")
    # ... 脚本主逻辑 ...
    oplog.set_result({"n_models": 20, "anchor_date": "2026-04-24"})
```

未来命令接入 run manifest 后，可以关联运行清单：

```python
with OperatorLog("ensemble_fusion", args=sys.argv[1:]) as oplog:
    oplog.set_run_manifest(
        run_id=run_id,
        manifest_path=f"output/manifests/ensemble_fusion/{run_id}.json",
    )
    oplog.set_plan_fingerprint(plan_fingerprint)
    # ... 脚本主逻辑 ...
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
    "experiment_name": "prod_train_weekly",
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

## 5. Training Context — 训练模式感知

**文件**: `quantpits/scripts/deep_analysis/training_context.py`

在 Phase 2 中引入，为所有分析代理提供当前工作区的训练模式清单和滚动管道状态信息。

### 核心能力

- **训练模式解析**: 从 `latest_train_records.json` 解析 `name@mode` 格式的模型键，识别每个模型使用的训练算法模式（static / cpcv / rolling / cpcv_rolling）。
- **跨模式检测**: `get_cross_mode_models()` 返回同时以多种训练模式训练的模型列表。
- **管道延迟计算**: `get_rolling_gap_days(mode)` 计算滚动管道上次 anchor 日期到当前日期的天数差，用于陈旧度检测。
- **模型键解析**: `resolve_model_key(record_id)` 从 record/run ID 反查完整 `name@mode` 标识，供下游 agent（如 Model Health）区分同一模型的不同训练模式。

**数据来源**（由 `from_workspace()` 工厂方法读取）:

| 文件 | 用途 |
|------|------|
| `data/latest_train_records.json` | 模型→run ID 映射、anchor 日期、`@` 后缀模式解析 |
| `config/rolling_config.yaml` | 滚动调度器配置 |
| `data/rolling_state.json` | Slide 滚动窗口进度 |
| `data/rolling_state_cpcv.json` | CPCV 滚动窗口进度 |
| `config/model_config.json` | `purged_cv` 段（CPCV 参数） |

---

## 6. Phase 2 — Agent 增强

### Training Health Agent (新增)

综合评估训练管道健康、滚动进度和交易执行趋势：

- **模式覆盖审计**: 检查每个模型的训练模式覆盖情况（static / cpcv / rolling / cpcv_rolling），标记缺少预期模式的模型。
- **滚动进度与陈旧度**: 检查滚动管道完成百分比，标记超过 90 天未更新的管道（warning），已完成管道标记 positive。
- **Alpha 衰减监控**: 对比 `rolling_metrics_20.csv` 和 `rolling_metrics_60.csv` 中的 `Idiosyncratic_Alpha` 指标。短期（20d）均值下穿长期（60d）均值且转负时，判定为选股能力衰减（warning）；短期向上穿越长期且为正时标记 positive。
- **执行摩擦检测**:
  - 滑点 z-score：对 `Exec_Slippage_Mean` 进行 60 日 z-score 检验，z < -2.0 → critical “执行摩擦崩盘”。
  - 延迟成本 z-score：对 `Delay_Cost_Mean` 进行 60 日 z-score 检验，z < -2.0 → warning “隔夜跳空恶化”。
- **Barra 因子漂移**: 对 `Exposure_Liquidity` 在 252 日滚动窗口中的百分位进行检测。≤ 5%（微盘漂移）→ critical；≥ 95%（大盘蓝筹超载）→ warning。
- **孤儿模型检测**: 发现已启用但不在任何活跃组合中的陈旧模型 → warning/info。

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
