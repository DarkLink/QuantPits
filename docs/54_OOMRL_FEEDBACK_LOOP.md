# OOM-RL 反馈闭环执行 — Phase 4

Phase 4 实现了 ActionItem 从建议到执行到验证到推广的完整闭环。

---

## 架构

```
action_items_{date}.json
         │
         ▼
┌─ feedback_loop.py (Orchestrator) ────────────────────────┐
│                                                          │
│  1. 计算优先级 + 时间预算                                  │
│  2. 创建/同步 Playground 工作区                            │
│  3. TrainingAdapter 修改 YAML 配置                        │
│  4. 在 Playground 中重训模型                               │
│  5. 单模型 IC 对比验证                                     │
│  6. 输出 feedback_report_{date}.json                      │
│  7. (手动) promote_config.py 推送到生产                    │
│                                                          │
│  模式:                                                    │
│    --report-only  仅生成报告，不执行                        │
│    --execute      执行但手动 promote                       │
│    --promote      推广已验证的变更到生产                     │
│    --auto-promote 全自动（Phase 4 暂不实现）                │
└──────────────────────────────────────────────────────────┘
```

---

## 1. Playground Manager

**文件**: `quantpits/scripts/deep_analysis/playground_manager.py`

Playground 是生产工作区的一个轻量级 sibling 副本，位于生产同级目录 `{workspace_name}_Playground`。

### 同步策略

| 目录/文件 | 同步方式 | 说明 |
|-----------|---------|------|
| `config/` | 完整拷贝 | Adapter 可自由修改 |
| `data/*.xlsx`, `data/*.csv` | symlink | 券商数据，只读共享 |
| `data/order_history/`, `data/history/`, `data/config_history/` | symlink 目录 | 只读历史数据 |
| `data/training_history.jsonl` | 实体拷贝 | Playground 训练独立写入 |
| `data/fusion_run_ledger.jsonl` | 实体拷贝 | Playground 回测独立写入 |
| `data/operator_log.jsonl` | 实体拷贝 | 操作日志隔离 |
| `latest_train_records.json` | 实体拷贝 | 训练记录独立 |
| `data/pretrained/` | 实体拷贝 | 预训练权重（当前未启用） |
| `output/`, `mlruns/`, `archive/` | 不同步 | 独立产出 |

### 工作区隔离

`env.set_root_dir(path)` 在运行时切换 `ROOT_DIR`，并同步更新 `train_utils` 的全部 9 个模块级路径常量。Playground 中的训练和回测自动写入隔离的 `data/` 和 `mlruns/`。

```python
from quantpits.utils import env
env.set_root_dir(playground_root)  # 切换到 Playground
# train_single_model() 等函数现在使用 playground 路径
env.set_root_dir(production_root)  # 恢复生产路径
```

### 核心接口

```python
class PlaygroundManager:
    def __init__(self, production_root: str)
    def create_or_sync(self) -> str     # 创建/同步 Playground, 返回 playground_root
    def get_playground_root(self) -> Optional[str]
    def clean(self)                     # 删除 Playground 目录
    def get_meta(self) -> dict          # 读取 _playground_meta.json
```

---

## 2. Training Adapter

**文件**: `quantpits/scripts/deep_analysis/adapters/training_adapter.py`

将 `adjust_hyperparam` 类型的 ActionItem 转化为 `workflow_config_*.yaml` 的实际修改。

### 修改流程

1. 从 `model_registry.yaml` 查找 target 对应的 `yaml_file`
2. 使用 `ruamel.yaml` 读取 YAML（保留注释和格式）
3. 校验当前值是否等于 `params[key]["from"]`（安全检查）
4. 校验 `params[key]["to"]` 是否在 `hyperparam_bounds.json` 范围内（二次校验）
5. 修改为 `params[key]["to"]`
6. 自动备份原文件到 `config/_backup/workflow_config_xxx.yaml.{timestamp}`
7. 写回 YAML（保留注释）

### 预训练依赖检查

`check_pretrain_deps(item)` 检查目标模型的 `pretrain_source` 依赖是否在 Playground 中存在：
- 如缺失 → 从生产 `data/pretrained/` 拷贝
- 如生产也没有 → 记录 warning 但不阻塞（降级为随机初始化）

### 核心接口

```python
class TrainingAdapter:
    def __init__(self, workspace_root: str)
    def apply(self, item: ActionItem) -> AdapterResult
    def preview(self, item: ActionItem) -> dict       # 干跑，不写文件
    def check_pretrain_deps(self, item: ActionItem) -> List[str]
```

### AdapterResult

```python
@dataclass
class AdapterResult:
    success: bool
    action_id: str
    adapter_type: str              # "training" | "search" | "fusion"
    modified_files: List[str]
    changes: List[dict]            # [{param, old, new, file}]
    error: str
```

### Adapter 注册

`adapters/__init__.py` 使用装饰器模式注册：

```python
@register_adapter("adjust_hyperparam")
class TrainingAdapter(BaseAdapter):
    ...
```

后续新增 SearchAdapter 或 FusionAdapter 只需装饰即可自动注册到 Orchestrator。

---

## 3. Feedback Loop Orchestrator

**文件**: `quantpits/scripts/deep_analysis/feedback_loop.py`

### 执行模式

| 模式 | 行为 |
|------|------|
| `--report-only` | 读取 ActionItems，生成变更预览 + 优先级排序，**不执行** |
| `--execute` | 创建 Playground → Adapter → 重训 → 单模型 IC 验证 → 报告 |
| `--promote` | 读取上次 `--execute` 的报告，将通过的变更推广到生产 |
| `--auto-promote` | Phase 4 暂不实现，返回提示信息 |

### 优先级调度

`compute_priority(item, signal_severity, training_history)` 基于四个维度计算优先级：

| 维度 | 权重逻辑 |
|------|---------|
| Signal 严重度 | critical=3.0, warning=2.0, info=1.0 |
| LLM 置信度 | confidence × 2.0 (0 ~ 2.0) |
| 风险等级 (tiebreaker) | high=0.5, medium=0.3, low=0.0 |
| 训练耗时 | 短训练优先：< 10 min +1.0, < 30 min +0.5 |

### 时间预算

`--max-duration-minutes N` 限制总执行时间：
- 基于 `training_history.jsonl` 中的历史训练时长估算每个模型的时间
- 未知模型默认估算 60 分钟
- 贪心选取能在预算内完成的最高优先级 ActionItem
- 超出预算的标记为 `deferred`，写入报告

### 人工干预

```bash
--models m1,m2          # 只处理指定模型（覆盖自动排序）
--skip-models m3,m4     # 排除特定模型
--max-duration-minutes 120  # 时间预算
```

### 验证策略

**主验证方式：单模型 IC 对比。** 重训后对比 Playground 与 Production 的单模型 IC：

- 通过条件：`playground_ic >= baseline_ic * 0.9`
- 优先条件：`ic_delta > 0`
- 失败处理：记录失败原因，**继续执行下一个**（不阻塞流程）

**可选的 Ensemble 级验证** (`--with-ensemble-backtest`): 从生产拷贝未重训模型的预测，仅替换重训模型的预测列，跑 combo 回测。当前为可选功能。

### FeedbackReport

```python
@dataclass
class FeedbackReport:
    run_date: str
    mode: str
    action_items_processed: int
    action_items_deferred: int       # 因时间预算不足推迟的
    adapter_results: List[dict]
    validation_results: List[dict]
    deferred_action_ids: List[str]
    promote_result: Optional[dict]
    summary: str
```

输出到 `output/deep_analysis/feedback_report_{date}.json`。

---

## 4. Config Promoter

**文件**: `quantpits/scripts/deep_analysis/promote_config.py`

将 Playground 中验证通过的配置变更推广到生产，生成完整审计轨迹。

### Promote 流程

1. `diff_snapshots(production, playground)` → 获取变更列表
2. `annotate_with_llm_context()` → 标注变更来源
3. 生成人类可读 Promote Summary (Markdown)
4. 拷贝修改过的文件到 Production（仅 config/，不拷贝 output/mlruns）
5. 保存新 config snapshot
6. 写入 `promote_history.jsonl`
7. 更新 `CHANGELOG.md`

> **重要**: Promote 仅推送**配置**，不包含模型权重。被 promote 的配置需要在下一次生产训练周期 (`static_train --all-enabled`) 中重新训练才能生效。

### Promote 状态生命周期

```
promoted_pending_retrain  →  active  →  (可选) rolled_back
```

| 状态 | 说明 |
|------|------|
| `promoted_pending_retrain` | 配置已推送，等待下次训练生效 |
| `active` | 训练完成，新配置已在生产中生效 |
| `rolled_back` | 因验证失败或问题回退 |

`static_train.py` 和 `rolling_train.py` 在训练完成后自动调用 `update_promote_status()` 将 `pending_retrain` → `active`（try/except 包裹，非阻塞）。

### 审计产物

每次 promote 生成两份记录：

1. **机器可读**: `data/promote_history.jsonl`
```json
{
    "promote_id": "uuid",
    "promoted_at": "2026-05-05 17:30:00",
    "action_item_ids": ["55b3a485-..."],
    "changes": [{"model": "gru_Alpha158", "param": "early_stop", "old": 10, "new": 20}],
    "source": "llm_critic",
    "status": "promoted_pending_retrain",
    "retrained_at": null,
    "rolled_back_at": null,
    "rollback_reason": null,
    "validation_result": {...},
    "human_readable_report": "data/promote_history/promote_2026-05-05.md"
}
```

2. **人类可读**: `data/promote_history/promote_{date}.md`
   - 变更摘要
   - Before/After 对比表
   - IC 验证结果对比表
   - ActionItem 溯源（Signal → ActionItem → Promote）
   - 回退指南（git checkout 命令）

### 变更历史总览

`data/CHANGELOG.md` 自动维护，按时间倒序列出每次变更：

```markdown
# Demo_Workspace 配置变更历史

## 2026-05-05: early_stop 调整 — 3 模型重训通过
- **来源**: LLM Critic (ActionItem 55b3a485, ac08cbd1, 6ee7122e)
- **变更**: alstm_Alpha158, lstm_Alpha360, gru_Alpha158 的 early_stop: 10 → 20
- **验证**: 3/3 模型 IC 改善 (avg +0.005), 已 promote
- **风险**: low
- [详细报告](promote_history/promote_2026-05-05.md)
```

---

## 5. 回退

生产 workspace 是独立 Git 仓库，回退利用 Git 版本控制：

```bash
# 1. 找到目标 promote 记录
cat data/promote_history/promote_2026-05-05.md

# 2. 恢复配置（通过 Git）
cd /path/to/Demo_Workspace
git checkout <pre-promote-commit> -- config/

# 3. 标记回退
# promote_config.py 中提供了 rollback 辅助函数
```

回退本身也是一个 promote 事件（方向相反），在审计轨迹中完整记录。

---

## 6. CLI 使用

```bash
# 报告模式 — 预览将要执行的变更
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --report-only

# 执行模式 — 在 Playground 中执行变更
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute

# 执行 + 时间预算 + 模型筛选
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute --max-duration-minutes 30 --models gru_Alpha158,alstm_Alpha158

# 干跑 — 预览适配器变更但不写文件
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute --dry-run

# 仅修改配置不重训
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute --skip-retrain

# 推广 — 将验证通过的变更推送到生产
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --promote
```
