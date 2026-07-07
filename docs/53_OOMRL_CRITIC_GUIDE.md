# OOM-RL LLM Critic — Phase 3

LLM Critic 将 Deep Analysis 的 Agent 发现转化为结构化的、可执行的 ActionItem，是 OOM-RL 从"分析系统"升级为"决策系统"的关键组件。

---

## 架构

```
7-Stage Pipeline (Stage 1-4)
      │
      ▼
Stage 5: Signal Extractor  ──→  List[Signal]   (纯规则层，从 findings 提取)
      │
      ▼
Stage 6: LLM Critic        ──→  List[ActionItem]  (LLM 综合决策)
      │
      ▼
ActionItem Validator  ──→  scope_status 标注 (in_scope / out_of_scope / rejected)
      │
      ▼
persist_action_items()  ──→  action_items_{date}.json + action_item_history.jsonl
```

---

## 1. Signal Extractor

**文件**: `quantpits/scripts/deep_analysis/signal_extractor.py`

纯规则层。从 Agent 的 `raw_metrics` 中提取结构化 Signal。**不做决策**——所有决策交由 LLM Critic。

### Signal 结构

```python
@dataclass
class Signal:
    signal_type: str    # 信号类型 (见下表)
    severity: str       # "critical" | "warning" | "info"
    scope: str          # 对应的 feedback_scope
    source_agent: str   # 产生该信号的 Agent 名称
    target: str         # 受影响的模型/组合名称
    metrics: dict       # 相关指标数据
    context: str        # 人类可读的一句话描述
```

### 16 种 Signal 类型

| Signal Type | 来源 Agent | Scope | 触发条件 |
|-------------|-----------|-------|---------|
| `underfitting` | Model Health | hyperparams | 模型在 underfitting_candidates 列表中 |
| `severe_underfitting` | Model Health | hyperparams | `actual_epochs < configured * 0.25` |
| `overfitting` | Model Health | hyperparams | 跑满 epoch 但 IC mean < 0.03 |
| `ic_decay` | Model Health | hyperparams | scorecard 中 ic_trend == "degrading" |
| `model_stale` | Model Health | hyperparams | stale_models 且 recommend_retrain=True |
| `oos_degradation` | Ensemble Eval | combo_search | OOS Calmar slope < -0.3 且 runs >= 5 |
| `oos_degradation_limited_sample` | Ensemble Eval | combo_search | 同上但 runs < 5 (低置信度) |
| `negative_contribution` | Ensemble Eval | model_selection | 模型在 consistently_negative 列表 |
| `poor_predictor` | Prediction Audit | model_selection | 模型在 underperformers 列表 |
| `regime_instability` | Market Regime | hyperparams | regime_switches >= 3 |
| `combo_stale` | Ensemble Eval | combo_search | 上次评估 > 30 天 |
| `training_window_mismatch` | Training Window Analyzer | training_config | 规则检测到窗口配置问题（bound check、ratio check、train-end gap、anchor staleness、regime mismatch） |
| `cross_agent_convergence` | Cross-Agent | (最佳 scope) | 同一 target 被 >= 2 个 agent 以 warning 标记 |
| `time_horizon_reversal` | Cross-Agent | (最佳 scope) | 时间维度上的 OOS 收益方向发生反转 |
| `combo_fragility` | Ensemble Eval / Cross-Agent | combo_search | 组合对单一模型的依赖度过高（脆弱性） |
| `orphan_model` | Training Health | model_selection | Predict-only 模式下发现遗留且未更新的模型 |

---

## 2. LLM Critic

**文件**: `quantpits/scripts/deep_analysis/llm_interface.py`

`LLMInterface.generate_action_items(signals)` 方法将 Signal 列表转化为 ActionItem 列表。

### 工作流程

1. 加载 `config/llm_config.json` 获取模型/API 配置
2. 加载 `config/feedback_scope.json` 获取 `active_scopes`
3. 加载 `config/hyperparam_bounds.json` 获取超参边界
4. 加载 `config/skills/` 中的 Markdown skill 文件作为系统提示词
5. 构建包含 active scopes、bounds 和 signals JSON 的 prompt
6. 调用 OpenAI 兼容 API（含 2 次 JSON 解析重试）
7. 解析 JSON 数组响应为 `List[ActionItem]`

### 降级策略

- 无 API key → 返回空列表
- API 调用失败 → 返回空列表
- JSON 解析失败 → 重试一次，仍失败则返回空列表

### LLM 配置

`config/llm_config.json`:
```json
{
    "critic_model": "deepseek-v4-pro",
    "summary_model": "deepseek-v4-pro",
    "available_models": ["deepseek-v4-pro", "gpt-4o", "claude-sonnet-4"],
    "api_key_env": "DEEPSEEK_API_KEY",
    "base_url": "https://api.deepseek.com",
    "temperature": 0.3,
    "max_tokens": 393216
}
```

### Skills 文件

`config/skills/` 目录下的 Markdown 文件会被拼接为 LLM 系统提示词：

| Skill 文件 | 内容 |
|-----------|------|
| `critic_system.md` | 核心决策原则（保守调整、数据驱动、可验证、范围约束）和 JSON 输出格式 |
| `hyperparam_tuning.md` | 超参调整领域知识（欠拟合→增加 n_epochs/hidden_size，过拟合→增加 dropout/l2_leaf_reg） |
| `model_selection.md` | 模型选择准则（禁用条件：负贡献 ≥3 次评估；启用条件：IC>0.03 且 ICIR>0.3） |
| `summary_system.md` | 综合摘要的写作指导 |

---

## 3. ActionItem

**文件**: `quantpits/scripts/deep_analysis/action_items.py`

### ActionItem 结构

```python
@dataclass
class ActionItem:
    action_id: str              # UUID, 自动生成
    action_type: str            # "adjust_hyperparam" | "disable_model" | "trigger_search" | "adjust_training_window"
    scope: str                  # "hyperparams" | "model_selection" | "combo_search" | "strategy_params" | "training_config"
    target: str                 # 目标模型/组合名称 (如 "alstm_Alpha158")
    params: dict                # e.g. {"early_stop": {"from": 10, "to": 20}}
    reason: str                 # LLM 给出的变更理由
    source_signals: list        # 触发此 ActionItem 的 signal_type 列表
    expected_outcome: str       # 预期效果描述
    confidence: float           # LLM 自评估置信度 [0, 1]
    risk_level: str             # "low" | "medium" | "high"

    # 验证层字段 (由 ActionItemValidator 设置)
    scope_status: str           # "pending" | "in_scope" | "out_of_scope" | "rejected"
    rejected_reason: str        # 拒绝原因
    validated_at: str           # 验证时间

    # Phase 4 执行上下文
    execution_context: dict     # {target_env, requires_retrain, requires_backtest,
                                #  estimated_duration_minutes, dependencies}
```

### ActionItemValidator

验证规则（按顺序）：
1. **Scope 检查**: `scope` 不在 `active_scopes` → `out_of_scope`
2. **值域检查**: `params[key]["to"]` 超出 `hyperparam_bounds.json` 的 `[min, max]` → `rejected`
3. **变更幅度检查**: 变更百分比超过 `max_change_pct` → `rejected`
4. **未知参数**: 不在 bounds 中的参数允许通过（warn）

### 超参边界

`config/hyperparam_bounds.json`:
```json
{
    "bounds": {
        "n_epochs":    {"min": 10, "max": 500,   "max_change_pct": 50},
        "lr":          {"min": 1e-5, "max": 1e-2, "max_change_pct": 100},
        "learning_rate": {"min": 1e-5, "max": 1.0, "max_change_pct": 100},
        "hidden_size": {"min": 16, "max": 512,    "max_change_pct": 100},
        "num_layers":  {"min": 1, "max": 8,       "max_change_pct": 100},
        "dropout":     {"min": 0.0, "max": 0.8,   "max_change_pct": null},
        "batch_size":  {"min": 64, "max": 16384,  "max_change_pct": 100},
        "iterations":  {"min": 100, "max": 10000, "max_change_pct": 50},
        "depth":       {"min": 2, "max": 16,      "max_change_pct": 100},
        "l2_leaf_reg": {"min": 0.0, "max": 100.0, "max_change_pct": null},
        "num_leaves":  {"min": 8, "max": 256,     "max_change_pct": 100},
        "early_stop":  {"min": 5, "max": 100,     "max_change_pct": null}
    }
}
```

### 数据持久化

```python
persist_action_items(items, workspace_root, run_date)
```

写入两个位置：
- `output/deep_analysis/action_items_{date}.json` — 完整快照
- `data/action_item_history.jsonl` — 追加审计轨迹（含 `_run_date`）

---

## 4. CLI 使用

```bash
# 运行 Deep Analysis 并启用 Critic（产出 ActionItems）
python -m quantpits.scripts.run_deep_analysis --critic

# 预览模式：生成 ActionItems 但不持久化到文件
python -m quantpits.scripts.run_deep_analysis --critic-dry-run

# 使用 LLM 生成综合摘要
python -m quantpits.scripts.run_deep_analysis --llm --critic

# 指定 Agent 子集运行
python -m quantpits.scripts.run_deep_analysis --critic --agents model_health,ensemble_eval
```
