# OOM-RL 闭环反馈系统 — 概览

## 概念

OOM-RL (Out-of-Money Reinforcement Learning) 是一个自动化反馈系统，将 Deep Analysis 的分析结果转化为可执行的模型优化动作，并在安全的 Playground 沙箱中验证后推送到生产环境。

```
每周运行                         按需运行 (Phase 4)
┌─────────────────────┐         ┌──────────────────────────┐
│ run_deep_analysis   │         │ run_feedback_loop        │
│  --critic           │         │  --execute / --promote   │
│                     │         │                          │
│ MAS Agents          │         │ Playground Manager       │
│   ↓                 │         │   ↓                      │
│ Signal Extractor    │         │ Training Adapter         │
│   ↓                 │         │   ↓                      │
│ LLM Critic          │ ──→     │ Retrain (Sandbox)        │
│   ↓                 │   AI    │   ↓                      │
│ ActionItems JSON    │         │ IC Validation            │
│                     │         │   ↓                      │
│                     │         │ Config Promoter          │
└─────────────────────┘         └──────────────────────────┘
```

## 四个阶段

| 阶段 | 交付物 | 说明 |
|------|--------|------|
| **Phase 1** — 数据基础设施 | `operator_log.py`, `config_ledger.py`, 训练收敛日志 | 操作审计、配置快照、训练指标自动收集 |
| **Phase 2** — Agent 信号增强 | Model Health / Ensemble Eval / Market Regime / Prediction Audit 增强 | 收敛检测、OOS 历史对比、LOO 贡献度、Regime 转换 |
| **Phase 3** — LLM Critic | `signal_extractor.py`, `llm_interface.py` (critic), `action_items.py` | 规则层信号提取 → LLM 决策 → 结构化 ActionItem |
| **Phase 4** — 执行层 | Playground, Adapters, Orchestrator, Promoter | ActionItem 的自动执行、验证和推广 |

## 数据流

```
训练脚本                        Deep Analysis                   Feedback Loop
────────                        ─────────────                   ─────────────
train_single_model()            run_deep_analysis.py            run_feedback_loop.py
  │                               │                               │
  ├─ training_history.jsonl      ├─ config_snapshot_{date}.json  ├─ Playground fork
  ├─ model_performance_{date}.json  ├─ agent findings              ├─ TrainingAdapter.apply()
  ├─ latest_train_records.json   ├─ signal_extractor.py          ├─ train_single_model()
  └─ operator_log.jsonl          ├─ llm_interface.py (critic)    ├─ single-model IC check
                                   ├─ action_items_{date}.json    ├─ promote_config.py
                                   └─ action_item_history.jsonl   └─ promote_history.jsonl
```

## 反馈范围控制

系统通过 `config/feedback_scope.json` 控制 LLM Critic 可以干预哪些环节：

```json
{
    "active_scopes": ["hyperparams"],
    "available_scopes": {
        "hyperparams": {
            "description": "模型超参调整 (n_epochs, lr, dropout 等)",
            "agents": ["model_health"],
            "adapters": ["training_adapter"],
            "enabled": true
        },
        "model_selection": {
            "description": "模型启用/禁用",
            "agents": ["model_health", "prediction_audit"],
            "adapters": ["training_adapter"],
            "enabled": false
        },
        "combo_search": {
            "description": "触发组合搜索和组合更新",
            "agents": ["ensemble_eval", "portfolio_risk"],
            "adapters": ["search_adapter", "fusion_adapter"],
            "enabled": false
        },
        "strategy_params": {
            "description": "TopK/DropN/流动性约束调整",
            "agents": ["execution_quality", "trade_pattern"],
            "adapters": ["fusion_adapter"],
            "enabled": false
        },
        "training_config": {
            "description": "训练数据划分配置调整（窗口大小、切片模式）",
            "agents": ["market_regime", "model_health"],
            "adapters": ["data_split_adapter"],
            "enabled": false,
            "focus_metrics": ["IC", "ICIR", "OOS Calmar"]
        }
    }
}
```

超出 `active_scopes` 的 ActionItem 会被标记为 `out_of_scope`，仅在报告中提示但不执行。

典型使用节奏：
1. **第一阶段**: `["hyperparams"]` — 先把模型超参调好
2. **第二阶段**: `["hyperparams", "model_selection"]` — 优化模型启用/禁用
3. **第三阶段**: `["combo_search"]` — 优化组合构成
4. **第四阶段**: 全部打开

## 文件索引

| 文档 | 内容 |
|------|------|
| [50 — Deep Analysis 指南](50_DEEP_ANALYSIS_GUIDE.md) | 基础 MAS 系统用法 |
| [51 — 本文档](51_OOMRL_FEEDBACK_OVERVIEW.md) | OOM-RL 闭环反馈概览 |
| [52 — 反馈数据基础设施](52_OOMRL_DATA_INFRASTRUCTURE.md) | OperatorLog, Config Ledger, 训练收敛日志 |
| [53 — LLM Critic 指南](53_OOMRL_CRITIC_GUIDE.md) | Signal 提取, Critic 模式, ActionItem, Skills |
| [54 — 反馈闭环执行指南](54_OOMRL_FEEDBACK_LOOP.md) | Playground, Adapter, Orchestrator, Promote |

## 快速上手

```bash
# 1. 运行 Deep Analysis + Critic（产出 ActionItems）
python -m quantpits.scripts.run_deep_analysis --critic

# 2. 预览 Feedback Loop 将要做什么
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --report-only

# 3. 在 Playground 中执行（挑 3 个高优先级模型）
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --execute --max-duration-minutes 30

# 4. 验证通过后推广到生产
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --promote

# 5. 推广的配置在下一次训练周期自动生效
python -m quantpits.scripts.static_train --all-enabled
```
