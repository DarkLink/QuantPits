# OOM-RL Closed-Loop Feedback System — Overview

## Concept

OOM-RL (Out-of-Money Reinforcement Learning) is an automated feedback system
that converts Deep Analysis findings into executable model optimization actions,
validates them in a sandboxed Playground workspace, and promotes proven changes
to production.

```
Weekly                              On-demand (Phase 4)
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

## The Four Phases

| Phase | Deliverables | Description |
|-------|-------------|-------------|
| **Phase 1** — Data Infrastructure | `operator_log.py`, `config_ledger.py`, training convergence logs | Operation audit trail, config snapshots, automatic training metrics collection |
| **Phase 2** — Agent Signal Enhancement | Model Health / Ensemble Eval / Market Regime / Prediction Audit upgrades | Convergence detection, OOS history comparison, LOO contributions, regime-switch detection |
| **Phase 3** — LLM Critic | `signal_extractor.py`, `llm_interface.py` (critic), `action_items.py` | Rule-based signal extraction → LLM decision → structured ActionItems |
| **Phase 4** — Execution Layer | Playground, Adapters, Orchestrator, Promoter | Automated ActionItem execution, validation, and promotion |

## Data Flow

```
Training Scripts                Deep Analysis                   Feedback Loop
────────────────                ─────────────                   ─────────────
train_single_model()            run_deep_analysis.py            run_feedback_loop.py
  │                               │                               │
  ├─ training_history.jsonl      ├─ config_snapshot_{date}.json  ├─ Playground fork
  ├─ model_performance_{date}.json  ├─ agent findings              ├─ TrainingAdapter.apply()
  ├─ latest_train_records.json   ├─ signal_extractor.py          ├─ train_single_model()
  └─ operator_log.jsonl          ├─ llm_interface.py (critic)    ├─ single-model IC check
                                   ├─ action_items_{date}.json    ├─ promote_config.py
                                   └─ action_item_history.jsonl   └─ promote_history.jsonl
```

## Feedback Scope Control

The system controls which domains the LLM Critic can influence via
`config/feedback_scope.json`:

| Scope | Description | Default | Adapter |
|-------|-------------|---------|---------|
| `hyperparams` | Model hyperparameter tuning (n_epochs, lr, dropout ...) | **enabled** | TrainingAdapter |
| `model_selection` | Model enable/disable | disabled | TrainingAdapter |
| `combo_search` | Trigger ensemble search and re-composition | disabled | (future) |
| `strategy_params` | TopK/DropN/liquidity constraint adjustments | disabled | (future) |

ActionItems outside `active_scopes` are marked `out_of_scope` and appear in
reports but do not execute.

Typical adoption rhythm:
1. **Stage 1**: `["hyperparams"]` — tune model hyperparameters first
2. **Stage 2**: `["hyperparams", "model_selection"]` — optimize which models are active
3. **Stage 3**: `["combo_search"]` — optimize ensemble composition
4. **Stage 4**: Full scope

## Document Index

| Document | Content |
|----------|---------|
| [50 — Deep Analysis Guide](50_DEEP_ANALYSIS_GUIDE.md) | Base MAS system usage |
| [51 — This document](51_OOMRL_FEEDBACK_OVERVIEW.md) | OOM-RL closed-loop feedback overview |
| [52 — Data Infrastructure](52_OOMRL_DATA_INFRASTRUCTURE.md) | OperatorLog, Config Ledger, training convergence logs |
| [53 — LLM Critic Guide](53_OOMRL_CRITIC_GUIDE.md) | Signal extraction, Critic mode, ActionItems, Skills |
| [54 — Feedback Loop Guide](54_OOMRL_FEEDBACK_LOOP.md) | Playground, Adapter, Orchestrator, Promote |

## Quick Start

```bash
# 1. Run Deep Analysis + Critic (produces ActionItems)
python -m quantpits.scripts.run_deep_analysis --critic

# 2. Preview what the Feedback Loop will do
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --report-only

# 3. Execute in Playground (pick 3 high-priority models)
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --execute --max-duration-minutes 30

# 4. Promote validated changes to production
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --promote

# 5. Promoted config takes effect on the next training cycle
python -m quantpits.scripts.static_train --all-enabled
```
