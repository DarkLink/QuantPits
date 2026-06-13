# Deep Analysis System (MAS) — Usage Guide

## Overview

The Deep Analysis System is a Multi-Agent System (MAS) that performs automated, 
multi-window post-trade analysis. Seven specialist agents analyze different aspects 
of the trading system and a Synthesizer cross-references findings to produce 
prioritized, actionable recommendations.

Starting from Phase 3, the system integrates OOM-RL (Out-of-Money Reinforcement
Learning) feedback capabilities: an LLM Critic converts analysis findings into
executable ActionItems, and the Phase 4 Feedback Loop automatically executes and
validates these recommendations in a sandboxed Playground workspace.

## Quick Start

```bash
# Activate workspace
source workspaces/Example_Workspace/run_env.sh

# Basic rule-based analysis
python -m quantpits.scripts.run_deep_analysis

# LLM-powered executive summary
python -m quantpits.scripts.run_deep_analysis --llm

# Critic mode — generate executable ActionItems (OOM-RL Phase 3)
python -m quantpits.scripts.run_deep_analysis --critic

# Critic dry-run — generate ActionItems without persisting to files
python -m quantpits.scripts.run_deep_analysis --critic-dry-run

# With frequency-change cutoff
python -m quantpits.scripts.run_deep_analysis --freq-change-date YYYY-MM-DD

# With operator notes
python -m quantpits.scripts.run_deep_analysis \
    --notes "Retrained catboost last week. Market volatile due to trade tensions."

# Specific agents only
python -m quantpits.scripts.run_deep_analysis --agents model_health,prediction_audit

# Run with label (prevents overwriting when running multiple times on same date)
python -m quantpits.scripts.run_deep_analysis --critic --run-label after-retrain

# Custom time windows
python -m quantpits.scripts.run_deep_analysis --windows 1y,3m,1m

# Enable training window analysis (rule-based, independent of Critic)
python -m quantpits.scripts.run_deep_analysis --critic --window-analysis
```

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--windows` | `full,weekly_era,1y,6m,3m,1m` | Comma-separated time windows |
| `--freq-change-date` | From config or `None` | Daily→weekly frequency cutoff date |
| `--output` | `output/deep_analysis_report.md` | Report output path |
| `--llm` | (flag) | Enable LLM executive summary (reads model/endpoint from llm_config.json) |
| `--llm-model` | (llm_config.json) | Override LLM model for summary |
| `--api-key` | (env var) | API key override (reads env var from llm_config.json api_key_env) |
| `--base-url` | (llm_config.json) | API base URL override |
| `--critic` | (flag) | **OOM-RL Phase 3** — Enable Critic mode, generate ActionItems |
| `--critic-dry-run` | (flag) | Critic preview mode, generate ActionItems without persisting |
| `--window-analysis` | (flag) | Enable rule-based training window analysis (TrainingWindowAnalyzer), runs independently of Critic |
| `--run-label` | `""` | Run label (e.g., "after-retrain"), injected into output filenames to prevent same-date overwrites |
| `--agents` | `all` | Comma-separated agent names |
| `--notes` | `""` | Free-text external context |
| `--notes-file` | `None` | File path containing external notes |
| `--shareable` | `false` | Redact sensitive data |
| `--no-snapshot` | `false` | Skip config snapshot |

> **OOM-RL workflow**: after `--critic` produces ActionItems, execute them via
> `run_feedback_loop.py`. See [54 — Feedback Loop Guide](54_OOMRL_FEEDBACK_LOOP.md).

## Agents

### 1. Market Regime (`market_regime`)
Detects current market trend, volatility regime, and drawdown state from CSI300 benchmark data.

- **Input**: `daily_amount_log_full.csv` (CSI300 column)
- **Outputs**: Trend label (Bull/Bear/Sideways), volatility percentile, drawdown depth

### 2. Model Health (`model_health`)
Evaluates individual model IC/ICIR trends, detects retrains, and snapshots hyperparameters.

- **Input**: `model_performance_*.json`, `workflow_config_*.yaml`, MLflow tags
- **Outputs**: IC/ICIR scorecard table, retrain timeline, hyperparameter summary, staleness warnings

### 3. Ensemble Evolution (`ensemble_eval`)
Tracks ensemble combo performance and detects three layers of composition changes.

- **Input**: `combo_comparison_*.csv`, `leaderboard_*.csv`, `ensemble_fusion_config_*.json`
- **Outputs**: Combo performance trend, 3-layer change event log, correlation drift

**Three layers of change detection:**
1. **Composition change**: Model list diff in `ensemble_fusion_config` files
2. **Active combo switch**: Default combo changed in `ensemble_config.json`
3. **Content mutation**: Same combo name, different models inside config

### 4. Execution Quality (`execution_quality`)
Analyzes trading execution friction using the existing `ExecutionAnalyzer`.

- **Input**: `trade_log_full.csv`
- **Outputs**: Friction trends, substitution bias, fee efficiency, ADV capacity

> **Note**: Execution timing analysis is deferred (TODO) pending granular intraday timestamp data.

### 5. Portfolio Risk (`portfolio_risk`)
Multi-window risk analysis with OLS statistical significance using the existing `PortfolioAnalyzer`.

- **Input**: `daily_amount_log_full.csv`, `trade_log_full.csv`, `holding_log_full.csv`
- **Outputs**: Multi-window CAGR/Sharpe/DD table, OLS alpha/beta t-stat/p-value, factor drift

### 6. Prediction Audit (`prediction_audit`)
Compares model predictions vs actual market outcomes.

- **Input**: `buy_suggestion_*.csv`, `sell_suggestion_*.csv`, `model_opinions_*.json`, Qlib forward returns
- **Outputs**: Buy/sell hit rates, consensus vs divergence analysis, holding retrospective

### 7. Trade Pattern (`trade_pattern`)
Analyzes trading behavior patterns and signal discipline.

- **Input**: `trade_classification.csv`, `trade_log_full.csv`, `holding_log_full.csv`
- **Outputs**: Signal/Substitute/Manual ratio, concentration trends, discipline score

### 8. Training Window Analyzer (`TrainingWindowAnalyzer`)

A **purely rule-driven**, independent analyzer that runs after the agents but before
signal extraction. Enabled via the `--window-analysis` flag.

It detects structural problems in the training data split configuration without
any LLM dependency:

- **Input**: `model_config.json` (window parameters), `training_history.jsonl`
  (anchor history), regime switch data from the Market Regime agent
- **Output**: List of `WindowAnalysisFinding` objects with severity
  (critical/warning/info), metrics, and actionable recommendations

**Six detection rules**:
1. **Window size bounds**: `train_set_windows` < 4 years → critical, > 15 years → info
2. **Validation ratio**: `valid / train` < 0.15 → early stopping unreliable
3. **Train-end gap**: In slide mode, `train_end = anchor - (valid + test)` years.
   When gap ≥ 5 years with ≥ 20 regime switches → critical
4. **Anchor staleness**: latest anchor > 90 days ago → warning, > 60 days → info
5. **Regime vs. window mismatch**: high-vol regime needs ≥ 10 years of training;
   ≥ 3 regime switches needs ≥ 8 years of training
6. **Frequency compatibility**: daily frequency with large window count → data overload

Findings flow into the LLM Critic as `training_window_mismatch` signals and
appear in all layered pipeline prompts as the `training_window_analysis` field.

## Data Discovery

The system scans both **active workspace** and **archive** directories:
- `output/` + `output/{ensemble,predictions,ranking}/`
- `archive/output/` + `archive/output/{ensemble,predictions,ranking}/`
- `data/` + `data/order_history/`

This ensures analysis uses all available historical data regardless of archival status.

## Frequency-Change Cutoff

When `--freq-change-date` is set (e.g., `YYYY-MM-DD` for a daily→weekly switch):
- A special `weekly_era` window is auto-generated from that date onward
- Shorter windows (1y, 6m, etc.) work normally
- The `full` window still covers all data but flags pre-cutoff data

Configuration can be persisted in `config/deep_analysis_config.json`:
```json
{
    "freq_change_date": "YYYY-MM-DD"
}
```

## External Notes

Inject operator context via `--notes` or `--notes-file`. This context is:
- Passed to all agents as part of `AnalysisContext`
- Included in LLM synthesis prompts
- Appended to the report as an appendix

Examples:
```bash
--notes "Retrained all Alpha360 models last week."
--notes "Market volatile due to tariff announcements. Reduced position sizes."
--notes-file operator_notes.txt
```

## Config Ledger

Each run automatically snapshots current configurations to `data/config_history/`:
- `config_snapshot_{date}.json` containing:
  - All `workflow_config_*.yaml` hyperparameters
  - `ensemble_config.json` (tracks active combo + composition)
  - `strategy_config.yaml`
  - `model_config.json` (training window parameters: train/valid/test sizes, slice mode, freq)

This enables future diff analysis for both hyperparameter tuning and training
window change impact assessment.

## Report Structure

```
# Deep Analysis Report — {date}
## Executive Summary          ← LLM-generated or template-based
## 1. Market Environment
## 2. Model Health Dashboard
    ### 2.1 IC/ICIR Scorecard
    ### 2.2 Retrain History
    ### 2.3 Hyperparameter Configuration
## 3. Ensemble Evolution
    ### 3.1 Combo Performance
    ### 3.2 Change Event Log
## 4. Execution Quality
## 5. Portfolio Risk & Attribution
    ### 5.1 Multi-Window Comparison
    ### 5.2 OLS Significance
## 6. Prediction Accuracy Audit
## 7. Trade Behavior
## 8. Holistic Change Impact Assessment
## 9. Prioritized Recommendations (P0/P1/P2)
## Appendix: External Notes
```

## Cross-Agent Synthesis Rules

The Synthesizer detects compound patterns across agents:

| Rule | Trigger | Assessment |
|------|---------|------------|
| Regime-driven IC decay | Model IC declining + High-vol market | "IC degradation may be regime-driven" |
| Liquidity drift | Negative alpha + Negative liquidity exposure | "Small-cap drift without selection edge" |
| Untradeable convictions | High substitution bias + Low hit rate | "Top picks frequently untradeable" |
| Ensemble value | Hit rate > 55% | "Fusion value confirmed" |
| No alpha | Alpha p>0.1 across all windows | "Cannot reject H₀ of zero alpha" |

## OOM-RL Closed-Loop Feedback

Starting from Phase 3, Deep Analysis integrates OOM-RL feedback capabilities.
When `--critic` is enabled, analysis findings flow through a pipeline that
converts them into executable model optimization actions:

```
Agent Findings → Signal Extractor → LLM Critic → ActionItems → Feedback Loop
```

### Related Documentation

| Document | Content |
|----------|---------|
| [51 — OOM-RL Overview](51_OOMRL_FEEDBACK_OVERVIEW.md) | System architecture, data flow, feedback scope control |
| [52 — Data Infrastructure](52_OOMRL_DATA_INFRASTRUCTURE.md) | OperatorLog, Config Ledger, training history, agent enhancements |
| [53 — LLM Critic Guide](53_OOMRL_CRITIC_GUIDE.md) | Signal extraction, Critic mode, ActionItem structure, Skills |
| [54 — Feedback Loop Guide](54_OOMRL_FEEDBACK_LOOP.md) | Playground, Adapter, Orchestrator, Promote, rollback |
| [55 — OOM-RL Weekly Operations Guide](55_OOMRL_WEEKLY_OPERATIONS.md) | Daily operations, intervention checks, edge cases |
| [56 — LLM Observability & Tracing](56_LLM_OBSERVABILITY_GUIDE.md) | LLM Traces, Reasoning capture, Langfuse, multi-model prep |

### Quick Flow

```bash
# Step 1: Deep Analysis + Critic + Window Analysis → produce ActionItems
python -m quantpits.scripts.run_deep_analysis --critic --window-analysis

# Step 2: Preview the Feedback Loop
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --report-only

# Step 3: Execute in Playground
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --execute

# Step 4: Promote to production
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --promote
```
