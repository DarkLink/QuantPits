# Deep Analysis System (MAS) — Usage Guide

## Overview

The Deep Analysis System is a Multi-Agent System (MAS) that performs automated, 
multi-window post-trade analysis. Seven specialist agents analyze different aspects 
of the trading system and a Synthesizer cross-references findings to produce 
prioritized, actionable recommendations.

## Quick Start

```bash
# Activate workspace
source workspaces/CSI300_Base/run_env.sh

# Basic rule-based analysis
python -m quantpits.scripts.run_deep_analysis

# With frequency-change cutoff (recommended for CSI300_Base)
python -m quantpits.scripts.run_deep_analysis --freq-change-date 2024-10-21

# With LLM-powered executive summary
OPENAI_API_KEY=sk-xxx python -m quantpits.scripts.run_deep_analysis \
    --llm openai --freq-change-date 2024-10-21

# With operator notes
python -m quantpits.scripts.run_deep_analysis \
    --notes "Retrained catboost last week. Market volatile due to trade tensions."

# Specific agents only
python -m quantpits.scripts.run_deep_analysis --agents model_health,prediction_audit

# Custom time windows
python -m quantpits.scripts.run_deep_analysis --windows 1y,3m,1m
```

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--windows` | `full,weekly_era,1y,6m,3m,1m` | Comma-separated time windows |
| `--freq-change-date` | From config or `None` | Daily→weekly frequency cutoff date |
| `--output` | `output/deep_analysis_report.md` | Report output path |
| `--llm` | `none` | LLM backend: `none` or `openai` |
| `--llm-model` | `gpt-4` | OpenAI model name |
| `--api-key` | `$OPENAI_API_KEY` | API key for LLM |
| `--base-url` | `None` | OpenAI-compatible API base URL |
| `--agents` | `all` | Comma-separated agent names |
| `--notes` | `""` | Free-text external context |
| `--notes-file` | `None` | File path containing external notes |
| `--shareable` | `false` | Redact sensitive data |
| `--no-snapshot` | `false` | Skip config snapshot |

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

## Data Discovery

The system scans both **active workspace** and **archive** directories:
- `output/` + `output/{ensemble,predictions,ranking}/`
- `archive/output/` + `archive/output/{ensemble,predictions,ranking}/`
- `data/` + `data/order_history/`

This ensures analysis uses all available historical data regardless of archival status.

## Frequency-Change Cutoff

When `--freq-change-date` is set (e.g., `2024-10-21` for CSI300_Base's daily→weekly switch):
- A special `weekly_era` window is auto-generated from that date onward
- Shorter windows (1y, 6m, etc.) work normally
- The `full` window still covers all data but flags pre-cutoff data

Configuration can be persisted in `config/deep_analysis_config.json`:
```json
{
    "freq_change_date": "2024-10-21"
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

This enables future diff analysis for hyperparameter tuning impact assessment.

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
