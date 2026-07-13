# Deep Analysis System (MAS) — Usage Guide

## Overview

The Deep Analysis System is a Multi-Agent System (MAS) that performs automated, multi-window post-trade analysis. Following recent updates, the system utilizes a strict **7-Stage Pipeline** architecture and supports loading custom workspace-local plugins via the **Pluggable Agent Registry**.
Multiple specialist agents analyze different aspects of the trading system, and a Synthesizer cross-references the findings to generate prioritized, actionable recommendations.

Starting from Phase 3, the system integrates OOM-RL (Out-of-Money Reinforcement Learning) feedback capabilities: an LLM Critic converts analysis findings into executable ActionItems, and the Phase 4 Feedback Loop automatically executes and validates these recommendations in a sandboxed Playground workspace.

## Core Architecture

### 1. 7-Stage Pipeline

Deep Analysis execution follows a strict 7-stage pipeline. Each stage self-declares its dependencies and outputs via the `@register_stage` decorator; the system builds an execution DAG dynamically. Use `--stage` to run any stage independently: compatible upstream checkpoints are reused, while the target stage itself is re-executed.

1. **`discover`**: Scans the workspace to discover and load all relevant snapshots, model predictions, and historical configuration data.
2. **`agents`**: Instantiates and executes registered specialist analysis Agents, producing structured `AgentFindings`.
3. **`synthesis`**: Passes all agent findings to the Synthesizer to generate cross-domain insights and summaries.
4. **`window_analysis`**: Runs the rule-driven `TrainingWindowAnalyzer` to assess training window configurations. **Always enabled** — static and CPCV/rolling rules run unconditionally; data-driven rules auto-activate when benchmark data is available.
5. **`signals`**: Runs `SignalExtractor` to convert raw agent metrics into standardized signals for Critic consumption.
6. **`critic`**: Executes the LLM Critic to transform signals into actionable optimization advice (`ActionItems`).
7. **`report`**: Synthesizes output from all stages and renders the final Markdown report.

**Checkpoints & Label Isolation**: Each stage auto-saves a checkpoint to `output/deep_analysis/checkpoints/`. The `--run-label` flag injects the label into checkpoint filenames — different labels are fully isolated, allowing multiple same-day experiments without interference. Checkpoint metadata records windows, agent selector, manifest path, and manifest content fingerprint. If a checkpoint is incompatible with the current request, it is skipped and the relevant upstream stage is re-run. `--resume-latest` continues from the newest checkpoint within the same label.

**Execution Plan Explanation**: Use `--explain-plan` to parse the DAG, workspace stage manifest, upstream checkpoint compatibility, and final execution plan without running any stages or writing checkpoints. It is useful before single-stage validation, plugin debugging, or any run where checkpoint reuse semantics need to be confirmed.

### 2. Pluggable Agent & Stage Registry

The system allows loading custom agents and pipeline stages via workspace-local manifests without polluting the global codebase.

- **Agent plugins**: Declare in `config/agent_manifest.json`. The system injects the workspace path into `sys.path`, imports the agent class, uses it only in the current run's local registry, and cleans up.
- **Stage plugins**: Declare in `config/pipeline_manifest.json`. Same isolation mechanism — temporary workspace-root `sys.path` injection and stage registry snapshot restoration after the run. Custom stages declare their DAG position via `insert_after`.

For detailed development steps, refer to [57 — Agent Plugin Development Guide](57_AGENT_PLUGIN_GUIDE.md).

## Quick Start

```bash
# Activate workspace
source workspaces/Demo_Workspace/run_env.sh

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

# Run with label (checkpoints fully isolated between different labels)
python -m quantpits.scripts.run_deep_analysis --critic --run-label after-retrain

# Run a single stage only (upstream auto-loaded from compatible checkpoints; target re-runs)
python -m quantpits.scripts.run_deep_analysis --stage signals --run-label exp-1

# Explain a single-stage plan without executing it
python -m quantpits.scripts.run_deep_analysis --stage agents:model_health --windows 1m --explain-plan

# Resume from latest checkpoint within the same label
python -m quantpits.scripts.run_deep_analysis --resume-latest --run-label exp-1

# Custom time windows
python -m quantpits.scripts.run_deep_analysis --windows 1y,3m,1m

# Load workspace-local custom agent plugin
python -m quantpits.scripts.run_deep_analysis --agents custom_mock_agent --agent-manifest config/agent_manifest.json

# Load workspace-local custom stage plugin
python -m quantpits.scripts.run_deep_analysis --stage custom_liquidity_check --stage-manifest config/pipeline_manifest.json
```

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stage` | `all` | Run a specific stage (`discover`, `agents`, `agents:NAME`, `synthesis`, `window_analysis`, `signals`, `critic`, `report`, `all`). Upstream stages auto-load from compatible same-label checkpoints; the target stage re-runs |
| `--run-label` | `""` | Run label (e.g., "after-retrain"). Injected into checkpoint and report filenames — different labels are fully isolated |
| `--resume-latest` | (flag) | Auto-find the latest checkpoint for the current label and resume from the next stage |
| `--resume-from` | `None` | Resume from a specific checkpoint file |
| `--manifest` | `None` | Deprecated compatibility alias for an agent manifest; prefer `--agent-manifest` |
| `--agent-manifest` | `None` | Agent manifest path (JSON/YAML), relative to the current workspace root; explicit missing paths fail fast |
| `--stage-manifest` | `None` | Stage manifest path (JSON/YAML), relative to the current workspace root; used to load custom pipeline stages |
| `--explain-plan` | (flag) | Print the DAG, checkpoint compatibility, and stages that would run; does not execute stages or write checkpoints |
| `--windows` | `full,weekly_era,1y,6m,3m,1m` | Comma-separated time windows |
| `--freq-change-date` | From config or `None` | Daily→weekly frequency cutoff date |
| `--output` | `output/deep_analysis_report.md` | Report output path |
| `--llm` | (flag) | Enable LLM executive summary (reads model/endpoint from llm_config.json) |
| `--llm-model` | (llm_config.json) | Override LLM model for summary |
| `--api-key` | (env var) | API key override (reads env var from llm_config.json api_key_env) |
| `--base-url` | (llm_config.json) | API base URL override |
| `--critic` | (flag) | **OOM-RL Phase 3** — Enable Critic mode, generate ActionItems |
| `--critic-dry-run` | (flag) | Critic preview mode, generate ActionItems without persisting |
| `--window-analysis` | (flag, retained) | Always runs by default; flag kept for backward compatibility |
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
signal extraction. Always enabled — static and CPCV/rolling rules run unconditionally;
data-driven rules auto-activate when benchmark data is available.

It detects structural problems in the training data split configuration without
any LLM dependency:

- **Input**: `model_config.json` (window parameters), `training_history.jsonl`
  (anchor history), regime switch data from the Market Regime agent
- **Output**: List of `WindowAnalysisFinding` objects with severity
  (critical/warning/info), metrics, and actionable recommendations

**Sixteen detection rules** in three groups:

**Static rules (R1-R6)** — always run:
1. **Window size bounds**: `train_set_windows` < 4 years → critical/warning, > 15 years → info
2. **Validation ratio**: `valid / train` < 0.15 → early stopping unreliable
3. **Train-end gap**: In slide mode, gap ≥ 5 years + ≥ 20 regime switches → critical
4. **Anchor staleness**: latest anchor > 90 days ago → warning, > 60 days → info
5. **Regime vs. window mismatch**: high-vol regime needs ≥ 10 years of training;
   ≥ 3 regime switches needs ≥ 8 years of training
6. **Frequency compatibility**: daily frequency with > 365 windows → data overload

**Data-driven rules (R7-R13)** — require `BenchmarkDataLoader` market benchmark data:
7. **Regime coverage**: training coverage < 40% observed regimes → warning;
   missing Bearish-HighVol → separate warning
8. **Volatility regime shift**: train vs test vol ratio > 1.5x or < 0.5x → warning/info
9. **Return distribution shift**: KS statistic > 0.20 → warning; mean shift > 1.0σ → info
10. **Drawdown coverage**: training lacks major drawdowns (>15%) present in full history → warning
11. **Boundary regime mismatch**: regime change at train→valid or valid→test boundary → warning
12. **Cliff edge**: regime will drop from training within 4 weeks → warning/info
13. **Coverage stability**: regime coverage stability < 0.90 across sliding windows → info

**CPCV/rolling rules (R14-R16)** — auto-run via `TrainingModeContext`:
14. **CPCV groups insufficient**: `n_groups` ≤ `n_test + n_val` → critical
15. **CPCV leak threat**: `purge_steps` > 10 or `embargo_steps` > 20 → warning
16. **Rolling staleness**: rolling state > 90 days without update → warning

Findings flow into the LLM Critic as `training_window_mismatch` signals and
appear in all layered pipeline prompts as the `training_window_analysis` field.

### 9. Training Health (`training_health`)

Evaluates training pipeline health, rolling progress, and trade execution trends:

- **Input**: `training_history.jsonl`, `rolling_metrics_20.csv`, `rolling_metrics_60.csv`,
  `latest_train_records.json`, plus workspace metadata provided by `TrainingContext`.
- **Outputs**:
  - **Mode coverage audit**: Checks each model's training mode coverage (static/cpcv/rolling/cpcv_rolling), flags models missing expected modes
  - **Rolling pipeline staleness**: Inspects rolling window progress, marks pipelines > 90 days stale
  - **Alpha decay monitoring**: Compares short/long-term idiosyncratic Alpha to detect stock-selection decay
  - **Execution friction**: Monitors z-score anomalies in slippage (`Exec_Slippage_Mean`) and delay cost (`Delay_Cost_Mean`)
  - **Factor drift detection**: Detects extreme percentile drift in Barra Liquidity Exposure (micro-cap / large-cap)
  - **Orphan model detection**: Identifies enabled models not belonging to any active combo

> **Training Context (`TrainingContext`)**: This agent relies on `training_context.py` for training mode inventory (name→mode parsing), rolling pipeline gap calculation, and model key resolution.

> **Rolling metrics CSV prerequisite**: Alpha decay, execution friction, and factor drift detection depend on `output/rolling_metrics_20.csv` and `rolling_metrics_60.csv`. These files are generated by the rolling analysis pipeline (`run_rolling_analysis.py`). **If files are missing, the agent explicitly reports this data gap** (at `info`/`warning` severity) rather than silently skipping. If rolling training is not configured, this gap is expected.

### 10. Training Mode Awareness

As of Phase 2b, the Deep Analysis system has full training mode awareness. `TrainingModeContext` (at `quantpits/scripts/deep_analysis/training_context.py`) is the single source of truth for training mode information, accessed by all agents via `AnalysisContext.training_context`.

#### TrainingModeContext Data Sources

| Source | Provides |
|--------|----------|
| `latest_train_records.json` | Model→mode mapping; V2 `model_records` adds per-model experiment, operation, prediction coverage, and source lineage |
| `training_history.jsonl` | Per-model last train date, mode, convergence status (Phase 2b) |
| `prediction_history.jsonl` | Per-model last predict-only event (Phase 2b) |
| `data/rolling_training_history.jsonl` | Rolling training events (slide + CPCV), separate from static (Phase 2c) |
| `data/rolling_prediction_history.jsonl` | Rolling predict-only events (slide + CPCV) (Phase 2c) |
| `config/rolling_config.yaml` | Rolling scheduler configuration |
| `data/rolling_state.json` | Slide rolling progress |
| `data/rolling_state_cpcv.json` | CPCV rolling progress |
| `config/model_config.json` | CPCV parameters (purged_cv) |

> **Phase 2c file separation**: Rolling training has independent code paths (calls `model.fit()` directly, bypassing `train_utils` wrappers) and produces "thin" log entries (no epoch-level data). Rolling events are therefore written to dedicated files, fully isolated from static training logs. Users not using rolling training never create these files — zero overhead.

#### `@suffix` Model Key Convention

Training records use `modelname@trainingmode` as compound keys:

| Suffix | Mode | Example |
|--------|------|---------|
| `@static` | Standard one-shot training | `lstm_Alpha158@static` |
| `@cpcv` | Purged K-Fold cross-validation | `linear_Alpha158@cpcv` |
| `@rolling` | Sliding window rolling training | `gru_Alpha360@rolling` |
| `@cpcv_rolling` | CPCV-strategy rolling windows | `lstm_Alpha158@cpcv_rolling` |

`TrainingModeContext.models_by_name` aggregates all modes per model: `{"lstm_Alpha158": {"static": "rid1", "rolling": "rid2"}}`.

#### Predict-Only Cycle Detection

The system auto-detects when the current cycle is predict-only (>50% of models have prediction as their latest operation). When `is_predict_only_cycle=True`, agent behavior is calibrated:

| Agent | Predict-Only Behavior |
|-------|----------------------|
| **ModelHealth** | Suppresses "stale" false positives (model wasn't retrained this cycle, not genuinely stale); labels `cycle_type: predict_only` |
| **TrainingHealth** | Suppresses "missing expected modes" (modes weren't run this cycle, not never-trained) |
| **PredictionAudit** | Adds cycle context — hit rates reflect model stability, not new training quality |
| **EnsembleEval** | Distinguishes model retirement from training mode migration (e.g., `static→rolling`) |

#### Using TrainingModeContext in Plugins

Custom agents access all training mode information via `ctx.training_context`:

```python
def analyze(self, ctx: AnalysisContext) -> AgentFindings:
    tc = ctx.training_context
    if tc:
        # Query the latest operation for a model
        last_op = tc.get_last_operation("lstm_Alpha158")
        # → {"type": "train", "date": "2026-07-03T...", "mode": "static", ...}
        
        # Check for predict-only cycle
        if tc.is_predict_only_cycle:
            ...
        
        # Get models trained in multiple modes
        cross_mode = tc.get_cross_mode_models()  # → ["lstm_Alpha158", ...]
        
        # Filter models by training mode
        rolling_models = tc.get_models_with_mode("rolling")
```

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
| [57 — Agent Plugin Guide](57_AGENT_PLUGIN_GUIDE.md) | Agent Manifest structure, development, and execution of workspace-local plugins |

### Quick Flow

```bash
# Step 1: Deep Analysis + Critic → produce ActionItems (window_analysis runs by default)
python -m quantpits.scripts.run_deep_analysis --critic

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

## Data File Format Reference

### `training_history.jsonl`

One JSON object per line, recording convergence info for each model training event. Written by `train_single_model()` and `train_cpcv_model()`.

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Model name (without @mode suffix) |
| `mode` | string | Training mode: `static`/`rolling`/`cpcv`/`cpcv_rolling` (Phase 2b+, old entries default to `static`) |
| `experiment_name` | string | MLflow experiment name |
| `record_id` | string | MLflow recorder UUID |
| `anchor_date` | string | Training anchor date (YYYY-MM-DD) |
| `trained_at` | string | ISO 8601 training completion timestamp |
| `duration_seconds` | float\|null | Wall-clock training time |
| `early_stopped` | bool | Whether early stopping triggered |
| `actual_epochs` | int\|null | Actual epochs run (num_boost_round for GBDT) |
| `configured_epochs` | int\|null | Configured epoch count |
| `best_epoch` | int\|null | Best validation score epoch |
| `best_score` | float\|null | Best validation score |
| `converged` | bool\|null | actual_epochs == configured_epochs |
| `score_type` | string | Metric type: `ic`/`rank_ic`/`loss`/`cpcv_folds` |
| `IC_Mean` | float\|null | Test-set IC mean |
| `ICIR` | float\|null | IC information ratio |
| `n_folds` | int\|null | CPCV fold count (only when `score_type=cpcv_folds`) |
| `fold_ic_mean` | float\|null | CPCV fold IC mean (CPCV only) |

### `prediction_history.jsonl`

One JSON object per line, recording each predict-only event. Written by `predict_single_model()` and `predict_cpcv_model()`.

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Model name |
| `mode` | string | Source model's training mode |
| `anchor_date` | string | Prediction anchor date |
| `predicted_at` | string | ISO 8601 prediction timestamp |
| `experiment_name` | string | MLflow experiment name |
| `record_id` | string | New recorder UUID |
| `source_record_id` | string | Source training recorder UUID |
| `IC_Mean` | float\|null | Prediction-period IC mean |
| `ICIR` | float\|null | Prediction-period ICIR |
| `prediction_type` | string\|null | `cpcv_ensemble` (CPCV predict only) |

### `rolling_training_history.jsonl` (Phase 2c)

One JSON object per line, recording slide or CPCV rolling training events. Written by `strategy_slide.py::train_window()` and `strategy_cpcv.py::train_window()`. Separated from `training_history.jsonl` to avoid log bloat from bulk rolling entries.

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Model name (without @mode suffix) |
| `mode` | string | `rolling` (slide mode) or `cpcv_rolling` (CPCV mode) |
| `experiment_name` | string | MLflow experiment name |
| `record_id` | string | MLflow recorder UUID |
| `anchor_date` | string | Training anchor date (YYYY-MM-DD) |
| `window_idx` | int | Sliding window index |
| `train_start` | string | Training segment start date |
| `train_end` | string | Training segment end date |
| `valid_start` | string\|null | Validation segment start (slide only) |
| `valid_end` | string\|null | Validation segment end (slide only) |
| `test_start` | string | Test segment start date |
| `test_end` | string | Test segment end date |
| `trained_at` | string | ISO 8601 training completion timestamp |
| `duration_seconds` | float | Training duration in seconds |
| `IC_Mean` | float\|null | Test-set IC mean |
| `ICIR` | float\|null | IC information ratio |
| `score_type` | string | `rolling_slide` or `cpcv_rolling_folds` |
| `n_folds` | int\|null | CPCV fold count (only `cpcv_rolling`) |
| `fold_ic_mean` | float\|null | Fold validation IC mean (only `cpcv_rolling`) |
| `Ann_Excess` | float\|null | Annualized excess return (CPCV only) |
| `Max_DD` | float\|null | Maximum drawdown (CPCV only) |
| `Information_Ratio` | float\|null | Information ratio (CPCV only) |

> **Note**: The following fields are intentionally absent (unavailable from `model.fit()`): `early_stopped`, `actual_epochs`, `converged`, `epoch_*` arrays. Setting them to `null` would be misleading; absence is truthful.

### `rolling_prediction_history.jsonl` (Phase 2c)

One JSON object per line, recording rolling predict-only events. Written by `strategy_slide.py::predict_latest()` and `strategy_cpcv.py::predict_latest()`.

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Model name |
| `mode` | string | `rolling` or `cpcv_rolling` |
| `anchor_date` | string | Prediction anchor date |
| `predicted_at` | string | ISO 8601 prediction timestamp |
| `experiment_name` | string | MLflow experiment name |
| `record_id` | null | Always `null` (`predict_latest()` does not create a new MLflow run) |
| `source_record_id` | string | Source training window recorder UUID |
| `window_idx` | int | Sliding window index used for prediction |
| `prediction_type` | string | `rolling_gap_predict` (slide) or `cpcv_rolling_ensemble` (CPCV) |

### `latest_train_records.json`

Workspace root file recording the latest training/prediction cycle's model registry.

```json
{
    "experiment_name": "<your_experiment_name>",
    "static_experiment_name": "<static_experiment_name>",
    "rolling_experiment_name": "",
    "cpcv_experiment_name": "<cpcv_experiment_name>",
    "anchor_date": "YYYY-MM-DD",
    "models": {
        "lstm_Alpha158@static": "<record-uuid>",
        "lstm_Alpha158@rolling": "<record-uuid>",
        "linear_Alpha158@cpcv": "<record-uuid>"
    }
}
```

`models` keys use the `modelname@trainingmode` compound key convention. `TrainingModeContext` parses the `@` separator to build the `models_by_name` mapping for cross-mode queries.

In Training Record V2, `model_records` is authoritative for current model identity; top-level
anchor and experiment fields are compatibility views. Freshness or lineage analysis should consume
the per-model fields and still verify actual recorder/history evidence rather than assuming one
global anchor applies to every model.
