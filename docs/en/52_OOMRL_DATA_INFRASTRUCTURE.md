# OOM-RL Data Infrastructure — Phases 1 & 2

Documents the structured data automatically collected by the OOM-RL system and
the agent signal enhancements delivered in Phase 2.

---

## 1. OperatorLog — Operation Audit Trail

**File**: `quantpits/utils/operator_log.py`

Each core script run automatically appends a JSONL record to
`data/operator_log.jsonl`. Integrated into 7 scripts: `static_train.py`,
`rolling_train.py`, `ensemble_fusion.py`, `pretrain.py`,
`minentropy_ensemble.py`, `brute_force_ensemble.py`, `brute_force_fast.py`.

### Record Format

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

### The `source` Field

| Value | Meaning |
|-------|---------|
| `human` | Manually run by a human operator |
| `llm_critic` | Triggered by the LLM Critic (Phase 4 Feedback Loop) |
| `scheduled` | Triggered by a scheduled/cron task |

### The `tags` Field

Supports `["test"]` and `["experiment"]` tags. Records tagged `test` can be
filtered out in downstream analysis.

### Usage

```python
from quantpits.utils.operator_log import OperatorLog

with OperatorLog("static_train", args=sys.argv[1:]) as oplog:
    oplog.set_source("llm_critic")
    oplog.set_action_item_id("55b3a485-dfc0-4e6e-bdb3-e843ab4f5905")
    # ... main script logic ...
    oplog.set_result({"n_models": 20, "anchor_date": "2026-04-24"})
```

---

## 2. Config Ledger

**File**: `quantpits/scripts/deep_analysis/config_ledger.py`

Snapshots the current production configuration on every `run_deep_analysis.py`
invocation, saved to `data/config_history/config_snapshot_{date}.json`.

### Snapshot Contents

- All hyperparameters from every `workflow_config_*.yaml` (per-model)
- `ensemble_config.json` (active combos and their composition)
- `strategy_config.yaml`

### Core Functions

| Function | Purpose |
|----------|---------|
| `snapshot_configs(workspace_root, snapshot_date)` | Full config snapshot |
| `save_snapshot(workspace_root, snapshot)` | Persist to `data/config_history/` |
| `load_previous_snapshot(workspace_root, before_date)` | Load most recent prior snapshot |
| `diff_snapshots(old, new)` | Field-level diff between two snapshots |
| `annotate_with_llm_context(records, reason, action_item_id, critic_score)` | Stamp change records with LLM provenance |

### Change Record Structure

```python
{
    "type": "hyperparam",          # hyperparam | ensemble | strategy | ensemble_switch
    "key": "gru_Alpha158.n_epochs",
    "old": 200,
    "new": 250,
    "impact_domain": "Hyperparameter",
    "semantic_label": "Tuning"     # Tuning | CapacityAdjustment | Regularization | ...
}
```

---

## 3. Training History — Convergence Logs

**File**: `data/training_history.jsonl`

`train_single_model()` automatically appends a record after each training run,
covering convergence diagnostics and model performance metrics.

### Record Format

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

### Key Fields

| Field | Meaning |
|-------|---------|
| `early_stopped` | Whether early stopping terminated training before max epochs |
| `actual_epochs` | Epochs actually trained |
| `configured_epochs` | Configured epoch limit |
| `best_epoch` / `best_score` | Best validation epoch and its score |
| `converged` | `actual_epochs == configured_epochs` (i.e., ran to completion) |
| `IC_Mean` / `ICIR` | Single-model IC mean and information ratio |
| `Ann_Excess` / `Max_DD` | Single-model backtest excess return and max drawdown |
| `duration_seconds` | Training wall-clock time (used by Phase 4 priority scheduling) |

### Important Notes

- Each training run appends a new record (never overwrites); history accumulates
- Predict-only cycles do NOT produce new records
- `model_performance_{date}.json` inherits its `convergence` field from the
  previous file via `merge_performance_file()`, preserving convergence data
  across predict-only overwrites

---

## 4. Fusion Run Ledger

**File**: `data/fusion_run_ledger.jsonl`

`ensemble_fusion.py` appends a record after each backtest run. Consumed by the
`EnsembleEvolutionAgent`.

### Key Fields

- `combo_name`: Ensemble combo name
- `run_date`: Run date
- `is_oos`: Whether this was an OOS (Out-of-Sample) evaluation
- `calmar`, `annualized_return`, `max_drawdown`: Backtest metrics
- `loo_contributions`: Leave-One-Out model contributions (`{model: {loo_ic,
  full_ic, delta}}`)

### OOS Evaluation

Trigger OOS fusion with `--only-last-years 1`:
```bash
python -m quantpits.scripts.ensemble_fusion --from-config-all --only-last-years 1
```

Default exclusion period controlled by `config/oos_config.json`:
```json
{
    "exclude_last_years": 1,
    "exclude_last_months": 0
}
```

---

## 5. Training Context — Training Mode Awareness

**File**: `quantpits/scripts/deep_analysis/training_context.py`

Introduced in Phase 2 to provide all analysis agents with context-awareness of the active workspace state and training mode. This prevents misleading alarms during specific operation cycles, such as predict-only cycles.

### Core Capabilities

- **Mode Recognition**: Automatically determines if the current phase runs in a `predict_only` state.
- **Orphan Model Guard**: Identifies stale or legacy models left in the configuration that are no longer updated during predict-only phases.
- **File & Schema Verification**: Exposes helper functions for validating schema consistency on prediction CSV columns.

---

## 6. Phase 2 — Agent Enhancements

### Training Health Agent (New)

- **Data Integrity Guard**: Strongly validates CSV output schemas of prediction files (e.g., must contain `datetime`, `instrument`, `score`).
- **Drawdown & Mode Awareness**: Incorporates `TrainingContext` to identify features expected in predict-only cycles and penalizes missing predictions.
- **Anomaly Interceptor**: Captures invalid or extremely large loss values during training.

### Model Health Agent

- **Training date tracing**: extracts each model's last real training date (not
  predict-only) from `training_history.jsonl`
- **Convergence detection**: classifies models as `underfitting` (early-stopped
  too soon), `overfitting` (ran full epochs but low IC), or normally converged
- **Staleness detection**: flags models not retrained for more than N weeks

### Ensemble Eval Agent

- **OOS history comparison**: reads OOS records from `fusion_run_ledger.jsonl`,
  computes OOS Calmar trend slope
- **LOO contribution**: consumes `model_contribution_{date}.json` generated by
  `ensemble_fusion.py`
- **Combo change detection**: three-tier change taxonomy (composition change /
  active combo switch / content mutation)

### Market Regime Agent

- Sliding-window detection of regime-switch events and frequency
- Output: trend label (bull/bear/range), volatility percentile, drawdown depth

### Prediction Audit Agent

- Decomposes `model_opinions_{date}.json` into per-model independent prediction
  hit rates
- Consensus vs. divergence analysis
