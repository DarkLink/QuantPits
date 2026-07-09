# OOM-RL Data Infrastructure — Phases 1, 2 & Runtime Linkage

Documents the structured data automatically collected by the OOM-RL system and
the agent signal enhancements delivered in Phase 2, plus compatible linkage
fields between runtime plans/manifests and operation logs.

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
    "run_id": null,
    "manifest_path": null,
    "plan_fingerprint": null,
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

### Runtime Manifest Linkage Fields

`OperatorLog` now has three backward-compatible optional fields for future
commands that integrate with `quantpits.runtime.RunManifest`:

| Field | Meaning |
|-------|---------|
| `run_id` | Stable run ID for this command execution |
| `manifest_path` | Workspace-relative path to the matching `RunManifest`, for example `output/manifests/{command}/{run_id}.json` |
| `plan_fingerprint` | Stable fingerprint of the dry-run plan / public plan dict |

Existing scripts do not need to change. When unset, these fields are `null`.
They record identifiers, paths, and fingerprints only; they do not include raw
configuration content.

### Usage

```python
from quantpits.utils.operator_log import OperatorLog

with OperatorLog("static_train", args=sys.argv[1:]) as oplog:
    oplog.set_source("llm_critic")
    oplog.set_action_item_id("55b3a485-dfc0-4e6e-bdb3-e843ab4f5905")
    # ... main script logic ...
    oplog.set_result({"n_models": 20, "anchor_date": "2026-04-24"})
```

Once a command writes a run manifest, it can link the audit entry to that
manifest:

```python
with OperatorLog("ensemble_fusion", args=sys.argv[1:]) as oplog:
    oplog.set_run_manifest(
        run_id=run_id,
        manifest_path=f"output/manifests/ensemble_fusion/{run_id}.json",
    )
    oplog.set_plan_fingerprint(plan_fingerprint)
    # ... main script logic ...
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

Introduced in Phase 2 to provide all analysis agents with the current workspace's training mode inventory and rolling pipeline status.

### Core Capabilities

- **Training Mode Parsing**: Parses `name@mode` keys from `latest_train_records.json` to identify each model's training algorithm mode (static / cpcv / rolling / cpcv_rolling).
- **Cross-Mode Detection**: `get_cross_mode_models()` returns models that exist in multiple training modes simultaneously.
- **Pipeline Gap Calculation**: `get_rolling_gap_days(mode)` computes the number of days between the last rolling anchor date and the current date, used for staleness detection.
- **Model Key Resolution**: `resolve_model_key(record_id)` reverse-looks up the full `name@mode` identifier from a record/run ID, enabling downstream agents (e.g., Model Health) to distinguish models trained in different modes.

**Data Sources** (read by `from_workspace()` factory):

| File | Purpose |
|------|---------|
| `data/latest_train_records.json` | Model→run ID mappings, anchor dates, `@` suffix mode parsing |
| `config/rolling_config.yaml` | Rolling scheduler configuration |
| `data/rolling_state.json` | Slide rolling window progress |
| `data/rolling_state_cpcv.json` | CPCV rolling window progress |
| `config/model_config.json` | `purged_cv` section (CPCV parameters) |

---

## 6. Phase 2 — Agent Enhancements

### Training Health Agent (New)

Comprehensively evaluates training pipeline health, rolling progress, and trade execution trends:

- **Mode Coverage Audit**: Checks each model's training mode coverage (static / cpcv / rolling / cpcv_rolling), flags models missing expected modes.
- **Rolling Progress & Staleness**: Inspects rolling pipeline completion percentage, marks pipelines over 90 days stale (warning), and completed pipelines (positive).
- **Alpha Decay Monitoring**: Compares `Idiosyncratic_Alpha` from `rolling_metrics_20.csv` and `rolling_metrics_60.csv`. Short-term (20d) mean falling below long-term (60d) mean into negative territory → warning "Alpha Decay". Short-term surging above long-term → positive "Alpha Surge".
- **Execution Friction Detection**:
  - Slippage z-score: Computes 60-day z-score of `Exec_Slippage_Mean`. z < -2.0 → critical "High Slippage Extreme".
  - Delay cost z-score: Computes 60-day z-score of `Delay_Cost_Mean`. z < -2.0 → warning "Delay Cost Degradation".
- **Barra Factor Drift**: Checks `Exposure_Liquidity` percentile over a 252-day rolling window. ≤5% (micro-cap drift) → critical; ≥95% (large-cap overload) → warning.
- **Orphan Model Detection**: Detects enabled models not in any active combo → warning/info.

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
