# QuantPits Training System Guide

## Overview

The training system consists of three main scripts that share the same utility modules and model registry:

| Script | Purpose | Training | Data Source | Save Semantics |
|------|------|------|--------|----------|
| `static_train.py --full` | Full training | ✅ | configs | `latest_train_records.json` |
| `static_train.py` | Incremental training | ✅ | configs | `latest_train_records.json` |
| `static_train.py --predict-only` | Prediction only | ❌ | Existing models | `latest_train_records.json` |
| `pretrain.py` | Base model pre-training | ✅ | configs | `data/pretrained/` (state_dict) |

Real static/CPCV commands publish through one typed execution kernel. Only one real command may hold
`data/locks/training_execution.lock` in a workspace; lightweight plans do not create it. The current
record and the run's performance output form a recoverable publication unit under
`data/training_transactions/<run-id>-<fingerprint>/`, with `latest_train_records.json` replaced last as
the canonical current pointer.
This complete contract currently applies to static/CPCV only. Rolling and CPCV-rolling still use
their existing independent lifecycle. Their real mutating commands share only the workspace training
execution lease and are not yet backed by State V3 or the publication coordinator.

### Training Record V2

New records use `schema_version: 2`. `model_records` is the authoritative identity map: every
`model@mode` entry carries its actual output experiment, recorder, operation, prediction coverage,
and optional source-recorder lineage. Top-level `models`, `*_experiment_name`, and `anchor_date`
remain compatibility projections and must not override per-model V2 identity.

Legacy V1 files remain readable as `legacy_unverified`. Audit or preview them without mutation:

```bash
python -m quantpits.tools.audit_training_records \
  --workspace workspaces/Demo_Workspace --json
```

Default JSON contains only counts and stable issue codes. Add `--preview` explicitly in a controlled
terminal to include the in-memory migration proposal.
Use `--verify-mlflow` to validate experiment/recorder identity and workspace containment;
`--verify-predictions` additionally checks persisted `pred.pkl` coverage. The audit never initializes
a missing backend. Full training publishes the current registry only when every selected model
succeeds and produces a verified entry.

A producer should publish `ready` only after validating the output recorder and persisted
`pred.pkl`. Failed or skipped operations must not replace an existing current pointer.

V2 uses strict version dispatch. Once `schema_version: 2` is declared, matching `models` and
`model_records` mappings are mandatory; readers never fall back to top-level V1 fields. A `ready`
entry must describe actual prediction start/end/rows, while predict-only entries also carry their
immediate source recorder, experiment, and operation. Ensemble execution rechecks recorder
experiment/artifact identity and persisted `pred.pkl` coverage. The registry is evidence, not a
substitute for runtime verification.

### Lightweight plans and run audit

`static_train.py` and `cv_train.py` share a plan-first command boundary:

```bash
python -m quantpits.scripts.static_train --full --explain-plan
python -m quantpits.scripts.static_train --predict-only --all-enabled --json-plan
python -m quantpits.scripts.cv_train --all-enabled --explain-plan
```

`--explain-plan` and `--json-plan` only read workspace-contained registry, config, workflow, current
record baseline, optional resume state, and required source-record inputs. They do not initialize
Qlib/MLflow, invoke the safeguard, change cwd,
or write files. Calendar-dependent anchors are reported as `deferred_to_qlib_calendar` rather than
guessed. Legacy `--dry-run` routes to the same lightweight plan.

Real execution writes `output/manifests/{static_train|cv_train}/<run_id>.json` by default and links
`data/operator_log.jsonl`. Use `--run-id` for an explicit identity or `--no-manifest` to disable only
the manifest. Both commands support `--workspace PATH`.

After safeguard approval, the service initializes Qlib once and binds exact dates, ordered targets,
source recorders, the output experiment, and the current-record baseline into an
`execution_fingerprint`. A runner receives exactly one planned target and cannot rediscover or
broaden the target set.

Real execution uses locked, compare-and-swap protected Training State V3. Its phases distinguish target
execution, publication preparation/commit, manifest closure, and terminal status. A same-identity
`--resume` classifies transaction preimages/postimages and deterministically rolls forward or closes
audit evidence; an unknown third version fails closed.

Phase 27 separates recovery evidence more explicitly. `--resume` without `--run-id` adopts the
persisted logical run ID. Every successful target writes immutable, SHA-256-addressed JSON evidence
before publication, so verified expensive work that has not yet been published is reused rather than
executed again. A receipt proves current-output commit, target evidence proves model work, and state
coordinates recovery. History events are receipt/target linked and manifests use write-or-adopt
semantics: identical evidence is adopted and conflicting evidence fails closed.
Recovery actions are selected by a pure classifier from state, target evidence, intent, receipt, and
current member observations. Unknown output bytes or identity mismatches never enter target or
publication execution. History, promotion, manifest, OperatorLog, and terminal-state work are
individually durable closure steps. A warning retains state; a same-identity `--resume` retries only
unfinished steps, and `run_state.json` is cleared only after closure is complete.
A receipt authorizes closure only when its schema/status, run/transaction, target order, complete
output ledger, and every current postimage fingerprint exactly match the intent. Missing, duplicate,
extra, or changed members fail closed. The manifest-linked OperatorLog JSONL entry is appended,
flushed, and fsynced before `operator_log_linked` completes; a write failure retains `closing` state.
Manifests use exact canonical bytes, and durable logical start/finish timing is not replaced by the
timestamp of a later closure attempt.
An older compatible State V3 is backfilled only when transaction-bound intent/receipt, complete
target evidence, current postimages, and recoverable timing prove every new field. Otherwise resume
fails closed with clear/explicit-migration guidance; no automatic migration is performed.

Publication semantics:

- Full runs commit the record/performance bundle only after every target returns verified evidence; otherwise existing bytes are preserved.
- Incremental and predict-only runs merge successes in one transaction and preserve failed pointers; the receipt is durable before the aggregate command failure is returned.
- Resume requires the same date/config/source/target resume fingerprint and skips only a published recorder that remains the current pointer; expected current-record baseline changes do not create false conflicts.
- Manifests list only outputs proven by the publication receipt; unpublished MLflow recorders remain outcome evidence only.
- Static-train promotion runs after a durable receipt. Promotion failure is a visible warning and does not roll back a verified current pointer.
- Typed static/CPCV runners do not write global history paths. The service appends stable-`event_id`, workspace-local history only for newly published receipt entries, so recovery can replay it idempotently.

---

## File Structure

```text
QuantPits/
├── quantpits/
│   ├── scripts/                      # Core system scripts
│   │   ├── static_train.py           # Unified static training entry point
│   │   ├── pretrain.py               # 🧠 Base model pre-training script
│   │   └── check_workflow_yaml.py    # 🔧 YAML config production validation & fix
│   ├── utils/                         # Shared utility modules
│   │   ├── train_utils.py            # Date calculus, YAML injection, model registry, record merging
│   │   ├── predict_utils.py          # Prediction data load/save
│   │   ├── config_loader.py          # Workspace-level config loading
│   │   ├── workspace.py              # Explicit WorkspaceContext and fingerprint helpers
│   │   ├── strategy.py               # Strategy config / backtest strategy construction
│   │   └── ...                       # More shared modules (see System Overview)
│   ├── config_contracts/              # Workspace config validation, normalization, fingerprints
│   ├── training/                      # plan/runner/lease/State V3/publication journal/service
│   └── docs/
│       └── 01_TRAINING_GUIDE.md      # This document
│
└── workspaces/
    └── <YourWorkspace>/              # Isolated trading environments
        ├── config/
        │   ├── model_registry.yaml   # 📋 Model registry (Core config)
        │   ├── model_config.json     # Date/Market parameters
        │   └── workflow_config_*.yaml# Qlib workflow bindings for each model
        ├── output/
        │   ├── predictions/          # (Managed by Qlib Recorders inside mlruns)
        │   └── model_performance_*.json # Model performance metrics (IC/ICIR)
        ├── data/
        │   ├── history/              # 📦 Auto-backed up historical files
        │   ├── pretrained/           # 🧠 Pre-trained base models (.pkl + .json)
        │   ├── run_state.json        # Locked, CAS-protected Training State V3
        │   ├── locks/                # Training execution/publication advisory locks
        │   ├── training_transactions/# Recoverable publication intents/receipts
        │   └── training_runs/        # Immutable target evidence/closure per logical run
        └── latest_train_records.json # Current training records
```

> Test policy: automatic full-suite validation is owner-operated. Implementation and review agents
> should not automatically run `pytest tests/`, coverage, Docker full-suite, or equivalent commands.
> A minimal focused test is reserved for a concrete high-risk contract that cannot be established by
> static review, and its reason and exact command must be reported at handoff.

---

## Model Registry (`config/model_registry.yaml`)

### Structure

Every model is organized by three dimensions: **algorithm** + **dataset** + **market**

```yaml
models:
  gru:                              # Model unique identifier
    algorithm: gru                  # Algorithm name
    dataset: Alpha158               # Data handler
    market: csi300                  # Target market (Metadata tag used for CLI filtering)
    yaml_file: config/workflow_config_gru.yaml  # Qlib workflow config
    enabled: true                   # Whether to participate in full training
    tags: [basemodel, ts]           # Classification tags (for filtering)
    pretrain_source: lstm_Alpha158  # (Optional) Declare dependency on base model
    notes: "Optional notes"         # Notes
```

#### Key Fields:
- **`tags: [basemodel]`**: Marks the model as a pre-trainable base model.
- **`pretrain_source`**: Tells the system which base model this upper-layer model depends on. The system will automatically look for the corresponding `_latest.pkl`.

> [!NOTE]
> **Distinction of Market Configurations**: The `market` field in the registry acts strictly as a **Model Metadata Tag** intended for CLI selection filtering via `--market` during incremental training or predictions. Actual data extraction bounds are perpetually steered by the global `market` setting inside `model_config.json`.

### Adding a New Model

1. Create a YAML workflow config at `config/workflow_config_xxx.yaml`
2. Add a model entry in `model_registry.yaml`
3. Use `static_train.py --models xxx` to train and verify it independently
4. Once confirmed, set `enabled` to `true`

### Disabling a Model

Set `enabled` to `false`. It will be automatically skipped during full training. Incremental training can still target it via `--models`.

### Available Tags

| Tag | Meaning | Models |
|------|------|------|
| `ts` | Time-series | gru, alstm, tcn, sfm, ... |
| `nn` | Neural Network | mlp, TabNet |
| `tree` | Tree-based | lightgbm, catboost |
| `attention` | Attention mechanism | alstm, transformer, TabNet |
| `baseline` | Baseline model | linear |
| `graph` | Graph model | gats |
| `cnn` | Convolutional NN | tcn |
| `basemodel` | Used as a base for others | lstm |

---

## Full Training (`static_train.py --full`)

### Usage Scenarios
- Routine production full retraining
- When all model records need a complete refresh

### Execution

```bash
cd QuantPits
python quantpits/scripts/static_train.py --full
```

### Behavior
1. Trains all `enabled: true` models in `model_registry.yaml`
2. Upon completion, **fully overwrites** `latest_train_records.json`
3. Stores exact record/performance preimages and postimages in the publication transaction directory
4. Verifies performance, replaces `latest_train_records.json` last, and writes a committed receipt

---

## Incremental Training (`static_train.py`)

### Usage Scenarios
- A new model was added, and you only want to train that one
- A model's hyperparameters were adjusted and it requires retraining
- A previously failed model needs to be re-run
- Avoiding the massive time/resource cost of full retraining

### Model Selection Methods

```bash
cd QuantPits

# 1. By name (comma-separated)
python quantpits/scripts/static_train.py --models gru,mlp

# 2. By algorithm
python quantpits/scripts/static_train.py --algorithm lstm

# 3. By dataset
python quantpits/scripts/static_train.py --dataset Alpha360

# 4. By tag
python quantpits/scripts/static_train.py --tag tree

# 5. By market
python quantpits/scripts/static_train.py --market csi300

# 6. All enabled models (merge mode)
python quantpits/scripts/static_train.py --all-enabled

# 7. Combinations
python quantpits/scripts/static_train.py --all-enabled --skip catboost_Alpha158
```

### Save Behavior (Merge Semantics)

| Condition | Behavior |
|------|------|
| Model with same name exists | Overwrites its recorder ID and performance stats |
| New model | Appended to the records |
| Untrained models | Previously existing records remain unchanged |

### Dry-run (Preview plan only)

```bash
# Preview which models will be trained without executing
python quantpits/scripts/static_train.py --models gru,mlp --dry-run
```

### Rerun / Resume (Crash Recovery)

If training is interrupted (model crashes or killed manually), the execution state is auto-saved to `data/run_state.json`.

```bash
# View last run state
python quantpits/scripts/static_train.py --show-state

# Resume unfinished training (skips successfully completed models)
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158 --resume

# Clear run state (start fresh)
python quantpits/scripts/static_train.py --clear-state
```

**Note**: `--resume` preserves receipt-proven models that remain current and reruns failed models. If
model work and publication completed but manifest/state closure was interrupted, resume closes audit
evidence without retraining.
The ordinary form does not require `--run-id`; if one is supplied explicitly, it must match state
exactly or planning fails before Qlib initialization.

The workspace owner explicitly runs full Python/Docker validation and the real Playground
static/CPCV predict-only release gate. Implementation tools do not automatically run the full suite
or delete/migrate legacy state.

### Viewing the Model Registry

```bash
# List all registered models
python quantpits/scripts/static_train.py --list

# Filter list by conditions
python quantpits/scripts/static_train.py --list --algorithm gru
python quantpits/scripts/static_train.py --list --dataset Alpha360
python quantpits/scripts/static_train.py --list --tag tree
```

---

## Date Handling

Training dates and frequency are controlled by `config/model_config.json`:

| Parameter | Description |
|------|------|
| `train_date_mode` | `last_trade_date` (uses recent trading day) or fixed date |
| `data_slice_mode` | `slide` (sliding window) or `fixed` (fixed dates) |
| `train_set_windows` | Training set window (years) |
| `valid_set_window` | Validation set window (years) |
| `test_set_window` | Test set window (years) |
| `freq` | Trading frequency (`week`/`day`) |

### Notes on Date Switching
- Full and incremental training share the same `model_config.json`
- If you change date parameters during incremental training, **the newly trained models will use the new dates**, while skipped models will remain on the old dates.
- It is highly advised to use incremental training strictly within the same anchor_date window. When rolling dates over, use full training.

---

## CPCV Mode: Purged Cross-Validation

CPCV (Combinatorial Purged Cross-Validation) follows Marcos Lopez de Prado's *Advances in Financial Machine Learning*. Validation periods are "punched out" from the training timeline with purge/embargo gaps, allowing training data from BOTH before AND after the validation period. This keeps models trained on recent market data while preventing information leakage.

### Why Use CPCV?

The traditional `slide` mode (e.g., 8-year train / 2-year valid / 3-year test) produces training data that is years old at prediction time — the model never sees recent market patterns. CPCV partitions the timeline into N equal groups, reserves the most recent groups as a fixed test set, and generates K cross-validation folds. Each fold uses discontiguous training chunks (before and after the validation group with purge gaps).

### Configuration

CPCV is independently enabled via the `purged_cv` config block in `config/model_config.json` —
**`data_slice_mode` does NOT affect CPCV.** `data_slice_mode` (`slide` / `fixed`) only controls
static training windowing and is orthogonal to CPCV:

```jsonc
{
    "data_slice_mode": "slide",    // Controls static training (slide/fixed); does NOT affect CPCV
    "purged_cv": {
        "n_groups": 10,            // Partition [start_time, anchor_date] into N equal-step groups
        "n_test_groups": 2,        // Last N groups are reserved as fixed test set (excluded from CV)
        "n_val_groups": 1,         // Each fold uses N consecutive groups as validation
        "purge_steps": 5,          // Symmetric purge: remove N steps from BOTH sides of validation
        "embargo_steps": 10        // Asymmetric embargo: additional N steps AFTER validation only
    },
    "start_time": "2015-01-01",   // Time range start (shared by CPCV and slide mode)
    "freq": "week"                 // Step unit: day or week
}
```

> [!NOTE]
> **Backward compatibility**: Existing configs with `data_slice_mode: "purged_cv"` continue to work.
> When both `data_slice_mode: "slide"` and the `purged_cv` block are configured, CPCV and static
> training can coexist in the same workspace — run `cv_train.py` and `static_train.py`
> independently without modifying the config file.

**Parameters**:

| Parameter | Meaning |
|-----------|---------|
| `n_groups` | Total groups partitioning `[start_time, anchor_date]` by equal calendar steps |
| `n_test_groups` | Trailing groups reserved as fixed test set (never used in CV) |
| `n_val_groups` | Consecutive groups used as validation per fold |
| `purge_steps` | Steps removed from BOTH sides of validation (frequency-agnostic: 1 step = 1 period of `freq`) |
| `embargo_steps` | Additional steps removed AFTER validation only |

Number of folds: **K = n_groups - n_test_groups - n_val_groups + 1**

### Running CPCV

```bash
# Full CPCV training on all enabled models
python quantpits/scripts/cv_train.py --full

# Incremental CPCV on specific models
python quantpits/scripts/cv_train.py --models lightgbm_Alpha158,gru_Alpha158

# CPCV by tag
python quantpits/scripts/cv_train.py --tag tree

# Preview fold plan
python quantpits/scripts/cv_train.py --all-enabled --dry-run

# Predict only with existing CPCV models
python quantpits/scripts/cv_train.py --predict-only --all-enabled
```

### Downstream Compatibility

CPCV-trained models are stored as `model_name@cpcv` keys in `latest_train_records.json`, coexisting with `@static` and `@rolling` models. Downstream scripts use `--training-mode cpcv`:

```bash
python quantpits/scripts/ensemble_fusion.py --from-config --training-mode cpcv
```

Each CPCV model stores K fold models (`model_fold_0.pkl` … `model_fold_K-1.pkl`) and the K-fold averaged prediction (`pred.pkl`) in the recorder. Downstream fusion loads `pred.pkl` directly — no K-fold awareness needed.

### Model Type Compatibility

| Model Type | Dataset Class | Method |
|-----------|--------------|--------|
| Tree/GBDT (LightGBM, XGBoost, CatBoost) | `PurgedDatasetH` | `pd.concat` of discontiguous time chunks (safe: point-in-time rows) |
| Linear (Linear) | `PurgedDatasetH` | Same |
| Deep Learning (LSTM, GRU, ALSTM, Transformer, TCN, GATs) | `PurgedTSDatasetH` | `ConcatTSDataSampler` logical concatenation (sliding windows never cross purge gaps) |
| TRA (Temporal Routing) | `PurgedMTSDatasetH` | Union mask on pre-computed slice indices |

### Preprocessing Notes

- **Prefer cross-sectional normalizers**: `CSZScoreNorm`, `CSRankNorm` (per-day, per-instrument — immune to temporal leakage)
- **Avoid temporal normalizers**: `ZScoreNorm`, `MinMaxNorm`, `RobustZScoreNorm` fit across time and will leak validation/test statistics. CPCV training emits a `UserWarning` when these are detected.

---

## Publication history and recovery evidence

Typed static/CPCV correctness no longer depends on timestamped `data/history/` backups. Every actual
publication uses:

```text
data/training_transactions/<transaction-id>/
├── intent.json               # Pre/post fingerprints and commit order
├── member-*.preimage         # Exact previous bytes when the member existed
├── member-*.postimage        # Fsynced intended bytes
└── receipt.json              # Verified committed outputs
```

Target and closure evidence for the same logical run is stored under:

```text
data/training_runs/<run-id>/
├── targets/<target-key>.json # Immutable typed result and recorder lineage
└── closure-<transaction-id>.json # Receipt-linked closure status and retry evidence
```

`data/history/` may still contain compatibility backups produced by legacy or rolling tools, but it is
not the recovery authority for new static/CPCV commands.

---

## Typical Workflows

### Scenario 1: Routine Training

```bash
cd QuantPits
python quantpits/scripts/static_train.py --full
python quantpits/scripts/ensemble_predict.py --method icir_weighted --backtest
```

### Scenario 1b: Predict Only after Data Update (No Retraining)

```bash
cd QuantPits
# Predict new data using existing models
python quantpits/scripts/static_train.py --predict-only --all-enabled
# The subsequent brute force/fusion pipeline remains unchanged
python quantpits/scripts/brute_force_fast.py --max-combo-size 3
python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158,alstm_Alpha158
```

> For details see [05_PREDICT_ONLY_GUIDE.md](05_PREDICT_ONLY_GUIDE.md)

### Scenario 2: Adding a New Model

```bash
# 1. Create YAML config
# 2. Add entry to model_registry.yaml (set enabled: false initially)
# 3. Train independently to verify
python quantpits/scripts/static_train.py --models new_model_name

# 4. If verified, change enabled: true
```

### Scenario 3: Retraining a Model After Param Tuning

```bash
# After modifying YAML config
python quantpits/scripts/static_train.py --models gru
```

### Scenario 4: Crash Recovery

```bash
# First run (interrupted mid-way)
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360
# ... gru completes, mlp fails, subsequent models haven't started ...

# View state
python quantpits/scripts/static_train.py --show-state

# Resume (groups gru into completed logic)
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360 --resume
```

### Scenario 5: Training Only Tree Models

```bash
python quantpits/scripts/static_train.py --tag tree
# Equivalent to: --models lightgbm_Alpha158,catboost_Alpha158
```

---

## Configuration Validation and Auto-Fix

To ensure all YAML workflow files meet the production frequency criteria (e.g., `label` predicting future returns based on frequency, `time_per_step` matching frequency, `ann_scaler` based on frequency), an automated validation script is provided. **It is recommended to run this after adding or mutating YAMLs.**

```bash
# Validate that all workflow_config_*.yaml conform to production parameters (day/week)
python quantpits/tools/check_workflow_yaml.py

# Attempt to automatically fix all malformed YAMLs (converts params to production requirements)
python quantpits/tools/check_workflow_yaml.py --fix
```

---

---

## Base Model Pre-training (`pretrain.py`)

Complex models (e.g., GATs, ADD, IGMTF) require a pre-trained base model (e.g., LSTM or GRU) for weight initialization.

### Usage Scenarios
- Providing initialization weights for upper-layer models.
- When features (d_feat) are modified, requiring new compatible base models.

### Core Semantics
- **Pre-training is not logged in records**: It does not modify `latest_train_records.json`.
- **Metadata Validation**: Each pre-trained file comes with a `.json` metadata file. If an upper model's `d_feat` doesn't match the pre-trained file, training will be blocked.

### Common Commands

```bash
# 1. List pre-trainable models and dependencies
python quantpits/scripts/pretrain.py --list

# 2. Pre-train a specific base model
python quantpits/scripts/pretrain.py --models lstm_Alpha158

# 3. Pre-train FOR a specific upper model (Recommended: Aligns dataset config)
# This ensures perfect compatibility even if features are modified.
python quantpits/scripts/pretrain.py --for gats_Alpha158_plus

# 4. Show existing pre-trained files
python quantpits/scripts/pretrain.py --show-pretrained

# 5. Force random weights (Skip pre-training)
# Available in all modes
python quantpits/scripts/static_train.py --models gats_Alpha158_plus --no-pretrain
```

---

## Concerning LSTM and GATs

- `gats_Alpha158_plus` depends on `lstm_Alpha158` by default.
- Full Workflow:
  1. Pre-train base model (Optional if already exists):
     `python quantpits/scripts/pretrain.py --for gats_Alpha158_plus`
  2. Train upper model:
     `python quantpits/scripts/static_train.py --models gats_Alpha158_plus`

- To bypass pre-training and use random weights, use the `--no-pretrain` flag.


---

## Comprehensive Parameter List

```text
python quantpits/scripts/static_train.py --help

Mode:
  --full                  Full training: train all enabled models, overwrite records
  --predict-only          Predict only: use existing models on latest data, no training

Model Selection:
  --models TEXT           Target model names, comma-separated
  --algorithm TEXT        Filter by algorithm
  --dataset TEXT          Filter by dataset
  --market TEXT           Filter by market
  --tag TEXT              Filter by tag
  --all-enabled           Trains all models where enabled=true

Exclusions & Skips:
  --skip TEXT             Skip target models, comma-separated
  --resume                Resume from last interruption

Run Controls:
  --dry-run               Print execution plan without training
  --experiment-name TEXT  MLflow experiment name override

Information:
  --list                  List the model registry
  --show-state            Show last interruption state
  --clear-state           Clear run state file
```
