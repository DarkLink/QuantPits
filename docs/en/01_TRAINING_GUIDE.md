# QuantPits Training System Guide

## Overview

The training system consists of three main scripts that share the same utility modules and model registry:

| Script | Purpose | Training | Data Source | Save Semantics |
|------|------|------|--------|----------|
| `prod_train_predict.py` | Full training+predict | ✅ | configs | `latest_train_records.json` |
| `incremental_train.py` | Incremental training+predict | ✅ | configs | `latest_train_records.json` |
| `prod_predict_only.py` | Prediction only | ❌ | Existing models | `latest_train_records.json` |
| `pretrain.py` | Base model pre-training | ✅ | configs | `data/pretrained/` (state_dict) |

Both scripts automatically back up the history to `data/history/` before modifying `latest_train_records.json`.

---

## File Structure

```text
QuantPits/
├── quantpits/
│   ├── scripts/                      # Core system logic
│   │   ├── prod_train_predict.py   # Full training script
│   │   ├── incremental_train.py      # Incremental training script
│   │   ├── prod_predict_only.py    # Prediction-only script (no training)
│   │   ├── pretrain.py               # 🧠 Base model pre-training script
│   │   ├── check_workflow_yaml.py    # 🔧 YAML config production validation & fix
│   │   └── train_utils.py            # Shared utility module
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
        │   ├── predictions/          # Prediction CSV results
        │   └── model_performance_*.json # Model performance metrics (IC/ICIR)
        ├── data/
        │   ├── history/              # 📦 Auto-backed up historical files
        │   ├── pretrained/           # 🧠 Pre-trained base models (.pkl + .json)
        │   └── run_state.json        # State tracker for incremental training
        └── latest_train_records.json # Current training records
```

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
3. Use `incremental_train.py --models xxx` to train and verify it independently
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

## Full Training (`prod_train_predict.py`)

### Usage Scenarios
- Routine production full retraining
- When all model records need a complete refresh

### Execution

```bash
cd QuantPits
python quantpits/scripts/prod_train_predict.py
```

### Behavior
1. Trains all `enabled: true` models in `model_registry.yaml`
2. Upon completion, **fully overwrites** `latest_train_records.json`
3. Auto-backs up to `data/history/train_records_YYYY-MM-DD_HHMMSS.json` before overwriting
4. Saves performance metrics to `output/model_performance_{anchor_date}.json`

---

## Incremental Training (`incremental_train.py`)

### Usage Scenarios
- A new model was added, and you only want to train that one
- A model's hyperparameters were adjusted and it requires retraining
- A previously failed model needs to be re-run
- Avoiding the massive time/resource cost of full retraining

### Model Selection Methods

```bash
cd QuantPits

# 1. By name (comma-separated)
python quantpits/scripts/incremental_train.py --models gru,mlp

# 2. By algorithm
python quantpits/scripts/incremental_train.py --algorithm lstm

# 3. By dataset
python quantpits/scripts/incremental_train.py --dataset Alpha360

# 4. By tag
python quantpits/scripts/incremental_train.py --tag tree

# 5. By market
python quantpits/scripts/incremental_train.py --market csi300

# 6. All enabled models (merge mode)
python quantpits/scripts/incremental_train.py --all-enabled

# 7. Combinations
python quantpits/scripts/incremental_train.py --all-enabled --skip catboost_Alpha158
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
python quantpits/scripts/incremental_train.py --models gru,mlp --dry-run
```

### Rerun / Resume (Crash Recovery)

If training is interrupted (model crashes or killed manually), the execution state is auto-saved to `data/run_state.json`.

```bash
# View last run state
python quantpits/scripts/incremental_train.py --show-state

# Resume unfinished training (skips successfully completed models)
python quantpits/scripts/incremental_train.py --models gru,mlp,alstm_Alpha158 --resume

# Clear run state (start fresh)
python quantpits/scripts/incremental_train.py --clear-state
```

**Note**: `--resume` only skips successfully completed models. **Failed models will be retrained.**

### Viewing the Model Registry

```bash
# List all registered models
python quantpits/scripts/incremental_train.py --list

# Filter list by conditions
python quantpits/scripts/incremental_train.py --list --algorithm gru
python quantpits/scripts/incremental_train.py --list --dataset Alpha360
python quantpits/scripts/incremental_train.py --list --tag tree
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

## History Backups

All vital files are auto-backed up to `data/history/` prior to modification:

```text
data/history/
├── train_records_2026-02-11_165306.json      # History of latest_train_records.json
├── train_records_2026-02-18_120000.json
├── model_performance_2026-02-06_165306.json  # History of performance metrics
└── run_state_2026-02-12_150000.json          # History of run states
```

This runs automatically without manual orchestration.

---

## Typical Workflows

### Scenario 1: Routine Training

```bash
cd QuantPits
python quantpits/scripts/prod_train_predict.py
python quantpits/scripts/ensemble_predict.py --method icir_weighted --backtest
```

### Scenario 1b: Predict Only after Data Update (No Retraining)

```bash
cd QuantPits
# Predict new data using existing models
python quantpits/scripts/prod_predict_only.py --all-enabled
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
python quantpits/scripts/incremental_train.py --models new_model_name

# 4. If verified, change enabled: true
```

### Scenario 3: Retraining a Model After Param Tuning

```bash
# After modifying YAML config
python quantpits/scripts/incremental_train.py --models gru
```

### Scenario 4: Crash Recovery

```bash
# First run (interrupted mid-way)
python quantpits/scripts/incremental_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360
# ... gru completes, mlp fails, subsequent models haven't started ...

# View state
python quantpits/scripts/incremental_train.py --show-state

# Resume (groups gru into completed logic)
python quantpits/scripts/incremental_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360 --resume
```

### Scenario 5: Training Only Tree Models

```bash
python quantpits/scripts/incremental_train.py --tag tree
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
# Available in both incremental_train and prod_predict_only
python quantpits/scripts/incremental_train.py --models gats_Alpha158_plus --no-pretrain
```

---

## Concerning LSTM and GATs

- `gats_Alpha158_plus` depends on `lstm_Alpha158` by default.
- Full Workflow:
  1. Pre-train base model (Optional if already exists):
     `python quantpits/scripts/pretrain.py --for gats_Alpha158_plus`
  2. Train upper model:
     `python quantpits/scripts/incremental_train.py --models gats_Alpha158_plus`

- To bypass pre-training and use random weights, use the `--no-pretrain` flag.


---

## Comprehensive Parameter List

```text
python quantpits/scripts/incremental_train.py --help

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
