# Predict-Only Execution Guide

## Overview

`scripts/weekly_predict_only.py` serves to operate explicit predictive generations across freshly updated datasets utilizing pre-existing architectures, **exempting completely from triggering parameter optimization training intervals.**

**Usage Context**: Employed after dataset appending sequences to rapidly spin fresh predictive targets without model retraining, seamlessly linking onto the Exhaustive Combo/Fusion streams.

**Pipeline Placement**: ~~Training Array~~ → **Prediction Output (This Step)** → Exhaustive Combo Mapping → Fusion Engine → Orders Gen

---

## Quick Start

```bash
cd QuantPits

# Broadcast prediction arrays across all active enabled architectures
python engine/scripts/weekly_predict_only.py --all-enabled

# Bound predictions against explicit named algorithm targets
python engine/scripts/weekly_predict_only.py --models gru,mlp,linear_Alpha158

# Sub-select via Tag classes (e.g. tree structures only)
python engine/scripts/weekly_predict_only.py --tag tree

# Run sequence preview inspection tracing mapping behavior
python engine/scripts/weekly_predict_only.py --all-enabled --dry-run
```

---

## Architecture Mechanics

```text
1. Parse source parameters natively from `latest_train_records.json`.
2. Determine executable boundaries matching conditional override requests.
3. For each active bound mapping:
   a. Hook natively onto the MLflow Recorder object loading its `.pkl` algorithm structure.
   b. Interface with `model_config.json` defining valid updated target chronological windows.
   c. Compile target Datasets + execute `.predict()` outputs.
   d. Instantiate novel Recorder mappings mapped inside the `Weekly_Predict_Only` branch identifier.
   e. Commit `pred.pkl` artifact persistence + Execute standard `SignalRecord` structures extracting IC parameters.
   f. Emulate parallel CSV data distributions terminating in `output/predictions/`.
4. Trigger Incremental Merge persisting logic to `latest_train_records.json`.
```

> [!IMPORTANT]
> Post-execution, the corresponding entries targeting active models within `latest_train_records.json` overwrite their respective `experiment_name` and `record_id` parameters mapped out to this explicit new execution trace. Downstream nodes parsing this dictionary seamlessly consume the updated hashes.

---

## Comprehensive Parameter Overrides

| Argument | Default Set | Description Overview |
|------|--------|------|
| `--models` | None | Strict manual explicit target array overrides (Comma-separated) |
| `--algorithm` | None | Logical sorting index class filters |
| `--dataset` | None | Data-handler bounding sorting |
| `--market` | None | Geographic asset bounds filters |
| `--tag` | None | Logical attribute index filters |
| `--all-enabled` | - | Instructs execution across all architectures denoting active booleans |
| `--skip` | None | Negates explicitly queried model components internally |
| `--source-records` | `latest_train_records.json` | The active persistence reference |
| `--dry-run` | - | Outputs intended mapping without structural state alteration |
| `--experiment-name` | `Weekly_Predict_Only` | Native MLflow tag container name overrides |
| `--list` | - | Render active model registry lists sequentially |

---

## Filter Selections Index

```bash
# 1. Nomenclature targets
python engine/scripts/weekly_predict_only.py --models gru,mlp

# 2. Logic groupings
python engine/scripts/weekly_predict_only.py --algorithm lstm

# 3. Handle structure filtering
python engine/scripts/weekly_predict_only.py --dataset Alpha360

# 4. Attribute label constraints
python engine/scripts/weekly_predict_only.py --tag tree

# 5. Native configuration activation
python engine/scripts/weekly_predict_only.py --all-enabled

# 6. Combined inclusive/exclusionary constraints
python engine/scripts/weekly_predict_only.py --all-enabled --skip catboost_Alpha158
```

---

## Artifact Outcomes

```text
output/
├── predictions/
│   ├── gru_2026-02-13.csv              # Emulated prediction CSV dumps
│   ├── mlp_2026-02-13.csv
│   └── ...
└── model_performance_2026-02-13.json   # Compounded IC/ICIR statistics 

latest_train_records.json               # Refreshed primary node dictionary 
```

---

## Persistence Operations (Merge Rules)

Functionality is rigidly analogous to `incremental_train.py`:

| Structural Occurrence | Resolution Behavior |
|------|------|
| Model matching name identifier previously loaded | Discards prior state targeting recorder logic pushing active ID |
| Entirely novel unmapped identifier class | Modifies file indexing inserting distinct parameters |
| Untouched / Filtered classes | Completely preserved unaltered spanning state iterations |

Note: Baseline logic actively maintains archival backups dumping to `data/history/` preemptive to any state overwriting protocol executing.

---

## Sequential Routine Cases

### Scenario 1: Nominal Update + Fast Track Pipeline

```bash
cd QuantPits

# Step 1: Predict via extant unretrained logic nodes
python engine/scripts/weekly_predict_only.py --all-enabled

# Step 2: Validate Exhaustive Compositions
python engine/scripts/brute_force_fast.py --max-combo-size 3

# Step 3: Instantiate Fused Orders Set
python engine/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# Step 4: Dispatch terminal outputs 
# (via order_gen script endpoints natively)
```

### Scenario 2: Target Segment Prediction Generation 

```bash
# Explicitly force execution strictly via tree logic frameworks 
python engine/scripts/weekly_predict_only.py --tag tree

# Target resultant mappings via ensemble bounding explicitly 
python engine/scripts/ensemble_fusion.py \
  --models lightgbm_Alpha158,catboost_Alpha158
```

### Scenario 3: Blind Preview Dry-Run Audit

```bash
# Analyze trace outputs dictating expected pipeline mapping targets solely
python engine/scripts/weekly_predict_only.py --all-enabled --dry-run

# Output full structured components array natively mapped through to YAML configs
python engine/scripts/weekly_predict_only.py --list
```

---

## Architecture Cross-Reference Module Index

| Script Endpoint | Context Usage | Modifies Parameters | Input Required | Primary Render Target |
|------|------|:--------:|------|------|
| `weekly_train_predict.py` | Full Network Overhaul | ✅ | configurations | `latest_train_records.json` |
| `incremental_train.py` | Focused Target Adjustments | ✅ | configurations | `latest_train_records.json` |
| **`weekly_predict_only.py`** | **Extrapolation (Prediction)** | **❌** | **Extant `.pkl`** | **`latest_train_records.json`** |
| `brute_force_ensemble.py` | Exhaustive Combo Sorting | - | train records | leaderboards CSV |
| `ensemble_fusion.py` | Compounded Backtest Validation | - | targeted outputs | Net fusion mappings |

> Operating seamlessly on matching target bounds `latest_train_records.json`, subsequent execution components require absolutely no logic variation downstream.

---

## Safety Guidelines

1. **Pre-requisite Baseline**: The pipeline demands previously active iteration tracking bounds. Ensure `latest_train_records.json` holds functional extant hash addresses prior to engaging predictability loops.
2. **Missing Identifiers**: Models failing validation parsing against source target records are natively suppressed generating terminal warnings without halting the global stream.
3. **Identifier Distinction**: Native operation routes completely through an independent `Weekly_Predict_Only` branch tag inside MLruns effectively protecting discrete training sequences records.
4. **Chronology Constraints**: Evaluation intervals bound strictly through `model_config.json` maintaining strict parity with explicit training mapping architecture.
5. **Autosave Backups**: Any mutable sequence dictating logic overrides implicitly generates historical snapshot representations automatically inside `data/history/` directories natively.
