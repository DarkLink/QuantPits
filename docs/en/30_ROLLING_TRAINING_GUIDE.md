# Rolling Training Guide

> The 30-series documentation focuses on **non-static training** paradigms — where training windows slide forward over time.

---

## Overview

Traditional static training (`static_train.py --full`, `static_train.py`) uses **fixed date ranges** to train models. As market regimes shift, static models gradually lose predictive power.

**Rolling Training** divides the timeline into multiple sliding windows and trains models independently on each window, keeping them continuously adapted to the latest market conditions.

### Static vs. Rolling

| Feature | Static Training | Rolling Training |
|---------|----------------|-----------------|
| Training Range | Fixed (e.g. 2015–2022) | Sliding windows (each window trained independently) |
| Number of Models | 1 per model | 1 per model × N windows |
| Adaptability | Low (relies on long-term statistical features) | High (slides with market regime) |
| Prediction Output | Single continuous segment | Multi-segment stitched (auto-concatenated into one file) |
| Downstream Compat. | `latest_train_records.json` | `latest_rolling_records.json` (switch via `--record-file`) |

### Coexistence Architecture

Rolling training is **fully independent** from static training and coexists within the same Workspace:

```text
output/
├── predictions/               # Static training predictions
│   └── rolling/               # Rolling training predictions (per-window CSV + stitched CSV)
data/
├── latest_train_records.json  # Static training records
├── latest_rolling_records.json# Rolling training records
└── rolling_state.json         # Rolling training run state (for crash recovery)
```

---

## Core Script

| Script | Purpose |
|--------|---------|
| `rolling_train.py` | Main rolling training script: cold start, daily mode, predict-only, crash recovery |

---

## Time Window Slicing

### Configuration

Configure in `config/rolling_config.yaml`:

```yaml
rolling_start: "2020-01-01"   # T: Start date
train_years: 3                # X: Training period (integer years)
valid_years: 1                # Y: Validation period (integer years)
test_step: "3M"               # Z: Test step size (nM or nY)
```

### Slicing Formula

For the `n`-th window (0-indexed):

```
Train: [T + nZ,       T + X + nZ − 1d]
Valid: [T + X + nZ,   T + X + Y + nZ − 1d]
Test:  [T + X + Y + nZ, T + X + Y + (n+1)Z − 1d]
```

> [!IMPORTANT]
> **Strictly non-overlapping**: Train, validation, and test segments have zero date overlap, including endpoints. `train_end + 1d = valid_start`, `valid_end + 1d = test_start`.

### Example

`T=2020-01-01, X=3Y, Y=1Y, Z=3M`:

| Window | Train | Valid | Test |
|:------:|-------|-------|------|
| W0 | 2020-01-01 ~ 2022-12-31 | 2023-01-01 ~ 2023-12-31 | 2024-01-01 ~ 2024-03-31 |
| W1 | 2020-04-01 ~ 2023-03-31 | 2023-04-01 ~ 2024-03-31 | 2024-04-01 ~ 2024-06-30 |
| W2 | 2020-07-01 ~ 2023-06-30 | 2023-07-01 ~ 2024-06-30 | 2024-07-01 ~ 2024-09-30 |
| W3 | 2020-10-01 ~ 2023-09-30 | 2023-10-01 ~ 2024-09-30 | 2024-10-01 ~ 2024-12-31 |

The last window's `test_end` is automatically truncated to `anchor_date` (the latest Qlib trading day).

---

## Execution Modes

### Mode 1: Cold Start

**Required for the first run.** Generates all windows and trains them sequentially.

```bash
# Cold start all enabled models
python quantpits/scripts/rolling_train.py --cold-start --all-enabled

# Specify models
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158

# Append new models (Merge Mode)
python quantpits/scripts/rolling_train.py --merge --models new_model_A

# Dry-run: preview window slicing only
python quantpits/scripts/rolling_train.py --cold-start --dry-run --all-enabled
```

Cold Start Add-on Features:
- `--merge`: Append new models to an existing cold-started state. Already-trained models will not be re-trained. It performs window training only for the newly added models, then merges their global predictions (`pred.pkl`) with existing ones.
- `--backtest`: Append this flag to automatically execute a complete Qlib backtest over the aggregate time frame after all training, predicting, and merging finishes. Standardized backtest artifacts (returns reports, positions) will be saved.

Cold start workflow:
1. Read parameters from `rolling_config.yaml`
2. Generate all rolling windows (up to anchor_date)
3. Train + predict for each window × model combination
4. Concatenate all windows' predictions into a continuous time series
5. Save `latest_rolling_records.json`

### Mode 2: Daily Mode

Automatically detects whether new windows need training:
- **New window detected** → Train new window + re-concatenate
- **No new window** → Predict using the latest model

```bash
python quantpits/scripts/rolling_train.py --all-enabled
```

### Mode 3: Predict Only

Use the most recently trained window's model to predict on new data:

```bash
python quantpits/scripts/rolling_train.py --predict-only --all-enabled
```

### Mode 4: Standalone Backtest Evaluation

If a run was previously executed and `latest_rolling_records.json` exists with concatenated predictions, but the comprehensive backtest was skipped (or is desired to be rerun with updated configs), you can use the standalone backtest mode. This mode skips all machine learning training and prediction steps. It directly runs a full Qlib Backtest simulation using the stored concatenated prediction scores (`pred.pkl`).

```bash
python quantpits/scripts/rolling_train.py --backtest-only
```

Standardized artifacts (`report_normal_<freq>.pkl`, `positions_normal_<freq>.pkl`, etc.) and indicator summaries will be stored under the designated comprehensive MLflow record.

### Crash Recovery

After interruption, automatically skips completed window × model pairs:

```bash
python quantpits/scripts/rolling_train.py --resume
```

### State Inspection

```bash
# View current state
python quantpits/scripts/rolling_train.py --show-state

# Clear state (start fresh)
python quantpits/scripts/rolling_train.py --clear-state
```

---

## Model Selection

Consistent with static training, all model filtering options are supported:

| Flag | Description |
|------|-------------|
| `--models m1,m2` | Select by name |
| `--algorithm alg` | Filter by algorithm |
| `--dataset ds` | Filter by dataset |
| `--tag tag` | Filter by tag |
| `--all-enabled` | All enabled models |
| `--skip m1,m2` | Exclude specific models |

---

## Downstream Integration

Rolling training predictions seamlessly connect to downstream scripts via `--record-file`:

```bash
# Brute force screening
python quantpits/scripts/brute_force_fast.py \
  --record-file latest_rolling_records.json

# Ensemble fusion
python quantpits/scripts/ensemble_fusion.py \
  --from-config --record-file latest_rolling_records.json
```

> [!TIP]
> The downstream workflow is identical for static and rolling training. The only difference is the `--record-file` argument. The default is `latest_train_records.json` (static); specify `latest_rolling_records.json` to switch to rolling.

---

## State Management & Crash Recovery

`rolling_state.json` tracks training progress:

```json
{
    "started_at": "2025-03-14 10:00:00",
    "rolling_config": {"test_step": "3M", ...},
    "anchor_date": "2025-03-14",
    "total_windows": 4,
    "completed_windows": {
        "0": {"linear_Alpha158": "rec_001", "gru_Alpha158": "rec_002"},
        "1": {"linear_Alpha158": "rec_003"}
    }
}
```

- State is saved after every completed window × model pair
- Use `--resume` after interruption to skip completed items
- `--clear-state` resets the state (old state is auto-backed up to `data/history/`)

---

## MLflow Experiment Naming

| Experiment Name | Contents |
|----------------|----------|
| `Rolling_Windows_{FREQ}` | Individual window training records |
| `Rolling_Combined_{FREQ}` | Stitched full prediction records |

Where `{FREQ}` is the trading frequency (e.g. `WEEK`, `DAY`).

---

## Configuration Reference

Full `config/rolling_config.yaml` example:

```yaml
# Rolling Training Configuration

rolling_start: "2020-01-01"   # T: Start date
train_years: 3                # X: Training period length (integer years)
valid_years: 1                # Y: Validation period length (integer years)
test_step: "3M"               # Z: Test step size
                              #   - nM: n months (e.g. 3M, 6M)
                              #   - nY: n years (e.g. 1Y)
```

> [!CAUTION]
> `train_years` and `valid_years` must be **integer years**. `test_step` must be `nM` (integer months) or `nY` (integer years). Fractional values are not supported.
