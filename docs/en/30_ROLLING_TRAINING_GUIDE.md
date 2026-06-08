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
| Downstream Compat. | Unified `latest_train_records.json` with `model@rolling` keys (switch via `--training-mode rolling`) |

### Coexistence Architecture

Rolling training is **fully independent** from static training and coexists within the same Workspace:

```text
output/
├── predictions/               # Static training predictions
│   └── rolling/               # Rolling training prediction records (Qlib Recorders in mlruns)
data/
├── latest_train_records.json  # Unified training records (incl. @rolling)
├── rolling_state.json         # Progress tracker (for resume)
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

### Quick Reference

| Flag | Effect on State | Training Scope | Use Case |
|------|----------------|---------------|---------|
| `--cold-start` | **Wipes everything** | All windows | First run, full rebuild |
| `--merge` | **No deletion**, fill gaps only | Missing windows only | New data arrived (new windows), add new models |
| `--retrain-models` | **Clears specific models** | All windows for those models | After hyperparameter / code changes |
| `--retrain-last` | **Deletes last window** | Last window | Data correction, forced last-window retrain |
| `--predict-only` | No change | No training, predict only | Quick prediction needed |
| `--resume` | No change | Unfinished windows | Resume after interruption |
| `--clear-state` | **Wipes everything** | — | Abandon all history, start over |

### Core Concept

All training modes share the same underlying logic:
1. Read `rolling_config.yaml`, generate full window list from current qlib data date
2. For each model × window, check if a record exists in `rolling_state.json`
3. **Record exists → skip**, **no record → train**
4. Concatenate all window test segments into one prediction file

This guarantees no window is trained twice. Different modes only differ in **which records get cleared** before training.

---

### Mode 1: Full Cold Start (`--cold-start`)

**Clears all records in `rolling_state.json`**, retrains every window for every model from scratch.

```bash
python quantpits/scripts/rolling_train.py --cold-start --all-enabled
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158
python quantpits/scripts/rolling_train.py --cold-start --dry-run --all-enabled  # preview only
```

> [!CAUTION]
> `--cold-start` is irreversible. All training history for all models is permanently lost.

---

### Mode 2: Fill-Missing (`--merge`)

**Does not delete any existing records.** Auto-detects and trains only missing windows. This is the most common maintenance command.

**Typical scenarios**:

1. **New windows appear after qlib data update** (e.g., month rolls over, W41 appears)
   ```bash
   python quantpits/scripts/rolling_train.py --merge --all-enabled
   ```
   Internally: generates window list → detects W41 not in state → trains only W41 → re-stitches predictions.

2. **Add brand-new models to existing state**
   ```bash
   python quantpits/scripts/rolling_train.py --merge --models new_model_A
   ```
   New model has zero records in state → trains all 41 windows. Existing models untouched.

> [!TIP]
   > This is the standard way to "fill-train the last window". No need for `--retrain-last` (which force-retrains an already-existing window). Just use `--merge` to detect and train newly-appeared windows.

---

### Mode 3: Rebuild Specific Models (`--retrain-models`)

**Clears the specified models' records from state**, retrains all their windows from scratch. Other models are completely unaffected.

```bash
# Single model
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158

# Batch
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158,gru_Alpha158
```

Use case: after hyperparameter adjustments, code changes, or wrapper upgrades.

---

### Mode 4: Retrain Last Window (`--retrain-last`)

**Deletes the last window's records from state**, then fills them via daily mode. Use for data corrections or when the last window must be forcibly retrained.

```bash
# All models' last window
python quantpits/scripts/rolling_train.py --retrain-last --all-enabled

# Limit to specific models
python quantpits/scripts/rolling_train.py --retrain-last --models gru_Alpha360
```

> [!TIP]
> If a **new** window appeared due to data update (not retraining an existing one), use `--merge` instead.

---

### Mode 5: Predict Only (`--predict-only`)

No training. Uses the latest window's model weights to predict on current data.

```bash
python quantpits/scripts/rolling_train.py --predict-only --all-enabled
```

**Window gap detection**: when qlib data has been updated but new windows haven't been trained:

1. Detects whether each model has untrained windows (state's latest window < current available window)
2. **Default behavior**: skips models with gaps, prints clear guidance with two options:
   - `--retrain-models <model>` (recommended, proper training)
   - `--allow-stale-predict` (expedient, predict with old weights)
3. **`--allow-stale-predict`**: when explicitly enabled, uses old weights to cover all available data. Later `--retrain-models` will overwrite the gap predictions.

---

### Mode 6: Resume (`--resume`)

After interruption, automatically skips completed windows and continues from where it stopped.

```bash
python quantpits/scripts/rolling_train.py --resume
```

---

### Mode 7: Standalone Backtest (`--backtest-only`)

Skips training and prediction. Runs a full Qlib backtest directly on existing stitched predictions.

```bash
python quantpits/scripts/rolling_train.py --backtest-only
```

---

### Daily Operations Cheat Sheet

```bash
# After weekly data update
python quantpits/scripts/rolling_train.py --merge --all-enabled       # fill new windows
python quantpits/scripts/rolling_train.py --predict-only --all-enabled  # predict

# After hyperparameter tuning
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158

# Data correction
python quantpits/scripts/rolling_train.py --retrain-last --all-enabled

# View status
python quantpits/scripts/rolling_train.py --show-state
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

Rolling training predictions seamlessly connect to downstream scripts via `--training-mode`:

```bash
# Brute force screening
python quantpits/scripts/brute_force_fast.py \
  --training-mode rolling

# Ensemble fusion
python quantpits/scripts/ensemble_fusion.py \
  --from-config --training-mode rolling
```

> [!TIP]
> The downstream workflow is identical for static and rolling training. Because they use a unified train records file, the only difference is appending `--training-mode rolling` to filter for rolling models. The default looks for static models (`@static`).

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
