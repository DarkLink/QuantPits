# 03 Ensemble Fusion Guide

## Overview

`scripts/ensemble_fusion.py` is used to conduct fusion prediction, backtesting, and risk analysis on **user-selected model combinations**.

**Multi-Combo Mode Supported**: Define multiple combos in `config/ensemble_config.json`, mark one as `default`, and run all combinations to compare performance simultaneously.

**Workflow Pipeline Placement**: Training → Brute Force → Combo Selection → **Fusion Backtesting (This Step)** → Order Generation

## Quick Start

```bash
cd QuantPits

# 1. Equal-weight fusion (directly specify models)
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# 2. Read the default combo from ensemble_config.json
python quantpits/scripts/ensemble_fusion.py --from-config

# 3. Run a specific named combo
python quantpits/scripts/ensemble_fusion.py --combo combo_A

# 4. Run all combos and generate comparisons
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

## Complete Parameter List

| Parameter | Default | Description |
|------|-------|------|
| `--models` | None | Comma-separated model name list (Directly specified, highest priority) |
| `--from-config` | false | Reads the `default` combo from `config/ensemble_config.json` |
| `--from-config-all` | false | Runs all combos and generates cross-combo comparisons |
| `--combo` | None | Runs a specifically named combo |
| `--method` | `equal` | Weighting mode: `equal` / `icir_weighted` / `manual` / `dynamic` |
| `--weights` | None | Manual weights string, e.g., `"gru:0.6,linear_Alpha158:0.4"` |
| `--freq` | `None` | Backtest frequency: `day` / `week` (Default: read from `model_config.json`) |
| `--record-file` | `latest_train_records.json` | Train records pointer |
| `--output-dir` | `output/ensemble` | Output directory bounds |
| `--no-backtest` | false | Skip backtesting execution |
| `--no-charts` | false | Skip chart generation |
| `--start-date` | None | Filter start date YYYY-MM-DD |
| `--end-date` | None | Filter end date YYYY-MM-DD |
| `--only-last-years N` | `0` | Use only the last N years of data (Designed exclusively for OOS testing) |
| `--only-last-months N` | `0` | Use only the last N months of data (Designed exclusively for OOS testing) |

## Multi-Combo Configurations

### Configuration Format (`config/ensemble_config.json`)

```json
{
  "combos": {
    "combo_A": {
      "models": ["gru", "linear_Alpha158", "TabNet_Alpha158", "mlp"],
      "method": "equal",
      "default": true,
      "description": "Original four-model equal weight ensemble"
    },
    "combo_B": {
      "models": ["gru", "linear_Alpha158", "alstm_Alpha158"],
      "method": "icir_weighted",
      "default": false,
      "description": "Three-model ICIR weighted"
    }
  },
  "min_model_ic": 0.00
}
```

**Key Notes**:
- `combos` dictionary, where each key represents a combo name.
- Each combo requires `models` and `method` fields.
- **Exactly one** combo must be flagged as `"default": true`.
- Script remains backward compatible with older flat formats (single `models` array + `ensemble_method`).

## Execution Modes

### Single Combo Mode

```bash
# Directly specify models (Bypasses config files)
python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method equal

# Read default combo from configuration
python quantpits/scripts/ensemble_fusion.py --from-config

# Run specific combo
python quantpits/scripts/ensemble_fusion.py --combo combo_B
```

### Multi Combo Mode

```bash
# Run all combos + generate inter-combo disparity comparison
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

### OOS (Out-Of-Sample) Verification Testing

If you utilized parameters like `--exclude-last-years 1` during the combo-seeking phase (via `brute_force_fast.py`) to fence off this year's data as OOS, you can leverage the following command to exclusively test pure forward OOS extrapolation performance just before taking the combo live:

```bash
# ========================================
# Execute performance tests exclusively bounding the recent 1 year OOS trajectory
# ========================================
python quantpits/scripts/ensemble_fusion.py --from-config --only-last-years 1
```

In this mode, generated net value metrics and attribution parameters will be **strictly bounded to only the final 1 year period**.

This mode will:
1. Load all combo-related prediction data synchronously once (shared pooling to prevent duplicated extraction costs).
2. Execute Stages 2-8 per combo sequentially (Correlation → Weighting → Fusion → Serialization → Backtest → Risk Analytics → Charting).
3. Produce tabular and visual cross-reference comparisons.

## Weighting Modes

### `equal` — Equal Weighting (Default)
Each model receives identical weighting. Simple and robust; serves as the baseline matrix.

### `icir_weighted` — ICIR Weighted
Distributes allocation coefficients strictly scaled to model ICIR metrics (Higher ICIR = Greater Weight).

### `manual` — Manual Interjection
Specified via `--weights` parameter string or via the `manual_weights` field inside combo configurations.

```bash
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158 \
  --method manual \
  --weights "gru:0.6,linear_Alpha158:0.4"
```

### `dynamic` — Dynamic Allocation
Leverages rolling 60-day window assessments targeting TopK position Sharpe Ratios to dynamically recalibrate distribution weights forward.

## Process Flow

```text
Stage 0: Initialize Qlib + Parse Configuration
Stage 1: Load selected predictions + Z-Score normalization (Shared across combos)
--- Iterated per combo ---
Stage 2: Correlation Analysis (Confined to combo models)
Stage 3: Compute Weights
Stage 4: Signal Fusion
Stage 5: Serialize Prediction Result Streams
Stage 6: Exhaustive Backtest (Muteable)
Stage 7: Risk Diagnostics + Leaderboards
Stage 8: Visual Rendering (Muteable)
--- Multi-Combo Addendum ---
Cross-combo Comparison Table + Merged Net Value Crossover Plot
```

## Output Artifacts

### Single Combo Mode (`--models` or `--from-config`)

```text
output/
├── predictions/
│   └── ensemble_{anchor_date}.csv            # Fused prediction vectors
└── ensemble/
    ├── ensemble_fusion_config_{date}.json     # Fused configuration state
    ├── correlation_matrix_{date}.csv          # Correlation matrix
    ├── leaderboard_{date}.csv                 # Performance leaderboards
    ├── ensemble_nav_{date}.png                # Net asset value trajectories
    └── ensemble_weights_{date}.png            # Dynamic weight mapping (dynamic mode)
```

### Multi Combo Mode (`--from-config-all` or `--combo`)

```text
output/
├── predictions/
│   ├── ensemble_combo_A_{date}.csv           # combo_A predictions
│   ├── ensemble_combo_B_{date}.csv           # combo_B predictions
│   └── ensemble_{date}.csv                   # default combo backward-compat shadow file
└── ensemble/
    ├── ensemble_fusion_config_combo_A_{date}.json
    ├── ensemble_fusion_config_combo_B_{date}.json
    ├── combo_comparison_{date}.csv           # Tabular cross-reference
    └── combo_comparison_{date}.png           # Comparative charted trajectories
```

> [!TIP]
> The `default` combo will redundantly output a nameless `ensemble_{date}.csv` artifact, guaranteeing absolute zero-modification compliance for downstream utilities like `order_gen.py`.

## Typical Operations Sequence

```bash
# Step 1: Train targeted algorithms
python quantpits/scripts/prod_train_predict.py

# Step 2: Brute force sweep uncovering optimized matrices
python quantpits/scripts/brute_force_ensemble.py --min-models 3 --max-models 6

# Step 3: Parse outputs, commit attractive combinations to config blocks
cat output/brute_force/leaderboard.csv
# Edit config/ensemble_config.json appropriately

# Step 4: Fire multi-combo validation
python quantpits/scripts/ensemble_fusion.py --from-config-all

# Step 5: Contrast outcomes, declare dominant default combo
cat output/ensemble/combo_comparison_*.csv

# Step 6: Dispatch orders reliant upon default combo matrix
python quantpits/scripts/order_gen.py
```

## Pipeline Architecture Mapping

| Script Scope | Intent | Input | Output |
|------|------|------|------|
| `prod_train_predict.py` | Train models | configs | `latest_train_records.json` |
| `brute_force_ensemble.py` | Combo Exhaustion | train records | leaderboards |
| **`ensemble_fusion.py`** | **Fusion Backtest** | **Targeted Combo sets** | **Fused Predictions + Risk Matrix** |
| `signal_ranking.py` | Top N Output | Fusion CSVs | Ranked CSV sets |
| `order_gen.py` | Target Execution | Fused CSVs + Current Pos | Buys/Sells + Multi-Model opinions |
