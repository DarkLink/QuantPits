# QuantPits End-to-End Walkthrough

This document is a **complete, hands-on walkthrough** for the QuantPits system, guiding you step-by-step through: environment setup → data preparation → model training → combination optimization → ensemble fusion → order generation → live-trading loop.

> [!TIP]
> This guide is designed for **first-time users**. All commands can be copied and pasted directly. For detailed parameter references and internals, please refer to the corresponding module documentation (01-08).

---

## Table of Contents

1. [Install Qlib & Dependencies](#1-install-qlib--dependencies)
2. [Prepare Market Data](#2-prepare-market-data)
3. [Initialize a Workspace](#3-initialize-a-workspace)
4. [Train Models](#4-train-models)
5. [Predict Only (Without Retraining)](#5-predict-only-without-retraining)
6. [Brute-Force Combination Search](#6-brute-force-combination-search)
7. [Ensemble Fusion](#7-ensemble-fusion)
8. [Generate Signal Ranking (Optional)](#8-generate-signal-ranking-optional)
9. [Process Live-Trade Data (Post-Trade)](#9-process-live-trade-data-post-trade)
10. [Generate Trade Orders (Order Gen)](#10-generate-trade-orders-order-gen)
11. [Live-Trading Analysis](#11-live-trading-analysis)

---

## 1. Install Qlib & Dependencies

QuantPits is built on top of [Microsoft Qlib](https://github.com/microsoft/qlib). You need to install Qlib and the system's dependencies first.

### 1.1 Install Qlib

```bash
pip install pyqlib
```

> For detailed installation instructions, see: https://github.com/microsoft/qlib#installation

### 1.2 Install QuantPits Dependencies

```bash
cd QuantPits
pip install -r requirements.txt
```

### 1.3 (Optional) Install as a Python Package

```bash
pip install -e .
```

### 1.4 (Optional) Install CuPy for GPU Acceleration

If you want GPU-accelerated brute-force combination search (`brute_force_fast.py`), install CuPy for your CUDA version:

```bash
# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x
```

---

## 2. Prepare Market Data

All scripts read Qlib daily-frequency market data from `~/.qlib/qlib_data/cn_data` by default.

### Option A: Qlib Official Data

```bash
python -m qlib.run.get_data qlib_data \
  --target_dir ~/.qlib/qlib_data/cn_data \
  --region cn \
  --version v2
```

> ⚠️ The initial download requires tens of GBs of disk space and may take a considerable amount of time.

### Option B: Third-Party Data Source (Recommended)

The [investment_data](https://github.com/chenditc/investment_data) project continuously publishes up-to-date Qlib-formatted market data on its GitHub Releases page:

1. Go to https://github.com/chenditc/investment_data/releases
2. Download the latest `qlib_bin.tar.gz`
3. Extract it to the Qlib data directory:

```bash
mkdir -p ~/.qlib/qlib_data/cn_data
tar -xzf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data
```

> See https://github.com/chenditc/investment_data for update frequency and data details.

### Custom Data Path

If your data is stored elsewhere, configure it in your workspace's `run_env.sh`:

```bash
export QLIB_DATA_DIR="/your/custom/path/cn_data"
export QLIB_REGION="cn"
```

---

## 3. Initialize a Workspace

QuantPits uses a **Workspace** mechanism to fully isolate configurations and data between instances. A `Demo_Workspace` template is included.

### 3.1 Quick Start with the Demo Workspace

```bash
cd QuantPits
source workspaces/Demo_Workspace/run_env.sh
```

Output: `Workspace activated: .../Demo_Workspace` confirms successful activation.

### 3.2 Create a New Workspace

```bash
python quantpits/scripts/init_workspace.py \
  --source workspaces/Demo_Workspace \
  --target workspaces/MyWorkspace
```

This will:
- Clone all configuration files from `Demo_Workspace/config/`
- Create empty `data/`, `output/`, `archive/`, and `mlruns/` directories
- Generate a `run_env.sh` environment activation script

### 3.3 Activate the New Workspace

```bash
source workspaces/MyWorkspace/run_env.sh
```

### 3.4 Workspace Directory Structure

```text
workspaces/MyWorkspace/
├── config/
│   ├── model_registry.yaml         # Model registry (which models to use)
│   ├── model_config.json           # Training config (data slicing, frequency)
│   ├── strategy_config.yaml        # Strategy config (backtest/order gen params)
│   ├── prod_config.json            # Production config (holdings, cash, live params)
│   ├── cashflow.json               # Cashflow (deposit/withdrawal records)
│   ├── ensemble_config.json        # Ensemble combination config
│   └── workflow_config_*.yaml      # Qlib model workflow configs
├── data/                           # Runtime data
├── output/                         # Output files
├── archive/                        # Historical archives
├── mlruns/                         # MLflow tracking
└── run_env.sh                      # Environment activation script
```

---

## 4. Train Models

> [!IMPORTANT]
> **Prerequisite**: Ensure a workspace is activated (`source workspaces/YourWorkspace/run_env.sh`). All commands below are run from the QuantPits root directory.

### 4.1 Prepare Model Workflow YAML

Each model requires a Qlib workflow configuration file defining the model class, dataset handler, and training recipe.

- YAML templates: https://github.com/microsoft/qlib/tree/main/examples/benchmarks
- Place them at: `config/workflow_config_<model_name>.yaml`

The Demo Workspace includes `workflow_config_demo_weekly.yaml` (weekly LinearModel + Alpha158).

### 4.2 Prepare Configuration Files

Core configuration files in `config/`:

| File | Purpose | Key Fields |
|------|---------|------------|
| `model_registry.yaml` | Model registry — defines which models to enable | algorithm, dataset, yaml_file, enabled, tags |
| `model_config.json` | Training parameters — data slicing and frequency | train/valid/test windows, slide/fixed mode, `freq: "week"` |
| `strategy_config.yaml` | Strategy parameters — TopK/DropN and backtest env | `topk: 20`, `n_drop: 3`, commission rates |
| `prod_config.json` | Production state — current holdings and cash | initial_cash, current_holding, current_cash |
| `cashflow.json` | Cashflow records — deposits and withdrawals by date | `{"cashflows": {"2026-01-15": 50000}}` |

### 4.3 Run Training

#### Full Training (Required for First Run)

Trains all `enabled: true` models in `model_registry.yaml`:

```bash
python quantpits/scripts/prod_train_predict.py
```

#### Incremental Training (On Demand)

Train only specific models, preserving existing records:

```bash
# By name
python quantpits/scripts/incremental_train.py --models demo_linear_Alpha158

# By tag
python quantpits/scripts/incremental_train.py --tag tree

# All enabled models (merge mode, non-destructive)
python quantpits/scripts/incremental_train.py --all-enabled

# Dry-run (preview only)
python quantpits/scripts/incremental_train.py --models demo_linear_Alpha158 --dry-run
```

Key output files:
- `latest_train_records.json` — Model training records (input for all downstream scripts)
- `output/predictions/<model>_<date>.csv` — Per-model prediction results
- `output/model_performance_<date>.json` — Model IC/ICIR metrics

> See [01_TRAINING_GUIDE.md](01_TRAINING_GUIDE.md) for details.

---

## 5. Predict Only (Without Retraining)

Once you have trained models, use them to generate new predictions on updated data without retraining:

```bash
# Predict with all enabled models
python quantpits/scripts/prod_predict_only.py --all-enabled

# Predict with specific models
python quantpits/scripts/prod_predict_only.py --models demo_linear_Alpha158

# Dry-run (preview only)
python quantpits/scripts/prod_predict_only.py --all-enabled --dry-run
```

Predictions are saved to `output/predictions/`, and `latest_train_records.json` is updated via merge semantics.

> See [05_PREDICT_ONLY_GUIDE.md](05_PREDICT_ONLY_GUIDE.md) for details.

---

## 6. Brute-Force Combination Search

When you have ≥ 2 models, brute-force enumeration helps you find the optimal model combination.

### 6.1 Prepare Combination Group File (Optional)

For a large number of models (15+), use grouped enumeration to dramatically reduce the search space. Create `config/combo_groups.yaml`:

```yaml
groups:
  LSTM_variants:
    - lstm_Alpha360
    - alstm_Alpha158
  Tree_models:
    - lightgbm_Alpha158
    - catboost_Alpha158
```

### 6.2 Run Brute-Force Search

#### Fast Mode (Vectorized, seconds-level, best for initial screening)

```bash
# Fast search (up to 3-model combos)
python quantpits/scripts/brute_force_fast.py --max-combo-size 3

# Full enumeration
python quantpits/scripts/brute_force_fast.py

# With OOS anti-overfitting validation (Recommended!)
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 10

# With group-based enumeration
python quantpits/scripts/brute_force_fast.py --use-groups

# GPU acceleration
python quantpits/scripts/brute_force_fast.py --use-gpu
```

> ⚠️ **Fast mode has limited precision** (no limit-up/down filtering, no exact position management). Best used for initial screening only.

#### Accurate Mode (Full Qlib Backtest, minutes-level)

```bash
# Accurately backtest top candidates (use after fast screening)
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

# Resume from interruption
python quantpits/scripts/brute_force_ensemble.py --resume
```

#### Recommended Workflow

1. **Coarse screening**: Run `brute_force_fast.py` across all combinations (seconds/minutes)
2. **Precise validation**: Validate Top 10-20 candidates with `brute_force_ensemble.py`
3. **Confirm**: Review `output/brute_force*/analysis_report_*.txt` and select optimal combinations

> See [02_BRUTE_FORCE_GUIDE.md](02_BRUTE_FORCE_GUIDE.md) for details.

---

## 7. Ensemble Fusion

Use selected model combinations to generate fused predictions and run backtests.

### 7.1 Prepare Ensemble Combination Config

Edit `config/ensemble_config.json` with your selected model combinations (multiple combos supported):

```json
{
  "combos": {
    "combo_A": {
      "models": ["gru", "linear_Alpha158", "TabNet_Alpha158"],
      "method": "equal",
      "default": true,
      "description": "3-model equal-weight combo"
    },
    "combo_B": {
      "models": ["gru", "linear_Alpha158", "alstm_Alpha158"],
      "method": "icir_weighted",
      "default": false,
      "description": "3-model ICIR-weighted combo"
    }
  },
  "min_model_ic": 0.00
}
```

### 7.2 Run Fusion

```bash
# Run all combos and generate cross-combo comparison (most common)
python quantpits/scripts/ensemble_fusion.py --from-config-all

# Run only the default combo
python quantpits/scripts/ensemble_fusion.py --from-config

# Specify models directly (without config file)
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# OOS validation (test on last 1 year only)
python quantpits/scripts/ensemble_fusion.py --from-config --only-last-years 1
```

Output files:
- `output/predictions/ensemble_<combo>_<date>.csv` — Fused predictions
- `output/ensemble/combo_comparison_<date>.csv` — Cross-combo comparison table
- `output/ensemble/combo_comparison_<date>.png` — NAV comparison chart

> See [03_ENSEMBLE_FUSION_GUIDE.md](03_ENSEMBLE_FUSION_GUIDE.md) for details.

---

## 8. Generate Signal Ranking (Optional)

Normalize fused prediction scores to a -100 ~ +100 recommendation index, suitable for sharing:

```bash
# Generate Top 300 ranking for all combos
python quantpits/scripts/signal_ranking.py --all-combos

# Default combo only
python quantpits/scripts/signal_ranking.py

# Customize Top N
python quantpits/scripts/signal_ranking.py --top-n 500
```

Output is saved to `output/ranking/Signal_<combo>_<date>_Top300.csv`.

> See [07_SIGNAL_RANKING_GUIDE.md](07_SIGNAL_RANKING_GUIDE.md) for details.

---

## 9. Process Live-Trade Data (Post-Trade)

> Steps 9-11 are only required when you have live-trading positions.

The Post-Trade script processes settlement files exported from your broker, updating holdings and cash. **Fully decoupled from training/prediction/fusion modules.**

### 9.1 Prepare Trade Data

1. Export settlement records (`.xlsx`) from your broker and place them in `data/`
2. File naming convention: `YYYY-MM-DD-table.xlsx` (e.g., `2026-02-24-table.xlsx`)
3. For non-trading days, use the empty template `emp-table.xlsx`

### 9.2 Configure Cashflow (If Applicable)

Edit `config/cashflow.json`:

```json
{
  "cashflows": {
    "2026-02-03": 50000,
    "2026-02-06": -20000
  }
}
```

### 9.3 Run Post-Trade

```bash
# Preview the processing plan (strongly recommended first)
python quantpits/scripts/prod_post_trade.py --dry-run

# Execute
python quantpits/scripts/prod_post_trade.py

# Specify broker (default: Guotai Junan)
python quantpits/scripts/prod_post_trade.py --broker gtja

# Verbose output
python quantpits/scripts/prod_post_trade.py --verbose
```

Files updated:
- `config/prod_config.json` — Latest holdings and cash balance
- `data/trade_log_full.csv` — Cumulative trade log
- `data/holding_log_full.csv` — Cumulative holding log
- `data/daily_amount_log_full.csv` — Daily capital summary

> See [04_POST_TRADE_GUIDE.md](04_POST_TRADE_GUIDE.md) for details.

---

## 10. Generate Trade Orders (Order Gen)

Generate buy/sell order suggestions based on fused predictions and current holdings.

```bash
# Use ensemble fusion predictions (most common)
python quantpits/scripts/order_gen.py

# Preview with multi-model judgment table
python quantpits/scripts/order_gen.py --dry-run --verbose

# Use a single model's prediction
python quantpits/scripts/order_gen.py --model gru

# Generate multi-model ranking visualization
python quantpits/scripts/plot_model_opinions.py
```

Output files:
- `output/buy_suggestion_<source>_<date>.csv` — Buy suggestions
- `output/sell_suggestion_<source>_<date>.csv` — Sell suggestions
- `output/model_opinions_<date>.csv` — Multi-model BUY/SELL/HOLD judgment table

> See [06_ORDER_GEN_GUIDE.md](06_ORDER_GEN_GUIDE.md) for details.

---

## 11. Live-Trading Analysis

### 11.1 Comprehensive Analysis Report

Perform a full audit covering model quality, ensemble correlation, execution slippage, and portfolio risk:

```bash
python quantpits/scripts/run_analysis.py \
  --models gru_Alpha158 transformer_Alpha360 TabNet_Alpha158 sfm_Alpha360 \
  --output output/analysis_report.md
```

### 11.2 Interactive Dashboards

```bash
# Macro portfolio performance dashboard
streamlit run ui/dashboard.py

# Rolling strategy health monitoring dashboard
streamlit run ui/rolling_dashboard.py --server.port 8503
```

### 11.3 Automated Rolling Health Report

```bash
# Generate rolling window metrics
python quantpits/scripts/run_rolling_analysis.py --windows 20 60

# Generate automated health report
python quantpits/scripts/run_rolling_health_report.py
```

> See [08_ANALYSIS_GUIDE.md](08_ANALYSIS_GUIDE.md) for details.

---

## Appendix: Typical Scenario Quick Reference

### Scenario A: First-Time Complete Run

```bash
cd QuantPits
source workspaces/Demo_Workspace/run_env.sh

# ① Full training
python quantpits/scripts/prod_train_predict.py

# ② Quick brute-force search
python quantpits/scripts/brute_force_fast.py --max-combo-size 3

# ③ Ensemble fusion
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

### Scenario B: Daily Minimum Loop (No Retraining)

```bash
cd QuantPits
source workspaces/MyWorkspace/run_env.sh

# ① Predict
python quantpits/scripts/prod_predict_only.py --all-enabled

# ② Fuse
python quantpits/scripts/ensemble_fusion.py --from-config-all

# ③ Post-Trade (if live trading)
python quantpits/scripts/prod_post_trade.py

# ④ Generate orders
python quantpits/scripts/order_gen.py
```

> Or use the Makefile for a one-command daily pipeline (predict → fuse → post-trade → orders):
> ```bash
> make run-daily-pipeline
> ```

### Scenario C: Re-evaluate Model Combinations

```bash
# ① Predict with existing models
python quantpits/scripts/prod_predict_only.py --all-enabled

# ② Fast brute-force search (with OOS validation)
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 10

# ③ Accurately validate top candidates
python quantpits/scripts/brute_force_ensemble.py --min-combo-size 3 --max-combo-size 3

# ④ Update ensemble_config.json, then re-fuse
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

### Scenario D: Model Performance Degradation — Retrain

```bash
# ① Incrementally retrain underperforming models
python quantpits/scripts/incremental_train.py --models gru,alstm_Alpha158

# ② Re-fuse with updated models
python quantpits/scripts/ensemble_fusion.py --from-config-all

# ③ Post-Trade + Order generation
python quantpits/scripts/prod_post_trade.py
python quantpits/scripts/order_gen.py
```

---

## Appendix: Key Reference Resources

| Resource | Link |
|----------|------|
| Qlib Official Repository | https://github.com/microsoft/qlib |
| Qlib Model Benchmark YAMLs | https://github.com/microsoft/qlib/tree/main/examples/benchmarks |
| Market Data Source (investment_data) | https://github.com/chenditc/investment_data |
| QuantPits System Overview | [00_SYSTEM_OVERVIEW.md](00_SYSTEM_OVERVIEW.md) |
| QuantPits Module Docs (01-08) | `docs/en/` directory |
