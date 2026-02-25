# QuantPits 

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Qlib](https://img.shields.io/badge/Tech_Stack-Qlib-brightgreen.svg)

An advanced, production-ready quantitative trading system built on top of [Microsoft Qlib](https://github.com/microsoft/qlib). This system provides a complete end-to-end pipeline for weekly and daily frequency trading, featuring modular architecture, multi-instance isolation (Workspaces), ensemble modeling, execution analytics, and interactive dashboards.

🌐 [中文版本 (README_zh.md)](./README_zh.md)

> **Note:** This repository is a read-only mirror of an internal monorepo. We welcome bug reports and issues, but please DO NOT submit Pull Requests at this time, as they cannot be merged into our internal system directly.

## 🚀 Key Features

* **Multi-Workspace Isolation**: Spin up independent "Pits" for different markets (e.g., CSI300, CSI500) or configurations without duplicating code.
* **Component-Based Pipeline**: 
  - **Train & Predict**: Support for both full and incremental training on multiple models (LSTM, GRU, Transformers, LightGBM, GATs).
  - **Brute Force & Ensemble**: High-performance (CuPy accelerated) brute force combination finding and intelligent signal fusion.
  - **Orders & Execution**: Generate actionable buy/sell signals with TopK/DropN logic and analyze micro-friction (slippage, delay costs).
* **Rich Observability**: Two interactive `streamlit` dashboards for macro portfolio performance and micro rolling health monitoring.
* **Resilient Infrastructure**: Automatic checkpoints, JSON tracking for model registries, and daily/weekly logs.

## 📂 Architecture Overview

The system strictly decouples the **Engine (Code)** from the **Workspace (Config & Data)**:

```text
QuantPits/
├── quantpits/                 # Core logic, scripts, and dashboards
│   ├── scripts/            # Pipeline execution scripts
│   ├── docs/               # Detailed system manuals (00-08)
│   ├── dashboard.py        # Macro performance streamlit app
│   └── rolling_dashboard.py# Temporal strategy health streamlit app
│
└── workspaces/             # Isolated trading instances
    └── Demo_Workspace/     # Example configured instance
        ├── config/         # Trading bounds, model registry, cashflow
        ├── data/           # Order logs, holding logs, daily amount
        ├── output/         # Model predictions, fusion results, reports
        ├── mlruns/         # MLflow tracking logs
        └── run_env.sh      # Environment activation script
```

## 🛠️ Quick Start

### 1. Requirements

Ensure you have a working installation of **Qlib**. The engine officially supports Python 3.8 to 3.12. Then install the extra dependencies:

```bash
pip install -r requirements.txt
# (Optional) Install the engine as a package for global access:
pip install -e .
```

*(Note: For GPU-accelerated brute force combinatorial backtesting, install `cupy-cuda11x` or `cupy-cuda12x` depending on your local CUDA version)*

### 2. Prepare Market Data

Before running the engine, ensure you have the required Qlib dataset downloaded to your local machine (e.g., `~/.qlib/qlib_data/cn_data`):

```bash
# Example: Download 1D data for the Chinese market
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
```
Make sure `workspaces/Demo_Workspace/run_env.sh` or your Qlib initialization points to this directory.

### 3. Activate a Workspace

Every action must be performed within the context of an active workspace. We provide a `Demo_Workspace` to get you started:

```bash
cd QuantPits/
source workspaces/Demo_Workspace/run_env.sh
```

### 4. Run the Pipeline

Once activated, you can execute the minimal routine loop using the quantpits scripts (or you can simply run `make run-daily-pipeline` from the repository root):

```bash
# 0. Update Daily Market Data
# Note: This engine assumes underlying Qlib data has been updated (e.g., via external Cron). 
# If not, update it first.

# 1. Generate predictions from existing models
python quantpits/scripts/prod_predict_only.py --all-enabled

# 2. Fuse predictions using your combo configs
python quantpits/scripts/ensemble_fusion.py --from-config-all

# 3. Process previous week's live trades (Post-Trade)
python quantpits/scripts/prod_post_trade.py

# 4. Generate new Buy/Sell orders based on current holdings
python quantpits/scripts/order_gen.py
```

### 5. Launch Dashboards

To view the interactive analytics of your active workspace:

```bash
# Portfolio Execution and Holding Dashboard
streamlit run quantpits/dashboard.py

# Rolling Strategy Health & Factor Drift Dashboard
streamlit run quantpits/rolling_dashboard.py
```

## 🏗️ Creating a New Workspace

To spin up a new strategy for a different index (e.g., CSI 500), use the scaffolding utility:

```bash
python quantpits/scripts/init_workspace.py \
  --source workspaces/Demo_Workspace \
  --target workspaces/CSI500_Base
```

This will cleanly clone the configuration files and generate empty `data/`, `output/`, and `mlruns/` directories, completely isolated from your other trading environments. You then simply `source workspaces/CSI500_Base/run_env.sh` to start operating in it.

## 📖 Documentation

For a deep dive into each module, refer to the documentation in `quantpits/docs/`:
- `00_SYSTEM_OVERVIEW.md` (System Architecture & Workflows)
- `01_TRAINING_GUIDE.md`
- `02_BRUTE_FORCE_GUIDE.md`
- `03_ENSEMBLE_FUSION_GUIDE.md`
- ...and more.

## 📜 License
MIT License.
