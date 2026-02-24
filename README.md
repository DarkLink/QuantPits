# QuantPits 

An advanced, production-ready quantitative trading system built on top of [Microsoft Qlib](https://github.com/microsoft/qlib). This system provides a complete end-to-end pipeline for weekly and daily frequency trading, featuring modular architecture, multi-instance isolation (Workspaces), ensemble modeling, execution analytics, and interactive dashboards.

🌐 [中文版本 (README_zh.md)](README_zh.md)

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
├── engine/                 # Core logic, scripts, and dashboards
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
        └── run_env.sh      # Environment activation script
```

## 🛠️ Quick Start

### 1. Requirements

Ensure you have a working installation of **Qlib**. Then install the extra dependencies:

```bash
pip install -r requirements.txt
```

*(Note: For GPU-accelerated brute force combinatorial backtesting, install `cupy-cuda12x`)*

### 2. Activate a Workspace

Every action must be performed within the context of an active workspace. We provide a `Demo_Workspace` to get you started:

```bash
cd QuantPits/
source workspaces/Demo_Workspace/run_env.sh
```

### 3. Run the Pipeline

Once activated, you can execute the minimal weekly loop using the engine scripts:

```bash
# 1. Generate predictions from existing models
python engine/scripts/weekly_predict_only.py --all-enabled

# 2. Fuse predictions using your combo configs
python engine/scripts/ensemble_fusion.py --from-config-all

# 3. Process previous week's live trades (Post-Trade)
python engine/scripts/weekly_post_trade.py

# 4. Generate new Buy/Sell orders based on current holdings
python engine/scripts/order_gen.py
```

### 4. Launch Dashboards

To view the interactive analytics of your active workspace:

```bash
# Portfolio Execution and Holding Dashboard
streamlit run engine/dashboard.py

# Rolling Strategy Health & Factor Drift Dashboard
streamlit run engine/rolling_dashboard.py
```

## 🏗️ Creating a New Workspace

To spin up a new strategy for a different index (e.g., CSI 500), use the scaffolding utility:

```bash
python engine/scripts/init_workspace.py \
  --source workspaces/Demo_Workspace \
  --target workspaces/CSI500_Base
```

This will cleanly clone the configuration files and generate empty `data/`, `output/`, and `mlruns/` directories, completely isolated from your other trading environments. You then simply `source workspaces/CSI500_Base/run_env.sh` to start operating in it.

## 📖 Documentation

For a deep dive into each module, refer to the documentation in `engine/docs/`:
- `00_SYSTEM_OVERVIEW.md` (System Architecture & Workflows)
- `01_TRAINING_GUIDE.md`
- `02_BRUTE_FORCE_GUIDE.md`
- `03_ENSEMBLE_FUSION_GUIDE.md`
- ...and more.

## 📜 License
MIT License.
