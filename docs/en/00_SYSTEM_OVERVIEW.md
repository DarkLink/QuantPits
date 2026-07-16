# QuantPits Quantitative Trading System Overview

## System Positioning

A **multi-frequency quantitative trading production system** (supporting weekly/daily) based on Qlib, adopting a "multi-model training → ensemble selection → fusion prediction → order generation" pipeline architecture.

---

## Workspace Architecture

The system strictly separates **Engine (Code)** from **Workspace (Data & Config)** to enable multi-instance parallelism:
- **Engine (`quantpits/`)**: Stores all globally shared core scripts (`scripts/`), analysis dashboards (`dashboard.py`), and system documentation (`docs/`).
- **Workspace (Directories under `workspaces/`)**: Any independent collection for live trading or simulation. Contains the instance-specific `config/`, `data/`, `output/`, and isolated MLflow data (new workspaces default to `mlflow.db`; legacy workspaces with existing data remain compatible with `mlruns/`).

**How to use workspaces?**
1. Create a new workspace: Use `python quantpits/tools/init_workspace.py --source workspaces/A --target workspaces/B` for quick scaffolding.
2. Activate a workspace: Enter the system directory and run `source workspaces/<Your_Workspace>/run_env.sh` to set `QLIB_WORKSPACE_DIR`.
3. Validate configs before running: `python -m quantpits.tools.validate_workspace --workspace workspaces/Demo_Workspace --read-only` performs a read-only check of core workspace configs and reports normalized fingerprints plus warnings/errors.
   Use `python -m quantpits.tools.audit_mlflow_workspace --workspace workspaces/Demo_Workspace` for a read-only lineage audit. A missing backend reports `tracking_backend_missing` without initializing a database. Only a safeguarded real ensemble execution may create one workspace-contained target experiment after Qlib initialization; plans and audits never create it.
4. (Optional) Custom data source: Uncomment `QLIB_DATA_DIR` / `QLIB_REGION` in `run_env.sh` to point the workspace at a different Qlib data directory (defaults to `~/.qlib/qlib_data/cn_data` and `cn`).
5. Execute scripts: Scripts will automatically route all file I/O into the currently activated workspace, and `env.init_qlib()` will initialize Qlib with the configured data path.
   > [!IMPORTANT]
   > **Initialization order**: Core scripts should import `env` before resolving workspace-scoped paths. This ensures `ROOT_DIR` is recognized, `MLFLOW_TRACKING_URI` points to the active workspace's MLflow backend (SQLite `mlflow.db` by default; legacy `mlruns/` file stores are auto-detected), and `env.init_qlib()` binds both the Qlib data provider and experiment manager to that workspace. To customize the backend, set `MLFLOW_TRACKING_URI` in `run_env.sh`.
   >
   > **Runtime context**: New code should prefer `env.get_workspace_context()` to obtain an explicit `WorkspaceContext` instead of copying `env.ROOT_DIR`, deriving import-time path constants, or changing the process `cwd` with `os.chdir()`. Legacy APIs remain compatible and production scripts will migrate gradually; `ensemble_fusion.py` has already removed its import-time cwd side effect.

---

## Core Design Philosophy

Production account valuation and model-feature/backtest prices have separate purpose boundaries. Post-trade records explicit `account_valuation` provenance and keeps market date, observation time, and account effective date distinct. Broker asset snapshots are read-only reconciliation evidence and do not change settlement/order/trade authority.

Every step in this system is **optional and combinable**. Only "Prediction" and "Order Generation" are mandatory routine steps, while others are triggered as needed:

| Step | Required Every Time? | Description |
|------|:---:|------|
| ① Training | ⚠️ At least once | Mandatory for the first time; subsequently only when models need refreshing |
| ② Prediction | ✅ Every time | Can be produced along with training, or run independently with existing models |
| ③ Ensemble Search | ❌ Occasionally | Exhaustive combinatorial search + multi-dimensional analysis; selected combos can be used for long periods |
| ④ Fusion Prediction | ✅ Every time | Supports multi-combo fusion and comparison; generates final fusion signals using selected combos |
| ⑤ Post-Trade | ✅ Every time | Processes last period's live data to update holdings and cash |
| ⑥ Order Generation | ✅ Every time | Generates buy/sell suggestions and multi-model opinions based on fused predictions and current holdings |
| ⑦ Signal Ranking | Optional | Normalizes scores into Top N rankings, suitable for sharing |
| ⑧ Deep Analysis | Recommended weekly | Multi-agent automated analysis + OOM-RL feedback (from analysis to auto-execution) |
| ⑨ Feedback Loop | On-demand | Execute LLM Critic ActionItems, validate in Playground, promote to production |

> [!IMPORTANT]
> Training is not required every time, but prediction is mandatory. **Ensemble Search** is a "heavy-duty" operation, and selected model combos can be used for a long time. The minimal routine operation loop is: **Prediction → Fusion → Post-Trade → Order Generation**. Run Deep Analysis weekly to monitor system health and get optimization recommendations.

---

## System Architecture

Post-trade uses one intake control plane while preserving three authority boundaries: settlement owns cash/holding state, orders capture execution intent, and trades capture intraday fills. The default `prod_post_trade --scope all` processes all three so execution analytics cannot silently miss order/fill logs. A source-fingerprint ledger discovers late historical exports and fails closed on content drift.

Account-state commits use staged bytes, baseline/target fingerprints, and a durable local recovery journal. Cashflows are consumed only for processed dates and the production cursor advances last. Execution ingestion remains a separate commit domain, while execution/state/classification share one RunManifest and OperatorLog audit lifecycle.

Account state is calculated as a complete batch by a pure `Decimal` engine. Filled quantities and valuation completeness are checked before derived logs and `prod_config.json` are committed in cursor-last order. `--dry-run` performs the same calculation without writes; this is a recoverable protocol, not a multi-file atomic transaction.

```mermaid
flowchart TB
    subgraph TRAIN["① Train (On-demand)"]
        TP["static_train.py --full<br/>Full Train"]
        IT["static_train.py<br/>Incremental Train"]
    end

    subgraph ROLLING["⑨ Rolling Train (On-demand)"]
        RT["rolling_train.py<br/>Cold Start/Daily/Predict"]
    end

    subgraph PREDICT["② Predict (Every time)"]
        PO["static_train.py --predict-only<br/>Predict Only"]
    end

    subgraph SEARCH["③ Ensemble Search (Occasionally)"]
        BF["brute_force_ensemble.py<br/>Exact Backtest"]
        BFF["brute_force_fast.py<br/>Fast Screening"]
        AE["analyze_ensembles.py<br/>Multi-dim Analysis + OOS Validation"]
    end

    EC["ensemble_config.json<br/>Combo Config (Manual Edit)"]

    subgraph FUSION["④ Fusion Predict (Every time/Multi-combo)"]
        EF["ensemble_fusion.py<br/>--from-config-all / --combo"]
    end

    subgraph POSTTRADE["⑤ Post-Trade (Every time)"]
        PT["prod_post_trade.py<br/>Process Live Trades"]
    end

    subgraph ORDERGEN["⑥ Order Generation (Every time)"]
        OG["order_gen.py<br/>Generate Orders + Opinions"]
    end

    subgraph SIGNAL["⑦ Signal Ranking (Optional/Sharing)"]
        SR["signal_ranking.py<br/>Top N Recommendation Index"]
    end

    subgraph ANALYSIS["⑧ Comprehensive Analysis (Monitoring)"]
        AN["run_analysis.py<br/>Full Review of Models/Fusions/Hazards"]
        DA["run_deep_analysis.py<br/>Multi-Agent Deep Analysis + OOM-RL"]
        FB["run_feedback_loop.py<br/>ActionItem Execution + Promotion"]
    end

    REG["model_registry.yaml<br/>Model Registry"]
    LTR["latest_train_records.json<br/>Train Records"]
    LRR["latest_train_records.json<br/>Unified Train Records<br/>(incl. @rolling)"]
    PRED_REC["output/predictions/<br/>Prediction Recorders"]
    ER["ensemble_records.json<br/>Fusion Record Pointer"]
    WC["prod_config.json<br/>Holdings/Cash"]
    AI["action_items_{date}.json<br/>OOM-RL ActionItems"]

    REG --> TRAIN
    REG --> PREDICT
    REG --> ROLLING
    TRAIN --> LTR
    PREDICT --> LTR
    ROLLING --> LRR
    LTR --> SEARCH
    LTR --> FUSION
    LRR -.->|--training-mode rolling| SEARCH
    LRR -.->|--training-mode rolling| FUSION
    BFF --> AE
    BF --> AE
    AE -.->|Recommended Combos → Manual Edit| EC
    EC --> EF
    PREDICT --> PRED_REC
    PRED_REC -.->|--model Single Model| ORDERGEN
    PRED_REC --> AN
    EF --> ER
    ER --> ORDERGEN
    ER --> SR
    PT --> WC
    WC --> ORDERGEN
    WC -.->|daily_amount| AN
    LTR --> DA
    WC -.->|daily_amount| DA
    DA --> AI
    AI --> FB
    FB -.->|promote config| REG
```

---

## Typical Execution Scenarios

⚠️ **All scenarios below must be executed in the system root directory (`QuantPits`) with the corresponding workspace activated beforehand.**
```bash
cd QuantPits
source workspaces/Demo_Workspace/run_env.sh
```

### Scenario A: Full Production Routine (Including Training)

Suitable for the first run or when models need to be refreshed.

```bash
# ① Full Training (Outputs predictions + training records)
python quantpits/scripts/static_train.py --full

# ③ Ensemble Search (Optional, time-consuming)
python quantpits/scripts/brute_force_fast.py --max-combo-size 3

# ④ Fusion Prediction (Using chosen combo)
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# ⑤ Process last period's live trades
python quantpits/scripts/prod_post_trade.py

# ⑥ Generate this period's orders
python quantpits/scripts/order_gen.py
```

### Scenario B: Regular Routine (No Retraining, Minimal Loop)

Existing models are available; directly predict after data updates. **This is the most common production workflow.**

```bash
# ② Predict new data using existing models
python quantpits/scripts/static_train.py --predict-only --all-enabled

# ④ Fusion Prediction (Multi-combo comparison)
python quantpits/scripts/ensemble_fusion.py --from-config-all

# ⑤ Process last period's live trades
python quantpits/scripts/prod_post_trade.py

# ⑥ Generate this period's orders
python quantpits/scripts/order_gen.py

# ⑦ Generate Signal Rankings (Optional, for sharing)
python quantpits/scripts/signal_ranking.py --all-combos

# ⑧ Run comprehensive system health analysis (Optional, for monitoring)
python quantpits/scripts/run_analysis.py \
  --models gru_Alpha158 transformer_Alpha360 TabNet_Alpha158 sfm_Alpha360
```

### Scenario C: Poor Fusion Performance, Fallback Retraining

When fusion prediction degrades, retrain specific models and re-evaluate.

```bash
# ① Incrementally train specific models
python quantpits/scripts/static_train.py --models gru,alstm_Alpha158

# ④ Refuse with the new models
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# ⑤⑥ Post-Trade + Order Gen (Same as Scenario B)
python quantpits/scripts/prod_post_trade.py
python quantpits/scripts/order_gen.py
```

### Scenario D: Re-evaluate Model Combinations

Re-run brute force periodically or when current combos are unsuitable.

```bash
# ② Predict using existing models (or automatically outputted after training)
python quantpits/scripts/static_train.py --predict-only --all-enabled

# ③ Fast Ensemble Search Screening
python quantpits/scripts/brute_force_fast.py --max-combo-size 5

# ③ Exact backtesting on Top candidates for confirmation
python quantpits/scripts/brute_force_ensemble.py \
  --min-combo-size 3 --max-combo-size 3

# ④ Fuse using the newly selected combo
python quantpits/scripts/ensemble_fusion.py \
  --models <new_selected_combo>

# ⑤⑥ Post-Trade + Order Gen (As above)
```

### Scenario E: Rolling Training (Adapting to Market Regime Shifts)

When static model predictions degrade due to market regime changes, use rolling training to keep models continuously adapted.

```bash
# ⑨ Cold start: first-time rolling training
python quantpits/scripts/rolling_train.py --cold-start --all-enabled

# ③④ Brute force + Fusion (using rolling predictions)
python quantpits/scripts/brute_force_fast.py --training-mode rolling
python quantpits/scripts/ensemble_fusion.py \
  --from-config --training-mode rolling

# ⑤⑥ Post-Trade + Order Gen (Same as other scenarios)
```

> For daily operation, simply run: `python quantpits/scripts/rolling_train.py --all-enabled` (auto-detects whether to train or predict)

### Scenario F: OOM-RL Closed-Loop Feedback (Deep Analysis → Auto-Execute)

For scenarios where weekly analysis reveals models needing hyperparameter tuning.

```bash
# ⑧ Deep Analysis + LLM Critic → produce ActionItems
python -m quantpits.scripts.run_deep_analysis --critic

# ⑨ Preview what the Feedback Loop will do (with priority sorting)
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --report-only

# ⑨ Execute in Playground sandbox (high-priority models only)
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --execute --max-duration-minutes 30

# ⑨ Promote validated changes to production
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --promote

# ① Next training cycle picks up the new config automatically
python -m quantpits.scripts.static_train --all-enabled
```

---

## Module Cheat Sheet

### ① Training Module

> See details in [01_TRAINING_GUIDE.md](01_TRAINING_GUIDE.md)

| Script | Purpose | Save Semantics |
|------|------|----------|
| `static_train.py --full` | Full training of all enabled models | **Full Overwrite** of `latest_train_records.json` |
| `static_train.py` | Selective training by name/algo/tags | **Incremental Merge** to `latest_train_records.json` |

Training Record V2 stores authoritative experiment, recorder, operation, prediction coverage, and
source lineage per `model@mode` under `model_records`. The legacy `models` map and top-level fields
remain compatibility projections.

- Models are uniformly managed in `config/model_registry.yaml`
- Date parameters are controlled by `config/model_config.json`
- Backtest strategy parameters (benchmark, frequency, capital) are controlled by `config/strategy_config.yaml` (preview)
- Typed static/CPCV stores exact pre/postimages and receipts under `data/training_transactions/`; legacy/rolling backups may remain in `data/history/`
- Incremental training supports `--resume` (resume from breakpoint) and `--dry-run` (preview)

### ⑨ Rolling Training Module

> See details in [30_ROLLING_TRAINING_GUIDE.md](30_ROLLING_TRAINING_GUIDE.md)

| Script | Purpose | Save Semantics |
|------|------|----------|
| `rolling_train.py` | Sliding window training + prediction stitching | **Unified** `latest_train_records.json` |

- Coexists with static training in the same record file, stored under `model@rolling` keys
- Supports cold start, daily mode (auto-detect train vs predict), predict-only, crash recovery
- Downstream scripts switch data source via `--training-mode rolling` parameter
- Config: `config/rolling_config.yaml` (start date, train years, valid years, step size)

### ② Prediction Module

> See details in [05_PREDICT_ONLY_GUIDE.md](05_PREDICT_ONLY_GUIDE.md)

| Script | Purpose |
|------|------|
| `static_train.py --predict-only` | Predict on new data using existing models, no retraining |

- Loads models from `latest_train_records.json` and creates a new Recorder to save predictions
- Supports the same model selection methods as training scripts (by name/algo/tags)
- **Incremental Merge** to update `latest_train_records.json`, making downstream script switching seamless

### ③ Ensemble Search Module

> See details in [02_BRUTE_FORCE_GUIDE.md](02_BRUTE_FORCE_GUIDE.md)

| Script | Purpose | Speed |
|------|------|------|
| `brute_force_fast.py` | Vectorized fast screening | ~0.001s/combo |
| `brute_force_ensemble.py` | Full Qlib Backtesting | ~5s/combo |
| `analyze_ensembles.py` | Multi-dimensional IS Analysis + OOS Blind Validation | As needed |

- **Recommended Workflow**: Fast screening → `analyze_ensembles.py` for multi-dimensional analysis & OOS validation → Configure selected combos in `ensemble_config.json`.
- Outputs model attribution reports, risk-return scatter plots, hierarchical clustering charts, etc.
- Results are neatly organized per-run under `output/ensemble_runs/{script}_{date}/`, stratified into IS and OOS layers.
- Supports `--resume` for crash recovery.

### ④ Fusion Prediction Module

> See details in [03_ENSEMBLE_FUSION_GUIDE.md](03_ENSEMBLE_FUSION_GUIDE.md)

| Script | Purpose |
|------|------|
| `ensemble_fusion.py` | Performs fusion + backtest + risk analysis on selected model combos |

- Supports multi-combo fusion: `--from-config-all` runs all enabled combos, while disabled combos require explicit research opt-in
- Ensemble execution strictly verifies workspace-contained MLflow lineage, per-model prediction freshness, and exact declared/loaded membership; silent combo shrinkage is rejected
- Multi-combo mode generates crossover comparison tables and net value comparison charts
- Weighting modes: `equal` / `icir_weighted` / `manual` / `dynamic`
- Outputs fused predictions to Qlib Recorder's pred.pkl
- Config file: `config/ensemble_config.json` (Multi-combo structure, backwards compatible)

### ⑤ Post-Trade Module

> See details in [04_POST_TRADE_GUIDE.md](04_POST_TRADE_GUIDE.md)

| Script | Purpose |
|------|------|
| `prod_post_trade.py` | Processes live trading data to update holdings and cash |

- **Completely independent** of the training/predict/fusion modules
- Reads executed trade records from brokerage exported files (`.xlsx`)
- Supports multiple deposits/withdrawals by date (`config/cashflow.json`)
- Updates holdings and cash balances in `config/prod_config.json`

### ⑥ Order Generation Module

> See details in [06_ORDER_GEN_GUIDE.md](06_ORDER_GEN_GUIDE.md)

| Script | Purpose |
|------|------|
| `order_gen.py` | Generates buy/sell order CSVs based on predictions + current holdings |

- Supports three prediction sources: ensemble fusion (default) / single model / custom file
- **Multi-Model Opinion Table**: Auto-loads predictions from all combos and single models, generating BUY/SELL/HOLD judgments for each instrument
- Includes visualization script `plot_model_opinions.py` to generate multi-model ranking line charts
- Relies on holdings and cash in `prod_config.json` (from Post-Trade)
- TopK + DropN strategy dictates buy/sell/hold decisions

### ⑦ Signal Ranking Module

> See details in [07_SIGNAL_RANKING_GUIDE.md](07_SIGNAL_RANKING_GUIDE.md)

| Script | Purpose |
|------|------|
| `signal_ranking.py` | Normalizes fusion scores into recommendation indices and generates Top N rankings |

- Normalizes scores to a -100 ~ +100 recommendation index
- Supports generation by combo (`--all-combos` / `--combo`)
- Independent from order generation, suitable for distributing to others

### ⑧ Deep Analysis Module (MAS + OOM-RL)

> See [50_DEEP_ANALYSIS_GUIDE.md](50_DEEP_ANALYSIS_GUIDE.md), [51_OOMRL_FEEDBACK_OVERVIEW.md](51_OOMRL_FEEDBACK_OVERVIEW.md)

| Script | Purpose |
|------|------|
| `run_deep_analysis.py` | Multi-Agent post-trade analysis: 7 specialist agents × multiple time windows |
| `run_deep_analysis.py --critic` | **OOM-RL Phase 3** — LLM Critic generates executable ActionItems |
| `run_feedback_loop.py` | **OOM-RL Phase 4** — Sandboxed Playground execution, validation, promotion |

- **Agent System**: Market Regime / Model Health / Ensemble Eval / Execution Quality / Portfolio Risk / Prediction Audit / Trade Pattern
- **OOM-RL feedback loop**: Signal Extractor (rules) → LLM Critic (decisions) → ActionItem (executable) → Feedback Loop (execute + validate + promote)
- **Playground isolation**: All experiments run in an independent sandbox workspace
- **Priority scheduling**: Auto-sorted by signal severity, confidence, and training cost
- **Rollback safety**: Production workspace is a standalone Git repo — `git checkout` to revert
- **Complete audit trail**: JSONL records + human-readable Markdown reports + CHANGELOG.md

### ⑨ Feedback Loop Module

> See [54_OOMRL_FEEDBACK_LOOP.md](54_OOMRL_FEEDBACK_LOOP.md)

| Script | Purpose |
|------|------|
| `run_feedback_loop.py --report-only` | Preview mode: sort + preview, no execution |
| `run_feedback_loop.py --execute` | Execute mode: apply changes in Playground and validate |
| `run_feedback_loop.py --promote` | Promote mode: push validated config to production |
| `run_feedback_loop.py --auto-promote` | Fully automatic (not yet implemented in Phase 4) |

- `--models` / `--skip-models` for manual model selection
- `--max-duration-minutes` for time-budget control
- `--dry-run` to preview adapter changes without writing files
- `--skip-retrain` to modify configs without retraining

---

## Data Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Configuration Layer                        │
│  model_registry.yaml    model_config.json    prod_config.json    strategy_config.yaml   │
│  cashflow.json          ensemble_config.json                    │
└────────────┬──────────────────┬──────────────────┬──────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Train / Predict  │  │ Live Trade       │  │ Deposits/        │
│                  │  │ Processing       │  │ Withdrawals      │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         ▼                     ▼                     ▼
latest_train_records.json   prod_config.json (Update Pos/Cash)
         │                             │
         ▼                             │
   ┌──────────────────┐                 │
   │ Ensemble Search    │                 │
   │ brute_force_*.py │                 │
   │ analyze_*.py     │                 │
   └────────┬─────────┘                 │
            │ Recommended Combos        │
            ▼                           │
   ensemble_config.json                 │
            │                           │
            ▼                           │
   ┌──────────────────┐                 │
   │ Ensemble Fusion  │                 │
   │ ensemble_fusion  │                 │
   └────────┬─────────┘                 │
            │                           │
            ▼                           │
   ensemble_records.json                │
   (combo → recorder_id)                │
            │                           ▼
            ▼                           ▼
  ┌────────────────────────────────────────┐
  │ Order Generation                       │
  │ order_gen.py                           │
  │ Input: ensemble_records + Pos/Cash     │
  │ Output: buy/sell_suggestion_*.csv      │
  └────────────────────────────────────────┘
```

---

## Key Files Explanation

### Config Files (`config/`)

| File | Purpose | Related Module |
|------|------|----------|
| `model_registry.yaml` | Model Registry: Defines model names, algorithms, datasets, labels | Train, Predict |
| `model_config.json` | Environment Params: Training windows, slicing mode, market, benchmark | Train, Predict, Environment |
| `strategy_config.yaml` | Strategy Params: Order generation boundaries (TopK, etc.), Backtest environment config | Order Gen, Backtesting |
| `prod_config.json` | Live State: Holdings, Cash, Processing state | Post-Trade, Order Gen |
| `cashflow.json` | Cashflow Records: Deposits/Withdrawals by date | Post-Trade |
| `ensemble_config.json` | Multi-combo Config: Combo definitions, weights, defaults | Fusion, Order Gen, Ranking |
| `ensemble_records.json` | Fusion Record Pointer: Mapping from combo to recorder_id + default flag | Fusion, Order Gen, Ranking |
| `rolling_config.yaml` | Rolling Training Params: Start date, train/valid years, step size | Rolling Train |
| `workflow_config_*.yaml` | Qlib Workflows: Training configurations for each model | Train |
| `feedback_scope.json` | OOM-RL Feedback Scope: Controls which domains the LLM Critic can influence | Deep Analysis, Feedback Loop |
| `hyperparam_bounds.json` | Hyperparameter bounds and per-change magnitude limits | LLM Critic, Training Adapter |
| `llm_config.json` | LLM API config: model, endpoint, key | Deep Analysis, LLM Critic |
| `oos_config.json` | OOS evaluation parameters: exclusion period | Deep Analysis |
| `config/skills/` | LLM Critic system prompts (Markdown) | LLM Critic |

### Output Files (`output/`)

| Dir/File | Purpose |
|-----------|------|
| `(Qlib Recorders)` | Prediction results for models and ensembles (Stored in mlruns) |
| `(Qlib Recorders)` | Rolling training predictions |
| `ensemble_runs/` | Brute force search outputs: dedicated subdirectories per run containing `is/`, `oos/`, and `summary.md` |
| `ensemble/` | Fusion configs, leaderboards, charts, inter-combo comparisons |
| `ranking/` | Top N signal ranking CSVs |
| `model_opinions_*.csv/json` | Multi-model BUY/SELL/HOLD judgment tables |
| `model_opinions_*_linechart.png` | Multi-model ranking visualization charts |
| `buy/sell_suggestion_*.csv` | Target buy/sell execution orders |
| `model_performance_*.json` | Model IC/ICIR metrics |
| `deep_analysis/` | Deep Analysis and OOM-RL outputs (action_items, feedback_report) |

### Running State (`data/`)

| File | Purpose |
|------|------|
| `history/` | Auto-backed up historical files |
| `order_history/` | Historical order suggestions, trade details, exported brokerage Excel files (supervised by archiver) |
| `run_state.json` | Locked/CAS-protected static/CPCV Training State V3; Rolling uses separate state |
| `training_runs/<run_id>/` | Immutable static/CPCV target evidence and recoverable closure state |
| `rolling_state.json` | Rolling training state tracker (crash recovery) |
| `migrate_records.py` | (Tool) Upgrades legacy dual-file record system to unified `model@mode` format |
| `trade_log_full.csv` | Cumulative trade log (both buys and sells) |
| `holding_log_full.csv` | Cumulative holdings log |
| `daily_amount_log_full.csv` | Daily capital summary |
| `training_history.jsonl` | Training convergence log (one record appended per training run) |
| `operator_log.jsonl` | Operation audit log (one record per script execution) |
| `fusion_run_ledger.jsonl` | Fusion run ledger (one record per backtest) |
| `action_item_history.jsonl` | OOM-RL ActionItem audit trail |
| `promote_history.jsonl` | Config promotion audit records |
| `CHANGELOG.md` | Human-readable changelog |
| `promote_history/` | Detailed Markdown reports for each promotion |
| `config_history/` | Configuration snapshot history |

### Archive Directory (`archive/`)

| Directory | Purpose |
|------|------|
| `output/` | Older output files (past versions of predictions, ensembles, brute forces) |

---

## Documentation Index

| ID | Document | Contents |
|:---:|------|------|
| 00 | [This Doc](00_SYSTEM_OVERVIEW.md) | System overview, architecture, typical scenarios, module cheat sheet |
| 01 | [TRAINING_GUIDE](01_TRAINING_GUIDE.md) | Full training, incremental training, model registry management |
| 02 | [BRUTE_FORCE_GUIDE](02_BRUTE_FORCE_GUIDE.md) | Ensemble Search and Backtesting (Standard + Fast vectors) |
| 03 | [ENSEMBLE_FUSION_GUIDE](03_ENSEMBLE_FUSION_GUIDE.md) | Model fusion prediction, weighting schemes, tracking |
| 04 | [POST_TRADE_GUIDE](04_POST_TRADE_GUIDE.md) | Live trading data processing, holding/capital updates |
| 05 | [PREDICT_ONLY_GUIDE](05_PREDICT_ONLY_GUIDE.md) | Prediction only (no retrain), update train records |
| 06 | [ORDER_GEN_GUIDE](06_ORDER_GEN_GUIDE.md) | Order generation, outputting buy/sell suggestions |
| 07 | [SIGNAL_RANKING_GUIDE](07_SIGNAL_RANKING_GUIDE.md) | Top N signal recommendation |
| 08 | [ANALYSIS_GUIDE](08_ANALYSIS_GUIDE.md) | Single model quality, fusion correlations, slippage costs and comprehensive multi-factor risk assessments |
| **30** | **[ROLLING_TRAINING_GUIDE](30_ROLLING_TRAINING_GUIDE.md)** | **Rolling training: sliding windows, cold start, crash recovery** |
| **50** | **[DEEP_ANALYSIS_GUIDE](50_DEEP_ANALYSIS_GUIDE.md)** | **Multi-Agent Deep Analysis + OOM-RL LLM Critic** |
| **51** | **[OOMRL_OVERVIEW](51_OOMRL_FEEDBACK_OVERVIEW.md)** | **OOM-RL closed-loop feedback system overview** |
| **52** | **[OOMRL_DATA](52_OOMRL_DATA_INFRASTRUCTURE.md)** | **Feedback data infrastructure** |
| **53** | **[OOMRL_CRITIC](53_OOMRL_CRITIC_GUIDE.md)** | **LLM Critic: Signal + ActionItem + Skills** |
| **54** | **[OOMRL_LOOP](54_OOMRL_FEEDBACK_LOOP.md)** | **Feedback Loop Execution: Playground + Adapter + Promote** |
| 70 | [WALKTHROUGH](70_WALKTHROUGH.md) | **End-to-End Hands-On Walkthrough (Start Here!)** |

---

## Shared Utilities

The `quantpits/utils/` directory provides shared capabilities for all scripts, eliminating code duplication:

| Module | Purpose | Consumers |
|------|------|----------|
| `train_utils.py` | Date calculus, YAML injection, model registry, record merging, history backups | Training, Prediction |
| `predict_utils.py` | Prediction data load/save, Recorder management | Prediction, Fusion, Ensemble Search |
| `config_loader.py` | Workspace-level config loading | Global |
| `config_contracts/` | Workspace config validation, normalization, and fingerprints for pre-run checks and future plan/manifest reuse | Global |
| `runtime/` | Shared `CommandPlan` / `RunManifest` / plan renderer / manifest writer runtime primitives | Integrated by `ensemble_fusion` and `order_gen` |
| `training/` | Static/CPCV plans, pure recovery classifier, immutable target evidence, workspace lease, State V3, retryable publication/closure, and service lifecycle | Training and predict-only |
| `order/` | Order command, prepared source, execution service, pure opinion calculation, atomic persistence, and actual artifact ledger | Order generation |
| `ensemble/` | Ensemble fusion service, I/O, and reporting layer: explicit config loading, workspace-bound execution paths, plan/render, manifest, OperatorLog linkage, prediction persistence, fusion ledger, correlation/LOO analytics, risk/leaderboard/chart reports, and execution lifecycle | Fusion |
| `strategy.py` | Strategy config / backtest strategy construction | Ensemble Search, Fusion, Analysis |
| `backtest_utils.py` | Qlib backtest execution and evaluation | Ensemble Search, Fusion, Analysis |
| `env.py` | Qlib initialization, workspace management, `set_root_dir()` runtime switching | Global |
| `workspace.py` | Explicit `WorkspaceContext`, workspace path helpers, stable fingerprint helpers | Global |
| `operator_log.py` | Operation audit log: auto-records every script execution | Global (7 scripts integrated) |
| `ensemble_utils.py` | Ensemble config parsing, combo management, records loading | Fusion, Signal Ranking, Order Gen |
| `search_utils.py` | Combo search shared logic: signal handling, backtest core, IS/OOS splitting, grouped combinations | Ensemble Search (Standard/Fast), MinEntropy, Analysis |
| `run_context.py` | Encapsulates per-run output directory management and IS/OOS stratifications | Ensemble Search, Analysis |
| `fusion_engine.py` | Weight calculation (equal/icir/manual/dynamic), signal fusion | Fusion |
| `backtest_report.py` | Detailed backtest analysis reports (using PortfolioAnalyzer) | Fusion |

### Workspace Config Validation

| Script | Purpose |
|------|------|
| `validate_workspace.py` | Read-only validation for core workspace configs with structured warnings/errors and normalized fingerprints |

```bash
python -m quantpits.tools.validate_workspace --workspace workspaces/Demo_Workspace --read-only
python -m quantpits.tools.validate_workspace --workspace workspaces/Demo_Workspace --json
python -m quantpits.tools.validate_workspace --workspace workspaces/Demo_Workspace --strict
```

The command does not write to the workspace. JSON output includes paths, summaries, and fingerprints only; it does not include full raw configs, holding details, or cash details.

### Runtime Plan / Manifest Infrastructure

`quantpits/runtime/` provides shared runtime planning and run manifest primitives. `ensemble_fusion.py` is the first production command integrated with this runtime model, and future heavy commands can reuse the same pattern:

| Module | Purpose |
|------|------|
| `command.py` | `CommandPlan`, `CommandResult`, input/output/state refs, and step descriptions |
| `ids.py` | Stable-format `run_id` generation |
| `render.py` | Human-readable dry-run plan rendering and public dict output |
| `manifest.py` | `RunManifest`, `manifest_from_result()`, and `write_run_manifest()` |

The default manifest path is `output/manifests/{command}/{run_id}.json`, and files are written only when the caller explicitly invokes `write_run_manifest()`. Manifest/public dict output records paths, summaries, fingerprints, records, and warnings; it does not include full raw configs. Real `ensemble_fusion.py` runs now write `output/manifests/ensemble_fusion/<run_id>.json` by default; dry-runs (`--explain-plan` / `--json-plan`) do not write manifests. `ensemble_fusion.py` keeps plan output workspace-relative and resolves relative output paths under the active workspace only at the real execution boundary.

`order_gen.py` also supports `--explain-plan` / `--json-plan`. These modes only read and validate workspace configuration; they do not initialize Qlib, load predictions or prices, or write files. `--dry-run` performs the full calculation but writes neither artifacts nor OperatorLog. Real execution uses the prepared recorder/config snapshot, atomically writes actual non-empty order/opinion outputs, writes `output/manifests/order_gen/<run_id>.json` by default, and links OperatorLog; `--no-manifest` disables only the manifest. Importing the command does not change cwd. Manifest/public summaries exclude full predictions, holdings, order rows, and raw config.

`static_train.py` and `cv_train.py` now share a plan-first command/service boundary. `--explain-plan`
and `--json-plan` return before safeguard and Qlib/MLflow initialization, write nothing, and do not
change cwd. Resume planning requires the newly selected ordered target keys to exactly match State V3
before runtime initialization. Real execution initializes Qlib once and binds exact dates, targets, source identity, and
the publication baseline into an execution fingerprint. A workspace lease, Training State V3, and
publication intent/receipt provide concurrency protection and crash recovery; the record is replaced
after performance as the current-pointer commit. The service owns target iteration, resume, promotion,
and the actual-output manifest. An incomplete full run never replaces the current record;
incremental/predict-only partial failures commit successes but still fail the command. Training Record V2 uses strict schema dispatch, while ensemble execution rechecks
workspace/experiment/artifact identity and actual prediction coverage.

`ensemble_fusion` also exposes a typed command boundary: `quantpits/ensemble/command.py` owns the argument contract and prepare/explain/execute routing, `quantpits/ensemble/service.py` owns the execution lifecycle, and the script layer owns process exit semantics only. The engine layer never calls `sys.exit()`, so notebooks, schedulers, tests, and other Python workflows can reuse the command runner and handle structured outcomes or typed domain errors themselves.

### Cross-command semantic verification lane

Developers can run `make test-semantic` (equivalent to `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/quantpits/semantic -q --tb=short --no-cov -p no:cacheprovider`) to verify identity, outcome, durable-effect, and recovery conservation across Training/Rolling, Ensemble, Post-trade, and Order. The lane uses only pytest-generated sanitized workspaces and deterministic fakes at external boundaries. It does not read a real workspace, initialize real Qlib/MLflow, require network or GPU access, or replace the owner-operated full suite and real-backend smoke checks.

`OperatorLog` has compatible optional fields for `run_id`, `manifest_path`, `plan_fingerprint`, and `transaction_id`. `ensemble_fusion.py`, `order_gen.py`, and post-trade link the relevant fields to their run manifests; commands without a transaction keep it as `null`.

### ⑧ File Archiving Tool

| Script | Purpose |
|------|------|
| `archive_dated_files.py` | Archives old dated output logs, retaining only the past N versions |

- Scans `output/` and `data/` for time-stamped files and groups them intuitively
- Trading data (suggestions, logs, excel files) is safely archived to `data/order_history/`
- Previous outputs migrated to `archive/output/`
- Supports `--dry-run`, `--keep N`, `--include-notebooks`, `--cleanup-legacy`
- **Routine use**: Just execute `python quantpits/tools/archive_dated_files.py` post-run each period


---

## Decision Reference

### When is retraining necessary?

- Model prediction quality notably drops (IC/ICIR decay)
- Phenomenal shifts manifest across widespread market styles
- Entirely new models are registered which necessitate a first-time fit
- Need to validate results following massive hyperparameter optimizations

### When is an ensemble search necessary?

- New models have been added or older ones removed
- Prediction characteristics drastically shift post-retraining
- Repeated underperformance sequentially over weeks
- Recommended routine reviews (e.g. at a monthly cadence)

### When should I run the OOM-RL feedback loop?

- Deep Analysis report shows `severe_underfitting` or `overfitting` models
- LLM Critic produced executable ActionItems (`action_items_{date}.json` is non-empty)
- You want to auto-validate hyperparameter tuning effects (Playground sandbox + backtest comparison)
- You need a human-readable audit trail of configuration changes

### What is the minimal daily / weekly operation?

```bash
# Routine minimal loop (5 commands, assuming workspace is activated)
python quantpits/scripts/static_train.py --predict-only --all-enabled     # Predict
python quantpits/scripts/ensemble_fusion.py --from-config-all       # Fuse
python quantpits/scripts/prod_post_trade.py                      # Post-Trade
python quantpits/scripts/order_gen.py                              # Generate Orders
python -m quantpits.scripts.run_deep_analysis                      # Deep Analysis (recommended)
```
