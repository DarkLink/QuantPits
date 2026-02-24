# Brute Force Ensemble Backtesting Guide

Automates the exhaustive combinatorial tracking of all model predictions, executing equal-weighted fusion backtests on each subset to uncover the optimal model ensemble pipeline.

## Quick Start

```bash
cd /path/to/QuantPits

# Quick test (max 3 models per combo, ~175 backtests)
python engine/scripts/brute_force_ensemble.py --max-combo-size 3

# Full exhaustive search (10 models = 1023 combos, highly time-intensive)
python engine/scripts/brute_force_ensemble.py

# Analysis only (bypasses recalculating backtests)
python engine/scripts/brute_force_ensemble.py --analysis-only

# Resume from interruption (via Ctrl+C payload or fatal exit)
python engine/scripts/brute_force_ensemble.py --resume

# Exhaustive search utilizing Model Grouping (only selections one per group, slashes overhead)
python engine/scripts/brute_force_ensemble.py --use-groups

# Specific groupings + throttle logical cores to limit RAM pressure
python engine/scripts/brute_force_ensemble.py --use-groups --group-config config/my_groups.yaml --n-jobs 2
```

## Script Flow

### Stage 1 — Load Predictions
- Reads `latest_train_records.json` to acquire the experiment name and available model mapping.
- Spawns Qlib Recorders to load `pred.pkl` prediction scores for each active model.
- Conducts cross-sectional Z-Score normalization daily.

### Stage 2 — Correlation Analysis
- Evaluates prediction correlation matrices.
- Saves raw correlation CSVs for subsequent debugging.

### Stage 3 — Brute Force Backtesting
- Generates all valid combinatorial subset models (from 1 to N).
- For each subset combo: Equal-weight fusion → TopkDropoutStrategy → Native Qlib Simulation.
- Extracts core portfolio mechanics: Annualized Return, Max Drawdown, Calmar Ratio, Excess Return.
- Supports `--resume` to ingest existing CSV batches without restart logic.

### Stage 4 — Result Analysis
- **Top Combo Ranks**: Ordered by Annualized Excess Return (Top 20) and Calmar Robustness (Top 10).
- **Model Attribution**: Examines frequency footprints of models inhabiting the Top/Bottom N, constructing net-win distributions.
- **Correlation vs Performance**: Highlights diversity multipliers, seeking out low-correlation/high-Calmar "Golden Pipelines".
- **Hierarchical Clustering**: Computes Ward linkage dendrograms utilizing Excess Return profiles.
- **Weight Optimization**: Comparative trials on Top 10 single models simulating Max Sharpe / Risk Parity optimization mappings.
- **Comprehensive Reporting**: Generates autonomous summaries of MVP models and superior fusions.

## Full Parameter List

| Parameter | Default | Description |
|------|--------|------|
| `--record-file` | `latest_train_records.json` | Train records pointer targeting model manifests. |
| `--max-combo-size` | `0` (All) | Upper limit of combined models (Or clusters if grouped). |
| `--min-combo-size` | `1` | Lower limit of combined models (Or clusters if grouped). |
| `--freq` | `week` | Simulation periodicity (`day` / `week`). |
| `--top-n` | `50` | Scale N target for Top/Bottom analysis metrics. |
| `--output-dir` | `output/brute_force` | Directory bounds. |
| `--resume` | - | Ingest target logic for crash recovery execution. |
| `--skip-analysis` | - | Skip analytical stages post-run. |
| `--analysis-only` | - | Jump to Analytics parsing existing outputs locally. |
| `--n-jobs` | `4` | Parallel concurrent simulations. |
| `--batch-size` | `50` | Combo count per persistence bucket (Impacts RAM intensity and checkpoint intervals). |
| `--use-groups` | - | Activate manual exclusionary subset routing (selects at most one model per defined group block). |
| `--group-config` | `config/combo_groups.yaml` | Overriding path for groupings definitions. |

## Output Files

All logs reside within `output/brute_force/`:

```text
output/brute_force/
├── correlation_matrix_{date}.csv     # Prediction correlations
├── brute_force_results_{date}.csv    # Base metrics (Terminal outputs)
├── model_attribution_{date}.csv      # Attribution vectors
├── model_attribution_{date}.png      # Attribution visual
├── risk_return_scatter_{date}.png    # Risk vs Return 2D plots
├── cluster_dendrogram_{date}.png     # Hierarchical mapping
├── optimization_weights_{date}.csv   # Simulation constraints weighting
├── optimization_equity_{date}.png    # Optimal simulation trajectory logs
└── analysis_report_{date}.txt        # Synthesis summary document
```

## Runtime Considerations

Assuming 10 active models (1023 sequential iterations):

| Range | Combos | ETA Scope |
|:---:|:---:|:---:|
| 1~3 | 175 | ~15 min |
| 1~5 | 637 | ~50 min |
| 1~10 (All) | 1023 | ~1.5 hours |

> **Best Practice**: Execute a preliminary `--max-combo-size 3` to test system integration before throwing heavy monolithic batches onto hardware.

## Advanced Usage

### Strict Combinatorial Sizes
```bash
# Evaluate only permutations featuring precisely 4 to 6 models
python engine/scripts/brute_force_ensemble.py --min-combo-size 4 --max-combo-size 6
```

### Daily Simulation Mapping
```bash
python engine/scripts/brute_force_ensemble.py --freq day
```

### High-Fidelity Deep Dive Analysis
```bash
# Top/Bottom 100 footprint attribution
python engine/scripts/brute_force_ensemble.py --analysis-only --top-n 100
```

## Checkpoint Sequencing & Safe Interruptions

The architecture bolsters **streaming checkpoints** alongside **Safe Interrupt protocols**, making volatile workloads secure from progression loss.

### Mechanics

- **Batch Chunking**: Global combinations are segmented into `--batch-size` partitions, aggressively persisting CSV logs at identical bounds.
- **Signal Tapping**: Interfaces attach to `SIGINT` (Ctrl+C) / `SIGTERM` handlers. Triggering causes:
  1. Queued termination bounds across active batch nodes.
  2. Commits currently executed pipelines entirely to IO safely.
  3. Halts gracefully.
  4. Subsequent Ctrl+C forcibly ends memory mapping.
- **Resume Routing**: Engaging `--resume` parses the destination output and automatically nulls pre-completed combos.

### Execution Trace

```bash
# Initiate
python engine/scripts/brute_force_ensemble.py --max-combo-size 5

# Ctrl+C applied here -> Wait for current thread sync, outputs termination prompt
# "⚠️ Interrupted safely! Completed: X/Y combos. Utilize --resume next session."

# Resume at later date
python engine/scripts/brute_force_ensemble.py --max-combo-size 5 --resume
```

### RAM Safeguards

| Parameter | Purpose |
|------|------|
| `--batch-size` | Dial down interval ranges to mitigate peak process retention (Default 50). |
| `--n-jobs` | Reduce thread spawns to alleviate simultaneous mapping loads (2-4 suggested). |

> Hard allocations are managed implicitly utilizing aggressive `gc.collect()` invokes immediately following each batch horizon.

---

## Model Grouping Exhaustion

Scaling beyond 15+ algorithms inherently dictates explosive permutation mapping magnitudes (32767+ combinations). Grouped exhaustion categorizes individual algorithms into distinct mutually-exclusive vectors, strictly imposing structural rules to pick a ceiling of 1 mapping per bounded group, significantly culling permutations.

### Configuration Layout

Group domains are sourced at `config/combo_groups.yaml`:

```yaml
groups:
  LSTM_variants:           # Nominal ID string
    - lstm_Alpha360        # Internal targeting index (Aligns to train_records hash values)
    - alstm_Alpha158
    - alstm_Alpha360

  Tree_models:
    - lightgbm_Alpha158
    - catboost_Alpha158

  # ... Additional grouping matrices ...
```

**Constraints**:
- IDs strictly provide cosmetic rendering boundaries.
- When toggling constrained groups, **unmapped models are natively suppressed** (skipped).
- Operates totally disparately from tags injected via `model_registry.yaml`.
- `--min/max-combo-size` scopes re-orient to bound the logic toward **quantities of active group containers invoked**.

### Execution

```bash
# Toggle utilizing default group constraints
python engine/scripts/brute_force_ensemble.py --use-groups

# Pass alternate YAML group routing definitions
python engine/scripts/brute_force_ensemble.py --use-groups --group-config config/custom_groups.yaml

# Filter combinatorial lengths utilizing bounded sizes across group capacities
python engine/scripts/brute_force_ensemble.py --use-groups --min-combo-size 3 --max-combo-size 4
```

### Output Reductions

| Scale | Volume Baseline | Exhaustive Combos | Grouped Combos |
|:---:|:---:|:---:|:---:|
| 15 Models, 6 groups × 2~3 nodes | 15 | 32767 | ~500 |
| 10 Models, 5 groups × 2 nodes | 10 | 1023 | ~62 |

> **Notice**: The vectorized fast port (`brute_force_fast.py`) is intrinsically tied to parity regarding `--use-groups` and `--group-config` overrides.

---

## ⚡ Accelerated Vectors (`brute_force_fast.py`)

Processing matrices expanding above a 10+ node density yield unacceptable wait loops executing natively on Qlib. `brute_force_fast.py` overrides the simulation core by utilizing **NumPy/CuPy vectorized matrix calculation methodologies**. It yields a **~5000x acceleration multiplier**.

### Quick Setup

```bash
cd /path/to/QuantPits

# Test sample
python engine/scripts/brute_force_fast.py --max-combo-size 3

# Unrestricted combinations
python engine/scripts/brute_force_fast.py

# ================================
# Premium Mitigation: Out-Of-Sample Chronology Splits
# Isolates combinations via purely In-Sample windows filtering the most recent year.
# Verifies dynamically selected Top 10 ensembles on the completely blind forward step.
# ================================
python engine/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 10

# Push compute layer to GPU architectures
python engine/scripts/brute_force_fast.py --use-gpu

# Local Analysis
python engine/scripts/brute_force_fast.py --analysis-only

# Crash recovery
python engine/scripts/brute_force_fast.py --resume

# Filter using Group limits targeting faster subset scaling
python engine/scripts/brute_force_fast.py --use-groups --group-config config/combo_groups_20.yaml
```

### Deviations Across Simulations

| Layer | Standard (Qlib Backend) | Vectorized Fast |
|------|:---:|:---:|
| Limit Up/Down Discards | ✅ | ❌ |
| Trade Costing Assumptions | ✅ Precision Check | ⚠️ Proxied via Aggregate Turnover Rates |
| TopK+Dropout Mechanics | ✅ Implemented | ⚠️ Simplified TopK only |
| Cash Reinvestment Logic | ✅ Dynamic Volume | ❌ Pure equal-balance assumptions |
| Time Constraint | ~5s/combo | ~0.001s/combo |

> **Sort Scaling Ratios Output Consistent Ranks**: The absolute numerical estimates manifest variance from reality; however, internal relational scoring logic remains mathematically consistent, establishing an ultra-efficient sorting filter scope. （**NO GOOD, USE WITH CAUTION**）

### Optimized Workflow Progression

1. **Broad Net**: Engage `python engine/scripts/brute_force_fast.py` scaling all exhaustive combinations (Completes rapidly).
2. **Precision Audit**: Extract top 10/20 groupings from the prior log, pushing them specifically against `brute_force_ensemble.py` ensuring deep realistic metrics.
3. **Execution Delivery**: Cement the validated combination scope into `ensemble_fusion.py` driving trading signal distribution.

### Dedicated Fast Arguments

| Argument | Built-In | Core Intent |
|------|--------|------|
| `--batch-size` | `512` | Massive vector matrix chunking sizes across array layers. |
| `--use-gpu` | - | Routes memory allocation to CuPy drivers scaling hardware acceleration. |
| `--no-gpu` | - | Forces host device array calculations. |
| `--cost-rate` | `0.002` | Friction proxies tied to turnover deviations mapping bidirectional 0.2%. |

> Auxiliary bounds (`--max-combo-size`, `--resume`, `--analysis-only`, `--use-groups`, `--group-config`, `--exclude-last-years`, `--auto-test-top` etc.) translate perfectly over from standard topologies.

---

## 🕰️ Hardened Paradigms: Preventing In-Sample Overfitting

While brute-forcing, programmatic optimization guarantees immense susceptibility against **In-Sample Overfitting** (Forming subsets identically geared exclusively toward the simulation period logic, crumbling upon forward deployment instances).

By employing `brute_force_ensemble.py` and `brute_force_fast.py` coupled with **Dynamic Relative Time Fencing**, the architecture quarantines forward temporal data retaining it as an explicit **Out-Of-Sample (OOS) Baseline Evaluator**:

### Validated Chronology Filtering

```bash
# 1. Search subsets generating outputs immediately scored on chronological OOS blind data.
# --exclude-last-years 1: Fences the latest year entirely outside algorithm views processing 2-year prior spans as base.
# --auto-test-top 5: Forces subsequent backtests against the blind year dynamically scoring robustness.
python engine/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 5
```

Deploying traces will output standardized bounds appending a `Stage 5: Autonomous Out-Of-Sample (OOS) Testing Scope (Top 5)` log verifying forward reality mappings.

### Date Modifiers

| Overrides | Contextual Range |
|------|------|
| `--exclude-last-years N` | Quarantines most recent chronological N years establishing dynamic IS boundaries prioritizing blind testing sets. |
| `--exclude-last-months N` | Replicates utilizing discrete month bounds. |
| `--auto-test-top N` | Isolates Top N algorithms testing them aggressively against forward data dumping `oos_validation_{date}.csv`.|
| `--start-date YYYY-MM-DD` | Absolute forced start. |
| `--end-date YYYY-MM-DD` | Subversive forced stop bounds deleting all subsequent rows. |

> Warning: Trading data is continually rolled per schedule natively. Implementing dynamic `exclude-last-years` / `exclude-last-months` fences inherently aligns testing without needing static arbitrary data edits per epoch.

---

### GPU Parallel Access

Engaging CuPy drivers natively scales architectures to parallel acceleration:

```bash
# CUDA 12.x Install Path
pip install cupy-cuda12x

# CUDA 11.x Install Path
pip install cupy-cuda11x
```

### Fast-Mode Benchmark Times

| Range Node Limits | Output Pool | Core CPU Path | CuPy GPU Execution Path |
|:---:|:---:|:---:|:---:|
| 1~3 | 175 | ~2s | ~1s |
| 1~5 | 637 | ~5s | ~2s |
| 1~10 (Full) | 1023 | ~10s | ~3s |
| 1~15 (Insane) | 32767 | ~5 min | ~1 min |

### Extracted Reports

Fast iteration layers route intrinsically to `output/brute_force_fast/` separating from native qlib output files:

```text
output/brute_force_fast/
├── correlation_matrix_{date}.csv          # Inter-prediction linkage
├── brute_force_fast_results_{date}.csv    # Vector simulations metrics
├── model_attribution_{date}.csv           # Structural frequency logs
├── model_attribution_{date}.png           # Structural graphic renders
├── risk_return_scatter_{date}.png         # Deviation distributions
├── optimization_weights_{date}.csv        # Bounded optimizations
└── analysis_report_fast_{date}.txt        # Native execution metrics
```
