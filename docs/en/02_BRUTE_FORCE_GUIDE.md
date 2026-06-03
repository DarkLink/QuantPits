# Ensemble Search and Backtesting Guide

Automates the exhaustive combinatorial tracking of all model predictions, executing equal-weighted fusion backtests on each subset to uncover the optimal model ensemble pipeline.

## Quick Start

```bash
cd /path/to/QuantPits

# Quick test (max 3 models per combo, ~175 backtests)
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

# Full exhaustive search (10 models = 1023 combos, highly time-intensive)
python quantpits/scripts/brute_force_ensemble.py

# Use Z-Score normalization (preserves model "confidence" info, intersection fusion)
python quantpits/scripts/brute_force_ensemble.py --norm-method zscore


# Resume from interruption (via Ctrl+C payload or fatal exit)
python quantpits/scripts/brute_force_ensemble.py --resume

# Exhaustive search utilizing Model Grouping (only selections one per group, slashes overhead)
python quantpits/scripts/brute_force_ensemble.py --use-groups

# Specific groupings + throttle logical cores to limit RAM pressure
python quantpits/scripts/brute_force_ensemble.py --use-groups --group-config config/my_groups.yaml --n-jobs 2
```

## Script Flow

### Stage 1 — Load Predictions
- Reads `latest_train_records.json` to acquire the experiment name and available model mapping.
- Spawns Qlib Recorders to load `pred.pkl` prediction scores for each active model.
- Conducts cross-sectional normalization daily (default `rank`: percentile rank → [0,1]; optional `zscore`).
- In rank mode, stocks not covered by a model are filled with 0.5 (abstain/neutral), enabling **union**-based fusion.

### Stage 2 — Correlation Analysis
- Evaluates prediction correlation matrices.
- Saves raw correlation CSVs for subsequent debugging.

### Stage 3 — Ensemble Search Backtesting
- Generates all valid combinatorial subset models (from 1 to N).
- For each subset combo: Equal-weight fusion → TopkDropoutStrategy → Native Qlib Simulation.
- Extracts core portfolio mechanics: Annualized Return, Max Drawdown, Calmar Ratio, Excess Return.
- Supports `--resume` to ingest existing CSV batches without restart logic.

### Stage 4 — Metadata Export
- After exhaustion, a `run_metadata.json` is generated under `output/ensemble_runs/{script}_{date}/`, capturing the exact environment context and date splits.

### Stage 5 — Decoupled OOS Analysis
- Ensemble search is fundamentally In-Sample (IS). All analysis and Out-Of-Sample (OOS) validation logics have been moved to the decoupled `analyze_ensembles.py` script.
- Executing `python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_<date>/run_metadata.json` constructs multidimensional candidate pools (Yield, Robustness, MVP) and runs completely blind OOS evaluation.
- After analysis completes, `summary.md` is auto-generated in the run directory, containing IS/OOS key metrics and a file inventory.

> [!NOTE]
> **Understanding Metric Discrepancies: Single Models vs. Ensemble Backtests**
>
> When evaluating model performance within fusion and ensemble search architectures, strict **Z-Score Normalization** and **Data Alignment** processing govern the engine. Therefore, because of TopK position bounding, backtest results of a single model here may exhibit reasonable, micro-level disparities from the raw metrics evaluated naturally post-training (e.g. via `run_analysis.py`):
> 1. **Isolated Normalization**: Each model calculates its daily cross-sectional Z-scores purely on its *own* non-null predicted universe. Scaling remains mathematically uniform, and a single model's signal scale cannot be skewed by other models' data coverage gaps prior to scoring.
> 2. **Delayed Intersection**: Strict intersection dropping (`dropna(how='any')`) is executed strictly at the exact combo scoring phase and is limited precisely to the subset of models within that specific combo iteration. This guarantees irrelevant sub-models don't unilaterally shrink the evaluated combination universe.
> 3. **Benchmarking Alignment**: The sub-model evaluation leaderboard dynamically slices historical records to match the precise temporal boundaries established by the current ensemble matrix index. This constructs a perfect "apples-to-apples" comparison avoiding overlapping timeframe distortion.

## Full Parameter List

| Parameter | Default | Description |
|------|--------|------|
| `--training-mode` | `static` | Filter models by mode (e.g. `static` or `rolling`) |
| `--record-file` | `latest_train_records.json` | Train records pointer targeting model manifests. |
| `--max-combo-size` | `0` (All) | Upper limit of combined models (Or clusters if grouped). |
| `--min-combo-size` | `1` | Lower limit of combined models (Or clusters if grouped). |
| `--freq` | `None` | Backtest frequency (`day` / `week`). Default: read from `strategy_config.yaml` |
| `--top-n` | `50` | Scale N target for Top/Bottom analysis metrics. |
| `--output-dir` | `output/ensemble_runs` | **Output root directory** (each run auto-creates a `{script}_{date}/` subdirectory beneath it). |
| `--resume` | - | Ingest target logic for crash recovery execution. |
| `--n-jobs` | `4` | Parallel concurrent simulations. |
| `--batch-size` | `50` | Combo count per persistence bucket (Impacts RAM intensity and checkpoint intervals). |
| `--use-groups` | - | Activate manual exclusionary subset routing (selects at most one model per defined group block). |
| `--group-config` | `config/combo_groups.yaml` | Overriding path for groupings definitions. |
| `--norm-method` | `rank` | Cross-sectional normalization: `rank` (percentile [0,1], recommended) or `zscore`. |

## Output Files

All outputs are stored in a **per-run subdirectory**, with IS search and OOS validation artifacts organized in separate layers:

```text
output/ensemble_runs/
└── brute_force_2026-04-03/         # Per-run directory (script + anchor_date)
    ├── run_metadata.json            # Run config and date splits (for the analyzer)
    ├── summary.md                   # One-page summary: IS/OOS key metrics + file index
    ├── is/                          # In-Sample artifacts
    │   ├── results.csv              # Backtest results (core file)
    │   ├── correlation_matrix.csv   # Prediction correlation matrix
    │   ├── analysis_report.txt      # IS comprehensive evaluation report
    │   ├── model_attribution.csv    # Model attribution table
    │   ├── model_attribution.png    # Attribution bar chart
    │   ├── risk_return_scatter.png  # Risk-return scatter plot
    │   └── cluster_dendrogram.png   # Hierarchical clustering chart
    └── oos/                         # Out-Of-Sample artifacts
        ├── oos_multi_analysis.csv   # Multi-dimensional candidate pool OOS results
        ├── oos_report.txt           # OOS synthesis report
        └── oos_risk_return.png      # OOS risk-return scatter plot
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
python quantpits/scripts/brute_force_ensemble.py --min-combo-size 4 --max-combo-size 6
```

### Daily Simulation Mapping
```bash
python quantpits/scripts/brute_force_ensemble.py --freq day
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
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 5

# Ctrl+C applied here -> Wait for current thread sync, outputs termination prompt
# "⚠️ Interrupted safely! Completed: X/Y combos. Utilize --resume next session."

# Step 1: Train targeted algorithms
python quantpits/scripts/static_train.py --full
 --max-combo-size 5 --resume
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
python quantpits/scripts/brute_force_ensemble.py --use-groups

# Pass alternate YAML group routing definitions
python quantpits/scripts/brute_force_ensemble.py --use-groups --group-config config/custom_groups.yaml

# Filter combinatorial lengths utilizing bounded sizes across group capacities
python quantpits/scripts/brute_force_ensemble.py --use-groups --min-combo-size 3 --max-combo-size 4
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
python quantpits/scripts/brute_force_fast.py --max-combo-size 3

# Unrestricted combinations
python quantpits/scripts/brute_force_fast.py

# ================================
# Premium Mitigation: Out-Of-Sample Chronology Splits
# Isolates combinations via purely In-Sample windows filtering the most recent year.
# Verifies dynamically selected Top 10 ensembles on the completely blind forward step.
# ================================
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1

# 2. Dispatch decoupled analyzer filtering dynamic ensembles generating autonomous OOS metrics.
python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_<date>/run_metadata.json

# Push compute layer to GPU architectures
python quantpits/scripts/brute_force_fast.py --use-gpu


# Crash recovery
python quantpits/scripts/brute_force_fast.py --resume

# Filter using Group limits targeting faster subset scaling
python quantpits/scripts/brute_force_fast.py --use-groups --group-config config/combo_groups_20.yaml
```

### Deviations Across Simulations

| Layer | Standard (Qlib Backend) | Vectorized Fast |
|------|:---:|:---:|
| Limit Up/Down Discards | ✅ | ❌ |
| Trade Costing Assumptions | ✅ Precision Check | ⚠️ Proxied via Aggregate Turnover Rates |
| TopK+Dropout Mechanics | ✅ Implemented | ⚠️ Simplified TopK only |
| Cash Reinvestment Logic | ✅ Dynamic Volume | ❌ Pure equal-balance assumptions |
| Time Constraint | ~5s/combo | ~0.001s/combo |

> **Note on Precision Deviations**: There are significant discrepancies between the fast version and the formal exhaustive search (since the fast version cannot smoothly handle positional states, leading to substantial errors for transactions that do not involve full-position swaps). Please use with caution.

### Optimized Workflow Progression

1.  **Broad Net**: Engage `python quantpits/scripts/brute_force_fast.py` scaling all exhaustive combinations (Completes rapidly).
2.  **Precision Audit**: Extract top 10/20 groupings from the prior log, pushing them specifically against `brute_force_ensemble.py` ensuring deep realistic metrics.
3.  **Execution Delivery**: Cement the validated combination scope into `ensemble_fusion.py` driving trading signal distribution.

### Dedicated Fast Arguments

| Argument | Built-In | Core Intent |
|------|--------|------|
| `--batch-size` | `512` | Massive vector matrix chunking sizes across array layers. |
| `--use-gpu` | - | Routes memory allocation to CuPy drivers scaling hardware acceleration. |
| `--no-gpu` | - | Forces host device array calculations. |
| `--cost-rate` | `0.002` | Friction proxies tied to turnover deviations mapping bidirectional 0.2%. |

> Auxiliary bounds (`--max-combo-size`, `--resume`, `--use-groups`, `--group-config`, `--exclude-last-years` etc.) translate perfectly over from standard topologies.

---

## 🕰️ Hardened Paradigms: Preventing In-Sample Overfitting

While brute-forcing, programmatic optimization guarantees immense susceptibility against **In-Sample Overfitting** (Forming subsets identically geared exclusively toward the simulation period logic, crumbling upon forward deployment instances).

By employing `brute_force_ensemble.py` and `brute_force_fast.py` coupled with **Dynamic Relative Time Fencing**, the architecture quarantines forward temporal data retaining it as an explicit **Out-Of-Sample (OOS) Baseline Evaluator**:

### Validated Chronology Filtering

```bash
# 1. Search subsets generating outputs immediately scored on chronological OOS blind data.
# --exclude-last-years 1: Fences the latest year entirely outside algorithm views processing 2-year prior spans as base.
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1

# 2. After completion, the script will print the metadata path, e.g.:
#   ✅ Metadata saved: output/ensemble_runs/brute_force_fast_2026-04-03/run_metadata.json
#   Run the following to analyze and validate OOS:
#     python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_2026-04-03/run_metadata.json

# 3. Dispatch decoupled analyzer filtering dynamic ensembles generating autonomous OOS metrics.
python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_<date>/run_metadata.json
```

Following combinatorial search, routing the exported JSON metadata payload into `analyze_ensembles.py` automatically constructs multidimensional candidate pools ensuring absolute unbiased OOS execution mapping scattering plots and evaluation reports.

### Analysis and Evaluation Artifacts

Upon execution, the script generates comprehensive visualizations and context-rich documents in the `is/` and `oos/` subdirectories of the run folder, plus an auto-generated `summary.md` at the root:
- **IS Evaluation Report (`is/analysis_report.txt`)**: Distilled IS insights per strategy pool (Yield, MVP, Diversity) — annualized returns, max drawdown, Calmar.
- **Risk Return Scatter Dashboard (`is/risk_return_scatter.png`)**: All IS combos plotted on a risk-return 2D plane with correlation fit overlay.
- **Prediction Cluster Dendrogram (`is/cluster_dendrogram.png`)**: Ward-distance hierarchical map identifying homogeneous model clusters.
- **Model Attribution Histogram (`is/model_attribution.png`)**: Cross-frequency chart comparing sub-model frequencies in Best vs. Worst performance buckets.
- **OOS Report (`oos/oos_report.txt`)**: Real-world blind OOS evaluation results for all candidate pools.
- **OOS Validation Scatter (`oos/oos_risk_return.png`)**: Risk-return plot tracking candidate performance on the untainted OOS domain.
- **Summary (`summary.md`)**: IS/OOS key metrics + run config + file index — viewable directly in VS Code preview.

### Analyzer Parameters

Beyond `brute_force_fast.py`, the post-generation `analyze_ensembles.py` script offers precise tuning over candidate pool extraction:

| Flags | Control Action |
|------|------|
| `--top-n N` | Dictates the universal baseline threshold of Top N combos hauled per strategic pillar into OOS (Default: 5). |
| `--top-n-yield`, `--top-n-robust`, etc. | Explicitly overrides Top N thresholds for granular pillars. Supported flags: `-yield`, `-robust`, `-defensive`, `-mvp`, `-diversity`. |
| `--training-mode MODE` | Enforces a strict pattern matcher culling any combinatorial sequences inclusive of model derivatives not terminating with `@MODE` (e.g., `static` or `incremental`) prior to OOS scoring. |
| `--max-workers N` | Threads to utilize for OOS backtest evaluations (Default: 4), scaling dramatically accelerates processing across extensive evaluation pools. |

### Date Modifiers

| Overrides | Contextual Range |
|------|------|
| `--exclude-last-years N` | Quarantines most recent chronological N years establishing dynamic IS boundaries prioritizing blind testing sets. |
| `--exclude-last-months N` | Replicates utilizing discrete month bounds. |
| `--start-date YYYY-MM-DD` | Absolute forced start. |
| `--end-date YYYY-MM-DD` | Subversive forced stop bounds deleting all subsequent rows. |

> Warning: Trading data is continually rolled per schedule natively. Implementing dynamic `exclude-last-years` / `exclude-last-months` fences inherently aligns testing without needing static arbitrary data edits per epoch.
> Note: Since prediction data rolls over time (added according to frequency), it is recommended to use `exclude-last-years` or `exclude-last-months` to dynamically keep the latest samples stripped, avoiding manual modification of absolute dates.

---

### GPU Parallel Access

Engaging CuPy drivers natively scales architectures to parallel acceleration:

```bash
# CUDA 12.x Install Path
pip install cupy-cuda11x # or cupy-cuda12x, depending on your CUDA version

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

Fast mode outputs share the same structure as standard mode, stored under `output/ensemble_runs/brute_force_fast_{date}/`:

```text
output/ensemble_runs/brute_force_fast_2026-04-03/
├── run_metadata.json               # Run config payload (for decoupled analyzer)
├── summary.md                      # One-page summary (populated after analyze_ensembles)
├── is/
│   ├── results.csv                 # Vector simulation metrics (core file)
│   ├── correlation_matrix.csv      # Inter-prediction linkage
│   ├── analysis_report.txt         # IS comprehensive report
│   ├── model_attribution.csv/.png  # Attribution analysis
│   ├── risk_return_scatter.png     # IS scatter plot
│   └── cluster_dendrogram.png      # Clustering chart
└── oos/
    ├── oos_multi_analysis.csv       # [Analyzer Output] Multidimensional OOS candidate metrics
    ├── oos_risk_return.png          # [Analyzer Output] OOS risk-return scatter plots
    └── oos_report.txt               # [Analyzer Output] OOS synthesis analytics report
```

> **Migrating legacy data**: To migrate existing flat files from `output/brute_force/` and `output/brute_force_fast/` into the new structure:
>
> ```bash
> # Preview
> python quantpits/scripts/migrate_ensemble_outputs.py --dry-run
>
> # Execute (original directories are preserved — clean up manually after verification)
> python quantpits/scripts/migrate_ensemble_outputs.py
> ```

---

## Next Steps: Configuring Ensemble Fusion

Once you have completed the ensemble search and analysis, move to the formal Ensemble Fusion phase.

### Step 1: Review the Analysis Reports

Identify top-performing combinations from your run directory:

```bash
# One-page summary (IS/OOS rankings + file list)
cat output/ensemble_runs/<run_dir>/summary.md

# Detailed OOS validation report
cat output/ensemble_runs/<run_dir>/oos/oos_report.txt
```

### Step 2: Add Selected Combos to Configuration

Copy the model names from the report into `config/ensemble_config.json`:

```json
{
  "combos": {
    "combo_A": {
      "models": ["gru", "linear_Alpha158", "TabNet_Alpha158"],
      "method": "equal",
      "default": true,
      "description": "High Calmar combo (from OOS validation)"
    },
    "combo_B": {
      "models": ["alstm_Alpha158", "linear_Alpha158", "sfm_Alpha360"],
      "method": "icir_weighted",
      "default": false,
      "description": "Low-correlation diversified combo"
    }
  },
  "min_model_ic": 0.00
}
```

> [!TIP]
> We recommend configuring 2-3 candidates with different strengths (e.g., high return vs. high robustness). These selections can be used for many cycles without needing to re-search.

### Step 3: Run Formal Fusion Backtests

```bash
# Run all combos and generate inter-combo comparison
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

> For detailed config formatting and instructions, see [03_ENSEMBLE_FUSION_GUIDE](03_ENSEMBLE_FUSION_GUIDE.md).

---

## Known Limitations (Future Work)

### Rank Granularity Mismatch Across Coverage Sizes

In rank mode, a model covering 300 stocks produces rank steps of `1/299`, while a model covering 500 stocks produces steps of `1/499`. Coarser-grained model signals get diluted by finer-grained quantization noise during averaging. Future versions may weight models by coverage breadth.

### NaN → 0.5 Abstention Assumption

In rank mode, NaN values (stocks not covered by a model) are filled with 0.5 (neutral). This assumes all models are equally "uncertain" about stocks they do not cover. When coverage is systematically biased (e.g., a model only covers large-cap stocks), filling 0.5 for small-caps may be overly optimistic or pessimistic.

### Fast Path NaN Handling Inconsistency

`brute_force_fast.py`'s `load_predictions` applies a global `.dropna()` (intersection) at merge time, while the standard path (`predict_utils.py`) preserves the union and only fills missing cells with 0.5. The same combo evaluated on different paths will produce results that are not directly comparable. The fast path may be deprecated in a future release.

