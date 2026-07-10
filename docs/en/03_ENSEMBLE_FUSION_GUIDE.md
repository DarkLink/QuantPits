# 03 Ensemble Fusion Guide

## Overview

`scripts/ensemble_fusion.py` is used to conduct fusion prediction, backtesting, and risk analysis on **user-selected model combinations**.

**Multi-Combo Mode Supported**: Define multiple combos in `config/ensemble_config.json`, mark one as `default`, and run all combinations to compare performance simultaneously.

**Workflow Pipeline Placement**: Training → Ensemble Search → Combo Selection → **Fusion Backtesting (This Step)** → Order Generation / Signal Ranking

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

# 5. Explain the execution plan only; do not initialize Qlib or write files
python quantpits/scripts/ensemble_fusion.py --from-config-all --explain-plan
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
| `--freq` | `None` | Backtest frequency: `day` / `week` (Default: read from the workspace merged config / `model_config.json`) |
| `--training-mode` | `None` | Filter models by mode (e.g. `static` or `rolling`); defaults to automatic resolution |
| `--record-file` | `latest_train_records.json` | Train records pointer |
| `--output-dir` | `output/ensemble` | Output directory; relative paths are resolved under the active workspace root |
| `--no-backtest` | false | Skip backtesting execution |
| `--no-charts` | false | Skip chart generation |
| `--start-date` | None | Filter start date YYYY-MM-DD |
| `--end-date` | None | Filter end date YYYY-MM-DD |
| `--only-last-years N` | `0` | Use only the last N years of data (Designed exclusively for OOS testing) |
| `--only-last-months N` | `0` | Use only the last N months of data (Designed exclusively for OOS testing) |
| `--detailed-analysis` | false | Generates a detailed backtest analysis report (similar to production reports) |
| `--verbose-backtest` | false | Enables verbose mode for Qlib backtesting |
| `--norm-method` | `rank` | Cross-sectional normalization: `rank` (percentile [0,1], recommended) or `zscore` |
| `--explain-plan` | false | Print a dry-run execution plan and exit without Qlib initialization or writes |
| `--json-plan` | false | Emit a machine-readable JSON plan; implies dry-run |
| `--run-id` | auto-generated | Set the run ID used by the plan and manifest |
| `--no-manifest` | false | Disable writing `output/manifests/ensemble_fusion/<run_id>.json` during actual execution |

## Dry-run and Run Manifests

Before a real run, inspect the execution plan:

```bash
python quantpits/scripts/ensemble_fusion.py --from-config-all --explain-plan
```

This command only reads workspace configs and train records. It prints the resolved combos, input fingerprints, planned writes, and expensive steps. Planned writes include report CSV/JSON files, the manifest, and NAV/dynamic-weight PNG placeholders when `--no-backtest` / `--no-charts` are not set. It does not trigger the safeguard, initialize Qlib, load recorders, or write to `output/`, `data/`, or `config/`.

Schedulers or CI jobs can use the JSON form:

```bash
python quantpits/scripts/ensemble_fusion.py --from-config-all --json-plan
```

Actual execution writes a run manifest by default:

```text
output/manifests/ensemble_fusion/<run_id>.json
```

The manifest records the `run_id`, plan fingerprint, input config fingerprints, resolved combos, execution status, and result summary. `data/operator_log.jsonl` links the same run with `run_id`, `manifest_path`, and `plan_fingerprint`. Use `--no-manifest` when you need the old no-manifest side-effect profile.

Implementation note: `ensemble_fusion.py` is now a thin CLI adapter. Plan rendering, manifest handling, OperatorLog linkage, and the execution lifecycle live in `quantpits/ensemble/service.py`. Importing the script no longer changes the process `cwd`; during real execution, the service resolves relative paths such as `--output-dir` and `--prediction-dir` under the active workspace root. Single-combo Stage 2-10 orchestration lives in `quantpits/ensemble/pipeline.py`; prediction persistence, fusion ledger writes, deterministic analytics, Qlib backtest execution, risk/leaderboard reporting, and chart generation live in `quantpits/ensemble/persistence.py`, `quantpits/ensemble/ledger.py`, `quantpits/ensemble/analytics.py`, `quantpits/ensemble/backtest.py`, `quantpits/ensemble/risk_report.py`, and `quantpits/ensemble/charts.py`. The script keeps thin same-name wrappers for existing patch/import compatibility.

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

## Configuring Ensembles from Search Results

Once you have identified recommended model combinations through brute force searching (see [02_BRUTE_FORCE_GUIDE](02_BRUTE_FORCE_GUIDE.md)), follow these steps to use them in the production pipeline:

### Step 1: Review search reports
Open the summary report in your run directory: `output/ensemble_runs/{your_run}/summary.md`. Look for combinations that perform well across both In-Sample (IS) and Out-Of-Sample (OOS) metrics.

### Step 2: Update `ensemble_config.json`
Select 2-3 robust combinations and add them to the `combos` section of `config/ensemble_config.json`. 

Example of mapping from `summary.md` to `ensemble_config.json`:
If `summary.md` shows a top combo with models `gru_Alpha158, transformer_Alpha360`, your config should look like:
```json
"combos": {
  "stable_growth": {
    "models": ["gru_Alpha158", "transformer_Alpha360"],
    "method": "equal",
    "default": true
  }
}
```

### Step 3: Verify and Set Default
Run the fusion script to compare all configured combos:
```bash
python quantpits/scripts/ensemble_fusion.py --from-config-all
```
Review the comparisons in `output/ensemble/combo_comparison_{date}.csv` and mark your preferred combo as `"default": true` for subsequent order generation.

> [!TIP]
> Selected combinations are intended to be persistent. You only need to re-run the search phase if you add or remove models, or if the market regime shifts significantly after retraining.


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

If you utilized parameters like `--exclude-last-years 1` during the combo-seeking phase (via `brute_force_fast.py`) to fence off this year's data as OOS, you can leverage the following command to exclusively test pure forward OOS extrapolation performance just before taking the combo live.

> [!IMPORTANT]
> **Comparison of OOS Validation Scenarios**
> | Phase | Tool | Purpose | Source |
> |------|------|------|------|
> | **Search Phase** | `brute_force --exclude-last-years` + `analyze_ensembles.py` | Prevent IS overfitting among **thousands of candidates**. | [02_GUIDE](02_BRUTE_FORCE_GUIDE.md) |
> | **Pre-deployment** | `ensemble_fusion.py --only-last-years` | Final verification of **selected candidates** on OOS data. | This Section |

You can execute performance tests exclusively bounding the recent 1 year OOS trajectory:


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
Stage 1: Load selected predictions + cross-sectional normalization (default: `rank`, shared across combos)
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

Actual execution also updates workspace state files: `config/ensemble_records.json` stores combo → recorder_id mappings and the default pointer, while completed backtests append one record to `data/fusion_run_ledger.jsonl` for downstream deep analysis. Dry-runs (`--explain-plan` / `--json-plan`) do not write these files.

### Single Combo Mode (`--models` or `--from-config`)

```text
output/
├── ensemble/
    ├── ensemble_fusion_config_{date}.json     # Fused configuration state
    ├── correlation_matrix_{date}.csv          # Correlation matrix
    ├── leaderboard_{date}.csv                 # Performance leaderboards
    ├── ensemble_nav_{date}.png                # Net asset value trajectories
    ├── ensemble_weights_{date}.png            # Dynamic weight mapping (dynamic mode)
    └── backtest_analysis_report_{date}.md     # [NEW] Detailed backtest analysis report (--detailed-analysis)
└── manifests/
    └── ensemble_fusion/
        └── <run_id>.json                      # Run manifest
```

### Multi Combo Mode (`--from-config-all` or `--combo`)

```text
output/
├── ensemble/
    ├── ensemble_fusion_config_combo_A_{date}.json
    ├── ensemble_fusion_config_combo_B_{date}.json
    ├── leaderboard_combo_A_{date}.csv       # combo_A performance leaderboard
    ├── leaderboard_combo_B_{date}.csv       # combo_B performance leaderboard
    ├── combo_comparison_{date}.csv           # Tabular cross-reference
    ├── combo_comparison_{date}.png           # Comparative charted trajectories
    └── backtest_analysis_report_{combo}_{date}.md # [NEW] Detailed analysis report for this specific combo
└── manifests/
    └── ensemble_fusion/
        └── <run_id>.json                     # Run manifest
```


> [!NOTE]
> **Understanding Metric Discrepancies: Single Models vs. Ensemble Backtests**
>
> Fusion and brute-force evaluation first apply daily cross-sectional normalization and coverage alignment. The default `--norm-method rank` uses percentile ranks and fills uncovered stocks with neutral `0.5`; explicit `--norm-method zscore` keeps the older Z-Score/intersection semantics. Because normalization, coverage handling, and TopK position bounding differ from raw post-training analysis, single-model backtest results here may reasonably differ slightly from raw metrics viewed through `run_analysis.py`:
> 1. **Isolated Normalization**: Each model is normalized within its own daily prediction universe, so another model's coverage gaps do not skew its score distribution before fusion.
> 2. **Coverage Alignment**: `rank` mode uses the union of model coverages and treats missing coverage as neutral abstention (`0.5`); `zscore` mode keeps NaNs and drops to the current combo intersection at scoring time (`dropna(how='any')`).
> 3. **Benchmarking Alignment**: The sub-model evaluation leaderboard dynamically slices historical records to match the precise temporal boundaries established by the current ensemble matrix index. This constructs a perfect "apples-to-apples" comparison avoiding overlapping timeframe distortion.

## Typical Operations Sequence

```bash
# Step 1: Train targeted algorithms
python quantpits/scripts/static_train.py --full

# Step 2: Exhaust combos targeting highest robustness
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1

# Step 3: Parse results, push selected structures into the configuration
cat output/ensemble_runs/brute_force_fast_*/summary.md
# Adjust config/ensemble_config.json appending multiple combosly

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
| `static_train.py --full` | Train models | configs | `latest_train_records.json` |
| `brute_force_ensemble.py` | Combo Exhaustion | train records | leaderboards |
| **`ensemble_fusion.py`** | **Fusion Backtest** | **Targeted Combo sets** | **Fused Predictions + Risk Matrix** |
| `signal_ranking.py` | Top N Output | Fusion Recorders | Ranked CSV sets |
| `order_gen.py` | Target Execution | Fused Recorders + Current Pos | Buys/Sells + Multi-Model opinions |

---

## Normalization Methods

### rank (default)

Cross-sectional percentile rank normalization, output strictly **[0, 1]**. Recommended for long-only TopK strategies:

- Each model's predictions are ranked within each trading day, then mapped to [0, 1].
- Stocks not covered by a model are filled with **0.5** (abstain/neutral vote).
- Fusion uses the **union** of all model coverages (no stocks lost due to partial coverage).
- Each model gets equal voting power in equal-weight fusion.

### zscore

Classic Z-Score normalization. Preserves model "confidence" magnitude:

- Unbounded output; extreme scores from one model can dominate the fused signal.
- NaN cells (stocks not covered) remain NaN → fusion drops to **intersection** via `dropna(how='any')`.
- Suitable when you want to differentiate model conviction levels.

### OOS Analysis Consistency

`analyze_ensembles.py` automatically reads the normalization method from `run_metadata.json`, ensuring the same normalization strategy is applied to both IS and OOS data.

---

## Known Limitations (Future Work)

### Rank Granularity Mismatch Across Coverage Sizes

In rank mode, a model covering 300 stocks produces rank steps of `1/299`, while a model covering 500 stocks produces steps of `1/499`. Coarser-grained model signals get diluted by finer-grained quantization noise during averaging. Future versions may weight models by coverage breadth.

### NaN → 0.5 Abstention Assumption

In rank mode, NaN values (stocks not covered by a model) are filled with 0.5 (neutral). This assumes all models are equally "uncertain" about stocks they do not cover. When coverage is systematically biased (e.g., a model only covers large-cap stocks), filling 0.5 for small-caps may be overly optimistic or pessimistic.

### Fast Path NaN Handling Inconsistency

`brute_force_fast.py`'s `load_predictions` applies a global `.dropna()` (intersection) at merge time, while the standard path (`predict_utils.py`) preserves the union and only fills missing cells with 0.5. The same combo evaluated on different paths will produce results that are not directly comparable. The fast path may be deprecated in a future release.
