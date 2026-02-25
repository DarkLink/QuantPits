# Signal Ranking Guide

## Overview

`scripts/signal_ranking.py` normalizes fusion prediction scores into a streamlined -100 to +100 recommendation index, generating Top N ranking CSV outputs.

**Optimal for sharing insights with others** — operates entirely decoupled from order generation mechanics and remains completely agnostic to active holding/portfolio data.

**Workflow Pipeline Placement**: Fusion Backtest → **Signal Ranking (This Step)**

---

## Quick Start

```bash
cd QuantPits

# 1. Generate Top 300 rankings parsing the default combo matrix
python quantpits/scripts/signal_ranking.py

# 2. Iterate mappings synthesizing distinct lists sequentially spanning all combos
python quantpits/scripts/signal_ranking.py --all-combos

# 3. Explicitly target a bespoke named combo parameter
python quantpits/scripts/signal_ranking.py --combo combo_A

# 4. Filter explicitly utilizing custom Target bounds
python quantpits/scripts/signal_ranking.py --top-n 500

# 5. Native explicit file overrides parsing custom directories
python quantpits/scripts/signal_ranking.py --prediction-file output/predictions/ensemble_2026-02-13.csv
```

---

## Comprehensive Parameter Overrides

| Argument | Default Set | Description Overview |
|------|-------|------|
| `--combo` | None | Strict explicit named target string |
| `--all-combos` | false | Activates loop parsing spanning identical execution over all tracked combinations |
| `--prediction-file` | None | Direct targeted CSV bounds override |
| `--top-n` | 300 | Extracted bounds index constraint |
| `--output-dir` | `output/ranking` | Directory bounds outputs |
| `--dry-run` | false | Engages stdout processing solely bypassing sequence IO writes |

---

## Scaling Index Formula

```text
1. Import Prediction vectors (Extract `score` matrix indices).
2. Filter constraints retrieving newest temporal boundaries natively.
3. Apply normalization scaling parameters: 
   signal = (score - min) / (max - min) * 200 - 100
   → Absolute Scope Scaling Range: -100 (Maximum negative bounds) ~ +100 (Maximum positive bounds)
4. Order structurally descending evaluating recommendations terminating extraction upon Nth limit.
5. Export serialization: Instrument Identifier Index, Normalized Scaled Score.
```

---

## Artifact Outcomes

```text
output/ranking/
├── Signal_default_2026-02-13_Top300.csv     # Target bound generated from Default configs
├── Signal_combo_A_2026-02-13_Top300.csv     # Discretely bound trace from combo A
└── Signal_combo_B_2026-02-13_Top300.csv     # Discretely bound trace from combo B
```

### Schema Structure

| Key Header | Bounds Profile Focus |
|------|------|
| `Stock_Code` (股票代码) | Asset target identifier index |
| `Recommendation_Index` (推荐指数) | Normalization output constrained bounds bounded -100 to 100 mapped to explicit dual float decimals precision parameters |

---

## Core Dependencies

> [!IMPORTANT]
> Operations strictly mandate sequential outputs produced directly via prior `ensemble_fusion.py` processing loops.
> Trace dependencies natively traverse boundaries located within the `output/predictions/` scope.

---

## Systems Relationship Cross-Reference

| Component Focus | Internal Usage Bounds | Parameter Source | Trace Output |
|------|------|------|------|
| `ensemble_fusion.py` | Compounded Backtesting Checks | Selected Logic Combinations | Merged Multi-Predict CSVs |
| **`signal_ranking.py`** | **Explicit Ranking Matrices** | **Predictive Fusions** | **Top N Scaled Output Metrics** |
| `order_gen.py` | Action Allocations Target Generation | Predictive Fusions + Live Holdings | Buy/Sell Execution Manifests |
