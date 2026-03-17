# Order Generation Guide

## Overview

`scripts/order_gen.py` is used to generate target buy/sell order suggestions based on ensemble fusion or single-model predictions.

**This script is the final step of the operational workflow.** It must be run after predictive data is available and actual post-trade conditions have been reconciled.

**Workflow Pipeline Placement**: Training → Brute Force → Fusion Backtest → Post-Trade → **Order Generation (This Step)**

| Script | Purpose |
|------|------|
| `order_gen.py` | Generates buy/sell suggestion CSV files natively dependent upon predictions + current holdings |

---

## Quick Start

```bash
cd QuantPits

# 1. Use default ensemble fused prediction (Most common operation)
python quantpits/scripts/order_gen.py

# 2. Use specific single-model prediction (Bypass fusion)
python quantpits/scripts/order_gen.py --model gru

# 3. Dry-run inspection (Exhibits mapping without writing serialization to disk)
python quantpits/scripts/order_gen.py --dry-run

# 4. Verbose outputs examining detailed ranks + multi-model opinions 
python quantpits/scripts/order_gen.py --verbose
```

---

## Configuration Requirements

The script loads unified workspace parameters implicitly leveraging `config_loader` (which aggregates `prod_config.json`, `model_config.json`, and `strategy_config.yaml`). Below are the vital operative boundaries steering execution logic:

- **market**: (Recommended) Target operational universe (e.g., `csi300`, `csi1000`). Customarily bounded centrally inside `model_config.json`. If omitted, the architecture auto-infers data scopes utilizing target prediction limits.
- **benchmark**: (Recommended) Index metric boundaries (e.g., `SH000300`, `SH000852`). Built into `model_config.json`.
- **current_cash**: Current liquid balance limits defining order sizes. Systematically maintained via Post-Trade executing upon `prod_config.json`.
- **current_holding**: Active held component quantities arrays. Systematically maintained via Post-Trade executing upon `prod_config.json`.
- **topk** / **n_drop** / **buy_suggestion_factor**: Operation positioning strategy boundaries natively registered executing inside `strategy_config.yaml`.

> [!NOTE]
> The engine features **automatic resilience**: even if the `market` configuration is mismatched, it will dynamically fetch pricing for all assets identified in the prediction source.

---

## Predictive Data Sources

The processing pipeline prioritizes prediction sources via a hierarchical tier:

| Priority | Parameter | Source | Description |
|:---:|------|------|------|
| 1 | `--model` | Single Model | Auto-loads target native Qlib prediction record |
| 2 | *(Default)* | Ensemble | Loads active `ensemble_records.json` default combo bounds |

### Scenario 1: Nominal Routine Process

```bash
# Operate exclusively utilizing the gru prediction trace
python quantpits/scripts/order_gen.py --model gru

# Operate exclusively utilizing the lightgbm architecture
python quantpits/scripts/order_gen.py --model lightgbm_Alpha158
```



## Execution Logic

```mermaid
flowchart TD
    A[Stage 0: Init + Load config] --> B[Stage 1: Ingest predictions]
    B --> C[Stage 2: Fetch pricing parameters]
    C --> D[Stage 3: Ranking + Position reconciliation]
    D --> D2[Stage 3.5: Multi-model opinion matrix]
    D2 --> E[Stage 4: Generate SELL vectors]
    E --> F[Stage 5: Generate BUY vectors]
    F --> G[Stage 6: Serialize targets to IO bounds]
```

### Anchor Dates

The architecture automatically infers the **most recent previous trading day** according to the Qlib calendar interface, remaining chronologically synchronized with the explicit training parameters.

### Position Reconciliation Formula

1. Sort all active equities descending universally per `score`.
2. Determine Candidate Pool bounds utilizing `TopK + DropN × Buy Suggestion Factor`.
3. If natively held equities currently reside within the Candidate Pool → **HOLD**.
4. If natively held equities reside *outside* the Candidate Pool, extract the absolute worst `DropN` subset bounds → **SELL**.
5. Discovered equities residing within Candidate bounds currently unheld → **BUY (Candidate)**.

### Multi-Model Opinion Matrix (Stage 3.5)

**Feature**: Auto-loads prediction matrices encompassing all active combos alongside single models, aggregating them to deliver comprehensive BUY/SELL/HOLD judgment assessments targeting focus assets (Held + Candidate pools).

Decision Protocols (Matched identically to Stage 3 positioning):
- **order_basis**: Inherently operates using tracking parameters strictly derived from Stage 1 limits (price-merged); guarantees **100% equivalence** to actual triggered permutations.
- **combo_*/model_***: Simulates identically scoped swapping logic mapping individually.

Signal Flags:
- **HOLD**: Equities persisting inside bounds, or those outside boundaries bypassing the worst bounds `DropN`.
- **SELL**: Equities located strictly outside bounds encompassing exclusively the absolute worst positional `DropN`.
- **BUY**: Unheld targets riding topmost rankings bounded symmetrically targeting active Sell quantities.
- **BUY***: Unheld targets assigned strictly as alternates (Substitutes protecting against execution faults e.g. trade suspensions).
- **--**: Unheld components neglected entirely by Top parameters scaling.

Export Targets:
- `model_opinions_{date}.csv` — Matrix grid mapping asset rows dynamically across diverse algorithm opinion structures.
- `model_opinions_{date}.json` — Lexicon rendering combo matrices schemas + index component logic sets + legend mapping blocks.

Activate terminal monitoring mapping appending `--verbose`:
```bash
python quantpits/scripts/order_gen.py --verbose --dry-run
```

#### Visualizing Model Placements

Utilize `plot_model_opinions.py` yielding accelerated ranked line chart extrapolations targeting discrete assets arrayed sequentially across the algorithm framework mapping:
```bash
python quantpits/tools/plot_model_opinions.py
# Natively auto-locates newest mapped artifacts, or explicit targeting: --input output/model_opinions_2026-02-24.csv
```

### Capital Mechanics

```text
Available Capital = Existing Balance + Estimated Sell Reclamation Proceeds + Inter-day Deposits
Lot Budget = Available Capital / Gross Buys Triggered
Bought Shares = floor(Lot Budget / Limit-Up Expected Bounds / 100) × 100
```

---

## Artifact Outcomes

```text
output/
├── sell_suggestion_{source}_{date}.csv       # Sell permutation targets
├── buy_suggestion_{source}_{date}.csv        # Buy permutation allocations
├── model_opinions_{date}.csv                 # Aggregated prediction multi-matrix
└── model_opinions_{date}.json                # Matrix component schemas
```

(`{source}` scales dynamically mapping toward `ensemble`, active algorithm parameters `gru`, or explicit `custom` flags.)

### Order Trajectories

| Field | Meaning |
|------|------|
| `instrument` | Asset code identifier |
| `datetime` | Target operational mapping date |
| `value` | Share count quantity |
| `estimated_amount` | Capital proxy boundary approximations |
| `score` | Native prediction index evaluation |
| `current_close` | Adjusted prevailing closing evaluation price vector |

### Multi-Model Map Columns

| Identity | Schema Reference |
|------|------|
| `instrument` | Matrix indexing asset vectors |
| `order_basis` | Unbiased decision mapping tracking generated action targets perfectly |
| `combo_{name}` | Standalone combo prediction logic trace mappings |
| `model_{name}` | Standalone explicit prediction components tracing bounds |

---

## Complete Parameter Reference

```text
python quantpits/scripts/order_gen.py --help

Optional Overrides:
  --model TEXT             Utilize explicitly defined algorithm (Loaded from Qlib records)
  --output-dir TEXT        Output targets (Default output)
  --dry-run               Terminal outputs solely bypassing file execution
  --verbose               Activate heavy debug/trace components exhibiting rank lists
```

---

### Stage 2: Fetch Pricing Parameters

The script now scans all instruments identified within the prediction data and **automatically** retrieves their latest adjusted full price, limit-up price, and limit-down price. This significantly enhances system robustness, ensuring that correct pricing data is secured for lot size calculation even if the `market` configuration is not perfectly aligned.

## Core Dependencies

> [!IMPORTANT]
> The engine mandates adherence strictly following:
> 1. Accessible Qlib record IDs tracing upstream generator algorithms (Fusion/Prediction mappings).
> 2. Synchronized array variables referencing native parameters scaling inside `config/prod_config.json` denoting absolute capital + asset boundaries explicitly synced (Managed via `post-trade`).
> 3. `config/strategy_config.yaml` stores native topk/n_drop strategy limit bounds explicitly mapping to execution definitions.

> [!TIP]
> Executing `--dry-run --verbose` is heavily advocated preventing critical misallocations prior to actual execution orders mapping.
