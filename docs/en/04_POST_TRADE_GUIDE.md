# Post-Trade Batch Processing Guide

## Overview

The Post-Trade script is designed to process live trading execution data: it parses executed broker history exported as files, updating holdings, cash balances, and cumulative logs.

**This script is completely decoupled from prediction, fusion, and backtest modules. It does not possess dependencies on any model output.**

| Script | Purpose |
|------|------|
| `prod_post_trade.py` | Batch processes trading day data to update holdings and capital |

---

## File Structure

```text
QuantPits/
├── quantpits/
│   ├── scripts/
│   │   └── prod_post_trade.py          # This script
│   └── docs/
│       └── 04_POST_TRADE_GUIDE.md        # This document
│
└── workspaces/
    └── <YourWorkspace>/                  # Isolated active workspace
        ├── config/
        │   ├── prod_config.json        # Holdings/Cash/Process state
        │   └── cashflow.json             # Deposit/Withdrawal mapping records
        └── data/
            ├── YYYY-MM-DD-table.xlsx     # Daily discrete exported trade spreadsheets
            ├── emp-table.xlsx            # Null placeholder templates (Used automatically for empty trading days)
            ├── trade_log_full.csv        # Cumulative holistic trade transaction ledger
            ├── trade_detail_YYYY-MM-DD.csv # Explicit daily trade itemization
            ├── trade_classification.csv  # Trade classification tags (Cumulative: Signal, Substitute, Manual)
            ├── holding_log_full.csv      # Cumulative snapshot holding logs
            └── daily_amount_log_full.csv # Cumulative capital balance trajectory
```

---

## Cashflow Configuration

### New Format (Recommended)

`config/cashflow.json` gracefully supports explicit multidate arrays of incoming/outgoing transfers:

```json
{
    "cashflows": {
        "2026-02-03": 50000,
        "2026-02-06": -20000
    }
}
```

- **Positive Integers** = Deposit (Injecting cash into the account entity)
- **Negative Integers** = Withdrawal (Extracting cash externally)
- Entries fire strictly and uniquely bounds to their target matching trade date.

### Legacy Format (Backward Compatibility support)

```json
{
    "cash_flow_today": 50000
}
```

Legacy logic will inject the unallocated integer entirety toward the **first sequential processing day** encountered in the pending batch backlog.

### Post-Processing Archival

Upon processing consumption, executed entries inside the `cashflows` key are safely serialized back out into the `processed` nested bounds:

```json
{
    "cashflows": {},
    "processed": {
        "2026-02-03": 50000,
        "2026-02-06": -20000
    }
}
```

---

## Execution Logic

```bash
cd QuantPits

# Standard Operation: Processes the backlog from the previous sync date up against today's bounds.
python quantpits/scripts/prod_post_trade.py

# Preview Mode: Previews what dates and cashflows will trigger chronologically, bypassing IO writes.
python quantpits/scripts/prod_post_trade.py --dry-run

# Target End Date Override:
python quantpits/scripts/prod_post_trade.py --end-date 2026-02-10

# Verbosity Trace: Ouputs individual transaction ledger items implicitly to stdout
python quantpits/scripts/prod_post_trade.py --verbose
```

---

## Procedural Engine

For each processing day chronologically, the script computes boundaries as follows:

```mermaid
flowchart TD
    A[Load Daily Export] --> B[Aggregate Sells]
    B --> C[Aggregate Buys]
    C --> D[Compute Dividends & Interest]
    D --> E[Inject Mapped Cashflows]
    E --> F[Roll Cash Balance Forward]
    F --> G[Reconcile Net Holdings Array]
    G --> H[Query Closing Market Vector]
    H --> I[Estimate Unrealized Fluctuations]
    I --> J[Serialize Daily Logs (CSV)]
    J --> K[Execute Trade Classification Engine]
```

### Capital Rollover Algorithm

```text
cash_after = cash_before + Total_Sell_Value - Total_Buy_Gross + Dividends_Interest + Targeted_Cashflow
```

### Database Descriptions

| Target Element | Contents | Overwrite Protocol |
|------|------|----------|
| `trade_log_full.csv` | Holistic executed ledger database | Append + Deduplication |
| `trade_classification.csv` | Quantitative signal vs Manual trade classification mappings | Regenerated via suggestions |
| `holding_log_full.csv` | Inter-day positional footprint snapshots | Append + Deduplication |
| `daily_amount_log_full.csv` | Aggregate account capitalization tracking | Append + Deduplication |
| `trade_detail_*.csv` | Discrete slice of single day trade logs | Daily Full Overwrite |

---

## Typical Workflow Operations

### Scenario 1: Nominal Routine Update

```bash
# 1. Distribute exported trade statements inside `data/` directory appropriately labelled `YYYY-MM-DD-table.xlsx`
# 2. Open `config/cashflow.json` and declare deposits if necessary
# 3. Synchronize pipeline:
python quantpits/scripts/prod_post_trade.py
```

### Scenario 2: Erratic Mid-Interval Adjustments / Injections

```bash
# Assign cashflows inside cashflow.json
cat config/cashflow.json
# {"cashflows": {"2026-02-03": 50000, "2026-02-06": -20000}}

# Safety check sequence outputs bounds
python quantpits/scripts/prod_post_trade.py --dry-run

# Execute mutations upon clearance
python quantpits/scripts/prod_post_trade.py
```

### Scenario 3: Blind Preview Inspection

```bash
# Trace output sequencing
python quantpits/scripts/prod_post_trade.py --dry-run
# -> Generates STDOUT mapping representing planned file ingestions and injected cash traces.

# Dispatch when trace validates correctness
python quantpits/scripts/prod_post_trade.py
```

---

## Parameter Arguments

```text
python quantpits/scripts/prod_post_trade.py --help

Optional Overrides:
  --end-date TEXT   Override cursor target date (YYYY-MM-DD); bypasses fetching current day
  --dry-run         Print sequence bounds solely; completely suspends JSON/CSV modification mutations
  --verbose         Elevate log thresholds emitting discrete stock purchase actions per slice
```

---

## Important Notices

> [!IMPORTANT]
> This script **strictly processes live trading data**. It is completely independent and decoupled from training (`prod_train_predict.py`), prediction (`prod_predict_only.py`), backtesting (`brute_force_ensemble.py`), and other modules.

> [!TIP]
> Executing `--dry-run` is heavily advocated whenever complex multiday deposits (`cashflow.json`) are utilized prior to confirming data bounds updates.

> [!WARNING]
> The structural file labeling of trading software exports must stringently adhere to `YYYY-MM-DD-table.xlsx`. If an explicit date slice lacks detection, the engine evaluates it by spawning an empty trace template for sequence integrity.
