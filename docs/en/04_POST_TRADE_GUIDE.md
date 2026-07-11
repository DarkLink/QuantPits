# Post-Trade Batch Processing Guide

## Overview

The unified post-trade command handles three complementary broker evidence streams: settlement, orders, and intraday fills. Its default `--scope all` updates account state and preserves the raw evidence required by execution analytics.

**This script is completely decoupled from prediction, fusion, and backtest modules. It does not possess dependencies on any model output.**

| Script | Purpose |
|------|------|
| `prod_post_trade.py` | Batch processes trading day data to update holdings and capital |
| `prod_post_trade_analytics.py` | Compatibility entry point for order/trade evidence ingestion only |

### Data authority boundaries

- `YYYY-MM-DD-table.xlsx` (settlement) is the sole authority for cash, holdings, fees, and dividends.
- `YYYY-MM-DD-order.xlsx` records execution intent, fill/cancel quantities, and status.
- `YYYY-MM-DD-trade.xlsx` records intraday fill timestamps, quantities, and prices.

These streams are not interchangeable. Execution-quality analysis needs orders and fills, while only settlement may change account state.

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
            ├── YYYY-MM-DD-table.xlsx     # Daily discrete exported trade spreadsheets (Current parsing matched to: Guotai Junan Securities export format)
            ├── YYYY-MM-DD-order.xlsx     # Order evidence
            ├── YYYY-MM-DD-trade.xlsx     # Intraday fill evidence
            ├── emp-table.xlsx            # Legacy helper only; no implicit command fallback
            ├── trade_log_full.csv        # Cumulative holistic trade transaction ledger
            ├── trade_detail_YYYY-MM-DD.csv # Explicit daily trade itemization
            ├── trade_classification.csv  # Trade classification tags (Cumulative: Signal, Substitute, Manual)
            ├── holding_log_full.csv      # Cumulative snapshot holding logs
            ├── daily_amount_log_full.csv # Cumulative capital balance trajectory
            ├── raw_order_log_full.csv    # Cumulative order evidence
            ├── raw_trade_log_full.csv    # Cumulative intraday fill evidence
            └── post_trade_ingestion_state.json # Source fingerprint receipts
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

The current compatibility state workflow reads cashflows but does not archive them automatically. Consumed-only archival and transactional state persistence are intentionally deferred to the state-transaction phase.

---

## Execution Logic

```bash
cd QuantPits

# Standard Operation: Processes the backlog from the previous sync date up against today's bounds.
python quantpits/scripts/prod_post_trade.py --scope all

# Light planning: no Qlib initialization, Excel parsing, or writes
python quantpits/scripts/prod_post_trade.py --explain-plan
python quantpits/scripts/prod_post_trade.py --json-plan

# Override targeted Broker via CLI flag (defaults to config JSON setting, falls back to `gtja`)
python quantpits/scripts/prod_post_trade.py --broker gtja

# Strict dry-run: parses, reconciles, values, and calculates full state without writes.
python quantpits/scripts/prod_post_trade.py --dry-run

# Explicit partial workflows; the plan warns about the skipped authority.
python quantpits/scripts/prod_post_trade.py --scope state
python quantpits/scripts/prod_post_trade.py --scope execution

# Target End Date Override:
python quantpits/scripts/prod_post_trade.py --end-date 2026-02-10

# Verbosity Trace: Ouputs individual transaction ledger items implicitly to stdout
python quantpits/scripts/prod_post_trade.py --verbose
```

Missing settlement evidence is an error by default; the primary command no longer silently substitutes `emp-table.xlsx`. Use `--allow-missing-settlement` only after explicitly confirming no activity. Evidence of fills still blocks that acknowledgement.

Execution ingestion uses source path + SHA-256 receipts in `post_trade_ingestion_state.json`, not a maximum-date cursor. Late historical exports are therefore discovered. Changed content at an already-receipted path fails closed instead of silently mixing corrected rows into history.

`--scope all` uses two independent windows. Settlement discovery starts after the account-state cursor, so historical state is never replayed by a normal run. Unless `--start-date` is explicit, order/trade discovery scans existing historical exports and lets source receipts decide what is pending. For state/all scopes, an explicit start date cannot precede the next state date; historical state replay requires a future audited backfill command.

---

## Procedural Engine

For each processing day, the command builds an immutable state change before persistence:

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

Account arithmetic uses `Decimal`. Partial sales remove average-cost basis: a
100-share position costing 1000 retains cost 600 after selling 40 shares;
proceeds are never subtracted directly from position cost. Negative dividend-tax
adjustments use their actual cash effect.

Every ending position and the benchmark must have a valid close. Missing
valuation fails before the first state write instead of silently dropping a
position or writing a fake zero benchmark.

### Database Descriptions

| Target Element | Contents | Overwrite Protocol |
|------|------|----------|
| `trade_log_full.csv` | Holistic executed ledger database | Append + Deduplication |
| `trade_classification.csv` | Quantitative signal vs Manual trade classification mappings | Regenerated via suggestions |
| `holding_log_full.csv` | Inter-day positional footprint snapshots | Append + Deduplication |
| `daily_amount_log_full.csv` | Aggregate account capitalization tracking | Append + Deduplication |
| `trade_detail_*.csv` | Discrete slice of single day trade logs | Daily Full Overwrite |

### Broker Adapters Framework

The system utilizes a **Broker Adapter** architecture to seamlessly handle formatting deviations (differing headers/rows) originating from alternate broker terminal exports. The central engine standardizes inputs by normalizing datasets into a rigid sequence using unified Schema arrays (enforcing Chinese nomenclature constraints downstream).

**Built-In Options (`brokers/`)：**
* `gtja`: Guotai Junan Securities adapter (Default behavior). Skips empty header prefixes, strips escape tabs, and maintains alignment against expected core schema variables.

To natively tether a new broker into the overarching pipeline:
1. Initialize a discrete driver under `quantpits/scripts/brokers/` extending `BaseBrokerAdapter`.
2. Implement strict `parse_settlement`, `parse_orders`, and `parse_trades` methods that normalize the broker format. Missing files, parse failures, and schema failures must raise typed errors; compatibility `read_*` methods may retain warning-plus-empty behavior.
3. Import and map the newly forged adapter inside `BROKER_REGISTRY` mapped via `brokers/__init__.py`.
4. Trigger workflow sequences utilizing `--broker <your_broker_name>` flags, or bind it persistently inside `prod_config.json`.

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
# Strict preflight: opens and validates Excel inputs but writes nothing
python quantpits/scripts/prod_post_trade.py --dry-run

# Light plan only (does not open Excel)
python quantpits/scripts/prod_post_trade.py --explain-plan

# Dispatch when trace validates correctness
python quantpits/scripts/prod_post_trade.py
```

---

## Parameter Arguments

```text
python quantpits/scripts/prod_post_trade.py --help

Optional Overrides:
  --scope {all,state,execution}
  --start-date TEXT Start date; state/all cannot precede the next state date
  --end-date TEXT   Override cursor target date (YYYY-MM-DD); bypasses fetching current day
  --allow-missing-settlement  Explicitly acknowledge no activity for missing statements
  --dry-run         Full parse, reconciliation, valuation and state calculation; writes nothing
  --explain-plan    Light text plan; no Qlib initialization or Excel parsing
  --json-plan       Light JSON plan; stdout is one JSON payload
  --broker TEXT     Override target Broker sequence behavior mappings
  --run-id TEXT     Optional run identity; excluded from semantic fingerprint
  --verbose         Elevate log thresholds emitting discrete stock purchase actions per slice
```

---

## Important Notices

> [!IMPORTANT]
> This script **strictly processes live trading data**. It is completely independent and decoupled from training (`static_train.py --full`), prediction (`static_train.py --predict-only`), backtesting (`brute_force_ensemble.py`), and other modules.

> [!TIP]
> Inspect `--explain-plan` first, then use `--dry-run` for strict input validation before a real run.

> [!WARNING]
> Name the three exports `YYYY-MM-DD-table.xlsx`, `YYYY-MM-DD-order.xlsx`, and `YYYY-MM-DD-trade.xlsx`. Missing settlement fails by default; no empty template is substituted unless `--allow-missing-settlement` explicitly acknowledges no activity.

> [!WARNING]
> Real state execution currently uses a recoverable cursor-last protocol:
> derived CSV files are atomically replaced one file at a time and
> `prod_config.json` advances last. This is not a multi-file atomic transaction;
> transactional cashflow archival and a crash journal remain follow-up work.
