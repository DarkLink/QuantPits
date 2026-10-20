# Read-only account groups

`compare_accounts` presents saved facts from independent actual and paper workspaces in one report.
It needs no workspace activation, initializes neither Qlib nor MLflow, and runs no training, post-trade,
or paper operations. It does not register, migrate, or merge accounts.

## Configuration and command

The group is an index stored at any explicitly selected path. Keep private configurations and reports
under ignored `brain/account_groups/` directories; do not commit private paths, identities, or balances.
This example uses synthetic paths only:

```yaml
name: example_comparison
members:
  - id: live
    label: Actual
    kind: actual
    workspace: ../live_workspace
    notes: Includes actual execution friction and entitlements.
  - id: old_paper
    label: Old paper
    kind: paper
    workspace: ../old_paper_workspace
  - id: new_paper
    label: New paper
    kind: paper
    workspace: ../new_paper_workspace
```

Member IDs must be unique. Declaration order and all requested members are retained, including failed
or incomplete members. Shared models, starting states, or workspaces do not merge accounts. Labels are
for display: account identity comes from the actual trade ledger's `资金账号` or the validated paper journal.
An absent or ambiguous identity remains unknown.

Relative paths resolve beside the **group file**, and `~` is supported. Optional `mlflow_uri` preserves
an explicit context without creating a client or querying its backend. Optional `qlib_data_dir` declares
a protected provider root, also relative to the group file. The default is `~/.qlib/qlib_data/cn_data`;
provider roots recorded in saved configuration or paper sources are protected too. Inherited
`QLIB_WORKSPACE_DIR` and `MLFLOW_TRACKING_URI` do not choose accounts.

```bash
python -m quantpits.scripts.compare_accounts \
  --group /path/to/group.yaml \
  --since 2024-04-05 --date 2024-04-12 \
  --output /path/to/comparison_reports
```

Both exact saved dates are required in `YYYY-MM-DD` format, with the start preceding the end.
`--output` is required. The directory and final files, after symlink resolution, must remain outside
every member workspace and known provider. Adding another account only requires another group member.

## Reports and accounting

Five files share the prefix `comparison_<since>_<date>`:

| File | Contents |
| --- | --- |
| `.md` | Finance, current progress, holdings comparisons, sources, and gaps |
| `.json` | Every requested member, endpoint positions, sources, available dates, errors, and input SHA-256 hashes |
| `_summary.csv` | Per-account NAV, cash, shares/position count, funding, P&L, return, and fees |
| `_holdings.csv` | Per-stock quantities, values, NAV weights, and differences; unavailable members stay blank |
| `_pairs.csv` | All account pairs, common/exclusive holdings, quantity differences, and weight overlap |

Actual NAV and external funding come from `data/daily_amount_log_full.csv`, using `收盘价值` and
`CASHFLOW`. Cash/positions come from the selected day's `holding_log_full.csv`; period fees use
`trade_log_full.csv`'s `费用合计`. Raw broker XLSX files and latest configuration holdings are not historical
accounting inputs. Missing, empty, or nonfinite cashflows/fees remain unknown. NAV can still be shown
when cashflows are unknown, while P&L and return stay blank.

Paper data passes `PaperStore.read()` validation. Endpoint NAV and positions come from the last saved
valuation at the exact date, including bootstrap valuation. A processing/trade date earlier than the
valuation date is valid. Saved `STALE`/`PARTIAL` observations retain their values/status but do not
qualify as reliable period returns or stock-weight comparisons.

```text
Net external funding = external changes with since < booking date <= date
Period P&L = ending NAV - starting NAV - net external funding
Return on starting capital = period P&L / starting NAV
```

Paper funding is the difference in saved valuations' cumulative applied `EXTERNAL` events. Dividends,
interest, and taxes belong to investment P&L. Paper fees include only settlements with
`since < settlement_date <= date`, not lifetime fees. Fees already affect NAV and are not deducted again.
This return is not TWR; nonpositive starting NAV leaves return blank with an explanation.

Requested dates are never shifted to accommodate an account. Missing endpoints or reliable valuations
show recorded available dates and gaps; there is no forward filling or zero NAV fabrication. Actual and
paper assets remain separate rather than becoming an actual total.

Each stock's weight is its selected-date value divided by that account's complete NAV. Weight overlap
is `sum(min(account A stock weight, account B stock weight))`. Cash is separate from stock counts.
Missing holdings/NAV make only the affected comparisons unavailable. CSVs distinguish exclusive stocks
and shared stocks with different quantities; unknown positions are not represented as zero holdings.

## Current progress and historical sources

Current processing/valuation dates, pending cycles/execution dates, and next signals are separate from
the selected historical period. Settled paper sources come from the corresponding frozen plan, and
pending sources from the current frozen plan. Mutable default `signal_source` settings never replace
those historical identities. Bootstrap sources and concise saved lineage summaries are included without
re-reading current signals, deserializing models, or recursively querying MLflow.

Actual accounts inspect existing `order_gen` manifests, supporting workspace names and absolute paths.
A successful manifest plus valid saved suggestion CSVs can show “suggestions generated.” That records
the suggested source, not proof of broker orders, fills, or complete pipeline success. Broker pending
status remains unknown. Unrecorded sources are shown as unknown without hiding balances.

Actual/paper differences also include execution friction, cash entitlements, and accounting assumptions.
Use member notes for unrecorded dividends/taxes or zero additional slippage. Differences are not pure
model contributions, and notes do not grant or revoke trading eligibility.

## Status and read-only boundary

| Exit code | Meaning |
| --- | --- |
| `0` | `complete`: required selected-period financial/valuation facts were readable for every member and reports were saved; optional sources may remain unknown |
| `2` | `partial`: required facts are missing, corrupt, or unstable for at least one member; all members and available results were saved |
| `1` | Invalid group/dates/output or a save failure; stderr contains a structured error |

`--help` needs no workspace. Ordinary member errors do not discard later members, while process-control
interruptions propagate. Invalid groups fail before account reads.

Stable reads and mutation observation are reused, with a final check of all read inputs and manifest
inventories after the whole group is read. A changed member loses its mixed snapshot and becomes
`unstable`; other members remain visible. No account lock, cache, MLflow database, or member output is
created. Reports commit per file, not as a five-file transaction; a failed save can be rerun.
