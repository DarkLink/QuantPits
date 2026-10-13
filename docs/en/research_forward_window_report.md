# Paper-account ledger and return reports

`quantpits.research.forward_window_report.build_forward_window_report` reads an explicit account’s first and continuing intent/settlement records. It verifies sources/predecessors and displays actual accounting records, late generation, late settlement and explicit HOLD. Original stdout and strict prospective qualification are not required. It does not replan, settle or update Production.

## Request and CLI

Supply UTF-8 JSON with exactly these required fields. Extra fields, duplicate keys, and non-finite numbers are rejected. `schema_version` is integer 1; cycle indices are strict integers 1..N with strictly increasing dates. Existing readers enforce canonical physical paths and private permissions (0700 directories, 0600 bundle members). No latest discovery occurs. All four roots must be explicit, even for a first-cycle-only request; only requested slots are observed.

```json
{
  "schema_version": 1,
  "first_intent_store_root": "/private/research/first_intents",
  "first_settlement_store_root": "/private/research/first_settlements",
  "continuing_intent_store_root": "/private/research/continuing_intents",
  "continuing_settlement_store_root": "/private/research/continuing_settlements",
  "epoch_id": "example.epoch",
  "requested_cycles": [
    {
      "cycle_index": 1,
      "current_cycle_id": "2026-09-04",
      "expected_intent_request_digest": null,
      "expected_settlement_request_digest": null
    },
    {
      "cycle_index": 2,
      "current_cycle_id": "2026-09-11",
      "expected_intent_request_digest": null,
      "expected_settlement_request_digest": null
    }
  ]
}
```

Replace digests with known lowercase SHA256 identities from original records. Null means no expected identity is bound: an existing slot becomes `UNBOUND_PRESENT`, never an automatically selected source for returns. An absent slot records observed absence. Without a verified intent, the requested date has `date_verified=false`.

```bash
# No workspace activation required; safe stdout summary only
python -m quantpits.scripts.report_forward_window --request-file /private/window-request.json

# Explicitly authorized private derived output
python -m quantpits.scripts.report_forward_window \
  --request-file /private/window-request.json \
  --output-dir /private/reports/window-001
```

Stdout excludes paths, epoch IDs, securities, amounts, and raw exceptions. The output parent must exist. The target must be absent or empty with mode 0700. Symlink paths, source descendants or ancestors, and nonempty targets are rejected. Files are created with mode 0600, without overwrites or automatic failure cleanup. On failure, `written_files` lists only files verified through their canonical names; `attempted_files` identifies attempted files, which may exist partially, and `output_state=PARTIAL_OR_UNCERTAIN`. Process-control interruptions propagate and may leave derived files behind.

| Status | Exit | Meaning |
| --- | --- | --- |
| COMPLETE | 0 | Every requested pair settled and joined, complete NAVs, valid common base |
| PARTIAL | 2 | Waiting, unbound identity, missing members, PARTIAL NAV, or invalid base |
| BLOCKED | 4 | Invalid request, conflict, broken chain, or uncertain source observation |
| OUTPUT_FAILED | 5 | Not all output completed; inspect attempted/verified file facts |

An ordinary row failure does not suppress later observations. An internal missing predecessor produces `CHAIN_BREAK` when the next settled pair cannot join it. Waiting does not claim exchange data has not arrived; current wall time does not invent a schedule. Source mutation during observation marks the window `UNCERTAIN`. Chain and original-record completeness remain separate from `prospective_claim=false/epoch_started=false/promotion_capability=false`.

## Valuation and comparison

The fixed rule is `FIRST_NEXT_OPEN_PRETRADE_BASE_TO_POSTTRADE_SAMPLES_V1`. Base B is both arms' identical, complete, positive first next-open **pretrade** `nav_before`; normalized start is explicitly 1. Every later sample is the cycle's next-open **posttrade** `nav_after`, at its frozen intent timestamp.

- Normalized NAV is post/B; window return is final post/B−1.
- First return is post/B−1; later returns are current post/direct predecessor post−1, not current post/pre.
- Challenger−champion excess return subtracts the two returns; curve difference subtracts normalized NAVs.
- Drawdown uses the running sample maximum including initial 1. It is discrete next-open drawdown, not daily or intraday maximum drawdown.
- Cost is actual fee+slippage, divided by B for cost rate. NAV already includes costs: never subtract them again. Gross turnover is `(gross_buy+gross_sell)/current nav_before`, without dividing by two. Cumulative turnover sums the complete requested periods.

Golden example: B=100, post1=99, pre2=110, post2=109 gives normalized 0.99/1.09, window return 0.09, second return 109/99−1, and first drawdown −0.01. Cross-period price changes enter subsequent post values and must not disappear through a post/pre formula.

Missing cycles, broken links, or any PARTIAL NAV make full-window returns, maximum drawdowns, and return difference null with reasons and all original rows preserved. Local points must connect to the original base; the report never changes its starting point. After a partial valuation, independently complete normalized points and directly adjacent complete returns may remain available, but incomplete sample history prevents further local drawdown. Actual costs may aggregate with complete accounting coverage; turnover requires every denominator to be complete and positive. An invalid base nulls normalization and returns while retaining cash and execution diagnostics.

Zero or negative post NAV remains visible with `NONPOSITIVE_NAV`. A positive B still permits normalization and drawdown below −1; a subsequent return with a nonpositive denominator is null. Losses are not clipped or rewritten as zero returns.

Overlap uses actual ranking Top-K, BUY/SELL instruments, after-state holdings, and quantities. K comes from the frozen definition. Set comparisons contain intersection/union counts and Jaccard; two empty sets yield 1 with `empty_both=true`. Identical arms and legitimate zero orders remain valid. The report preserves price provenance, execution assumptions, frozen definitions, and actual continuation references. Corporate actions are UNMODELED and external cash flows are excluded. Synthetic execution assumptions do not describe broker execution attribution. Bootstrap-to-first-open holding returns are outside the window. Original technical records support local time observations only, never proof that a synthetic fixture was a real forward run.

## Timing and HOLD disclosure

Each row’s accounting_time retains preparation start, actual creation/data verification, settlement and source verification timestamps separately. Facts absent from old records remain null. AFTER_OPEN_SIMULATION depends on original creation/verification versus original opening; late settlement does not reclassify earlier prospective stock selection. settlement_delay_seconds is actual elapsed time after opening, not an unlock gate. Without stdout, original_success=LOCAL_BUNDLE_VERIFIED_WITHOUT_STDOUT does not fabricate historical commit success.

Requests enumerate saved records; skipped trading weeks receive no invented rows. JSON, CSV and Markdown display HOLD and the caller’s holding assumption. period_start_trade_date/period_end_trade_date describe actual adjacent valuation endpoints. Interval returns and actual execution costs remain available across HOLD, but missing intermediate valuations make max_drawdown null with HOLD_INTERVAL_NOT_SAMPLED/DRAWDOWN_HISTORY_INCOMPLETE. COMPLETE refers to the requested ledger and endpoint valuations, not full daily valuation through HOLD. Unmodeled corporate-action disclosure remains.

## Outputs and identities

- `report.json`: all requested rows, actual request/manifest/operation references and member raw digests, cross-bundle joins, `source_context`, arm metrics, reasons, and capabilities.
- `cycles.csv`: exactly CHAMPION and CHALLENGER rows per actual request. Existing amount/ratio columns remain; appended fields are `created_at_utc,generation_relation,settled_at_utc,settlement_delay_seconds,hold_start_date,hold_end_date,hold_assumption,hold_reason,period_start_trade_date,period_end_trade_date`. Null is empty; reasons use `|`.
- `report.md`: requested scope, common base, gaps, returns/costs, and per-arm samples, with explicit nulls. Detailed overlap and price provenance are in JSON.

Calculations use a local high-precision Decimal context. Amounts are deterministic decimal strings; ratios use 24 fractional digits with ROUND_HALF_EVEN and trailing zeros removed. Rounding never feeds subsequent calculations. JSON/CSV/Markdown share numeric strings. `request_digest` excludes local roots; `semantic_digest` excludes `private_bindings` and `report_generated_at`, retaining actual source identities and implementation SHA256 values. Editing report JSON grants no publication or reuse authority. Legacy first-cycle v1 bundles do not require absent D2 fields; existing readers validate continuing v2 physical bindings and schedules.

D3 preserves production and execution source observations in each `report.json` row at `source_context.input_provenance.source_change`. Markdown shows per-cycle comparisons and unavailability reasons; legacy records show NOT_RECORDED. Source observations do not downgrade verified accounting results to PARTIAL.
