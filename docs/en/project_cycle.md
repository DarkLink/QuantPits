# Project cycle execution

`python -m quantpits.scripts.run_cycle` coordinates one weekly cycle across existing actual/paper workspaces. It reuses the [account comparison](account_comparison.md) group file and native domain commands. Static producers use predict-only with explicit models and fixed fit records; this entry never starts a fit.

Keep the private project and its `cycles/` directory outside all account/provider trees. Initialize paper accounts with their existing bootstrap command first. A synthetic project:

```yaml
version: 1
group: group.yaml
signals:
  shared:
    producer: live
    models: [model_a, model_b]
    fit_records: data/training_sources/selected_fit.json
    combo: chosen
    norm_method: rank
  candidate:
    producer: paper_b
    models: [model_a, model_b]
    fit_records: data/training_sources/selected_fit.json
    combo: chosen
accounts:
  live: {signal: shared}
  paper_a: {signal: shared}
  paper_b: {signal: candidate}
```

Use the unchanged comparison group schema (`name`, `members` with `id`, `kind`, `workspace`, optional `mlflow_uri` and `qlib_data_dir`). Set explicit backends and providers for execution. Each member must have a binding; each producer workspace supplies one signal group, shared by any number of consumers. Models are registry names without `@static`; the selected native combo must resolve exactly that set. Group/rule paths resolve beside the project; fit paths resolve within the producer workspace. Each child receives explicit workspace, tracking URI, provider and region, overriding inherited environment values.

## Dates and execution

Choose `since < value-through <= anchor < trade-date`. Anchor fixes prediction coverage; value-through advances old accounts; trade-date is the selected calendar session, including a non-default session. Future execution prices are not required to save a new plan.

Optional cycle inputs:

```json
{
  "actual": {"live": {
    "settlement_bundle": "data/2024-05-13-2024-05-17-table.xlsx",
    "allow_missing_settlement": false
  }},
  "paper_cycles": {"paper_a": "paper_next_a", "paper_b": "paper_next_b"},
  "hold": [],
  "skip": []
}
```

Settlement bundles use native filenames and paths relative to the actual workspace. Omit the bundle to discover native daily files. Missing evidence waits unless the user explicitly selects the existing empty-settlement assumption. `paper_cycles` maps new plan IDs; old pending plans always settle under their original journal IDs. HOLD advances old accounts and valuations while omitting new orders/plans. Skip accepts explicit step IDs from plan; dependent steps remain waiting.

```bash
python -m quantpits.scripts.run_cycle plan --project /path/to/project.yaml --cycle week_one \
  --anchor 2024-05-17 --value-through 2024-05-17 --trade-date 2024-05-21 \
  --since 2024-05-10 --inputs /path/to/inputs.json
# Once workspace writes are authorized, run with the same arguments:
python -m quantpits.scripts.run_cycle run --project /path/to/project.yaml --cycle week_one \
  --anchor 2024-05-17 --value-through 2024-05-17 --trade-date 2024-05-21 \
  --since 2024-05-10 --inputs /path/to/inputs.json
python -m quantpits.scripts.run_cycle status --project /path/to/project.yaml --cycle week_one
python -m quantpits.scripts.run_cycle resume --project /path/to/project.yaml --cycle week_one
```

Plan/status create no backend, account locks or project progress. Predictions/fusion awaiting generation have no invented recorder identity. Run/resume save progress, step logs and existing comparison reports under `cycles/<id>/`. Exit codes: 0 complete/plan, 2 waiting or failed steps, 1 configuration error, 130 interruption.

## Sources and recovery

Standalone commands also support `static_train --anchor YYYY-MM-DD` and `order_gen --anchor YYYY-MM-DD --trade-date YYYY-MM-DD --ensemble-record-file <fixed.json>`. The explicit ensemble file retains `combos/default_combo` and may provide `predictions.<combo>.path/sha256` for cross-workspace reads. A missing selected combo fails explicitly. Omitting the new options preserves legacy defaults; `--record-file` continues to select model records.

Prediction snapshots require successful native manifests, all requested members, and native publication receipts. Records come from the original retained transaction postimage, even when latest has changed. Fusion exports the exact manifest recorder at anchor. Producers store fixed fit/prediction records, an ensemble mapping, anchor CSV and evidence under `output/cycles/<project-key>/<cycle>/<signal>/`. Consumers read these files; the coordinator never temporarily replaces `config/ensemble_records.json` to select a source. Native fusion retains its normal publication behavior.

Missing broker evidence blocks only the actual account's new orders. Independent paper accounts continue. Recovery adopts verified completed native results, resumes unfinished training under the original run/state, recovers native actual transactions and uses existing paper journals. Interrupts reach children and the parent waits for their exit. Report-only failures regenerate derived reports. Completed prediction reuse does not depend on current account cash or source-code hashes; unfinished native runs retain their exact recovery contracts.

Use a new cycle ID for changed dates, members or sources. Existing paper plans remain frozen; use native explicit cancel when needed. Never remove native state, transactions or leases to bypass recovery.

Paper advancement saves a separate native close valuation for each trading date after the saved account frontier through `value-through`. Dates before a pending execution use the old positions; on its execution date the original cycle settles before the close valuation. No-activity and HOLD accounts still receive daily valuations. Holidays create no rows. `since` is the report baseline; historical gaps before the frontier are retained, and later positions are never used to backfill earlier values. New plans depend on every required advancement date.

Known future calendar dates may be planned while actual data waits. An unavailable calendar retains the original frontier and pending IDs until the same request can be expanded. A failed or stale date holds subsequent dates for that account; independent accounts continue. Resume adopts completed native dates before proceeding. Previously saved cycles retain their original recorded scope.

## Reference rules

Set project `rules: dated_rules.json`, optional per-account `rules`, or `pending_rules: {original_cycle_id: dated_rules.json}`. Explicit files use native paper rule schema and are frozen when consumed.

Alternatively, `prepare_rules: normal_session.json` uses PaperMarket anchor cash closes and records observation receipts. Members must share a provider. The template must declare `reference_assumption: anchor_close_for_normal_session`, a source description, board assumptions and instrument exceptions. Existing instrument reference closes take precedence. This is an explicit normal-session assumption; missing prices retain native local handling, and whole-session data gaps wait. It does not modify market data.

For detailed differences, separately run `compare_accounts --explain` with the same group and valuation dates. The cycle produces existing financial comparison reports and native account suggestions by default.
