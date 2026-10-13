# Settling the first paper-account orders

Settlement reads the original cash/positions, orders and execution assumption from a complete local intent bundle and uses next-open prices for its original trade_date. Settling in the evening or a later week does not move the trade date to today. No model inference, order replanning or Production update occurs.

## Inputs and read-only preparation

Provide the explicit `intent_store_root / epoch_id` and original request digest. `publication_success_record_path` is optional corroboration. Current inspection verifies the complete request, every data member, manifest, completion and account/source joins. Missing stdout is persisted as `publication_success.json=null` with `source_basis=VERIFIED_LOCAL_INTENT_BUNDLE_V1`; no COMMITTED output is synthesized. Supplied stdout must match the original COMMITTED operation, digests, account and actual records. ADOPTED/VERIFIED cannot masquerade as original publication output.

Use a dedicated physical settlement root with mode 0700; an optional stdout file must have mode 0600. Paths must be absolute without symlink aliases. Owners select the real account and write scope before running these examples.

```bash
python -m quantpits.scripts.settle_forward_intent prepare \
  --intent-store-root /absolute/research/intent_pairs \
  --epoch-id EPOCH_ID \
  --expected-intent-request-digest C4_REQUEST_SHA256 \

  --qlib-provider-root /absolute/qlib/provider \
  --settlement-store-root /absolute/research/settlements
```

`prepare` writes nothing and returns READY with the settlement request digest. The request binds original publication identity, optional stdout bytes, accounting inputs, observed calendar/prices, results and implementation fingerprints. Missing stdout has an explicit local-bundle basis. Completion records this source verification at `source_verified_at_utc` and actual settlement at `settled_at_utc`; it does not invent original final publication timing. Paths, inode, mtime and current time are excluded from the semantic request; price revisions change the request.

Dates come from verified C4 content. Local UTC must have reached the saved next open. The actual `calendars/day.txt` must contain the trade date, preserve the historical prefix through the anchor and retain the original next session. Later calendar extensions are allowed. Before opening, before the trade date enters the actual calendar, or when a nonempty requested union has no valid observation, D1 returns WAITING_FOR_DATA without creating a no-fill settlement.

Open/factor observations cover the union of both arms' holdings and intents, then project to each arm's exact set. `OBSERVED_NEXT_OPEN_IN_REQUESTED_UNION_V1` requires at least one OBSERVED member; remaining missing/invalid members retain their reasons and follow B0 behavior. Unreadable sources and malformed binary layouts or start headers are precondition errors, not missing prices. This does not establish market-wide data completeness. If both arms have no holdings and no intents, the receipt is `NOT_REQUIRED_EMPTY_EXPOSURE`; price observation is unnecessary, but time/calendar checks still apply. Zero orders with holdings still require valuation observations.

## Publication and recovery

Use the same arguments with action `publish` and add `--expected-request-digest D1_REQUEST_SHA256`. Publication freshly observes and computes the request before writing. The only slot is `settlement_store_root / epoch_id`; changing the request never selects a different directory.

The writer uses create-only mkdir, exclusive files, 0700/0600 modes, fsync and canonical-path rereads. It verifies data and manifest before creating completion, then rereads the completed pair. Source guards cover the exact C4 bundle, original success record, actual calendar, and requested open/factor files. An ordinary failure in one arm preserves diagnostics for the other arm but prevents pair publication. Process-control interruptions propagate.

```bash
python -m quantpits.scripts.settle_forward_intent inspect \
  --settlement-store-root /absolute/research/settlements \
  --epoch-id EPOCH_ID \
  --expected-request-digest D1_REQUEST_SHA256
```

Publishing to an existing slot only inspects/adopts it, without reading current C4, provider or log inputs. Never repair, remove, overwrite or retry under another epoch after an incomplete write. Inspect the original slot after failure or interruption. Post-write ordinary failures report UNCERTAIN with `did_write=true`, or null when uncertain, never an invented zero-write result. Interruptions emit no success JSON.

| Status | Exit | Meaning |
| --- | --- | --- |
| READY / COMMITTED / ADOPTED / VERIFIED | 0 | Prepared read-only / newly committed / adopted read-only / independently inspected |
| WAITING_FOR_DATA / PRECONDITION_BLOCKED / REQUEST_MISMATCH | 2 | Wait or reject before writing |
| CONFLICT | 4 | Existing slot conflicts with expected identity or content contract |
| INCOMPLETE / UNCERTAIN | 5 | Incomplete evidence or uncertain postcondition |

Safe JSON exposes dates, digests, operation identity, ordered arm statuses/counts, and accounting/valuation/chain capabilities. It excludes private paths, epoch, instruments, amounts and raw exceptions. Help requires no workspace. There is no `--now` or forced-forward switch.

## Stored evidence and offline consumption

The slot contains request.json, source_intent.json, publication_success.json (original stdout bytes or JSON null), calendar_day.txt, next_open_prices.json, settlements.json, manifest.json and completion.json. Limits remain 4 MiB per data member, 32 MiB total, and 1 MiB per manifest/completion. Digest ordering remains logical data → request → inventory/manifest → completion.

`source_intent.json` retains the request, manifest, completion and original calendar bytes from the same successful C4 reader observation, plus canonical prior/intents/assumption for each arm. Offline inspection checks these saved references; it does not claim to re-adopt all external C4 sources. C4's `d1_metadata` returns defensive copies and confers no publication authority.

The Python APIs in `quantpits.research.forward_settlement` are:

- `prepare_first_forward_settlement(...)`
- `publish_first_forward_settlement(..., expected_request_digest=...)`
- `inspect_first_forward_settlement(settlement_store_root, epoch_id, expected_request_digest=...)`

The offline reader checks exact inventory, canonical schemas, source joins, float32 words, derived cash prices and quote projection. Strict loaders reconstruct inputs, and a real `ShadowPortfolioTransition.apply` recomputes and compares the entire stored transition. No live provider, Production, MLflow, model or original C4 directory is needed. Successful `after_states` return typed states in CHAMPION/CHALLENGER order, with predecessor links for prior, intents, source request/manifest/operation and settlement request. READY does not authorize after-state consumption.

`accounting_pair_complete` and `state_chain_ready` do not require every order to fill or valuation to be complete. COMPLETE accounting with PARTIAL valuation may yield consumable after-states, with null NAV and full missing-instrument lists in the transition. B0 retains signed cash, whole-order fill/no-fill, fees, slippage and no-deficit-worsening. `nav_before/nav_after` reconcile accounting at the same next open; they are not investment returns across weeks. Inner evidence remains `RETROSPECTIVE_TECHNICAL_REPLAY` with `prospective_claim=false`; the outer source linkage carries the original forward record without starting the epoch again. Corporate actions remain `UNMODELED`, with original warnings preserved.

## Continuing cycles

See the [continuation entry point](research_forward_continuation.md), sharing price observation, both-arm accounting and transactional writes. Existing positional API arguments remain compatible. When omitting stdout, pass the provider as `qlib_provider_root=...`; do not shift it into the old stdout position. `continuation_metadata` is a same-read defensive copy, not publication authority or whole-history verification.


## Copying records and format compatibility

New `completion.schema_version=2` records use `FORWARD_RECORD_BINDING_V2`. Their `target_binding_digest` binds record kind, epoch, cycle/index, request digest and manifest digest. Continuing `store_bindings` use `FORWARD_CHAIN_STORE_V2`, binding the intent/settlement role, first request/manifest/operation, frozen selectors, definition request and schedule. Neither binding includes paths, device, inode, mtime or host identity. Request/bundle versions and the safe stdout schema are unchanged; request version 2 alone does not indicate portability.

Copy the four runtime stores and optionally preserve any original COMMITTED stdout bytes, restore directory mode 0700 and file mode 0600, and explicitly select the new roots. Reading, inspection and adoption neither rewrite records nor renew the original publication time or prospective claim. Settlement consumes complete copied intents with optional stdout; D3 reads copied chains; D2 continues from verified after-states without resetting bootstrap or epoch. Production, provider, definitions and models remain explicit inputs. This does not migrate an entire workspace or MLflow URIs.

V1 completions retain physical binding checks at their original location, with subsequent new records written as V2. A new V2 settlement can retain an unchanged V1 source completion and still be copied: its own verified content binding identifies the new settlement. The V1 intent itself remains bound to its original directory. Copied V1 records return `LEGACY_PHYSICAL_BINDING_MISMATCH`; no automatic conversion or rewriting occurs. Unknown, missing or ill-typed versions are rejected. Canonical-path, directory identity, file fingerprint, source mutation and create-only checks remain active during operations. Independent copies do not share a global lock; select one active copy after recovery. Rollback must retain readers for V2, PAPER_ACCOUNTING_TIME_V1, optional stdout and HOLD without downgrading published records. This format change does not change economic rules or introduce a production approval gate.

Settlement accepts new paper-time records, historical strict/maintenance records and source_change V1 observations. Source-code differences do not block verified orders; omitting stdout does not weaken content, account or source checks.
