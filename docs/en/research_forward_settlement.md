# D1: settlement of the first forward intent pair

D1 settles one exact C4 intent pair using its original prior states, intents and execution assumption, with the corresponding next-open observations and existing B0 accounting. It saves both full transitions and after-states. It does not load models, replan orders, update Production, implement D2 continuation, or produce D3 investment returns.

## Inputs and read-only preparation

Provide the exact C4 `intent_store_root / epoch_id`, its request digest, and the safe JSON returned by the original **COMMITTED** publication. ADOPTED, VERIFIED and UNCERTAIN records cannot substitute for that original success. Request, manifest, operation, target binding, dates and every local time gate must match. This is single-owner local observation evidence, without an external trusted timestamp.

Use a dedicated physical settlement root with mode 0700 and a success record with mode 0600. Paths must be absolute and contain no symlink aliases. Development uses temporary workspaces; these command examples do not authorize private settlement.

```bash
python -m quantpits.scripts.settle_forward_intent prepare \
  --intent-store-root /absolute/research/intent_pairs \
  --epoch-id EPOCH_ID \
  --expected-intent-request-digest C4_REQUEST_SHA256 \
  --publication-success-record-path /absolute/research/c4_success.json \
  --qlib-provider-root /absolute/qlib/provider \
  --settlement-store-root /absolute/research/settlements
```

`prepare` writes nothing and returns READY with the D1 request digest. The request binds original publication identity, the exact success-record bytes, both accounting inputs, observed calendar and price receipt, results and implementation fingerprints. Physical paths, inode, mtime and current sampling time are excluded. Price revisions change the request; metadata-only touches do not.

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

The slot contains `request.json`, `source_intent.json`, exact original `publication_success.json` bytes, actual `calendar_day.txt`, `next_open_prices.json`, `settlements.json`, `manifest.json` and `completion.json`. Data members are limited to 4 MiB each and 32 MiB total; manifest and completion are each limited to 1 MiB. Digest order is logical data → request → member inventory/manifest → completion, without circular hashing.

`source_intent.json` retains the request, manifest, completion and original calendar bytes from the same successful C4 reader observation, plus canonical prior/intents/assumption for each arm. Offline inspection checks these saved references; it does not claim to re-adopt all external C4 sources. C4's `d1_metadata` returns defensive copies and confers no publication authority.

The Python APIs in `quantpits.research.forward_settlement` are:

- `prepare_first_forward_settlement(...)`
- `publish_first_forward_settlement(..., expected_request_digest=...)`
- `inspect_first_forward_settlement(settlement_store_root, epoch_id, expected_request_digest=...)`

The offline reader checks exact inventory, canonical schemas, source joins, float32 words, derived cash prices and quote projection. Strict loaders reconstruct inputs, and a real `ShadowPortfolioTransition.apply` recomputes and compares the entire stored transition. No live provider, Production, MLflow, model or original C4 directory is needed. Successful `after_states` return typed states in CHAMPION/CHALLENGER order, with predecessor links for prior, intents, source request/manifest/operation and settlement request. READY does not authorize after-state consumption.

`accounting_pair_complete` and `state_chain_ready` do not require every order to fill or valuation to be complete. COMPLETE accounting with PARTIAL valuation may yield consumable after-states, with null NAV and full missing-instrument lists in the transition. B0 retains signed cash, whole-order fill/no-fill, fees, slippage and no-deficit-worsening. `nav_before/nav_after` reconcile accounting at the same next open; they are not investment returns across weeks. Inner evidence remains `RETROSPECTIVE_TECHNICAL_REPLAY` with `prospective_claim=false`; the outer source linkage carries the original forward record without starting the epoch again. Corporate actions remain `UNMODELED`, with original warnings preserved.

## Continuing cycles

D2 has a separate [continuation entry point](research_forward_continuation.md). This module shares NEXT_OPEN observation, both-arm B0 recomputation and create-only writes while strictly distinguishing first v1 and continuing v2 sources and their original success records. `inspect_first_forward_settlement` additionally exposes a defensive `continuation_metadata` copy containing the request and source intent request/completion verified in that same read. This metadata grants no publication authority and makes no claim about the entire history. First-cycle safe JSON and persisted v1 contracts remain unchanged.
