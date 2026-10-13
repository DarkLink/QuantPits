# Continuing paper accounts and explicit HOLD intervals

Each cycle continues from each arm’s own settled cash and positions: first intent → first settlement → subsequent intent/settlement. No bootstrap reset, saved terminal stdout or Monday deadline is required. Late generation retains its actual timestamp and after-open simulation label; late settlement keeps the original trade date. Owners select workspace, provider, account, sources and write scope.

## Schedule and predecessor

`WEEKLY_LAST_SESSION_V1` uses the last trading session of each ISO week in the market timezone. The next anchor is the last session of the first subsequent week containing sessions. A future calendar must cover the target week and contain a session in a later week. Entirely closed weeks can be skipped; the actual and future calendars must agree over the observed range. The original first anchor must also be a week-ending session.

Index 2 joins the first settlement; later indexes join index−1 in the same epoch. Indexes count actual saved accounting records. Readers verify each portfolio ID, after-state and digest; divergent cash/positions and PARTIAL NAV remain valid. Without explicit HOLD, skipping a nonempty trading week still returns GAP_DETECTED. Entirely closed weeks follow the actual calendar. Missing, corrupt or unsettled predecessors cannot advance. Ordinary source-code changes are observations.

## Explicit HOLD: continuous holding without generated orders

To skip at least one nonempty trading week, supply all three flags to prepare-intent and publish-intent:

```bash
--hold-start-date "$PREDECESSOR_TRADE_DATE" \
--hold-end-date "$SELECTED_ANCHOR" \
--hold-reason "No generated orders; assume continuous holding for this interval"
```

The interval is `(predecessor trade date, selected anchor]`. Start must equal the settled predecessor trade_date and end must equal the selected anchor. The next trade date remains the first session after that anchor. The calendar verifies week-ending anchors and covers a later trading week. An unnecessary HOLD without a skipped trading week is rejected. The retained NO_ORDERS_CONTINUOUS_HOLD_USER_ASSUMPTION_V1 describes the caller’s simulation choice, not an observation that no historical orders existed.

Cash, position quantities and costs remain the predecessor’s until new selected-cycle orders execute. No empty weekly records or skipped-week trading fees are created. Preparation displays predecessor date, interval, anchor and trade_date. The selected account’s continuing intent store is inspected: a later unsettled intent returns HOLD_PENDING_INTENT, incomplete/corrupt/unknown records return HOLD_RECORD_UNCERTAIN, and later settled records also prevent reverting to an older predecessor. Process existing records first and continue from their after-states.

Missing prices retain positions and existing partial/no-fill/null-NAV semantics. Dividends/splits remain unmodeled and disclosed. Reports retain actual indexes and interval returns without inventing intermediate valuations.

## Storage and manual operation

Explicitly provision two dedicated roots and their epoch parents with mode `0700`:

```text
<intent-root>/<epoch>/2
<settlement-root>/<epoch>/2
<intent-root>/<epoch>/3
<settlement-root>/<epoch>/3
```

Writers create only the current canonical integer slot. They do not create parents recursively, overwrite slots or use leading zeros, hashes or attempt IDs. Preserve conflicts, incomplete bundles and uncertain writes for inspection/adoption using the original expected request. V2 completion store_bindings identify the frozen chain and intent/settlement roles, checked on subsequent joins. The actual target roots must remain non-overlapping.

1. Inspect the settled predecessor and exact original first-intent identity.
2. Complete the current Production seal and signal capsule. Prepare the intent with the frozen definition/bootstrap/model selectors, current cycle/signal and exact predecessor.
3. Review the READY digest, predecessor trade date, selected anchor/trade date, counts and optional HOLD. Publish with the same arguments and expected digest when convenient; saving stdout is optional.
4. Once the original next-open has passed and prices are available, prepare settlement directly from the complete local intent, then publish the reviewed digest. Settlement has no deadline; stdout is optional corroboration.
5. Inspect settlement and use it as the next predecessor, incrementing the index by one.

Shadow failures are reported separately and do not modify or automatically block Production. No scheduler or Make target is added.

## API and CLI

`quantpits.research.forward_continuation` provides:

- `prepare_next_forward_intent_publication` / `publish_next_forward_intent_pair`: the fourteen current-source arguments and explicit first/predecessor references remain required. `decision_deadline_utc=None` and `publication_mode="PAPER"` are optional. HOLD requires all of `hold_start_date`, `hold_end_date`, `hold_reason`.
- `prepare_next_forward_settlement` / `publish_next_forward_settlement`: retain old positional arguments, with optional `publication_success_record_path=None`. Pass `qlib_provider_root=...` by keyword when omitting stdout.
- Both publication APIs additionally require `expected_request_digest`. Preparation observations cannot be deserialized into publication authority; publication reobserves inputs and replans.
- `inspect_next_forward_intent_pair(root, epoch, index, expected_request_digest=...)` exposes `d1_inputs`. `inspect_next_forward_settlement(...)` exposes `after_states` and defensive-copy `continuation_metadata`. Inspection needs no live models, provider or Production workspace.

```bash
python -m quantpits.scripts.continue_forward --help
python -m quantpits.scripts.continue_forward inspect-intent \
  --intent-store-root "$INTENT_ROOT" --epoch-id "$EPOCH" \
  --cycle-index 2 --expected-request-digest "$INTENT_REQUEST"
python -m quantpits.scripts.continue_forward inspect-settlement \
  --settlement-store-root "$SETTLEMENT_ROOT" --epoch-id "$EPOCH" \
  --cycle-index 2 --expected-request-digest "$SETTLEMENT_REQUEST"
```

Other actions are `prepare-intent`, `publish-intent`, `prepare-settlement`, and `publish-settlement`. CLI flags use hyphens instead of API underscores. Help lists every argument without accessing a workspace. Exit codes: success 0; precondition/waiting/request mismatch 2; VERSION_BREAK 3; conflict 4; incomplete/uncertain 5.

## Evidence scope

Continuing requests/bundles have separate domains and schema v2. Original C4/D1 v1 remains readable. Each bundle contains this cycle's inputs, direct predecessor references and frozen starting identity; it does not recursively embed history or copy models.

Default paper publication reports cycle_intent_published=true with prospective_claim=false and epoch_started=false. Persisted timing describes facts, not historical strict prospective qualification. First and continuing settlements both accept complete local bundles directly. Supplied stdout is checked; a first-cycle log, ADOPTED or VERIFIED cannot replace the current original commit output. Historical strict records and optional STRICT_PREOPEN publication keep their original time contract.

`state_chain_ready` describes actual B0 recomputation and both after-states in this settlement. `predecessor_join_observed` comes from fresh preparation. Unread history is not verified and `whole_chain_verified` is never true. Limits remain 4 MiB per data member, 32 MiB total and 1 MiB per record, preserving ordinary failures, process-control propagation and uncertain post-write states.

See [first preparation](research_forward_preparation.md), [first orders](research_forward_publication.md), [settlement](research_forward_settlement.md) and [account reports](research_forward_window_report.md).

C3/C4/D2 treat arbitrary source differences or unavailable source-equivalence comparisons as observations. Paper accounting can continue when the other model, configuration, signal and account checks pass. Historical production code need not equal current research code. Current working files are still checked against the loaded source location, and their content fingerprint is recorded independently of HEAD. Corrupt signals, rebuilt-ranking contradictions, model/configuration changes and wrong predecessors still block accounting. The existing public decision-surface API retains its comparison semantics; the daily path grants neither strict same-segment equivalence nor maintenance admission.

New requests use `input_provenance.source_change`, with independent `schema_version=1` and `SOURCE_DIFFERENCE_OBSERVATION_ONLY_V1`. `production_comparison` records the two production source observations; `production_commit`/`production_digest` identify the current prediction source; `execution_commit`/`execution_tree`/`execution_implementation_digest`/`execution_code_digest` identify current research execution. `production_execution_comparison` is EQUAL, DIFFERENT or INCOMPARABLE, with an unavailability reason where applicable. These observations affect input/request digests, not account continuity. Legacy `maintenance_admission` records retain their existing reader contract. Portable completion V2 bindings are independent of this field version. Code rollback must retain a reader for new records; historical records are not rewritten. Paper timing and optional stdout follow the rules above; legacy strict records, duplicate-write checks and interruption contracts remain compatible.


## Copying records and format compatibility

New `completion.schema_version=2` records use `FORWARD_RECORD_BINDING_V2`. Their `target_binding_digest` binds record kind, epoch, cycle/index, request digest and manifest digest. Continuing `store_bindings` use `FORWARD_CHAIN_STORE_V2`, binding the intent/settlement role, first request/manifest/operation, frozen selectors, definition request and schedule. Neither binding includes paths, device, inode, mtime or host identity. Request/bundle versions and the safe stdout schema are unchanged; request version 2 alone does not indicate portability.

Copy the four runtime stores and optionally preserve any original COMMITTED stdout bytes, restore directory mode 0700 and file mode 0600, and explicitly select the new roots. Reading, inspection and adoption neither rewrite records nor renew the original publication time or prospective claim. Settlement consumes complete copied intents with optional stdout; D3 reads copied chains; D2 continues from verified after-states without resetting bootstrap or epoch. Production, provider, definitions and models remain explicit inputs. This does not migrate an entire workspace or MLflow URIs.

V1 completions retain physical binding checks at their original location, with subsequent new records written as V2. A new V2 settlement can retain an unchanged V1 source completion and still be copied: its own verified content binding identifies the new settlement. The V1 intent itself remains bound to its original directory. Copied V1 records return `LEGACY_PHYSICAL_BINDING_MISMATCH`; no automatic conversion or rewriting occurs. Unknown, missing or ill-typed versions are rejected. Canonical-path, directory identity, file fingerprint, source mutation and create-only checks remain active during operations. Independent copies do not share a global lock; select one active copy after recovery. Rollback must retain readers for V2, PAPER_ACCOUNTING_TIME_V1, optional stdout and HOLD without downgrading published records. This format change does not change economic rules or introduce a production approval gate.
