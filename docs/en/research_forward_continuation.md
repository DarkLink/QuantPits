# Shadow Forward continuing cycles (D2)

D2 continues each arm from its verified settlement: `C4 intent 1 → D1 settlement 1 → intent 2 → settlement 2 → intent 3 …`. Synthetic development chains do not establish real market observations. Real operations require explicit workspace, provider, epoch, source selectors, roots and write authorization.

## Schedule and predecessor

`WEEKLY_LAST_SESSION_V1` uses the last trading session of each ISO week in the market timezone. The next anchor is the last session of the first subsequent week containing sessions. A future calendar must cover the target week and contain a session in a later week. Entirely closed weeks can be skipped; the actual and future calendars must agree over the observed range. The original first anchor must also be a week-ending session.

Index 2 accepts only a D1 settlement. Later indexes accept only the same epoch's D2 settlement at index minus one. Actual readers verify both arms and join each role, portfolio ID, canonical after-state and digest. Divergent cash/positions and PARTIAL NAV are valid. Missing, incomplete or unsettled predecessors cannot advance. Gaps, unavailable schedules, expired deadlines and VERSION_BREAK stop new intents; no historical backfill or fabricated zero-order cycle is created. Previously sealed intents can still settle using their original assumption.

## Storage and manual operation

Explicitly provision two dedicated roots and their epoch parents with mode `0700`:

```text
<intent-root>/<epoch>/2
<settlement-root>/<epoch>/2
<intent-root>/<epoch>/3
<settlement-root>/<epoch>/3
```

Writers create only the current canonical integer slot. They do not create parents recursively, overwrite slots or use leading zeros, hashes or attempt IDs. Preserve conflicts, incomplete bundles and uncertain writes for inspection/adoption using the original expected request. Physical bindings of both roots are retained in completion metadata, included together in the original success record’s target_binding_digest, and checked on subsequent fresh joins.

1. Inspect the settled predecessor and exact original first-intent identity.
2. Complete the current Production seal and signal capsule. Prepare the intent with the frozen definition/bootstrap/model selectors, current cycle/signal and exact predecessor.
3. Review the READY request digest, index, dates, counts and deadline. Publish with the same parameters and expected request before next-open. Save this operation's original COMMITTED safe JSON at an authorized location.
4. Once next-open data arrives, prepare settlement with this intent and its original success record, then publish the reviewed request.
5. Inspect settlement and use it as the next predecessor, incrementing the index by one.

Shadow failures are reported separately and do not modify or automatically block Production. No scheduler or Make target is added.

## API and CLI

`quantpits.research.forward_continuation` provides:

- `prepare_next_forward_intent_publication` / `publish_next_forward_intent_pair`: the fourteen C3 current-source arguments, plus keywords `intent_store_root, settlement_store_root, epoch_id, cycle_index, first_intent_store_root, expected_first_intent_request_digest, predecessor_settlement_store_root, predecessor_cycle_index, expected_predecessor_request_digest, decision_deadline_utc, next_open_utc, market_timezone`.
- `prepare_next_forward_settlement` / `publish_next_forward_settlement`: `intent_store_root, epoch_id, cycle_index, expected_intent_request_digest, publication_success_record_path, qlib_provider_root`, plus keyword `settlement_store_root`.
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

A fresh successful publication reports `cycle_intent_published=true, prospective_claim=true, epoch_started=false`. It retains all local UTC/monotonic, 600-second and pre-open gates. Settlement strictly links this cycle's original success JSON and completion. A first-cycle log, ADOPTED or VERIFIED output cannot substitute for it. Inspection/adoption grants no new publication claim.

`state_chain_ready` describes actual B0 recomputation and both after-states in this settlement. `predecessor_join_observed` comes from fresh preparation. Unread history is not verified and `whole_chain_verified` is never true. Limits remain 4 MiB per data member, 32 MiB total and 1 MiB per record, preserving ordinary failures, process-control propagation and uncertain post-write states.

See [first settlement](research_forward_settlement.md). D3 common-window reporting and real observation periods remain separate work.
