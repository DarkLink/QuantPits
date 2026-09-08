# First Shadow Forward intent preparation (C3)

C3 computes one in-memory pair from an explicit current cycle, a frozen four-model Champion, its frozen three-model Challenger, and one matched bootstrap. It reads inputs and plans intents without storing orders, executing trades, or starting an epoch.

Source the appropriate workspace environment first, then supply physical absolute paths and existing selectors explicitly:

```bash
python -B -m quantpits.scripts.prepare_forward_intent \
  --production-workspace "$PRODUCTION_ROOT" \
  --research-workspace "$RESEARCH_ROOT" \
  --engine-root "$ENGINE_ROOT" --qlib-provider "$PROVIDER_ROOT" \
  --current-cycle "$CURRENT_CYCLE" --activation "$ACTIVATION_PATH" \
  --definition-store "$DEFINITION_STORE" --evidence-store "$DEFINITION_EVIDENCE_STORE" \
  --bootstrap-store "$BOOTSTRAP_STORE" --bootstrap-set-id "$BOOTSTRAP_SET_ID" \
  --signal-capsule-store "$SIGNAL_STORE" --signal-capsule-id "$SIGNAL_CAPSULE_ID" \
  --model-capsule-store "$MODEL_STORE" --model-capsule-id "$MODEL_CAPSULE_ID"
```

These variables illustrate explicit invocation; the program does not discover them. `--help` needs no workspace. The Python API `quantpits.research.forward_intent_preparation.prepare_first_forward_intent(...)` accepts the corresponding positional arguments and returns `FirstForwardIntentPreparation`. The CLI supports `main(argv)`. Neither initializes Qlib/MLflow nor changes environment, cwd, or argv.

Reference and bootstrap source cycles come from Production. Definitions, definition evidence, bootstrap, and capsules come from Research. Model adoption uses the definition evidence cycle; weekly signals and ranking use the current cycle. `--evidence-store` means the Research definition-evidence store. There is no latest discovery, output directory, publication, or repair option.

| Status | Exit | Meaning |
| --- | --- | --- |
| `PREPARED` | 0 | Both plans complete; joins, four-source parity, and final stability passed |
| `VERSION_BREAK` | 3 | The existing decision-surface observer detected a frozen version change |
| `PRECONDITION_BLOCKED` | 2 | Missing/incompatible inputs, incomplete scores, parity/runtime-code mismatch, planning failure, or unstable observation |

stdout contains one canonical safe JSON line with dates, fixed roles, statuses, counts, and digests. It omits instruments, cash, quantities, private named IDs, paths, and exception text. Unobserved counts are null. Failed prerequisites leave both roles `NOT_RUN`. An ordinary arm failure still permits the other arm's diagnostic computation, but no partial pair is delivered. Process-control exceptions propagate.

Only a `PREPARED` API result retains the complete in-memory pair. Safe JSON cannot restore that pair or authorize C4 writes. `intent_publication_capability`, `epoch_started`, `prospective_claim`, `promotion_capability`, and `did_write` are always false.

Each source must completely cover the sealed anchor universe. Missing scores, duplicates, foreign instruments, NaN/inf, and sealed unscored rows block preparation without filling or shrinking the universe. Fusion preserves Stage A's average-tie percentiles, pandas mean in frozen member order, and canonical tie-break. Identical arm rankings, 100% overlap, and zero orders are valid.

One union anchor-close observation is projected separately into B1. The day/day_future calendars must agree through the anchor; trade date is the first future session after it. Price provenance is `CURRENT_PROVIDER_ANCHOR_OBSERVATION`, with historical materialization still `unverified`. Calendar byte-match to the seal is reported separately; ordinary calendar extension does not establish historical price reproduction. The existing reader hashes complete close/factor files, but decision arithmetic interprets only the anchor values. It does not read open files or use next-open values.

Legal B1 missing prices, pending forced exits, buy shortages, signed cash, and zero orders remain valid facts. C3 performs no settlement or post-trade cash/no-deficit-worsening check. Historical technical preparation cannot backfill prospective evidence. C4 requires a separately chosen future cycle and fresh observation.

The existing `inspect_decision_surface` CLI adds `--reference-source research|production`, retaining `research` as its default. Fresh split-root layouts explicitly select `production`, without existence-based fallback. The capacity fix applies only to fresh definition evidence read-only ADOPT; prepare/publish writer inventory budgets remain unchanged.

## Prediction-copy identity and historical compatibility

Predict-only continues to retain the actual `model.pkl`, or the complete CPCV fold set. Prediction run IDs and direct-parent tags remain audit identities. New `training_origin_record_id`, `training_origin_experiment`, and `training_origin_status=VERIFIED` tags are derived by traversing explicit parent links. Missing history, cycles, model-name conflicts, or contradictory cached origins produce `UNRESOLVED`; an already available model can still be used for prediction without claiming verified ancestry. Process-control interruptions propagate.

Existing definitions, capsules, manifests, and seals are unchanged. Exact legacy source matches remain valid. When direct source IDs or artifact inventories differ, a read-only compatibility observation compares the traced training origin and the actual sealed model contents. Each model file must first match its cycle's raw SHA-256. The versioned content fingerprint statically parses pickle instructions without loading models, importing Torch, or executing GLOBAL/REDUCE. It normalizes only framing, equivalent byte-length encodings, and recognized Torch legacy storage allocation IDs. Tensor values, types, shapes/strides, alias relationships, configuration, and optimizer state remain part of the fingerprint. CatBoost's embedded model bytes remain exact. This conservative copy protocol is not a general pickle equivalence algorithm.

Prediction/label files, code-status/diff/cache snapshots, and portfolio/signal analysis reports do not define model identity. Their existing signal, seal, and runtime checks remain applicable. Unknown auxiliary inputs retain exact relative-name and content checks. The independent code, ensemble, market, and order-policy checks are unchanged.

Historical compatibility currently supports only physical workspace-local `mlruns/<experiment-id>/<recorder-id>/artifacts` file stores. It reads the explicitly named experiment metadata, ancestry tags, and sealed model/auxiliary files, with mutation observation and before/after checks. Live ancestry is supplementary evidence observed now, not a claim retroactively attributed to the old seal. No MLflow initialization or latest-recorder selection occurs. Missing/ambiguous ancestry, symlinks, or integrity failures yield INCOMPARABLE (PRECONDITION_BLOCKED in C3). Verified changes to training origin, model parameters, or configuration remain VERSION_BREAK.

This repair neither reconstructs deleted historical runs nor waives C3's runtime-code/seal match. A historical cycle may still be blocked after prediction code changes; do not rewrite its seal or disable the check to force PREPARED.

### Stale experiment tags (2026-09-08)

Some historical `source_experiment` tags point to the wrong experiment. Research compatibility now locates the **exact recorder ID uniquely within the explicitly selected file backend**, checking the run ID and experiment ID in recorder metadata against its physical directory. It obtains the experiment name from that actual experiment's metadata. It neither edits historical tags nor substitutes another recorder or a latest selection. Duplicate run IDs, conflicting metadata and truly missing records remain blocking. Duplicate experiment names do not override a unique run ID. Experiment namespace and selected recorder metadata remain watched through the observation window.

The prediction writer's default ancestry lookup remains conservative for its explicit backend/experiment and records UNRESOLVED when it cannot establish the root. Research's independent file-backend observation can resolve such legacy ancestry; an UNRESOLVED cached tag does not prove that the training run was deleted.
