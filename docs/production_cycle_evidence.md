# Production cycle evidence

`cycle-evidence capture` is an explicit post-cycle observation command. It does
not train models, change portfolio state, promote configuration, submit orders,
or modify an existing M1–M5 result. It reads exact run manifests and publishes
one create-only bundle under `data/evidence/v1/cycles/<cycle-id>/`.
The workspace comes from `--workspace` or `QLIB_WORKSPACE_DIR`; one of them is
required. Source the workspace `run_env.sh` before using the environment form.
Qlib data follows `QLIB_DATA_DIR` and otherwise uses the engine default
`~/.qlib/qlib_data/cn_data`; `--qlib-data-dir` can bind an explicit location.

Run it after the cycle has finished:

```bash
python -m quantpits.scripts.cycle_evidence capture \
  --cycle-id 2099-01-02 \
  --research-epoch-id SYNTHETIC_OBSERVATION_V1 \
  --post-trade-manifest output/manifests/post-trade/example.json \
  --prediction-manifest output/manifests/static_train/example.json \
  --ensemble-manifest output/manifests/ensemble_fusion/example.json \
  --order-manifest output/manifests/order_gen/example.json \
  --deep-analysis-run output/deep_analysis/example \
  --decision-event data/decisions/example.json
```

Use `--dry-run` to perform the same inspection and ranking in memory without
creating lock, staging, output, MLflow, environment, or working-directory state.
It returns `preview_complete` or `preview_partial`; previews include a
non-authoritative candidate digest but never claim that a final bundle exists.

`sealed_complete` means every Phase 37A-required fact was observed and sealed.
`sealed_partial` preserves the observed facts and diagnostics but grants no
complete-evidence capability. An absent decision is recorded as `not_recorded`,
never inferred as `NO_ACTION`. Absence of a future Dolt-to-Qlib materialization
receipt is recorded exactly as `unverified` and does not by itself make the
Phase 37A bundle partial.

Historical universe files may contain multiple non-overlapping membership
intervals for one instrument. Eligibility is resolved at the exact evidence
anchor; two active intervals for the same instrument are rejected. Portfolio
cash, holding amount, and holding value accept strict finite decimal strings as
well as JSON numbers and are sealed as normalized decimal text, without binary
float conversion. Whitespace, exponent-form strings or JSON numbers,
non-finite values, and negative holding amount/value remain invalid.

Run-manifest references retain their `inputs`/`outputs` collection and `kind`.
An absent M3 `outputs` reference of kind `record` is treated as a logical
recorder locator only when the same exact recorder and its contained artifact
tree were independently verified; every unproven or filesystem reference must
still exist and be comparable.

Deep Analysis remains an explicit optional source. Point
`--deep-analysis-run` at one operator-created run capsule containing exactly
the report, checkpoints, and trace directories for that M5 run. The collector
does not guess a run from mutable “latest” files or change M5 output behavior;
omitting the capsule produces an honest visible partial seal.

Replaying byte-equivalent cycle inputs adopts the existing bundle. A changed
input under the same cycle ID returns `conflict`; the existing namespace is
never overwritten or repaired in place.

A capture that has written staging but cannot publish returns
`failed_no_final`. Its private `.staging` directory is retained for
owner-controlled cleanup; the command does not recursively delete a mutable
public path after failure.
