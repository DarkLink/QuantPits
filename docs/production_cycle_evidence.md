# Production cycle evidence

`cycle-evidence capture` is an explicit post-cycle observation command. It does
not train models, change portfolio state, promote configuration, submit orders,
or modify an existing M1–M5 result. It reads exact run manifests and publishes
one create-only bundle under `data/evidence/v1/cycles/<cycle-id>/`.

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

`sealed_complete` means every Phase 37A-required fact was observed and sealed.
`sealed_partial` preserves the observed facts and diagnostics but grants no
complete-evidence capability. An absent decision is recorded as `not_recorded`,
never inferred as `NO_ACTION`. Absence of a future Dolt-to-Qlib materialization
receipt is recorded exactly as `unverified` and does not by itself make the
Phase 37A bundle partial.

Replaying byte-equivalent cycle inputs adopts the existing bundle. A changed
input under the same cycle ID returns `conflict`; the existing namespace is
never overwritten or repaired in place.
