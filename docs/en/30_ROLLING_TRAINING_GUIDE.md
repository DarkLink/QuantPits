# Rolling Training Guide (Overview)

> Phase 28A/28B/28C establish an import-pure CLI, a filesystem-only authoritative Prepared Plan,
> and an in-lease Resolved Plan plus legacy execution adapter.
> Importing the module and running `--help` require no prior `source run_env.sh` and do not change the
> current directory. `--dry-run`, `--explain-plan`, and `--json-plan` render the same typed plan,
> freezing the action, ordered targets, workflow and workspace input fingerprints, legacy-state
> classification, expected effects, and plan fingerprint. This route does not initialize Qlib or
> MLflow, acquire the lease, or write files. Relative workflow YAML paths are resolved against the
> selected workspace and may not escape it.
>
> A Prepared Plan does not invent calendar facts: its anchor, slide windows, and CPCV folds remain
> runtime-deferred. A real command renders safeguard directly from the Prepared `WorkspaceContext`
> without importing legacy `env`; after acquiring the shared lease it rechecks Prepared inputs,
> activates that same explicit workspace, initializes Qlib once, and freezes the exact
> anchor, ordered windows/CPCV folds, stable window keys, and execution fingerprint. The adapter
> consumes only Prepared targets and Resolved windows; it does not rescan the registry or generate a
> second window set. Rolling still uses unversioned legacy state and legacy record
> merge, without the newer evidence/publication closure parity.

> Phase 27 transition boundary: Rolling/CPCV-rolling still use their legacy window/state
> orchestration and are not yet backed by the static/CPCV execution service. All real mutating
> commands now share the workspace training execution lease with static/CPCV, preventing concurrent
> current-record replacement. `--show-state` and `--dry-run` do not acquire that lease. Full
> service-owned execution/evidence/publication/closure parity remains explicitly deferred.
> `--dry-run` is a strict filesystem-only Prepared Plan: it renders configuration and input
> fingerprints, the action, ordered targets, state classification, and expected effects without
> initializing Qlib, resolving exact window/fold dates, or creating OperatorLog, state, manifest, or
> lock files. Exact windows are resolved by real execution.
> `--show-state` likewise uses lock-free strict readonly classification. `--clear-state` backs up/deletes
> state, so it still passes safeguard and holds the shared execution lease.

> The 30-series covers **rolling training** — paradigms where training windows slide forward over time.
>
> | Document | Content |
> |----------|---------|
> | **30 (this doc)** | Architecture overview, four modes, key system, quick start |
> | [31 · Slide Mode](31_SLIDE_ROLLING.md) | Window math, examples, use cases |
> | [32 · CPCV Mode](32_CPCV_ROLLING.md) | Walk-Forward CPCV, fold structure, gap analysis, parameter constraints |
> | [33 · Operations](33_OPERATIONS.md) | CLI reference, state management, daily workflow, troubleshooting |

---

## 1. Four Training Modes

| # | Mode | Script | Method | Output Key |
|---|------|------|--------|------------|
| 1 | Static | `static_train.py --full` | Fixed date-range slide | `model@static` |
| 2 | CPCV single | `cv_train.py --all-enabled` | K-fold + purge/embargo | `model@cpcv` |
| 3 | Rolling + slide | `rolling_train.py` | Slide per window | `model@rolling` |
| 4 | Rolling + CPCV | `rolling_train.py` | Walk-Forward CPCV per window | `model@cpcv_rolling` |

All four modes coexist in the same workspace. Downstream scripts select models via `--training-mode`.

### Train → Today Gap

| Mode | Gap | Bottleneck |
|------|-----|------------|
| Static slide (5yr train / 2yr valid / 2yr test) | ~4 years | fixed valid + test |
| CPCV single (10 groups / 2 test) | ~2 years | test_set = 2 groups |
| Rolling + slide (5yr train / 1yr valid / 3M step) | ~15 months | valid = 1 year |
| **Rolling + CPCV (5yr train / 3M step)** | **~16 months** | test_step + purge |

> CPCV's advantage is not just gap size — its right training segment covers data AFTER the validation period, which slide mode never sees.

---

## 2. Architecture: WHEN vs HOW Decoupled

```
rolling_train.py      ← CLI + orchestration + strategy dispatch
    │
    ├─ training_method: "slide" | "cpcv"  (--training-method overrides config)
    │
    ├─ strategy_slide.py     ← Slide: train/valid/test contiguous per window
    ├─ strategy_cpcv.py      ← CPCV: K-fold Walk-Forward CPCV per window
    │
    └─ orchestration.py      ← Shared: window loop + stitching + saving + backtest
```

```
quantpits/rolling/
├── command.py            # filesystem-only PreparedRollingRun
├── windows.py            # runtime ResolvedRollingRun + stable window keys
└── legacy.py             # exact-scope legacy adapter + baseline recheck

rolling/
├── state.py              # State management (separate file per method)
├── memory.py             # 3-tier memory cleanup
├── backtest.py           # Backtest
├── orchestration.py      # Shared orchestration
├── strategy_slide.py     # Slide strategy
├── strategy_cpcv.py      # CPCV strategy
├── windows.py            # [compat layer]
├── training.py           # [compat layer]
└── prediction.py         # [compat layer]
```

---

## 3. Configuration

`config/rolling_config.yaml`:

```yaml
rolling_start: "2015-01-01"
train_years: 5                  # slide: train length / CPCV: train domain length
valid_years: 1                  # slide: valid length / CPCV: unused (kept for compat)
test_step: "3M"                 # rolling step (nM or nY)

training_method: "cpcv"         # "slide" (default) | "cpcv"

# CPCV params (only when training_method=cpcv)
cpcv_n_groups: 10               # groups in train domain
cpcv_n_val_groups: 1            # validation groups per fold
cpcv_purge_steps: 3             # symmetric purge (trading weeks)
cpcv_embargo_steps: 5           # asymmetric embargo (trading weeks)
```

---

## 4. Quick Start

```bash
conda activate qlib_cupy
source workspaces/<name>/run_env.sh

# Or skip sourcing and select the example workspace explicitly
python -m quantpits.scripts.rolling_train --workspace workspaces/Demo_Workspace --help

# Equal form; direct script, python -m, and main(argv=[...]) use the same parsed context
python -m quantpits.scripts.rolling_train --workspace=workspaces/Demo_Workspace --help

# ---- Slide rolling ----
python quantpits/scripts/rolling_train.py --cold-start --dry-run \
  --models linear_Alpha158 --training-method slide

# Single-document JSON form of the same authoritative Prepared Plan
python -m quantpits.scripts.rolling_train \
  --workspace workspaces/Demo_Workspace --cold-start --all-enabled --json-plan

# ---- CPCV rolling (with fold details) ----
python quantpits/scripts/rolling_train.py --cold-start --dry-run \
  --models linear_Alpha158 --training-method cpcv --show-folds

# ---- Train ----
python quantpits/scripts/rolling_train.py --cold-start \
  --models linear_Alpha158 --training-method cpcv

# ---- Update after new data ----
python quantpits/scripts/rolling_train.py --merge --all-enabled

# ---- Predict only (no retraining) ----
python quantpits/scripts/rolling_train.py --predict-only --all-enabled

# ---- Backtest ----
python quantpits/scripts/rolling_train.py --backtest-only \
  --models linear_Alpha158 --training-method cpcv
```

The Prepared Plan does not render exact Qlib-calendar windows or folds. In plan mode, `--show-folds`
only exposes the deferred reason; exact content is available after real execution resolves dates.
This preserves a zero-write route with no Qlib/MLflow initialization. Conflicting primary actions,
such as `--cold-start --resume`, fail with `rolling_action_conflict` before safeguard, lease, or
backend activation. `--show-state` distinguishes `missing`, `valid_legacy`, `corrupt`, and
`unsupported` instead of treating damaged state as empty.

`--workspace PATH`, `--workspace=PATH`, and programmatic `main(argv=[...])` all use the Prepared
context as the sole workspace identity; legacy `env` does not reselect it from process `sys.argv`.
Real execution is ordered as explicit-context safeguard → shared lease → input baseline recheck → workspace
activation → Qlib init → Resolved Plan → legacy adapter → OperatorLog → lease release.
`--clear-state` needs no Qlib initialization, but still rechecks its state baseline and performs the
backup/removal inside the lease; an already-missing state is recorded as `skipped`. Missing valid
anchors for `daily`/`predict-only`, or no completed window for `retrain-last`, fail before backend
initialization with `rolling_state_precondition_failed`. OperatorLog, adapter outcome, and CLI exit
share one command-level status. Successful actions retain `legacy_partial_visibility`; they do not
prove immutable evidence for every target×window or manifest/receipt closure. A `predict-only`
action that produces no prediction is recorded as `skipped`.
`backtest-only` fails nonzero with `rolling_backtest_precondition_failed` when current records are
missing or empty, the requested Rolling family is absent, or no historical Rolling record exists
for the selected targets. It reports `success / legacy_partial_visibility` only after records are
found and the legacy backtest is invoked.

The project owner controls and runs the Phase 28 full Python suite and workspace gates. A no-write
gate should use `Demo_Workspace` or a disposable validation workspace explicitly selected by the
owner, with before/after snapshots of configuration, state, current records, OperatorLog, and MLflow
paths. Plan commands must not trigger safeguard, lease acquisition, or backend initialization. The
production workspace remains read-only. A real adapter/bootstrap smoke is an owner acceptance gate
and may run only with explicit authorization in a disposable validation workspace. Committed code
and documentation must not contain private workspace identities, absolute paths, or runtime data.

> `--training-method` overrides `rolling_config.yaml` — switch modes without editing files.

---

## 5. Key System

| Mode | Key | `--training-mode` |
|------|-----|-------------------|
| Static | `model@static` | `static` |
| CPCV single | `model@cpcv` | `cpcv` |
| Rolling + slide | `model@rolling` | `rolling` |
| Rolling + CPCV | `model@cpcv_rolling` | `cpcv_rolling` |

Keys coexist in `latest_train_records.json`. `--cold-start` only clears the current method's state file.

V2 keeps output experiment/recorder identity per rolling and CPCV-rolling model. The derived
`cpcv_rolling_experiment_name` compatibility field is preserved across unrelated merges.
Real rolling commands verify the combined recorder, persisted prediction coverage, and artifact
workspace containment. Verification failure does not downgrade into an unverified current pointer.

```bash
# Downstream
python quantpits/scripts/brute_force_fast.py --training-mode rolling
python quantpits/scripts/brute_force_fast.py --training-mode cpcv_rolling
python quantpits/scripts/ensemble_fusion.py --from-config --training-mode rolling
```

---

## 6. Further Reading

- [31 · Slide Mode](31_SLIDE_ROLLING.md) — window math and examples
- [32 · CPCV Mode](32_CPCV_ROLLING.md) — Walk-Forward CPCV design, fold structure, gap analysis
- [33 · Operations](33_OPERATIONS.md) — CLI flags, state management, daily workflow
