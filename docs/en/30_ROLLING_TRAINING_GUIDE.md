# Rolling Training Guide (Overview)

> Phase 27 transition boundary: Rolling/CPCV-rolling still use their legacy window/state
> orchestration and are not yet backed by the static/CPCV execution service. All real mutating
> commands now share the workspace training execution lease with static/CPCV, preventing concurrent
> current-record replacement. `--show-state` and `--dry-run` do not acquire that lease. Full
> plan/evidence/publication/closure parity remains explicitly deferred.
> `--dry-run` is a strict filesystem-only preview: it renders configuration, action, and model
> selection without initializing Qlib, resolving exact window/fold dates, or creating OperatorLog,
> state, manifest, or lock files. Exact windows are resolved by real execution.
> `--show-state` likewise uses a lock-free read-only state load. `--clear-state` backs up/deletes
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

# ---- Slide rolling ----
python quantpits/scripts/rolling_train.py --cold-start --dry-run \
  --models linear_Alpha158 --training-method slide

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

In Phase 27, `--dry-run` does not render exact Qlib-calendar windows or folds; `--show-folds` takes
effect only after real execution resolves dates. This preserves a zero-write, zero-Qlib preview.

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
