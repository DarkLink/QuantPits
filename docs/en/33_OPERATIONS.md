# 33 · Operations Guide

> Daily operations: CLI reference, state management, workflows, troubleshooting.

## 1. CLI Reference

### Run Modes

| Flag | Effect | Use Case |
|------|--------|----------|
| `--cold-start` | Clear state, train all windows | First run, full rebuild |
| `--merge` | Keep state, train only missing windows | After qlib data update, append new models |
| `--retrain-models M1,M2` | Clear those models' state, rebuild | Hyperparameter/code changes |
| `--retrain-last` | Clear last window, retrain | Data correction |
| `--predict-only` | No training, predict with latest model | Quick prediction after data update |
| `--resume` | Continue from breakpoint | Training interrupted |
| `--backtest` | Run backtest after training | Attached to other modes |
| `--backtest-only` | Backtest only, skip training | Report on existing predictions |

### Control Flags

| Flag | Purpose |
|------|---------|
| `--workspace PATH` | Select a workspace explicitly; otherwise use `QLIB_WORKSPACE_DIR` |
| `--dry-run` | Compatibility entry for the authoritative Prepared Plan; exact dates are runtime-deferred |
| `--explain-plan` | Human-readable form of the same Prepared Plan |
| `--json-plan` | Single JSON document for that plan, including input/state/effect/plan fingerprints |
| `--run-id ID` | Explicit lease operation identity for a real run |
| `--show-folds` | CPCV: show per-window fold details |
| `--training-method slide\|cpcv` | Override config file setting |
| `--cache-size N` | Handler cache max (MB), 0=disable. CPCV K-fold reuse |
| `--allow-stale-predict` | Allow old weights for new data in predict-only |

Preparing a plan is strictly filesystem-read-only: it does not initialize Qlib/MLflow, acquire the
shared lease, or write an OperatorLog. The plan freezes registry-ordered targets and
workspace-contained workflows, classifies legacy state, and truthfully declares state, record,
history, MLflow artifact, and OperatorLog effects of a real command. Calendar anchors, windows, and
CPCV folds remain deferred in the Prepared Plan. Real execution rechecks input baselines inside the
shared lease, initializes Qlib once, and binds the exact anchor, ordered windows/folds, stable window
keys, and execution fingerprint in a Resolved Plan. `--workspace PATH`, `--workspace=PATH`, and
`main(argv=[...])` share the Prepared context; safeguard does not import legacy `env`, and activation
occurs only after safeguard → lease → baseline recheck. The legacy adapter consumes that frozen
scope without rescanning the registry or generating another window set.

`daily`/`predict-only` without an anchor and `retrain-last` without a completed window fail before
backend initialization with `rolling_state_precondition_failed`. An already-missing
`--clear-state`, and `--predict-only` that creates no prediction, are explicitly `skipped`.
OperatorLog, adapter outcome, and CLI exit share one command-level status. Successful actions retain
`legacy_partial_visibility`; they do not claim per-window evidence parity.
`--backtest-only` exits nonzero with `rolling_backtest_precondition_failed` when current records are
missing or empty, the requested Rolling family is absent, or selected targets have no historical
record. OperatorLog records the same failure. It reports `success / rolling_backtest_completed`
only when every selected model completes recorder/prediction loading, Qlib backtest, and required
metrics/artifact publication; any single-model or mixed-batch failure exits nonzero. The OperatorLog
backtest summary includes requested/attempted/succeeded/failed counts and concise model/stage/reason
failure entries. `legacy_partial_visibility` applies only to training-window evidence, not to the
authoritative backtest batch.

With `--backtest` attached to a primary action, training, merge, or prediction outputs may already
be persisted before the backtest fails. The entire command still exits nonzero with
`did_execute=true`. This truthfully reports partial execution; it is not a transaction rollback, so
inspect OperatorLog and current records before deleting or retrying generated output.
Inside the shared lease and before backend initialization, every declared write path receives a
symlink-aware containment check. State, record, history, MLflow, or OperatorLog paths that resolve
outside the workspace fail with `rolling_output_outside_workspace`; do not share writable runtime
state through cross-workspace symlinks.

The project owner runs the full regression suite and workspace gates. No-write validation uses only
`Demo_Workspace` or a disposable validation workspace explicitly selected by the owner; production
workspaces remain read-only. A real Rolling adapter/bootstrap smoke requires separate authorization
in a disposable validation workspace. Committed code and documentation must not contain private
workspace identities, absolute paths, or runtime data.

### Info Commands

| Flag | Purpose |
|------|---------|
| `--show-state` | Strict readonly classification: `missing` / `valid_legacy` / `corrupt` / `unsupported` |
| `--clear-state` | Clear current method's training state (auto-backup) |

> `--show-state` / `--clear-state` accept `--training-method` to target a specific mode.

## 2. State Management

| Method | State File |
|--------|-----------|
| slide | `data/rolling_state.json` |
| CPCV | `data/rolling_state_cpcv.json` |

Fully independent. `--cold-start` clears only the current method's state.

### Cross-Mode Isolation

| Operation | slide state | CPCV state | `latest_train_records.json` |
|-----------|-------------|------------|---------------------------|
| `--cold-start --training-method slide` | **Cleared** | Untouched | slide keys overwritten, CPCV keys preserved |
| `--cold-start --training-method cpcv` | Untouched | **Cleared** | CPCV keys overwritten, slide keys preserved |

Records use `merge_train_records()` — different `@mode` suffixes never collide.

## 3. Daily Workflow

```bash
# First time
python quantpits/scripts/rolling_train.py --cold-start --all-enabled \
  --training-method slide

# After weekly data update
python quantpits/scripts/rolling_train.py --merge --all-enabled
python quantpits/scripts/rolling_train.py --predict-only --all-enabled
python quantpits/scripts/rolling_train.py --backtest-only --all-enabled

# Rebuild after hyperparameter change
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158

# Compare slide vs CPCV (keys differ, can coexist)
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158 \
  --training-method slide
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158 \
  --training-method cpcv
```

## 4. Predict-Only Behavior

| Scenario | Behavior |
|----------|----------|
| No gap (all windows up to date) | Skip — prints "Nothing to predict" |
| Gap + `--allow-stale-predict` | Predict only gap range `[test_end+1d, anchor_date]` |
| Gap without flag | Skip with guidance |

Gap detection checks both window index and date difference.

## 5. Backtest Output

A successful batch uses `rolling_backtest_completed`, with
`n_requested == n_succeeded` and `n_failed == 0`. Recorder/prediction preconditions use
`rolling_backtest_precondition_failed`; Qlib execution or invalid results use
`rolling_backtest_execution_failed`; metrics/artifact write-back uses
`rolling_backtest_publication_failed`. These codes come from structured stage classification, not
from parsing the human-readable output below.

Matches training-time `PortAnaRecord` format:

```
The following are analysis results of benchmark return(1week).
                       risk
mean               0.002227
std                0.016799
annualized_return  0.115800
information_ratio  0.955909
max_drawdown      -0.066849
...
The following are analysis results of indicators(week).
     value
ffr    1.0
pa     0.0
pos    0.0
```

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `purge_steps >= smallest CV group size` | Window too short | Increase `train_years` or reduce `n_groups` |
| `Embargo pushes test_start past test_end` | embargo too large | Reduce `embargo_steps` |
| Handler cache not working | `--cache-size 0` disables it | Remove or set positive value |
| CPCV predict-only reloads data 8× | No gap → shouldn't run | Fixed: gap=0 now skips |
