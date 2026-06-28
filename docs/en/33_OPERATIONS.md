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
| `--dry-run` | Preview windows, no training |
| `--show-folds` | CPCV: show per-window fold details |
| `--training-method slide\|cpcv` | Override config file setting |
| `--cache-size N` | Handler cache max (MB), 0=disable. CPCV K-fold reuse |
| `--allow-stale-predict` | Allow old weights for new data in predict-only |

### Info Commands

| Flag | Purpose |
|------|---------|
| `--show-state` | Show current method's training state |
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
