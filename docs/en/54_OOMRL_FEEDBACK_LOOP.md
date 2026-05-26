# OOM-RL Feedback Loop Execution — Phase 4

Phase 4 closes the loop: ActionItems move from recommendations through sandboxed
execution to validated production changes with full audit trail.

---

## Architecture

```
action_items_{date}.json
         │
         ▼
┌─ feedback_loop.py (Orchestrator) ────────────────────────┐
│                                                          │
│  1. Compute priority + apply time budget                  │
│  2. Fork / sync Playground workspace                      │
│  3. TrainingAdapter modifies YAML configs                 │
│  4. Retrain models in Playground sandbox                  │
│  5. Validate via single-model IC comparison               │
│  6. Output feedback_report_{date}.json                    │
│  7. (Manual) promote_config.py pushes to production       │
│                                                          │
│  Modes:                                                   │
│    --report-only  Preview only, no execution              │
│    --execute      Execute, await manual promote            │
│    --promote      Push validated changes to production     │
│    --auto-promote Fully automatic (not yet implemented)   │
└──────────────────────────────────────────────────────────┘
```

---

## 1. Playground Manager

**File**: `quantpits/scripts/deep_analysis/playground_manager.py`

The Playground is a lightweight sibling copy of the production workspace at
`{workspace_name}_Playground`.

### Sync Strategy

| Path | Method | Rationale |
|------|--------|-----------|
| `config/` | Full copy | Adapter modifies freely |
| `data/*.xlsx`, `data/*.csv` | symlink | Broker data, read-only, hundreds of MB |
| `data/order_history/`, `data/history/`, `data/config_history/` | symlinked dirs | Read-only historical data |
| `data/training_history.jsonl` | Entity copy | Playground training writes independently |
| `data/fusion_run_ledger.jsonl` | Entity copy | Playground backtests write independently |
| `data/operator_log.jsonl` | Entity copy | Operation log isolation |
| `latest_train_records.json` | Entity copy | Training record isolation |
| `data/pretrained/` | Entity copy | Pretrained weights (currently unused) |
| `output/`, `mlruns/`, `archive/` | NOT synced | Independent output |

### Workspace Isolation

`env.set_root_dir(path)` switches `ROOT_DIR` at runtime and patches all 9
`train_utils` module-level path constants. Training and backtesting in the
Playground automatically write to isolated `data/` and `mlruns/` directories.

```python
from quantpits.utils import env
env.set_root_dir(playground_root)   # Switch to Playground
# train_single_model() now uses playground paths
env.set_root_dir(production_root)   # Restore production paths
```

### Core API

```python
class PlaygroundManager:
    def __init__(self, production_root: str)
    def create_or_sync(self) -> str     # Create/sync Playground, returns playground_root
    def get_playground_root(self) -> Optional[str]
    def clean(self)                     # Remove the Playground directory
    def get_meta(self) -> dict          # Read _playground_meta.json
```

---

## 2. Training Adapter

**File**: `quantpits/scripts/deep_analysis/adapters/training_adapter.py`

Converts `adjust_hyperparam` ActionItems into actual `workflow_config_*.yaml`
modifications using `ruamel.yaml` (preserves comments and formatting).

### Modification Flow

1. Look up `yaml_file` from `model_registry.yaml` for the target model
2. Read YAML with `ruamel.yaml` (round-trip mode)
3. Validate current value equals `params[key]["from"]` (safety check)
4. Validate `params[key]["to"]` against `hyperparam_bounds.json` (double-check)
5. Apply the change
6. Auto-backup to `config/_backup/workflow_config_xxx.yaml.{timestamp}`
7. Write back YAML (preserves comments)

### Pretrained Dependency Check

`check_pretrain_deps(item)` verifies the target model's `pretrain_source` is
available in the Playground:
- Missing → copy from production `data/pretrained/`
- Not in production either → warn but don't block (falls back to random init)

### Core API

```python
class TrainingAdapter:
    def __init__(self, workspace_root: str)
    def apply(self, item: ActionItem) -> AdapterResult
    def preview(self, item: ActionItem) -> dict       # Dry-run, no file writes
    def check_pretrain_deps(self, item: ActionItem) -> List[str]
```

### AdapterResult

```python
@dataclass
class AdapterResult:
    success: bool
    action_id: str
    adapter_type: str              # "training" | "search" | "fusion"
    modified_files: List[str]
    changes: List[dict]            # [{param, old, new, file}]
    error: str
```

### Adapter Registry

`adapters/__init__.py` uses a decorator-based registration pattern:

```python
@register_adapter("adjust_hyperparam")
class TrainingAdapter(BaseAdapter):
    ...
```

Future adapters (Search, Fusion) self-register by adding the decorator.

---

## 3. Feedback Loop Orchestrator

**File**: `quantpits/scripts/deep_analysis/feedback_loop.py`

### Execution Modes

| Mode | Behavior |
|------|----------|
| `--report-only` | Read ActionItems, generate preview + priority sort, **no execution** |
| `--execute` | Create Playground → Adapter → Retrain → Single-model IC validation → Report |
| `--promote` | Push previously validated changes to production |
| `--auto-promote` | Not yet implemented in Phase 4 |

### Priority Scheduling

`compute_priority(item, signal_severity, training_history)` scores ActionItems
on four dimensions:

| Dimension | Weighting |
|-----------|-----------|
| Signal severity | critical=3.0, warning=2.0, info=1.0 |
| LLM confidence | confidence × 2.0 (range 0 ~ 2.0) |
| Risk level (tiebreaker) | high=0.5, medium=0.3, low=0.0 |
| Training cost | Short training preferred: < 10 min +1.0, < 30 min +0.5 |

### Time Budget

`--max-duration-minutes N` caps total execution:
- Estimates per-model time from `training_history.jsonl` historical durations
- Unknown models default to 60-minute estimate
- Greedy selection: highest-priority items that fit within budget
- Excess items are marked `deferred` in the report

### Manual Override

```bash
--models m1,m2           # Only process specified models (overrides priority)
--skip-models m3,m4      # Exclude specific models
--max-duration-minutes 120   # Time budget
```

### Validation Strategy

**Primary: single-model IC comparison.** After retraining, compare Playground
vs. Production IC:

- Pass condition: `playground_ic >= baseline_ic * 0.9`
- Prefer condition: `ic_delta > 0`
- On failure: log the reason, **continue to the next item** (non-blocking)

**Optional Ensemble-level validation** (`--with-ensemble-backtest`): copies
non-retrained model predictions from production, substitutes only the retrained
model's prediction column, and runs `run_single_combo()`. Currently optional.

### FeedbackReport

```python
@dataclass
class FeedbackReport:
    run_date: str
    mode: str
    action_items_processed: int
    action_items_deferred: int       # Deferred due to time budget
    adapter_results: List[dict]
    validation_results: List[dict]
    deferred_action_ids: List[str]
    promote_result: Optional[dict]
    summary: str
```

Output to `output/deep_analysis/feedback_report_{date}.json`.

---

## 4. Config Promoter

**File**: `quantpits/scripts/deep_analysis/promote_config.py`

Pushes Playground-validated config changes to production with complete audit
trail.

### Promote Flow

1. `diff_snapshots(production, playground)` → change list
2. `annotate_with_llm_context()` → provenance stamping
3. Generate human-readable Promote Summary (Markdown)
4. Copy modified files to Production (`config/` only, no `output/` or `mlruns/`)
5. Save new config snapshot
6. Write `promote_history.jsonl`
7. Update `CHANGELOG.md`

> **Important**: Promote only pushes **config files**, not model weights.
> Promoted config takes effect on the next production training cycle
> (`static_train --all-enabled`).

### Promote Status Lifecycle

```
promoted_pending_retrain  →  active  →  (optional) rolled_back
```

| Status | Meaning |
|--------|---------|
| `promoted_pending_retrain` | Config pushed, awaiting next training cycle |
| `active` | Training complete, new config is live in production |
| `rolled_back` | Reverted due to validation failure or issue |

`static_train.py` and `rolling_train.py` automatically call
`update_promote_status()` after training (try/except wrapped, non-blocking) to
transition `pending_retrain` → `active`.

### Audit Artifacts

Each promote produces two records:

1. **Machine-readable**: `data/promote_history.jsonl`
```json
{
    "promote_id": "uuid",
    "promoted_at": "2026-05-05 17:30:00",
    "action_item_ids": ["55b3a485-..."],
    "changes": [{"model": "gru_Alpha158", "param": "early_stop", "old": 10, "new": 20}],
    "source": "llm_critic",
    "status": "promoted_pending_retrain",
    "retrained_at": null,
    "rolled_back_at": null,
    "rollback_reason": null,
    "validation_result": {...},
    "human_readable_report": "data/promote_history/promote_2026-05-05.md"
}
```

2. **Human-readable**: `data/promote_history/promote_{date}.md`
   - Change summary
   - Before/After comparison table
   - IC validation results table
   - ActionItem traceability (Signal → ActionItem → Promote)
   - Rollback guide (git checkout commands)

### Changelog

`data/CHANGELOG.md` is auto-maintained in reverse chronological order:

```markdown
# Demo_Workspace Configuration Changelog

## 2026-05-05: early_stop tuning — 3 models retrained, all passed
- **Source**: LLM Critic (ActionItem 55b3a485, ac08cbd1, 6ee7122e)
- **Changes**: alstm_Alpha158, lstm_Alpha360, gru_Alpha158 early_stop: 10 → 20
- **Validation**: 3/3 models IC improved (avg +0.005), promoted
- **Risk**: low
- [Detailed report](promote_history/promote_2026-05-05.md)
```

---

## 5. Rollback

Production workspace is a standalone Git repo. Rollback leverages Git:

```bash
# 1. Locate the target promote record
cat data/promote_history/promote_2026-05-05.md

# 2. Restore config via Git
cd /path/to/Demo_Workspace
git checkout <pre-promote-commit> -- config/

# 3. Mark as rolled back
# The promote_config module provides rollback helper functions
```

A rollback is itself a promote event (in the opposite direction), fully
recorded in the audit trail.

---

## 6. CLI Usage

```bash
# Report mode — preview what would be executed
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --report-only

# Execute mode — run in Playground
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute

# Execute with time budget and model filter
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute --max-duration-minutes 30 --models gru_Alpha158,alstm_Alpha158

# Dry-run — preview adapter changes without writing files
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute --dry-run

# Skip retrain — only modify configs
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --execute --skip-retrain

# Promote — push validated changes to production
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_2026-05-01.json \
    --promote
```
