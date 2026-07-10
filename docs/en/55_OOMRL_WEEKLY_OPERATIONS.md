# OOM-RL Weekly Operations Guide

Complete weekly operational workflow from data import through order generation, with RLFF feedback loop integration points.

---

## Full Workflow Overview

```
Saturday/Sunday (after data is ready)

  ── Order-blocking (must complete this week) ──
  1. Import new data                 qlib data update
         │
  2. Settlement + Post Trade          prod_post_trade.py
         │
  3. Run Deep Analysis                run_deep_analysis --critic
         │
  4. Evaluate Action Items            Pick high-confidence, single-param, or validated-direction items
         │
  5. Playground single-variable test  Isolate each param; never combo-tune
         │
  6. Promote winners → Retrain        static_train.py --models <promoted> then --predict-only --all-enabled
         │
  7. Fusion + Order Generation        ensemble_fusion.py → order_gen.py

  ── Non-blocking (can do later) ──
  8. Combo Search                     brute_force_ensemble.py --use-groups
         │
  9. Evaluate candidates → update     ensemble_config.json
```

**Core principles**:
- **Fix models first, then search combos** — combo search depends on model prediction quality
- **All adjustments must be Playground-verified before promoting to production**
- **Never skip step 5 and modify production config directly**
- **Deep analysis must run after post_trade** (needs latest cash/holdings)
- **Combo search does not block this week's orders** — it is an optimization step

---

## Step-by-Step Operations

### Step 1: Import Latest Market Data

```bash
# Update qlib data to the latest trading day
# Command depends on data source, typically:
qlib data update --region cn --freq day
```

### Step 2: Settlement Data + Post Trade

```bash
conda activate qlib_cupy
cd <project_root>
source workspaces/Demo_Workspace/run_env.sh

# Import broker settlement files, update cash and holdings
python -m quantpits.scripts.prod_post_trade
```

This updates `cash` and `holdings` in `prod_config.json`, which is required for downstream analysis.

### Step 3: Run Deep Analysis (Critic Mode)

```bash
# Optional: explain which checkpoints will be reused and which stages will run
python -m quantpits.scripts.run_deep_analysis --critic --window-analysis --explain-plan

python -m quantpits.scripts.run_deep_analysis --critic --window-analysis
```

**Output files** (under `output/deep_analysis/`):
- `action_items_{date}[_{label}].json` — LLM-generated ActionItem list
- `feedback_report_{date}[_{label}].json` — Feedback loop evaluation report
- `deep_analysis_report_{date}[_{label}].md` — Full analysis report

**Key metrics to watch**:
- `feedback_summary.accuracy` — feedback loop quality (null = no suggestions were executed last week)
- `global_diagnosis.health_status` — system health
- ActionItem `confidence` and `risk_level`
- `scope_recommendations` — blocked items due to scope restrictions

### Step 4: Evaluate Action Items

Read `action_items_{date}.json` and assess each item.

**Check model_knowledge.yaml first:**

```bash
# Check historical tuning experience for the model
cat config/model_knowledge.yaml | grep -A20 "<model_name>"
```

This file records `known_effective_params` and `known_ineffective_params` per model.
If an action item suggests a direction in `known_ineffective_params` → skip immediately.

**Decision matrix:**

| Dimension | Execute | Skip |
|-----------|---------|------|
| **model_knowledge** | Direction matches known_effective | Direction matches known_ineffective → skip |
| **confidence** | ≥ 0.5 worth considering, ≥ 0.7 usually reliable | < 0.4 reference only |
| **risk_level** | low — safe | high — needs extra analysis |
| **diversifier** | Check if Orthogonal_Wildcards member | Low IC but is_diversifier=true → keep |
| **action_type** | adjust_hyperparam / retrain via Playground | disable_model needs LOO delta evidence |
| **scope** | hyperparams / model_selection can execute directly | combo_search needs manual script |

**Special notes**:
- `disable_model` + `confidence > 0.5` + no LOO delta → violates constraint, do not execute
- `trigger_search` → run `brute_force_ensemble.py` manually (search_adapter not yet implemented)
- `keep_as_diversifier` diagnosis → Per-Model correctly identified orthogonal diversifier, no action needed

### Step 5: Playground Verification

**Never modify production workspace config directly.**

**Single-variable-first principle**: If an action item suggests changing ≥2 params simultaneously (combo tuning), isolate each parameter in separate Playground rounds. Combined changes can mask the case where "param A helps but param B hurts."

**Historical lesson**: krnn_Alpha360's dropout↑ + lr↓ combo caused IC degradation. Single-variable experiments revealed dropout↑ was harmful; lr↓ alone gave +10%. A direct combo execution would have wasted one training cycle and produced wrong conclusions.

#### 5a. Sync Playground

```bash
python -c "
from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
pm = PlaygroundManager('<project_root>/workspaces/Demo_Workspace')
pm.create_or_sync()
"
```

#### 5b. Experiment Loop (Core)

Each model follows this loop independently. Change only one parameter per round.

```
┌─────────────────────────────────────────────────────────┐
│  1. Record baseline IC (from production training_history) │
│         │                                                │
│  2. Modify Playground config (1 param only)               │
│         │                                                │
│  3. static_train --workspace Playground --models <m>      │
│         │                                                │
│  4. Compare Playground IC vs baseline IC                  │
│         │                                                │
│  5. Decision:                                            │
│     ├─ IC improves ≥5% or best_epoch improves → ✅ PASS  │
│     ├─ IC drops >5% or overfitting worsens → 🔄 RETRY    │
│     └─ IC flat → 🟡 check convergence (best_epoch?)      │
│         │                                                │
│  6. 🔄 On retry:                                         │
│     a) Record failure to model_knowledge.yaml            │
│     b) Reset config: pg_mgr.sync_single_config(<model>)  │
│     c) Pick different param or direction                  │
│     d) Go back to step 2                                  │
│         │                                                │
│  7. 1-2 retries with no improvement → ❌ GIVE_UP         │
│     Record in known_ineffective_params                    │
└─────────────────────────────────────────────────────────┘
```

**Reset config to production baseline** (required before retry):
```bash
python -c "
from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
pg_mgr = PlaygroundManager('<project_root>/workspaces/Demo_Workspace')
pg_mgr.sync_single_config('<model_name>')  # pull clean config from production
"
```

**Record experiments to model_knowledge.yaml** (always, pass or fail):
```yaml
# Pass → append to known_effective_params
known_effective_params:
  - param: dropout
    direction: increase
    evidence: "0.2→0.4: IC +5%, best_epoch 0→24 (YYYY-MM-DD)"

# Fail → append to known_ineffective_params
known_ineffective_params:
  - param: num_leaves
    direction: decrease
    evidence: "210→128: IC -9%, capacity too low (YYYY-MM-DD)"
```

**Call ExperimentAnalyzer for next-param suggestions** (optional, when unsure what to try next):
```bash
python -c "
from quantpits.scripts.deep_analysis.llm_interface import LLMInterface
llm = LLMInterface()
result = llm.suggest_next_experiment(
    model_name='model_name',
    playground_ic=0.037, baseline_ic=0.041,
    param_tried='num_leaves', from_val=210, to_val=128,
    workspace_root='<project_root>/workspaces/Demo_Workspace',
    rounds_remaining=2,
)
print(result)
"
```

#### 5c. Verification Criteria

- IC improves ≥5% or ICIR improves → ✅ PASS
- IC flat but best_epoch improves (e.g. 0→7) → ✅ Architectural fix, PASS
- IC drops >5% with healthy convergence → ❌ FAIL, retry
- 1-2 retries with no improvement → ❌ GIVE_UP, record in model_knowledge.yaml

### Step 6: Promote Successful Adjustments

```bash
# Method A: Manual promote (recommended)
# Copy Playground-verified config to production
cp workspaces/Demo_Workspace_Playground/config/workflow_config_{model}.yaml \
   workspaces/Demo_Workspace/config/workflow_config_{model}.yaml

# Method B: Programmatic promote via Python
python -c "
import shutil
from quantpits.scripts.deep_analysis.config_ledger import snapshot_configs, save_snapshot, generate_changelog

# Pre-promote snapshot + copy + post-promote snapshot + changelog
# See config_ledger.py for full API
"
```

**Promote rules**:
- Only promote adjustments with clear IC/ICIR improvement
- Ambiguous results (IC flat, risk=medium) → observe next cycle
- Worse results (IC degraded) → do NOT promote, revert in Playground

### Step 7: Training / Prediction

```bash
# 7a. Full train for promoted models (config changed, must retrain)
python -m quantpits.scripts.static_train \
    --workspace workspaces/Demo_Workspace \
    --models <promoted_model1>,<promoted_model2>

# 7b. Predict-only for remaining models (config unchanged)
python -m quantpits.scripts.static_train \
    --workspace workspaces/Demo_Workspace \
    --predict-only --all-enabled
```

**Order matters** — run 7a first (updates latest_train_records.json), then 7b.

### Step 8: Fusion + Order Generation

```bash
# Fuse model predictions into ensemble signals
python -m quantpits.scripts.ensemble_fusion

# Inspect order inputs, prediction source, and expected outputs without initializing Qlib
python -m quantpits.scripts.order_gen --explain-plan

# Generate buy/sell orders
python -m quantpits.scripts.order_gen
```

---

### Step 9: Combo Search (non-blocking, can do later)

> [!NOTE]
> Combo search is an optimization step, not required every week. It depends on model prediction quality — **fix models first, then search combos**. Candidate combos need manual review before updating `ensemble_config.json`, and do not affect orders already generated this week.

```bash
# Group-based exhaustive search (recommended)
python -m quantpits.scripts.brute_force_ensemble \
    --use-groups \
    --group-config config/combo_groups_27.yaml \
    --max-combo-size 4

# Resume if interrupted
python -m quantpits.scripts.brute_force_ensemble \
    --use-groups \
    --group-config config/combo_groups_27.yaml \
    --max-combo-size 4 \
    --resume

# Then run OOS analysis
python -m quantpits.scripts.analyze_ensembles.py \
    --metadata output/ensemble_runs/brute_force_{date}/run_metadata.json
```

**After search**:
1. Review `output/ensemble_runs/brute_force_*/summary.md` — IS/OOS top combos
2. Compare candidate combos with existing combos on OOS Calmar/Sharpe/Excess
3. If candidates are clearly better → update `ensemble_config.json`; effective next `ensemble_fusion`
4. If comparable → keep existing config, wait for more data

---

## Playground Safety Model

```
Production (Demo_Workspace)          Playground (Demo_Workspace_Playground)
┌─────────────────────────┐       ┌──────────────────────────────┐
│ config/                 │  ───→ │ config/        (full copy)    │
│   workflow_config_*.yaml│  sync │   workflow_config_*.yaml     │
│   model_registry.yaml   │       │   model_registry.yaml        │
│ data/                   │       │ data/          (mixed)       │
│   *.jsonl               │  copy │   *.jsonl      (entity copy) │
│   order_history/        │  link │   order_history/(symlink)     │
│   history/              │  link │   history/     (symlink)     │
│ output/                 │       │ output/        (independent) │
│ mlruns/                 │       │ mlruns/        (independent) │
└─────────────────────────┘       └──────────────────────────────┘
```

- **config/**: Full copy — adapter can freely modify
- **data/*.jsonl**: Entity copy — isolated training writes
- **data/history/**: Symlink — shared read-only historical data
- **output/ + mlruns/**: Fully independent — no impact on production

Create/sync Playground:
```bash
python -c "
from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
pm = PlaygroundManager('<project_root>/workspaces/Demo_Workspace')
print(pm.create_or_sync())
"
```

---

## Scope Control

Deep Analysis Critic is controlled by `feedback_scope.json`, which determines what the LLM can intervene on:

| Scope | Status | Adapter |
|-------|--------|---------|
| `hyperparams` | ✅ enabled | TrainingAdapter |
| `model_selection` | ✅ enabled | ModelSelectionAdapter |
| `combo_search` | ✅ enabled | search_adapter (not yet implemented; manual execution) |
| `strategy_params` | ❌ disabled | fusion_adapter (not yet implemented) |

Configured in `workspaces/Demo_Workspace/config/feedback_scope.json`.

---

## FAQ

### Q: Playground training errors with "lr must be float"

A: YAML parsers may treat `5e-05` as a string. Use `0.00005` instead of scientific notation.

### Q: feedback_summary.accuracy is always null

A: Action Items have never been actually executed. Execute at least 3 adjustments; the next deep_analysis will have evaluable feedback.

### Q: Synthesizer claims "dropout was already adjusted to 0.5 on XX date" but it wasn't

A: This is hallucination. All entries in `action_item_history.jsonl` default to `executed: false`. The Synthesizer now distinguishes "suggested" from "applied." If hallucination persists, check the "suggested vs executed" rules in `config/skills/synthesizer_system.md`.

### Q: gats_Alpha158_origin_N keeps getting recommended for disable

A: This model has avg_corr ≈ 0.03 and belongs to the Orthogonal_Wildcards group (is_diversifier=true). Low standalone IC is expected behavior (orthogonal diversification value > standalone prediction). Do not execute disable suggestions. The Synthesizer's LOO delta hard constraint caps confidence at 0.5.

### Q: Triage shows "0 combos → Per-Combo"

A: When all models have historical flags, the Triage may decide "fix models first, then analyze combos," skipping combo routing. This is a one-off LLM decision variance, not a code bug. If it happens two consecutive cycles, investigate the Triage routing logic. Even with 0 combos, the Synthesizer usually adds a `trigger_search` item to fill the combo analysis gap.

### Q: How to execute trigger_search ActionItems

A: search_adapter is not yet implemented. Run manually:
```bash
python -m quantpits.scripts.brute_force_ensemble --help
# or
python -m quantpits.scripts.minentropy_ensemble --help
```

---

## Quick Reference

```bash
# Full weekly operations (minimal command set)
conda activate qlib_cupy
source workspaces/Demo_Workspace/run_env.sh

# 1-2. Data + settlement
python -m quantpits.scripts.prod_post_trade

# 3. Analysis
python -m quantpits.scripts.run_deep_analysis --critic --window-analysis

# 4. Review action items
cat workspaces/Demo_Workspace/output/deep_analysis/action_items_$(date +%Y-%m-%d).json | python -m json.tool | less

# 5. Playground verification (single-variable first!)
python -c "from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager; PlaygroundManager('<project_root>/workspaces/Demo_Workspace').create_or_sync()"
# ...modify Playground config (one param at a time)...
python -m quantpits.scripts.static_train --workspace workspaces/Demo_Workspace_Playground --models <model>

# 6. Promote (after confirming IC improvement)
# ...cp config to production...

# 7. Training + prediction (full train first, then predict-only)
python -m quantpits.scripts.static_train --models <promoted_models>
python -m quantpits.scripts.static_train --predict-only --all-enabled

# 8. Fusion + orders
python -m quantpits.scripts.ensemble_fusion
python -m quantpits.scripts.order_gen

# 9. Combo search (can do later, non-blocking)
python -m quantpits.scripts.brute_force_ensemble \
    --use-groups --group-config config/combo_groups_27.yaml \
    --max-combo-size 4
```

---

## Related Documents

- [50_DEEP_ANALYSIS_GUIDE.md](50_DEEP_ANALYSIS_GUIDE.md) — Deep Analysis System
- [51_OOMRL_FEEDBACK_OVERVIEW.md](51_OOMRL_FEEDBACK_OVERVIEW.md) — Feedback Loop Overview
- [53_OOMRL_CRITIC_GUIDE.md](53_OOMRL_CRITIC_GUIDE.md) — LLM Critic Guide
- [54_OOMRL_FEEDBACK_LOOP.md](54_OOMRL_FEEDBACK_LOOP.md) — Feedback Loop Execution
- [04_POST_TRADE_GUIDE.md](04_POST_TRADE_GUIDE.md) — Post Trade Guide
- [01_TRAINING_GUIDE.md](01_TRAINING_GUIDE.md) — Training Guide
- [03_ENSEMBLE_FUSION_GUIDE.md](03_ENSEMBLE_FUSION_GUIDE.md) — Fusion Guide
- [06_ORDER_GEN_GUIDE.md](06_ORDER_GEN_GUIDE.md) — Order Generation Guide
