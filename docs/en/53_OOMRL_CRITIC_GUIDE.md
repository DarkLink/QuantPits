# OOM-RL LLM Critic — Phase 3

The LLM Critic converts Deep Analysis agent findings into structured, executable
ActionItems. It is the pivot point where OOM-RL graduates from an analysis
system to a decision-making system.

---

## Architecture

```
7-Stage Pipeline (Stages 1-4)
      │
      ▼
Stage 5: Signal Extractor  ──→  List[Signal]   (pure rule layer, extracts from findings)
      │
      ▼
Stage 6: LLM Critic        ──→  List[ActionItem]  (LLM holistic decision-making)
      │
      ▼
ActionItem Validator  ──→  scope_status annotation (in_scope / out_of_scope / rejected)
      │
      ▼
persist_action_items()  ──→  action_items_{date}.json + action_item_history.jsonl
```

---

## 1. Signal Extractor

**File**: `quantpits/scripts/deep_analysis/signal_extractor.py`

A pure rule-based layer. Extracts structured Signals from agent `raw_metrics`.
**Does not make decisions** — all decisions are deferred to the LLM Critic.

### Signal Structure

```python
@dataclass
class Signal:
    signal_type: str    # Signal type (see table below)
    severity: str       # "critical" | "warning" | "info"
    scope: str          # Corresponding feedback_scope key
    source_agent: str   # Agent name that produced the raw data
    target: str         # Affected model/combo name
    metrics: dict       # Relevant metric data
    context: str        # Human-readable one-line description
```

### The 16 Signal Types

| Signal Type | Source Agent | Scope | Trigger Condition |
|-------------|-------------|-------|-------------------|
| `underfitting` | Model Health | hyperparams | Model in underfitting_candidates |
| `severe_underfitting` | Model Health | hyperparams | `actual_epochs < configured * 0.25` |
| `overfitting` | Model Health | hyperparams | Full epochs but IC mean < 0.03 |
| `ic_decay` | Model Health | hyperparams | scorecard ic_trend == "degrading" |
| `model_stale` | Model Health | hyperparams | stale_models with recommend_retrain=True |
| `oos_degradation` | Ensemble Eval | combo_search | OOS Calmar slope < -0.3, runs ≥ 5 |
| `oos_degradation_limited_sample` | Ensemble Eval | combo_search | Same but runs < 5 (low confidence) |
| `negative_contribution` | Ensemble Eval | model_selection | Model in consistently_negative list |
| `poor_predictor` | Prediction Audit | model_selection | Model in underperformers list |
| `regime_instability` | Market Regime | hyperparams | regime_switches ≥ 3 |
| `combo_stale` | Ensemble Eval | combo_search | Last evaluation > 30 days ago |
| `training_window_mismatch` | Training Window Analyzer | training_config | Rule-detected window config issue (bounds, ratio, train-end gap, anchor staleness, regime mismatch) |
| `cross_agent_convergence` | Cross-Agent | (best scope) | Same target flagged by ≥ 2 agents at warning |
| `time_horizon_reversal` | Cross-Agent | (best scope) | Returns trend direction reverses across OOS horizons |
| `combo_fragility` | Ensemble Eval / Cross-Agent | combo_search | Combo exhibits excessive dependency on a single model |
| `orphan_model` | Training Health | model_selection | Legacy/unused model left in config during predict-only phases |

---

## 2. LLM Critic

**File**: `quantpits/scripts/deep_analysis/llm_interface.py`

`LLMInterface.generate_action_items(signals)` converts a list of Signals into a
list of ActionItems.

### Pipeline

1. Load `config/llm_config.json` for model / API settings
2. Load `config/feedback_scope.json` for `active_scopes`
3. Load `config/hyperparam_bounds.json` for parameter bounds
4. Load `config/skills/` Markdown skill files as the system prompt
5. Build user prompt with active scopes, bounds, and signals JSON
6. Call OpenAI-compatible API (with one retry for JSON parse errors)
7. Parse JSON array response into `List[ActionItem]`

### Graceful Degradation

- No API key → returns empty list
- API failure → returns empty list
- JSON parse failure → retries once, then returns empty list

### LLM Configuration

`config/llm_config.json`:
```json
{
    "critic_model": "deepseek-v4-pro",
    "summary_model": "deepseek-v4-pro",
    "available_models": ["deepseek-v4-pro", "gpt-4o", "claude-sonnet-4"],
    "api_key_env": "DEEPSEEK_API_KEY",
    "base_url": "https://api.deepseek.com",
    "temperature": 0.3,
    "max_tokens": 393216
}
```

### Skill Files

Markdown files under `config/skills/` are concatenated to form the system prompt:

| Skill File | Content |
|-----------|---------|
| `critic_system.md` | Core decision principles (conservative tuning, data-driven, verifiable, scope-aware) and JSON output format |
| `hyperparam_tuning.md` | Domain knowledge: underfitting → increase n_epochs/hidden_size; overfitting → increase dropout/l2_leaf_reg |
| `model_selection.md` | Model selection criteria (disable: negative contribution ≥ 3 evals; enable: IC > 0.03 and ICIR > 0.3) |
| `summary_system.md` | Executive summary writing guidance |

---

## 3. ActionItem

**File**: `quantpits/scripts/deep_analysis/action_items.py`

### ActionItem Structure

```python
@dataclass
class ActionItem:
    action_id: str              # UUID, auto-generated
    action_type: str            # "adjust_hyperparam" | "disable_model" | "trigger_search" | "adjust_training_window"
    scope: str                  # "hyperparams" | "model_selection" | "combo_search" | "strategy_params" | "training_config"
    target: str                 # Target model/combo name (e.g., "alstm_Alpha158")
    params: dict                # e.g., {"early_stop": {"from": 10, "to": 20}}
    reason: str                 # LLM rationale for the change
    source_signals: list        # signal_type values that triggered this
    expected_outcome: str       # Expected effect description
    confidence: float           # LLM self-assessed confidence [0, 1]
    risk_level: str             # "low" | "medium" | "high"

    # Validation layer (set by ActionItemValidator)
    scope_status: str           # "pending" | "in_scope" | "out_of_scope" | "rejected"
    rejected_reason: str        # Reason for rejection
    validated_at: str           # Validation timestamp

    # Phase 4 execution context
    execution_context: dict     # {target_env, requires_retrain, requires_backtest,
                                #  estimated_duration_minutes, dependencies}
```

### ActionItemValidator

Validation rules (applied in order):
1. **Scope check**: `scope` not in `active_scopes` → `out_of_scope`
2. **Value range check**: `params[key]["to"]` outside `hyperparam_bounds.json`
   `[min, max]` → `rejected`
3. **Change magnitude check**: percentage change exceeds `max_change_pct` → `rejected`
4. **Unknown params**: allowed through with a warning

### Hyperparameter Bounds

`config/hyperparam_bounds.json`:
```json
{
    "bounds": {
        "n_epochs":    {"min": 10, "max": 500,   "max_change_pct": 50},
        "lr":          {"min": 1e-5, "max": 1e-2, "max_change_pct": 100},
        "learning_rate": {"min": 1e-5, "max": 1.0, "max_change_pct": 100},
        "hidden_size": {"min": 16, "max": 512,    "max_change_pct": 100},
        "num_layers":  {"min": 1, "max": 8,       "max_change_pct": 100},
        "dropout":     {"min": 0.0, "max": 0.8,   "max_change_pct": null},
        "batch_size":  {"min": 64, "max": 16384,  "max_change_pct": 100},
        "iterations":  {"min": 100, "max": 10000, "max_change_pct": 50},
        "depth":       {"min": 2, "max": 16,      "max_change_pct": 100},
        "l2_leaf_reg": {"min": 0.0, "max": 100.0, "max_change_pct": null},
        "num_leaves":  {"min": 8, "max": 256,     "max_change_pct": 100},
        "early_stop":  {"min": 5, "max": 100,     "max_change_pct": null}
    }
}
```

### Data Persistence

```python
persist_action_items(items, workspace_root, run_date)
```

Writes to two locations:
- `output/deep_analysis/action_items_{date}.json` — full snapshot
- `data/action_item_history.jsonl` — append audit trail (includes `_run_date`)

---

## 4. CLI Usage

```bash
# Run Deep Analysis with Critic (produces ActionItems)
python -m quantpits.scripts.run_deep_analysis --critic

# Preview mode: generate ActionItems without persisting to files
python -m quantpits.scripts.run_deep_analysis --critic-dry-run

# With LLM executive summary
python -m quantpits.scripts.run_deep_analysis --llm --critic

# Limit to specific agents
python -m quantpits.scripts.run_deep_analysis --critic --agents model_health,ensemble_eval
```
