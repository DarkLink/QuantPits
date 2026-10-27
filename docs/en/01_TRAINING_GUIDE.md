# QuantPits Training System Guide

## Overview

The training system consists of three main scripts that share the same utility modules and model registry:

| Script | Purpose | Training | Data Source | Save Semantics |
|------|------|------|--------|----------|
| `static_train.py --full` | Full training | ✅ | configs | `latest_train_records.json` |
| `static_train.py` | Incremental training | ✅ | configs | `latest_train_records.json` |
| `static_train.py --predict-only` | Prediction only | ❌ | Existing models | `latest_train_records.json` |
| `pretrain.py` | Base model pre-training | ✅ | configs | `data/pretrained/` (state_dict) |

Real static/CPCV commands publish through one typed execution kernel. Only one real command may hold
`data/locks/training_execution.lock` in a workspace; lightweight plans do not create it. The current
record and the run's performance output form a recoverable publication unit under
`data/training_transactions/<run-id>-<fingerprint>/`, with `latest_train_records.json` replaced last as
the canonical current pointer.
This complete contract currently applies to static/CPCV only. Rolling and CPCV-rolling still use
their existing independent lifecycle. Their real mutating commands share only the workspace training
execution lease and are not yet backed by State V3 or the publication coordinator.

Model reuse after Qlib's artifact hardening checks the actual API signature: our own selected model
artifacts use explicit `trusted=True`, older APIs retain their original call, and data artifacts keep
default loading. Verify reload / predict-only from the saved source after the first fit. No model
parameter change or historical retraining is required for this adaptation. See the
[Predict-Only guide](05_PREDICT_ONLY_GUIDE.md) for dependency provenance and missing-model fallback.

### Training Record V2

New records use `schema_version: 2`. `model_records` is the authoritative identity map: every
`model@mode` entry carries its actual output experiment, recorder, operation, prediction coverage,
and optional source-recorder lineage. Top-level `models`, `*_experiment_name`, and `anchor_date`
remain compatibility projections and must not override per-model V2 identity.

Legacy V1 files remain readable as `legacy_unverified`. Audit or preview them without mutation:

```bash
python -m quantpits.tools.audit_training_records \
  --workspace workspaces/Demo_Workspace --json
```

Default JSON contains only counts and stable issue codes. Add `--preview` explicitly in a controlled
terminal to include the in-memory migration proposal.
Use `--verify-mlflow` to validate experiment/recorder identity and workspace containment;
`--verify-predictions` additionally checks persisted `pred.pkl` coverage. In that mode the audit binds
a read-only Qlib recorder runtime explicitly to `--workspace`; runtime initialization neither creates
nor migrates a missing backend and does not write to the workspace. Full training publishes the
current registry only when every selected model succeeds and produces a verified entry.

For a custom backend, pass its actual URI with `--mlflow-uri`, for example
`--mlflow-uri "sqlite:///$PWD/mlflow.db"` from the target workspace. SQLite model and prediction
artifacts may also reside in `mlruns/`; that directory alone does not identify the metadata backend.
Training reactivation of the same resolved workspace preserves the selected `MLFLOW_TRACKING_URI`;
switching to another workspace still resolves its backend afresh.

A producer should publish `ready` only after validating the output recorder and persisted
`pred.pkl`. Failed or skipped operations must not replace an existing current pointer.

In static TRA training, `MTSDatasetH` builds prediction samples from learning data. Training retains `DropnaLabel`;
after fitting, prediction and `SignalRecord` use a separate prediction dataset with only that
processor removed, so the latest dates can produce signals without realized labels. Other learning
and inference processors and model parameters retain their configured values. Unknown labels stay
missing. Regular `DatasetH` and `TSDatasetH` reuse their original dataset, and coverage checks still
inspect the persisted `pred.pkl`.

Static full training computes its final test/forward prediction once after fitting. Static
`--predict-only` also calls the model once per actually executed target attempt, then saves that
result and Qlib's native labels. Series results become a `score` DataFrame as in SignalRecord;
DataFrames retain their format. Missing or repeated standard `SignalRecord` configurations neither
discard the result nor add inference calls. Configured `SigAnaRecord` and `PortAnaRecord` continue
to consume the saved prediction; YAML analysis/backtests, prediction dates and the `model.pkl`
retention policy stay unchanged. Unknown forward labels remain missing without dropping the
prediction endpoint. Completed targets reuse existing evidence; retrying an uncommitted failed
attempt may infer again. This does not introduce a cross-process computation cache.

This count excludes validation, early stopping, mini-backtests and batched forwards inside fit.
Training parameters, checkpoint selection and topk/n_drop used for training selection remain intact.
Stochastic or stateful models now save the first final prediction, which can differ from the old
second result. The engine neither infers again nor resets RNG state to reproduce that old sequence.
A training success history event is created only after persisted coverage validation and recorder
completion; save, record-construction or close failures produce no success event. Direct helper
history-write failures remain warnings; native runners still delegate history to the service.

Static train/predict-only compute, save artifacts and run requested analysis inside the original recorder context. Context exit uses Qlib's existing end_run to wait for asynchronous writes; completion metadata and persisted predictions are then checked through the captured recorder before producing a success entry/history. Close errors, invalid metadata/predictions and interrupts produce no success evidence. A recorder can be FINISHED while a later evidence check fails the command; FINISHED alone cannot authorize publication. Existing models need no retraining, and historical FAILED runs are not automatically repaired.
Existing performance fields, including without-cost metrics, retain their definitions.

Static full/incremental/predict-only commands can explicitly select research strategy and backtest settings:

```bash
python -m quantpits.scripts.static_train --workspace /path/to/research_workspace \
  --predict-only --models small --source-records data/training_sources/selected.json \
  --research-config config/strategy_config.yaml --anchor 2024-05-17
```

The YAML uses the existing `strategy: {name, params}` and `backtest: {account, exchange_kwargs}` structure; select the default strategy file or another file with that structure.
`backtest.account` must be finite positive numeric cash, excluding booleans, strings and initial holdings. Strategy and exchange fields retain their existing factory meanings.
Relative paths start at the workspace; absolute paths and resolved symlinks must also stay inside it. Explicit mode reads neither `prod_config.json` nor a separately merged default strategy. Missing or invalid research inputs fail without an account fallback.
Market, benchmark, frequency and date splits still come from `model_config.json` and the existing calendar; the research YAML accepts only the strategy/backtest structure above. `--anchor` remains available.
IC and requested PortAna analysis remain enabled. Changing only research cash/fees leaves model tasks unchanged and does not require retraining a saved model. Topk/n_drop remain training inputs when workflow aliases or fit selection consume them.
Plans and manifests record the canonical relative path, content fingerprint and actual conditions. Editing the file after preparation or changing conditions when resuming the same run causes an input conflict; use a new run for different conditions.
Without the option, merged configuration, `current_full_cash`/existing fallback and legacy request/resume identities are unchanged. This option is static-only. Legacy custom date hooks that accept no research conditions are rejected before execution side effects; use the default date resolver.

Training and prediction retain their configured accompanying signal/backtest analysis. For saved
project model/ensemble signals, [run_cycle evaluate](project_cycle.md#compare-saved-model-and-ensemble-signals)
provides common diagnostic cash, empty positions and windows. Changing diagnostic dates or cash
later does not require fitting, predicting or fusing again, and does not alter source metrics or accounts.

Static model/data tasks and record/backtest settings now have separate construction APIs in
`quantpits.training.static_config`, usable without workspace environment variables:

```python
from quantpits.training.static_config import build_model_task, build_record_config

model_task = build_model_task(template, compute_params)
records = build_record_config(
    template, research_params, port_analysis_config=explicit_portfolio_config,
)
```

Both copy their inputs. Model construction neither builds analysis settings, reads accounts nor
initializes Qlib/MLflow; record construction does not alter model/data inputs. The portfolio context
is an explicit Qlib strategy/backtest configuration, required when the template requests a root
PortAna configuration. Pretrained paths can be supplied explicitly; workspace path resolution,
no_pretrain and d_feat checks remain in the existing runtime adapter. Unknown model kwargs,
processor settings and topk/n_drop used for fit selection remain intact.

Static train/predict use the separate results and retain configured SigAna/PortAna. Static handler
cache enumeration reads only model/data tasks, without building records or backtest settings.
The legacy `inject_config` facade still composes the full view for rolling/pretrain callers.
YAML layout, aliases and default analysis remain compatible. Old templates whose root
data_handler_config is not aliased to the actual handler retain their previous behavior; inspect
task.dataset.handler rather than only the root field.

This does not move workspace configuration files or remove legacy merged reads from the full CLI.
calculate_dates/load_training_config retain flat params, account retains its current_full_cash
source, and full-run/resume and record configuration fingerprint definitions remain unchanged.
Research cash/cost/report changes leave the same model/dataset inputs and handler key intact;
new complete runs still follow the original request-identity rules.

V2 uses strict version dispatch. Once `schema_version: 2` is declared, matching `models` and
`model_records` mappings are mandatory; readers never fall back to top-level V1 fields. A `ready`
entry must describe actual prediction start/end/rows, while predict-only entries also carry their
immediate source recorder, experiment, and operation. Ensemble execution rechecks recorder
experiment/artifact identity and persisted `pred.pkl` coverage. The registry is evidence, not a
substitute for runtime verification.

### Preview and export an explicit model source

After a native static training run, export references for explicitly selected models. Omitting `--output` previews only, without opening a backend or loading a model:

```bash
python -m quantpits.scripts.model_source --workspace /path/to/producer \
  --run-id selected_static_run --models model_a,model_b \
  --output data/training_sources/chosen_models.json

python -m quantpits.scripts.static_train --workspace /path/to/producer \
  --predict-only --models model_a,model_b \
  --source-records data/training_sources/chosen_models.json
```

Use `--record-file data/training_sources/existing.json` instead of `--run-id` to select an explicit V1/V2 document, including latest when deliberately named. Run selection reads the original manifest, publication receipt and retained record postimage; later latest changes do not affect it. Only native static runs are reconstructed. Existing rolling/CPCV source files remain usable. File selection preserves operation and direct ancestry without guessing a prediction copy's fit root.

The preview retains request order and shows key, recorder, experiment, operation, declared status, direct source and existing date ranges. Overall run status is reported separately. Missing, failed, duplicate or ambiguous members are reported individually, return nonzero, and prevent writing a reduced success set. An unselected failed member does not block an available explicit subset. Use `model@mode` for an ambiguous bare name.

The output uses the existing training-record JSON contract and feeds `--source-records` directly. V1 retains legacy keys and metadata; V2 uses the existing serializer and preserves member entries. The tool never upgrades `legacy_unverified` to `ready`. It checks reference evidence, while existing consumers still verify recorder contents, model loading and prediction coverage. Export is not a new training or model certification result.

Input/output paths must remain inside the explicit workspace and cannot refer to the same file. Existing equivalent JSON is reported as `reused` without rewriting its bytes; different content requires a new path. Control interrupts propagate. An interrupt after atomic creation can leave a complete file, which a retry reports as reused. Only the requested source file is written: no model.pkl copy, training, prediction, latest/project changes, publication or paper account updates. Actual switching and new-cycle selection remain separate explicit requests.

To use an export for a new project cycle, put `"fit_records": {"candidate": "data/training_sources/chosen_models.json"}` in that cycle's `--inputs` JSON. Replace candidate with an existing signal ID; paths resolve inside its producer workspace. Project defaults remain unchanged and old cycles retain their own frozen sources. See [project cycles](project_cycle.md#dates-and-execution). A cycle cannot be reselected; ordinary requests need no new argument.

### Lightweight plans and run audit

`static_train.py` and `cv_train.py` share a plan-first command boundary:

```bash
python -m quantpits.scripts.static_train --full --explain-plan
python -m quantpits.scripts.static_train --predict-only --all-enabled --json-plan
python -m quantpits.scripts.cv_train --all-enabled --explain-plan
```

`--explain-plan` and `--json-plan` only read workspace-contained registry, config, workflow, current
record baseline, optional resume state, and required source-record inputs. They do not initialize
Qlib/MLflow, invoke the safeguard, change cwd,
or write files. Calendar-dependent anchors are reported as `deferred_to_qlib_calendar` rather than
guessed. Legacy `--dry-run` routes to the same lightweight plan.

Real execution writes `output/manifests/{static_train|cv_train}/<run_id>.json` by default and links
`data/operator_log.jsonl`. Use `--run-id` for an explicit identity or `--no-manifest` to disable only
the manifest. Both commands support `--workspace PATH`.

Under the training execution lease, the service prepares the actual output experiment before
the first runner or cache. New experiments use the workspace-local artifact location
`output/mlflow_artifacts/<experiment_name>`, independent of the launch cwd. Existing contained
experiments retain their IDs and locations. An output experiment pointing outside the workspace
fails before model work; it is not moved or renamed. This check applies to outputs; existing model
sources retain their read contract. Lightweight plans create no backend, experiment, or artifact directory.

After safeguard approval, the service initializes Qlib once and binds exact dates, ordered targets,
source recorders, the output experiment, and the current-record baseline into an
`execution_fingerprint`. A runner receives exactly one planned target and cannot rediscover or
broaden the target set. Before preparing a cache or starting a predict-only runner, the service also
rechecks each source recorder's actual experiment and artifact containment through the
workspace-bound MLflow backend. A conflicting legacy declaration fails closed instead of being copied
into a new V2 entry.

Real execution uses locked, compare-and-swap protected Training State V3. Its phases distinguish target
execution, publication preparation/commit, manifest closure, and terminal status. A same-identity
`--resume` classifies transaction preimages/postimages and deterministically rolls forward or closes
audit evidence; an unknown third version fails closed.

When date/calendar resolution fails before execution, the service observes the absence of execution
state, target evidence, and publication for this request, then records the request identity and
`pre_execution_failure` in the failed manifest. After correcting runtime conditions, retry the same
arguments and `--run-id` without `--resume`. Exact failed bytes are first archived in
`data/training_attempts/<run_id>/pre-execution-<sha256>.json`, then the canonical manifest path is freed.
When state exists, use the usual `--resume`: validated source/date/request identity permits archiving
a residual early failure while successful members remain reusable. Historical failures without state
and the observation identity are not assumed to be unexecuted. Different requests or sources,
successful manifests, and uncertain publication remain conflicts.

Phase 27 separates recovery evidence more explicitly. `--resume` without `--run-id` adopts the
persisted logical run ID. Every successful target writes immutable, SHA-256-addressed JSON evidence
before publication, so verified expensive work that has not yet been published is reused rather than
executed again. A receipt proves current-output commit, target evidence proves model work, and state
coordinates recovery. History events are receipt/target linked and manifests use write-or-adopt
semantics: identical evidence is adopted and conflicting evidence fails closed.
Recovery actions are selected by a pure classifier from state, target evidence, intent, receipt, and
current member observations. Unknown output bytes or identity mismatches never enter target or
publication execution. History, promotion, manifest, OperatorLog, and terminal-state work are
individually durable closure steps. A warning retains state; a same-identity `--resume` retries only
unfinished steps, and `run_state.json` is cleared only after closure is complete.
A receipt authorizes closure only when its schema/status, run/transaction, target order, complete
output ledger, and every current postimage fingerprint exactly match the intent. Missing, duplicate,
extra, or changed members fail closed. The manifest-linked OperatorLog JSONL entry is appended,
flushed, and fsynced before `operator_log_linked` completes; a write failure retains `closing` state.
Manifests use exact canonical bytes, and durable logical start/finish timing is not replaced by the
timestamp of a later closure attempt.
An older compatible State V3 is backfilled only when transaction-bound intent/receipt, complete
target evidence, current postimages, and recoverable timing prove every new field. Otherwise resume
fails closed with clear/explicit-migration guidance; no automatic migration is performed.

Publication semantics:

- Full runs commit the record/performance bundle only after every target returns verified evidence; otherwise existing bytes are preserved.
- Incremental and predict-only runs merge successes in one transaction and preserve failed pointers; the receipt is durable before the aggregate command failure is returned.
- Resume requires the same date/config/source/target resume fingerprint and skips only a published recorder that remains the current pointer; expected current-record baseline changes do not create false conflicts.
- Manifests list only outputs proven by the publication receipt; unpublished MLflow recorders remain outcome evidence only.
- Static-train promotion runs after a durable receipt. Promotion failure is a visible warning and does not roll back a verified current pointer.
- Typed static/CPCV runners do not write global history paths. The service appends stable-`event_id`, workspace-local history only for newly published receipt entries, so recovery can replay it idempotently.

---

## File Structure

```text
QuantPits/
├── quantpits/
│   ├── scripts/                      # Core system scripts
│   │   ├── static_train.py           # Unified static training entry point
│   │   ├── pretrain.py               # 🧠 Base model pre-training script
│   │   └── check_workflow_yaml.py    # 🔧 YAML config production validation & fix
│   ├── utils/                         # Shared utility modules
│   │   ├── train_utils.py            # Date calculus, YAML injection, model registry, record merging
│   │   ├── predict_utils.py          # Prediction data load/save
│   │   ├── config_loader.py          # Workspace-level config loading
│   │   ├── workspace.py              # Explicit WorkspaceContext and fingerprint helpers
│   │   ├── strategy.py               # Strategy config / backtest strategy construction
│   │   └── ...                       # More shared modules (see System Overview)
│   ├── config_contracts/              # Workspace config validation, normalization, fingerprints
│   ├── training/                      # plan/runner/lease/State V3/publication journal/service
│   └── docs/
│       └── 01_TRAINING_GUIDE.md      # This document
│
└── workspaces/
    └── <YourWorkspace>/              # Isolated trading environments
        ├── config/
        │   ├── model_registry.yaml   # 📋 Model registry (Core config)
        │   ├── model_config.json     # Date/Market parameters
        │   └── workflow_config_*.yaml# Qlib workflow bindings for each model
        ├── output/
        │   ├── predictions/          # (Managed by Qlib Recorders inside mlruns)
        │   └── model_performance_*.json # Model performance metrics (IC/ICIR)
        ├── data/
        │   ├── history/              # 📦 Auto-backed up historical files
        │   ├── pretrained/           # 🧠 Pre-trained base models (.pkl + .json)
        │   ├── run_state.json        # Locked, CAS-protected Training State V3
        │   ├── locks/                # Training execution/publication advisory locks
        │   ├── training_transactions/# Recoverable publication intents/receipts
        │   └── training_runs/        # Immutable target evidence/closure per logical run
        └── latest_train_records.json # Current training records
```

> Test policy: automatic full-suite validation is owner-operated. Implementation and review agents
> should not automatically run `pytest tests/`, coverage, Docker full-suite, or equivalent commands.
> A minimal focused test is reserved for a concrete high-risk contract that cannot be established by
> static review, and its reason and exact command must be reported at handoff.

---

## Model Registry (`config/model_registry.yaml`)

### Structure

Every model is organized by three dimensions: **algorithm** + **dataset** + **market**

```yaml
models:
  gru:                              # Model unique identifier
    algorithm: gru                  # Algorithm name
    dataset: Alpha158               # Data handler
    market: csi300                  # Target market (Metadata tag used for CLI filtering)
    yaml_file: config/workflow_config_gru.yaml  # Qlib workflow config
    enabled: true                   # Whether to participate in full training
    tags: [basemodel, ts]           # Classification tags (for filtering)
    pretrain_source: lstm_Alpha158  # (Optional) Declare dependency on base model
    notes: "Optional notes"         # Notes
```

#### Key Fields:
- **`tags: [basemodel]`**: Marks the model as a pre-trainable base model.
- **`pretrain_source`**: Tells the system which base model this upper-layer model depends on. The system will automatically look for the corresponding `_latest.pkl`.

> [!NOTE]
> **Distinction of Market Configurations**: The `market` field in the registry acts strictly as a **Model Metadata Tag** intended for CLI selection filtering via `--market` during incremental training or predictions. Actual data extraction bounds are perpetually steered by the global `market` setting inside `model_config.json`.

### Adding a New Model

1. Create a YAML workflow config at `config/workflow_config_xxx.yaml`
2. Add a model entry in `model_registry.yaml`
3. Use `static_train.py --models xxx` to train and verify it independently
4. Once confirmed, set `enabled` to `true`

### Disabling a Model

Set `enabled` to `false`. It will be automatically skipped during full training. Incremental training can still target it via `--models`.

### Available Tags

| Tag | Meaning | Models |
|------|------|------|
| `ts` | Time-series | gru, alstm, tcn, sfm, ... |
| `nn` | Neural Network | mlp, TabNet |
| `tree` | Tree-based | lightgbm, catboost |
| `attention` | Attention mechanism | alstm, transformer, TabNet |
| `baseline` | Baseline model | linear |
| `graph` | Graph model | gats |
| `cnn` | Convolutional NN | tcn |
| `basemodel` | Used as a base for others | lstm |

---

## Full Training (`static_train.py --full`)

### Usage Scenarios
- Routine production full retraining
- When all model records need a complete refresh

### Execution

```bash
cd QuantPits
python quantpits/scripts/static_train.py --full
```

### Behavior
1. Trains all `enabled: true` models in `model_registry.yaml`
2. Upon completion, **fully overwrites** `latest_train_records.json`
3. Stores exact record/performance preimages and postimages in the publication transaction directory
4. Verifies performance, replaces `latest_train_records.json` last, and writes a committed receipt

---

## Incremental Training (`static_train.py`)

### Usage Scenarios
- A new model was added, and you only want to train that one
- A model's hyperparameters were adjusted and it requires retraining
- A previously failed model needs to be re-run
- Avoiding the massive time/resource cost of full retraining

### Model Selection Methods

```bash
cd QuantPits

# 1. By name (comma-separated)
python quantpits/scripts/static_train.py --models gru,mlp

# 2. By algorithm
python quantpits/scripts/static_train.py --algorithm lstm

# 3. By dataset
python quantpits/scripts/static_train.py --dataset Alpha360

# 4. By tag
python quantpits/scripts/static_train.py --tag tree

# 5. By market
python quantpits/scripts/static_train.py --market csi300

# 6. All enabled models (merge mode)
python quantpits/scripts/static_train.py --all-enabled

# 7. Combinations
python quantpits/scripts/static_train.py --all-enabled --skip catboost_Alpha158
```

### Save Behavior (Merge Semantics)

| Condition | Behavior |
|------|------|
| Model with same name exists | Overwrites its recorder ID and performance stats |
| New model | Appended to the records |
| Untrained models | Previously existing records remain unchanged |

### Dry-run (Preview plan only)

```bash
# Preview which models will be trained without executing
python quantpits/scripts/static_train.py --models gru,mlp --dry-run
```

### Rerun / Resume (Crash Recovery)

If training is interrupted (model crashes or killed manually), the execution state is auto-saved to `data/run_state.json`.

```bash
# View last run state
python quantpits/scripts/static_train.py --show-state

# Resume unfinished training (skips successfully completed models)
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158 --resume

# Clear run state (start fresh)
python quantpits/scripts/static_train.py --clear-state
```

Explicit `--clear-state` is an operator-confirmed restart boundary, so it can also delete an
unresumable legacy-schema state without first parsing it as V3. The terminal-only cleanup used by
automatic closure still parses V3 strictly and rejects legacy or nonterminal state. Before clearing
production state, preserve evidence with `--show-state` and a file fingerprint.

**Note**: `--resume` preserves receipt-proven models that remain current and reruns failed models. If
model work and publication completed but manifest/state closure was interrupted, resume closes audit
evidence without retraining.
The ordinary form does not require `--run-id`; if one is supplied explicitly, it must match state
exactly or planning fails before Qlib initialization. The newly selected ordered target-key tuple
must also exactly match state; missing, extra, reordered, or differently selected targets are rejected
during planning before Qlib/MLflow initialization, date resolution, cache preparation, or model work.

The workspace owner explicitly runs full Python/Docker validation and the real Playground
static/CPCV predict-only release gate. Implementation tools do not automatically run the full suite
or delete/migrate legacy state.

### Viewing the Model Registry

```bash
# List all registered models
python quantpits/scripts/static_train.py --list

# Filter list by conditions
python quantpits/scripts/static_train.py --list --algorithm gru
python quantpits/scripts/static_train.py --list --dataset Alpha360
python quantpits/scripts/static_train.py --list --tag tree
```

---

## Date Handling

Training dates and frequency are controlled by `config/model_config.json`:

Fixed mode requires a valid `model_config.current_date` in `YYYY-MM-DD` format. The account's
`prod_config.current_date` remains the post-trade cursor: changing it does not change training or
prediction windows, and placeholder account dates need not be synchronized. `last_trade_date` still
uses the actual provider's observed trading calendar. This separation does not trigger fitting in
predict-only mode, which continues to load the selected old source model. Legacy `calculate_dates()`
callers follow the same date rule.

| Parameter | Description |
|------|------|
| `train_date_mode` | `last_trade_date` (uses recent trading day) or fixed date |
| `data_slice_mode` | `slide` (sliding window) or `fixed` (fixed dates) |
| `train_set_windows` | Training set window (years) |
| `valid_set_window` | Validation set window (years) |
| `test_set_window` | Test set window (years) |
| `freq` | Trading frequency (`week`/`day`) |

### Notes on Date Switching
- Full and incremental training share the same `model_config.json`
- If you change date parameters during incremental training, **the newly trained models will use the new dates**, while skipped models will remain on the old dates.
- It is highly advised to use incremental training strictly within the same anchor_date window. When rolling dates over, use full training.

---

## CPCV Mode: Purged Cross-Validation

CPCV (Combinatorial Purged Cross-Validation) follows Marcos Lopez de Prado's *Advances in Financial Machine Learning*. Validation periods are "punched out" from the training timeline with purge/embargo gaps, allowing training data from BOTH before AND after the validation period. This keeps models trained on recent market data while preventing information leakage.

### Why Use CPCV?

The traditional `slide` mode (e.g., 8-year train / 2-year valid / 3-year test) produces training data that is years old at prediction time — the model never sees recent market patterns. CPCV partitions the timeline into N equal groups, reserves the most recent groups as a fixed test set, and generates K cross-validation folds. Each fold uses discontiguous training chunks (before and after the validation group with purge gaps).

### Configuration

CPCV is independently enabled via the `purged_cv` config block in `config/model_config.json` —
**`data_slice_mode` does NOT affect CPCV.** `data_slice_mode` (`slide` / `fixed`) only controls
static training windowing and is orthogonal to CPCV:

```jsonc
{
    "data_slice_mode": "slide",    // Controls static training (slide/fixed); does NOT affect CPCV
    "purged_cv": {
        "n_groups": 10,            // Partition [start_time, anchor_date] into N equal-step groups
        "n_test_groups": 2,        // Last N groups are reserved as fixed test set (excluded from CV)
        "n_val_groups": 1,         // Each fold uses N consecutive groups as validation
        "purge_steps": 5,          // Symmetric purge: remove N steps from BOTH sides of validation
        "embargo_steps": 10        // Asymmetric embargo: additional N steps AFTER validation only
    },
    "start_time": "2015-01-01",   // Time range start (shared by CPCV and slide mode)
    "freq": "week"                 // Step unit: day or week
}
```

> [!NOTE]
> **Backward compatibility**: Existing configs with `data_slice_mode: "purged_cv"` continue to work.
> When both `data_slice_mode: "slide"` and the `purged_cv` block are configured, CPCV and static
> training can coexist in the same workspace — run `cv_train.py` and `static_train.py`
> independently without modifying the config file.

**Parameters**:

| Parameter | Meaning |
|-----------|---------|
| `n_groups` | Total groups partitioning `[start_time, anchor_date]` by equal calendar steps |
| `n_test_groups` | Trailing groups reserved as fixed test set (never used in CV) |
| `n_val_groups` | Consecutive groups used as validation per fold |
| `purge_steps` | Steps removed from BOTH sides of validation (frequency-agnostic: 1 step = 1 period of `freq`) |
| `embargo_steps` | Additional steps removed AFTER validation only |

Number of folds: **K = n_groups - n_test_groups - n_val_groups + 1**

### Running CPCV

```bash
# Full CPCV training on all enabled models
python quantpits/scripts/cv_train.py --full

# Incremental CPCV on specific models
python quantpits/scripts/cv_train.py --models lightgbm_Alpha158,gru_Alpha158

# CPCV by tag
python quantpits/scripts/cv_train.py --tag tree

# Preview fold plan
python quantpits/scripts/cv_train.py --all-enabled --dry-run

# Predict only with existing CPCV models
python quantpits/scripts/cv_train.py --predict-only --all-enabled
```

### Downstream Compatibility

CPCV-trained models are stored as `model_name@cpcv` keys in `latest_train_records.json`, coexisting with `@static` and `@rolling` models. Downstream scripts use `--training-mode cpcv`:

```bash
python quantpits/scripts/ensemble_fusion.py --from-config --training-mode cpcv
```

Each CPCV model stores K fold models (`model_fold_0.pkl` … `model_fold_K-1.pkl`) and the K-fold averaged prediction (`pred.pkl`) in the recorder. Downstream fusion loads `pred.pkl` directly — no K-fold awareness needed.

### Model Type Compatibility

| Model Type | Dataset Class | Method |
|-----------|--------------|--------|
| Tree/GBDT (LightGBM, XGBoost, CatBoost) | `PurgedDatasetH` | `pd.concat` of discontiguous time chunks (safe: point-in-time rows) |
| Linear (Linear) | `PurgedDatasetH` | Same |
| Deep Learning (LSTM, GRU, ALSTM, Transformer, TCN, GATs) | `PurgedTSDatasetH` | `ConcatTSDataSampler` logical concatenation (sliding windows never cross purge gaps) |
| TRA (Temporal Routing) | `PurgedMTSDatasetH` | Union mask on pre-computed slice indices |

### Preprocessing Notes

- **Prefer cross-sectional normalizers**: `CSZScoreNorm`, `CSRankNorm` (per-day, per-instrument — immune to temporal leakage)
- **Avoid temporal normalizers**: `ZScoreNorm`, `MinMaxNorm`, `RobustZScoreNorm` fit across time and will leak validation/test statistics. CPCV training emits a `UserWarning` when these are detected.

---

## Publication history and recovery evidence

Typed static/CPCV correctness no longer depends on timestamped `data/history/` backups. Every actual
publication uses:

```text
data/training_transactions/<transaction-id>/
├── intent.json               # Pre/post fingerprints and commit order
├── member-*.preimage         # Exact previous bytes when the member existed
├── member-*.postimage        # Fsynced intended bytes
└── receipt.json              # Verified committed outputs
```

Target and closure evidence for the same logical run is stored under:

```text
data/training_runs/<run-id>/
├── targets/<target-key>.json # Immutable typed result and recorder lineage
└── closure-<transaction-id>.json # Receipt-linked closure status and retry evidence
```

`data/history/` may still contain compatibility backups produced by legacy or rolling tools, but it is
not the recovery authority for new static/CPCV commands.

---

## Typical Workflows

### Scenario 1: Routine Training

```bash
cd QuantPits
python quantpits/scripts/static_train.py --full
python quantpits/scripts/ensemble_predict.py --method icir_weighted --backtest
```

### Scenario 1b: Predict Only after Data Update (No Retraining)

```bash
cd QuantPits
# Predict new data using existing models
python quantpits/scripts/static_train.py --predict-only --all-enabled
# The subsequent brute force/fusion pipeline remains unchanged
python quantpits/scripts/brute_force_fast.py --max-combo-size 3
python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158,alstm_Alpha158
```

> For details see [05_PREDICT_ONLY_GUIDE.md](05_PREDICT_ONLY_GUIDE.md)

### Scenario 2: Adding a New Model

```bash
# 1. Create YAML config
# 2. Add entry to model_registry.yaml (set enabled: false initially)
# 3. Train independently to verify
python quantpits/scripts/static_train.py --models new_model_name

# 4. If verified, change enabled: true
```

### Scenario 3: Retraining a Model After Param Tuning

```bash
# After modifying YAML config
python quantpits/scripts/static_train.py --models gru
```

### Scenario 4: Crash Recovery

```bash
# First run (interrupted mid-way)
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360
# ... gru completes, mlp fails, subsequent models haven't started ...

# View state
python quantpits/scripts/static_train.py --show-state

# Resume (groups gru into completed logic)
python quantpits/scripts/static_train.py --models gru,mlp,alstm_Alpha158,sfm_Alpha360 --resume
```

### Scenario 5: Training Only Tree Models

```bash
python quantpits/scripts/static_train.py --tag tree
# Equivalent to: --models lightgbm_Alpha158,catboost_Alpha158
```

---

## Configuration Validation and Auto-Fix

To ensure all YAML workflow files meet the production frequency criteria (e.g., `label` predicting future returns based on frequency, `time_per_step` matching frequency, `ann_scaler` based on frequency), an automated validation script is provided. **It is recommended to run this after adding or mutating YAMLs.**

```bash
# Validate that all workflow_config_*.yaml conform to production parameters (day/week)
python quantpits/tools/check_workflow_yaml.py

# Attempt to automatically fix all malformed YAMLs (converts params to production requirements)
python quantpits/tools/check_workflow_yaml.py --fix
```

---

---

## Base Model Pre-training (`pretrain.py`)

Complex models (e.g., GATs, ADD, IGMTF) require a pre-trained base model (e.g., LSTM or GRU) for weight initialization.

### Usage Scenarios
- Providing initialization weights for upper-layer models.
- When features (d_feat) are modified, requiring new compatible base models.

### Core Semantics
- **Pre-training is not logged in records**: It does not modify `latest_train_records.json`.
- **Metadata Validation**: Each pre-trained file comes with a `.json` metadata file. If an upper model's `d_feat` doesn't match the pre-trained file, training will be blocked.

### Common Commands

```bash
# 1. List pre-trainable models and dependencies
python quantpits/scripts/pretrain.py --list

# 2. Pre-train a specific base model
python quantpits/scripts/pretrain.py --models lstm_Alpha158

# 3. Pre-train FOR a specific upper model (Recommended: Aligns dataset config)
# This ensures perfect compatibility even if features are modified.
python quantpits/scripts/pretrain.py --for gats_Alpha158_plus

# 4. Show existing pre-trained files
python quantpits/scripts/pretrain.py --show-pretrained

# 5. Force random weights (Skip pre-training)
# Available in all modes
python quantpits/scripts/static_train.py --models gats_Alpha158_plus --no-pretrain
```

---

## Concerning LSTM and GATs

- `gats_Alpha158_plus` depends on `lstm_Alpha158` by default.
- Full Workflow:
  1. Pre-train base model (Optional if already exists):
     `python quantpits/scripts/pretrain.py --for gats_Alpha158_plus`
  2. Train upper model:
     `python quantpits/scripts/static_train.py --models gats_Alpha158_plus`

- To bypass pre-training and use random weights, use the `--no-pretrain` flag.


---

## Comprehensive Parameter List

```text
python quantpits/scripts/static_train.py --help

Mode:
  --full                  Full training: train all enabled models, overwrite records
  --predict-only          Predict only: use existing models on latest data, no training

Model Selection:
  --models TEXT           Target model names, comma-separated
  --algorithm TEXT        Filter by algorithm
  --dataset TEXT          Filter by dataset
  --market TEXT           Filter by market
  --tag TEXT              Filter by tag
  --all-enabled           Trains all models where enabled=true

Exclusions & Skips:
  --skip TEXT             Skip target models, comma-separated
  --resume                Resume from last interruption

Run Controls:
  --dry-run               Print execution plan without training
  --experiment-name TEXT  MLflow experiment name override

Information:
  --list                  List the model registry
  --show-state            Show last interruption state
  --clear-state           Clear run state file
```
