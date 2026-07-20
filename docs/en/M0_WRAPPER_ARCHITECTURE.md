# M0: Model Wrapper Architecture

## Overview

QuantPits model wrappers live in `quantpits/utils/model_wrappers/`. They extend qlib's
pytorch models without modifying the upstream qlib repo, adding:

- **IC / Rank IC / IR / loss-based early stopping** via a single `metric` YAML parameter
- **IC loss** (1 - Pearson) as a trainable objective alongside standard MSE
- **Loss history tracking** for convergence diagnostics
- **Portfolio IR evaluation** via lightweight TopK backtest on the validation set

All wrappers follow the principle: **one file per model architecture**.
Engine code contains zero hardcoded strategy parameters (TopK comes from workspace config).

---

## Directory Structure

```
quantpits/utils/model_wrappers/
├── mixins/                       # Reusable components (stacked via multiple inheritance)
│   ├── ic.py                     # ICMetricMixin + ICLoss
│   ├── strategy.py               # StrategyMetricMixin (IR backtest, IC loss, metric dispatch)
│   └── loss_history.py           # LossHistoryMixin (captures per-epoch train/valid loss)
│
├── custom/                       # Production wrappers — one file per model architecture
│   ├── pytorch_lstm.py           # LSTM (numpy-based, metric: ir/ic/rank_ic/loss)
│   ├── pytorch_alstm_ts.py       # ALSTM TS (DataLoader + ConcatDataset + hooks)
│   ├── pytorch_gats_plus.py      # GATsPlus standalone (DailyBatchSampler, pretrained base)
│   ├── pytorch_lstm_ic_loss.py   # LSTM IC-loss standalone (custom test_epoch → preds/labels)
│   ├── pytorch_general_nn.py     # GeneralPTNN (minimize-based, IR negated)
│   └── ... (24 wrappers total)
│
├── lh/                           # LossHistory thin wrappers — one per model, mirrors custom/
│   ├── pytorch_lstm.py           # class LSTM(LossHistoryMixin, _CustomLSTM): pass
│   └── ... (19 wrappers)
│
└── experiment/                   # Scratch space for one-off experiments
    └── .gitkeep
```

### Naming convention

| Pattern | Example | Meaning |
|---------|---------|---------|
| `pytorch_<arch>.py` | `pytorch_lstm.py` | Standard wrapper (numpy or DataLoader) |
| `pytorch_<arch>_ts.py` | `pytorch_alstm_ts.py` | Time-series variant (DataLoader-based fit) |
| `pytorch_<arch>_ic_loss.py` | `pytorch_lstm_ic_loss.py` | Standalone model with custom `fit()` |
| `pytorch_<arch>_rank.py` | `pytorch_lstm_rank.py` | Rank-variant standalone |

---

## Mixin Stack

Each wrapper file composes a class via multiple inheritance. The MRO follows a
fixed layering:

```
Wrapper (custom/pytorch_xxx.py)
  └─ StrategyMetricMixin (mixins/strategy.py)
       ├─ metric_fn: dispatches ir → IC, ic → Pearson, rank_ic → Spearman, loss → -MSE
       ├─ loss_fn:   dispatches ic → 1-Pearson, mse → parent
       └─ fit():     dispatches ir → IR-backtest loop, else → parent
  └─ QlibBase (qlib.contrib.model.pytorch_xxx)
       ├─ train_epoch, test_epoch, predict
       └─ inner nn.Module (lstm_model, gru_model, etc.)
```

### What each mixin provides

| Mixin | Provides | Used by |
|-------|----------|---------|
| `StrategyMetricMixin` | `metric_fn` (ir/ic/rank_ic/loss), `loss_fn` (ic/mse), `fit()` (IR loop), mini-backtest, inner-module discovery | All `custom/` wrappers |
| `ICMetricMixin` | `metric_fn` (ic/rank_ic), `_batch_pearson_ic`, `_batch_rank_ic` | Only `custom/pytorch_general_nn.py` (minimize negate) |
| `LossHistoryMixin` | Monkey-patches `test_epoch` in `fit()` to capture per-epoch loss | All `lh/` wrappers |

`StrategyMetricMixin` subsumes `ICMetricMixin` for most models. The only
exception is `GeneralPTNN` which needs `ICMetricMixin` separately for minimize-based
negation.

---

## `metric` Dispatch

A single YAML key `metric` controls both the validation score (for early stopping)
and the batch-level score (for train/valid logging):

```
YAML metric value
    │
    ├── "ir"  ──→ StrategyMetricMixin.fit() IR-backtest loop
    │            metric_fn falls back to Pearson IC for batch logging
    │
    ├── "ic"  ──→ parent fit() (ICMetricMixin or base model)
    │
    ├── "rank_ic" ──→ parent fit()
    │
    └── "loss" / "mse" / "" ──→ parent fit(), negative MSE
```

When `metric: ir`, `fit()` collects full validation predictions after each epoch,
runs a lightweight TopK-equal-weight backtest (no transaction costs), and uses the
annualised portfolio IR as the early-stopping signal. Traditional IC / Rank IC is
still recorded to `evals_result` for diagnostics.

---

## Model Paradigms

### 1. Numpy-based (most models)

Standard qlib fit pattern: `dataset.prepare()` returns DataFrames, `train_epoch(x, y)`
takes numpy arrays. `StrategyMetricMixin` provides default hook implementations.

Models: LSTM, GRU, TCN, ALSTM, Transformer, Localformer, SFM, Sandwich, ADARNN,
KRNN, IGMTF, TabNet, TRA (13 models, all Alpha360).

### 2. DataLoader-based with hooks (TS variants)

`train_epoch(data_loader)` takes a DataLoader. Each model builds DataLoaders
differently (ConcatDataset with weights, simple DataLoader, DailyBatchSampler).
The wrapper overrides four hooks in `StrategyMetricMixin`:

| Hook | Purpose |
|------|---------|
| `_prepare_fit_data(dataset)` | Build train/valid DataLoaders → return (train, valid, index) |
| `_train_one_epoch(train_data)` | Call `self.train_epoch(data_loader)` |
| `_eval_one_epoch(data)` | Call `self.test_epoch(data_loader)` → (loss, score) |
| `_collect_predictions(valid_data, valid_index)` | Forward pass over valid DataLoader → (preds, labels, loss) |

Models: ALSTM_TS, Transformer_TS, Localformer_TS, TCN_TS, GATs_TS (5 models).

### 3. Standalone (custom fit)

Models with their own `fit()` that cannot use the standard hook pattern.
IR support is added via a `_fit_ir()` method directly in the wrapper file,
called from a patched `fit()`:

```python
def fit(self, dataset, evals_result=None, save_path=None):
    if getattr(self, "metric", "") == "ir":
        return self._fit_ir(dataset, evals_result, save_path)
    return super().fit(dataset, evals_result=evals_result, save_path=save_path)
```

Models: GATsPlus (DailyBatchSampler + pretrained base), LSTMICModel (custom
test_epoch returning preds/labels), ADD (daily batches + market label).

### 4. Minimize-based (GeneralPTNN)

GeneralPTNN's `fit()` uses minimization: `best_score = np.inf`, `val_score < best_score`.
IR is higher-is-better, so it must be **negated** to work with this framework.
The wrapper handles this in `_fit_ir()` by negating IR before comparison.

Models: GRU_Alpha158, MLP_Alpha158.

---

## TopK Injection

TopK is never hardcoded in engine code. The flow:

```
strategy_config.yaml (topk: 22)
  → load_workspace_config()
    → params['topk']
      → train_single_model() sets model.topk = params['topk']
        → StrategyMetricMixin.fit() uses self.topk
```

---

## Not Supported

Tree / linear models (LightGBM, CatBoost, Linear) use the sklearn API and do not
go through the PyTorch wrapper system. They keep the original qlib module paths.

---

## Wrapper Capability Matrix

`quantpits/model_capabilities/` provides a read-only, machine-readable matrix of
wrapper × dataset × action × execution-family capabilities. The public catalog
contains atomic rows for all 43 repository wrapper modules (24 under `custom/`,
19 under `lh/`) plus one sanitized `LinearModel + DatasetH` passthrough declaration.
Each model is expanded over `train`, `incremental`, `predict_only`, and `resume`,
and over `static`, `cpcv`, `rolling`, and `cpcv_rolling`. The catalog never reads a
workspace registry, workflow, recorder, or MLflow backend.

Terminal statuses are fixed:

| Status | Meaning | Allowed capability |
|---|---|---|
| `supported_verified` | Every required predicate for the exact row was inspected and passed | render/query + a future execution-preflight proposal |
| `unsupported` | The exact dataset/action/family combination is incompatible | render only |
| `conditional` | An optional dependency or device condition is unavailable | render; re-inspect after the condition changes |
| `coverage_unsafe` | A prediction tail/gap/unique/finite or processor-tail predicate failed | render only |
| `not_comparable` | Complete protocol/artifact facts were not established | render only |
| `invalid_declaration` | The strict raw identity is invalid or duplicated | audit only |
| `probe_failed` | The isolated row probe had an ordinary operational failure | render only |

A successful import, class lookup, or constructor/fit/predict signature is only a
predicate fact; none of them alone implies `supported_verified`. Serialized replay,
public construction, and `dataclasses.replace()` cannot manufacture inspector
provenance, aggregate counts, or preflight capability. The matrix is not yet wired
into the static, CPCV, or Rolling runners, and a positive row is not evidence of
training, recovery, publication, or model quality.

The default inspector is wired to a controlled generated-protocol adapter. In an
isolated temporary directory, the adapter invokes the exact actual wrapper, a tiny
dataset protocol, artifact reload, and prediction-coverage predicates. Its envelope
binds the model/wrapper, dataset module/class/protocol, action, execution family,
processor, artifact, and dependency profiles. The measurement DTO is not public API;
test injection is always `harness_self_test_only` and cannot grant positive authority,
even when every supplied value passes. A row with no exact adapter truthfully returns
`not_comparable / protocol_adapter_not_available`. The current actual adapter covers
only the `train/static` rows for the two LSTM wrappers and cannot authorize a
neighboring action/family from that measurement. Controlled import also binds a default constructor, checks
`fit(dataset, evals_result)` and `predict(dataset)`, denies workspace/backend
activation, and re-propagates process-control interrupts from the child process.

```python
from quantpits.model_capabilities import ModelCapabilityInspector

matrix = ModelCapabilityInspector().inspect_catalog()
payload = matrix.to_public_dict()
```
