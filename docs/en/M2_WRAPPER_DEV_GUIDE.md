# M2: Model Wrapper Developer Guide

## When to create a wrapper

You need a new wrapper when adding a model that doesn't yet have one, or when
migrating a model from the original qlib `module_path` to the QuantPits wrapper
system. The wrapper provides IC/IR early-stopping and IC loss support.

## Step-by-step

1. **Identify the model's paradigm** — numpy-based, DataLoader-based, or standalone
2. **Create `custom/pytorch_<arch>.py`** — follow the pattern for that paradigm
3. **Create `lh/pytorch_<arch>.py`** — thin LossHistory wrapper (5 lines)
4. **Update the YAML** — change `module_path` and add `metric`/`loss` kwargs
5. **Import test** — `python -c "from quantpits.utils.model_wrappers.custom.pytorch_<arch> import <Class>"`

---

## Paradigm 1: Numpy-based (most common)

The qlib base model uses `train_epoch(x, y)` with numpy arrays. No hooks needed.

**Template** (`custom/pytorch_xxx.py`):

```python
from qlib.contrib.model.pytorch_xxx import XXXModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class XXXModel(StrategyMetricMixin, _Base):
    pass
```

**LH template** (`lh/pytorch_xxx.py`):

```python
from quantpits.utils.model_wrappers.custom.pytorch_xxx import XXXModel as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin


class XXXModel(LossHistoryMixin, _Base):
    pass
```

**YAML**:

```yaml
model:
    class: XXXModel
    module_path: quantpits.utils.model_wrappers.custom.pytorch_xxx
    kwargs:
        metric: ir
        loss: ic
```

Models using this: `pytorch_lstm.py`, `pytorch_gru.py`, `pytorch_alstm.py`,
`pytorch_transformer.py`, `pytorch_localformer.py`, `pytorch_tcn.py`,
`pytorch_sfm.py`, `pytorch_sandwich.py`, `pytorch_adarnn.py`, `pytorch_krnn.py`,
`pytorch_igmtf.py`, `pytorch_tabnet.py`, `pytorch_tra.py`.

---

## Paradigm 2: DataLoader-based with hooks

The qlib base model uses `train_epoch(data_loader)`. Override four hooks
in `StrategyMetricMixin`. The DataLoader construction varies per model —
copy the pattern from the base model's `fit()` method.

**Template** (`custom/pytorch_xxx.py`):

```python
import numpy as np
from torch.utils.data import DataLoader
from qlib.data.dataset.handler import DataHandlerLP

from qlib.contrib.model.pytorch_xxx import XXXModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class XXXModel(StrategyMetricMixin, _Base):

    def _prepare_fit_data(self, dataset):
        """Build DataLoaders.  Copy logic from parent fit()."""
        dl_train = dataset.prepare("train", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset.")
        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        # ... weight computation, ConcatDataset if needed ...
        train_loader = DataLoader(...)
        valid_loader = DataLoader(..., shuffle=False, drop_last=False)
        return train_loader, valid_loader, dl_valid.get_index()

    def _train_one_epoch(self, train_data):
        self.train_epoch(train_data)

    def _eval_one_epoch(self, data):
        return self.test_epoch(data)

    def _collect_predictions(self, valid_data, valid_index):
        return self._forward_all_dataloader(valid_data)
```

### Hook reference

| Hook | Input | Output | Notes |
|------|-------|--------|-------|
| `_prepare_fit_data(dataset)` | qlib dataset | `(train_data, valid_data, valid_index)` | DataLoader or numpy |
| `_train_one_epoch(train_data)` | opaque from prepare | None | Just calls `self.train_epoch()` |
| `_eval_one_epoch(data)` | opaque from prepare | `(loss, score)` | Just calls `self.test_epoch()` |
| `_collect_predictions(valid_data, valid_index)` | opaque + index | `(preds, labels, loss)` | For IR backtest |

### Special cases

**ConcatDataset with weights** (ALSTM_TS):
```python
wl_train = np.ones(len(dl_train))
wl_valid = np.ones(len(dl_valid))
train_loader = DataLoader(ConcatDataset(dl_train, wl_train), ...)
```

**Channel-first transpose** (TCN_TS): Override `_collect_predictions` to
include `data = torch.transpose(data, 1, 2)` before feature extraction.

**DailyBatchSampler** (GATs_TS): Override `_collect_predictions` to include
`data = data.squeeze()` and use `feature = data[:, 0:-1]`.

Models using this: `pytorch_alstm_ts.py`, `pytorch_transformer_ts.py`,
`pytorch_localformer_ts.py`, `pytorch_tcn_ts.py`, `pytorch_gats_ts.py`.

---

## Paradigm 3: Standalone (custom fit)

The model has its own `fit()` that cannot use the hook pattern. Add a
`_fit_ir()` method and dispatch from `fit()`:

```python
class XXXModel(_Base):
    def fit(self, dataset, evals_result=None, save_path=None):
        if getattr(self, "metric", "") == "ir":
            return self._fit_ir(dataset, evals_result, save_path)
        return super().fit(dataset, evals_result, save_path)

    def _fit_ir(self, dataset, evals_result, save_path):
        from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin as _SM

        # ... replicate parent's DataLoader/numpy setup ...
        # ... training loop: train_epoch → collect preds → compute IR ...
        # Use _SM._compute_ic(), _SM._compute_rank_ic(), _SM._run_mini_backtest()
```

The `_fit_ir()` method should:
1. Replicate the parent's data preparation (DataLoader construction, etc.)
2. Run the training loop with epoch-by-epoch IR computation
3. Use `_SM` static methods for IC/Rank IC/mini-backtest
4. Early-stop on IR (maximise, `best_score = -np.inf`)

Models using this: `pytorch_gats_plus.py`, `pytorch_lstm_ic_loss.py`,
`pytorch_add.py`.

---

## Paradigm 4: Minimize-based (GeneralPTNN)

GeneralPTNN's `fit()` minimises: `best_score = np.inf`, `val_score < best_score`.
IR is higher-is-better — negate it.

```python
class GeneralPTNN(ICMetricMixin, _Base):
    def metric_fn(self, pred, label):
        val = super().metric_fn(pred.squeeze(), label.squeeze())
        return -val  # negate for minimization

    def fit(self, dataset, ...):
        if self.metric == "ir":
            return self._fit_ir(dataset, ...)
        return super().fit(dataset, ...)
```

In `_fit_ir()`:
```python
best_score = np.inf  # minimize
neg_ir = -val_ir
if neg_ir < best_score:
    best_score = neg_ir
```

Models using this: `pytorch_general_nn.py`.

---

## Validation

After creating a wrapper:

```bash
# Import test
python -c "from quantpits.utils.model_wrappers.custom.pytorch_xxx import XXXModel"

# Dry-run training
python quantpits/scripts/static_train.py \
  --workspace workspaces/CSI300_Base_Playground \
  --models <model_name> --dry-run

# Full training in Playground
python quantpits/scripts/static_train.py \
  --workspace workspaces/CSI300_Base_Playground \
  --models <model_name>
```

Key things to verify from the training log:
1. `metric=` shows the expected value (ir/ic/rank_ic)
2. `val_ir` is computed and tracked each epoch
3. `New best IR:` appears when IR improves
4. `best IR: X @ epoch N` is correct at the end
5. No `AttributeError` about missing inner module
