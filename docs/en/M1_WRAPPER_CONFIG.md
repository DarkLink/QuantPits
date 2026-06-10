# M1: Model Wrapper Configuration

## YAML Reference

Each model's `workflow_config_<model>.yaml` specifies the wrapper via `module_path`
and `class`. A typical configuration:

```yaml
model:
    class: LSTM
    module_path: quantpits.utils.model_wrappers.custom.pytorch_lstm
    kwargs:
        d_feat: 6
        hidden_size: 64
        num_layers: 2
        dropout: 0.3
        n_epochs: 200
        lr: 0.001
        early_stop: 20
        batch_size: 2048
        metric: ir          # ← early-stopping criterion
        loss: ic             # ← training loss
        GPU: 0
```

### `metric` — Early-stopping criterion

| Value | Behaviour | Batch-level fallback |
|-------|-----------|---------------------|
| `ir` | Portfolio IR via lightweight TopK backtest on validation set | Pearson IC |
| `ic` | Per-epoch Pearson IC (maximise) | — |
| `rank_ic` | Per-epoch Spearman Rank IC (maximise) | — |
| `""`, `loss`, `mse` | Negative MSE loss (maximise = minimise loss) | — |

### `loss` — Training objective

| Value | Behaviour | Supported by |
|-------|-----------|-------------|
| `ic` | 1 - Pearson(pred, label) | All `custom/` wrappers (via `StrategyMetricMixin.loss_fn`) |
| `mse` | Mean squared error | All models (qlib default) |

### `topk` — Backtest parameter (workspace-level)

Not set in per-model YAML. Injected from `strategy_config.yaml` via
`train_single_model()`:

```yaml
# strategy_config.yaml
strategy:
    name: topk_dropout
    params:
        topk: 22
```

---

## Model → Wrapper Mapping

### IR-ready models (metric: ir supported)

```
model                    class              module_path (custom.pytorch_*)
────────────────────────  ─────────────────  ───────────────────────────
lstm_Alpha360             LSTM               pytorch_lstm
lstm_Alpha158             LSTMICModel        pytorch_lstm_ic_loss
gru_Alpha360              GRU                pytorch_gru
gru_Alpha158              GeneralPTNN        pytorch_general_nn        (minimize mode)
mlp_Alpha158              GeneralPTNN        pytorch_general_nn        (minimize mode)
alstm_Alpha360            ALSTM              pytorch_alstm
alstm_Alpha158            ALSTM              pytorch_alstm_ts
transformer_Alpha360      TransformerModel   pytorch_transformer
transformer_Alpha158      TransformerModelIC pytorch_transformer_ts
localformer_Alpha360      LocalformerModel   pytorch_localformer
localformer_Alpha158      LocalformerModelIC pytorch_localformer_ts
tcn_Alpha360              TCN                pytorch_tcn
tcn_Alpha158              TCNIC              pytorch_tcn_ts
sfm_Alpha360              SFM                pytorch_sfm
sandwich_Alpha360         Sandwich           pytorch_sandwich
adarnn_Alpha360           ADARNN             pytorch_adarnn
krnn_Alpha360             KRNN               pytorch_krnn
igmtf_Alpha360            IGMTF              pytorch_igmtf
TabNet_Alpha158           TabnetModel        pytorch_tabnet
tra_Alpha360              TRAModelIC         pytorch_tra
tra_Alpha158_full         TRAModelIC         pytorch_tra
gats_Alpha158_plus        GATsPlus           pytorch_gats_plus         (standalone)
gats_Alpha158_origin_N    GATs               pytorch_gats_ts
add_Alpha360              ADD                pytorch_add               (daily batches)
```

### Non-pytorch models (metric: ir not supported)

```
model                    class              module_path                API
────────────────────────  ─────────────────  ─────────────────────────  ──────
lightgbm_Alpha158         LGBModel           qlib.contrib.model.gbdt    sklearn
catboost_Alpha158         CatBoostModel      qlib.contrib.model.catboost_model
linear_Alpha158           LinearModel        qlib.contrib.model.linear
```

### Disabled / deprecated

```
tcts_Alpha360, gru2mlp_Alpha158, tft_Alpha158
```

---

## LossHistory Variants

Each model has an `lh/` counterpart that records per-epoch train/valid loss
alongside the validation score. To use, change `module_path` from `custom.pytorch_xxx`
to `lh.pytorch_xxx`:

```yaml
# Standard
module_path: quantpits.utils.model_wrappers.custom.pytorch_lstm

# With loss history
module_path: quantpits.utils.model_wrappers.lh.pytorch_lstm
```

Both use the same `class` name and accept the same kwargs.

---

## Experiment Directory

`experiment/` is an empty scratch directory for one-off model experiments.
Add experimental wrappers here; they won't be picked up by production pipelines
unless explicitly referenced in a workspace YAML.
