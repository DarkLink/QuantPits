# M1: 模型 Wrapper 配置参考

## YAML 配置

每个模型的 `workflow_config_<模型名>.yaml` 通过 `module_path` 和 `class` 指定 wrapper。
典型配置：

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
        metric: ir          # ← 早停指标
        loss: ic             # ← 训练 loss
        GPU: 0
```

### `metric` — 早停指标

| 值 | 行为 | batch 级退路 |
|---|------|-------------|
| `ir` | 验证集轻量 TopK 回测，年化组合 IR | Pearson IC |
| `ic` | 每 epoch 的 Pearson IC（最大化） | — |
| `rank_ic` | 每 epoch 的 Spearman Rank IC（最大化） | — |
| `""`, `loss`, `mse` | 负 MSE loss（最大化 = 最小化 loss） | — |

### `loss` — 训练目标

| 值 | 行为 | 支持范围 |
|---|------|---------|
| `ic` | 1 - Pearson(pred, label) | 所有 `custom/` wrapper（通过 `StrategyMetricMixin.loss_fn`） |
| `mse` | 均方误差 | 所有模型（qlib 默认） |

### `topk` — 回测参数（workspace 级）

不在模型 YAML 中配置。从 `strategy_config.yaml` 自动注入：

```yaml
# strategy_config.yaml
strategy:
    name: topk_dropout
    params:
        topk: 22
```

---

## 模型 → Wrapper 对照表

### 支持 IR 的模型（metric: ir）

```
模型名                   class              module_path (custom.pytorch_*)
───────────────────────  ─────────────────  ───────────────────────────
lstm_Alpha360             LSTM               pytorch_lstm
lstm_Alpha158             LSTMICModel        pytorch_lstm_ic_loss
gru_Alpha360              GRU                pytorch_gru
gru_Alpha158              GeneralPTNN        pytorch_general_nn        (minimize 模式)
mlp_Alpha158              GeneralPTNN        pytorch_general_nn        (minimize 模式)
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
gats_Alpha158_plus        GATsPlus           pytorch_gats_plus        (standalone)
gats_Alpha158_origin_N    GATs               pytorch_gats_ts
add_Alpha360              ADD                pytorch_add              (每日批次)
```

### 非 PyTorch 模型（metric: ir 不支持）

```
模型名                   class              module_path                API
───────────────────────  ─────────────────  ─────────────────────────  ──────
lightgbm_Alpha158         LGBModel           qlib.contrib.model.gbdt    sklearn
catboost_Alpha158         CatBoostModel      qlib.contrib.model.catboost_model
linear_Alpha158           LinearModel        qlib.contrib.model.linear
```

### 已禁用 / 废弃

```
tcts_Alpha360, gru2mlp_Alpha158, tft_Alpha158
```

---

## LossHistory 变体

每个模型有对应的 `lh/` 版本，在验证分数之外额外记录每 epoch 的 train/valid loss。
使用时将 `module_path` 从 `custom.pytorch_xxx` 改为 `lh.pytorch_xxx`：

```yaml
# 标准版
module_path: quantpits.utils.model_wrappers.custom.pytorch_lstm

# 含 loss history
module_path: quantpits.utils.model_wrappers.lh.pytorch_lstm
```

两者 class 名相同，接受相同的 kwargs。

---

## Experiment 目录

`experiment/` 是空目录，用于一次性模型实验。将实验 wrapper 放在此处，不会被生产管线自动引用，
除非 workspace YAML 显式指定路径。
