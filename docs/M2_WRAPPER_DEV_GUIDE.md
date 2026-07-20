# M2: 模型 Wrapper 开发指南

## 何时创建 wrapper

添加新模型或从 qlib 原始 `module_path` 迁移到 QuantPits wrapper 系统时。
wrapper 提供 IC/IR 早停和 IC loss 支持。

## 步骤

1. **判断模型范式** — numpy-based、DataLoader-based 还是 standalone
2. **创建 `custom/pytorch_<架构>.py`** — 按对应范式模板编写
3. **创建 `lh/pytorch_<架构>.py`** — LossHistory 薄封装（5 行）
4. **更新 YAML** — 改 `module_path`，加 `metric`/`loss` kwargs
5. **验证** — 导入测试 + Playground 试跑

---

## 范式 1：Numpy-based（最常见）

qlib 基类使用 `train_epoch(x, y)`，参数为 numpy 数组。无需覆盖 hook。

**模板**（`custom/pytorch_xxx.py`）：

```python
from qlib.contrib.model.pytorch_xxx import XXXModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class XXXModel(StrategyMetricMixin, _Base):
    pass
```

**LH 模板**（`lh/pytorch_xxx.py`）：

```python
from quantpits.utils.model_wrappers.custom.pytorch_xxx import XXXModel as _Base
from quantpits.utils.model_wrappers.mixins.loss_history import LossHistoryMixin


class XXXModel(LossHistoryMixin, _Base):
    pass
```

**YAML**：

```yaml
model:
    class: XXXModel
    module_path: quantpits.utils.model_wrappers.custom.pytorch_xxx
    kwargs:
        metric: ir
        loss: ic
```

适用模型：`pytorch_lstm.py`、`pytorch_gru.py`、`pytorch_alstm.py`、`pytorch_transformer.py`、
`pytorch_localformer.py`、`pytorch_tcn.py`、`pytorch_sfm.py`、`pytorch_sandwich.py`、
`pytorch_adarnn.py`、`pytorch_krnn.py`、`pytorch_igmtf.py`、`pytorch_tabnet.py`、`pytorch_tra.py`。

---

## 范式 2：DataLoader-based + hooks

qlib 基类使用 `train_epoch(data_loader)`。覆盖 `StrategyMetricMixin` 中的四个 hook。
DataLoader 构建方式因模型而异——从基类 `fit()` 方法复制逻辑。

**模板**（`custom/pytorch_xxx.py`）：

```python
import numpy as np
from torch.utils.data import DataLoader
from qlib.data.dataset.handler import DataHandlerLP

from qlib.contrib.model.pytorch_xxx import XXXModel as _Base
from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin


class XXXModel(StrategyMetricMixin, _Base):

    def _prepare_fit_data(self, dataset):
        """构建 DataLoader。从父类 fit() 复制 DataLoader 构建逻辑。"""
        dl_train = dataset.prepare("train", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"],
                                   data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset.")
        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        # ... 权重计算、ConcatDataset（如需要）...
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

### Hook 参考

| Hook | 输入 | 输出 | 备注 |
|------|------|------|------|
| `_prepare_fit_data(dataset)` | qlib dataset | `(train_data, valid_data, valid_index)` | DataLoader 或 numpy |
| `_train_one_epoch(train_data)` | 由 prepare 返回 | None | 直接调 `self.train_epoch()` |
| `_eval_one_epoch(data)` | 由 prepare 返回 | `(loss, score)` | 直接调 `self.test_epoch()` |
| `_collect_predictions(valid_data, valid_index)` | 由 prepare 返回 + index | `(preds, labels, loss)` | 供 IR 回测使用 |

### 特殊情况

**ConcatDataset + 权重**（ALSTM_TS）：
```python
wl_train = np.ones(len(dl_train))
wl_valid = np.ones(len(dl_valid))
train_loader = DataLoader(ConcatDataset(dl_train, wl_train), ...)
```

**Channel-first transpose**（TCN_TS）：覆盖 `_collect_predictions`，
在特征提取前加 `data = torch.transpose(data, 1, 2)`。

**DailyBatchSampler**（GATs_TS）：覆盖 `_collect_predictions`，
加 `data = data.squeeze()`，用 `feature = data[:, 0:-1]`。

适用模型：`pytorch_alstm_ts.py`、`pytorch_transformer_ts.py`、`pytorch_localformer_ts.py`、
`pytorch_tcn_ts.py`、`pytorch_gats_ts.py`。

---

## 范式 3：Standalone（自定义 fit）

模型有自己的 `fit()`，无法用 hook 模式。在 wrapper 文件中添加 `_fit_ir()` 方法，
从 `fit()` 派发：

```python
class XXXModel(_Base):
    def fit(self, dataset, evals_result=None, save_path=None):
        if getattr(self, "metric", "") == "ir":
            return self._fit_ir(dataset, evals_result, save_path)
        return super().fit(dataset, evals_result, save_path)

    def _fit_ir(self, dataset, evals_result, save_path):
        from quantpits.utils.model_wrappers.mixins.strategy import StrategyMetricMixin as _SM

        # ... 复制父类的 DataLoader/numpy 设置 ...
        # ... 训练循环：train_epoch → 收集preds → 计算IR ...
        # 使用 _SM._compute_ic(), _SM._compute_rank_ic(), _SM._run_mini_backtest()
```

`_fit_ir()` 方法应：
1. 复制父类的数据准备逻辑（DataLoader 构建等）
2. 运行训练循环，每 epoch 计算 IR
3. 调用 `_SM` 静态方法计算 IC / Rank IC / mini-backtest
4. 以 IR 做早停（最大化，`best_score = -np.inf`）

适用模型：`pytorch_gats_plus.py`、`pytorch_lstm_ic_loss.py`、`pytorch_add.py`。

---

## 范式 4：Minimize-based（GeneralPTNN）

GeneralPTNN 的 `fit()` 最小化：`best_score = np.inf`，`val_score < best_score`。
IR 越高越好——取反适配。

```python
class GeneralPTNN(ICMetricMixin, _Base):
    def metric_fn(self, pred, label):
        val = super().metric_fn(pred.squeeze(), label.squeeze())
        return -val  # 取反以适配最小化

    def fit(self, dataset, ...):
        if self.metric == "ir":
            return self._fit_ir(dataset, ...)
        return super().fit(dataset, ...)
```

`_fit_ir()` 中：
```python
best_score = np.inf  # 最小化
neg_ir = -val_ir
if neg_ir < best_score:
    best_score = neg_ir
```

适用模型：`pytorch_general_nn.py`。

---

## 验证

创建 wrapper 后：

```bash
# 导入测试
python -c "from quantpits.utils.model_wrappers.custom.pytorch_xxx import XXXModel"

# Dry-run
python quantpits/scripts/static_train.py \
  --workspace workspaces/Demo_Workspace \
  --models <模型名> --dry-run

# Playground 全量训练
python quantpits/scripts/static_train.py \
  --workspace workspaces/Demo_Workspace \
  --models <模型名>
```

训练日志中需确认：
1. `metric=` 显示预期值（ir/ic/rank_ic）
2. 每 epoch 有 `val_ir` 计算和跟踪
3. IR 提升时出现 `New best IR:` 日志
4. 最终 `best IR: X @ epoch N` 正确
5. 无 `AttributeError` 关于 inner module 缺失

---

## 能力 catalog 与短合同验证

新增或删除 repository wrapper 时，必须同步更新 `quantpits/model_capabilities/catalog.py` 的 public、sanitized
model profile。不要从 workspace 的 `model_registry.yaml` 自动发现声明；filesystem inventory 与 catalog 必须精确
对账，`custom/pytorch_add.py::ADD` 等 custom-only wrapper 也不能遗漏。

每条 catalog declaration 会按 exact dataset protocol、action 和 execution family 展开为 atomic row。需要明确选择：

- `point_in_time`、`time_series`、`memory_time_series`、`daily_market_label` 或 `multi_label`；
- inference processor 是否保留无标签 prediction tail；
- CPCV 是否有显式 `PurgedDatasetH` / `PurgedTSDatasetH` / `PurgedMTSDatasetH` projection；
- artifact type/source 是否可比较，以及 optional dependency / CPU / GPU 条件。

daily/multi-label profile 没有显式 CPCV projection 时会 fail closed，不能回退成普通 `PurgedDatasetH`。import 或
signature smoke 通过也不代表 prediction coverage 安全。

如果希望 row 获得 `supported_verified`，还必须在受控 adapter 中为 exact canonical identity 建立 actual-wrapper
generated protocol：完整 identity 包含 dataset module/class、action/family、processor、artifact 和 dependency profile，
不能只匹配 wrapper module/class。adapter 必须实际验证默认 constructor、`fit(dataset, evals_result)`、
`predict(dataset)`、generated dataset/processor、artifact reload/source 以及 tail/gap/unique/finite coverage。不要公开或
手工构造 protocol measurement；测试 callback 只用于 harness negative self-test，永远不能制造 positive provenance。
暂未实现 exact adapter 的 wrapper/profile 应保留 `not_comparable`，不能复制邻近 row 的 observation。当前 actual
adapter 仅为两个 LSTM wrapper 的 `train/static` identity 建立了上述事实。

owner 可在 final candidate 上运行短合同命令：

```bash
python3.12 -m pytest tests/quantpits/model_capabilities/ -q --tb=short --no-cov
python3.12 -m pytest tests/quantpits/semantic/test_model_capability_conservation.py -q --tb=short --no-cov
```

这些 generated/tiny tests 不读取 workspace、不初始化 Qlib provider/MLflow，也不替代 Playground/发布前的真实
训练验证。能力矩阵当前只提供 render/query 和后续 preflight proposal，不会自动修改 workspace config 或启动 runner。
