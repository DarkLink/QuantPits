# 滚动训练指南 (Rolling Training)

> 30 系列文档专注于**非静态训练**逻辑——即训练窗口随时间推进而滚动的训练范式。

---

## 概述

传统静态训练（`static_train.py --full`、`static_train.py`）使用**固定的日期区间**训练模型。当市场风格发生漂移时，静态模型的预测质量会逐渐衰减。

**滚动训练 (Rolling Training)** 通过将时间轴切分为多个滑动窗口，在每个窗口上独立训练模型，从而使模型始终适应最新的市场状态。

### 静态 vs. 滚动

| 特性 | 静态训练 | 滚动训练 |
|------|---------|---------|
| 训练区间 | 固定（如 2015–2022） | 滑动窗口（每窗口独立训练） |
| 模型数量 | 每模型 1 个 | 每模型 × N 个窗口 |
| 适应性 | 低（依赖长期统计特征） | 高（随市场风格滑动更新） |
| 预测输出 | 单段连续预测 | 多段拼接（自动拼接为连续文件） |
| 下游兼容性 | 统一写入 `latest_train_records.json` 下的 `model@rolling` 键（通过 `--training-mode rolling` 切换） |

### 共存架构

滚动训练与静态训练**完全独立**，共存于同一 Workspace：

```text
output/
├── predictions/               # 静态训练预测
data/
├── latest_train_records.json  # 统一训练记录 (含 @rolling)
├── rolling_state.json         # 滚动运行状态进度（断点续跑）
```

---

## 核心脚本

| 脚本 | 用途 |
|------|------|
| `rolling_train.py` | 滚动训练主脚本：冷启动、日常模式、仅预测、断点恢复 |

---

## 时间窗口划分

### 配置参数

在 `config/rolling_config.yaml` 中配置：

```yaml
rolling_start: "2020-01-01"   # T: 起始日期
train_years: 3                # X: 训练区间（整数年）
valid_years: 1                # Y: 验证区间（整数年）
test_step: "3M"               # Z: 测试步长（nM 或 nY）
```

### 划分公式

对于第 `n` 个窗口（从 0 开始）：

```
Train: [T + nZ,       T + X + nZ − 1d]
Valid: [T + X + nZ,   T + X + Y + nZ − 1d]
Test:  [T + X + Y + nZ, T + X + Y + (n+1)Z − 1d]
```

> [!IMPORTANT]
> **绝对不重叠**：训练、验证、测试三段之间没有任何日期重叠，包括端点。`train_end + 1d = valid_start`，`valid_end + 1d = test_start`。

### 示例

`T=2020-01-01, X=3年, Y=1年, Z=3M`：

| 窗口 | 训练区间 | 验证区间 | 测试区间 |
|:---:|---------|---------|---------|
| W0 | 2020-01-01 ~ 2022-12-31 | 2023-01-01 ~ 2023-12-31 | 2024-01-01 ~ 2024-03-31 |
| W1 | 2020-04-01 ~ 2023-03-31 | 2023-04-01 ~ 2024-03-31 | 2024-04-01 ~ 2024-06-30 |
| W2 | 2020-07-01 ~ 2023-06-30 | 2023-07-01 ~ 2024-06-30 | 2024-07-01 ~ 2024-09-30 |
| W3 | 2020-10-01 ~ 2023-09-30 | 2023-10-01 ~ 2024-09-30 | 2024-10-01 ~ 2024-12-31 |

最后一个窗口的 `test_end` 自动截断至 `anchor_date`（Qlib 最新交易日）。

---

## 运行模式

### 参数速查

| 参数 | 对 state 的影响 | 训练范围 | 使用场景 |
|------|----------------|---------|---------|
| `--cold-start` | **清空全部** | 全部窗口 | 首次使用、彻底重建 |
| `--merge` | **不删除**，只补缺 | 仅缺失窗口 | qlib 数据更新后补训新窗口、追加新模型 |
| `--retrain-models` | **清空指定模型** | 该模型全部窗口 | 模型超参/代码变更后重建 |
| `--retrain-last` | **删除最后窗口** | 最后窗口 | 数据修正、最后窗口重训 |
| `--predict-only` | 不修改 | 不训练，仅预测 | 仅需要最新预测时 |
| `--resume` | 不修改 | 未完成的窗口 | 中断后继续 |
| `--clear-state` | **清空全部** | — | 放弃所有历史，重新开始 |

### 核心概念

所有涉及训练的 mode 共享同一个底层逻辑：
1. 从 `rolling_config.yaml` 读取参数，结合当前 qlib 数据日期生成完整 window 列表
2. 对每个 model × window，检查 `rolling_state.json` 中是否已有记录
3. **有记录 → 跳过**，**无记录 → 训练**
4. 拼接全部 window 的 test 段预测 + 可选 `--backtest`

这保证了同一窗口不会重复训练，不同模式只是"哪些记录需要清空"不同。

---

### 模式一：全量冷启动 (`--cold-start`)

**清空 `rolling_state.json` 中全部记录**，从头训练所有模型的全部窗口。

```bash
python quantpits/scripts/rolling_train.py --cold-start --all-enabled
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158
python quantpits/scripts/rolling_train.py --cold-start --dry-run --all-enabled  # 仅查看
```

> [!CAUTION]
> `--cold-start` 不可逆，所有模型的全部训练记录永久丢失。

---

### 模式二：补训 (`--merge`)

**不删除任何已有记录**，自动检测并训练缺失的窗口。是日常最常用的维护命令。

**典型场景**：

1. **qlib 数据更新后补训新窗口**（如月末进入下月，出现 W41）
   ```bash
   python quantpits/scripts/rolling_train.py --merge --all-enabled
   ```
   内部流程：生成窗口列表 → 检测到 W41 不在 state 中 → 仅训练 W41 → 重新拼接预测。

2. **追加全新模型到已有状态**
   ```bash
   python quantpits/scripts/rolling_train.py --merge --models new_model_A
   ```
   新模型在 state 中无任何记录 → 训练全部 41 个窗口。已有模型不受影响。

> [!TIP]
> 这就是"补训最后一个窗口"的标准做法。不需要 `--retrain-last`（那是强制重训已存在的窗口），直接用 `--merge` 检测并训练新出现的窗口。

---

### 模式三：重建指定模型 (`--retrain-models`)

**清空指定模型在 state 中的全部记录**，从头训练该模型的所有窗口。其他模型完全不受影响。

```bash
# 单个模型
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158

# 批量
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158,gru_Alpha158
```

适用场景：模型超参调整后、代码变更后、wrapper 升级后。

---

### 模式四：重训最后窗口 (`--retrain-last`)

**删除 state 中最后窗口的记录**，然后通过日常模式补训。适用于数据修正等需要强制重训最后一个窗口的场景。

```bash
# 全部模型的最后窗口
python quantpits/scripts/rolling_train.py --retrain-last --all-enabled

# 限定模型范围
python quantpits/scripts/rolling_train.py --retrain-last --models gru_Alpha360
```

> [!TIP]
> 如果是 qlib 数据更新导致出现**新窗口**（不是重训已有窗口），请用 `--merge`。

---

### 模式五：仅预测 (`--predict-only`)

不训练，使用最新 window 权重对当前数据预测。

```bash
python quantpits/scripts/rolling_train.py --predict-only --all-enabled
```

**窗口落后检测**：当 qlib 数据已更新但未补训新窗口时：

1. 检测每个模型是否有未训练的窗口（state 最新 window < 当前可用 window）
2. **默认行为**：跳过有 gap 的模型，提示 `--retrain-models` 或 `--allow-stale-predict`
3. **`--allow-stale-predict`**：显式开启后用旧权重尽力预测，事后可通过补训覆盖

---

### 模式六：断点恢复 (`--resume`)

训练中断时，跳过已完成的 window，从断点继续。

```bash
python quantpits/scripts/rolling_train.py --resume
```

---

### 模式七：独立回测 (`--backtest-only`)

跳过训练和预测，直接对已有拼接预测执行全量回测。

```bash
python quantpits/scripts/rolling_train.py --backtest-only
```

---

### 日常运维速查

```bash
# 每周数据更新后
python quantpits/scripts/rolling_train.py --merge --all-enabled    # 补训新窗口
python quantpits/scripts/rolling_train.py --predict-only --all-enabled  # 预测

# 调参后重建
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158

# 数据修正
python quantpits/scripts/rolling_train.py --retrain-last --all-enabled

# 查看状态
python quantpits/scripts/rolling_train.py --show-state
```

---

## 模型选择

与静态训练一致，支持所有模型筛选方式：

| 参数 | 说明 |
|------|------|
| `--models m1,m2` | 按名称指定 |
| `--algorithm alg` | 按算法筛选 |
| `--dataset ds` | 按数据集筛选 |
| `--tag tag` | 按标签筛选 |
| `--all-enabled` | 所有 enabled 模型 |
| `--skip m1,m2` | 排除指定模型 |

---

## 下游衔接

滚动训练的预测结果通过 `--training-mode rolling` 参数无缝衔接下游脚本：

```bash
# 穷举
python quantpits/scripts/brute_force_fast.py \
  --training-mode rolling

# 融合
python quantpits/scripts/ensemble_fusion.py \
  --from-config --training-mode rolling
```

> [!TIP]
> 静态和滚动训练的下游流程完全相同，由于使用统一记录文件，仅通过 `--training-mode rolling` 即可过滤对应的滚动模型。默认寻找的是静态模型 (`@static`)。

---

## 状态管理与断点恢复

`rolling_state.json` 记录训练进度，结构如下：

```json
{
    "started_at": "2025-03-14 10:00:00",
    "rolling_config": {"test_step": "3M", ...},
    "anchor_date": "2025-03-14",
    "total_windows": 4,
    "completed_windows": {
        "0": {"linear_Alpha158": "rec_001", "gru_Alpha158": "rec_002"},
        "1": {"linear_Alpha158": "rec_003"}
    }
}
```

- 每完成一个 window × model，立即保存状态
- 中断后使用 `--resume` 恢复，自动跳过已完成项
- `--clear-state` 清除状态重新开始（旧状态自动备份到 `data/history/`）

---

## MLflow 实验命名

| 实验名 | 内容 |
|--------|------|
| `Rolling_Windows_{FREQ}` | 各 window 的单独训练记录 |
| `Rolling_Combined_{FREQ}` | 拼接后的完整预测记录 |

其中 `{FREQ}` 为交易频率（如 `WEEK`、`DAY`）。

---

## 配置文件参考

`config/rolling_config.yaml` 完整示例：

```yaml
# Rolling Training Configuration
# 滚动训练配置

rolling_start: "2020-01-01"   # T: 起始日期
train_years: 3                # X: 训练区间长度（整数年）
valid_years: 1                # Y: 验证区间长度（整数年）
test_step: "3M"               # Z: 测试步长
                              #   - nM: n 个月 (如 3M, 6M)
                              #   - nY: n 年 (如 1Y)
```

> [!CAUTION]
> `train_years` 和 `valid_years` 必须为**整数年**。`test_step` 必须为 `nM`（整数月）或 `nY`（整数年），不支持小数。
