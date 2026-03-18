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
| 下游兼容性 | `latest_train_records.json` | `latest_rolling_records.json`（通过 `--record-file` 切换） |

### 共存架构

滚动训练与静态训练**完全独立**，共存于同一 Workspace：

```text
output/
├── predictions/               # 静态训练预测
│   └── rolling/               # 滚动训练预测（per-window CSV + 拼接 CSV）
data/
├── latest_train_records.json  # 静态训练记录
├── latest_rolling_records.json# 滚动训练记录
└── rolling_state.json         # 滚动训练运行状态（中间态，断点恢复用）
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

### 模式一：冷启动

**首次运行必须执行冷启动。** 生成所有 windows 并逐个训练。

```bash
# 全量冷启动
python quantpits/scripts/rolling_train.py --cold-start --all-enabled

# 指定模型
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158

# 追加新模型 (Merge Mode)
python quantpits/scripts/rolling_train.py --merge --models new_model_A

# Dry-run: 仅查看窗口划分
python quantpits/scripts/rolling_train.py --cold-start --dry-run --all-enabled
```

冷启动扩展功能：
- `--merge`：在已有的一批训练结果之上追加新的模型。已经训练完毕的模型将不再重复训练，仅针对新加入的模型执行全窗口的冷启动，完成后统一合并 `pred.pkl`。
- `--backtest`：附加此参数，在所有窗口训练预测合并完成后，将自动执行一次针对整体时间的完整 Qlib 回测，生成包含收益率报告与仓位数据的标准化产物。

冷启动流程：
1. 从 `rolling_config.yaml` 读取参数
2. 生成所有 rolling windows（到 anchor_date 为止）
3. 对每个 window × 每个模型执行训练 + 预测
4. 拼接所有 windows 的预测为连续时间序列
5. 保存 `latest_rolling_records.json`

### 模式二：日常模式

自动检测是否有新 window 需要训练：
- **有新 window** → 训练新 window + 重新拼接
- **无新 window** → 使用最近模型执行预测

```bash
python quantpits/scripts/rolling_train.py --all-enabled
```

### 模式三：仅预测

使用最近一个 window 训练的模型对最新数据预测：

```bash
python quantpits/scripts/rolling_train.py --predict-only --all-enabled
```

### 模式四：单独回测评估

如果之前已经运行过冷启动或合并流程，且 `latest_rolling_records.json` 存在预测记录，但缺失回测报告（或希望使用新配置重新回测），可以使用独立回测模式。此模式将跳过所有的训练和预测环节，直接使用历史拼接好的全局预测分 (`pred.pkl`) 执行完整的 Qlib 回测。

```bash
python quantpits/scripts/rolling_train.py --backtest-only
```

生成的标准化产物 (`report_normal_<freq>.pkl`, `positions_normal_<freq>.pkl` 等) 以及 indicator 分析指标将保存在 MLflow 的 Combined 实验记录下。

### 断点恢复

训练中断时，自动跳过已完成的 window × model：

```bash
python quantpits/scripts/rolling_train.py --resume
```

### 状态查看

```bash
# 查看当前状态
python quantpits/scripts/rolling_train.py --show-state

# 清除状态（重新开始）
python quantpits/scripts/rolling_train.py --clear-state
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

滚动训练的预测结果通过 `--record-file` 参数无缝衔接下游脚本：

```bash
# 穷举
python quantpits/scripts/brute_force_fast.py \
  --record-file latest_rolling_records.json

# 融合
python quantpits/scripts/ensemble_fusion.py \
  --from-config --record-file latest_rolling_records.json
```

> [!TIP]
> 静态和滚动训练的下游流程完全相同，仅通过 `--record-file` 切换数据来源。默认值为 `latest_train_records.json`（静态），指定 `latest_rolling_records.json` 即切换到滚动。

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
