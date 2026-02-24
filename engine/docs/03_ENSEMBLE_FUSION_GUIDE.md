# 03 Ensemble Fusion 使用指南

## 概述

`scripts/ensemble_fusion.py` 用于对**用户选定的模型组合**进行融合预测、回测和风险分析。

**支持多组合模式**：在 `config/ensemble_config.json` 中定义多个 combo，标记一个 `default`，一次运行所有组合并对比绩效。

**工作流位置**: 训练 → 暴力穷举 → 选组合 → **融合回测（本步）** → 订单生成

## 快速开始

```bash
cd QuantPits

# 1. 等权融合（直接指定模型）
python engine/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# 2. 从 ensemble_config.json 读取 default combo
python engine/scripts/ensemble_fusion.py --from-config

# 3. 运行指定 combo
python engine/scripts/ensemble_fusion.py --combo combo_A

# 4. 运行所有 combo 并生成对比
python engine/scripts/ensemble_fusion.py --from-config-all
```

## 完整参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--models` | 无 | 逗号分隔的模型名列表（直接指定，优先级最高） |
| `--from-config` | false | 从 `config/ensemble_config.json` 读取 default combo |
| `--from-config-all` | false | 运行所有 combo 并生成跨组合对比 |
| `--combo` | 无 | 运行指定名称的 combo |
| `--method` | `equal` | 权重模式: `equal` / `icir_weighted` / `manual` / `dynamic` |
| `--weights` | 无 | 手动权重，如 `"gru:0.6,linear_Alpha158:0.4"` |
| `--freq` | `week` | 回测频率: `day` / `week` |
| `--record-file` | `latest_train_records.json` | 训练记录文件 |
| `--output-dir` | `output/ensemble` | 输出目录 |
| `--no-backtest` | false | 跳过回测 |
| `--no-charts` | false | 跳过图表生成 |
| `--start-date` | 无 | 过滤数据的开始日期 YYYY-MM-DD |
| `--end-date` | 无 | 过滤数据的结束日期 YYYY-MM-DD |
| `--only-last-years N` | `0` | 仅使用最后 N 年数据 (专为 OOS 测试设计) |
| `--only-last-months N` | `0` | 仅使用最后 N 个月数据 (专为 OOS 测试设计) |

## 多组合配置

### 配置格式 (`config/ensemble_config.json`)

```json
{
  "combos": {
    "combo_A": {
      "models": ["gru", "linear_Alpha158", "TabNet_Alpha158", "mlp"],
      "method": "equal",
      "default": true,
      "description": "原始四模型等权组合"
    },
    "combo_B": {
      "models": ["gru", "linear_Alpha158", "alstm_Alpha158"],
      "method": "icir_weighted",
      "default": false,
      "description": "三模型 ICIR 加权"
    }
  },
  "min_model_ic": 0.00
}
```

**要点**：
- `combos` 字典，每个 key 是 combo 名称
- 每个 combo 需要 `models` 和 `method` 字段
- 恰好一个 combo 标记 `"default": true`
- 脚本兼容旧格式（单 `models` + `ensemble_method`）

## 运行模式

### 单组合模式

```bash
# 直接指定模型（不使用配置文件）
python engine/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method equal

# 从配置文件读取 default combo
python engine/scripts/ensemble_fusion.py --from-config

# 运行指定 combo
python engine/scripts/ensemble_fusion.py --combo combo_B
```

### 多组合模式

```bash
# 运行所有 combo + 生成跨组合对比
python engine/scripts/ensemble_fusion.py --from-config-all
```

### OOS 验证测试模式 (Out-Of-Sample)

如果你在寻找最佳组合时（使用 `brute_force_fast.py`）使用了 `--exclude-last-years 1` 等参数排除了今年的数据作为 OOS。在最终选定组合准备发车前，你可以使用以下命令单独测试组合在这最后 1 年的 OOS 纯外推表现：

```bash
# ========================================
# 仅在最近 1 年的 OOS 数据上测试此组合的表现
# ========================================
python engine/scripts/ensemble_fusion.py --from-config --only-last-years 1
```

此模式下，生成的回测净值、归因指标都将**严格限制在最后 1 年的数据上**。

此模式会：
1. 一次性加载所有 combo 涉及的模型预测（共享数据，避免重复加载）
2. 逐 combo 执行 Stage 2-8（相关性分析 → 权重 → 融合 → 保存 → 回测 → 风险分析 → 图表）
3. 生成跨组合对比表和净值对比图

## 权重模式

### `equal` — 等权（默认）
每个模型权重相同。简单可靠，作为基线。

### `icir_weighted` — ICIR 加权
按模型的 ICIR 指标分配权重，ICIR 越高权重越大。

### `manual` — 手动指定
通过 `--weights` 参数或 combo 配置中的 `manual_weights` 字段指定。

```bash
python engine/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158 \
  --method manual \
  --weights "gru:0.6,linear_Alpha158:0.4"
```

### `dynamic` — 动态权重
使用 60 天滚动窗口评估各模型 TopK 持仓的 Sharpe，动态分配权重。

## 处理流程

```
Stage 0: 初始化 Qlib + 加载配置
Stage 1: 加载选定模型预测 + Z-Score 归一化（所有 combo 共享）
--- 以下逐 combo 执行 ---
Stage 2: 相关性分析（仅该 combo 模型）
Stage 3: 权重计算
Stage 4: 信号融合
Stage 5: 保存预测结果
Stage 6: 回测 (可跳过)
Stage 7: 风险分析 + 排行榜
Stage 8: 可视化 (可跳过)
--- 多组合模式额外步骤 ---
跨组合对比表 + 净值对比图
```

## 输出文件

### 单组合模式（`--models` 或 `--from-config`）

```
output/
├── predictions/
│   └── ensemble_{anchor_date}.csv            # 融合预测
└── ensemble/
    ├── ensemble_fusion_config_{date}.json     # 融合配置
    ├── correlation_matrix_{date}.csv          # 相关性矩阵
    ├── leaderboard_{date}.csv                 # 绩效排行榜
    ├── ensemble_nav_{date}.png                # 净值曲线
    └── ensemble_weights_{date}.png            # 动态权重图 (dynamic 模式)
```

### 多组合模式（`--from-config-all` 或 `--combo`）

```
output/
├── predictions/
│   ├── ensemble_combo_A_{date}.csv           # combo_A 融合预测
│   ├── ensemble_combo_B_{date}.csv           # combo_B 融合预测
│   └── ensemble_{date}.csv                   # default combo 兼容文件
└── ensemble/
    ├── ensemble_fusion_config_combo_A_{date}.json
    ├── ensemble_fusion_config_combo_B_{date}.json
    ├── combo_comparison_{date}.csv           # 跨组合对比表
    └── combo_comparison_{date}.png           # 净值对比图
```

> [!TIP]
> Default combo 会额外保存一份不带 combo 名的 `ensemble_{date}.csv`，确保向后兼容 `order_gen.py` 等下游脚本。

## 典型工作流

```bash
# Step 1: 训练所有模型
python engine/scripts/weekly_train_predict.py

# Step 2: 暴力穷举找最优组合
python engine/scripts/brute_force_ensemble.py --min-models 3 --max-models 6

# Step 3: 查看结果，选择多个组合写入配置
cat output/brute_force/leaderboard.csv
# 编辑 config/ensemble_config.json 添加多个 combo

# Step 4: 运行所有组合融合回测
python engine/scripts/ensemble_fusion.py --from-config-all

# Step 5: 查看对比结果，确认 default combo
cat output/ensemble/combo_comparison_*.csv

# Step 6: 基于 default combo 生成订单
python engine/scripts/order_gen.py
```

## 与其他脚本的关系

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `weekly_train_predict.py` | 训练模型 | configs | `latest_train_records.json` |
| `brute_force_ensemble.py` | 穷举组合 | train records | leaderboard |
| **`ensemble_fusion.py`** | **融合回测** | **选定模型/多组合** | **融合预测 + 绩效 + 对比** |
| `signal_ranking.py` | 信号排名 | 融合预测 | Top N 排名 CSV |
| `order_gen.py` | 生成订单 | 融合预测 + 持仓 | 买卖建议 + 多模型判断 |
