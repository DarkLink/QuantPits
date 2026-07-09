# 03 Ensemble Fusion 使用指南

## 概述

`scripts/ensemble_fusion.py` 用于对**用户选定的模型组合**进行融合预测、回测和风险分析。

**支持多组合模式**：在 `config/ensemble_config.json` 中定义多个 combo，标记一个 `default`，一次运行所有组合并对比绩效。

**工作流位置**: 训练 → 组合搜索 → 选组合 → **融合回测（本步）** → 订单生成

## 快速开始

```bash
cd QuantPits

# 1. 等权融合（直接指定模型）
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# 2. 从 ensemble_config.json 读取 default combo
python quantpits/scripts/ensemble_fusion.py --from-config

# 3. 运行指定 combo
python quantpits/scripts/ensemble_fusion.py --combo combo_A

# 4. 运行所有 combo 并生成对比
python quantpits/scripts/ensemble_fusion.py --from-config-all

# 5. 只解释执行计划，不初始化 Qlib、不写文件
python quantpits/scripts/ensemble_fusion.py --from-config-all --explain-plan
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
| `--freq` | `None` | 回测频率: `day` / `week` (默认从 workspace merged config / `model_config.json` 读取) |
| `--training-mode` | `None` | 限定模型训练模式（如 `static` 或 `rolling`）；默认自动解析 |
| `--record-file` | `latest_train_records.json` | 指定训练记录文件 |
| `--output-dir` | `output/ensemble` | 输出目录 |
| `--no-backtest` | false | 跳过回测 |
| `--no-charts` | false | 跳过图表生成 |
| `--start-date` | 无 | 过滤数据的开始日期 YYYY-MM-DD |
| `--end-date` | 无 | 过滤数据的结束日期 YYYY-MM-DD |
| `--only-last-years N` | `0` | 仅使用最后 N 年数据 (专为 OOS 测试设计) |
| `--only-last-months N` | `0` | 仅使用最后 N 个月数据 (专为 OOS 测试设计) |
| `--detailed-analysis` | false | 生成详尽的回测分析报告（类似实盘分析） |
| `--verbose-backtest` | false | 开启 Qlib 回测的详细模式 |
| `--norm-method` | `rank` | 截面归一化方法: `rank` (percentile rank [0,1]，推荐) 或 `zscore` |
| `--explain-plan` | false | 打印 dry-run 执行计划后退出，不初始化 Qlib、不写文件 |
| `--json-plan` | false | 输出机器可读 JSON plan；隐含 dry-run |
| `--run-id` | 自动生成 | 指定本次 plan/manifest 的运行 ID |
| `--no-manifest` | false | 真实执行时不写 `output/manifests/ensemble_fusion/<run_id>.json` |

## Dry-run 与运行清单

正式运行前建议先查看执行计划：

```bash
python quantpits/scripts/ensemble_fusion.py --from-config-all --explain-plan
```

该命令只读取 workspace 配置和训练记录，输出将运行的 combo、输入 fingerprint、计划写入的文件和耗时步骤；不会触发 safeguard、不会初始化 Qlib、不会加载 recorder，也不会写 `output/`、`data/` 或 `config/`。

调度器或 CI 可以使用 JSON 版本：

```bash
python quantpits/scripts/ensemble_fusion.py --from-config-all --json-plan
```

真实执行默认会写入运行清单：

```text
output/manifests/ensemble_fusion/<run_id>.json
```

清单记录 `run_id`、plan fingerprint、输入配置 fingerprint、resolved combos、执行状态和结果摘要，并会通过 `data/operator_log.jsonl` 关联 `run_id`、`manifest_path` 和 `plan_fingerprint`。如果需要保持旧的无 manifest 副作用，可加 `--no-manifest`。

实现上，`ensemble_fusion.py` 现在是薄 CLI adapter；plan/render/manifest/OperatorLog linkage 和执行生命周期集中在 `quantpits/ensemble/service.py`。核心融合、回测、图表和 recorder 写入函数仍保留在脚本中，以保持既有行为兼容。

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

## 从搜索结果配置组合

当你通过组合搜索（详见 [02_BRUTE_FORCE_GUIDE](02_BRUTE_FORCE_GUIDE.md)）找到候选组合后，按以下步骤将它们转化为融合配置。

> [!NOTE]
> 这一步是**手动决策步骤**。搜索脚本只负责推荐，最终选哪些组合由你决定。选好后，本文档的融合脚本接管后续所有自动化工作。

### 步骤 1：查看搜索报告确认候选组合

```bash
# 查看一页纸摘要（IS 排名 + OOS 验证结果）
cat output/ensemble_runs/<run_dir>/summary.md

# 或在浏览器打开交互式 HTML 报告（内容更完整）
# output/ensemble_runs/<run_dir>/oos/oos_multi_analysis.html
```

建议从报告中选取 **2-3 个候选组合**，涵盖不同维度：
- 一个 IS Calmar 最高的组合（高收益）
- 一个 OOS 表现稳定的组合（高稳健）
- 一个模型间相关性低的组合（高多样性）

### 步骤 2：编辑 ensemble_config.json

将选定组合的 `models` 列表复制到 `config/ensemble_config.json`，**必须标记一个 `default`**：

```json
{
  "combos": {
    "combo_A": {
      "models": ["gru", "linear_Alpha158", "TabNet_Alpha158"],
      "method": "equal",
      "default": true,
      "description": "高 Calmar 组合（OOS 验证通过）"
    },
    "combo_B": {
      "models": ["alstm_Alpha158", "linear_Alpha158", "sfm_Alpha360"],
      "method": "icir_weighted",
      "default": false,
      "description": "低相关性多样化组合"
    }
  },
  "min_model_ic": 0.00
}
```

配置格式说明详见下方 [多组合配置](#多组合配置) 章节。

### 步骤 3：运行融合回测验证

```bash
# 运行所有 combo 并生成跨组合对比，确认 default 选择是否合理
python quantpits/scripts/ensemble_fusion.py --from-config-all

# 查看对比结果
cat output/ensemble/combo_comparison_*.csv
```

配置完成后，后续每次生产例行只需 `--from-config-all` 即可，无需重新搜索。

### 什么时候需要重新进行组合搜索？

---

## 运行模式

### 单组合模式

```bash
# 直接指定模型（不使用配置文件）
python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method equal

# 从配置文件读取 default combo
python quantpits/scripts/ensemble_fusion.py --from-config

# 运行指定 combo
python quantpits/scripts/ensemble_fusion.py --combo combo_B
```

### 多组合模式

```bash
# 运行所有 combo + 生成跨组合对比
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

### OOS 验证测试模式 (Out-Of-Sample)

> [!NOTE]
> **系统中存在两种 OOS 验证，用途不同：**
>
> | 阶段 | 工具 | 用途 |
> |------|------|------|
> | **搜索阶段** | `brute_force_fast --exclude-last-years` + `analyze_ensembles.py` | 在**海量候选**中防止 IS 过拟合，筛选出真正有泛化能力的组合 |
> | **发车前验证** | `ensemble_fusion.py --only-last-years`（本节） | 对**已选定的组合**在 OOS 数据上做最终确认，确保发车前心里有底 |

如果你在搜索时（使用 `brute_force_fast.py`）使用了 `--exclude-last-years 1` 等参数排除了今年的数据作为 OOS，在最终选定组合准备发车前，可以使用以下命令单独测试组合在这最后 1 年的 OOS 纯外推表现：

```bash
# ========================================
# 仅在最近 1 年的 OOS 数据上测试此组合的表现
# ========================================
python quantpits/scripts/ensemble_fusion.py --from-config --only-last-years 1
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
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158 \
  --method manual \
  --weights "gru:0.6,linear_Alpha158:0.4"
```

### `dynamic` — 动态权重
使用 60 天滚动窗口评估各模型 TopK 持仓的 Sharpe，动态分配权重。

## 处理流程

```
Stage 0: 初始化 Qlib + 加载配置
Stage 1: 加载选定模型预测 + 截面归一化（默认 `rank`，所有 combo 共享）
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
├── ensemble/
    ├── ensemble_fusion_config_{date}.json     # 融合配置
    ├── correlation_matrix_{date}.csv          # 相关性矩阵
    ├── leaderboard_{date}.csv                 # 绩效排行榜
    ├── ensemble_nav_{date}.png                # 净值曲线
    ├── ensemble_weights_{date}.png            # 动态权重图 (dynamic 模式)
    └── backtest_analysis_report_{date}.md     # [NEW] 详尽回测分析报告 (--detailed-analysis)
└── manifests/
    └── ensemble_fusion/
        └── <run_id>.json                      # 运行清单
```

### 多组合模式（`--from-config-all` 或 `--combo`）

```
output/
├── ensemble/
    ├── ensemble_fusion_config_combo_A_{date}.json
    ├── ensemble_fusion_config_combo_B_{date}.json
    ├── combo_comparison_{date}.csv           # 跨组合对比表
    ├── combo_comparison_{date}.png           # 净值对比图
    └── backtest_analysis_report_{combo}_{date}.md # [NEW] 该组合的详尽分析报告
└── manifests/
    └── ensemble_fusion/
        └── <run_id>.json                     # 运行清单
```


> [!NOTE]
> **关于单模型表现与融合回测的评测差异说明**
>
> 融合与穷举脚本在评估模型表现时，会先做按日截面归一化和数据对齐处理。默认 `--norm-method rank` 使用 percentile rank，并对模型未覆盖股票填充中性分 `0.5`；如果显式使用 `--norm-method zscore`，则保留 Z-Score 与交集对齐语义。因此由于归一化、覆盖范围和 TopK 截断的存在，单模型在此处的回测结果可能与训练期间通过 `run_analysis.py` 查看到的原始预测分值回测结果存在合理且微小的差异：
> 1. **独立归一化隔离**：每个模型的预测分值会首先基于自身预测股票池按天归一化，避免其他模型的数据缺失在融合前污染该模型的分布。
> 2. **覆盖范围对齐**：`rank` 模式使用并集股票池并以 `0.5` 表示中性弃权；`zscore` 模式保留 NaN，并在特定组合打分阶段按当前组合涉及模型取交集（`dropna(how='any')`）。
> 3. **评估排名的对齐**：所有提供参照的基准数据（如单模型的历史排行榜回测指标）均会严格根据当前评价矩阵实际生成的时间窗口进行动态切片对齐，从而为您提供“同时间段”的一致性比对。

## 典型工作流

```bash
# Step 1: 训练所有模型
python quantpits/scripts/static_train.py --full

# Step 2: 组合搜索找最优组合
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1

# Step 3: 查看结果，选择多个组合写入配置
cat output/ensemble_runs/brute_force_fast_*/summary.md
# 编辑 config/ensemble_config.json 添加多个 combo

# Step 4: 运行所有组合融合回测
python quantpits/scripts/ensemble_fusion.py --from-config-all

# Step 5: 查看对比结果，确认 default combo
cat output/ensemble/combo_comparison_*.csv

# Step 6: 基于 default combo 生成订单
python quantpits/scripts/order_gen.py
```

## 与其他脚本的关系

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `static_train.py --full` | 训练模型 | configs | `latest_train_records.json` |
| `brute_force_ensemble.py` | 组合搜索 | train records | leaderboard |
| **`ensemble_fusion.py`** | **融合回测** | **选定模型/多组合** | **融合预测 + 绩效 + 对比** |
| `signal_ranking.py` | 信号排名 | 融合 Recorder | Top N 排名 CSV |
| `order_gen.py` | 生成订单 | 融合 Recorder + 持仓 | 买卖建议 + 多模型判断 |

---

## 归一化方法

### rank (默认)

截面 percentile rank 归一化，输出严格 **[0, 1]**。适合纯多头 TopK 选股：

- 每个模型按当天预测分排名 → 映射到 [0, 1]
- 模型未覆盖的股票填充 **0.5**（视为弃权/中性投票）
- 融合时使用**并集**股票池（不会因为某个模型缺数据而丢失股票）
- 各模型在等权融合时获得完全平等的投票权

### zscore

经典 Z-Score 归一化，保留模型的"确定度"信息：

- 输出无界，极值股票的分数会主导融合信号
- 模型未覆盖的股票保留 NaN → 融合时 `dropna` 取**交集**
- 适合需要区分模型置信度的场景

### OOS 分析时的一致性

`analyze_ensembles.py` 会自动从 `run_metadata.json` 读取搜索时使用的归一化方法，保证 IS/OOS 使用相同的归一化策略。

---

## 已知限制 (Future Work)

### 模型覆盖股票数不一致时的 rank 粒度差异

在 rank 模式下，覆盖 300 只股票的模型 rank 步长为 `1/299`，而覆盖 500 只股票的模型步长为 `1/499`。粗粒度模型的信号会被细粒度模型的"量化噪声"稀释。将来可考虑按覆盖股票数对模型加权。

### NaN → 0.5 的"弃权"假设

rank 模式下，模型未覆盖的股票填充 0.5（中性分）。这假设所有模型对未覆盖股票的"不确定度"相同。当模型覆盖有系统性偏差时（例如某模型只覆盖大市值股票），对小票填充 0.5 可能过度乐观或悲观。

### Fast 路径 NaN 处理不一致

`brute_force_fast.py` 的 `load_predictions` 在合并时提前执行全局 `.dropna()`（交集），而标准路径保留并集、仅对缺失填 0.5。同一 combo 在不同路径下的结果不可直接对比。Fast 路径可能在未来版本被弃用。
