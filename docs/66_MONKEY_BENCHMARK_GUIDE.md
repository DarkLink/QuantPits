# 随机猴子统计基准指南 (Random Monkey Benchmark)

随机猴子基准系统用于通过 Monte Carlo 模拟与假设检验，量化回答核心问题：**“量化模型是否具有统计显著的 Alpha 预测能力？策略表现来自模型预测信号还是来自持仓交易规则本身？”**

---

## 快速开始

```bash
cd /path/to/QuantPits
source workspaces/<workspace_name>/run_env.sh

# 1. 默认运行：1000 次模拟，自动读取 strategy_config.yaml 配置并输出统计图表
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --plot

# 2. 快速模式：5000 次模拟，仅计算 IC / ICIR / Rank IC（跳过回测，快速完成）
python quantpits/scripts/monkey_benchmark.py --n-trials 5000 --ic-only

# 3. 指定对比模型（与已训练模型进行显著性检验）
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --models <model_1>,<model_2>

# 4. 自定义策略参数覆盖（例如测试纯持有 Buy and Hold 模式）
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --topk 20 --n-drop 0

# 5. 全选成分股等权基准验证（TopK 设为大于成分股数量的值，如 --topk 9999）
python quantpits/scripts/monkey_benchmark.py --n-trials 10 --topk 9999
```

---

## 核心设计与两层架构

系统设计为两层互补架构，兼顾**单模型标准 Pipeline 对比**与**大规模统计分布检验**：

```
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 1: RandomModel (Qlib 兼容伪模型类)                                │
│  • 继承 Qlib Model 基类，fit() 空操作，predict() 生成动态均匀随机打分        │
│  • 注册至 model_registry.yaml，支持 static_train / ensemble_fusion 全流程 │
│  • 适用于：单次基准对比、拉入模型融合验证抗噪能力                       │
└────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 2: monkey_benchmark.py (向量化 Monte Carlo 统计引擎)             │
│  • 数据集与行情特征仅加载 1 次                                          │
│  • 循环 N 次并行模拟，生成经验统计分布                                  │
│  • 严格对齐 TopkDropout (TopK + DropN) 调仓与动态成分股过滤             │
│  • 自动计算 IC/ICIR、年化收益、Sharpe、最大回撤分布与假设检验 p-value      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 核心机制与防失真保证

### 1. 严格动态成分股池 (Dynamic Universe Masking)
* **背景**：指数成分股存在定期调整，历史累计出现的标的池较大，但在任一特定交易日 $t$ 内有效成分股数量有限（$N_t$ 只）。
* **隔离机制**：系统通过 Qlib 动态成分股解析器在每一个交易日 $t$ 提取当日实际在池的有效股票。非当日成分股的打分被强制置为 $-\infty$：
  * 随机选股**绝不可能**选入池外标的；
  * 出池股票在下一个调仓日立即被强制移出；
  * 杜绝将历史池外标的当成 0 收益从而稀释组合收益的失真问题。

### 2. 精确对齐 TopkDropout (TopK + DropN) 状态机
实盘量化策略并非每期将持仓推倒重选（全量推倒会产生极高的换手摩擦）。系统完整实现了 TopkDropout 调仓状态机：

| 参数设定 | 策略模式 | 调仓状态机行为 |
|---|---|---|
| **`n_drop = 0`** | **零淘汰纯持有 (Buy and Hold)** | 首日按打分买入 TopK 支股票后**不再主动换仓**（换手率=0，除非成分股退市/出池触发被动替换）。 |
| **`0 < n_drop < top_k`** (如 `n_drop = 3`) | **TopkDropout (实盘策略模式)** | 每期仅淘汰当前持仓中打分最靠后的 $N_{\text{drop}}$ 支股票并买入新候选补齐，换手率受控在 $N_{\text{drop}}/K$ 附近。 |
| **`n_drop >= top_k`** 或未设置 | **全量调仓 (Full Rebalance)** | 每期根据最新打分重新全选 TopK 支股票。 |

### 3. 基准收益与时间对齐 (Forward 1-Day Alignment)
* 自动从工作区 `model_config.json` 读取配置的基准（如 `benchmark` 字段）。
* 在交易日 $t$ 选股成交，组合持仓收益与基准收益均严格对应 $t$ 日收盘至 $t+1$ 日收盘（前向 1 日），保证时间序列严格对齐。

---

## 报告解读与假设检验

运行脚本后，系统会输出三部分关键分析：

### 1. 猴子基准统计分布 (Monte Carlo Distribution)
展示 $N$ 次模拟中各项指标的经验均值、标准差、90% 置信区间与极值范围：

```text
📊 [猴子基准统计分布 (Monte Carlo Distribution)]
                          指标          均值 ± 标准差     中位数  90% 置信区间 (5%~95%)     极端范围 (Min~Max)
                  IC (Pearson) -0.0000 ± +0.0035 +0.0000 [-0.0055, +0.0055] [-0.0110, +0.0110]
                ICIR (Pearson) -0.0000 ± +0.0600 +0.0000 [-0.0950, +0.0950] [-0.1900, +0.1900]
            Rank IC (Spearman) -0.0000 ± +0.0035 +0.0000 [-0.0055, +0.0055] [-0.0110, +0.0110]
          Rank ICIR (Spearman) -0.0000 ± +0.0600 +0.0000 [-0.0950, +0.0950] [-0.1900, +0.1900]
              年化收益率 (CAGR)   +1.50% ± +7.50%  +1.50% [-10.50%, +13.50%] [-15.00%, +20.00%]
            累计总收益率 (Total)   +1.40% ± +7.20%  +1.40% [-10.00%, +13.00%] [-14.50%, +19.50%]
                   Sharpe 比率 +0.1500 ± +0.4500 +0.1500 [-0.5500, +0.7500] [-0.8500, +1.2000]
             最大回撤 (Max DD)  -13.50% ± +3.50% -12.50% [-19.50%, -9.50%]  [-24.00%, -7.00%]
      年化超额收益 (Ann Excess)   -4.00% ± +7.50%  -4.00% [-16.50%, +8.00%]  [-20.50%, +15.50%]
```

### 2. 真实模型 vs 猴子假设检验 (Hypothesis Testing)
计算每个已训练模型在猴子经验分布中的百分位（Percentile）与单侧经验 $p\text{-value}$：

$$p\text{-value} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Metric}_{\text{Monkey}, i} \ge \text{Metric}_{\text{Model}})$$

```text
🎯 [真实模型 vs 猴子基准检验 (Hypothesis Testing)]
模型名称                 IC     ICIR  Rank IC  IC 百分位  p-value 显著性结论
model_candidate_a   +0.0400  +0.5000  +0.0380     99.8%   0.0020  ✅ 极显著 (p<0.01)
model_candidate_b   +0.0350  +0.4500  +0.0330     99.4%   0.0060  ✅ 极显著 (p<0.01)
baseline_linear     +0.0120  +0.1500  +0.0110     95.2%   0.0480  ✅ 显著 (p<0.05)
weak_candidate      +0.0010  +0.0150  +0.0010     60.0%   0.4000  ❌ 未能打败猴子 (p≥0.05)
```

### 3. 可视化直方图输出
指定 `--plot` 参数后，图表将自动保存至工作区的 `output/monkey_benchmark/` 目录：
* `monkey_ic_distribution.png`：Pearson IC 分布图（标注真实模型垂线及 90% 置信区间）；
* `monkey_rank_ic_distribution.png`：Rank IC 分布图；
* `monkey_ann_ret_distribution.png`：年化收益率分布图；
* `monkey_sharpe_distribution.png`：Sharpe 比率分布图；
* `monkey_max_dd_distribution.png`：最大回撤分布图；
* `monkey_trials_distribution.csv`：所有 Monte Carlo 实验的原始指标明细表。

---

## 完整参数列表

| 参数 | 默认值 | 类型 | 说明 |
|---|---|---|---|
| `--n-trials` | `1000` | int | Monte Carlo 随机模拟次数 |
| `--models` | 所有已训练模型 | str | 指定对比的真实模型名称（逗号分隔） |
| `--topk` | 从 `strategy_config.yaml` 读取 | int | 目标持仓数量 |
| `--n-drop` | 从 `strategy_config.yaml` 读取 | int | 调仓时淘汰股票数量（`0`=纯持有，`>=topk`=全量调仓） |
| `--ic-only` | `False` | flag | 快速模式：仅计算 IC/Rank IC，跳过回测 |
| `--plot` | `False` | flag | 是否生成并保存分布直方图 PNG |
| `--workspace` | 当前环境工作区 | str | 指定执行分析的工作区路径 |

---

## 典型分析场景与实验方法

### 场景 A：模型 Alpha 显著性检验（常规上线门禁）
```bash
# 跑 1000 次模拟，检验模型是否显著超越随机基准 (p < 0.05)
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --plot
```

### 场景 B：策略持仓规则贡献度拆解（Strategy vs Alpha Attribution）
通过对比以下三组实验，精准拆解收益来源：
1. **纯持有猴子**：`--topk <K> --n-drop 0`（完全无调仓的随机选股基准）；
2. **TopK Dropout 猴子**：`--topk <K> --n-drop <N>`（带有动量惯性调仓规则的随机基准）；
3. **真实模型策略**：查看真实模型在同一 `TopK=<K>, DropN=<N>` 下的回测表现。
* 若 `(2) > (1)`：说明 TopK Dropout 调仓规则本身带来了正向持仓收益（策略贡献）；
* 若 `(3) > (2)`：说明模型打分提供了超越交易规则的真实 Alpha 预测能力。

### 场景 C：单模型 Pipeline 融合去噪对比（Layer 1）
```bash
# 1. 在模型注册表中配置猴子模型并训练
python quantpits/scripts/static_train.py --models random_monkey_Alpha158

# 2. 将猴子模型与真实模型一起进行融合回测，观察融合抗噪能力
python quantpits/scripts/ensemble_fusion.py --models <real_model>,random_monkey_Alpha158
```
