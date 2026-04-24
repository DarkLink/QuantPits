# 深度分析系统 (MAS) — 使用指南

## 概述

深度分析系统是一个多代理系统 (MAS)，用于执行自动化的、多窗口的盘后分析。七个专业代理分析交易系统的不同方面，综合器 (Synthesizer) 交叉引用发现的结果，生成具有优先级的、可操作的建议。

## 快速开始

```bash
# 激活工作区
source workspaces/Example_Workspace/run_env.sh

# 基础规则分析
python -m quantpits.scripts.run_deep_analysis

# 带频率变更截止日期
python -m quantpits.scripts.run_deep_analysis --freq-change-date YYYY-MM-DD

# 使用 LLM 生成执行摘要
OPENAI_API_KEY=sk-xxx python -m quantpits.scripts.run_deep_analysis \
    --llm openai --freq-change-date YYYY-MM-DD

# 附带操作员笔记
python -m quantpits.scripts.run_deep_analysis \
    --notes "上周重新训练了 catboost。由于贸易紧张局势，市场波动较大。"

# 仅运行特定代理
python -m quantpits.scripts.run_deep_analysis --agents model_health,prediction_audit

# 自定义时间窗口
python -m quantpits.scripts.run_deep_analysis --windows 1y,3m,1m
```

## CLI 参数

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `--windows` | `full,weekly_era,1y,6m,3m,1m` | 逗号分隔的时间窗口 |
| `--freq-change-date` | 来自配置或 `None` | 日频→周频切换的截止日期 |
| `--output` | `output/deep_analysis_report.md` | 报告输出路径 |
| `--llm` | `none` | LLM 后端：`none` 或 `openai` |
| `--llm-model` | `gpt-4` | OpenAI 模型名称 |
| `--api-key` | `$OPENAI_API_KEY` | LLM API 密钥 |
| `--base-url` | `None` | OpenAI 兼容的 API 基础 URL |
| `--agents` | `all` | 逗号分隔的代理名称 |
| `--notes` | `""` | 自由文本形式的外部上下文 |
| `--notes-file` | `None` | 包含外部笔记的文件路径 |
| `--shareable` | `false` | 脱敏敏感数据 |
| `--no-snapshot` | `false` | 跳过配置快照 |

## 代理

### 1. 市场状态 (`market_regime`)
从 CSI300 基准数据中检测当前市场趋势、波动率状态和回撤状态。

- **输入**: `daily_amount_log_full.csv` (CSI300 列)
- **输出**: 趋势标签 (牛市/熊市/震荡)、波动率百分位、回撤深度

### 2. 模型健康 (`model_health`)
评估单个模型的 IC/ICIR 趋势，检测重新训练事件，并对超参数进行快照。

- **输入**: `model_performance_*.json`, `workflow_config_*.yaml`, MLflow 标签
- **输出**: IC/ICIR 评分表、重训时间线、超参数摘要、陈旧性警告

### 3. 组合演进 (`ensemble_eval`)
跟踪组合的表现，并检测三个层面的构成变化。

- **输入**: `combo_comparison_*.csv`, `leaderboard_*.csv`, `ensemble_fusion_config_*.json`
- **输出**: 组合表现趋势、三层变化事件日志、相关性漂移

**三层变化检测：**
1. **构成变化 (Composition change)**: `ensemble_fusion_config` 文件中的模型列表差异
2. **活跃组合切换 (Active combo switch)`**: `ensemble_config.json` 中默认组合的更改
3. **内容变异 (Content mutation)`**: 组合名称相同，但配置内的模型不同

### 4. 执行质量 (`execution_quality`)
使用现有的 `ExecutionAnalyzer` 分析交易执行摩擦。

- **输入**: `trade_log_full.csv`
- **输出**: 摩擦趋势、替代偏差、费率效率、ADV 容量

> **注意**: 执行时机分析暂缓 (TODO)，等待更细粒度的日内时间戳数据。

### 5. 组合风险 (`portfolio_risk`)
使用现有的 `PortfolioAnalyzer` 进行多窗口风险分析，包含 OLS 统计显著性。

- **输入**: `daily_amount_log_full.csv`, `trade_log_full.csv`, `holding_log_full.csv`
- **输出**: 多窗口 CAGR/Sharpe/DD 表、OLS alpha/beta t-stat/p-value、因子漂移

### 6. 预测审计 (`prediction_audit`)
对比模型预测与实际市场结果。

- **输入**: `buy_suggestion_*.csv`, `sell_suggestion_*.csv`, `model_opinions_*.json`, Qlib 前向收益率
- **输出**: 买入/卖出胜率 (hit rates)、共识 vs 分歧分析、持仓回顾

### 7. 交易模式 (`trade_pattern`)
分析交易行为模式和信号纪律。

- **输入**: `trade_classification.csv`, `trade_log_full.csv`, `holding_log_full.csv`
- **输出**: 信号/替代/手动比例、集中度趋势、纪律评分

## 数据发现

系统会同时扫描 **活动工作区** 和 **归档** 目录：
- `output/` + `output/{ensemble,predictions,ranking}/`
- `archive/output/` + `archive/output/{ensemble,predictions,ranking}/`
- `data/` + `data/order_history/`

这确保了无论归档状态如何，分析都能使用所有可用的历史数据。

## 频率变更截止日期

当设置了 `--freq-change-date` 时 (例如从日频切换到周频的日期 `YYYY-MM-DD`)：
- 系统会自动从该日期起生成一个特殊的 `weekly_era` 窗口
- 较短的窗口 (1y, 6m 等) 正常工作
- `full` 窗口仍然覆盖所有数据，但会标记截止日期前的数据

配置可以持久化在 `config/deep_analysis_config.json` 中：
```json
{
    "freq_change_date": "YYYY-MM-DD"
}
```

## 外部笔记

通过 `--notes` 或 `--notes-file` 注入操作员上下文。这些上下文将：
- 作为 `AnalysisContext` 的一部分传递给所有代理
- 包含在 LLM 综合提示词中
- 作为附录添加到报告中

示例：
```bash
--notes "上周重新训练了所有 Alpha360 模型。"
--notes "由于关税公告，市场波动较大。减小了头寸规模。"
--notes-file operator_notes.txt
```

## 配置分类账 (Config Ledger)

每次运行都会自动将当前配置快照到 `data/config_history/`：
- `config_snapshot_{date}.json` 包含：
  - 所有 `workflow_config_*.yaml` 超参数
  - `ensemble_config.json` (跟踪活跃组合及其构成)
  - `strategy_config.yaml`

这使得未来可以通过差异分析来评估超参数调整的影响。

## 报告结构

```
# 深度分析报告 — {date}
## 执行摘要          ← LLM 生成或基于模板
## 1. 市场环境
## 2. 模型健康仪表盘
    ### 2.1 IC/ICIR 评分表
    ### 2.2 重训历史
    ### 2.3 超参数配置
## 3. 组合演进
    ### 3.1 组合表现
    ### 3.2 变化事件日志
## 4. 执行质量
## 5. 组合风险与归因
    ### 5.1 多窗口对比
    ### 5.2 OLS 显著性
## 6. 预测准确性审计
## 7. 交易行为
## 8. 整体变化影响评估
## 9. 优先建议 (P0/P1/P2)
## 附录：外部笔记
```

## 跨代理综合规则

综合器检测跨代理的复合模式：

| 规则 | 触发条件 | 评估结果 |
|------|---------|------------|
| 状态驱动的 IC 衰减 | 模型 IC 下降 + 高波动市场 | "IC 退化可能是由市场状态驱动的" |
| 流动性漂移 | 负 Alpha + 负流动性暴露 | "小市值漂移，且缺乏选股优势" |
| 无法交易的信念 | 高替代偏差 + 低胜率 | "首选标的经常无法交易" |
| 融合价值 | 胜率 > 55% | "融合价值已确认" |
| 无 Alpha | 所有窗口的 Alpha p>0.1 | "无法拒绝零 Alpha 的零假设 (H₀)" |
