# 深度分析系统 (MAS) — 使用指南

## 概述

深度分析系统是一个多代理系统 (MAS)，用于执行自动化的、多窗口的盘后分析。自最新的重构以来，系统采用了严格的 **7 阶段流水线 (7-Stage Pipeline)** 架构，并支持通过 **Pluggable Agent Registry** 在工作区本地加载自定义插件。
系统内置多个专业代理，分析交易系统的不同方面，综合器 (Synthesizer) 交叉引用发现的结果，生成具有优先级的、可操作的建议。

自 Phase 3 起，系统集成了 OOM-RL (Out-of-Money Reinforcement Learning) 反馈能力：LLM Critic 将分析结果转化为可执行的 ActionItem，Phase 4 的 Feedback Loop 在 Playground 沙箱中自动执行并验证这些建议。

## 核心架构

### 1. 7-Stage Pipeline

深度分析执行遵循严格的 7 阶段流水线。每个阶段通过 `@register_stage` 装饰器自声明依赖关系和产出字段，系统根据声明动态构建执行 DAG。可以通过 `--stage` 参数**单独运行**任意阶段——系统只会复用目标阶段的上游 checkpoint，目标阶段本身会重新执行。

1. **`discover`**: 扫描工作区，发现并加载所有相关的快照、模型预测和历史配置数据。
2. **`agents`**: 实例化并执行注册的分析代理 (Agents)，生成结构化的 `AgentFindings`。
3. **`synthesis`**: 将所有代理的发现传递给综合器，生成跨域的洞察和摘要。
4. **`window_analysis`**: 运行纯规则驱动的 `TrainingWindowAnalyzer`，分析训练时间窗口的合理性。**默认启用**，静态规则 + CPCV/滚动规则始终运行，数据驱动规则在 benchmark 数据可用时自动激活。
5. **`signals`**: 运行 `SignalExtractor`，将代理指标转换为标准化信号 (Signals) 以供 Critic 消费。
6. **`critic`**: 运行 LLM Critic，将提取的信号转化为可操作的优化建议 (ActionItems)。
7. **`report`**: 汇总所有阶段的输出并渲染 Markdown 最终报告。

**Checkpoint 与 Label 隔离**: 每个阶段完成后自动保存 checkpoint 到 `output/deep_analysis/checkpoints/`。通过 `--run-label` 指定的标签会注入 checkpoint 文件名，不同 label 之间的 checkpoint 完全隔离——同日多次运行不同参数的实验互不干扰。Checkpoint metadata 会记录窗口、agent selector、manifest 路径和内容指纹；当本次请求与 checkpoint 不兼容时，系统会跳过旧 checkpoint 并重跑对应上游阶段。`--resume-latest` 可从同 label 的最新 checkpoint 续跑。

**执行计划解释**: 使用 `--explain-plan` 可以只解析 DAG、workspace stage manifest、上游 checkpoint 兼容性和最终执行计划，不运行任何阶段，也不会写入 checkpoint。它适合在单阶段验证、插件调试或 checkpoint 语义不确定时先确认“哪些阶段会跑、哪些 checkpoint 会被复用或跳过”。

### 2. 可插拔代理与阶段注册 (Pluggable Agent & Stage Registry)

系统支持通过工作区本地清单文件加载自定义代理和流水线阶段，全程不污染全局环境。

- **代理插件**: 通过 `config/agent_manifest.json` 声明。系统注入工作区路径到 `sys.path`，导入 agent 类，并仅在本次运行的局部 registry 中使用，完成后清理路径。
- **阶段插件**: 通过 `config/pipeline_manifest.json` 声明。采用相同的隔离加载机制——工作区路径临时注入 `sys.path`，运行结束后恢复 stage registry 快照。自定义阶段通过 `insert_after` 声明在 DAG 中的插入位置。

详细开发流程请参阅 [57 — 代理插件开发指南](57_AGENT_PLUGIN_GUIDE.md)。

## 快速开始

```bash
# 激活工作区
source workspaces/Demo_Workspace/run_env.sh

# 基础规则分析
python -m quantpits.scripts.run_deep_analysis

# 使用 LLM 生成执行摘要
python -m quantpits.scripts.run_deep_analysis --llm

# 运行 Critic 模式 — 生成可执行的 ActionItems (OOM-RL Phase 3)
python -m quantpits.scripts.run_deep_analysis --critic

# Critic 预览模式 — 生成 ActionItems 但不持久化
python -m quantpits.scripts.run_deep_analysis --critic-dry-run

# 带频率变更截止日期
python -m quantpits.scripts.run_deep_analysis --freq-change-date YYYY-MM-DD

# 附带操作员笔记
python -m quantpits.scripts.run_deep_analysis \
    --notes "上周重新训练了 catboost。由于贸易紧张局势，市场波动较大。"

# 仅运行特定代理
python -m quantpits.scripts.run_deep_analysis --agents model_health,prediction_audit

# 带标签运行（同日多次运行不覆盖，不同 label 的 checkpoint 完全隔离）
python -m quantpits.scripts.run_deep_analysis --critic --run-label after-retrain

# 独立运行单个阶段（上游自动从兼容 checkpoint 加载，目标阶段重新执行）
python -m quantpits.scripts.run_deep_analysis --stage signals --run-label exp-1

# 查看单阶段执行计划，不实际执行
python -m quantpits.scripts.run_deep_analysis --stage agents:model_health --windows 1m --explain-plan

# 从最新 checkpoint 续跑
python -m quantpits.scripts.run_deep_analysis --resume-latest --run-label exp-1

# 自定义时间窗口
python -m quantpits.scripts.run_deep_analysis --windows 1y,3m,1m

# 加载工作区本地自定义代理插件
python -m quantpits.scripts.run_deep_analysis --agents custom_mock_agent --agent-manifest config/agent_manifest.json

# 加载工作区本地自定义阶段插件
python -m quantpits.scripts.run_deep_analysis --stage custom_liquidity_check --stage-manifest config/pipeline_manifest.json
```

## CLI 参数

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `--stage` | `all` | 单独运行指定阶段 (`discover`, `agents`, `agents:NAME`, `synthesis`, `window_analysis`, `signals`, `critic`, `report`, `all`)。上游阶段自动从同 label 的兼容 checkpoint 加载，目标阶段重新执行 |
| `--run-label` | `""` | 运行标签（如 "after-retrain"）。Checkpoint 和报告文件名均注入此标签，不同 label 之间完全隔离 |
| `--resume-latest` | (flag) | 自动查找同 label 的最新 checkpoint 并从下一阶段续跑 |
| `--resume-from` | `None` | 从指定 checkpoint 文件恢复 |
| `--manifest` | `None` | 兼容旧用法的 agent manifest 参数；推荐改用 `--agent-manifest` |
| `--agent-manifest` | `None` | Agent 注册清单 (JSON/YAML)，路径相对于当前工作区；显式指定但不存在时会报错 |
| `--stage-manifest` | `None` | Stage 注册清单 (JSON/YAML)，路径相对于当前工作区；用于加载自定义流水线阶段 |
| `--explain-plan` | (flag) | 只打印 DAG、checkpoint 兼容性和将要运行的阶段；不执行阶段、不写 checkpoint |
| `--windows` | `full,weekly_era,1y,6m,3m,1m` | 逗号分隔的时间窗口 |
| `--freq-change-date` | 来自配置或 `None` | 日频→周频切换的截止日期 |
| `--output` | `output/deep_analysis_report.md` | 报告输出路径 |
| `--llm` | (flag) | 启用 LLM 执行摘要（模型/endpoint 读取 llm_config.json） |
| `--llm-model` | (llm_config.json) | 覆盖摘要 LLM 模型 |
| `--api-key` | (env var) | API 密钥覆盖（默认读取 llm_config.json 中 api_key_env 指向的环境变量） |
| `--base-url` | (llm_config.json) | API base URL 覆盖 |
| `--critic` | (flag) | **OOM-RL Phase 3** — 启用 Critic 模式，生成 ActionItems |
| `--critic-dry-run` | (flag) | Critic 预览模式，生成 ActionItems 但不持久化到文件 |
| `--window-analysis` | (flag, 默认启用) | 规则驱动的训练窗口分析，现默认运行，flag 保留向后兼容 |
| `--agents` | `all` | 逗号分隔的代理名称 |
| `--notes` | `""` | 自由文本形式的外部上下文 |
| `--notes-file` | `None` | 包含外部笔记的文件路径 |
| `--shareable` | `false` | 脱敏敏感数据 |
| `--no-snapshot` | `false` | 跳过配置快照 |

> **OOM-RL 工作流**: `--critic` 产出 ActionItems 后，通过 `run_feedback_loop.py` 执行。详见 [54 — 反馈闭环执行指南](54_OOMRL_FEEDBACK_LOOP.md)。

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

### 8. 训练窗口分析器 (`TrainingWindowAnalyzer`)

一个**纯规则驱动**的独立分析器，在 Agent 运行之后、Signal 提取之前自动执行。静态规则 + CPCV/滚动规则始终运行；数据驱动规则在 benchmark 数据可用时自动启用。

它检测训练数据划分配置中的结构性问题，不依赖 LLM：

- **输入**: `model_config.json`（窗口参数）、`training_history.jsonl`（anchor 历史）、Market Regime Agent 的 regime 切换数据
- **输出**: `WindowAnalysisFinding` 列表，包含 severity（critical/warning/info）、metrics、和可执行的建议

**16 类检测规则**，分为三组：

**静态规则 (R1-R6)** — 始终运行：
1. **窗口大小边界**: `train_set_windows` < 4 年 → critical/warning，> 15 年 → info
2. **验证比率**: `valid / train` < 0.15 → early stopping 不可靠
3. **训练终点距离 (train-end gap)**: 在 slide 模式下 gap ≥ 5 年 + regime 切换 ≥ 20 → critical
4. **Anchor 陈旧度**: 最新 anchor > 90 天 → warning，> 60 天 → info
5. **Regime vs 窗口不匹配**: 高波动需 ≥ 10 年；≥ 3 次切换需 ≥ 8 年
6. **频率兼容性**: 日频 + > 365 窗口 → 数据量过大

**数据驱动规则 (R7-R13)** — 需要 `BenchmarkDataLoader` 提供的市场基准数据：
7. **Regime 覆盖**: 训练覆盖 < 40% observed regimes → warning；缺失 Bearish-HighVol → 独立 warning
8. **波动率漂移**: train vs test 波动率比 > 1.5x 或 < 0.5x → warning/info
9. **收益分布漂移**: KS 统计量 > 0.20 → warning；mean shift > 1.0σ → info
10. **回撤覆盖**: 训练窗口缺少全历史中存在的大回撤 (>15%) → warning
11. **边界 Regime 不匹配**: train→valid 或 valid→test 边界发生 regime 切换 → warning
12. **断崖效应 (Cliff Edge)**: regime 将在 4 周内滑出训练窗口 → warning/info
13. **覆盖稳定性**: 滑动窗口间 regime 覆盖稳定性 < 0.90 → info

**CPCV/滚动规则 (R14-R16)** — 通过 `TrainingModeContext` 自动运行：
14. **CPCV 组数不足**: `n_groups` ≤ `n_test + n_val` → critical
15. **CPCV 泄露威胁**: `purge_steps` > 10 或 `embargo_steps` > 20 → warning
16. **滚动陈旧度**: rolling state > 90 天未更新 → warning

分析结果通过 `training_window_mismatch` 信号注入 LLM Critic，同时在所有分层流水线 prompt 中以 `training_window_analysis` 字段提供。

### 9. 训练健康 (`training_health`)

检测训练管道健康、滚动进度和交易执行趋势：

- **输入**: `training_history.jsonl`、`rolling_metrics_20.csv`、`rolling_metrics_60.csv`、`latest_train_records.json`，以及 `TrainingContext` 提供的元数据
- **输出**: 
  - **模式覆盖审计**: 检查每个模型的训练模式覆盖（static/cpcv/rolling/cpcv_rolling），标记缺失预期模式
  - **滚动管道陈旧度**: 检查滚动窗口进度，标记超过 90 天未更新的管道
  - **Alpha 衰减监控**: 对比短/长期特质 Alpha，检测选股能力退化
  - **执行摩擦检测**: 监控滑点 (`Exec_Slippage_Mean`) 和延迟成本 (`Delay_Cost_Mean`) 的 z-score 异常
  - **因子漂移检测**: 检测 Barra Liquidity Exposure 的极端百分位漂移（微盘/大盘）
  - **孤儿模型检测**: 识别已启用但不属于任何活跃组合的陈旧模型

> **训练上下文 (TrainingContext)**: 该代理深度依赖 `training_context.py`。该模块提供训练模式清单（名称→模式解析）、滚动管道延迟计算、和模型键解析功能。

> **滚动指标 CSV 前置条件**: Alpha 衰减、执行摩擦和因子漂移检测依赖 `output/rolling_metrics_20.csv` 和 `rolling_metrics_60.csv`。这些文件由滚动分析管道生成（`run_rolling_analysis.py`）。**如果文件缺失，agent 会显式报告此数据缺口**（`info`/`warning` 级别），而非静默跳过。若滚动训练未配置，此缺口为预期行为。

### 10. 训练模式感知 (Training Mode Awareness)

自 Phase 2b 起，Deep Analysis 系统具备完整的训练模式感知能力。`TrainingModeContext`（位于 `quantpits/scripts/deep_analysis/training_context.py`）是训练模式信息的唯一真相来源，所有 agent 通过 `AnalysisContext.training_context` 字段访问。

#### TrainingModeContext 数据源

| 数据源 | 提供内容 |
|--------|----------|
| `latest_train_records.json` | 模型→模式映射 (`lstm_Alpha158@static`)，anchor_date，experiment_name |
| `training_history.jsonl` | 静态/CV 训练的最近训练日期、模式、收敛状态（Phase 2b） |
| `prediction_history.jsonl` | 静态/CV 训练的最近仅预测事件（Phase 2b） |
| `data/rolling_training_history.jsonl` | 滚动训练事件（slide + CPCV），与静态文件分离（Phase 2c） |
| `data/rolling_prediction_history.jsonl` | 滚动仅预测事件（slide + CPCV）（Phase 2c） |
| `config/rolling_config.yaml` | 滚动调度器配置 |
| `data/rolling_state.json` | 滑动滚动进度 (slide) |
| `data/rolling_state_cpcv.json` | CPCV 滚动进度 |
| `config/model_config.json` | CPCV 参数 (purged_cv) |

> **Phase 2c 文件分离设计**: 滚动训练有独立的代码路径（直接调用 `model.fit()`，不经过 `train_utils` 包装函数），且产生"薄"日志（无 epoch 级数据）。因此滚动事件写入独立文件，与静态训练日志完全隔离。未启用滚动的用户不会产生这些文件，零额外开销。

#### `@suffix` 模型键约定

训练记录使用 `模型名@训练模式` 作为复合键：

| 后缀 | 训练模式 | 示例 |
|------|----------|------|
| `@static` | 标准单次训练 | `lstm_Alpha158@static` |
| `@cpcv` | Purged K-Fold 交叉验证 | `linear_Alpha158@cpcv` |
| `@rolling` | 滑动窗口滚动训练 | `gru_Alpha360@rolling` |
| `@cpcv_rolling` | CPCV 策略下的滚动窗口 | `lstm_Alpha158@cpcv_rolling` |

`TrainingModeContext.models_by_name` 按模型名聚合所有模式：`{"lstm_Alpha158": {"static": "rid1", "rolling": "rid2"}}`。

#### Predict-Only 周期检测

系统自动检测当前周期是否为仅预测周期（>50% 模型的最新操作为预测而非训练）。当 `is_predict_only_cycle=True` 时，agent 行为校准如下：

| Agent | Predict-Only 行为 |
|-------|-------------------|
| **ModelHealth** | 抑制 "stale" 误报（模型仅未在本周期重训，并非真正陈旧）；标注 `cycle_type: predict_only` |
| **TrainingHealth** | 抑制 "缺失预期模式" 警告（模式未在本周期运行，并非从未训练） |
| **PredictionAudit** | 添加周期类型上下文，提示命中率反映模型稳定性而非新训练质量 |
| **EnsembleEval** | 区分模型退役与训练模式迁移（如 `static→rolling` 切换） |

#### 插件开发中使用 TrainingModeContext

自定义 agent 通过 `ctx.training_context` 访问所有训练模式信息：

```python
def analyze(self, ctx: AnalysisContext) -> AgentFindings:
    tc = ctx.training_context
    if tc:
        # 查询模型最新操作
        last_op = tc.get_last_operation("lstm_Alpha158")
        # → {"type": "train", "date": "2026-07-03T...", "mode": "static", ...}
        
        # 检查是否为仅预测周期
        if tc.is_predict_only_cycle:
            ...
        
        # 获取多模式模型
        cross_mode = tc.get_cross_mode_models()  # → ["lstm_Alpha158", ...]
        
        # 按模式筛选模型
        rolling_models = tc.get_models_with_mode("rolling")
```

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
  - `model_config.json` (训练窗口参数：train/valid/test 窗口大小、slice mode、freq 等)

这使得未来可以通过差异分析来评估超参数调整和训练窗口变更的影响。

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

## OOM-RL 闭环反馈

从 Phase 3 开始，Deep Analysis 集成了 OOM-RL 反馈能力。启用 `--critic` 标志后，分析结果会通过以下管道转化为可执行的模型优化动作：

```
Agent Findings → Signal Extractor → LLM Critic → ActionItems → Feedback Loop
```

### 相关文档

| 文档 | 内容 |
|------|------|
| [51 — OOM-RL 概览](51_OOMRL_FEEDBACK_OVERVIEW.md) | 系统架构、数据流、反馈范围控制 |
| [52 — 数据基础设施](52_OOMRL_DATA_INFRASTRUCTURE.md) | OperatorLog、Config Ledger、训练收敛日志、Agent 增强 |
| [53 — LLM Critic 指南](53_OOMRL_CRITIC_GUIDE.md) | Signal 提取、Critic 模式、ActionItem 结构、Skills |
| [54 — 反馈闭环执行](54_OOMRL_FEEDBACK_LOOP.md) | Playground、Adapter、Orchestrator、Promote、回退 |
| [55 — OOM-RL 每周操作指南](55_OOMRL_WEEKLY_OPERATIONS.md) | 日常运维、干预检查、特殊情况处理 |
| [56 — LLM 观测与追踪](56_LLM_OBSERVABILITY_GUIDE.md) | LLM Traces、Reasoning 记录、Langfuse、多模型研讨预留 |
| [57 — 代理插件开发指南](57_AGENT_PLUGIN_GUIDE.md) | Agent Manifest 规范、Workspace-Local Plugin 的开发与集成 |

### 快速流程

```bash
# Step 1: Deep Analysis + Critic → 产出 ActionItems（window_analysis 默认运行）
python -m quantpits.scripts.run_deep_analysis --critic

# Step 2: 预览 Feedback Loop
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --report-only

# Step 3: 在 Playground 中执行
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --execute

# Step 4: 推广到生产
python -m quantpits.scripts.run_feedback_loop \
    --action-items output/deep_analysis/action_items_$(date +%Y-%m-%d).json \
    --promote
```

## 数据文件格式参考

### `training_history.jsonl`

每行一个 JSON 对象，记录每次模型训练的收敛信息。由 `train_single_model()` 和 `train_cpcv_model()` 写入。

| 字段 | 类型 | 描述 |
|------|------|------|
| `model_name` | string | 模型名称（不含 @mode 后缀） |
| `mode` | string | 训练模式：`static`/`rolling`/`cpcv`/`cpcv_rolling`（Phase 2b+，旧条目默认 `static`） |
| `experiment_name` | string | MLflow 实验名称 |
| `record_id` | string | MLflow recorder UUID |
| `anchor_date` | string | 训练锚日期 (YYYY-MM-DD) |
| `trained_at` | string | ISO 8601 训练完成时间戳 |
| `duration_seconds` | float\|null | 训练耗时 |
| `early_stopped` | bool | 是否早停 |
| `actual_epochs` | int\|null | 实际 epoch 数（GBDT 为 num_boost_round） |
| `configured_epochs` | int\|null | 配置的 epoch 数 |
| `best_epoch` | int\|null | 最佳验证分数 epoch |
| `best_score` | float\|null | 最佳验证分数 |
| `converged` | bool\|null | actual_epochs == configured_epochs |
| `score_type` | string | 指标类型：`ic`/`rank_ic`/`loss`/`cpcv_folds` |
| `IC_Mean` | float\|null | 测试集 IC 均值 |
| `ICIR` | float\|null | IC 信息比 |
| `n_folds` | int\|null | CPCV fold 数（仅 `score_type=cpcv_folds`） |
| `fold_ic_mean` | float\|null | CPCV fold IC 均值（仅 CPCV） |

### `prediction_history.jsonl`

每行一个 JSON 对象，记录每次仅预测事件。由 `predict_single_model()` 和 `predict_cpcv_model()` 写入。

| 字段 | 类型 | 描述 |
|------|------|------|
| `model_name` | string | 模型名称 |
| `mode` | string | 源模型的训练模式 |
| `anchor_date` | string | 预测锚日期 |
| `predicted_at` | string | ISO 8601 预测时间戳 |
| `experiment_name` | string | MLflow 实验名称 |
| `record_id` | string | 新 recorder UUID |
| `source_record_id` | string | 源训练 recorder UUID |
| `IC_Mean` | float\|null | 预测期 IC 均值 |
| `ICIR` | float\|null | 预测期 ICIR |
| `prediction_type` | string\|null | `cpcv_ensemble`（仅 CPCV 预测） |

### `rolling_training_history.jsonl` (Phase 2c)

每行一个 JSON 对象，记录滑动窗口或 CPCV 滚动训练事件。由 `strategy_slide.py::train_window()` 和 `strategy_cpcv.py::train_window()` 写入。与 `training_history.jsonl` 分离，避免滚动批量条目的日志膨胀。

| 字段 | 类型 | 描述 |
|------|------|------|
| `model_name` | string | 模型名称（不含 @mode 后缀） |
| `mode` | string | `rolling`（slide 模式）或 `cpcv_rolling`（CPCV 模式） |
| `experiment_name` | string | MLflow 实验名称 |
| `record_id` | string | MLflow recorder UUID |
| `anchor_date` | string | 训练锚日期 (YYYY-MM-DD) |
| `window_idx` | int | 滑动窗口索引 |
| `train_start` | string | 训练段起始日期 |
| `train_end` | string | 训练段结束日期 |
| `valid_start` | string\|null | 验证段起始（仅 slide） |
| `valid_end` | string\|null | 验证段结束（仅 slide） |
| `test_start` | string | 测试段起始日期 |
| `test_end` | string | 测试段结束日期 |
| `trained_at` | string | ISO 8601 训练完成时间戳 |
| `duration_seconds` | float | 训练耗时（秒） |
| `IC_Mean` | float\|null | 测试集 IC 均值 |
| `ICIR` | float\|null | IC 信息比 |
| `score_type` | string | `rolling_slide` 或 `cpcv_rolling_folds` |
| `n_folds` | int\|null | CPCV fold 数（仅 `cpcv_rolling`） |
| `fold_ic_mean` | float\|null | Fold 验证 IC 均值（仅 `cpcv_rolling`） |
| `Ann_Excess` | float\|null | 年化超额收益（仅 CPCV） |
| `Max_DD` | float\|null | 最大回撤（仅 CPCV） |
| `Information_Ratio` | float\|null | 信息比（仅 CPCV） |

> **注意**: 以下字段不可用（`model.fit()` 不返回 epoch 级数据）故不存在：`early_stopped`、`actual_epochs`、`converged`、`epoch_*` 数组。用 `null` 代替是误导的；字段不存在本身就是事实。

### `rolling_prediction_history.jsonl` (Phase 2c)

每行一个 JSON 对象，记录滚动仅预测事件。由 `strategy_slide.py::predict_latest()` 和 `strategy_cpcv.py::predict_latest()` 写入。

| 字段 | 类型 | 描述 |
|------|------|------|
| `model_name` | string | 模型名称 |
| `mode` | string | `rolling` 或 `cpcv_rolling` |
| `anchor_date` | string | 预测锚日期 |
| `predicted_at` | string | ISO 8601 预测时间戳 |
| `experiment_name` | string | MLflow 实验名称 |
| `record_id` | null | 始终 `null`（`predict_latest()` 不创建新 MLflow run） |
| `source_record_id` | string | 源训练窗口的 recorder UUID |
| `window_idx` | int | 用于预测的滑动窗口索引 |
| `prediction_type` | string | `rolling_gap_predict`（slide）或 `cpcv_rolling_ensemble`（CPCV） |

### `latest_train_records.json`

工作区根目录，记录最新训练/预测周期的模型注册信息。

```json
{
    "experiment_name": "<your_experiment_name>",
    "static_experiment_name": "<static_experiment_name>",
    "rolling_experiment_name": "",
    "cpcv_experiment_name": "<cpcv_experiment_name>",
    "anchor_date": "YYYY-MM-DD",
    "models": {
        "lstm_Alpha158@static": "<record-uuid>",
        "lstm_Alpha158@rolling": "<record-uuid>",
        "linear_Alpha158@cpcv": "<record-uuid>"
    }
}
```

`models` 键使用 `模型名@训练模式` 复合键。`TrainingModeContext` 解析 `@` 分隔符以构建 `models_by_name` 映射，支持跨模式查询。
