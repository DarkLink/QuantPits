# 综合分析模块 (Comprehensive Analysis Module)

本模块用于为量化系统提供专业级的多维审查视角，包括且不限于：单模型的有效性衰减、融合组合的边际贡献差异、真实的执行滑点成本及基于实盘资金的传统风险评估。

核心入口脚本为：`scripts/run_analysis.py`

---

## 一、模块功能概览

总控脚本基于 4 个相互独立的分析组件 (`scripts/analysis/`) 生成完整的 Markdown 审查报告：

1. **单模型表现 (`single_model_analyzer.py`)**：通过 Rank IC 与 T+1 ~ T+5 半衰期变化追踪原始信号的强度和退化速度。同时考察 ICIR (信息比率)、十分位收益排序 (Decile Spread) 和多头超额发掘能力 (Long-only IC)。
2. **组合及相关性 (`ensemble_analyzer.py`)**：评估多模型跨截面的信号斯皮尔曼 (Spearman) 相关性，并通过 Leave-One-Out (留一法) 测算各子模型当前的**边际夏普贡献**，同时输出等权融合打分后的整体业务指标。
3. **真实摩擦测算 (`execution_analyzer.py`)**：切分真实的交易记录，量化 **Delay Cost**（交易日收盘到下一交易日开盘的天然跳空缺口）及 **Execution Slippage**（开盘价到真实成交价的打滑成本）。同时通过 MFE/MAE 计算日内的极端路径。
4. **组合风险评估 (`portfolio_analyzer.py`)**：基于 `daily_amount_log_full.csv` 中的 **收盘价值** 和 `trade_log_full.csv` 中的 **CASHFLOW/成交金额** 严谨地推断出真实复权收益，并输出标准度量指标（CAGR、Vol、Sharpe 1.35% Rf、Max Drawdown 等）、资金运作效率（换手率、盈亏比）与风格敞口 (Barra Style Exposures)。

---

## 二、基础使用方法

最常见的情景是：在你变更了融合配置或新增了一周的数据与持仓之后，快速验证系统健康度和组合多样性。

### 1. 例行全面检验

```bash
cd QuantPits

# 使用当前所有指定的模型作为融合主体，输出 Markdown 分析报告
python quantpits/scripts/run_analysis.py \
  --models gru_Alpha158 transformer_Alpha360 TabNet_Alpha158 sfm_Alpha360 \
  --output output/analysis_report.md
```

### 2. 参数说明

| 参数 | 可选 / 必选 | 说明 |
|------|-----------|------|
| `--models` | 必选 | 需要纳入分析和相关性评估的模型名称清单，例如 `gru_Alpha158` 或 `mlp`。 |
| `--start-date` | 可选 | 截取数据的开始日期 (YYYY-MM-DD)。默认自动从预测文件和日志中推断。 |
| `--end-date`| 可选 | 截取数据的结束日期 (YYYY-MM-DD)。默认自动从预测文件和日志中推断。 |
| `--market`| 可选 | 设定的市场池或基准，默认为 `csi300` |
| `--output` | 可选 | 将生成的最终 Markdown 审查报告完整路径（默认仅打印在终端） |

### 3. 交互式可视化看板 (Interactive Visual Dashboard)

除了静态的 Markdown 报告，系统还提供基于 Streamlit + Plotly 构建的多维度交互式可视化监控看板。该看板不仅直接调用现有的核心组件计算最新指标，还极大地方便了对策略回撤、微观执行滑点以及动态因子暴露的深度复盘。

```bash
cd QuantPits

# 启动交互式看板（启动后在浏览器打开 http://localhost:8501 ）
streamlit run dashboard.py
```

在前端界面中，你可以自由选择测算的时间范围 (Start Date, End Date) 及对标基准 (Market Benchmark)，点击生成即可得到四大可视化模块：
- **全局收益与风险 (Macro Performance & Risk)**：对数收益率曲线、动态水下回撤面积图、滚动 20 日 Sharpe & Alpha 追踪。
- **微观执行与摩擦 (Micro-Execution & Friction)**：MFE/MAE 联合散点图 (利用悬停查看单笔逐笔交易)、执行摩擦与滑点分布直方图、错失的高优信替代偏差比拼。
- **因子暴露与归因 (Factor Exposure & Attribution)**：基于滚动 20 日回归的 Barra 风格因子漂移图、极其直观的收益归因三层堆叠面积图 (Beta + Style + Idiosyncratic)。
- **持仓与盈亏特征 (Holdings & Trade Analytics)**：单只持仓超限监控、基于 FIFO 严格撮合的持仓周期与盈亏结果散点图、基于 "月份 vs 星期" 的日历胜率热力图。

### 4. 滚动参数可视化看板 (Rolling Analysis Dashboard)

针对固定期间报告无法识别策略“风格漂移”和“性能退化”的问题，系统专门提供了滚动评估看板。它通过滑动的 60 日窗口 (步进 1 日)，动态捕捉系统表现的环境变迁。

1. 首先运行数据生成脚本。你可以自由地指定一个或多个所需的滑动窗口长度（例如 20日 和 60日）：
```bash
cd QuantPits
python quantpits/scripts/run_rolling_analysis.py --windows 20 60
```

2. 然后启动专门的滚动可视化看板：
```bash
streamlit run rolling_dashboard.py --server.port 8503
```

- 该看板包含四大滚动监控模块：
  1. **风格漂移监控 (Rolling Factor Exposure)**：动态揭示策略在 Size、Momentum、Volatility 等因子上的拥挤或偏离度。
  2. **阿尔法生命周期 (Rolling Return Attribution)**：堆叠柱状图，将每日收益明确切分为 Beta、Style Alpha 和纯粹的不跟风 Idiosyncratic Alpha。
### 5. 自动化滚动异常体检报告 (Automated Rolling Health Report)

为了进一步减少人工看盘的时间，系统现增加了一键式“滚动体检”自动化诊断工具。该脚本会智能提取并对比不同滑动窗口(`output/rolling_metrics_{20/60}.csv`) 之间的多维指标，侦测异象并自动生成 Markdown 摘要与处理建议。

```bash
cd QuantPits

# 注意：运行此检验前必须先生成好底层的窗口参数数据
python quantpits/scripts/run_rolling_analysis.py --windows 20 60
python quantpits/scripts/run_rolling_health_report.py
```
这将在 `output/rolling_health_report.md` 生成最新的状态报文，其引擎内部会执行以下三级异常监测：
1. **Z-Score 异常检测 (Friction limits)**：实时判定系统在最近一个分析周期的数据与持仓之后，快速验证系统健康度和组合多样性。
2. **均线死叉断层 (Alpha Decay)**：利用最近 5 天平滑的短期 Idio Alph 下穿 60 日 Idio Alpha（死跌入水下），定性判断底层模型的纯净选股能力是否正在面临彻底失效与断层衰减。
3. **极小/极大历史分位突破 (Factor Drift)**：判断最新的诸如 Size (市值)、Momentum 因子 Beta 的值，是否突破了这台策略一年历史环境以来的 5% 深水重灾极点。如果触及极高极低边界，会提出强制风格因子中性化的抢救建议。

---

## 三、报告解读与行动建议

当你在检查生成的 `analysis_report.md` 时，重点关注以下红线或绿灯信号：

### 1. 单模型及融合打分表现 (Model Quality)
- **Rank IC Mean & ICIR**：
  - **绿灯**：IC > 0.035 且 ICIR > 0.15 即被认为是有效的 Alpha 来源。
  - **红线行动**：若某个主力的 `T+1 IC` 持续低迷至 0.01 或变负，应当停止其参与 Ensemble Fusion，并调用 `incremental_train.py` 重训。
- **Decile Spread (十分位多空收益差)**：衡量模型打分前 10% 标的与后 10% 标的的收益差（基于次日）。该值应显著为正。
- **Long-Only IC (Top 22)**：严格只计算模型预测头部分数（因实际只有对应仓位）与收益的 Spearman 相关性。由于头部样本少，稍微为正或在 0 附近即可接受；如果深负说明模型在“高优股票”中存在严重的偏好误导。
- **IC Decay (衰减曲线)**：健康的信号应当在 T+1 时最强，随后平滑衰减。如果直线下坠，说明信号极度偏向高频或失效过快。

### 2. 相关性与边际贡献 (Ensemble Correlation)
- **目标**：模型间的 Spearman 最好保持在 `0.2` 到 `0.5` 左右。太高说明同质化，太低可能有一个跑偏。
- **关键操作点**：查看 `Marginal Contribution to Sharpe`。如果显示 "Drop `A` -> impact on Sharpe: `+0.2`"，这表示**剔除模型 A 反而让整个组合的模拟夏普更高**。这通常是因为它的信号噪音遮蔽了别人。如果你还在用 Equal Weight 组合，需要考虑人工踢出 A 或者调低其权重。

### 3. 执行滑点与摩擦 (Execution Friction & Path Dependency)
- **摩擦组成 (Total Friction)**：买卖总摩擦在 0.2% 以内是非常优秀的执行水平。
  - **绿灯**：如果发现 **Execution Slippage 经常大比率偏负**（比如超过 -1%），这说明我们的挂单策略非常激进且总是被迫买到了当天的非常高点。如果不含复权除息逻辑引发的数据失真，实盘操作中必须减慢买入节奏。
- **摩擦面值 (Absolute Slippage Amount)**：将负滑点比例等比例转换为损失的人民币（面额）数值。如果发现买入组或卖出组绝对总金额特别高，需配合订单日明细盘查。
- **市场容量监测 (ADV Participation Rate)**：我们的策略日均交易额在该只股票当天的整个大盘**真实总计交易面额**中的吃水比例（买入卖出分开计算，以 100 股一手和因子解复权的方式完全还原底层 Qlib RMB 规模）。若均值在 0.5% 内相对安全，一旦达到数个百分点（如最大值飙升破 15%-25%），意味着资金体量开始形成严重的市场冲击，需严格控制买入量天花板。
- **显性费用与分红补偿**：新加入了真实的 Explicit Costs（包含佣金、印花税等）计算，并在流水中提取红利入账与红利税进行抵扣，提供真实的净影响。
- **订单执行差异与替代偏差 (Order Discrepancy & Substitution Bias)**：
  - **Substitution Bias Loss**：由于涨停等原因导致未能买入 Suggestion 中的 Top 模型标的，而被迫买入后排替补标的所损失的 5 日 alpha 收益。如果此项常驻负值且较大，需要注意高优标的的可买入性。

### 4. 传统评估、风险与资金效率 (Return, Risk & Efficiency)

这部分数据基于**实盘流水表 (`daily_amount_log_full.csv`和`trade_log_full.csv`)** 剔除出入金 (Cashflow) 后推断的复权净值计算：

- **基础收益与回撤 (CAGR & Max Drawdown)**：与交易终端 (例如实盘系统) 看板对齐。如果剧烈不符，请首先检查 CSV 中近期是否存在出入金格式变更未被兼容。*计算 Sharpe 时使用的无风险收益率为 1.35% (年化)。*新增单独显示的 ** Absolute Return** （累积回报），以抑制系统短时间内截取数据时 CAGR 引发的“年化幻觉”。
- **相对基准风险 (Relative Risk to Benchmark)**：
  - **Tracking Error (跟踪误差)**：计算策略跑赢/跑输沪深 300 基准的主动收益的标准差。数值越小，代表策略偏离大环境的程度越低。
  - **Information Ratio (信息比率)**：即 `主动年化超额收益 / 年化 Tracking Error`。衡量承担每单位非系统风险所带来的超额回报。
  - **Calmar Ratio (卡尔玛比率)**：即 `CAGR / Max_Drawdown` 的绝对值。衡量忍受单位回撤时的总增长。
- **资金效率 (Efficiency)**：
  - **Turnover_Rate_Annual (年化换手率)**：通常日度优选策略的年化换手率在 1000% - 2500% 左右（单边 2%-5%/日）。太高意味着每天全仓换发，交易成本将抹平 alpha。
  - **Win/Loss Ratio (盈亏比)**：平均盈利日的收益除以平均亏损日的绝对收益。
- **水下时间考量 (Time Under Water)**：
  - **Max / Avg Time under Water Days**：资产在创下某一个历史新高后，随后的连跌或震荡导致它“直到重新突破前高”所耗费的交易日数量。
  - **Days Below Initial Capital**：即资产跌破初始本金的真实亏损天数，与水下时间配合服用，可判别“横盘期”与“实亏期”。
- **Barra 风格暴露 (Proxy Style Exposures)**：
  - 基于实际资产日收益对简单的 Barra 代理特征 (Size, Momentum, Volatility) 进行横截面回归。若某一项系数绝对值异常偏高（例如 Size 远负于 -0.5 等），说明策略严重暴露于某些市值/热度因子陷阱中。
- **Performance Attribution (收益归因分析)**：
  - 在大类因子暴露回归分析的基础上，进一步将模型的总 CAGR 等权重拆分。
  - **Beta Return**：源于纯粹跟踪大盘 Beta (比如市场涨了，你也随之被动涨出的收益部分)。
  - **Style Alpha**：源于模型在市值、动能、波动率三大类因子上的线性暴露。这里运用**年化算术收益率**(Arithmetic Return) 与 Beta 相乘计算得出，彻底排除了几何复利爆炸偏差。
- **Idiosyncratic Alpha (特异性阿尔法)**：即 `总 CAGR - Beta Return - Style Alpha`。这是本策略极具价值的核心指标，代表了纯粹不跟风的，完全由模型自身的择时、独门因子或交易结构带来的残差超额收益（即真 Alpha）。数值越大（如持续为正的大几个点），系统愈发高级且不可替代。

### 5. 交易分类与手动干预影响 (Trade Classification & Manual Impact)

这部分数据通过匹配原始交易记录（`trade_log`）和推荐买卖列表（`buy/sell_suggestion`）进行比对得出。通过该系统，将混合在一起的实盘表现干净地分离为“量化系统业绩”和“手动干预业绩”。

- **分类矩阵 (Classification Distribution)**：
  - **SIGNAL (S)**：模型主控信号。该笔交易的标的处于推荐列表的头部名额内（名额 = 当日实际产生的交易笔数）。这是严谨量化回测的基石。
  - **SUBSTITUTE (A)**：备用替代信号。该笔交易的标的在推荐列表中，但排名靠后（未进入当日前排）。通常发生在头部标的“一字涨停无法买入”或“停牌”等交易受阻引发的顺延动作，此分类可用于追溯因流动性妥协而付出的代价。
  - **MANUAL (M)**：完全手动操作。标的根本不在当期推荐列表中。**在随后的所有量化执行摩擦计算（滑点）中，该类交易将被全数剔除**。
  
- **量化净表现 (Quantitative-Only Performance)**：
  - 剔除了手动交易干扰后的、纯机器算法贡献的年化收益率。这一指标能最直观地解答：“如果我不进行那些手动干预操作，系统本身到底能赚多少？”
  
- **手动盘点 (Manual Trade Details)**：
  - 精确列出在此分析区间的每一笔手动交易的买卖日期、金额方向以及相关的胜率拉据影响。长期而言，“手动 PnL”与“量化 PnL”的对比是交易团队能否严格遵守纪律的最直观镜子。
