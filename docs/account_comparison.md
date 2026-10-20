# 关联账户只读对比

`compare_accounts` 将多个独立实盘／paper workspace 的已保存事实放在同一份报告中。它不要求激活
workspace，不初始化 Qlib／MLflow，不运行训练、交割或 paper 步骤，也不注册、迁移或合并账户。

## 组配置与命令

组配置是一个操作索引，可以放在任意路径。真实配置及报告建议保存在忽略的 `brain/account_groups/`
下，不把私人路径、账户 ID 或余额提交到 Git。公开示例仅使用合成路径：

```yaml
name: example_comparison
members:
  - id: live
    label: Actual
    kind: actual
    workspace: ../live_workspace
    notes: Includes actual execution friction and entitlements.
  - id: old_paper
    label: Old paper
    kind: paper
    workspace: ../old_paper_workspace
  - id: new_paper
    label: New paper
    kind: paper
    workspace: ../new_paper_workspace
```

成员 `id` 必须唯一，报告保留声明顺序，包括失败或缺数据的成员。同模型、同起点或同 workspace
不会合并成员。`label` 只用于展示；实际账户身份从实盘交易日志的 `资金账号` 或验证后的 paper
journal 读取，缺失／不唯一时显示未知，不使用 label 冒充账户 ID。

路径相对于**组文件所在目录**解析，支持 `~`。成员可选 `mlflow_uri` 仅保留显式上下文，第一版不
创建 client 或查询后端。可选 `qlib_data_dir` 指定应保护的 provider 根，同样按组文件目录解析；
未指定时采用 `~/.qlib/qlib_data/cn_data`，并保护已保存配置／paper 来源中记录的 provider。
读取不采用继承的 `QLIB_WORKSPACE_DIR`／`MLFLOW_TRACKING_URI` 来切换账户。

```bash
python -m quantpits.scripts.compare_accounts \
  --group /path/to/group.yaml \
  --since 2024-04-05 --date 2024-04-12 \
  --output /path/to/comparison_reports
```

`--since` 和 `--date` 必填，均为精确的 `YYYY-MM-DD` 已保存日期，且起点早于末日。
`--output` 必须显式指定。输出目录及最终文件解析符号链接后都不得进入任何组成员 workspace 或
已知 provider；不借用成员的 `output/`。新账户只需加入组，不改变比较算法。

## 报告与口径

每次生成五份同名报告，文件名前缀为 `comparison_<since>_<date>`：

| 文件 | 内容 |
| --- | --- |
| `.md` | 财务、当前进度、成对持仓对照、来源及缺口 |
| `.json` | 全部请求成员、首尾持仓、来源、可用日期、错误和已读文件 SHA-256 |
| `_summary.csv` | 每个账户的期间 NAV、现金、股数／股票数、入金、损益、收益及费用 |
| `_holdings.csv` | 每股在各账户的数量、市值、NAV 权重和差异；不可用成员留空 |
| `_pairs.csv` | 所有账户对的共有／各自独有股票、数量差异与权重重合 |

增加 `--explain` 可在同一 Markdown／JSON 中展示资产差额和当期成交对照，并额外保存
`_asset_differences.csv` 与 `_executions.csv`，共七份文件；不加参数仍生成原五份。

实盘 NAV／外部入金来自 `data/daily_amount_log_full.csv` 的 `收盘价值`／`CASHFLOW`；现金及持仓
来自所选日的 `holding_log_full.csv`，期间费用来自 `trade_log_full.csv` 的 `费用合计`。
不读取原始券商 XLSX，不拿最新 `prod_config` 持仓替代历史。现金流或费用缺失、空值、非有限值
均保留为未知；缺现金流时仍可展示 NAV，但损益／收益留空。

paper 使用 `PaperStore.read()` 校验完整账本，再从精确日期的最后一份已保存估值读取 NAV 和
当日持仓，包括 bootstrap 估值。交易／事件最后处理日早于估值日是合法情况，不要求两者相等。
`STALE`／`PARTIAL` 估值保留观察值和状态，但不作为可靠收益或股票权重对照的依据。

财务统一使用以下口径：

```text
净外部入金 = since < 入账日期 <= date 的外部资金变化
期间损益 = 期末 NAV - 期初 NAV - 净外部入金
起点本金收益率 = 期间损益 / 期初 NAV
```

paper 的外部资金用已保存估值的已应用 `EXTERNAL` 累计差；分红、利息、税费属于投资损益，
不当作外部入金。期间费用只累计 `since < settlement_date <= date` 的结算，既不使用全历史
累计费用，也不从已含费用的 NAV 再扣一次。收益不是 TWR；期初 NAV 非正时收益留空并说明。

所有账户使用请求的同一首尾日期。缺期初／期末或可靠估值时显示可用日期与缺口，不改起点，
不前向填充或补零。资产各自展示，不将实盘与 paper 相加成真实总资产。

持仓比较的股票权重为当日市值除以各自完整 NAV，重合度为
`sum(min(账户 A 股票权重, 账户 B 股票权重))`。现金占比单列，不计股票数；持仓与 NAV 不完整
时仅对应的账户对不可用。CSV 区分独有股票和同股数量不同，未知成员留空，不声称其持仓为零。
`present_members`／`absent_members`／`unknown_members` 分别记录持有、已知未持有和未知成员，
`quantity_differs` 独立表达已观察数量差异。只有部分成员同量持有时标为 `shared_subset_same_quantity`。

## 差额与成交说明（`--explain`）

每对账户按声明顺序展示“左减右”：所选末日 NAV 差＝现金差＋逐证券期末市值差＋舍入残差。
两边持仓均完整时，缺席证券才按零计算；未知持仓使该账户对不可用，不影响其他对。
Markdown 突出数量或市值差非零的证券，JSON／CSV 保留全部明细及双方实际数据日、数量、市值、
权重。单端沿已有 0.01 元容差，成对残差容差为 0.02 元，超出时标 `unreconciled`；残差始终单列。
这是某日资产差额，不是逐股期间收益贡献。可靠期间数据另列期初 NAV 差、净外部入金差和期间
损益差的桥接，不能把期初资产或入金不同归给模型。

只对 `(since, date]` 内已观察到的成交／结算日对齐证券和买卖方向。实盘复用 post-trade 的明确
交割类别解析，同日同股同方向多笔累计数量、成交金额、费用和笔数，汇总均价为金额合计除以
数量合计；Markdown 均价显示四位小数，JSON／CSV 保留计算值。红利即使带股份数量也不算买入；
红利、补税、利息和持仓调整另列，未知类别留缺口。
paper 展示已验证冻结计划的主备角色、候选次序、建议数量及已记录的 FILLED／SKIPPED 决策、
模拟数量、价格、费用和原因。`NO_SLOT` 是备选未使用，不是券商拒单。待执行计划单列，不能混入
已成交记录。实盘未成交或没有委托记录时，撤单、涨跌停和人工选择原因均保持未知。

实盘建议只来自保存的成功 manifest／建议 CSV；同一执行日存在多个成功版本时全部保留并标歧义，
不按文件名挑选“采用版本”。有原始 SHA-256 摘要时核对；旧 manifest 无摘要时标
`current_readable_no_historical_digest`（当前可读文件），不证明历史文件内容或下单。来源对照仅用
明确 workspace、anchor、recorder 引用；确认融合来源时可说明引用相同及执行数量是否不同，
不依赖文件名、模型名或 CSV／pickle 摘要推断信号等价。

## 当前进度与历史来源

当前最后处理日／估值日、pending cycle／执行日和下一期信号独立于所选历史期间展示。paper
的已结算来源取相应结算绑定的冻结计划，pending 来源取当前冻结计划；不使用现在配置里的默认
`signal_source` 替换旧来源。报告同时保留 bootstrap 来源和已有 lineage 的简短摘要，不重新读取
当前信号文件、不反序列化模型或递归追查 MLflow。

实盘只读已有 `order_gen` manifest，兼容 workspace 名称或绝对路径。成功 manifest 加上有效的
已保存建议 CSV 可显示“建议已生成”。它仅证明当期建议来源，不能证明已经下单、成交或整个
流水线成功；实际 broker pending 状态仍标未知。来源未记录时明确显示未知，不妨碍展示余额。
Markdown 展开 workspace、组合、experiment／recorder／文件及信号／执行日，区分“期间订单建议来源”、
“已结算计划来源”和“当前待执行来源”。损坏或缺失的可选元数据、旧建议文件单列 `source_gaps`，
保留独立有效的财务数据；journal／财务损坏和读取不稳定仍按成员错误处理。

实盘与 paper 的差异还包含成交摩擦、权益和记账口径。当前未录入分红／补税的 paper 及额外滑点
为零的假设应写为成员备注；对比差异不能直接解释为纯模型贡献，备注不授予或取消交易资格。

## 状态与只读边界

| CLI 退出码 | 含义 |
| --- | --- |
| `0` | 所有成员必需财务／估值事实完整且报告已保存；指定 `--explain` 时说明也完整 |
| `2` | 成员必需事实缺失／损坏／不稳定，或请求的说明部分不可用；全部成员及有效结果仍保存 |
| `1` | 组配置／日期／输出无效或保存失败；stderr 输出结构化错误 |

JSON／stdout 的 `status` 保留基础财务状态；`--explain` 另有 `explanations.status`／
`explanation_status`，说明缺口不会把有效余额改成失败。实际选择原因未知、来源关系无法证明、
建议多版本有歧义均可作为完整的事实展示；已知所需文件不可读、交割类别不能解析或差额不能对账
则使说明为 `partial`。默认五文件命令仍允许可选来源未知。

`--help` 不要求 workspace。普通成员错误不会吞掉后续成员，`KeyboardInterrupt` 等进程控制中断
传播。组配置无效在读取账户前失败。

读取复用稳定文件读取与 mutation observer，整个组读取结束时再核对已读文件和 manifest 清单；
某成员读取期间变化时丢弃其混合快照，标为 `unstable`，其他成员继续展示。不创建账户锁、缓存、
MLflow 数据库或成员输出。报告按文件保存，不宣称派生文件跨文件原子提交；保存失败可重跑。
