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

## 当前进度与历史来源

当前最后处理日／估值日、pending cycle／执行日和下一期信号独立于所选历史期间展示。paper
的已结算来源取相应结算绑定的冻结计划，pending 来源取当前冻结计划；不使用现在配置里的默认
`signal_source` 替换旧来源。报告同时保留 bootstrap 来源和已有 lineage 的简短摘要，不重新读取
当前信号文件、不反序列化模型或递归追查 MLflow。

实盘只读已有 `order_gen` manifest，兼容 workspace 名称或绝对路径。成功 manifest 加上有效的
已保存建议 CSV 可显示“建议已生成”。它仅证明当期建议来源，不能证明已经下单、成交或整个
流水线成功；实际 broker pending 状态仍标未知。来源未记录时明确显示未知，不妨碍展示余额。

实盘与 paper 的差异还包含成交摩擦、权益和记账口径。当前未录入分红／补税的 paper 及额外滑点
为零的假设应写为成员备注；对比差异不能直接解释为纯模型贡献，备注不授予或取消交易资格。

## 状态与只读边界

| CLI 退出码 | 含义 |
| --- | --- |
| `0` | `complete`：所有成员的所选期间必需财务／估值事实均可读，报告已保存；可选来源仍可能未知 |
| `2` | `partial`：至少一个成员缺必需事实、损坏或读取不稳定；全部成员及可用结果仍已保存 |
| `1` | 组配置／日期／输出无效或保存失败；stderr 输出结构化错误 |

`--help` 不要求 workspace。普通成员错误不会吞掉后续成员，`KeyboardInterrupt` 等进程控制中断
传播。组配置无效在读取账户前失败。

读取复用稳定文件读取与 mutation observer，整个组读取结束时再核对已读文件和 manifest 清单；
某成员读取期间变化时丢弃其混合快照，标为 `unstable`，其他成员继续展示。不创建账户锁、缓存、
MLflow 数据库或成员输出。报告按文件保存，不宣称五个派生文件跨文件原子提交；保存失败可重跑。
