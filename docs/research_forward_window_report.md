# 纸面账户账本与收益报告

`quantpits.research.forward_window_report.build_forward_window_report` 只读消费显式账户的首期和后续 intent／settlement。逐一验证来源与前序，并展示实际保存的记账记录、晚生成、晚结算及显式 HOLD；不要求旧 stdout 或严格事前实验资格。不会重新规划、结算或写 Production。

## 显式请求与 CLI

使用 UTF-8 JSON；以下字段全部必需，不接受额外字段、重复 key、非有限数字。`schema_version` 必须为整数 1；`cycle_index` 必须为严格整数 1..N，日期严格递增。四个源 root 使用现有 reader 的物理路径及私有权限合同（目录 0700、bundle 成员 0600）；不会自动扫描 latest。即使本窗口只有首期，四个 root 仍须显式提供；只观察实际请求涉及的槽。

```json
{
  "schema_version": 1,
  "first_intent_store_root": "/private/research/first_intents",
  "first_settlement_store_root": "/private/research/first_settlements",
  "continuing_intent_store_root": "/private/research/continuing_intents",
  "continuing_settlement_store_root": "/private/research/continuing_settlements",
  "epoch_id": "example.epoch",
  "requested_cycles": [
    {
      "cycle_index": 1,
      "current_cycle_id": "2026-09-04",
      "expected_intent_request_digest": null,
      "expected_settlement_request_digest": null
    },
    {
      "cycle_index": 2,
      "current_cycle_id": "2026-09-11",
      "expected_intent_request_digest": null,
      "expected_settlement_request_digest": null
    }
  ]
}
```

用已知原记录中的小写 SHA256 替换 digest。null 只表示未绑定预期身份：槽存在时输出 `UNBOUND_PRESENT`，不会从 manifest 自动选择身份、纳入收益；槽不存在时记录观察到的缺失。日期未有 verified intent 支持时 `date_verified=false`。

```bash
# 无需 source 工作区；仅 stdout 安全摘要
python -m quantpits.scripts.report_forward_window --request-file /private/window-request.json

# 明确授权的 private 派生输出目录
python -m quantpits.scripts.report_forward_window \
  --request-file /private/window-request.json \
  --output-dir /private/reports/window-001
```

stdout 不含路径、epoch、证券、金额和原异常。输出目录的父目录须已存在；目标须不存在或为空且权限为 0700。拒绝符号链接路径、源目录内部或源目录祖先、非空目标。三个文件均以 0600 创建，不覆盖已有文件，也不自动清理失败输出。失败摘要的 `written_files` 只列经规范路径验证的文件；`attempted_files` 表示尝试过的文件名，失败文件可能部分存在，`output_state=PARTIAL_OR_UNCERTAIN`。控制中断传播，已有派生文件可能保留。

| 状态 | 退出码 | 含义 |
| --- | --- | --- |
| COMPLETE | 0 | 请求全部结算、实际链相连、两臂完整 NAV 和有效共同基数 |
| PARTIAL | 2 | 合法等待、未绑定、缺失成员、PARTIAL NAV 或共同基数无效 |
| BLOCKED | 4 | 非法请求、冲突、断链或源读取不确定 |
| OUTPUT_FAILED | 5 | 输出未全部完成；查看文件尝试/验证事实 |

普通一行失败后仍观察后续请求。内部缺失使下一节点的实际 predecessor 无法接合时显示 `CHAIN_BREAK`。等待不声称交易所数据尚未到达；不使用当前日期推测交易日。源在本轮读取过程中变化时窗口标记 `UNCERTAIN`。安全摘要中的完整链和原记录能力均独立于报告 `prospective_claim=false/epoch_started=false/promotion_capability=false`。

## 估值与比较

规则固定为 `FIRST_NEXT_OPEN_PRETRADE_BASE_TO_POSTTRADE_SAMPLES_V1`。共同起点 B 是首期 next-open **交易前**两臂完整、相同且正的 `nav_before`；显式 normalized start=1。每期采样点为 next-open **交易后** `nav_after`，时刻来自已冻结 intent。

- 归一化净值为 post/B，窗口收益为末期 post/B−1。
- 首期收益为 post/B−1，以后为本期 post/直接前期 post−1；不使用本期 post/pre 代替跨期收益。
- challenger−champion 收益差为两个收益率相减，曲线差为两个 normalized 值相减。
- 回撤以包含基准 1 的历史最高采样值为分母；这只是离散 next-open 采样回撤，不能当成日频/盘中最大回撤。
- 成本为 actual fee+slippage，成本率以 B 为分母。NAV 已含成本，不二次扣费。换手为 `(gross_buy+gross_sell)/本期 nav_before`，不除以 2；累计换手为完整请求各期换手之和。

手算 B=100、post1=99、pre2=110、post2=109：归一化为 0.99、1.09，窗口收益 0.09，第二期收益 109/99−1，首期回撤 −0.01。价格跨期变化已进入后续 post，不能被 post/pre 公式遗漏。

缺期、断链或任一 PARTIAL NAV 使完整窗口收益/最大回撤/收益差为 null，并保留所有原行及原因。只展示实际连接到原起点的局部点值，不重选基数。PARTIAL 估值之后可保留独立完整点值和直接相邻完整收益，但历史回撤缺样时不再给局部回撤。实际成本可在完整会计覆盖下累计；换手必须所有分母完整且正。无效基数使所有 normalized/收益为 null，现金和成交诊断仍保留。

post 为零或负时保留真实金额及 `NONPOSITIVE_NAV`，正 B 下仍可计算 normalized 和回撤（可能低于 −1）；下一期收益的非正分母产生 null。不截断损失、不改写为零收益。

重叠指标消费实际 ranking Top-K、BUY/SELL 标的、after-state 持仓及数量；K 来自冻结定义。集合输出交/并数量和 Jaccard；两个空集合定义为 1 且 `empty_both=true`。合法相同两臂与零订单均保留。保存价格观察 provenance、execution assumption、冻结定义和实际 continuation 引用；corporate action 为 UNMODELED，无外部现金流。本结果不等同券商实际成交归因，也不覆盖 bootstrap 到首期 next-open 之前的持有收益。技术记录只能证明本机时间观察，不把 synthetic fixture 当作真实前向运行。

## 时间与 HOLD 说明

每行 `accounting_time` 分别保存实际创建、写前数据读验、实际结算和本次来源读验时间；旧记录没有的事实为 null。`generation_relation` 的 AFTER_OPEN_SIMULATION 由原创建时间与原开盘时刻决定，晚结算不会将旧事前选股改称事后模拟。`settlement_delay_seconds` 是实际结算距原开盘的秒数，不是解锁门槛。缺原 stdout 时 `original_success=LOCAL_BUNDLE_VERIFIED_WITHOUT_STDOUT`，不补造旧提交成功。

请求索引按实际保存记录排列，HOLD 跳过的交易周不另建请求行。JSON、CSV、Markdown 展示区间及调用者持有假设。`period_start_trade_date / period_end_trade_date` 表示相邻实际估值之间的收益区间；HOLD 的跨期收益与累计实际成交成本可计算，但区间内无净值样本，`max_drawdown=null` 并显示 HOLD_INTERVAL_NOT_SAMPLED／DRAWDOWN_HISTORY_INCOMPLETE。报告 COMPLETE 只表示所请求账本与端点估值完整，不表示 HOLD 内日频估值完整；公司行为未建模披露保持。

## 文件与可复核身份

- `report.json`：完整 requested rows、每节点实际 request/manifest/operation 三元引用与原成员 raw digests、跨 bundle joins、`source_context`、两臂指标、原因和能力。
- `cycles.csv`：每实际请求固定 CHAMPION、CHALLENGER 两行。原金额／比例列保持，末尾增加 `created_at_utc,generation_relation,settled_at_utc,settlement_delay_seconds,hold_start_date,hold_end_date,hold_assumption,hold_reason,period_start_trade_date,period_end_trade_date`。null 为空，原因用 `|` 连接。
- `report.md`：范围、共同基数、缺口、收益/成本和逐臂逐期值；null 明示。证券重叠和完整价格 provenance 在 JSON 中。

计算使用局部高精度 Decimal；金额为确定性十进制字符串，比例输出 24 位小数 ROUND_HALF_EVEN 后去尾零，舍入不进入后续计算。JSON/CSV/Markdown 使用相同数值字符串。`request_digest` 不含本地 roots；`semantic_digest` 排除 `private_bindings` 和 `report_generated_at`，保留实际来源身份及实现文件 SHA256。报告是派生诊断文件，修改其 JSON 不会赋予任何发布或复用权限。旧 v1 首期不要求原来不存在的 D2 字段，后续 v2 的物理 binding 和 schedule 由既有 reader 验证。

D3 在 `report.json` 每期 `source_context.input_provenance.source_change` 中保留生产与执行来源比较；Markdown 逐期展示差异及不可比较原因。旧记录显示 NOT_RECORDED。源码说明不将已验证会计结果降为 PARTIAL。
