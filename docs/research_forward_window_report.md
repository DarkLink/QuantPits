# 前向共同窗口报告（D3）

`quantpits.research.forward_window_report.build_forward_window_report` 只读消费一个显式 epoch 的 C4/D1 首期和 D2 后续 intent/settlement。它逐一读取从 index 1 到指定 N 的完整请求，验证实际 source 与 predecessor，生成可重算的 private 派生报告。它不运行模型、provider 估值、生产账户流水线或重新结算，不产生 epoch-start、prospective 或 promotion 资格。

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

## 文件与可复核身份

- `report.json`：完整 requested rows、每节点实际 request/manifest/operation 三元引用与原成员 raw digests、跨 bundle joins、`source_context`、两臂指标、原因和能力。
- `cycles.csv`：每请求固定 CHAMPION、CHALLENGER 两行。列序为 `cycle_index,current_cycle_id,trade_date,next_open_utc,status,row_reason_codes,role,portfolio_id,valuation_status,nav_before,nav_after,cash_after,total_fee,slippage_cost,gross_buy,gross_sell,filled_count,no_fill_count,normalized_nav,period_return,drawdown,cost,cost_rate,gross_turnover,reason_codes`。null 为空单元格，原因用 `|` 连接。
- `report.md`：范围、共同基数、缺口、收益/成本和逐臂逐期值；null 明示。证券重叠和完整价格 provenance 在 JSON 中。

计算使用局部高精度 Decimal；金额为确定性十进制字符串，比例输出 24 位小数 ROUND_HALF_EVEN 后去尾零，舍入不进入后续计算。JSON/CSV/Markdown 使用相同数值字符串。`request_digest` 不含本地 roots；`semantic_digest` 排除 `private_bindings` 和 `report_generated_at`，保留实际来源身份及实现文件 SHA256。报告是派生诊断文件，修改其 JSON 不会赋予任何发布或复用权限。旧 v1 首期不要求原来不存在的 D2 字段，后续 v2 的物理 binding 和 schedule 由既有 reader 验证。
