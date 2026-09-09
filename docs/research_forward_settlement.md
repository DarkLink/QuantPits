# D1：首对前向 intent 结算

D1 将一个明确的 C4 首对 intent，按其原 prior、intent 和 execution assumption，用对应交易日的 next-open 数据执行 B0 会计，保存两臂完整 transition 和 after-state。它不加载模型、不重新规划订单、不更新 Production，也不实现 D2 续跑或 D3 收益报告。

## 输入与只读准备

先取得 C4 的 exact `intent_store_root / epoch_id`、原 request digest，以及首次 **COMMITTED** 返回的 safe JSON 文件。ADOPTED、VERIFIED、UNCERTAIN 记录不能代替首次成功记录。记录须与 C4 request、manifest、operation、target binding、日期及全部本地时间 gate 匹配。该记录只代表单 owner 本机观察，没有外部可信时间戳。

创建专用、物理路径的 settlement store root，权限为 0700；原 safe JSON 权限为 0600。所有路径须为绝对路径，不接受符号链接别名。开发测试使用临时工作区；以下命令形式不构成私有结算授权。

```bash
python -m quantpits.scripts.settle_forward_intent prepare \
  --intent-store-root /absolute/research/intent_pairs \
  --epoch-id EPOCH_ID \
  --expected-intent-request-digest C4_REQUEST_SHA256 \
  --publication-success-record-path /absolute/research/c4_success.json \
  --qlib-provider-root /absolute/qlib/provider \
  --settlement-store-root /absolute/research/settlements
```

`prepare` 不创建文件，返回 `READY` 和 D1 `request_digest`。请求包含原始发布身份、原成功记录 bytes 摘要、两臂会计输入、实际日历和价格 receipt、结果及实现指纹；路径、inode、mtime 和当前采样时刻不进入语义摘要。价格内容修订会改变请求；只有文件时间改变不会改变请求。

日期取自已验证 C4。必须已到原 next-open UTC，实际 `calendars/day.txt` 必须包含该交易日，截止 anchor 的历史前缀须与 C4 一致，anchor 后下一 session 须仍为原 trade date。允许后续日历扩展。尚未开盘、交易日尚未进入实际日历、非空请求集合全无有效价格时返回 `WAITING_FOR_DATA`，不封存整对 no-fill。

价格按两臂 holdings ∪ intents 的 union 观察 open/factor，再按每臂 exact 集合投影。规则 `OBSERVED_NEXT_OPEN_IN_REQUESTED_UNION_V1` 只要求该 union 至少一个 OBSERVED；其余 missing/invalid 保留原因，按 B0 处理。不可读文件、损坏的二进制布局或起始位置头属于前置错误，不伪装成缺价。它不证明整个市场数据完整。两臂均无持仓且无订单时，receipt 为 `NOT_REQUIRED_EMPTY_EXPOSURE`，无需价格观察，但仍检查时间和日历。零订单而有持仓仍需估值价格。

## 发布与恢复

将相同参数的 action 改为 `publish`，增加 `--expected-request-digest D1_REQUEST_SHA256`。发布会重新读取、观察和计算，只有 fresh request 相同才写入。固定槽为 `settlement_store_root / epoch_id`，不会按 request hash 换目录。

新槽采用 create-only mkdir、排他文件创建、0700/0600、fsync、规范路径重读，先验明完整 data/manifest，再创建 completion，最后重读。来源 guards 覆盖 C4 exact bundle、原成功记录、实际 calendar 及 union 的 open/factor。单臂普通会计异常会继续另一臂诊断，但不发布半对；进程控制中断传播。

```bash
python -m quantpits.scripts.settle_forward_intent inspect \
  --settlement-store-root /absolute/research/settlements \
  --epoch-id EPOCH_ID \
  --expected-request-digest D1_REQUEST_SHA256
```

既有槽的 publish 只执行独立 inspect/adopt，不再访问当前 C4、provider 或日志。残件不补写、不删除、不覆盖，不换 epoch 重试；写后失败或中断只检查原槽。普通写后失败返回 `UNCERTAIN`，`did_write` 为 true 或无法确定时的 null，不谎报零写。中断没有成功 JSON。

| 状态 | 退出码 | 含义 |
| --- | --- | --- |
| READY / COMMITTED / ADOPTED / VERIFIED | 0 | 只读准备 / 本次提交 / 只读复用 / 独立检查 |
| WAITING_FOR_DATA / PRECONDITION_BLOCKED / REQUEST_MISMATCH | 2 | 写前等待或拒绝 |
| CONFLICT | 4 | 原槽与预期或内容合同冲突 |
| INCOMPLETE / UNCERTAIN | 5 | 残缺或后置状态无法确认 |

safe JSON 仅输出日期、摘要、操作身份、固定两臂状态和计数，以及 accounting/valuation/chain 能力，不输出路径、epoch、证券、金额或原始异常。`--help` 无需 workspace；没有 `--now` 或强制前向开关。

## 持久内容与离线消费

槽包含 `request.json`、`source_intent.json`、原 bytes 的 `publication_success.json`、实际 `calendar_day.txt`、`next_open_prices.json`、`settlements.json`、`manifest.json`、`completion.json`。数据成员单个不超过 4 MiB、合计不超过 32 MiB，manifest/completion 各不超过 1 MiB。摘要顺序为 logical 数据 → request → 成员 inventory/manifest → completion，无循环 hash。

`source_intent.json` 保存 C4 同次 reader 已验证的 request、manifest、completion 和原 calendar bytes，以及两臂 canonical prior/intents/assumption。它记录来源引用，离线检查不声称重新采用全部外部 C4 来源。C4 `d1_metadata` 是只读副本，不授予发布资格。

Python API 位于 `quantpits.research.forward_settlement`：

- `prepare_first_forward_settlement(...)`
- `publish_first_forward_settlement(..., expected_request_digest=...)`
- `inspect_first_forward_settlement(settlement_store_root, epoch_id, expected_request_digest=...)`

离线 reader 验证 exact inventory、canonical schema、来源关联、float32 words、派生现金价和 quote projection，再用严格 loader 重建输入并真实调用 `ShadowPortfolioTransition.apply`，比对完整保存结果。它无需 live provider、Production、MLflow、模型或原 C4 目录。成功 observation 的 `after_states` 按 CHAMPION、CHALLENGER 顺序返回 typed 状态及 prior/intent/source request/manifest/operation/settlement request predecessor 关联。READY 不授予 after-state 消费资格。

`accounting_pair_complete` 与 `state_chain_ready` 不要求全部成交或完整估值。会计 COMPLETE、估值 PARTIAL 时仍可交付 after-state；NAV 为 null，完整缺价名单保存在 transition。B0 保留 signed cash、whole-order fill/no-fill、费用、滑点和 no-deficit-worsening。`nav_before/nav_after` 是同一 next-open 的会计对账，不是跨周投资收益。内层仍为 `RETROSPECTIVE_TECHNICAL_REPLAY`、`prospective_claim=false`，外层仅表明承接了原前向记录，不再次启动 epoch。保留 `UNMODELED` corporate action 模式及 warning，不能据此声称处理了除权分红。
