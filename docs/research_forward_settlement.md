# 纸面账户首期结算

结算从完整本地 intent 读取原现金／持仓、订单与成交假设，按原定 `trade_date` 的 next-open 行情记账。周一晚上或下周再结算都可以；不会把成交日改成今天。模块不加载模型、不重选订单，也不写 Production。

## 输入与只读准备

提供明确的 `intent_store_root / epoch_id` 和原 request digest 即可；`publication_success_record_path` 是可选核对材料。本次 reader 必须验证完整 request、所有数据成员、manifest、completion 和账户／来源关联。缺 stdout 保存 `publication_success.json=null` 与 `source_basis=VERIFIED_LOCAL_INTENT_BUNDLE_V1`，不会合成旧 COMMITTED。显式提供 stdout 时仍核对其原始 COMMITTED 身份、digest、账户和记录；ADOPTED／VERIFIED 不能冒充它。

创建专用物理 settlement root，权限 0700；可选 stdout 文件须 0600。路径为绝对路径且无符号链接别名。以下示例由 owner 选定账户和写入范围后运行。

```bash
python -m quantpits.scripts.settle_forward_intent prepare \
  --intent-store-root /absolute/research/intent_pairs \
  --epoch-id EPOCH_ID \
  --expected-intent-request-digest C4_REQUEST_SHA256 \

  --qlib-provider-root /absolute/qlib/provider \
  --settlement-store-root /absolute/research/settlements
```

`prepare` 不创建文件，返回 READY 和结算 request digest。请求绑定原发布身份、可选 stdout bytes 的摘要、两臂会计输入、实际日历／价格 receipt、结果与实现指纹；缺 stdout 的明确本地依据也进入摘要。完成记录保存本次 `source_verified_at_utc` 和实际 `settled_at_utc`，不补造原发布最后成功时间。路径、inode、mtime 和当前采样时刻不进入语义摘要；价格内容修订会改变请求。

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

槽包含 `request.json`、`source_intent.json`、可选原 stdout bytes（或 JSON null）的 `publication_success.json`、实际 `calendar_day.txt`、`next_open_prices.json`、`settlements.json`、`manifest.json`、`completion.json`。数据成员单个不超过 4 MiB、合计不超过 32 MiB，manifest/completion 各不超过 1 MiB。摘要顺序沿用 logical 数据 → request → inventory/manifest → completion。

`source_intent.json` 保存 C4 同次 reader 已验证的 request、manifest、completion 和原 calendar bytes，以及两臂 canonical prior/intents/assumption。它记录来源引用，离线检查不声称重新采用全部外部 C4 来源。C4 `d1_metadata` 是只读副本，不授予发布资格。

Python API 位于 `quantpits.research.forward_settlement`：

- `prepare_first_forward_settlement(...)`
- `publish_first_forward_settlement(..., expected_request_digest=...)`
- `inspect_first_forward_settlement(settlement_store_root, epoch_id, expected_request_digest=...)`

离线 reader 验证 exact inventory、canonical schema、来源关联、float32 words、派生现金价和 quote projection，再用严格 loader 重建输入并真实调用 `ShadowPortfolioTransition.apply`，比对完整保存结果。它无需 live provider、Production、MLflow、模型或原 C4 目录。成功 observation 的 `after_states` 按 CHAMPION、CHALLENGER 顺序返回 typed 状态及 prior/intent/source request/manifest/operation/settlement request predecessor 关联。READY 不授予 after-state 消费资格。

`accounting_pair_complete` 与 `state_chain_ready` 不要求全部成交或完整估值。会计 COMPLETE、估值 PARTIAL 时仍可交付 after-state；NAV 为 null，完整缺价名单保存在 transition。B0 保留 signed cash、whole-order fill/no-fill、费用、滑点和 no-deficit-worsening。`nav_before/nav_after` 是同一 next-open 的会计对账，不是跨周投资收益。内层仍为 `RETROSPECTIVE_TECHNICAL_REPLAY`、`prospective_claim=false`，外层仅表明承接了原前向记录，不再次启动 epoch。保留 `UNMODELED` corporate action 模式及 warning，不能据此声称处理了除权分红。

## 连续周期接入

续期使用[账户接续入口](research_forward_continuation.md)，共用行情观察、两臂会计重算和事务写入。首期及续期 API 的旧位置参数保持兼容；省略 stdout 时建议用 keyword `qlib_provider_root=...`，不要把 provider 移到旧 stdout 参数位置。`continuation_metadata` 是同次读验的副本，不授予发布资格或全链完成事实。


## 记录复制与格式兼容

新写入的 `completion.schema_version=2` 使用 `FORWARD_RECORD_BINDING_V2`：`target_binding_digest` 绑定记录种类、epoch、周期/index、request 和 manifest 摘要。连续记录的 `store_bindings` 使用 `FORWARD_CHAIN_STORE_V2`，区分 intent/settlement 角色，并绑定首期 request/manifest/operation、冻结 selectors、definition request 和 schedule。它们不包含路径、device、inode、mtime 或主机身份。request/bundle 自身的版本及 safe stdout 的 schema 不变，不能仅凭 request 的版本 2 判断是否可迁移。

可将四个运行 store 原样复制到新目录，原 COMMITTED stdout 若存在可同时保留，恢复目录 0700、文件 0600 权限，再显式指定新根。reader/inspect/adopt 不改写记录、不刷新首次成功时间，也不产生新的 prospective claim。结算消费复制后的完整 intent；stdout 可选；D3 可读取复制链；D2 从已结算 after-state 接续，无需重建 bootstrap 或 epoch。Production、provider、定义和模型仍按现有参数提供，这不包含整个工作区或 MLflow URI 的迁移。

V1 completion 在原位置继续按旧物理绑定读取，后续新记录写 V2；新 V2 settlement 可保留未经改写的 V1 来源 completion，并以自身已校验的内容绑定支持复制；V1 intent 本身仍受原目录限制。复制 V1 后会返回 `LEGACY_PHYSICAL_BINDING_MISMATCH`，本功能不迁移或重写旧记录。未知、缺失或类型混乱的版本被拒绝。操作期间仍检查规范路径、目录身份、文件指纹、来源变动和 create-only 冲突。两个独立副本不共享全局锁，恢复后应选定一个活动副本。代码回退须保留能读取 V2、PAPER_ACCOUNTING_TIME_V1、可选 stdout 和 HOLD 的 reader，不得降级改写已发布记录；这不是经济规则变更或新的生产审批门槛。

结算可消费新的纸面时间记录、旧严格时间／maintenance 记录及 `source_change` V1 观察。源码差异不阻断已验证订单结算；缺 stdout 不放宽内容、账户或来源校验。
