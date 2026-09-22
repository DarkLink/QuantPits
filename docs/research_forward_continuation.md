# Shadow Forward 连续周期（D2）

D2 从已结算的首期开始，按同一冻结定义延续两臂状态：`C4 intent 1 → D1 settlement 1 → intent 2 → settlement 2 → intent 3 …`。真实运维需明确工作区、provider、根目录、epoch、前序请求和写入授权。开发测试的 synthetic 链不代表真实周期已经发生。

## 周度与前序

采用 `WEEKLY_LAST_SESSION_V1`：anchor 是市场时区 ISO 周最后交易日，下期为之后第一个有交易日的周末交易日。future calendar 必须覆盖目标周，并至少包含其后一个周的交易日；全周休市可跳过。实际日历在已观察范围须与 future 相容。首期 anchor 也必须满足周末规则。

`cycle_index=2` 只接 D1，之后只接同 epoch 的 `index-1` D2 settlement。reader 必须实际验证完整两臂，逐角色接合 portfolio ID、after-state 内容与摘要；两臂现金/持仓分化和 PARTIAL NAV 合法。缺失、残缺、未结算前序不可继续。`GAP_DETECTED`、`SCHEDULE_UNAVAILABLE`、deadline 过期或 VERSION_BREAK 均停止新增 intent，不补历史、不生成假零订单。已封存的旧 intent 仍可按原 assumption 结算。

## 目录和手工顺序

预先明确准备两个专用 root 及各自的 epoch 父目录，权限 `0700`：

```text
<intent-root>/<epoch>/2
<settlement-root>/<epoch>/2
<intent-root>/<epoch>/3
<settlement-root>/<epoch>/3
```

writer 只 create 当前整数槽；不递归创建父目录，不允许前导零、attempt/hash 换槽。同槽冲突、缺 completion、写后不确定均保留现场，只用原 expected request inspect/adopt。V2 completion 的 store_bindings 绑定冻结链起点和 intent/settlement 角色，后续接入复验链身份；实际两个目标 root 仍须互不重叠。

1. 用 D1 或 D2 inspect 核对已结算前序及原首期身份。
2. 完成当前 Production 周期 seal 与 signal capsule，再准备本期 intent。沿用原 definition/bootstrap/model selectors，指定本期 cycle/signal 和 exact predecessor。
3. 审阅 READY 的 request digest、index、日期、订单计数与 deadline。在 next-open 前，用同参数和 expected digest 发布一次 pair，保存此次原始 COMMITTED safe JSON 到授权位置。
4. next-open 数据到达后，以本期 intent 和本期原始日志 prepare settlement，审阅 request 后 publish settlement。
5. 离线 inspect settlement；下一期将其作为唯一 predecessor，index 加一。

Shadow 失败独立报告，不修改或自动阻塞生产交易。没有自动调度或 Make target。

## API 与 CLI

领域入口 `quantpits.research.forward_continuation` 提供：

- `prepare_next_forward_intent_publication` / `publish_next_forward_intent_pair`：沿用 C3 十四个当前来源参数；额外 keyword 为 `intent_store_root, settlement_store_root, epoch_id, cycle_index, first_intent_store_root, expected_first_intent_request_digest, predecessor_settlement_store_root, predecessor_cycle_index, expected_predecessor_request_digest, decision_deadline_utc, next_open_utc, market_timezone`。
- `prepare_next_forward_settlement` / `publish_next_forward_settlement`：`intent_store_root, epoch_id, cycle_index, expected_intent_request_digest, publication_success_record_path, qlib_provider_root`，keyword `settlement_store_root`。
- 两种 publish 另需 `expected_request_digest`。prepare 不授予可反序列化的发布资格；publish fresh 重读和规划。
- `inspect_next_forward_intent_pair(root, epoch, index, expected_request_digest=...)` 返回 `d1_inputs`；`inspect_next_forward_settlement(...)` 返回 `after_states` 和 defensive-copy `continuation_metadata`。离线检查不需要在线模型/provider/Production。

```bash
python -m quantpits.scripts.continue_forward --help
python -m quantpits.scripts.continue_forward inspect-intent \
  --intent-store-root "$INTENT_ROOT" --epoch-id "$EPOCH" \
  --cycle-index 2 --expected-request-digest "$INTENT_REQUEST"
python -m quantpits.scripts.continue_forward inspect-settlement \
  --settlement-store-root "$SETTLEMENT_ROOT" --epoch-id "$EPOCH" \
  --cycle-index 2 --expected-request-digest "$SETTLEMENT_REQUEST"
```

其余动作为 `prepare-intent / publish-intent / prepare-settlement / publish-settlement`；参数名把 API 下划线替换为连字符。help 输出完整参数名单，不读取工作区。退出码为成功 0、前置/等待/请求不符 2、VERSION_BREAK 3、冲突 4、残缺/不确定 5。

## 证据边界

连续期使用独立 request/bundle domain 与 schema v2，C4/D1 v1 保持可读。每期仅保存本期输入、直接前序引用和冻结起点身份；不递归嵌入历史或复制模型。

首次本期 COMMITTED 为 `cycle_intent_published=true, prospective_claim=true, epoch_started=false`，仍执行本机 UTC/monotonic、600 秒和全部 pre-open gate。D1/D2 settlement 严格校验本期原始日志与 completion，首期日志、ADOPTED 或 VERIFIED 不能替代。本期 reader/adoption 不新授予上述发布事实。

`state_chain_ready` 只表示本 settlement 的实际 B0 重算与两臂 after-state 已验证、可用于后续接入；`predecessor_join_observed` 来自 fresh preparation；不递归验证全链，不输出 `whole_chain_verified=true`。单个数据成员 4 MiB、总计 32 MiB、记录 1 MiB；边界错误保留普通失败/中断/写后不确定语义。

相关文档：[首期准备](research_forward_preparation.md)、[首期结算](research_forward_settlement.md)。D3 共同窗口报告和真实观察周期另行推进。

D2 使用同一 C3 有限维护兼容判断，拒绝时保留全部组件原因；通过时把 `RESEARCH_MAINTENANCE_ADMISSION_V1` 来源材料保存在本期 request。旧首期 V1 与新兼容 request 可接续，协议材料变化本身不触发经济断段；每期仍须实际检查源码/来源、冻结定义、前序各自 after-state、周度日历和时间 gate。新材料不改变原始成功记录字段或 D1/D3 的成功记录要求。


## 记录复制与格式兼容

新写入的 `completion.schema_version=2` 使用 `FORWARD_RECORD_BINDING_V2`：`target_binding_digest` 绑定记录种类、epoch、周期/index、request 和 manifest 摘要。连续记录的 `store_bindings` 使用 `FORWARD_CHAIN_STORE_V2`，区分 intent/settlement 角色，并绑定首期 request/manifest/operation、冻结 selectors、definition request 和 schedule。它们不包含路径、device、inode、mtime 或主机身份。request/bundle 自身的版本及 safe stdout 的 schema 不变，不能仅凭 request 的版本 2 判断是否可迁移。

可将四个运行 store 与必要的原 COMMITTED stdout 原样复制到新目录，恢复目录 0700、文件 0600 权限，再显式指定新根。reader/inspect/adopt 不改写记录、不刷新首次成功时间，也不产生新的 prospective claim。D1 消费复制后的 intent 和原 stdout；D3 可读取复制链；D2 从已结算 after-state 接续，无需重建 bootstrap 或 epoch。Production、provider、定义和模型仍按现有参数提供，这不包含整个工作区或 MLflow URI 的迁移。

V1 completion 在原位置继续按旧物理绑定读取，后续新记录写 V2；新 V2 settlement 可保留未经改写的 V1 来源 completion，并以自身已校验的内容绑定支持复制；V1 intent 本身仍受原目录限制。复制 V1 后会返回 `LEGACY_PHYSICAL_BINDING_MISMATCH`，本功能不迁移或重写旧记录。未知、缺失或类型混乱的版本被拒绝。操作期间仍检查规范路径、目录身份、文件指纹、来源变动和 create-only 冲突。两个独立副本不共享全局锁，恢复后应选定一个活动副本。代码回退须保留 V2 reader，不得降级改写已发布记录；这不是经济规则变更或新的生产审批门槛。
