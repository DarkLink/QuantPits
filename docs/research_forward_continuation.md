# 纸面账户接续与明确跳期持有

每期从两臂各自已经结算的现金／持仓继续：首期 intent → 首期 settlement → 后续 intent／settlement。无需重建 bootstrap、保存终端 stdout 或守候周一 deadline。默认允许晚生成，标记实际创建时间及事后模拟；晚结算仍使用原定成交日。工作区、provider、账户、来源和写入范围由 owner 明确选择。

## 周度与前序

采用 `WEEKLY_LAST_SESSION_V1`：anchor 是市场时区 ISO 周最后交易日，下期为之后第一个有交易日的周末交易日。future calendar 必须覆盖目标周，并至少包含其后一个周的交易日；全周休市可跳过。实际日历在已观察范围须与 future 相容。首期 anchor 也必须满足周末规则。

`cycle_index=2` 接首期 settlement，之后接同 epoch 的 index−1 settlement；索引表示实际保存的记账记录数量。逐角色核对 portfolio ID、after-state 内容／digest，两臂现金／持仓分化和 PARTIAL NAV 合法。没有显式 HOLD 时跳过非空交易周仍返回 GAP_DETECTED；全周休市按真实日历跳过。缺失、损坏或未结算前序不能继续。普通源码差异仅作为观察保存。

## 显式 HOLD：一段时间不生成订单，持续持有

当所选 anchor 跳过了至少一个有交易日的周，prepare-intent / publish-intent 必须一起提供：

```bash
--hold-start-date "$PREDECESSOR_TRADE_DATE" \
--hold-end-date "$SELECTED_ANCHOR" \
--hold-reason "该区间未生成订单，按持续持有假设补记"
```

区间是 `(前序成交日, 所选 anchor]`：start 必须等于前序已结算 trade_date，end 必须等于当前 anchor。下一次成交仍在 anchor 后首个交易日；日历要求目标 anchor 是该交易周最后一日，并覆盖其后交易周。没有真正跳过交易周时拒绝多余 HOLD。该字段表示调用者的模拟选择 `NO_ORDERS_CONTINUOUS_HOLD_USER_ASSUMPTION_V1`，不是系统观察到历史中没有订单。

HOLD 保留前序两臂的现金、数量和成本，下一所选期才执行新订单；不会为跳过周创建空流水或收费。prepare 显示前序日期、区间、所选 anchor 和 trade_date。扫描所选账户的 continuing intent store：后于所选前序的未结算 intent 返回 `HOLD_PENDING_INTENT`，残缺／损坏或身份不明记录返回 `HOLD_RECORD_UNCERTAIN`；已存在更后结算也拒绝从旧前序回退。先处理已有记录，再选择其 after-state。

缺价沿原 partial/no-fill/NAV-null 语义保留持仓；不使用零价格或删除持仓。分红、拆并股仍未建模并持续披露。报告保留实际记录索引和跨越区间收益，HOLD 内不虚构估值样本。

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
3. 审阅 READY 的 request digest、前序成交日、所选 anchor／trade_date、订单计数和可选 HOLD。方便时用同参数及预期 digest 发布，stdout 可不保存。
4. 原定 next-open 已到且行情可用后，直接从本期完整本地 intent prepare settlement，再用审核后的 digest publish。没有结算 deadline；stdout 是可选核对材料。
5. 离线 inspect settlement；下一期将其作为唯一 predecessor，index 加一。

Shadow 失败独立报告，不修改或自动阻塞生产交易。没有自动调度或 Make target。

## API 与 CLI

领域入口 `quantpits.research.forward_continuation` 提供：

- `prepare_next_forward_intent_publication` / `publish_next_forward_intent_pair`：沿用十四个当前来源参数与显式首期／前序引用；`decision_deadline_utc=None`、`publication_mode="PAPER"` 可省略。显式 HOLD 增加 `hold_start_date`、`hold_end_date`、`hold_reason` 三项，必须一起填写。
- `prepare_next_forward_settlement` / `publish_next_forward_settlement`：旧参数位置保持兼容；`publication_success_record_path=None` 可省略，provider 建议按 keyword `qlib_provider_root=...` 传入。
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

默认纸面发布 `cycle_intent_published=true`，但 `prospective_claim=false, epoch_started=false`；持久时间说明用于展示事实，不授予旧事前实验资格。首期／续期结算均支持完整本地记录直接入账。原 stdout 显式提供时核对；首期日志、ADOPTED 或 VERIFIED 不能替代本期原始提交输出。旧严格记录及可选 STRICT_PREOPEN 发布仍按原时间合同验证。

`state_chain_ready` 只表示本 settlement 的实际 B0 重算与两臂 after-state 已验证、可用于后续接入；`predecessor_join_observed` 来自 fresh preparation；不递归验证全链，不输出 `whole_chain_verified=true`。单个数据成员 4 MiB、总计 32 MiB、记录 1 MiB；边界错误保留普通失败/中断/写后不确定语义。

相关文档：[首期准备](research_forward_preparation.md)、[首期保存](research_forward_publication.md)、[首期结算](research_forward_settlement.md)、[账本报告](research_forward_window_report.md)。

C3/C4/D2 默认将任意源码差异或源码等价比较不可用作为观察信息；只要其余模型、配置、信号与账户校验通过，就可以继续纸面记账。历史生产封存代码不必等于当前研究代码，当前工作文件仍与实际加载来源核对，并记录工作文件内容指纹，不能仅用 HEAD 代表执行内容。真实信号损坏、排名重建矛盾、模型/配置变化和错误前序仍拒绝。旧公开 decision-surface API 保留原比较语义；新日常路径不授予严格同段或维护许可。

新 request 的 `input_provenance.source_change` 使用独立 `schema_version=1` / `SOURCE_DIFFERENCE_OBSERVATION_ONLY_V1`：`production_comparison` 保存两期生产源码比较，`production_commit`/`production_digest` 保存本期预测生产来源，`execution_commit`/`execution_tree`/`execution_implementation_digest`/`execution_code_digest` 保存当前研究执行来源；`production_execution_comparison` 为 EQUAL、DIFFERENT 或 INCOMPARABLE，不可观察时保存原因。它参与 input/request digest，不决定账户断段。旧 `maintenance_admission` 记录继续按旧合同读取。completion V2 的可迁移绑定与该字段版本独立；回退程序须保留可读取新字段的 reader，不改写历史记录。新纸面时间／可选 stdout 语义见上文；旧严格记录、防重复和中断合同保持兼容。


## 记录复制与格式兼容

新写入的 `completion.schema_version=2` 使用 `FORWARD_RECORD_BINDING_V2`：`target_binding_digest` 绑定记录种类、epoch、周期/index、request 和 manifest 摘要。连续记录的 `store_bindings` 使用 `FORWARD_CHAIN_STORE_V2`，区分 intent/settlement 角色，并绑定首期 request/manifest/operation、冻结 selectors、definition request 和 schedule。它们不包含路径、device、inode、mtime 或主机身份。request/bundle 自身的版本及 safe stdout 的 schema 不变，不能仅凭 request 的版本 2 判断是否可迁移。

可将四个运行 store 原样复制到新目录，原 COMMITTED stdout 若存在可同时保留，恢复目录 0700、文件 0600 权限，再显式指定新根。reader/inspect/adopt 不改写记录、不刷新首次成功时间，也不产生新的 prospective claim。结算消费复制后的完整 intent；stdout 可选；D3 可读取复制链；D2 从已结算 after-state 接续，无需重建 bootstrap 或 epoch。Production、provider、定义和模型仍按现有参数提供，这不包含整个工作区或 MLflow URI 的迁移。

V1 completion 在原位置继续按旧物理绑定读取，后续新记录写 V2；新 V2 settlement 可保留未经改写的 V1 来源 completion，并以自身已校验的内容绑定支持复制；V1 intent 本身仍受原目录限制。复制 V1 后会返回 `LEGACY_PHYSICAL_BINDING_MISMATCH`，本功能不迁移或重写旧记录。未知、缺失或类型混乱的版本被拒绝。操作期间仍检查规范路径、目录身份、文件指纹、来源变动和 create-only 冲突。两个独立副本不共享全局锁，恢复后应选定一个活动副本。代码回退须保留能读取 V2、PAPER_ACCOUNTING_TIME_V1、可选 stdout 和 HOLD 的 reader，不得降级改写已发布记录；这不是经济规则变更或新的生产审批门槛。
