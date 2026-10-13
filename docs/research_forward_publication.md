# 纸面账户首期订单保存

首次订单从已有 matched bootstrap 的两臂账户开始，保存到显式 `intent_store_root / epoch_id`。默认 `publication_mode=PAPER`，方便时即可生成、保存订单，再按原定成交日结算；开盘后生成会标记 `AFTER_OPEN_SIMULATION`。无需行政 deadline 或保存终端 stdout。原始信号 anchor、来源 capsule 与成交假设仍必须明确，不能改用 latest。续期和跳期持有见[账户接续](research_forward_continuation.md)。

领域 API 位于 `quantpits/research/forward_intent_publication.py`：

- `prepare_first_forward_intent_publication`：原 14 个显式来源参数，加 keyword-only `intent_store_root`、`epoch_id`、`next_open_utc`、`market_timezone`、`opening_policy`；`decision_deadline_utc=None` 和 `publication_mode="PAPER"` 可省略。只读返回 READY 和待审阅 request digest。
- `publish_first_forward_intent_pair`：相同参数加 `expected_request_digest`。目标不存在时重新执行 C3；摘要必须完全一致，来源 guards 持有到提交核验结束。
- `inspect_first_forward_intent_pair(intent_store_root, epoch_id, *, expected_request_digest)`：只读检查原槽，不需要 Production/provider/MLflow 或模型推理。成功观察的 `d1_inputs` 固定顺序返回两臂 `role / prior / intents / execution_assumption`，使用既有严格会计类型。

C3 public API 仍只读并在返回前关闭 guards；私有移交接缝只供 C4 使用。调用者不能把序列化的 PREPARED 或 READY 结果作为发布资格。

CLI：`python -B -m quantpits.scripts.publish_forward_intent --help`。action 为 `prepare`、`publish`、`inspect`。prepare/publish 沿用 preparation 参数名，并增加 `--intent-store-root --epoch-id --next-open-utc --market-timezone --opening-policy`；publish 另需 `--expected-request-digest`。`--decision-deadline-utc` 可省略，旧严格实验可显式选择 `--publication-mode STRICT_PREOPEN`。inspect 只接受 root、epoch 和预期 digest。没有历史创建时间回填参数。

root 必须预先存在，是规范绝对物理路径、0700 私有目录。epoch 目录 create-only；文件 O_EXCL/0600、fsync 后从规范路径重读。同一 epoch 即使更换 cycle 也不创建第二槽。原请求已存在时只读 ADOPT，不重新读取 live 来源、不覆盖、不自动补文件或删除残留。

bundle 固定九个数据成员：`request.json`、`definitions.json`、`priors.json`、两臂 `*_ranking.csv`、`plans.json`、`anchor_prices.json`、`calendar_day.txt`、`calendar_future.txt`；另有 `manifest.json` 与 `completion.json`。每个数据成员至多 4 MiB，合计至多 32 MiB，两个记录各至多 1 MiB。bootstrap manifest 与两个完整 prior 分别保留，未声称复制 bootstrap 全部原始文件。模型与信号 capsule 保留引用；实际使用的日历、价格 words/rows 则保存在本 bundle。

先对八个非 request 逻辑成员建立 SHA256 inventory，再计算版本化 request body 的 canonical SHA256。definitions 中的 C4 request digest 在这一逻辑摘要中剔除；然后生成最终九成员 raw inventory 的 manifest。completion 绑定 manifest 的 raw digest，故没有自引用。C4 serializer/reader/writer 和 CLI 源码具有独立实现指纹。物理 root 身份仅用于本次 I/O 稳定性检查；新 completion 使用下述 V2 内容绑定。

reader 校验完整集合、canonical schema、摘要、角色/来源/日期、完整 prior、ranking、计划/intent 关联，以及 float32 words 到 cash close 和各臂价格 projection。保存的 planner report 仍是封存报告，reader 不重建 planner authority，也不重新执行生产 planner。合法零订单、signed cash、pending 和 shortage 保留。

默认时间策略 `PAPER_ACCOUNTING_TIME_V1` 只保存实际时间观察。deadline 可省略；若提供，仅作为行政时间事实保存，不参与准入。累计运行超过旧 600 秒、wall/monotonic 差超过 2 秒或 wall clock 回拨都记录事实／问题码，不丢弃正确会计输入。无效 wall 时间保存为 null。`accounting_time` 显示真实创建、写前数据读验时间及 `BEFORE_OPEN / AFTER_OPEN_SIMULATION / UNKNOWN`；这些标签不代表严格事前认证。next-open 经显式时区转换的日期仍须等于所选 anchor 后第一交易日。

completion 保存 `accounting_time_observations`，截止写入前数据 bundle 的读验；最后完整发布观察仍只存在当次可选 stdout 中。纸面 COMMITTED、reader 和 ADOPT 均不授予 `prospective_claim` 或旧严格 `epoch_started`，账目可用性由完整内容读验决定。`LOCAL_PREOPEN_OBSERVATION_V1` 是旧严格时间合同；仅显式 `publication_mode=STRICT_PREOPEN` 使用原 deadline、600 秒和 2 秒门槛，旧 COMMITTED/stdout 的历史含义保留。

退出码：READY/COMMITTED/ADOPTED/VERIFIED 为 0；前置拒绝或 request mismatch 为 2；VERSION_BREAK 为 3；CONFLICT 为 4；UNCERTAIN/INCOMPLETE 为 5。写后失败报告实际已写或 UNKNOWN，进程控制中断传播。缺 completion 不允许补写；完整持久记录可按当前读验直接纸面结算，这不证明原发布曾返回最后 COMMITTED。通常 safe stdout 不含 private epoch、路径、证券、资金或原始异常；HOLD 请求会展示调用者给出的区间与说明。

开发测试只用临时工作区。实际写入应使用 owner 选定的工作区、provider、账户和范围；本指南命令不自动授权生产运行或模型训练。普通纸面操作不需要行政 deadline 或 stdout 文件，仍需审阅 prepare 的日期、来源、订单计数和 request digest。

C3/C4/D2 默认将任意源码差异或源码等价比较不可用作为观察信息；只要其余模型、配置、信号与账户校验通过，就可以继续纸面记账。历史生产封存代码不必等于当前研究代码，当前工作文件仍与实际加载来源核对，并记录工作文件内容指纹，不能仅用 HEAD 代表执行内容。真实信号损坏、排名重建矛盾、模型/配置变化和错误前序仍拒绝。旧公开 decision-surface API 保留原比较语义；新日常路径不授予严格同段或维护许可。

新 request 的 `input_provenance.source_change` 使用独立 `schema_version=1` / `SOURCE_DIFFERENCE_OBSERVATION_ONLY_V1`：`production_comparison` 保存两期生产源码比较，`production_commit`/`production_digest` 保存本期预测生产来源，`execution_commit`/`execution_tree`/`execution_implementation_digest`/`execution_code_digest` 保存当前研究执行来源；`production_execution_comparison` 为 EQUAL、DIFFERENT 或 INCOMPARABLE，不可观察时保存原因。它参与 input/request digest，不决定账户断段。旧 `maintenance_admission` 记录继续按旧合同读取。completion V2 的可迁移绑定与该字段版本独立；回退程序须保留可读取新字段的 reader，不改写历史记录。新纸面时间／可选 stdout 语义见上文；旧严格记录、防重复和中断合同保持兼容。

共同缺行的前向输入保留完整 eligible 清册及每臂 coverage_counts。部分输入的 input_provenance 必须携带 coverage_policy=COMMON_ANCHOR_RANK_EQUAL_V1；reader 检查两臂 scored 集合一致、缺失原因为 missing_prediction、统计与排名/计划一致，价格集合为各臂 scored ∪ holdings，联合 receipt 不得漏项或多项。角色 COMPLETE 只表示该臂计算完成。旧完整输入无需新增字段；inspect/adopt 仍不能生成原始成功时间或 prospective_claim。


## 记录复制与格式兼容

新写入的 `completion.schema_version=2` 使用 `FORWARD_RECORD_BINDING_V2`：`target_binding_digest` 绑定记录种类、epoch、周期/index、request 和 manifest 摘要。连续记录的 `store_bindings` 使用 `FORWARD_CHAIN_STORE_V2`，区分 intent/settlement 角色，并绑定首期 request/manifest/operation、冻结 selectors、definition request 和 schedule。它们不包含路径、device、inode、mtime 或主机身份。request/bundle 自身的版本及 safe stdout 的 schema 不变，不能仅凭 request 的版本 2 判断是否可迁移。

可将四个运行 store 原样复制到新目录，原 COMMITTED stdout 若存在可同时保留，恢复目录 0700、文件 0600 权限，再显式指定新根。reader/inspect/adopt 不改写记录、不刷新首次成功时间，也不产生新的 prospective claim。结算消费复制后的完整 intent；stdout 可选；D3 可读取复制链；D2 从已结算 after-state 接续，无需重建 bootstrap 或 epoch。Production、provider、定义和模型仍按现有参数提供，这不包含整个工作区或 MLflow URI 的迁移。

V1 completion 在原位置继续按旧物理绑定读取，后续新记录写 V2；新 V2 settlement 可保留未经改写的 V1 来源 completion，并以自身已校验的内容绑定支持复制；V1 intent 本身仍受原目录限制。复制 V1 后会返回 `LEGACY_PHYSICAL_BINDING_MISMATCH`，本功能不迁移或重写旧记录。未知、缺失或类型混乱的版本被拒绝。操作期间仍检查规范路径、目录身份、文件指纹、来源变动和 create-only 冲突。两个独立副本不共享全局锁，恢复后应选定一个活动副本。代码回退须保留能读取 V2、PAPER_ACCOUNTING_TIME_V1、可选 stdout 和 HOLD 的 reader，不得降级改写已发布记录；这不是经济规则变更或新的生产审批门槛。
