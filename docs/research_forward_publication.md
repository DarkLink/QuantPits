# 首对前向 intent 发布（C4）

C4 将一次 fresh C3 的 Champion/Challenger 完整 intent pair 保存到显式 `intent_store_root / epoch_id`。首版仅支持 matched bootstrap 在开端前未交易的声明 `MATCHED_BOOTSTRAP_UNTRADED_V1`，不结算、不生成收益、不推进后续周期。

领域 API 位于 `quantpits/research/forward_intent_publication.py`：

- `prepare_first_forward_intent_publication`：C3 原 14 个显式参数，加 keyword-only `intent_store_root`、`epoch_id`、`decision_deadline_utc`、`next_open_utc`、`market_timezone`、`opening_policy`。只读返回 READY 和待审阅 request digest。
- `publish_first_forward_intent_pair`：相同参数加 `expected_request_digest`。目标不存在时重新执行 C3；摘要必须完全一致，来源 guards 持有到提交核验结束。
- `inspect_first_forward_intent_pair(intent_store_root, epoch_id, *, expected_request_digest)`：只读检查原槽，不需要 Production/provider/MLflow 或模型推理。成功观察的 `d1_inputs` 固定顺序返回两臂 `role / prior / intents / execution_assumption`，使用既有严格会计类型。

C3 public API 仍只读并在返回前关闭 guards；私有移交接缝只供 C4 使用。调用者不能把序列化的 PREPARED 或 READY 结果作为发布资格。

CLI：`python -B -m quantpits.scripts.publish_forward_intent --help`。action 为 `prepare`、`publish`、`inspect`。prepare/publish 沿用 C3 参数名，并增加 `--intent-store-root --epoch-id --decision-deadline-utc --next-open-utc --market-timezone --opening-policy`；publish 另需 `--expected-request-digest`。inspect 只接受后者及 target root/epoch。没有 `--now` 或历史时间回填参数。

root 必须预先存在，是规范绝对物理路径、0700 私有目录。epoch 目录 create-only；文件 O_EXCL/0600、fsync 后从规范路径重读。同一 epoch 即使更换 cycle 也不创建第二槽。原请求已存在时只读 ADOPT，不重新读取 live 来源、不覆盖、不自动补文件或删除残留。

bundle 固定九个数据成员：`request.json`、`definitions.json`、`priors.json`、两臂 `*_ranking.csv`、`plans.json`、`anchor_prices.json`、`calendar_day.txt`、`calendar_future.txt`；另有 `manifest.json` 与 `completion.json`。每个数据成员至多 4 MiB，合计至多 32 MiB，两个记录各至多 1 MiB。bootstrap manifest 与两个完整 prior 分别保留，未声称复制 bootstrap 全部原始文件。模型与信号 capsule 保留引用；实际使用的日历、价格 words/rows 则保存在本 bundle。

先对八个非 request 逻辑成员建立 SHA256 inventory，再计算版本化 request body 的 canonical SHA256。definitions 中的 C4 request digest 在这一逻辑摘要中剔除；然后生成最终九成员 raw inventory 的 manifest。completion 绑定 manifest 的 raw digest，故没有自引用。C4 serializer/reader/writer 和 CLI 源码具有独立实现指纹。物理 root 身份只进入执行绑定摘要和 completion，不改变语义 request。

reader 校验完整集合、canonical schema、摘要、角色/来源/日期、完整 prior、ranking、计划/intent 关联，以及 float32 words 到 cash close 和各臂价格 projection。保存的 planner report 仍是封存报告，reader 不重建 planner authority，也不重新执行生产 planner。合法零订单、signed cash、pending 和 shortage 保留。

时间策略固定 `LOCAL_PREOPEN_OBSERVATION_V1`：aware UTC 本机 clock 配合 monotonic，调用内累计差异不超过 2 秒、预算不超过 600 秒，且各 gate 严格早于 decision deadline。next-open 经显式时区转换的日期必须等于保存日历的下一交易日，deadline 不晚于 next-open；交易所开盘时刻由操作者明确确认。程序不联网认证时钟或交易所日历。

completion 的时刻只声明其创建前已经完成的数据 bundle 核验，不能证明 completion 自身未来的 fsync。COMMITTED 的最终 gate 记录在当次 safe JSON 中；正式运行必须将原始成功 safe JSON 保存到事先授权位置。只有当次 COMMITTED 授予本地观察等级的 prospective/epoch_started。reader/ADOPT 分别报告 `bundle_verified`、`completion_record_verified`、`recorded_time_claim`、`d1_readable`，始终不重新授予 prospective 或启动 epoch。

退出码：READY/COMMITTED/ADOPTED/VERIFIED 为 0；前置拒绝或 request mismatch 为 2；VERSION_BREAK 为 3；CONFLICT 为 4；UNCERTAIN/INCOMPLETE 为 5。写后失败报告实际已写或 UNKNOWN；进程控制中断传播。缺 completion 不允许补写；已有 completion 但最后 gate 超时/不确定时，后来的字节检查不能代替原始成功记录。safe stdout 不含 private epoch、路径、证券、资金或原始异常。

开发测试仅使用临时工作区。真实 publication 另需具体候选、Production/Research/provider 绑定、冻结 selectors、新周期/截止、request digest、唯一 target、safe 记录位置和最多 600 秒预算的授权。历史 C3 PREPARED 不是未来周期依据。开发验收与首次真实运行分别记录：实现及合成合同验收后，后续开发可使用该 reader/schema，无需等待市场周期；首次真实发布仍须独立通过时间与来源检查，才能声明前向 epoch 已启动。
