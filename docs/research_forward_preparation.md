# 首次 Shadow Forward intent 准备（C3）

C3 从显式 current cycle、冻结四模型 Champion、冻结三模型 Challenger 和同一 matched bootstrap，计算一对内存 intent。它只读取输入并计算，不保存订单、执行交易或开始 epoch。

操作员先按工作区约定 source 正确环境，再显式指定所有物理绝对路径与已有 selector：

```bash
python -B -m quantpits.scripts.prepare_forward_intent \
  --production-workspace "$PRODUCTION_ROOT" \
  --research-workspace "$RESEARCH_ROOT" \
  --engine-root "$ENGINE_ROOT" --qlib-provider "$PROVIDER_ROOT" \
  --current-cycle "$CURRENT_CYCLE" --activation "$ACTIVATION_PATH" \
  --definition-store "$DEFINITION_STORE" --evidence-store "$DEFINITION_EVIDENCE_STORE" \
  --bootstrap-store "$BOOTSTRAP_STORE" --bootstrap-set-id "$BOOTSTRAP_SET_ID" \
  --signal-capsule-store "$SIGNAL_STORE" --signal-capsule-id "$SIGNAL_CAPSULE_ID" \
  --model-capsule-store "$MODEL_STORE" --model-capsule-id "$MODEL_CAPSULE_ID"
```

这些变量是调用示例，不由程序自动发现。`--help` 无需工作区。Python API `quantpits.research.forward_intent_preparation.prepare_first_forward_intent(...)` 使用相同顺序的显式参数，返回 `FirstForwardIntentPreparation`；CLI 支持 `main(argv)`。无需 Qlib/MLflow 初始化，也不会更改环境、cwd 或 argv。

definition/reference cycle 与 bootstrap source cycle 从 Production 读取；definition、evidence、bootstrap 和 capsules 从 Research 读取。model capsule 使用 definition evidence cycle，weekly signal 与 ranking 使用 current cycle。`--evidence-store` 是 Research definition-evidence store。没有 latest discovery、输出目录、发布或修复选项。

| 状态 | 退出码 | 含义 |
| --- | --- | --- |
| `PREPARED` | 0 | 两臂规划完成，输入关联、四源 parity 与最终稳定性通过 |
| `VERSION_BREAK` | 3 | 既有 decision-surface observer 观察到冻结版本变化 |
| `PRECONDITION_BLOCKED` | 2 | 输入缺失、不相容、缺分、parity/实际源码不符、规划异常或观察不稳定 |

stdout 仅一行 canonical safe JSON，包含日期、固定 role、状态、计数和摘要，不输出证券、现金、数量、私有命名 ID、路径或异常文本。未观察的计数为 null；前置失败时两臂 `NOT_RUN`。普通单臂规划失败后仍观察另一臂，整对不交付；进程控制中断传播。

只有 `PREPARED` 的 API 结果保留完整内存 pair。safe JSON 不能恢复 pair，也不是 C4 的写入授权。`intent_publication_capability`、`epoch_started`、`prospective_claim`、`promotion_capability` 和 `did_write` 始终 false。

首版要求每个模型在 anchor 上完整覆盖 sealed universe：缺分、重复、foreign、NaN/inf 和 sealed unscored 都会阻塞，不填分或缩池。融合沿用 Stage A 的平均并列 percentile、冻结成员顺序的 pandas mean 和 canonical tie-break。两臂相同排名、100% overlap、零订单均合法。

只观察一次两臂 union 的 anchor close，再分别投影到 B1。day/day_future 的历史 session 顺序必须一致，trade date 取 anchor 后首个 future session。价格来源为 `CURRENT_PROVIDER_ANCHOR_OBSERVATION`；历史物化关系仍为 `unverified`。calendar 与 seal 的 byte-match 单独报告，calendar 正常扩展不等于历史价格重现。现有 reader 为 hash 读取完整 close/factor 文件；决策仅解释 anchor 值，不读取 open 文件、不使用 next-open。

B1 的合法缺价、pending forced exit、buy shortage、负现金和零订单保留。C3 不结算，也不验证结算后的现金或 no-deficit-worsening。历史日期的技术准备不能回填 prospective evidence；C4 需另选尚未开盘的周期并重新观察输入。

旧 `inspect_decision_surface` 新增 `--reference-source research|production`，默认仍为 `research`；fresh split-root 布局显式选 `production`，不会按存在性 fallback。此次容量修复仅适用于 fresh definition evidence 的只读 ADOPT；prepare/publish writer 的全树预算未改变。

## Predict-only 副本的身份与历史兼容

`predict-only` 继续保存本次实际使用的 `model.pkl`（CPCV 保存完整 fold 集合）。预测运行 ID 与直接父 recorder 用于审计，不代表一次新的训练。新增的 `training_origin_record_id` / `training_origin_experiment` tags 来自沿明确父链的实际追溯；`training_origin_status=VERIFIED` 表示本次追溯完成。历史父记录缺失、循环、模型名矛盾或缓存根身份冲突时，预测仍可使用已加载的模型，记录 `UNRESOLVED`，不伪造原始训练身份。进程控制中断继续传播。

冻结 definition、capsule、manifest 和 seal 不改写。原有精确来源匹配仍可使用；当历史封存中的直接来源或 artifacts 清单变化时，decision surface 通过额外的只读兼容观察比较：

- 明确的训练来源链终点，不能根据 latest 或仅根据缓存根 tag 推断；
- 每个模型/fold 的实际文件，先核验其 raw SHA-256 与所属周期的封存清单完全一致；
- 完整模型对象的版本化内容指纹，及未被明确分类为报告的辅助文件。

内容指纹使用非执行的 pickle opcode 解析，不反序列化模型、不导入 Torch、不执行 GLOBAL/REDUCE。仅规范化 FRAME、等价字节长度编码以及已识别的 Torch legacy storage 分配 ID；保留张量内容、dtype、shape/stride、别名关系、模型配置和优化器状态。CatBoost 内嵌模型 bytes 保持精确比较。它是保守的复制兼容协议，不是任意 pickle 的通用语义等价判定。未知编码不能因解析失败被认定为相同。

`pred.pkl`、`label.pkl`、`code_status.txt` / `code_diff.txt` / `code_cached.txt`、`portfolio_analysis/` 和 `sig_analysis/` 不参与模型身份；其各自封存、信号和运行校验继续生效。未知辅助输入仍按相对文件名和 raw digest 比较。推理代码、融合规则、市场和订单策略仍由原有独立组件检查。

历史兼容观察目前仅支持工作区内显式 `mlruns/<experiment-id>/<recorder-id>/artifacts` 的物理文件后端；会读取对应实验元数据和父链 tags，以及封存引用的模型/辅助文件，并对所选输入进行变化观察和前后核对。这些 live ancestry 信息是**本次观察的补充证据**，不是旧 seal 已封存的事实；不初始化 MLflow，不搜索最新 recorder。缺失、歧义、符号链接、内容不符或来源无法证明时返回 INCOMPARABLE（C3 为 PRECONDITION_BLOCKED）。确实不同且可验证的训练来源、权重或配置仍为 VERSION_BREAK。

只读修复不补造已清理的历史记录，也不豁免 C3 的实际引擎源码与 seal 匹配要求。修改预测代码之后，历史周期可能仍因源码不符阻塞；不能为了得到 PREPARED 修改旧 seal 或关闭校验。
