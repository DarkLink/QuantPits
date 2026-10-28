# 项目周期运行

`python -m quantpits.scripts.run_cycle` 用一个私有项目配置组织多个账户的一轮周任务。账户列表沿用 [账户组比较](account_comparison.md) 的组文件；模型、会计和 paper journal 仍由原领域入口负责。入口支持静态模型的 predict-only，不启动 fit。

## 配置

实际账户的周期订单原位保存在 `output/cycles/<project>/<cycle>/orders/`；直接运行旧 order_gen 的默认位置仍为 `output/`。分类、替代买入分析、dashboard 和 deep_analysis 会共同发现该账户的周期目录、`output/` 顶层及 `data/order_history/`，按文件执行日匹配，不从周期 anchor 推断日期。原文件受 manifest／progress 引用，不应为分析移动、覆盖或自动归档它们。

同日同方向的相同字节副本只计一次；不同建议列明冲突。报告可通过重复 `--suggestion-file <工作区相对路径或绝对路径>` 选择实际文件，API 使用 `suggestion_files=[绝对路径, ...]`，详见[分析指南](08_ANALYSIS_GUIDE.md)。未解析来源使订单分析不完整；post_trade 已提交账务保留，派生分类记录失败／警告。本次兼容不会自动重分类旧历史，也不改变 paper／交易准入或周期恢复身份。

项目文件及其 `cycles/` 目录放在账户、行情目录之外。下面是合成示例；先为三个成员分别建立已有的实际/paper 工作区，再填写组文件。paper 必须已通过原 `paper_trade bootstrap` 初始化。

```yaml
# group.yaml：与 compare_accounts 共用，不添加项目字段
name: Example project
members:
  - id: live
    kind: actual
    workspace: /path/to/actual
    mlflow_uri: sqlite:////path/to/actual/mlflow.db
    qlib_data_dir: /path/to/provider
  - id: paper_a
    kind: paper
    workspace: /path/to/paper_a
    mlflow_uri: sqlite:////path/to/paper_a/mlflow.db
    qlib_data_dir: /path/to/provider
  - id: paper_b
    kind: paper
    workspace: /path/to/paper_b
    mlflow_uri: sqlite:////path/to/paper_b/mlflow.db
    qlib_data_dir: /path/to/provider
```

```yaml
# project.yaml：group/rules 相对本文件；fit_records 相对 producer workspace
version: 1
group: group.yaml
signals:
  shared:
    producer: live
    models: [model_a, model_b] # 注册表中的明确静态模型名，不能用 all-enabled
    fit_records: data/training_sources/selected_fit.json
    combo: chosen
    norm_method: rank
  candidate:
    producer: paper_b
    models: [model_a, model_b]
    fit_records: data/training_sources/selected_fit.json
    combo: chosen
    norm_method: rank
accounts:
  live: {signal: shared}
  paper_a: {signal: shared}
  paper_b: {signal: candidate}
```

每个信号组每周期只预测、融合一次，绑定账户共享固定信号。每个 producer workspace 配置一个信号组，combo 的实际模型集合必须与该组明确列表一致。账户及 producer 可以共用一个 workspace；两个账户的 workspace 必须独立。组中每个成员都必须绑定来源，不能通过删掉失败成员缩小请求。

建议在组中明确写入后端 URI 和 provider。子进程会显式覆盖 workspace、tracking URI、provider、region，不依赖调用者 source 的环境。写入路径和后端必须在所属 workspace 内，进度、规则与汇总写在项目目录；生产执行须另有相应工作区授权。

## 周期输入与命令

`anchor` 是本期预测末日，`value-through` 是旧账估值末日，`trade-date` 是用户选定执行日，`since` 是已有比较基线。要求 `since < value-through <= anchor < trade-date`。执行日按实际/未来交易日历核对，可以选择非默认交易日；未来执行行情尚未到达时只生成计划。

```json
{
  "fit_records": {"candidate": "data/training_sources/explicit_candidate.json"},
  "actual": {
    "live": {
      "settlement_bundle": "data/2024-05-13-2024-05-17-table.xlsx",
      "allow_missing_settlement": false
    }
  },
  "paper_cycles": {"paper_a": "paper_next_a", "paper_b": "paper_next_b"},
  "hold": [],
  "skip": []
}
```

`settlement_bundle` 必须相对该实盘 workspace，格式沿原 post_trade；也可省略，使用原逐日文件发现。资料缺失时等待，只有明确 `allow_missing_settlement: true` 才沿原命令的空日假设处理。`paper_cycles` 是本期新计划的 ID 映射；到期 pending 自动按账本里的原 ID 结算。`hold` 仍推进旧账和估值，但不生成该成员的新订单/计划。`skip` 接受 plan 中的明确步骤 ID；依赖被跳过步骤的后续步骤继续等待。

可选 `fit_records` 只为本次新cycle覆盖指定signal的来源，其他signal沿project默认；普通无覆盖操作无需添加字段。路径相对该signal的 **producer workspace**，也可用其中的绝对路径，不相对project目录或paper消费者。来源可以由 [model_source 导出](01_TRAINING_GUIDE.md#明确预览与导出模型来源)，也可直接使用已有合法V1／V2文件；原模型成员、operation和直接source检查保持，不新增“只能选择train”的限制。plan的signals及覆盖预测步骤会显示当期选择，status保留保存请求中的来源。

未知signal、错误类型、空路径或跨workspace符号链接明确报错；文件暂缺只等待其依赖分支，不能回退project默认或latest。请求保存路径，首次可用内容由原 `fit_sha256` 绑定；冻结前内容变更会拒绝，冻结为周期signal目录的 `fit_records.json` 后以该副本为准，原文件删改不使已冻结周期失效。resume／status／独立evaluate不重新读取外部inputs选源，evaluate继续使用该周期保存的预测证据。

同cycle不能更换覆盖，需使用新cycle ID。不要为换下一期模型修改共享project：原project/group identity校验仍生效，改变producer、账户、模型集合或combo后不保证旧cycle恢复。旧inputs不补空映射，旧JSON／身份保持；显式 `"fit_records": {}` 属于本次新请求。回退代码前，已开始使用此字段的周期须保留相应代码和冻结资料至完成或明确处置，不能删字段或改摘要伪装成旧请求。此功能不会自动切换生产绑定、结算pending或执行训练。

```bash
python -m quantpits.scripts.run_cycle plan --project /path/to/project.yaml --cycle week_one \
  --anchor 2024-05-17 --value-through 2024-05-17 --trade-date 2024-05-21 \
  --since 2024-05-10 --inputs /path/to/inputs.json

# 获得实际写入授权后，使用相同参数运行
python -m quantpits.scripts.run_cycle run --project /path/to/project.yaml --cycle week_one \
  --anchor 2024-05-17 --value-through 2024-05-17 --trade-date 2024-05-21 \
  --since 2024-05-10 --inputs /path/to/inputs.json
python -m quantpits.scripts.run_cycle status --project /path/to/project.yaml --cycle week_one
python -m quantpits.scripts.run_cycle resume --project /path/to/project.yaml --cycle week_one
```

plan/status 不初始化后端、不创建账户锁或项目进度。plan 对未生成的预测与融合标记 `pending_generation`，不预填 recorder。run/resume 保存 `cycles/<id>/progress.json`、各步骤日志和既有比较报告。退出码：完整/只读 plan 为 0，失败或等待为 2，配置错误为 1，中断为 130。

## 新周期可选导出完整订单候选

在原inputs中加入明确选择（可与其他字段组合）：

```json
{"order_candidates": ["live"]}
```

列表须为本group中不重复的actual成员ID，不能选择paper或未知成员。所选成员不能同时HOLD或skip其 `<member>-new` order步骤；移除导出选择或取消该HOLD／skip即可。省略字段不增加默认值，显式空列表是一个明确的新请求。共享project、账户配置及paper冻结计划不需要修改。

被选成员的**原order调用一次**生成原CSV和 `output/cycles/<project-stem>/<cycle>/orders/order_candidates.json`，文件位于该成员workspace。plan显示 `candidate_output`，status的原step证据提供候选路径／digest／count及files摘要。采用前核对原native manifest、候选格式、执行日期和本期固定信号来源；原生完成后中断时resume沿同一个manifest／候选采用，不再计算order。实际没有目标时允许完整空候选。

本功能不增加步骤、不读取未来开盘观察，也不自动运行prepare_orders。日后自行提供当前现金／持仓和现金开盘观察即可预演：

```bash
python -m quantpits.scripts.prepare_orders --workspace /path/to/actual_workspace \
  --candidates output/cycles/project/new_cycle/orders/order_candidates.json \
  --observation data/current_open_observation.json --preview
```

观察字段、预计费用与未完成卖出的处理见[订单指南](06_ORDER_GEN_GUIDE.md#保存完整候选并按当前现金开盘观察准备数量)。预览不执行交易、不写账户、不重新预测／融合。原request保存导出选择，resume不重新读取外部inputs；已完成候选损坏会被status发现。旧cycle继续沿原请求恢复，无候选仍有效，不能通过改外部inputs或共享project给旧请求补导出；需要时选择新cycle。

## 固定来源与恢复

单命令也支持 `static_train --anchor YYYY-MM-DD`，以及 `order_gen --anchor YYYY-MM-DD --trade-date YYYY-MM-DD --ensemble-record-file <fixed.json>`。显式融合文件保留原 `combos/default_combo` 格式，可附带 `predictions.<combo>.path/sha256` 供跨 workspace 只读消费；请求组合缺失直接报错。未指定新参数时沿用旧日期与来源默认行为，`--record-file` 继续负责单模型记录。

预测必须确认原生 manifest、完整成员及发布 receipt；从该事务保留的原始 postimage 提取本期记录，避免复制后来变化的 latest。融合按本次 manifest 中的明确 recorder 读取，导出 anchor 信号与固定映射。制品位于 producer 的 `output/cycles/<project-key>/<cycle>/<signal>/`，包括 `fit_records.json`、`predictions.json`、`ensemble.json`、`signal.csv` 和 receipt 引用。账户引用这些制品，不覆写 `config/ensemble_records.json` 切换下游来源。原生融合命令仍正常发布自身默认映射。

券商资料不足只阻断该实盘账户的新订单；paper 继续独立结算、估值和生成新计划。成功步骤核实原生结果后采用，未完成的训练继续原 run/state，实盘继续原事务，paper 采用原 journal。Ctrl+C/终止信号向子进程传播并等待退出；项目保留中断状态。仅报告失败时，resume 只重做派生汇总。源码或账户现金变化不要求重跑已完成预测；未完成的原生 run 仍按其原恢复合同校验。

周期日期、成员或来源配置改变时使用新项目 cycle ID。已有 paper 计划继续冻结；需要取消时使用原 paper 的显式 cancel，再按新请求操作。不要删除原生 lease、state、事务或账本来绕开恢复。

paper 从已有估值前沿向 `value-through` 逐交易日推进，每个日期单独记录原生结果。到期执行日前先估原持仓；执行日先按旧 cycle ID 结算，再估当天及后续收盘。无成交和 HOLD 仍逐日估值；节假日不造记录，`since` 只用于报告比较，不作为回放起点。已有历史缺口不自动回填；若原生账户状态已推进到更晚日期，也不会使用新持仓补更早净值。新计划依赖本轮所有必要日期完成。

已知未来日历可用于列出等待日期，执行仍逐日核对实际行情；日历范围尚不足时保存原账户前沿和 pending ID，补齐后展开同一请求。已落账且通过原生校验的 STALE 若有有效缓存价和 NAV，表示该日已处理，后续日期正常推进，无需补出历史报价。run/status/resume 的估值步骤在 `valuation` 中展示原生状态、NAV、各持仓价格及其日期，STALE 保留原标记；项目进度仅保存原生证明引用。PARTIAL/未知 NAV 保留真实缺口，命令失败、未落账或整日行情尚未到齐仍等待或恢复，独立账户继续。resume 采用已保存的可用日期，已有旧周期按原保存范围读取，不通过重新规划改写历史。

比较报告沿用原生质量口径；比较端点为 STALE 时仍可能返回 partial，项目报告步骤显示等待，已处理的账户日期不重放。

## 普通日前参考规则

可在项目指定共同 `rules: dated_rules.json`，账户的 `rules` 可覆盖；`pending_rules: {original_cycle_id: dated_rules.json}` 可为旧期提供明确补充。规则沿原 paper 格式，使用时冻结本周期文件。

也可设 `prepare_rules: normal_session.json`，复用 PaperMarket 观察 anchor 的现金收盘价，并生成本期 dated rules。此方式要求各成员共用 provider；模板必须显式声明：

```json
{
  "reference_assumption": "anchor_close_for_normal_session",
  "source": "owner-selected normal session; known exceptions explicitly supplied",
  "boards": {"SH_MAIN": {"normal_limit": true, "limit_rate": "0.10"}},
  "instruments": {}
}
```

数值是示例。板块假设和个股例外由使用者填写，已给的个股 `reference_close` 不被覆盖。准备结果和观察 receipt 放在周期目录，不修改行情。未知或缺价沿原 paper 的局部处理；整个交易日数据不足保持等待。

更详细差异可对同一组、同一估值日期另外调用 `compare_accounts --explain`；周期入口默认生成已有财务比较及账户建议。

## 从已保存信号比较模型与组合

对已有周期可单独重新评估，无需重跑预测、融合或交易步骤：

```bash
python -m quantpits.scripts.run_cycle evaluate --project /path/to/project.yaml --cycle week_one
python -m quantpits.scripts.run_cycle evaluate --project /path/to/project.yaml --cycle week_one \
  --start-date 2024-05-13 --end-date 2024-05-17 --account 100000
```

也可在原 `run`／`resume` 命令添加 `--evaluate`，核心周期 complete 后调用同一比较能力。
此时 `--start-date`／`--end-date`／`--account` 仅控制诊断评估，不进入原交易请求身份；
`--account` 默认100000，为各信号新建的空仓回测账户资金，不修改实盘或paper现金。
未加 `--evaluate` 的旧命令与status行为保留。

每个signal组读取本周期固定predictions／ensemble对应的原recorder完整pred.pkl，校验来源及字节；
不读latest换源、不加载模型、不重新融合。anchor的signal.csv不能替代历史融合预测。
同组成员及组合使用共同窗口、初始现金／空仓、策略、频率、基准、成交价与费用；
默认窗口取各保存信号共同覆盖的起止日期。显式区间缺覆盖会列出缺失日期与原覆盖，不能悄悄裁短后报complete。
股票覆盖保留各自信号，组间分别展示条件和窗口，不做跨组优劣排名。

每次评估新建 `cycles/<cycle>/evaluations/evaluation-*/`，内含总结果JSON／Markdown及各组
`result.json`、`comparison.md`、`summary.csv`、逐信号轨迹和worker日志。
净值以诊断初始现金为1计，首期费用和从初始资金发生的回撤保留；Qlib仍使用原生信号滞后及回测频率，
报告标明信号窗口、首末报告bar及样本数，首个bar可能尚未成交。
这些是对保存信号的历史诊断回放，与连续实盘／paper净值分开。

原YAML的SigAna／PortAna继续执行。新表不重算IC／RankIC或补历史标签，IC保持未评估，
原recorder指标仍是原标签观察期的记录；未知尾部标签不填0。诊断不会覆盖源指标、原报告、融合权重或分数。
首版重新做必要回测，不承诺已消除原records与新比较之间的回测重复。

单独evaluate缺来源／评估失败时返回非零，组与成员失败仍完整保留；某组失败不吞掉其他组。
`run/resume --evaluate`的普通评估失败放在独立evaluation结果中，核心complete及原退出语义保留；
核心未完成则不启动附带评估。Ctrl+C等中断沿原CLI传播。评估不新增核心步骤、进度或恢复状态机。
