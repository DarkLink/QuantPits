# 项目周期运行

`python -m quantpits.scripts.run_cycle` 用一个私有项目配置组织多个账户的一轮周任务。账户列表沿用 [账户组比较](account_comparison.md) 的组文件；模型、会计和 paper journal 仍由原领域入口负责。入口支持静态模型的 predict-only，不启动 fit。

## 配置

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

## 固定来源与恢复

单命令也支持 `static_train --anchor YYYY-MM-DD`，以及 `order_gen --anchor YYYY-MM-DD --trade-date YYYY-MM-DD --ensemble-record-file <fixed.json>`。显式融合文件保留原 `combos/default_combo` 格式，可附带 `predictions.<combo>.path/sha256` 供跨 workspace 只读消费；请求组合缺失直接报错。未指定新参数时沿用旧日期与来源默认行为，`--record-file` 继续负责单模型记录。

预测必须确认原生 manifest、完整成员及发布 receipt；从该事务保留的原始 postimage 提取本期记录，避免复制后来变化的 latest。融合按本次 manifest 中的明确 recorder 读取，导出 anchor 信号与固定映射。制品位于 producer 的 `output/cycles/<project-key>/<cycle>/<signal>/`，包括 `fit_records.json`、`predictions.json`、`ensemble.json`、`signal.csv` 和 receipt 引用。账户引用这些制品，不覆写 `config/ensemble_records.json` 切换下游来源。原生融合命令仍正常发布自身默认映射。

券商资料不足只阻断该实盘账户的新订单；paper 继续独立结算、估值和生成新计划。成功步骤核实原生结果后采用，未完成的训练继续原 run/state，实盘继续原事务，paper 采用原 journal。Ctrl+C/终止信号向子进程传播并等待退出；项目保留中断状态。仅报告失败时，resume 只重做派生汇总。源码或账户现金变化不要求重跑已完成预测；未完成的原生 run 仍按其原恢复合同校验。

周期日期、成员或来源配置改变时使用新项目 cycle ID。已有 paper 计划继续冻结；需要取消时使用原 paper 的显式 cancel，再按新请求操作。不要删除原生 lease、state、事务或账本来绕开恢复。

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
