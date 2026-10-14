# 独立 paper 周期

`python -m quantpits.scripts.paper_trade` 提供一次初始化、计划、开盘结算、收盘估值、显式事件和报告。一个账户一个 workspace；账户可以只读共享其他 workspace 的融合信号及行情。账户不需要旧研究流程的成对角色、seal 或 capsule。原研究 B0/B1 的固定数量与费用语义保持原义。

## 配置与来源

账户使用自身 `config/model_config.json` 的 `market` 和 `config/strategy_config.yaml` 的 `topk_dropout` 参数：`topk`、`n_drop`、`buy_suggestion_factor`、可选 `sell_out_of_universe`。股票池从 provider 的 `instruments/<market>.txt` 按 anchor 的有效区间读取；不会从持仓推断池，也不会切回标准指数池。

在账户 workspace 创建 `config/paper_config.json`。下面全部是合成配置；费用必须按账户实际约定填写，不能直接使用回测费率。正常设限声明必须适用于所选执行日；未知、新上市、除权或特殊规则证券应给出明确的当日参考，否则该笔跳过。

```json
{
  "account_id": "example_account",
  "signal_source": {
    "workspace": "/path/to/signal_workspace",
    "mlflow_uri": "sqlite:////path/to/signal_workspace/mlflow.db",
    "experiment_id": "<EXPERIMENT_ID>",
    "recorder_id": "<RECORDER_ID>",
    "lineage_files": ["data/training_sources/chosen_fit.json"]
  },
  "execution": {
    "base_fee": "4.00",
    "base_rate": "0.00008",
    "sell_tax_rate": "0.0007",
    "transfer_rates": {"SH": "0.00002", "SZ": "0"},
    "lot_size": 100,
    "cash_reserve": "0.00",
    "slippage_rate": "0",
    "observed_max_gross": "15000.00"
  },
  "market_rules": {
    "source": "explicit rules checked for the selected session",
    "boards": {
      "SH_MAIN": {"normal_limit": true, "limit_rate": "0.10"},
      "SZ_MAIN": {"normal_limit": true, "limit_rate": "0.10"},
      "CHINEXT": {"normal_limit": true, "limit_rate": "0.20"}
    },
    "instruments": {}
  }
}
```

`base_fee` 是基础费用下限，基础费用为 `max(base_fee, gross × base_rate)`；已包含的规费不要再扣。卖出税款与沪深单列过户费分别舍入到分，再加总。没有比例基础费用时显式设 `base_rate=0`。金额超过已观察规模时逐笔显示费用估计标记。上述税率也是合成值，不代表当前市场税率。普通买入支持 100 股整手，清仓允许尾仓零股；本入口暂不支持科创板、北交所、ETF、融资和非零额外滑点。

来源也可用 `"prediction": "output/predictions/chosen_signal.csv"` 代替 experiment/recorder。CSV 必须包含 `datetime,instrument,score`，pickle 必须是带这两个索引层的 DataFrame/Series；只读取指定 anchor。缺分成员保留在排名覆盖中，不补零。MLflow 来源显式检查 experiment、FINISHED recorder 及本地制品；SQLite 使用只读连接，file 后端只读已有 metadata，不初始化或迁移后端。来源 backend 和制品必须归所选信号 workspace。

每期保存来源 URI、recorder 或文件路径及 hash、实际池快照、价格观察、可选 fit/reload/manifest 原文。以后更换来源不会改变已保存计划。`--signal-workspace`、`--signal-mlflow-uri`、`--recorder-id` 配合 `--experiment-id` 或 `--prediction` 可以覆盖本期来源；重复 `--lineage-file` 可冻结本期附加来源材料。无需每个账户重训共享模型。

## 首期与续期命令

以下日期是示例，运行时由使用者显式选择。先填写上述配置，以及账户的市场/策略文件。起点文件是一个只读 `prod_config.json` 快照，必须有匹配日期的 `current_date` 或 `last_processed_date`、`current_cash` 和 `current_holding`；每个持仓的 `value` 是股数、`amount` 是总成本。也可以提供相同格式的自行设定起点。

```bash
PAPER_WS=/path/to/paper_workspace
PROVIDER=/path/to/qlib_provider
SOURCE_ACCOUNT=/path/to/read_only_account/config/prod_config.json
ANCHOR=2024-05-10
TRADE_DATE=2024-05-13
VALUE_DATE=2024-05-17

python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  bootstrap --source-config "$SOURCE_ACCOUNT" --date "$ANCHOR"
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  plan --cycle week_one --anchor "$ANCHOR" --trade-date "$TRADE_DATE"

# 执行日行情更新后；此前返回 WAITING_FOR_DATA，退出码 2，不变更余额。
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  settle --cycle week_one
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  value --date "$VALUE_DATE"
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  report --date "$VALUE_DATE" --since "$ANCHOR" --reference-config "$SOURCE_ACCOUNT" --save

# 旧期结清，按既有 predict-only/ensemble 命令准备明确的新信号后。
NEXT_ANCHOR=2024-05-17
NEXT_TRADE_DATE=2024-05-20
NEXT_SIGNAL=/path/to/signal_workspace/output/predictions/next_signal.csv
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  plan --cycle week_two --anchor "$NEXT_ANCHOR" --trade-date "$NEXT_TRADE_DATE" \
  --prediction "$NEXT_SIGNAL"
```

`bootstrap` 只复制一次，初始 NAV 使用当日现金收盘价；继承成本不变，收益从这个 NAV 起算。候选 `prod_config.json` 的训练占位余额不参与账户。续期直接使用 paper 结算后的现金、股数和成本。CLI 本身不训练、不生成模型预测、不调用生产流水线；新的模型信号按已有训练/融合入口准备，随后只读消费。

执行日必须晚于 anchor。已知非交易日明确报错；未来日历未覆盖时先保存 `PENDING_CALENDAR`，结算前用实际日历核对。没有整个执行日行情，或整个请求面都没有 open 且没有明确的全部停牌事实时，保留待结算。日期不会自动顺延。若暂存日期后来确定不可用，显式取消并用新周期 ID 重建：

```bash
python -m quantpits.scripts.paper_trade --workspace "$PAPER_WS" --qlib-data-dir "$PROVIDER" \
  cancel --cycle week_two --reason "explicit change of selected execution date"
# 再用 week_two_revised、用户选择的 trade-date 执行 plan。
```

取消保留原计划及原因，不能撤销已结算的交易。每个账户同时保留一个待结算周期；同名周期/事件的不同内容明确报错。

## 执行与估值口径

计划直接使用生产 position analysis 的卖出选择与完整 `buy_candidates` 顺序，在保守数量为零的过滤前保存主备名单。建议数量只是 anchor 收盘价预算参考。执行采用所选日期的现金 open（Qlib open/factor，取分精度），先卖后买。H/L 不参与；执行日 C 只用于估值。没有盘口、排队、成交量限制或盘中部分成交模型。

卖出缺价、已知停牌、触限或不可卖时保留持仓，继续其他证券。未卖出不变现、不释放名额；同日买入股数不能再卖。可用现金只包括原现金、实际卖出净收入以及结算前已入账的显式事件。买入名额是目标买入数与卖后 `topk` 空位的较小值。

每个名额预算固定为 `(clean_cash - cash_reserve) / slots` 向下取分；reserve 非负且不超过 clean_cash。逐候选计算预算与实时现金内含费的最大整手数量，实际数量可高于或低于建议。缺价、触限、买不起一手含费或池外买入都留下原因并递补，跳过不消费名额。名额/备选耗尽后保留现金；不会引入名单外证券或给已买证券追单。

open 触及任一涨跌停价时，买卖都跳过，这是本 paper 的保守假设。可靠逐证券当日上下限优先；否则仅在分类明确、显式正常设限、factor 未变且前一实际交易日现金 C 可用时，按配置比例计算并四舍五入到价格最小单位。代码前缀只确定支持的类别，不能证明正常设限；不会从 ST 名称自动推断比例。无法确定的类别/期间、除权 factor 变化或参考不明，只跳过当笔。

可为 `plan` 或 `settle` 提供 `--rules /path/to/session_rules.json`，使用同样的 `source/boards/instruments` 结构并增加准确的 `date`。逐证券项可覆盖 `suspended`、`normal_limit`、`limit_rate`、`reference_close`，或成对提供 `limit_up/limit_down`。补规则只用于当期交易约束，不更换信号、候选或前序账户；已结算后不同规则不再重算成交。

`value --date` 单独追加收盘观察，不产生交易或费用；没有调仓也可估值。个股缺 C 可沿用账户已保存的可靠估值价，显示 `price_date/stale`；完全无可靠价时只给已知市值，NAV/收益为 null。整日行情未到返回待估值。报告以 `--since` 的已保存估值计算期间收益，默认前一个观察日期；有日估值时请明确指定周起点。只有首尾观察不提供日频回撤。

## 显式事件与实盘参考

`event --file /path/to/event.json` 登记有来源且属于本账户的真实事件，不自动复制实盘分红。以下仍是合成例子：

```json
{
  "event_id": "notice_example",
  "effective_date": "2024-05-15",
  "booking_date": "2024-05-15",
  "kind": "DIVIDEND",
  "cash": "12.30",
  "instrument": "SH600001",
  "quantity_delta": 0,
  "cost_delta": "0",
  "source": "confirmed entitlement and payment notice for this account"
}
```

支持 `DIVIDEND`（正现金）、`TAX`（负现金）、`EXTERNAL`（外部入出金）和 `SHARES`（已有持仓股数/成本变动，现金为零）。事件按 booking_date 在下一次 plan/settle/value 中入账，同身份重试不重复收入。有效日期保留；后知事件不能悄悄回写历史，必须显式指定不早于账户前沿的 booking_date，追加一次修正。事件不能导致负现金或非法股数。权益资格由录入者按本账户权益日期确认；未知事件披露为未计。

报告分列外部净入金和 `NAV变化 - 净入金` 的损益，分红及补税属于投资损益。外部资金存在时不声称 NAV/初始 NAV 就是单位收益，也不把简单剔除净入金收益率叫时间加权收益率。

`--reference-config` 仅读实际账户同日快照；日期未对齐显示 `PENDING_UPDATE`，不阻断 paper。当前实盘快照不能证明期间外部现金流，故实盘期间收益需再传 `--reference-events`（相同事件格式的数组，`[]` 是明确无已知现金流）和续期的 `--reference-start-config`（期间起点快照）。首期以保存的复制起点为基准。缺少这些事实时现金/持仓/NAV 仍显示，实盘收益明确待补。

## 保存与恢复

唯一正式余额由 `<workspace>/data/paper/account.json` 中的 bootstrap 和连续账务记录派生。它保存源快照/hash、冻结计划、逐笔成交/跳过、事件、取消记录和追加估值；不会另写 `prod_config.json`。`data/paper/account.lock` 使用已有 execution lease，作用范围在本账户。`report --save` 在 `output/paper/report_<date>.json` 和 `.md` 保存派生报告，不改变余额。

新计划使用当前生产订单规则。历史计划按保存的候选顺序、建议及来源读取和消费；reader 检查身份、日期、池内归属、数量/金额结构、冻结价格及账户约束，建议金额作为当时的预算参考保留。生产建议算法修订后，旧计划仍可结算/估值；读取与同周期计划复用保持既有计划及账本 digest 原样。实际现金、费用、成本与前序链仍由账务复算校验。

写入一次原子替换并从规范路径重读核实；reader 校验来源价格、账户、前序链、计划、逐笔费用及 after-state。中断前未提交则按同周期重试；已提交则返回原结果，不能重复交易/扣费。不要通过删账本、清锁、覆盖现金或重做 bootstrap 来恢复。新账户使用新的 workspace，不复制旧账户待结计划和锁。CLI 的成功退出码为 0，待数据为 2，合同/输入错误为 1。
