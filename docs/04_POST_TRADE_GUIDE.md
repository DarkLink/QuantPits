# Post-Trade 批量处理使用指南

## 概览

Post-Trade 命令统一处理三类互补的券商证据：交割单、委托和逐笔成交。默认 `--scope all`，既更新现金/持仓，也维护执行分析所需的原始委托/成交日志。

**本脚本与预测、融合、回测等模块完全解耦，不依赖任何模型输出。**

| 脚本 | 用途 |
|------|------|
| `prod_post_trade.py` | 批量处理交易日数据，更新持仓和资金 |
| `prod_post_trade_analytics.py` | 兼容入口，仅摄取委托和逐笔成交证据 |

### 数据权威边界

- `YYYY-MM-DD-table.xlsx`（settlement）是现金、持仓、费用和红利的唯一权威来源。
- `YYYY-MM-DD-order.xlsx`（order）记录委托意图、成交率、撤单和状态。
- `YYYY-MM-DD-trade.xlsx`（trade）记录逐笔成交时间、数量和价格。

三者不能互相替代。执行质量分析依赖 order/trade；账户状态只允许 settlement 改变。

---

## 文件结构

```text
QuantPits/
├── quantpits/
│   ├── scripts/
│   │   └── prod_post_trade.py          # 本脚本
│   └── docs/
│       └── 04_POST_TRADE_GUIDE.md        # 本文档
│
└── workspaces/
    └── <YourWorkspace>/                  # 激活的隔离工作区
        ├── config/
            ├── prod_config.json        # 持仓/现金/处理状态
        │   └── cashflow.json             # 出入金记录
        └── data/
            ├── YYYY-MM-DD-table.xlsx     # 交易软件导出文件（每日一个，当前来源：国泰君安交割单脱敏导出）
            ├── YYYY-MM-DD-order.xlsx     # 委托证据
            ├── YYYY-MM-DD-trade.xlsx     # 逐笔成交证据
            ├── emp-table.xlsx            # 仅保留给旧 helper；主命令不再隐式回退
            ├── trade_log_full.csv        # 交易日志（累计）
            ├── trade_detail_YYYY-MM-DD.csv # 每日交易详情
            ├── trade_classification.csv  # 交易分类打标（累计：量化信号、替代、手工）
            ├── holding_log_full.csv      # 持仓日志（累计）
            ├── daily_amount_log_full.csv # 每日资金汇总（累计）
            ├── raw_order_log_full.csv    # 委托证据（累计）
            ├── raw_trade_log_full.csv    # 逐笔成交证据（累计）
            └── post_trade_ingestion_state.json # source fingerprint receipts
```

---

## Cashflow 配置

### 新格式（推荐）

`config/cashflow.json` 支持按日期指定多次出入金：

```json
{
    "cashflows": {
        "2026-02-03": 50000,
        "2026-02-06": -20000
    }
}
```

- **正数** = 入金（向账户转入资金）
- **负数** = 出金（从账户转出资金）
- 仅在对应日期的处理中生效

### 旧格式（向后兼容）

```json
{
    "cash_flow_today": 50000
}
```

旧格式会将全部金额应用到批次的**第一个交易日**。

### 处理后归档

当前兼容状态流程读取 cashflow，但不会自动归档。消费范围受控的归档和事务写入将在后续 state-transaction 阶段完成；不要假设普通运行已经移动条目。

---

## 运行方式

```bash
cd QuantPits

# 正常运行：处理上次处理日到今天的所有交易日
python quantpits/scripts/prod_post_trade.py --scope all

# 轻量计划：不初始化 Qlib、不打开 Excel、不写文件
python quantpits/scripts/prod_post_trade.py --explain-plan
python quantpits/scripts/prod_post_trade.py --json-plan

# 指定券商进行处理 (默认读取 prod_config.json 中的配置，兜底为 gtja)
python quantpits/scripts/prod_post_trade.py --broker gtja

# 严格 dry-run：打开并验证输入，但不写文件
python quantpits/scripts/prod_post_trade.py --dry-run

# 明确只处理某一侧（计划会给出另一侧被跳过的 warning）
python quantpits/scripts/prod_post_trade.py --scope state
python quantpits/scripts/prod_post_trade.py --scope execution

# 指定结束日期
python quantpits/scripts/prod_post_trade.py --end-date 2026-02-10

# 详细输出：显示每笔交易明细
python quantpits/scripts/prod_post_trade.py --verbose
```

缺失交割单默认是错误，不再静默使用 `emp-table.xlsx`。确认某些交易日确实无活动时，可显式使用 `--allow-missing-settlement`；若 order/trade 已证明存在成交，该参数仍不能绕过一致性检查。

execution ingestion 使用 `post_trade_ingestion_state.json` 中的 source path + SHA-256 receipt，而不是最大日期游标，因此后到的旧日期文件仍会被发现。同一路径内容变化会 hard fail，避免更正文件与历史日志静默混合。

`--scope all` 使用两个独立窗口：settlement 只从 `prod_config.json` 的 state cursor 下一交易日开始，避免重复执行历史账户状态；order/trade 在未显式指定 `--start-date` 时扫描现存历史 export，并由 receipt 判断是否需要摄取。显式 `--start-date` 不允许早于 state cursor 的下一日；历史 state replay 需要后续专用 backfill 工具，不能借普通命令完成。

---

## 处理逻辑

对每个交易日，脚本按以下顺序处理：

```mermaid
flowchart TD
    A[加载交易文件] --> B[处理卖出]
    B --> C[处理买入]
    C --> D[处理红利/利息]
    D --> E[应用 Cashflow]
    E --> F[更新现金余额]
    F --> G[更新持仓]
    G --> H[获取收盘价]
    H --> I[计算浮盈]
    I --> J[写入各项资金与持仓日志]
    J --> K[执行交易分类分类匹配]
```

### 现金更新公式

```
cash_after = cash_before + 卖出收入 - 买入支出 + 红利利息 + cashflow
```

### 数据文件说明

| 文件 | 内容 | 更新方式 |
|------|------|----------|
| `trade_log_full.csv` | 全部交易记录 | 追加 + 去重 |
| `trade_classification.csv` | 核心量化/手工买卖归因打标 | 自动依赖建议文件推算 |
| `holding_log_full.csv` | 每日持仓快照 | 追加 + 去重 |
| `daily_amount_log_full.csv` | 每日资金汇总 | 追加 + 去重 |
| `trade_detail_*.csv` | 单日交易详情 | 每日覆写 |

### 券商交割单适配器 (Broker Adapter)

系统采用 **Broker Adapter** 架构处理不同券商由于导出的 Excel/CSV 表头和格式不同的问题。核心处理逻辑要求内部有一套标准化 Schema（包含 `证券代码`, `交易类别`, `资金发生数` 等标准中文字段）。

**内置适配器 (`brokers/`)：**
* `gtja`: 国泰君安交割单适配器（默认）。它负责处理前5行无用表头、剥离内部字符串的 `\t`，并直接沿用了原有的中文字段。

如果你需要接入新的券商，只需：
1. 在 `quantpits/scripts/brokers/` 下创建一个继承自 `BaseBrokerAdapter` 的新适配器类。
2. 实现 strict `parse_settlement` / `parse_orders` / `parse_trades`，将券商格式映射成标准 DataFrame；缺文件、解析失败和 schema 错误必须抛出 typed error。兼容 `read_*` 方法可保留 warning + empty 行为。
3. 在 `brokers/__init__.py` 的 `BROKER_REGISTRY` 中注册你的适配器。
4. 运行时加上 `--broker your_broker_name` 或在 `prod_config.json` 中配置 `"broker": "your_broker_name"`。

---

## 典型工作流

### 场景 1：例行处理

```bash
# 1. 将交易软件导出文件放入 data/ 目录，命名为 YYYY-MM-DD-table.xlsx
# 2. 如有出入金，编辑 config/cashflow.json
# 3. 运行脚本
python quantpits/scripts/prod_post_trade.py
```

### 场景 2：两次处理之间有多次出入金

```bash
# 编辑 cashflow.json，按日期填写每次出入金
cat config/cashflow.json
# {"cashflows": {"2026-02-03": 50000, "2026-02-06": -20000}}

# 预览确认
python quantpits/scripts/prod_post_trade.py --dry-run

# 确认无误后执行
python quantpits/scripts/prod_post_trade.py
```

### 场景 3：先预览再执行

```bash
# 严格预检：会打开并校验 Excel，但不写文件
python quantpits/scripts/prod_post_trade.py --dry-run

# 只查看轻量计划（不打开 Excel）
python quantpits/scripts/prod_post_trade.py --explain-plan

# 确认后实际运行
python quantpits/scripts/prod_post_trade.py
```

---

## 完整参数一览

```
python quantpits/scripts/prod_post_trade.py --help

可选参数:
  --scope {all,state,execution}
  --start-date TEXT 起始日期；state/all 不得早于 state cursor 的下一日
  --end-date TEXT   结束日期 (YYYY-MM-DD)，默认为今天
  --allow-missing-settlement  显式确认缺失交割单的日期无活动
  --dry-run         严格解析和完整性预检，不写入任何文件
  --explain-plan    轻量文本计划；不初始化 Qlib、不打开 Excel
  --json-plan       轻量 JSON 计划；stdout 只输出一个 JSON payload
  --broker TEXT     券商标识 (默认优先读取 prod_config.json 中的 broker，兜底为 gtja)
  --run-id TEXT     可选运行标识，不影响语义 fingerprint
  --verbose         详细输出每日交易明细
```

---

## 注意事项

> [!IMPORTANT]
> 本脚本**仅处理实盘交易数据**，与训练 (`static_train.py --full`)、预测 (`static_train.py --predict-only`)、回测 (`brute_force_ensemble.py`) 等模块完全独立，互不耦合。

> [!TIP]
> 建议先用 `--explain-plan` 查看轻量计划，再用 `--dry-run` 完成严格输入预检。

> [!WARNING]
> 三类文件必须严格命名为 `YYYY-MM-DD-table.xlsx`、`YYYY-MM-DD-order.xlsx`、`YYYY-MM-DD-trade.xlsx`。缺失 settlement 默认失败，不会隐式使用空模板；只有显式 `--allow-missing-settlement` 才能确认无活动。
