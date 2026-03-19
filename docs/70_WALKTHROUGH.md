# QuantPits 完整实战操作手册

本文档是 QuantPits 系统的 **端到端实战演练指南**，从零开始逐步引导您完成：环境搭建 → 数据准备 → 模型训练 → 组合优选 → 融合预测 → 订单生成 → 实盘闭环。

> [!TIP]
> 本手册面向**首次接触的用户**，所有命令均可直接复制粘贴执行。各步骤的深入参数和原理请参阅对应的模块文档 (01-08)。

---

## 目录

1. [安装 Qlib 及依赖](#1-安装-qlib-及依赖)
2. [准备行情数据](#2-准备行情数据)
3. [初始化工作区](#3-初始化工作区)
4. [训练模型](#4-训练模型)
5. [仅预测（不重训）](#5-仅预测不重训)
6. [组合穷举优选](#6-组合穷举优选)
7. [融合预测](#7-融合预测)
8. [生成信号排名（可选）](#8-生成信号排名可选)
9. [处理实盘数据（Post-Trade）](#9-处理实盘数据post-trade)
10. [生成交易订单（Order Gen）](#10-生成交易订单order-gen)
11. [实盘分析](#11-实盘分析)

---

## 1. 安装 Qlib 及依赖

QuantPits 基于 [Microsoft Qlib](https://github.com/microsoft/qlib) 构建，需要先安装 Qlib 及本系统的依赖包。

### 1.1 安装 Qlib

> [!WARNING]
> Qlib 在 Windows 或某些 Python 版本下可能因 C++ 编译或 NumPy 版本冲突而安装失败。**强烈推荐使用Linux/WSL 环境)** 以获得最佳兼容性（本项目目前使用的是Python 3.12）。

```bash
pip install pyqlib
```

> 详细安装文档参考：https://github.com/microsoft/qlib#installation

### 1.2 下载项目与安装依赖

```bash
git clone https://github.com/DarkLink/QuantPits.git
cd QuantPits
pip install -r requirements.txt
```

### 1.3（可选）安装为 Python 包

```bash
pip install -e .
```

### 1.4（可选）安装 CuPy GPU 加速

> [!NOTE]
> CuPy 仅用于加速组合优选时的**暴力穷举**过程，与 Qlib 模型训练无关。模型训练本身的 GPU 依赖（如 LightGBM、CatBoost 等）需参考 Qlib 或对应算法的官方文档配置。

如果需要使用 GPU 加速暴力穷举组合（`brute_force_fast.py`），按 CUDA 版本安装 CuPy：

```bash
# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x
```

---

## 2. 准备行情数据

所有脚本默认从 `~/.qlib/qlib_data/cn_data` 读取 Qlib 日频行情数据。

### 方式一：Qlib 官方数据

```bash
python -m qlib.run.get_data qlib_data \
  --target_dir ~/.qlib/qlib_data/cn_data \
  --region cn \
  --version v2
```

> ⚠️ 初次下载需要十几 GB 硬盘空间，可能需要较长时间。

### 方式二：使用第三方数据源（推荐）

推荐使用 [investment_data](https://github.com/chenditc/investment_data) 项目，该项目会持续发布最新的 Qlib 格式行情数据。可通过以下命令直接下载并解压：

```bash
mkdir -p ~/.qlib/qlib_data/cn_data
# 使用 wget 下载最新版本数据
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
# 解压到数据目录
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
```

> 详见 https://github.com/chenditc/investment_data 了解数据更新频率和内容说明。

### 自定义数据路径

如果数据不在默认位置，可在工作区的 `run_env.sh` 中配置：

```bash
export QLIB_DATA_DIR="/your/custom/path/cn_data"
export QLIB_REGION="cn"
```
如果使用Windows系统的PowerShell，可在工作区的`run_env.ps1`中配置：

```bash
$env:QLIB_DATA_DIR = "X:/your/custom/path/cn_data"
$env:QLIB_REGION = "cn"
```
---

## 3. 初始化工作区

QuantPits 采用 **Workspace（工作区）** 机制实现配置与数据的完全隔离。系统自带 `Demo_Workspace` 作为模板。

> [!IMPORTANT]
> 每次执行训练、预测或实盘操作前，**必须先激活工作区**。如果重启终端，请务必重新执行 `source` 命令激活！

### 3.1 使用 Demo 工作区快速体验

```bash
# 确保当前位于项目根目录：
# cd QuantPits

source workspaces/Demo_Workspace/run_env.sh

# For Windows
. ./workspaces/Demo_Workspace/run_env.ps1
```

输出 `Workspace activated: .../Demo_Workspace` 表示激活成功。

**特别提醒：**在 Windows 上，如果你要让 PowerShell 执行 .ps1 脚本，如果这是你第一次运行脚本，PowerShell 可能会报“禁止执行脚本”的错误。此时你需要以**管理员身份**运行 PowerShell，并输入：
```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3.2 创建新工作区

```bash
python quantpits/tools/init_workspace.py \
  --source workspaces/Demo_Workspace \
  --target workspaces/MyWorkspace
```

该命令会：
- 克隆 `Demo_Workspace/config/` 下的全部配置文件
- 创建空的 `data/`、`output/`、`archive/`、`mlruns/` 目录
- 生成 `run_env.sh` 环境激活脚本

### 3.3 激活新工作区

```bash
source workspaces/MyWorkspace/run_env.sh

# For Windows
. ./workspaces/Demo_Workspace/run_env.ps1
```

### 3.4 工作区目录结构

```text
workspaces/MyWorkspace/
├── config/
│   ├── model_registry.yaml         # 模型注册表（启用哪些模型）
│   ├── model_config.json           # 训练配置（数据划片、频次等）
│   ├── strategy_config.yaml        # 策略配置（回测/订单生成参数）
│   ├── prod_config.json            # 生产配置（持仓、现金、实盘参数）
│   ├── cashflow.json               # 现金流（出入金记录）
│   ├── ensemble_config.json        # 融合组合配置
│   └── workflow_config_*.yaml      # Qlib 模型工作流配置
├── data/                           # 运行时数据
├── output/                         # 输出文件
├── archive/                        # 历史归档
├── mlruns/                         # MLflow 追踪
├── run_env.ps1                     # 环境激活脚本(Win)
└── run_env.sh                      # 环境激活脚本(Linux)
```

---

## 4. 训练模型

> [!IMPORTANT]
> **前提**：确保已激活工作区（`source workspaces/YourWorkspace/run_env.sh`），以下所有命令均在 QuantPits 根目录执行。

### 4.1 准备模型工作流 YAML

每个模型需要一个 Qlib 工作流配置文件，定义模型类、数据集处理器、训练方式等。

- YAML 模板来源：https://github.com/microsoft/qlib/tree/main/examples/benchmarks
- 放置位置：`config/workflow_config_<model_name>.yaml`

Demo 工作区已自带示例文件 `workflow_config_demo_weekly.yaml`（周频 LinearModel + Alpha158）。

### 4.2 准备配置文件

工作区 `config/` 下的核心配置文件：

| 文件 | 作用 | 示例值 |
|------|------|--------|
| `model_registry.yaml` | 模型注册表，定义启用哪些模型 | 每个模型需指定算法、数据集、YAML 文件、是否启用 |
| `model_config.json` | 训练参数，控制数据划片和频次 | 训练/验证/测试窗口、滑动/固定模式、`freq: "week"` |
| `strategy_config.yaml` | 策略参数，定义 TopK/DropN 和回测环境 | `topk: 20`, `n_drop: 3`, 手续费率 |
| `prod_config.json` | 实盘参数，记录当前持仓和现金 | 初始资金、当前持仓列表 |
| `cashflow.json` | 出入金记录，按日期的现金进出 | `{"cashflows": {"2026-01-15": 50000}}` |

### 4.3 运行训练

#### 全量训练（首次必须）

训练 `model_registry.yaml` 中所有 `enabled: true` 的模型：

```bash
python quantpits/scripts/static_train.py --full
```

#### 增量训练（按需）

只训练指定模型，已有记录保持不变：

```bash
# 按名称指定
python quantpits/scripts/static_train.py --models demo_linear_Alpha158

# 按标签筛选
python quantpits/scripts/static_train.py --tag tree

# 所有 enabled 模型（merge 模式，不覆写）
python quantpits/scripts/static_train.py --all-enabled

# 预览训练计划（不实际执行）
python quantpits/scripts/static_train.py --models demo_linear_Alpha158 --dry-run
```

训练结束后生成的核心文件：
- `latest_train_records.json` — 模型训练记录（下游所有脚本的输入）
- `output/model_performance_<date>.json` — 模型 IC/ICIR 指标

> [!NOTE]
> 文件命名中的 `<date>` 为**交易日/预测日**（而非运行脚本的真实系统时间），方便您在复盘历史数据时按日期定位。

> 详见 [01_TRAINING_GUIDE.md](01_TRAINING_GUIDE.md)

---

## 5. 仅预测（不重训）

已有训练好的模型后，数据更新时无需重训，直接用已有模型预测新数据：

```bash
# 预测所有 enabled 模型
python quantpits/scripts/static_train.py --predict-only --all-enabled

# 只预测指定模型
python quantpits/scripts/static_train.py --predict-only --models demo_linear_Alpha158

# 预览计划
python quantpits/scripts/static_train.py --predict-only --all-enabled --dry-run
```

预测结果会自动保存到 `output/predictions/`，同时以 Merge 方式更新 `latest_train_records.json`。

> 详见 [05_PREDICT_ONLY_GUIDE.md](05_PREDICT_ONLY_GUIDE.md)

---

## 6. 组合穷举优选

当模型数量 ≥ 2 时，可通过暴力穷举找到最优模型组合。

> [!WARNING]
> **注意**：穷举和融合需要至少 **2 个**已训练的模型。如果使用默认的 Demo 工作区（仅有 1 个模型），请先在 `model_registry.yaml` 中启用并训练其他模型再执行此步骤。

### 6.1 准备组合分组文件（可选）

如果模型数量较多（15+），建议使用分组穷举以减少组合数。创建 `config/combo_groups.yaml`：

```yaml
groups:
  LSTM_variants:
    - lstm_Alpha360
    - alstm_Alpha158
  Tree_models:
    - lightgbm_Alpha158
    - catboost_Alpha158
```

### 6.2 运行穷举

#### 快速穷举（向量化，秒级，适合粗筛）

```bash
# 快速穷举（最多 3 个模型组合）
python quantpits/scripts/brute_force_fast.py --max-combo-size 3

# 完整穷举
python quantpits/scripts/brute_force_fast.py

# 带防过拟合的 OOS 验证（推荐！）
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 10

# 使用分组穷举
python quantpits/scripts/brute_force_fast.py --use-groups

# GPU 加速
python quantpits/scripts/brute_force_fast.py --use-gpu
```

> ⚠️ **快速版精度有限**（不含涨跌停过滤、精确持仓管理），适合初筛使用。

#### 精确穷举（Qlib 完整回测，分钟级）

```bash
# 精确穷举 Top 候选（建议先用快速版筛选后再精确验证）
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

# 从中断处继续
python quantpits/scripts/brute_force_ensemble.py --resume
```

#### 推荐工作流

1. **粗筛**：`brute_force_fast.py` 跑完所有组合（秒/分钟级）
2. **精确验证**：将 Top 10-20 组合用 `brute_force_ensemble.py` 精确验证
3. **确认**：查看 `output/brute_force*/analysis_report_*.txt` 选定最优组合

> 详见 [02_BRUTE_FORCE_GUIDE.md](02_BRUTE_FORCE_GUIDE.md)

---

## 7. 融合预测

使用选定的模型组合进行融合预测和回测。

### 7.1 准备融合组合配置

编辑 `config/ensemble_config.json`，定义多个 combo（选定合适的模型组合）：

```json
{
  "combos": {
    "combo_A": {
      "models": ["gru", "linear_Alpha158", "TabNet_Alpha158"],
      "method": "equal",
      "default": true,
      "description": "三模型等权组合"
    },
    "combo_B": {
      "models": ["gru", "linear_Alpha158", "alstm_Alpha158"],
      "method": "icir_weighted",
      "default": false,
      "description": "三模型 ICIR 加权"
    }
  },
  "min_model_ic": 0.00
}
```

### 7.2 运行融合

```bash
# 运行所有 combo 并生成跨组合对比（最常用）
python quantpits/scripts/ensemble_fusion.py --from-config-all

# 只运行 default combo
python quantpits/scripts/ensemble_fusion.py --from-config

# 直接指定模型（不使用配置文件）
python quantpits/scripts/ensemble_fusion.py \
  --models gru,linear_Alpha158,alstm_Alpha158

# OOS 验证（仅在最近 1 年数据上测试）
python quantpits/scripts/ensemble_fusion.py --from-config --only-last-years 1
```

输出文件：
- `output/ensemble/combo_comparison_<date>.csv` — 跨组合对比表
- `output/ensemble/combo_comparison_<date>.png` — 净值对比图

> 详见 [03_ENSEMBLE_FUSION_GUIDE.md](03_ENSEMBLE_FUSION_GUIDE.md)

---

## 8. 生成信号排名（可选）

将融合预测分数归一化为 -100 ~ +100 的推荐指数，适合分享给他人参考：

```bash
# 为所有 combo 生成 Top 300 排名
python quantpits/scripts/signal_ranking.py --all-combos

# 仅 default combo
python quantpits/scripts/signal_ranking.py

# 自定义 Top N
python quantpits/scripts/signal_ranking.py --top-n 500
```

输出保存在 `output/ranking/Signal_<combo>_<date>_Top300.csv`。

> 详见 [07_SIGNAL_RANKING_GUIDE.md](07_SIGNAL_RANKING_GUIDE.md)

---

## 9. 处理实盘数据（Post-Trade）

> 以下步骤 9-11 仅在有实盘交易时执行（目前示例为国泰君安系统的实盘交易数据）。

Post-Trade 脚本处理交易软件导出的成交记录，更新持仓和现金。**与训练/预测/融合模块完全解耦。**

### 9.1 准备交易数据

1. 从交易软件导出成交记录（`.xlsx`），放入 `data/` 目录
2. 文件命名格式：`YYYY-MM-DD-table.xlsx`（如 `2026-02-24-table.xlsx`）
3. 无交易日使用空模板 `emp-table.xlsx` (全空的Excel文件)

### 9.2 配置出入金（如有）

编辑 `config/cashflow.json`：

```json
{
  "cashflows": {
    "2026-02-03": 50000,
    "2026-02-06": -20000
  }
}
```

### 9.3 运行 Post-Trade

```bash
# 预览处理计划（强烈建议先预览）
python quantpits/scripts/prod_post_trade.py --dry-run

# 正式处理
python quantpits/scripts/prod_post_trade.py

# 指定券商（默认国泰君安）
python quantpits/scripts/prod_post_trade.py --broker gtja

# 详细输出
python quantpits/scripts/prod_post_trade.py --verbose
```

处理后更新：
- `config/prod_config.json` — 最新持仓和现金余额
- `data/trade_log_full.csv` — 累计交易日志
- `data/holding_log_full.csv` — 累计持仓日志
- `data/daily_amount_log_full.csv` — 每日资金汇总

> 详见 [04_POST_TRADE_GUIDE.md](04_POST_TRADE_GUIDE.md)

---

## 10. 生成交易订单（Order Gen）

基于融合预测 + 当前持仓，生成本期的买卖订单建议。

```bash
# 使用 ensemble 融合预测（最常用）
python quantpits/scripts/order_gen.py

# 先预览，查看多模型判断表
python quantpits/scripts/order_gen.py --dry-run --verbose

# 使用单模型预测
python quantpits/scripts/order_gen.py --model gru

# 生成多模型排名可视化
python quantpits/tools/plot_model_opinions.py
```

输出文件：
- `output/buy_suggestion_<source>_<date>.csv` — 买入建议
- `output/sell_suggestion_<source>_<date>.csv` — 卖出建议
- `output/model_opinions_<date>.csv` — 多模型 BUY/SELL/HOLD 判断表

> 详见 [06_ORDER_GEN_GUIDE.md](06_ORDER_GEN_GUIDE.md)

---

## 11. 实盘分析

### 11.1 综合分析报告

对模型质量、融合相关性、执行滑点、组合风险进行全面审查：

```bash
python quantpits/scripts/run_analysis.py \
  --models gru_Alpha158 transformer_Alpha360 TabNet_Alpha158 sfm_Alpha360 \
  --output output/analysis_report.md
```

### 11.2 交互式可视化看板

```bash
# 宏观资产组合评估面板
streamlit run ui/dashboard.py

# 滚动策略健康监测面板
streamlit run ui/rolling_dashboard.py --server.port 8503
```

### 11.3 滚动异常体检

```bash
# 生成底层窗口参数数据
python quantpits/scripts/run_rolling_analysis.py --windows 20 60

# 生成自动化健康报告
python quantpits/scripts/run_rolling_health_report.py
```

> 详见 [08_ANALYSIS_GUIDE.md](08_ANALYSIS_GUIDE.md)

---

## 附录：典型操作场景速查

### 场景 A：首次完整运行

> [!WARNING]
> 步骤 ② 和 ③ 的穷举融合需要至少训练 2 个以上的模型。如果在 Demo 工作区下测试，请先修改 `model_registry.yaml` 开启额外模型。

```bash
# 确保当前位于项目根目录：
# cd QuantPits
source workspaces/Demo_Workspace/run_env.sh

# ① 全量训练
python quantpits/scripts/static_train.py --full

# ② 快速穷举找组合
python quantpits/scripts/brute_force_fast.py --max-combo-size 3

# ③ 融合预测
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

### 场景 B：日常最小闭环（不重训）

```bash
cd QuantPits
source workspaces/MyWorkspace/run_env.sh

# ① 预测
python quantpits/scripts/static_train.py --predict-only --all-enabled

# ② 融合
python quantpits/scripts/ensemble_fusion.py --from-config-all

# ③ Post-Trade（如有实盘）
python quantpits/scripts/prod_post_trade.py

# ④ 订单生成
python quantpits/scripts/order_gen.py
```

> 也可以使用 Makefile 一键执行日常流程（预测 → 融合 → Post-Trade → 订单）：
> ```bash
> make run-daily-pipeline  # (仅限 Linux/macOS/WSL)
> ```

### 场景 C：重新评估模型组合

```bash
# ① 用已有模型预测
python quantpits/scripts/static_train.py --predict-only --all-enabled

# ② 快速穷举（带 OOS 验证）
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 10

# ③ 精确验证 Top 候选
python quantpits/scripts/brute_force_ensemble.py --min-combo-size 3 --max-combo-size 3

# ④ 更新 ensemble_config.json，选新组合融合
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

### 场景 D：模型效果下降，回退重训

```bash
# ① 增量训练某些模型
python quantpits/scripts/static_train.py --models gru,alstm_Alpha158

# ② 用新模型重新融合
python quantpits/scripts/ensemble_fusion.py --from-config-all

# ③ Post-Trade + 订单生成
python quantpits/scripts/prod_post_trade.py
python quantpits/scripts/order_gen.py
```

---

## 附录：关键参考资源

| 资源 | 链接 |
|------|------|
| Qlib 官方仓库 | https://github.com/microsoft/qlib |
| Qlib 模型 Benchmark YAML | https://github.com/microsoft/qlib/tree/main/examples/benchmarks |
| 行情数据源 (investment_data) | https://github.com/chenditc/investment_data |
| QuantPits 系统总览文档 | [00_SYSTEM_OVERVIEW.md](00_SYSTEM_OVERVIEW.md) |
| QuantPits 模块文档 (01-08) | `docs/` 目录 |
