# QuantPits 量化交易系统

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Qlib](https://img.shields.io/badge/Tech_Stack-Qlib-brightgreen.svg)

基于 [Microsoft Qlib](https://github.com/microsoft/qlib) 构建的先进、生产级别的量化交易系统。本系统提供了一个用于支持周频及日频交易的完整端到端流水道，核心特点包括高度模块化架构、多实例隔离运行（Workspace 机制）、模型融合（Ensemble）、执行归因分析以及全交互式的可视化数据面板。

🌐 [English Version (README.md)](./README.md)

> **注意：** 欢迎各种形式的贡献！如果您发现 Bug 或有功能建议，请随时提交 Issue 或 Pull Request。

## 🚀 核心特性

* **多工作区（Workspace）级隔离**：能够为不同的市场（如沪深300、中证500）或不同的策略配置拉起独立的“交易控制台”，无需复制系统底层核心代码。
* **组件化流水线**：
  - **训练与预测**：完全支持多模型的全量训练与增量训练更新（包含 LSTM, GRU, Transformers, LightGBM, GATs 等）。
  - **暴力回测与融合框架**：内置基于 CuPy 显存加速演算的高性能组合暴力穷举寻优，以及智能化的信号融合架构。
  - **订单与实盘执行**：利用系统内置的 TopK/DropN 原生逻辑自动生成可操作的买卖订单，并在事前和事后全面分析微观执行摩擦损耗（包含价差滑点、持仓延时成本等）。
* **全息化监测面板**：系统内置两大原生交互式 `streamlit` 数据看板，分别用于追踪“宏观资产组合表现”与“微观滚动策略健康状态监控”。
* **高可用基础设施**：自动历史备份检查点、由原生 JSON 承载的模型注册表状态追踪、完整规范的日内/周频执行日志。

## 📂 架构总览

系统在底层结构上严格将 **引擎逻辑（Engine 代码端）** 与 **隔离工作区（Workspace 配置及数据端）** 实现物理剥离：

```text
QuantPits/
├── docs/                   # 详细的系统开发及应用操作手册（00-08）
├── ui/                     # 交互式数据图表面板
│   ├── dashboard.py        # 宏观资管业绩评估 Streamlit 面板
│   └── rolling_dashboard.py# 时序策略执行健康监测 Streamlit 面板
├── quantpits/              # 核心逻辑及执行引擎组件
│   └── scripts/            # Pipeline 流水线脚本矩阵
│
└── workspaces/             # 隔离式的实盘配置存储区
    └── Demo_Workspace/     # 示范性的可配置交易运行库
        ├── config/         # 交易边界约束、模型注册表、出入金路由
        ├── data/           # 订单簿记录、持仓簿流转、单日资金盘口
        ├── output/         # 单一预测结果、融合模型阵列结果、评估报告
        ├── mlruns/         # MLflow 追踪日志
        └── run_env.sh      # 工作区安全隔离的环境激活脚本
```

## 🛠️ 快速启动指南

### 1. 依赖安装

请确保当前环境内具备基础的 **Qlib** 安装生态。引擎官方支持 Python 3.8 至 3.12。随后加载并安装额外依赖：

```bash
pip install -r requirements.txt
# (可选) 将 engine 作为一个系统级包安装以获得全局可见性:
pip install -e .
```

*(注意: 针对采用 GPU 硬加速的穷举排列组合模块，需视您本地电脑的 CUDA 版本决定安装 `cupy-cuda11x` 或 `cupy-cuda12x`)*

### 2. 准备底层行情数据

在执行后续流程之前，请确保您已经在本地准备好了 Qlib 所需的日频或高频特征数据（通常放在 `~/.qlib/qlib_data/cn_data`）：

```bash
# 示例：下载中国市场的 1D 日频数据
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
```

> **注意：** 该数据集包含海量历史行情，初次下载可能需要占用十几 GB 的硬盘空间和较长的一段时间。请耐心等待。
请确保您配置的 Workspace 数据源路径能够准确命中该目录。

### 3. 激活工作区

所有的模块运行都要求显式地具备明确且已激活的隔离工作区上下文。系统内置了一套完整的 `Demo_Workspace` 供基础调试：

```bash
cd QuantPits/
source workspaces/Demo_Workspace/run_env.sh
```

### 4. 主动式流水线运行

环境挂载完毕后，您可以直接按顺序触发引擎脚本执行基本的生产流程逻辑循环（或直接在仓库根目录执行 `make run-daily-pipeline`）：

```bash
# 0. 更新每日市场数据
# 注意：本引擎假设 Qlib 底层数据已由外部 Cron 任务定时更新完毕。
# 如果未更新，请在此步骤前优先更新。

# 1. 全量训练模型（首次运行或需要刷新模型时必做）
python -m quantpits.scripts.prod_train_predict

# 2. 使用所有已使能的模型触发全量增量预测推断
python -m quantpits.scripts.prod_predict_only --all-enabled

# 3. 调用当前库表配置好的融合配比组合完成多维度参数预测网格
python -m quantpits.scripts.ensemble_fusion --from-config-all

# 4. 处理回溯实盘执行状态变更（Post-Trade 落单归档）
python -m quantpits.scripts.prod_post_trade

# 5. 根据当前最新的组合建议及最新持仓执行全新订单信号推演
python -m quantpits.scripts.order_gen
```

### 5. 驱动可视化数据面板

渲染查看已激活工作区内的深度分析数据图层：

```bash
# 资产组合执行及持仓情况综合评估面板
streamlit run ui/dashboard.py

# 时序策略微观执行损耗及因子漂移监测面板
streamlit run ui/rolling_dashboard.py
```

## 🏗️ 创设新实例工作区

如果您希望针对截然不同的标的物池（如建立一个专注于中证500的实例分支），可以使用自带的脚手架指令：

```bash
python -m quantpits.scripts.init_workspace \
  --source workspaces/Demo_Workspace \
  --target workspaces/CSI500_Base
```

这条指令将会无损克隆整个源配置体系，并从零开始建立全新的空 `data/`、`output/` 以及独立映射的 `mlruns/` 沙盒阵列，完全杜绝交叉污染干扰已存在的部署实例。创设后仅需调用常规命令 `source workspaces/CSI500_Base/run_env.sh` 即可登入此全新库。

## 📖 深度说明文档

如需从零剖析具体各个计算节点以及架构组件的底层原理与完整参数，请前往 `docs/` 阅读系统手册：
- `00_SYSTEM_OVERVIEW.md` (系统架构部署与流水线总览)
- `01_TRAINING_GUIDE.md` (全量训练及模型配置向导)
- `02_BRUTE_FORCE_GUIDE.md` (穷举回测及GPU加速矩阵操作向导)
- `03_ENSEMBLE_FUSION_GUIDE.md` ...以此类推。

所有文档均已提供中文与纯正的英文(`en/`)双语版本支持。

## 📜 授权协议
MIT License.
