# 组合搜索组合回测指南

将所有模型的预测进行组合搜索，对每个组合做等权融合回测，找到最优模型组合。

## 快速开始

```bash
cd /path/to/QuantPits

# 快速测试（最多 3 个模型的组合，约 175 次回测）
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

# 全量搜索（10 个模型 = 1023 个组合，耗时较长）
python quantpits/scripts/brute_force_ensemble.py

# 使用 Z-Score 归一化（保留模型"确定度"信息，交集融合）
python quantpits/scripts/brute_force_ensemble.py --norm-method zscore


# 从上次中断处继续（Ctrl+C 中断或崩溃后均可）
python quantpits/scripts/brute_force_ensemble.py --resume

# 使用模型分组搜索（每组只选一个，大幅减少组合数）
python quantpits/scripts/brute_force_ensemble.py --use-groups

# 指定自定义分组 + 减少线程数控制内存
python quantpits/scripts/brute_force_ensemble.py --use-groups --group-config config/my_groups.yaml --n-jobs 2
```

## 脚本流程

### Stage 1 — 加载预测数据
- 读取 `latest_train_records.json` 获取实验名和模型列表
- 从 Qlib Recorder 加载每个模型的 `pred.pkl` 预测值
- 按天做截面归一化（默认 `rank`: percentile rank → [0,1]; 可选 `zscore`）
- rank 模式下，模型未覆盖的股票填充 0.5（视为弃权/中性），融合时使用**并集**股票池

### Stage 2 — 相关性分析
- 计算预测值相关性矩阵
- 保存 CSV 供后续分析

### Stage 3 — 组合搜索回测
- 生成所有模型组合（从 1 个到 N 个）
- 对每个组合：等权融合 → TopkDropoutStrategy → Qlib 回测
- 提取指标：年化收益、最大回撤、Calmar、超额收益等
- 支持 `--resume` 从已有 CSV 继续

### Stage 4 — 元数据导出
- 搜索结束后，将在 `output/ensemble_runs/{script}_{date}/` 目录下生成 `run_metadata.json`，供独立分析脚本消费。

### Stage 5 — 独立多维分析与 OOS 验证
- 搜索本质上是 In-Sample (IS) 寻优。所有分析、挑选和 Out-Of-Sample (OOS) 验证逻辑解耦到了独立的 `analyze_ensembles.py` 脚本中。
- 通过运行 `python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_<date>/run_metadata.json`，系统将基于 Yield, Robustness, MVP 等多维度构建候选池，并在 OOS 数据上自动且无偏地打分排名。
- 分析完成后，在 run 目录内自动生成 `summary.md` 一页纸摘要，包含 IS/OOS 关键指标和文件清单。

> [!NOTE]
> **关于单模型表现与融合回测的评测差异说明**
>
> 融合与搜索脚本在评估模型表现时，引入了严格的 **Z-Score 归一化**（Z-Score Normalization）和 **数据对齐**（Data Alignment）处理，因此由于 TopK 截断的存在，单模型在此处的回测结果可能与训练期间通过 `run_analysis.py` 查看到的原始预测分值回测结果存在合理且微小的差异：
> 1. **独立归一化隔离**：每个模型的预测分值会首先仅基于自身非为空的预测股票池进行按天的 Z-Score 归一化处理。这保证了模型之间的评分尺度统一，且某个模型的数据缺失不会影响并在归一化前污染其他模型的分布。
> 2. **延迟交集对齐**：仅在计算最终特定组合的均值或加权打分时，系统才会对当前组合涉及的模型取交集（即执行 `dropna(how='any')`），这避免了无关模型的数据缺失引发当前组合评测池的不当缩水。
> 3. **评估排名的对齐**：所有提供参照的基准数据（如单模型的历史排行榜回测指标）均会严格根据当前评价矩阵实际生成的时间窗口进行动态切片对齐，从而为您提供“同时间段”的一致性比对。

## 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--training-mode` | `static` | 限定模型训练模式（如 `static` 或 `rolling`） |
| `--record-file` | `latest_train_records.json` | 指定训练记录文件 |
| `--max-combo-size` | `0` (全部) | 最大组合模型数（分组模式下指选几个组） |
| `--min-combo-size` | `1` | 最小组合模型数（分组模式下指选几个组） |
| `--freq` | `None` | 回测交易频率 (`day` / `week`)，默认从 `strategy_config.yaml` 读取 |
| `--top-n` | `50` | 分析时 Top/Bottom N |
| `--output-dir` | `output/ensemble_runs` | **输出根目录**（每次运行自动在其下创建 `{script}_{date}/` 子目录） |
| `--resume` | - | 从已有 CSV 继续（支持崩溃/中断后恢复） |
| `--n-jobs` | `4` | 并发回测线程数 |
| `--batch-size` | `50` | 每批处理组合数（影响 checkpoint 粒度和内存） |
| `--use-groups` | - | 启用分组搜索模式（每组只选一个模型） |
| `--group-config` | `config/combo_groups.yaml` | 分组配置文件路径 |
| `--norm-method` | `rank` | 截面归一化方法: `rank` (percentile rank [0,1]，推荐) 或 `zscore` |

## 输出文件

所有输出按**每次运行**存储在独立子目录下，搜索过程（IS）和 OOS 验证产出物分层组织：

```
output/ensemble_runs/
└── brute_force_2026-04-03/         # 每次运行独立目录（script + anchor_date）
    ├── run_metadata.json            # 运行配置及日期切分（供分析脚本使用）
    ├── summary.md                   # 一页纸汇总：IS/OOS 关键指标 + 文件清单
    ├── is/                          # IS（样本内）产出物
    │   ├── results.csv              # 回测结果（核心文件）
    │   ├── correlation_matrix.csv   # 预测值相关性矩阵
    │   ├── analysis_report.txt      # IS 综合评估报告
    │   ├── model_attribution.csv    # 模型归因表
    │   ├── model_attribution.png    # 归因条形图
    │   ├── risk_return_scatter.png  # 风险-收益散点图
    │   └── cluster_dendrogram.png   # 层次聚类图
    └── oos/                         # OOS（样本外）产出物
        ├── oos_multi_analysis.csv   # 多维候选池 OOS 评测结果
        ├── oos_report.txt           # OOS 综合分析报告
        └── oos_risk_return.png      # OOS 风险-收益散点图
```

## 运行时间估计

以 10 个模型为例（共 1023 个组合）：

| 模型组合大小 | 组合数量 | 预估耗时 |
|:---:|:---:|:---:|
| 1~3 | 175 | ~15 分钟 |
| 1~5 | 637 | ~50 分钟 |
| 1~10 (全部) | 1023 | ~1.5 小时 |

> **建议**：首次使用 `--max-combo-size 3` 快速验证，确认无误后再跑完整穷举。

## 进阶用法

### 仅看特定大小的组合
```bash
# 只看 4-6 个模型的组合
python quantpits/scripts/brute_force_ensemble.py --min-combo-size 4 --max-combo-size 6
```

### 日频回测
```bash
python quantpits/scripts/brute_force_ensemble.py --freq day
```


## 断点续跑与安全中断

脚本支持 **增量 checkpoint** 和 **信号安全中断**，确保长时间穷举任务不会丢失进度。

### 工作原理

- **分批执行**：将所有组合分为 `--batch-size` 大小的批次，每批完成后立刻追加写入 CSV
- **信号处理**：注册 `SIGINT` (Ctrl+C) / `SIGTERM` handler，收到信号后：
  1. 等待当前批次中已提交的回测任务完成
  2. 将已完成的结果保存到 CSV
  3. 打印中断提示并安全退出
  4. 再次按 Ctrl+C 可强制退出
- **Resume**：`--resume` 读取已有 CSV 自动跳过已完成组合

### 使用方式

```bash
# 启动穷举
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 5

# 运行中按 Ctrl+C → 当前批次完成后安全退出
# 输出: "⚠️ 已安全中断！已完成: X/Y 组合，使用 --resume 继续"

# 从中断处继续
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 5 --resume
```

### 内存控制

| 参数 | 作用 |
|------|------|
| `--batch-size` | 减小批次大小可降低内存峰值（默认 50） |
| `--n-jobs` | 减少线程数可降低并发内存占用（建议 2-4） |

> 每个批次完成后会自动调用 `gc.collect()` 释放中间对象。

---

## 模型分组穷举

当模型数较多时（15+ 个），全穷举的组合数爆炸（32767+）。分组穷举模式将模型按**人工定义的组别**分组，每组只选一个模型参与组合，大幅减少有效组合数。

### 配置文件

分组配置在 `config/combo_groups.yaml`，格式如下：

```yaml
groups:
  LSTM_variants:           # 组名（自定义）
    - lstm_Alpha360        # 组内模型（与 train_records 中的 key 一致）
    - alstm_Alpha158
    - alstm_Alpha360

  Tree_models:
    - lightgbm_Alpha158
    - catboost_Alpha158

  # ... 更多组 ...
```

**规则**：
- 组名自定义，仅用于展示
- 开启分组模式时，**未出现在任何组中的模型将被排除**（不自动参与）
- 此文件与 `model_registry.yaml` 完全独立，不自动关联 tags
- `--min/max-combo-size` 在分组模式下控制**选几个组参与**

### 使用方式

```bash
# 使用默认分组配置
python quantpits/scripts/brute_force_ensemble.py --use-groups

# 指定自定义分组
python quantpits/scripts/brute_force_ensemble.py --use-groups --group-config config/custom_groups.yaml

# 分组 + 只看 3~4 个组的组合
python quantpits/scripts/brute_force_ensemble.py --use-groups --min-combo-size 3 --max-combo-size 4
```

### 组合数对比示例

| 场景 | 模型数 | 全穷举组合数 | 分组穷举组合数 |
|:---:|:---:|:---:|:---:|
| 15 个模型, 6 组 × 2~3 个 | 15 | 32767 | ~500 |
| 10 个模型, 5 组 × 2 个 | 10 | 1023 | ~62 |

> **提示**：`brute_force_fast.py` (快速模式) 也完全支持上述 `--use-groups` 和 `--group-config` 分组穷举功能，用法一致。

---

## ⚡ 快速模式 (`brute_force_fast.py`)

当模型数量较多（>10）、组合数量巨大时，原版 qlib 回测非常慢。`brute_force_fast.py` 使用 **NumPy/CuPy 向量化矩阵运算** 替代 qlib 的逐单模拟，速度提升 **~5000 倍**。

### 快速开始

```bash
cd /path/to/QuantPits

# 快速测试
python quantpits/scripts/brute_force_fast.py --max-combo-size 3

# 完整穷举
python quantpits/scripts/brute_force_fast.py

# ================================
# 防止过拟合的高阶按时间窗口测试 (推荐!)
# 在除最后 1 年外的所有数据上寻找最佳组合 (In-Sample)
# 并在最后 1 年的数据上自动验证前 10 名组合 (Out-Of-Sample)
# ================================
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1

# 2. 调用独立的分析脚本进行多维筛选和自动 OOS 验证
python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_<上次运行的anchor_date>/run_metadata.json

# 使用 GPU 加速
python quantpits/scripts/brute_force_fast.py --use-gpu


# 从上次中断处继续
python quantpits/scripts/brute_force_fast.py --resume

# 使用模型分组穷举加速测试
python quantpits/scripts/brute_force_fast.py --use-groups --group-config config/combo_groups_20.yaml
```

### 精度差异

| 特性 | 原版 (qlib backtest) | 快速版 (vectorized) |
|------|:---:|:---:|
| 涨跌停过滤 | ✅ | ❌ |
| 交易费用 | ✅ 精确  | ⚠️ 按换手率估算 |
| TopK+Dropout | ✅ 完整 | ⚠️ 仅 TopK |
| 资金管理 | ✅ 完整 | ❌ 假设全仓等权 |
| 速度 | ~5s/combo | ~0.001s/combo |

> **注意精度差异**：快速版（fast）和正式穷举会有很大的差异（因为 fast 无法顺利处理持仓，对于非全仓交换的会有很大误差）。请谨慎使用。

### 推荐工作流

1. **粗筛**：`python quantpits/scripts/brute_force_fast.py` 跑完所有组合（分钟级）
2. **精确验证**：将快速版 Top 10-20 组合，用原版 `brute_force_ensemble.py` 精确回测
3. **融合**：选定组合后用 `ensemble_fusion.py` 进行正式融合

### 快速版专有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | `512` | 批量处理大小 |
| `--use-gpu` | - | 强制使用 GPU (需要 CuPy) |
| `--no-gpu` | - | 强制禁用 GPU |
| `--cost-rate` | `0.002` | 单次换手交易费用率 (双边 0.2%) |

> 其他参数（`--max-combo-size`, `--resume`, `--use-groups`, `--group-config`, `--exclude-last-years` 等）与原版相同。

---

## 🕰️ 高级技巧：防止 In-Sample 过拟合

在机器寻找最佳组合时，极其容易出现 **In-Sample 过拟合** (选出的组合刚好适应了当前整个回测期，未来表现拉胯)。

为了解决这个问题，`brute_force_ensemble.py` 和 `brute_force_fast.py` 支持**动态相对时间窗口切分**，允许将最新的预测数据留作 **Out-Of-Sample (OOS) 测试集**：

### OOS 验证工作流

```bash
# 1. 寻找组合并立即在 OOS 数据上验证
# --exclude-last-years 1: 把最近的 1 年数据剔除，仅使用前 2 年作为 In-Sample 寻找组合
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1

# 2. 运行结束后，脚本会打印出 metadata 路径，例如：
#   ✅ 元数据已保存: output/ensemble_runs/brute_force_fast_2026-04-03/run_metadata.json
#   请使用以下命令进行分析与 OOS 验证:
#     python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_2026-04-03/run_metadata.json

# 3. 调用独立的分析脚本，在被剔除的最近 1 年 OOS 数据上自动验证模型池
python quantpits/scripts/analyze_ensembles.py --metadata output/ensemble_runs/brute_force_fast_<date>/run_metadata.json
```

执行后，穷举结束后，通过调用 `analyze_ensembles.py` 喂入刚才生成的元数据 JSON，系统将自动构建多维候选池（Yield, Robustness 等），并在被剔除的 OOS 数据上展开真实回测，生成散点图与 OOS 评价报告。

### 分析与评估产出物 (Analysis Artifacts)

运行上述分析脚本后，系统将在 run 目录下的 `is/` 和 `oos/` 子目录中生成详尽的评估文件，并在根目录自动生成 `summary.md` 一页纸摘要：
- **IS 综合评估报告 (`is/analysis_report.txt`)**：包含各策略候选池（Yield, MVP, Diversity）在 IS 阶段的年化收益、最大回撤、Calmar 等表现。
- **风险收益全景散点图 (`is/risk_return_scatter.png`)**：展示 IS 阶段所有组合的风险-收益二维全景以及相关性拟合。
- **内部聚合特征树状图 (`is/cluster_dendrogram.png`)**：基于模型间预测相关性绘制的 Ward 聚类分析图，识别模型同质化。
- **模型归因重要度 (`is/model_attribution.png`)**：基于最优与最底组合进行归因频率统计，发现高价值基础模型。
- **OOS 综合报告 (`oos/oos_report.txt`)**：多维候选池在 OOS 独立集中的真实盲测结果。
- **OOS 验证散点图 (`oos/oos_risk_return.png`)**：多维候选池在 OOS 集中的风险-收益绘图追踪。
- **汇总摘要 (`summary.md`)**：IS/OOS 关键指标 + 运行配置 + 文件清单，可直接在 VS Code 预览。

### 分析脚本参数说明 (Analyzer Parameters)

针对独立运行的 `analyze_ensembles.py` 脚本，你还可以追加下列高级指令控制候选池颗粒度：

| 参数 | 说明 |
|------|------|
| `--top-n N` | 设定每个维度（Yield, MVP, Defensive 等）默认提取 Top N 组合进入 OOS (默认: 5) |
| `--top-n-yield`, `--top-n-robust`, 等 | 单独覆盖某一特定派系的 Top N 提取数量。支持的覆盖项包括： `-yield`, `-robust`, `-defensive`, `-mvp`, `-diversity`。 |
| `--training-mode MODE` | 在评估 OOS 之前，硬性过滤所有组合，仅保留所有成员均匹配给定触发模式（如 `static` 或 `incremental`）的组合进行打分。 |
| `--max-workers N` | OOS 回测时的并发线程数，默认为 4。如果候选池极大，可适当调高加速验证过程。 |

### 日期参数说明

| 参数 | 说明 |
|------|------|
| `--exclude-last-years N` | 在寻找组合时，剔除最近 N 年的数据作为 OOS 集。 |
| `--exclude-last-months N` | 同上，剔除最近 N 个月的数据。 |
| `--start-date YYYY-MM-DD` | 绝对日期过滤：强行指定 IS 阶段的最早开始日期。 |
| `--end-date YYYY-MM-DD` | 绝对日期过滤：强行截断，即不使用该日期之后的数据。 |

> 注：因为预测数据是随时间滚动（按交易频次新增）的，推荐使用 `exclude-last-years` 或 `exclude-last-months` 动态保持始终剥离最新的样本，避免每次手动修改绝对日期。

---

### GPU 加速

安装 CuPy 即可启用 GPU 加速（自动检测）：

```bash
# CUDA 12.x
pip install cupy-cuda11x # 或者安装 cupy-cuda12x，取决于你的 CUDA 版本

# CUDA 11.x
pip install cupy-cuda11x
```

### 运行时间估计 (快速模式)

| 模型组合大小 | 组合数量 | CPU 预估耗时 | GPU 预估耗时 |
|:---:|:---:|:---:|:---:|
| 1~3 | 175 | ~2 秒 | ~1 秒 |
| 1~5 | 637 | ~5 秒 | ~2 秒 |
| 1~10 (全部) | 1023 | ~10 秒 | ~3 秒 |
| 1~15 | 32767 | ~5 分钟 | ~1 分钟 |

### 输出文件

快速版产出物与标准版结构一致，存储在 `output/ensemble_runs/brute_force_fast_{date}/`：

```
output/ensemble_runs/brute_force_fast_2026-04-03/
├── run_metadata.json               # 运行配置及参数（供独立分析脚本使用）
├── summary.md                      # 一页纸汇总（analyze_ensembles 生成后填充）
├── is/
│   ├── results.csv                 # 回测结果（核心文件）
│   ├── correlation_matrix.csv      # 预测值相关性矩阵
│   ├── analysis_report.txt         # IS 综合评估报告
│   ├── model_attribution.csv/.png  # 模型归因
│   ├── risk_return_scatter.png     # IS 散点图
│   └── cluster_dendrogram.png      # 聚类图
└── oos/
    ├── oos_multi_analysis.csv       # 【分析脚本生成】多维候选池 OOS 评测结果
    ├── oos_risk_return.png          # 【分析脚本生成】OOS 风险-收益散点图
    └── oos_report.txt               # 【分析脚本生成】OOS 综合分析报告
```

> **迁移旧数据**：如需将 `output/brute_force/` 和 `output/brute_force_fast/` 中的旧文件迁移到新结构，可使用：
>
> ```bash
> # 预览
> python quantpits/scripts/migrate_ensemble_outputs.py --dry-run
> 
> # 执行（原目录不删除，确认后可手动清理）
> python quantpits/scripts/migrate_ensemble_outputs.py
> ```

---

## 下一步：配置融合组合

组合搜索和 OOS 验证完成后，进入正式的融合预测阶段：

### 步骤 1：查看分析报告

打开本次运行目录下的报告，确认候选组合：

```bash
# 一页纸摘要（IS/OOS 排名 + 文件清单）
cat output/ensemble_runs/<run_dir>/summary.md

# 或用浏览器打开 HTML 版（表格更完整）
# output/ensemble_runs/<run_dir>/summary.html

# OOS 详细验证报告
cat output/ensemble_runs/<run_dir>/oos/oos_report.txt
```

### 步骤 2：将选定组合写入配置

从报告的 `models` 列中复制组合成员，编辑 `config/ensemble_config.json`：

```json
{
  "combos": {
    "combo_A": {
      "models": ["gru", "linear_Alpha158", "TabNet_Alpha158"],
      "method": "equal",
      "default": true,
      "description": "高 Calmar 组合（来自 OOS 验证）"
    },
    "combo_B": {
      "models": ["alstm_Alpha158", "linear_Alpha158", "sfm_Alpha360"],
      "method": "icir_weighted",
      "default": false,
      "description": "低相关性多样化组合"
    }
  },
  "min_model_ic": 0.00
}
```

> [!TIP]
> 建议配置 2-3 个候选组合，涵盖不同维度（如高收益、高稳健、低相关性）。
> 选好的组合可持续使用多个周期，无需每次重新穷举。

### 步骤 3：运行正式融合回测

```bash
# 运行所有 combo 并生成跨组合对比，最终确认 default
python quantpits/scripts/ensemble_fusion.py --from-config-all
```

> 详细的配置格式和融合回测说明，请参阅 [03_ENSEMBLE_FUSION_GUIDE](03_ENSEMBLE_FUSION_GUIDE.md)。

---

## 已知限制 (Future Work)

### 模型覆盖股票数不一致时的 rank 粒度差异

在 rank 模式下，覆盖 300 只股票的模型 rank 步长为 `1/299`，而覆盖 500 只股票的模型步长为 `1/499`。粗粒度模型的信号会被细粒度模型的"量化噪声"稀释。将来可考虑按覆盖股票数对模型加权。

### NaN → 0.5 的"弃权"假设

rank 模式下，模型未覆盖的股票填充 0.5（中性分）。这假设所有模型对未覆盖股票的"不确定度"相同。当模型覆盖有系统性偏差时（例如某模型只覆盖大市值股票），对小票填充 0.5 可能过度乐观或悲观。

### Fast 路径 NaN 处理不一致

`brute_force_fast.py` 的 `load_predictions` 在合并时提前执行全局 `.dropna()`（交集），而标准路径 (`predict_utils.py`) 保留并集、仅对缺失填 0.5。同一 combo 在不同路径下的结果不可直接对比。
