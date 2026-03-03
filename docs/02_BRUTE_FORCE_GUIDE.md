# 暴力穷举组合回测指南

将所有模型的预测进行暴力组合穷举，对每个组合做等权融合回测，找到最优模型组合。

## 快速开始

```bash
cd /path/to/QuantPits

# 快速测试（最多 3 个模型的组合，约 175 次回测）
python quantpits/scripts/brute_force_ensemble.py --max-combo-size 3

# 完整穷举（10 个模型 = 1023 个组合，耗时较长）
python quantpits/scripts/brute_force_ensemble.py

# 仅分析已有结果（不重新跑回测）
python quantpits/scripts/brute_force_ensemble.py --analysis-only

# 从上次中断处继续（Ctrl+C 中断或崩溃后均可）
python quantpits/scripts/brute_force_ensemble.py --resume

# 使用模型分组穷举（每组只选一个，大幅减少组合数）
python quantpits/scripts/brute_force_ensemble.py --use-groups

# 指定自定义分组 + 减少线程数控制内存
python quantpits/scripts/brute_force_ensemble.py --use-groups --group-config config/my_groups.yaml --n-jobs 2
```

## 脚本流程

### Stage 1 — 加载预测数据
- 读取 `latest_train_records.json` 获取实验名和模型列表
- 从 Qlib Recorder 加载每个模型的 `pred.pkl` 预测值
- 按天做 Z-Score 归一化

### Stage 2 — 相关性分析
- 计算预测值相关性矩阵
- 保存 CSV 供后续分析

### Stage 3 — 暴力穷举回测
- 生成所有模型组合（从 1 个到 N 个）
- 对每个组合：等权融合 → TopkDropoutStrategy → Qlib 回测
- 提取指标：年化收益、最大回撤、Calmar、超额收益等
- 支持 `--resume` 从已有 CSV 继续

### Stage 4 — 结果分析
- **Top 组合展示**：按年化超额排序 Top 20、最稳健 Top 10
- **模型归因**：统计 Top/Bottom N 中各模型出现频率，计算净胜分
- **相关性 vs 绩效**：计算多样性红利，找低相关性高 Calmar 的"黄金组合"
- **层次聚类**：基于超额收益相关性做 Ward 聚类
- **权重优化**：对 Top 10 单模型做 Max Sharpe / Risk Parity 优化对比
- **综合报告**：自动输出最佳组合、MVP 核心模型

## 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--record-file` | `latest_train_records.json` | 训练记录文件 |
| `--max-combo-size` | `0` (全部) | 最大组合模型数（分组模式下指选几个组） |
| `--min-combo-size` | `1` | 最小组合模型数（分组模式下指选几个组） |
| `--freq` | `None` | 回测交易频率 (`day` / `week`)，默认从 `strategy_config.yaml` 读取 |
| `--top-n` | `50` | 分析时 Top/Bottom N |
| `--output-dir` | `output/brute_force` | 输出目录 |
| `--resume` | - | 从已有 CSV 继续（支持崩溃/中断后恢复） |
| `--skip-analysis` | - | 跳过分析阶段 |
| `--analysis-only` | - | 仅分析已有结果 |
| `--n-jobs` | `4` | 并发回测线程数 |
| `--batch-size` | `50` | 每批处理组合数（影响 checkpoint 粒度和内存） |
| `--use-groups` | - | 启用分组穷举模式（每组只选一个模型） |
| `--group-config` | `config/combo_groups.yaml` | 分组配置文件路径 |

## 输出文件

所有输出保存在 `output/brute_force/` 目录下：

```
output/brute_force/
├── correlation_matrix_{date}.csv     # 预测值相关性矩阵
├── brute_force_results_{date}.csv    # 回测结果（核心文件）
├── model_attribution_{date}.csv      # 模型归因表
├── model_attribution_{date}.png      # 归因条形图
├── risk_return_scatter_{date}.png    # 风险-收益散点图
├── cluster_dendrogram_{date}.png     # 层次聚类图
├── optimization_weights_{date}.csv   # 优化权重
├── optimization_equity_{date}.png    # 优化策略净值曲线
└── analysis_report_{date}.txt        # 综合分析报告
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

### 更细粒度的归因分析
```bash
# Top/Bottom 100 的归因
python quantpits/scripts/brute_force_ensemble.py --analysis-only --top-n 100
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
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 10

# 使用 GPU 加速
python quantpits/scripts/brute_force_fast.py --use-gpu

# 仅分析已有结果
python quantpits/scripts/brute_force_fast.py --analysis-only

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

> 其他参数（`--max-combo-size`, `--resume`, `--analysis-only`, `--use-groups`, `--group-config`, `--exclude-last-years`, `--auto-test-top` 等）与原版相同。

---

## 🕰️ 高级技巧：防止 In-Sample 过拟合

在机器寻找最佳组合时，极其容易出现 **In-Sample 过拟合** (选出的组合刚好适应了当前整个回测期，未来表现拉胯)。

为了解决这个问题，`brute_force_ensemble.py` 和 `brute_force_fast.py` 支持**动态相对时间窗口切分**，允许将最新的预测数据留作 **Out-Of-Sample (OOS) 测试集**：

### OOS 验证工作流

```bash
# 1. 寻找组合并立即在 OOS 数据上验证
# --exclude-last-years 1: 把最近的 1 年数据剔除，仅使用前 2 年作为 In-Sample 寻找组合
# --auto-test-top 5: 在被剔除的最近 1 年数据上，验证选出的 Top 5 组合
python quantpits/scripts/brute_force_fast.py --exclude-last-years 1 --auto-test-top 5
```

执行后，除了生成正常的 IS 报告外，还会在终端输出 `Stage 5: 自动 Out-Of-Sample (OOS) 验证 (Top 5)`，展示这几个最佳组合在从未见过的新数据上的真实表现。

### 日期参数说明

| 参数 | 说明 |
|------|------|
| `--exclude-last-years N` | 在寻找组合时，剔除最近 N 年的数据作为 OOS 集。 |
| `--exclude-last-months N` | 同上，剔除最近 N 个月的数据。 |
| `--auto-test-top N` | 选择 Top N 在 OOS 集上执行真实回测并生成评估报告 (`oos_validation_{date}.csv`)。|
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

输出保存在 `output/brute_force_fast/`（与原版目录分开）：

```
output/brute_force_fast/
├── correlation_matrix_{date}.csv          # 预测值相关性矩阵
├── brute_force_fast_results_{date}.csv    # 回测结果（核心文件）
├── model_attribution_{date}.csv           # 模型归因表
├── model_attribution_{date}.png           # 归因条形图
├── risk_return_scatter_{date}.png         # 风险-收益散点图
├── optimization_weights_{date}.csv        # 优化权重
└── analysis_report_fast_{date}.txt        # 综合分析报告
```
