# 信号排名使用指南

## 概览

`scripts/signal_ranking.py` 将融合预测分数归一化为 -100 ~ +100 的推荐指数，生成 Top N 排名 CSV。

**适合分享给他人作为参考** — 独立于订单生成，不依赖持仓信息。

**工作流位置**: 融合回测 → **信号排名（本步）**

---

## 快速开始

```bash
cd QuantPits

# 1. 为 default combo 生成 Top 300
python quantpits/scripts/signal_ranking.py

# 2. 为所有 combo 各生成一份
python quantpits/scripts/signal_ranking.py --all-combos

# 3. 为指定 combo 生成
python quantpits/scripts/signal_ranking.py --combo combo_A

# 4. 自定义 Top N
python quantpits/scripts/signal_ranking.py --top-n 500

# 5. 指定预测文件
python quantpits/scripts/signal_ranking.py --prediction-file output/predictions/ensemble_2026-02-13.csv
```

---

## 完整参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--combo` | 无 | 指定 combo 名称 |
| `--all-combos` | false | 为所有 combo 各生成一份 |
| `--prediction-file` | 无 | 直接指定预测文件路径 |
| `--top-n` | 300 | 输出 Top N 个标的 |
| `--output-dir` | `output/ranking` | 输出目录 |
| `--dry-run` | false | 仅打印，不写入文件 |

---

## 信号评分逻辑

```
1. 加载融合预测 CSV（score 列）
2. 取最新日期数据
3. 归一化: signal = (score - min) / (max - min) * 200 - 100
   → 分数范围: -100（最弱）~ +100（最强）
4. 按推荐指数降序排名，取 Top N
5. 输出 CSV: 股票代码, 推荐指数
```

---

## 输出文件

```
output/ranking/
├── Signal_default_2026-02-13_Top300.csv     # default combo
├── Signal_combo_A_2026-02-13_Top300.csv     # combo_A
└── Signal_combo_B_2026-02-13_Top300.csv     # combo_B
```

### 文件格式

| 列 | 说明 |
|------|------|
| `股票代码` | 标的代码（index） |
| `推荐指数` | -100 ~ +100 归一化分数，2位小数 |

---

## 典型工作流

```bash
# Step 1: 运行融合回测
python quantpits/scripts/ensemble_fusion.py --from-config-all

# Step 2: 生成所有 combo 的信号排名
python quantpits/scripts/signal_ranking.py --all-combos

# Step 3: 分享 CSV 给他人
ls output/ranking/
```

---

## 前置条件

> [!IMPORTANT]
> 需要先运行 `ensemble_fusion.py` 生成融合预测 CSV。
> 预测文件应位于 `output/predictions/` 目录。

---

## 与其他脚本的关系

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `ensemble_fusion.py` | 融合回测 | 选定模型 | 融合预测 CSV |
| **`signal_ranking.py`** | **信号排名** | **融合预测** | **Top N 排名 CSV** |
| `order_gen.py` | 生成订单 | 融合预测 + 持仓 | 买卖建议 |
