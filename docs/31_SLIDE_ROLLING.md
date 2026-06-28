# 31 · Slide 模式详解

> 滚动训练中最经典的模式：每个窗口内按 train → valid → test 连续三段进行训练和预测。

---

## 1. 窗口划分公式

设 $T = \text{rolling\_start}$，$\Delta = \text{test\_step}$，$X = \text{train\_years}$，$Y = \text{valid\_years}$，$A = \text{anchor\_date}$。

对于第 $w$ 个窗口（$w \ge 0$）：

$$
\begin{aligned}
\text{offset} &= \Delta \times w \\[4pt]
\text{train\_start} &= T + \text{offset} \\
\text{train\_end} &= \text{train\_start} + X\text{年} - 1\text{天} \\[4pt]
\text{valid\_start} &= \text{train\_end} + 1\text{天} \\
\text{valid\_end} &= \text{valid\_start} + Y\text{年} - 1\text{天} \\[4pt]
\text{test\_start} &= \text{valid\_end} + 1\text{天} \\
\text{test\_end} &= \min(\text{test\_start} + \Delta - 1\text{天},\; A)
\end{aligned}
$$

停止条件：$\text{test\_start} > A$。

**三段连续、绝对不重叠**：$\text{train\_end} + 1\text{天} = \text{valid\_start}$，$\text{valid\_end} + 1\text{天} = \text{test\_start}$。

---

## 2. 示例

配置：`rolling_start=2015-01-01, train_years=5, valid_years=1, test_step=3M, anchor_date=2024-12-31`

| w | Train | Valid | Test |
|:--|:------|:------|:-----|
| 0 | 2015-01-01 ~ 2019-12-31 | 2020-01-01 ~ 2020-12-31 | 2021-01-01 ~ 2021-03-31 |
| 1 | 2015-04-01 ~ 2020-03-31 | 2020-04-01 ~ 2021-03-31 | 2021-04-01 ~ 2021-06-30 |
| ... | ... | ... | ... |
| 14 | 2018-07-01 ~ 2023-06-30 | 2023-07-01 ~ 2024-06-30 | 2024-07-01 ~ 2024-09-30 |
| 15 | 2018-10-01 ~ 2023-09-30 | 2023-10-01 ~ 2024-09-30 | 2024-10-01 ~ 2024-12-31 |

共 16 个窗口。最后一个窗口的 `test_end` 截断至 `anchor_date`。

每个窗口训练 **1 个模型**。

---

## 3. Train → Today Gap

最后窗口（W15）：
- Train 结束于 2023-09-30
- Anchor 为 2024-12-31
- **Gap = valid（1年）+ test_step 尾部（~3月）≈ 15 个月**

---

## 4. 适用场景

| 适合 | 不太适合 |
|------|---------|
| 模型训练快（单 model/window） | 需要极致贴近最新市场 |
| 需要明确的 valid 段做早停 | Train 域相对较短 |
| 窗口数多但每窗口成本低 | — |

---

## 5. 与 CPCV 模式的关系

Slide 模式等价于 CPCV 的特殊情况：$K=1$ fold，valid = train 末尾 $Y$ 年。详见 [32 · CPCV 模式](32_CPCV_ROLLING.md)。
