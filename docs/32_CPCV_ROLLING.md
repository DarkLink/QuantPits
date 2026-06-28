# 32 · CPCV 模式详解（Walk-Forward CPCV）

> CPCV 滚动模式将 Purged Cross-Validation 嵌入每个滚动窗口。Rolling 大循环定义 Test 边界，CPCV 只对 Train 域做纯粹的 K-fold 交叉验证。

---

## 1. 设计原理：为什么 `n_test_groups = 0`

### 旧设计的问题（Test Set Inception）

如果把 CPCV 原样塞进 Rolling（保留内部 test set），会出现：

| 问题 | 原因 |
|------|------|
| 测试集重叠 | Rolling test_step=3M，CPCV 内部 test_set≈6月 → 相邻窗口预测重叠 |
| 尾部压缩 | 窗口越接近 anchor_date，时间范围越短，purge 吞噬训练段 → 跳过 |
| 职责混淆 | CPCV 和 Rolling 各定义一套 test set，互相干扰 |

### 修正方案

**Rolling 大循环定义 Test 边界，CPCV 只对 Train 域做交叉验证。`n_test_groups` 强制为 0。**

---

## 2. 窗口划分公式

对于第 $w$ 个窗口（$w \ge 0$）：

$$
\begin{aligned}
\text{offset} &= \Delta \times w \\[4pt]
\text{test\_start} &= T + \text{offset} + X\text{年} \\
\text{test\_end} &= \min(\text{test\_start} + \Delta - 1\text{天},\; A) \\[4pt]
\text{train\_end} &= \text{test\_start} - 1\text{天} \\
\text{train\_start} &= \text{train\_end} - X\text{年} + 1\text{天}
\end{aligned}
$$

停止条件：$\text{test\_start} > A$。

**Train 域始终严格 $X$ 年长，Test 域严格 $\Delta$ 长，无重叠，无跳过。**

---

## 3. CPCV Fold 结构

在 $[\text{train\_start}, \text{train\_end}]$ 这个 $X$ 年的区间上：

| 参数 | 含义 |
|------|------|
| $G = \text{n\_groups}$ | Train 域等周期分组数 |
| $V = \text{n\_val\_groups}$ | 每 Fold 验证组数 |
| $K = G - V + 1$ | 折数 |
| $\text{purge}$ | 验证集两侧各移除 N 个交易周 |
| $\text{embargo}$ | 验证集之后额外延迟 N 交易周（仅影响 Right Train） |

对于 fold $f \in [0, K-1]$：

| 段 | 范围 |
|----|------|
| **Validation** | 组 $[f,\; f+V-1]$ |
| **Left Train** | 组 $[0,\; f-1]$，右边界削去 $\text{purge}$ |
| **Right Train** | 组 $[f+V,\; G-1]$，左边界削去 $\text{purge} + \text{embargo}$ |

$K$ 折 Ensemble → 预测 $[\text{test\_start}, \text{test\_end}]$。

---

## 4. 完整示例

配置：`rolling_start=2015-01-01, train_years=5, test_step=3M, n_groups=10, n_val_groups=1, purge=3, embargo=5, anchor_date=2024-12-31`

### 窗口总览

| w | Train Domain（5年, 10折） | Test（3M, 不重叠） |
|:--|:--|:--|
| 0 | 2015-01-01 ~ 2019-12-31 | 2020-01-01 ~ 2020-03-31 |
| 1 | 2015-04-01 ~ 2020-03-31 | 2020-04-01 ~ 2020-06-30 |
| ... | ... | ... |
| 14 | 2018-07-01 ~ 2023-06-30 | 2023-07-01 ~ 2023-09-30 |
| 15 | 2018-10-01 ~ 2023-09-30 | 2023-10-01 ~ 2024-12-31 |

共 16 个窗口，每个窗口 10 折。最后一个窗口 test_end 截断至 anchor_date。

### Window 0 的 10 个 Fold

Train 域 = [2015-01-01, 2019-12-31]，约 260 交易周，分 10 组，每组约 26 周。

| Fold | Valid | Left Train | Right Train |
|:-----|:------|:-----------|:------------|
| 0 | Group 0 | — | Groups 1-9 (经 purge+embargo 削边) |
| 1 | Group 1 | Group 0 (削边) | Groups 2-9 (削边) |
| ... | ... | ... | ... |
| 8 | Group 8 | Groups 0-7 (削边) | Group 9 (削边) |
| 9 | Group 9 | Groups 0-8 (削边) | — |

> Fold 0~8 的 Right Train 包含 Group 9（Train 域最后 1 组）→ 训练数据紧贴 test_start。

---

## 5. Train → Today Gap

最后窗口（W15）：

- Train 域结束于 2023-09-30
- Test = [2023-10-01, 2024-12-31]
- Fold 0~8（9 折）Right Train 包含最后 1 组 → 训练数据至 `train_end - purge` ≈ 2023-09-07
- **这 9 折的训练数据距 anchor ≈ test（15月）+ purge（3周）≈ 15.7 个月**
- Fold 9 不含最后一段，但仅占 ensemble 1/10
- **加权 Gap ≈ 16 个月**

> 如果缩短 test_step（如 1M）或增大 train_years，gap 可进一步压缩。

| 模式 | Gap（示例配置） |
|------|----------------|
| 静态 slide | ~4 年 |
| CPCV 单次 | ~2 年 |
| 滚动 + slide | ~15 月 |
| **滚动 + CPCV** | **~16 月**（但训练含验证期之后的近期数据） |

> CPCV 的核心优势不仅是 gap 数值，更是 Right Train 覆盖了 slide 模式中完全盲区的"验证期之后"的数据。

---

## 6. 参数约束（Fail-Fast）

虽然 Train 域固定 $X$ 年，但不合理的 purge/embargo 仍会吞噬训练段。

### 两级校验

| 层级 | 条件 | 行为 |
|------|------|------|
| Error | `purge + embargo ≥ 80% × group_size` | `ValueError`，阻止启动 |
| Warning | `purge + embargo ≥ 50% × group_size` | `UserWarning` |

### 安全参考

| train_years | n_groups | 每组约 | 安全 purge+embargo |
|-------------|----------|--------|-------------------|
| 5 | 10 | ~26 周 | ≤ 20 (推荐 3+5=8) |
| 5 | 8 | ~32 周 | ≤ 25 |
| 3 | 10 | ~15 周 | ≤ 12 (推荐 2+3=5) |
| 3 | 8 | ~19 周 | ≤ 15 |

---

## 7. 计算成本

每窗口训练 $K = G - V + 1$ 个 fold。以 $G=10, V=1$ 为例，$K=10$。

- $N$ 窗口 × $K$ fold = $N \times K$ 次训练/模型
- 每 fold 间 GPU 清理（`torch.cuda.empty_cache()` + `gc.collect()`）
- 使用 `--cache-size` 可在单窗口内复用 DataHandler，减少 K-fold 间重复加载

---

## 8. 与 Slide 模式的统一

Slide 等价于 CPCV 的特殊情况：

| 参数 | Slide | CPCV |
|------|-------|------|
| `train_start` | $T + w\Delta$ | $T + w\Delta$ |
| `train_end` | $T + w\Delta + X\text{年} - 1\text{天}$ | $\text{test\_start} - 1\text{天}$ |
| `test_start` | $\text{train\_end} + Y\text{年} + 1\text{天}$ | $\text{train\_end} + 1\text{天}$ |
| Validation | 独立 valid 段（$Y$ 年） | 嵌入 K-fold（每 fold 1 group） |
| 每窗口训练次数 | 1 | $K$ |

Slide 中 $Y$ 年的 valid 段 ≈ CPCV 中 $K$ 折的 ensemble validation。Slide 是 CPCV 在 $K=1$ 时的退化。
