# 31 · Slide Mode

> Classic rolling: each window uses contiguous train → valid → test segments.

## 1. Window Formula

Let $T = \text{rolling\_start}$, $\Delta = \text{test\_step}$, $X = \text{train\_years}$, $Y = \text{valid\_years}$, $A = \text{anchor\_date}$.

For window $w$ ($w \ge 0$):

$$
\begin{aligned}
\text{offset} &= \Delta \times w \\[4pt]
\text{train\_start} &= T + \text{offset} \\
\text{train\_end} &= \text{train\_start} + X\text{yr} - 1\text{d} \\[4pt]
\text{valid\_start} &= \text{train\_end} + 1\text{d} \\
\text{valid\_end} &= \text{valid\_start} + Y\text{yr} - 1\text{d} \\[4pt]
\text{test\_start} &= \text{valid\_end} + 1\text{d} \\
\text{test\_end} &= \min(\text{test\_start} + \Delta - 1\text{d},\; A)
\end{aligned}
$$

Stop when $\text{test\_start} > A$. Segments are contiguous with no overlap.

## 2. Example

Config: `rolling_start=2015-01-01, train_years=5, valid_years=1, test_step=3M, anchor_date=2024-12-31`

| w | Train | Valid | Test |
|:--|:------|:------|:-----|
| 0 | 2015-01-01 ~ 2019-12-31 | 2020-01-01 ~ 2020-12-31 | 2021-01-01 ~ 2021-03-31 |
| ... | ... | ... | ... |
| 15 | 2018-10-01 ~ 2023-09-30 | 2023-10-01 ~ 2024-09-30 | 2024-10-01 ~ 2024-12-31 |

16 windows total. 1 model per window.

## 3. Gap

Last window (W15): train ends 2023-09-30, anchor is 2024-12-31. **Gap ≈ 15 months**.

## 4. Use Cases

| Good fit | Less fit |
|----------|----------|
| Fast training (1 model/window) | Need extreme recency |
| Explicit validation for early stopping | Short train domain |

## 5. Relationship to CPCV

Slide is CPCV with $K=1$ fold and valid = last $Y$ years. See [32 · CPCV Mode](32_CPCV_ROLLING.md).
