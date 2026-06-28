# 32 · CPCV Mode (Walk-Forward CPCV)

> CPCV rolling embeds Purged Cross-Validation inside each rolling window. The rolling loop defines test boundaries; CPCV performs pure K-fold CV on the train domain only.

## 1. Why `n_test_groups = 0`

The original design (CPCV with internal test set inside rolling) suffered from Test Set Inception:

| Problem | Cause |
|---------|-------|
| Overlapping test sets | Rolling step=3M but CPCV test≈6mo |
| Shrinking tail | Windows near anchor get compressed, purge destroys training |
| Confused responsibilities | Two layers each define their own test set |

**Fix**: Rolling defines test boundaries. CPCV does CV on train domain only. `n_test_groups` forced to 0.

## 2. Window Formula

For window $w$ ($w \ge 0$):

$$
\begin{aligned}
\text{offset} &= \Delta \times w \\[4pt]
\text{test\_start} &= T + \text{offset} + X\text{yr} \\
\text{test\_end} &= \min(\text{test\_start} + \Delta - 1\text{d},\; A) \\[4pt]
\text{train\_end} &= \text{test\_start} - 1\text{d} \\
\text{train\_start} &= \text{train\_end} - X\text{yr} + 1\text{d}
\end{aligned}
$$

Train domain is always exactly $X$ years. Test is exactly $\Delta$. No overlap, no skipping.

## 3. CPCV Fold Structure

On $[\text{train\_start}, \text{train\_end}]$ ($X$ years):

| Param | Meaning |
|-------|---------|
| $G = \text{n\_groups}$ | Equal-period groups |
| $V = \text{n\_val\_groups}$ | Validation groups per fold |
| $K = G - V + 1$ | Number of folds |
| purge | Periods removed on both sides of validation |
| embargo | Extra delay after validation (right side only) |

For fold $f \in [0, K-1]$:

| Segment | Range |
|---------|-------|
| **Validation** | Groups $[f, f+V-1]$ |
| **Left Train** | Groups $[0, f-1]$, trimmed by purge on right edge |
| **Right Train** | Groups $[f+V, G-1]$, trimmed by purge+embargo on left edge |

$K$ folds ensemble → predict $[\text{test\_start}, \text{test\_end}]$.

## 4. Example

Config: `rolling_start=2015-01-01, train_years=5, test_step=3M, n_groups=10, n_val_groups=1, purge=3, embargo=5, anchor_date=2024-12-31`

| w | Train Domain (5yr, 10 folds) | Test (3M, non-overlapping) |
|:--|:--|:--|
| 0 | 2015-01-01 ~ 2019-12-31 | 2020-01-01 ~ 2020-03-31 |
| 15 | 2018-10-01 ~ 2023-09-30 | 2023-10-01 ~ 2024-12-31 |

16 windows, 10 folds each.

### Window 0 Folds

| Fold | Valid | Right Train includes Group 9? |
|:-----|:------|:------------------------------|
| 0 | Group 0 | Yes |
| 1-8 | Groups 1-8 | Yes |
| 9 | Group 9 | No (last group is validation) |

> Folds 0-8 have right training reaching to `train_end - purge` — very close to test_start.

## 5. Gap

Last window (W15): 9/10 folds have training to `train_end - purge` ≈ 2023-09-07. Anchor is 2024-12-31. **Weighted gap ≈ 16 months**.

| Mode | Gap |
|------|-----|
| Static slide | ~4 years |
| Rolling + slide | ~15 months |
| **Rolling + CPCV** | **~16 months** (but training includes post-validation data) |

## 6. Parameter Constraints (Fail-Fast)

| Tier | Condition | Action |
|------|-----------|--------|
| Error | `purge + embargo ≥ 80% × group_size` | `ValueError` — reject |
| Warning | `purge + embargo ≥ 50% × group_size` | `UserWarning` |

| train_years | n_groups | Group ≈ | Safe purge+embargo |
|-------------|----------|---------|---------------------|
| 5 | 10 | ~26 wks | ≤ 20 (rec. 3+5=8) |
| 3 | 10 | ~15 wks | ≤ 12 (rec. 2+3=5) |

## 7. Compute Cost

$K = G - V + 1$ folds per window. With $G=10, V=1$: $K=10$.

- $N$ windows × $K$ folds trainings per model
- Per-fold GPU cleanup between folds
- `--cache-size` enables DataHandler reuse across folds within a window

## 8. Relationship to Slide

Slide is CPCV with $K=1$, valid = last $Y$ years. Both share the same window formula when `valid_years=0`.
