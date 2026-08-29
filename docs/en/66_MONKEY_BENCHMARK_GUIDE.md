# Random Monkey Benchmark Guide

The Random Monkey Benchmark system uses Monte Carlo simulation and hypothesis testing to rigorously answer a core quantitative question: **"Can our models statistically beat random stock-picking monkeys? Does performance come from genuine predictive Alpha or simply from portfolio execution rules?"**

---

## Quickstart

```bash
cd /path/to/QuantPits
source workspaces/<workspace_name>/run_env.sh

# 1. Default run: 1,000 trials, auto-loads configuration from strategy_config.yaml, plots charts
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --plot

# 2. Fast mode: 5,000 trials, IC / ICIR / Rank IC only (skips backtest, finishes in seconds)
python quantpits/scripts/monkey_benchmark.py --n-trials 5000 --ic-only

# 3. Model comparison: Compute empirical p-values against specific trained models
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --models <model_1>,<model_2>

# 4. Custom strategy parameters (e.g. testing Buy and Hold mode)
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --topk 20 --n-drop 0

# 5. Full equal-weighted market universe benchmark (TopK set larger than constituent count, e.g. 9999)
python quantpits/scripts/monkey_benchmark.py --n-trials 10 --topk 9999
```

---

## Architecture & Two-Layer Design

The system provides a two-layer architecture for both **single-model pipeline integration** and **large-scale statistical distribution testing**:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 1: RandomModel (Qlib-Compatible Model Class)                    │
│  • Inherits from Qlib Model; fit() is no-op, predict() returns random  │
│  • Registered in model_registry.yaml; supports static_train / ensemble │
│  • Use case: Single baseline run, ensemble fusion noise-rejection test │
└────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 2: monkey_benchmark.py (Vectorized Monte Carlo Statistical Engine│
│  • Data & features loaded once with minimal preparation overhead       │
│  • Loops N parallel trials to construct empirical distributions        │
│  • Strictly aligned with TopkDropout (TopK + DropN) & dynamic universe │
│  • Outputs IC/ICIR, CAGR, Sharpe, Max DD distributions and p-values    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Core Mechanisms & Safeguards

### 1. Dynamic Universe Masking
* **Context**: Index constituents rotate over time. Across a long historical window, the cumulative set of historical tickers can be large, but on any specific trading day $t$, only $N_t$ active constituents exist.
* **Isolation**: The engine queries Qlib's dynamic market filter to extract active constituents on date $t$. Non-constituent scores are masked to $-\infty$:
  * Monkeys **cannot** select out-of-universe tickers.
  * Delisted or rotated-out stocks are forcibly dropped on the next rebalance.
  * Eliminates the zero-return dilution distortion where inactive historical tickers dilute portfolio returns.

### 2. Exact TopkDropout (TopK + DropN) State Machine
Live production strategies do not dump 100% of holdings every rebalance period (which would incur excessive turnover and friction). The engine precisely reproduces the `TopkDropout` state machine:

| Parameter | Mode | Execution Behavior |
|---|---|---|
| **`n_drop = 0`** | **Buy and Hold (Zero Dropout)** | Buys TopK stocks on Day 0 and **holds without active turnover** (turnover = 0, except forced out-of-universe replacement). |
| **`0 < n_drop < top_k`** (e.g. `n_drop = 3`) | **TopkDropout (Production Default)** | Only drops the worst $N_{\text{drop}}$ ranked holdings per period and replaces them with top unheld candidates; turnover is bounded around $N_{\text{drop}}/K$. |
| **`n_drop >= top_k`** or unconfigured | **Full Rebalance** | Re-ranks and replaces all non-TopK holdings each period. |

### 3. Benchmark Alignment (Forward 1-Day)
* Automatically reads the configured benchmark from `model_config.json`.
* Both portfolio holdings and benchmark returns are aligned forward 1-day ($t$ close to $t+1$ close), ensuring strict temporal consistency without lookahead bias.

---

## Report & Statistical Interpretation

Running the script outputs three comprehensive analytical sections:

### 1. Monte Carlo Distribution
Displays empirical mean, standard deviation, 90% confidence interval, and extreme ranges across $N$ trials:

```text
📊 [Monkey Benchmark Distribution (Monte Carlo)]
                          Metric          Mean ± Std       Median  90% Conf Interval (5%~95%)     Extreme Range (Min~Max)
                  IC (Pearson) -0.0000 ± +0.0035 +0.0000 [-0.0055, +0.0055] [-0.0110, +0.0110]
                ICIR (Pearson) -0.0000 ± +0.0600 +0.0000 [-0.0950, +0.0950] [-0.1900, +0.1900]
            Rank IC (Spearman) -0.0000 ± +0.0035 +0.0000 [-0.0055, +0.0055] [-0.0110, +0.0110]
          Rank ICIR (Spearman) -0.0000 ± +0.0600 +0.0000 [-0.0950, +0.0950] [-0.1900, +0.1900]
        Annual Return (CAGR)   +1.50% ± +7.50%  +1.50% [-10.50%, +13.50%] [-15.00%, +20.00%]
         Total Return (Total)   +1.40% ± +7.20%  +1.40% [-10.00%, +13.00%] [-14.50%, +19.50%]
                Sharpe Ratio +0.1500 ± +0.4500 +0.1500 [-0.5500, +0.7500] [-0.8500, +1.2000]
           Max Drawdown (MDD) -13.50% ± +3.50% -12.50% [-19.50%, -9.50%]  [-24.00%, -7.00%]
     Annual Excess (vs Bench)   -4.00% ± +7.50%  -4.00% [-16.50%, +8.00%]  [-20.50%, +15.50%]
```

### 2. Hypothesis Testing (Real Models vs Monkeys)
Computes empirical percentile rank and one-sided $p\text{-value}$ for each trained model:

$$p\text{-value} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Metric}_{\text{Monkey}, i} \ge \text{Metric}_{\text{Model}})$$

```text
🎯 [Hypothesis Testing: Real Models vs Monkey Benchmark]
Model Name               IC     ICIR  Rank IC  IC %ile   p-value Significance Verdict
model_candidate_a   +0.0400  +0.5000  +0.0380     99.8%   0.0020  ✅ Highly Significant (p<0.01)
model_candidate_b   +0.0350  +0.4500  +0.0330     99.4%   0.0060  ✅ Highly Significant (p<0.01)
baseline_linear     +0.0120  +0.1500  +0.0110     95.2%   0.0480  ✅ Significant (p<0.05)
weak_candidate      +0.0010  +0.0150  +0.0010     60.0%   0.4000  ❌ Failed to Beat Monkey (p>=0.05)
```

### 3. Visualizations
With `--plot`, generated figures are saved to `output/monkey_benchmark/`:
* `monkey_ic_distribution.png`: Pearson IC histogram with model vertical lines and 90% confidence bands.
* `monkey_rank_ic_distribution.png`: Spearman Rank IC histogram.
* `monkey_ann_ret_distribution.png`: Annualized return distribution.
* `monkey_sharpe_distribution.png`: Sharpe ratio distribution.
* `monkey_max_dd_distribution.png`: Maximum drawdown distribution.
* `monkey_trials_distribution.csv`: Raw trial-by-trial metrics table.

---

## CLI Parameters Reference

| Parameter | Default | Type | Description |
|---|---|---|---|
| `--n-trials` | `1000` | int | Number of Monte Carlo simulation trials |
| `--models` | All trained models | str | Comma-separated list of real models to compare against |
| `--topk` | From `strategy_config.yaml` | int | Target portfolio holding count |
| `--n-drop` | From `strategy_config.yaml` | int | Drop count per rebalance (`0`=Buy & Hold, `>=topk`=Full Rebalance) |
| `--ic-only` | `False` | flag | Fast mode: IC / Rank IC only (skips backtest) |
| `--plot` | `False` | flag | Generate and save histogram PNG charts |
| `--workspace` | Current environment | str | Path to workspace root |

---

## Typical Analysis Scenarios

### Scenario A: Production Alpha Significance Gate
```bash
# Run 1,000 trials to verify model statistical superiority (p < 0.05)
python quantpits/scripts/monkey_benchmark.py --n-trials 1000 --plot
```

### Scenario B: Strategy Attribution (Execution Rule vs Alpha Signal)
Compare the following three configurations to decouple alpha source:
1. **Static Monkey**: `--topk <K> --n-drop 0` (Pure Buy & Hold random baseline);
2. **Dropout Monkey**: `--topk <K> --n-drop <N>` (Random baseline with TopK Dropout momentum/inertia rule);
3. **Real Model Strategy**: Model backtest under identical `TopK=<K>, DropN=<N>`.
* If `(2) > (1)`: The TopK Dropout execution rule itself generates positive structural excess.
* If `(3) > (2)`: The model's predictive scores provide genuine predictive Alpha beyond the rule.

### Scenario C: Single Model Pipeline Fusion Test (Layer 1)
```bash
# 1. Train the RandomModel in the workspace
python quantpits/scripts/static_train.py --models random_monkey_Alpha158

# 2. Fuse with real models in ensemble backtest to evaluate noise tolerance
python quantpits/scripts/ensemble_fusion.py --models <real_model>,random_monkey_Alpha158
```
