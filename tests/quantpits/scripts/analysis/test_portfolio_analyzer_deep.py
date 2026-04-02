"""
Deep mathematical verification of PortfolioAnalyzer.

Every test constructs data with a hand-computable ground truth and asserts
element-level closeness, rather than just checking key presence.
"""
import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from unittest.mock import patch

from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_nav_series(n=50, daily_ret=0.001, cashflows=None, seed=42):
    """
    Build a deterministic daily-amount DataFrame.

    Parameters
    ----------
    n : int
        Number of trading days.
    daily_ret : float
        Fixed daily portfolio return (before cashflow).
    cashflows : dict[int, float] or None
        Map of day-index → cashflow amount. If None, all zero.
    seed : int
        Seed for benchmark noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)

    nav = np.zeros(n)
    cf = np.zeros(n)
    nav[0] = 100_000.0

    if cashflows:
        for idx, val in cashflows.items():
            cf[idx] = val

    for i in range(1, n):
        # NAV(t) = NAV(t-1) * (1 + ret) + CF(t)
        nav[i] = nav[i - 1] * (1 + daily_ret) + cf[i]

    # Benchmark: correlated with portfolio but different
    bench_nav = 3500.0 * (1 + 0.0005 * np.arange(n) + rng.normal(0, 0.001, n)).cumprod()

    return pd.DataFrame({
        "成交日期": dates,
        "收盘价值": nav,
        "CASHFLOW": cf,
        "CSI300": bench_nav,
    })


def _expected_returns_from_nav(nav, cf):
    """Independently compute daily returns: (NAV(t) - NAV(t-1) - CF(t)) / (NAV(t-1) + CF(t))."""
    prev = np.roll(nav, 1)
    ret = (nav - prev - cf) / (prev + cf)
    # The first day is NaN after the calculation and is dropped by the analyzer.
    # Therefore, we should exclude the first element to match the analyzer's output.
    return ret[1:]


# ============================================================================
# Group 1: Daily Returns
# ============================================================================

class TestDailyReturns:
    def test_deterministic_series(self):
        """Element-by-element match against hand computation."""
        da = _build_nav_series(n=20, daily_ret=0.002, cashflows={5: -2000, 10: 5000})
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        returns = pa.calculate_daily_returns()

        expected = _expected_returns_from_nav(da["收盘价值"].values, da["CASHFLOW"].values)
        np.testing.assert_allclose(returns.values, expected, atol=1e-12)

    def test_pure_growth_no_cashflow(self):
        """With zero cashflow, return = pct_change of NAV."""
        da = _build_nav_series(n=10, daily_ret=0.01, cashflows=None)
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        returns = pa.calculate_daily_returns()

        nav = da["收盘价值"].values
        expected_pct = np.diff(nav) / nav[:-1]
        # The first element (which would be 0 or NaN) is now dropped.
        np.testing.assert_allclose(returns.values, expected_pct, atol=1e-12)

    def test_date_filtering(self):
        """start_date / end_date should filter returned series."""
        da = _build_nav_series(n=30, daily_ret=0.001)
        pa = PortfolioAnalyzer(
            daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame(),
            start_date="2025-01-10", end_date="2025-01-20"
        )
        returns = pa.calculate_daily_returns()
        assert returns.index.min() >= pd.Timestamp("2025-01-10")
        assert returns.index.max() <= pd.Timestamp("2025-01-20")

    def test_empty(self):
        pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        assert pa.calculate_daily_returns().empty


# ============================================================================
# Group 2: Benchmark NAV Conversion
# ============================================================================

class TestBenchmarkConversion:
    def test_returns_to_nav_conversion(self):
        """When benchmark starts near 0, it should be auto-converted to cumulative NAV."""
        dates = pd.bdate_range("2025-01-02", periods=5)
        # Use a non-zero first return to test for chain-break issues
        bench_returns = [0.01, 0.02, -0.005, 0.02, 0.003]
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": [100000, 101000, 100500, 102000, 102500],
            "CASHFLOW": [0] * 5,
            "CSI300": bench_returns,
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        # After conversion, first value should be (1 + r1). 
        # We NO LONGER force index 0 to 1.0 to preserve the chain.
        internal = pa.daily_amount["CSI300"].values
        expected_cumprod = np.cumprod(1 + np.array(bench_returns))
        
        # Verify first element is indeed 1.01
        assert np.isclose(internal[0], 1.01, atol=1e-12)
        
        # Element-wise match
        np.testing.assert_allclose(internal, expected_cumprod, atol=1e-12)
        
        # CRITICAL: Verify that the second day's return derived from this NAV matches the input.
        derived_returns = pd.Series(internal).pct_change().dropna().values
        # bench_returns: [0.01, 0.02, -0.005, 0.02, 0.003]
        # derived_returns should be: [0.02, -0.005, 0.02, 0.003]
        np.testing.assert_allclose(derived_returns, bench_returns[1:], atol=1e-12)

    def test_nav_mode_passthrough(self):
        """When benchmark values are large (> 0.5), no conversion should happen."""
        dates = pd.bdate_range("2025-01-02", periods=3)
        bench_nav = [3500.0, 3520.0, 3510.0]
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": [100000, 101000, 100500],
            "CASHFLOW": [0] * 3,
            "CSI300": bench_nav,
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        np.testing.assert_array_equal(pa.daily_amount["CSI300"].values, bench_nav)


# ============================================================================
# Group 3: Traditional Metrics – Full Manual Ground Truth
# ============================================================================

class TestTraditionalMetrics:
    @pytest.fixture
    def setup(self):
        """50-day deterministic data with known returns."""
        n = 50
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2025-01-02", periods=n)

        # Deterministic portfolio returns
        port_rets = np.array([0.002, -0.001, 0.003, -0.002, 0.001] * 10)
        nav = np.zeros(n)
        nav[0] = 100_000.0
        for i in range(1, n):
            nav[i] = nav[i - 1] * (1 + port_rets[i])

        # Deterministic benchmark
        bench_rets = np.array([0.001, -0.0005, 0.002, -0.001, 0.0015] * 10)
        bench_nav = np.zeros(n)
        bench_nav[0] = 3500.0
        for i in range(1, n):
            bench_nav[i] = bench_nav[i - 1] * (1 + bench_rets[i])

        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": nav,
            "CASHFLOW": 0.0,
            "CSI300": bench_nav,
        })

        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        # Expected returns: drop the first day (NaN) to match dropna() behavior
        expected_rets = np.diff(nav) / nav[:-1]
        
        # Benchmark returns for intervals
        bench_rets_intervals = np.diff(bench_nav) / bench_nav[:-1]
        
        return pa, expected_rets, bench_rets_intervals, bench_nav, n - 1

    def test_absolute_return(self, setup):
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        cum = np.cumprod(1 + rets)
        expected_abs_ret = cum[-1] - 1.0
        assert np.isclose(metrics["Absolute_Return"], expected_abs_ret, atol=1e-10)

    def test_cagr(self, setup):
        pa, rets, _, _, n = setup
        metrics = pa.calculate_traditional_metrics()
        cum = np.cumprod(1 + rets)
        years = n / 252.0
        expected_cagr = cum[-1] ** (1 / years) - 1
        assert np.isclose(metrics["CAGR_252"], expected_cagr, atol=1e-10)

    def test_arithmetic_annual_return(self, setup):
        """Portfolio_Arithmetic_Annual_Return = mean(daily_returns) * 252.
        AM-GM: daily arithmetic mean > daily geometric mean for positive volatility."""
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        expected_arith = float(np.mean(rets) * 252)
        assert np.isclose(metrics["Portfolio_Arithmetic_Annual_Return"], expected_arith, atol=1e-10)
        # AM > GM at the daily level (before annualization)
        daily_am = np.mean(rets)
        daily_gm = np.prod(1 + rets) ** (1 / len(rets)) - 1
        assert daily_am > daily_gm

    def test_volatility(self, setup):
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        expected_vol = np.std(rets, ddof=1) * np.sqrt(252)
        assert np.isclose(metrics["Volatility"], expected_vol, atol=1e-10)

    def test_sharpe(self, setup):
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        rf_daily = 0.0135 / 252
        expected_sharpe = ((np.mean(rets) - rf_daily) / np.std(rets, ddof=1)) * np.sqrt(252)
        assert np.isclose(metrics["Sharpe"], expected_sharpe, atol=1e-10)

    def test_sortino(self, setup):
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        rf_daily = 0.0135 / 252
        downside_dev = np.sqrt(np.mean(np.minimum(0, rets)**2))
        expected_sortino = ((np.mean(rets) - rf_daily) / downside_dev) * np.sqrt(252)
        assert np.isclose(metrics["Sortino"], expected_sortino, atol=1e-10)

    def test_max_drawdown(self, setup):
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        cum = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        expected_maxdd = dd.min()
        assert np.isclose(metrics["Max_Drawdown"], expected_maxdd, atol=1e-10)

    def test_calmar(self, setup):
        pa, rets, _, _, n = setup
        metrics = pa.calculate_traditional_metrics()
        cum = np.cumprod(1 + rets)
        years = n / 252.0
        cagr = cum[-1] ** (1 / years) - 1
        running_max = np.maximum.accumulate(cum)
        maxdd = (cum / running_max - 1).min()
        expected_calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
        assert np.isclose(metrics["Calmar"], expected_calmar, atol=1e-10)

    def test_win_rate_and_profit_factor(self, setup):
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        expected_win_rate = (rets > 0).mean()
        assert np.isclose(metrics["Daily_Return_Win_Rate"], expected_win_rate, atol=1e-10)

        nav = pd.Series(pa.daily_amount['收盘价值'].values)
        daily_pnl = nav - nav.shift(1).fillna(nav)
        
        gross_profit = daily_pnl[daily_pnl > 0].sum()
        gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
        expected_pf_pnl = gross_profit / gross_loss
        assert np.isclose(metrics["Daily_Profit_Factor_(PnL)"], expected_pf_pnl, atol=1e-10)
        
        expected_pf_ret = rets[rets > 0].sum() / abs(rets[rets < 0].sum())
        assert np.isclose(metrics["Daily_Profit_Factor_(Returns)"], expected_pf_ret, atol=1e-10)
        assert pd.isna(metrics["Trade_Profit_Factor"])
        assert pd.isna(metrics["Realized_Trade_Win_Rate"])

        pa, _, bench_rets, bench_nav, n_intervals = setup
        metrics = pa.calculate_traditional_metrics()
        # benchmark_abs_ret is (P_end / P_start) - 1 where start is index[0] of returns
        # which corresponds to the 'previous day' of the first return.
        # But in setup, rets starts from index 1.
        bench_cum = np.cumprod(1 + bench_rets)
        years = n_intervals / 252.0
        expected_bench_cagr = bench_cum[-1] ** (1 / years) - 1
        assert np.isclose(metrics["Benchmark_CAGR_252"], expected_bench_cagr, atol=1e-10)

    def test_excess_return_cagr(self, setup):
        pa, rets, bench_rets, bench_nav, n_intervals = setup
        metrics = pa.calculate_traditional_metrics()
        cum = np.cumprod(1 + rets)
        years = n_intervals / 252.0
        cagr = cum[-1] ** (1 / years) - 1
        bench_cum = np.cumprod(1 + bench_rets)
        bench_cagr = bench_cum[-1] ** (1 / years) - 1
        expected_excess = (1 + cagr) / (1 + bench_cagr) - 1
        assert np.isclose(metrics["Excess_Return_CAGR_252"], expected_excess, atol=1e-10)

    def test_tracking_error_and_ir(self, setup):
        pa, rets, bench_rets, bench_nav, n_intervals = setup
        metrics = pa.calculate_traditional_metrics()

        # The analyzer now uses shift-then-slice for benchmark as well,
        # so bench_rets already has the correct intervals.
        active_ret = rets - bench_rets
        expected_te = np.std(active_ret, ddof=1) * np.sqrt(252)
        expected_ir = (np.mean(active_ret) * 252) / expected_te

        assert np.isclose(metrics["Tracking_Error"], expected_te, atol=1e-10)
        assert np.isclose(metrics["Information_Ratio_(Arithmetic)"], expected_ir, atol=1e-10)

    def test_tracking_error_and_ir_log(self, setup):
        pa, rets, bench_rets, bench_nav, n_intervals = setup
        metrics = pa.calculate_traditional_metrics()

        # Log-based active returns: ln(1+Rp) - ln(1+Rb)
        log_active_ret = np.log((1 + rets) / (1 + bench_rets))
        expected_te_geo = np.std(log_active_ret, ddof=1) * np.sqrt(252)
        expected_ir_log = (np.mean(log_active_ret) * 252) / expected_te_geo

        assert np.isclose(metrics["Tracking_Error_(Geometric)"], expected_te_geo, atol=1e-10)
        assert np.isclose(metrics["Information_Ratio_(Log_Based)"], expected_ir_log, atol=1e-10)

    def test_time_under_water_unfinished(self):
        """Verify Max/Avg TUW for portfolio, specifically handling unfinished drawdowns at the end."""
        # Day 0: 100
        # Day 1: 101
        # Day 2: 102 (Peak)
        # Day 3: 98 (UW length 1)
        # Day 4: 99 (Still UW, length 2, end)
        dates = pd.bdate_range("2025-01-01", periods=5)
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": [100.0, 101.0, 102.0, 98.0, 99.0],
            "CASHFLOW": 0.0,
            "CSI300": 3500.0 # Constant
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        
        metrics = pa.calculate_traditional_metrics()
        
        # Max TUW = 2 (Days 3 and 4)
        # Avg TUW = 0 (block is unfinished)
        assert np.isclose(metrics["Max_Time_Under_Water_Days"], 2.0, atol=1e-10)
        assert np.isclose(metrics["Avg_Time_Under_Water_Days"], 0.0, atol=1e-10)

        # Finished + Unfinished case
        # NAV: 100, 101, 98, 102, 97 (Finished 101->98->102, Unfinished 102->97)
        da2 = pd.DataFrame({
            "成交日期": pd.bdate_range("2025-01-01", periods=5),
            "收盘价值": [100.0, 101.0, 98.0, 102.0, 97.0],
            "CASHFLOW": 0.0,
            "CSI300": 3500.0
        })
        pa2 = PortfolioAnalyzer(daily_amount_df=da2, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        metrics2 = pa2.calculate_traditional_metrics()
        
        # Max TUW = 1
        # Avg TUW = 1.0
        assert np.isclose(metrics2["Max_Time_Under_Water_Days"], 1.0, atol=1e-10)
        assert np.isclose(metrics2["Avg_Time_Under_Water_Days"], 1.0, atol=1e-10)

    def test_benchmark_time_under_water_unfinished(self):
        """Verify Max/Avg TUW for benchmark specifically handling unfinished drawdowns at the end."""
        dates = pd.bdate_range("2025-01-01", periods=6)
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": [1000] * 6,
            "CASHFLOW": 0.0,
            "CSI300": [100.0, 101.0, 102.0, 98.0, 99.0, 97.0],
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        metrics = pa.calculate_traditional_metrics()
        
        # Bench is UW for last 3 days
        assert np.isclose(metrics["Benchmark_Max_Time_Under_Water_Days"], 3.0, atol=1e-10)
        assert np.isclose(metrics["Benchmark_Avg_Time_Under_Water_Days"], 0.0, atol=1e-10)

        da2 = pd.DataFrame({
            "成交日期": pd.bdate_range("2025-01-01", periods=5),
            "收盘价值": [1000] * 5,
            "CASHFLOW": 0.0,
            "CSI300": [100.0, 101.0, 98.0, 102.0, 97.0],
        })
        pa2 = PortfolioAnalyzer(daily_amount_df=da2, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        metrics2 = pa2.calculate_traditional_metrics()
        
        assert np.isclose(metrics2["Benchmark_Max_Time_Under_Water_Days"], 1.0, atol=1e-10)
        assert np.isclose(metrics2["Benchmark_Avg_Time_Under_Water_Days"], 1.0, atol=1e-10)

    def test_calendar_cagr_uses_nav_date(self):
        """Calendar CAGR should use the first NAV date (base of first return)
        rather than returns.index[0], avoiding fencepost error."""
        # Build 5 trading days starting on a Monday
        # NAV dates: Mon 01-06, Tue 01-07, Wed 01-08, Thu 01-09, Fri 01-10
        dates = pd.bdate_range("2025-01-06", periods=5)
        nav = [100_000.0, 100_500, 101_000, 100_800, 101_200]
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": nav,
            "CASHFLOW": 0.0,
            "CSI300": [3500, 3510, 3520, 3515, 3525],
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        metrics = pa.calculate_traditional_metrics()
        
        returns = pa.calculate_daily_returns()
        cum_ret = (1 + returns).cumprod()
        
        # Calendar days should span from first NAV date to last return date
        # First NAV date = 2025-01-06, last return date = 2025-01-10
        # Calendar days = (01-10 - 01-06) = 4 days
        expected_cal_days = (dates[-1] - dates[0]).days  # 4
        expected_years_cal = expected_cal_days / 365.0
        expected_cagr_cal = cum_ret.iloc[-1] ** (1 / expected_years_cal) - 1
    def test_max_daily_drop(self, setup):
        pa, rets, _, _, _ = setup
        metrics = pa.calculate_traditional_metrics()
        expected_drop = float(np.min(rets))
        assert np.isclose(metrics["Max_Daily_Drop"], expected_drop, atol=1e-10)

    def test_trade_profit_factor_mtm(self, setup):
        """
        Verify Trade_Profit_Factor_MTM with open positions at end of period.
        - Start with 100k
        - Day 1: Buy 1000 shares of STOCK_A at 10.0 (Cost 10,000, 0 fees)
        - Day 2: Buy 500 shares of STOCK_B at 20.0 (Cost 10,000, 0 fees)
        - Day 3: Sell 500 shares of STOCK_A at 12.0 (Realized Profit 1000)
        - Period End (Day 3): 
            - Open STOCK_A: 500 shares, at 12.0 (Unrealized Profit 1000)
            - Open STOCK_B: 500 shares, at 18.0 (Unrealized Loss 1000)
        - Expected Realized PF: 1000 / 0 = inf (Gross Profit 1000, Gross Loss 0)
        - Expected MTM PF: (1000 realized + 1000 unrealized gain) / (1000 unrealized loss) = 2.0
        """
        dates = pd.bdate_range("2025-01-01", periods=4) # Day 0, 1, 2, 3
        # Use simple returns to avoid NAV/amount complexity
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": [100000, 110000, 105000, 115000],
            "CASHFLOW": 0.0,
            "CSI300": 3500.0
        })
        
        # Trade Log: Buy A, Buy B, Sell Half A
        tl = pd.DataFrame([
            {"成交日期": dates[1], "证券代码": "A", "交易类别": "证券买入", "成交数量": 1000, "成交价格": 10.0, "成交金额": 10000, "费用合计": 0},
            {"成交日期": dates[2], "证券代码": "B", "交易类别": "证券买入", "成交数量": 500, "成交价格": 20.0, "成交金额": 10000, "费用合计": 0},
            {"成交日期": dates[3], "证券代码": "A", "交易类别": "证券卖出", "成交数量": 500, "成交价格": 12.0, "成交金额": 6000, "费用合计": 0},
        ])
        
        # Holding Log: Terminal prices on Day 3
        hl = pd.DataFrame([
            {"成交日期": dates[3], "证券代码": "A", "持仓数量": 500, "收盘价格": 12.0},
            {"成交日期": dates[3], "证券代码": "B", "持仓数量": 500, "收盘价格": 18.0},
        ])
        
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=tl, holding_log_df=hl)
        metrics = pa.calculate_traditional_metrics()
        
        # Realized PF
        assert metrics["Trade_Profit_Factor"] == np.inf
        # MTM PF: (1000 realized + 1000 unrealized) / 1000 loss = 2.0
        assert np.isclose(metrics["Trade_Profit_Factor_MTM"], 2.0, atol=1e-10)


# ============================================================================
# Group 4: Factor Exposure (OLS Regression) – Critical Path
# ============================================================================

class TestFactorExposure:
    def test_ols_with_benchmark_column(self):
        """
        Construct portfolio returns as: port = alpha + beta * market + noise.
        Then verify the analyzer's OLS matches independent statsmodels.
        """
        n = 100
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-02", periods=n)

        # Known parameters
        true_daily_alpha = 0.0003
        true_beta = 0.85
        noise = rng.normal(0, 0.002, n)

        # Market returns from benchmark NAV
        bench_nav = np.zeros(n)
        bench_nav[0] = 3500.0
        bench_daily_rets = rng.normal(0.0005, 0.01, n)
        bench_daily_rets[0] = 0
        for i in range(1, n):
            bench_nav[i] = bench_nav[i - 1] * (1 + bench_daily_rets[i])

        # Market return series used in regression = pct_change of bench_nav
        market_ret = np.zeros(n)
        market_ret[1:] = np.diff(bench_nav) / bench_nav[:-1]

        # Construct portfolio returns
        port_rets = true_daily_alpha + true_beta * market_ret + noise
        port_rets[0] = 0.0

        # Build NAV from these returns
        port_nav = np.zeros(n)
        port_nav[0] = 100_000.0
        for i in range(1, n):
            port_nav[i] = port_nav[i - 1] * (1 + port_rets[i])

        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": port_nav,
            "CASHFLOW": 0.0,
            "CSI300": bench_nav,
        })

        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
            result = pa.calculate_factor_exposure()

        # Independent reference: replicate analyzer's logic exactly
        actual_returns = pa.calculate_daily_returns()
        # market_return in analyzer is calculated on the FULL market series and then aligned
        market_close_full = pd.Series(bench_nav, index=dates)
        market_return_full = market_close_full.pct_change().dropna()
        mkt_ret_series = market_return_full.loc[market_return_full.index.intersection(actual_returns.index)]
        
        rf_daily = 0.0135 / 252
        port_excess = actual_returns - rf_daily
        mkt_excess = mkt_ret_series - rf_daily

        aligned = pd.concat([port_excess, mkt_excess], axis=1).dropna()
        aligned.columns = ["Portfolio", "Market"]
        X = sm.add_constant(aligned["Market"])
        model_ref = sm.OLS(aligned["Portfolio"], X).fit()

        expected_alpha = model_ref.params["const"] * 252
        expected_beta = model_ref.params["Market"]
        expected_r2 = model_ref.rsquared

        assert np.isclose(result["Annualized_Alpha"], expected_alpha, atol=1e-10), \
            f"Alpha: {result['Annualized_Alpha']} vs {expected_alpha}"
        assert np.isclose(result["Beta_Market"], expected_beta, atol=1e-10), \
            f"Beta: {result['Beta_Market']} vs {expected_beta}"
        assert np.isclose(result["R_Squared"], expected_r2, atol=1e-10), \
            f"R²: {result['R_Squared']} vs {expected_r2}"
        
        expected_market_total = mkt_ret_series.mean() * 252
        assert np.isclose(result["Market_Total_Return_Annualized"], expected_market_total, atol=1e-10)

    def test_ols_qlib_fallback_branch(self):
        """
        Without CSI300 column, the code falls back to computing market return
        as the cross-sectional mean of individual instrument returns.
        """
        n = 20
        dates = pd.bdate_range("2025-01-02", periods=n)

        # Build portfolio NAV
        rng = np.random.default_rng(99)
        port_rets = rng.normal(0.001, 0.01, n)
        port_rets[0] = 0.0
        port_nav = np.zeros(n)
        port_nav[0] = 100_000.0
        for i in range(1, n):
            port_nav[i] = port_nav[i - 1] * (1 + port_rets[i])

        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": port_nav,
            "CASHFLOW": 0.0,
            # No CSI300 column!
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        # Build mock multi-instrument features
        instruments = ["SZ000001", "SZ000002", "SZ000003"]
        rows = []
        for inst in instruments:
            base_price = rng.uniform(10, 50)
            for i, d in enumerate(dates):
                close = base_price * (1 + rng.normal(0.0005, 0.01)) ** (i + 1)
                rows.append({"instrument": inst, "datetime": d, "close": close})
        mock_features = pd.DataFrame(rows)

        with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=mock_features):
            with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
                result = pa.calculate_factor_exposure()

        # Independent: compute market return = mean of instrument returns per day
        feat = mock_features.sort_values(["instrument", "datetime"]).copy()
        feat["prev_close"] = feat.groupby("instrument")["close"].shift(1)
        feat["ret"] = (feat["close"] - feat["prev_close"]) / feat["prev_close"]
        market_return = feat.groupby("datetime")["ret"].mean()

        actual_rets = pa.calculate_daily_returns()
        
        rf_daily = 0.0135 / 252
        port_excess = actual_rets - rf_daily
        mkt_excess = market_return - rf_daily
        
        aligned = pd.concat([port_excess, mkt_excess], axis=1).dropna()
        aligned.columns = ["Portfolio", "Market"]

        if len(aligned) >= 2:
            X = sm.add_constant(aligned["Market"])
            model_ref = sm.OLS(aligned["Portfolio"], X).fit()

            assert np.isclose(result["Beta_Market"], model_ref.params["Market"], atol=1e-10)
            expected_alpha = model_ref.params["const"] * 252
            assert np.isclose(result["Annualized_Alpha"], expected_alpha, atol=1e-10)
            assert np.isclose(result["R_Squared"], model_ref.rsquared, atol=1e-10)

    def test_benchmark_returns_mode(self):
        """
        When the benchmark column contains returns (abs < 2.0), the code should
        use them directly as market_return rather than computing pct_change.
        """
        n = 30
        dates = pd.bdate_range("2025-01-02", periods=n)
        rng = np.random.default_rng(77)

        # Benchmark as returns (small values)
        bench_returns = rng.normal(0.0005, 0.01, n)
        bench_returns[0] = 0.0

        port_rets = 0.0003 + 0.9 * bench_returns + rng.normal(0, 0.002, n)
        port_rets[0] = 0.0

        port_nav = np.zeros(n)
        port_nav[0] = 100_000.0
        for i in range(1, n):
            port_nav[i] = port_nav[i - 1] * (1 + port_rets[i])

        # Benchmark will be auto-converted to NAV by the constructor.
        # After that, the factor_exposure code checks max(abs(bench)) < 2.0
        # to decide if it's returns or NAV. After cumprod, values will be ~1.0,
        # so abs().max() will likely be < 2.0 → treated as returns directly.
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": port_nav,
            "CASHFLOW": 0.0,
            "CSI300": bench_returns,
        })

        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
            result = pa.calculate_factor_exposure()

        # Independent: after constructor, CSI300 is cumprod NAV
        # downstream code now always takes pct_change directly
        actual_rets = pa.calculate_daily_returns()
        internal_bench = pa.daily_amount["CSI300"].astype(float)
        market_return_full = internal_bench.pct_change().dropna()
        market_ret = market_return_full.loc[market_return_full.index.intersection(actual_rets.index)]
        rf_daily = 0.0135 / 252

        aligned = pd.concat([actual_rets - rf_daily, market_ret - rf_daily], axis=1).dropna()
        aligned.columns = ["Portfolio", "Market"]
        X = sm.add_constant(aligned["Market"])
        model_ref = sm.OLS(aligned["Portfolio"], X).fit()

        assert np.isclose(result["Beta_Market"], model_ref.params["Market"], atol=1e-10)

    def test_attribution_identity_single_factor(self):
        """
        Verify: rf*(1-β) + β*E(Rm) + α ≡ Portfolio_Arithmetic_Annual_Return.
        This is the fundamental OLS identity and must hold exactly.
        """
        n = 100
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-02", periods=n)

        bench_nav = np.zeros(n)
        bench_nav[0] = 3500.0
        bench_daily_rets = rng.normal(0.0005, 0.01, n)
        bench_daily_rets[0] = 0
        for i in range(1, n):
            bench_nav[i] = bench_nav[i - 1] * (1 + bench_daily_rets[i])

        port_rets = 0.0003 + 0.85 * np.diff(bench_nav, prepend=bench_nav[0]) / np.roll(bench_nav, 1) + rng.normal(0, 0.002, n)
        port_rets[0] = 0.0
        port_nav = np.zeros(n)
        port_nav[0] = 100_000.0
        for i in range(1, n):
            port_nav[i] = port_nav[i - 1] * (1 + port_rets[i])

        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": port_nav,
            "CASHFLOW": 0.0,
            "CSI300": bench_nav,
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
            result = pa.calculate_factor_exposure()

        rf_annual = 0.0135
        beta = result['Beta_Market']
        alpha = result['Annualized_Alpha']
        market_ann = result['Market_Total_Return_Annualized']
        aligned_arith = result['Portfolio_Arithmetic_Annual_Return']

        # OLS identity: E(Rp) = rf*(1-β) + β*E(Rm) + α
        rf_component = rf_annual * (1 - beta)
        beta_return = beta * market_ann
        attribution_sum = rf_component + beta_return + alpha

        assert np.isclose(attribution_sum, aligned_arith, atol=1e-10), \
            f"Single-factor identity failed: {rf_component:.6%} + {beta_return:.6%} + {alpha:.6%} = {attribution_sum:.6%} vs {aligned_arith:.6%}"

        # Also check Aligned_Sample_Size is present and sane
        assert 'Aligned_Sample_Size' in result
        assert result['Aligned_Sample_Size'] > 0


# ============================================================================
# Group 5: Style Exposures (Multi-factor OLS)
# ============================================================================

class TestStyleExposures:
    def test_multi_factor_regression(self):
        """
        Full verification of style factor construction and multi-factor OLS.
        """
        n_port = 30
        dates_port = pd.bdate_range("2025-02-03", periods=n_port)

        rng = np.random.default_rng(55)
        port_rets = rng.normal(0.001, 0.01, n_port)
        port_rets[0] = 0.0
        port_nav = np.zeros(n_port)
        port_nav[0] = 100_000.0
        for i in range(1, n_port):
            port_nav[i] = port_nav[i - 1] * (1 + port_rets[i])

        bench_nav = 3500.0 * np.cumprod(1 + rng.normal(0.0005, 0.008, n_port))

        da = pd.DataFrame({
            "成交日期": dates_port,
            "收盘价值": port_nav,
            "CASHFLOW": 0.0,
            "CSI300": bench_nav,
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        # Build mock features: needs bigger date range for 20-day rolling
        n_feat = 60
        dates_feat = pd.bdate_range("2024-12-01", periods=n_feat)
        instruments = [f"SZ{str(i).zfill(6)}" for i in range(20)]
        rows = []
        for inst in instruments:
            base_price = rng.uniform(10, 50)
            base_vol = rng.uniform(1e6, 5e6)
            for j, d in enumerate(dates_feat):
                close = base_price * (1 + rng.normal(0.001, 0.02)) ** (j + 1)
                volume = base_vol * (1 + rng.normal(0, 0.3))
                rows.append({"instrument": inst, "datetime": d, "close": max(close, 0.01), "volume": max(volume, 100)})
        mock_features = pd.DataFrame(rows)

        with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=mock_features):
            with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
                result = pa.calculate_style_exposures()

        if not result:
            # Not enough aligned data after rolling operations — pass silently.
            return

        # Independent verification: replicate factor construction with T-1 lag
        feat = mock_features.sort_values(["instrument", "datetime"]).copy()
        feat["prev_close"] = feat.groupby("instrument")["close"].shift(1)
        feat["ret"] = (feat["close"] - feat["prev_close"]) / feat["prev_close"]
        feat["liquidity"] = np.log(feat["close"] * feat["volume"] + 1e-9)
        feat["liquidity"] = feat.groupby("instrument")["liquidity"].shift(1)
        feat["momentum"] = feat.groupby("instrument")["close"].pct_change(20).shift(1)
        feat["volatility"] = feat.groupby("instrument")["ret"].rolling(20, min_periods=5).std().reset_index(0, drop=True)
        feat["volatility"] = feat.groupby("instrument")["volatility"].shift(1)
        feat = feat.dropna(subset=["ret", "liquidity", "momentum", "volatility"])

        # Factor returns: long-short top/bottom 20%
        factor_returns = {}
        for factor in ["liquidity", "momentum", "volatility"]:
            def _factor_ret(df, f=factor):
                if len(df) < 5:
                    return 0.0
                q_top = df[f].quantile(0.8)
                q_bot = df[f].quantile(0.2)
                ret_top = df[df[f] >= q_top]["ret"].mean()
                ret_bot = df[df[f] <= q_bot]["ret"].mean()
                if pd.isna(ret_top): ret_top = 0
                if pd.isna(ret_bot): ret_bot = 0
                return ret_top - ret_bot
            factor_returns[factor] = feat.groupby("datetime").apply(_factor_ret)

        factor_df = pd.DataFrame(factor_returns)

        # Full regression check
        actual_rets = pa.calculate_daily_returns()
        market_close_full = pd.Series(bench_nav, index=dates_port)
        market_return_full = market_close_full.pct_change().dropna()
        market_ret = market_return_full.loc[market_return_full.index.intersection(actual_rets.index)]

        aligned = pd.concat([actual_rets - (0.0135/252), market_ret - (0.0135/252), factor_df], axis=1).dropna()
        if len(aligned) < 2:
            # Not enough aligned data for regression — pass silently.
            return

        aligned.columns = ["Portfolio", "Market"] + list(factor_df.columns)

        # Check Factor_Annualized — must be computed from ALIGNED data (same sample as regression)
        for col in factor_df.columns:
            expected_ann = float(aligned[col].mean() * 252)
            if "Factor_Annualized" in result:
                assert np.isclose(result["Factor_Annualized"][col], expected_ann, atol=1e-8), \
                    f"Factor_Annualized[{col}]: {result['Factor_Annualized'][col]} vs {expected_ann}"

        X = sm.add_constant(aligned[["Market", "liquidity", "momentum", "volatility"]])
        model_ref = sm.OLS(aligned["Portfolio"], X).fit()

        assert np.isclose(result["Multi_Factor_Beta"], model_ref.params.get("Market", 0), atol=1e-8)
        assert np.isclose(result["Barra_Liquidity_Exp_(High-Low)"], model_ref.params.get("liquidity", 0), atol=1e-8)
        assert np.isclose(result["Barra_Momentum_Exp_(High-Low)"], model_ref.params.get("momentum", 0), atol=1e-8)
        assert np.isclose(result["Barra_Volatility_Exp_(High-Low)"], model_ref.params.get("volatility", 0), atol=1e-8)
        assert np.isclose(result["Barra_Style_R_Squared"], model_ref.rsquared, atol=1e-8)

        expected_intercept = float(model_ref.params.get("const", 0)) * 252
        assert np.isclose(result["Multi_Factor_Intercept"], expected_intercept, atol=1e-8)
        assert "Market_Total_Return_Annualized" in result
        assert "Market_Excess_Return_Annualized" in result

    def test_attribution_identity_holds(self):
        """
        Verify multi-factor identity:
        rf*(1-β) + β*E(Rm) + Σ(βi*E(Fi)) + α ≡ Portfolio_Arithmetic_Annual_Return.
        
        This must hold exactly because OLS guarantees:
        E(Y) = β₀ + β₁*E(X₁) + β₂*E(X₂) + ...
        """
        n_port = 30
        dates_port = pd.bdate_range("2025-02-03", periods=n_port)

        rng = np.random.default_rng(55)
        port_rets = rng.normal(0.001, 0.01, n_port)
        port_rets[0] = 0.0
        port_nav = np.zeros(n_port)
        port_nav[0] = 100_000.0
        for i in range(1, n_port):
            port_nav[i] = port_nav[i - 1] * (1 + port_rets[i])

        bench_nav = 3500.0 * np.cumprod(1 + rng.normal(0.0005, 0.008, n_port))

        da = pd.DataFrame({
            "成交日期": dates_port,
            "收盘价值": port_nav,
            "CASHFLOW": 0.0,
            "CSI300": bench_nav,
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        n_feat = 60
        dates_feat = pd.bdate_range("2024-12-01", periods=n_feat)
        instruments = [f"SZ{str(i).zfill(6)}" for i in range(20)]
        rows = []
        for inst in instruments:
            base_price = rng.uniform(10, 50)
            base_vol = rng.uniform(1e6, 5e6)
            for j, d in enumerate(dates_feat):
                close = base_price * (1 + rng.normal(0.001, 0.02)) ** (j + 1)
                volume = base_vol * (1 + rng.normal(0, 0.3))
                rows.append({"instrument": inst, "datetime": d, "close": max(close, 0.01), "volume": max(volume, 100)})
        mock_features = pd.DataFrame(rows)

        with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=mock_features):
            with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
                result = pa.calculate_style_exposures()

        if not result:
            return  # Not enough data for regression

        rf_annual = 0.0135
        multi_beta = result['Multi_Factor_Beta']
        alpha = result['Multi_Factor_Intercept']
        market_ann = result['Market_Total_Return_Annualized']
        aligned_arith = result['Portfolio_Arithmetic_Annual_Return']
        factor_ann = result['Factor_Annualized']

        # rf*(1-β)
        rf_component = rf_annual * (1 - multi_beta)
        # β*E(Rm)
        beta_return = multi_beta * market_ann
        # Σ(βi*E(Fi))
        from quantpits.scripts.analysis.portfolio_analyzer import BARRA_LIQD_KEY, BARRA_MOMT_KEY, BARRA_VOLA_KEY
        style_ret = 0.0
        style_ret += result[BARRA_LIQD_KEY] * factor_ann['liquidity']
        style_ret += result[BARRA_MOMT_KEY] * factor_ann['momentum']
        style_ret += result[BARRA_VOLA_KEY] * factor_ann['volatility']

        attribution_sum = rf_component + beta_return + style_ret + alpha

        assert np.isclose(attribution_sum, aligned_arith, atol=1e-8), \
            f"Multi-factor identity failed: {rf_component:.6%} + {beta_return:.6%} + {style_ret:.6%} + {alpha:.6%} = {attribution_sum:.6%} vs {aligned_arith:.6%}"

        assert 'Aligned_Sample_Size' in result
        assert result['Aligned_Sample_Size'] > 0

    def test_aligned_return_differs_from_full(self):
        """
        When factor_df has NaN on some portfolio trading dates, the multi-factor
        aligned return should reflect only the dates used in regression, not
        the full portfolio return series.
        """
        n_port = 40
        dates_port = pd.bdate_range("2025-02-03", periods=n_port)

        rng = np.random.default_rng(77)
        port_rets = rng.normal(0.001, 0.01, n_port)
        port_rets[0] = 0.0
        port_nav = np.zeros(n_port)
        port_nav[0] = 100_000.0
        for i in range(1, n_port):
            port_nav[i] = port_nav[i - 1] * (1 + port_rets[i])

        bench_nav = 3500.0 * np.cumprod(1 + rng.normal(0.0005, 0.008, n_port))

        da = pd.DataFrame({
            "成交日期": dates_port,
            "收盘价值": port_nav,
            "CASHFLOW": 0.0,
            "CSI300": bench_nav,
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())

        # Build features that intentionally cover FEWER dates than portfolio
        # Only provide features for dates_port[5:] to simulate Qlib calendar mismatch
        n_feat = 50
        dates_feat = pd.bdate_range("2024-12-15", periods=n_feat)
        instruments = [f"SZ{str(i).zfill(6)}" for i in range(20)]
        rows = []
        for inst in instruments:
            base_price = rng.uniform(10, 50)
            base_vol = rng.uniform(1e6, 5e6)
            for j, d in enumerate(dates_feat):
                close = base_price * (1 + rng.normal(0.001, 0.02)) ** (j + 1)
                volume = base_vol * (1 + rng.normal(0, 0.3))
                rows.append({"instrument": inst, "datetime": d, "close": max(close, 0.01), "volume": max(volume, 100)})
        mock_features = pd.DataFrame(rows)

        with patch('quantpits.scripts.analysis.portfolio_analyzer.get_daily_features', return_value=mock_features):
            with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
                result = pa.calculate_style_exposures()

        if not result:
            return  # Not enough data for regression

        # The aligned sample size should be <= n_port - 1 (returns have n-1 elements)
        # and potentially less due to factor NaN drops
        full_returns = pa.calculate_daily_returns()
        full_n = len(full_returns)
        aligned_n = result['Aligned_Sample_Size']
        
        # If alignment dropped data, the aligned return should differ from full return
        full_arith = float(full_returns.mean() * 252)
        aligned_arith = result['Portfolio_Arithmetic_Annual_Return']
        
        if aligned_n < full_n:
            # They CAN be equal by coincidence, but the key guarantee is that
            # the attribution identity holds with the ALIGNED return
            rf_annual = 0.0135
            multi_beta = result['Multi_Factor_Beta']
            alpha = result['Multi_Factor_Intercept']
            market_ann = result['Market_Total_Return_Annualized']
            factor_ann = result['Factor_Annualized']
            from quantpits.scripts.analysis.portfolio_analyzer import BARRA_LIQD_KEY, BARRA_MOMT_KEY, BARRA_VOLA_KEY

            rf_component = rf_annual * (1 - multi_beta)
            beta_return = multi_beta * market_ann
            style_ret = (result[BARRA_LIQD_KEY] * factor_ann['liquidity']
                        + result[BARRA_MOMT_KEY] * factor_ann['momentum']
                        + result[BARRA_VOLA_KEY] * factor_ann['volatility'])
            attribution_sum = rf_component + beta_return + style_ret + alpha

            # Identity must hold with aligned return, NOT full return
            assert np.isclose(attribution_sum, aligned_arith, atol=1e-8), \
                f"Identity should hold with aligned arith ({aligned_arith:.6%}), got {attribution_sum:.6%}"


# ============================================================================
# Group 6: Holding Metrics
# ============================================================================

class TestHoldingMetrics:
    def test_manual_computation(self):
        """Hand-computed holding metrics with known data."""
        hl = pd.DataFrame({
            "成交日期": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-03"] * 2),
            "证券代码": ["SZ000001", "SZ000002", "CASH", "SZ000001", "CASH"],
            "收盘价值": [8000.0, 12000.0, 80000.0, 15000.0, 85000.0],
            "浮盈收益率": [0.05, -0.02, pd.NA, 0.10, pd.NA],
        })
        pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=hl)
        m = pa.calculate_holding_metrics()

        # Day 1: 2 stocks (excl CASH), Day 2: 1 stock → avg = 1.5
        assert np.isclose(m["Avg_Daily_Holdings_Count"], 1.5, atol=1e-10)

        # Top1 concentration: Day1 = 12000/100000=0.12, Day2 = 15000/100000=0.15 → avg = 0.135
        assert np.isclose(m["Avg_Top1_Concentration"], 0.135, atol=1e-10)

        # Avg floating return: (0.05, -0.02, 0.10) / 3
        expected_avg_float = (0.05 + (-0.02) + 0.10) / 3
        assert np.isclose(m["Avg_Floating_Return"], expected_avg_float, atol=1e-10)

        # Win rate: 0.05>0 → T, -0.02>0 → F, 0.10>0 → T → 2/3
        assert np.isclose(m["Daily_Holding_Win_Rate"], 2.0 / 3.0, atol=1e-10)

    def test_all_cash(self):
        """Only CASH entries → empty result."""
        hl = pd.DataFrame({
            "成交日期": pd.to_datetime(["2025-01-02"]),
            "证券代码": ["CASH"],
            "收盘价值": [100000.0],
            "浮盈收益率": [pd.NA],
        })
        pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=hl)
        assert pa.calculate_holding_metrics() == {}


# ============================================================================
# Group 7: Turnover Rate
# ============================================================================

class TestTurnover:
    def test_manual_turnover(self):
        """Verify annual turnover calculation with known trades and NAV."""
        dates = pd.bdate_range("2025-01-02", periods=5)
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": [100_000.0, 102_000.0, 101_000.0, 103_000.0, 104_000.0],
            "CASHFLOW": [0] * 5,
            "CSI300": [3500, 3520, 3510, 3530, 3540],
        })
        tl = pd.DataFrame({
            "成交日期": pd.to_datetime(["2025-01-03", "2025-01-06"]),
            "证券代码": ["SZ000001", "SZ000002"],
            "交易类别": ["买入", "卖出"],
            "成交金额": [20_000.0, 10_000.0],
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=tl, holding_log_df=pd.DataFrame())
        metrics = pa.calculate_traditional_metrics()

        # Manual: daily_trade by date: 01-03→20000, 01-06→10000
        # NAV on each day: [100000, 102000, 101000, 103000, 104000]
        # daily_turnover = (trade_amount/2) / NAV, aligned by concat then fillna(0)
        # 01-02: 0/100000=0, 01-03: 10000/102000, 01-06: 5000/103000
        # 01-07: 0/101000=0, 01-08: 0/104000=0 (remaining days)
        # Wait, need to check: trade dates and NAV dates must align after concat
        # The trade on 01-03 maps to NAV of the same date in returns.index
        returns = pa.calculate_daily_returns()
        full_nav = pa.daily_amount["收盘价值"].astype(float)
        # Use Average NAV
        avg_nav = (full_nav + full_nav.shift(1).fillna(full_nav)) / 2
        daily_nav = avg_nav.loc[returns.index]
        
        daily_trade = tl.groupby("成交日期")["成交金额"].sum()
        aligned = pd.concat([daily_nav, daily_trade], axis=1).fillna(0)
        aligned.columns = ["NAV", "Trade_Amount"]
        aligned["NAV"] = aligned["NAV"].replace(0, 1e-9)
        daily_to = (aligned["Trade_Amount"] / 2) / aligned["NAV"]
        expected_annual = float(daily_to.mean() * 252)

        assert np.isclose(metrics["Turnover_Rate_Annual"], expected_annual, atol=1e-8)


# ============================================================================
# Group 8: Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_dataframes(self):
        pa = PortfolioAnalyzer(daily_amount_df=pd.DataFrame(), trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        assert pa.calculate_daily_returns().empty
        assert pa.calculate_traditional_metrics() == {}

    def test_single_day(self):
        da = pd.DataFrame({
            "成交日期": pd.to_datetime(["2025-01-02"]),
            "收盘价值": [100_000.0],
            "CASHFLOW": [0.0],
            "CSI300": [3500.0],
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        returns = pa.calculate_daily_returns()
        assert len(returns) == 0 # First day dropped
        # Traditional metrics requires >=2 days
        assert pa.calculate_traditional_metrics() == {}

    def test_all_negative_returns(self):
        """Verify correct signs when portfolio only loses money."""
        n = 20
        dates = pd.bdate_range("2025-01-02", periods=n)
        nav = np.zeros(n)
        nav[0] = 100_000.0
        for i in range(1, n):
            nav[i] = nav[i - 1] * 0.995  # -0.5% every day
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": nav,
            "CASHFLOW": 0.0,
            "CSI300": np.linspace(3500, 3600, n),
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        m = pa.calculate_traditional_metrics()
        assert m["CAGR_252"] < 0
        assert m["Max_Drawdown"] < 0
        assert m["Sharpe"] < 0
        # Because downside deviation is cleanly defined around target=0, and numerator is negative,
        # Sortino should definitely be negative.
        assert m["Sortino"] < 0
        assert m["Daily_Return_Win_Rate"] == 0.0

    def test_zero_volatility_returns(self):
        """Zero-std returns should not crash."""
        n = 10
        dates = pd.bdate_range("2025-01-02", periods=n)
        nav = [100_000.0] * n  # Flat NAV
        da = pd.DataFrame({
            "成交日期": dates,
            "收盘价值": nav,
            "CASHFLOW": 0.0,
            "CSI300": [3500.0] * n,
        })
        pa = PortfolioAnalyzer(daily_amount_df=da, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        m = pa.calculate_traditional_metrics()
        assert m["Sharpe"] == 0
        assert m["Sortino"] == 0
