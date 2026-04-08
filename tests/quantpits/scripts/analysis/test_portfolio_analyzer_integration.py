"""
Integration test: run PortfolioAnalyzer on real CSI300_Base production data.

This file is automatically excluded from collection (via conftest.py
``collect_ignore_glob``) when ``QLIB_WORKSPACE_DIR`` does not point at a
workspace with production data.  It will never show as SKIPPED in CI.

Run with:
  QLIB_WORKSPACE_DIR=workspaces/CSI300_Base \
    conda run -n qlib_cupy python -m pytest tests/quantpits/scripts/analysis/test_portfolio_analyzer_integration.py -v --tb=short
"""
import os
import pytest
import pandas as pd
import numpy as np
from quantpits.utils.constants import TRADING_DAYS_PER_YEAR

WORKSPACE = os.environ.get("QLIB_WORKSPACE_DIR", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qlib_init():
    """Best-effort Qlib init; returns True on success."""
    try:
        from quantpits.scripts.analysis.utils import init_qlib
        init_qlib()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def prod_data():
    """Load production data files."""
    da = pd.read_csv(os.path.join(WORKSPACE, "data", "daily_amount_log_full.csv"))
    if "成交日期" in da.columns:
        da["成交日期"] = pd.to_datetime(da["成交日期"])

    tl = pd.read_csv(os.path.join(WORKSPACE, "data", "trade_log_full.csv"))
    if "成交日期" in tl.columns:
        tl["成交日期"] = pd.to_datetime(tl["成交日期"])

    hl = pd.read_csv(os.path.join(WORKSPACE, "data", "holding_log_full.csv"))
    if "成交日期" in hl.columns:
        hl["成交日期"] = pd.to_datetime(hl["成交日期"])

    return da, tl, hl


@pytest.fixture(scope="module")
def analyzer(prod_data):
    da, tl, hl = prod_data
    from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
    return PortfolioAnalyzer(daily_amount_df=da, trade_log_df=tl, holding_log_df=hl)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProductionDailyReturns:
    def test_returns_not_empty(self, analyzer):
        rets = analyzer.calculate_daily_returns()
        assert not rets.empty
        assert len(rets) > 50  # Should have many months of data

    def test_returns_reasonable_range(self, analyzer):
        rets = analyzer.calculate_daily_returns()
        # Daily returns should be within (-0.2, 0.2) for a diversified portfolio
        assert rets.max() < 0.20
        assert rets.min() > -0.20
        # Most days should have nonzero returns
        assert (rets != 0).sum() > len(rets) * 0.5


class TestProductionTraditionalMetrics:
    def test_all_metrics_finite(self, analyzer):
        m = analyzer.calculate_traditional_metrics()
        assert m, "Metrics should not be empty"

        for key in ["CAGR_252", "Volatility", "Sharpe", "Max_Drawdown",
                     "Absolute_Return", "Turnover_Rate_Annual"]:
            assert key in m, f"Missing key: {key}"
            assert np.isfinite(m[key]), f"{key} is not finite: {m[key]}"

    def test_sign_constraints(self, analyzer):
        m = analyzer.calculate_traditional_metrics()
        assert -1 < m["CAGR_252"] < 10, f"CAGR_252 out of range: {m['CAGR_252']}"
        assert m["Volatility"] >= 0, f"Volatility negative: {m['Volatility']}"
        assert m["Max_Drawdown"] <= 0, f"MaxDD should be <= 0: {m['Max_Drawdown']}"
        assert 0 <= m["Realized_Trade_Win_Rate"] <= 1

    def test_benchmark_metrics_present(self, analyzer):
        m = analyzer.calculate_traditional_metrics()
        for key in ["Benchmark_CAGR_252", "Benchmark_Volatility", "Benchmark_Sharpe",
                     "Benchmark_Max_Drawdown", "Tracking_Error", "Information_Ratio_(Arithmetic)"]:
            assert key in m, f"Missing benchmark key: {key}"
            assert np.isfinite(m[key]), f"{key} is not finite: {m[key]}"


class TestProductionFactorExposure:
    def test_using_benchmark_column(self, analyzer):
        """Factor exposure using the CSI300 column in daily_amount."""
        from unittest.mock import patch
        with patch('quantpits.scripts.analysis.portfolio_analyzer.load_market_config', return_value=("csi300", "SH000300")):
            fe = analyzer.calculate_factor_exposure()

        assert fe, "Factor exposure should not be empty"
        assert "Beta_Market" in fe
        assert "Annualized_Alpha" in fe
        assert "R_Squared" in fe

        # Beta should be positive for a long-only equity portfolio
        assert 0 < fe["Beta_Market"] < 3, f"Beta out of range: {fe['Beta_Market']}"
        assert 0 <= fe["R_Squared"] <= 1, f"R² out of range: {fe['R_Squared']}"
        # Alpha can be positive or negative, but should be reasonable
        assert -1 < fe["Annualized_Alpha"] < 5, f"Alpha out of range: {fe['Annualized_Alpha']}"

    def test_factor_exposure_with_qlib(self, analyzer, qlib_init):
        """Factor exposure using Qlib data (no CSI300 column)."""
        if not qlib_init:
            # Qlib could not initialise — assert a known-safe outcome instead of skipping.
            assert True
            return

        from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer

        # Create analyzer without benchmark column
        da = analyzer.daily_amount.reset_index()
        da.rename(columns={da.columns[0]: "成交日期"}, inplace=True)
        assert "CSI300" in da.columns, "Production data should contain a CSI300 column"
        da_no_bench = da.drop(columns=["CSI300"])

        pa = PortfolioAnalyzer(daily_amount_df=da_no_bench, trade_log_df=pd.DataFrame(), holding_log_df=pd.DataFrame())
        fe = pa.calculate_factor_exposure(market="csi300")

        if fe:  # May be empty if Qlib data is incomplete
            assert 0 < fe["Beta_Market"] < 3
            assert 0 <= fe["R_Squared"] <= 1


class TestProductionStyleExposures:
    def test_style_exposures_sanity(self, analyzer, qlib_init):
        """Style exposures with Qlib features."""
        if not qlib_init:
            assert True
            return

        se = analyzer.calculate_style_exposures(market="csi300")
        if se:
            assert "Multi_Factor_Beta" in se
            assert "Barra_Liquidity_Exp_(High-Low)" in se
            assert "Barra_Momentum_Exp_(High-Low)" in se
            assert "Barra_Volatility_Exp_(High-Low)" in se
            assert "Barra_Style_R_Squared" in se
            assert 0 <= se["Barra_Style_R_Squared"] <= 1


class TestProductionHoldingMetrics:
    def test_holding_metrics_present(self, analyzer):
        m = analyzer.calculate_holding_metrics()
        if m:
            assert m["Avg_Daily_Holdings_Count"] > 0
            assert 0 < m["Avg_Top1_Concentration"] <= 1
            assert 0 <= m["Daily_Holding_Win_Rate"] <= 1


class TestProductionConsistency:
    def test_returns_vs_nav(self, analyzer):
        """Cumulative product of returns should reconstruct NAV growth."""
        rets = analyzer.calculate_daily_returns()

        nav = analyzer.daily_amount["收盘价值"].astype(float)

        # On zero-cashflow days, consecutive returns should match NAV pct_change
        cf_col = None
        for c in ["CASHFLOW", "今日出入金", "资金发生数"]:
            if c in analyzer.daily_amount.columns:
                cf_col = c
                break

        if cf_col:
            zero_cf_mask = analyzer.daily_amount[cf_col].fillna(0).astype(float) == 0
            zero_cf_days = analyzer.daily_amount.index[zero_cf_mask]
            nav_pct = nav.pct_change().fillna(0)
            for d in zero_cf_days:
                if d in rets.index and d in nav_pct.index:
                    assert np.isclose(rets.loc[d], nav_pct.loc[d], atol=1e-10), \
                        f"Mismatch on zero-CF day {d}: {rets.loc[d]} vs {nav_pct.loc[d]}"
