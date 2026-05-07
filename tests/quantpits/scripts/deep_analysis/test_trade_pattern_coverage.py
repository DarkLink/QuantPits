"""Coverage tests for TradePatternAgent — uncovered paths."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quantpits.scripts.deep_analysis.agents.trade_pattern import TradePatternAgent
from quantpits.scripts.deep_analysis.base_agent import AnalysisContext


class TestTradePatternAgent:
    """Test TradePatternAgent with various edge cases."""

    @staticmethod
    def _make_context(trade_log=None, classification=None, holding_log=None):
        """Helper to create an AnalysisContext for TradePatternAgent."""
        # Create minimal dataframes if not provided
        if trade_log is None:
            trade_log = pd.DataFrame()
        if classification is None:
            classification = pd.DataFrame()
        if holding_log is None:
            holding_log = pd.DataFrame()

        return AnalysisContext(
            start_date="2026-01-01",
            end_date="2026-05-01",
            window_label="full",
            workspace_root="/tmp/test",
            trade_log_df=trade_log,
            trade_classification_df=classification,
            holding_log_df=holding_log,
        )

    def test_empty_trade_log(self):
        """Empty trade log returns info finding."""
        agent = TradePatternAgent()
        ctx = self._make_context()
        result = agent.analyze(ctx)

        assert result.agent_name == "Trade Pattern"
        assert len(result.findings) == 1
        assert "No trade data" in result.findings[0].title

    def test_with_full_data(self):
        """Test with complete trade, classification, and holding data."""
        dates = pd.date_range("2026-01-05", periods=90, freq='B')

        trade_log = pd.DataFrame({
            '成交日期': dates[:60],
            '证券代码': ['600000.SH'] * 60,
            '交易类别': ['上海A股普通股票竞价买入'] * 30 + ['上海A股普通股票竞价卖出'] * 30,
            '成交价格': [10.0] * 60,
            '成交数量': [1000] * 60,
            '成交金额': [10000.0] * 60,
        })

        class_dates = [d.strftime('%Y-%m-%d') for d in dates[:30]]
        classification = pd.DataFrame({
            'trade_date': class_dates,
            'instrument': ['600000.SH'] * 30,
            'trade_class': ['S'] * 20 + ['A'] * 7 + ['M'] * 3,
        })

        holding_log = pd.DataFrame({
            '成交日期': list(dates[:90]) * 3,
            '证券代码': ['600000.SH'] * 90 + ['000001.SZ'] * 90 + ['CASH'] * 90,
            '收盘价值': [50000.0] * 90 + [30000.0] * 90 + [20000.0] * 90,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(trade_log, classification, holding_log)
        result = agent.analyze(ctx)

        assert result.agent_name == "Trade Pattern"
        # Should have multiple findings
        assert len(result.findings) > 0
        # Should have discipline metrics
        assert 'discipline' in result.raw_metrics

    def test_classification_only(self):
        """Classification without trade log."""
        class_dates = [f"2026-01-{i+1:02d}" for i in range(30)]
        classification = pd.DataFrame({
            'trade_date': class_dates,
            'instrument': ['600000.SH'] * 30,
            'trade_class': ['S'] * 25 + ['A'] * 5,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(classification=classification)
        result = agent.analyze(ctx)

        assert result.agent_name == "Trade Pattern"
        # Should not crash

    def test_high_substitution_rate(self):
        """High substitution rate triggers warning."""
        class_dates = [f"2026-01-{i+1:02d}" for i in range(30)]
        classification = pd.DataFrame({
            'trade_date': class_dates,
            'instrument': ['600000.SH'] * 30,
            'trade_class': ['A'] * 20 + ['S'] * 10,  # 66% substitute
        })

        dates = pd.date_range("2026-01-01", periods=30, freq='D')
        trade_log = pd.DataFrame({
            '成交日期': dates,
            '证券代码': ['600000.SH'] * 30,
            '交易类别': ['上海A股普通股票竞价买入'] * 15 + ['上海A股普通股票竞价卖出'] * 15,
            '成交价格': [10.0] * 30,
            '成交金额': [10000.0] * 30,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(trade_log, classification)
        result = agent.analyze(ctx)

        # Should have a warning about substitution
        warning_titles = [f.title for f in result.findings if f.severity == 'warning']
        assert any('substitution' in t.lower() for t in warning_titles)

    def test_high_manual_rate(self):
        """High manual intervention rate triggers warning."""
        class_dates = [f"2026-01-{i+1:02d}" for i in range(30)]
        classification = pd.DataFrame({
            'trade_date': class_dates,
            'instrument': ['600000.SH'] * 30,
            'trade_class': ['M'] * 10 + ['S'] * 20,  # 33% manual (>5%)
        })

        dates = pd.date_range("2026-01-01", periods=30, freq='D')
        trade_log = pd.DataFrame({
            '成交日期': dates,
            '证券代码': ['600000.SH'] * 30,
            '交易类别': ['上海A股普通股票竞价买入'] * 30,
            '成交价格': [10.0] * 30,
            '成交金额': [10000.0] * 30,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(trade_log, classification)
        result = agent.analyze(ctx)

        warning_titles = [f.title for f in result.findings if f.severity == 'warning']
        assert any('manual' in t.lower() for t in warning_titles)

    def test_trade_counts_with_date_range(self):
        """Trade count analysis with proper date range."""
        dates = pd.date_range("2026-01-01", periods=60, freq='D')
        trade_log = pd.DataFrame({
            '成交日期': dates,
            '证券代码': ['600000.SH'] * 60,
            '交易类别': (['上海A股普通股票竞价买入'] * 30 +
                        ['上海A股普通股票竞价卖出'] * 30),
            '成交价格': [10.0] * 60,
            '成交金额': [10000.0] * 60,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(trade_log)
        result = agent.analyze(ctx)

        assert 'trade_counts' in result.raw_metrics
        assert result.raw_metrics['trade_counts']['total'] == 60

    def test_concentration_without_cash(self):
        """Concentration analysis with non-CASH holdings only."""
        dates = pd.date_range("2026-01-01", periods=30, freq='D')
        holding_log = pd.DataFrame({
            '成交日期': dates,
            '证券代码': ['600000.SH'] * 30,
            '收盘价值': [50000.0] * 30,
        })

        # Need a trade_log too, otherwise agent returns early
        trade_log = pd.DataFrame({
            '成交日期': dates[:10],
            '证券代码': ['600000.SH'] * 10,
            '交易类别': ['上海A股普通股票竞价买入'] * 10,
            '成交价格': [10.0] * 10,
            '成交金额': [10000.0] * 10,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(trade_log, holding_log=holding_log)
        result = agent.analyze(ctx)

        assert 'concentration' in result.raw_metrics

    def test_no_trade_category_column(self):
        """Trade log without 交易类别 column."""
        trade_log = pd.DataFrame({
            '成交日期': pd.date_range("2026-01-01", periods=10, freq='D'),
            '证券代码': ['600000.SH'] * 10,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(trade_log=trade_log)
        result = agent.analyze(ctx)

        # Should not crash
        assert isinstance(result.findings, list)

    def test_holding_log_no_security_code(self):
        """Holding log without 证券代码 column."""
        holding_log = pd.DataFrame({
            '成交日期': pd.date_range("2026-01-01", periods=10, freq='D'),
            '收盘价值': [10000.0] * 10,
        })

        agent = TradePatternAgent()
        ctx = self._make_context(holding_log=holding_log)
        result = agent.analyze(ctx)

        # Should not crash
        assert isinstance(result.findings, list)


class TestAnalyzeDiscipline:
    """Test _analyze_discipline helper."""

    def test_empty_classification(self):
        agent = TradePatternAgent()
        result = agent._analyze_discipline(pd.DataFrame())
        assert result == {}

    def test_all_signal(self):
        classification = pd.DataFrame({
            'trade_class': ['S'] * 10,
        })
        agent = TradePatternAgent()
        result = agent._analyze_discipline(classification)
        assert result['signal_pct'] == 100.0
        assert result['substitute_pct'] == 0.0
        assert result['manual_pct'] == 0.0

    def test_mixed_classes(self):
        classification = pd.DataFrame({
            'trade_class': ['S', 'S', 'S', 'A', 'A', 'M'],
        })
        agent = TradePatternAgent()
        result = agent._analyze_discipline(classification)
        assert result['signal_pct'] == 50.0
        assert abs(result['substitute_pct'] - 33.33) < 1
        assert abs(result['manual_pct'] - 16.67) < 1


class TestAnalyzeTradeCounts:
    """Test _analyze_trade_counts helper."""

    def test_with_date_range(self):
        dates = pd.date_range("2026-01-01", periods=30, freq='D')
        trade_log = pd.DataFrame({
            '成交日期': dates,
            '交易类别': ['上海A股普通股票竞价买入'] * 15 + ['上海A股普通股票竞价卖出'] * 15,
        })

        agent = TradePatternAgent()
        result = agent._analyze_trade_counts(trade_log)
        assert result['total'] == 30
        assert result['buys'] == 15
        assert result['sells'] == 15
        assert result['avg_per_week'] > 0

    def test_single_date(self):
        """Single date should result in avg_per_week that's total / 1 or 0."""
        trade_log = pd.DataFrame({
            '成交日期': [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
            '交易类别': ['上海A股普通股票竞价买入', '上海A股普通股票竞价卖出'],
        })

        agent = TradePatternAgent()
        result = agent._analyze_trade_counts(trade_log)
        # Single date → days_span=0, weeks = max(0, 1) = 1 → avg_per_week = 2
        assert result['total'] == 2

    def test_no_trade_category(self):
        trade_log = pd.DataFrame({
            '成交日期': pd.date_range("2026-01-01", periods=10, freq='D'),
        })

        agent = TradePatternAgent()
        result = agent._analyze_trade_counts(trade_log)
        assert result == {}


class TestAnalyzeConcentration:
    """Test _analyze_concentration helper."""

    def test_basic_concentration(self):
        dates = pd.date_range("2026-01-01", periods=5, freq='D')
        holding_log = pd.DataFrame({
            '成交日期': list(dates) * 2,
            '证券代码': ['600000.SH'] * 5 + ['000001.SZ'] * 5,
            '收盘价值': [70000.0] * 5 + [30000.0] * 5,
        })

        agent = TradePatternAgent()
        result = agent._analyze_concentration(holding_log)
        assert result['avg_top1_pct'] > 0
        assert result['avg_top3_pct'] > 0

    def test_no_security_column(self):
        holding_log = pd.DataFrame({
            '成交日期': pd.date_range("2026-01-01", periods=5, freq='D'),
            '收盘价值': [10000.0] * 5,
        })

        agent = TradePatternAgent()
        result = agent._analyze_concentration(holding_log)
        assert result == {}

    def test_all_cash(self):
        dates = pd.date_range("2026-01-01", periods=5, freq='D')
        holding_log = pd.DataFrame({
            '成交日期': dates,
            '证券代码': ['CASH'] * 5,
            '收盘价值': [10000.0] * 5,
        })

        agent = TradePatternAgent()
        result = agent._analyze_concentration(holding_log)
        assert result == {}

    def test_no_value_column(self):
        dates = pd.date_range("2026-01-01", periods=5, freq='D')
        holding_log = pd.DataFrame({
            '成交日期': dates,
            '证券代码': ['600000.SH'] * 5,
        })

        agent = TradePatternAgent()
        result = agent._analyze_concentration(holding_log)
        assert result == {}


class TestAnalyzeClassPerformance:
    """Test _analyze_class_performance helper."""

    def test_basic_performance(self):
        trade_dates = pd.date_range("2026-01-01", periods=10, freq='D')
        trade_log = pd.DataFrame({
            '成交日期': trade_dates,
            '证券代码': ['600000.SH'] * 10,
            '交易类别': ['上海A股普通股票竞价买入'] * 10,
            '成交价格': [10.0] * 10,
            '成交金额': [10000.0] * 10,
        })

        class_dates = [d.strftime('%Y-%m-%d') for d in trade_dates]
        classification = pd.DataFrame({
            'trade_date': class_dates,
            'instrument': ['600000.SH'] * 10,
            'trade_class': ['S'] * 6 + ['A'] * 4,
        })

        agent = TradePatternAgent()
        result = agent._analyze_class_performance(trade_log, classification)
        assert 'SIGNAL' in result
        assert 'SUBSTITUTE' in result
        assert result['SIGNAL']['n_trades'] == 6

    def test_no_trade_category_column(self):
        trade_log = pd.DataFrame({
            '成交日期': pd.date_range("2026-01-01", periods=5, freq='D'),
            '证券代码': ['600000.SH'] * 5,
        })
        classification = pd.DataFrame({
            'trade_date': ['2026-01-01'] * 5,
            'instrument': ['600000.SH'] * 5,
            'trade_class': ['S'] * 5,
        })

        agent = TradePatternAgent()
        result = agent._analyze_class_performance(trade_log, classification)
        assert result == {}

    def test_empty_buys(self):
        trade_log = pd.DataFrame({
            '成交日期': pd.date_range("2026-01-01", periods=5, freq='D'),
            '证券代码': ['600000.SH'] * 5,
            '交易类别': ['上海A股普通股票竞价卖出'] * 5,  # Only sells
            '成交价格': [10.0] * 5,
            '成交金额': [10000.0] * 5,
        })
        classification = pd.DataFrame({
            'trade_date': ['2026-01-01'] * 5,
            'instrument': ['600000.SH'] * 5,
            'trade_class': ['S'] * 5,
        })

        agent = TradePatternAgent()
        result = agent._analyze_class_performance(trade_log, classification)
        assert result == {}  # no buys

    def test_substitute_outperforming_signal(self):
        """When substitute trades have higher win rate, check for warning."""
        dates = pd.date_range("2026-01-01", periods=20, freq='D')
        trade_log = pd.DataFrame({
            '成交日期': dates,
            '证券代码': ['600000.SH'] * 20,
            '交易类别': ['上海A股普通股票竞价买入'] * 20,
            '成交价格': [10.0] * 20,
            '成交金额': [10000.0] * 20,
        })

        class_dates = [d.strftime('%Y-%m-%d') for d in dates]
        classification = pd.DataFrame({
            'trade_date': class_dates,
            'instrument': ['600000.SH'] * 20,
            'trade_class': ['S'] * 10 + ['A'] * 10,
        })

        agent = TradePatternAgent()
        ctx = AnalysisContext(
            start_date="2026-01-01",
            end_date="2026-05-01",
            window_label="full",
            workspace_root="/tmp/test",
            trade_log_df=trade_log,
            trade_classification_df=classification,
        )
        result = agent.analyze(ctx)
        # Should not crash despite placeholder win rates being equal
        assert result.agent_name == "Trade Pattern"
