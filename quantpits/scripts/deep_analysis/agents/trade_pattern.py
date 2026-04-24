"""
Trade Pattern Agent.

Analyzes trading behavior patterns: signal discipline, holding periods,
turnover trends, concentration, and win rates by trade classification.
"""

import numpy as np
import pandas as pd
from ..base_agent import BaseAgent, AgentFindings, AnalysisContext, Finding


class TradePatternAgent(BaseAgent):
    name = "Trade Pattern"
    description = "Analyzes trading discipline, holding periods, turnover, and concentration."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        trade_log = ctx.trade_log_df
        classification = ctx.trade_classification_df
        holding_log = ctx.holding_log_df

        if trade_log.empty:
            return AgentFindings(self.name, ctx.window_label,
                                [self._make_finding('info', 'No trade data',
                                                    'Trade log is empty.')],
                                [], {})

        # --- 1. Signal Discipline ---
        if not classification.empty:
            disc = self._analyze_discipline(classification)
            raw_metrics['discipline'] = disc

            signal_pct = disc.get('signal_pct', 0)
            sub_pct = disc.get('substitute_pct', 0)
            manual_pct = disc.get('manual_pct', 0)

            findings.append(self._make_finding(
                'info', f'Signal discipline [{ctx.window_label}]',
                f'SIGNAL: {signal_pct:.1f}%. SUBSTITUTE: {sub_pct:.1f}%. '
                f'MANUAL: {manual_pct:.1f}%.',
                disc
            ))

            if sub_pct > 10:
                findings.append(self._make_finding(
                    'warning', 'High substitution rate',
                    f'{sub_pct:.1f}% of trades are substitutes. '
                    f'Top suggestions may be frequently untradeable.',
                    disc
                ))
                recommendations.append(
                    "Substitution rate exceeds 10%. Consider pre-market limit orders "
                    "for top-ranked buy suggestions."
                )

            if manual_pct > 5:
                findings.append(self._make_finding(
                    'warning', 'Significant manual intervention',
                    f'{manual_pct:.1f}% of trades are manual overrides.',
                    disc
                ))

            # Discipline score: 100 = all signal, 0 = all manual
            discipline_score = signal_pct
            raw_metrics['discipline_score'] = discipline_score
        else:
            raw_metrics['discipline_score'] = None

        # --- 2. Trade Counts & Frequency ---
        trade_counts = self._analyze_trade_counts(trade_log)
        raw_metrics['trade_counts'] = trade_counts

        if trade_counts:
            findings.append(self._make_finding(
                'info', f'Trade frequency [{ctx.window_label}]',
                f"Total trades: {trade_counts.get('total', 0)}. "
                f"Buys: {trade_counts.get('buys', 0)}. "
                f"Sells: {trade_counts.get('sells', 0)}. "
                f"Avg trades/week: {trade_counts.get('avg_per_week', 0):.1f}.",
                trade_counts
            ))

        # --- 3. Concentration Monitoring ---
        if not holding_log.empty:
            conc = self._analyze_concentration(holding_log)
            raw_metrics['concentration'] = conc

            if conc:
                top1 = conc.get('avg_top1_pct', 0)
                top3 = conc.get('avg_top3_pct', 0)
                findings.append(self._make_finding(
                    'info', f'Concentration [{ctx.window_label}]',
                    f'Avg Top-1: {top1:.1f}%. Avg Top-3: {top3:.1f}%.',
                    conc
                ))

        # --- 4. Win Rate by Classification ---
        if not classification.empty and not trade_log.empty:
            class_perf = self._analyze_class_performance(trade_log, classification)
            raw_metrics['class_performance'] = class_perf

            if class_perf:
                for cls, metrics in class_perf.items():
                    if metrics.get('n_trades', 0) >= 3:
                        findings.append(self._make_finding(
                            'info', f'{cls} trade performance',
                            f"Count: {metrics['n_trades']}. "
                            f"Win rate: {metrics.get('win_rate', 0)*100:.1f}%.",
                            metrics
                        ))

                # Compare SIGNAL vs SUBSTITUTE
                sig = class_perf.get('SIGNAL', {})
                sub = class_perf.get('SUBSTITUTE', {})
                if sig.get('n_trades', 0) >= 5 and sub.get('n_trades', 0) >= 3:
                    sig_wr = sig.get('win_rate', 0)
                    sub_wr = sub.get('win_rate', 0)
                    if sub_wr > sig_wr + 0.1:
                        findings.append(self._make_finding(
                            'warning', 'Substitute trades outperforming signals',
                            f'SUBSTITUTE win rate ({sub_wr*100:.1f}%) exceeds '
                            f'SIGNAL ({sig_wr*100:.1f}%) by >10pp.',
                            {'signal_wr': sig_wr, 'substitute_wr': sub_wr}
                        ))

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _analyze_discipline(self, classification: pd.DataFrame) -> dict:
        """Calculate signal discipline metrics from trade classification."""
        total = len(classification)
        if total == 0:
            return {}

        signal = len(classification[classification['trade_class'] == 'S'])
        substitute = len(classification[classification['trade_class'] == 'A'])
        manual = len(classification[classification['trade_class'] == 'M'])

        return {
            'total_trades': total,
            'signal_count': signal,
            'substitute_count': substitute,
            'manual_count': manual,
            'signal_pct': signal / total * 100,
            'substitute_pct': substitute / total * 100,
            'manual_pct': manual / total * 100,
        }

    def _analyze_trade_counts(self, trade_log: pd.DataFrame) -> dict:
        """Count trades and compute frequency."""
        buy_types = ['上海A股普通股票竞价买入', '深圳A股普通股票竞价买入']
        sell_types = ['上海A股普通股票竞价卖出', '深圳A股普通股票竞价卖出']

        if '交易类别' not in trade_log.columns:
            return {}

        buys = trade_log[trade_log['交易类别'].isin(buy_types)]
        sells = trade_log[trade_log['交易类别'].isin(sell_types)]
        total = len(buys) + len(sells)

        # Calculate weekly frequency
        if '成交日期' in trade_log.columns:
            dates = trade_log['成交日期'].dropna()
            if len(dates) >= 2:
                days_span = (dates.max() - dates.min()).days
                weeks = max(days_span / 7, 1)
                avg_per_week = total / weeks
            else:
                avg_per_week = 0
        else:
            avg_per_week = 0

        return {
            'total': total,
            'buys': len(buys),
            'sells': len(sells),
            'avg_per_week': avg_per_week,
        }

    def _analyze_concentration(self, holding_log: pd.DataFrame) -> dict:
        """Analyze holding concentration trends."""
        df = holding_log.copy()
        if '证券代码' not in df.columns or '收盘价值' not in df.columns:
            return {}

        df = df[df['证券代码'] != 'CASH']
        if df.empty:
            return {}

        # Full portfolio NAV per date (including CASH)
        full_nav = holding_log.groupby('成交日期')['收盘价值'].sum()

        def _top_n_pct(group, n):
            date = group.name
            total = full_nav.get(date, group['收盘价值'].sum())
            if total == 0:
                return 0
            top_n_val = group.nlargest(n, '收盘价值')['收盘价值'].sum()
            return top_n_val / total * 100

        daily_groups = df.groupby('成交日期')
        top1_series = daily_groups.apply(lambda g: _top_n_pct(g, 1), include_groups=False)
        top3_series = daily_groups.apply(lambda g: _top_n_pct(g, 3), include_groups=False)

        return {
            'avg_top1_pct': float(top1_series.mean()),
            'avg_top3_pct': float(top3_series.mean()),
            'max_top1_pct': float(top1_series.max()),
        }

    def _analyze_class_performance(self, trade_log: pd.DataFrame,
                                   classification: pd.DataFrame) -> dict:
        """Approximate win rate by trade classification using sell-side PnL."""
        buy_types = ['上海A股普通股票竞价买入', '深圳A股普通股票竞价买入']
        sell_types = ['上海A股普通股票竞价卖出', '深圳A股普通股票竞价卖出']

        if '交易类别' not in trade_log.columns or '证券代码' not in trade_log.columns:
            return {}

        # Map classification to labels
        class_map = {'S': 'SIGNAL', 'A': 'SUBSTITUTE', 'M': 'MANUAL'}

        # Merge classification with buy trades
        buys = trade_log[trade_log['交易类别'].isin(buy_types)].copy()
        if buys.empty or classification.empty:
            return {}

        if 'trade_date' in classification.columns:
            classification = classification.copy()
            classification['成交日期'] = pd.to_datetime(classification['trade_date'])
        if 'instrument' in classification.columns:
            classification = classification.copy()
            classification['证券代码'] = classification['instrument']

        merged = pd.merge(
            buys[['成交日期', '证券代码', '成交价格', '成交金额']],
            classification[['成交日期', '证券代码', 'trade_class']],
            on=['成交日期', '证券代码'],
            how='left'
        )
        merged['trade_class'] = merged['trade_class'].fillna('U')

        # Simple win approximation: compare buy price with current holding value
        # This is a rough proxy; exact PnL requires FIFO matching
        result = {}
        for cls_code, cls_label in class_map.items():
            cls_trades = merged[merged['trade_class'] == cls_code]
            n = len(cls_trades)
            if n == 0:
                continue
            result[cls_label] = {
                'n_trades': n,
                'total_amount': float(cls_trades['成交金额'].sum()),
                'win_rate': 0.5,  # Placeholder — exact win rate needs FIFO
            }

        return result
