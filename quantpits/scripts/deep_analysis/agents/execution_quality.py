"""
Execution Quality Agent.

Analyzes trading execution friction trends (delay cost, slippage, substitution bias)
across multiple time windows using the existing ExecutionAnalyzer.
"""

import os
import numpy as np
import pandas as pd
from ..base_agent import BaseAgent, AgentFindings, AnalysisContext, Finding


class ExecutionQualityAgent(BaseAgent):
    name = "Execution Quality"
    description = "Analyzes execution friction trends, substitution bias, and fee efficiency."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        trade_log = ctx.trade_log_df
        if trade_log.empty:
            return AgentFindings(self.name, ctx.window_label,
                                [self._make_finding('info', 'No trade data',
                                                    'Trade log is empty for this window.')],
                                [], {})

        # --- Initialize ExecutionAnalyzer ---
        try:
            from quantpits.scripts.analysis.execution_analyzer import ExecutionAnalyzer
            from quantpits.scripts.analysis.utils import init_qlib
            init_qlib()

            ea = ExecutionAnalyzer(
                trade_log_df=trade_log,
                start_date=ctx.start_date,
                end_date=ctx.end_date
            )
        except Exception as e:
            return AgentFindings(self.name, ctx.window_label,
                                [self._make_finding('warning', 'ExecutionAnalyzer init failed',
                                                    str(e))],
                                [], {'error': str(e)})

        # --- 1. Slippage & Delay Analysis ---
        try:
            friction_df = ea.calculate_slippage_and_delay()
            if friction_df is not None and not friction_df.empty:
                is_buy = friction_df['交易类别'].str.contains('买入', na=False)
                is_sell = friction_df['交易类别'].str.contains('卖出', na=False)

                buy_trades = friction_df[is_buy]
                sell_trades = friction_df[is_sell]

                # Volume-weighted metrics
                if not buy_trades.empty and buy_trades['成交金额'].sum() > 0:
                    buy_total_vol = buy_trades['成交金额'].sum()
                    buy_delay = float((buy_trades['Delay_Cost'] * buy_trades['成交金额']).sum() / buy_total_vol)
                    buy_slip = float((buy_trades['Exec_Slippage'] * buy_trades['成交金额']).sum() / buy_total_vol)
                    buy_total = buy_delay + buy_slip
                    buy_adv_mean = float(buy_trades['ADV_Participation_Rate'].dropna().mean()) if 'ADV_Participation_Rate' in buy_trades.columns else 0
                    buy_adv_max = float(buy_trades['ADV_Participation_Rate'].dropna().max()) if 'ADV_Participation_Rate' in buy_trades.columns else 0
                else:
                    buy_delay = buy_slip = buy_total = buy_adv_mean = buy_adv_max = 0

                if not sell_trades.empty and sell_trades['成交金额'].sum() > 0:
                    sell_total_vol = sell_trades['成交金额'].sum()
                    sell_delay = float((sell_trades['Delay_Cost'] * sell_trades['成交金额']).sum() / sell_total_vol)
                    sell_slip = float((sell_trades['Exec_Slippage'] * sell_trades['成交金额']).sum() / sell_total_vol)
                    sell_total = sell_delay + sell_slip
                else:
                    sell_delay = sell_slip = sell_total = 0

                raw_metrics['buy_total_friction'] = buy_total
                raw_metrics['buy_delay_cost'] = buy_delay
                raw_metrics['buy_exec_slippage'] = buy_slip
                raw_metrics['sell_total_friction'] = sell_total
                raw_metrics['sell_delay_cost'] = sell_delay
                raw_metrics['sell_exec_slippage'] = sell_slip
                raw_metrics['buy_adv_mean'] = buy_adv_mean
                raw_metrics['buy_adv_max'] = buy_adv_max

                findings.append(self._make_finding(
                    'info', f'Friction summary [{ctx.window_label}]',
                    f'Buy: total={buy_total*100:.3f}% '
                    f'(delay={buy_delay*100:.3f}%, slip={buy_slip*100:.3f}%). '
                    f'Sell: total={sell_total*100:.3f}% '
                    f'(delay={sell_delay*100:.3f}%, slip={sell_slip*100:.3f}%).',
                    raw_metrics
                ))

                # Alert on high friction
                if abs(buy_total) > 0.003:
                    findings.append(self._make_finding(
                        'warning', 'Elevated buy-side friction',
                        f'Total buy friction: {buy_total*100:.3f}% '
                        f'(Delay: {buy_delay*100:.3f}%, Slip: {buy_slip*100:.3f}%).',
                        {'buy_total': buy_total}
                    ))
                if abs(sell_total) > 0.003:
                    findings.append(self._make_finding(
                        'warning', 'Elevated sell-side friction',
                        f'Total sell friction: {sell_total*100:.3f}%.',
                        {'sell_total': sell_total}
                    ))

                # ADV capacity
                max_adv = max(buy_adv_max, 0)
                if max_adv > 0.01:
                    findings.append(self._make_finding(
                        'warning', 'ADV capacity concern',
                        f'Max ADV participation: {max_adv*100:.3f}%. '
                        f'Approaching market impact threshold.',
                        {'max_adv': max_adv}
                    ))
            else:
                findings.append(self._make_finding(
                    'info', 'Slippage data unavailable',
                    'Could not compute slippage — Qlib market data may be missing.',
                ))
        except Exception as e:
            raw_metrics['friction_error'] = str(e)
            findings.append(self._make_finding(
                'info', 'Friction analysis error',
                f'Could not compute friction metrics: {e}'))

        # --- 2. Explicit Costs ---
        try:
            costs = ea.analyze_explicit_costs()
            raw_metrics['explicit_costs'] = costs

            if costs:
                fee_ratio = costs.get('fee_ratio', 0)
                total_fees = costs.get('total_fees', 0)
                total_div = costs.get('total_dividend', 0)
                div_offset = total_div / total_fees if total_fees > 0 else 0

                raw_metrics['avg_fee_ratio'] = fee_ratio
                raw_metrics['dividend_offset_ratio'] = div_offset

                findings.append(self._make_finding(
                    'info', 'Fee efficiency',
                    f'Avg fee ratio: {fee_ratio*100:.4f}%. '
                    f'Total fees: ¥{total_fees:,.0f}. '
                    f'Dividends: ¥{total_div:,.0f} '
                    f'(offset: {div_offset*100:.1f}%).',
                    costs
                ))
        except Exception as e:
            raw_metrics['cost_error'] = str(e)

        # --- 3. Substitution Bias ---
        try:
            # analyze_order_discrepancies needs the order_history directory
            order_dirs = [
                os.path.join(ctx.workspace_root, 'data', 'order_history'),
                os.path.join(ctx.workspace_root, 'output'),
            ]
            for order_dir in order_dirs:
                if os.path.isdir(order_dir):
                    sub_result = ea.analyze_order_discrepancies(order_dir)
                    if sub_result:
                        raw_metrics['substitution_bias'] = sub_result
                        theo_bias = sub_result.get('theoretical_substitute_bias_impact', 0)
                        real_bias = sub_result.get('realized_substitute_bias_impact', 0)
                        n_missed = sub_result.get('total_missed_count', 0)
                        n_sub = sub_result.get('total_substitute_count', 0)

                        findings.append(self._make_finding(
                            'info', 'Substitution bias',
                            f'Theoretical impact: {theo_bias*100:.2f}%. '
                            f'Realized impact: {real_bias*100:.2f}%. '
                            f'Missed buys: {n_missed}. Substitutes: {n_sub}.',
                            sub_result
                        ))

                        if abs(real_bias) > 0.02:
                            findings.append(self._make_finding(
                                'warning', 'Significant substitution bias',
                                f'Realized substitution bias impact: {real_bias*100:.2f}%. '
                                f'Missed {n_missed} top buys.',
                                sub_result
                            ))
                            recommendations.append(
                                "Substitution bias is significant. Consider pre-market "
                                "limit orders for top-ranked buy suggestions."
                            )
                        break  # Use first directory that returns results
        except Exception as e:
            raw_metrics['substitution_bias_error'] = str(e)

        # --- 4. Execution Timing TODO ---
        findings.append(self._make_finding(
            'info', 'Execution timing analysis: TODO',
            'Execution timing analysis pending granular intraday timestamp data. '
            'Consider recording precise execution timestamps in future trade logs.',
            {'status': 'deferred'}
        ))

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)
