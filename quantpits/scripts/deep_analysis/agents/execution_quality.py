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

        # --- 4. Execution Timing ---
        try:
            import pandas as pd
            order_log_path = os.path.join(ctx.workspace_root, 'data', 'raw_order_log_full.csv')
            trade_log_path = os.path.join(ctx.workspace_root, 'data', 'raw_trade_log_full.csv')
            
            if os.path.exists(order_log_path) and os.path.exists(trade_log_path):
                # Load data
                order_df = pd.read_csv(order_log_path)
                trade_df = pd.read_csv(trade_log_path)
                
                # Create datetime strings for filtering
                order_df['datetime'] = pd.to_datetime(order_df['委托日期'].astype(str) + ' ' + order_df['委托时间'])
                
                if '成交日期' in trade_df.columns:
                    trade_df['datetime'] = pd.to_datetime(trade_df['成交日期'].astype(str) + ' ' + trade_df['成交时间'])
                elif '日期' in trade_df.columns:
                    trade_df['datetime'] = pd.to_datetime(trade_df['日期'].astype(str) + ' ' + trade_df['成交时间'])
                else:
                    trade_df['datetime'] = pd.to_datetime(trade_df['成交日期'].astype(str) + ' ' + trade_df['成交时间'])
                    
                # Filter by window dates
                if ctx.start_date:
                    order_df = order_df[order_df['datetime'] >= pd.to_datetime(ctx.start_date)]
                    trade_df = trade_df[trade_df['datetime'] >= pd.to_datetime(ctx.start_date)]
                if ctx.end_date:
                    # Make sure end of day is included by moving to the next day if we are at 00:00:00
                    order_df = order_df[order_df['datetime'] <= pd.to_datetime(ctx.end_date) + pd.Timedelta(days=1)]
                    trade_df = trade_df[trade_df['datetime'] <= pd.to_datetime(ctx.end_date) + pd.Timedelta(days=1)]
                    
                if not order_df.empty:
                    # Fill Rate & Cancel Rate
                    total_order = order_df['委托数量'].sum()
                    total_fill = order_df['成交数量'].sum()
                    total_cancel = order_df['撤单数量'].sum()
                    
                    fill_rate = float(total_fill / total_order) if total_order > 0 else 0.0
                    cancel_rate = float(total_cancel / total_order) if total_order > 0 else 0.0
                    
                    raw_metrics['fill_rate'] = fill_rate
                    raw_metrics['cancel_rate'] = cancel_rate
                    
                    # Latency Calculation using merge_asof
                    order_df['dir'] = order_df['交易类别'].apply(lambda x: 'buy' if '买入' in str(x) else ('sell' if '卖出' in str(x) else 'other'))
                    trade_df['dir'] = trade_df['交易类别'].apply(lambda x: 'buy' if '买入' in str(x) else ('sell' if '卖出' in str(x) else 'other'))
                    
                    # Ensure symbol formatting is consistent (as string)
                    order_df['证券代码'] = order_df['证券代码'].astype(str).str.zfill(6)
                    trade_df['证券代码'] = trade_df['证券代码'].astype(str).str.zfill(6)
                    
                    # Keep only buy/sell
                    order_df = order_df[order_df['dir'].isin(['buy', 'sell'])].sort_values('datetime')
                    trade_df = trade_df[trade_df['dir'].isin(['buy', 'sell'])].sort_values('datetime')
                    
                    # Filter needed columns
                    o_sub = order_df[['datetime', '证券代码', 'dir']].rename(columns={'datetime': 'order_time'})
                    t_sub = trade_df[['datetime', '证券代码', 'dir']].rename(columns={'datetime': 'trade_time'})
                    
                    if not o_sub.empty and not t_sub.empty:
                        # merge each trade to the latest order before it
                        merged = pd.merge_asof(
                            t_sub, o_sub,
                            left_on='trade_time',
                            right_on='order_time',
                            by=['证券代码', 'dir'],
                            direction='backward'
                        )
                        
                        merged['latency'] = (merged['trade_time'] - merged['order_time']).dt.total_seconds()
                        valid_latency = merged['latency'].dropna()
                        valid_latency = valid_latency[valid_latency >= 0]
                        
                        if not valid_latency.empty:
                            lat_mean = float(valid_latency.mean())
                            lat_median = float(valid_latency.median())
                            lat_p90 = float(valid_latency.quantile(0.9))
                            
                            raw_metrics['latency_mean_sec'] = lat_mean
                            raw_metrics['latency_median_sec'] = lat_median
                            raw_metrics['latency_p90_sec'] = lat_p90
                            
                            timing_detail = (
                                f"Fill Rate: {fill_rate*100:.1f}% | Cancel Rate: {cancel_rate*100:.1f}%. "
                                f"Latency: Mean {lat_mean:.2f}s, Median {lat_median:.2f}s, 90th Pct {lat_p90:.2f}s."
                            )
                            
                            findings.append(self._make_finding(
                                'info', 'Execution timing analysis',
                                timing_detail,
                                {
                                    'fill_rate': fill_rate,
                                    'cancel_rate': cancel_rate,
                                    'latency_mean_sec': lat_mean,
                                    'latency_median_sec': lat_median,
                                    'latency_p90_sec': lat_p90
                                }
                            ))
                            
                            if cancel_rate > 0.15:
                                findings.append(self._make_finding(
                                    'warning', 'Elevated order cancel rate',
                                    f"Cancel rate is {cancel_rate*100:.1f}%, indicating excessive order modifications or failures.",
                                    {'cancel_rate': cancel_rate}
                                ))
                                recommendations.append("Order cancel rate is high. Investigate order placement logic or market liquidity issues.")
                        else:
                            findings.append(self._make_finding(
                                'info', 'Execution timing analysis',
                                f"Fill Rate: {fill_rate*100:.1f}% | Cancel Rate: {cancel_rate*100:.1f}%. (No valid latency data found)",
                                {'fill_rate': fill_rate, 'cancel_rate': cancel_rate}
                            ))
                    else:
                        findings.append(self._make_finding(
                            'info', 'Execution timing analysis',
                            f"Fill Rate: {fill_rate*100:.1f}% | Cancel Rate: {cancel_rate*100:.1f}%. (No valid latency data found)",
                            {'fill_rate': fill_rate, 'cancel_rate': cancel_rate}
                        ))
                else:
                    findings.append(self._make_finding(
                        'info', 'Execution timing analysis',
                        'No order data available in this window.',
                        {'status': 'no_data'}
                    ))
            else:
                findings.append(self._make_finding(
                    'info', 'Execution timing analysis',
                    'Execution timing analysis pending granular intraday timestamp data. '
                    'Consider recording precise execution timestamps in future trade logs.',
                    {'status': 'deferred'}
                ))
        except Exception as e:
            raw_metrics['execution_timing_error'] = str(e)
            findings.append(self._make_finding(
                'warning', 'Execution timing analysis error',
                f'Failed to compute execution timing: {e}',
                {'error': str(e)}
            ))

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)
