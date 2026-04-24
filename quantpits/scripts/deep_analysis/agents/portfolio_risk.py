"""
Portfolio Risk Agent.

Analyzes portfolio returns, factor exposures, drawdown patterns, and OLS
statistical significance across multiple time windows using the existing
PortfolioAnalyzer.
"""

import numpy as np
import pandas as pd
from ..base_agent import BaseAgent, AgentFindings, AnalysisContext, Finding


class PortfolioRiskAgent(BaseAgent):
    name = "Portfolio Risk"
    description = "Analyzes portfolio risk metrics, factor drift, and OLS significance."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        if ctx.daily_amount_df.empty:
            return AgentFindings(self.name, ctx.window_label,
                                [self._make_finding('info', 'No portfolio data',
                                                    'Daily amount log is empty.')],
                                [], {})

        # --- Initialize PortfolioAnalyzer ---
        try:
            from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
            from quantpits.scripts.analysis.utils import init_qlib
            init_qlib()

            pa = PortfolioAnalyzer(
                daily_amount_df=ctx.daily_amount_df,
                trade_log_df=ctx.trade_log_df,
                holding_log_df=ctx.holding_log_df,
                start_date=ctx.start_date,
                end_date=ctx.end_date,
            )
        except Exception as e:
            return AgentFindings(self.name, ctx.window_label,
                                [self._make_finding('critical', 'PortfolioAnalyzer init failed',
                                                    str(e))],
                                [], {'error': str(e)})

        # --- 1. Traditional Metrics ---
        try:
            trad = pa.calculate_traditional_metrics()
            raw_metrics['traditional'] = trad

            if trad:
                cagr = trad.get('CAGR_252', 0)
                sharpe = trad.get('Sharpe', 0)
                max_dd = trad.get('Max_Drawdown', 0)
                calmar = trad.get('Calmar', 0)
                excess_cagr = trad.get('Excess_Return_CAGR_252', 0)
                sortino = trad.get('Sortino', 0)

                findings.append(self._make_finding(
                    'info', f'Performance [{ctx.window_label}]',
                    f'CAGR: {cagr*100:.2f}%. Sharpe: {sharpe:.3f}. '
                    f'Sortino: {sortino:.3f}. Calmar: {calmar:.3f}. '
                    f'Max DD: {max_dd*100:.2f}%. Excess CAGR: {excess_cagr*100:.2f}%.',
                    trad
                ))

                # Alert on concerning metrics
                if excess_cagr < -0.05:
                    findings.append(self._make_finding(
                        'warning', 'Underperforming benchmark',
                        f'Excess return CAGR is {excess_cagr*100:.2f}%, '
                        f'significantly below benchmark.',
                        {'excess_cagr': excess_cagr}
                    ))

                if max_dd < -0.15:
                    findings.append(self._make_finding(
                        'critical', 'Severe drawdown',
                        f'Maximum drawdown of {max_dd*100:.2f}% is severe.',
                        {'max_dd': max_dd}
                    ))
                    recommendations.append(
                        "Maximum drawdown exceeds 15%. Review position sizing and risk limits."
                    )
        except Exception as e:
            raw_metrics['traditional_error'] = str(e)

        # --- 2. Factor Exposure + OLS Significance ---
        try:
            factor = pa.calculate_factor_exposure()
            raw_metrics['factor_exposure'] = factor

            if factor:
                beta = factor.get('Beta_Market', 0)
                alpha = factor.get('Annualized_Alpha', 0)
                alpha_t = factor.get('Annualized_Alpha_t', 0)
                alpha_p = factor.get('Annualized_Alpha_p', 1)
                beta_t = factor.get('Beta_Market_t', 0)
                beta_p = factor.get('Beta_Market_p', 1)
                r_sq = factor.get('R_Squared', 0)

                # OLS significance assessment
                alpha_sig = "***" if alpha_p < 0.01 else "**" if alpha_p < 0.05 else "*" if alpha_p < 0.1 else "n.s."
                beta_sig = "***" if beta_p < 0.01 else "**" if beta_p < 0.05 else "*" if beta_p < 0.1 else "n.s."

                findings.append(self._make_finding(
                    'info', f'OLS Factor Exposure [{ctx.window_label}]',
                    f'Alpha: {alpha*100:.2f}% (t={alpha_t:.2f}, p={alpha_p:.4f} {alpha_sig}). '
                    f'Beta: {beta:.3f} (t={beta_t:.2f}, p={beta_p:.4f} {beta_sig}). '
                    f'R²: {r_sq:.3f}.',
                    factor
                ))

                if alpha_p > 0.1:
                    findings.append(self._make_finding(
                        'warning', 'Alpha not statistically significant',
                        f'Single-factor alpha p-value is {alpha_p:.4f} (> 0.1). '
                        f'Cannot reject H₀ that alpha = 0 at 10% significance.',
                        {'alpha': alpha, 'alpha_p': alpha_p, 'alpha_t': alpha_t}
                    ))

                if abs(beta - 1.0) > 0.3:
                    direction = "under-exposed" if beta < 0.7 else "over-exposed"
                    findings.append(self._make_finding(
                        'info', f'Portfolio {direction} to market (β={beta:.3f})',
                        f'Beta significantly different from 1.0.',
                        {'beta': beta}
                    ))
        except Exception as e:
            raw_metrics['factor_error'] = str(e)

        # --- 3. Multi-Factor Style Exposures ---
        try:
            style = pa.calculate_style_exposures()
            raw_metrics['style_exposure'] = style

            if style:
                mf_alpha = style.get('Multi_Factor_Intercept', 0)
                mf_alpha_t = style.get('Multi_Factor_Intercept_t', 0)
                mf_alpha_p = style.get('Multi_Factor_Intercept_p', 1)
                mf_sig = "***" if mf_alpha_p < 0.01 else "**" if mf_alpha_p < 0.05 else "*" if mf_alpha_p < 0.1 else "n.s."

                liq_exp = style.get('Barra_Liquidity_Exp_(High-Low)', 0)
                mom_exp = style.get('Barra_Momentum_Exp_(High-Low)', 0)
                vol_exp = style.get('Barra_Volatility_Exp_(High-Low)', 0)

                findings.append(self._make_finding(
                    'info', f'Multi-Factor Alpha [{ctx.window_label}]',
                    f'Intercept: {mf_alpha*100:.2f}% (t={mf_alpha_t:.2f}, p={mf_alpha_p:.4f} {mf_sig}). '
                    f'Exposures — Liquidity: {liq_exp:.3f}, Momentum: {mom_exp:.3f}, '
                    f'Volatility: {vol_exp:.3f}.',
                    style
                ))

                # Extreme factor exposures
                if abs(liq_exp) > 0.3:
                    direction = "small-cap/illiquid" if liq_exp < 0 else "large-cap/liquid"
                    findings.append(self._make_finding(
                        'warning', f'Extreme liquidity exposure: {direction}',
                        f'Liquidity loading = {liq_exp:.3f}.',
                        {'liquidity_exposure': liq_exp}
                    ))
        except Exception as e:
            raw_metrics['style_error'] = str(e)

        # --- 4. Holding Metrics ---
        try:
            hold = pa.calculate_holding_metrics()
            raw_metrics['holding_metrics'] = hold

            if hold:
                avg_count = hold.get('Avg_Daily_Holdings_Count', 0)
                top1_conc = hold.get('Avg_Top1_Concentration', 0)
                win_rate = hold.get('Daily_Holding_Win_Rate', 0)

                findings.append(self._make_finding(
                    'info', f'Holding metrics [{ctx.window_label}]',
                    f'Avg holdings: {avg_count:.0f}. '
                    f'Top-1 concentration: {top1_conc*100:.1f}%. '
                    f'Daily holding win rate: {win_rate*100:.1f}%.',
                    hold
                ))

                if top1_conc > 0.15:
                    findings.append(self._make_finding(
                        'warning', 'High concentration risk',
                        f'Top-1 holding concentration is {top1_conc*100:.1f}%, '
                        f'exceeding 15% threshold.',
                        {'top1_concentration': top1_conc}
                    ))
        except Exception as e:
            raw_metrics['holding_error'] = str(e)

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)
