"""
Synthesizer for the MAS Deep Analysis System.

Cross-references findings from all agents to detect compound patterns,
correlate change events with performance shifts, and produce prioritized
recommendations.
"""

import numpy as np
from typing import List, Dict, Optional
from .base_agent import AgentFindings, Finding


class Synthesizer:
    """
    Cross-agent reasoning engine.
    
    Takes findings from all agents across all windows and produces:
    1. Cross-referenced compound findings
    2. Change event impact assessment
    3. Priority-ranked recommendations
    """

    def __init__(self, all_findings: List[AgentFindings], external_notes: str = ""):
        self.all_findings = all_findings
        self.external_notes = external_notes

        # Index findings by agent and window
        self._by_agent = {}
        self._by_window = {}
        for af in all_findings:
            self._by_agent.setdefault(af.agent_name, []).append(af)
            self._by_window.setdefault(af.window_label, []).append(af)

    def synthesize(self) -> dict:
        """
        Run cross-agent synthesis.
        
        Returns:
            {
                'cross_findings': List[Finding],
                'change_impact': List[dict],
                'recommendations': List[dict],  # {priority, text, rationale}
                'health_status': str,
                'executive_summary_data': dict,
            }
        """
        cross_findings = []
        change_impact = []
        recommendations = []

        # --- 1. Cross-Reference Rules ---
        cross_findings.extend(self._check_alpha_decay_regime())
        cross_findings.extend(self._check_liquidity_drift())
        cross_findings.extend(self._check_substitution_hit_rate())
        cross_findings.extend(self._check_ensemble_value())
        cross_findings.extend(self._check_alpha_significance())
        cross_findings.extend(self._check_time_horizon_reversal())
        cross_findings.extend(self._check_ic_combo_contradiction())

        # --- 2. Holistic Change Impact Assessment ---
        change_impact = self._assess_change_impact()

        # --- 3. Collect & Prioritize Recommendations ---
        recommendations = self._prioritize_recommendations(cross_findings)

        # --- 4. Overall Health Status ---
        health_status = self._compute_health_status()

        # --- 5. Executive Summary Data ---
        exec_data = self._build_executive_summary_data()

        return {
            'cross_findings': cross_findings,
            'change_impact': change_impact,
            'recommendations': recommendations,
            'health_status': health_status,
            'executive_summary_data': exec_data,
            'external_notes': self.external_notes,
        }

    # ------------------------------------------------------------------
    # Cross-Reference Rules
    # ------------------------------------------------------------------

    def _check_alpha_decay_regime(self) -> List[Finding]:
        """Check: Model IC declining + Market regime change."""
        findings = []
        model_findings = self._get_agent_findings('Model Health')
        market_findings = self._get_agent_findings('Market Regime')

        model_declining = any(
            f.severity == 'warning' and 'declining' in f.title.lower()
            for af in model_findings for f in af.findings
        )
        market_volatile = any(
            'High-Vol' in af.raw_metrics.get('regime', '')
            for af in market_findings
        )

        if model_declining and market_volatile:
            findings.append(Finding(
                severity='warning',
                category='Cross-Agent',
                title='IC decline coincides with high-volatility regime',
                detail='Model IC degradation may be regime-driven rather than fundamental '
                       'model failure. Monitor for recovery when volatility normalizes.',
                data={'model_declining': True, 'market_volatile': True}
            ))

        return findings

    def _check_time_horizon_reversal(self) -> List[Finding]:
        """
        Check: Short-term window (1m/3m) shows positive return/excess,
        but long-term window (1y) shows negative return/excess.
        """
        findings = []
        ee_runs = self._by_agent.get('Ensemble Evolution', [])
        if not ee_runs:
            return findings

        window_best = {}
        for af in ee_runs:
            best_combo = af.raw_metrics.get('best_combo')
            if best_combo:
                window_best[af.window_label] = best_combo

        # We look for '3m' or '1m' vs '1y'
        short_window = window_best.get('3m', window_best.get('1m'))
        long_window = window_best.get('1y', window_best.get('full'))

        if short_window and long_window:
            short_ret = short_window.get('excess_return', 0.0)
            long_ret = long_window.get('excess_return', 0.0)

            if short_ret > 0 and long_ret < -0.05:
                findings.append(Finding(
                    severity='warning',
                    category='Cross-Agent',
                    title='OOS Time Horizon Reversal (多周期倒挂)',
                    detail=f"策略在短期窗口 ({short_window.get('name', 'short')}, excess={short_ret:.2f}%) 表现占优，但在一阶长期窗口 ({long_window.get('name', 'long')}, excess={long_ret:.2f}%) 却陷入严重亏损。",
                    data={'short_return': short_ret, 'long_return': long_ret}
                ))
        return findings

    def _check_ic_combo_contradiction(self) -> List[Finding]:
        """
        Check: Model IC trend is improving or stable, but combo returns are declining.

        Uses structured raw_metrics from Model Health (scorecard.*.ic_trend) and
        Ensemble Evolution ({combo}_return_change, oos_trend.is_oos_decay) —
        no keyword matching on free-text finding fields.
        """
        findings = []
        model_runs = self._by_agent.get('Model Health', [])
        ee_runs = self._by_agent.get('Ensemble Evolution', [])
        if not model_runs or not ee_runs:
            return findings

        # 1. Any model with improving or stable IC trend (from scorecard)
        ic_improving = False
        for af in model_runs:
            scorecard = af.raw_metrics.get('scorecard', {})
            for _model_name, sc in scorecard.items():
                if sc.get('ic_trend') in ('improving', 'stable'):
                    ic_improving = True
                    break
            if ic_improving:
                break

        # 2. Any combo with declining returns
        combo_degrading = False
        degrading_combo_name = ""
        for af in ee_runs:
            # Check per-combo return_change (negative delta = degrading)
            for key, val in af.raw_metrics.items():
                if key.endswith('_return_change') and isinstance(val, (int, float)):
                    if val < -0.03:  # -3% absolute return decline
                        combo_degrading = True
                        degrading_combo_name = key.replace('_return_change', '')
                        break
            # Also check OOS decay signal
            if not combo_degrading:
                oos = af.raw_metrics.get('oos_trend', {})
                if oos.get('is_oos_decay'):
                    combo_degrading = True
                    degrading_combo_name = "OOS"
            if combo_degrading:
                break

        if ic_improving and combo_degrading:
            findings.append(Finding(
                severity='warning',
                category='Cross-Agent',
                title='IC-Combo Performance Contradiction (指标背离)',
                detail=f"背离警示：单模型IC趋势稳定/回升，但融合组合 ({degrading_combo_name}) 表现超额却出现持续恶化。"
                       f"这可能暗示模型间的相关性结构漂移或过度拟合。",
                data={'ic_improving': ic_improving, 'combo_degrading': combo_degrading,
                      'degrading_combo': degrading_combo_name}
            ))
        return findings

    def _check_liquidity_drift(self) -> List[Finding]:
        """Check: Idiosyncratic Alpha negative + Liquidity exposure increasing."""
        findings = []
        port_findings = self._get_agent_findings('Portfolio Risk')

        for af in port_findings:
            style = af.raw_metrics.get('style_exposure', {})
            factor = af.raw_metrics.get('factor_exposure', {})

            liq_exp = style.get('Barra_Liquidity_Exp_(High-Low)', 0)
            alpha = factor.get('Annualized_Alpha', 0)
            alpha_p = factor.get('Annualized_Alpha_p', 1)

            if liq_exp < -0.2 and alpha < 0:
                findings.append(Finding(
                    severity='warning',
                    category='Cross-Agent',
                    title='Small-cap drift without selection edge',
                    detail=f'Portfolio has negative liquidity exposure ({liq_exp:.3f}) '
                           f'combined with negative alpha ({alpha*100:.2f}%). '
                           f'Strategy is drifting toward illiquid stocks without generating '
                           f'selection returns.',
                    data={'liq_exposure': liq_exp, 'alpha': alpha, 'alpha_p': alpha_p}
                ))
                break  # One finding is enough

        return findings

    def _check_substitution_hit_rate(self) -> List[Finding]:
        """Check: Substitution bias growing + hit rate declining."""
        findings = []
        exec_findings = self._get_agent_findings('Execution Quality')
        pred_findings = self._get_agent_findings('Prediction Audit')

        high_sub_bias = any(
            f.severity == 'warning' and 'substitution' in f.title.lower()
            for af in exec_findings for f in af.findings
        )
        low_hit_rate = any(
            f.severity == 'warning' and 'hit rate' in f.title.lower()
            for af in pred_findings for f in af.findings
        )

        if high_sub_bias and low_hit_rate:
            findings.append(Finding(
                severity='critical',
                category='Cross-Agent',
                title='Top convictions increasingly untradeable',
                detail='High substitution bias combined with low hit rate suggests '
                       'the model\'s top picks are frequently limit-up or illiquid, '
                       'forcing execution of weaker alternatives.',
                data={'high_sub_bias': True, 'low_hit_rate': True}
            ))

        return findings

    def _check_ensemble_value(self) -> List[Finding]:
        """Check: Does ensemble outperform single models?"""
        findings = []
        pred_findings = self._get_agent_findings('Prediction Audit')

        for af in pred_findings:
            buy_hits = af.raw_metrics.get('buy_hit_rate', {})
            overall = buy_hits.get('overall', {})
            hit_rate = overall.get('hit_rate', 0)

            if hit_rate > 0.55:
                findings.append(Finding(
                    severity='positive',
                    category='Cross-Agent',
                    title='Ensemble fusion value confirmed',
                    detail=f'Buy suggestion hit rate of {hit_rate*100:.1f}% indicates '
                           f'effective signal combination.',
                    data={'hit_rate': hit_rate}
                ))
                break

        return findings

    def _check_alpha_significance(self) -> List[Finding]:
        """Check: Alpha significance across windows."""
        findings = []
        port_findings = self._get_agent_findings('Portfolio Risk')

        all_insignificant = True
        for af in port_findings:
            factor = af.raw_metrics.get('factor_exposure', {})
            p = factor.get('Annualized_Alpha_p', 1)
            if p < 0.1:
                all_insignificant = False
                break

        if all_insignificant and len(port_findings) >= 2:
            findings.append(Finding(
                severity='warning',
                category='Cross-Agent',
                title='No statistically significant alpha across any window',
                detail='Alpha is not significant (p>0.1) in any time window. '
                       'Cannot reject the null hypothesis that the strategy '
                       'generates zero stock selection alpha.',
                data={'all_insignificant': True}
            ))

        return findings

    # ------------------------------------------------------------------
    # Change Impact Assessment
    # ------------------------------------------------------------------

    def _assess_change_impact(self) -> List[dict]:
        """
        Correlate retrain/combo change events from Model Health and Ensemble Evolution
        with performance changes from Portfolio Risk.
        """
        impact = []

        # Get change events
        ensemble_findings = self._get_agent_findings('Ensemble Evolution')
        model_findings = self._get_agent_findings('Model Health')

        change_events = []
        for af in ensemble_findings:
            for event in af.raw_metrics.get('change_events', []):
                change_events.append(event)
        for af in model_findings:
            for event in af.raw_metrics.get('retrain_events', []):
                change_events.append({
                    'type': 'retrain',
                    'model': event.get('model'),
                    'date': event.get('date'),
                })

        if not change_events:
            return impact

        # Get performance metrics per window for comparison
        port_metrics_by_window = {}
        for af in self._get_agent_findings('Portfolio Risk'):
            trad = af.raw_metrics.get('traditional', {})
            if trad:
                port_metrics_by_window[af.window_label] = {
                    'sharpe': trad.get('Sharpe', 0),
                    'cagr': trad.get('CAGR_252', 0),
                    'max_dd': trad.get('Max_Drawdown', 0),
                }

        # Build impact summary
        for event in change_events:
            impact.append({
                'event': event,
                'portfolio_windows': port_metrics_by_window,
                'assessment': 'Performance comparison requires before/after window analysis. '
                              'See multi-window metrics for trends.',
            })

        return impact

    # ------------------------------------------------------------------
    # Recommendation Prioritization
    # ------------------------------------------------------------------

    def _prioritize_recommendations(self, cross_findings: List[Finding]) -> List[dict]:
        """Collect and prioritize all recommendations."""
        recs = []

        # From individual agents
        for af in self.all_findings:
            for rec in af.recommendations:
                recs.append({
                    'priority': 'P1',
                    'source': af.agent_name,
                    'text': rec,
                })

        # From cross-findings
        for cf in cross_findings:
            if cf.severity == 'critical':
                recs.append({
                    'priority': 'P0',
                    'source': 'Cross-Agent',
                    'text': cf.detail,
                })
            elif cf.severity == 'warning':
                recs.append({
                    'priority': 'P1',
                    'source': 'Cross-Agent',
                    'text': cf.detail,
                })

        # Deduplicate and sort
        import re
        seen = set()
        unique_recs = []
        for r in recs:
            # Remove details in parentheses to deduplicate variations
            key = re.sub(r'\(.*?\)', '', r['text']).strip()
            if key not in seen:
                seen.add(key)
                unique_recs.append(r)

        priority_order = {'P0': 0, 'P1': 1, 'P2': 2}
        unique_recs.sort(key=lambda x: priority_order.get(x['priority'], 9))

        return unique_recs

    # ------------------------------------------------------------------
    # Health Status
    # ------------------------------------------------------------------

    def _compute_health_status(self) -> str:
        """Compute overall system health status."""
        n_critical = sum(
            1 for af in self.all_findings
            for f in af.findings if f.severity == 'critical'
        )
        n_warning = sum(
            1 for af in self.all_findings
            for f in af.findings if f.severity == 'warning'
        )

        if n_critical >= 2:
            return "🔴 CRITICAL — Multiple systemic issues detected. Intervention recommended."
        elif n_critical >= 1:
            return "🟠 ALERT — Critical issue detected. Review urgently."
        elif n_warning >= 4:
            return "🟡 WARNING — Multiple anomalies detected. Close monitoring needed."
        elif n_warning >= 1:
            return "🟡 CAUTION — Minor anomalies detected."
        else:
            return "🟢 HEALTHY — No systemic issues detected."

    # ------------------------------------------------------------------
    # Executive Summary Data
    # ------------------------------------------------------------------

    def _build_executive_summary_data(self) -> dict:
        """Build structured data for executive summary generation."""
        data = {
            'total_findings': sum(len(af.findings) for af in self.all_findings),
            'critical_count': sum(
                1 for af in self.all_findings
                for f in af.findings if f.severity == 'critical'),
            'warning_count': sum(
                1 for af in self.all_findings
                for f in af.findings if f.severity == 'warning'),
            'positive_count': sum(
                1 for af in self.all_findings
                for f in af.findings if f.severity == 'positive'),
            'windows_analyzed': list(set(af.window_label for af in self.all_findings)),
            'agents_run': list(set(af.agent_name for af in self.all_findings)),
        }

        # Pull key metrics from agents
        for af in self.all_findings:
            if af.agent_name == 'Market Regime' and af.window_label in ('1m', '3m', 'weekly_era'):
                data['market_regime'] = af.raw_metrics.get('regime', 'Unknown')
            if af.agent_name == 'Portfolio Risk' and af.window_label == '1y':
                trad = af.raw_metrics.get('traditional', {})
                data['cagr_1y'] = trad.get('CAGR_252')
                data['sharpe_1y'] = trad.get('Sharpe')

        return data

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_agent_findings(self, agent_name: str) -> List[AgentFindings]:
        return self._by_agent.get(agent_name, [])
