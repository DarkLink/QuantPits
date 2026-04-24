"""
Report Generator for the MAS Deep Analysis System.

Formats synthesis results into a structured Markdown report.
"""

from datetime import datetime
from typing import List, Dict, Optional
from .base_agent import AgentFindings, Finding


class ReportGenerator:
    """Formats analysis results into a structured Markdown report."""

    def __init__(self, all_findings: List[AgentFindings],
                 synthesis_result: dict,
                 executive_summary: str):
        self.all_findings = all_findings
        self.synthesis = synthesis_result
        self.executive_summary = executive_summary

        # Index findings by agent
        self._by_agent = {}
        for af in all_findings:
            self._by_agent.setdefault(af.agent_name, []).append(af)

    def generate(self) -> str:
        """Generate the full Markdown report."""
        sections = [
            self._header(),
            self._executive_summary(),
            self._section_market(),
            self._section_model_health(),
            self._section_ensemble(),
            self._section_execution(),
            self._section_portfolio(),
            self._section_prediction(),
            self._section_trade_pattern(),
            self._section_change_impact(),
            self._section_recommendations(),
            self._section_appendix(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ------------------------------------------------------------------
    # Report Sections
    # ------------------------------------------------------------------

    def _header(self) -> str:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        health = self.synthesis.get('health_status', '')
        exec_data = self.synthesis.get('executive_summary_data', {})
        windows = exec_data.get('windows_analyzed', [])
        agents = exec_data.get('agents_run', [])

        return (
            f"# Deep Analysis Report — {now}\n\n"
            f"**Health Status:** {health}\n\n"
            f"**Scope:** {len(windows)} windows ({', '.join(windows)}) × "
            f"{len(agents)} agents"
        )

    def _executive_summary(self) -> str:
        return f"## Executive Summary\n\n{self.executive_summary}"

    def _section_market(self) -> str:
        findings = self._by_agent.get('Market Regime', [])
        if not findings:
            return ""

        lines = ["## 1. Market Environment"]
        for af in findings:
            lines.append(f"\n### Window: {af.window_label}")
            regime = af.raw_metrics.get('regime', 'Unknown')
            vol = af.raw_metrics.get('volatility_20d')
            vol_pct = af.raw_metrics.get('volatility_percentile', 0)
            dd = af.raw_metrics.get('current_drawdown', 0)

            lines.append(f"- **Regime:** {regime}")
            if vol:
                lines.append(f"- **20d Volatility:** {vol*100:.1f}% "
                           f"({vol_pct:.0f}th percentile)")
            lines.append(f"- **Current Drawdown:** {dd*100:.1f}%")

        return "\n".join(lines)

    def _section_model_health(self) -> str:
        findings = self._by_agent.get('Model Health', [])
        if not findings:
            return ""

        lines = ["## 2. Model Health Dashboard"]

        # Find the most comprehensive window's scorecard
        best_af = max(findings, key=lambda af: len(af.raw_metrics.get('scorecard', {})),
                      default=None)
        if not best_af:
            return "\n".join(lines)

        # 2.1 IC/ICIR Scorecard
        scorecard = best_af.raw_metrics.get('scorecard', {})
        if scorecard:
            lines.append(f"\n### 2.1 IC/ICIR Scorecard (Window: {best_af.window_label})")
            lines.append("| Model | IC Mean | IC Latest | ICIR | Trend | Snapshots |")
            lines.append("|-------|---------|-----------|------|-------|-----------|")

            sorted_models = sorted(scorecard.items(),
                                  key=lambda x: x[1].get('ic_mean', 0) or 0,
                                  reverse=True)
            for model, sc in sorted_models:
                ic_mean = f"{sc['ic_mean']:.4f}" if sc.get('ic_mean') is not None else "N/A"
                ic_latest = f"{sc['ic_latest']:.4f}" if sc.get('ic_latest') is not None else "N/A"
                icir = f"{sc['icir_mean']:.4f}" if sc.get('icir_mean') is not None else "N/A"
                trend = sc.get('ic_trend', '?')
                trend_icon = "📈" if trend == 'improving' else "📉" if trend == 'degrading' else "➡️"
                n = sc.get('n_snapshots', 0)
                lines.append(f"| {model} | {ic_mean} | {ic_latest} | {icir} | "
                           f"{trend_icon} {trend} | {n} |")

        # 2.2 Retrain History
        retrain_events = best_af.raw_metrics.get('retrain_events', [])
        if retrain_events:
            lines.append("\n### 2.2 Retrain History")
            for event in retrain_events[-10:]:
                lines.append(f"- **{event['date']}**: {event['model']} retrained")

        # 2.3 Hyperparameter Snapshot
        hyperparams = best_af.raw_metrics.get('hyperparams', {})
        if hyperparams:
            lines.append("\n### 2.3 Hyperparameter Configuration")
            lines.append("| Model | Class | Key Params |")
            lines.append("|-------|-------|------------|")
            for model, hp in sorted(hyperparams.items()):
                cls = hp.get('class', '?')
                params = {k: v for k, v in hp.items()
                         if k not in ('class', 'label', '_path', '_error', 'model_key',
                                     'feature_set', 'model_module')}
                param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:5])
                lines.append(f"| {model} | {cls} | {param_str} |")

        return "\n".join(lines)

    def _section_ensemble(self) -> str:
        findings = self._by_agent.get('Ensemble Evolution', [])
        if not findings:
            return ""

        lines = ["## 3. Ensemble Evolution"]

        for af in findings:
            # 3.1 Combo Performance
            trends = af.raw_metrics.get('combo_trends', {})
            if trends:
                lines.append(f"\n### 3.1 Combo Performance (Window: {af.window_label})")
                for combo, series in trends.items():
                    if series:
                        latest = series[-1]
                        lines.append(
                            f"- **{combo}**: Return={latest.get('total_return', 0):.2f}%, "
                            f"Calmar={latest.get('calmar_ratio', 0):.2f}, "
                            f"Excess={latest.get('excess_return', 0):.2f}%"
                        )

            # 3.2 Change Events
            events = af.raw_metrics.get('change_events', [])
            if events:
                lines.append("\n### 3.2 Change Event Log")
                for e in events:
                    etype = e.get('type', '?')
                    icon = "🔄" if etype == 'composition_change' else "🔀" if etype == 'active_switch' else "📝"
                    lines.append(f"- {icon} **{e.get('date', '?')}** [{etype}]: "
                               f"{e.get('detail', e.get('combo', ''))}")
            break  # Only show first window for these

        return "\n".join(lines)

    def _section_execution(self) -> str:
        findings = self._by_agent.get('Execution Quality', [])
        if not findings:
            return ""

        lines = ["## 4. Execution Quality"]
        for af in findings:
            lines.append(f"\n### Window: {af.window_label}")
            buy_f = af.raw_metrics.get('buy_total_friction')
            sell_f = af.raw_metrics.get('sell_total_friction')
            if buy_f is not None:
                buy_d = af.raw_metrics.get('buy_delay_cost', 0)
                buy_s = af.raw_metrics.get('buy_exec_slippage', 0)
                lines.append(f"- Buy friction: {buy_f*100:.3f}% "
                           f"(delay: {buy_d*100:.3f}%, slip: {buy_s*100:.3f}%)")
            if sell_f is not None:
                sell_d = af.raw_metrics.get('sell_delay_cost', 0)
                sell_s = af.raw_metrics.get('sell_exec_slippage', 0)
                lines.append(f"- Sell friction: {sell_f*100:.3f}% "
                           f"(delay: {sell_d*100:.3f}%, slip: {sell_s*100:.3f}%)")

            costs = af.raw_metrics.get('explicit_costs', {})
            if costs:
                lines.append(f"- Fee ratio: {costs.get('fee_ratio', 0)*100:.4f}%")

            sub = af.raw_metrics.get('substitution_bias', {})
            if sub:
                theo = sub.get('theoretical_substitute_bias_impact', 0)
                real = sub.get('realized_substitute_bias_impact', 0)
                n_missed = sub.get('total_missed_count', 0)
                lines.append(f"- Substitution bias: theoretical={theo*100:.2f}%, "
                           f"realized={real*100:.2f}% (missed buys: {n_missed})")

        return "\n".join(lines)

    def _section_portfolio(self) -> str:
        findings = self._by_agent.get('Portfolio Risk', [])
        if not findings:
            return ""

        lines = ["## 5. Portfolio Risk & Attribution"]

        # 5.1 Multi-Window Comparison
        lines.append("\n### 5.1 Multi-Window Comparison")
        lines.append("| Window | CAGR | Sharpe | Max DD | Calmar | Excess CAGR |")
        lines.append("|--------|------|--------|--------|--------|-------------|")
        for af in findings:
            trad = af.raw_metrics.get('traditional', {})
            if trad:
                cagr = trad.get('CAGR_252', 0)
                sharpe = trad.get('Sharpe', 0)
                max_dd = trad.get('Max_Drawdown', 0)
                calmar = trad.get('Calmar', 0)
                excess = trad.get('Excess_Return_CAGR_252', 0)
                lines.append(
                    f"| {af.window_label} | {cagr*100:.2f}% | {sharpe:.3f} | "
                    f"{max_dd*100:.2f}% | {calmar:.3f} | {excess*100:.2f}% |"
                )

        # 5.2 OLS Significance
        lines.append("\n### 5.2 OLS Significance")
        for af in findings:
            factor = af.raw_metrics.get('factor_exposure', {})
            if factor:
                alpha = factor.get('Annualized_Alpha', 0)
                alpha_t = factor.get('Annualized_Alpha_t', 0)
                alpha_p = factor.get('Annualized_Alpha_p', 1)
                beta = factor.get('Beta_Market', 0)
                sig = "***" if alpha_p < 0.01 else "**" if alpha_p < 0.05 else "*" if alpha_p < 0.1 else "n.s."
                lines.append(
                    f"- **{af.window_label}**: α={alpha*100:.2f}% "
                    f"(t={alpha_t:.2f}, p={alpha_p:.4f} {sig}), β={beta:.3f}"
                )

        return "\n".join(lines)

    def _section_prediction(self) -> str:
        findings = self._by_agent.get('Prediction Audit', [])
        if not findings:
            return ""

        lines = ["## 6. Prediction Accuracy Audit"]

        for af in findings:
            lines.append(f"\n### Window: {af.window_label}")

            # Buy hit rate
            buy = af.raw_metrics.get('buy_hit_rate', {}).get('overall', {})
            if buy:
                lines.append(f"- **Buy Hit Rate:** {buy.get('hit_rate', 0)*100:.1f}% "
                           f"({buy.get('n_suggestions', 0)} suggestions, "
                           f"avg return: {buy.get('avg_return', 0)*100:.2f}%)")

            # Sell quality
            sell = af.raw_metrics.get('sell_hit_rate', {}).get('overall', {})
            if sell:
                lines.append(f"- **Sell Signal Quality:** {sell.get('hit_rate', 0)*100:.1f}% "
                           f"underperformed ({sell.get('n_suggestions', 0)} sells)")

            # Consensus
            cons = af.raw_metrics.get('consensus_analysis', {})
            if cons.get('n_high_consensus', 0) > 0:
                lines.append(
                    f"- **Consensus vs Divergence:** "
                    f"High-consensus avg return: {cons.get('high_consensus_avg_return', 0)*100:.2f}%, "
                    f"High-divergence: {cons.get('high_divergence_avg_return', 0)*100:.2f}%"
                )

        return "\n".join(lines)

    def _section_trade_pattern(self) -> str:
        findings = self._by_agent.get('Trade Pattern', [])
        if not findings:
            return ""

        lines = ["## 7. Trade Behavior"]
        for af in findings:
            disc = af.raw_metrics.get('discipline', {})
            if disc:
                lines.append(f"\n### Window: {af.window_label}")
                lines.append(f"- SIGNAL: {disc.get('signal_pct', 0):.1f}% "
                           f"({disc.get('signal_count', 0)} trades)")
                lines.append(f"- SUBSTITUTE: {disc.get('substitute_pct', 0):.1f}% "
                           f"({disc.get('substitute_count', 0)} trades)")
                lines.append(f"- MANUAL: {disc.get('manual_pct', 0):.1f}% "
                           f"({disc.get('manual_count', 0)} trades)")

                score = af.raw_metrics.get('discipline_score')
                if score is not None:
                    lines.append(f"- **Discipline Score:** {score:.1f}/100")

        return "\n".join(lines)

    def _section_change_impact(self) -> str:
        impact = self.synthesis.get('change_impact', [])
        if not impact:
            return ""

        lines = ["## 8. Holistic Change Impact Assessment"]
        for ci in impact[:10]:
            event = ci.get('event', {})
            etype = event.get('type', '?')
            lines.append(
                f"- **{event.get('date', '?')}** [{etype}]: "
                f"{event.get('model', event.get('combo', '?'))}"
            )

        lines.append(f"\n*{len(impact)} change events detected. "
                    f"See multi-window metrics above for before/after comparison.*")
        return "\n".join(lines)

    def _section_recommendations(self) -> str:
        recs = self.synthesis.get('recommendations', [])
        if not recs:
            return ""

        lines = ["## 9. Prioritized Recommendations"]

        for priority in ['P0', 'P1', 'P2']:
            group = [r for r in recs if r['priority'] == priority]
            if group:
                label = {'P0': 'Urgent Actions', 'P1': 'Important Adjustments',
                         'P2': 'Monitor Items'}[priority]
                lines.append(f"\n### {priority} — {label}")
                for r in group:
                    lines.append(f"- ({r['source']}) {r['text']}")

        return "\n".join(lines)

    def _section_appendix(self) -> str:
        notes = self.synthesis.get('external_notes', '')
        if not notes:
            return ""

        return f"## Appendix: External Notes\n\n{notes}"
