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
            period_ret = af.raw_metrics.get('period_return')
            window_vol = af.raw_metrics.get('window_volatility')
            max_dd = af.raw_metrics.get('max_drawdown', 0)
            curr_dd = af.raw_metrics.get('current_drawdown', 0)

            lines.append(f"- **Regime:** {regime}")
            if period_ret is not None:
                lines.append(f"- **Period Return:** {period_ret*100:.2f}%")
            if window_vol is not None:
                lines.append(f"- **Annualized Volatility:** {window_vol*100:.1f}%")
            lines.append(f"- **Max Drawdown:** {max_dd*100:.1f}% (Current: {curr_dd*100:.1f}%)")

            # Regime switches (per-window)
            regime_switches = af.raw_metrics.get('regime_switches', {})
            if regime_switches.get('switch_count', 0) > 0:
                lines.append(f"\n- **Regime Switches:** {regime_switches['switch_count']} detected")
                lines.append(f"  (current {regime_switches.get('current_regime', '?')} "
                            f"streak: {regime_switches.get('current_streak_days', '?')} days)")
                for sw in regime_switches.get('switches', [])[:5]:
                    lines.append(f"  - {sw.get('approx_date', '?')}: "
                               f"{sw.get('from', '?')} → {sw.get('to', '?')}")

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
            # Group by date for readability
            from collections import defaultdict
            by_date = defaultdict(list)
            for event in retrain_events:
                by_date[event['date']].append(event['model'])
            for date in sorted(by_date.keys(), reverse=True):
                models = by_date[date]
                if len(models) <= 5:
                    lines.append(f"- **{date}**: {', '.join(models)} retrained")
                else:
                    lines.append(f"- **{date}**: {len(models)} models retrained "
                               f"({', '.join(models[:3])}, ...)")

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

        # 2.4 Convergence Summary
        convergence = best_af.raw_metrics.get('convergence_summary', {})
        if convergence and convergence.get('total_models', 0) > 0:
            lines.append("\n### 2.4 Convergence Summary")
            lines.append(f"- **Models tracked:** {convergence['total_models']}")
            lines.append(f"- **Early-stopped:** {convergence['pct_early_stopped']*100:.0f}%")
            lines.append(f"- **Avg duration:** {convergence['avg_duration_s']:.0f}s")
            underfitting = convergence.get('underfitting_candidates', [])
            if underfitting:
                lines.append(f"- **Underfitting candidates:** {', '.join(underfitting)} "
                           f"(early-stopped < 50% of configured epochs)")
            full_epoch = convergence.get('full_epoch_models', [])
            if full_epoch:
                lines.append(f"- **Full-epoch models:** {', '.join(full_epoch)}")

        return "\n".join(lines)

    def _section_ensemble(self) -> str:
        findings = self._by_agent.get('Ensemble Evolution', [])
        if not findings:
            return ""

        lines = ["## 3. Ensemble Evolution"]

        af = findings[0]  # Use first (most comprehensive) window

        # 3.1 Combo Performance — per-combo latest snapshot with trend indicator.
        # Metrics are annualized full-period backtest results; change over time
        # reflects how the combo's evaluation evolves as new data comes in.
        trends = af.raw_metrics.get('combo_trends', {})
        if trends:
            lines.append("\n### 3.1 Combo Performance")
            lines.append("| Combo | Return | Calmar | Excess | Period | Trend |")
            lines.append("|-------|--------|--------|--------|--------|-------|")
            # Sort: default first if present, then by latest calmar desc
            def _sort_key(item):
                name, series = item
                is_default = 0 if name == 'default' else 1
                latest_cal = series[-1].get('calmar_ratio') if series else 0
                latest_cal = float(latest_cal) if latest_cal is not None else 0
                return (is_default, -latest_cal)
            for combo, series in sorted(trends.items(), key=_sort_key):
                if not series:
                    continue
                latest = series[-1]
                first = series[0]
                ret = latest.get('total_return')
                cal = latest.get('calmar_ratio')
                exc = latest.get('excess_return')
                ret_str = f"{float(ret):.1f}%" if ret is not None else "N/A"
                cal_str = f"{float(cal):.2f}" if cal is not None else "N/A"
                exc_str = f"{float(exc):.1f}%" if exc is not None else "N/A"

                # Date range
                first_date = first.get('_date', '?')
                latest_date = latest.get('_date', '?')
                if first_date == latest_date:
                    period_str = first_date
                else:
                    period_str = f"{first_date}→{latest_date}"

                # Trend arrow: compare first vs latest return
                first_ret = first.get('total_return')
                trend_str = "➡️"
                if first_ret is not None and ret is not None and len(series) >= 2:
                    delta = float(ret) - float(first_ret)
                    if delta > 1:
                        trend_str = f"📈 +{delta:.1f}pp"
                    elif delta < -1:
                        trend_str = f"📉 {delta:.1f}pp"
                    else:
                        trend_str = "➡️ flat"

                lines.append(
                    f"| **{combo}** | {ret_str} | {cal_str} | {exc_str} "
                    f"| {period_str} | {trend_str} |"
                )

        # 3.2 Change Events (from first window only)
        lines.append("\n### 3.2 Change Event Log")
        events = af.raw_metrics.get('change_events', [])
        if events:
            for e in events:
                etype = e.get('type', '?')
                icon = "🔄" if etype == 'composition_change' else "🔀" if etype == 'active_switch' else "📝"
                lines.append(f"- {icon} **{e.get('date', '?')}** [{etype}]: "
                           f"{e.get('detail', e.get('combo', ''))}")
        else:
            lines.append("- No ensemble composition changes detected in this period.")

        # 3.3 OOS Performance Trend
        oos_trend = af.raw_metrics.get('oos_trend', {})
        if oos_trend and oos_trend.get('oos_runs', 0) > 0:
            lines.append("\n### 3.3 OOS Performance Trend")
            lines.append(f"- **OOS runs analyzed:** {oos_trend['oos_runs']}")
            slope = oos_trend.get('oos_calmar_slope')
            if slope is not None:
                direction = "improving" if slope > 0 else "degrading"
                lines.append(f"- **Calmar slope:** {slope:.3f} ({direction})")
            latest = oos_trend.get('latest_oos_calmar')
            best = oos_trend.get('best_oos_calmar')
            if latest is not None and best is not None:
                lines.append(f"- **Latest OOS Calmar:** {latest:.2f}, **Best:** {best:.2f}")
            decay = oos_trend.get('is_oos_decay', [])
            if decay:
                lines.append(f"- **IS→OOS Decay:**")
                for d in decay:
                    lines.append(f"  - {d['combo']}: {d['decay_ratio']*100:.1f}% "
                               f"(IS={d['full_calmar']:.2f}, OOS={d['oos_calmar']:.2f})")

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

            fill_rate = af.raw_metrics.get('fill_rate')
            if fill_rate is not None:
                cancel_rate = af.raw_metrics.get('cancel_rate', 0)
                lines.append(f"- Timing: Fill Rate {fill_rate*100:.1f}%, Cancel Rate {cancel_rate*100:.1f}%")
                
                lat_mean = af.raw_metrics.get('latency_mean_sec')
                if lat_mean is not None:
                    lat_median = af.raw_metrics.get('latency_median_sec', 0)
                    lat_p90 = af.raw_metrics.get('latency_p90_sec', 0)
                    lines.append(f"- Latency: Mean {lat_mean:.2f}s, Median {lat_median:.2f}s, P90 {lat_p90:.2f}s")

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

        # Per-model IC proxy: data is identical across windows (same model_opinions),
        # so render once after all windows.
        af0 = findings[0]
        pm = af0.raw_metrics.get('per_model_hit_rate', {})
        if pm and pm.get('per_model_ic'):
            lines.append(f"\n### 6.1 Per-Model IC Proxy")
            lines.append(f"- **Ensemble overall proxy IC:** {pm.get('ensemble_overall_proxy_ic', 0):.4f}")
            sorted_models = sorted(pm['per_model_ic'].items(), key=lambda x: x[1], reverse=True)
            for model, ic in sorted_models[:10]:
                flag = " ⚠️" if model in pm.get('underperformers', []) else ""
                lines.append(f"  - {model}: {ic:.4f}{flag}")
            underperformers = pm.get('underperformers', [])
            if underperformers:
                lines.append(f"- **Underperformers:** {', '.join(underperformers)}")

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

        # Group by event type
        from collections import Counter
        type_counts = Counter(ci.get('event', {}).get('type', '?') for ci in impact)
        lines.append(f"\n*{len(impact)} total change events detected.*")
        lines.append(f"\nBreakdown by type:")
        for etype, count in type_counts.most_common():
            lines.append(f"- **{etype}**: {count} events")

        # Show non-retrain events first (retrains already covered in 2.2)
        non_retrain = [ci for ci in impact
                      if ci.get('event', {}).get('type') != 'retrain']
        if non_retrain:
            lines.append(f"\n### Non-Retrain Changes")
            for ci in non_retrain[:15]:
                event = ci.get('event', {})
                etype = event.get('type', '?')
                date = event.get('date', '?')
                detail = event.get('detail', event.get('model', event.get('combo', '?')))
                lines.append(f"- **{date}** [{etype}]: {detail}")

        # Summarize retrains (already detailed in 2.2)
        retrain_count = type_counts.get('retrain', 0)
        if retrain_count:
            lines.append(f"\n*Retrain events ({retrain_count} total) are detailed in "
                        f"Section 2.2 (Retrain History).*")

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
