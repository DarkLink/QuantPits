"""
Market Regime Detection Agent.

Analyzes benchmark (CSI300) data to determine current market regime:
trend direction, volatility state, and drawdown depth.
"""

import numpy as np
import pandas as pd
from ..base_agent import BaseAgent, AgentFindings, AnalysisContext, Finding


class MarketRegimeAgent(BaseAgent):
    name = "Market Regime"
    description = "Detects market trend, volatility regime, and drawdown state from benchmark data."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        df = ctx.daily_amount_df
        if df.empty or 'CSI300' not in df.columns:
            return AgentFindings(self.name, ctx.window_label,
                                [self._make_finding('info', 'No benchmark data',
                                                    'CSI300 column not found in daily amount log.')],
                                [], {})

        df = df.sort_values('成交日期').copy()
        bench = df.set_index('成交日期')['CSI300'].dropna().astype(float)

        if len(bench) < 20:
            return AgentFindings(self.name, ctx.window_label,
                                [self._make_finding('info', 'Insufficient data',
                                                    f'Only {len(bench)} data points.')],
                                [], {})

        # --- Trend Detection ---
        period_return = bench.iloc[-1] / bench.iloc[0] - 1 if bench.iloc[0] != 0 else 0.0

        # Linear regression slope on all available data in the window
        lookback = len(bench)
        y = np.array(bench.values, dtype=float)
        x = np.arange(lookback, dtype=float)
        slope = np.polyfit(x, y, 1)[0] if lookback > 1 else 0
        slope_pct = slope / bench.iloc[0] * 100 * 252 if bench.iloc[0] != 0 else 0.0 # Annualized slope %

        raw_metrics['current_price'] = float(bench.iloc[-1])
        raw_metrics['period_return'] = float(period_return)
        raw_metrics['annualized_slope_pct'] = float(slope_pct)

        if period_return > 0.05:
            trend_label = "Bullish"
        elif period_return < -0.05:
            trend_label = "Bearish"
        else:
            trend_label = "Sideways"

        raw_metrics['trend_label'] = trend_label

        # --- Volatility Regime ---
        daily_ret = bench.pct_change().dropna()
        window_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) >= 5 else 0.0

        raw_metrics['window_volatility'] = float(window_vol)

        if window_vol > 0.20:
            vol_label = "High-Vol"
        elif window_vol < 0.12:
            vol_label = "Low-Vol"
        else:
            vol_label = "Normal-Vol"

        raw_metrics['volatility_label'] = vol_label

        if window_vol > 0.20:
            findings.append(self._make_finding(
                'warning', 'Elevated market volatility',
                f'Annualized volatility for this window is {window_vol*100:.1f}%.',
                {'window_volatility': window_vol}
            ))

        # --- Drawdown State ---
        cum_ret = (1 + daily_ret).cumprod()
        rolling_max = cum_ret.cummax()
        # Ensure we don't divide by zero
        drawdown = (cum_ret / rolling_max - 1) if not rolling_max.empty else pd.Series([0.0])
        current_dd = float(drawdown.iloc[-1]) if not drawdown.empty else 0.0
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        # Underwater duration
        if current_dd < -0.001 and not cum_ret.empty:
            peak_idx = cum_ret.idxmax()
            underwater_days = (bench.index[-1] - peak_idx).days
        else:
            underwater_days = 0

        raw_metrics['current_drawdown'] = current_dd
        raw_metrics['max_drawdown'] = max_dd
        raw_metrics['underwater_days'] = underwater_days

        if current_dd < -0.10:
            findings.append(self._make_finding(
                'critical', 'Deep market drawdown',
                f'CSI300 is {current_dd*100:.1f}% below peak within window, '
                f'underwater for {underwater_days} days.',
                {'drawdown': current_dd, 'underwater_days': underwater_days}
            ))
            recommendations.append(
                "Consider reducing position sizes during deep market drawdowns."
            )
        elif current_dd < -0.05:
            findings.append(self._make_finding(
                'warning', 'Moderate market drawdown',
                f'CSI300 is {current_dd*100:.1f}% below window peak.',
                {'drawdown': current_dd}
            ))

        # --- Regime Summary ---
        regime = trend_label
        if vol_label == "High-Vol":
            regime = f"High-Vol {trend_label}"

        raw_metrics['regime'] = regime

        findings.append(self._make_finding(
            'info', f'Market regime: {regime}',
            f'Period Return: {period_return*100:.2f}%. '
            f'Volatility: {window_vol*100:.1f}%. '
            f'Max Drawdown: {max_dd*100:.1f}%.',
            raw_metrics
        ))

        # --- Sliding Window Regime Switch Detection ---
        switches = self._detect_regime_switches(bench)
        raw_metrics['regime_switches'] = switches
        
        if switches.get('switch_count', 0) >= 3:
            findings.append(self._make_finding(
                'warning', 'Frequent market regime switching',
                f"Market has switched regimes {switches['switch_count']} times in this window, "
                "indicating high instability and potential model generalization pressure.",
                switches
            ))
        elif switches.get('current_streak_days', 0) > 30:
            findings.append(self._make_finding(
                'info', 'Stable market regime',
                f"Current regime ({regime}) has been stable for {switches['current_streak_days']} days.",
                switches
            ))

        return AgentFindings(
            agent_name=self.name,
            window_label=ctx.window_label,
            findings=findings,
            recommendations=recommendations,
            raw_metrics=raw_metrics,
        )

    def _detect_regime_switches(self, bench: pd.Series, window: int = 20, step: int = 5) -> dict:
        """Detect transitions between market regimes using a sliding window."""
        switches = []
        last_regime = None
        
        # We need at least 'window' days to start
        if len(bench) < window:
            return {"switch_count": 0, "current_streak_days": len(bench)}

        # Iterate with sliding window
        for i in range(0, len(bench) - window + 1, step):
            segment = bench.iloc[i : i + window]
            seg_ret = segment.iloc[-1] / segment.iloc[0] - 1
            
            # Simple regime labeling for the segment
            if seg_ret > 0.02:
                reg = "Bullish"
            elif seg_ret < -0.02:
                reg = "Bearish"
            else:
                reg = "Sideways"
            
            if last_regime and reg != last_regime:
                switches.append({
                    "from": last_regime,
                    "to": reg,
                    "approx_date": segment.index[-1].strftime('%Y-%m-%d')
                })
            last_regime = reg

        # Calculate current streak: days since the last regime switch
        current_streak = 0
        if last_regime:
            last_date = bench.index[-1]
            if switches:
                switch_date = pd.Timestamp(switches[-1]['approx_date'])
                current_streak = (last_date - switch_date).days
            else:
                # No switches — current regime covers the entire window
                current_streak = len(bench)
        
        return {
            "switch_count": len(switches),
            "switches": switches,
            "current_streak_days": current_streak,
            "current_regime": last_regime
        }
