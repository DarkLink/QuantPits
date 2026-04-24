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
        ma20 = bench.rolling(20).mean()
        ma60 = bench.rolling(60).mean() if len(bench) >= 60 else pd.Series(dtype=float)

        # Current values
        curr_price = bench.iloc[-1]
        curr_ma20 = ma20.iloc[-1] if not ma20.empty else np.nan
        curr_ma60 = ma60.iloc[-1] if not ma60.empty and not np.isnan(ma60.iloc[-1]) else np.nan

        # Linear regression slope on last 60 days (or all available)
        lookback = min(60, len(bench))
        y = bench.iloc[-lookback:].values
        x = np.arange(lookback)
        slope = np.polyfit(x, y, 1)[0] if lookback > 1 else 0
        slope_pct = slope / bench.iloc[-lookback] * 100  # as % of starting price

        raw_metrics['current_price'] = float(curr_price)
        raw_metrics['ma20'] = float(curr_ma20) if not np.isnan(curr_ma20) else None
        raw_metrics['ma60'] = float(curr_ma60) if curr_ma60 and not np.isnan(curr_ma60) else None
        raw_metrics['trend_slope_pct_per_day'] = float(slope_pct)

        # MA crossover state
        if not np.isnan(curr_ma20) and curr_ma60 and not np.isnan(curr_ma60):
            if curr_ma20 > curr_ma60 and curr_price > curr_ma20:
                trend_label = "Bullish"
            elif curr_ma20 < curr_ma60 and curr_price < curr_ma20:
                trend_label = "Bearish"
            else:
                trend_label = "Sideways"
        elif not np.isnan(curr_ma20):
            trend_label = "Bullish" if curr_price > curr_ma20 else "Bearish"
        else:
            trend_label = "Unknown"

        raw_metrics['trend_label'] = trend_label

        # --- Volatility Regime ---
        daily_ret = bench.pct_change().dropna()
        vol_20d = daily_ret.iloc[-20:].std() * np.sqrt(252) if len(daily_ret) >= 20 else np.nan

        # Historical percentile (use all available data)
        if len(daily_ret) >= 60:
            rolling_vol = daily_ret.rolling(20).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            if len(rolling_vol) > 0 and not np.isnan(vol_20d):
                vol_percentile = float((rolling_vol < vol_20d).mean() * 100)
            else:
                vol_percentile = 50.0
        else:
            vol_percentile = 50.0

        raw_metrics['volatility_20d'] = float(vol_20d) if not np.isnan(vol_20d) else None
        raw_metrics['volatility_percentile'] = vol_percentile

        if vol_percentile > 80:
            vol_label = "High-Volatility"
            findings.append(self._make_finding(
                'warning', 'Elevated market volatility',
                f'20-day annualized volatility ({vol_20d*100:.1f}%) is at the '
                f'{vol_percentile:.0f}th percentile of historical distribution.',
                {'vol_20d': vol_20d, 'percentile': vol_percentile}
            ))
        elif vol_percentile < 20:
            vol_label = "Low-Volatility"
        else:
            vol_label = "Normal"

        raw_metrics['volatility_label'] = vol_label

        # --- Drawdown State ---
        cum_ret = (1 + daily_ret).cumprod()
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret / rolling_max - 1)
        current_dd = float(drawdown.iloc[-1])

        # Underwater duration
        if current_dd < -0.001:
            # Find when drawdown started
            peak_idx = cum_ret.idxmax()
            underwater_days = (bench.index[-1] - peak_idx).days
        else:
            underwater_days = 0

        raw_metrics['current_drawdown'] = current_dd
        raw_metrics['max_drawdown'] = float(drawdown.min())
        raw_metrics['underwater_days'] = underwater_days

        if current_dd < -0.10:
            findings.append(self._make_finding(
                'critical', 'Deep market drawdown',
                f'CSI300 is {current_dd*100:.1f}% below peak, '
                f'underwater for {underwater_days} days.',
                {'drawdown': current_dd, 'underwater_days': underwater_days}
            ))
            recommendations.append(
                "Consider reducing position sizes during deep market drawdowns."
            )
        elif current_dd < -0.05:
            findings.append(self._make_finding(
                'warning', 'Moderate market drawdown',
                f'CSI300 is {current_dd*100:.1f}% below peak.',
                {'drawdown': current_dd}
            ))

        # --- Regime Summary ---
        regime = trend_label
        if vol_label == "High-Volatility":
            regime = f"High-Vol {trend_label}"

        raw_metrics['regime'] = regime

        findings.append(self._make_finding(
            'info', f'Market regime: {regime}',
            f'Trend: {trend_label} (slope: {slope_pct:.3f}%/day). '
            f'Volatility: {vol_label} ({vol_percentile:.0f}th pctile). '
            f'Drawdown: {current_dd*100:.1f}%.',
            raw_metrics
        ))

        return AgentFindings(
            agent_name=self.name,
            window_label=ctx.window_label,
            findings=findings,
            recommendations=recommendations,
            raw_metrics=raw_metrics,
        )
