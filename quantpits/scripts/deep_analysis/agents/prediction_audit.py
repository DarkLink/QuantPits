"""
Prediction Audit Agent.

Core new capability: compares model predictions vs actual market outcomes.
Analyzes buy/sell suggestion hit rates, ensemble vs single model performance,
and model consensus vs divergence patterns.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from ..base_agent import BaseAgent, AgentFindings, AnalysisContext, Finding


class PredictionAuditAgent(BaseAgent):
    name = "Prediction Audit"
    description = ("Compares model predictions vs actuals: hit rates, "
                   "ensemble vs single, consensus vs divergence.")

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        # --- 1. Buy Suggestion Hit Rate ---
        buy_hits = self._analyze_suggestion_hits(ctx, direction='buy')
        raw_metrics['buy_hit_rate'] = buy_hits

        if buy_hits:
            overall = buy_hits.get('overall', {})
            hit_rate = overall.get('hit_rate', 0)
            n_suggestions = overall.get('n_suggestions', 0)
            avg_return = overall.get('avg_return', 0)

            findings.append(self._make_finding(
                'info', f'Buy suggestion hit rate [{ctx.window_label}]',
                f'Hit rate: {hit_rate*100:.1f}% ({n_suggestions} suggestions). '
                f'Avg T+5 return: {avg_return*100:.2f}%.',
                overall
            ))

            if hit_rate < 0.4 and n_suggestions >= 5:
                findings.append(self._make_finding(
                    'warning', 'Low buy suggestion hit rate',
                    f'Only {hit_rate*100:.1f}% of buy suggestions had positive T+5 returns.',
                    overall
                ))
                recommendations.append(
                    "Buy suggestion hit rate is below 40%. Review model selection criteria "
                    "and ensemble composition."
                )
            elif hit_rate > 0.6 and n_suggestions >= 5:
                findings.append(self._make_finding(
                    'positive', 'Strong buy suggestion accuracy',
                    f'{hit_rate*100:.1f}% of buy suggestions had positive T+5 returns.',
                    overall
                ))

        # --- 2. Sell Suggestion Quality ---
        sell_hits = self._analyze_suggestion_hits(ctx, direction='sell')
        raw_metrics['sell_hit_rate'] = sell_hits

        if sell_hits:
            overall = sell_hits.get('overall', {})
            hit_rate = overall.get('hit_rate', 0)
            n_suggestions = overall.get('n_suggestions', 0)

            findings.append(self._make_finding(
                'info', f'Sell signal quality [{ctx.window_label}]',
                f'Sell accuracy: {hit_rate*100:.1f}% of sold stocks underperformed '
                f'portfolio avg in T+5 ({n_suggestions} sells).',
                overall
            ))

        # --- 3. Model Consensus vs Divergence ---
        consensus = self._analyze_consensus(ctx)
        raw_metrics['consensus_analysis'] = consensus

        if consensus:
            high_cons_ret = consensus.get('high_consensus_avg_return', 0)
            high_div_ret = consensus.get('high_divergence_avg_return', 0)
            n_high_cons = consensus.get('n_high_consensus', 0)
            n_high_div = consensus.get('n_high_divergence', 0)

            if n_high_cons > 0 and n_high_div > 0:
                findings.append(self._make_finding(
                    'info', 'Consensus vs Divergence analysis',
                    f'High-consensus picks avg T+5 return: {high_cons_ret*100:.2f}% '
                    f'({n_high_cons} stocks). '
                    f'High-divergence picks: {high_div_ret*100:.2f}% ({n_high_div} stocks).',
                    consensus
                ))

                if high_cons_ret > high_div_ret:
                    findings.append(self._make_finding(
                        'positive', 'Consensus picks outperform',
                        'Stocks with high model consensus outperform divergence picks.',
                        consensus
                    ))
                else:
                    findings.append(self._make_finding(
                        'warning', 'Divergence picks outperform consensus',
                        'High-divergence stocks outperform high-consensus stocks. '
                        'Ensemble may be suppressing valuable contrarian signals.',
                        consensus
                    ))

        # --- 4. Holding Retrospective ---
        retrospective = self._analyze_holding_retrospective(ctx)
        raw_metrics['holding_retrospective'] = retrospective

        if retrospective.get('summary'):
            findings.append(self._make_finding(
                'info', 'Holding retrospective',
                retrospective['summary'],
                retrospective
            ))

        # --- 5. Per-model Hit Rate Analysis ---
        per_model = self._analyze_per_model_hit_rate(ctx)
        raw_metrics['per_model_hit_rate'] = per_model
        
        if per_model and per_model.get('underperformers'):
            findings.append(self._make_finding(
                'warning', 'Individual models underperforming ensemble',
                f"Models {', '.join(per_model['underperformers'])} have significantly lower hit rates "
                "than the ensemble average.",
                per_model
            ))

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _analyze_suggestion_hits(self, ctx: AnalysisContext,
                                 direction: str = 'buy') -> dict:
        """
        Analyze hit rate of buy/sell suggestions.
        
        For buy: % of suggested stocks with positive T+5 return.
        For sell: % of sold stocks that underperformed portfolio average T+5.
        """
        files = ctx.buy_suggestion_files if direction == 'buy' else ctx.sell_suggestion_files
        if not files:
            return {}

        try:
            from quantpits.scripts.analysis.utils import init_qlib
            from qlib.data import D
            init_qlib()
        except Exception:
            return {'error': 'Qlib initialization failed'}

        all_hits = []
        all_returns = []
        per_date = []

        for path in files:
            date_str = self._extract_date(path)
            if not date_str:
                continue

            try:
                df = pd.read_csv(path)
                if 'instrument' not in df.columns:
                    continue
            except Exception:
                continue

            instruments = df['instrument'].tolist()
            if not instruments:
                continue

            # Get forward returns (T+5 for weekly)
            try:
                fwd_date = pd.Timestamp(date_str)
                end_fwd = fwd_date + pd.Timedelta(days=10)
                field = "Ref($close, -5) / $close - 1"
                fwd_returns = D.features(
                    instruments, [field],
                    start_time=date_str, end_time=end_fwd.strftime('%Y-%m-%d')
                )
                fwd_returns.columns = ['fwd_return']

                # Get returns on the suggestion date
                fwd_on_date = fwd_returns.xs(
                    pd.Timestamp(date_str), level='datetime', drop_level=False
                ) if pd.Timestamp(date_str) in fwd_returns.index.get_level_values('datetime') else pd.DataFrame()

                if fwd_on_date.empty:
                    # Try next available date
                    available_dates = fwd_returns.index.get_level_values('datetime').unique()
                    future_dates = [d for d in available_dates if d >= pd.Timestamp(date_str)]
                    if future_dates:
                        fwd_on_date = fwd_returns.xs(future_dates[0], level='datetime', drop_level=False)

                if fwd_on_date.empty:
                    continue

                fwd_on_date = fwd_on_date.reset_index()
                fwd_on_date = fwd_on_date[fwd_on_date['instrument'].isin(instruments)]

                if direction == 'buy':
                    hits = (fwd_on_date['fwd_return'] > 0).sum()
                    total = len(fwd_on_date)
                    avg_ret = fwd_on_date['fwd_return'].mean()
                else:
                    # For sell: hit = sold stock underperformed market average
                    market_avg = fwd_on_date['fwd_return'].mean()
                    hits = (fwd_on_date['fwd_return'] < market_avg).sum()
                    total = len(fwd_on_date)
                    avg_ret = fwd_on_date['fwd_return'].mean()

                if total > 0:
                    all_hits.append(hits)
                    all_returns.extend(fwd_on_date['fwd_return'].dropna().tolist())
                    per_date.append({
                        'date': date_str,
                        'hit_rate': hits / total,
                        'n': total,
                        'avg_return': float(avg_ret),
                    })
            except Exception:
                continue

        if not per_date:
            return {}

        total_hits = sum(all_hits)
        total_n = sum(d['n'] for d in per_date)

        return {
            'overall': {
                'hit_rate': total_hits / total_n if total_n > 0 else 0,
                'n_suggestions': total_n,
                'avg_return': float(np.mean(all_returns)) if all_returns else 0,
                'n_dates': len(per_date),
            },
            'per_date': per_date,
        }

    def _analyze_consensus(self, ctx: AnalysisContext) -> dict:
        """
        Analyze whether high-consensus picks (all models agree) outperform
        high-divergence picks (models disagree).
        """
        if not ctx.model_opinions_files:
            return {}

        try:
            from quantpits.scripts.analysis.utils import init_qlib
            from qlib.data import D
            init_qlib()
        except Exception:
            return {'error': 'Qlib init failed'}

        high_cons_returns = []
        high_div_returns = []

        for path in ctx.model_opinions_files[-5:]:  # Last 5 for efficiency
            date_str = self._extract_date(path)
            if not date_str:
                continue

            try:
                with open(path, 'r') as f:
                    opinions = json.load(f)
            except Exception:
                continue

            # Parse model_to_combos or combo_composition
            # We want per-instrument signals from model_opinions CSV
            csv_path = path.replace('.json', '.csv')
            if not os.path.exists(csv_path):
                continue

            try:
                opinion_df = pd.read_csv(csv_path)
                if 'instrument' not in opinion_df.columns:
                    continue
            except Exception:
                continue

            # Count BUY signals per stock across models
            model_cols = [c for c in opinion_df.columns
                         if c.startswith('model_') or c.startswith('combo_')]
            if not model_cols:
                continue

            def _count_buys(row):
                count = 0
                for col in model_cols:
                    val = str(row.get(col, ''))
                    if 'BUY' in val.upper():
                        count += 1
                return count

            opinion_df['buy_count'] = opinion_df.apply(_count_buys, axis=1)
            total_models = len(model_cols)
            if total_models == 0:
                continue

            opinion_df['consensus_ratio'] = opinion_df['buy_count'] / total_models

            # High consensus: > 80% models agree
            high_cons = opinion_df[opinion_df['consensus_ratio'] > 0.8]['instrument'].tolist()
            # High divergence: 30-70% models disagree
            high_div = opinion_df[
                (opinion_df['consensus_ratio'] > 0.3) &
                (opinion_df['consensus_ratio'] < 0.7)
            ]['instrument'].tolist()

            if not high_cons and not high_div:
                continue

            # Get forward returns
            try:
                all_instr = list(set(high_cons + high_div))
                fwd_date = pd.Timestamp(date_str)
                end_fwd = fwd_date + pd.Timedelta(days=10)
                field = "Ref($close, -5) / $close - 1"
                fwd = D.features(all_instr, [field],
                                start_time=date_str,
                                end_time=end_fwd.strftime('%Y-%m-%d'))
                fwd.columns = ['fwd_return']
                fwd = fwd.reset_index()

                # Match to suggestion date
                available_dates = fwd['datetime'].unique()
                target = min(available_dates, key=lambda d: abs(d - fwd_date))
                fwd_day = fwd[fwd['datetime'] == target]

                for inst in high_cons:
                    row = fwd_day[fwd_day['instrument'] == inst]
                    if not row.empty:
                        high_cons_returns.append(float(row['fwd_return'].iloc[0]))

                for inst in high_div:
                    row = fwd_day[fwd_day['instrument'] == inst]
                    if not row.empty:
                        high_div_returns.append(float(row['fwd_return'].iloc[0]))
            except Exception:
                continue

        result = {
            'n_high_consensus': len(high_cons_returns),
            'n_high_divergence': len(high_div_returns),
        }
        if high_cons_returns:
            result['high_consensus_avg_return'] = float(np.mean(high_cons_returns))
        if high_div_returns:
            result['high_divergence_avg_return'] = float(np.mean(high_div_returns))

        return result

    def _analyze_holding_retrospective(self, ctx: AnalysisContext) -> dict:
        """Summarize current holdings' performance since entry."""
        if ctx.holding_log_df.empty:
            return {}

        df = ctx.holding_log_df.copy()
        if '证券代码' not in df.columns or '成交日期' not in df.columns:
            return {}

        # Exclude CASH
        df = df[df['证券代码'] != 'CASH']
        if df.empty:
            return {}

        # Get latest date holdings
        latest_date = df['成交日期'].max()
        current_holdings = df[df['成交日期'] == latest_date]

        if current_holdings.empty:
            return {}

        n_holdings = len(current_holdings)
        if '浮盈收益率' in current_holdings.columns:
            avg_float = current_holdings['浮盈收益率'].mean()
            win_rate = (current_holdings['浮盈收益率'] > 0).mean()
            summary = (
                f'{n_holdings} holdings as of {latest_date.strftime("%Y-%m-%d")}. '
                f'Avg floating return: {avg_float*100:.2f}%. '
                f'Win rate: {win_rate*100:.1f}%.'
            )
        else:
            summary = f'{n_holdings} holdings as of {latest_date.strftime("%Y-%m-%d")}.'

        return {
            'summary': summary,
            'n_holdings': n_holdings,
            'latest_date': latest_date.strftime('%Y-%m-%d'),
        }

    @staticmethod
    def _parse_opinion_rank(value) -> Optional[float]:
        """Parse numeric rank from model_opinions label like 'HOLD (7)' -> 7.0."""
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)
        if isinstance(value, str):
            m = re.search(r'\((\d+)\)', value)
            if m:
                return float(m.group(1))
        return None

    def _analyze_per_model_hit_rate(self, ctx: AnalysisContext) -> dict:
        """
        Analyze per-model prediction quality via rank correlation with ensemble.

        The model_opinions CSV uses string labels like 'BUY (3)', 'HOLD (7)'
        where the number is the intra-model prediction rank (lower = stronger).
        We parse the rank, then compute Spearman correlation between each
        model's ranking and the ensemble order_basis ranking as an IC proxy.
        """
        if not ctx.model_opinions_files:
            return {}

        per_model_stats = {}
        all_models = set()

        for path in ctx.model_opinions_files[-3:]:
            csv_path = path.replace('.json', '.csv')
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            # Use order_basis as reference, then combo_, then score as fallback
            ref_col = None
            for candidate in ['order_basis'] + \
                             [c for c in df.columns if c.startswith('combo_')] + \
                             ['score']:
                if candidate in df.columns:
                    ref_col = candidate
                    break

            if ref_col is None:
                continue

            ref_ranks = df[ref_col].apply(self._parse_opinion_rank)
            valid_ref = ref_ranks.notna()
            if valid_ref.sum() < 5:
                continue

            model_cols = [c for c in df.columns if c.startswith('model_')]
            for col in model_cols:
                model_name = col.replace('model_', '')
                all_models.add(model_name)

                model_ranks = df[col].apply(self._parse_opinion_rank)
                valid = valid_ref & model_ranks.notna()
                if valid.sum() < 5:
                    continue

                # Spearman rank correlation as IC proxy
                ic = float(ref_ranks[valid].corr(model_ranks[valid], method='spearman'))
                if model_name not in per_model_stats:
                    per_model_stats[model_name] = []
                per_model_stats[model_name].append(ic)

        if not per_model_stats:
            return {}

        avg_ic = {m: np.mean(v) for m, v in per_model_stats.items()}
        overall_avg = np.mean(list(avg_ic.values()))

        underperformers = [m for m, ic in avg_ic.items()
                          if ic < overall_avg * 0.5 and ic < overall_avg - 0.05]

        return {
            "ensemble_overall_proxy_ic": float(overall_avg),
            "per_model_ic": {m: round(v, 4) for m, v in avg_ic.items()},
            "underperformers": underperformers,
            "snapshots_analyzed": len(per_model_stats.get(list(per_model_stats.keys())[0], [])) if per_model_stats else 0,
        }

    @staticmethod
    def _extract_date(path: str) -> Optional[str]:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(path))
        return m.group(1) if m else None
