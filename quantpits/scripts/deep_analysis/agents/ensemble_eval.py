"""
Ensemble Evolution Agent.

Tracks ensemble combo performance trends, detects 3 layers of composition changes
(model list diff, active combo switch, intra-combo content mutation), and analyzes
post-change impact.
"""

import os
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..base_agent import BaseAgent, AgentFindings, AnalysisContext, Finding


class EnsembleEvolutionAgent(BaseAgent):
    name = "Ensemble Evolution"
    description = ("Tracks combo performance, detects composition changes, "
                   "and analyzes post-change impact.")

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        # --- 1. Combo Performance Trend ---
        combo_trends = self._load_combo_trends(ctx)
        raw_metrics['combo_trends'] = combo_trends

        if combo_trends:
            for combo_name, series in combo_trends.items():
                if len(series) < 2:
                    continue
                returns = [s['total_return'] for s in series if 'total_return' in s]
                calmars = [s['calmar_ratio'] for s in series if 'calmar_ratio' in s]

                if len(returns) >= 2:
                    ret_trend = returns[-1] - returns[0]
                    raw_metrics[f'{combo_name}_return_change'] = ret_trend

                    if len(calmars) >= 2:
                        calmar_latest = calmars[-1]
                        calmar_first = calmars[0]

                        if calmar_latest < calmar_first * 0.7:
                            findings.append(self._make_finding(
                                'warning', f'{combo_name}: Calmar degrading',
                                f'Calmar ratio declined from {calmar_first:.2f} '
                                f'to {calmar_latest:.2f} over the analysis window.',
                                {'combo': combo_name, 'calmar_start': calmar_first,
                                 'calmar_end': calmar_latest}
                            ))

            # Best combo identification
            best_combo = self._identify_best_combo(combo_trends)
            if best_combo:
                raw_metrics['best_combo'] = best_combo
                findings.append(self._make_finding(
                    'info', f'Best performing combo: {best_combo["name"]}',
                    f'Based on latest snapshot: excess_return={best_combo.get("excess_return", "N/A")}, '
                    f'calmar={best_combo.get("calmar_ratio", "N/A")}.',
                    best_combo
                ))

        # --- 2. Three-Layer Change Detection ---
        change_events = self._detect_changes(ctx)
        raw_metrics['change_events'] = change_events

        for event in change_events:
            severity = 'info'
            if event['type'] == 'composition_change':
                title = f"Combo '{event['combo']}' composition changed on {event['date']}"
                detail = f"Models changed: {event.get('detail', '')}"
            elif event['type'] == 'active_switch':
                title = f"Active combo switched on {event['date']}"
                detail = f"Switched from '{event.get('old')}' to '{event.get('new')}'"
                severity = 'warning'
            elif event['type'] == 'content_mutation':
                title = f"Combo '{event['combo']}' content mutated on {event['date']}"
                detail = f"Same name but models changed: {event.get('detail', '')}"
                severity = 'warning'
            else:
                title = f"Ensemble change: {event['type']}"
                detail = str(event)

            findings.append(self._make_finding(severity, title, detail, event))

        # --- 3. Correlation Drift ---
        corr_drift = self._analyze_correlation_drift(ctx)
        raw_metrics['correlation_drift'] = corr_drift
        if corr_drift.get('significant_drift'):
            findings.append(self._make_finding(
                'warning', 'Model correlation drift detected',
                f"Inter-model correlations have shifted significantly. "
                f"Mean drift: {corr_drift.get('mean_drift', 0):.3f}.",
                corr_drift
            ))
            recommendations.append(
                "Review ensemble composition — correlation structure has changed, "
                "which may affect diversification benefits."
            )

        # --- 4. Model Contribution from Leaderboards ---
        contributions = self._analyze_contributions(ctx)
        raw_metrics['model_contributions'] = contributions

        if contributions.get('consistently_negative'):
            for model in contributions['consistently_negative']:
                findings.append(self._make_finding(
                    'warning', f'{model}: Consistently negative contribution',
                    f'This model has been a drag on ensemble performance across recent snapshots.',
                    {'model': model}
                ))
                recommendations.append(
                    f"Consider removing {model} from its ensemble combo(s)."
                )

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_combo_trends(self, ctx: AnalysisContext) -> Dict[str, list]:
        """Load combo_comparison_*.csv files into per-combo time series."""
        result = {}

        for path in ctx.combo_comparison_files:
            date_str = self._extract_date(path)
            if not date_str:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            for _, row in df.iterrows():
                combo = row.get('combo', '')
                if not combo:
                    continue
                entry = {'_date': date_str}
                for col in ['total_return', 'annualized_return', 'annualized_excess',
                            'max_drawdown', 'calmar_ratio', 'excess_return', 'models']:
                    if col in row:
                        entry[col] = row[col]
                if combo not in result:
                    result[combo] = []
                result[combo].append(entry)

        for combo in result:
            result[combo].sort(key=lambda x: x['_date'])

        return result

    def _detect_changes(self, ctx: AnalysisContext) -> List[dict]:
        """
        Detect 3 layers of ensemble changes:
        1. Composition change: model list diff in fusion configs
        2. Active combo switch: default combo changed in ensemble_config
        3. Content mutation: same combo name, different models in ensemble_config
        """
        events = []

        # Layer 1: Composition change from ensemble_fusion_config files
        fusion_configs = {}  # (combo, date) -> model_list
        for path in ctx.ensemble_fusion_config_files:
            date_str = self._extract_date(path)
            if not date_str:
                continue
            # Extract combo name from filename: ensemble_fusion_config_{combo}_{date}.json
            basename = os.path.basename(path)
            m = re.match(r'ensemble_fusion_config_(.+?)_\d{4}-\d{2}-\d{2}', basename)
            if not m:
                continue
            combo = m.group(1)
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                models = config.get('selected_models', config.get('models', []))
                if isinstance(models, dict):
                    models = list(models.keys())
                fusion_configs[(combo, date_str)] = sorted(models) if models else []
            except Exception:
                continue

        # Group by combo and detect changes
        combos = set(c for c, d in fusion_configs.keys())
        for combo in combos:
            dates = sorted([d for c, d in fusion_configs.keys() if c == combo])
            for i in range(1, len(dates)):
                old_models = fusion_configs.get((combo, dates[i-1]), [])
                new_models = fusion_configs.get((combo, dates[i]), [])
                if old_models != new_models:
                    added = set(new_models) - set(old_models)
                    removed = set(old_models) - set(new_models)
                    events.append({
                        'type': 'composition_change',
                        'combo': combo,
                        'date': dates[i],
                        'detail': f"Added: {list(added)}, Removed: {list(removed)}",
                        'old_models': old_models,
                        'new_models': new_models,
                    })

        # Layers 2 & 3 from config_history snapshots (if available)
        history_dir = os.path.join(ctx.workspace_root, 'data', 'config_history')
        if os.path.isdir(history_dir):
            import glob
            snapshots = sorted(glob.glob(os.path.join(history_dir, 'config_snapshot_*.json')))
            for i in range(1, len(snapshots)):
                try:
                    with open(snapshots[i-1], 'r') as f:
                        old_snap = json.load(f)
                    with open(snapshots[i], 'r') as f:
                        new_snap = json.load(f)
                except Exception:
                    continue

                old_ec = old_snap.get('ensemble_config', {})
                new_ec = new_snap.get('ensemble_config', {})
                snap_date = new_snap.get('snapshot_date', self._extract_date(snapshots[i]))

                # Layer 2: Active combo switch
                old_default = old_ec.get('default_combo')
                new_default = new_ec.get('default_combo')
                if old_default and new_default and old_default != new_default:
                    events.append({
                        'type': 'active_switch',
                        'date': snap_date,
                        'old': old_default,
                        'new': new_default,
                    })

                # Layer 3: Intra-combo content mutation
                old_groups = old_ec.get('combo_groups', {})
                new_groups = new_ec.get('combo_groups', {})
                for name in set(old_groups.keys()) & set(new_groups.keys()):
                    if json.dumps(old_groups[name], sort_keys=True) != \
                       json.dumps(new_groups[name], sort_keys=True):
                        events.append({
                            'type': 'content_mutation',
                            'combo': name,
                            'date': snap_date,
                            'detail': f"Old: {old_groups[name]}, New: {new_groups[name]}",
                        })

        events.sort(key=lambda x: x.get('date', ''))
        return events

    def _analyze_correlation_drift(self, ctx: AnalysisContext) -> dict:
        """Compare latest correlation matrix vs earlier ones."""
        if len(ctx.correlation_matrix_files) < 2:
            return {'significant_drift': False, 'reason': 'insufficient_data'}

        try:
            early = pd.read_csv(ctx.correlation_matrix_files[0], index_col=0)
            latest = pd.read_csv(ctx.correlation_matrix_files[-1], index_col=0)

            # Align columns
            common = list(set(early.columns) & set(latest.columns))
            if len(common) < 2:
                return {'significant_drift': False, 'reason': 'no_common_models'}

            early_sub = early.loc[common, common]
            latest_sub = latest.loc[common, common]

            diff = (latest_sub - early_sub).abs()
            mean_drift = float(diff.values[np.triu_indices_from(diff.values, k=1)].mean())

            return {
                'significant_drift': mean_drift > 0.15,
                'mean_drift': mean_drift,
                'n_models': len(common),
            }
        except Exception as e:
            return {'significant_drift': False, 'error': str(e)}

    def _analyze_contributions(self, ctx: AnalysisContext) -> dict:
        """Analyze model contributions from leaderboard files."""
        model_excess = {}  # model -> list of excess returns

        for path in ctx.leaderboard_files:
            try:
                df = pd.read_csv(path)
                if 'name' in df.columns and 'annualized_excess' in df.columns:
                    for _, row in df.iterrows():
                        name = row['name']
                        if name == 'Ensemble':
                            continue
                        excess = row.get('annualized_excess', 0)
                        if name not in model_excess:
                            model_excess[name] = []
                        model_excess[name].append(excess)
            except Exception:
                continue

        consistently_negative = []
        for model, excesses in model_excess.items():
            if len(excesses) >= 2 and all(e < 0 for e in excesses):
                consistently_negative.append(model)

        return {
            'model_excess': {k: {'mean': np.mean(v), 'count': len(v)}
                            for k, v in model_excess.items()},
            'consistently_negative': consistently_negative,
        }

    def _identify_best_combo(self, combo_trends: Dict[str, list]) -> Optional[dict]:
        """Identify the best performing combo from latest snapshots."""
        candidates = []
        for combo_name, series in combo_trends.items():
            if series:
                latest = series[-1]
                candidates.append({
                    'name': combo_name,
                    'total_return': latest.get('total_return', 0),
                    'calmar_ratio': latest.get('calmar_ratio', 0),
                    'excess_return': latest.get('excess_return', 0),
                })

        if not candidates:
            return None

        # Sort by calmar_ratio as primary metric
        candidates.sort(key=lambda x: x.get('calmar_ratio', 0), reverse=True)
        return candidates[0]

    @staticmethod
    def _extract_date(path: str) -> Optional[str]:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(path))
        return m.group(1) if m else None
