"""
Ensemble Evolution Agent.

Tracks ensemble combo performance trends, detects 3 layers of composition changes
(model list diff, active combo switch, intra-combo content mutation), and analyzes
post-change impact.
"""

import os
import json
import re
import glob
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
                returns = [s['total_return'] for s in series if s.get('total_return') is not None]
                calmars = [s['calmar_ratio'] for s in series if s.get('calmar_ratio') is not None]

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
        # --- 4. Model Contribution Analysis ---
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

        # --- 5. OOS History Analysis ---
        oos_trend = self._load_oos_history(ctx)
        raw_metrics['oos_trend'] = oos_trend
        
        if oos_trend and oos_trend.get('latest_oos_calmar'):
            slope = oos_trend.get('oos_calmar_slope', 0)
            if slope < -0.1:
                findings.append(self._make_finding(
                    'warning', 'OOS performance degrading',
                    f"OOS Calmar ratio is declining (slope={slope:.3f}). "
                    f"Latest: {oos_trend['latest_oos_calmar']:.2f}, Best: {oos_trend['best_oos_calmar']:.2f}.",
                    oos_trend
                ))
            
            if oos_trend.get('latest_oos_calmar', 0) < oos_trend.get('best_oos_calmar', 0) * 0.8:
                findings.append(self._make_finding(
                    'warning', 'Significant OOS drawdown',
                    f"Latest OOS Calmar ({oos_trend['latest_oos_calmar']:.2f}) is more than 20% below "
                    f"historical best ({oos_trend['best_oos_calmar']:.2f}).",
                    oos_trend
                ))

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_combo_trends(self, ctx: AnalysisContext) -> Dict[str, list]:
        """
        Load per-combo performance time series.

        Primary: fusion_run_ledger.jsonl (written by every ensemble_fusion.py run).
        Supplement: combo_comparison_*.csv (written only in multi-combo --from-config-all mode).
        """
        result = {}  # combo_name -> [{_date, total_return, ...}, ...]

        # --- 1. Primary: fusion_run_ledger.jsonl ---
        ledger_path = os.path.join(ctx.workspace_root, 'data', 'fusion_run_ledger.jsonl')
        ledger_dedup = {}  # (combo, date) -> entry
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        combo = rec.get('combo_name', 'default')
                        date_str = rec.get('run_date', '')
                        if not date_str:
                            continue
                        # Apply window filter (agent context date range)
                        if date_str < ctx.start_date or date_str > ctx.end_date:
                            continue
                        m = rec.get('metrics', {})
                        
                        ann_ret = m.get('annualized_return')
                        ann_ret_pct = ann_ret * 100 if ann_ret is not None else None
                        ann_exc = m.get('annualized_excess')
                        ann_exc_pct = ann_exc * 100 if ann_exc is not None else None
                        mdd = m.get('max_drawdown')
                        mdd_pct = mdd * 100 if mdd is not None else None

                        entry = {
                            '_date': date_str,
                            '_source': 'ledger',
                            'models': ','.join(rec.get('models', [])),
                            'method': rec.get('method'),
                            'is_oos': rec.get('eval_window', {}).get('is_oos', False),
                            'total_return': ann_ret_pct,  # fallback to annualized
                            'annualized_return': ann_ret_pct,
                            'annualized_excess': ann_exc_pct,
                            'excess_return': ann_exc_pct, # fallback
                            'max_drawdown': mdd_pct,
                            'calmar_ratio': m.get('calmar'),
                            'information_ratio': m.get('information_ratio'),
                        }
                        ledger_dedup[(combo, date_str)] = entry
            except Exception:
                pass
                
        for (combo, date_str), entry in ledger_dedup.items():
            result.setdefault(combo, []).append(entry)

        # --- 2. Supplement: combo_comparison_*.csv (multi-combo runs) ---
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
                # Skip if already covered by ledger on the same date
                existing_dates = {e['_date'] for e in result.get(combo, [])}
                if date_str in existing_dates:
                    continue
                entry = {'_date': date_str, '_source': 'csv'}
                for col in ['total_return', 'annualized_return', 'annualized_excess',
                            'max_drawdown', 'calmar_ratio', 'excess_return', 'models']:
                    if col in row:
                        entry[col] = row[col]
                # Normalize CSV metrics to match ledger convention:
                # CSV total_return/excess_return are cumulative; ledger uses annualized.
                if 'annualized_return' in entry:
                    entry['total_return'] = float(entry['annualized_return'])
                if 'annualized_excess' in entry:
                    entry['excess_return'] = float(entry['annualized_excess'])
                else:
                    # CSV may lack annualized_excess; don't mix cumulative with annualized
                    entry['excess_return'] = None
                result.setdefault(combo, []).append(entry)

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
        """Analyze model contributions from leaderboard and LOO contribution files."""
        model_excess = {}  # model -> list of excess returns
        loo_deltas = {}    # model -> list of LOO deltas

        # 1. From leaderboards
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

        # 2. From LOO contribution files
        for path in ctx.model_contribution_files:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                contribs = data.get('contributions', {})
                for model, metrics in contribs.items():
                    delta = metrics.get('delta', 0)
                    if model not in loo_deltas:
                        loo_deltas[model] = []
                    loo_deltas[model].append(delta)
            except Exception:
                continue

        consistently_negative = []
        # Check both metrics: negative excess return AND negative LOO delta (drag on ensemble)
        all_models = set(list(model_excess.keys()) + list(loo_deltas.keys()))
        
        for model in all_models:
            excesses = model_excess.get(model, [])
            deltas = loo_deltas.get(model, [])
            
            # Consistently negative if:
            # - Excess return is negative across snapshots
            # - OR LOO delta is negative (meaning removing it IMPROVES the ensemble)
            is_neg = False
            if excesses and len(excesses) >= 2 and all(e < 0 for e in excesses):
                is_neg = True
            if deltas and len(deltas) >= 2 and all(d < 0 for d in deltas):
                is_neg = True
                
            if is_neg:
                consistently_negative.append(model)

        return {
            'model_excess': {k: {'mean': np.mean(v), 'count': len(v)}
                            for k, v in model_excess.items()},
            'loo_deltas': {k: {'mean': np.mean(v), 'count': len(v)}
                          for k, v in loo_deltas.items()},
            'consistently_negative': consistently_negative,
        }

    def _load_performance_history(self, ctx: AnalysisContext) -> List[dict]:
        """
        Load unified combo performance history from all available sources.

        Priority / merge strategy:
          1. fusion_run_ledger.jsonl  — every ensemble_fusion.py execution
          2. run_metadata.json files  — brute_force / minentropy search runs (fallback)

        Records are returned in chronological order and NOT filtered by OOS flag,
        allowing callers to split full vs OOS views themselves.
        """
        records = []
        seen = set()  # (date, combo) dedup key

        # --- Source 1: fusion_run_ledger.jsonl ---
        ledger_path = os.path.join(ctx.workspace_root, 'data', 'fusion_run_ledger.jsonl')
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        date_str = rec.get('run_date', '')
                        combo = rec.get('combo_name', 'default')
                        key = (date_str, combo)
                        if key in seen:
                            continue
                        seen.add(key)
                        m = rec.get('metrics', {})
                        ew = rec.get('eval_window', {})
                        records.append({
                            'date': date_str,
                            'combo': combo,
                            'source': 'fusion_ledger',
                            'is_oos': ew.get('is_oos', False),
                            'only_last_years': ew.get('only_last_years', 0),
                            'only_last_months': ew.get('only_last_months', 0),
                            'eval_start': ew.get('start'),
                            'eval_end': ew.get('end'),
                            'calmar': m.get('calmar'),
                            'annualized_return': m.get('annualized_return'),
                            'annualized_excess': m.get('annualized_excess'),
                            'max_drawdown': m.get('max_drawdown'),
                            'loo_contributions': rec.get('loo_contributions', {}),
                            'models': rec.get('models', []),
                        })
            except Exception:
                pass

        # --- Source 2: run_metadata.json (brute_force / minentropy fallback) ---
        ensemble_dir = os.path.join(ctx.workspace_root, 'output', 'ensemble')
        metadata_paths = glob.glob(
            os.path.join(ensemble_dir, '**', 'run_metadata.json'), recursive=True
        ) if os.path.isdir(ensemble_dir) else []

        for path in metadata_paths:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                oos = data.get('oos_metrics') or data.get('oos')
                if not oos:
                    continue
                date_str = (
                    self._extract_date(path)
                    or self._extract_date(os.path.dirname(path))
                    or data.get('run_date', '')
                )
                if not date_str:
                    continue
                combo = data.get('combo_name', 'brute_force')
                key = (date_str, combo)
                if key in seen:
                    continue  # already covered by ledger
                seen.add(key)
                records.append({
                    'date': date_str,
                    'combo': combo,
                    'source': 'run_metadata',
                    'is_oos': True,  # run_metadata always contains OOS results
                    'only_last_years': data.get('exclude_last_years', 0),
                    'only_last_months': data.get('exclude_last_months', 0),
                    'eval_start': data.get('oos_start_date'),
                    'eval_end': data.get('oos_end_date'),
                    'calmar': oos.get('oos_calmar', oos.get('calmar')),
                    'annualized_return': oos.get('ann_ret'),
                    'annualized_excess': oos.get('oos_excess_return', oos.get('excess_return')),
                    'max_drawdown': oos.get('max_dd'),
                    'loo_contributions': {},
                    'models': [],
                })
            except Exception:
                continue

        records.sort(key=lambda x: x['date'])
        return records

    def _load_oos_history(self, ctx: AnalysisContext) -> dict:
        """
        Analyze OOS performance trend across all historical runs.

        Uses fusion_run_ledger.jsonl as primary source (every ensemble_fusion.py
        run that uses --only-last-years/months is tagged is_oos=True), with
        run_metadata.json from brute_force as fallback.

        Also computes IS→OOS decay ratio when the same combo has both full and
        OOS snapshots on similar dates.
        """
        all_records = self._load_performance_history(ctx)
        if not all_records:
            return {}

        # Split into OOS-only records
        oos_records = [r for r in all_records if r.get('is_oos')]
        # All records (full + OOS) for decay ratio calculation
        full_records = [r for r in all_records if not r.get('is_oos')]

        if not oos_records:
            # No dedicated OOS runs, but we still report that data exists
            return {
                'runs_analyzed': len(all_records),
                'oos_runs': 0,
                'note': 'No OOS-flagged runs found. Use --only-last-years or --only-last-months to tag OOS evaluations.',
            }

        calmars = [r['calmar'] for r in oos_records if r.get('calmar') is not None]
        excesses = [r['annualized_excess'] for r in oos_records if r.get('annualized_excess') is not None]

        result = {
            'runs_analyzed': len(all_records),
            'oos_runs': len(oos_records),
            'history': oos_records[-5:],  # Last 5 OOS runs
        }

        if calmars:
            result['latest_oos_calmar'] = float(calmars[-1])
            result['best_oos_calmar'] = float(max(calmars))
            if len(calmars) >= 2:
                x = np.arange(len(calmars))
                slope, _ = np.polyfit(x, np.array(calmars, dtype=float), 1)
                result['oos_calmar_slope'] = float(slope)

        if excesses:
            result['latest_oos_excess'] = float(excesses[-1])
            result['avg_oos_excess'] = float(np.mean(excesses))

        # IS→OOS decay: compare most recent full vs most recent OOS for same combo
        decay_ratios = []
        if full_records and oos_records:
            combos = set(r['combo'] for r in oos_records)
            for combo in combos:
                oos_combo = [r for r in oos_records if r['combo'] == combo and r.get('calmar')]
                full_combo = [r for r in full_records if r['combo'] == combo and r.get('calmar')]
                if oos_combo and full_combo:
                    latest_oos = oos_combo[-1]['calmar']
                    latest_full = full_combo[-1]['calmar']
                    if latest_full and latest_full != 0:
                        decay = (latest_oos - latest_full) / abs(latest_full)
                        decay_ratios.append({'combo': combo, 'decay_ratio': round(decay, 4),
                                             'oos_calmar': latest_oos, 'full_calmar': latest_full})
        if decay_ratios:
            result['is_oos_decay'] = decay_ratios

        return result

    def _identify_best_combo(self, combo_trends: Dict[str, list]) -> Optional[dict]:
        """Identify the best performing combo from latest snapshots."""
        candidates = []
        for combo_name, series in combo_trends.items():
            if series:
                latest = series[-1]
                candidates.append({
                    'name': combo_name,
                    'total_return': latest.get('total_return') if latest.get('total_return') is not None else 0.0,
                    'calmar_ratio': latest.get('calmar_ratio') if latest.get('calmar_ratio') is not None else 0.0,
                    'excess_return': latest.get('excess_return') if latest.get('excess_return') is not None else 0.0,
                })

        if not candidates:
            return None

        # Sort by calmar_ratio as primary metric
        candidates.sort(key=lambda x: x['calmar_ratio'], reverse=True)
        return candidates[0]

    @staticmethod
    def _extract_date(path: str) -> Optional[str]:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(path))
        return m.group(1) if m else None
