"""
Model Health Agent.

Analyzes individual model IC/ICIR trends, retrain detection, hyperparameter
snapshots, and cross-window stability.
"""

import os
import json
import glob
import re
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from ..base_agent import BaseAgent, AgentFindings, AnalysisContext, Finding


class ModelHealthAgent(BaseAgent):
    name = "Model Health"
    description = "Evaluates model IC/ICIR trends, retrain events, and hyperparameter configurations."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        # --- 1. Load model performance time series ---
        perf_series = self._load_performance_series(ctx)
        if not perf_series:
            findings.append(self._make_finding(
                'info', 'No model performance data',
                'No model_performance_*.json files found in the analysis window.'))
            return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)

        # --- 2. IC/ICIR Trend Analysis ---
        scorecard = {}
        for model_name, series in perf_series.items():
            ic_values = [s['IC_Mean'] for s in series if 'IC_Mean' in s]
            icir_values = [s['ICIR'] for s in series if 'ICIR' in s]
            dates = [s['_date'] for s in series if 'IC_Mean' in s]

            if len(ic_values) < 2:
                scorecard[model_name] = {
                    'ic_mean': ic_values[0] if ic_values else None,
                    'icir_mean': icir_values[0] if icir_values else None,
                    'ic_trend': 'insufficient_data',
                    'n_snapshots': len(ic_values),
                }
                continue

            ic_arr = np.array(ic_values, dtype=float)
            x = np.arange(len(ic_arr), dtype=float)

            # Linear regression for trend
            slope, intercept = np.polyfit(x, ic_arr, 1)
            ic_mean = float(ic_arr.mean())
            ic_std = float(ic_arr.std())
            icir_mean = float(np.mean(icir_values)) if icir_values else None

            # Trend label based on slope significance
            if ic_std > 0:
                t_stat = slope / (ic_std / np.sqrt(len(x)))
            else:
                t_stat = 0

            if t_stat > 1.5:
                trend = 'improving'
            elif t_stat < -1.5:
                trend = 'degrading'
            else:
                trend = 'stable'

            scorecard[model_name] = {
                'ic_mean': ic_mean,
                'ic_latest': float(ic_arr[-1]),
                'icir_mean': icir_mean,
                'ic_trend': trend,
                'ic_slope': float(slope),
                'ic_trend_t': float(t_stat),
                'n_snapshots': len(ic_values),
                'date_range': f"{dates[0]} → {dates[-1]}",
            }

            # Generate findings for concerning models
            if trend == 'degrading' and ic_mean > 0:
                findings.append(self._make_finding(
                    'warning', f'{model_name}: IC declining',
                    f'IC trend is degrading (slope={slope:.4f}/week, t={t_stat:.2f}). '
                    f'Mean IC: {ic_mean:.4f}, Latest IC: {ic_arr[-1]:.4f}.',
                    {'model': model_name, 'ic_mean': ic_mean, 'slope': slope}
                ))
            elif ic_mean < 0:
                findings.append(self._make_finding(
                    'critical', f'{model_name}: Negative IC',
                    f'Model has negative mean IC ({ic_mean:.4f}), indicating predictions '
                    f'are inversely correlated with actual returns.',
                    {'model': model_name, 'ic_mean': ic_mean}
                ))
                recommendations.append(
                    f"Consider disabling {model_name} from active ensemble combinations."
                )
            elif trend == 'improving':
                findings.append(self._make_finding(
                    'positive', f'{model_name}: IC improving',
                    f'IC trend is positive (slope={slope:.4f}/week). '
                    f'Mean IC: {ic_mean:.4f}.',
                    {'model': model_name, 'ic_mean': ic_mean, 'slope': slope}
                ))

        raw_metrics['scorecard'] = scorecard

        # --- 3. Retrain Detection ---
        retrain_events = self._detect_retrains(ctx)
        raw_metrics['retrain_events'] = retrain_events

        if retrain_events:
            for event in retrain_events:
                findings.append(self._make_finding(
                    'info', f"{event['model']}: Retrained on {event['date']}",
                    f"Record ID changed from {event.get('old_record', '?')[:8]}... "
                    f"to {event.get('new_record', '?')[:8]}...",
                    event
                ))

        # Staleness check
        stale_models = self._check_staleness(ctx, retrain_events, scorecard)
        for model_name, info in stale_models.items():
            if info.get('recommend_retrain'):
                findings.append(self._make_finding(
                    'warning', f'{model_name}: Potentially stale',
                    f"Last retrain: {info.get('last_retrain', 'unknown')}. "
                    f"IC trend: {info.get('ic_trend', '?')}. "
                    f"Consider retraining.",
                    info
                ))
                recommendations.append(
                    f"Model {model_name} may benefit from retraining "
                    f"(last retrain: {info.get('last_retrain', 'unknown')}, "
                    f"IC trend: {info.get('ic_trend', '?')})."
                )
        raw_metrics['stale_models'] = stale_models

        # --- 4. Hyperparameter Snapshot ---
        hyperparams = self._load_hyperparams(ctx)
        raw_metrics['hyperparams'] = hyperparams

        # --- 5. Static vs Rolling ---
        rolling_status = self._check_rolling_status(ctx)
        raw_metrics['rolling_status'] = rolling_status
        if rolling_status == 'not_active':
            findings.append(self._make_finding(
                'info', 'Rolling training not active',
                'Rolling training mode has not been started. '
                'All models are using static training mode.'))

        # --- 6. Top/Bottom summary ---
        if scorecard:
            sorted_models = sorted(
                [(k, v) for k, v in scorecard.items() if v.get('ic_mean') is not None],
                key=lambda x: x[1]['ic_mean'], reverse=True
            )
            if len(sorted_models) >= 3:
                top3 = [f"{m} (IC={v['ic_mean']:.4f})" for m, v in sorted_models[:3]]
                bot3 = [f"{m} (IC={v['ic_mean']:.4f})" for m, v in sorted_models[-3:]]
                findings.append(self._make_finding(
                    'info', 'Model ranking summary',
                    f"Top 3: {', '.join(top3)}. Bottom 3: {', '.join(bot3)}.",
                    {'top3': sorted_models[:3], 'bottom3': sorted_models[-3:]}
                ))

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_performance_series(self, ctx: AnalysisContext) -> Dict[str, list]:
        """Load model_performance_*.json files into per-model time series."""
        result = {}  # model_name -> [{date, IC_Mean, ICIR, record_id}, ...]

        for path in ctx.model_performance_files:
            date_str = self._extract_date_from_path(path)
            if not date_str:
                continue
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            for model_name, metrics in data.items():
                if not isinstance(metrics, dict):
                    continue
                entry = {'_date': date_str}
                entry.update(metrics)
                if model_name not in result:
                    result[model_name] = []
                result[model_name].append(entry)

        # Sort each series by date
        for model_name in result:
            result[model_name].sort(key=lambda x: x['_date'])

        return result

    def _detect_retrains(self, ctx: AnalysisContext) -> List[dict]:
        """Detect model retrains by tracking record_id changes across snapshots."""
        events = []
        perf_series = self._load_performance_series(ctx)

        record_mode_cache = {}

        def is_predict_only(record_id: str) -> bool:
            if record_id in record_mode_cache:
                return record_mode_cache[record_id]

            mlruns_dir = os.path.join(ctx.workspace_root, 'mlruns')
            if not os.path.exists(mlruns_dir):
                record_mode_cache[record_id] = False
                return False

            mode_files = glob.glob(os.path.join(mlruns_dir, '*', record_id, 'tags', 'mode'))
            if mode_files:
                try:
                    with open(mode_files[0], 'r') as f:
                        mode = f.read().strip()
                        is_predict = (mode == 'predict_only')
                        record_mode_cache[record_id] = is_predict
                        return is_predict
                except Exception:
                    pass

            record_mode_cache[record_id] = False
            return False

        for model_name, series in perf_series.items():
            prev_record = None
            for entry in series:
                curr_record = entry.get('record_id')
                
                if curr_record:
                    if is_predict_only(curr_record):
                        continue
                        
                    if prev_record and curr_record != prev_record:
                        events.append({
                            'model': model_name,
                            'date': entry['_date'],
                            'old_record': prev_record,
                            'new_record': curr_record,
                        })
                    prev_record = curr_record

        return sorted(events, key=lambda x: x['date'])

    def _check_staleness(self, ctx: AnalysisContext,
                         retrain_events: List[dict],
                         scorecard: dict) -> Dict[str, dict]:
        """Check if models are stale (long since retrain + declining IC)."""
        stale = {}
        # Build last retrain date per model
        last_retrain = {}
        for event in retrain_events:
            last_retrain[event['model']] = event['date']

        for model_name, sc in scorecard.items():
            lr_date = last_retrain.get(model_name)
            trend = sc.get('ic_trend', 'stable')

            # Recommend retrain if IC is degrading and we know the model hasn't been retrained
            # or if there's no retrain record at all (always static)
            recommend = False
            if trend == 'degrading':
                recommend = True
            elif trend == 'stable' and sc.get('ic_mean', 0) < 0.02:
                recommend = True

            if recommend:
                stale[model_name] = {
                    'last_retrain': lr_date or 'no retrain record',
                    'ic_trend': trend,
                    'ic_mean': sc.get('ic_mean'),
                    'recommend_retrain': True,
                }

        return stale

    def _load_hyperparams(self, ctx: AnalysisContext) -> Dict[str, dict]:
        """Parse workflow_config_*.yaml files for hyperparameter summary."""
        config_dir = os.path.join(ctx.workspace_root, 'config')
        result = {}

        yaml_pattern = os.path.join(config_dir, 'workflow_config_*.yaml')
        for yaml_path in sorted(glob.glob(yaml_pattern)):
            basename = os.path.basename(yaml_path)
            m = re.match(r'workflow_config_(.+)\.yaml', basename)
            if not m:
                continue
            model_key = m.group(1)

            try:
                with open(yaml_path, 'r') as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                continue

            info = {}
            task = cfg.get('task', {})
            model_cfg = task.get('model', {})
            if model_cfg:
                info['class'] = model_cfg.get('class', 'Unknown')
                kwargs = model_cfg.get('kwargs', {})
                for key in ['d_model', 'd_feat', 'hidden_size', 'num_layers', 'dropout',
                            'n_epochs', 'lr', 'batch_size', 'loss', 'optimizer',
                            'loss_function', 'iterations', 'learning_rate',
                            'depth', 'l2_leaf_reg', 'num_leaves', 'max_depth',
                            'n_estimators', 'early_stop', 'GPU']:
                    if key in kwargs:
                        info[key] = kwargs[key]

            dh = cfg.get('data_handler_config', {})
            label_list = dh.get('label', [])
            if label_list and isinstance(label_list, list):
                info['label'] = label_list[0]

            result[model_key] = info

        return result

    def _check_rolling_status(self, ctx: AnalysisContext) -> str:
        """Check if rolling training is active."""
        records_path = os.path.join(ctx.workspace_root, 'latest_train_records.json')
        if not os.path.exists(records_path):
            return 'unknown'

        try:
            with open(records_path, 'r') as f:
                records = json.load(f)
            models = records.get('models', {})
            rolling_models = [k for k in models if '@rolling' in k]
            if rolling_models:
                return 'active'
            return 'not_active'
        except Exception:
            return 'unknown'

    @staticmethod
    def _extract_date_from_path(path: str) -> Optional[str]:
        """Extract date from a file path like model_performance_2026-04-17.json."""
        m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(path))
        return m.group(1) if m else None
