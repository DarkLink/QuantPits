"""
Signal Extractor for the MAS Deep Analysis System (Phase 3).

Pure rule-based layer that converts Agent raw_metrics into structured Signals.
Signals are the sole input to the LLM Critic Agent, avoiding direct LLM
exposure to voluminous raw data.
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .base_agent import AgentFindings

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A structured observation extracted from Agent raw_metrics."""
    signal_type: str       # "underfitting" | "overfitting" | "ic_decay" | ...
    severity: str          # "critical" | "warning" | "info"
    scope: str             # feedback_scope key: "hyperparams" | "model_selection" | ...
    source_agent: str      # Agent name that produced the raw data
    target: str            # Affected model/combo name
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: str = ""      # One-line natural language description

    def to_dict(self) -> dict:
        return asdict(self)


class SignalExtractor:
    """
    Extract structured Signals from Agent findings and synthesis results.

    This is a pure rule layer — it does NOT make decisions.  The LLM Critic
    is responsible for interpreting signals and producing ActionItems.
    """

    def __init__(self, reference_date: Optional[str] = None,
                 workspace_root: Optional[str] = None,
                 window_analysis_findings: Optional[List] = None):
        """
        Args:
            reference_date: YYYY-MM-DD string used for staleness checks.
                            Defaults to today.
            workspace_root: Path to workspace root dir.  If provided, the
                            extractor can read training_history.jsonl for
                            per-epoch loss analysis (optimizer thrashing).
            window_analysis_findings: Pre-computed findings from
                            TrainingWindowAnalyzer (if available).
        """
        self._ref_date = (
            datetime.strptime(reference_date, "%Y-%m-%d")
            if reference_date else datetime.now()
        )
        self._workspace_root = workspace_root
        self._window_analysis_findings = window_analysis_findings or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        all_findings: List[AgentFindings],
        synthesis_result: dict,
    ) -> List[Signal]:
        """
        Extract signals from agent findings and synthesis result.

        Args:
            all_findings: List of AgentFindings from the coordinator.
            synthesis_result: Output from Synthesizer.synthesize().

        Returns:
            List of structured Signal objects.
        """
        # Index findings by agent name for efficient lookup
        by_agent: Dict[str, List[AgentFindings]] = {}
        for af in all_findings:
            by_agent.setdefault(af.agent_name, []).append(af)

        signals: List[Signal] = []

        # --- Model Health signals ---
        signals.extend(self._extract_model_health(by_agent))

        # --- Ensemble Eval signals ---
        signals.extend(self._extract_ensemble_eval(by_agent))

        # --- Market Regime signals ---
        signals.extend(self._extract_market_regime(by_agent))

        # --- Prediction Audit signals ---
        signals.extend(self._extract_prediction_audit(by_agent))

        # --- Cross-agent convergence ---
        signals.extend(self._extract_cross_agent_convergence(signals))

        # --- Training history based (optimizer thrashing, etc.) ---
        if self._workspace_root:
            signals.extend(self._extract_optimizer_thrashing())

        # --- Training Window Analysis signals (rule-based, independent) ---
        if self._window_analysis_findings:
            signals.extend(self._extract_training_window())

        return signals

    def extract_triage_input(
        self,
        all_findings: List[AgentFindings],
        synthesis_result: dict,
        signals: List[Signal],
    ) -> dict:
        """
        Build a structured input payload for the Triage LLM.

        Returns a dict with: market_context, model_ranking, family_stats,
        combo_summary, signal_distribution.  Designed to give the Triage
        LLM a compact but complete picture of the current state.
        """
        by_agent: Dict[str, List[AgentFindings]] = {}
        for af in all_findings:
            by_agent.setdefault(af.agent_name, []).append(af)

        # --- 1. Market context ---
        market_context = self._build_market_context(by_agent, synthesis_result)

        # --- 2. Model ranking table ---
        combo_membership = self._load_combo_membership()
        historical_flags = self._load_historical_flags()
        model_ranking = self._build_model_ranking(by_agent, combo_membership, historical_flags)

        # --- 3. Architecture family statistics ---
        family_stats = self._build_family_stats(model_ranking)

        # --- 4. Combo performance summary ---
        combo_summary = self._build_combo_summary(by_agent)

        # --- 5. Signal distribution ---
        signal_dist = self._build_signal_distribution(signals)

        return {
            "market_context": market_context,
            "model_ranking": model_ranking,
            "family_stats": family_stats,
            "combo_summary": combo_summary,
            "signal_distribution": signal_dist,
            "training_window_analysis": [
                f.to_dict() for f in self._window_analysis_findings
            ] if self._window_analysis_findings else [],
        }

    # ------------------------------------------------------------------
    # Triage input builders
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_architecture(model_name: str) -> str:
        """Classify a model into its architecture family from its name."""
        name_lower = model_name.lower()
        # Alpha360 families — check more specific patterns first
        if 'sfm' in name_lower and '360' in name_lower:
            return 'Alpha360_SFM'
        if 'adarnn' in name_lower and '360' in name_lower:
            return 'Alpha360_ADARNN'
        if 'tcts' in name_lower and '360' in name_lower:
            return 'Alpha360_TCTS'
        if 'tra_' in name_lower and '360' in name_lower and 'transformer' not in name_lower:
            return 'Alpha360_TRA'
        if 'alstm' in name_lower and '360' in name_lower:
            return 'Alpha360_Attention'
        if 'igmtf' in name_lower and '360' in name_lower:
            return 'Alpha360_Attention'
        if 'tabnet' in name_lower and '360' in name_lower:
            return 'Alpha360_TabNet'
        if 'add' in name_lower and '360' in name_lower:
            return 'Alpha360_ADD'
        if any(a in name_lower for a in ['transformer', 'localformer']) and '360' in name_lower:
            return 'Alpha360_Transformer'
        if any(a in name_lower for a in ['tcn', 'sandwich', 'krnn']) and '360' in name_lower:
            return 'Alpha360_Structural'
        if any(a in name_lower for a in ['gru', 'lstm', 'rnn']) and '360' in name_lower:
            return 'Alpha360_RNN'
        # Alpha158 families
        if any(a in name_lower for a in ['gru', 'lstm', 'gats']) and '158' in name_lower:
            return 'Alpha158_RNN'
        if any(a in name_lower for a in ['transformer', 'localformer', 'tft']) and '158' in name_lower:
            return 'Alpha158_Transformer'
        if any(a in name_lower for a in ['tabnet', 'catboost', 'lightgbm', 'mlp']) and '158' in name_lower:
            return 'Alpha158_Tabular'
        if 'tra_' in name_lower and '158' in name_lower and 'transformer' not in name_lower:
            return 'Alpha158_TRA'
        if 'tcts' in name_lower and '360' in name_lower:
            return 'Alpha360_TCTS'
        # Catch-all
        if '360' in name_lower:
            return 'Alpha360_Other'
        if '158' in name_lower:
            return 'Alpha158_Other'
        return 'Unknown'

    def _load_combo_membership(self) -> Dict[str, List[str]]:
        """Load ensemble_config.json and return {model_name: [combo_names]}."""
        if not self._workspace_root:
            return {}
        path = os.path.join(self._workspace_root, 'config', 'ensemble_config.json')
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r') as f:
                config = json.load(f)
        except Exception:
            return {}
        membership: Dict[str, List[str]] = {}
        combos = config.get('combos', {})
        for combo_name, combo_info in combos.items():
            if not isinstance(combo_info, dict):
                continue
            for m in combo_info.get('models', []):
                if isinstance(m, str):
                    membership.setdefault(m, []).append(combo_name)
        return membership

    def _load_historical_flags(self) -> Dict[str, dict]:
        """Scan last 3 action_item snapshots and flag historically problematic models.

        Returns {model_name: {historical_signal_types, run_count, last_action_types}}.
        This is a deterministic rule layer — the Triage LLM uses these flags to
        ensure historically troubled models are routed even when they lack new data.
        """
        if not self._workspace_root:
            return {}
        import glob as _glob
        pattern = os.path.join(
            self._workspace_root, 'output', 'deep_analysis', 'action_items_*.json',
        )
        files = sorted(_glob.glob(pattern))
        if not files:
            return {}

        # Look at last 3 runs, tracking per-file presence
        recent_files = files[-3:]
        flags: Dict[str, dict] = {}

        for fpath in recent_files:
            try:
                with open(fpath, 'r') as f:
                    items = json.load(f)
            except Exception:
                continue
            if not isinstance(items, list):
                continue

            seen_in_this_file: set = set()
            for item in items:
                target = item.get('target', '')
                if not target:
                    continue
                if target not in flags:
                    flags[target] = {
                        'historical_signal_types': [],
                        'run_count': 0,
                        'last_action_types': [],
                    }
                entry = flags[target]
                if target not in seen_in_this_file:
                    seen_in_this_file.add(target)
                    entry['run_count'] += 1

                action_type = item.get('action_type', '')
                if action_type and action_type not in entry['last_action_types']:
                    entry['last_action_types'].append(action_type)

                reason = (item.get('reason', '') or '').lower()
                sigs = entry['historical_signal_types']
                if any(kw in reason for kw in [
                    'underfitting', 'underfit', 'early_stop',
                    'best_epoch', '欠拟合', 'early_stopped too',
                ]):
                    if 'underfitting' not in sigs:
                        sigs.append('underfitting')
                if any(kw in reason for kw in [
                    'ic_decay', 'ic decay', 'ic 衰减', 'ic attenuat',
                    'degrading ic', 'ic trend', 'ic 持续',
                ]):
                    if 'ic_decay' not in sigs:
                        sigs.append('ic_decay')
                if any(kw in reason for kw in [
                    'ic≈0', 'ic ≈ 0', 'ic ~= 0', 'ic 接近零',
                    'ic 极低', 'ic 贫弱', 'negative_ic', 'negative ic',
                ]):
                    if 'negative_ic' not in sigs:
                        sigs.append('negative_ic')

        return flags

    def _build_market_context(self, by_agent, synthesis_result) -> dict:
        """Extract market regime and recent portfolio performance."""
        ctx = {}
        for af in by_agent.get('Market Regime', []):
            rm = af.raw_metrics
            switches = rm.get('regime_switches', {})
            ctx['current_regime'] = switches.get('current_regime', 'unknown')
            ctx['regime_switch_count'] = switches.get('switch_count', 0)
        exec_data = synthesis_result.get('executive_summary_data', {})
        ctx['market_regime_label'] = exec_data.get('market_regime', 'unknown')
        ctx['cagr_1y'] = exec_data.get('cagr_1y')
        ctx['sharpe_1y'] = exec_data.get('sharpe_1y')
        return ctx

    def _build_model_ranking(
        self, by_agent, combo_membership: Dict[str, List[str]],
        historical_flags: Optional[Dict[str, dict]] = None,
    ) -> list:
        """Build a ranked list of models with key metrics (deduplicated across windows)."""
        if historical_flags is None:
            historical_flags = {}
        # Collect entries across all windows, keep the one with most snapshots
        best_per_model: Dict[str, dict] = {}
        for af in by_agent.get('Model Health', []):
            scorecard = af.raw_metrics.get('scorecard', {})
            conv = af.raw_metrics.get('convergence_summary', {})
            details = conv.get('model_details', {})
            for model_name, sc in scorecard.items():
                ic_mean = sc.get('ic_mean')
                if ic_mean is None:
                    continue
                n_snap = sc.get('n_snapshots', 0)
                if model_name in best_per_model and best_per_model[model_name]['n_snapshots'] >= n_snap:
                    continue
                entry = {
                    'model': model_name,
                    'ic_mean': round(ic_mean, 4),
                    'icir_mean': round(sc.get('icir_mean'), 4) if sc.get('icir_mean') is not None else None,
                    'ic_trend': sc.get('ic_trend', '?'),
                    'n_snapshots': n_snap,
                    'family': self._classify_architecture(model_name),
                    'in_combos': combo_membership.get(model_name, []),
                }
                d = details.get(model_name, {})
                if d:
                    entry['best_epoch'] = d.get('best_epoch')
                    entry['actual_epochs'] = d.get('actual_epochs')
                    entry['early_stopped'] = d.get('early_stopped')
                # Merge historical tracking flags from past ActionItem runs
                hf = historical_flags.get(model_name)
                if hf:
                    entry['historical_flags'] = hf
                best_per_model[model_name] = entry

        ranking = list(best_per_model.values())
        ranking.sort(
            key=lambda x: x.get('icir_mean') or x.get('ic_mean') or 0,
            reverse=True,
        )
        return ranking

    def _build_family_stats(self, model_ranking: list) -> dict:
        """Compute per-family aggregate statistics."""
        families: Dict[str, list] = {}
        for m in model_ranking:
            families.setdefault(m['family'], []).append(m)

        stats = {}
        for family, models in families.items():
            ic_values = [m['ic_mean'] for m in models if m['ic_mean'] is not None]
            icir_values = [m['icir_mean'] for m in models if m['icir_mean'] is not None]
            if not ic_values:
                continue
            stats[family] = {
                'count': len(models),
                'avg_ic': round(sum(ic_values) / len(ic_values), 4),
                'best_ic': round(max(ic_values), 4),
                'worst_ic': round(min(ic_values), 4),
                'avg_icir': round(sum(icir_values) / len(icir_values), 4) if icir_values else None,
                'models': [m['model'] for m in models],
            }
        return stats

    def _build_combo_summary(self, by_agent) -> dict:
        """Extract combo performance summary from Ensemble Evolution."""
        summary = {}
        for af in by_agent.get('Ensemble Evolution', []):
            rm = af.raw_metrics
            combo_trends = rm.get('combo_trends', {})
            for combo_name, entries in combo_trends.items():
                if not entries:
                    continue
                latest = max(entries, key=lambda e: e.get('_date', ''))
                summary[combo_name] = {
                    'latest_excess': latest.get('excess_return'),
                    'latest_calmar': latest.get('calmar'),
                    'latest_sharpe': latest.get('sharpe'),
                    'n_entries': len(entries),
                }
            # Add OOS trend
            oos = rm.get('oos_trend', {})
            if oos:
                summary['_oos_trend'] = {
                    'calmar_slope': oos.get('oos_calmar_slope'),
                    'runs': oos.get('oos_runs'),
                    'latest_calmar': oos.get('latest_oos_calmar'),
                    'split_definition': oos.get('split_definition', {}),
                }
        return summary

    def _build_signal_distribution(self, signals: List[Signal]) -> dict:
        """Build summary statistics of extracted signals."""
        by_type: Dict[str, dict] = {}
        by_severity: Dict[str, int] = {}
        for s in signals:
            by_severity[s.severity] = by_severity.get(s.severity, 0) + 1
            if s.signal_type not in by_type:
                by_type[s.signal_type] = {'count': 0, 'severity': s.severity, 'targets': []}
            by_type[s.signal_type]['count'] += 1
            if s.target not in by_type[s.signal_type]['targets']:
                by_type[s.signal_type]['targets'].append(s.target)

        return {
            'total_signals': len(signals),
            'by_severity': by_severity,
            'by_type': {
                st: {'count': info['count'], 'sample_targets': info['targets'][:5]}
                for st, info in sorted(by_type.items(), key=lambda x: -x[1]['count'])
            },
            'unique_targets': len(set(s.target for s in signals)),
        }

    # ------------------------------------------------------------------
    # Model Health extraction
    # ------------------------------------------------------------------

    def _extract_model_health(
        self, by_agent: Dict[str, List[AgentFindings]]
    ) -> List[Signal]:
        signals: List[Signal] = []

        for af in by_agent.get("Model Health", []):
            rm = af.raw_metrics

            # --- convergence_summary ---
            conv = rm.get("convergence_summary", {})
            model_details = conv.get("model_details", {})

            # Rule 1: underfitting_candidates (non-empty)
            for model in conv.get("underfitting_candidates", []):
                detail = model_details.get(model, {})
                signals.append(Signal(
                    signal_type="underfitting",
                    severity="warning",
                    scope="hyperparams",
                    source_agent="Model Health",
                    target=model,
                    metrics={
                        "actual_epochs": detail.get("actual_epochs"),
                        "configured_epochs": detail.get("configured_epochs"),
                        "early_stopped": detail.get("early_stopped"),
                    },
                    context=(
                        f"{model} appears underfitting: trained {detail.get('actual_epochs', '?')}"
                        f"/{detail.get('configured_epochs', '?')} epochs"
                    ),
                ))

            # Rule 2: severe_underfitting — only when best_epoch is extremely
            # early AND the model stopped before using a meaningful fraction of
            # configured epochs.  NN healthy early-stop (epoch 20-50 for
            # n_epochs=200) is normal and should NOT trigger this signal.
            # Thresholds are aligned with ModelHealthAgent._analyze_convergence.
            for model, detail in model_details.items():
                actual = detail.get("actual_epochs")
                configured = detail.get("configured_epochs")
                if actual is not None and configured is not None and configured > 0:
                    best_ep = detail.get("best_epoch")
                    is_severe = False
                    if best_ep is not None and best_ep <= 1 and actual < configured * 0.3:
                        is_severe = True
                    elif best_ep is None and actual < configured * 0.15:
                        is_severe = True

                    if is_severe:
                        # Avoid duplicate if already flagged as underfitting
                        if not any(
                            s.signal_type == "underfitting" and s.target == model
                            for s in signals
                        ):
                            signals.append(Signal(
                                signal_type="severe_underfitting",
                                severity="warning",
                                scope="hyperparams",
                                source_agent="Model Health",
                                target=model,
                                metrics={
                                    "actual_epochs": actual,
                                    "configured_epochs": configured,
                                    "best_epoch": best_ep,
                                    "ratio": round(actual / configured, 3),
                                },
                                context=(
                                    f"{model} severe underfitting: only "
                                    f"{actual}/{configured} epochs "
                                    f"(best_epoch={best_ep}, "
                                    f"{actual/configured*100:.0f}%)"
                                ),
                            ))
                        else:
                            # Upgrade existing underfitting to severe
                            for s in signals:
                                if s.signal_type == "underfitting" and s.target == model:
                                    s.signal_type = "severe_underfitting"
                                    s.metrics["ratio"] = round(actual / configured, 3)
                                    s.metrics["best_epoch"] = best_ep
                                    s.context = (
                                        f"{model} severe underfitting: only "
                                        f"{actual}/{configured} epochs "
                                        f"(best_epoch={best_ep}, "
                                        f"{actual/configured*100:.0f}%)"
                                    )
                                    break

            # Rule 3: overfitting — full_epoch_models with low IC
            scorecard = rm.get("scorecard", {})
            for model in conv.get("full_epoch_models", []):
                sc = scorecard.get(model, {})
                ic_mean = sc.get("ic_mean")
                if ic_mean is not None and ic_mean < 0.03:
                    signals.append(Signal(
                        signal_type="overfitting",
                        severity="warning",
                        scope="hyperparams",
                        source_agent="Model Health",
                        target=model,
                        metrics={
                            "ic_mean": ic_mean,
                            "full_epoch": True,
                        },
                        context=(
                            f"{model} possible overfitting: ran full epochs "
                            f"but IC mean only {ic_mean:.4f}"
                        ),
                    ))

            # Rule 4: ic_decay
            for model, sc in scorecard.items():
                if sc.get("ic_trend") == "degrading":
                    signals.append(Signal(
                        signal_type="ic_decay",
                        severity="warning",
                        scope="hyperparams",
                        source_agent="Model Health",
                        target=model,
                        metrics={
                            "ic_trend": "degrading",
                            "ic_mean": sc.get("ic_mean"),
                        },
                        context=f"{model} IC trend is degrading",
                    ))

            # Rule 5: model_stale
            stale = rm.get("stale_models", {})
            for model, info in stale.items():
                if info.get("recommend_retrain"):
                    signals.append(Signal(
                        signal_type="model_stale",
                        severity="info",
                        scope="hyperparams",
                        source_agent="Model Health",
                        target=model,
                        metrics={
                            "last_retrain": info.get("last_retrain"),
                            "ic_trend": info.get("ic_trend"),
                        },
                        context=(
                            f"{model} is stale and recommended for retraining "
                            f"(last retrain: {info.get('last_retrain', 'unknown')})"
                        ),
                    ))

        return signals

    # ------------------------------------------------------------------
    # Ensemble Eval extraction
    # ------------------------------------------------------------------

    def _extract_ensemble_eval(
        self, by_agent: Dict[str, List[AgentFindings]]
    ) -> List[Signal]:
        signals: List[Signal] = []

        for af in by_agent.get("Ensemble Evolution", []):
            rm = af.raw_metrics

            # --- oos_trend ---
            oos = rm.get("oos_trend", {})
            slope = oos.get("oos_calmar_slope")
            runs = oos.get("oos_runs", 0)

            if slope is not None and slope < -0.3:
                if runs >= 5:
                    # Rule 6: oos_degradation (sufficient samples)
                    signals.append(Signal(
                        signal_type="oos_degradation",
                        severity="warning",
                        scope="combo_search",
                        source_agent="Ensemble Evolution",
                        target="ensemble",
                        metrics={
                            "oos_calmar_slope": slope,
                            "oos_runs": runs,
                            "latest_oos_calmar": oos.get("latest_oos_calmar"),
                            "best_oos_calmar": oos.get("best_oos_calmar"),
                            "split_definition": oos.get("split_definition", {}),
                        },
                        context=(
                            f"OOS Calmar degrading (slope={slope:.3f}, "
                            f"{runs} runs)"
                        ),
                    ))
                else:
                    # Rule 7: oos_degradation_limited_sample
                    signals.append(Signal(
                        signal_type="oos_degradation_limited_sample",
                        severity="info",
                        scope="combo_search",
                        source_agent="Ensemble Evolution",
                        target="ensemble",
                        metrics={
                            "oos_calmar_slope": slope,
                            "oos_runs": runs,
                            "sample_size_warning": True,
                            "split_definition": oos.get("split_definition", {}),
                        },
                        context=(
                            f"OOS Calmar slope negative ({slope:.3f}) but only "
                            f"{runs} runs — limited statistical reliability"
                        ),
                    ))

            # --- model_contributions ---
            contrib = rm.get("model_contributions", {})
            # Rule 8: negative_contribution
            for model in contrib.get("consistently_negative", []):
                loo = contrib.get("loo_deltas", {}).get(model, {})
                excess = contrib.get("model_excess", {}).get(model, {})
                signals.append(Signal(
                    signal_type="negative_contribution",
                    severity="warning",
                    scope="model_selection",
                    source_agent="Ensemble Evolution",
                    target=model,
                    metrics={
                        "loo_delta_mean": loo.get("mean"),
                        "loo_delta_count": loo.get("count"),
                        "excess_mean": excess.get("mean"),
                    },
                    context=(
                        f"{model} consistently negative contribution "
                        f"(LOO delta mean: {loo.get('mean', '?')})"
                    ),
                ))

            # --- combo_trends ---
            combo_trends = rm.get("combo_trends", {})
            # Rule 11: combo_stale
            for combo, entries in combo_trends.items():
                if combo == "default":
                    continue
                if not entries:
                    continue
                # Find the latest _date entry
                latest_date_str = None
                for entry in entries:
                    d = entry.get("_date")
                    if d and (latest_date_str is None or d > latest_date_str):
                        latest_date_str = d
                if latest_date_str:
                    try:
                        latest_dt = datetime.strptime(latest_date_str, "%Y-%m-%d")
                        days_ago = (self._ref_date - latest_dt).days
                        if days_ago > 30:
                            signals.append(Signal(
                                signal_type="combo_stale",
                                severity="warning",
                                scope="combo_search",
                                source_agent="Ensemble Evolution",
                                target=combo,
                                metrics={
                                    "latest_date": latest_date_str,
                                    "days_since_eval": days_ago,
                                },
                                context=(
                                    f"Combo '{combo}' last evaluated {days_ago} "
                                    f"days ago ({latest_date_str})"
                                ),
                            ))
                    except ValueError:
                        pass  # Skip unparseable dates

        return signals

    # ------------------------------------------------------------------
    # Market Regime extraction
    # ------------------------------------------------------------------

    def _extract_market_regime(
        self, by_agent: Dict[str, List[AgentFindings]]
    ) -> List[Signal]:
        signals: List[Signal] = []

        for af in by_agent.get("Market Regime", []):
            rm = af.raw_metrics

            # --- regime_switches ---
            switches = rm.get("regime_switches", {})
            switch_count = switches.get("switch_count", 0)

            # Rule 10: regime_instability
            if switch_count >= 3:
                signals.append(Signal(
                    signal_type="regime_instability",
                    severity="info",
                    scope="hyperparams",
                    source_agent="Market Regime",
                    target="market",
                    metrics={
                        "switch_count": switch_count,
                        "current_regime": switches.get("current_regime"),
                        "current_streak_days": switches.get("current_streak_days"),
                    },
                    context=(
                        f"Market regime switched {switch_count} times — "
                        f"training window may need adjustment"
                    ),
                ))

        return signals

    # ------------------------------------------------------------------
    # Prediction Audit extraction
    # ------------------------------------------------------------------

    def _extract_prediction_audit(
        self, by_agent: Dict[str, List[AgentFindings]]
    ) -> List[Signal]:
        signals: List[Signal] = []

        for af in by_agent.get("Prediction Audit", []):
            rm = af.raw_metrics

            # --- per_model_hit_rate ---
            hit = rm.get("per_model_hit_rate", {})

            # Rule 9: poor_predictor
            for model in hit.get("underperformers", []):
                ic_val = hit.get("per_model_ic", {}).get(model)
                signals.append(Signal(
                    signal_type="poor_predictor",
                    severity="info",
                    scope="model_selection",
                    source_agent="Prediction Audit",
                    target=model,
                    metrics={
                        "per_model_ic": ic_val,
                        "ensemble_overall_proxy_ic": hit.get(
                            "ensemble_overall_proxy_ic"
                        ),
                    },
                    context=(
                        f"{model} underperforming in prediction audit "
                        f"(Spearman IC: {ic_val})"
                    ),
                ))

        return signals

    # ------------------------------------------------------------------
    # Training history based signals
    # ------------------------------------------------------------------

    def _extract_optimizer_thrashing(self) -> List[Signal]:
        """Detect optimizer thrashing from per-epoch train loss in
        training_history.jsonl.

        Requires ``epoch_train_loss`` in training records, which is only
        present when models were trained with a ``_lh`` (LossHistory) wrapper.

        Thresholds from llm_agent_tuning_guide.md §2 策略D:
        - Adjacent-epoch relative change > 5% counts as one thrash event.
        - > 30% of epochs affected → critical; > 15% → warning.
        """
        history_path = os.path.join(
            self._workspace_root, "data", "training_history.jsonl")
        if not os.path.exists(history_path):
            return []

        # Only inspect the latest record per model
        latest: Dict[str, dict] = {}
        try:
            with open(history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    model = rec.get("model_name", "")
                    losses = rec.get("epoch_train_loss")
                    if not losses or len(losses) < 3:
                        continue
                    ts = rec.get("trained_at", "")
                    if model not in latest or ts > latest[model].get("trained_at", ""):
                        latest[model] = rec
        except Exception:
            logger.warning("Failed to read training_history for thrashing detection",
                           exc_info=True)
            return []

        signals: List[Signal] = []
        for model, rec in latest.items():
            losses = rec["epoch_train_loss"]

            # Relative change between consecutive epochs
            rel_changes = []
            for i in range(1, len(losses)):
                prev, cur = losses[i - 1], losses[i]
                if prev and cur and abs(prev) > 1e-8:
                    rel_changes.append(abs(cur - prev) / abs(prev))

            if not rel_changes:
                continue

            thrash_count = sum(1 for c in rel_changes if c > 0.05)
            thrash_ratio = thrash_count / len(rel_changes)

            if thrash_ratio > 0.3:
                severity = "critical"
            elif thrash_ratio > 0.15:
                severity = "warning"
            else:
                continue

            signals.append(Signal(
                signal_type="optimizer_thrashing",
                severity=severity,
                scope="hyperparams",
                source_agent="Model Health",
                target=model,
                metrics={
                    "thrash_ratio": round(thrash_ratio, 3),
                    "thrash_count": thrash_count,
                    "total_epochs": len(rel_changes),
                    "threshold": 0.05,
                },
                context=(
                    f"{model} train loss oscillates across epochs "
                    f"({thrash_count}/{len(rel_changes)} epochs, "
                    f"ratio {thrash_ratio:.0%})"
                ),
            ))

        return signals

    # ------------------------------------------------------------------
    # Training Window Analysis signals (from TrainingWindowAnalyzer)
    # ------------------------------------------------------------------

    def _extract_training_window(self) -> List[Signal]:
        """Convert TrainingWindowAnalyzer findings into Signals."""
        signals: List[Signal] = []

        severity_map = {
            "critical": "critical",
            "warning": "warning",
            "info": "info",
        }

        for finding in self._window_analysis_findings:
            finding_type = getattr(finding, "finding_type", "")
            severity = severity_map.get(
                getattr(finding, "severity", "warning"), "warning"
            )

            signals.append(Signal(
                signal_type="training_window_mismatch",
                severity=severity,
                scope="training_config",
                source_agent="Training Window Analyzer",
                target=getattr(finding, "target", "global"),
                metrics={
                    "finding_type": finding_type,
                    **(getattr(finding, "metrics", {}) or {}),
                    "recommendation": getattr(finding, "recommendation", ""),
                },
                context=getattr(finding, "context", ""),
            ))

        return signals

    # ------------------------------------------------------------------
    # Cross-agent convergence
    # ------------------------------------------------------------------

    def _extract_cross_agent_convergence(
        self, existing_signals: List[Signal]
    ) -> List[Signal]:
        """
        Rule 12: If the same target is flagged by ≥2 different agents
        at warning severity, emit a cross_agent_convergence signal.
        """
        # Group warning-level signals by target
        target_agents: Dict[str, set] = {}
        target_scopes: Dict[str, List[str]] = {}

        for s in existing_signals:
            if s.severity == "warning":
                target_agents.setdefault(s.target, set()).add(s.source_agent)
                target_scopes.setdefault(s.target, []).append(s.scope)

        convergence_signals: List[Signal] = []
        for target, agents in target_agents.items():
            if len(agents) >= 2:
                # Pick the highest-priority scope
                scope_priority = {
                    "hyperparams": 0,
                    "model_selection": 1,
                    "combo_search": 2,
                    "strategy_params": 3,
                }
                scopes = target_scopes.get(target, [])
                best_scope = min(
                    scopes,
                    key=lambda s: scope_priority.get(s, 99),
                    default="hyperparams",
                )

                convergence_signals.append(Signal(
                    signal_type="cross_agent_convergence",
                    severity="warning",
                    scope=best_scope,
                    source_agent="Cross-Agent",
                    target=target,
                    metrics={
                        "contributing_agents": sorted(agents),
                        "signal_count": sum(
                            1 for s in existing_signals
                            if s.target == target and s.severity == "warning"
                        ),
                    },
                    context=(
                        f"{target} flagged by {len(agents)} agents "
                        f"({', '.join(sorted(agents))}) — higher confidence"
                    ),
                ))

        return convergence_signals
