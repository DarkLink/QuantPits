"""
Signal Extractor for the MAS Deep Analysis System (Phase 3).

Pure rule-based layer that converts Agent raw_metrics into structured Signals.
Signals are the sole input to the LLM Critic Agent, avoiding direct LLM
exposure to voluminous raw data.
"""

import logging
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

    def __init__(self, reference_date: Optional[str] = None):
        """
        Args:
            reference_date: YYYY-MM-DD string used for staleness checks.
                            Defaults to today.
        """
        self._ref_date = (
            datetime.strptime(reference_date, "%Y-%m-%d")
            if reference_date else datetime.now()
        )

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

        return signals

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

            # Rule 2: severe_underfitting — actual < configured * 0.25
            for model, detail in model_details.items():
                actual = detail.get("actual_epochs")
                configured = detail.get("configured_epochs")
                if actual is not None and configured is not None and configured > 0:
                    if actual < configured * 0.25:
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
                                    "ratio": round(actual / configured, 3),
                                },
                                context=(
                                    f"{model} severe underfitting: only "
                                    f"{actual}/{configured} epochs "
                                    f"({actual/configured*100:.0f}%)"
                                ),
                            ))
                        else:
                            # Upgrade existing underfitting to severe
                            for s in signals:
                                if s.signal_type == "underfitting" and s.target == model:
                                    s.signal_type = "severe_underfitting"
                                    s.metrics["ratio"] = round(actual / configured, 3)
                                    s.context = (
                                        f"{model} severe underfitting: only "
                                        f"{actual}/{configured} epochs "
                                        f"({actual/configured*100:.0f}%)"
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
