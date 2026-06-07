"""
Feedback Loop Orchestrator for RLFF Phase 4.

End-to-end orchestration of: ActionItem → Playground → Adapter → Retrain
→ Validation → Report / Promote.

Execution modes:
- report-only:   Load ActionItems, preview changes, generate report. No writes.
- execute:        Create Playground, apply changes, retrain, validate, report.
- promote:        Promote validated changes from previous --execute to production.
- auto-promote:   (Not yet implemented) Auto-promote if validation passes.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

from quantpits.scripts.deep_analysis.action_items import ActionItem
from quantpits.scripts.deep_analysis.signal_extractor import Signal
from quantpits.scripts.deep_analysis.adapters.base_adapter import AdapterResult
from quantpits.scripts.deep_analysis.adapters.training_adapter import TrainingAdapter
from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
from quantpits.scripts.deep_analysis.promote_config import ConfigPromoter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Single-model IC validation result."""

    model: str
    baseline_ic: float = 0.0
    playground_ic: float = 0.0
    ic_delta: float = 0.0
    ic_improved: bool = False
    passed: bool = False
    ensemble: Optional[dict] = None  # Optional EnsembleValidation data
    playground_icir: Optional[float] = None
    playground_excess: Optional[float] = None
    convergence: Optional[dict] = None  # epoch/loss data for experiment analysis
    round_idx: int = 0  # which experiment round produced this result
    params_changed: Optional[dict] = None  # {param: {from, to}} that produced this result


@dataclass
class FeedbackReport:
    """Complete report from a feedback loop run."""

    run_date: str = ""
    mode: str = ""
    action_items_processed: int = 0
    action_items_deferred: int = 0
    adapter_results: List[dict] = field(default_factory=list)
    validation_results: List[dict] = field(default_factory=list)
    deferred_action_ids: List[str] = field(default_factory=list)
    promote_result: Optional[dict] = None
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Priority computation
# ------------------------------------------------------------------

def compute_priority(
    item: ActionItem,
    signal_severity: str = "warning",
    training_history: Dict[str, float] = None,
) -> float:
    """Compute execution priority for an ActionItem.

    Higher score = higher priority.

    Dimensions:
    - signal_severity: "critical"=3, "warning"=2, "info"=1
    - confidence: LLM confidence [0, 1] → weighted ×2
    - risk_level: tiebreaker bonus
    - training_cost: shorter training → bonus

    Args:
        item: The ActionItem to score.
        signal_severity: Severity from the source Signal.
        training_history: {model_name: avg_duration_seconds}

    Returns:
        Priority score (float).
    """
    if training_history is None:
        training_history = {}

    score = 0.0

    # Severity contribution
    severity_weight = {"critical": 3.0, "warning": 2.0, "info": 1.0}
    score += severity_weight.get(signal_severity, 2.0)

    # Confidence contribution
    score += item.confidence * 2.0

    # Risk level as tiebreaker
    risk_bonus = {"high": 0.5, "medium": 0.3, "low": 0.0}
    score += risk_bonus.get(item.risk_level, 0.0)

    # Training cost: shorter training → higher priority
    if item.target in training_history:
        dur_min = training_history[item.target] / 60
        if dur_min < 10:
            score += 1.0
        elif dur_min < 30:
            score += 0.5

    return score


def _load_training_duration_history(workspace_root: str) -> Dict[str, float]:
    """Load average training duration per model from training_history.jsonl.

    Returns:
        {model_name: avg_duration_seconds}
    """
    history_path = os.path.join(workspace_root, "data", "training_history.jsonl")
    if not os.path.exists(history_path):
        return {}

    durations: Dict[str, List[float]] = {}
    try:
        with open(history_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    model = rec.get("model_name", "")
                    dur = rec.get("duration_seconds")
                    if model and dur is not None:
                        durations.setdefault(model, []).append(float(dur))
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        return {}

    return {model: sum(durs) / len(durs) for model, durs in durations.items() if durs}


def _infer_signal_severity(item: ActionItem) -> str:
    """Infer signal severity from ActionItem's source_signals."""
    signals = item.source_signals or []
    for sig in signals:
        sig_lower = sig.lower() if isinstance(sig, str) else ""
        if "severe" in sig_lower or "critical" in sig_lower:
            return "critical"
        if "warning" in sig_lower:
            return "warning"
    # Default based on risk_level
    if item.risk_level == "high":
        return "critical"
    return "warning"


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

class FeedbackLoop:
    """End-to-end Feedback Loop Orchestrator."""

    def __init__(self, workspace_root: str, mode: str = "report-only"):
        """
        Args:
            workspace_root: Production workspace root.
            mode: "report-only" | "execute" | "promote" | "auto-promote"
        """
        self.workspace_root = os.path.abspath(workspace_root)
        self.mode = mode
        self._run_label = ""

    def run(
        self,
        action_items_path: str,
        models: List[str] = None,
        skip_models: List[str] = None,
        max_duration_minutes: int = None,
        dry_run: bool = False,
        skip_retrain: bool = False,
        max_experiment_rounds: int = 3,
        resume: bool = False,
        run_label: str = "",
    ) -> FeedbackReport:
        """Execute the feedback loop.

        Args:
            action_items_path: Path to action_items_{date}.json
            models: Only process these models (override priority sort).
            skip_models: Exclude these models.
            max_duration_minutes: Time budget; excess items are deferred.
            dry_run: Adapter dry-run (preview only, no file writes).
            skip_retrain: Skip retraining (only apply config changes).
            max_experiment_rounds: Max retrain rounds per model in
                Playground (ExperimentAnalyzer decides retry vs give_up).
            resume: Skip params already tried in a previous experiment;
                let ExperimentAnalyzer suggest fresh params.

        Returns:
            FeedbackReport with results.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._run_label = run_label
        report = FeedbackReport(run_date=date_str, mode=self.mode)

        # playground-only shortcut: no ActionItems needed
        if self.mode == "playground-only":
            return self._run_playground_only(
                report, models, skip_models,
                max_experiment_rounds=max_experiment_rounds,
                skip_retrain=skip_retrain,
            )

        # 1. Load and filter ActionItems
        items = self._load_action_items(action_items_path)
        if not items:
            report.summary = "No in-scope ActionItems found."
            self._save_report(report)
            return report

        # 2. Filter by models / skip_models
        if models:
            items = [it for it in items if it.target in models]
        if skip_models:
            items = [it for it in items if it.target not in skip_models]

        if not items:
            report.summary = "All ActionItems filtered out by model selection."
            self._save_report(report)
            return report

        # 3. Priority sort + time budget
        training_history = _load_training_duration_history(self.workspace_root)
        scored = []
        for item in items:
            severity = _infer_signal_severity(item)
            priority = compute_priority(item, severity, training_history)
            scored.append((priority, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply time budget
        selected_items = []
        deferred_items = []
        if max_duration_minutes is not None:
            budget_seconds = max_duration_minutes * 60
            total_est = 0.0
            for priority, item in scored:
                est = training_history.get(item.target, 3600)  # default 1h
                if total_est + est <= budget_seconds:
                    selected_items.append(item)
                    total_est += est
                else:
                    deferred_items.append(item)
        else:
            selected_items = [item for _, item in scored]

        report.action_items_processed = len(selected_items)
        report.action_items_deferred = len(deferred_items)
        report.deferred_action_ids = [it.action_id for it in deferred_items]

        # ----- Mode dispatch -----

        if self.mode == "auto-promote":
            report.summary = "auto-promote mode is not yet implemented."
            self._save_report(report)
            return report

        if self.mode == "promote":
            return self._run_promote(action_items_path, selected_items, report)

        if self.mode == "report-only":
            return self._run_report_only(selected_items, report, training_history)

        if self.mode == "execute":
            return self._run_execute(
                selected_items, report, training_history,
                dry_run=dry_run, skip_retrain=skip_retrain,
                max_experiment_rounds=max_experiment_rounds,
                resume=resume,
            )

        report.summary = f"Unknown mode: {self.mode}"
        self._save_report(report)
        return report

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _run_report_only(
        self,
        items: List[ActionItem],
        report: FeedbackReport,
        training_history: Dict[str, float],
    ) -> FeedbackReport:
        """Generate preview report without executing anything."""
        adapter = TrainingAdapter(self.workspace_root)

        previews = []
        for item in items:
            if item.action_type == "adjust_hyperparam":
                preview = adapter.preview(item)
                severity = _infer_signal_severity(item)
                priority = compute_priority(item, severity, training_history)
                preview["priority_score"] = priority
                preview["action_id"] = item.action_id
                preview["confidence"] = item.confidence
                preview["risk_level"] = item.risk_level
                preview["reason"] = item.reason
                previews.append(preview)

        report.adapter_results = previews
        report.summary = (
            f"Report-only: {len(previews)} ActionItems previewed, "
            f"{report.action_items_deferred} deferred."
        )
        self._save_report(report)
        return report

    def _run_execute(
        self,
        items: List[ActionItem],
        report: FeedbackReport,
        training_history: Dict[str, float],
        dry_run: bool = False,
        skip_retrain: bool = False,
        max_experiment_rounds: int = 3,
        resume: bool = False,
    ) -> FeedbackReport:
        """Full execution with optional multi-round experiment loop.

        When resume=True, the initial ActionItem's adapter is skipped
        (its param was already applied in a previous run).  The experiment
        loop calls the ExperimentAnalyzer immediately to pick a fresh param.
        """
        pg_mgr = PlaygroundManager(self.workspace_root)
        playground_root = pg_mgr.create_or_sync()
        logger.info("Playground created at %s", playground_root)

        adapter = TrainingAdapter(playground_root)
        adapter_results = []
        validation_results = []

        for item in items:
            if item.action_type != "adjust_hyperparam":
                logger.warning(
                    "Skipping unsupported action_type '%s' for item %s",
                    item.action_type, item.action_id,
                )
                print(
                    f"   ⏭️  Skipping '{item.action_type}' for {item.target} "
                    f"(conf={item.confidence}) — only 'adjust_hyperparam' is "
                    f"auto-executable. Manual action may be needed."
                )
                continue

            missing = adapter.check_pretrain_deps(item)
            if missing:
                logger.warning(
                    "Missing pretrain deps for %s: %s (will use random init)",
                    item.target, missing,
                )

            if dry_run:
                preview = adapter.preview(item)
                adapter_results.append({
                    "action_id": item.action_id,
                    "dry_run": True,
                    **preview,
                })
                continue

            # Apply adapter — skip if resuming (param already applied)
            apply_first = not resume
            if resume:
                print(f"   ⏭️  Resume: skipping initial adapter for {item.target} "
                      f"— ExperimentAnalyzer will suggest fresh param")
                adapter_results.append({
                    "action_id": item.action_id,
                    "success": True,
                    "changes": [],
                    "skipped": True,
                    "reason": "resume — param already applied in previous run",
                })

            if apply_first:
                result = adapter.apply(item)
                adapter_results.append({
                    "action_id": result.action_id,
                    "success": result.success,
                    "changes": result.changes,
                    "error": result.error,
                })
                if not result.success:
                    logger.error("Adapter failed for %s: %s", item.action_id, result.error)
                    continue

            if not skip_retrain and max_experiment_rounds > 0:
                all_vrs = self._run_experiment_loop(
                    item=item,
                    playground_root=playground_root,
                    training_history=training_history,
                    adapter=adapter,
                    max_rounds=max_experiment_rounds,
                    skip_first_train=resume,
                    pg_mgr=pg_mgr,
                )
                for vr in all_vrs:
                    validation_results.append(asdict(vr))

        report.adapter_results = adapter_results
        report.validation_results = validation_results

        n_success = sum(1 for r in adapter_results if r.get("success"))
        n_passed = sum(1 for v in validation_results if v.get("passed"))
        report.summary = (
            f"Execute: {n_success}/{len(adapter_results)} adapters succeeded, "
            f"{n_passed}/{len(validation_results)} validations passed, "
            f"{report.action_items_deferred} deferred."
        )

        self._save_report(report)
        return report

    @staticmethod
    def _detect_overfitting(convergence: Optional[dict]) -> bool:
        """Check if training shows overfitting based on epoch ratio.

        A model is overfitting when best_epoch is in the first 15% of
        actual training epochs AND there were at least 5 actual epochs
        (to avoid false positives on very short runs).
        """
        if not convergence:
            return False
        best_ep = convergence.get("best_epoch")
        actual_ep = convergence.get("actual_epochs")
        if best_ep is None or actual_ep is None or actual_ep <= 5:
            return False
        return best_ep / actual_ep < 0.15

    def _run_experiment_loop(
        self,
        item: ActionItem,
        playground_root: str,
        training_history: Dict[str, float],
        adapter,
        max_rounds: int,
        skip_first_train: bool = False,
        llm=None,
        pg_mgr: Optional[PlaygroundManager] = None,
    ) -> List[ValidationResult]:
        """Multi-round experiment: retrain → validate → maybe retry.

        Independent experiment mode: before each round, the model's YAML is
        reset to production state and only the current round's single param
        change is applied.  This ensures clear causality per round.

        Results are persisted to ``experiment_history.jsonl`` for resume
        support.

        Returns all ValidationResults (one per training round), not just
        the best.  Callers can inspect per-round convergence.
        """
        # Load resume state if available
        exp_history = self._load_experiment_history(item.target)
        if exp_history and exp_history.get("status") == "in_progress":
            changes_tried = [
                {"param": r["param"], "from": r["from"], "to": r["to"],
                 "ic_result": r.get("playground_ic", 0)}
                for r in exp_history.get("rounds", [])
            ]
            print(f"   📂 Resumed experiment for {item.target}: "
                  f"{len(changes_tried)} prior rounds loaded")
        else:
            changes_tried = []
            exp_history = None

        all_results: List[ValidationResult] = []
        best_vr = None
        current_item = item
        baseline_ic = self._get_model_ic(self.workspace_root, item.target) or 0.0

        # Initialize experiment record for persistence
        experiment_id = (
            exp_history["experiment_id"] if exp_history
            else f"exp_{item.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        for round_idx in range(max_rounds):
            remaining = max_rounds - round_idx - 1

            if skip_first_train and round_idx == 0:
                # Resume mode: skip the first training round, go straight to
                # ExperimentAnalyzer to pick a fresh param
                print(f"\n   🔬 Experiment resume for {item.target} "
                      f"({remaining} rounds available) — "
                      f"calling ExperimentAnalyzer to suggest next param")
                vr = None  # No training yet; go to LLM analysis
            else:
                print(f"\n   🔬 Experiment round {round_idx + 1}/{max_rounds} "
                      f"for {item.target} ({remaining} remaining)")
                vr = self._retrain_and_validate(
                    current_item, playground_root, training_history,
                )
                if vr is None:
                    break

            # Track change history & improvement check (skip if resume first round)
            if vr is not None:
                if current_item.params:
                    for param, vals in current_item.params.items():
                        changes_tried.append({
                            "param": param,
                            "from": vals.get("from"),
                            "to": vals.get("to"),
                            "ic_result": vr.playground_ic,
                        })

                vr.round_idx = round_idx + 1
                vr.params_changed = dict(current_item.params) if current_item.params else None
                all_results.append(vr)
                # Per-round convergence summary
                c = vr.convergence or {}
                overfitting = self._detect_overfitting(c)
                print(f"   📊 Round {vr.round_idx} summary: "
                      f"IC={vr.playground_ic:.4f} (Δ{vr.ic_delta:+.4f}), "
                      f"best@{c.get('best_epoch','?')}/{c.get('actual_epochs','?')}, "
                      f"overfit={'YES' if overfitting else 'no'}")
                if best_vr is None or vr.playground_ic > best_vr.playground_ic:
                    best_vr = vr

                # Persist this round
                self._save_experiment_round(
                    experiment_id=experiment_id,
                    model=item.target,
                    baseline_ic=baseline_ic,
                    round_data={
                        "round": round_idx + 1,
                        "param": list(current_item.params.keys())[0] if current_item.params else None,
                        "from": list(current_item.params.values())[0].get("from") if current_item.params else None,
                        "to": list(current_item.params.values())[0].get("to") if current_item.params else None,
                        "playground_ic": vr.playground_ic,
                        "ic_delta": vr.ic_delta,
                        "convergence": vr.convergence,
                        "verdict": "overfitting" if overfitting else (
                            "improved" if vr.ic_improved else "degraded"
                        ),
                    },
                    status="in_progress",
                )

                min_abs_improvement = 0.002
                min_rel_improvement = 0.05
                meaningful = (
                    vr.ic_improved
                    and vr.ic_delta >= min_abs_improvement
                    and (vr.baseline_ic == 0 or vr.ic_delta / vr.baseline_ic >= min_rel_improvement)
                )
                if overfitting:
                    print(f"   ⚠️  Model is overfitting (best @ epoch {c.get('best_epoch')}, "
                          f"actual {c.get('actual_epochs')}) — IC improvement is not trustworthy")
                if meaningful and not overfitting:
                    print(f"   ✅ Experiment: IC improved ({vr.ic_delta:+.4f}) — stopping")
                    self._save_experiment_round(
                        experiment_id, item.target, baseline_ic, None,
                        status="completed",
                    )
                    break
                if remaining == 0:
                    print(f"   ⏹️  Experiment: max rounds exhausted "
                          f"(best IC: {best_vr.playground_ic:.4f})")
                    self._save_experiment_round(
                        experiment_id, item.target, baseline_ic, None,
                        status="exhausted",
                    )
                    break
                if overfitting:
                    print(f"   🔬 Overfitting detected — "
                          f"trying next param ({remaining} remaining)")
                elif vr.ic_improved:
                    print(f"   🔬 IC delta {vr.ic_delta:+.4f} is noise-level — "
                          f"trying next param ({remaining} remaining)")
                else:
                    print(f"   🔬 IC degraded ({vr.ic_delta:+.4f}) — "
                          f"trying next param ({remaining} remaining)")

            # --- Call ExperimentAnalyzer (LLM) ---
            if llm is None:
                try:
                    from quantpits.scripts.deep_analysis.llm_interface import (
                        LLMInterface,
                    )
                except ImportError:
                    print("   ⚠️  Cannot import LLMInterface — skipping")
                    break
                # Load API key from workspace config
                ws_cfg = {}
                cfg_p = os.path.join(self.workspace_root, "config", "llm_config.json")
                if os.path.exists(cfg_p):
                    try:
                        with open(cfg_p, "r") as f:
                            ws_cfg = json.load(f)
                    except Exception:
                        pass
                key_env = ws_cfg.get("api_key_env", "OPENAI_API_KEY")
                key = os.environ.get(key_env, "")
                llm = LLMInterface(api_key=key, base_url=ws_cfg.get("base_url"))
                if not llm.is_available():
                    print(f"   ⚠️  No API key (env: {key_env}) — skipping")
                    break

            # Load current params and available interventions
            current_params = llm._load_current_params(
                playground_root, [Signal(
                    signal_type="ic_decay", severity="warning",
                    scope="hyperparams", source_agent="Experiment",
                    target=item.target, context="experiment loop",
                )],
            )
            recent_history = llm._load_recent_action_history(
                self.workspace_root, limit=20,
            )
            interventions = llm._compute_available_interventions(
                [Signal(signal_type="ic_decay", severity="warning",
                        scope="hyperparams", source_agent="Experiment",
                        target=item.target, context="experiment loop")],
                recent_history, current_params,
            )
            hyperparam_bounds = llm._load_hyperparam_bounds(self.workspace_root)

            # For resume first round (no training yet), signal that this is
            # the INITIAL suggestion — IC values are placeholder, not a result
            is_first_round = (vr is None)
            b_ic = vr.baseline_ic if vr else baseline_ic
            p_ic = vr.playground_ic if vr else baseline_ic
            convergence = vr.convergence if vr else None

            analysis = llm.analyze_experiment_result(
                model_name=item.target,
                baseline_ic=b_ic,
                playground_ic=p_ic,
                changes_tried=changes_tried,
                convergence=convergence,
                current_params=current_params,
                available_interventions=interventions,
                hyperparam_bounds=hyperparam_bounds,
                max_rounds_remaining=remaining,
                workspace_root=self.workspace_root,
                is_first_round=is_first_round,
            )

            if analysis is None or analysis.get("decision") != "retry":
                self._save_experiment_round(
                    experiment_id, item.target, baseline_ic, None,
                    status="give_up",
                )
                break

            # Build a new ActionItem for the next round
            next_param = analysis.get("next_param")
            next_from = analysis.get("next_from")
            next_to = analysis.get("next_to")
            if not next_param or next_to is None:
                print("   ⚠️  ExperimentAnalyzer retry missing param/value — stopping")
                break

            # Create synthetic ActionItem for this round
            next_params = {next_param: {"from": next_from, "to": next_to}}
            current_item = ActionItem(
                action_type="adjust_hyperparam",
                scope="hyperparams",
                target=item.target,
                params=next_params,
                reason=f"[Experiment round {round_idx+2}] {analysis.get('rationale', '')}",
                source_signals=item.source_signals,
                expected_outcome=analysis.get("rationale", ""),
                confidence=0.4,  # Lower confidence for experimental retries
                risk_level="medium",
                action_id=f"{item.action_id}_round{round_idx+2}",
            )

            # Independent experiment mode: reset config to production state
            # before applying this round's single change
            if pg_mgr is not None:
                pg_mgr.sync_single_config(item.target)
                logger.debug(
                    "Reset %s config to production (independent experiment mode)",
                    item.target,
                )

            result = adapter.apply(current_item)
            if not result.success:
                print(f"   ⚠️  Adapter failed in round {round_idx+2}: {result.error}")
                break

        return all_results

    def _run_playground_only(
        self,
        report: FeedbackReport,
        models: Optional[List[str]],
        skip_models: Optional[List[str]],
        max_experiment_rounds: int = 3,
        skip_retrain: bool = False,
    ) -> FeedbackReport:
        """Lightweight mode: ExperimentAnalyzer drives param search directly.

        No ActionItems, no Triage, no Critic — just load model configs,
        let the ExperimentAnalyzer suggest params, train, validate, loop.
        """
        if not models:
            report.summary = "--models is required for playground-only mode"
            self._save_report(report)
            return report

        pg_mgr = PlaygroundManager(self.workspace_root)
        playground_root = pg_mgr.create_or_sync()
        logger.info("Playground ready at %s", playground_root)

        adapter = TrainingAdapter(playground_root)
        adapter_results = []
        validation_results = []

        # Load shared context (same for all models in this run)
        try:
            from quantpits.scripts.deep_analysis.llm_interface import LLMInterface
        except ImportError:
            report.summary = "Cannot import LLMInterface"
            self._save_report(report)
            return report

        # Load API key from workspace llm_config.json (same as Critic)
        ws_config = {}
        cfg_path = os.path.join(self.workspace_root, "config", "llm_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    ws_config = json.load(f)
            except Exception:
                pass
        api_key_env = ws_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        base_url = ws_config.get("base_url")

        llm = LLMInterface(api_key=api_key, base_url=base_url)
        if not llm.is_available():
            report.summary = f"No API key available (env: {api_key_env})"
            self._save_report(report)
            return report

        for model_name in models:
            if skip_models and model_name in skip_models:
                continue

            print(f"\n{'='*50}\n🧪 Playground Experiment: {model_name}\n{'='*50}")

            # Build a synthetic ActionItem for the experiment loop
            # (no pre-selected param — ExperimentAnalyzer picks the first one)
            dummy_item = ActionItem(
                action_type="adjust_hyperparam",
                scope="hyperparams",
                target=model_name,
                params={},  # Empty — ExperimentAnalyzer will fill in
                reason="[playground-only mode] ExperimentAnalyzer-driven search",
                source_signals=["ic_decay"],
                expected_outcome="",
                confidence=0.4,
                risk_level="medium",
                action_id=f"playground_{model_name}",
            )

            if not skip_retrain and max_experiment_rounds > 0:
                all_vrs = self._run_experiment_loop(
                    item=dummy_item,
                    playground_root=playground_root,
                    training_history={},
                    adapter=adapter,
                    max_rounds=max_experiment_rounds,
                    skip_first_train=False,
                    llm=llm,
                    pg_mgr=pg_mgr,
                )
                for vr in all_vrs:
                    validation_results.append(asdict(vr))

        report.adapter_results = adapter_results
        report.validation_results = validation_results
        n_passed = sum(1 for v in validation_results if v.get("passed"))
        report.summary = (
            f"Playground-only: {len(validation_results)} models tested, "
            f"{n_passed} validations passed."
        )
        self._save_report(report)
        return report

    def _run_promote(
        self,
        action_items_path: str,
        items: List[ActionItem],
        report: FeedbackReport,
    ) -> FeedbackReport:
        """Promote validated changes from Playground to Production."""
        pg_mgr = PlaygroundManager(self.workspace_root)
        playground_root = pg_mgr.get_playground_root()

        if not playground_root:
            report.summary = "No Playground found. Run --execute first."
            report.promote_result = {"success": False, "error": "No playground"}
            self._save_report(report)
            return report

        # Load the latest feedback report to get validation results
        latest_report = self._load_latest_feedback_report()
        validation_results = latest_report.get("validation_results", []) if latest_report else []

        # Only promote items that passed validation
        passed_ids = [
            vr.get("model") for vr in validation_results if vr.get("passed")
        ]
        promote_item_ids = [
            it.action_id for it in items
            if it.target in passed_ids or not validation_results
        ]

        if not promote_item_ids and validation_results:
            report.summary = "No ActionItems passed validation. Nothing to promote."
            self._save_report(report)
            return report

        promoter = ConfigPromoter(playground_root, self.workspace_root)
        result = promoter.promote(
            action_item_ids=promote_item_ids,
            validation_results=validation_results,
            reason="Promoted via feedback loop after validation",
        )

        report.promote_result = {
            "success": result.success,
            "promoted_files": result.promoted_files,
            "error": result.error,
        }
        report.summary = (
            f"Promote: {'success' if result.success else 'failed'}. "
            f"{len(result.promoted_files)} files promoted."
        )

        self._save_report(report)
        return report

    # ------------------------------------------------------------------
    # Retrain & validate
    # ------------------------------------------------------------------

    def _retrain_and_validate(
        self,
        item: ActionItem,
        playground_root: str,
        training_history: Dict[str, float],
    ) -> Optional[ValidationResult]:
        """Retrain a single model in the Playground and validate IC.

        This switches the workspace to playground via env.set_root_dir(),
        calls train_single_model(), then switches back.
        """
        try:
            from quantpits.utils import env

            # Get baseline IC from production (before switching)
            baseline_ic = self._get_model_ic(self.workspace_root, item.target)

            # Switch to playground
            original_root = env.ROOT_DIR
            env.set_root_dir(playground_root)

            try:
                # Lazy imports to avoid qlib dependency at module level
                from quantpits.utils.train_utils import (
                    load_model_registry,
                    train_single_model,
                    calculate_dates,
                )
                from quantpits.utils.operator_log import OperatorLog

                registry = load_model_registry()
                model_info = registry.get(item.target, {})
                yaml_file = model_info.get("yaml_file", "")
                if yaml_file:
                    yaml_file = os.path.join(playground_root, yaml_file)

                # Calculate dates
                env.init_qlib()
                params = calculate_dates()

                # Set up operator log
                with OperatorLog(
                    "feedback_loop_retrain",
                    args=[item.target],
                    log_file=os.path.join(playground_root, "data", "operator_log.jsonl"),
                ) as op_log:
                    op_log.set_source("llm_critic")
                    op_log.set_action_item_id(item.action_id)

                    result = train_single_model(
                        model_name=item.target,
                        yaml_file=yaml_file,
                        params=params,
                        experiment_name="feedback_loop_playground",
                    )

                    op_log.set_result({
                        "success": result.get("success", False),
                        "record_id": result.get("record_id"),
                    })

                if not result.get("success"):
                    logger.error(
                        "Training failed for %s: %s",
                        item.target, result.get("error"),
                    )
                    return ValidationResult(
                        model=item.target,
                        baseline_ic=baseline_ic or 0.0,
                        passed=False,
                    )

                # Get playground IC from train_single_model result directly
                # (model_performance_*.json is not generated in Playground)
                perf = result.get("performance", {}) or {}
                playground_ic = perf.get("IC_Mean")
                playground_icir = perf.get("ICIR")
                playground_excess = perf.get("Ann_Excess")
                convergence = perf.get("convergence", {})

            finally:
                # Always restore production workspace
                env.set_root_dir(original_root)

            # Validate
            if baseline_ic is None or playground_ic is None:
                return ValidationResult(
                    model=item.target,
                    baseline_ic=baseline_ic or 0.0,
                    playground_ic=playground_ic or 0.0,
                    playground_icir=playground_icir,
                    playground_excess=playground_excess,
                    convergence=convergence,
                    passed=True,  # Cannot compare; pass by default
                )

            ic_delta = playground_ic - baseline_ic
            # Tiered pass condition depending on baseline strength
            if baseline_ic > 0.01:
                # Normal baseline: IC should not drop more than 10%
                passed = playground_ic >= baseline_ic * 0.9
            elif baseline_ic > 0:
                # Weak baseline (IC < 0.01): any non-degradation passes
                passed = playground_ic >= baseline_ic
            else:
                # Negative or zero baseline: must become positive
                passed = playground_ic > 0

            return ValidationResult(
                model=item.target,
                baseline_ic=baseline_ic,
                playground_ic=playground_ic,
                ic_delta=ic_delta,
                ic_improved=ic_delta > 0,
                passed=passed,
                playground_icir=playground_icir,
                playground_excess=playground_excess,
                convergence=convergence,
            )

        except Exception as e:
            logger.exception("Retrain/validate failed for %s", item.target)
            return ValidationResult(model=item.target, passed=False)

    def _get_model_ic(self, workspace_root: str, model_name: str) -> Optional[float]:
        """Extract the latest IC for a model from model_performance files."""
        import glob

        perf_dir = os.path.join(workspace_root, "output")
        pattern = os.path.join(perf_dir, "model_performance_*.json")
        files = sorted(glob.glob(pattern))

        if not files:
            return None

        # Read the latest performance file
        try:
            with open(files[-1], "r") as f:
                data = json.load(f)

            # Per-model structure: {"model_name": {"IC_Mean": ..., ...}, ...}
            model_data = data.get(model_name, {})
            if isinstance(model_data, dict):
                ic = model_data.get("IC_Mean") or model_data.get("ic")
                if ic is not None:
                    return float(ic)

            # Fallback: direct IC field or nested "all" structure
            ic = data.get("ic")
            if ic is not None:
                return float(ic)

            all_data = data.get("all", {})
            ic = all_data.get("IC_Mean") or all_data.get("ic")
            if ic is not None:
                return float(ic)
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Experiment history persistence (B2)
    # ------------------------------------------------------------------

    def _experiment_history_path(self) -> str:
        """Path to the experiment history JSONL file."""
        return os.path.join(self.workspace_root, "data", "experiment_history.jsonl")

    def _load_experiment_history(self, model_name: str) -> Optional[dict]:
        """Load the most recent in-progress experiment for a model.

        Returns:
            The experiment dict if found, else None.
        """
        path = self._experiment_history_path()
        if not os.path.exists(path):
            return None

        latest = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("model") == model_name:
                            latest = entry
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return None

        # Only return if the experiment is still in-progress
        if latest and latest.get("status") == "in_progress":
            return latest
        return None

    def _save_experiment_round(
        self,
        experiment_id: str,
        model: str,
        baseline_ic: float,
        round_data: Optional[dict],
        status: str = "in_progress",
    ):
        """Append or update an experiment record in experiment_history.jsonl.

        Each experiment_id gets one line.  On subsequent rounds, the entire
        line is replaced (read-all → filter → rewrite) to keep the file
        compact and resumable.
        """
        path = self._experiment_history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Load all existing records
        records: List[dict] = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass

        # Find or create the record for this experiment
        target_rec = None
        for rec in records:
            if rec.get("experiment_id") == experiment_id:
                target_rec = rec
                break

        if target_rec is None:
            target_rec = {
                "experiment_id": experiment_id,
                "model": model,
                "started_at": datetime.now().isoformat(),
                "baseline_ic": baseline_ic,
                "rounds": [],
                "status": status,
            }
            records.append(target_rec)

        # Append round data if provided
        if round_data is not None:
            target_rec["rounds"].append(round_data)

        # Update status and timestamp
        target_rec["status"] = status
        target_rec["updated_at"] = datetime.now().isoformat()

        # Compute best round
        best_ic = baseline_ic
        best_round = None
        for r in target_rec.get("rounds", []):
            ic = r.get("playground_ic", 0)
            if ic > best_ic:
                best_ic = ic
                best_round = r.get("round")
        target_rec["best_ic"] = best_ic
        target_rec["best_round"] = best_round

        # Rewrite the file (atomic via tmp)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning("Failed to save experiment history: %s", e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Report I/O
    # ------------------------------------------------------------------

    def _save_report(self, report: FeedbackReport):
        """Save FeedbackReport to output/deep_analysis/."""
        out_dir = os.path.join(self.workspace_root, "output", "deep_analysis")
        os.makedirs(out_dir, exist_ok=True)
        _label_suffix = f"_{self._run_label}" if self._run_label else ""
        path = os.path.join(out_dir, f"feedback_report_{report.run_date}{_label_suffix}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Feedback report saved to %s", path)

    def _load_latest_feedback_report(self) -> Optional[dict]:
        """Load the most recent feedback_report JSON."""
        import glob

        out_dir = os.path.join(self.workspace_root, "output", "deep_analysis")
        pattern = os.path.join(out_dir, "feedback_report_*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        try:
            with open(files[-1], "r") as f:
                return json.load(f)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # ActionItem loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_action_items(path: str) -> List[ActionItem]:
        """Load ActionItems from JSON, filter to in_scope only."""
        if not os.path.exists(path):
            logger.warning("ActionItems file not found: %s", path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load ActionItems: %s", e)
            return []

        items = []
        for d in data:
            item = ActionItem.from_dict(d)
            if item.scope_status == "in_scope":
                items.append(item)

        logger.info("Loaded %d in-scope ActionItems from %s", len(items), path)
        return items
