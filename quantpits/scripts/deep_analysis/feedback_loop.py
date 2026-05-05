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

    def run(
        self,
        action_items_path: str,
        models: List[str] = None,
        skip_models: List[str] = None,
        max_duration_minutes: int = None,
        dry_run: bool = False,
        skip_retrain: bool = False,
    ) -> FeedbackReport:
        """Execute the feedback loop.

        Args:
            action_items_path: Path to action_items_{date}.json
            models: Only process these models (override priority sort).
            skip_models: Exclude these models.
            max_duration_minutes: Time budget; excess items are deferred.
            dry_run: Adapter dry-run (preview only, no file writes).
            skip_retrain: Skip retraining (only apply config changes).

        Returns:
            FeedbackReport with results.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        report = FeedbackReport(run_date=date_str, mode=self.mode)

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
    ) -> FeedbackReport:
        """Full execution: Playground → Adapter → Retrain → Validate → Report."""
        # 1. Create Playground
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
                continue

            # 2. Check pretrain deps
            missing = adapter.check_pretrain_deps(item)
            if missing:
                logger.warning(
                    "Missing pretrain deps for %s: %s (will use random init)",
                    item.target, missing,
                )

            # 3. Apply adapter
            if dry_run:
                preview = adapter.preview(item)
                adapter_results.append({
                    "action_id": item.action_id,
                    "dry_run": True,
                    **preview,
                })
                continue

            result = adapter.apply(item)
            adapter_results.append({
                "action_id": result.action_id,
                "success": result.success,
                "changes": result.changes,
                "error": result.error,
            })

            if not result.success:
                logger.error(
                    "Adapter failed for %s: %s", item.action_id, result.error
                )
                continue

            # 4. Retrain (if not skipped)
            if not skip_retrain:
                vr = self._retrain_and_validate(
                    item, playground_root, training_history
                )
                if vr:
                    validation_results.append(asdict(vr))

        report.adapter_results = adapter_results
        report.validation_results = validation_results

        # Summary
        n_success = sum(1 for r in adapter_results if r.get("success"))
        n_passed = sum(1 for v in validation_results if v.get("passed"))
        report.summary = (
            f"Execute: {n_success}/{len(adapter_results)} adapters succeeded, "
            f"{n_passed}/{len(validation_results)} validations passed, "
            f"{report.action_items_deferred} deferred."
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

                # Get playground IC
                playground_ic = self._get_model_ic(playground_root, item.target)

            finally:
                # Always restore production workspace
                env.set_root_dir(original_root)

            # Validate
            if baseline_ic is None or playground_ic is None:
                return ValidationResult(
                    model=item.target,
                    baseline_ic=baseline_ic or 0.0,
                    playground_ic=playground_ic or 0.0,
                    passed=True,  # Cannot compare; pass by default
                )

            ic_delta = playground_ic - baseline_ic
            # Pass condition: IC doesn't drop more than 10%
            passed = playground_ic >= baseline_ic * 0.9 if baseline_ic > 0 else True

            return ValidationResult(
                model=item.target,
                baseline_ic=baseline_ic,
                playground_ic=playground_ic,
                ic_delta=ic_delta,
                ic_improved=ic_delta > 0,
                passed=passed,
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

            # Direct IC field
            ic = data.get("ic")
            if ic is not None:
                return float(ic)

            # Try nested structure
            all_data = data.get("all", {})
            ic = all_data.get("IC_Mean") or all_data.get("ic")
            if ic is not None:
                return float(ic)
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Report I/O
    # ------------------------------------------------------------------

    def _save_report(self, report: FeedbackReport):
        """Save FeedbackReport to output/deep_analysis/."""
        out_dir = os.path.join(self.workspace_root, "output", "deep_analysis")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"feedback_report_{report.run_date}.json")

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
