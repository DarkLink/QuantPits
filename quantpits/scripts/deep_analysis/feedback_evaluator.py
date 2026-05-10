"""
Feedback Evaluator for the layered LLM analysis pipeline.

Provides closed-loop evaluation of previous LLM recommendations against
actual model performance changes, using existing dated files only (no git).

Two components:
1. HistoryReader — reconstructs what happened between two analysis dates
2. FeedbackEvaluator — judges previous ActionItem quality, generates self_corrections
"""

import json
import logging
import os
import re
import glob
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class FeedbackSnapshot:
    """What happened between the last analysis and now."""
    period_start: str = ""   # YYYY-MM-DD of previous analysis
    period_end: str = ""     # YYYY-MM-DD of current analysis
    operator_actions: List[dict] = field(default_factory=list)
    model_list_changes: dict = field(default_factory=dict)   # added, removed, enabled, disabled
    hyperparam_changes: List[dict] = field(default_factory=list)
    combo_changes: List[dict] = field(default_factory=list)
    performance_deltas: Dict[str, dict] = field(default_factory=dict)  # model_name -> {ic_delta, icir_delta, ...}

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# History Reader
# ------------------------------------------------------------------


class HistoryReader:
    """
    Reconstructs the timeline of user actions and model changes between
    two analysis dates using dated files in the workspace.
    """

    def __init__(self, workspace_root: str):
        self._workspace_root = workspace_root

    def read(
        self,
        current_date: str,
        previous_date: Optional[str] = None,
    ) -> FeedbackSnapshot:
        """
        Build a FeedbackSnapshot covering the period from previous_date to current_date.

        If previous_date is None, auto-discovers the most recent analysis date
        before current_date.
        """
        if previous_date is None:
            previous_date = self._find_last_analysis_date(current_date)

        snapshot = FeedbackSnapshot(
            period_start=previous_date or "unknown",
            period_end=current_date,
        )

        if not previous_date:
            return snapshot

        snapshot.operator_actions = self._read_operator_actions(previous_date, current_date)
        snapshot.model_list_changes = self._detect_model_list_changes(previous_date, current_date)
        snapshot.hyperparam_changes = self._detect_hyperparam_changes(previous_date, current_date)
        snapshot.performance_deltas = self._compute_performance_deltas(previous_date, current_date)
        snapshot.combo_changes = self._detect_combo_changes(previous_date, current_date)

        return snapshot

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_last_analysis_date(self, current_date: str) -> Optional[str]:
        """Find the most recent action_items_{date}.json before current_date."""
        output_dir = os.path.join(self._workspace_root, "output", "deep_analysis")
        if not os.path.isdir(output_dir):
            return None

        pattern = re.compile(r"action_items_(\d{4}-\d{2}-\d{2})\.json$")
        dates = []
        for fname in os.listdir(output_dir):
            m = pattern.search(fname)
            if m:
                d = m.group(1)
                if d < current_date:
                    dates.append(d)

        return max(dates) if dates else None

    def _find_closest_snapshot(
        self, target_date: str, glob_pattern: str
    ) -> Optional[str]:
        """Find the file matching glob_pattern with date closest to target_date."""
        files = sorted(glob.glob(os.path.join(self._workspace_root, glob_pattern)))
        pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
        best_path = None
        best_diff = None
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")

        for path in files:
            m = pattern.search(os.path.basename(path))
            if not m:
                continue
            file_dt = datetime.strptime(m.group(1), "%Y-%m-%d")
            diff = abs((target_dt - file_dt).days)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_path = path

        return best_path

    # ------------------------------------------------------------------
    # Operator log
    # ------------------------------------------------------------------

    def _read_operator_actions(
        self, start_date: str, end_date: str
    ) -> List[dict]:
        """Read operator_log.jsonl and filter human actions between dates."""
        log_path = os.path.join(self._workspace_root, "data", "operator_log.jsonl")
        if not os.path.exists(log_path):
            return []

        actions = []
        try:
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts = entry.get("timestamp_start", "")
                    if not ts:
                        continue

                    date_str = ts[:10]  # YYYY-MM-DD
                    if start_date < date_str <= end_date:
                        if entry.get("source") == "human":
                            actions.append({
                                "date": date_str,
                                "script": entry.get("script", ""),
                                "args": entry.get("args", []),
                                "duration_s": entry.get("duration_seconds"),
                                "log_id": entry.get("log_id", ""),
                            })
        except Exception as e:
            logger.warning("Failed to read operator_log: %s", e)

        return sorted(actions, key=lambda a: a["date"])

    # ------------------------------------------------------------------
    # Model list changes
    # ------------------------------------------------------------------

    def _detect_model_list_changes(
        self, prev_date: str, curr_date: str
    ) -> dict:
        """Compare model_performance snapshots to detect added/removed models."""
        prev_path = self._find_closest_snapshot(
            prev_date, "output/model_performance_*.json"
        )
        curr_path = self._find_closest_snapshot(
            curr_date, "output/model_performance_*.json"
        )

        if not prev_path or not curr_path:
            return {}

        try:
            with open(prev_path) as f:
                prev_data = json.load(f)
            with open(curr_path) as f:
                curr_data = json.load(f)
        except Exception:
            return {}

        prev_models = set(prev_data.keys())
        curr_models = set(curr_data.keys())

        return {
            "added": sorted(curr_models - prev_models),
            "removed": sorted(prev_models - curr_models),
            "prev_count": len(prev_models),
            "curr_count": len(curr_models),
        }

    # ------------------------------------------------------------------
    # Hyperparam changes
    # ------------------------------------------------------------------

    def _detect_hyperparam_changes(
        self, prev_date: str, curr_date: str
    ) -> List[dict]:
        """
        Detect hyperparameter changes by comparing the LAST training_history
        record for each model before prev_date vs before curr_date.
        """
        history_path = os.path.join(
            self._workspace_root, "data", "training_history.jsonl"
        )
        if not os.path.exists(history_path):
            return []

        # Build {model_name: [records sorted by trained_at]}
        records: Dict[str, list] = {}
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
                    records.setdefault(model, []).append(rec)
        except Exception as e:
            logger.warning("Failed to read training_history: %s", e)
            return []

        # Sort each model's records by trained_at
        for model in records:
            records[model].sort(key=lambda r: r.get("trained_at", ""))

        # For each model, find the latest record before prev_date and before curr_date
        changes = []
        param_keys_of_interest = {
            "n_epochs", "early_stop", "lr", "learning_rate", "batch_size",
            "dropout", "hidden_size", "num_layers", "iterations", "depth",
            "l2_leaf_reg", "num_leaves", "n_estimators",
        }

        for model, recs in records.items():
            prev_rec = None
            curr_rec = None
            for rec in recs:
                trained_at = rec.get("trained_at", "")
                if not trained_at:
                    continue
                date_str = trained_at[:10]
                if date_str <= prev_date:
                    prev_rec = rec
                if date_str <= curr_date:
                    curr_rec = rec

            if prev_rec and curr_rec and prev_rec != curr_rec:
                for key in param_keys_of_interest:
                    prev_val = prev_rec.get(key)
                    curr_val = curr_rec.get(key)
                    if prev_val is not None and curr_val is not None and prev_val != curr_val:
                        changes.append({
                            "model": model,
                            "param": key,
                            "from": prev_val,
                            "to": curr_val,
                            "prev_trained_at": prev_rec.get("trained_at"),
                            "curr_trained_at": curr_rec.get("trained_at"),
                        })

        return changes

    # ------------------------------------------------------------------
    # Performance deltas
    # ------------------------------------------------------------------

    def _compute_performance_deltas(
        self, prev_date: str, curr_date: str
    ) -> Dict[str, dict]:
        """Compare IC/ICIR changes between two model_performance snapshots."""
        prev_path = self._find_closest_snapshot(
            prev_date, "output/model_performance_*.json"
        )
        curr_path = self._find_closest_snapshot(
            curr_date, "output/model_performance_*.json"
        )

        if not prev_path or not curr_path:
            return {}

        try:
            with open(prev_path) as f:
                prev_data = json.load(f)
            with open(curr_path) as f:
                curr_data = json.load(f)
        except Exception:
            return {}

        deltas = {}
        for model in set(prev_data.keys()) & set(curr_data.keys()):
            prev_metrics = prev_data[model]
            curr_metrics = curr_data[model]
            if not isinstance(prev_metrics, dict) or not isinstance(curr_metrics, dict):
                continue

            prev_ic = prev_metrics.get("IC_Mean")
            curr_ic = curr_metrics.get("IC_Mean")
            prev_icir = prev_metrics.get("ICIR")
            curr_icir = curr_metrics.get("ICIR")

            if prev_ic is not None and curr_ic is not None:
                deltas[model] = {
                    "ic_delta": round(curr_ic - prev_ic, 6),
                    "icir_delta": round(curr_icir - prev_icir, 6) if prev_icir is not None and curr_icir is not None else None,
                    "prev_ic": round(prev_ic, 6),
                    "curr_ic": round(curr_ic, 6),
                    "prev_icir": round(prev_icir, 6) if prev_icir is not None else None,
                    "curr_icir": round(curr_icir, 6) if curr_icir is not None else None,
                }

        return deltas

    # ------------------------------------------------------------------
    # Combo changes
    # ------------------------------------------------------------------

    def _detect_combo_changes(
        self, prev_date: str, curr_date: str
    ) -> List[dict]:
        """Detect combo configuration changes between two dates."""
        # For now, check if combo_comparison files exist for both dates
        prev_csv = self._find_closest_snapshot(
            prev_date, "output/combo_comparison_*.csv"
        )
        curr_csv = self._find_closest_snapshot(
            curr_date, "output/combo_comparison_*.csv"
        )

        changes = []
        if prev_csv and curr_csv and prev_csv != curr_csv:
            changes.append({
                "type": "combo_comparison_updated",
                "prev_file": os.path.basename(prev_csv),
                "curr_file": os.path.basename(curr_csv),
            })

        return changes


# ------------------------------------------------------------------
# Feedback Evaluator
# ------------------------------------------------------------------


class FeedbackEvaluator:
    """
    Judges the quality of previous LLM recommendations by comparing them
    against actual outcomes (model performance changes, operator actions).

    Outputs a FeedbackEval that the Synthesizer uses for self-correction.
    """

    def __init__(self, workspace_root: str):
        self._workspace_root = workspace_root

    def evaluate(
        self,
        snapshot: FeedbackSnapshot,
        current_date: str,
    ) -> dict:
        """
        Evaluate previous ActionItems against observed outcomes.

        Returns a FeedbackEval dict with:
        - per_item_evaluations: quality label for each previous ActionItem
        - quality_summary: aggregate stats
        - self_corrections: rules for the Synthesizer
        """
        if not snapshot.period_start or snapshot.period_start == "unknown":
            return {
                "evaluated": False,
                "reason": "No previous analysis found — first run or missing history.",
                "per_item_evaluations": [],
                "quality_summary": {},
                "self_corrections": [],
            }

        # Load previous ActionItems
        prev_items = self._load_action_items(snapshot.period_start)
        if not prev_items:
            return {
                "evaluated": False,
                "reason": f"No action_items found for {snapshot.period_start}.",
                "per_item_evaluations": [],
                "quality_summary": {},
                "self_corrections": [],
            }

        # Evaluate each item
        evaluations = []
        for item in prev_items:
            ev = self._evaluate_item(item, snapshot)
            evaluations.append(ev)

        # Summarize
        quality_summary = self._summarize(evaluations)

        # Generate self-corrections
        self_corrections = self._generate_self_corrections(evaluations, quality_summary)

        return {
            "evaluated": True,
            "period_start": snapshot.period_start,
            "period_end": snapshot.period_end,
            "per_item_evaluations": evaluations,
            "quality_summary": quality_summary,
            "self_corrections": self_corrections,
        }

    # ------------------------------------------------------------------
    # Item evaluation
    # ------------------------------------------------------------------

    def _evaluate_item(self, item: dict, snapshot: FeedbackSnapshot) -> dict:
        """
        Classify a single previous ActionItem's quality.

        Returns: {
            action_id, action_type, target, params, reason (original),
            quality: correct_effective | correct_ignored | incorrect | pending_verification,
            evidence: {...}
        }
        """
        action_type = item.get("action_type", "")
        target = item.get("target", "")
        params = item.get("params", {})

        ev = {
            "action_id": item.get("action_id", ""),
            "action_type": action_type,
            "target": target,
            "params": params,
            "original_reason": item.get("reason", "")[:200],
            "quality": "pending_verification",
            "evidence": {},
        }

        # 1. Was this item executed by the user?
        executed = self._was_executed(item, snapshot)
        ev["evidence"]["executed"] = executed

        # 2. Did the target model's performance change?
        perf_delta = snapshot.performance_deltas.get(target, {})
        ic_delta = perf_delta.get("ic_delta")
        ev["evidence"]["ic_delta"] = ic_delta
        ev["evidence"]["icir_delta"] = perf_delta.get("icir_delta")

        # 3. Were the suggested hyperparams actually changed?
        params_changed = self._were_params_changed(target, params, snapshot)
        ev["evidence"]["params_changed"] = params_changed

        # 4. Classify quality
        if action_type == "adjust_hyperparam":
            ev["quality"] = self._classify_param_adjustment(
                executed, params_changed, ic_delta
            )
        elif action_type == "disable_model":
            ev["quality"] = self._classify_disable(executed, ic_delta, target, snapshot)
        elif action_type == "enable_model":
            ev["quality"] = self._classify_enable(executed, target, snapshot)
        else:
            ev["quality"] = "pending_verification"

        return ev

    @staticmethod
    def _was_executed(item: dict, snapshot: FeedbackSnapshot) -> bool:
        """Check if the ActionItem was executed by looking at operator log."""
        target = item.get("target", "")
        action_type = item.get("action_type", "")

        for action in snapshot.operator_actions:
            args = action.get("args", [])
            args_str = " ".join(args) if args else ""

            if action_type == "adjust_hyperparam":
                script = action.get("script", "")
                if script in ("static_train",) and target in args_str:
                    return True
            elif action_type in ("disable_model", "enable_model"):
                script = action.get("script", "")
                if target in args_str:
                    return True
            elif action_type == "trigger_search":
                script = action.get("script", "")
                if "ensemble" in script.lower() or "search" in script.lower():
                    return True

        return False

    @staticmethod
    def _were_params_changed(
        target: str, params: dict, snapshot: FeedbackSnapshot
    ) -> bool:
        """Check if the suggested params were actually changed."""
        for change in snapshot.hyperparam_changes:
            if change["model"] != target:
                continue
            for param_name, param_val in params.items():
                if isinstance(param_val, dict) and "to" in param_val:
                    if change["param"] == param_name:
                        return True
        return False

    @staticmethod
    def _classify_param_adjustment(
        executed: bool, params_changed: bool, ic_delta: Optional[float]
    ) -> str:
        if executed and ic_delta is not None and ic_delta > 0.001:
            return "correct_effective"
        elif executed and ic_delta is not None and ic_delta < -0.001:
            return "incorrect"
        elif params_changed and ic_delta is not None and abs(ic_delta) < 0.001:
            return "pending_verification"
        elif not executed:
            return "correct_ignored"
        return "pending_verification"

    def _classify_disable(
        self, executed: bool, ic_delta: Optional[float],
        target: str, snapshot: FeedbackSnapshot,
    ) -> str:
        """A disable suggestion is incorrect if the model had positive LOO delta."""
        if not executed:
            return "correct_ignored"
        # If the model was added back to combos or IC improved without it,
        # it's hard to judge without LOO context. Default to pending.
        if ic_delta is not None and ic_delta < -0.01:
            return "correct_effective"  # model got worse → disable was right
        return "pending_verification"

    @staticmethod
    def _classify_enable(
        executed: bool, target: str, snapshot: FeedbackSnapshot,
    ) -> str:
        if not executed:
            return "correct_ignored"
        # Check if the model appears in the new model list
        added = snapshot.model_list_changes.get("added", [])
        if target in added:
            return "correct_effective"
        return "pending_verification"

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize(evaluations: List[dict]) -> dict:
        total = len(evaluations)
        if total == 0:
            return {"total": 0}

        counts = {}
        for ev in evaluations:
            q = ev.get("quality", "pending_verification")
            counts[q] = counts.get(q, 0) + 1

        return {
            "total": total,
            "correct_effective": counts.get("correct_effective", 0),
            "correct_ignored": counts.get("correct_ignored", 0),
            "incorrect": counts.get("incorrect", 0),
            "pending_verification": counts.get("pending_verification", 0),
            "accuracy": (
                (counts.get("correct_effective", 0) + counts.get("correct_ignored", 0))
                / total
                if total > 0
                else 0
            ),
        }

    # ------------------------------------------------------------------
    # Self-corrections
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_self_corrections(
        evaluations: List[dict], quality_summary: dict
    ) -> List[dict]:
        """
        Generate self_corrections rules from detected error patterns.

        Each rule has: pattern (what went wrong), rule (how to avoid it),
        severity (how often it occurs).
        """
        rules = []

        # Pattern 1: incorrect param adjustments
        incorrect_params = [
            ev for ev in evaluations
            if ev.get("quality") == "incorrect"
            and ev.get("action_type") == "adjust_hyperparam"
        ]
        if incorrect_params:
            rules.append({
                "pattern": "Param adjustments that degraded IC",
                "rule": (
                    "When suggesting param changes for models with ≤ 1 training "
                    "record, set confidence ≤ 0.5. Single-run IC is unreliable."
                ),
                "affected_models": [ev["target"] for ev in incorrect_params],
                "severity": len(incorrect_params),
            })

        # Pattern 2: disable suggestions for combo members
        for ev in evaluations:
            if ev.get("quality") == "incorrect" and ev.get("action_type") == "disable_model":
                rules.append({
                    "pattern": "Disable suggestion for model with combo value",
                    "rule": (
                        "Do NOT suggest disabling a model based solely on low "
                        "single-model IC. Verify LOO delta first — a model with "
                        "IC ≈ 0 but positive LOO delta is a valuable diversifier."
                    ),
                    "affected_models": [ev["target"]],
                    "severity": 1,
                })

        # Pattern 3: batch same-param adjustments
        same_param_items = {}
        for ev in evaluations:
            if ev.get("action_type") != "adjust_hyperparam":
                continue
            for param_name in ev.get("params", {}).keys():
                same_param_items.setdefault(param_name, []).append(ev["target"])

        for param, models in same_param_items.items():
            if len(models) >= 3:
                rules.append({
                    "pattern": f"Batch {param} adjustment on {len(models)} models",
                    "rule": (
                        f"Do NOT apply the same {param} change to ≥ 3 models. "
                        f"Pick the 2 worst-affected models as experiments."
                    ),
                    "affected_models": models,
                    "severity": len(models),
                })

        return rules

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_action_items(self, date_str: str) -> List[dict]:
        """Load action_items_{date}.json from the deep_analysis output directory."""
        filename = f"action_items_{date_str}.json"
        path = os.path.join(
            self._workspace_root, "output", "deep_analysis", filename
        )
        if not os.path.exists(path):
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning("Failed to load action_items for %s: %s", date_str, e)
            return []


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------


def run_feedback_loop(
    workspace_root: str,
    current_date: Optional[str] = None,
) -> dict:
    """
    Run the full feedback loop: HistoryReader → FeedbackEvaluator.

    Returns the FeedbackEval dict ready for Synthesizer injection.
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    reader = HistoryReader(workspace_root)
    snapshot = reader.read(current_date)

    evaluator = FeedbackEvaluator(workspace_root)
    return evaluator.evaluate(snapshot, current_date)
