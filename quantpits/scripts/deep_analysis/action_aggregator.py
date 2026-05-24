"""
Action Aggregator for the layered LLM analysis pipeline.

Collects ActionItems from all upstream LLM outputs (Per-Model, Per-Combo,
Execution/Risk), deduplicates, filters by active scope, and pre-detects
conflicts for the Synthesizer to resolve.
"""

import json
import logging
import os
from typing import List, Dict, Optional

from .action_items import ActionItem

logger = logging.getLogger(__name__)


class ActionAggregator:
    """
    Rule-based aggregation layer between upstream LLM calls and the Synthesizer.

    Responsibilities:
    1. Collect ActionItems from Per-Model, Per-Combo, and Exec/Risk outputs
    2. Deduplicate (same target + action_type + param keys → keep highest confidence)
    3. Tag each item with its source (for traceability)
    4. Filter by active_scopes
    5. Pre-detect conflicts for the Synthesizer to resolve
    """

    def __init__(self, workspace_root: str):
        self._workspace_root = workspace_root
        self._active_scopes = self._load_active_scopes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        model_diagnoses: Dict[str, dict],
        combo_diagnoses: Dict[str, dict],
        execution_risk_output: Optional[dict],
    ) -> dict:
        """
        Aggregate ActionItems from all upstream sources.

        Returns a dict with:
        - raw_items: list of all ActionItem dicts (before dedup)
        - deduped_items: list after deduplication
        - in_scope_count / out_of_scope_count
        - conflicts: list of pre-detected conflicts for Synthesizer
        """
        raw_items: List[dict] = []

        # Collect from Per-Model diagnoses
        for model_name, diag in model_diagnoses.items():
            for item in diag.get("action_items", []):
                item["_source"] = f"Per-Model({model_name})"
                item["_source_diagnosis"] = diag.get("diagnosis", "?")
                if "target" not in item:
                    item["target"] = model_name
                raw_items.append(item)

        # Collect from Per-Combo diagnoses
        for combo_name, diag in combo_diagnoses.items():
            for item in diag.get("action_items", []):
                item["_source"] = f"Per-Combo({combo_name})"
                item["_source_diagnosis"] = diag.get("diagnosis", "?")
                if "target" not in item:
                    item["target"] = combo_name
                raw_items.append(item)

        # Collect from Execution/Risk output
        if execution_risk_output:
            for item in execution_risk_output.get("action_items", []):
                item["_source"] = "Execution/Risk"
                raw_items.append(item)

        # Deduplicate
        deduped = self._deduplicate(raw_items)

        # Scope filter
        in_scope = [i for i in deduped if i.get("scope", "") in self._active_scopes]
        out_of_scope = [i for i in deduped if i.get("scope", "") not in self._active_scopes]

        # Pre-detect conflicts
        conflicts = self._detect_conflicts(model_diagnoses, combo_diagnoses, deduped)

        return {
            "raw_count": len(raw_items),
            "deduped_count": len(deduped),
            "in_scope_count": len(in_scope),
            "out_of_scope_count": len(out_of_scope),
            "deduped_items": deduped,
            "conflicts": conflicts,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_active_scopes(self) -> List[str]:
        path = os.path.join(self._workspace_root, "config", "feedback_scope.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("active_scopes", [])
        except Exception:
            return []

    @staticmethod
    def _deduplicate(items: List[dict]) -> List[dict]:
        """
        Deduplicate ActionItems by (target, action_type, param_keys).
        When duplicates exist, keep the one with highest confidence.
        """
        groups: Dict[str, List[dict]] = {}
        for item in items:
            target = item.get("target", "")
            action_type = item.get("action_type", "")
            params = item.get("params") or {}
            param_keys = tuple(sorted(params.keys()))
            key = f"{target}|{action_type}|{param_keys}"
            groups.setdefault(key, []).append(item)

        result = []
        for key, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Keep highest confidence; if tie, keep first source
                best = max(group, key=lambda x: (x.get("confidence", 0), len(x.get("_source", ""))))
                if len(group) > 1:
                    sources = [i.get("_source", "?") for i in group]
                    best["_dedup_note"] = f"Merged from {len(group)} sources: {sources}"
                result.append(best)

        return result

    def _detect_conflicts(
        self,
        model_diagnoses: Dict[str, dict],
        combo_diagnoses: Dict[str, dict],
        deduped_items: List[dict],
    ) -> List[dict]:
        """
        Detect conflicts between upstream LLM outputs.

        Primary conflict pattern: Per-Model says disable model X,
        but Per-Combo says keep model X in the combo.
        """
        conflicts = []

        # Conflict 1: disable vs keep
        disable_targets = set()
        for model_name, diag in model_diagnoses.items():
            if diag.get("diagnosis") in ("should_disable",):
                disable_targets.add(model_name)
            for item in diag.get("action_items", []):
                if item.get("action_type") == "disable_model":
                    disable_targets.add(item.get("target", model_name))

        for combo_name, diag in combo_diagnoses.items():
            for member, assessment in diag.get("member_assessments", {}).items():
                if member in disable_targets and assessment.get("keep", True):
                    conflicts.append({
                        "type": "disable_vs_keep",
                        "target": member,
                        "combo": combo_name,
                        "per_model_says": "disable",
                        "per_combo_says": f"keep (role: {assessment.get('role', '?')})",
                        "loo_delta": assessment.get("loo_delta"),
                    })

        # Conflict 2: contradictory param suggestions from different sources
        param_suggestions: Dict[str, Dict[str, list]] = {}
        for item in deduped_items:
            target = item.get("target", "")
            for param, values in (item.get("params") or {}).items():
                if isinstance(values, dict) and "to" in values:
                    key = f"{target}|{param}"
                    param_suggestions.setdefault(key, {}).setdefault(
                        str(values["to"]), []
                    ).append(item.get("_source", "?"))

        for key, to_values in param_suggestions.items():
            if len(to_values) > 1:
                target, param = key.split("|", 1)
                conflicts.append({
                    "type": "contradictory_param",
                    "target": target,
                    "param": param,
                    "suggestions": {
                        val: sources for val, sources in to_values.items()
                    },
                })

        return conflicts
