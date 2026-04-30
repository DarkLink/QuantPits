"""
ActionItem data model and Validator for the MAS Deep Analysis System (Phase 3).

Defines the ActionItem structure produced by the LLM Critic and the
ActionItemValidator that enforces feedback_scope and hyperparam_bounds
constraints.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    """A concrete action recommendation produced by the LLM Critic."""

    action_id: str = ""                        # UUID, auto-generated
    action_type: str = ""                      # "adjust_hyperparam" | "disable_model" | "trigger_search" | ...
    scope: str = ""                            # "hyperparams" | "model_selection" | "combo_search" | "strategy_params"
    target: str = ""                           # Target model/combo name
    params: Dict[str, Any] = field(default_factory=dict)   # e.g. {"n_epochs": {"from": 100, "to": 150}}
    reason: str = ""                           # LLM rationale
    source_signals: List[str] = field(default_factory=list)  # Signal types that triggered this
    expected_outcome: str = ""                 # Expected effect description
    confidence: float = 0.0                    # LLM self-assessed confidence [0, 1]
    risk_level: str = "medium"                 # "low" | "medium" | "high"

    # Validation layer fields
    scope_status: str = "pending"              # "in_scope" | "out_of_scope" | "rejected"
    rejected_reason: str = ""
    validated_at: str = ""

    # Phase 4 forward-compatibility
    execution_context: Dict[str, Any] = field(default_factory=lambda: {
        "target_env": "playground",
        "requires_retrain": True,
        "requires_backtest": True,
        "estimated_duration_minutes": 60,
        "dependencies": [],
    })

    def __post_init__(self):
        if not self.action_id:
            self.action_id = str(uuid.uuid4())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ActionItem":
        """Create an ActionItem from a dict (e.g. parsed from LLM JSON output)."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        item = cls(**filtered)
        # Ensure action_id is always set
        if not item.action_id:
            item.action_id = str(uuid.uuid4())
        return item


class ActionItemValidator:
    """
    Validate ActionItems against feedback_scope and hyperparam_bounds.

    Validation rules:
    1. scope not in active_scopes → scope_status = "out_of_scope"
    2. adjust_hyperparam with value outside bounds → scope_status = "rejected"
    3. adjust_hyperparam with change exceeding max_change_pct → scope_status = "rejected"
    4. Unknown hyperparam names → pass with warning
    5. All checks passed → scope_status = "in_scope"
    """

    def __init__(
        self,
        feedback_scope_path: str,
        hyperparam_bounds_path: str,
        workspace_root: str,
    ):
        self.workspace_root = workspace_root
        self._active_scopes = self._load_active_scopes(feedback_scope_path)
        self._bounds = self._load_bounds(hyperparam_bounds_path)

    @staticmethod
    def _load_active_scopes(path: str) -> List[str]:
        """Load active_scopes from feedback_scope.json."""
        if not os.path.exists(path):
            logger.warning("feedback_scope.json not found at %s — no scopes active", path)
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("active_scopes", [])
        except Exception as e:
            logger.warning("Failed to load feedback_scope.json: %s", e)
            return []

    @staticmethod
    def _load_bounds(path: str) -> Dict[str, dict]:
        """Load hyperparam bounds configuration."""
        if not os.path.exists(path):
            logger.warning("hyperparam_bounds.json not found at %s — no bounds enforced", path)
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("bounds", {})
        except Exception as e:
            logger.warning("Failed to load hyperparam_bounds.json: %s", e)
            return {}

    def validate(self, items: List[ActionItem]) -> List[ActionItem]:
        """
        Validate each ActionItem and set scope_status + rejected_reason.

        Items are modified in-place and also returned for convenience.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for item in items:
            item.validated_at = now

            # Step 1: Scope check
            if item.scope not in self._active_scopes:
                item.scope_status = "out_of_scope"
                item.rejected_reason = (
                    f"Scope '{item.scope}' not in active_scopes "
                    f"{self._active_scopes}"
                )
                continue

            # Step 2: Hyperparam bounds check (only for adjust_hyperparam)
            if item.action_type == "adjust_hyperparam":
                rejected = self._check_hyperparam_bounds(item)
                if rejected:
                    item.scope_status = "rejected"
                    item.rejected_reason = rejected
                    continue

            # All checks passed
            item.scope_status = "in_scope"

        return items

    def _check_hyperparam_bounds(self, item: ActionItem) -> Optional[str]:
        """
        Check hyperparam params against bounds.

        Returns rejection reason string if failed, None if passed.
        """
        for param_name, change_spec in item.params.items():
            if not isinstance(change_spec, dict):
                continue

            new_val = change_spec.get("to")
            old_val = change_spec.get("from")

            if param_name not in self._bounds:
                logger.warning(
                    "Hyperparam '%s' not in bounds config — allowing by default",
                    param_name,
                )
                continue

            bounds = self._bounds[param_name]
            b_min = bounds.get("min")
            b_max = bounds.get("max")
            max_pct = bounds.get("max_change_pct")

            # Value range check
            if new_val is not None:
                if b_min is not None and new_val < b_min:
                    return (
                        f"Parameter '{param_name}' value {new_val} "
                        f"below minimum {b_min}"
                    )
                if b_max is not None and new_val > b_max:
                    return (
                        f"Parameter '{param_name}' value {new_val} "
                        f"above maximum {b_max}"
                    )

            # Change magnitude check
            if (
                max_pct is not None
                and old_val is not None
                and new_val is not None
                and old_val != 0
            ):
                change_pct = abs(new_val - old_val) / abs(old_val) * 100
                if change_pct > max_pct:
                    return (
                        f"Parameter '{param_name}' change "
                        f"{old_val} → {new_val} ({change_pct:.1f}%) "
                        f"exceeds max_change_pct ({max_pct}%)"
                    )

        return None


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------

def persist_action_items(
    items: List[ActionItem],
    workspace_root: str,
    run_date: Optional[str] = None,
) -> str:
    """
    Save validated ActionItems to workspace.

    Writes:
    - output/deep_analysis/action_items_{date}.json  (full snapshot)
    - data/action_item_history.jsonl                  (append audit trail)

    Returns the path to the snapshot JSON.
    """
    date_str = run_date or datetime.now().strftime("%Y-%m-%d")

    # --- Snapshot ---
    out_dir = os.path.join(workspace_root, "output", "deep_analysis")
    os.makedirs(out_dir, exist_ok=True)

    snapshot_path = os.path.join(out_dir, f"action_items_{date_str}.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(
            [item.to_dict() for item in items],
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("ActionItems snapshot saved to %s", snapshot_path)

    # --- Audit trail ---
    data_dir = os.path.join(workspace_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    history_path = os.path.join(data_dir, "action_item_history.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        for item in items:
            record = item.to_dict()
            record["_run_date"] = date_str
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("ActionItems appended to %s", history_path)

    return snapshot_path
