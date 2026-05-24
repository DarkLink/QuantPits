"""
Model Selection Adapter for RLFF Phase 4.

Translates ``disable_model`` / ``enable_model`` ActionItems into modifications
of ``model_registry.yaml``'s ``enabled`` field.

Safety mechanisms:
- LOO delta pre-check: refuses to disable a model with positive LOO delta
  (the model is a valuable diversifier).
- Combo linkage check: warns if a disabled model is in an active combo and
  suggests generating a ``replace_member`` ActionItem.
- Backup: original model_registry.yaml backed up to config/_backup/ before
  modification.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from ruamel.yaml import YAML

from quantpits.scripts.deep_analysis.action_items import ActionItem
from quantpits.scripts.deep_analysis.adapters import register_adapter
from quantpits.scripts.deep_analysis.adapters.base_adapter import (
    AdapterResult,
    BaseAdapter,
)

logger = logging.getLogger(__name__)

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False


@register_adapter("disable_model")
@register_adapter("enable_model")
class ModelSelectionAdapter(BaseAdapter):
    """Enable/disable models in model_registry.yaml."""

    adapter_type = "model_selection"

    def __init__(self, workspace_root: str):
        super().__init__(workspace_root)
        self._registry_path = os.path.join(
            workspace_root, "config", "model_registry.yaml"
        )
        self._ensemble_path = os.path.join(
            workspace_root, "config", "ensemble_config.json"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, item: ActionItem) -> AdapterResult:
        """Enable or disable a model in model_registry.yaml.

        Returns:
            AdapterResult with success/failure and change details.
        """
        try:
            # 1. Load registry
            if not os.path.exists(self._registry_path):
                return AdapterResult(
                    success=False,
                    action_id=item.action_id,
                    adapter_type=self.adapter_type,
                    error=f"model_registry.yaml not found at {self._registry_path}",
                )

            with open(self._registry_path, "r", encoding="utf-8") as f:
                doc = _yaml.load(f)

            models = doc.get("models", {})
            if item.target not in models:
                return AdapterResult(
                    success=False,
                    action_id=item.action_id,
                    adapter_type=self.adapter_type,
                    error=f"Model '{item.target}' not found in model_registry.yaml",
                )

            model_entry = models[item.target]
            current_enabled = model_entry.get("enabled", True)
            new_enabled = item.action_type == "enable_model"

            # No-op check
            if current_enabled == new_enabled:
                state = "enabled" if new_enabled else "disabled"
                return AdapterResult(
                    success=True,
                    action_id=item.action_id,
                    adapter_type=self.adapter_type,
                    error=f"Model '{item.target}' is already {state} — no change needed",
                )

            # 2. Pre-check for disable: LOO delta
            if item.action_type == "disable_model":
                loo_err = self._check_loo_delta(item)
                if loo_err:
                    return AdapterResult(
                        success=False,
                        action_id=item.action_id,
                        adapter_type=self.adapter_type,
                        error=loo_err,
                    )

            # 3. Combo linkage check for disable
            warnings = []
            if item.action_type == "disable_model":
                warnings = self._check_combo_membership(item.target)

            # 4. Backup
            self._backup()

            # 5. Apply change
            model_entry["enabled"] = new_enabled

            with open(self._registry_path, "w", encoding="utf-8") as f:
                _yaml.dump(doc, f)

            logger.info(
                "ModelSelectionAdapter: %s %s → enabled=%s",
                item.action_type, item.target, new_enabled,
            )

            return AdapterResult(
                success=True,
                action_id=item.action_id,
                adapter_type=self.adapter_type,
                modified_files=[self._registry_path],
                changes=[{
                    "param": "enabled",
                    "old": current_enabled,
                    "new": new_enabled,
                    "file": self._registry_path,
                }],
                error="; ".join(warnings) if warnings else "",
            )

        except Exception as e:
            logger.exception(
                "ModelSelectionAdapter.apply failed for action %s", item.action_id
            )
            return AdapterResult(
                success=False,
                action_id=item.action_id,
                adapter_type=self.adapter_type,
                error=str(e),
            )

    def preview(self, item: ActionItem) -> dict:
        """Dry-run: return what would change without writing files.

        Returns:
            dict with keys: target, registry_file, planned_change, errors, warnings
        """
        result = {
            "target": item.target,
            "registry_file": self._registry_path,
            "planned_change": None,
            "errors": [],
            "warnings": [],
        }

        if not os.path.exists(self._registry_path):
            result["errors"].append(f"model_registry.yaml not found at {self._registry_path}")
            return result

        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                doc = _yaml.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to parse model_registry.yaml: {e}")
            return result

        models = doc.get("models", {})
        if item.target not in models:
            result["errors"].append(f"Model '{item.target}' not found in model_registry.yaml")
            return result

        model_entry = models[item.target]
        current_enabled = model_entry.get("enabled", True)
        new_enabled = item.action_type == "enable_model"

        if current_enabled == new_enabled:
            result["errors"].append(
                f"Model '{item.target}' is already {'enabled' if new_enabled else 'disabled'}"
            )
            return result

        result["planned_change"] = {
            "param": "enabled",
            "current": current_enabled,
            "new": new_enabled,
        }

        # LOO delta check for disable
        if item.action_type == "disable_model":
            loo_err = self._check_loo_delta(item)
            if loo_err:
                result["errors"].append(loo_err)

        # Combo linkage check
        if item.action_type == "disable_model":
            result["warnings"] = self._check_combo_membership(item.target)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_loo_delta(self, item: ActionItem) -> Optional[str]:
        """Check LOO delta before disabling a model.

        Returns an error string if the model should NOT be disabled,
        None if it's safe to proceed.
        """
        loo_delta = item.params.get("loo_delta", 0) if item.params else 0

        if loo_delta is not None and loo_delta > 0:
            return (
                f"LOO delta for '{item.target}' is positive ({loo_delta:+.4f}). "
                f"This model is a valuable diversifier — refusing to disable. "
                f"Marked as 'diversifier retained'."
            )

        return None

    def _check_combo_membership(self, model_name: str) -> List[str]:
        """Check if a model is a member of any active combo.

        Returns a list of warning strings (empty if OK).
        """
        warnings = []
        if not os.path.exists(self._ensemble_path):
            return warnings

        try:
            with open(self._ensemble_path, "r") as f:
                ensemble_config = json.load(f)
        except Exception:
            return warnings

        combos = ensemble_config.get("combos", {})
        for combo_name, combo_def in combos.items():
            members = combo_def.get("models", [])
            # Combo members use "{model}@static" naming convention
            for member in members:
                member_base = member.rsplit("@", 1)[0] if "@" in member else member
                if member_base == model_name:
                    is_default = combo_def.get("default", False)
                    tag = " (default)" if is_default else ""
                    warnings.append(
                        f"Model '{model_name}' is in combo '{combo_name}'{tag}. "
                        f"Consider generating a replace_member ActionItem."
                    )

        return warnings

    def _backup(self):
        """Create a timestamped backup of model_registry.yaml before modification."""
        backup_dir = os.path.join(self.workspace_root, "config", "_backup")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.basename(self._registry_path)
        backup_path = os.path.join(backup_dir, f"{basename}.{timestamp}")
        shutil.copy2(self._registry_path, backup_path)
        logger.debug("Backed up %s → %s", self._registry_path, backup_path)
