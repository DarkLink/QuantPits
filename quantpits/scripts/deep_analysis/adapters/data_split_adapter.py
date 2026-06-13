"""
Data Split Adapter — modifies ``model_config.json`` for ``adjust_training_window``
ActionItems.

Unlike TrainingAdapter (which edits YAML), this adapter modifies the single
JSON file that controls all models' data split configuration.

Safety mechanisms:
- "from" value verification: refuses to apply if current value doesn't match
- Backup: model_config.json backed up to config/_backup/ before modification
- Bounds double-check: re-validates against training_window_bounds.json
- Atomic write: writes to temp file, then os.replace() for crash safety
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Optional

from quantpits.scripts.deep_analysis.action_items import ActionItem
from quantpits.scripts.deep_analysis.adapters import register_adapter
from quantpits.scripts.deep_analysis.adapters.base_adapter import (
    AdapterResult,
    BaseAdapter,
)

logger = logging.getLogger(__name__)


@register_adapter("adjust_training_window")
class DataSplitAdapter(BaseAdapter):
    """Modify model_config.json for ``adjust_training_window`` ActionItems."""

    adapter_type = "data_split"

    # Fields in model_config.json that this adapter is allowed to modify
    ALLOWED_FIELDS = {
        "train_set_windows",
        "valid_set_window",
        "test_set_window",
        "data_slice_mode",
    }

    def __init__(self, workspace_root: str):
        super().__init__(workspace_root)
        self._config_path = os.path.join(
            workspace_root, "config", "model_config.json"
        )
        self._bounds_path = os.path.join(
            workspace_root, "config", "training_window_bounds.json"
        )
        self._bounds: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, item: ActionItem) -> AdapterResult:
        """Apply training window changes to model_config.json."""
        try:
            if not os.path.exists(self._config_path):
                return AdapterResult(
                    success=False,
                    action_id=item.action_id,
                    adapter_type=self.adapter_type,
                    error=f"model_config.json not found at {self._config_path}",
                )

            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            changes = []
            for param_name, change_spec in item.params.items():
                if not isinstance(change_spec, dict):
                    continue

                if param_name not in self.ALLOWED_FIELDS:
                    return AdapterResult(
                        success=False,
                        action_id=item.action_id,
                        adapter_type=self.adapter_type,
                        error=(
                            f"Field '{param_name}' is not an allowed "
                            f"model_config.json field. Allowed: "
                            f"{self.ALLOWED_FIELDS}"
                        ),
                    )

                expected_from = change_spec.get("from")
                new_val = change_spec.get("to")
                current_val = config.get(param_name)

                # Verify "from" value matches current config
                if expected_from is not None and not self._values_match(
                    current_val, expected_from
                ):
                    return AdapterResult(
                        success=False,
                        action_id=item.action_id,
                        adapter_type=self.adapter_type,
                        error=(
                            f"Field '{param_name}' current value "
                            f"{current_val!r} does not match expected "
                            f"'from' {expected_from!r}. Config may have "
                            f"been manually modified."
                        ),
                    )

                # Bounds double-check
                bounds_err = self._check_bounds(param_name, new_val)
                if bounds_err:
                    return AdapterResult(
                        success=False,
                        action_id=item.action_id,
                        adapter_type=self.adapter_type,
                        error=bounds_err,
                    )

                changes.append({
                    "param": param_name,
                    "old": current_val,
                    "new": new_val,
                    "file": self._config_path,
                })

            if not changes:
                return AdapterResult(
                    success=False,
                    action_id=item.action_id,
                    adapter_type=self.adapter_type,
                    error="No valid parameter changes to apply.",
                )

            # Backup original
            self._backup()

            # Apply changes
            for change in changes:
                config[change["param"]] = change["new"]

            # Atomic write
            tmp_path = self._config_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, self._config_path)

            logger.info(
                "DataSplitAdapter applied %d changes to %s for action %s",
                len(changes), self._config_path, item.action_id,
            )

            return AdapterResult(
                success=True,
                action_id=item.action_id,
                adapter_type=self.adapter_type,
                modified_files=[self._config_path],
                changes=changes,
            )

        except Exception as e:
            logger.exception(
                "DataSplitAdapter.apply failed for action %s", item.action_id
            )
            tmp_path = self._config_path + ".tmp"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return AdapterResult(
                success=False,
                action_id=item.action_id,
                adapter_type=self.adapter_type,
                error=str(e),
            )

    def preview(self, item: ActionItem) -> dict:
        """Dry-run: return what would change without writing files."""
        result = {
            "target": item.target,
            "config_file": self._config_path,
            "planned_changes": [],
            "errors": [],
        }

        if not os.path.exists(self._config_path):
            result["errors"].append(
                f"model_config.json not found at {self._config_path}"
            )
            return result

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to parse model_config.json: {e}")
            return result

        for param_name, change_spec in item.params.items():
            if not isinstance(change_spec, dict):
                continue

            expected_from = change_spec.get("from")
            new_val = change_spec.get("to")
            current_val = config.get(param_name)

            entry = {
                "param": param_name,
                "current": current_val,
                "expected_from": expected_from,
                "to": new_val,
                "from_match": (
                    self._values_match(current_val, expected_from)
                    if expected_from is not None
                    else True
                ),
            }

            bounds_err = self._check_bounds(param_name, new_val)
            if bounds_err:
                entry["bounds_error"] = bounds_err

            if param_name not in self.ALLOWED_FIELDS:
                entry["disallowed_field"] = True

            result["planned_changes"].append(entry)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_bounds(self) -> dict:
        if self._bounds is None:
            if os.path.exists(self._bounds_path):
                try:
                    with open(self._bounds_path, "r") as f:
                        data = json.load(f)
                    self._bounds = data.get("bounds", {})
                except Exception:
                    self._bounds = {}
            else:
                self._bounds = {}
        return self._bounds

    def _check_bounds(self, param_name: str, new_val) -> Optional[str]:
        """Check a new value against training_window_bounds.json."""
        bounds = self._load_bounds()
        if param_name not in bounds:
            return None  # Unknown param — allow by default

        b = bounds[param_name]
        b_min = b.get("min")
        b_max = b.get("max")

        if new_val is not None:
            if (
                b_min is not None
                and isinstance(new_val, (int, float))
                and new_val < b_min
            ):
                return (
                    f"Parameter '{param_name}' value {new_val} "
                    f"below minimum {b_min}"
                )
            if (
                b_max is not None
                and isinstance(new_val, (int, float))
                and new_val > b_max
            ):
                return (
                    f"Parameter '{param_name}' value {new_val} "
                    f"above maximum {b_max}"
                )

        allowed = b.get("allowed_values")
        if allowed and new_val not in allowed:
            return (
                f"Parameter '{param_name}' value '{new_val}' "
                f"not in allowed {allowed}"
            )

        return None

    @staticmethod
    def _values_match(actual, expected) -> bool:
        """Compare values tolerantly (numeric or string)."""
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False
        try:
            return abs(float(actual) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return str(actual) == str(expected)

    def _backup(self):
        """Create a timestamped backup of model_config.json."""
        backup_dir = os.path.join(self.workspace_root, "config", "_backup")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.basename(self._config_path)
        backup_path = os.path.join(backup_dir, f"{basename}.{timestamp}")
        shutil.copy2(self._config_path, backup_path)
        logger.debug("Backed up %s -> %s", self._config_path, backup_path)
