"""
Training Adapter for RLFF Phase 4.

Translates ``adjust_hyperparam`` ActionItems into concrete modifications of
``workflow_config_*.yaml`` files.  Uses ``ruamel.yaml`` to preserve YAML
comments and formatting.

Safety mechanisms:
- "from" value verification: refuses to apply if the current value doesn't
  match the ActionItem's ``params[key]["from"]`` expectation.
- Backup: original file backed up before modification.
- Bounds double-check: re-validates against ``hyperparam_bounds.json``.
"""

import json
import logging
import os
import re
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

# ruamel.yaml instance configured for round-trip (preserves comments/formatting)
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False


@register_adapter("adjust_hyperparam")
class TrainingAdapter(BaseAdapter):
    """Modify workflow_config YAML files for ``adjust_hyperparam`` ActionItems."""

    adapter_type = "training"

    def __init__(self, workspace_root: str):
        super().__init__(workspace_root)
        self._registry = None
        self._bounds = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, item: ActionItem) -> AdapterResult:
        """Apply a single ActionItem's hyperparam changes to the YAML file.

        Returns:
            AdapterResult with success/failure and change details.
        """
        try:
            # 1. Resolve target model → YAML file
            yaml_path = self._resolve_yaml_path(item.target)
            if yaml_path is None:
                return AdapterResult(
                    success=False,
                    action_id=item.action_id,
                    adapter_type=self.adapter_type,
                    error=f"Cannot resolve YAML file for model '{item.target}'",
                )

            # 2. Load YAML (round-trip)
            with open(yaml_path, "r", encoding="utf-8") as f:
                doc = _yaml.load(f)

            kwargs = self._get_model_kwargs(doc)
            if kwargs is None:
                return AdapterResult(
                    success=False,
                    action_id=item.action_id,
                    adapter_type=self.adapter_type,
                    error=f"Cannot locate task.model.kwargs in {yaml_path}",
                )

            # 3. Validate & apply each param change
            changes = []
            for param_name, change_spec in item.params.items():
                if not isinstance(change_spec, dict):
                    continue

                expected_from = change_spec.get("from")
                new_val = change_spec.get("to")

                # "from" value verification
                current_val = kwargs.get(param_name)
                if expected_from is not None and not self._values_match(current_val, expected_from):
                    return AdapterResult(
                        success=False,
                        action_id=item.action_id,
                        adapter_type=self.adapter_type,
                        error=(
                            f"Param '{param_name}' current value {current_val!r} "
                            f"does not match expected 'from' value {expected_from!r} "
                            f"in {yaml_path}. "
                            f"File may have been manually modified."
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
                    "file": yaml_path,
                })

            # 4. Backup original file
            self._backup(yaml_path)

            # 5. Write changes
            for change in changes:
                kwargs[change["param"]] = change["new"]

            with open(yaml_path, "w", encoding="utf-8") as f:
                _yaml.dump(doc, f)

            logger.info(
                "TrainingAdapter applied %d changes to %s for action %s",
                len(changes), yaml_path, item.action_id,
            )

            return AdapterResult(
                success=True,
                action_id=item.action_id,
                adapter_type=self.adapter_type,
                modified_files=[yaml_path],
                changes=changes,
            )

        except Exception as e:
            logger.exception("TrainingAdapter.apply failed for action %s", item.action_id)
            return AdapterResult(
                success=False,
                action_id=item.action_id,
                adapter_type=self.adapter_type,
                error=str(e),
            )

    def preview(self, item: ActionItem) -> dict:
        """Dry-run: return what would change without writing files.

        Returns:
            dict with keys: target, yaml_file, planned_changes, errors
        """
        result = {
            "target": item.target,
            "yaml_file": None,
            "planned_changes": [],
            "errors": [],
        }

        yaml_path = self._resolve_yaml_path(item.target)
        if yaml_path is None:
            result["errors"].append(f"Cannot resolve YAML file for model '{item.target}'")
            return result
        result["yaml_file"] = yaml_path

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                doc = _yaml.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to parse YAML: {e}")
            return result

        kwargs = self._get_model_kwargs(doc)
        if kwargs is None:
            result["errors"].append(f"Cannot locate task.model.kwargs in {yaml_path}")
            return result

        for param_name, change_spec in item.params.items():
            if not isinstance(change_spec, dict):
                continue

            expected_from = change_spec.get("from")
            new_val = change_spec.get("to")
            current_val = kwargs.get(param_name)

            entry = {
                "param": param_name,
                "current": current_val,
                "expected_from": expected_from,
                "to": new_val,
                "from_match": self._values_match(current_val, expected_from) if expected_from is not None else True,
            }

            bounds_err = self._check_bounds(param_name, new_val)
            if bounds_err:
                entry["bounds_error"] = bounds_err

            result["planned_changes"].append(entry)

        return result

    def check_pretrain_deps(self, item: ActionItem) -> List[str]:
        """Check pretrained model dependencies for the target model.

        Returns:
            List of missing pretrain_source names (empty if all OK).
        """
        registry = self._load_registry()
        model_info = registry.get(item.target, {})
        pretrain_source = model_info.get("pretrain_source")

        if not pretrain_source:
            return []

        # Check if pretrained file exists in playground
        pretrained_dir = os.path.join(self.workspace_root, "data", "pretrained")
        latest_path = os.path.join(pretrained_dir, f"{pretrain_source}_latest.pkl")
        if os.path.exists(latest_path):
            return []

        return [pretrain_source]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_yaml_path(self, model_name: str) -> Optional[str]:
        """Look up the workflow YAML file for a model via model_registry.yaml."""
        registry = self._load_registry()
        model_info = registry.get(model_name, {})
        yaml_rel = model_info.get("yaml_file")

        if yaml_rel:
            yaml_path = os.path.join(self.workspace_root, yaml_rel)
            if os.path.exists(yaml_path):
                return yaml_path

        # Fallback: try conventional naming
        fallback = os.path.join(
            self.workspace_root, "config", f"workflow_config_{model_name}.yaml"
        )
        if os.path.exists(fallback):
            return fallback

        return None

    def _load_registry(self) -> dict:
        """Load model_registry.yaml (cached)."""
        if self._registry is None:
            registry_file = os.path.join(self.workspace_root, "config", "model_registry.yaml")
            if os.path.exists(registry_file):
                import yaml
                with open(registry_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                self._registry = data.get("models", {})
            else:
                self._registry = {}
        return self._registry

    def _load_bounds(self) -> dict:
        """Load hyperparam_bounds.json (cached)."""
        if self._bounds is None:
            bounds_file = os.path.join(self.workspace_root, "config", "hyperparam_bounds.json")
            if os.path.exists(bounds_file):
                with open(bounds_file, "r") as f:
                    data = json.load(f)
                self._bounds = data.get("bounds", {})
            else:
                self._bounds = {}
        return self._bounds

    @staticmethod
    def _get_model_kwargs(doc) -> Optional[dict]:
        """Extract task.model.kwargs from a parsed YAML document."""
        try:
            return doc["task"]["model"]["kwargs"]
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _values_match(actual, expected) -> bool:
        """Compare values tolerantly (int/float coercion)."""
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False
        # Numeric comparison with tolerance
        try:
            return abs(float(actual) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return str(actual) == str(expected)

    def _check_bounds(self, param_name: str, new_val) -> Optional[str]:
        """Check a new value against hyperparam_bounds.json.

        Returns:
            Error string if out of bounds, None if OK.
        """
        bounds = self._load_bounds()
        if param_name not in bounds:
            return None  # Unknown param — allow by default

        b = bounds[param_name]
        b_min = b.get("min")
        b_max = b.get("max")

        if new_val is not None:
            if b_min is not None and new_val < b_min:
                return f"Parameter '{param_name}' value {new_val} below minimum {b_min}"
            if b_max is not None and new_val > b_max:
                return f"Parameter '{param_name}' value {new_val} above maximum {b_max}"

        return None

    def _backup(self, yaml_path: str):
        """Create a timestamped backup of the YAML file before modification."""
        backup_dir = os.path.join(self.workspace_root, "config", "_backup")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.basename(yaml_path)
        backup_path = os.path.join(backup_dir, f"{basename}.{timestamp}")
        shutil.copy2(yaml_path, backup_path)
        logger.debug("Backed up %s → %s", yaml_path, backup_path)
