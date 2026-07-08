"""
Training Mode Context for the MAS Deep Analysis System (Phase 2).

Discovers and models training modes and rolling states across:
- latest_train_records.json (experiment name mappings, model-to-mode runs)
- config/rolling_config.yaml (rolling scheduler config)
- data/rolling_state.json (slide rolling progress)
- data/rolling_state_cpcv.json (CPCV rolling progress)
- config/model_config.json (purged_cv parameters)
"""

import os
import json
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> list:
    """Read a JSONL file, returning a list of parsed dicts.

    Silently handles missing files, empty lines, and malformed JSON.
    """
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (FileNotFoundError, PermissionError):
        pass
    return entries


class TrainingModeContext:
    """Provides structured training configurations and model mode mappings."""

    def __init__(
        self,
        workspace_root: str,
        anchor_date: Optional[str] = None,
        all_model_keys: Optional[Dict[str, str]] = None,
        models_by_name: Optional[Dict[str, Dict[str, str]]] = None,
        available_modes: Optional[List[str]] = None,
        rolling_config: Optional[dict] = None,
        rolling_states: Optional[Dict[str, dict]] = None,
        cpcv_config: Optional[dict] = None,
        last_prediction: Optional[Dict[str, dict]] = None,
        last_train: Optional[Dict[str, dict]] = None,
        is_predict_only_cycle: bool = False,
    ):
        self.workspace_root = workspace_root
        self.anchor_date = anchor_date
        self.all_model_keys = all_model_keys or {}
        self.models_by_name = models_by_name or {}
        self.available_modes = available_modes or []
        self.rolling_config = rolling_config or {}
        self.rolling_states = rolling_states or {}
        self.cpcv_config = cpcv_config or {}
        self.last_prediction = last_prediction or {}
        self.last_train = last_train or {}
        self.is_predict_only_cycle = is_predict_only_cycle

    def get_cross_mode_models(self) -> List[str]:
        """Return model names that have runs registered in multiple training modes."""
        cross_models = []
        for name, modes_dict in self.models_by_name.items():
            if len(modes_dict) > 1:
                cross_models.append(name)
        return sorted(cross_models)

    def get_models_with_mode(self, mode: str) -> List[str]:
        """Return all model names trained under a specific mode (e.g. static, cpcv, rolling)."""
        models = []
        for name, modes_dict in self.models_by_name.items():
            if mode in modes_dict:
                models.append(name)
        return sorted(models)

    def get_rolling_gap_days(self, mode: str) -> Optional[int]:
        """Calculate gap in days between the last rolling completed date and the current anchor_date."""
        state = self.rolling_states.get(mode)
        if not state or not state.get("anchor_date") or not self.anchor_date:
            return None
        try:
            r_date = datetime.strptime(state["anchor_date"], "%Y-%m-%d")
            c_date = datetime.strptime(self.anchor_date, "%Y-%m-%d")
            return (c_date - r_date).days
        except Exception as e:
            logger.warning(f"Failed to calculate rolling gap for '{mode}': {e}")
            return None

    def resolve_model_key(self, record_id: str) -> Optional[str]:
        """Resolve the full model key (e.g. 'lstm_Alpha158@static') from a record ID."""
        for key, rid in self.all_model_keys.items():
            if rid == record_id:
                return key
        return None

    def get_last_operation(self, model_name: str) -> Optional[dict]:
        """Return the most recent operation (train or predict) for a model.

        Returns:
            dict with keys: type ('train'|'predict'), date, mode, record_id.
            None if no operation found for this model.
        """
        train_info = self.last_train.get(model_name, {})
        pred_info = self.last_prediction.get(model_name, {})
        t_date = train_info.get("date", "")
        p_date = pred_info.get("date", "")
        if not t_date and not p_date:
            return None
        if t_date >= p_date:
            return {"type": "train", **train_info}
        return {"type": "predict", **pred_info}

    @classmethod
    def from_workspace(cls, workspace_root: str) -> "TrainingModeContext":
        """Factory method to build a TrainingModeContext from workspace filesystem."""
        all_model_keys = {}
        models_by_name = {}
        available_modes = set()
        anchor_date = None

        # 1. Parse latest_train_records.json
        records_path = os.path.join(workspace_root, "latest_train_records.json")
        if os.path.exists(records_path):
            try:
                with open(records_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                anchor_date = records.get("anchor_date")
                
                # Suffix parsing logic (e.g., lstm_Alpha158@static -> name="lstm_Alpha158", mode="static")
                models_dict = records.get("models", {})
                for key, run_id in models_dict.items():
                    all_model_keys[key] = run_id
                    if "@" in key:
                        name, mode = key.split("@", 1)
                        available_modes.add(mode)
                        models_by_name.setdefault(name, {})[mode] = run_id
                    else:
                        # Fallback for keys without suffix: assume static
                        available_modes.add("static")
                        models_by_name.setdefault(key, {})["static"] = run_id
            except Exception as e:
                logger.error(f"Failed to load latest_train_records.json: {e}")

        # 2. Parse config/rolling_config.yaml
        rolling_config = {}
        rc_path = os.path.join(workspace_root, "config", "rolling_config.yaml")
        if os.path.exists(rc_path):
            try:
                with open(rc_path, "r", encoding="utf-8") as f:
                    rolling_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load config/rolling_config.yaml: {e}")

        # 3. Parse rolling states (rolling_state.json -> slide, rolling_state_cpcv.json -> cpcv)
        rolling_states = {}
        state_files = {
            "rolling": "rolling_state.json",
            "cpcv_rolling": "rolling_state_cpcv.json"
        }
        for mode, filename in state_files.items():
            state_path = os.path.join(workspace_root, "data", filename)
            if os.path.exists(state_path):
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        rolling_states[mode] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")

        # 4. Parse config/model_config.json
        cpcv_config = {}
        mc_path = os.path.join(workspace_root, "config", "model_config.json")
        if os.path.exists(mc_path):
            try:
                with open(mc_path, "r", encoding="utf-8") as f:
                    mc_data = json.load(f)
                cpcv_config = mc_data.get("purged_cv", {})
            except Exception as e:
                logger.error(f"Failed to load config/model_config.json: {e}")

        # 5. Parse prediction_history.jsonl for predict-only tracking
        last_prediction = {}
        pred_path = os.path.join(workspace_root, "data", "prediction_history.jsonl")
        for entry in _read_jsonl(pred_path):
            name = entry.get("model_name")
            if name and entry.get("predicted_at", "") >= last_prediction.get(
                name, {}
            ).get("date", ""):
                last_prediction[name] = {
                    "date": entry["predicted_at"],
                    "mode": entry.get("mode", "static"),
                    "record_id": entry.get("record_id"),
                }

        # 6. Parse training_history.jsonl for per-model last-train tracking
        last_train = {}
        hist_path = os.path.join(workspace_root, "data", "training_history.jsonl")
        for entry in _read_jsonl(hist_path):
            name = entry.get("model_name")
            if name and entry.get("trained_at", "") >= last_train.get(
                name, {}
            ).get("date", ""):
                last_train[name] = {
                    "date": entry["trained_at"],
                    "mode": entry.get("mode", "static"),
                    "record_id": entry.get("record_id"),
                    "epochs": entry.get("actual_epochs"),
                    "early_stopped": entry.get("early_stopped"),
                }

        # 7. Detect predict-only cycle (>50% of models have predict as latest op)
        predict_only_cycle = False
        if last_prediction and anchor_date:
            all_models = set(last_prediction.keys()) | set(last_train.keys())
            if all_models:
                pred_latest = sum(
                    1
                    for m in all_models
                    if last_prediction.get(m, {}).get("date", "")
                    >= last_train.get(m, {}).get("date", "")
                )
                predict_only_cycle = pred_latest / len(all_models) > 0.5

        return cls(
            workspace_root=workspace_root,
            anchor_date=anchor_date,
            all_model_keys=all_model_keys,
            models_by_name=models_by_name,
            available_modes=sorted(list(available_modes)),
            rolling_config=rolling_config,
            rolling_states=rolling_states,
            cpcv_config=cpcv_config,
            last_prediction=last_prediction,
            last_train=last_train,
            is_predict_only_cycle=predict_only_cycle,
        )
