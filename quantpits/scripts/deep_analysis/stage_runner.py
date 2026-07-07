"""
Stage Runner for the MAS Deep Analysis System (Phase 1).

Orchestrates sequential stages of deep analysis:
1. discover: file scanning and initial window generation
2. agents: specialist agent execution
3. synthesis: cross-agent synthesis
4. window_analysis: training window rules
5. signals: signal extraction
6. critic: action item generation and validation
7. report: summary and markdown rendering

Supports checkpointing (save/load) of each stage's output.
"""

from __future__ import annotations

import os
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    stage: str
    date: str
    label: str
    timestamp: str
    predecessor: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class StageRunner:
    """Orchestrates staged execution of deep analysis with checkpointing."""

    STAGES = [
        'discover',
        'agents',
        'synthesis',
        'window_analysis',
        'signals',
        'critic',
        'report'
    ]

    def __init__(self, workspace_root: str, run_date: str, run_label: str = ""):
        self.workspace_root = workspace_root
        self.run_date = run_date
        self.run_label = run_label.strip()
        self.checkpoint_dir = os.path.join(workspace_root, 'output', 'deep_analysis', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Serialization / Deserialization helpers
    # ------------------------------------------------------------------

    def _serialize_val(self, val: Any) -> Any:
        if val is None or isinstance(val, (str, int, float, bool)):
            return val
        if hasattr(val, '_mock_return_value') or 'mock' in str(type(val)).lower():
            return f"MockObject({type(val).__name__})"
        if isinstance(val, (list, tuple, set)):
            return [self._serialize_val(x) for x in val]
        if isinstance(val, dict):
            return {str(k): self._serialize_val(v) for k, v in val.items()}
        if isinstance(val, pd.DataFrame):
            # If large, use parquet sidecar
            if len(val) > 10000:
                sidecar_name = f"sidecar_{uuid.uuid4().hex}.parquet"
                sidecar_path = os.path.join(self.checkpoint_dir, sidecar_name)
                val.to_parquet(sidecar_path)
                return {
                    "_type": "parquet_ref",
                    "path": sidecar_name
                }
            else:
                return {
                    "_type": "dataframe",
                    "columns": list(val.columns),
                    "records": val.to_dict(orient='records')
                }
        if isinstance(val, pd.Series):
            return {
                "_type": "series",
                "name": val.name,
                "index": list(val.index),
                "data": list(val.values)
            }
        if hasattr(val, 'to_dict'):
            try:
                return val.to_dict()
            except Exception:
                pass
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.isoformat()
        if hasattr(val, '__dataclass_fields__'):
            return asdict(val)
        return str(val)

    def _deserialize_val(self, val: Any) -> Any:
        if isinstance(val, dict):
            if val.get("_type") == "dataframe":
                # Convert records dict back to pandas DataFrame
                return pd.DataFrame(val["records"], columns=val["columns"])
            if val.get("_type") == "parquet_ref":
                sidecar_path = os.path.join(self.checkpoint_dir, val["path"])
                if os.path.exists(sidecar_path):
                    return pd.read_parquet(sidecar_path)
                logger.error(f"Sidecar Parquet file not found: {sidecar_path}")
                return pd.DataFrame()
            if val.get("_type") == "series":
                return pd.Series(val["data"], index=val["index"], name=val["name"])

            # Findings reconstruction
            if "severity" in val and "category" in val and "title" in val and "detail" in val:
                # Do NOT import at module-level to avoid circular imports
                from quantpits.scripts.deep_analysis.base_agent import Finding
                return Finding(
                    severity=val["severity"],
                    category=val["category"],
                    title=val["title"],
                    detail=val["detail"],
                    data=self._deserialize_val(val.get("data", {}))
                )
            if "agent_name" in val and "window_label" in val and "findings" in val:
                from quantpits.scripts.deep_analysis.base_agent import AgentFindings
                return AgentFindings(
                    agent_name=val["agent_name"],
                    window_label=val["window_label"],
                    findings=self._deserialize_val(val["findings"]),
                    recommendations=val.get("recommendations", []),
                    raw_metrics=self._deserialize_val(val.get("raw_metrics", {}))
                )

            # Signal reconstruction
            if "signal_type" in val and "severity" in val and "scope" in val and "source_agent" in val:
                from quantpits.scripts.deep_analysis.signal_extractor import Signal
                return Signal(
                    signal_type=val["signal_type"],
                    severity=val["severity"],
                    scope=val["scope"],
                    source_agent=val["source_agent"],
                    target=val.get("target", ""),
                    metrics=self._deserialize_val(val.get("metrics", {})),
                    context=val.get("context", "")
                )

            # ActionItem reconstruction
            if "action_id" in val and "action_type" in val and "scope" in val:
                from quantpits.scripts.deep_analysis.action_items import ActionItem
                return ActionItem(
                    action_id=val["action_id"],
                    action_type=val["action_type"],
                    scope=val["scope"],
                    target=val.get("target", ""),
                    params=self._deserialize_val(val.get("params", {})),
                    reason=val.get("reason", ""),
                    source_signals=val.get("source_signals", []),
                    expected_outcome=val.get("expected_outcome", ""),
                    confidence=val.get("confidence", 0.0),
                    risk_level=val.get("risk_level", "medium"),
                    scope_status=val.get("scope_status", "pending"),
                    rejected_reason=val.get("rejected_reason", ""),
                    validated_at=val.get("validated_at", ""),
                    execution_context=val.get("execution_context", {})
                )

            return {k: self._deserialize_val(v) for k, v in val.items()}

        if isinstance(val, list):
            return [self._deserialize_val(x) for x in val]
        return val

    # ------------------------------------------------------------------
    # Checkpoint File operations
    # ------------------------------------------------------------------

    def save_checkpoint(self, stage: str, data: dict, predecessor: Optional[str] = None) -> str:
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage name: {stage}. Must be one of {self.STAGES}")

        meta = CheckpointMeta(
            stage=stage,
            date=self.run_date,
            label=self.run_label,
            timestamp=datetime.now().isoformat(),
            predecessor=predecessor
        )

        serialized_data = self._serialize_val(data)
        payload = {
            "_meta": meta.to_dict(),
            "data": serialized_data
        }

        label_suffix = f"_{self.run_label}" if self.run_label else ""
        filename = f"{stage}_{self.run_date}{label_suffix}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Checkpoint saved for stage '{stage}' at {filepath}")
        return filepath

    def load_checkpoint(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        data = self._deserialize_val(payload.get("data", {}))
        meta_dict = payload.get("_meta", {})
        
        # Verify dates / stages match if applicable
        logger.info(f"Loaded checkpoint for stage '{meta_dict.get('stage')}' dated {meta_dict.get('date')}")
        return {
            "meta": meta_dict,
            "data": data
        }

    def find_latest_checkpoint(self, stage: str) -> Optional[str]:
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage name: {stage}")

        label_suffix = f"_{self.run_label}" if self.run_label else ""
        prefix = f"{stage}_{self.run_date}{label_suffix}"
        
        candidates = []
        if os.path.isdir(self.checkpoint_dir):
            for entry in os.listdir(self.checkpoint_dir):
                if entry.startswith(prefix) and entry.endswith('.json'):
                    candidates.append(os.path.join(self.checkpoint_dir, entry))

        if not candidates:
            # Fallback to no-label if label was specified but not found
            if self.run_label:
                fallback_prefix = f"{stage}_{self.run_date}"
                for entry in os.listdir(self.checkpoint_dir):
                    if entry.startswith(fallback_prefix) and entry.endswith('.json') and not entry[len(fallback_prefix):].startswith('_'):
                        candidates.append(os.path.join(self.checkpoint_dir, entry))

        if not candidates:
            return None

        # Return the newest based on file mtime
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return candidates[0]
