"""
DeepAnalysisState — typed container for cross-stage pipeline state.

Replaces the unstructured dict-based state passing in run_deep_analysis.py
with a typed dataclass that stages can read from and write to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DeepAnalysisState:
    """Cross-stage state for the deep analysis pipeline.

    Each stage reads its inputs from state fields populated by upstream
    stages and writes its outputs to the fields declared in its
    ``@register_stage(provides=[...])`` decorator.
    """

    # -- pipeline metadata --
    workspace_root: str = ""
    run_date: str = ""
    run_label: str = ""
    external_notes: str = ""
    freq_change_date: Optional[str] = None

    # -- Stage 1: discover --
    data_start_date: Optional[str] = None
    data_end_date: Optional[str] = None
    discovered_files: Optional[dict] = None
    windows: List[dict] = field(default_factory=list)

    # -- Stage 2: agents --
    all_findings: list = field(default_factory=list)

    # -- Stage 3: synthesis --
    synthesis_result: Optional[dict] = None

    # -- Stage 4: window_analysis --
    window_findings: list = field(default_factory=list)
    window_analysis_context: Optional[dict] = None

    # -- Stage 5: signals --
    signals: list = field(default_factory=list)

    # -- Stage 6: critic --
    action_items: list = field(default_factory=list)
    executive_summary: Optional[str] = None

    # -- Stage 7: report --
    report_md: Optional[str] = None
    output_path: Optional[str] = None

    # -- progress tracking --
    completed_stages: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        """Check whether *key* has been populated (non-None, non-empty)."""
        val = getattr(self, key, None)
        if val is None:
            return False
        if isinstance(val, (list, dict)) and len(val) == 0:
            return False
        return True

    def mark_completed(self, stage: str) -> None:
        """Record *stage* as having finished successfully."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    @property
    def last_completed_stage(self) -> Optional[str]:
        """The most recently completed stage, or None."""
        return self.completed_stages[-1] if self.completed_stages else None

    def is_stage_done(self, stage_name: str, provides: List[str]) -> bool:
        """Return True when every field in *provides* is populated."""
        return all(self.has(key) for key in provides)

    # ------------------------------------------------------------------
    # serialization (delegates to StageRunner's existing serializers)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to plain dict for checkpoint serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DeepAnalysisState":
        """Reconstruct from a dict (e.g. loaded from checkpoint)."""
        # Only pick known fields to stay resilient to extra keys
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
