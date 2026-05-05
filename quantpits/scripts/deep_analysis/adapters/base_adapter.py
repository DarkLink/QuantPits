"""
Base types for Feedback Loop Adapters.

Provides the AdapterResult dataclass and BaseAdapter abstract base class
used by all concrete adapter implementations (TrainingAdapter, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any

from quantpits.scripts.deep_analysis.action_items import ActionItem


@dataclass
class AdapterResult:
    """Result of applying an ActionItem via an Adapter."""

    success: bool
    action_id: str
    adapter_type: str              # "training" | "search" | "fusion"
    modified_files: List[str] = field(default_factory=list)
    changes: List[dict] = field(default_factory=list)   # [{param, old, new, file}]
    error: str = ""


class BaseAdapter(ABC):
    """Abstract base class for all Feedback Loop Adapters.

    Each adapter translates a specific action_type into concrete workspace
    modifications (e.g., YAML edits, config updates).
    """

    adapter_type: str = "base"

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    @abstractmethod
    def apply(self, item: ActionItem) -> AdapterResult:
        """Execute the ActionItem, modifying workspace files.

        Args:
            item: A validated, in-scope ActionItem.

        Returns:
            AdapterResult indicating success/failure and what changed.
        """
        ...

    @abstractmethod
    def preview(self, item: ActionItem) -> dict:
        """Dry-run: return what *would* change without writing files.

        Returns:
            dict describing the planned modifications.
        """
        ...
