"""
Base classes and data types for the MAS Deep Analysis System.

Defines the core abstractions: Finding, AgentFindings, AnalysisContext, and BaseAgent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd


@dataclass
class Finding:
    """A single observation or alert produced by an agent."""
    severity: str          # "critical" | "warning" | "info" | "positive"
    category: str          # Agent domain name (e.g., "Model Health")
    title: str             # One-line summary
    detail: str            # Full explanation
    data: Dict[str, Any] = field(default_factory=dict)  # Structured metrics for LLM

    def to_dict(self) -> dict:
        return {
            'severity': self.severity,
            'category': self.category,
            'title': self.title,
            'detail': self.detail,
            'data': self.data,
        }


@dataclass
class AgentFindings:
    """Complete output from one agent run on one time window."""
    agent_name: str
    window_label: str      # "full" | "weekly_era" | "1y" | "6m" | "3m" | "1m"
    findings: List[Finding] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'agent_name': self.agent_name,
            'window_label': self.window_label,
            'findings': [f.to_dict() for f in self.findings],
            'recommendations': self.recommendations,
            'raw_metrics': self.raw_metrics,
        }


@dataclass
class AnalysisContext:
    """
    Shared context passed to each agent for a specific time window.
    
    The Coordinator populates this with pre-loaded data and discovered file paths.
    DataFrames are pre-sliced to the window's date range.
    """
    start_date: str
    end_date: str
    window_label: str
    workspace_root: str
    external_notes: str = ""

    # Pre-loaded shared DataFrames (sliced per window by Coordinator)
    daily_amount_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    holding_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_classification_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Discovered file paths (merged from output/ + archive/ + order_history/)
    model_performance_files: List[str] = field(default_factory=list)
    ensemble_fusion_config_files: List[str] = field(default_factory=list)
    combo_comparison_files: List[str] = field(default_factory=list)
    leaderboard_files: List[str] = field(default_factory=list)
    correlation_matrix_files: List[str] = field(default_factory=list)
    model_opinions_files: List[str] = field(default_factory=list)
    buy_suggestion_files: List[str] = field(default_factory=list)
    sell_suggestion_files: List[str] = field(default_factory=list)
    model_contribution_files: List[str] = field(default_factory=list)

    # Frequency change info
    freq_change_date: Optional[str] = None
    is_pre_cutoff_window: bool = False  # True if window contains mostly pre-cutoff data


class BaseAgent(ABC):
    """Abstract base class for all specialist agents."""

    name: str = "BaseAgent"
    description: str = ""

    @abstractmethod
    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        """
        Run analysis on the given context and return findings.
        
        Args:
            ctx: AnalysisContext with pre-loaded data and file paths
            
        Returns:
            AgentFindings with observations, recommendations, and raw metrics
        """
        ...

    def _make_finding(self, severity: str, title: str, detail: str,
                      data: Optional[dict] = None) -> Finding:
        """Helper to create a Finding with this agent's category."""
        return Finding(
            severity=severity,
            category=self.name,
            title=title,
            detail=detail,
            data=data or {},
        )
