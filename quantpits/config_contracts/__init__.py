"""Workspace configuration validation and normalization contracts."""

from quantpits.config_contracts.core import (
    ConfigArtifact,
    ValidationMessage,
    WorkspaceValidationResult,
)
from quantpits.config_contracts.runtime_bridge import input_refs_from_validation
from quantpits.config_contracts.workspace import validate_workspace

__all__ = [
    "ConfigArtifact",
    "ValidationMessage",
    "WorkspaceValidationResult",
    "input_refs_from_validation",
    "validate_workspace",
]
