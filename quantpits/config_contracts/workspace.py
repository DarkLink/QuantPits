"""Workspace-level configuration validation entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from quantpits.config_contracts.core import ConfigArtifact, ValidationMessage, WorkspaceValidationResult, has_errors
from quantpits.config_contracts.io import ConfigSpec, load_config_artifact
from quantpits.config_contracts.normalizers import (
    normalize_cashflow_config,
    normalize_ensemble_config,
    normalize_ensemble_records,
    normalize_latest_train_records,
    normalize_model_config,
    normalize_model_registry,
    normalize_prod_config,
    normalize_rolling_config,
    normalize_strategy_config,
)
from quantpits.config_contracts.validators import (
    validate_cashflow_config,
    validate_ensemble_config,
    validate_ensemble_cross_refs,
    validate_ensemble_records,
    validate_latest_train_records,
    validate_model_config,
    validate_model_registry,
    validate_prod_config,
    validate_rolling_config,
    validate_strategy_config,
)
from quantpits.utils.workspace import WorkspaceContext


def _specs(ctx: WorkspaceContext, *, include_optional: bool) -> List[ConfigSpec]:
    specs = [
        ConfigSpec("model_config", "config/model_config.json", True, "json", normalize_model_config, validate_model_config),
        ConfigSpec("strategy_config", "config/strategy_config.yaml", True, "yaml", normalize_strategy_config, validate_strategy_config),
        ConfigSpec("prod_config", "config/prod_config.json", True, "json", normalize_prod_config, validate_prod_config),
        ConfigSpec("ensemble_config", "config/ensemble_config.json", True, "json", normalize_ensemble_config, validate_ensemble_config),
        ConfigSpec(
            "model_registry",
            "config/model_registry.yaml",
            True,
            "yaml",
            normalize_model_registry,
            lambda data: validate_model_registry(data, workspace_root=ctx.root),
        ),
    ]
    if include_optional:
        specs.extend(
            [
                ConfigSpec("ensemble_records", "config/ensemble_records.json", False, "json", normalize_ensemble_records, validate_ensemble_records),
                ConfigSpec("rolling_config", "config/rolling_config.yaml", False, "yaml", normalize_rolling_config, validate_rolling_config),
                ConfigSpec("latest_train_records", "latest_train_records.json", False, "json", normalize_latest_train_records, validate_latest_train_records),
                ConfigSpec("cashflow", "config/cashflow.json", False, "json", normalize_cashflow_config, validate_cashflow_config),
            ]
        )
    return specs


def _artifact_by_name(artifacts: List[ConfigArtifact], name: str) -> Optional[ConfigArtifact]:
    for artifact in artifacts:
        if artifact.name == name:
            return artifact
    return None


def _add_cross_file_messages(artifacts: List[ConfigArtifact]) -> List[ValidationMessage]:
    messages: List[ValidationMessage] = []
    ensemble = _artifact_by_name(artifacts, "ensemble_config")
    registry = _artifact_by_name(artifacts, "model_registry")
    train_records = _artifact_by_name(artifacts, "latest_train_records")
    if ensemble is None or not isinstance(ensemble.normalized, dict):
        return messages

    registry_models = None
    if registry is not None and isinstance(registry.normalized, dict):
        models = registry.normalized.get("models")
        if isinstance(models, dict):
            registry_models = models.keys()

    train_record_models = None
    if train_records is not None and isinstance(train_records.normalized, dict):
        models = train_records.normalized.get("models")
        if isinstance(models, dict):
            train_record_models = models.keys()

    messages.extend(
        validate_ensemble_cross_refs(
            ensemble.normalized,
            registry_models=registry_models,
            train_record_models=train_record_models,
        )
    )
    return messages


def validate_workspace(
    ctx: WorkspaceContext,
    *,
    include_optional: bool = True,
    strict: bool = False,
) -> WorkspaceValidationResult:
    artifacts: List[ConfigArtifact] = []
    messages: List[ValidationMessage] = []

    for spec in _specs(ctx, include_optional=include_optional):
        artifact, artifact_messages = load_config_artifact(ctx, spec, strict=strict)
        artifacts.append(artifact)
        messages.extend(artifact_messages)

    messages.extend(_add_cross_file_messages(artifacts))
    return WorkspaceValidationResult(
        workspace=ctx.root,
        ok=not has_errors(messages),
        artifacts=tuple(artifacts),
        messages=tuple(messages),
    )
