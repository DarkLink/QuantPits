"""Public API for the read-only model capability matrix."""

from .catalog import AUTHORITATIVE_CATALOG, authoritative_catalog, declared_repository_models, repository_wrapper_inventory
from .contracts import (
    ACTIONS,
    ARTIFACT_PROTOCOLS,
    DATASET_PROTOCOLS,
    DEPENDENCY_PROFILES,
    EXECUTION_FAMILIES,
    PROCESSOR_PROFILES,
    TERMINAL_STATUSES,
    CapabilityReplay,
    ModelCapabilityContractError,
    ModelCapabilityIdentity,
    ModelCapabilityMatrix,
    ModelCapabilityResult,
    RawModelCapabilityDeclaration,
)
from .inspector import ModelCapabilityInspector, inspect_authoritative_catalog
from .probes import GeneratedProtocolFixture, ImportObservation, ProtocolMeasurements, generated_protocol_fixture

__all__ = [
    "ACTIONS", "ARTIFACT_PROTOCOLS", "AUTHORITATIVE_CATALOG", "CapabilityReplay", "DATASET_PROTOCOLS",
    "DEPENDENCY_PROFILES", "EXECUTION_FAMILIES", "GeneratedProtocolFixture", "ImportObservation",
    "ModelCapabilityContractError", "PROCESSOR_PROFILES",
    "ModelCapabilityIdentity", "ModelCapabilityInspector", "ModelCapabilityMatrix",
    "ModelCapabilityResult", "ProtocolMeasurements", "RawModelCapabilityDeclaration",
    "TERMINAL_STATUSES", "authoritative_catalog", "declared_repository_models", "generated_protocol_fixture",
    "inspect_authoritative_catalog", "repository_wrapper_inventory",
]
