"""Read-only model capability truth owner."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .catalog import AUTHORITATIVE_CATALOG, declared_repository_models, repository_wrapper_inventory
from .contracts import (
    ModelCapabilityIdentity,
    ModelCapabilityMatrix,
    ModelCapabilityResult,
    RawModelCapabilityDeclaration,
    UnassignedDeclaration,
    _capability_matrix,
    _predicate_fact,
    _raw_fingerprint,
    _terminal_result,
)
from .probes import (
    ControlledImportProbe, ControlledProtocolProbe, ImportObservation, ProtocolProbeFailure, ZeroWriteObserver,
    _ProtocolMeasurements, _action_protocol,
    classify_prediction_coverage,
)


ProtocolProbe = Callable[[RawModelCapabilityDeclaration], Any]
ImportProbe = Callable[[str, str], ImportObservation]
_PROBE_ADAPTER_TOKEN = object()


class ModelCapabilityInspector:
    """Inspect an ordered raw inventory and preserve every declaration."""

    def __init__(
        self,
        import_probe: Optional[ImportProbe] = None,
        protocol_probe: Optional[ProtocolProbe] = None,
        _adapter_authority: object = None,
        _protected_roots: Optional[Sequence[Path]] = None,
    ) -> None:
        if (import_probe is not None or protocol_probe is not None) and _adapter_authority is not _PROBE_ADAPTER_TOKEN:
            raise TypeError("custom capability probes are inspector-internal adapters")
        controlled = ControlledImportProbe()
        self._import_probe = import_probe or controlled.observe
        if protocol_probe is None:
            self._protocol_probe = ControlledProtocolProbe().observe
        else:
            def harness_only_probe(declaration: RawModelCapabilityDeclaration) -> Any:
                observation = protocol_probe(declaration)
                if isinstance(observation, ProtocolProbeFailure):
                    return observation
                if not isinstance(observation, _ProtocolMeasurements):
                    raise TypeError("custom protocol probes must return internal measurement envelopes")
                return observation.as_harness_only()
            self._protocol_probe = harness_only_probe
        self._environment_isolated = import_probe is None or _adapter_authority is _PROBE_ADAPTER_TOKEN
        self._protected_roots = tuple(_protected_roots or (Path(__file__).resolve().parents[1],))

    @classmethod
    def _with_probes(
        cls,
        import_probe: ImportProbe,
        protocol_probe: Optional[ProtocolProbe] = None,
        protected_roots: Optional[Sequence[Path]] = None,
    ) -> "ModelCapabilityInspector":
        return cls(
            import_probe, protocol_probe, _adapter_authority=_PROBE_ADAPTER_TOKEN,
            _protected_roots=protected_roots,
        )

    def inspect(self, raw_declarations: Sequence[Any]) -> ModelCapabilityMatrix:
        with ZeroWriteObserver(self._protected_roots):
            return self._inspect_observed(raw_declarations)

    def _inspect_observed(self, raw_declarations: Sequence[Any]) -> ModelCapabilityMatrix:
        if isinstance(raw_declarations, (str, bytes)) or not isinstance(raw_declarations, Sequence):
            raise TypeError("raw_declarations must be an ordered sequence")
        raw_fingerprints = tuple(_raw_fingerprint(self._public_raw(item)) for item in raw_declarations)
        parsed = []
        unassigned = []
        for position, raw in enumerate(raw_declarations):
            try:
                declaration = raw if isinstance(raw, RawModelCapabilityDeclaration) else RawModelCapabilityDeclaration.from_dict(raw)
                parsed.append((position, declaration))
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                required = bool(raw.get("required", True)) if isinstance(raw, Mapping) and type(raw.get("required", True)) is bool else True
                unassigned.append(UnassignedDeclaration(position, raw_fingerprints[position], "invalid_raw_declaration", required))

        identity_counts = Counter(ModelCapabilityIdentity.from_declaration(item).fingerprint for _position, item in parsed)
        import_cache = {}  # type: Dict[Tuple[str, str], ImportObservation]
        results = []
        for position, declaration in parsed:
            identity = ModelCapabilityIdentity.from_declaration(declaration)
            if identity_counts[identity.fingerprint] > 1:
                facts = (
                    _predicate_fact("identity_canonical", "passed", "actual_class_static_observation", True, True, "identity_canonical"),
                    _predicate_fact("catalog_assigned", "failed", "actual_class_static_observation", 1, identity_counts[identity.fingerprint], "duplicate_identity"),
                )
                results.append(_terminal_result(
                    declaration, facts, "invalid_declaration", "duplicate_identity", False,
                    position, raw_fingerprints[position],
                ))
                continue
            try:
                key = (declaration.model_module, declaration.model_class)
                if key not in import_cache:
                    import_cache[key] = self._import_probe(*key)
                results.append(self._inspect_one(
                    declaration, import_cache[key], position, raw_fingerprints[position],
                ))
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                facts = (
                    _predicate_fact("identity_canonical", "passed", "actual_class_static_observation", True, True, "identity_canonical"),
                    _predicate_fact("catalog_assigned", "passed", "actual_class_static_observation", True, True, "catalog_assigned"),
                )
                results.append(_terminal_result(
                    declaration, facts, "probe_failed", "row_probe_exception", True,
                    position, raw_fingerprints[position],
                ))
        return _capability_matrix(raw_fingerprints, results, unassigned)

    def inspect_catalog(self) -> ModelCapabilityMatrix:
        with ZeroWriteObserver(self._protected_roots):
            filesystem = repository_wrapper_inventory()
            declared = tuple(module for module, _class_name in declared_repository_models())
            drift = []
            for module in sorted(set(filesystem) - set(declared)):
                drift.append({"inventory_module": module, "inventory_drift": "undeclared", "required": True})
            for module in sorted(set(declared) - set(filesystem)):
                drift.append({"inventory_module": module, "inventory_drift": "missing", "required": True})
            return self._inspect_observed(tuple(AUTHORITATIVE_CATALOG) + tuple(drift))

    @staticmethod
    def _public_raw(value: Any) -> Any:
        return value.to_public_dict() if isinstance(value, RawModelCapabilityDeclaration) else value

    def _inspect_one(
        self,
        declaration: RawModelCapabilityDeclaration,
        imported: ImportObservation,
        raw_position: int,
        raw_fingerprint: str,
    ) -> ModelCapabilityResult:
        dependency_available = not imported.dependency_missing and (
            declaration.dependency_profile != "python_gpu" or imported.gpu_available
        )
        dependency_reason = imported.reason
        if declaration.dependency_profile == "python_gpu" and not imported.gpu_available:
            dependency_reason = "gpu_required_cpu_only"
        device_required = declaration.dependency_profile == "python_gpu"
        device_expected = True if device_required else "not_required"
        device_observed = imported.gpu_available if device_required else "not_required"
        facts = [
            _predicate_fact("identity_canonical", "passed", "actual_class_static_observation", True, True, "identity_canonical"),
            _predicate_fact("catalog_assigned", "passed", "actual_class_static_observation", True, True, "catalog_assigned"),
            _predicate_fact("dependency_available", "passed" if dependency_available else "failed", "actual_class_static_observation", True, dependency_available, dependency_reason),
            _predicate_fact("module_imported", "passed" if imported.imported else "failed", "actual_class_static_observation", True, imported.imported, imported.reason),
            _predicate_fact("class_resolved", "passed" if imported.class_resolved else "failed", "actual_class_static_observation", True, imported.class_resolved, imported.reason),
            _predicate_fact("constructor_signature", "passed" if imported.constructor_signature else "failed", "actual_class_static_observation", True, imported.constructor_signature, imported.reason),
            _predicate_fact("fit_signature", "passed" if imported.fit_signature else "failed", "actual_class_static_observation", True, imported.fit_signature, imported.reason),
            _predicate_fact("predict_signature", "passed" if imported.predict_signature else "failed", "actual_class_static_observation", True, imported.predict_signature, imported.reason),
            _predicate_fact("device_available", "passed" if not device_required or imported.gpu_available else "failed", "actual_class_static_observation", device_expected, device_observed, dependency_reason),
            _predicate_fact("dataset_declaration", "passed", "actual_class_static_observation", declaration.dataset_protocol, declaration.dataset_protocol, "declared_protocol_canonical"),
            _predicate_fact("environment_isolated", "passed" if self._environment_isolated else "failed", "actual_class_static_observation", True, self._environment_isolated, "controlled_probe_environment"),
        ]
        if not dependency_available:
            return _terminal_result(declaration, facts, "conditional", dependency_reason, True, raw_position, raw_fingerprint)
        if not imported.imported or not imported.class_resolved:
            return _terminal_result(declaration, facts, "probe_failed", imported.reason, True, raw_position, raw_fingerprint)
        if not imported.constructor_signature or not imported.fit_signature or not imported.predict_signature:
            return _terminal_result(declaration, facts, "unsupported", "wrapper_signature_incompatible", True, raw_position, raw_fingerprint)
        if (
            declaration.execution_family in ("cpcv", "cpcv_rolling")
            and declaration.dataset_protocol in ("daily_market_label", "multi_label")
        ):
            facts.append(_predicate_fact(
                "cpcv_projection", "failed", "actual_class_static_observation",
                "explicit_projection", "not_declared", "cpcv_projection_missing",
            ))
            return _terminal_result(declaration, facts, "unsupported", "cpcv_projection_missing", True, raw_position, raw_fingerprint)

        observation = self._protocol_probe(declaration)
        if isinstance(observation, ProtocolProbeFailure):
            facts.append(_predicate_fact(
                "protocol_adapter", "failed", "actual_class_static_observation",
                "exact_generated_protocol_adapter", observation.reason, observation.reason,
            ))
            return _terminal_result(declaration, facts, "not_comparable", observation.reason, True, raw_position, raw_fingerprint)
        coverage = classify_prediction_coverage(observation)
        observation_kind = observation.measurement_source
        expected_identity = ModelCapabilityIdentity.from_declaration(declaration)
        observed_identity = observation.identity
        identity_matches = observed_identity == expected_identity
        wrapper_identity_matches = (
            observed_identity.model_module == expected_identity.model_module
            and observed_identity.model_class == expected_identity.model_class
        )
        actual_authority = observation._actual_authority
        processor_tail_safe = (
            observation.processor_input_index == observation.expected_index
            and observation.processor_output_index == observation.expected_index
        )
        artifact_roundtrip = observation.artifact_expected_type == observation.artifact_observed_type
        artifact_source_matches = observation.artifact_expected_source == observation.artifact_observed_source
        prediction_shape_valid = (
            len(observation.observed_index) == len(observation.scores)
            and all(type(item) in (int, float) for item in observation.scores)
        )
        facts.extend((
            _predicate_fact("protocol_adapter", "passed" if actual_authority else "failed", observation_kind, "inspector_controlled_actual_wrapper", "inspector_controlled_actual_wrapper" if actual_authority else "caller_supplied_harness", "protocol_adapter_authority_observed"),
            _predicate_fact("capability_identity_match", "passed" if identity_matches else "failed", observation_kind, expected_identity.to_public_dict(), observed_identity.to_public_dict(), "complete_capability_identity_observed"),
            _predicate_fact("wrapper_identity_match", "passed" if wrapper_identity_matches else "failed", observation_kind, (declaration.model_module, declaration.model_class), (observation.model_module, observation.model_class), "wrapper_identity_observed"),
            _predicate_fact("wrapper_kind", "passed" if observation.wrapper_kind == declaration.wrapper_kind else "failed", observation_kind, declaration.wrapper_kind, observation.wrapper_kind, "wrapper_kind_observed"),
            _predicate_fact("dataset_identity", "passed" if (observation.dataset_module, observation.dataset_class) == (declaration.dataset_module, declaration.dataset_class) else "failed", observation_kind, (declaration.dataset_module, declaration.dataset_class), (observation.dataset_module, observation.dataset_class), "dataset_identity_observed"),
            _predicate_fact("dataset_protocol", "passed" if observation.dataset_protocol == declaration.dataset_protocol else "failed", observation_kind, declaration.dataset_protocol, observation.dataset_protocol, "dataset_protocol_observed"),
            _predicate_fact("action_identity", "passed" if observation.action == declaration.action else "failed", observation_kind, declaration.action, observation.action, "action_identity_observed"),
            _predicate_fact("action_protocol", "passed" if observation.action_protocol == _action_protocol(declaration.action) else "failed", observation_kind, _action_protocol(declaration.action), observation.action_protocol, "action_protocol_observed"),
            _predicate_fact("execution_family_identity", "passed" if observation.execution_family == declaration.execution_family else "failed", observation_kind, declaration.execution_family, observation.execution_family, "execution_family_identity_observed"),
            _predicate_fact("processor_profile_identity", "passed" if observation.processor_profile == declaration.processor_profile else "failed", observation_kind, declaration.processor_profile, observation.processor_profile, "processor_profile_identity_observed"),
            _predicate_fact("artifact_protocol_identity", "passed" if observation.artifact_protocol == declaration.artifact_protocol else "failed", observation_kind, declaration.artifact_protocol, observation.artifact_protocol, "artifact_protocol_identity_observed"),
            _predicate_fact("dependency_profile_identity", "passed" if observation.dependency_profile == declaration.dependency_profile else "failed", observation_kind, declaration.dependency_profile, observation.dependency_profile, "dependency_profile_identity_observed"),
            _predicate_fact("processor_tail_safe", "passed" if processor_tail_safe else "failed", observation_kind, observation.expected_index, observation.processor_output_index, "processor_tail_observed"),
            _predicate_fact("artifact_roundtrip", "passed" if artifact_roundtrip and artifact_source_matches else "failed", observation_kind, {"type": observation.artifact_expected_type, "source": observation.artifact_expected_source}, {"type": observation.artifact_observed_type, "source": observation.artifact_observed_source}, "artifact_source_observed"),
            _predicate_fact("prediction_shape", "passed" if prediction_shape_valid else "failed", observation_kind, (len(observation.observed_index), True), (len(observation.scores), all(type(item) in (int, float) for item in observation.scores)), "prediction_shape_observed"),
        ))
        for name in ("prediction_tail", "prediction_gap", "prediction_unique", "prediction_finite"):
            facts.append(_predicate_fact(name, "passed" if coverage[name] else "failed", observation_kind, True, coverage[name], "%s_observed" % name))
        if not identity_matches:
            return _terminal_result(declaration, facts, "not_comparable", "capability_identity_mismatch", True, raw_position, raw_fingerprint)
        if not processor_tail_safe or not all(coverage.values()):
            if not processor_tail_safe:
                coverage_reason = "processor_tail_unsafe"
            elif not coverage["prediction_tail"]:
                coverage_reason = "prediction_tail_missing"
            elif not coverage["prediction_unique"]:
                coverage_reason = "prediction_duplicate_index"
            elif not coverage["prediction_finite"]:
                coverage_reason = "prediction_non_finite"
            else:
                coverage_reason = "prediction_internal_gap"
            return _terminal_result(declaration, facts, "coverage_unsafe", coverage_reason, True, raw_position, raw_fingerprint)
        if not artifact_roundtrip or not artifact_source_matches:
            return _terminal_result(declaration, facts, "not_comparable", "artifact_not_comparable", True, raw_position, raw_fingerprint)
        if not prediction_shape_valid:
            return _terminal_result(declaration, facts, "not_comparable", "prediction_shape_not_comparable", True, raw_position, raw_fingerprint)
        if not actual_authority:
            return _terminal_result(declaration, facts, "not_comparable", "wrapper_probe_not_authoritative", True, raw_position, raw_fingerprint)
        return _terminal_result(declaration, facts, "supported_verified", "all_required_predicates_passed", True, raw_position, raw_fingerprint)


def inspect_authoritative_catalog() -> ModelCapabilityMatrix:
    return ModelCapabilityInspector().inspect_catalog()
