import json

from quantpits.model_capabilities.inspector import ModelCapabilityInspector
from quantpits.model_capabilities.contracts import RawModelCapabilityDeclaration
from quantpits.model_capabilities.probes import ImportObservation, _harness_protocol_measurements


def _raw(module):
    return {
        "model_module": module, "model_class": "ExampleModel", "wrapper_kind": "custom",
        "dataset_module": "public.datasets.example", "dataset_class": "DatasetH",
        "dataset_protocol": "point_in_time", "action": "predict_only",
        "execution_family": "static", "processor_profile": "safe_inference",
        "artifact_protocol": "artifact_v1", "dependency_profile": "python_example",
        "required_predicates": (
            "identity_canonical", "catalog_assigned", "dependency_available", "module_imported",
            "class_resolved", "constructor_signature", "fit_signature", "predict_signature",
            "device_available", "protocol_adapter", "capability_identity_match", "action_protocol",
            "dataset_protocol", "processor_tail_safe", "artifact_roundtrip", "prediction_shape",
            "prediction_tail", "prediction_gap", "prediction_unique", "prediction_finite",
            "wrapper_identity_match", "environment_isolated",
        ),
        "required": True,
    }


def _import_observation(module, _class_name):
    available = module != "public.models.unavailable"
    return ImportObservation(
        available, available, available, available, available, False, not available,
        "observed" if available else "dependency_missing",
    )


def _protocol(_declaration):
    declaration = (
        _declaration if isinstance(_declaration, RawModelCapabilityDeclaration)
        else RawModelCapabilityDeclaration.from_dict(_declaration)
    )
    return _harness_protocol_measurements(
        declaration,
        ("2026-07-17", "2026-07-20"), ("2026-07-17", "2026-07-20"),
        (0.1, 0.2),
    )


def test_available_capabilities_never_shrink_declared_rows():
    raw = (_raw("public.models.available"), _raw("public.models.unavailable"))
    matrix = ModelCapabilityInspector._with_probes(_import_observation, _protocol).inspect(raw)
    assert matrix.n_declarations == len(matrix.results) == 2
    assert [item.identity.model_module for item in matrix.results] == [
        "public.models.available", "public.models.unavailable",
    ]
    assert [item.status for item in matrix.results] == ["not_comparable", "conditional"]
    assert matrix.status == "none_supported"
    assert matrix.preflight_allowed is False


def test_capability_render_fingerprint_and_order_are_stable():
    raw = (_raw("public.models.first"), _raw("public.models.second"))
    first = ModelCapabilityInspector._with_probes(_import_observation, _protocol).inspect(raw)
    second = ModelCapabilityInspector._with_probes(_import_observation, _protocol).inspect(raw)
    assert first.fingerprint == second.fingerprint
    assert json.dumps(first.to_public_dict(), sort_keys=True) == json.dumps(second.to_public_dict(), sort_keys=True)
    assert [item.identity.model_module for item in first.results] == [
        "public.models.first", "public.models.second",
    ]
