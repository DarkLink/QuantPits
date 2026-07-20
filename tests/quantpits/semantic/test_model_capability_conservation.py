import json

from quantpits.model_capabilities.inspector import ModelCapabilityInspector
from quantpits.model_capabilities.probes import ImportObservation, ProtocolMeasurements


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
            "device_available",
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
    return ProtocolMeasurements(
        _declaration.model_module, _declaration.model_class,
        "actual_wrapper_generated_protocol_probe",
        ("2026-07-17", "2026-07-20"), ("2026-07-17", "2026-07-20"),
        (0.1, 0.2), "point_in_time", ("2026-07-17", "2026-07-20"),
        ("2026-07-17", "2026-07-20"), "ExampleModel", "ExampleModel",
        "source_a", "source_a",
    )


def test_available_capabilities_never_shrink_declared_rows():
    raw = (_raw("public.models.available"), _raw("public.models.unavailable"))
    matrix = ModelCapabilityInspector._with_probes(_import_observation, _protocol).inspect(raw)
    assert matrix.n_declarations == len(matrix.results) == 2
    assert [item.identity.model_module for item in matrix.results] == [
        "public.models.available", "public.models.unavailable",
    ]
    assert [item.status for item in matrix.results] == ["supported_verified", "conditional"]
    assert matrix.status == "partially_supported"
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
