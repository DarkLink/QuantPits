import ast
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from quantpits.model_capabilities.contracts import (
    ModelCapabilityContractError,
    ModelCapabilityIdentity,
    ModelCapabilityMatrix,
    PredicateFact,
    RawModelCapabilityDeclaration,
    _capability_matrix,
    _raw_fingerprint,
    _terminal_result,
)
from quantpits.model_capabilities.inspector import ModelCapabilityInspector
from quantpits.model_capabilities.probes import ImportObservation, _harness_protocol_measurements
from quantpits.model_capabilities.catalog import AUTHORITATIVE_CATALOG


REQUIRED = (
    "identity_canonical", "catalog_assigned", "dependency_available", "module_imported",
    "class_resolved", "constructor_signature", "fit_signature", "predict_signature",
    "device_available", "protocol_adapter", "capability_identity_match", "action_protocol", "dataset_protocol",
    "processor_tail_safe", "artifact_roundtrip", "prediction_shape", "prediction_tail",
    "prediction_gap", "prediction_unique", "prediction_finite", "wrapper_identity_match",
    "environment_isolated",
)


def _raw(**overrides):
    values = {
        "model_module": "public.models.example", "model_class": "ExampleModel",
        "wrapper_kind": "custom", "dataset_module": "public.datasets.example",
        "dataset_class": "DatasetH", "dataset_protocol": "point_in_time",
        "action": "predict_only", "execution_family": "static",
        "processor_profile": "safe_inference", "artifact_protocol": "artifact_v1",
        "dependency_profile": "python_example", "required_predicates": REQUIRED,
        "required": True,
    }
    values.update(overrides)
    return values


def _import_ok(_module, _class_name):
    return ImportObservation(True, True, True, True, True, False, False, "observed")


def _protocol_ok(_declaration):
    return _harness_protocol_measurements(
        _declaration,
        ("2026-07-17", "2026-07-20"), ("2026-07-17", "2026-07-20"),
        (0.1, 0.2),
    )


@lru_cache(maxsize=1)
def _supported_matrix():
    declaration = next(
        item for item in AUTHORITATIVE_CATALOG
        if item.model_module.endswith("custom.pytorch_lstm")
        and item.action == "train" and item.execution_family == "static"
    )
    return ModelCapabilityInspector().inspect((declaration,))


@pytest.mark.parametrize("field,value", [
    ("model_module", None), ("model_module", ""), ("model_module", True),
    ("model_module", "/private/model.py"), ("model_module", "file:///private/model.py"),
    ("model_module", "workspaces/private/model"),
    ("model_class", 1), ("wrapper_kind", "CUSTOM"), ("dataset_protocol", []),
    ("action", "predict-only"), ("execution_family", 1.0),
    ("processor_profile", "unknown_processor"), ("artifact_protocol", "unknown_artifact"),
    ("dependency_profile", "unknown_dependency"),
])
def test_capability_identity_rejects_closest_invalid_representations(field, value):
    values = _raw()
    values.pop("required_predicates")
    values.pop("required")
    values[field] = value
    with pytest.raises(ModelCapabilityContractError):
        ModelCapabilityIdentity(**values)

    serialized = _raw()
    serialized[field] = value
    with pytest.raises(ModelCapabilityContractError):
        RawModelCapabilityDeclaration.from_dict(serialized)

    missing = _raw()
    missing.pop("model_module")
    with pytest.raises(ModelCapabilityContractError):
        RawModelCapabilityDeclaration.from_dict(missing)


def test_public_replay_and_replace_cannot_manufacture_supported_authority():
    import quantpits.model_capabilities as public_api

    assert not hasattr(public_api, "ProtocolMeasurements")
    matrix = _supported_matrix()
    result = matrix.results[0]
    assert result.status == "supported_verified"
    assert result.preflight_allowed is True

    with pytest.raises(ModelCapabilityContractError):
        replace(result, status="supported_verified")
    with pytest.raises(ModelCapabilityContractError):
        replace(matrix, results=matrix.results)
    with pytest.raises(ModelCapabilityContractError):
        PredicateFact("prediction_tail", "passed", "audit_replay", True, True, "forged")
    with pytest.raises(TypeError):
        ModelCapabilityInspector(_import_ok, _protocol_ok)

    replay = result.from_dict(result.to_public_dict())
    aggregate_replay = ModelCapabilityMatrix.from_dict(matrix.to_public_dict())
    assert replay.claimed_status == "supported_verified"
    assert replay.preflight_allowed is False
    assert aggregate_replay.claimed_status == "all_required_supported"
    assert aggregate_replay.preflight_allowed is False


def test_aggregate_counts_derive_from_raw_inventory_and_terminal_rows():
    declaration = next(
        item for item in AUTHORITATIVE_CATALOG
        if item.model_module.endswith("custom.pytorch_lstm")
        and item.action == "train" and item.execution_family == "static"
    )
    matrix = ModelCapabilityInspector().inspect((declaration, {"required": True}))
    assert matrix.n_declarations == 2
    assert len(matrix.results) == 1
    assert matrix.n_unassigned_declarations == 1
    assert matrix.n_supported == 1
    assert matrix.n_blocked == 0
    assert matrix.status == "inventory_invalid"
    rendered = matrix.to_public_dict()
    assert rendered["n_declarations"] == 2
    assert rendered["n_results"] == 1
    assert rendered["n_unassigned_declarations"] == 1


def test_impossible_terminal_status_and_capability_combinations_are_rejected():
    result = _supported_matrix().results[0]
    with pytest.raises(ModelCapabilityContractError):
        replace(result, did_probe=False)
    replay = result.from_dict(result.to_public_dict())
    assert replay.preflight_allowed is False

    declaration = RawModelCapabilityDeclaration.from_dict(_raw())
    raw_fingerprint = _raw_fingerprint(declaration.to_public_dict())
    with pytest.raises(ModelCapabilityContractError):
        _terminal_result(
            declaration, (), "supported_verified", "forged_support", False,
            0, raw_fingerprint,
        )
    with pytest.raises(ModelCapabilityContractError):
        _terminal_result(
            declaration, (), "conditional", "missing_condition_fact", True,
            0, raw_fingerprint,
        )
    with pytest.raises(ModelCapabilityContractError):
        _terminal_result(
            declaration, (), "unsupported", "unobserved_unsupported", False,
            0, raw_fingerprint,
        )
    with pytest.raises(ModelCapabilityContractError):
        _terminal_result(
            declaration, (), "invalid_declaration", "invalid_after_probe", True,
            0, raw_fingerprint,
        )


def test_aggregate_rejects_foreign_member_and_inventory_fingerprint():
    first = _supported_matrix()
    foreign_raw = _raw(model_module="public.models.foreign")
    foreign = ModelCapabilityInspector._with_probes(_import_ok, _protocol_ok).inspect((foreign_raw,))
    with pytest.raises(ModelCapabilityContractError):
        _capability_matrix(first.raw_fingerprints, foreign.results, ())
    with pytest.raises(ModelCapabilityContractError):
        first.query(foreign.results[0].identity)


def test_model_capability_modules_parse_as_python38():
    package = Path(__file__).resolve().parents[3] / "quantpits" / "model_capabilities"
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
        assert isinstance(tree, ast.Module)
