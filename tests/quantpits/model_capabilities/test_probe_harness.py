import os
from pathlib import Path

import pytest

from quantpits.model_capabilities.inspector import ModelCapabilityInspector
from quantpits.model_capabilities.probes import (
    ImportObservation,
    ProtocolMeasurements,
    ControlledImportProbe,
    ZeroWriteObserver,
    classify_prediction_coverage,
    generated_protocol_fixture,
)

from .test_contracts import _raw


def _import_ok(_module, _class_name):
    return ImportObservation(True, True, True, True, True, False, False, "observed")


def _observation(index, scores, **overrides):
    values = {
        "model_module": "public.models.example", "model_class": "ExampleModel",
        "measurement_source": "actual_wrapper_generated_protocol_probe",
        "expected_index": ("2026-07-17", "2026-07-20", "2026-07-21"),
        "observed_index": tuple(index), "scores": tuple(scores),
        "dataset_protocol": "point_in_time",
        "processor_input_index": ("2026-07-17", "2026-07-20", "2026-07-21"),
        "processor_output_index": ("2026-07-17", "2026-07-20", "2026-07-21"),
        "artifact_expected_type": "ExampleModel", "artifact_observed_type": "ExampleModel",
        "artifact_expected_source": "source_a", "artifact_observed_source": "source_a",
    }
    values.update(overrides)
    return ProtocolMeasurements(**values)


def test_generated_protocol_fixtures_cover_canonical_dataset_profiles():
    for protocol in (
        "point_in_time", "time_series", "memory_time_series", "daily_market_label", "multi_label",
    ):
        fixture = generated_protocol_fixture(protocol)
        assert fixture.dataset_protocol == protocol
        assert fixture.features.shape == (3, 2)
        assert fixture.expected_index[-1] == "2026-07-21"
        assert fixture.labels.shape[0] == 3
        if protocol == "daily_market_label":
            assert fixture.market_labels.shape[0] == 3
        if protocol == "multi_label":
            assert fixture.labels.shape[1] == 2


def test_harness_self_test_measurements_cannot_grant_actual_wrapper_support():
    measurement = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, 0.2, 0.3),
        measurement_source="harness_self_test_only",
    )
    result = ModelCapabilityInspector._with_probes(
        _import_ok, lambda _row: measurement,
    ).inspect((_raw(),)).results[0]
    assert result.status == "not_comparable"
    assert result.preflight_allowed is False


def test_label_dependent_processor_tail_drop_is_detected():
    observed = _observation(
        ("2026-07-17", "2026-07-20"), (0.1, 0.2),
        processor_output_index=("2026-07-17", "2026-07-20"),
    )
    matrix = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: observed).inspect((_raw(),))
    result = matrix.results[0]
    assert result.status == "coverage_unsafe"
    facts = {item.name: item for item in result.predicates}
    assert facts["processor_tail_safe"].outcome == "failed"
    assert facts["prediction_tail"].outcome == "failed"


def test_prediction_tail_gap_duplicate_and_nonfinite_are_distinct_failures():
    missing_tail = classify_prediction_coverage(_observation(
        ("2026-07-17", "2026-07-20"), (0.1, 0.2),
    ))
    gap = classify_prediction_coverage(_observation(
        ("2026-07-17", "2026-07-21"), (0.1, 0.2),
    ))
    duplicate = classify_prediction_coverage(_observation(
        ("2026-07-17", "2026-07-20", "2026-07-20"), (0.1, 0.2, 0.3),
    ))
    nonfinite = classify_prediction_coverage(_observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, float("nan"), 0.3),
    ))
    assert missing_tail["prediction_tail"] is False
    assert gap["prediction_gap"] is False and gap["prediction_tail"] is True
    assert duplicate["prediction_unique"] is False
    assert nonfinite["prediction_finite"] is False

    cases = (
        (_observation(("2026-07-17", "2026-07-20"), (0.1, 0.2)), "prediction_tail_missing"),
        (_observation(("2026-07-17", "2026-07-21"), (0.1, 0.2)), "prediction_internal_gap"),
        (_observation(("2026-07-17", "2026-07-20", "2026-07-21", "2026-07-21"), (0.1, 0.2, 0.3, 0.4)), "prediction_duplicate_index"),
        (_observation(("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, float("nan"), 0.3)), "prediction_non_finite"),
    )
    for measurement, expected_reason in cases:
        result = ModelCapabilityInspector._with_probes(
            _import_ok, lambda _row, item=measurement: item,
        ).inspect((_raw(),)).results[0]
        assert result.status == "coverage_unsafe"
        assert result.reason == expected_reason


def test_artifact_roundtrip_and_foreign_source_are_classified():
    exact = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, 0.2, 0.3),
    )
    foreign = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, 0.2, 0.3),
        artifact_observed_source="source_b",
    )
    missing = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, 0.2, 0.3),
        artifact_observed_type="MissingArtifact",
    )
    foreign_wrapper = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, 0.2, 0.3),
        model_module="public.models.foreign",
    )
    exact_result = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: exact).inspect((_raw(),)).results[0]
    foreign_result = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: foreign).inspect((_raw(),)).results[0]
    missing_result = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: missing).inspect((_raw(),)).results[0]
    wrapper_result = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: foreign_wrapper).inspect((_raw(),)).results[0]
    assert exact_result.status == "supported_verified"
    assert exact_result.preflight_allowed is True
    assert foreign_result.status == "not_comparable"
    assert foreign_result.preflight_allowed is False
    assert missing_result.status == "not_comparable"
    assert wrapper_result.status == "not_comparable"
    assert wrapper_result.reason == "wrapper_probe_not_authoritative"


def test_wrong_dataset_protocol_and_prediction_shape_fail_closed():
    wrong_protocol = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, 0.2, 0.3),
        dataset_protocol="time_series",
    )
    wrong_shape = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, True, 0.3),
    )
    protocol_result = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: wrong_protocol).inspect((_raw(),)).results[0]
    shape_result = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: wrong_shape).inspect((_raw(),)).results[0]
    assert protocol_result.status == "unsupported"
    assert protocol_result.reason == "wrong_dataset_protocol"
    assert shape_result.status == "not_comparable"
    assert shape_result.reason == "prediction_shape_not_comparable"


def test_controlled_import_probe_detects_missing_module_and_class():
    runner = ControlledImportProbe(timeout_seconds=5)
    missing_module = runner.observe("public.module.does_not_exist", "Missing")
    missing_class = runner.observe("quantpits.model_capabilities.catalog", "Missing")
    assert missing_module.imported is False
    assert missing_module.dependency_missing is True
    assert missing_class.imported is True
    assert missing_class.class_resolved is False
    assert missing_class.reason == "class_missing"


def test_probe_observer_detects_repository_cache_and_symlink_escape(tmp_path):
    protected = tmp_path / "repository"
    protected.mkdir()
    cache = protected / "cache.bin"
    cache.write_bytes(b"aa")
    with pytest.raises(RuntimeError, match="protected root"):
        with ZeroWriteObserver((protected,)):
            cache.write_bytes(b"bb")

    outside = tmp_path / "outside"
    outside.mkdir()
    (protected / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="external symlink"):
        with ZeroWriteObserver((protected,)):
            pass


def test_catalog_probe_is_workspace_and_backend_independent(monkeypatch):
    original_cwd = Path.cwd()
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", "sentinel_workspace_value")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sentinel_backend_value")
    observation = _observation(
        ("2026-07-17", "2026-07-20", "2026-07-21"), (0.1, 0.2, 0.3),
    )
    matrix = ModelCapabilityInspector._with_probes(_import_ok, lambda _row: observation).inspect((_raw(),))
    assert matrix.results[0].status == "supported_verified"
    assert Path.cwd() == original_cwd
    assert os.environ["QLIB_WORKSPACE_DIR"] == "sentinel_workspace_value"
    assert os.environ["MLFLOW_TRACKING_URI"] == "sentinel_backend_value"
