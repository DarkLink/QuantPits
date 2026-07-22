import os
from pathlib import Path

import pytest

from quantpits.model_capabilities.inspector import ModelCapabilityInspector
from quantpits.model_capabilities.catalog import AUTHORITATIVE_CATALOG
from quantpits.model_capabilities.contracts import RawModelCapabilityDeclaration
from quantpits.model_capabilities.probes import (
    ImportObservation,
    ControlledImportProbe,
    ZeroWriteObserver,
    _harness_protocol_measurements,
    classify_prediction_coverage,
    generated_protocol_fixture,
)

from .test_contracts import _raw


def _import_ok(_module, _class_name):
    return ImportObservation(True, True, True, True, True, False, False, "observed")


def _observation(index, scores, **overrides):
    identity_fields = {
        "model_module", "model_class", "wrapper_kind", "dataset_module", "dataset_class",
        "dataset_protocol", "action", "execution_family", "processor_profile",
        "artifact_protocol", "dependency_profile",
    }
    raw = _raw()
    measurement_overrides = {}
    for key, value in overrides.items():
        if key in identity_fields:
            raw[key] = value
        else:
            measurement_overrides[key] = value
    declaration = RawModelCapabilityDeclaration.from_dict(raw)
    return _harness_protocol_measurements(
        declaration,
        ("2026-07-17", "2026-07-20", "2026-07-21"),
        tuple(index), tuple(scores), **measurement_overrides
    )


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
    assert exact_result.status == "not_comparable"
    assert exact_result.reason == "wrapper_probe_not_authoritative"
    assert exact_result.preflight_allowed is False
    assert foreign_result.status == "not_comparable"
    assert foreign_result.preflight_allowed is False
    assert missing_result.status == "not_comparable"
    assert wrapper_result.status == "not_comparable"
    assert wrapper_result.reason == "capability_identity_mismatch"

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
    assert protocol_result.status == "not_comparable"
    assert protocol_result.reason == "capability_identity_mismatch"
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


def test_controlled_import_probe_detects_real_constructor_and_fit_signature_negatives():
    runner = ControlledImportProbe(timeout_seconds=5)
    required_constructor = runner.observe(
        "quantpits.model_capabilities.probes", "_ConstructorRequiresArgument",
    )
    missing_evals = runner.observe(
        "quantpits.model_capabilities.probes", "_FitWithoutEvalsResult",
    )
    assert required_constructor.imported is True
    assert required_constructor.constructor_signature is False
    assert required_constructor.fit_signature is True
    assert missing_evals.imported is True
    assert missing_evals.constructor_signature is True
    assert missing_evals.fit_signature is False


def test_incomplete_actual_protocol_rows_remain_not_comparable():
    matrix = ModelCapabilityInspector._with_probes(_import_ok).inspect(AUTHORITATIVE_CATALOG)
    assert matrix.n_supported == 0
    for result in matrix.results:
        if (
            result.identity.model_module.endswith("pytorch_lstm")
            and result.identity.action == "train"
            and result.identity.execution_family == "static"
        ):
            assert result.status == "not_comparable"
            assert result.reason == "protocol_adapter_not_available"
            assert result.preflight_allowed is False


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

    lifecycle = tmp_path / "lifecycle"
    lifecycle.mkdir()
    declaration = RawModelCapabilityDeclaration.from_dict(_raw())
    observation = _harness_protocol_measurements(
        declaration,
        ("2026-07-17", "2026-07-20", "2026-07-21"),
        ("2026-07-17", "2026-07-20", "2026-07-21"),
        (0.1, 0.2, 0.3),
    )

    def writing_protocol(_row):
        (lifecycle / "probe-write").write_text("detected", encoding="utf-8")
        return observation

    with pytest.raises(RuntimeError, match="protected root"):
        ModelCapabilityInspector._with_probes(
            _import_ok, writing_protocol, protected_roots=(lifecycle,),
        ).inspect((_raw(),))


def test_default_inspector_observer_covers_repository_public_boundary(tmp_path):
    repository = tmp_path / "repository"
    for relative in (".git", "plan", "workspaces", "quantpits", "docs", "tests", "output"):
        (repository / relative).mkdir(parents=True)
    declaration = RawModelCapabilityDeclaration.from_dict(_raw())
    observation = _harness_protocol_measurements(
        declaration, ("2026-07-21",), ("2026-07-21",), (0.1,),
    )

    for relative in ("root-cache.bin", "docs/cache.bin", "tests/cache.bin", "output/cache.bin"):
        target = repository / relative

        def writing_protocol(_row, path=target):
            path.write_bytes(b"observed")
            return observation

        with pytest.raises(RuntimeError, match="protected root"):
            ModelCapabilityInspector._with_probes(
                _import_ok, writing_protocol, repository_root=repository,
            ).inspect((_raw(),))
        target.unlink()

    inspector = ModelCapabilityInspector._with_probes(_import_ok, repository_root=repository)
    assert inspector._protected_roots == (repository.resolve(),)
    assert inspector._protected_exclusions == (".git", "plan", "workspaces")

    for relative in (".git/ignored", "plan/ignored", "workspaces/private-ignored"):
        target = repository / relative

        def writing_excluded(_row, path=target):
            path.write_bytes(b"excluded")
            return observation

        result = ModelCapabilityInspector._with_probes(
            _import_ok, writing_excluded, repository_root=repository,
        ).inspect((_raw(),)).results[0]
        assert result.status == "not_comparable"
        target.unlink()


def test_catalog_probe_is_workspace_and_backend_independent(monkeypatch, tmp_path):
    original_cwd = Path.cwd()
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", "sentinel_workspace_value")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sentinel_backend_value")
    declaration = next(
        item for item in AUTHORITATIVE_CATALOG
        if item.model_module.endswith("custom.pytorch_lstm")
        and item.action == "train" and item.execution_family == "static"
    )
    matrix = ModelCapabilityInspector._with_probes(_import_ok).inspect((declaration,))
    assert matrix.results[0].status == "not_comparable"
    assert matrix.results[0].reason == "protocol_adapter_not_available"
    unavailable = ImportObservation(False, False, False, False, False, False, True, "dependency_missing")
    catalog_matrix = ModelCapabilityInspector._with_probes(
        lambda _module, _class: unavailable,
    ).inspect_catalog()
    assert catalog_matrix.n_declarations == len(AUTHORITATIVE_CATALOG)
    assert Path.cwd() == original_cwd
    assert os.environ["QLIB_WORKSPACE_DIR"] == "sentinel_workspace_value"
    assert os.environ["MLFLOW_TRACKING_URI"] == "sentinel_backend_value"

    (tmp_path / "backend_hook.py").write_text(
        "import qlib\nqlib.init()\nclass Model: pass\n", encoding="utf-8",
    )
    (tmp_path / "workspace_import.py").write_text(
        "import quantpits.utils.env\nclass Model: pass\n", encoding="utf-8",
    )
    controlled = ControlledImportProbe(timeout_seconds=5)
    hook = controlled.observe("backend_hook", "Model", (tmp_path,))
    workspace = controlled.observe("workspace_import", "Model", (tmp_path,))
    assert hook.reason == "forbidden_backend_access"
    assert workspace.reason == "forbidden_backend_access"
