from pathlib import Path

import pytest

from quantpits.model_capabilities.contracts import ModelCapabilityContractError, RawModelCapabilityDeclaration
from quantpits.model_capabilities.inspector import ModelCapabilityInspector
from quantpits.model_capabilities.probes import ControlledImportProbe, ImportObservation

from .test_contracts import _protocol_ok, _raw


def _import_ok(_module, _class_name):
    return ImportObservation(True, True, True, True, True, False, False, "observed")


def test_duplicate_canonical_rows_remain_visible_and_block_support():
    raw = _raw()
    matrix = ModelCapabilityInspector._with_probes(_import_ok, _protocol_ok).inspect((raw, dict(raw)))
    assert matrix.n_declarations == 2
    assert len(matrix.results) == 2
    assert [item.status for item in matrix.results] == ["invalid_declaration", "invalid_declaration"]
    assert [item.reason for item in matrix.results] == ["duplicate_identity", "duplicate_identity"]
    assert matrix.status == "inventory_invalid"
    with pytest.raises(ModelCapabilityContractError, match="unknown or conflicted"):
        matrix.query(matrix.results[0].identity)


def test_import_and_constructor_success_do_not_imply_supported():
    matrix = ModelCapabilityInspector._with_probes(_import_ok).inspect((_raw(),))
    result = matrix.results[0]
    assert result.status == "not_comparable"
    assert {"module_imported", "class_resolved", "fit_signature", "predict_signature"}.issubset(result.checked_predicates)
    assert result.preflight_allowed is False


def test_protocol_measurement_is_bound_to_exact_action_family_and_profiles():
    train = RawModelCapabilityDeclaration.from_dict(_raw(action="train", execution_family="static"))
    measurement = _protocol_ok(train)
    same_row = ModelCapabilityInspector._with_probes(
        _import_ok, lambda _row: measurement,
    ).inspect((train,)).results[0]
    foreign_row = ModelCapabilityInspector._with_probes(
        _import_ok, lambda _row: measurement,
    ).inspect((_raw(action="resume", execution_family="rolling"),)).results[0]
    assert same_row.status == "not_comparable"
    assert same_row.reason == "wrapper_probe_not_authoritative"
    assert same_row.preflight_allowed is False
    assert foreign_row.status == "not_comparable"
    assert foreign_row.reason == "capability_identity_mismatch"
    facts = {item.name: item for item in foreign_row.predicates}
    assert facts["action_identity"].outcome == "failed"
    assert facts["action_protocol"].outcome == "failed"
    assert facts["execution_family_identity"].outcome == "failed"
    assert facts["capability_identity_match"].outcome == "failed"


def test_optional_dependency_absence_is_conditional_not_dropped():
    def dependency_missing(_module, _class_name):
        return ImportObservation(False, False, False, False, False, False, True, "dependency_missing")

    matrix = ModelCapabilityInspector._with_probes(dependency_missing).inspect((_raw(),))
    assert matrix.n_declarations == len(matrix.results) == 1
    assert matrix.results[0].status == "conditional"
    assert matrix.results[0].identity.model_class == "ExampleModel"
    assert matrix.n_conditional == 1


def test_import_class_signature_dependency_and_device_failures_are_typed():
    def varied(module, _class_name):
        if module.endswith("module_missing"):
            return ImportObservation(False, False, False, False, False, False, True, "dependency_missing")
        if module.endswith("class_missing"):
            return ImportObservation(True, False, False, False, False, False, False, "class_missing")
        if module.endswith("bad_constructor"):
            return ImportObservation(True, True, False, True, True, False, False, "constructor_incompatible")
        return ImportObservation(True, True, True, True, True, False, False, "observed")

    matrix = ModelCapabilityInspector._with_probes(varied).inspect((
        _raw(model_module="public.models.module_missing"),
        _raw(model_module="public.models.class_missing"),
        _raw(model_module="public.models.bad_constructor"),
        _raw(model_module="public.models.gpu_required", dependency_profile="python_gpu"),
    ))
    assert [item.status for item in matrix.results] == [
        "conditional", "probe_failed", "unsupported", "conditional",
    ]
    assert [item.reason for item in matrix.results] == [
        "dependency_missing", "class_missing", "wrapper_signature_incompatible", "gpu_required_cpu_only",
    ]
    gpu_facts = {item.name: item for item in matrix.results[-1].predicates}
    assert gpu_facts["device_available"].outcome == "failed"
    assert gpu_facts["dependency_available"].outcome == "failed"


def test_daily_and_multilabel_cpcv_without_projection_fail_closed():
    for protocol in ("daily_market_label", "multi_label"):
        matrix = ModelCapabilityInspector._with_probes(_import_ok, _protocol_ok).inspect((
            _raw(dataset_protocol=protocol, execution_family="cpcv"),
        ))
        assert matrix.results[0].status == "unsupported"
        assert matrix.results[0].reason == "cpcv_projection_missing"


def test_row_exception_fails_closed_and_preserves_later_rows():
    calls = []

    def failing_first(module, _class_name):
        calls.append(module)
        if module == "public.models.first":
            raise RuntimeError("ordinary operational failure")
        return _import_ok(module, _class_name)

    matrix = ModelCapabilityInspector._with_probes(failing_first).inspect((
        _raw(model_module="public.models.first"),
        _raw(model_module="public.models.second"),
    ))
    assert calls == ["public.models.first", "public.models.second"]
    assert [item.status for item in matrix.results] == ["probe_failed", "not_comparable"]
    assert [item.identity.model_module for item in matrix.results] == ["public.models.first", "public.models.second"]


def test_process_control_interrupt_propagates_and_cleans_probe_resources(tmp_path):
    marker = tmp_path / "probe-active"

    def interrupting_probe(_declaration):
        marker.write_text("active", encoding="utf-8")
        try:
            raise KeyboardInterrupt()
        finally:
            marker.unlink()

    with pytest.raises(KeyboardInterrupt):
        ModelCapabilityInspector._with_probes(_import_ok, interrupting_probe).inspect((_raw(),))
    assert not marker.exists()

    (tmp_path / "interrupt_module.py").write_text("raise KeyboardInterrupt()\n", encoding="utf-8")
    (tmp_path / "exit_module.py").write_text("raise SystemExit(9)\n", encoding="utf-8")
    controlled = ControlledImportProbe(timeout_seconds=5)
    with pytest.raises(KeyboardInterrupt):
        controlled.observe("interrupt_module", "Unused", (tmp_path,))
    with pytest.raises(SystemExit):
        controlled.observe("exit_module", "Unused", (tmp_path,))
    assert not tuple(tmp_path.glob("__pycache__"))
