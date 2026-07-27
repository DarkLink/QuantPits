import pytest

from quantpits.rolling import materialize_rolling_aggregate_candidates

from tests.quantpits.rolling.aggregate_support import (
    FakeCandidateBackend,
    aggregate_case,
)


@pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_process_control_interrupts_propagate_without_forged_batch(tmp_path, control):
    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend()
    backend.controls[0] = control
    with pytest.raises(control.__class__):
        materialize_rolling_aggregate_candidates(
            aggregate, repository, source, backend,
        )


@pytest.mark.parametrize("point,expected_write", [
    ("before_candidate_namespace", False),
    ("after_candidate_namespace", True),
    ("after_candidate_tags", True),
    ("after_candidate_prediction", True),
    ("after_candidate_manifest", True),
    ("after_candidate_reinspection", True),
])
def test_candidate_fault_matrix_matches_frozen_timeline(
    tmp_path, point, expected_write,
):
    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend()
    backend.fault_point = point
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "failed"
    assert result.target_results[0].did_write is expected_write


@pytest.mark.parametrize("drift_kind", [
    "repository", "backend", "protected",
])
def test_backend_or_workspace_drift_denies_candidate_success(
    tmp_path, drift_kind,
):
    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend()
    if drift_kind == "repository":
        original = repository.inspect_readonly
        calls = [0]

        def drift():
            calls[0] += 1
            if calls[0] == 1:
                return original()
            return object()

        repository.inspect_readonly = drift
    elif drift_kind == "backend":
        calls = [0]

        def backend_identity(_scope):
            calls[0] += 1
            return ("%064x" % calls[0])[-64:]

        backend.backend_identity = backend_identity
    else:
        calls = [0]

        def protected_snapshot(_scope):
            calls[0] += 1
            return ("%064x" % calls[0])[-64:]

        backend.protected_snapshot = protected_snapshot
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "indeterminate"
