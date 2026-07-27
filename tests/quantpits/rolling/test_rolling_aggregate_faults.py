import pytest

from quantpits.rolling import materialize_rolling_aggregate_candidates

from tests.quantpits.rolling.aggregate_support import (
    FakeCandidateBackend,
    aggregate_case,
)


@pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_process_control_interrupts_propagate_without_forged_batch(tmp_path, control):
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend(context)
    backend.controls[0] = control
    with pytest.raises(control.__class__):
        materialize_rolling_aggregate_candidates(
            aggregate, repository, source, backend,
        )


@pytest.mark.parametrize("point,expected_write", [
    ("before_candidate_namespace", False),
    ("after_candidate_experiment_namespace", True),
    ("after_candidate_namespace", True),
    ("after_candidate_tags", True),
    ("after_candidate_prediction", True),
    ("after_candidate_manifest", True),
    ("after_candidate_reinspection", True),
])
def test_candidate_fault_matrix_matches_frozen_timeline(
    tmp_path, point, expected_write,
):
    context, _scope, repository, source, aggregate = aggregate_case(
        tmp_path, n_targets=2,
    )
    backend = FakeCandidateBackend(context)
    fired = []

    def fail_once(observed):
        if observed == point and not fired:
            fired.append(observed)
            raise RuntimeError(observed)

    backend._fault = fail_once
    source_calls = []
    original_prediction_bytes = source.prediction_bytes

    def observed_prediction_bytes(request):
        source_calls.append(request.unit_key)
        return original_prediction_bytes(request)

    source.prediction_bytes = observed_prediction_bytes
    state_before = repository.state_path.read_bytes()
    protected_before = backend.protected_snapshot(aggregate)
    backend_before = backend.backend_identity(aggregate)
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    inventory = backend.inventory(aggregate)
    assert result.status == "failed"
    assert result.target_results[0].did_write is expected_write
    assert result.target_results[0].status == "failed"
    assert result.target_results[0].candidate is None
    assert result.target_results[1].status == "materialized_success"
    assert tuple(item.target_key for item in result.target_results) == (
        aggregate.target_keys
    )
    assert backend.calls == [
        (aggregate.target_keys[0], aggregate.candidate_keys[0]),
        (aggregate.target_keys[1], aggregate.candidate_keys[1]),
    ]
    assert fired == [point]
    assert tuple(source_calls[:4]) == aggregate.requested_unit_keys
    assert tuple(source_calls[4:]) == aggregate.requested_unit_keys
    assert inventory["raw_count"] == (
        1 if point in (
            "before_candidate_namespace",
            "after_candidate_experiment_namespace",
        ) else 2
    )
    assert inventory["raw_count"] == len(inventory["candidates"])
    assert inventory["experiment_present"] is True
    assert result.to_public_dict()["n_failed"] == 1
    assert result.to_public_dict()["n_materialized"] == 1
    assert "publication_input" not in result.capabilities
    assert repository.state_path.read_bytes() == state_before
    assert backend.protected_snapshot(aggregate) == protected_before
    assert backend.backend_identity(aggregate) == backend_before
    assert not (context.data_dir / "latest_train_records.json").exists()


@pytest.mark.parametrize("drift_kind", [
    "repository", "backend", "protected",
])
def test_backend_or_workspace_drift_denies_candidate_success(
    tmp_path, drift_kind,
):
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend(context)
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
