from dataclasses import replace

import pytest

from quantpits.rolling import (
    RollingAggregateContractError,
    RollingAggregateSourceFailure,
    RollingAggregateSourceUnit,
    build_rolling_aggregate_scope,
    inspect_rolling_aggregate_sources,
)

from tests.quantpits.rolling.aggregate_support import aggregate_case


def test_aggregate_scope_requires_matching_units_complete_authority(tmp_path):
    _context, scope, repository, _source, aggregate = aggregate_case(tmp_path)
    rebuilt = build_rolling_aggregate_scope(
        scope, repository.inspect_readonly(), "second-attempt",
    )
    assert rebuilt.requested_unit_keys == aggregate.requested_unit_keys
    with pytest.raises(RollingAggregateContractError):
        replace(aggregate, aggregate_attempt_id="forged")


def test_source_join_revalidates_every_state_claim_field(tmp_path):
    context, _scope, _repository, source, aggregate = aggregate_case(tmp_path)
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    foreign = replace(requests[0], recorder_id="foreign-recorder")
    observed = inspect_rolling_aggregate_sources(
        context, aggregate, (foreign,) + requests[1:], source,
    )
    assert observed.status == "observation_drifted"
    assert all(
        item.classification == "observation_drifted"
        for item in observed.unit_results
    )


@pytest.mark.parametrize("n_targets,n_windows", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_aggregate_set_identity_order_and_cardinality_are_conserved(
    tmp_path, n_targets, n_windows,
):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path, n_targets, n_windows,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    observed = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert observed.requested_unit_keys == aggregate.requested_unit_keys
    assert len(observed.unit_results) == n_targets * n_windows


def test_aggregate_rejects_forged_or_replayed_source_authority(tmp_path):
    context, _scope, _repository, source, aggregate = aggregate_case(tmp_path)
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    observed = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    item = observed.unit_results[0]
    with pytest.raises(RollingAggregateContractError):
        RollingAggregateSourceUnit(
            item.unit_key, item.request_fingerprint,
            item.evidence_fingerprint, item.recorder_id, item.sessions,
            item.canonical_rows, item.canonical_values,
            item.content_fingerprint,
        )
    with pytest.raises(RollingAggregateContractError):
        RollingAggregateSourceFailure(
            item.unit_key, "observation_drifted",
            "rolling_aggregate_source_observation_drifted",
        )


def test_aggregate_results_reject_impossible_cross_field_combinations():
    from quantpits.rolling import RollingAggregateTargetResult

    with pytest.raises(RollingAggregateContractError):
        RollingAggregateTargetResult(
            "demo_linear@rolling",
            (("demo_linear@rolling", "rolling:2026-01-01:2026-01-02:" + "a" * 64),),
            "materialized_success", False, None,
            "rolling_aggregate_target_materialized_success",
        )
    for observed_write in (False, True, None):
        result = RollingAggregateTargetResult(
            "demo_linear@rolling",
            ((
                "demo_linear@rolling",
                "rolling:2026-01-01:2026-01-02:" + "a" * 64,
            ),),
            "indeterminate", observed_write, None,
            "rolling_aggregate_target_indeterminate",
        )
        assert result.did_write is observed_write

    for status, observed_write in (
        ("indeterminate", 0),
        ("indeterminate", 1),
        ("failed", 0),
        ("failed", 1),
    ):
        with pytest.raises(RollingAggregateContractError):
            RollingAggregateTargetResult(
                "demo_linear@rolling",
                ((
                    "demo_linear@rolling",
                    "rolling:2026-01-01:2026-01-02:" + "a" * 64,
                ),),
                status, observed_write, None,
                "rolling_aggregate_target_%s" % status,
            )
