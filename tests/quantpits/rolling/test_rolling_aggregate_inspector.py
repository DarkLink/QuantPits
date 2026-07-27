import io
from pathlib import Path

import pandas as pd
import pytest

from quantpits.rolling import (
    RollingAggregateContractError,
    inspect_rolling_aggregate_sources,
)

from tests.quantpits.rolling.aggregate_support import aggregate_case
from quantpits.utils.workspace import fingerprint_value


def _rewrite_source(source, request, frame):
    output = io.BytesIO()
    frame.to_pickle(output)
    original = source.prediction_bytes
    source.prediction_bytes = lambda item: (
        output.getvalue() if item.unit_key == request.unit_key
        else original(item)
    )


def test_aggregate_coverage_is_exact_requested_session_union(tmp_path):
    context, _scope, _repository, source, aggregate = aggregate_case(tmp_path)
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    frame = pd.read_pickle(io.BytesIO(source.prediction_bytes(requests[0])))
    _rewrite_source(source, requests[0], frame.iloc[:-1])
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "incomplete"
    assert result.unit_results[0].classification == "incomplete"


def test_aggregate_rejects_overlap_without_keep_last_semantics(tmp_path):
    context, _scope, _repository, source, aggregate = aggregate_case(tmp_path)
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    second = pd.read_pickle(io.BytesIO(source.prediction_bytes(requests[1])))
    first = pd.read_pickle(io.BytesIO(source.prediction_bytes(requests[0])))
    second.index = pd.MultiIndex.from_tuples(
        [first.index[0], second.index[1]], names=second.index.names,
    )
    _rewrite_source(source, requests[1], second)
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "incomplete"
    assert result.unit_results[1].classification == "incomplete"


def test_aggregate_rejects_every_foreign_source_dimension(tmp_path):
    context, _scope, _repository, source, aggregate = aggregate_case(tmp_path)
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    with pytest.raises(RollingAggregateContractError):
        inspect_rolling_aggregate_sources(
            context, aggregate, tuple(reversed(requests)), source,
        )


def test_aggregate_rejects_foreign_index_level_without_collapsing_it(
    tmp_path,
):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    frame = pd.read_pickle(io.BytesIO(source.prediction_bytes(requests[0])))
    frame.index = pd.MultiIndex.from_tuples(
        [
            (session, instrument, "foreign-partition")
            for session, instrument in frame.index
        ],
        names=("datetime", "instrument", "foreign_partition"),
    )
    _rewrite_source(source, requests[0], frame)
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "incomplete"
    assert result.unit_results[0].classification == "incomplete"


def test_aggregate_rejects_non_string_instrument_without_coercion(
    tmp_path,
):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    frame = pd.read_pickle(io.BytesIO(source.prediction_bytes(requests[0])))
    frame.index = pd.MultiIndex.from_tuples(
        [
            (session, position)
            for position, (session, _instrument) in enumerate(frame.index)
        ],
        names=("datetime", "instrument"),
    )
    _rewrite_source(source, requests[0], frame)
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "incomplete"
    assert result.unit_results[0].classification == "incomplete"


def test_source_backend_observation_failure_is_publicly_drifted(tmp_path):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )

    def unavailable(_request):
        raise OSError("injected observation failure")

    source.prediction_bytes = unavailable
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "observation_drifted"
    assert tuple(
        item.classification for item in result.unit_results
    ) == ("observation_drifted", "observation_drifted")
    assert all(
        item.capabilities == ("render",)
        for item in result.unit_results
    )


def test_source_partial_and_duplicate_evidence_are_incomplete(
    tmp_path,
):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path / "partial",
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    root = Path(
        source.candidates[requests[0].unit_key]["artifact_root_uri"][
            len("file://"):
        ]
    )
    (root / "pred.pkl").unlink()
    partial = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert partial.status == "incomplete"
    assert "aggregate_source" not in (
        partial.unit_results[0].capabilities
    )

    context2, _scope2, _repository2, source2, aggregate2 = (
        aggregate_case(tmp_path / "duplicate")
    )
    requests2 = source2.requests_for_state(
        aggregate2.execution_scope,
        aggregate2.state_repository_view.inspection.snapshot,
    )
    original_inventory = source2.inventory

    def duplicate_inventory(observed_requests):
        inventory = original_inventory(observed_requests)
        rows = inventory["candidates"] + (
            inventory["candidates"][0],
        )
        return {
            "fingerprint": fingerprint_value(rows),
            "candidates": rows,
        }

    source2.inventory = duplicate_inventory
    duplicate = inspect_rolling_aggregate_sources(
        context2, aggregate2, requests2, source2,
    )
    assert duplicate.status == "incomplete"
    assert "aggregate_source" not in (
        duplicate.unit_results[0].capabilities
    )


def test_candidate_write_parent_is_physically_contained_and_stable(tmp_path):
    from quantpits.rolling.errors import RollingExecutionBackendError
    from quantpits.rolling.mlflow_execution_backend import _local_artifact_root

    workspace = tmp_path / "Demo_Workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    escaped = workspace / "escaped"
    escaped.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RollingExecutionBackendError):
        _local_artifact_root(escaped.as_uri(), workspace)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -float("inf"), 2 ** 54])
def test_aggregate_score_normalization_is_loss_visible_and_finite(tmp_path, value):
    context, _scope, _repository, source, aggregate = aggregate_case(tmp_path)
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    frame = pd.read_pickle(io.BytesIO(source.prediction_bytes(requests[0])))
    if value == 2 ** 54:
        frame = frame.astype("int64")
    frame.iloc[0, 0] = value
    _rewrite_source(source, requests[0], frame)
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "incomplete"
    assert result.unit_results[0].classification == "incomplete"
