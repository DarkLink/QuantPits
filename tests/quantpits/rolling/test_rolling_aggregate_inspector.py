import io

import pandas as pd
import pytest

from quantpits.rolling import (
    RollingAggregateContractError,
    inspect_rolling_aggregate_sources,
)

from tests.quantpits.rolling.aggregate_support import aggregate_case


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
    with pytest.raises(RollingAggregateContractError):
        inspect_rolling_aggregate_sources(
            context, aggregate, requests, source,
        )


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
    with pytest.raises(RollingAggregateContractError):
        inspect_rolling_aggregate_sources(
            context, aggregate, requests, source,
        )


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
    with pytest.raises(RollingAggregateContractError):
        inspect_rolling_aggregate_sources(
            context, aggregate, requests, source,
        )
