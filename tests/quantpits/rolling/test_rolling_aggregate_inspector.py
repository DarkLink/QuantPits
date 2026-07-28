import hashlib
import io
from pathlib import Path
import shutil

import pandas as pd
import pytest

import quantpits.rolling.mlflow_execution_backend as execution_backend_module
from quantpits.rolling import (
    QlibMlflowExecutionBackend,
    RollingAggregateContractError,
    inspect_rolling_aggregate_sources,
)
from quantpits.rolling.errors import RollingExecutionBackendError

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


@pytest.mark.parametrize(
    "instrument",
    [
        "DEMO\x00FOREIGN", "DEMO\nFOREIGN",
        "DEMO\x7fFOREIGN", "DEMO\u0085FOREIGN",
    ],
)
def test_aggregate_rejects_instrument_control_characters(
    tmp_path, instrument,
):
    def inject_control(_unit, frame):
        frame.index = pd.MultiIndex.from_tuples(
            [
                (session, instrument)
                for session, _original in frame.index
            ],
            names=("datetime", "instrument"),
        )
        return frame

    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path, prediction_transform=inject_control,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "incomplete"
    assert result.unit_results[0].classification == "incomplete"


@pytest.mark.parametrize("replacement_kind", ["same_shape", "extra_byte"])
def test_source_backend_cannot_replace_frozen_prediction_bytes(
    tmp_path, replacement_kind,
):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    original_backend = source.prediction_bytes
    original = original_backend(requests[0])
    if replacement_kind == "same_shape":
        frame = pd.read_pickle(io.BytesIO(original))
        frame.iloc[0, 0] = float(frame.iloc[0, 0]) + 0.5
        output = io.BytesIO()
        frame.to_pickle(output)
        replacement = output.getvalue()
        assert len(replacement) == len(original)
        assert hashlib.sha256(replacement).digest() != hashlib.sha256(
            original
        ).digest()
        source.prediction_bytes = lambda request: (
            replacement
            if request.unit_key == requests[0].unit_key
            else original_backend(request)
        )
    else:
        source.prediction_bytes = lambda request: (
            original + b"x"
            if request.unit_key == requests[0].unit_key
            else original_backend(request)
        )

    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )

    assert result.status == "incomplete"
    assert result.unit_results[0].classification == "incomplete"
    assert "aggregate_source" not in result.unit_results[0].capabilities


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
    from quantpits.rolling.mlflow_execution_backend import _local_artifact_root

    workspace = tmp_path / "Demo_Workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    escaped = workspace / "escaped"
    escaped.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RollingExecutionBackendError):
        _local_artifact_root(escaped.as_uri(), workspace)


@pytest.mark.parametrize("alias_level", ["root", "ancestor"])
def test_source_artifact_public_root_alias_denies_aggregate_authority(
    tmp_path, alias_level,
):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    request = requests[0]
    public_roots = {
        item.recorder_id: Path(
            source.candidates[item.unit_key]["artifact_root_uri"][
                len("file://"):
            ]
        )
        for item in requests
    }
    public_root = public_roots[request.recorder_id]
    public_node = (
        public_root if alias_level == "root" else public_root.parent
    )
    physical_node = public_node.with_name(
        public_node.name + "-physical",
    )
    class Recorder:
        def __init__(self, root):
            self.root = root

        def get_artifact_uri(self):
            return self.root.as_uri()

    adapter = QlibMlflowExecutionBackend(context)
    adapter._recorder = lambda _experiment, recorder_id: Recorder(
        public_roots[recorder_id],
    )
    original_inspect = source.inspect

    def inspect_then_alias(scope, observed_requests):
        inspected = original_inspect(scope, observed_requests)
        public_node.rename(physical_node)
        public_node.symlink_to(
            physical_node, target_is_directory=True,
        )
        return inspected

    source.inspect = inspect_then_alias
    source.prediction_bytes = adapter.prediction_bytes

    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "observation_drifted"
    assert result.unit_results[0].classification == "observation_drifted"
    assert "aggregate_source" not in result.unit_results[0].capabilities


def test_source_artifact_root_identity_is_rechecked_after_prediction_read(
    tmp_path, monkeypatch,
):
    context, _scope, _repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    public_roots = {
        item.recorder_id: Path(
            source.candidates[item.unit_key]["artifact_root_uri"][
                len("file://"):
            ]
        )
        for item in requests
    }
    public_root = public_roots[requests[0].recorder_id]

    class Recorder:
        def __init__(self, root):
            self.root = root

        def get_artifact_uri(self):
            return self.root.as_uri()

    adapter = QlibMlflowExecutionBackend(context)
    adapter._recorder = lambda _experiment, recorder_id: Recorder(
        public_roots[recorder_id],
    )
    source.prediction_bytes = adapter.prediction_bytes
    original_read = execution_backend_module._secure_read
    replaced = {"done": False}

    def read_then_replace(root, artifact_root, logical_key):
        observed = original_read(root, artifact_root, logical_key)
        if artifact_root == public_root and not replaced["done"]:
            replaced["done"] = True
            displaced = public_root.with_name(
                public_root.name + "-displaced",
            )
            public_root.rename(displaced)
            shutil.copytree(displaced, public_root)
        return observed

    monkeypatch.setattr(
        execution_backend_module, "_secure_read", read_then_replace,
    )
    result = inspect_rolling_aggregate_sources(
        context, aggregate, requests, source,
    )
    assert result.status == "observation_drifted"
    assert result.unit_results[0].classification == "observation_drifted"
    assert "aggregate_source" not in result.unit_results[0].capabilities


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
