import json
from pathlib import Path

import pytest

from quantpits.rolling import (
    QlibMlflowAggregateBackend,
    materialize_rolling_aggregate_candidates,
)
from quantpits.rolling.aggregate import (
    _candidate_from_observation,
    _candidate_manifest_contract_fingerprint,
)
from quantpits.rolling.errors import RollingAggregateBackendError
from quantpits.utils.workspace import fingerprint_value

from tests.quantpits.rolling.aggregate_support import (
    FakeCandidateBackend,
    aggregate_case,
)


def _real_backend_case(tmp_path, fault_hook=None):
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    fixture = FakeCandidateBackend(context)
    fixture_result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, fixture,
    )
    assert fixture_result.status == "success"
    fixture_candidate = fixture.candidates[aggregate.candidate_keys[0]]
    backend = QlibMlflowAggregateBackend(context, fault_hook=fault_hook)
    backend._assert_tracking = lambda: None
    return (
        context, repository, source, aggregate, backend,
        fixture_candidate["prediction_bytes"],
        fixture_candidate["manifest"],
    )


def test_real_backend_requires_active_finished_run_for_reuse(tmp_path):
    (
        _context, repository, source, aggregate, backend,
        _prediction, manifest,
    ) = _real_backend_case(
        tmp_path,
        fault_hook=lambda point: (
            (_ for _ in ()).throw(RuntimeError(point))
            if point == "after_candidate_manifest" else None
        ),
    )
    first = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert first.status == "failed"
    assert first.target_results[0].did_write is True
    backend._fault_hook = None
    observation = backend.inspect_candidate(
        aggregate, aggregate.target_keys[0], aggregate.candidate_keys[0],
        _candidate_manifest_contract_fingerprint(manifest),
    )
    assert observation == {"classification": "partial"}
    retry = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert retry.status == "blocked"
    assert retry.target_results[0].candidate is None
    assert "publication_input" not in retry.capabilities


def test_real_backend_experiment_namespace_is_counted_and_retryable(tmp_path):
    (
        _context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(
        tmp_path,
        fault_hook=lambda point: (
            (_ for _ in ()).throw(RuntimeError(point))
            if point == "after_candidate_experiment_namespace" else None
        ),
    )
    first = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert first.status == "failed"
    assert first.target_results[0].did_write is True
    inventory = backend.inventory(aggregate)
    assert inventory["experiment_present"] is True
    assert inventory["raw_count"] == 0
    backend._fault_hook = None
    retry = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert retry.status == "success"
    assert retry.target_results[0].status == "materialized_success"


def test_real_backend_provenance_tamper_cannot_become_reusable(tmp_path):
    (
        _context, _repository, _source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    target = aggregate.target_keys[0]
    candidate_key = aggregate.candidate_keys[0]
    original = backend.create_candidate(
        aggregate, target, candidate_key, prediction, manifest,
    )
    assert original["classification"] == "valid"
    recorder = next(iter(
        backend._recorders("Rolling_Aggregate_Candidates").values()
    ))
    artifact_uri = recorder.get_artifact_uri()
    artifact_root = Path(
        artifact_uri[7:] if artifact_uri.startswith("file://")
        else artifact_uri
    )
    manifest_path = artifact_root / "aggregate_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_recorder_ids"][0] = "foreign-source-recorder"
    core = dict(payload)
    core.pop("manifest_content_fingerprint")
    payload["manifest_content_fingerprint"] = fingerprint_value(core)
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    observed = backend.inspect_candidate(
        aggregate, target, candidate_key,
        _candidate_manifest_contract_fingerprint(manifest),
    )
    assert observed["classification"] == "identity_mismatch"
    expected = {
        "content_fingerprint": manifest["content_fingerprint"],
        "row_count": manifest["row_count"],
        "manifest_contract_fingerprint":
            _candidate_manifest_contract_fingerprint(manifest),
    }
    rebuilt = _candidate_from_observation(
        aggregate, target, candidate_key, observed, expected,
    )
    assert rebuilt.classification == "identity_mismatch"
    assert rebuilt.capabilities == ("render",)


def test_real_backend_deleted_finished_run_is_audit_only(tmp_path):
    (
        context, _repository, _source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    target = aggregate.target_keys[0]
    candidate_key = aggregate.candidate_keys[0]
    created = backend.create_candidate(
        aggregate, target, candidate_key, prediction, manifest,
    )
    assert created["classification"] == "valid"
    from mlflow.tracking import MlflowClient
    MlflowClient(tracking_uri=str(context.mlflow_uri)).delete_run(
        created["recorder_id"]
    )
    assert backend.inspect_candidate(
        aggregate, target, candidate_key,
        _candidate_manifest_contract_fingerprint(manifest),
    ) == {"classification": "partial"}


def test_real_backend_rejects_foreign_data_parent_before_lock_write(tmp_path):
    (
        context, _repository, _source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    context.data_dir.rename(context.root / "original_data")
    context.data_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RollingAggregateBackendError):
        backend.create_candidate(
            aggregate, aggregate.target_keys[0],
            aggregate.candidate_keys[0], prediction, manifest,
        )
    assert not (outside / "locks").exists()


def test_existing_candidate_experiment_requires_exact_frozen_artifact_root(
    tmp_path,
):
    (
        context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path)
    from mlflow.tracking import MlflowClient
    forbidden = context.output_dir / "forbidden-candidate-root"
    forbidden.mkdir(parents=True)
    MlflowClient(tracking_uri=str(context.mlflow_uri)).create_experiment(
        "Rolling_Aggregate_Candidates",
        artifact_location=forbidden.as_uri(),
    )
    before = tuple(forbidden.rglob("*"))
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "blocked"
    assert tuple(forbidden.rglob("*")) == before == ()
    assert "publication_input" not in result.capabilities


def test_reuse_requires_terminal_candidate_lock(tmp_path):
    (
        context, repository, source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    created = backend.create_candidate(
        aggregate, aggregate.target_keys[0],
        aggregate.candidate_keys[0], prediction, manifest,
    )
    assert created["classification"] == "valid"
    import fcntl
    lock_path = (
        context.data_dir / "locks"
        / "rolling_aggregate_candidate.lock"
    )
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = materialize_rolling_aggregate_candidates(
            aggregate, repository, source, backend,
        )
    assert result.status == "indeterminate"
    assert result.target_results[0].candidate is None
    assert "publication_input" not in result.capabilities


def test_reuse_never_recreates_a_missing_terminal_lock(tmp_path):
    (
        context, repository, source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    created = backend.create_candidate(
        aggregate, aggregate.target_keys[0],
        aggregate.candidate_keys[0], prediction, manifest,
    )
    assert created["classification"] == "valid"
    lock_path = (
        context.data_dir / "locks"
        / "rolling_aggregate_candidate.lock"
    )
    lock_path.unlink()
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "indeterminate"
    assert not lock_path.exists()
    assert "publication_input" not in result.capabilities


def test_candidate_backend_tracking_drift_blocks_before_write(tmp_path):
    (
        _context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path)

    def drifted():
        raise RollingAggregateBackendError("injected tracking drift")

    backend._assert_tracking = drifted
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "blocked"
    assert "publication_input" not in result.capabilities
