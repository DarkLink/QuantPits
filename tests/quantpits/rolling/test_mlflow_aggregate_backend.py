import json
from pathlib import Path
import shutil

import pytest

import quantpits.rolling.mlflow_aggregate_backend as aggregate_backend_module
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


def test_no_create_terminal_lock_open_is_read_only(tmp_path, monkeypatch):
    parent = tmp_path / "locks"
    parent.mkdir()
    lock = parent / "rolling_aggregate_candidate.lock"
    lock.write_bytes(b"")
    parent_meta = parent.stat()
    original_open = aggregate_backend_module.os.open
    observed = {}

    def capture_open(path, flags, *args, **kwargs):
        if path == lock.name and kwargs.get("dir_fd") is not None:
            observed["flags"] = flags
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        aggregate_backend_module.os, "open", capture_open,
    )
    handle = aggregate_backend_module._open_regular_child(
        parent, lock.name, (parent_meta.st_dev, parent_meta.st_ino),
        create_if_missing=False,
    )
    with handle:
        assert handle.read() == b""
    write_flags = (
        aggregate_backend_module.os.O_WRONLY
        | aggregate_backend_module.os.O_RDWR
        | aggregate_backend_module.os.O_APPEND
        | aggregate_backend_module.os.O_CREAT
    )
    assert observed["flags"] & write_flags == 0


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


@pytest.mark.parametrize(
    "node_kind",
    ["symlink_file", "symlink_directory", "directory", "fifo"],
)
def test_real_backend_rejects_non_regular_extra_candidate_artifact_nodes(
    tmp_path, node_kind,
):
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
    recorder = next(iter(
        backend._recorders("Rolling_Aggregate_Candidates").values()
    ))
    artifact_uri = recorder.get_artifact_uri()
    artifact_root = Path(
        artifact_uri[7:] if artifact_uri.startswith("file://")
        else artifact_uri
    )
    extra = artifact_root / "escaped"
    if node_kind == "directory":
        extra.mkdir()
    elif node_kind == "fifo":
        import os
        os.mkfifo(str(extra))
    else:
        outside = tmp_path / "outside"
        outside.mkdir(exist_ok=True)
        target_path = (
            outside if node_kind == "symlink_directory"
            else outside / "foreign.bin"
        )
        if node_kind == "symlink_file":
            target_path.write_bytes(b"foreign")
        extra.symlink_to(
            target_path,
            target_is_directory=node_kind == "symlink_directory",
        )

    observed = backend.inspect_candidate(
        aggregate, target, candidate_key,
        _candidate_manifest_contract_fingerprint(manifest),
    )

    assert observed == {"classification": "partial"}


@pytest.mark.parametrize("alias_level", ["root", "ancestor"])
def test_real_backend_public_artifact_root_alias_denies_candidate_authority(
    tmp_path, alias_level,
):
    (
        _context, repository, source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    target = aggregate.target_keys[0]
    candidate_key = aggregate.candidate_keys[0]
    created = backend.create_candidate(
        aggregate, target, candidate_key, prediction, manifest,
    )
    assert created["classification"] == "valid"
    recorder = next(iter(
        backend._recorders("Rolling_Aggregate_Candidates").values()
    ))
    artifact_uri = recorder.get_artifact_uri()
    public_root = Path(
        artifact_uri[7:] if artifact_uri.startswith("file://")
        else artifact_uri
    )
    public_node = (
        public_root if alias_level == "root" else public_root.parent
    )
    physical_node = public_node.with_name(
        public_node.name + "-physical",
    )
    public_node.rename(physical_node)
    public_node.symlink_to(physical_node, target_is_directory=True)

    observed = backend.inspect_candidate(
        aggregate, target, candidate_key,
        _candidate_manifest_contract_fingerprint(manifest),
    )
    assert observed == {"classification": "corrupt"}
    expected = {
        "content_fingerprint": manifest["content_fingerprint"],
        "row_count": manifest["row_count"],
        "manifest_contract_fingerprint":
            _candidate_manifest_contract_fingerprint(manifest),
    }
    rebuilt = _candidate_from_observation(
        aggregate, target, candidate_key, observed, expected,
    )
    assert "candidate_reference" not in rebuilt.capabilities

    terminal = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert terminal.status == "blocked"
    assert "publication_input" not in terminal.capabilities


def test_real_backend_rechecks_public_root_identity_after_artifact_reads(
    tmp_path, monkeypatch,
):
    (
        _context, _repository, _source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    target = aggregate.target_keys[0]
    candidate_key = aggregate.candidate_keys[0]
    created = backend.create_candidate(
        aggregate, target, candidate_key, prediction, manifest,
    )
    assert created["classification"] == "valid"
    recorder = next(iter(
        backend._recorders("Rolling_Aggregate_Candidates").values()
    ))
    artifact_uri = recorder.get_artifact_uri()
    public_root = Path(
        artifact_uri[7:] if artifact_uri.startswith("file://")
        else artifact_uri
    )
    displaced = public_root.with_name(public_root.name + "-displaced")
    original_decode = aggregate_backend_module._strict_json_object
    replaced = {"done": False}

    def decode_then_replace(data):
        payload = original_decode(data)
        if not replaced["done"]:
            replaced["done"] = True
            public_root.rename(displaced)
            shutil.copytree(displaced, public_root)
        return payload

    monkeypatch.setattr(
        aggregate_backend_module, "_strict_json_object",
        decode_then_replace,
    )
    observed = backend.inspect_candidate(
        aggregate, target, candidate_key,
        _candidate_manifest_contract_fingerprint(manifest),
    )
    assert observed == {"classification": "drifted"}
    expected = {
        "content_fingerprint": manifest["content_fingerprint"],
        "row_count": manifest["row_count"],
        "manifest_contract_fingerprint":
            _candidate_manifest_contract_fingerprint(manifest),
    }
    rebuilt = _candidate_from_observation(
        aggregate, target, candidate_key, observed, expected,
    )
    assert "candidate_reference" not in rebuilt.capabilities


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
    assert result.status == "blocked"
    assert result.target_results[0].did_write is False
    assert result.target_results[0].candidate is None
    assert "publication_input" not in result.capabilities


def test_create_lock_busy_is_blocked_without_candidate_write(tmp_path):
    (
        context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path)
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
    assert result.status == "blocked"
    assert result.target_results[0].did_write is False
    assert backend.inventory(aggregate)["raw_count"] == 0
    assert "publication_input" not in result.capabilities


def test_reuse_never_recreates_a_missing_terminal_lock_parent(tmp_path):
    (
        context, repository, source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    created = backend.create_candidate(
        aggregate, aggregate.target_keys[0],
        aggregate.candidate_keys[0], prediction, manifest,
    )
    assert created["classification"] == "valid"
    lock_dir = context.data_dir / "locks"
    for child in lock_dir.iterdir():
        child.unlink()
    lock_dir.rmdir()
    assert not lock_dir.exists()
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "blocked"
    assert result.target_results[0].did_write is False
    assert not lock_dir.exists()
    assert "publication_input" not in result.capabilities


def test_materialized_terminal_recheck_never_recreates_deleted_lock(
    tmp_path,
):
    (
        context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path)
    original_create = backend.create_candidate
    lock_path = (
        context.data_dir / "locks"
        / "rolling_aggregate_candidate.lock"
    )

    def create_then_delete_lock(*args, **kwargs):
        observation = original_create(*args, **kwargs)
        lock_path.unlink()
        return observation

    backend.create_candidate = create_then_delete_lock
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "indeterminate"
    assert result.target_results[0].did_write is True
    assert not lock_path.exists()
    assert "publication_input" not in result.capabilities


def test_reuse_lock_deleted_between_precheck_and_open_is_not_recreated(
    tmp_path, monkeypatch,
):
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
    original_open = aggregate_backend_module._open_regular_child
    observed = {"removed": False}

    def delete_before_open(
        parent, name, parent_identity, create_if_missing,
    ):
        if not create_if_missing:
            assert lock_path.exists()
            lock_path.unlink()
            observed["removed"] = True
        return original_open(
            parent, name, parent_identity,
            create_if_missing=create_if_missing,
        )

    monkeypatch.setattr(
        aggregate_backend_module, "_open_regular_child",
        delete_before_open,
    )
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert observed["removed"] is True
    assert result.status == "blocked"
    assert result.target_results[0].did_write is False
    assert not lock_path.exists()
    assert "publication_input" not in result.capabilities


def test_reuse_unopenable_terminal_lock_node_is_blocked_without_write(
    tmp_path,
):
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
    lock_path.mkdir()

    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "blocked"
    assert result.target_results[0].did_write is False
    assert lock_path.is_dir()
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
