import json
import os
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
from quantpits.rolling.errors import (
    RollingAggregateBackendError,
    RollingExecutionBackendError,
)
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


@pytest.mark.parametrize("node_kind", ["hardlink", "fifo"])
def test_regular_child_create_rejects_aliased_or_special_node_before_write(
    tmp_path, node_kind,
):
    parent = tmp_path / "staging"
    parent.mkdir()
    child = parent / "pred.pkl"
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"must-not-change")
    if node_kind == "hardlink":
        os.link(sentinel, child)
    else:
        os.mkfifo(child)
    parent_meta = parent.stat()

    with pytest.raises(RollingAggregateBackendError):
        aggregate_backend_module._open_regular_child(
            parent, child.name,
            (parent_meta.st_dev, parent_meta.st_ino),
            create_if_missing=True, require_missing=True,
        )

    assert sentinel.read_bytes() == b"must-not-change"


def test_regular_child_exclusive_create_uses_excl_and_unlinked_node(
    tmp_path, monkeypatch,
):
    parent = tmp_path / "staging"
    parent.mkdir()
    parent_meta = parent.stat()
    original_open = aggregate_backend_module.os.open
    observed = {}

    def capture_open(path, flags, *args, **kwargs):
        if path == "pred.pkl" and kwargs.get("dir_fd") is not None:
            observed["flags"] = flags
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        aggregate_backend_module.os, "open", capture_open,
    )
    with aggregate_backend_module._open_regular_child(
        parent, "pred.pkl",
        (parent_meta.st_dev, parent_meta.st_ino),
        create_if_missing=True, require_missing=True,
    ) as handle:
        handle.write(b"candidate")
        handle.flush()

    assert observed["flags"] & os.O_EXCL
    assert (parent / "pred.pkl").stat().st_nlink == 1


def test_regular_child_observation_rejects_post_creation_hardlink(
    tmp_path,
):
    parent = tmp_path / "staging"
    parent.mkdir()
    parent_meta = parent.stat()
    with aggregate_backend_module._open_regular_child(
        parent, "pred.pkl",
        (parent_meta.st_dev, parent_meta.st_ino),
        create_if_missing=True, require_missing=True,
    ) as handle:
        handle.write(b"candidate")
        handle.flush()
    os.link(parent / "pred.pkl", tmp_path / "alias.pkl")

    with pytest.raises(RollingAggregateBackendError):
        aggregate_backend_module._observe_regular_child(
            parent, "pred.pkl",
            (parent_meta.st_dev, parent_meta.st_ino),
        )


@pytest.mark.parametrize("expected_present", [False, True])
def test_directory_establishment_rejects_presence_or_identity_race(
    tmp_path, monkeypatch, expected_present,
):
    parent = tmp_path / "data"
    parent.mkdir()
    child = parent / "candidate-root"
    if expected_present:
        child.mkdir()
        child_meta = child.stat()
        expected_identity = (child_meta.st_dev, child_meta.st_ino)
    else:
        expected_identity = None
    parent_meta = parent.stat()
    parent_identity = (parent_meta.st_dev, parent_meta.st_ino)
    original_mkdir = aggregate_backend_module.os.mkdir

    def replace_during_mkdir(path, mode=0o777, *, dir_fd=None):
        if path != child.name or dir_fd is None:
            return original_mkdir(path, mode, dir_fd=dir_fd)
        if expected_present:
            os.rename(
                child.name, "displaced",
                src_dir_fd=dir_fd, dst_dir_fd=dir_fd,
            )
        original_mkdir(child.name, mode, dir_fd=dir_fd)
        raise FileExistsError(child.name)

    monkeypatch.setattr(
        aggregate_backend_module.os, "mkdir", replace_during_mkdir,
    )
    with pytest.raises(
        RollingAggregateBackendError,
        match="presence drifted|identity drifted",
    ):
        aggregate_backend_module._establish_child_directory(
            parent,
            child.name,
            parent_identity,
            expected_present,
            expected_identity,
        )


@pytest.mark.parametrize("backend_kind", ["sqlite", "file"])
def test_tracking_backend_observer_detects_same_public_path_replacement(
    tmp_path, backend_kind,
):
    workspace = tmp_path / "Demo_Workspace"
    workspace.mkdir()
    if backend_kind == "sqlite":
        node = workspace / "mlflow.db"
        node.write_bytes(b"metadata")
        uri = "sqlite:///%s" % node
    else:
        node = workspace / "mlruns"
        node.mkdir()
        (node / "metadata").write_bytes(b"metadata")
        uri = node.as_uri()
    public_before, identity_before = (
        aggregate_backend_module._observe_tracking_backend(
            uri, workspace,
        )
    )
    displaced = node.with_name(node.name + "-displaced")
    node.rename(displaced)
    if backend_kind == "sqlite":
        shutil.copy2(displaced, node)
    else:
        shutil.copytree(displaced, node)
    public_after, identity_after = (
        aggregate_backend_module._observe_tracking_backend(
            uri, workspace,
        )
    )

    assert public_after == public_before == node
    assert identity_after != identity_before


@pytest.mark.parametrize("node_kind", ["database", "wal", "shm", "journal"])
def test_tracking_backend_observer_rejects_hardlinked_sqlite_nodes(
    tmp_path, node_kind,
):
    workspace = tmp_path / "Demo_Workspace"
    workspace.mkdir()
    database = workspace / "mlflow.db"
    database.write_bytes(b"database")
    target = database if node_kind == "database" else Path(
        str(database) + "-" + node_kind
    )
    if node_kind != "database":
        sentinel = tmp_path / ("sentinel-" + node_kind)
        sentinel.write_bytes(b"sidecar")
        os.link(sentinel, target)
    else:
        original = tmp_path / "database-original"
        database.rename(original)
        os.link(original, database)

    with pytest.raises(RollingExecutionBackendError):
        aggregate_backend_module._observe_tracking_backend(
            "sqlite:///%s" % database, workspace,
        )


@pytest.mark.parametrize(
    "node_kind",
    ["file_symlink", "directory_symlink", "ancestor_symlink", "fifo"],
)
def test_tracking_backend_observer_rejects_aliases_and_special_nodes(
    tmp_path, node_kind,
):
    workspace = tmp_path / "Demo_Workspace"
    workspace.mkdir()
    physical = workspace / "physical"
    if node_kind == "directory_symlink":
        physical.mkdir()
        node = workspace / "mlruns"
        node.symlink_to(physical, target_is_directory=True)
        uri = node.as_uri()
    elif node_kind == "ancestor_symlink":
        physical.mkdir()
        (physical / "mlflow.db").write_bytes(b"metadata")
        ancestor = workspace / "alias"
        ancestor.symlink_to(physical, target_is_directory=True)
        uri = "sqlite:///%s" % (ancestor / "mlflow.db")
    else:
        node = workspace / "mlflow.db"
        if node_kind == "file_symlink":
            physical.write_bytes(b"metadata")
            node.symlink_to(physical)
        else:
            os.mkfifo(str(node))
        uri = "sqlite:///%s" % node

    with pytest.raises(RollingExecutionBackendError):
        aggregate_backend_module._observe_tracking_backend(
            uri, workspace,
        )


def _real_backend_case(tmp_path, fault_hook=None):
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    from mlflow.tracking import MlflowClient

    MlflowClient(
        tracking_uri=str(context.mlflow_uri),
    ).get_experiment_by_name(
        "__rolling_aggregate_fixture_initialization__",
    )
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


@pytest.mark.parametrize("node_kind", ["hardlink", "fifo"])
def test_candidate_lock_rejects_hardlink_and_special_nodes(
    tmp_path, node_kind,
):
    (
        context, _repository, _source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path)
    lock_dir = context.data_dir / "locks"
    lock_dir.mkdir(exist_ok=True)
    lock = lock_dir / "rolling_aggregate_candidate.lock"
    sentinel = tmp_path / "lock-sentinel.bin"
    sentinel.write_bytes(b"must-not-change")
    if node_kind == "hardlink":
        os.link(sentinel, lock)
    else:
        os.mkfifo(lock)
    called = []

    with pytest.raises(RollingAggregateBackendError):
        backend.with_candidate_lock(
            aggregate, lambda: called.append(True),
            create_if_missing=True,
        )

    assert called == []
    assert sentinel.read_bytes() == b"must-not-change"


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


def test_real_backend_artifact_root_namespace_is_counted_before_experiment(
    tmp_path,
):
    (
        _context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(
        tmp_path,
        fault_hook=lambda point: (
            (_ for _ in ()).throw(RuntimeError(point))
            if point == "after_candidate_artifact_root" else None
        ),
    )
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    inventory = backend.inventory(aggregate)

    assert result.status == "failed"
    assert result.target_results[0].did_write is True
    assert inventory["artifact_root_present"] is True
    assert inventory["experiment_present"] is False
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


@pytest.mark.parametrize(
    ("fault_point", "namespace"),
    (
        ("after_candidate_namespace", "experiment"),
        ("after_candidate_prediction", "experiment"),
        ("after_candidate_prediction", "recorder"),
        ("after_candidate_prediction", "prediction"),
        ("after_candidate_manifest", "staging"),
        ("after_candidate_manifest", "manifest"),
    ),
)
def test_real_backend_rejects_namespace_replacement_during_candidate_create(
    tmp_path, fault_point, namespace,
):
    holder = {}
    replaced = {"done": False}

    def replace_directory(path):
        displaced = path.with_name(path.name + "-displaced")
        path.rename(displaced)
        shutil.copytree(displaced, path)

    def fault(point):
        if point != fault_point or replaced["done"]:
            return
        context = holder["context"]
        backend = holder["backend"]
        if namespace == "experiment":
            path = (
                context.data_dir
                / "rolling_aggregate_candidates_rolling"
            )
        elif namespace == "recorder":
            recorder = next(iter(
                backend._recorders(
                    "Rolling_Aggregate_Candidates",
                ).values()
            ))
            artifact_uri = recorder.get_artifact_uri()
            path = Path(
                artifact_uri[7:]
                if artifact_uri.startswith("file://")
                else artifact_uri
            )
        elif namespace == "staging":
            path = next(
                context.data_dir.glob(".quantpits-aggregate-*")
            )
        else:
            recorder = next(iter(
                backend._recorders(
                    "Rolling_Aggregate_Candidates",
                ).values()
            ))
            artifact_uri = recorder.get_artifact_uri()
            root = Path(
                artifact_uri[7:]
                if artifact_uri.startswith("file://")
                else artifact_uri
            )
            path = root / (
                "pred.pkl"
                if namespace == "prediction"
                else "aggregate_manifest.json"
            )
        if path.is_dir():
            replace_directory(path)
        else:
            displaced = path.with_name(path.name + "-displaced")
            path.rename(displaced)
            shutil.copy2(displaced, path)
        replaced["done"] = True

    (
        context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path, fault_hook=fault)
    holder.update(context=context, backend=backend)

    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )

    assert replaced["done"] is True
    assert result.status in ("failed", "indeterminate")
    assert result.target_results[0].did_write is True
    assert result.target_results[0].candidate is None
    assert "candidate_reference" not in (
        result.target_results[0].capabilities
    )
    assert "publication_input" not in result.capabilities


def test_candidate_root_identity_survives_create_to_terminal_recheck(
    tmp_path,
):
    (
        _context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path)
    original_create = backend.create_candidate
    replaced = {"done": False}

    def create_then_replace(*args, **kwargs):
        observation = original_create(*args, **kwargs)
        recorder = next(iter(
            backend._recorders(
                "Rolling_Aggregate_Candidates",
            ).values()
        ))
        artifact_uri = recorder.get_artifact_uri()
        public_root = Path(
            artifact_uri[7:]
            if artifact_uri.startswith("file://")
            else artifact_uri
        )
        displaced = public_root.with_name(
            public_root.name + "-displaced",
        )
        public_root.rename(displaced)
        shutil.copytree(displaced, public_root)
        replaced["done"] = True
        return observation

    backend.create_candidate = create_then_replace
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )

    assert replaced["done"] is True
    assert result.status == "indeterminate"
    assert result.target_results[0].did_write is True
    assert result.target_results[0].candidate is None
    assert "publication_input" not in result.capabilities


def test_existing_experiment_root_identity_survives_inventory_to_first_write(
    tmp_path,
):
    holder = {}
    replaced = {"done": False}

    def fault(point):
        if point != "before_candidate_namespace" or replaced["done"]:
            return
        root = (
            holder["context"].data_dir
            / "rolling_aggregate_candidates_rolling"
        )
        displaced = root.with_name(root.name + "-displaced")
        root.rename(displaced)
        shutil.copytree(displaced, root)
        replaced["done"] = True

    (
        context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path, fault_hook=fault)
    holder["context"] = context
    experiment_root = (
        context.data_dir / "rolling_aggregate_candidates_rolling"
    )
    experiment_root.mkdir()
    from mlflow.tracking import MlflowClient
    MlflowClient(
        tracking_uri=str(context.mlflow_uri),
    ).create_experiment(
        "Rolling_Aggregate_Candidates",
        artifact_location=experiment_root.as_uri(),
    )

    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )

    assert replaced["done"] is True
    assert result.status in ("failed", "indeterminate")
    assert result.target_results[0].candidate is None
    assert "publication_input" not in result.capabilities


@pytest.mark.parametrize(
    "replacement_kind", ("new_inode", "hardlink", "same_inode"),
)
def test_candidate_direct_artifact_identity_is_stable_during_inspection(
    tmp_path, monkeypatch, replacement_kind,
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
    artifact_root = Path(
        artifact_uri[7:] if artifact_uri.startswith("file://")
        else artifact_uri
    )
    prediction_path = artifact_root / "pred.pkl"
    displaced = tmp_path / "pred-displaced.pkl"
    original_decode = aggregate_backend_module._strict_json_object
    replaced = {"done": False}

    def decode_then_replace(data):
        payload = original_decode(data)
        if not replaced["done"]:
            if replacement_kind == "same_inode":
                original = prediction_path.read_bytes()
                before = prediction_path.stat()
                prediction_path.write_bytes(original)
                os.utime(
                    prediction_path,
                    ns=(
                        before.st_atime_ns,
                        before.st_mtime_ns + 1_000_000,
                    ),
                )
            else:
                prediction_path.rename(displaced)
                if replacement_kind == "hardlink":
                    os.link(displaced, prediction_path)
                else:
                    shutil.copy2(displaced, prediction_path)
            replaced["done"] = True
        return payload

    monkeypatch.setattr(
        aggregate_backend_module, "_strict_json_object",
        decode_then_replace,
    )
    observed = backend.inspect_candidate(
        aggregate, target, candidate_key,
        _candidate_manifest_contract_fingerprint(manifest),
    )

    assert replaced["done"] is True
    assert observed == {"classification": "drifted"}


def test_real_backend_rejects_same_uri_tracking_node_replacement(
    tmp_path,
):
    holder = {}
    replaced = {"done": False}
    identities = {}

    def fault(point):
        if point != "after_candidate_reinspection" or replaced["done"]:
            return
        database = holder["context"].root / "mlflow.db"
        identities["before"] = (
            database.stat().st_dev, database.stat().st_ino,
        )
        replacement = database.with_name("mlflow.db.replacement")
        shutil.copy2(database, replacement)
        os.replace(replacement, database)
        identities["after"] = (
            database.stat().st_dev, database.stat().st_ino,
        )
        replaced["done"] = True

    (
        context, repository, source, aggregate, backend,
        _prediction, _manifest,
    ) = _real_backend_case(tmp_path, fault_hook=fault)
    holder["context"] = context

    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )

    assert replaced["done"] is True
    assert identities["before"] != identities["after"]
    assert result.status in ("failed", "indeterminate")
    assert result.target_results[0].did_write is True
    assert result.target_results[0].candidate is None
    assert "publication_input" not in result.capabilities


@pytest.mark.parametrize("observer", ["inventory", "inspection", "lock"])
def test_real_backend_rechecks_tracking_node_across_every_observer(
    tmp_path, monkeypatch, observer,
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
    database = context.root / "mlflow.db"
    replaced = {"done": False}

    def replace_database():
        if replaced["done"]:
            return
        replacement = database.with_name("mlflow.db.replacement")
        shutil.copy2(database, replacement)
        os.replace(replacement, database)
        replaced["done"] = True

    if observer in ("inventory", "inspection"):
        original_recorders = backend._recorders

        def recorders_then_replace(experiment_name):
            observed = original_recorders(experiment_name)
            replace_database()
            return observed

        monkeypatch.setattr(
            backend, "_recorders", recorders_then_replace,
        )
    if observer == "inventory":
        with pytest.raises(RollingAggregateBackendError):
            backend.inventory(aggregate)
    elif observer == "inspection":
        observed = backend.inspect_candidate(
            aggregate, target, candidate_key,
            _candidate_manifest_contract_fingerprint(manifest),
        )
        assert observed == {"classification": "drifted"}
    else:
        with pytest.raises(RollingAggregateBackendError):
            backend.with_candidate_lock(
                aggregate, replace_database,
                create_if_missing=False,
            )
    assert replaced["done"] is True


def test_real_backend_tracking_node_drift_blocks_before_candidate_write(
    tmp_path, monkeypatch,
):
    (
        context, _repository, _source, aggregate, backend,
        prediction, manifest,
    ) = _real_backend_case(tmp_path)
    original_inventory = backend.inventory
    replaced = {"done": False}

    def inventory_then_replace(scope):
        observed = original_inventory(scope)
        if not replaced["done"]:
            database = context.root / "mlflow.db"
            replacement = database.with_name(
                "mlflow.db.replacement",
            )
            shutil.copy2(database, replacement)
            os.replace(replacement, database)
            replaced["done"] = True
        return observed

    monkeypatch.setattr(
        backend, "inventory", inventory_then_replace,
    )
    with pytest.raises(RollingAggregateBackendError):
        backend.create_candidate(
            aggregate, aggregate.target_keys[0],
            aggregate.candidate_keys[0], prediction, manifest,
        )

    assert replaced["done"] is True
    assert not (
        context.data_dir / "locks"
        / "rolling_aggregate_candidate.lock"
    ).exists()
    assert backend._recorders(
        "Rolling_Aggregate_Candidates",
    ) == {}


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
