"""Contained local Qlib/MLflow backend for immutable aggregate candidates."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from quantpits.rolling.aggregate import (
    AGGREGATE_PROTOCOL_VERSION,
    CANDIDATE_MANIFEST_CORE_FIELDS,
    CANDIDATE_EXPERIMENTS,
    RollingAggregateScope,
    _canonical_frame,
    _candidate_manifest_contract_fingerprint,
    _content_fingerprint,
    _index_fingerprint,
    _value_fingerprint,
)
from quantpits.rolling.errors import (
    RollingAggregateBackendError,
    RollingAggregateContractError,
)
from quantpits.rolling.mlflow_execution_backend import (
    _local_artifact_root,
    _tracking_uri_identity,
)
from quantpits.rolling.evidence import _secure_read
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


def _json_bytes(payload):
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _strict_json_object(data):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    payload = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    if not isinstance(payload, dict):
        raise ValueError("manifest root is not an object")
    return payload


def _contained_directory_identity(path, workspace_root, field):
    try:
        root = workspace_root.resolve(strict=True)
        root_meta = os.lstat(str(workspace_root))
        node_meta = os.lstat(str(path))
        physical = path.resolve(strict=True)
        physical.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RollingAggregateBackendError(
            "%s is physically foreign or unavailable" % field
        ) from exc
    if (
        stat.S_ISLNK(root_meta.st_mode)
        or not stat.S_ISDIR(root_meta.st_mode)
        or stat.S_ISLNK(node_meta.st_mode)
        or not stat.S_ISDIR(node_meta.st_mode)
    ):
        raise RollingAggregateBackendError(
            "%s is not a real directory" % field
        )
    return (node_meta.st_dev, node_meta.st_ino)


def _prospective_contained_path(uri, workspace_root, field):
    parsed = urlparse(str(uri))
    if parsed.scheme not in ("", "file") or parsed.netloc not in ("", None):
        raise RollingAggregateBackendError(
            "%s is not a local path" % field
        )
    raw = unquote(parsed.path if parsed.scheme else str(uri))
    path = Path(raw).expanduser().absolute()
    if path.is_symlink():
        raise RollingAggregateBackendError("%s is a symlink" % field)
    try:
        parent = path.parent.resolve(strict=True)
        parent.relative_to(workspace_root.resolve(strict=True))
        if path.exists():
            path.resolve(strict=True).relative_to(
                workspace_root.resolve(strict=True)
            )
    except (OSError, ValueError) as exc:
        raise RollingAggregateBackendError(
            "%s is physically foreign" % field
        ) from exc
    return path


def _ensure_child_directory(parent, name, parent_identity):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(str(parent), flags)
    try:
        meta = os.fstat(parent_fd)
        if (meta.st_dev, meta.st_ino) != parent_identity:
            raise RollingAggregateBackendError(
                "directory parent identity drifted before mkdir"
            )
        try:
            os.mkdir(name, dir_fd=parent_fd)
        except FileExistsError:
            pass
    finally:
        os.close(parent_fd)


def _open_regular_child(parent, name, parent_identity):
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(str(parent), parent_flags)
    try:
        parent_meta = os.fstat(parent_fd)
        if (parent_meta.st_dev, parent_meta.st_ino) != parent_identity:
            raise RollingAggregateBackendError(
                "file parent identity drifted before open"
            )
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    return os.fdopen(file_fd, "a+b")


class _MlflowRecorderView:
    def __init__(self, run):
        self._run = run
        self.info = {"id": str(run.info.run_id)}

    def list_tags(self):
        return dict(self._run.data.tags)

    def get_artifact_uri(self):
        return str(self._run.info.artifact_uri)

    def get_status(self):
        return str(self._run.info.status)

    def get_lifecycle_stage(self):
        return str(getattr(self._run.info, "lifecycle_stage", "active"))


class QlibMlflowAggregateBackend:
    """Append-only candidate writer and independent public-name inspector."""

    def __init__(self, context, fault_hook=None):
        if not isinstance(context, WorkspaceContext):
            raise RollingAggregateContractError(
                "aggregate backend requires WorkspaceContext"
            )
        self.context = context
        self._fault_hook = fault_hook
        self._staging_write_bytes = 0

    def _fault(self, point):
        if self._fault_hook is not None:
            self._fault_hook(point)

    @property
    def staging_write_bytes(self):
        return self._staging_write_bytes

    @property
    def backend_fingerprint(self):
        return fingerprint_value({"tracking_uri": str(self.context.mlflow_uri)})

    def _assert_tracking(self):
        from qlib.workflow import R

        current = str(R.get_uri())
        present, contained = _tracking_uri_identity(
            current, self.context.root,
        )
        if (
            current != str(self.context.mlflow_uri)
            or not present
            or not contained
        ):
            raise RollingAggregateBackendError(
                "aggregate tracking backend is foreign or unavailable"
            )

    def backend_identity(self, aggregate_scope):
        self._assert_tracking()
        return fingerprint_value({
            "workspace_fingerprint":
                aggregate_scope.execution_scope.run_identity.workspace_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "tracking_uri": str(self.context.mlflow_uri),
        })

    def protected_snapshot(self, aggregate_scope):
        """Hash publication/state/config facts without exposing their content."""

        relative_paths = (
            "data/rolling_state.json",
            "data/rolling_state_cpcv.json",
            "data/latest_train_records.json",
            "data/rolling_training_history.jsonl",
            "data/rolling_prediction_history.jsonl",
            "data/operator_log.jsonl",
            "config/prod_config.json",
            "config/model_registry.yaml",
        )
        root = self.context.root.resolve(strict=True)
        facts = []
        for relative in relative_paths:
            path = self.context.root / relative
            if not path.exists():
                facts.append((relative, "missing", None, None))
                continue
            if path.is_symlink() or not path.is_file():
                raise RollingAggregateBackendError(
                    "protected path is not a regular contained file"
                )
            physical = path.resolve(strict=True)
            try:
                physical.relative_to(root)
            except ValueError as exc:
                raise RollingAggregateBackendError(
                    "protected path escaped the workspace"
                ) from exc
            data = physical.read_bytes()
            facts.append((
                relative, "file", len(data),
                hashlib.sha256(data).hexdigest(),
            ))
        return fingerprint_value(facts)

    def _recorders(self, experiment_name):
        from mlflow.entities import ViewType
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=str(self.context.mlflow_uri))
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                return {}
            runs = []
            page_token = None
            while True:
                page = client.search_runs(
                    [str(experiment.experiment_id)],
                    run_view_type=ViewType.ALL,
                    max_results=1000,
                    page_token=page_token,
                )
                runs.extend(page)
                page_token = getattr(page, "token", None)
                if not page_token:
                    break
                if len(runs) >= 50000:
                    raise RollingAggregateBackendError(
                        "candidate recorder inventory exceeds its bound"
                    )
            return {
                str(run.info.run_id): _MlflowRecorderView(run)
                for run in runs
            }
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            raise RollingAggregateBackendError(
                "candidate recorder inventory is unavailable"
            ) from exc

    def _recorder(self, experiment_name, recorder_id):
        recorders = self._recorders(experiment_name)
        try:
            return recorders[recorder_id]
        except KeyError as exc:
            raise RollingAggregateBackendError(
                "candidate recorder is unavailable by public name"
            ) from exc

    def inventory(self, aggregate_scope):
        if not isinstance(aggregate_scope, RollingAggregateScope):
            raise RollingAggregateContractError(
                "inventory requires RollingAggregateScope"
            )
        self._assert_tracking()
        experiment_name = CANDIDATE_EXPERIMENTS[aggregate_scope.family]
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=str(self.context.mlflow_uri))
        try:
            experiment_present = (
                client.get_experiment_by_name(experiment_name) is not None
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            raise RollingAggregateBackendError(
                "candidate experiment inventory is unavailable"
            ) from exc
        rows = []
        for recorder_id, recorder in sorted(
            self._recorders(experiment_name).items()
        ):
            try:
                tags = recorder.list_tags()
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                tags = {}
            rows.append({
                "recorder_id": str(recorder_id),
                "candidate_key": tags.get("candidate_key"),
                "scope_fingerprint": tags.get("scope_fingerprint"),
                "aggregate_attempt_id": tags.get("aggregate_attempt_id"),
                "target_key": tags.get("target_key"),
                "run_status": recorder.get_status(),
                "lifecycle_stage": recorder.get_lifecycle_stage(),
            })
        return {
            "raw_count": len(rows),
            "fingerprint": fingerprint_value(rows),
            "candidates": tuple(rows),
            "experiment_present": experiment_present,
        }

    def inspect_candidate(
        self, aggregate_scope, target_key, candidate_key,
        expected_manifest_contract_fingerprint,
    ):
        if (
            not isinstance(expected_manifest_contract_fingerprint, str)
            or len(expected_manifest_contract_fingerprint) != 64
        ):
            raise RollingAggregateContractError(
                "candidate inspection requires the reobserved manifest contract"
            )
        self._assert_tracking()
        experiment_name = CANDIDATE_EXPERIMENTS[aggregate_scope.family]
        matches = []
        for recorder_id, recorder in sorted(
            self._recorders(experiment_name).items()
        ):
            try:
                tags = recorder.list_tags()
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                return {"classification": "not_comparable"}
            if tags.get("candidate_key") == candidate_key:
                matches.append((str(recorder_id), recorder, tags))
        if not matches:
            return {"classification": "missing"}
        if len(matches) != 1:
            return {"classification": "duplicate"}
        recorder_id, recorder, tags = matches[0]
        if (
            recorder.get_lifecycle_stage() != "active"
            or recorder.get_status() != "FINISHED"
        ):
            return {"classification": "partial"}
        expected_tags = {
            "aggregate_protocol": AGGREGATE_PROTOCOL_VERSION,
            "scope_fingerprint": aggregate_scope.scope_fingerprint,
            "aggregate_attempt_id": aggregate_scope.aggregate_attempt_id,
            "target_key": target_key,
            "candidate_key": candidate_key,
        }
        if any(str(tags.get(key)) != str(value) for key, value in expected_tags.items()):
            return {"classification": "foreign"}
        try:
            root = _local_artifact_root(
                recorder.get_artifact_uri(), self.context.root,
            )
            artifact_files = tuple(sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*") if path.is_file()
            ))
            if artifact_files != (
                "aggregate_manifest.json", "pred.pkl",
            ):
                return {"classification": "partial"}
            workspace_root = self.context.root.resolve(strict=True)
            pred_snapshot, pred_status, _detail, _checked = _secure_read(
                workspace_root, root, "pred.pkl",
            )
            manifest_snapshot, manifest_status, _detail, _checked = _secure_read(
                workspace_root, root, "aggregate_manifest.json",
            )
            if pred_status != "valid" or manifest_status != "valid":
                return {"classification": "drifted"}
            pred = pred_snapshot.data
            manifest_raw = manifest_snapshot.data
            manifest = _strict_json_object(manifest_raw)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return {"classification": "corrupt"}
        claimed = manifest.get("manifest_content_fingerprint")
        core = dict(manifest)
        core.pop("manifest_content_fingerprint", None)
        metadata_fields = {
            "candidate_experiment", "candidate_recorder_id",
            "candidate_pred_size", "candidate_pred_sha256",
            "manifest_content_fingerprint",
        }
        exact_fields = CANDIDATE_MANIFEST_CORE_FIELDS | metadata_fields
        checked_predicates = [
            "source_identity_order_cardinality",
            "source_state_join",
            "source_terminal_evidence",
            "source_session_exactness",
            "source_non_overlap",
            "candidate_index_exactness",
            "candidate_value_exactness",
            "candidate_content_exactness",
        ]
        list_fields = (
            "member_unit_keys", "source_request_fingerprints",
            "source_evidence_fingerprints", "source_recorder_ids",
            "source_sessions", "source_row_counts",
            "source_content_fingerprints", "expected_sessions",
            "checked_predicates",
        )
        if (
            set(manifest) != exact_fields
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 2
            or manifest.get("protocol") != AGGREGATE_PROTOCOL_VERSION
            or manifest.get("scope_fingerprint")
            != aggregate_scope.scope_fingerprint
            or manifest.get("aggregate_attempt_id")
            != aggregate_scope.aggregate_attempt_id
            or manifest.get("target_key") != target_key
            or manifest.get("candidate_key") != candidate_key
            or manifest.get("candidate_experiment") != experiment_name
            or type(manifest.get("row_count")) is not int
            or manifest.get("row_count") <= 0
            or any(not isinstance(manifest.get(field), list)
                   for field in list_fields)
            or manifest.get("checked_predicates") != checked_predicates
        ):
            return {"classification": "identity_mismatch"}
        member_count = len(manifest["member_unit_keys"])
        if (
            member_count <= 0
            or any(
                len(manifest[field]) != member_count
                for field in (
                    "source_request_fingerprints",
                    "source_evidence_fingerprints",
                    "source_recorder_ids",
                    "source_sessions",
                    "source_row_counts",
                    "source_content_fingerprints",
                )
            )
            or any(
                not isinstance(item, list) or len(item) != 2
                for item in manifest["member_unit_keys"]
            )
            or any(
                not isinstance(item, list) or not item
                for item in manifest["source_sessions"]
            )
            or any(
                type(item) is not int or item <= 0
                for item in manifest["source_row_counts"]
            )
            or sum(manifest["source_row_counts"]) != manifest["row_count"]
            or [session
                for sessions in manifest["source_sessions"]
                for session in sessions] != manifest["expected_sessions"]
        ):
            return {"classification": "identity_mismatch"}
        try:
            rows, values = _canonical_frame(
                pred, tuple(manifest["expected_sessions"]),
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return {"classification": "not_comparable"}
        if (
            claimed != fingerprint_value(core)
            or manifest.get("candidate_pred_size") != len(pred)
            or manifest.get("candidate_pred_sha256")
            != hashlib.sha256(pred).hexdigest()
            or manifest.get("candidate_recorder_id") != recorder_id
            or len(rows) != manifest.get("row_count")
            or _index_fingerprint(rows)
            != manifest.get("candidate_index_fingerprint")
            or _value_fingerprint(values)
            != manifest.get("candidate_value_fingerprint")
            or _content_fingerprint(rows, values)
            != manifest.get("content_fingerprint")
        ):
            return {"classification": "corrupt"}
        try:
            manifest_contract_fingerprint = (
                _candidate_manifest_contract_fingerprint({
                    key: manifest[key]
                    for key in CANDIDATE_MANIFEST_CORE_FIELDS
                })
            )
        except RollingAggregateContractError:
            return {"classification": "identity_mismatch"}
        if (
            manifest_contract_fingerprint
            != expected_manifest_contract_fingerprint
        ):
            return {"classification": "identity_mismatch"}
        return {
            "classification": "valid",
            "candidate_key": candidate_key,
            "target_key": target_key,
            "scope_fingerprint": aggregate_scope.scope_fingerprint,
            "aggregate_attempt_id": aggregate_scope.aggregate_attempt_id,
            "recorder_id": recorder_id,
            "manifest_fingerprint": hashlib.sha256(manifest_raw).hexdigest(),
            "content_fingerprint": manifest.get("content_fingerprint"),
            "row_count": manifest.get("row_count"),
            "manifest_contract_fingerprint":
                manifest_contract_fingerprint,
        }

    def create_candidate(
        self, aggregate_scope, target_key, candidate_key,
        prediction_bytes, manifest,
    ):
        if not isinstance(prediction_bytes, bytes) or not isinstance(manifest, dict):
            raise RollingAggregateContractError(
                "candidate write requires bytes and a manifest mapping"
            )
        expected_manifest_contract_fingerprint = (
            _candidate_manifest_contract_fingerprint(manifest)
        )
        if (
            manifest.get("schema_version") != 2
            or manifest.get("protocol") != AGGREGATE_PROTOCOL_VERSION
            or manifest.get("scope_fingerprint")
            != aggregate_scope.scope_fingerprint
            or manifest.get("aggregate_attempt_id")
            != aggregate_scope.aggregate_attempt_id
            or manifest.get("target_key") != target_key
            or manifest.get("candidate_key") != candidate_key
            or not isinstance(manifest.get("expected_sessions"), list)
        ):
            raise RollingAggregateContractError(
                "candidate manifest identity is invalid"
            )
        rows, values = _canonical_frame(
            prediction_bytes, tuple(manifest["expected_sessions"]),
        )
        if (
            len(rows) != manifest.get("row_count")
            or _index_fingerprint(rows)
            != manifest.get("candidate_index_fingerprint")
            or _value_fingerprint(values)
            != manifest.get("candidate_value_fingerprint")
            or _content_fingerprint(rows, values)
            != manifest.get("content_fingerprint")
        ):
            raise RollingAggregateContractError(
                "candidate manifest and prediction bytes disagree"
            )
        self._assert_tracking()
        before = self.inventory(aggregate_scope)
        existing = self.inspect_candidate(
            aggregate_scope, target_key, candidate_key,
            expected_manifest_contract_fingerprint,
        )
        if existing.get("classification") != "missing":
            return existing
        root_identity = _contained_directory_identity(
            self.context.root, self.context.root, "workspace root",
        )
        data_identity = _contained_directory_identity(
            self.context.data_dir, self.context.root, "workspace data parent",
        )
        mlruns_identity = _contained_directory_identity(
            self.context.mlruns_dir, self.context.root,
            "candidate artifact store parent",
        )
        lock_dir = self.context.data_dir / "locks"
        _ensure_child_directory(
            self.context.data_dir, "locks", data_identity,
        )
        lock_parent_identity = _contained_directory_identity(
            lock_dir, self.context.root, "aggregate lock parent",
        )
        lock_path = lock_dir / "rolling_aggregate_candidate.lock"
        if lock_path.is_symlink():
            raise RollingAggregateBackendError(
                "aggregate lock path must not be a symlink"
            )
        try:
            import fcntl
        except ImportError as exc:
            raise RollingAggregateBackendError(
                "aggregate lock platform is unsupported"
            ) from exc
        with _open_regular_child(
            lock_dir, lock_path.name, lock_parent_identity,
        ) as lock_handle:
            lock_open_meta = os.fstat(lock_handle.fileno())
            lock_public_meta = os.lstat(str(lock_path))
            lock_identity = (lock_open_meta.st_dev, lock_open_meta.st_ino)
            if (
                not stat.S_ISREG(lock_open_meta.st_mode)
                or not stat.S_ISREG(lock_public_meta.st_mode)
                or (lock_public_meta.st_dev, lock_public_meta.st_ino)
                != lock_identity
            ):
                raise RollingAggregateBackendError(
                    "aggregate lock node identity is not canonical"
                )

            def assert_base_identities():
                self._assert_tracking()
                if _contained_directory_identity(
                    self.context.root, self.context.root, "workspace root",
                ) != root_identity:
                    raise RollingAggregateBackendError(
                        "workspace root identity drifted"
                    )
                if _contained_directory_identity(
                    self.context.data_dir, self.context.root,
                    "workspace data parent",
                ) != data_identity:
                    raise RollingAggregateBackendError(
                        "workspace data parent identity drifted"
                    )
                if _contained_directory_identity(
                    lock_dir, self.context.root, "aggregate lock parent",
                ) != lock_parent_identity:
                    raise RollingAggregateBackendError(
                        "aggregate lock parent identity drifted"
                    )
                if _contained_directory_identity(
                    self.context.mlruns_dir, self.context.root,
                    "candidate artifact store parent",
                ) != mlruns_identity:
                    raise RollingAggregateBackendError(
                        "candidate artifact store parent identity drifted"
                    )
                public_lock = os.lstat(str(lock_path))
                open_lock = os.fstat(lock_handle.fileno())
                if (
                    public_lock.st_dev, public_lock.st_ino,
                ) != lock_identity or (
                    open_lock.st_dev, open_lock.st_ino,
                ) != lock_identity:
                    raise RollingAggregateBackendError(
                        "aggregate lock node identity drifted"
                    )

            try:
                fcntl.flock(
                    lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except OSError as exc:
                raise RollingAggregateBackendError(
                    "aggregate candidate lock is busy"
                ) from exc
            if self.inspect_candidate(
                aggregate_scope, target_key, candidate_key,
                expected_manifest_contract_fingerprint,
            ).get("classification") != "missing":
                raise RollingAggregateBackendError(
                    "candidate identity appeared after lock acquisition"
                )
            self._fault("before_candidate_namespace")
            assert_base_identities()
            experiment_name = CANDIDATE_EXPERIMENTS[aggregate_scope.family]
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=str(self.context.mlflow_uri))
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                artifact_location = (
                    self.context.data_dir
                    / ("rolling_aggregate_candidates_%s"
                       % aggregate_scope.family)
                ).as_uri()
                experiment_id = client.create_experiment(
                    experiment_name, artifact_location=artifact_location,
                )
                created_experiment = client.get_experiment(experiment_id)
                experiment_artifact_uri = created_experiment.artifact_location
                experiment_root = _prospective_contained_path(
                    experiment_artifact_uri, self.context.root,
                    "candidate experiment artifact root",
                )
                self._fault("after_candidate_experiment_namespace")
                assert_base_identities()
                if _contained_directory_identity(
                    self.context.mlruns_dir, self.context.root,
                    "candidate artifact store parent",
                ) != mlruns_identity:
                    raise RollingAggregateBackendError(
                        "candidate artifact store parent identity drifted"
                    )
            else:
                experiment_artifact_uri = experiment.artifact_location
                experiment_root = _prospective_contained_path(
                    experiment_artifact_uri, self.context.root,
                    "candidate experiment artifact root",
                )
                experiment_id = str(experiment.experiment_id)
            experiment_parent_identity = _contained_directory_identity(
                experiment_root.parent, self.context.root,
                "candidate experiment artifact parent",
            )
            run = client.create_run(
                experiment_id,
                tags={
                    "aggregate_protocol": AGGREGATE_PROTOCOL_VERSION,
                    "scope_fingerprint": aggregate_scope.scope_fingerprint,
                    "aggregate_attempt_id":
                        aggregate_scope.aggregate_attempt_id,
                    "target_key": target_key,
                    "candidate_key": candidate_key,
                    "candidate_kind": "non_current",
                },
            )
            recorder_id = str(run.info.run_id)
            try:
                self._fault("after_candidate_namespace")
                assert_base_identities()
                self._fault("after_candidate_tags")
                assert_base_identities()
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                try:
                    client.set_terminated(recorder_id, status="KILLED")
                except Exception:
                    pass
                raise
            except Exception:
                try:
                    client.set_terminated(recorder_id, status="FAILED")
                except Exception:
                    pass
                raise
            try:
                payload = dict(manifest)
                payload.update({
                    "candidate_experiment": experiment_name,
                    "candidate_recorder_id": recorder_id,
                    "candidate_pred_size": len(prediction_bytes),
                    "candidate_pred_sha256":
                        hashlib.sha256(prediction_bytes).hexdigest(),
                })
                payload["manifest_content_fingerprint"] = fingerprint_value(
                    payload
                )
                with tempfile.TemporaryDirectory(
                    prefix=".quantpits-aggregate-",
                    dir=str(self.context.data_dir),
                ) as temp_dir:
                    temporary = Path(temp_dir)
                    staging_identity = _contained_directory_identity(
                        temporary, self.context.root,
                        "candidate staging directory",
                    )
                    manifest_bytes = _json_bytes(payload)
                    with _open_regular_child(
                        temporary, "pred.pkl", staging_identity,
                    ) as staging_pred:
                        staging_pred.write(prediction_bytes)
                    with _open_regular_child(
                        temporary, "aggregate_manifest.json",
                        staging_identity,
                    ) as staging_manifest:
                        staging_manifest.write(manifest_bytes)
                    self._staging_write_bytes += (
                        len(prediction_bytes) + len(manifest_bytes)
                    )
                    if _contained_directory_identity(
                        temporary, self.context.root,
                        "candidate staging directory",
                    ) != staging_identity:
                        raise RollingAggregateBackendError(
                            "candidate staging directory identity drifted"
                        )
                    client.log_artifact(
                        recorder_id, str(temporary / "pred.pkl"),
                    )
                    self._fault("after_candidate_prediction")
                    assert_base_identities()
                    if _contained_directory_identity(
                        temporary, self.context.root,
                        "candidate staging directory",
                    ) != staging_identity:
                        raise RollingAggregateBackendError(
                            "candidate staging directory identity drifted"
                        )
                    client.log_artifact(
                        recorder_id,
                        str(temporary / "aggregate_manifest.json"),
                    )
                    self._fault("after_candidate_manifest")
                    assert_base_identities()
                client.set_terminated(recorder_id, status="FINISHED")
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                try:
                    client.set_terminated(recorder_id, status="KILLED")
                except Exception:
                    pass
                raise
            except Exception:
                try:
                    client.set_terminated(recorder_id, status="FAILED")
                except Exception:
                    pass
                raise
            recorder = self._recorder(experiment_name, recorder_id)
            candidate_artifact_root = _local_artifact_root(
                recorder.get_artifact_uri(), self.context.root,
            )
            _contained_directory_identity(
                candidate_artifact_root, self.context.root,
                "candidate artifact root",
            )
            if _contained_directory_identity(
                self.context.root, self.context.root, "workspace root",
            ) != root_identity:
                raise RollingAggregateBackendError(
                    "workspace root identity drifted"
                )
            if _contained_directory_identity(
                self.context.data_dir, self.context.root,
                "workspace data parent",
            ) != data_identity:
                raise RollingAggregateBackendError(
                    "workspace data parent identity drifted"
                )
            _local_artifact_root(
                experiment_artifact_uri, self.context.root,
            )
            if _contained_directory_identity(
                experiment_root.parent, self.context.root,
                "candidate experiment artifact parent",
            ) != experiment_parent_identity:
                raise RollingAggregateBackendError(
                    "candidate experiment artifact parent identity drifted"
                )
            completed_lock_meta = os.lstat(str(lock_dir))
            if (
                completed_lock_meta.st_dev,
                completed_lock_meta.st_ino,
            ) != lock_parent_identity:
                raise RollingAggregateBackendError(
                    "aggregate lock parent identity drifted"
                )
            completed_lock_node = os.lstat(str(lock_path))
            if (
                completed_lock_node.st_dev,
                completed_lock_node.st_ino,
            ) != lock_identity:
                raise RollingAggregateBackendError(
                    "aggregate lock node identity drifted"
                )
            after = self.inventory(aggregate_scope)
            if after["raw_count"] != before["raw_count"] + 1:
                raise RollingAggregateBackendError(
                    "candidate write did not create exactly one recorder"
                )
        observed = self.inspect_candidate(
            aggregate_scope, target_key, candidate_key,
            expected_manifest_contract_fingerprint,
        )
        self._fault("after_candidate_reinspection")
        return observed
