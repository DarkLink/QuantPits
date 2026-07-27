"""Contained local Qlib/MLflow backend for immutable aggregate candidates."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from quantpits.rolling.aggregate import (
    AGGREGATE_PROTOCOL_VERSION,
    CANDIDATE_EXPERIMENTS,
    RollingAggregateScope,
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


class _MlflowRecorderView:
    def __init__(self, run):
        self._run = run
        self.info = {"id": str(run.info.run_id)}

    def list_tags(self):
        return dict(self._run.data.tags)

    def get_artifact_uri(self):
        return str(self._run.info.artifact_uri)


class QlibMlflowAggregateBackend:
    """Append-only candidate writer and independent public-name inspector."""

    def __init__(self, context, fault_hook=None):
        if not isinstance(context, WorkspaceContext):
            raise RollingAggregateContractError(
                "aggregate backend requires WorkspaceContext"
            )
        self.context = context
        self._fault_hook = fault_hook

    def _fault(self, point):
        if self._fault_hook is not None:
            self._fault_hook(point)

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
            runs = client.search_runs(
                [str(experiment.experiment_id)],
                run_view_type=ViewType.ALL,
                max_results=50000,
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
            })
        return {
            "raw_count": len(rows),
            "fingerprint": fingerprint_value(rows),
            "candidates": tuple(rows),
        }

    def inspect_candidate(
        self, aggregate_scope, target_key, candidate_key,
    ):
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
                continue
            if tags.get("candidate_key") == candidate_key:
                matches.append((str(recorder_id), recorder, tags))
        if not matches:
            return {"classification": "missing"}
        if len(matches) != 1:
            return {"classification": "duplicate"}
        recorder_id, recorder, tags = matches[0]
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
        exact_fields = {
            "schema_version", "protocol", "scope_fingerprint",
            "aggregate_attempt_id", "target_key", "candidate_key",
            "member_unit_keys", "source_recorder_ids",
            "source_evidence_fingerprints",
            "source_content_fingerprints", "expected_sessions",
            "row_count", "content_fingerprint", "candidate_experiment",
            "candidate_recorder_id", "candidate_pred_size",
            "candidate_pred_sha256", "manifest_content_fingerprint",
        }
        if (
            set(manifest) != exact_fields
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1
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
            or not isinstance(manifest.get("member_unit_keys"), list)
            or not isinstance(manifest.get("source_recorder_ids"), list)
            or not isinstance(
                manifest.get("source_evidence_fingerprints"), list,
            )
            or not isinstance(
                manifest.get("source_content_fingerprints"), list,
            )
            or not isinstance(manifest.get("expected_sessions"), list)
        ):
            return {"classification": "identity_mismatch"}
        try:
            from quantpits.rolling.aggregate import (
                _canonical_frame,
                _content_fingerprint,
            )
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
            or _content_fingerprint(rows, values)
            != manifest.get("content_fingerprint")
        ):
            return {"classification": "corrupt"}
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
        }

    def create_candidate(
        self, aggregate_scope, target_key, candidate_key,
        prediction_bytes, manifest,
    ):
        if not isinstance(prediction_bytes, bytes) or not isinstance(manifest, dict):
            raise RollingAggregateContractError(
                "candidate write requires bytes and a manifest mapping"
            )
        self._assert_tracking()
        before = self.inventory(aggregate_scope)
        existing = self.inspect_candidate(
            aggregate_scope, target_key, candidate_key,
        )
        if existing.get("classification") != "missing":
            return existing
        lock_dir = self.context.data_dir / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        root = self.context.root.resolve(strict=True)
        try:
            lock_dir.resolve(strict=True).relative_to(root)
            lock_meta = os.lstat(str(lock_dir))
        except (OSError, ValueError) as exc:
            raise RollingAggregateBackendError(
                "aggregate lock parent is physically foreign"
            ) from exc
        if (
            stat.S_ISLNK(lock_meta.st_mode)
            or not stat.S_ISDIR(lock_meta.st_mode)
        ):
            raise RollingAggregateBackendError(
                "aggregate lock parent is not a real directory"
            )
        lock_parent_identity = (lock_meta.st_dev, lock_meta.st_ino)
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
        with lock_path.open("a+b") as lock_handle:
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
            ).get("classification") != "missing":
                raise RollingAggregateBackendError(
                    "candidate identity appeared after lock acquisition"
                )
            self._fault("before_candidate_namespace")
            experiment_name = CANDIDATE_EXPERIMENTS[aggregate_scope.family]
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=str(self.context.mlflow_uri))
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(
                    experiment_name,
                )
                created_experiment = client.get_experiment(experiment_id)
                _local_artifact_root(
                    created_experiment.artifact_location,
                    self.context.root,
                )
            else:
                _local_artifact_root(
                    experiment.artifact_location, self.context.root,
                )
                experiment_id = str(experiment.experiment_id)
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
            self._fault("after_candidate_namespace")
            self._fault("after_candidate_tags")
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
                    prefix="quantpits-aggregate-"
                ) as temp_dir:
                    temporary = Path(temp_dir)
                    (temporary / "pred.pkl").write_bytes(prediction_bytes)
                    (temporary / "aggregate_manifest.json").write_bytes(
                        _json_bytes(payload)
                    )
                    client.log_artifact(
                        recorder_id, str(temporary / "pred.pkl"),
                    )
                    self._fault("after_candidate_prediction")
                    client.log_artifact(
                        recorder_id,
                        str(temporary / "aggregate_manifest.json"),
                    )
                    self._fault("after_candidate_manifest")
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
            _local_artifact_root(
                recorder.get_artifact_uri(), self.context.root,
            )
            completed_lock_meta = os.lstat(str(lock_dir))
            if (
                completed_lock_meta.st_dev,
                completed_lock_meta.st_ino,
            ) != lock_parent_identity:
                raise RollingAggregateBackendError(
                    "aggregate lock parent identity drifted"
                )
            after = self.inventory(aggregate_scope)
            if after["raw_count"] != before["raw_count"] + 1:
                raise RollingAggregateBackendError(
                    "candidate write did not create exactly one recorder"
                )
        observed = self.inspect_candidate(
            aggregate_scope, target_key, candidate_key,
        )
        self._fault("after_candidate_reinspection")
        return observed
