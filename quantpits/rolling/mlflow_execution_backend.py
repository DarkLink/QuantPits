"""Local Qlib/MLflow adapter for execution-bound manifests and evidence reads."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from quantpits.rolling.errors import (
    RollingExecutionBackendError,
    RollingExecutionContractError,
)
from quantpits.rolling.evidence import (
    RollingArtifactExpectation,
    RollingUnitEvidenceRequest,
    inspect_rolling_evidence,
)
from quantpits.rolling.execution import RollingExecutionScope, RollingUnitRunnerObservation
from quantpits.rolling.identity import RollingRunIdentity, workspace_fingerprint
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


_INVENTORY_TOKEN = object()


@dataclass(frozen=True)
class _RecorderInventoryBaseline:
    run_fingerprint: str
    attempt_id: str
    unit_key: tuple
    recorders: tuple
    _authority: InitVar[object] = None
    _backend_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority):
        if _authority is not _INVENTORY_TOKEN:
            raise RollingExecutionContractError(
                "recorder inventory baselines are backend-observer-owned"
            )
        object.__setattr__(self, "_backend_authority", True)


def _local_artifact_root(uri, workspace_root):
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file"):
        raise RollingExecutionBackendError("only a contained local artifact backend is supported")
    raw = unquote(parsed.path if parsed.scheme else uri)
    path = Path(raw).expanduser().resolve(strict=True)
    root = Path(workspace_root).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError:
        raise RollingExecutionBackendError("recorder artifact root escapes the workspace")
    if not path.is_dir():
        raise RollingExecutionBackendError("recorder artifact root is not a directory")
    return path


def _tracking_uri_identity(uri, workspace_root):
    parsed = urlparse(str(uri))
    if parsed.scheme not in ("file", "sqlite") or parsed.netloc not in ("", None):
        return False, False
    raw = unquote(parsed.path)
    if parsed.scheme == "sqlite" and raw.startswith("//"):
        raw = raw[1:]
    path = Path(raw).expanduser().absolute()
    root = Path(workspace_root).resolve(strict=True)
    try:
        parent = path.parent.resolve(strict=True)
        parent.relative_to(root)
        if path.is_symlink():
            return False, False
        if path.exists():
            path.resolve(strict=True).relative_to(root)
            present = path.is_file() if parsed.scheme == "sqlite" else path.is_dir()
        else:
            present = parsed.scheme == "sqlite"
    except (OSError, ValueError):
        return False, False
    return present, True


def _artifact(path, logical_key, role):
    node = path / logical_key
    try:
        physical = node.resolve(strict=True)
        physical.relative_to(path.resolve(strict=True))
    except (OSError, ValueError):
        raise RollingExecutionBackendError("recorder artifact is missing or physically escaped")
    if node.is_symlink() or not physical.is_file():
        raise RollingExecutionBackendError("recorder artifact is not a regular file")
    data = physical.read_bytes()
    return RollingArtifactExpectation(
        logical_key, role, len(data), hashlib.sha256(data).hexdigest(),
    )


class QlibMlflowExecutionBackend:
    """Bind one local recorder to Phase 32 immutable evidence inspection."""

    def __init__(self, context):
        if not isinstance(context, WorkspaceContext):
            raise RollingExecutionContractError("backend requires WorkspaceContext")
        self.context = context
        self._requests = {}
        self._selector_candidates = {}

    @property
    def backend_fingerprint(self):
        return fingerprint_value({"tracking_uri": self.context.mlflow_uri})

    @staticmethod
    def calendar_sessions(start, end):
        from qlib.data import D

        return tuple(D.calendar(start_time=start, end_time=end, freq="day"))

    def tracking_identity(self):
        from qlib.workflow import R

        current = str(R.get_uri())
        expected = str(self.context.mlflow_uri)
        present, contained = _tracking_uri_identity(current, self.context.root)
        return {
            "workspace_fingerprint": workspace_fingerprint(self.context.root),
            "backend_fingerprint": self.backend_fingerprint,
            "present": present,
            "contained": current == expected and contained,
            "foreign": current != expected or not contained,
        }

    @staticmethod
    def _recorder(experiment_name, recorder_id):
        from qlib.workflow import R

        return R.get_recorder(
            recorder_id=recorder_id, experiment_name=experiment_name,
        )

    def _request_from_recorder(self, scope, unit, attempt_id, experiment_name, experiment_id, recorder):
        artifact_root = _local_artifact_root(recorder.get_artifact_uri(), self.context.root)
        def expected_or_missing(key, role):
            try:
                return _artifact(artifact_root, key, role)
            except RollingExecutionBackendError:
                return RollingArtifactExpectation(key, role, 0, "0" * 64)

        manifest_expectation = expected_or_missing(
            "execution_manifest.json", "supporting",
        )
        expectations = (
            manifest_expectation,
            expected_or_missing("model.pkl", "supporting"),
            expected_or_missing("pred.pkl", "prediction"),
        )
        try:
            manifest = json.loads((artifact_root / "execution_manifest.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            manifest = None
        expected_manifest = {
            "protocol": "execution_bound_v1",
            "run_fingerprint": scope.run_identity.fingerprint,
            "attempt_id": attempt_id,
            "target_key": unit.unit_key[0],
            "window_key": unit.unit_key[1],
            "source_operation": scope.run_identity.action,
            "workflow_fingerprint": unit.target.workflow_fingerprint,
            "capability_fingerprint": unit.target.capability_result_fingerprint,
            "experiment_name": experiment_name,
            "experiment_id": str(experiment_id),
            "recorder_id": recorder.info["id"],
            "expected_prediction_sessions": list(unit.window.expected_sessions),
            "artifacts": [
                expectations[1].to_fingerprint_dict(),
                expectations[2].to_fingerprint_dict(),
            ],
        }
        manifest_valid = isinstance(manifest, dict) and not any(
            manifest.get(key) != value for key, value in expected_manifest.items()
        )
        if manifest_valid:
            manifest_core = dict(manifest)
            claimed_content = manifest_core.pop("manifest_content_fingerprint", None)
            manifest_valid = claimed_content == fingerprint_value(manifest_core)
        run = RollingRunIdentity(
            workspace_fingerprint=scope.run_identity.workspace_fingerprint,
            family=scope.run_identity.family,
            action=scope.run_identity.action,
            plan_fingerprint=scope.run_identity.plan_fingerprint,
            config_fingerprint=scope.run_identity.config_fingerprint,
            anchor_date=scope.run_identity.anchor_date,
            target_keys=scope.run_identity.target_keys,
            window_keys=scope.run_identity.window_keys,
            runtime_params_fingerprint=scope.run_identity.runtime_params_fingerprint,
            attempt_id=attempt_id,
        )
        request = RollingUnitEvidenceRequest(
            run, unit.unit_key[0], unit.window.identity,
            "execution_bound_v1", unit.unit_key[0], scope.run_identity.action,
            experiment_name, str(experiment_id), recorder.info["id"],
            expectations, unit.window.expected_sessions,
        )
        return request, artifact_root, manifest_valid

    @staticmethod
    def _candidate(request, artifact_root, backend_fingerprint, manifest_valid=True):
        candidate = {
            "workspace_fingerprint": request.run_identity.workspace_fingerprint,
            "backend_fingerprint": backend_fingerprint,
            "experiment_name": request.experiment_name,
            "experiment_id": request.experiment_id,
            "recorder_id": request.recorder_id,
            "run_fingerprint": request.run_identity.fingerprint,
            "attempt_id": request.run_identity.attempt_id,
            "plan_fingerprint": request.run_identity.plan_fingerprint,
            "config_fingerprint": request.run_identity.config_fingerprint,
            "target_key": request.target_key,
            "window_key": request.window_key,
            "source_protocol": request.source_protocol,
            "source_publication_key": request.source_publication_key,
            "source_operation": request.source_operation,
            "source_manifest_fingerprint": request.source_manifest_fingerprint,
            "artifact_root_uri": artifact_root.as_uri(),
        }
        if not manifest_valid:
            candidate["source_manifest_fingerprint"] = fingerprint_value({
                "invalid_execution_manifest": request.recorder_id,
            })
        return candidate

    def _discover_original_selector(self, scope, unit, attempt_id):
        from qlib.workflow import R

        found = []
        for experiment_name, experiment in sorted(R.list_experiments().items()):
            recorders = R.list_recorders(experiment_name=experiment_name)
            for recorder_id, recorder in sorted(recorders.items()):
                try:
                    tags = recorder.list_tags()
                except (KeyboardInterrupt, SystemExit, GeneratorExit):
                    raise
                except Exception:
                    continue
                if not all(str(tags.get(key)) == str(value) for key, value in {
                    "execution_protocol": scope.execution_protocol_version,
                    "run_fingerprint": scope.run_identity.fingerprint,
                    "attempt_id": attempt_id,
                    "target_key": unit.unit_key[0],
                    "window_key": unit.unit_key[1],
                    "source_operation": scope.run_identity.action,
                }.items()):
                    continue
                try:
                    request, root, manifest_valid = self._request_from_recorder(
                        scope, unit, attempt_id, experiment_name, experiment.id, recorder,
                    )
                except (KeyboardInterrupt, SystemExit, GeneratorExit):
                    raise
                except Exception as exc:
                    raise RollingExecutionBackendError(
                        "original selector candidate is not safely observable"
                    ) from exc
                found.append((request, self._candidate(
                    request, root, self.backend_fingerprint, manifest_valid,
                )))
        return tuple(found)

    @staticmethod
    def _recorder_inventory():
        from qlib.workflow import R

        observed = []
        for experiment_name, experiment in sorted(R.list_experiments().items()):
            for recorder_id in sorted(R.list_recorders(
                experiment_name=experiment_name,
            )):
                observed.append((str(experiment_name), str(experiment.id), str(recorder_id)))
        return tuple(observed)

    def capture_recorder_inventory(self, scope, unit, attempt_id):
        return _RecorderInventoryBaseline(
            scope.run_identity.fingerprint, attempt_id, unit.unit_key,
            self._recorder_inventory(), _authority=_INVENTORY_TOKEN,
        )

    def commit_execution_manifest(self, scope, unit, observation, recorder_baseline):
        if not isinstance(scope, RollingExecutionScope):
            raise RollingExecutionContractError("manifest commit requires typed scope")
        if not isinstance(observation, RollingUnitRunnerObservation):
            raise RollingExecutionContractError("manifest commit requires typed observation")
        if (
            not isinstance(recorder_baseline, _RecorderInventoryBaseline)
            or not recorder_baseline._backend_authority
            or recorder_baseline.run_fingerprint != scope.run_identity.fingerprint
            or recorder_baseline.attempt_id != observation.attempt_id
            or recorder_baseline.unit_key != unit.unit_key
        ):
            raise RollingExecutionContractError(
                "manifest commit requires the matching recorder inventory baseline"
            )
        after = self._recorder_inventory()
        created = tuple(item for item in after if item not in set(recorder_baseline.recorders))
        expected_created = (
            observation.experiment_name,
            str(observation.experiment_id),
            observation.recorder_id,
        )
        if created != (expected_created,):
            raise RollingExecutionBackendError(
                "runner did not create exactly its one claimed recorder"
            )
        from qlib.workflow import R
        experiment = R.get_exp(
            experiment_name=observation.experiment_name, create=False,
        )
        if str(experiment.id) != str(observation.experiment_id):
            raise RollingExecutionBackendError("runner experiment identity changed")
        recorder = self._recorder(observation.experiment_name, observation.recorder_id)
        if str(recorder.info.get("id")) != observation.recorder_id:
            raise RollingExecutionBackendError("runner recorder identity changed")
        try:
            tags = recorder.list_tags()
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            raise RollingExecutionBackendError("runner recorder tags are unavailable") from exc
        expected_tags = {
            "execution_protocol": scope.execution_protocol_version,
            "run_fingerprint": scope.run_identity.fingerprint,
            "attempt_id": observation.attempt_id,
            "target_key": unit.unit_key[0],
            "window_key": unit.unit_key[1],
            "source_operation": scope.run_identity.action,
        }
        if not all(str(tags.get(key)) == str(value) for key, value in expected_tags.items()):
            raise RollingExecutionBackendError("runner recorder provenance tags disagree")
        artifact_root = _local_artifact_root(recorder.get_artifact_uri(), self.context.root)
        model = _artifact(artifact_root, "model.pkl", "supporting")
        prediction = _artifact(artifact_root, "pred.pkl", "prediction")
        manifest_payload = {
            "protocol": "execution_bound_v1",
            "run_fingerprint": scope.run_identity.fingerprint,
            "attempt_id": observation.attempt_id,
            "target_key": unit.unit_key[0],
            "window_key": unit.unit_key[1],
            "source_operation": scope.run_identity.action,
            "workflow_fingerprint": unit.target.workflow_fingerprint,
            "capability_fingerprint": unit.target.capability_result_fingerprint,
            "experiment_name": observation.experiment_name,
            "experiment_id": observation.experiment_id,
            "recorder_id": observation.recorder_id,
            "expected_prediction_sessions": list(unit.window.expected_sessions),
            "artifacts": [model.to_fingerprint_dict(), prediction.to_fingerprint_dict()],
        }
        manifest_payload["manifest_content_fingerprint"] = fingerprint_value(manifest_payload)
        manifest_data = (
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="quantpits-execution-manifest-") as temp_dir:
            local = Path(temp_dir) / "execution_manifest.json"
            local.write_bytes(manifest_data)
            recorder.log_artifact(str(local))
        if self._recorder_inventory() != after:
            raise RollingExecutionBackendError("recorder inventory drifted during manifest commit")
        manifest = _artifact(artifact_root, "execution_manifest.json", "supporting")
        run = RollingRunIdentity(
            workspace_fingerprint=scope.run_identity.workspace_fingerprint,
            family=scope.run_identity.family,
            action=scope.run_identity.action,
            plan_fingerprint=scope.run_identity.plan_fingerprint,
            config_fingerprint=scope.run_identity.config_fingerprint,
            anchor_date=scope.run_identity.anchor_date,
            target_keys=scope.run_identity.target_keys,
            window_keys=scope.run_identity.window_keys,
            runtime_params_fingerprint=scope.run_identity.runtime_params_fingerprint,
            attempt_id=observation.attempt_id,
        )
        request = RollingUnitEvidenceRequest(
            run, unit.unit_key[0], unit.window.identity,
            "execution_bound_v1", unit.unit_key[0], scope.run_identity.action,
            observation.experiment_name, observation.experiment_id,
            observation.recorder_id, (manifest, model, prediction),
            unit.window.expected_sessions,
        )
        self._requests[unit.unit_key] = request
        return request

    def requests_for_state(self, scope, state):
        requests = []
        for unit, claim in zip(scope.units, state.units):
            cached = self._requests.get(unit.unit_key)
            if cached is not None and cached.recorder_id == claim.record_id:
                requests.append(cached)
                continue
            values = claim.extensions
            artifacts = values.get("artifacts") if isinstance(values, dict) else None
            if not isinstance(artifacts, list):
                discovered = self._discover_original_selector(
                    scope, unit, state.attempt_id,
                )
                if discovered:
                    self._selector_candidates[unit.unit_key] = tuple(
                        candidate for _request, candidate in discovered
                    )
                    request = discovered[0][0]
                    requests.append(request)
                    self._requests[unit.unit_key] = request
                    continue
            if not isinstance(artifacts, list):
                # An unresolved selector has no committed recorder.  A public,
                # impossible placeholder lets the inspector classify missing;
                # it is never used to discover a mutable current recorder.
                artifacts = [{
                    "logical_key": "pred.pkl", "role": "prediction",
                    "size_bytes": 0, "fingerprint": "0" * 64,
                }]
            expectations = tuple(RollingArtifactExpectation(
                item["logical_key"], item["role"], item["size_bytes"], item["fingerprint"],
            ) for item in artifacts)
            original_attempt = values.get("attempt_id", state.attempt_id) if isinstance(values, dict) else state.attempt_id
            run = RollingRunIdentity(
                workspace_fingerprint=scope.run_identity.workspace_fingerprint,
                family=scope.run_identity.family,
                action=scope.run_identity.action,
                plan_fingerprint=scope.run_identity.plan_fingerprint,
                config_fingerprint=scope.run_identity.config_fingerprint,
                anchor_date=scope.run_identity.anchor_date,
                target_keys=scope.run_identity.target_keys,
                window_keys=scope.run_identity.window_keys,
                runtime_params_fingerprint=scope.run_identity.runtime_params_fingerprint,
                attempt_id=original_attempt,
            )
            request = RollingUnitEvidenceRequest(
                run, unit.unit_key[0], unit.window.identity,
                "execution_bound_v1", unit.unit_key[0], scope.run_identity.action,
                values.get("experiment_name", "unresolved-experiment") if isinstance(values, dict) else "unresolved-experiment",
                values.get("experiment_id", "unresolved-experiment-id") if isinstance(values, dict) else "unresolved-experiment-id",
                (
                    values.get("recorder_id")
                    if isinstance(values, dict) and values.get("recorder_id")
                    else claim.record_id or "unresolved-%s" % (unit.position,)
                ),
                expectations, unit.window.expected_sessions,
            )
            frozen_manifest = values.get("source_manifest_fingerprint") if isinstance(values, dict) else None
            if frozen_manifest is not None and request.source_manifest_fingerprint != frozen_manifest:
                raise RollingExecutionBackendError("persisted source manifest identity changed")
            requests.append(request)
            self._requests[unit.unit_key] = request
        return tuple(requests)

    def inventory(self, requests):
        candidates = []
        for request in requests:
            discovered = self._selector_candidates.get(request.unit_key)
            if discovered is not None:
                candidates.extend(discovered)
                continue
            try:
                recorder = self._recorder(request.experiment_name, request.recorder_id)
                artifact_root = _local_artifact_root(recorder.get_artifact_uri(), self.context.root)
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception:
                continue
            candidates.append(self._candidate(
                request, artifact_root, self.backend_fingerprint,
            ))
        return {
            "fingerprint": fingerprint_value(candidates),
            "candidates": tuple(candidates),
        }

    def inspect(self, scope, requests):
        return inspect_rolling_evidence(self.context, requests, self)
