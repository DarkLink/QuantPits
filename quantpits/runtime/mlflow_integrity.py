"""Workspace containment checks for local MLflow resources.

The pure URI helpers in this module deliberately do not import Qlib or MLflow.
Runtime metadata inspection is kept behind a small injectable protocol so plans and
unit tests stay lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.parse import unquote, urlparse


class MlflowWorkspaceIntegrityError(RuntimeError):
    """Base error for unsafe or ambiguous MLflow lineage."""

    def __init__(self, message: str, report: "MlflowIntegrityReport | None" = None):
        super().__init__(message)
        self.report = report


class DuplicateMlflowExperimentError(MlflowWorkspaceIntegrityError):
    pass


class ExternalMlflowArtifactError(MlflowWorkspaceIntegrityError):
    pass


class RecorderIdentityError(MlflowWorkspaceIntegrityError):
    pass


@dataclass(frozen=True)
class MlflowIntegrityIssue:
    code: str
    severity: str
    resource_kind: str
    message: str
    experiment_name: str | None = None
    experiment_id: str | None = None
    recorder_id: str | None = None


@dataclass(frozen=True)
class MlflowResourceRef:
    resource_kind: str
    uri: str
    canonical_path: Path | None
    workspace_relative_path: str | None
    contained: bool
    scheme: str

    def public_path(self) -> str:
        return self.workspace_relative_path or "<external>"


@dataclass(frozen=True)
class ExperimentIntegrity:
    name: str
    active_ids: tuple[str, ...]
    selected_id: str | None
    artifact_location: MlflowResourceRef | None
    access_mode: str
    issues: tuple[MlflowIntegrityIssue, ...] = ()


@dataclass(frozen=True)
class RecorderIntegrity:
    recorder_id: str
    experiment_name: str
    experiment_id: str
    artifact_uri: MlflowResourceRef
    issues: tuple[MlflowIntegrityIssue, ...] = ()


@dataclass(frozen=True)
class MlflowIntegrityReport:
    workspace_root: Path
    tracking: MlflowResourceRef
    experiments: tuple[ExperimentIntegrity, ...] = ()
    recorders: tuple[RecorderIntegrity, ...] = ()
    issues: tuple[MlflowIntegrityIssue, ...] = ()

    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace_root.name,
            "tracking": _resource_dict(self.tracking),
            "experiments": [
                {
                    "name": item.name,
                    "active_ids": list(item.active_ids),
                    "selected_id": item.selected_id,
                    "access_mode": item.access_mode,
                    "artifact": _resource_dict(item.artifact_location),
                    "issues": [_issue_dict(issue) for issue in item.issues],
                }
                for item in self.experiments
            ],
            "recorders": [
                {
                    "recorder_id": item.recorder_id,
                    "experiment_name": item.experiment_name,
                    "experiment_id": item.experiment_id,
                    "artifact": _resource_dict(item.artifact_uri),
                    "issues": [_issue_dict(issue) for issue in item.issues],
                }
                for item in self.recorders
            ],
            "issues": [_issue_dict(issue) for issue in self.issues],
        }


def _issue_dict(issue: MlflowIntegrityIssue) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "code": issue.code,
            "severity": issue.severity,
            "resource_kind": issue.resource_kind,
            "message": issue.message,
            "experiment_name": issue.experiment_name,
            "experiment_id": issue.experiment_id,
            "recorder_id": issue.recorder_id,
        }.items()
        if value is not None
    }


def _resource_dict(ref: MlflowResourceRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "kind": ref.resource_kind,
        "scheme": ref.scheme,
        "contained": ref.contained,
        "path": ref.public_path(),
    }


def resolve_mlflow_resource_uri(
    uri: str,
    *,
    workspace_root: str | Path,
    resource_kind: str = "recorder",
    base_path: str | Path | None = None,
) -> MlflowResourceRef:
    """Resolve a local MLflow URI and classify canonical workspace containment."""

    root = Path(workspace_root).expanduser().resolve()
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    candidate: Path | None

    if scheme == "sqlite":
        raw = unquote(parsed.path)
        candidate = Path(raw)
    elif scheme == "file":
        raw = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            candidate = None
        else:
            candidate = Path(raw)
    elif scheme == "":
        candidate = Path(unquote(uri))
        if not candidate.is_absolute():
            candidate = Path(base_path).resolve() / candidate if base_path else root / candidate
    else:
        candidate = None

    if candidate is None:
        return MlflowResourceRef(resource_kind, uri, None, None, False, scheme or "local")

    canonical = candidate.expanduser().resolve()
    try:
        relative = canonical.relative_to(root).as_posix()
        contained = True
    except ValueError:
        relative = None
        contained = False
    return MlflowResourceRef(
        resource_kind=resource_kind,
        uri=uri,
        canonical_path=canonical,
        workspace_relative_path=relative,
        contained=contained,
        scheme=scheme or "local",
    )


class MlflowMetadataClient(Protocol):
    def tracking_uri(self) -> str: ...
    def experiments_by_name(self, name: str) -> Iterable[Any]: ...
    def recorder(self, recorder_id: str, experiment_name: str | None = None) -> Any: ...


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
        info = getattr(obj, "info", None)
        if isinstance(info, dict) and name in info:
            return info[name]
    return default


def inspect_mlflow_workspace(
    *,
    workspace_root: str | Path,
    client: MlflowMetadataClient,
    experiment_names: Iterable[str] = (),
    recorder_requests: Iterable[tuple[str, str | None]] = (),
    write_experiments: Iterable[str] = (),
) -> MlflowIntegrityReport:
    """Inspect selected MLflow metadata without mutating the backend."""

    root = Path(workspace_root).resolve()
    issues: list[MlflowIntegrityIssue] = []
    tracking = resolve_mlflow_resource_uri(
        client.tracking_uri(), workspace_root=root, resource_kind="tracking"
    )
    if not tracking.contained:
        issues.append(MlflowIntegrityIssue(
            "unsupported_tracking_scheme" if tracking.canonical_path is None else "tracking_outside_workspace",
            "error", "tracking",
            "MLflow tracking backend is unsupported or outside the active workspace.",
        ))

    write_names = set(write_experiments)
    experiments: list[ExperimentIntegrity] = []
    for name in dict.fromkeys(tuple(experiment_names) + tuple(write_names)):
        active = [item for item in client.experiments_by_name(name) if str(_field(item, "lifecycle_stage", default="active")) == "active"]
        ids = tuple(str(_field(item, "experiment_id", "id")) for item in active)
        item_issues: list[MlflowIntegrityIssue] = []
        if len(active) > 1:
            item_issues.append(MlflowIntegrityIssue(
                "duplicate_active_experiment",
                "error" if name in write_names else "warning", "experiment",
                "Experiment name resolves to multiple active experiments.",
                experiment_name=name,
            ))
        elif name in write_names and not active:
            item_issues.append(MlflowIntegrityIssue(
                "target_experiment_missing", "error", "experiment",
                "Writable experiment name must resolve to exactly one active experiment.",
                experiment_name=name,
            ))
        selected = active[0] if len(active) == 1 else None
        artifact = None
        if selected is not None:
            location = str(_field(selected, "artifact_location", default=""))
            artifact = resolve_mlflow_resource_uri(location, workspace_root=root, resource_kind="experiment")
            if not artifact.contained:
                item_issues.append(MlflowIntegrityIssue(
                    "experiment_artifact_outside_workspace", "error", "experiment",
                    "Experiment artifact location is outside the active workspace.",
                    experiment_name=name, experiment_id=ids[0],
                ))
        elif active:
            for candidate in active:
                candidate_id = str(_field(candidate, "experiment_id", "id"))
                location = str(_field(candidate, "artifact_location", default=""))
                candidate_ref = resolve_mlflow_resource_uri(
                    location, workspace_root=root, resource_kind="experiment"
                )
                if not candidate_ref.contained:
                    item_issues.append(MlflowIntegrityIssue(
                        "experiment_artifact_outside_workspace", "error", "experiment",
                        "Experiment artifact location is outside the active workspace.",
                        experiment_name=name, experiment_id=candidate_id,
                    ))
        issues.extend(item_issues)
        experiments.append(ExperimentIntegrity(name, ids, ids[0] if selected is not None else None, artifact, "write" if name in write_names else "read", tuple(item_issues)))

    recorders: list[RecorderIntegrity] = []
    for recorder_id, expected_name in recorder_requests:
        try:
            recorder = client.recorder(recorder_id, expected_name)
        except Exception:
            issue = MlflowIntegrityIssue("recorder_not_found", "error", "recorder", "Requested recorder was not found.", experiment_name=expected_name, recorder_id=recorder_id)
            issues.append(issue)
            continue
        actual_id = str(_field(recorder, "id", "run_id", default=recorder_id))
        experiment_id = str(_field(recorder, "experiment_id", default=""))
        actual_name = str(_field(recorder, "experiment_name", default=expected_name or ""))
        artifact_uri = str(_field(recorder, "artifact_uri", default=""))
        ref = resolve_mlflow_resource_uri(artifact_uri, workspace_root=root, resource_kind="recorder")
        item_issues: list[MlflowIntegrityIssue] = []
        if actual_id != recorder_id or (expected_name and actual_name and actual_name != expected_name):
            item_issues.append(MlflowIntegrityIssue("recorder_experiment_mismatch", "error", "recorder", "Recorder identity does not match the requested lineage.", experiment_name=expected_name, recorder_id=recorder_id))
        if not ref.contained:
            item_issues.append(MlflowIntegrityIssue("recorder_artifact_outside_workspace", "error", "recorder", "Recorder artifacts are outside the active workspace.", experiment_name=expected_name, recorder_id=recorder_id))
        issues.extend(item_issues)
        recorders.append(RecorderIntegrity(recorder_id, actual_name, experiment_id, ref, tuple(item_issues)))

    return MlflowIntegrityReport(root, tracking, tuple(experiments), tuple(recorders), tuple(issues))


def require_mlflow_integrity(report: MlflowIntegrityReport) -> None:
    if report.has_errors():
        codes = ", ".join(sorted({issue.code for issue in report.issues}))
        if "duplicate_active_experiment" in codes:
            raise DuplicateMlflowExperimentError(f"MLflow integrity check failed: {codes}", report)
        if "outside_workspace" in codes:
            raise ExternalMlflowArtifactError(f"MLflow integrity check failed: {codes}", report)
        raise MlflowWorkspaceIntegrityError(f"MLflow integrity check failed: {codes}", report)
