from pathlib import Path

from quantpits.runtime.mlflow_integrity import (
    ensure_writable_experiment,
    inspect_tracking_backend_presence,
    inspect_mlflow_workspace,
    resolve_mlflow_resource_uri,
)


def test_missing_backend_presence_is_pure(tmp_path):
    root = tmp_path / "Demo_Workspace"; root.mkdir()
    target = root / "mlflow.db"
    result = inspect_tracking_backend_presence("sqlite:///%s" % target, workspace_root=root)
    assert not result.metadata_ready
    assert result.issue.code == "tracking_backend_missing"
    assert not target.exists()


def test_safe_target_experiment_bootstrap(tmp_path):
    root = tmp_path / "Demo_Workspace"; root.mkdir()
    client = Client(root)
    def create(name, location):
        client.experiments.append({"name": name, "experiment_id": "7", "artifact_location": location})
    client.create_experiment = create
    result = ensure_writable_experiment(
        "Ensemble_Fusion", workspace_root=root, client=client,
        artifact_root=root / "output" / "mlflow_artifacts" / "Ensemble_Fusion",
    )
    assert result.selected_id == "7"
    assert result.artifact_location.contained
    assert not (root / "mlruns").exists()


class Client:
    def __init__(self, root, experiments=(), recorders=None):
        self.root = root
        self.experiments = list(experiments)
        self.recorders = recorders or {}

    def tracking_uri(self):
        return f"file://{self.root / 'mlruns'}"

    def experiments_by_name(self, name):
        return [item for item in self.experiments if item["name"] == name]

    def recorder(self, recorder_id, experiment_name=None):
        return self.recorders[recorder_id]


def test_uri_containment_uses_canonical_paths_and_rejects_sibling_prefix(tmp_path):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    inside = resolve_mlflow_resource_uri(f"file://{root / 'mlruns' / '1'}", workspace_root=root)
    outside = resolve_mlflow_resource_uri(
        f"file://{tmp_path / 'Demo_Workspace_Copy' / 'mlruns'}", workspace_root=root
    )
    assert inside.contained and inside.workspace_relative_path == "mlruns/1"
    assert not outside.contained and outside.public_path() == "<external>"


def test_sqlite_and_percent_encoded_paths(tmp_path):
    root = tmp_path / "Demo Workspace"
    root.mkdir()
    sqlite = resolve_mlflow_resource_uri(f"sqlite:///{root / 'mlflow.db'}", workspace_root=root, resource_kind="tracking")
    encoded = resolve_mlflow_resource_uri(
        f"file://{str(root / 'ml runs').replace(' ', '%20')}", workspace_root=root
    )
    assert sqlite.contained
    assert encoded.contained


def test_symlink_workspace_and_canonical_target_share_containment(tmp_path):
    canonical = tmp_path / "canonical" / "Demo_Workspace"
    canonical.mkdir(parents=True)
    link = tmp_path / "Demo_Link"
    link.symlink_to(canonical, target_is_directory=True)
    ref = resolve_mlflow_resource_uri(
        f"file://{canonical / 'mlruns/1'}", workspace_root=link
    )
    assert ref.contained
    assert ref.workspace_relative_path == "mlruns/1"


def test_write_experiment_must_be_unique_and_contained(tmp_path):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    experiments = [
        {"name": "Ensemble_Fusion", "experiment_id": "1", "artifact_location": f"file://{root / 'mlruns/1'}"},
        {"name": "Ensemble_Fusion", "experiment_id": "2", "artifact_location": f"file://{root / 'mlruns/2'}"},
    ]
    report = inspect_mlflow_workspace(
        workspace_root=root,
        client=Client(root, experiments),
        write_experiments=("Ensemble_Fusion",),
    )
    assert report.has_errors()
    assert "duplicate_active_experiment" in {item.code for item in report.issues}
    assert str(tmp_path) not in str(report.to_public_dict())


def test_exact_recorder_external_artifact_is_error(tmp_path):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    report = inspect_mlflow_workspace(
        workspace_root=root,
        client=Client(root, recorders={
            "r1": {"id": "r1", "experiment_name": "Source", "experiment_id": "1", "artifact_uri": f"file://{tmp_path / 'external'}"}
        }),
        recorder_requests=(("r1", "Source"),),
    )
    assert "recorder_artifact_outside_workspace" in {item.code for item in report.issues}
