import json

from quantpits.runtime.mlflow_integrity import inspect_mlflow_workspace
from quantpits.tools import audit_mlflow_workspace as tool


class Client:
    def __init__(self, root):
        self.root = root

    def tracking_uri(self):
        return f"file://{self.root / 'mlruns'}"

    def experiments_by_name(self, name):
        return [{"name": name, "experiment_id": "1", "artifact_location": f"file://{self.root / 'mlruns/1'}"}]

    def recorder(self, recorder_id, experiment_name=None):
        return {"id": recorder_id, "experiment_id": "1", "experiment_name": experiment_name or "Source", "artifact_uri": f"file://{self.root / 'mlruns/1/run/artifacts'}"}

    def all_recorder_requests(self):
        return [("all-run", "Source")]


def test_audit_cli_json_is_read_only_and_redacted(tmp_path, monkeypatch, capsys):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    (root / "mlruns" / "0").mkdir(parents=True)
    (root / "mlruns" / "0" / "meta.yaml").write_text("name: Default\n", encoding="utf-8")
    monkeypatch.setattr(tool, "_MlflowClientAdapter", lambda: Client(root))
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
    before = sorted(path.relative_to(root) for path in root.rglob("*"))
    code = tool.run(tool.build_parser().parse_args([
        "--workspace", str(root), "--experiment", "Source", "--recorder-id", "r1", "--json"
    ]))
    payload = json.loads(capsys.readouterr().out)
    after = sorted(path.relative_to(root) for path in root.rglob("*"))
    assert code == 0
    assert payload["workspace"] == "Demo_Workspace"
    assert str(tmp_path) not in json.dumps(payload)
    assert before == after


def test_audit_cli_all_runs_uses_expensive_scan_only_when_requested(tmp_path, monkeypatch, capsys):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    (root / "mlruns" / "0").mkdir(parents=True)
    (root / "mlruns" / "0" / "meta.yaml").write_text("name: Default\n", encoding="utf-8")
    client = Client(root)
    monkeypatch.setattr(tool, "_MlflowClientAdapter", lambda: client)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
    code = tool.run(tool.build_parser().parse_args([
        "--workspace", str(root), "--all-runs", "--json"
    ]))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [item["recorder_id"] for item in payload["recorders"]] == ["all-run"]


def test_missing_backend_audit_does_not_construct_client_or_create_files(tmp_path, monkeypatch, capsys):
    root = tmp_path / "Demo_Workspace"; root.mkdir()
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(tool, "_MlflowClientAdapter", lambda: (_ for _ in ()).throw(AssertionError("client constructed")))
    code = tool.run(tool.build_parser().parse_args(["--workspace", str(root), "--json"]))
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["issues"][0]["code"] == "tracking_backend_missing"
    assert list(root.iterdir()) == []
