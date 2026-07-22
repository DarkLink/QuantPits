from contextlib import contextmanager
from types import SimpleNamespace

from quantpits.rolling import LinearSlideUnitRunner
from quantpits.rolling.identity import workspace_fingerprint
from quantpits.rolling.unit_runner import _WORKER_MARKER, _worker_payload
from quantpits.utils.workspace import WorkspaceContext

from tests.quantpits.rolling.execution_support import linear_capability_result, make_scope


def test_unit_adapter_does_not_write_current_combined_backtest_history_or_promotion(monkeypatch, tmp_path):
    root = (tmp_path / "workspace").resolve()
    for name in ("config", "data", "mlruns", "output"):
        (root / name).mkdir(parents=True, exist_ok=True)
    context = WorkspaceContext.from_root(root)
    scope = make_scope(context, linear_capability_result())

    class FakeModel:
        def fit(self, dataset):
            self.fitted = dataset

        def predict(self, dataset, segment):
            return {"segment": segment}

    class FakeDataset:
        pass

    class FakeRecorder:
        info = {"id": "recorder-1"}

        def __init__(self):
            self.saved = {}

        def save_objects(self, **kwargs):
            self.saved.update(kwargs)

        def get_artifact_uri(self):
            artifact_root = context.mlruns_dir / "fake-artifacts"
            artifact_root.mkdir(parents=True, exist_ok=True)
            return artifact_root.resolve().as_uri()

    class FakeExperiment:
        id = "experiment-1"

    class FakeR:
        recorder = FakeRecorder()
        tags = {}

        @classmethod
        @contextmanager
        def start(cls, experiment_name):
            yield

        @classmethod
        def get_recorder(cls):
            return cls.recorder

        @classmethod
        def set_tags(cls, **kwargs):
            cls.tags.update(kwargs)

        @classmethod
        def get_exp(cls, **kwargs):
            return FakeExperiment()

        @classmethod
        def get_uri(cls):
            return context.mlflow_uri

    def fake_init(config):
        return FakeModel() if config["class"] == "LinearModel" else FakeDataset()

    import qlib.utils
    import qlib.workflow
    import qlib

    monkeypatch.setattr(qlib, "init", lambda **_kwargs: None)
    monkeypatch.setattr(qlib.utils, "init_instance_by_config", fake_init)
    monkeypatch.setattr(qlib.workflow, "R", FakeR)
    forbidden = (
        context.data_dir / "latest_train_records.json",
        context.data_dir / "rolling_prediction_history.jsonl",
        context.data_dir / "promote_history.jsonl",
        context.output_dir / "Rolling_Combined_forbidden",
    )
    unit = scope.units[0]
    window = unit.window.identity
    payload = {
        "workspace_root": str(context.root),
        "workspace_fingerprint": workspace_fingerprint(context.root),
        "mlflow_uri": context.mlflow_uri,
        "qlib_data_dir": str(context.qlib_data_dir),
        "qlib_region": context.qlib_region,
        "workflow_relative_path": unit.target.workflow_relative_path,
        "workflow_fingerprint": unit.target.workflow_fingerprint,
        "target_key": unit.unit_key[0],
        "window": {
            "family": window.family, "train_start": window.train_start,
            "train_end": window.train_end, "valid_start": window.valid_start,
            "valid_end": window.valid_end, "test_start": window.test_start,
            "test_end": window.test_end,
            "effective_config_fingerprint": window.effective_config_fingerprint,
            "folds": [],
        },
        "window_key": unit.unit_key[1],
        "run_fingerprint": scope.run_identity.fingerprint,
        "source_operation": scope.run_identity.action,
        "attempt_id": "attempt-1", "experiment_name": "exact-unit-experiment",
        "runtime_params": {"market": "synthetic", "benchmark": "synthetic"},
    }
    worker_result = _worker_payload(payload)
    assert worker_result["status"] == "candidate_success"
    assert set(FakeR.recorder.saved) == {"model.pkl", "pred.pkl"}
    assert "execution_protocol" in FakeR.tags

    monkeypatch.setattr(
        "quantpits.rolling.unit_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=_WORKER_MARKER + '{"experiment_id":"experiment-1",'
            '"experiment_name":"exact-unit-experiment",'
            '"recorder_id":"recorder-1","status":"candidate_success"}\n',
            stderr="",
        ),
    )
    runner = LinearSlideUnitRunner(
        context, {"market": "synthetic", "benchmark": "synthetic"},
        "exact-unit-experiment",
    )
    observation = runner.execute(scope, unit, "attempt-1")
    assert observation.candidate_status == "candidate_success"
    assert all(not path.exists() for path in forbidden)
